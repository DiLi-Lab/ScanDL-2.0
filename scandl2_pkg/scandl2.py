


import os
import sys 
import time
import json
import joblib
import argparse

from functools import partial 

from typing import Union, List, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.distributed as dist 
from torch.utils.data import DataLoader

import numpy as np 
import pandas as pd

from tqdm import tqdm 

from transformers import (
    set_seed, 
    BertTokenizerFast, 
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    GPT2Model,
    AutoConfig,
    BertModel,
)
from transformers.models.bert.modeling_bert import BertEncoder
from datasets import DatasetDict
from datasets import Dataset as Dataset2

from scandl2_pkg.scandl_module.original_scandl.sp_rounding import denoised_fn_round 
from scandl2_pkg.scandl_module.original_scandl.utils import dist_util, logger
from scandl2_pkg.scandl_module.original_scandl.utils.nn import * 

from scandl2_pkg.scandl2_utils import text_dataset_loader, FixdurDataset

from scandl2_pkg.scandl_module.scripts.sp_load_celer_zuco import _collate_batch_helper
from scandl2_pkg.scandl_module.scripts.sp_basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from scandl2_pkg.fix_dur_module.model_seq2seq import Seq2SeqModel
from scandl2_pkg.fix_dur_module.utils_data import aggregate_input_embeddings, padding_and_mask_seq2seq

from scandl2_pkg.PATHS import (
    SENT_SCANDL_MODULE,
    SENT_FIXDUR_MODULE,
    PAR_SCANDL_MODULE,
    PAR_FIXDUR_MODULE,
)


class ScanDL2(nn.Module):
    def __init__(
        self,
        text_type: str = 'sentence',  # sentence, paragraph
        bsz: Optional[int] = 2,
        save: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        super(ScanDL2, self).__init__()

        self.save = save
        self.filename = filename

        # initialize the ScanDL module and the Fixdur Module 
        self.scandl_module = ScanDLModule(
            text_type=text_type,
            bsz=bsz,
        )
        self.fixdur_module = FixdurModule(
            text_type=text_type,
            bsz=bsz,
        )


    def forward(
        self,
        texts: Union[str, List[str]],
    ):
        # check if the input is in the correct format
        self._validate_inputs(texts=texts)

        # get the fixation location predictions from the ScanDL module
        scandl_module_output = self.scandl_module(texts=texts)

        # get the fixation duration predictions from the Fixdur module
        fixdur_module_output = self.fixdur_module(scandl_module_output=scandl_module_output)

        if self.save is not None:
            filename = self.filename if self.filename is not None else f'scandl2_outputs.json'
            filename = f'{filename}.json' if not filename.endswith('.json') else filename
            self._save_results(results=fixdur_module_output, filename=filename)
        
        return fixdur_module_output


    def _save_results(self, results, filename):
        if not os.path.exists(self.save):
            os.makedirs(self.save)
        with open(os.path.join(self.save, filename), 'w') as f:
            json.dump(results, f)
        print(f'--- ScanDL 2.0 outputs saved to {os.path.join(self.save, filename)}.')
    

    def _validate_inputs(self, texts: Union[str, List[str]]) -> None:
        if not isinstance(texts, (str, list)) or (isinstance(texts, list) and not all(isinstance(t, str) for t in texts)):
            raise TypeError("Invalid input: 'texts' must be of type 'str' or 'List[str]'.")


class ScanDLModule(nn.Module):
    
    def __init__(
        self,
        text_type: str,  # sentence, paragraph
        bsz: int,
    ):
        super(ScanDLModule, self).__init__()

        base_path = 'scandl2_pkg'
        if text_type == 'paragraph':
            self.path_to_config = os.path.join(base_path, 'config_emtec.json')
            self.path_to_scandl_module = PAR_SCANDL_MODULE
        elif text_type == 'sentence':
            self.path_to_config = os.path.join(base_path, 'config.json')
            self.path_to_scandl_module = SENT_SCANDL_MODULE
        else:
            raise NotImplementedError(f'Text type {text_type} not implemented.')

        # get the args 
        self.args = self._get_args()
        self.args.batch_size = bsz

        # seting up the environment
        dist_util.setup_dist()
        logger.configure()
        self.world_size = dist.get_world_size() or 1
        self.rank = dist.get_rank() or 0
        #set_seed(self.args.seed2)

        # load the tokenizer
        self.tokenizer = self._load_tokenizer()

        # load the ScanDL module and the Diffusion
        self.scandl_module, self.diffusion = self._load_scandl_module(path_to_scandl_module=self.path_to_scandl_module)
        self.sn_sp_repr_embedding = self._get_sn_sp_repr_emb()

        

    def forward(
        self, 
        texts: Union[str, List[str]],
    ) -> Dict[str, Union[List[List[str]], List[List[int]], List[str]]]:
        
        data_loader = self._preprocess_text(texts=texts)

        predicted_sp_words, predicted_sp_ids = [], []
        original_sn = []

        print('\t\t### ScanDL Module generates fixation locations ...')

        unique_idx = list()
        idx_ctr = 0

        for batch_idx, batch in tqdm(enumerate(data_loader)):

            mask = batch['mask'].to(dist_util.dev())
            sn_sp_repr = batch['sn_sp_repr'].to(dist_util.dev())
            sn_input_ids = batch['sn_input_ids'].to(dist_util.dev())
            indices_pos_enc = batch['indices_pos_enc'].to(dist_util.dev())
            sn_repr_len = batch['sn_repr_len'].to(dist_util.dev())
            words_for_mapping = batch['words_for_mapping']

            sn_sp_emb, pos_enc, sn_input_ids_emb = self.scandl_module.get_embeds(
                sn_sp_repr=sn_sp_repr,
                sn_input_ids=sn_input_ids,
                indices_pos_enc=indices_pos_enc,
            )

            x_start = sn_sp_emb 
            noise = torch.randn_like(x_start)
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
            x_noised = torch.where(mask == 0, x_start, noise)

            self.args.use_ddim = False
            step_gap = 1

            sample_fn = (
                self.diffusion.p_sample_loop if not self.args.use_ddim else self.diffusion.ddim_sample_loop
            )

            sample_shape = (x_start.shape[0], self.args.seq_len, self.args.hidden_dim)
            subwords = [self.tokenizer.convert_ids_to_tokens(i) for i in sn_input_ids]

            samples = sample_fn(
                model=self.scandl_module,
                shape=sample_shape,
                noise=x_noised,
                sn_input_ids_emb=sn_input_ids_emb,
                pos_enc=pos_enc,
                mask_sn_padding=None,
                mask_transformer_att=None,
                clip_denoised=self.args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, self.args, self.sn_sp_repr_embedding),
                model_kwargs=None,
                top_p=self.args.top_p,
                clamp_step=self.args.clamp_step,
                clamp_first=self.args.clamp_first_bool,
                mask=mask,
                x_start=x_start,
                gap=step_gap,
            )
            sample = samples[-1]

            logits = self.scandl_module.get_logits(sample)
            cands = torch.topk(logits, k=1, dim=-1)

            for instance_idx, (pred_seq, orig_words, sn_len) in enumerate(
                zip(
                    cands.indices, words_for_mapping, sn_repr_len
                )
            ):
                pred_seq_sp = pred_seq[sn_len:]
                words_split = orig_words.split()
                predicted_sp = [words_split[i] for i in pred_seq_sp]
                pred_sp_ids = [e.item() for e in pred_seq_sp]

                # cut off trailing pad tokens 
                while len(predicted_sp) > 1 and predicted_sp[-1] == '[PAD]':
                    predicted_sp.pop()
                while len(pred_sp_ids) > 1 and pred_sp_ids[-1] == self.args.seq_len - 1:
                    pred_sp_ids.pop()
                while len(words_split) > 1 and words_split[-1] == '[PAD]':
                    words_split.pop()

                # remove CLS and SEP tokens from predictions
                if predicted_sp[0] == '[CLS]':
                    predicted_sp = predicted_sp[1:]
                    pred_sp_ids = pred_sp_ids[1:]
                if predicted_sp[-1] == '[SEP]':
                    predicted_sp = predicted_sp[:-1]
                    pred_sp_ids = pred_sp_ids[:-1]
                words_split = words_split[1:-1]

                # filter out erroneously predicted PAD tokens (they will raise an error in the fixdur module)
                pred_sp_ids, predicted_sp = self._remove_special_tokens(
                    predicted_sp_ids=pred_sp_ids,
                    predicted_sp_words=predicted_sp,
                    token='[PAD]',
                )
                pred_sp_ids, predicted_sp = self._remove_special_tokens(
                    predicted_sp_ids=pred_sp_ids,
                    predicted_sp_words=predicted_sp,
                    token='[CLS]',
                )
                pred_sp_ids, predicted_sp = self._remove_special_tokens(
                    predicted_sp_ids=pred_sp_ids,
                    predicted_sp_words=predicted_sp,
                    token='[SEP]',
                )

                predicted_sp_words.append(predicted_sp)
                predicted_sp_ids.append(pred_sp_ids)
                original_sn.append(words_split)

                idx_ctr += 1
                unique_idx.append(idx_ctr)

        predictions = {
            'predicted_sp_words': predicted_sp_words,
            'predicted_sp_ids': predicted_sp_ids,
            'original_sn': original_sn,
            'unique_idx': unique_idx,
        }
        return predictions

    
    def _remove_special_tokens(
        self,
        predicted_sp_ids: List[int],
        predicted_sp_words: List[str],
        token: str,  # '[CLS]' or '[SEP]' or '[PAD]'
    ):
        filtered_sp_ids, filtered_sp_words = [], []
        for sp_word, sp_id in zip(predicted_sp_words, predicted_sp_ids):
            if sp_word != token:
                filtered_sp_ids.append(sp_id)
                filtered_sp_words.append(sp_word)
        return filtered_sp_ids, filtered_sp_words


    def _preprocess_text(
        self,
        texts: Union[str, List[str]],
    ):
        data = {
            'mask': [],
            'sn_sp_repr': [],
            'sn_input_ids': [],
            'indices_pos_enc': [],
            'words_for_mapping': [],
            'sn_repr_len': [],
        }

        if isinstance(texts, str):
            texts = [texts]
        
        for sn_idx, sn in enumerate(texts):

            if sn.startswith('[CLS]') and sn.endswith('[SEP]'):
                sn = sn
            elif sn.startswith('[CLS]'):
                sn = sn + ' [SEP]'
            elif sn.endswith('[SEP]'):
                sn = '[CLS] ' + sn
            else:
                sn = '[CLS] ' + sn + ' [SEP]'
            
            encoded_sn = self.tokenizer.encode_plus(
                sn.split(),
                add_special_tokens=False,
                padding=False,
                return_attention_mask=False,
                is_split_into_words=True,
                truncation=False,
            )

            if len(encoded_sn) > self.args.seq_len / 2:
                print(f'Sentence {sn} is too long. Continue.')

            sn_word_ids = encoded_sn.word_ids()
            sn_input_ids = encoded_sn['input_ids']

            sn_sp_repr = sn_word_ids  

            mask = [0] * len(sn_word_ids)
            indices_pos_enc = list(range(0, len(sn_word_ids))) + list(range(0, self.args.seq_len - len(sn_word_ids)))
            words_for_mapping = sn.split() + (self.args.seq_len - len(sn.split())) * ['[PAD]']

            data['mask'].append(mask)
            data['sn_sp_repr'].append(sn_sp_repr)
            data['sn_input_ids'].append(sn_input_ids)
            data['indices_pos_enc'].append(indices_pos_enc)
            data['words_for_mapping'].append(' '.join(words_for_mapping))
            data['sn_repr_len'].append(len(sn_word_ids))

        # padding
        data['mask'] = _collate_batch_helper(
            examples=data['mask'],
            pad_token_id=1,
            max_length=self.args.seq_len,
        )
        data['sn_sp_repr'] = _collate_batch_helper(
            examples=data['sn_sp_repr'],
            pad_token_id=self.args.seq_len - 1,
            max_length=self.args.seq_len,
        )
        data['sn_input_ids'] = _collate_batch_helper(
            examples=data['sn_input_ids'],
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.args.seq_len,
        )

        split = 'inference'
        dataset = Dataset2.from_dict(data)
        dataset_dict = DatasetDict()
        dataset_dict[split] = dataset
        data_loader = text_dataset_loader(
            data=dataset_dict,
            data_args=self.args,
            split=split,
            deterministic=True,
        )
        return data_loader


    def _load_scandl_module(
        self,
        path_to_scandl_module: str,

    ):
        logger.log('### Loading ScanDL Diffusion Module ...')
        scandl_module, diffusion = create_model_and_diffusion(
            **args_to_dict(
                self.args, load_defaults_config(config_path=self.path_to_config).keys()
            )
        )
        # TODO Name scandl module, not model
        scandl_module.load_state_dict(
            dist_util.load_state_dict(
                os.path.join(self.path_to_scandl_module, 'ema_0.9999_080000.pt'),
                map_location='cpu'
            )
        )
        pytorch_total_params = sum(p.numel() for p in scandl_module.parameters())
        logger.log(f'### Total number of parameters: {pytorch_total_params}')
        scandl_module.eval().requires_grad_(False).to(dist_util.dev())
        return scandl_module, diffusion


    def _get_sn_sp_repr_emb(self):
        sn_sp_repr_embedding = nn.Embedding(
            num_embeddings=self.args.hidden_t_dim,
            embedding_dim=self.args.hidden_dim,
            _weight=self.scandl_module.sn_sp_repr_embedding.weight.clone().cpu(),
        )
        return sn_sp_repr_embedding


    def _get_args(self):
        args = self._get_parser().parse_args()
        # load the training arguments
        with open(os.path.join(self.path_to_scandl_module, 'training_args.json')) as f:
            training_args = json.load(f)
        training_args['batch_size'] = args.batch_size 
        args.__dict__.update(training_args)
        if args.clamp_first == 'yes':
            args.clamp_first_bool = True 
        else:
            args.clamp_first_bool = False 
        # TODO self.args.clamp_first_bool as argument
        # set mask_padding to False 
        args.mask_padding = False
        return args 
    

    def _load_tokenizer(self):
        tokenizer = BertTokenizerFast.from_pretrained(self.args.config_name)
        self.args.vocab_size = tokenizer.vocab_size
        return tokenizer


    def _get_parser(self) -> argparse.ArgumentParser:
        defaults = dict(
                model_path='',
                step=0,
                out_dir='',
                top_p=0,
                clamp_first='yes',
                test_set_sns='mixed',
                atten_vis=False,
                notes='-',
                tsne_vis=False,
                sp_vis=False,
                no_inst=0,
                atten_vis_sp=False,
                load_ids='-',
                load_test_data='-',
                setting='-',
                fold=0,
            )
        decode_defaults = dict(
            split='valid',
            clamp_step=0,
            seed2=105,
            clip_denoised=False,
        )

        defaults.update(load_defaults_config(config_path=self.path_to_config))
        defaults.update(decode_defaults)
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        
        return parser


class FixdurModule(nn.Module):
    
    def __init__(
        self,
        text_type: str,  # sentence, paragraph
        bsz: Optional[int] = 2,
    ):
        super(FixdurModule, self).__init__()

        base_path = 'scandl2_pkg'
        if text_type == 'paragraph':
            self.path_to_config = os.path.join(base_path, 'config_emtec.json')
            self.path_to_fixdur_module = PAR_FIXDUR_MODULE
        elif text_type == 'sentence':
            self.path_to_config = os.path.join(base_path, 'config.json')
            self.path_to_fixdur_module = SENT_FIXDUR_MODULE
        else:
            raise NotImplementedError(f'Text type {text_type} not implemented.')

        self.bsz = bsz
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = load_defaults_config(config_path=self.path_to_config)
        self.args = self._get_args(config=self.config)
        self.hyperparameters = self._get_hyperparams(path_to_fixdur_module=self.path_to_fixdur_module)
        
        # load GPT-2 model and tokenizer, and BERT embeddings 
        self.gpt2_model, self.tokenizer, self.bert_embeddings = self._load_gpt_and_bert(config=self.config)

        # load the Fixdur module and the MinMax Scaler
        self.fixdur_module = self._load_fixdur_module()
        self.scaler = self._load_scaler()

    
    def _get_args(self, config: Dict[str, Any]) -> Dict[str, Any]:
        args = {
            'max_length': config['seq_len'],
            'normalize': True,
            'output_attentions': False,
            'bsz': self.bsz,
            'corpus': config['corpus'],
            'sp_pad_token': config['seq_len'] - 1,
        }
        return args 


    def _get_hyperparams(self, path_to_fixdur_module: str) -> Dict[str, Any]:
        with open(os.path.join(path_to_fixdur_module, 'hyperparameters.json')) as f:
            return json.load(f)
    

    def _load_gpt_and_bert(self, config: Dict[str, Any]):
        """
        Load GPT-2 and GPT-2 tokenizer to get the contextualized embeddings.
        Load BERT model (for embeddings of CLS and PAD tokens)
        """
        # GPT-2
        gpt_config_name = config['gpt_config_name']
        tokenizer = GPT2TokenizerFast.from_pretrained(gpt_config_name, add_prefix_space=True)
        gpt2_model = GPT2Model.from_pretrained(gpt_config_name)
        tokenizer.pad_token = tokenizer.eos_token
        # freeze parameters 
        for param in gpt2_model.parameters():
            param.requires_grad = False
        
        # BERT
        bert_config_name = config['config_name']
        bert_embeddings = BertModel.from_pretrained(bert_config_name).embeddings.word_embeddings 
        # freeze parameters 
        for param in bert_embeddings.parameters():
            param.requires_grad = False
        
        return gpt2_model, tokenizer, bert_embeddings

    
    def _load_fixdur_module(self):
        fixdur_module_config = AutoConfig.from_pretrained('bert-base-cased')
        fixdur_module_config.num_attention_heads = self.hyperparameters['num_heads']
        fixdur_module_config.num_hidden_layers = self.hyperparameters['num_layers']
        fixdur_module = Seq2SeqModel(
            config=fixdur_module_config,
            output_dim=self.args['max_length'],
            num_linear=self.hyperparameters['num_linear'],
            dropout=self.hyperparameters['dropout'],
        )
        fixdur_module.load_state_dict(
            torch.load(
                os.path.join(self.path_to_fixdur_module, 'seq2seq_fixdur.pt')
            )
        )
        fixdur_module.eval()
        fixdur_module.to(self.device)
        return fixdur_module


    def _load_scaler(self):
        scaler = joblib.load(
            os.path.join(
                self.path_to_fixdur_module, 'min_max_scaler.pkl'
            )
        )
        return scaler
    

    def _prepare_data(
        self,  
        scandl_module_output: Dict[str, Union[List[List[str]], List[List[int]], List[str]]],
    ):
        data_dict = {
            'sp_embeddings': [],
            'attention_mask': [],
            'unique_idx': [],
        }
        for idx in range(len(scandl_module_output['predicted_sp_words'])):

            sn_words = scandl_module_output['original_sn'][idx]
            sp_ids = scandl_module_output['predicted_sp_ids'][idx]
            unique_id = scandl_module_output['unique_idx'][idx]

            # make the scanpath ids start at 0 for re-ordering of the embeddings 
            sp_ids = [i - 1 for i in sp_ids]

            sp_words = scandl_module_output['predicted_sp_words'][idx]

            # get the sentence encoding 
            sn_enc = self.tokenizer(
                sn_words,
                add_special_tokens=False,
                return_tensors='pt',
                is_split_into_words=True,
            )
            sn_word_ids = torch.Tensor(sn_enc.word_ids())

            # get the embeddings
            with torch.no_grad():
                last_hidden = self.gpt2_model(sn_enc.input_ids).last_hidden_state
            
            # aggregate the embeddings to word level 
            sn_embeddings = aggregate_input_embeddings(
                embeddings=last_hidden,
                word_ids=sn_word_ids,
                aggregate='mean',
            )

            # convert sp_ids to tensor
            sp_ids = torch.Tensor(sp_ids).long()

            # re-order the embeddings as scanpath 
            try:
                sp_embeddings = sn_embeddings[:, sp_ids, :]
            except:
                breakpoint()

            # pad the embeddings to max input length and get the attentino mask 
            sp_embeddings_padded, attention_mask = padding_and_mask_seq2seq(
                sp_embeddings=sp_embeddings,
                bert_embeddings=self.bert_embeddings,
                max_length=self.args['max_length'],
                inference=True,
            )

            data_dict['sp_embeddings'].append(sp_embeddings_padded)
            data_dict['attention_mask'].append(attention_mask)
            data_dict['unique_idx'].append(unique_id)

        return data_dict


    def forward(
        self,
        scandl_module_output: Dict[str, Union[List[List[str]], List[List[int]], List[str]]],
    ) -> Dict[str, Union[List[List[str]], List[List[int]], List[str], List[List[float]]]]:

        output_dict = {
            'predicted_sp_words': [],
            'predicted_sp_ids': [],
            'original_sn': [],
            'predicted_fix_durs': [],
            'unique_idx': [],
        }

        data_df = pd.DataFrame(scandl_module_output)

        data_dict = self._prepare_data(
            scandl_module_output=scandl_module_output,
        )
        dataset = FixdurDataset(data=data_dict)
        data_loader = DataLoader(
            dataset,
            batch_size=self.bsz,
            shuffle=False,
        )

        print('\t\t### FixDur Module generates fixation durations ...')
        for batch_idx, batch in tqdm(enumerate(data_loader)):

            sp_embeddings = batch['sp_embeddings'].squeeze(1).to(self.device)
            attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
            unique_indices = batch['unique_idx']
            

            out = self.fixdur_module(
                sp_embeddings=sp_embeddings,
                attention_mask=attention_mask,
                output_attentions=self.args['output_attentions'],
            )

            # scale the output back to the original range
            out_transformed = self.scaler.inverse_transform(out.detach().cpu().numpy())
            out_transformed_rounded = np.round(out_transformed, 2)

            # iterate over the individual predictions
            for out_idx, out_instance in enumerate(out_transformed_rounded):

                predicted_fix_durs = out_instance 
                unique_idx = unique_indices[out_idx].item()

                # find predicted_sp_words, predicted_sp_ids, and original_sn in data_df conditioned on unique_idx
                predicted_sp_words = data_df.loc[data_df['unique_idx'] == unique_idx, 'predicted_sp_words'].values[0]
                predicted_sp_ids = data_df.loc[data_df['unique_idx'] == unique_idx, 'predicted_sp_ids'].values[0]
                original_sn = data_df.loc[data_df['unique_idx'] == unique_idx, 'original_sn'].values[0]

                sp_len = len(predicted_sp_ids)

                # cut off the predicted_fix_durs to the length of the scanpath
                # the predicted fixation durations still contain predictions for the CLS and SEP token as well
                pred_fix_durs = predicted_fix_durs[:sp_len + 2].tolist()[1:-1]
                pred_fix_durs = [round(d, 2) for d in pred_fix_durs]

                # add to output_dict
                output_dict['predicted_sp_words'].append(predicted_sp_words)
                output_dict['predicted_sp_ids'].append(predicted_sp_ids)
                output_dict['original_sn'].append(original_sn)
                output_dict['predicted_fix_durs'].append(pred_fix_durs)
                output_dict['unique_idx'].append(unique_idx)

                print(f'fixdur original sn: {original_sn}')

        return output_dict
