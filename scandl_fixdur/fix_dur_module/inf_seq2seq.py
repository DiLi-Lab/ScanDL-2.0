"""
Inference on the Fixation Duration Module.
"""



from argparse import ArgumentParser

import sys
import os
import json 
import joblib

import torch 

import numpy as np
import pandas as pd

from typing import Optional

from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Model, AutoConfig, BertModel 
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler

from scandl_fixdur.fix_dur_module.model_seq2seq import Seq2SeqModel
from scandl_fixdur.fix_dur_module.utils_data import prepare_seq2seq_data_hp, Seq2SeqDatasetHP



sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from CONSTANTS import SCANDL_MODULE_INF_PATH, FIXDUR_MODULE_TRAIN_PATH, FIXDUR_MODULE_INF_PATH
from CONSTANTS import SCANDL_MODULE_INF_PATH_EMTEC, FIXDUR_MODULE_TRAIN_PATH_EMTEC, FIXDUR_MODULE_INF_PATH_EMTEC
from CONSTANTS import SCANDL_MODULE_INF_PATH_BSC, FIXDUR_MODULE_TRAIN_PATH_BSC, FIXDUR_MODULE_INF_PATH_BSC



def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--setting',
        type=str,
        required=True,
        help='The evaluation setting.',
        choices=['reader', 'sentence', 'combined', 'cross_dataset'],
    )
    parser.add_argument(
        '--emtec',
        action='store_true',
        help='Whether to run inference on the EMTeC dataset.',
    )
    parser.add_argument(
        '--bsc',
        action='store_true',
        help='Whether to run inference on the BSC dataset.',
    )
    return parser 


def load_defaults_config(corpus: Optional[str] = None) -> Dict[str, Any]:
    """
    Load defaults for training args.
    """
    if corpus == 'emtec':
        config_name = 'config_emtec.json'
    elif corpus == 'bsc':
        config_name = 'config_bsc.json'
    else:
        config_name = 'config.json'
    with open(f'diffusion_only/scandl_diff_dur/{config_name}', 'r') as f:
        return json.load(f)


def batch_iteration(
    model: Seq2SeqModel,
    test_loader: DataLoader,
    scaler: StandardScaler,
    output_dict: Dict[str, List[Any]],
    scandl_output_df: pd.DataFrame,
    device: torch.device,
    setting: str,
    corpus: Optional[str] = None,
):
    for batch_idx, batch in tqdm(enumerate(test_loader)):

        sp_embeddings = batch['sp_embeddings'].to(device)
        attention_mask = batch['attention_masks'].to(device)
        sn_ids = batch['sn_ids']
        reader_ids = batch['reader_ids']
        predicted_sp_ids_batch = batch['predicted_sp_ids']
        original_fix_durs_batch = batch['original_fix_durs']

        out = model(
            sp_embeddings=sp_embeddings,
            attention_mask=attention_mask,
        )

        # scale the output back 
        out_transformed = scaler.inverse_transform(out.detach().cpu().numpy())
        out_transformed_rounded = np.round(out_transformed, 2)

        # iterate over the output batch 
        for out_idx, out_instance in enumerate(out_transformed_rounded):

            # cut off the pad tokens from the prediction 
            sp_len = len(eval(predicted_sp_ids_batch[out_idx]))
            predicted_fix_durs = out_instance[:sp_len].tolist()
            predicted_fix_durs = [round(fix_dur, 2) for fix_dur in predicted_fix_durs]

            # add the prediction to the output dictionary
            output_dict['predicted_fix_durs'].append(predicted_fix_durs) 

            if corpus == 'emtec':
                sn_id = sn_ids[out_idx]
                reader_id = reader_ids[out_idx]
            elif corpus == 'bsc':
                reader_id = reader_ids[out_idx].item()
                sn_id = sn_ids[out_idx].item()
            else:
                if setting == 'cross_dataset':
                    sn_id = sn_ids[out_idx].item()
                    reader_id = reader_ids[out_idx]
                else:
                    sn_id = sn_ids[out_idx]
                    reader_id = reader_ids[out_idx].item()

            predicted_sp_words = scandl_output_df.loc[(scandl_output_df['sn_ids'] == sn_id) & (scandl_output_df['reader_ids'] == reader_id), 'predicted_sp_words'].values[0]
            original_sp_words = scandl_output_df.loc[(scandl_output_df['sn_ids'] == sn_id) & (scandl_output_df['reader_ids'] == reader_id), 'original_sp_words'].values[0]
            predicted_sp_ids = scandl_output_df.loc[(scandl_output_df['sn_ids'] == sn_id) & (scandl_output_df['reader_ids'] == reader_id), 'predicted_sp_ids'].values[0]
            original_sp_ids = scandl_output_df.loc[(scandl_output_df['sn_ids'] == sn_id) & (scandl_output_df['reader_ids'] == reader_id), 'original_sp_ids'].values[0]
            original_sn = scandl_output_df.loc[(scandl_output_df['sn_ids'] == sn_id) & (scandl_output_df['reader_ids'] == reader_id), 'original_sn'].values[0]

            original_fix_durs = eval(original_fix_durs_batch[out_idx])

            # add to output dictionary 
            output_dict['predicted_sp_words'].append(predicted_sp_words)
            output_dict['original_sp_words'].append(original_sp_words)
            output_dict['predicted_sp_ids'].append(predicted_sp_ids)
            output_dict['original_sp_ids'].append(original_sp_ids)
            output_dict['original_fix_durs'].append(original_fix_durs)
            output_dict['original_sn'].append(original_sn)
            output_dict['sn_ids'].append(sn_id)
            output_dict['reader_ids'].append(reader_id)
            
    return output_dict


def run_inference(
    path_to_seq2seq_train: str,
    path_to_seq2seq_inf: str,
    path_to_scandl_orig: str,
    hyperparameters: Dict[str, Any],
    tokenizer: GPT2TokenizerFast,
    gpt2_model: GPT2Model,
    bert_embeddings: BertModel,
    max_length: int,
    device: torch.device,
    output_dict_keys: List[str],
    bsz: int,
    setting: str,
    sp_pad_token: int,
    corpus: Optional[str] = None,
):
    """
    Run inference on the Seq2Seq model.
    """
    # load the scandl predictions
    scandl_output_filename = next((s for s in os.listdir(path_to_scandl_orig) if s.endswith('PAD_rank0.json')), None)

    with open(os.path.join(path_to_scandl_orig, scandl_output_filename), 'r') as f:
        scandl_output = json.load(f)
    
    # convert to data frames for accessing values when saving the data 
    scandl_output_df = pd.DataFrame(scandl_output)

    # prepare the data for the Seq2Seq model
    data_dict = prepare_seq2seq_data_hp(
        scandl_output=scandl_output,
        tokenizer=tokenizer,
        gpt2_model=gpt2_model,
        bert_embeddings=bert_embeddings,
        max_length=max_length,
        sp_pad_token=sp_pad_token,
    )

    output_dict = {key: [] for key in output_dict_keys}
    
    num_heads = hyperparameters['num_heads']
    num_layers = hyperparameters['num_layers']
    num_linear = hyperparameters['num_linear']
    dropout = hyperparameters['dropout']
    use_attention_mask = hyperparameters['use_attention_mask']

    # load the seq2seq model and adapt the hyperparameters 
    config = AutoConfig.from_pretrained('bert-base-cased')
    config.num_attention_heads = num_heads
    config.num_hidden_layers = num_layers

    # initialise model 
    model = Seq2SeqModel(
        config=config,
        output_dim=max_length,
        num_linear=num_linear,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(os.path.join(path_to_seq2seq_train, 'seq2seq_fixdur.pt')))
    model.eval()
    model.to(device)

    # load scaler
    scaler = joblib.load(os.path.join(path_to_seq2seq_train, 'min_max_scaler.pkl'))

    # run inference on ScanDL's PAD output 
    test_dataset = Seq2SeqDatasetHP(data_dict)
    test_loader = DataLoader(
        test_dataset,
        batch_size=bsz,
        shuffle=False,
    )
    output_dict = batch_iteration(
        model=model,
        test_loader=test_loader,
        scaler=scaler,
        output_dict=output_dict,
        scandl_output_df=scandl_output_df,
        device=device,
        setting=setting,
        corpus=corpus,
    )

    
    # save the output 
    with open(os.path.join(path_to_seq2seq_inf, 'output_dict.json'), 'w') as f:
        json.dump(output_dict, f)
    

def main(): 

    args = get_parser().parse_args()
    setting = args.setting

    np.set_printoptions(suppress=True)

    config = load_defaults_config(corpus='emtec' if args.emtec else 'bsc' if args.bsc else None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_length = config['seq_len']
    normalize = True 
    output_attentions = False 
    bsz = 6 if args.emtec else 12
    corpus = config['corpus']

    # load GPT-2 and GPT-2 tokenizer to get the contextualized embeddings
    gpt_config_name = config['gpt_config_name']
    tokenizer = GPT2TokenizerFast.from_pretrained(gpt_config_name, add_prefix_space=True)
    gpt2_model = GPT2Model.from_pretrained(gpt_config_name)
    tokenizer.pad_token = tokenizer.eos_token
    # freeze parameters
    for param in gpt2_model.parameters():
        param.requires_grad = False

    # load BERT model (for embeddings of CLS and PAD tokens)   
    bert_config_name = config['config_name']
    bert_embeddings = BertModel.from_pretrained(bert_config_name).embeddings.word_embeddings
    # freeze parameters
    for param in bert_embeddings.parameters():
        param.requires_grad = False

    output_dict_keys = ['predicted_sp_words', 'original_sp_words', 'predicted_sp_ids', 'original_sp_ids', 'predicted_fix_durs', 'original_fix_durs', 'original_sn', 'sn_ids', 'reader_ids']

    # paths 
    if args.emtec:
        fixdur_module_train = FIXDUR_MODULE_TRAIN_PATH_EMTEC
        fixdur_module_inf = FIXDUR_MODULE_INF_PATH_EMTEC
        scandl_module_inf = SCANDL_MODULE_INF_PATH_EMTEC
    elif args.bsc:
        fixdur_module_train = FIXDUR_MODULE_TRAIN_PATH_BSC
        fixdur_module_inf = FIXDUR_MODULE_INF_PATH_BSC
        scandl_module_inf = SCANDL_MODULE_INF_PATH_BSC
    else:
        fixdur_module_train = FIXDUR_MODULE_TRAIN_PATH
        fixdur_module_inf = FIXDUR_MODULE_INF_PATH
        scandl_module_inf = SCANDL_MODULE_INF_PATH

    sp_pad_token = max_length - 1

    path_to_hyperparameters = os.path.join(fixdur_module_train, setting, 'hyperparameters.json')
    with open(path_to_hyperparameters, 'r') as f:
        hyperparameters = json.load(f)

    if setting == 'cross_dataset':
        path_to_seq2seq_train = os.path.join(fixdur_module_train, setting)
        path_to_seq2seq_inf = os.path.join(fixdur_module_inf, setting)
        path_to_scandl_orig = os.path.join(scandl_module_inf, setting)
        if not os.path.exists(path_to_seq2seq_inf):
            os.makedirs(path_to_seq2seq_inf)

        print(f'--- running inference on setting {setting} ---')
        # run inference 
        run_inference(
            path_to_seq2seq_train=path_to_seq2seq_train,
            path_to_seq2seq_inf=path_to_seq2seq_inf,
            path_to_scandl_orig=path_to_scandl_orig,
            hyperparameters=hyperparameters,
            tokenizer=tokenizer,
            gpt2_model=gpt2_model,
            bert_embeddings=bert_embeddings,
            max_length=max_length,
            device=device,
            output_dict_keys=output_dict_keys,
            bsz=bsz,
            setting=setting,
            sp_pad_token=sp_pad_token,
            corpus=corpus,
        )
        
    else:

        for fold_idx in range(5):
            path_to_seq2seq_train = os.path.join(fixdur_module_train, setting, f'fold-{fold_idx}')
            path_to_seq2seq_inf = os.path.join(fixdur_module_inf, setting, f'fold-{fold_idx}')
            path_to_scandl_orig = os.path.join(scandl_module_inf, setting, f'fold-{fold_idx}')
            if not os.path.exists(path_to_seq2seq_inf):
                os.makedirs(path_to_seq2seq_inf)
            
            print(f'--- running inference on setting {setting} and fold {fold_idx} ---')
            # run inference
            run_inference(
                path_to_seq2seq_train=path_to_seq2seq_train,
                path_to_seq2seq_inf=path_to_seq2seq_inf,
                path_to_scandl_orig=path_to_scandl_orig,
                hyperparameters=hyperparameters,
                tokenizer=tokenizer,
                gpt2_model=gpt2_model,
                bert_embeddings=bert_embeddings,
                max_length=max_length,
                device=device,
                output_dict_keys=output_dict_keys,
                bsz=bsz,
                setting=setting,
                sp_pad_token=sp_pad_token,
                corpus=corpus,
            )


if __name__ == '__main__':
    raise SystemExit(main())
