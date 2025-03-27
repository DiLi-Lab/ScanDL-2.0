"""
The training script for training the fixation duration module.
"""

import joblib
import sys 
import json 
import os
from typing import Dict
from argparse import ArgumentParser

from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Model, AutoConfig, BertModel 
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, DatasetDict
from sklearn.preprocessing import MinMaxScaler

from scandl2_pkg.fix_dur_module.utils_data import prepare_seq2seq_data, get_embeddings_seq2seq, Seq2SeqDataset, split_train_val_data
from scandl2_pkg.fix_dur_module.model_seq2seq import Seq2SeqModel
from scandl2_pkg.fix_dur_module.utils_train import EarlyStopping, train

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from CONSTANTS import COMPLETE_FIXDUR_MODULE_TRAIN_PATH_BSC, COMPLETE_FIXDUR_MODULE_TRAIN_PATH_CELER, COMPLETE_FIXDUR_MODULE_TRAIN_PATH_EMTEC


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='The maximum sequence length.',
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=12,
        help='The number of attention heads in the Transformer encoder.',
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=12,
        help='The number of layers in the Transformer encoder.',
    )
    parser.add_argument(
        '--num-linear',
        type=int,
        default=8,
        help='The number of linear layers.',
    )
    parser.add_argument(
        '--bsz',
        type=int,
        default=128,
        help='The batch size.',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='The dropout rate.',
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=400,
    )
    parser.add_argument(
        '--sp-pad-token',
        type=int,
        default=127,
        help='the padding token appended to the sp, usually seq_len-1',
    )
    parser.add_argument(
        '--use-attention-mask',
        action='store_true',
        help='Whether to use the attention mask in the Transformer encoder.',
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        choices=['emtec', 'bsc', 'celer'],
        help='The dataset to train on.',
    )
    return parser


def main():


    args = get_parser().parse_args()

    max_length = args.max_length
    output_attentions = False
    learning_rate = 1e-4
    num_epochs = args.num_epochs
    patience = 25
    normalize = True 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.data == 'emtec':
        path_save_model = COMPLETE_FIXDUR_MODULE_TRAIN_PATH_EMTEC
        path_to_data = 'processed_data_all_emtec'
    elif args.data == 'bsc':
        raise NotImplementedError('Training on BSC data is not yet implemented.')
        path_save_model = COMPLETE_FIXDUR_MODULE_TRAIN_PATH_BSC
        path_to_data = 'processed_data_all_bsc'
    elif args.data == 'celer':
        path_save_model = COMPLETE_FIXDUR_MODULE_TRAIN_PATH_CELER
        path_to_data = 'processed_data_all_celer'
    else:
        raise ValueError('Unknown dataset.')
    
    if not os.path.exists(path_save_model):
        os.makedirs(path_save_model)
    model_name = 'seq2seq_fixdur.pt'

    hypeparameters = {
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'num_linear': args.num_linear,
        'bsz': args.bsz,
        'dropout': args.dropout,
        'use_attention_mask': args.use_attention_mask,
    }
    with open(os.path.join(path_save_model, 'hyperparameters.json'), 'w') as f:
        json.dump(hypeparameters, f)

    # load GPT-2 and GPT-2 tokenizer to get the contextualized embeddings
    if args.data == 'bsc':
        raise NotImplementedError('Training on BSC data is not yet implemented.')
        gpt_config_name = 'benjamin/gpt2-wechsel-chinese'
    else:    
        gpt_config_name = 'gpt2'
    
    tokenizer = GPT2TokenizerFast.from_pretrained(gpt_config_name, add_prefix_space=True)
    gpt2_model = GPT2Model.from_pretrained(gpt_config_name)
    tokenizer.pad_token = tokenizer.eos_token
    # freeze parameters
    for param in gpt2_model.parameters():
        param.requires_grad = False

    # load BERT config (for model architecture) and BERT model (for embeddings of CLS and PAD tokens)
    
    if args.data == 'bsc':
        raise NotImplementedError('Training on BSC data is not yet implemented.')
        bert_config_name = 'bert-base-chinese'
    else:
        bert_config_name = 'bert-base-cased'
    config = AutoConfig.from_pretrained(bert_config_name)
    bert_embeddings = BertModel.from_pretrained(bert_config_name).embeddings.word_embeddings
    # freeze parameters
    for param in bert_embeddings.parameters():
        param.requires_grad = False

    # change the parameters in the config
    config.num_attention_heads = args.num_heads
    config.num_hidden_layers = args.num_layers


    # training 
    print('--- load and prepare data ...')
    train_data = load_from_disk(os.path.join('scandl2_pkg', path_to_data, 'train'))
    new_data = DatasetDict()
    new_data['train'] = train_data


    # prepare the data for training
    data = prepare_seq2seq_data(
        data=new_data,
        tokenizer=tokenizer,
        gpt2_model=gpt2_model,
        bert_embeddings=bert_embeddings,
        aggregate='mean',
        max_length=max_length,
        sp_pad_token=args.sp_pad_token,
    )

    fix_dur_colname = 'fix_durs'

    if normalize:
        min_max_scaler = MinMaxScaler()
        fix_durs = [t.cpu().detach().numpy() for t in data['fix_durs']]
        flattened = np.concatenate(fix_durs).reshape(-1, 1)
        # fit the scaler on the training data
        min_max_scaler.fit(flattened)
        # normalize the fixation durations
        flattened_normalized = min_max_scaler.transform(flattened)
        # reshape 
        split_indices = [len(t) for t in fix_durs]
        normalized_data = np.split(flattened_normalized.flatten(), np.cumsum(split_indices)[:-1])
        # convert back to tensors 
        normalized_tensors = [torch.tensor(t) for t in normalized_data]
        data['fix_durs_normalized'] = normalized_tensors
        # save the scaler (needed for inference)
        joblib.dump(min_max_scaler, os.path.join(path_save_model, 'min_max_scaler.pkl'))
        fix_dur_colname = 'fix_durs_normalized'
    
    # split data into train and val data (val data for early stopping)
    train_data, val_data = split_train_val_data(
        data=data,
        val_size=0.1,
    )

    # create dataset and dataloader
    train_dataset = Seq2SeqDataset(
        data=train_data,
        normalize=normalize,
    )
    val_dataset = Seq2SeqDataset(
        data=val_data,
        normalize=normalize,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bsz,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bsz,
        shuffle=False,
    )

    # model, loss, optimizer, scheduler, early stopping

    model = Seq2SeqModel(
        config=config, 
        output_dim=max_length,
        num_linear=args.num_linear, 
        dropout=args.dropout,
    )
    model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(
        patience=patience, 
        path=os.path.join(path_save_model, model_name),
    )

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.05 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # training 
    train(
        model=model,
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        early_stopping=early_stopping,
        scheduler=scheduler,
        device=device,
        fix_dur_colname=fix_dur_colname,
        output_attentions=output_attentions,
        use_attention_mask=args.use_attention_mask,
    )



if __name__ == '__main__':
    raise SystemExit(main())
