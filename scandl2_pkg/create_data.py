"""
Create the data for training ScanDL on all data.
"""


import argparse
import os
import json
import numpy as np
import pandas as pd
import sys


from diffusion_only.scripts.sp_load_celer_zuco import load_celer, load_celer_speakers, process_celer
from diffusion_only.scripts.sp_load_celer_zuco import load_zuco, process_zuco, get_kfold, get_kfold_indices_combined
from diffusion_only.scripts.sp_load_celer_zuco import load_emtec, process_emtec
from diffusion_only.scripts.sp_load_celer_zuco import load_bsc, process_bsc
from diffusion_only.scripts.sp_load_celer_zuco import flatten_data, unflatten_data
from transformers import set_seed, BertTokenizerFast

sys.path.append('./')
sys.path.append('../')


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder-name',
        type=str,
        default='processed_data_all',
        help='Name of the folder to save the processed data in.',
    )
    parser.add_argument(
        '--max-fix-dur',
        type=int,
        help='max fixatino duration value. greater fixation durations are replaced with this value.',
        default=999,
    )
    parser.add_argument(
        '--data',
        type=str,
        choices=['celer', 'emtec', 'bsc'],
        required=True,
    )
    defaults = dict()
    defaults.update(load_defaults_config(parser.parse_args()))

    add_dict_to_argparser(parser, defaults)
    return parser


def load_defaults_config(args):
    """
    Load defaults for training args.
    """
    if args.data == 'emtec':
        config_name = 'config_emtec.json'
    elif args.data == 'bsc':
        config_name = 'config_bsc.json'
    else:
        config_name = 'config.json'
    with open(f'diffusion_only/scandl_diff_dur/{config_name}', 'r') as f:
        return json.load(f)


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def main():

    base_folder_name = 'scandl2_pkg'

    print('Loading argument parser...')
    args = create_argparser().parse_args()
    set_seed(args.seed)
    
    if args.data == 'celer':

        tokenizer = BertTokenizerFast.from_pretrained(args.config_name)
        data_path = args.folder_name + '_celer'
        if not os.path.exists(os.path.join(base_folder_name, data_path)):
            os.makedirs(os.path.join(base_folder_name, data_path))
        
        # load Celer data
        word_info_df, eyemovement_df = load_celer()
        reader_list = load_celer_speakers(only_native_speakers=args.celer_only_L1)
        sn_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()

        data, splitting_IDs_dict = process_celer(
            sn_list=sn_list,
            reader_list=reader_list,
            word_info_df=word_info_df,
            eyemovement_df=eyemovement_df,
            tokenizer=tokenizer,
            args=args,
            inference='cv',
            max_fix_dur=args.max_fix_dur,
        )
        flattened_data = flatten_data(data)
        flattened_data = np.array(flattened_data, dtype=object).tolist()
        train_data = unflatten_data(flattened_data=flattened_data, split='train')
        train_data.save_to_disk(os.path.join(base_folder_name, data_path))
    

    elif args.data == 'bsc':

        raise NotImplementedError('BSC data not implemented yet.')

    elif args.data == 'emtec':

        tokenizer = BertTokenizerFast.from_pretrained(args.config_name)
        data_path = args.folder_name + '_emtec'
        if not os.path.exists(os.path.join(base_folder_name, data_path)):
            os.makedirs(os.path.join(base_folder_name, data_path))
        
        # load EMTeC data
        print('Loading EMTeC data...')
        fixations_df, stimuli_df = load_emtec()
        data, splitting_IDs_dict = process_emtec(
            fixations_df=fixations_df,
            stimuli_df=stimuli_df,
            tokenizer=tokenizer,
            args=args,
            inference='cv',
            max_fix_dur=args.max_fix_dur,
        )
        flattened_data = flatten_data(data)
        flattened_data = np.array(flattened_data, dtype=object).tolist()
        train_data = unflatten_data(flattened_data=flattened_data, split='train')
        train_data.save_to_disk(os.path.join(base_folder_name, data_path))


    else:
        raise NotImplementedError('Data not implemented yet.')


if __name__ == '__main__':
    raise SystemExit(main())
