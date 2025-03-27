


import os
import json 
import yaml

import pandas as pd 

from argparse import ArgumentParser

from typing import List, Dict, Any, Optional


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--config-name',
        type=str,
        required=True,
        help='name of the model config',
    )
    parser.add_argument(
        '--fixdur-module',
        action='store_true',
        help='whether we are looking at the output of the separate fixdur module',
    )
    parser.add_argument(
        '--human',
        action='store_true',
        help='whether we are looking at human data',
    )
    return parser


def convert_file(
        data: Dict[str, List[Any]],
        setting: str,
        sp_words_colname: str,
        sp_ids_colname: str,
        fix_durs_colname: str,
        fold: Optional[str] = None,
    ) -> pd.DataFrame:
    """ convert the fixations output into a dataframe, and create dataframe to hold the sentence information. """

    fixations_dict = {
        'sn_id': [],
        'reader_id': [],
        'instance_idx': [],
        'fixation_index': [],
        'fixation_duration': [],
        'word_id': [],
        'word': []
    }

    orig_text_dict = {
        'sn_id': [],
        'reader_id': [],
        'instance_idx': [],
        'word_id': [],
        'word': []
    }

    for i in range(len(data['predicted_sp_words'])):

        # remove CLS and SEP tokens 
        sp_words = data[sp_words_colname][i].split()
        sp_ids = data[sp_ids_colname][i]
        fix_durs = data[fix_durs_colname][i]

        if len(fix_durs) == len(sp_words):
            sp_words = sp_words[1:-1]
            sp_ids = sp_ids[1:-1]
            fix_durs = fix_durs[1:-1]

        elif len(fix_durs) == len(sp_words) + 1:
            sp_words = sp_words[1:]
            sp_ids = sp_ids[1:-1]
            fix_durs = fix_durs[1:-1]
        
        else:
            print(i, '\t', sp_words, fix_durs) #f"Length mismatch: {len(sp_words)}, {len(sp_ids)}, {len(fix_durs)}"
            continue

        # make sp_ids start from 0 
        sp_ids = [word_id - 1 for word_id in sp_ids]

        try:
            assert len(sp_words) == len(sp_ids) == len(fix_durs)
        except:
            breakpoint()
            print(i, '\t', sp_words, fix_durs) #f"Length mismatch: {len(sp_words)}, {len(sp_ids)}, {len(fix_durs)}"
            continue

        sp_len = len(sp_words)

        fixation_indices = [idx + 1 for idx in range(sp_len)]

        sn_id = [data['sn_ids'][i]] * sp_len
        reader_id = [data['reader_ids'][i]] * sp_len
        instance_idx = [i] * sp_len

        fixations_dict['sn_id'].extend(sn_id)
        fixations_dict['reader_id'].extend(reader_id)
        fixations_dict['instance_idx'].extend(instance_idx)
        fixations_dict['fixation_index'].extend(fixation_indices)
        fixations_dict['fixation_duration'].extend(fix_durs)
        fixations_dict['word_id'].extend(sp_ids)
        fixations_dict['word'].extend(sp_words)

        orig_words = data['original_sn'][i].split()[1:-1]
        sn_len = len(orig_words)
        orig_word_ids = list(range(sn_len))
        sn_id_words = [data['sn_ids'][i]] * sn_len
        reader_id_words = [data['reader_ids'][i]] * sn_len
        instance_idx_words = [i] * sn_len

        orig_text_dict['sn_id'].extend(sn_id_words)
        orig_text_dict['reader_id'].extend(reader_id_words)
        orig_text_dict['instance_idx'].extend(instance_idx_words)
        orig_text_dict['word_id'].extend(orig_word_ids)
        orig_text_dict['word'].extend(orig_words)

    fixations_df = pd.DataFrame(fixations_dict)
    orig_text_df = pd.DataFrame(orig_text_dict)


    return fixations_df, orig_text_df



def merge_json_files(file_paths: List[str]):

    for file_idx, file_path in enumerate(file_paths):
        with open(file_path, 'r') as f:
            data = json.load(f)

        if file_idx == 0:
            merged_data = data
        else:
            for key in data.keys():
                merged_data[key].extend(data[key])
        
    return merged_data


def main():

    # open the constants file
    with open('CONSTANTS_ANALYSES.yaml', 'r') as f:
        constants = yaml.safe_load(f)
    
    path_to_seq2seq_data = constants['path_to_seq2seq_data']

    args = get_parser().parse_args()

    config_name = args.config_name
    fixdur_module = args.fixdur_module
    human = args.human

    if fixdur_module:
        sp_words_colname = 'predicted_sp_words'
        sp_ids_colname = 'predicted_sp_ids'
        fix_durs_colname = 'predicted_fix_durs'
        fix_df_filename = 'fixations_df-scandl_fixdur.csv'
        orig_texts_filename = 'orig_text_df-scandl_fixdur.csv'
    elif human:
        sp_words_colname = 'original_sp_words'
        sp_ids_colname = 'original_sp_ids'
        fix_durs_colname = 'original_fix_durs'
        fix_df_filename = 'fixations_df-human.csv'
        orig_texts_filename = 'orig_text_df-human.csv'
    else:
        raise NotImplementedError('Please specify whether the data is from the fixdur module or human data.')

    path_to_settings = os.path.join(path_to_seq2seq_data, config_name)
    settings = ['reader', 'sentence', 'combined', 'cross_dataset']

    for setting in settings:
        
        if setting == 'cross_dataset':
            
            with open(os.path.join(path_to_settings, setting, 'output_dict_pad.json'), 'r') as f:
                data = json.load(f)

            
            fixations_df, orig_text_df = convert_file(
                data=data, 
                setting=setting,
                sp_words_colname=sp_words_colname,
                sp_ids_colname=sp_ids_colname,
                fix_durs_colname=fix_durs_colname,
            )
            # remove predicted PAD tokens that the model predicts for cross-dataset setting
            fixations_df = fixations_df[fixations_df['word'] != '[PAD]']
            # and stuck SEP tokens 
            fixations_df = fixations_df[fixations_df['word'] != '[SEP]']
            fixations_df.to_csv(os.path.join(path_to_seq2seq_data, config_name, setting, fix_df_filename), sep='\t', index=False)
            orig_text_df.to_csv(os.path.join(path_to_seq2seq_data, config_name, setting, orig_texts_filename), sep='\t', index=False)

            
        
        else:

            folds = os.listdir(os.path.join(path_to_settings, setting))
            paths = [os.path.join(path_to_settings, setting, fold, 'output_dict_pad.json') for fold in folds if fold.startswith('fold')]
            merged_data = merge_json_files(paths)
            fixations_df, orig_text_df = convert_file(
                data=merged_data,
                setting=setting,
                sp_words_colname=sp_words_colname,
                sp_ids_colname=sp_ids_colname,
                fix_durs_colname=fix_durs_colname,
                )
            fixations_df.to_csv(os.path.join(path_to_seq2seq_data, config_name, setting, fix_df_filename), sep='\t', index=False)
            orig_text_df.to_csv(os.path.join(path_to_seq2seq_data, config_name, setting, orig_texts_filename), sep='\t', index=False)
                



if __name__ == '__main__':
    raise SystemExit(main())

