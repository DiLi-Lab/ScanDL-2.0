
import os
import yaml
import numpy as np
import pandas as pd 
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Optional


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--config-name',
        type=str,
        required=False,
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
    parser.add_argument(
        '--baseline',
        type=str,
        required=False,
        choices=['ez_reader', 'swift'],
        help='whether we are looking at baseline data',
    )
    return parser



def compute_reading_measures(
    fixations_df: pd.DataFrame,
    words_df: pd.DataFrame,
    baseline: Optional[str] = None,
    setting: Optional[str] = None,
):

    # append one extra dummy fixation to have the next fixation for the actual last fixation
    fixations_df = pd.concat(
        [
            fixations_df,
            pd.DataFrame(
                [[0 for _ in range(len(fixations_df.columns))]], columns=fixations_df.columns,
            ),
        ],
        ignore_index=True,
    )

    text_aois = words_df['word_id'].tolist()
    text_strs = words_df['word'].tolist()

    # iterate over the words in that text 
    word_dict = dict() 
    for word_index, word in zip(text_aois, text_strs):
        word_row = {
            'word': word,
            'word_index': word_index,
            'FFD': 0,       # first-fixation duration
            'SFD': 0,       # single-fixation duration
            'FD': 0,        # first duration
            'FPRT': 0,      # first-pass reading time
            'FRT': 0,       # first-reading time
            'TFT': 0,       # total-fixation time
            'RRT': 0,       # re-reading time
            'RPD_inc': 0,   # inclusive regression-path duration
            'RPD_exc': 0,   # exclusive regression-path duration
            'RBRT': 0,      # right-bounded reading time
            'Fix': 0,       # fixation (binary)
            'FPF': 0,       # first-pass fixation (binary)
            'RR': 0,        # re-reading (binary)
            'FPReg': 0,     # first-pass regression (binary)
            'TRC_out': 0,   # total count of outgoing regressions
            'TRC_in': 0,    # total count of incoming regressions
            'SL_in': 0,     # incoming saccade length
            'SL_out': 0,    # outgoing saccade length
            'TFC': 0,       # total fixation count
        }

        word_dict[int(word_index)] = word_row

        right_most_word, cur_fix_word_idx, next_fix_word_idx, next_fix_dur = -1, -1, -1, -1

    # TODO what to do with negative fixation durations (wrongly predicted by the model)?

    for index, fixation in fixations_df.iterrows():

        aoi = int(fixation['word_id'])

        # update variables 
        last_fix_word_idx = cur_fix_word_idx 

        cur_fix_word_idx = next_fix_word_idx 
        cur_fix_dur = next_fix_dur 

        next_fix_word_idx = aoi 
        next_fix_dur = fixation['fixation_duration']

        # the 0 that we added as dummy fixation at the end of the fixations df 
        if next_fix_dur == 0:
            # we set the idx to the idx of the actual last fixation such that there is no error later 
            next_fix_word_idx = cur_fix_word_idx

        if right_most_word < cur_fix_word_idx:
            right_most_word = cur_fix_word_idx 
        
        if cur_fix_word_idx == -1:
            continue
        
        try:
            word_dict[cur_fix_word_idx]['TFT'] += int(cur_fix_dur)
        except:
            breakpoint()

        word_dict[cur_fix_word_idx]['TFC'] += 1

        if word_dict[cur_fix_word_idx]['FD'] == 0:
            word_dict[cur_fix_word_idx]['FD'] += int(cur_fix_dur)
        
        if right_most_word == cur_fix_word_idx:
            if word_dict[cur_fix_word_idx]['TRC_out'] == 0:
                word_dict[cur_fix_word_idx]['FPRT'] += int(cur_fix_dur)
                if last_fix_word_idx < cur_fix_word_idx:
                    word_dict[cur_fix_word_idx]['FFD'] += int(cur_fix_dur)
        else:
            if right_most_word < cur_fix_word_idx:
                print('error')
            word_dict[right_most_word]['RPD_exc'] += int(cur_fix_dur)

        if cur_fix_word_idx < last_fix_word_idx:
            word_dict[cur_fix_word_idx]['TRC_in'] += 1
        if cur_fix_word_idx > next_fix_word_idx:
            word_dict[cur_fix_word_idx]['TRC_out'] += 1
        if cur_fix_word_idx == right_most_word:
            word_dict[cur_fix_word_idx]['RBRT'] += int(cur_fix_dur)
        if word_dict[cur_fix_word_idx]['FRT'] == 0 and (
            not next_fix_word_idx == cur_fix_word_idx or next_fix_dur == 0
        ):
            word_dict[cur_fix_word_idx]['FRT'] = word_dict[cur_fix_word_idx]['TFT']
        if word_dict[cur_fix_word_idx]['SL_in'] == 0:
            word_dict[cur_fix_word_idx]['SL_in'] = cur_fix_word_idx - last_fix_word_idx
        if word_dict[cur_fix_word_idx]['SL_out'] == 0:
            word_dict[cur_fix_word_idx]['SL_out'] = next_fix_word_idx - cur_fix_word_idx

    # Compute the remaining reading measures from the ones computed above
    for word_indices, word_rm in sorted(word_dict.items()):
        if word_rm['FFD'] == word_rm['FPRT']:
            word_rm['SFD'] = word_rm['FFD']
        word_rm['RRT'] = word_rm['TFT'] - word_rm['FPRT']
        word_rm['FPF'] = int(word_rm['FFD'] > 0)
        word_rm['RR'] = int(word_rm['RRT'] > 0)
        word_rm['FPReg'] = int(word_rm['RPD_exc'] > 0)
        word_rm['Fix'] = int(word_rm['TFT'] > 0)
        word_rm['RPD_inc'] = word_rm['RPD_exc'] + word_rm['RBRT']

        # if it is the first word, we create the df (index of first word is 0)
        if word_indices == 0:
            rm_df = pd.DataFrame([word_rm])
            
        else:
            rm_df = pd.concat([rm_df, pd.DataFrame([word_rm])])

    # if baseline == 'ez_reader':
    #     if not setting == 'cross_dataset':
    #         reader_id = fixations_df['reader_id'][0]
    #         rm_df['reader_id'] = reader_id

    sn_id = fixations_df['sn_id'][0]
    instance_idx = fixations_df['instance_idx'][0]
    
    rm_df['sn_id'] = sn_id
    rm_df['instance_idx'] = instance_idx
    
    return rm_df




def main():

    parser = get_parser()
    args = parser.parse_args()

    # open the constants file
    with open('CONSTANTS_ANALYSES.yaml', 'r') as f:
        constants = yaml.safe_load(f)


    if args.baseline:

        if args.baseline == 'ez_reader':
            rm_filename = 'reading_measures-ez_reader.csv'
            fix_df_filename = 'fixations_df-baselines-ez_reader.csv'
            orig_texts_filename = 'orig_text_df-baselines.csv'
            path_to_settings = constants['path_to_baseline_data']
        elif args.baseline == 'swift':
            rm_filename = 'reading_measures-swift.csv'
            fix_df_filename = 'fixations_df-baselines-swift.csv'
            orig_texts_filename = 'orig_text_df-baselines.csv'
            path_to_settings = constants['path_to_baseline_data']
    else:
        if not args.config_name:
            raise ValueError('Please specify the config name.')
        if args.fixdur_module:
            rm_filename = 'reading_measures-scandl_fixdur.csv'
            fix_df_filename = 'fixations_df-scandl_fixdur.csv'
            orig_texts_filename = 'orig_text_df-scandl_fixdur.csv'
        elif args.human:
            rm_filename = 'reading_measures-human.csv'
            fix_df_filename = 'fixations_df-human.csv'
            orig_texts_filename = 'orig_text_df-human.csv'
        path_to_settings = os.path.join(constants['path_to_seq2seq_data'], args.config_name)

    # if args.fixdur_module:
    #     if not args.config_name:
    #         raise ValueError('Please specify the config name for the fixdur module data.')
    # elif args.human:
    #     if not args.config_name:
    #         raise ValueError('Please specify the config name for the human data.')
    
    # path_to_seq2seq_data = constants['path_to_seq2seq_data']
    # path_to_baseline_data = constants['path_to_baseline_data']
    # config = args.config_name

    #configs = [c for c in os.listdir(path_to_seq2seq_data) if c.startswith('config')]
    settings = ['reader', 'sentence', 'cross_dataset', 'combined']

    # if args.fixdur_module:
    #     rm_filename = 'reading_measures-scandl_fixdur.csv'
    #     fix_df_filename = 'fixations_df-scandl_fixdur.csv'
    #     orig_texts_filename = 'orig_text_df-scandl_fixdur.csv'
    # elif args.human:
    #     rm_filename = 'reading_measures-human.csv'
    #     fix_df_filename = 'fixations_df-human.csv'
    #     orig_texts_filename = 'orig_text_df-human.csv'
    # elif args.ez_reader:
    #     rm_filename = 'reading_measures-ez_reader.csv'
    #     fix_df_filename = 'fixations_df-baselines-ez_reader'
    #     orig_texts_filename = 'orig_text_df-baselines.csv'
    # else:
    #     raise NotImplementedError('Please specify whether the data is from the fixdur module or human data.')


    for setting in settings:
        # if setting == 'cross_dataset':
        #     continue
        all_fixations_df = pd.read_csv(os.path.join(path_to_settings, setting, fix_df_filename), sep='\t')
        all_words_df = pd.read_csv(os.path.join(path_to_settings, setting, orig_texts_filename), sep='\t')

        # word indexing starts at 0 for the original words but at 1 for EZ reader and swift predictions
        #all_words_df['word_id'] = all_words_df['word_id'] + 1
        all_fixations_df['word_id'] = all_fixations_df['word_id'] - 1

        # if ez-reader: replace predicted 'inf' values with 2000
        if args.baseline == 'ez_reader':
            all_fixations_df['fixation_duration'].replace([np.inf, -np.inf], 2000, inplace=True)
            all_fixations_df['fixation_duration'].fillna(2000, inplace=True)

        print(f'Computing reading measures for {setting}...')

        for idx, fixations_df in tqdm(all_fixations_df.groupby('instance_idx')):
            instance_idx = fixations_df['instance_idx'].unique().item()
            words_df = all_words_df[all_words_df['instance_idx'] == instance_idx]
            #breakpoint()

            

            rm_df = compute_reading_measures(fixations_df, words_df, baseline=args.baseline)

            if idx == 0:
                all_rm_df = rm_df
            else:
                all_rm_df = pd.concat([all_rm_df, rm_df])
            
        all_rm_df.to_csv(os.path.join(path_to_settings, setting, rm_filename), sep='\t', index=False)

    
    # all_fixations_df = pd.read_csv('fixations_df.csv', sep='\t')
    # all_words_df = pd.read_csv('orig_text_df.csv', sep='\t')

    # for idx, fixations_df in tqdm(all_fixations_df.groupby('instance_idx')):
    #     instance_idx = fixations_df['instance_idx'].unique().item()
    #     words_df = all_words_df[all_words_df['instance_idx'] == instance_idx]

    #     rm_df = compute_reading_measures(fixations_df, words_df)

    #     if idx == 0:
    #         all_rm_df = rm_df
    #     else:
    #         all_rm_df = pd.concat([all_rm_df, rm_df])

    # all_rm_df.to_csv('reading_measures.csv', sep='\t', index=False)


if __name__ == '__main__':
    raise SystemExit(main())
