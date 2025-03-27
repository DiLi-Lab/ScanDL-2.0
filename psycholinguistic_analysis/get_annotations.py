
import os
from tqdm import tqdm
import numpy as np
import pandas as pd 
from SurprisalScorer import SurprisalScorer
from wordfreq import zipf_frequency

from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--config-name',
        type=str,
        help='the fixdur config name',
        default='config134_seq2seq_fixdur-normalize-heads_12-layers_12-linear_8-dropout_0.5-att_mask_True',
    )
    return parser


def main(): 

    args = get_parser().parse_args()
    config_name = args.config_name

    rms_scandl_filename = 'reading_measures-scandl_fixdur.csv'
    rms_human_filename = 'reading_measures-human.csv'

    path_to_config = '../generation_outputs/fixdur_module'
    path_to_settings = os.path.join(path_to_config, config_name)

    settings = ['reader', 'sentence', 'combined', 'cross_dataset']

    scorer = SurprisalScorer(model_name='gpt2')

    all_rms_dfs = list()

    

    
    # add ez-reader data

    rms_ez_reader_filename = 'reading_measures-ez_reader.csv'
    rms_swift_filename = 'reading_measures-swift.csv'
    path_to_baseline_settings = '../docs/baseline_outputs'

    for idx, setting in enumerate(settings):
        
        # # TODO cross_dataset setting
        # if setting == 'cross_dataset':
        #     continue

        path_to_rms = os.path.join(path_to_baseline_settings, setting)
        
        print(f'--- {setting} ez_reader ---')
        rms_ez_reader = pd.read_csv(os.path.join(path_to_rms, rms_ez_reader_filename), sep='\t')
        surprisal_list_ez_reader = list()
        for idx, group in tqdm(rms_ez_reader.groupby('instance_idx')):
            text = ' '.join(group['word'].tolist())
            _, surprisal, _ = scorer.score(text, BOS=True)
            surprisal_list_ez_reader += surprisal.tolist()
        rms_ez_reader['surprisal'] = surprisal_list_ez_reader
        rms_ez_reader['model'] = 'ez-reader'
        rms_ez_reader['setting'] = setting
        rms_ez_reader['reader_id'] = np.nan


        print(f'--- {setting} swift ---')
        rms_swift = pd.read_csv(os.path.join(path_to_rms, rms_swift_filename), sep='\t')
        surprisal_list_ez_reader = list()
        for idx, group in tqdm(rms_swift.groupby('instance_idx')):
            text = ' '.join(group['word'].tolist())
            _, surprisal, _ = scorer.score(text, BOS=True)
            surprisal_list_ez_reader += surprisal.tolist()
        rms_swift['surprisal'] = surprisal_list_ez_reader
        rms_swift['model'] = 'swift'
        rms_swift['setting'] = setting
        rms_swift['reader_id'] = np.nan

        all_rms_dfs.append(rms_ez_reader)
        all_rms_dfs.append(rms_swift)
    

    
    for idx, setting in enumerate(settings):
        path_to_rms = os.path.join(path_to_settings, setting)

        # surprisal for scandl fix dur output
        print(f'--- {setting} scandl fixdur ---')
        rms_scandl = pd.read_csv(os.path.join(path_to_rms, rms_scandl_filename), sep='\t')
        surprisal_list_scandl = list()
        for idx, group in tqdm(rms_scandl.groupby('instance_idx')):
            text = ' '.join(group['word'].tolist())
            _, surprisal, _ = scorer.score(text, BOS=True)
            surprisal_list_scandl += surprisal.tolist()
        rms_scandl['surprisal'] = surprisal_list_scandl
        rms_scandl['model'] = 'scandl-fixdur'
        rms_scandl['setting'] = setting

        # surprisal for human data
        print(f'--- {setting} human ---')
        rms_human = pd.read_csv(os.path.join(path_to_rms, rms_human_filename), sep='\t')
        surprisal_list_human = list()
        for idx, group in tqdm(rms_human.groupby('instance_idx')):
            text = ' '.join(group['word'].tolist())
            _, surprisal, _ = scorer.score(text, BOS=True)
            surprisal_list_human += surprisal.tolist()
        rms_human['surprisal'] = surprisal_list_human
        rms_human['model'] = 'human'
        rms_human['setting'] = setting

        all_rms_dfs.append(rms_scandl)
        all_rms_dfs.append(rms_human)



    all_rms_df = pd.concat(all_rms_dfs)
    all_rms_df = all_rms_df.drop(columns=['instance_idx'])

    # add Zipf frequency and word length
    all_rms_df['zipf_freq'] = all_rms_df['word'].apply(lambda w: zipf_frequency(w, 'en'))
    all_rms_df['word_length'] = all_rms_df['word'].apply(lambda w: len(w))

    all_rms_df.to_csv('reading_measures_annotated.csv', sep='\t', index=False)





if __name__ == '__main__':
    raise SystemExit(main())
