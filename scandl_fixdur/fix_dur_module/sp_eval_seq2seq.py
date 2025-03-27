"""
Run the evaluation on the Seq2Seq fixation duration module output.
"""

import os
import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from CONSTANTS import FIXDUR_MODULE_INF_PATH, FIXDUR_MODULE_INF_PATH_BSC, FIXDUR_MODULE_INF_PATH_EMTEC

import numpy as np
import pandas as pd
import json
from scandl_fixdur.scandl_module.scripts.scanpath_similarity import levenshtein_distance
from scandl_fixdur.scandl_module.scripts.scanpath_similarity import levenshtein_similarity
from scandl_fixdur.scandl_module.scripts.scanpath_similarity import levenshtein_normalized_distance
from scandl_fixdur.scandl_module.scripts.scanpath_similarity import levenshtein_normalized_similarity
from scandl_fixdur.fix_dur_module.scasim import scasim
from typing import Optional
from argparse import ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--emtec',
        action='store_true',
        help='Run evaluation for EMTeC'
    )
    parser.add_argument(
        '--bsc',
        action='store_true',
        help='Run evaluation for BSC'
    )
    # argument 'setting' that allows for several settings as list
    parser.add_argument(
        '--setting',
        nargs='+',
        default=['reader', 'sentence', 'combined', 'cross_dataset'],
        help='Settings to run evaluation for'
    )
    return parser



def run_eval(
    path_to_output: str,
    output_filename: str,
    seed: Optional[int] = None,
):

    with open(os.path.join(path_to_output, output_filename), 'r') as f:
        outputs = json.load(f)

     # run the evaluation on the output of the fixation duration module

    print(' ... evaluating scandl fix-dur ...')

    ## fixation locations

    original_sp_ids = outputs['original_sp_ids']
    predicted_sp_ids = outputs['predicted_sp_ids']

    # replace empty lists with [PAD] or they will throw an error for levenshtein distance 
    original_sp_ids = [orig_sp_ids if orig_sp_ids != [] else [511] for orig_sp_ids in original_sp_ids]
    predicted_sp_ids = [pred_sp_ids if pred_sp_ids != [] else [511] for pred_sp_ids in predicted_sp_ids]

    ## fixation durations 

    original_fix_durs = outputs['original_fix_durs']
    predicted_fix_durs = outputs['predicted_fix_durs']

    # replace empty lists with [0] or they will throw an error for levnshtein distance
    original_fix_durs = [orig_fix_durs if orig_fix_durs != [] else [0] for orig_fix_durs in original_fix_durs]
    predicted_fix_durs = [pred_fix_durs if pred_fix_durs != [] else [0] for pred_fix_durs in predicted_fix_durs]

    ls_list, ld_list, nld_list, nls_list = [], [], [], []
    scasim_list, scasim_normalized_fixations_list, scasim_normalized_durations_list = [], [], []

    for idx, (pred_fix_locs, orig_fix_locs, pred_fix_durs, orig_fix_durs) in enumerate(zip(
        predicted_sp_ids, original_sp_ids, predicted_fix_durs, original_fix_durs
    )):

        # Compute the Levenshtein Distance

        ld_list.append(levenshtein_distance(gt=orig_fix_locs, pred=pred_fix_locs))
        ls_list.append(levenshtein_similarity(gt=orig_fix_locs, pred=pred_fix_locs))
        nld_list.append(levenshtein_normalized_distance(gt=orig_fix_locs, pred=pred_fix_locs))
        nls_list.append(levenshtein_normalized_similarity(gt=orig_fix_locs, pred=pred_fix_locs))

        # Compute ScaSim 

        # create dummy y-values for original fixation locations and predicted fixation locations
        dummy_y_orig_fix_locs = [1] * len(orig_fix_locs)
        dummy_y_pred_fix_locs = [1] * len(pred_fix_locs)

        # zip together the predicted fixation locations, the dummy y-values and the predicted fixation durations as list of triples
        predicted_sp = list(zip(pred_fix_locs, dummy_y_pred_fix_locs, pred_fix_durs))
        # zip together the original fixation locations, the dummy y-values and the original fixation durations as list of triples
        original_sp = list(zip(orig_fix_locs, dummy_y_orig_fix_locs, orig_fix_durs))

        # compute ScaSim
        sim = scasim(s=predicted_sp, t=original_sp)
        sim_normalized_fixations = scasim(s=predicted_sp, t=original_sp, normalize='fixations')
        sim_normalized_durations = scasim(s=predicted_sp, t=original_sp, normalize='durations')

        scasim_list.append(sim)
        scasim_normalized_fixations_list.append(sim_normalized_fixations)
        scasim_normalized_durations_list.append(sim_normalized_durations)

    eval_dict_scandl_fixdur = {
        'ls': np.round(np.mean(np.array(ls_list)), 3).item(),
        'ld': np.round(np.mean(np.array(ld_list)), 3).item(),
        'nld': np.round(np.mean(np.array(nld_list)), 3).item(),
        'nls': np.round(np.mean(np.array(nls_list)), 3).item(),
        'scasim': np.round(np.mean(np.array(scasim_list)), 3).item(),
        'scasim_normalized_nfix': np.round(np.mean(np.array(scasim_normalized_fixations_list)), 3).item(),
        'scasim_normalized_fixdur': np.round(np.mean(np.array(scasim_normalized_durations_list)), 3).item(),
    }


    # run the evaluation on the human baseline

    print(' ... evaluating human baseline ...')

    outputs_df = pd.DataFrame(outputs)

    for row_idx, row in outputs_df.iterrows():

        original_sp_ids = row['original_sp_ids']
        original_fix_durs = row['original_fix_durs']

        sn_id = row['sn_ids']

        # randomly sample another scanpath on the same sentence 
        matching_rows = outputs_df[outputs_df['sn_ids'] == sn_id]

        # if no other scanpath on the same sentence, skip
        if len(matching_rows) <= 1:
            continue
        
        sampled_row = matching_rows.sample(n=1, random_state=seed).iloc[0]
        sampled_sp_ids = sampled_row['original_sp_ids']
        sampled_fix_durs = sampled_row['original_fix_durs']

        ls_list, ld_list, nld_list, nls_list = [], [], [], []
        scasim_list, scasim_normalized_fixations_list, scasim_normalized_durations_list = [], [], []

        # compute levenshtein distance and similarity
        ld_list.append(levenshtein_distance(gt=original_sp_ids, pred=sampled_sp_ids))
        ls_list.append(levenshtein_similarity(gt=original_sp_ids, pred=sampled_sp_ids))
        nld_list.append(levenshtein_normalized_distance(gt=original_sp_ids, pred=sampled_sp_ids))
        nls_list.append(levenshtein_normalized_similarity(gt=original_sp_ids, pred=sampled_sp_ids))

        # compute scasim

        # create dummy y-values for original and sampled fixation locations 
        dummy_y_orig_sp_ids = [1] * len(original_sp_ids)
        dummy_y_sampled_sp_ids = [1] * len(sampled_sp_ids)

        # zip together the sampled fixation locations, the dummy y-values and the sampled fixation durations as list of triples
        sampled_sp = list(zip(sampled_sp_ids, dummy_y_sampled_sp_ids, sampled_fix_durs))
        # zip together the original fixation locations, the dummy y-values and the original fixation durations as list of triples
        orig_sp = list(zip(original_sp_ids, dummy_y_orig_sp_ids, original_fix_durs))

        # compute scasim
        sim = scasim(s=orig_sp, t=sampled_sp)
        sim_normalized_fixations = scasim(s=orig_sp, t=sampled_sp, normalize='fixations')
        sim_normalized_durations = scasim(s=orig_sp, t=sampled_sp, normalize='durations')

        scasim_list.append(sim)
        scasim_normalized_fixations_list.append(sim_normalized_fixations)
        scasim_normalized_durations_list.append(sim_normalized_durations)

    eval_dict_human = {
        'ls': np.round(np.mean(np.array(ls_list)), 3).item(),
        'ld': np.round(np.mean(np.array(ld_list)), 3).item(),
        'nld': np.round(np.mean(np.array(nld_list)), 3).item(),
        'nls': np.round(np.mean(np.array(nls_list)), 3).item(),
        'scasim': np.round(np.mean(np.array(scasim_list)), 3).item(),
        'scasim_normalized_nfix': np.round(np.mean(np.array(scasim_normalized_fixations_list)), 3).item(),
        'scasim_normalized_fixdur': np.round(np.mean(np.array(scasim_normalized_durations_list)), 3).item(),
    }
    
    return eval_dict_scandl_fixdur, eval_dict_human


def main():

    parser = get_parser()
    args = parser.parse_args()

    if args.emtec:
        fixdur_inf_path = FIXDUR_MODULE_INF_PATH_EMTEC
    elif args.bsc:
        fixdur_inf_path = FIXDUR_MODULE_INF_PATH_BSC
    else:
        fixdur_inf_path = FIXDUR_MODULE_INF_PATH

    seed = 10

    df_filename_scandl_fixdur = 'scandl-fixdur-eval-metrics.csv'
    folds_filename_scandl_fixdur = 'scandl-fixdur-eval-metrics-folds.json'
    df_filename_human = 'human-eval-metrics.csv'
    folds_filename_human = 'human-eval-metrics-folds.json'
    
    all_results_scandl_fixdur = {
        'setting': [],
        'ls_mean': [],
        'ls_std': [],
        'ld_mean': [],
        'ld_std': [],
        'nld_mean': [],
        'nld_std': [],
        'nls_mean': [],
        'nls_std': [],
        'scasim_mean': [],
        'scasim_std': [],
        'scasim_normalized_nfix_mean': [],
        'scasim_normalized_nfix_std': [],
        'scasim_normalized_fixdur_mean': [],
        'scasim_normalized_fixdur_std': [],
    }
    all_results_human = {
        'setting': [],
        'ls_mean': [],
        'ls_std': [],
        'ld_mean': [],
        'ld_std': [],
        'nld_mean': [],
        'nld_std': [],
        'nls_mean': [],
        'nls_std': [],
        'scasim_mean': [],
        'scasim_std': [],
        'scasim_normalized_nfix_mean': [],
        'scasim_normalized_nfix_std': [],
        'scasim_normalized_fixdur_mean': [],
        'scasim_normalized_fixdur_std': [],
    }

    all_folds_scandl_fixdur = {}    
    all_folds_human = {}
    
    settings = args.setting 

    for setting in settings: 

        if setting == 'cross_dataset':

            print(f' --- running evaluation for setting {setting} ---')

            path_to_output = os.path.join(fixdur_inf_path, setting)
            
            eval_dict_scandl_fixdur, eval_dict_human = run_eval(
                path_to_output=path_to_output,
                output_filename='output_dict.json',
                seed=seed,
            )
            for key in all_results_scandl_fixdur.keys():
                if key.endswith('mean'):
                    all_results_scandl_fixdur[key].append(eval_dict_scandl_fixdur[key[:-5]])
                    all_results_human[key].append(eval_dict_human[key[:-5]])
                elif key == 'setting':
                    all_results_scandl_fixdur[key].append(setting)
                    all_results_human[key].append(setting)
                else:
                    all_results_scandl_fixdur[key].append(np.nan)
                    all_results_human[key].append(np.nan)
        
        else:

            
            all_folds_scandl_fixdur[setting] = {}
            all_folds_human[setting] = {}
            

            folds = os.listdir(os.path.join(fixdur_inf_path, setting))
            fold_dicts_scandl_fixdur = list()
            fold_dicts_human = list()

            for fold in folds: 

                all_folds_scandl_fixdur[setting][fold] = {}
                all_folds_human[setting][fold] = {}

                print(f' --- running evaluation for setting {setting} fold {fold} ---')

                path_to_output = os.path.join(fixdur_inf_path, setting, fold)
                fold_dict_scandl_fixdur, fold_dict_human = run_eval(
                    path_to_output=path_to_output,
                    output_filename='output_dict.json',
                    seed=seed,
                )
                fold_dicts_scandl_fixdur.append(fold_dict_scandl_fixdur)
                fold_dicts_human.append(fold_dict_human)

                for key in fold_dict_scandl_fixdur.keys():
                    if key not in all_folds_scandl_fixdur[setting][fold]:
                        all_folds_scandl_fixdur[setting][fold][key] = fold_dict_scandl_fixdur[key]
                        all_folds_human[setting][fold][key] = fold_dict_human[key]
                    else:
                        all_folds_scandl_fixdur[setting][fold][key].append(fold_dict_scandl_fixdur[key])
                        all_folds_human[setting][fold][key].append(fold_dict_human[key])

            combined_fold_dict_scandl_fixdur = {}
            combined_fold_dict_human = {}
            for d in fold_dicts_scandl_fixdur:
                for key, value in d.items():
                    if key in combined_fold_dict_scandl_fixdur:
                        combined_fold_dict_scandl_fixdur[key].append(value)
                    else:
                        combined_fold_dict_scandl_fixdur[key] = [value]
            for d in fold_dicts_human:
                for key, value in d.items():
                    if key in combined_fold_dict_human:
                        combined_fold_dict_human[key].append(value)
                    else:
                        combined_fold_dict_human[key] = [value]
            

            all_results_scandl_fixdur['setting'].append(setting)
            all_results_human['setting'].append(setting)
            for key in combined_fold_dict_scandl_fixdur.keys():
                if key in ['setting']:
                    continue
                all_results_scandl_fixdur[f'{key}_mean'].append(np.mean(np.array(combined_fold_dict_scandl_fixdur[key])))
                all_results_scandl_fixdur[f'{key}_std'].append(np.std(np.array(combined_fold_dict_scandl_fixdur[key])))
                all_results_human[f'{key}_mean'].append(np.mean(np.array(combined_fold_dict_human[key])))
                all_results_human[f'{key}_std'].append(np.std(np.array(combined_fold_dict_human[key])))


    with open(os.path.join(fixdur_inf_path, folds_filename_scandl_fixdur), 'w') as f:
        json.dump(all_folds_scandl_fixdur, f)
    with open(os.path.join(fixdur_inf_path, folds_filename_human), 'w') as f:
        json.dump(all_folds_human, f)
    

    all_results_df_scandl_fixdur = pd.DataFrame(all_results_scandl_fixdur)
    all_results_df_human = pd.DataFrame(all_results_human)
    all_results_df_scandl_fixdur.to_csv(os.path.join(fixdur_inf_path, df_filename_scandl_fixdur), index=False, sep='\t')
    all_results_df_human.to_csv(os.path.join(fixdur_inf_path, df_filename_human), index=False, sep='\t')
    


if __name__ == '__main__':
    raise SystemExit(main())