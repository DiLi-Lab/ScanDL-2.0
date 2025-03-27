import os
import numpy as np
import pandas as pd
import json
from argparse import ArgumentParser
from tqdm import tqdm

import sys 
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from CONSTANTS import DIFFUSION_ONLY_INF_PATH

from diffusion_only.scripts.scanpath_similarity import levenshtein_distance
from diffusion_only.scripts.scanpath_similarity import levenshtein_similarity
from diffusion_only.scripts.scanpath_similarity import levenshtein_normalized_distance
from diffusion_only.scripts.scanpath_similarity import levenshtein_normalized_similarity
from diffusion_only.scripts.scasim import scasim


def main():
    
    

    metrics_dict = {
        'ld': list(),
        'ls': list(),
        'nld': list(),
        'nls': list(),
        'scasim': list(),
        'scasim_normalized_fixations': list(),
        'scasim_normalized_durations': list(),
    }

    output_filename = [f for f in os.listdir(DIFFUSION_ONLY_INF_PATH) if f.endswith('remove-PAD_rank0.json')][0]

    with open(os.path.join(DIFFUSION_ONLY_INF_PATH, output_filename), 'r') as f:
        model_output = json.load(f)
    
    ## fixation locations

    original_sp_ids = model_output['original_sp_ids']
    predicted_sp_ids = model_output['predicted_sp_ids']

    # replace empty lists with [PAD] or they will throw an error for levenshtein distance
    original_sp_ids = [orig_sp_ids if orig_sp_ids != [] else [511] for orig_sp_ids in original_sp_ids]
    predicted_sp_ids = [pred_sp_ids if pred_sp_ids != [] else [511] for pred_sp_ids in predicted_sp_ids]

    ## fixation durations

    original_fix_durs = model_output['original_fix_durs']
    predicted_fix_durs = model_output['predicted_fix_durs']

    # replace empty lists with [0] or they will throw an error for levenshtein distance
    original_fix_durs = [orig_fix_durs if orig_fix_durs != [] else [0] for orig_fix_durs in original_fix_durs]
    predicted_fix_durs = [pred_fix_durs if pred_fix_durs != [] else [0] for pred_fix_durs in predicted_fix_durs]

    ld_list, ls_list, nld_list, nls_list = list(), list(), list(), list()
    scasim_list, scasim_normalized_fixations_list, scasim_normalized_durations_list = list(), list(), list()

    for idx, (pred_fix_loc, orig_fix_loc) in tqdm(enumerate(zip(predicted_sp_ids, original_sp_ids))):

        # Levenshtein distance and similarity
        
        ld_list.append(levenshtein_distance(gt=orig_fix_loc, pred=pred_fix_loc))
        ls_list.append(levenshtein_similarity(gt=orig_fix_loc, pred=pred_fix_loc))
        nld_list.append(levenshtein_normalized_distance(gt=orig_fix_loc, pred=pred_fix_loc))
        nls_list.append(levenshtein_normalized_similarity(gt=orig_fix_loc, pred=pred_fix_loc))

        # ScaSim

        # create dummy y-values for original_sp_ids and predicted_sp_ids
        dummy_y_orig_fix_loc = [1] * len(orig_fix_loc)
        dummy_y_pred_fix_loc = [1] * len(pred_fix_loc)

        # zip together the predicted_sp_ids and predicted_fix_durs list as list of triples
        predicted_sp = list(zip(pred_fix_loc, dummy_y_pred_fix_loc, predicted_fix_durs[idx]))
        # zip together the original_sp_ids and original_fix_durs list as list of triples
        original_sp = list(zip(orig_fix_loc, dummy_y_orig_fix_loc, original_fix_durs[idx]))

        # compute ScaSim
        sim = scasim(s=predicted_sp, t=original_sp)
        sim_normalized_fixations = scasim(s=predicted_sp, t=original_sp, normalize='fixations')
        sim_normalized_durations = scasim(s=predicted_sp, t=original_sp, normalize='durations')

        scasim_list.append(sim)
        scasim_normalized_fixations_list.append(sim_normalized_fixations)
        scasim_normalized_durations_list.append(sim_normalized_durations)

    ld = np.round(np.mean(np.array(ld_list)), 5).item()
    ls = np.round(np.mean(np.array(ls_list)), 5).item()
    nld = np.round(np.mean(np.array(nld_list)), 5).item()
    nls = np.round(np.mean(np.array(nls_list)), 5).item()

    scasim_mean = np.round(np.mean(np.array(scasim_list)), 5).item()
    scasim_normalized_fixations_mean = np.round(np.mean(np.array(scasim_normalized_fixations_list)), 5).item()
    scasim_normalized_durations_mean = np.round(np.mean(np.array(scasim_normalized_durations_list)), 5).item()

    metrics_dict['ld'].append(ld)
    metrics_dict['ls'].append(ls)
    metrics_dict['nld'].append(nld)
    metrics_dict['nls'].append(nls)
    metrics_dict['scasim'].append(scasim_mean)
    metrics_dict['scasim_normalized_fixations'].append(scasim_normalized_fixations_mean)
    metrics_dict['scasim_normalized_durations'].append(scasim_normalized_durations_mean)
    
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(os.path.join(DIFFUSION_ONLY_INF_PATH, 'diffusion-only-metrics.csv'), index=False)


            



if __name__ == '__main__':
    raise SystemExit(main())