"""
Create the data splits for all settings (new reader, new sentence, combined (= new reader/new sentence), and cross dataset
(i.e., train on celer test on zuco) to keep train and test data consistent across all baselines and for hyper-parameter tuning.
Save all data sets as well as only the reader and sn ids (for the baselines).
"""

import argparse
import os
import json
import numpy as np
import pandas as pd


from diffusion_only.scripts.sp_load_celer_zuco import load_celer, load_celer_speakers, process_celer
from diffusion_only.scripts.sp_load_celer_zuco import load_zuco, process_zuco, get_kfold, get_kfold_indices_combined
from diffusion_only.scripts.sp_load_celer_zuco import load_emtec, process_emtec
from diffusion_only.scripts.sp_load_celer_zuco import load_bsc, process_bsc
from diffusion_only.scripts.sp_load_celer_zuco import flatten_data, unflatten_data
from transformers import set_seed, BertTokenizerFast

os.environ["WANDB_MODE"] = "offline"


def create_argparser() -> argparse.ArgumentParser:
    """ Loads the config from the file scandl/config.json and adds all keys and values in the config dict
    to the argument parser where config values are the argparse arguments' default values. """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder-name',
        type=str,
        default='processed_data',
        help='name of the folder in which the processed data is saved',
    )
    parser.add_argument(
        '--max-fix-dur',
        type=int,
        help='max fixation duration value. greater fixation durations are replaced by this value.',
        default=999,
    )
    parser.add_argument(
        '--emtec',
        action='store_true',
        help='preprocess EMTeC data.',
    )
    parser.add_argument(
        '--bsc',
        action='store_true',
        help='preprocess BSC data.',
    )

    defaults = dict()
    defaults.update(load_defaults_config(parser.parse_args()))  # load defaults from config.json

    add_dict_to_argparser(parser, defaults)  # update latest args according to argparse
    return parser


def load_defaults_config(args):
    """
    Load defaults for training args.
    """
    if args.emtec:
        config_name = 'config_emtec.json'
    elif args.bsc:
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

    
    print('Loading argument parser...')
    args = create_argparser().parse_args()
    set_seed(args.seed)
    
    
    if not args.emtec and not args.bsc:

        tokenizer = BertTokenizerFast.from_pretrained(args.config_name)

        cv_settings = [('reader', 'cv'), ('sentence', 'cv'), ('scanpath', 'cv'), ('combined', 'cv')]
        cross_dataset_settings = [('scanpath', 'zuco')]

        # load celer for all settings except cross-dataset

        print('Loading word info and eye movement df...')
        word_info_df, eyemovement_df = load_celer()
        reader_list = load_celer_speakers(only_native_speakers=args.celer_only_L1)
        sn_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()

        print('Loading data for within dataset evaluation...')
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
        # flatten the data for subsequent splitting
        flattened_data = flatten_data(data)

        for data_split_criterion, inference in cv_settings:

            data_path = os.path.join(args.folder_name, data_split_criterion)
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            args.data_split_criterion = data_split_criterion
            args.inference = inference

            if args.data_split_criterion != 'combined':
                for fold_idx, (train_idx, test_idx) in enumerate(
                    get_kfold(
                        data=flattened_data,
                        splitting_IDs_dict=splitting_IDs_dict,
                        splitting_criterion=args.data_split_criterion,
                        n_splits=args.n_folds,
                    )
                ):
                    fold_path = os.path.join(data_path, f'fold-{fold_idx}')
                    if not os.path.exists(fold_path):
                        os.makedirs(fold_path)

                    train_data = np.array(flattened_data, dtype=object)[train_idx].tolist()
                    test_data = np.array(flattened_data, dtype=object)[test_idx].tolist()

                    # save the train and test IDs separately (though they are also contained within train_data/test_data)
                    train_ids_reader = np.array(splitting_IDs_dict['reader'])[train_idx].tolist()
                    train_ids_sn = np.array(splitting_IDs_dict['sentence'])[train_idx].tolist()
                    test_ids_reader = np.array(splitting_IDs_dict['reader'])[test_idx].tolist()
                    test_ids_sn = np.array(splitting_IDs_dict['sentence'])[test_idx].tolist()
                    train_ids = [[tr_s, tr_r] for tr_s, tr_r in zip(train_ids_sn, train_ids_reader)]
                    test_ids = [[tr_s, tr_r] for tr_s, tr_r in zip(test_ids_sn, test_ids_reader)]
                    with open(os.path.join(fold_path, 'test_ids.npy'), 'wb') as f:
                        np.save(f, test_ids, allow_pickle=True)
                    with open(os.path.join(fold_path, 'train_ids.npy'), 'wb') as f:
                        np.save(f, train_ids, allow_pickle=True)

                    # save the train data
                    train_data_save = unflatten_data(flattened_data=train_data, split='train')
                    train_data_save.save_to_disk(os.path.join(fold_path, 'train_data'))

                    # save the test data
                    test_data_save = unflatten_data(flattened_data=test_data, split='test')
                    test_data_save.save_to_disk(os.path.join(fold_path, 'test_data'))

            else:  # new reader/new sentence setting
                reader_indices, sentence_indices = get_kfold_indices_combined(
                    data=flattened_data,
                    splitting_IDs_dict=splitting_IDs_dict,
                )

                reader_IDs = splitting_IDs_dict['reader']
                sn_IDs = splitting_IDs_dict['sentence']

                for fold_idx, ((reader_train_idx, reader_test_idx), (sn_train_idx, sn_test_idx)) in enumerate(
                    zip(reader_indices, sentence_indices)
                ):
                    fold_path = os.path.join(data_path, f'fold-{fold_idx}')
                    if not os.path.exists(fold_path):
                        os.makedirs(fold_path)

                    # create data sets with only unique readers and sentences in test set
                    unique_reader_test_IDs = set(np.array(reader_IDs)[reader_test_idx].tolist())
                    unique_sn_test_IDs = set(np.array(sn_IDs)[sn_test_idx].tolist())

                    # subset the data: if an ID is both in the IDs for sentence and reader sampled for the test set,
                    # add the data point to the test data; if it is in neither of them, add to train data. if
                    # in one of them, unfortunately discard
                    train_data, test_data = list(), list()
                    train_ids, test_ids = list(), list()
                    for i in range(len(flattened_data)):
                        if reader_IDs[i] in unique_reader_test_IDs and sn_IDs[i] in unique_sn_test_IDs:
                            test_data.append(flattened_data[i])
                            test_ids.append([sn_IDs[i], reader_IDs[i]])
                        elif reader_IDs[i] not in unique_reader_test_IDs and sn_IDs[i] not in unique_sn_test_IDs:
                            train_data.append(flattened_data[i])
                            train_ids.append([sn_IDs[i], reader_IDs[i]])
                        else:
                            continue

                    # save the train data
                    train_data_save = unflatten_data(flattened_data=train_data, split='train')
                    train_data_save.save_to_disk(os.path.join(fold_path, 'train_data'))

                    # save the test data
                    test_data_save = unflatten_data(flattened_data=test_data, split='test')
                    test_data_save.save_to_disk(os.path.join(fold_path, 'test_data'))

                    # save the train and test ids
                    with open(os.path.join(fold_path, 'test_ids.npy'), 'wb') as f:
                        np.save(f, test_ids, allow_pickle=True)
                    with open(os.path.join(fold_path, 'train_ids.npy'), 'wb') as f:
                        np.save(f, train_ids, allow_pickle=True)

        del data
        del splitting_IDs_dict
        del word_info_df
        del eyemovement_df
        del reader_list
        del sn_list

        # load and save the data for the cross-dataset evaluation (i.e., train celer, test zuco)
        for data_split_criterion, inference in cross_dataset_settings:
            args.data_split_criterion = data_split_criterion
            args.inference = inference

            data_path = os.path.join(args.folder_name, 'cross_dataset')
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            split_sizes = {'val_size': 0.1}

            word_info_df, eyemovement_df = load_celer()
            reader_list = load_celer_speakers(only_native_speakers=args.celer_only_L1)
            sn_list = np.unique(word_info_df[word_info_df['list'].isin(reader_list)].sentenceid.values).tolist()
            print('Loading data for cross dataset evaluation...')
            train_data, val_data = process_celer(
                sn_list=sn_list,
                reader_list=reader_list,
                word_info_df=word_info_df,
                eyemovement_df=eyemovement_df,
                tokenizer=tokenizer,
                args=args,
                split='train-val',
                split_sizes=split_sizes,
                splitting_criterion=args.data_split_criterion,
                max_fix_dur=args.max_fix_dur,
            )
            train_data.save_to_disk(os.path.join(data_path, 'train_data'))
            val_data.save_to_disk(os.path.join(data_path, 'val_data'))

            # save the train IDs (i.e., including the validation IDs --> used for the baselines)
            train_sn_ids = train_data['train']['sn_ids'] + val_data['val']['sn_ids']
            train_reader_ids = train_data['train']['reader_ids'] + val_data['val']['reader_ids']
            train_ids = [[sn_id, reader_id] for sn_id, reader_id in zip(train_sn_ids, train_reader_ids)]
            with open(os.path.join(data_path, 'train_ids.npy'), 'wb') as f:
                np.save(f, train_ids, allow_pickle=True)

            # loading ZuCo: onla ZuCo (1), not ZuCo2.0
            # only tasks 1 (Sentiment) and task 2 (Wikipedia), which are normal reading
            word_info_df, eyemovement_df = load_zuco(task='zuco11')  # task: 'zuco11', 'zuco12'
            word_info_df2, eyemovement_df2 = load_zuco(task='zuco12')
            # combine the two corpora
            word_info_df2.SN = word_info_df2.SN.values + word_info_df.SN.values.max()
            eyemovement_df2.sn = eyemovement_df2.sn.values + eyemovement_df.sn.values.max()
            word_info_df = pd.concat([word_info_df, word_info_df2])
            eyemovement_df = pd.concat([eyemovement_df, eyemovement_df2])

            # lists with unique sentence and reader IDs
            sn_list = np.unique(eyemovement_df.sn.values).tolist()
            reader_list = np.unique(eyemovement_df.id.values).tolist()

            # call the split 'train' so that the data is not split at all; use all zuco data for inference
            print('Loading ZuCo data...')
            test_data = process_zuco(
                sn_list=sn_list,
                reader_list=reader_list,
                word_info_df=word_info_df,
                eyemovement_df=eyemovement_df,
                tokenizer=tokenizer,
                args=args,
                split='train',
                splitting_criterion=args.data_split_criterion,
                max_fix_dur=args.max_fix_dur,
            )
            test_data.save_to_disk(os.path.join(data_path, 'test_data'))

            # save test IDs
            test_ids = [[sn_id, reader_id] for sn_id, reader_id in zip(
                test_data['train']['sn_ids'], test_data['train']['reader_ids']
            )]
            with open(os.path.join(data_path, 'test_ids.npy'), 'wb') as f:
                np.save(f, test_ids, allow_pickle=True)
    
    elif args.emtec:

        tokenizer = BertTokenizerFast.from_pretrained(args.config_name)

        cv_settings = [('reader', 'cv'), ('sentence', 'cv'), ('combined', 'cv')]

        folder_name = args.folder_name + '_emtec'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # load emtec data 
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
        # flatten the data for subsequent splitting
        flattened_data = flatten_data(data)

        for data_split_criterion, inference in cv_settings:

            data_path = os.path.join(folder_name, data_split_criterion)
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            
            args.data_split_criterion = data_split_criterion
            args.inference = inference

            if args.data_split_criterion != 'combined':
                for fold_idx, (train_idx, test_idx) in enumerate(
                    get_kfold(
                        data=flattened_data,
                        splitting_IDs_dict=splitting_IDs_dict,
                        splitting_criterion=args.data_split_criterion,
                        n_splits=args.n_folds,
                    )
                ):
                    fold_path = os.path.join(data_path, f'fold-{fold_idx}')
                    if not os.path.exists(fold_path):
                        os.makedirs(fold_path)

                    train_data = np.array(flattened_data, dtype=object)[train_idx].tolist()
                    test_data = np.array(flattened_data, dtype=object)[test_idx].tolist()

                    # save the train and test IDs separately (though they are also contained within train_data/test_data)
                    train_ids_reader = np.array(splitting_IDs_dict['reader'])[train_idx].tolist()
                    train_ids_sn = np.array(splitting_IDs_dict['sentence'])[train_idx].tolist()
                    test_ids_reader = np.array(splitting_IDs_dict['reader'])[test_idx].tolist()
                    test_ids_sn = np.array(splitting_IDs_dict['sentence'])[test_idx].tolist()
                    train_ids = [[tr_s, tr_r] for tr_s, tr_r in zip(train_ids_sn, train_ids_reader)]
                    test_ids = [[tr_s, tr_r] for tr_s, tr_r in zip(test_ids_sn, test_ids_reader)]
                    with open(os.path.join(fold_path, 'test_ids.npy'), 'wb') as f:
                        np.save(f, test_ids, allow_pickle=True)
                    with open(os.path.join(fold_path, 'train_ids.npy'), 'wb') as f:
                        np.save(f, train_ids, allow_pickle=True)
                    
                    # save the train data
                    train_data_save = unflatten_data(flattened_data=train_data, split='train')
                    train_data_save.save_to_disk(os.path.join(fold_path, 'train_data'))

                    # save the test data
                    test_data_save = unflatten_data(flattened_data=test_data, split='test')
                    test_data_save.save_to_disk(os.path.join(fold_path, 'test_data'))
            
            else:  # new reader/new sentence setting
                reader_indices, sentence_indices = get_kfold_indices_combined(
                    data=flattened_data,
                    splitting_IDs_dict=splitting_IDs_dict,
                )

                reader_IDs = splitting_IDs_dict['reader']
                sn_IDs = splitting_IDs_dict['sentence']

                for fold_idx, ((reader_train_idx, reader_test_idx), (sn_train_idx, sn_test_idx)) in enumerate(
                    zip(reader_indices, sentence_indices)
                ):
                    fold_path = os.path.join(data_path, f'fold-{fold_idx}')
                    if not os.path.exists(fold_path):
                        os.makedirs(fold_path)
                    
                    # create data sets with only unique readers and sentences in test set
                    unique_reader_test_IDs = set(np.array(reader_IDs)[reader_test_idx].tolist())
                    unique_sn_test_IDs = set(np.array(sn_IDs)[sn_test_idx].tolist())

                    # subset the data: if an ID is both in the IDs for sentence and reader sampled for the test set,
                    # add the data point to the test data; if it is in neither of them, add to train data. if
                    # in one of them, unfortunately discard
                    train_data, test_data = list(), list()
                    train_ids, test_ids = list(), list()
                    for i in range(len(flattened_data)):
                        if reader_IDs[i] in unique_reader_test_IDs and sn_IDs[i] in unique_sn_test_IDs:
                            test_data.append(flattened_data[i])
                            test_ids.append([sn_IDs[i], reader_IDs[i]])
                        elif reader_IDs[i] not in unique_reader_test_IDs and sn_IDs[i] not in unique_sn_test_IDs:
                            train_data.append(flattened_data[i])
                            train_ids.append([sn_IDs[i], reader_IDs[i]])
                        else:
                            continue
                    
                    # save the train data
                    train_data_save = unflatten_data(flattened_data=train_data, split='train')
                    train_data_save.save_to_disk(os.path.join(fold_path, 'train_data'))

                    # save the test data
                    test_data_save = unflatten_data(flattened_data=test_data, split='test')
                    test_data_save.save_to_disk(os.path.join(fold_path, 'test_data'))

                    # save the train and test ids
                    with open(os.path.join(fold_path, 'test_ids.npy'), 'wb') as f:
                        np.save(f, test_ids, allow_pickle=True)
                    with open(os.path.join(fold_path, 'train_ids.npy'), 'wb') as f:
                        np.save(f, train_ids, allow_pickle=True)

    elif args.bsc:

        tokenizer = BertTokenizerFast.from_pretrained(args.config_name)

        cv_settings = [('reader', 'cv'), ('sentence', 'cv'), ('combined', 'cv')]

        folder_name = args.folder_name + '_bsc'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # load bsc data 
        word_info_df, pos_info_df, eyemovement_df = load_bsc()
        # list of sentence ids and reader ids
        #sn_list = np.unique(eyemovement_df.sn.values).tolist()
        #reader_list = np.unique(eyemovement_df.id.values).tolist()

        data, splitting_IDs_dict = process_bsc(
            word_info_df=word_info_df,
            eyemovement_df=eyemovement_df,
            tokenizer=tokenizer,
            args=args,
            inference='cv',
            max_fix_dur=args.max_fix_dur,
        )
        # flatten the data for subsequent splitting
        flattened_data = flatten_data(data)

        for data_split_criterion, inference in cv_settings:

            data_path = os.path.join(folder_name, data_split_criterion)
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            
            args.data_split_criterion = data_split_criterion
            args.inference = inference

            if args.data_split_criterion != 'combined':
                for fold_idx, (train_idx, test_idx) in enumerate(
                    get_kfold(
                        data=flattened_data,
                        splitting_IDs_dict=splitting_IDs_dict,
                        splitting_criterion=args.data_split_criterion,
                        n_splits=args.n_folds,
                    )
                ):
                    fold_path = os.path.join(data_path, f'fold-{fold_idx}')
                    if not os.path.exists(fold_path):
                        os.makedirs(fold_path)

                    train_data = np.array(flattened_data, dtype=object)[train_idx].tolist()
                    test_data = np.array(flattened_data, dtype=object)[test_idx].tolist()

                    # save the train and test IDs separately (though they are also contained within train_data/test_data)
                    train_ids_reader = np.array(splitting_IDs_dict['reader'])[train_idx].tolist()
                    train_ids_sn = np.array(splitting_IDs_dict['sentence'])[train_idx].tolist()
                    test_ids_reader = np.array(splitting_IDs_dict['reader'])[test_idx].tolist()
                    test_ids_sn = np.array(splitting_IDs_dict['sentence'])[test_idx].tolist()
                    train_ids = [[tr_s, tr_r] for tr_s, tr_r in zip(train_ids_sn, train_ids_reader)]
                    test_ids = [[tr_s, tr_r] for tr_s, tr_r in zip(test_ids_sn, test_ids_reader)]
                    with open(os.path.join(fold_path, 'test_ids.npy'), 'wb') as f:
                        np.save(f, test_ids, allow_pickle=True)
                    with open(os.path.join(fold_path, 'train_ids.npy'), 'wb') as f:
                        np.save(f, train_ids, allow_pickle=True)
                    
                    # save the train data
                    train_data_save = unflatten_data(flattened_data=train_data, split='train')
                    train_data_save.save_to_disk(os.path.join(fold_path, 'train_data'))

                    # save the test data
                    test_data_save = unflatten_data(flattened_data=test_data, split='test')
                    test_data_save.save_to_disk(os.path.join(fold_path, 'test_data'))

            else:  # new reader/new sentence setting
                reader_indices, sentence_indices = get_kfold_indices_combined(
                    data=flattened_data,
                    splitting_IDs_dict=splitting_IDs_dict,
                )

                reader_IDs = splitting_IDs_dict['reader']
                sn_IDs = splitting_IDs_dict['sentence']

                for fold_idx, ((reader_train_idx, reader_test_idx), (sn_train_idx, sn_test_idx)) in enumerate(
                    zip(reader_indices, sentence_indices)
                ):
                    fold_path = os.path.join(data_path, f'fold-{fold_idx}')
                    if not os.path.exists(fold_path):
                        os.makedirs(fold_path)
                    
                    # create data sets with only unique readers and sentences in test set
                    unique_reader_test_IDs = set(np.array(reader_IDs)[reader_test_idx].tolist())
                    unique_sn_test_IDs = set(np.array(sn_IDs)[sn_test_idx].tolist())

                    # subset the data: if an ID is both in the IDs for sentence and reader sampled for the test set,
                    # add the data point to the test data; if it is in neither of them, add to train data. if
                    # in one of them, unfortunately discard
                    train_data, test_data = list(), list()
                    train_ids, test_ids = list(), list()
                    for i in range(len(flattened_data)):
                        if reader_IDs[i] in unique_reader_test_IDs and sn_IDs[i] in unique_sn_test_IDs:
                            test_data.append(flattened_data[i])
                            test_ids.append([sn_IDs[i], reader_IDs[i]])
                        elif reader_IDs[i] not in unique_reader_test_IDs and sn_IDs[i] not in unique_sn_test_IDs:
                            train_data.append(flattened_data[i])
                            train_ids.append([sn_IDs[i], reader_IDs[i]])
                        else:
                            continue
                    
                    # save the train data
                    train_data_save = unflatten_data(flattened_data=train_data, split='train')
                    train_data_save.save_to_disk(os.path.join(fold_path, 'train_data'))

                    # save the test data
                    test_data_save = unflatten_data(flattened_data=test_data, split='test')
                    test_data_save.save_to_disk(os.path.join(fold_path, 'test_data'))

                    # save the train and test ids
                    with open(os.path.join(fold_path, 'test_ids.npy'), 'wb') as f:
                        np.save(f, test_ids, allow_pickle=True)
                    with open(os.path.join(fold_path, 'train_ids.npy'), 'wb') as f:
                        np.save(f, train_ids, allow_pickle=True)
        


if __name__ == '__main__':
    raise SystemExit(main())
