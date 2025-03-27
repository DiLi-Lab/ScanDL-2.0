"""
Train ScanDL.
"""



import argparse
import json
import os
import numpy as np
import wandb

import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from transformers import set_seed, BertTokenizerFast
from datasets import load_from_disk, DatasetDict, Dataset

from scandl_diff_dur.utils import dist_util, logger
from scandl_diff_dur.step_sample import create_named_schedule_sampler
from scripts.sp_basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from scripts.sp_train_util import TrainLoop
from scripts.sp_load_celer_zuco import load_celer, load_celer_speakers, process_celer, celer_zuco_dataset_and_loader
from scripts.sp_load_celer_zuco import get_kfold, get_kfold_indices_combined
from scripts.sp_load_celer_zuco import flatten_data, unflatten_data

os.environ["WANDB_MODE"] = "offline"


def create_argparser():
    """ Loads the config from the file diffuseq/config.json and adds all keys and values in the config dict
    to the argument parser where config values are the argparse arguments' default values. """
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()

    add_dict_to_argparser(parser, defaults)  # update latest args according to argparse
    return parser



def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)

    if os.path.exists('../processed_data/hp-tuning-data/train'):
        train_data = load_from_disk('../processed_data/hp-tuning-data/train')
    else:
        # create triple cross-validation data 
        dataset = load_from_disk('../processed_data/reader/fold-0/train_data')

        # the test data should only contain new readers
        # extract unique reader ids
        reader_ids = list(set(dataset['train']['reader_ids']))
        # shuffle the ids
        np.random.shuffle(reader_ids)
        # proportions for train and test sets
        train_size = int(0.75 * len(reader_ids))
        test_size = len(reader_ids) - train_size

        train_ids = set(reader_ids[:train_size])
        test_ids = set(reader_ids[train_size:])

        # create new datasets with non-overlapping reader ids
        def filter_dataset(dataset, reader_ids):
            return dataset.filter(lambda x: x['reader_ids'] in reader_ids)

        # apply filtering 
        train_dataset = filter_dataset(dataset['train'], train_ids)
        test_dataset = filter_dataset(dataset['train'], test_ids)

        split_dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        }) 
        if not os.path.exists('../processed_data/hp-tuning-data/'):
            os.makedirs('../processed_data/hp-tuning-data/')
        split_dataset_dict.save_to_disk('../processed_data/hp-tuning-data/')
        train_data = split_dataset_dict['train']



    assert args.seq_len == args.hidden_t_dim

    # set up distributed processing group
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    rank = dist.get_rank() or 0

    tokenizer = BertTokenizerFast.from_pretrained(args.config_name)

    args.vocab_size = tokenizer.vocab_size

    if rank == 0:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)

    # load train data for HP tuning
    #print('\t\t--- load train data from disk!')
    #train_data = load_from_disk('hp_tuning/hp_tuning_data/train')
    
    flattened_data = flatten_data(train_data.to_dict())

    # train_data = load_from_disk(os.path.join(path_to_data_dir, 'train_data'))
    # print('\t\t--- loaded train data from disk!')
    # flattened_data = flatten_data(train_data['train'].to_dict())
    train_data, val_data = train_test_split(flattened_data, test_size=0.1, shuffle=True, random_state=77)

    # unflatten the data
    train_data = unflatten_data(flattened_data=train_data, split='train')
    val_data = unflatten_data(flattened_data=val_data, split='val')

    train_loader = celer_zuco_dataset_and_loader(
        data=train_data,
        data_args=args,
        split='train',
    )
    val_loader = celer_zuco_dataset_and_loader(
        data=val_data,
        data_args=args,
        split='val',
    )

    logger.log("### Creating model and diffusion...")
    print('#' * 30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    model.to(dist_util.dev())

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    # args.schedule_sampler = lossaware
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log('### Training...')

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=val_loader,
        eval_interval=args.eval_interval,
    ).run_loop()



if __name__ == '__main__':
    main()