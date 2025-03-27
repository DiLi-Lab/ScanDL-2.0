"""
Train ScanDL 2.0 on all data of the current dataset.
"""

import argparse
import json
import os
import numpy as np
#import wandb

import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from transformers import set_seed, BertTokenizerFast
from datasets import load_from_disk, DatasetDict

from original_scandl.utils import dist_util, logger
from original_scandl.step_sample import create_named_schedule_sampler
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


def create_argparser():
    """ Loads the config from the file scandl/config.json and adds all keys and values in the config dict
    to the argument parser where config values are the argparse arguments' default values. """
   
    defaults = dict(
        checkpoint_path='',
        vocab='bert',
        use_plm_init='no',
        lr=1e-4,
        batch_size=64,
        microbatch=64,
        diffusion_steps=2000,
        noise_schedule='sqrt',
        schedule_sampler='lossaware',
        seq_len=128,
        resume_checkpoint='none',
        hidden_t_dim=128,
        seed=101,
        hidden_dim=256,
        learning_steps=80000,
        save_interval=5000,
        #config_name='bert-base-cased',
        notes='-',
        data_split_criterion='',
        num_transformer_layers=12,
        num_transformer_heads=8,
        corpus='',
        inference='',
        load_train_data='-',
    )
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)  # update latest args according to argparse
    return parser


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)

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


    # load train data 
    train = load_from_disk(os.path.join('..', args.load_train_data, 'train'))
    train_data = DatasetDict()
    train_data['train'] = train
    print('\t\t--- loaded train data ---')

    train_loader = celer_zuco_dataset_and_loader(
        data=train_data,
        data_args=args,
        split='train',
    )

    logger.log("### Creating model and diffusion...")
    if torch.cuda.is_available():
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

    # if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
    #     wandb.init(
    #         project=os.getenv("WANDB_PROJECT", "ScanDL"),
    #         name=args.checkpoint_path,
    #     )
    #     wandb.config.update(args.__dict__, allow_val_change=True)

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
        #eval_data=val_loader,
        eval_interval=args.eval_interval,
    ).run_loop()

if __name__ == '__main__':
    raise SystemExit(main())
