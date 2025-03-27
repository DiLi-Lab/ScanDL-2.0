"""
Run inference on ScanDL fixdur for the hyper-parameter tuned models.
"""

import argparse
import os
import json
import sys
import time
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from transformers import set_seed
from transformers import BertTokenizerFast
from datasets import load_from_disk, DatasetDict

from scandl_diff_dur.sp_rounding import denoised_fn_round
from scandl_diff_dur.utils import dist_util, logger
from scandl_diff_dur.utils.nn import *
from scripts.sp_load_celer_zuco import celer_zuco_dataset_and_loader
from scripts.sp_load_celer_zuco import load_zuco, process_zuco

from scripts.sp_basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)


def get_parser() -> argparse.ArgumentParser:
    defaults = dict(
        model_path='',
        step=0,
        out_dir='',
        top_p=0,
        clamp_first='yes',
        test_set_sns='mixed',
        atten_vis=False,
        notes='-',
        tsne_vis=False,
        sp_vis=False,
        no_inst=0,
        atten_vis_sp=False,
        load_ids='-',
        load_test_data='-',
        eta=0.0,
    )
    decode_defaults = dict(
        split='valid',
        clamp_step=0,
        seed2=105,
        clip_denoised=False,
    )
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


@torch.no_grad()
def main():

    args = get_parser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1  # the number of processes in the current process group
    rank = dist.get_rank() or 0

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    if args.clamp_first == 'yes':
        clamp_first_bool = True
    else:
        clamp_first_bool = False

    # set mask_padding to False (otherwise would be leak)
    args.mask_padding = False

    tokenizer = BertTokenizerFast.from_pretrained(args.config_name)

    args.vocab_size = tokenizer.vocab_size

    if args.tsne_vis and args.batch_size > 12:
        raise RuntimeError('choose a smaller batch size (<=12) when plotting tSNE plots because tensors are all '
                           'moved to CPU.')

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location='cpu')
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    sn_sp_repr_embedding = nn.Embedding(
        num_embeddings=args.hidden_t_dim,
        embedding_dim=args.hidden_dim,
        _weight=model.sn_sp_repr_embedding.weight.clone().cpu(),
    )

    fix_dur_embedding = nn.Embedding(
        num_embeddings=1000,
        embedding_dim=args.hidden_dim,
        _weight=model.sn_sp_fix_dur_embedding.weight.clone().cpu(),
    )

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    test_data = load_from_disk(args.load_test_data)
    new_data = {'test': test_data}
    test_data = DatasetDict(new_data)

    test_loader = celer_zuco_dataset_and_loader(
        data=test_data,
        data_args=args,
        split='test',
        loop=False,
        deterministic=True,
    )

    out_path = args.out_dir 
    if rank == 0:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
    out_path_incrementally_pad = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}_clamp-first-{args.clamp_first}_running_remove-PAD_rank{rank}.json")
    out_path_all_pad = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}_clamp-first-{args.clamp_first}_all_remove-PAD_rank{rank}.json")


    start_t = time.time()

    all_test_data = []
    idx = 0

    try:
        while True:
            batch = next(test_loader)
            if idx % world_size == rank:  # split data per nodes/GPUs
                all_test_data.append(batch)
            idx += 1
    except StopIteration:
        print('### End of reading iteration ...')

    if idx % world_size and rank >= idx % world_size:
        all_test_data.append({})  # dummy data for Remainder: for dist.barrier()   ??

    if rank == 0:
        from tqdm import tqdm
        iterator = tqdm(all_test_data)
    else:
        iterator = iter(all_test_data)

    predicted_sp_words_pad = []
    predicted_sp_ids_pad = []
    original_sp_words_pad = []
    original_sp_ids_pad = []
    predicted_fix_durs_pad = []
    original_fix_durs_pad = []
    original_sn_pad = []
    sn_ids_pad = []
    reader_ids_pad = []


    data_dict_initial_pad = {
        'predicted_sp_words': list(),
        'original_sp_words': list(),
        'predicted_sp_ids': list(),
        'original_sp_ids': list(),
        'predicted_fix_durs': list(),
        'original_fix_durs': list(),
        'original_sn': list(),
        'sn_ids': list(),
        'reader_ids': list(),
    }


    instance_ctr = 0

    for batch_idx, batch in enumerate(iterator):

        if not batch:
            for i in range(world_size):
                dist.barrier()
            continue

        mask = batch['mask'].to(dist_util.dev())
        sn_sp_repr = batch['sn_sp_repr'].to(dist_util.dev())
        sn_input_ids = batch['sn_input_ids'].to(dist_util.dev())
        indices_pos_enc = batch['indices_pos_enc'].to(dist_util.dev())
        sn_sp_fix_dur = batch['sn_sp_fix_dur'].to(dist_util.dev())
        sn_repr_len = batch['sn_repr_len'].to(dist_util.dev())
        words_for_mapping = batch['words_for_mapping']
        sn_ids = batch['sn_ids']
        reader_ids = batch['reader_ids']

        # needed for attention visualisation
        subwords = [tokenizer.convert_ids_to_tokens(i) for i in sn_input_ids]

        sn_sp_emb, pos_enc, sn_input_ids_emb, fix_dur_emb = model.get_embeds(
            sn_sp_repr=sn_sp_repr,
            sn_input_ids=sn_input_ids,
            indices_pos_enc=indices_pos_enc,
            sn_sp_fix_dur=sn_sp_fix_dur,
        )

        x_start = sn_sp_emb + fix_dur_emb
        noise = torch.randn_like(x_start)
        mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())

        # replace the scan path with Gaussian noise
        x_noised = torch.where(mask == 0, x_start, noise)

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps // args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)
        samples_fix_loc, samples_fix_dur = sample_fn(
            model=model,
            shape=sample_shape,
            noise=x_noised,
            sn_input_ids_emb=sn_input_ids_emb,
            pos_enc=pos_enc,
            mask_sn_padding=None,
            mask_transformer_att=None,
            clip_denoised=args.clip_denoised,
            denoised_fn_fix_loc=partial(denoised_fn_round, args, sn_sp_repr_embedding),
            denoised_fn_fix_dur=partial(denoised_fn_round, args, fix_dur_embedding),
            model_kwargs=None,
            eta=0.0, # TODO eta=args.eta, default 0.0
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=clamp_first_bool,
            mask=mask,
            x_start=x_start,
            subwords_list=subwords,
            atten_vis=args.atten_vis,
            atten_vis_fn=None,
            atten_vis_path=out_path,
            batch_idx=batch_idx,
            gap=step_gap,
            rank=rank,
            atten_vis_sp=args.atten_vis_sp,
            langevin_fn=None,  # TODO should be defined somewhere. if it is provided, it is used for ddim, if none, it's not used
        )

        # len(samples) is the number of diffusion steps
        print(samples_fix_loc[0].shape)  # samples for each diffusion step; [batch_size, args.seq_len, hidden=768]

        # take the output of the very last denoising step
        sample_fix_loc = samples_fix_loc[-1]
        sample_fix_dur = samples_fix_dur[-1]

        print('decoding for seq2seq')
        print(sample_fix_dur.shape)

        logits_fix_loc = model.get_logits(sample_fix_loc)
        cands_fix_loc = torch.topk(logits_fix_loc, k=1, dim=-1)

        logits_fix_dur = model.get_logits_fix_dur(sample_fix_dur)
        cands_fix_dur = torch.topk(logits_fix_dur, k=1, dim=-1)

        predicted_sp_words_list_atten = list()

        for instance_idx, (pred_seq, orig_words, orig_ids, pred_fix, orig_fix, sn_len, sn_id, reader_id) in enumerate(
                zip(cands_fix_loc.indices, words_for_mapping, sn_sp_repr, cands_fix_dur.indices, sn_sp_fix_dur, sn_repr_len, sn_ids, reader_ids)):

            pred_seq_sp = pred_seq[sn_len:]
            orig_ids_sp = orig_ids[sn_len:]
            pred_fix_durs = pred_fix[sn_len:]
            orig_fix_durs = orig_fix[sn_len:]

            # # # cut off all trailing PAD tokens # # #

            words_split = orig_words.split()
            # map the predicted indices to the words
            predicted_sp = [words_split[i] for i in pred_seq_sp]
            # the predicted sp IDs
            pred_sp_ids = [e.item() for e in pred_seq_sp]
            # original scan path in words
            original_sp = [words_split[i] for i in orig_ids_sp]
            # the original sp IDs
            orig_sp_ids = [e.item() for e in orig_ids_sp]
            # the predicted fixation durations
            pred_fix_durs = [e.item() for e in pred_fix_durs]
            # the original fixation durations
            orig_fix_durs = [e.item() for e in orig_fix_durs]

            # # # cut off all trailing PAD tokens # # #

            while len(predicted_sp) > 1 and predicted_sp[-1] == '[PAD]':
                predicted_sp.pop()
            while len(pred_sp_ids) > 1 and pred_sp_ids[-1] == args.seq_len - 1:
                pred_sp_ids.pop()
            while len(original_sp) > 1 and original_sp[-1] == '[PAD]':
                original_sp.pop()
            while len(orig_sp_ids) > 1 and orig_sp_ids[-1] == args.seq_len - 1:
                orig_sp_ids.pop()
            while len(words_split) > 1 and words_split[-1] == '[PAD]':
                words_split.pop()
            # TODO this is not supposed to be the pad token; also SEP token removed by accident, bc also encoded as 0
            while len(pred_fix_durs) > 1 and pred_fix_durs[-1] == 0:
                pred_fix_durs.pop()
            while len(orig_fix_durs) > 1 and orig_fix_durs[-1] == 0:
                orig_fix_durs.pop()
            
            # add again one 0 to orig_fix_durs and pred_fix_durs because the SEP token is (should) also be encoded as 0
            orig_fix_durs.append(0)
            pred_fix_durs.append(0)

            predicted_sp_words_list_atten.append(predicted_sp)

            if args.sp_vis:
                # plot the true scanpath
                plot_scanpath_image(
                    sentence=' '.join(words_split),
                    scanpath=orig_sp_ids,
                    instance_id=instance_idx,
                    name='true',
                    rank=rank,
                    batch_id=batch_idx,
                    output_dir=os.path.join(out_path, 'sp_plots'),
                )
                # plot the predicted scanpath
                plot_scanpath_image(
                    sentence=' '.join(words_split),
                    scanpath=pred_sp_ids,
                    instance_id=instance_idx,
                    name='pred',
                    rank=rank,
                    batch_id=batch_idx,
                    output_dir=os.path.join(out_path, 'sp_plots'),
                )

            # predicted scan path in words
            predicted_sp_words_pad.append(' '.join(predicted_sp))
            # original scan path in words
            original_sp_words_pad.append(' '.join(original_sp))
            # predicted scan path IDs
            predicted_sp_ids_pad.append(pred_sp_ids)
            # true/gold label scan path IDs
            original_sp_ids_pad.append(orig_sp_ids)
            # predicted fixation durations
            predicted_fix_durs_pad.append(pred_fix_durs)
            # original fixation durations
            original_fix_durs_pad.append(orig_fix_durs)

            # add the original sn
            original_sn_pad.append(' '.join(words_split))

            # add the IDs to the list

            if args.inference == 'zuco':
                reader_ids_pad.append(reader_id)
                sn_ids_pad.append(sn_id.item())
            else:
                reader_ids_pad.append(reader_id.item())
                sn_ids_pad.append(sn_id)

            if batch_idx == 0 and instance_idx == 0:

                data_dict_initial_pad['predicted_sp_words'].append(' '.join(predicted_sp))
                data_dict_initial_pad['original_sp_words'].append(' '.join(original_sp))
                data_dict_initial_pad['predicted_sp_ids'].append(pred_sp_ids)
                data_dict_initial_pad['original_sp_ids'].append(orig_sp_ids)
                data_dict_initial_pad['predicted_fix_durs'].append(pred_fix_durs)
                data_dict_initial_pad['original_fix_durs'].append(orig_fix_durs)
                data_dict_initial_pad['original_sn'].append(' '.join(words_split))

                if args.inference == 'zuco':
                    data_dict_initial_pad['reader_ids'].append(reader_id)
                    data_dict_initial_pad['sn_ids'].append(sn_id.item())
                else:
                    data_dict_initial_pad['reader_ids'].append(reader_id.item())
                    data_dict_initial_pad['sn_ids'].append(sn_id)

                for i in range(world_size):
                    if i == rank:
                        with open(out_path_incrementally_pad, 'w') as f:
                            json.dump(data_dict_initial_pad, f)
                    dist.barrier()

            else:

                for i in range(world_size):
                    if i == rank:
                        with open(out_path_incrementally_pad, 'rb') as f:
                            loaded_data_dict = json.load(f)
                    dist.barrier()

                loaded_data_dict['predicted_sp_words'].append(' '.join(predicted_sp))
                loaded_data_dict['original_sp_words'].append(' '.join(original_sp))
                loaded_data_dict['predicted_sp_ids'].append(pred_sp_ids)
                loaded_data_dict['original_sp_ids'].append(orig_sp_ids)
                loaded_data_dict['predicted_fix_durs'].append(pred_fix_durs)
                loaded_data_dict['original_fix_durs'].append(orig_fix_durs)
                loaded_data_dict['original_sn'].append(' '.join(words_split))

                if args.inference == 'zuco':
                    loaded_data_dict['reader_ids'].append(reader_id)
                    loaded_data_dict['sn_ids'].append(sn_id.item())
                else:
                    loaded_data_dict['reader_ids'].append(reader_id.item())
                    loaded_data_dict['sn_ids'].append(sn_id)

                for i in range(world_size):
                    if i == rank:
                        with open(out_path_incrementally_pad, 'w') as f:
                            json.dump(loaded_data_dict, f)
                    dist.barrier()

        del batch
        del mask
        del sn_sp_repr
        del sn_input_ids
        del indices_pos_enc
        del sn_repr_len
        del words_for_mapping
        del x_start
        del noise
        del x_noised
        del samples_fix_loc
        del samples_fix_dur
        del sample_fix_loc
        del sample_fix_dur

        instance_ctr += args.batch_size
        if args.no_inst != 0:
            if instance_ctr >= args.no_inst:
                break

    data_dict_complete_pad = {
        'predicted_sp_words': predicted_sp_words_pad,
        'original_sp_words': original_sp_words_pad,
        'predicted_sp_ids': predicted_sp_ids_pad,
        'original_sp_ids': original_sp_ids_pad,
        'predicted_fix_durs': predicted_fix_durs_pad,
        'original_fix_durs': original_fix_durs_pad,
        'original_sn': original_sn_pad,
        'sn_ids': sn_ids_pad,
        'reader_ids': reader_ids_pad,
    }
    data_dict_complete_sep = {
        'predicted_sp_words': predicted_sp_words_sep,
        'original_sp_words': original_sp_words_sep,
        'predicted_sp_ids': predicted_sp_ids_sep,
        'original_sp_ids': original_sp_ids_sep,
        'predicted_fix_durs': predicted_fix_durs_sep,
        'original_fix_durs': original_fix_durs_sep,
        'original_sn': original_sn_sep,
        'sn_ids': sn_ids_sep,
        'reader_ids': reader_ids_sep,
    }

    for i in range(world_size):
        if i == rank:  # write files sequentially
            with open(out_path_all_pad, 'w') as f:
                json.dump(data_dict_complete_pad, f)
        dist.barrier()
    for i in range(world_size):
        if i == rank:  # write files sequentially
            with open(out_path_all_sep, 'w') as f:
                json.dump(data_dict_complete_sep, f)
        dist.barrier()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == '__main__':
    main()
