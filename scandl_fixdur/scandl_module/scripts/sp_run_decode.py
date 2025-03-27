import os
import sys
import glob
import argparse

from CONSTANTS import SCANDL_MODULE_TRAIN_PATH, SCANDL_MODULE_INF_PATH
from CONSTANTS import SCANDL_MODULE_TRAIN_PATH_EMTEC, SCANDL_MODULE_INF_PATH_EMTEC
from CONSTANTS import SCANDL_MODULE_TRAIN_PATH_BSC, SCANDL_MODULE_INF_PATH_BSC


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--model_dir', type=str, default='', help='path to the folder of diffusion model')
    parser.add_argument('--seed', type=int, default=101, help='random seed')
    parser.add_argument('--step', type=int, default=2000, help='if less than diffusion training steps, like 1000, use ddim sampling')

    parser.add_argument('--bsz', type=int, default=8, help='batch size')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='dataset split used to decode')

    parser.add_argument('--top_p', type=int, default=-1, help='top p used in sampling, default is off')
    parser.add_argument('--pattern', type=str, default='ema', help='training pattern')
    parser.add_argument('--cv', help='if given, inference is performed on models trained in k-fold CV.',
                        action='store_true',)
    parser.add_argument('--unique_sns', type=str, choices=['mixed', 'universal-only'],
                        help='in the reader split, this decides whether in the test set are both unique '
                             'and universal sentences or only universal sentences.',
                        default='mixed')
    parser.add_argument('--atten_vis', action='store_true',
                        help='if given, transformer attention for sn is visualised.')
    parser.add_argument('--tsne_vis', action='store_true',
                        help='if given, the denoising process is visualised at different denoising steps.')
    parser.add_argument('--sp_vis', action='store_true',
                        help='if given, the true and predicted scanpaths are plotted.')
    parser.add_argument('--no_inst', type=int, required=False, default=0,
                        help='if given, inference is stopped after after the specified number of instances. used for'
                             ' subsetting number of visualisations.')
    parser.add_argument('--atten_vis_sp', action='store_true',
                        help='if given, the attention heatmap is plotted for both sn and sp at t=0.')

    parser.add_argument(
        '--clamp_first',
        type=str,
        help='if yes, the model output is piped through the denoising fn',
        default='yes',
        choices=['yes', 'no']
    )
    parser.add_argument(
        '--run_only_on',
        type=str,
        help='for partial inference, i.e. not on all models and all folds. indicate path to specific model in specific fold.',
        required=False,
        default='',
    )
    parser.add_argument(
        '--notes',
        type=str,
        default='-',
        help='additional info to put in output name of folder',
        required=False,
    )
    parser.add_argument(
        '--no_gpus',
        type=int,
        required=False,
        default=5,
    )
    parser.add_argument(
        '--load_ids',
        type=str,
        default='-',
        help='if test ids saved during training, load them from this folder',
        required=False,
    )
    parser.add_argument(
        '--load_test_data',
        type=str,
        default='-',
        help='if given, load the test data from the specified path',
        required=False,
    )
    parser.add_argument(
        '--setting',
        type=str,
        choices=['reader', 'sentence', 'combined', 'cross_dataset'],
        required=True,
    )
    parser.add_argument(
        '--fold',
        type=int,
        choices=[0, 1, 2, 3, 4],
        required=True,
    )
    parser.add_argument(
        '--checkpoint-name',
        type=str,
        default='ema_0.9999_080000.pt',
    )
    parser.add_argument(
        '--emtec',
        action='store_true',
        help='if given, inference is performed on models trained on emtec data.',
    )
    parser.add_argument(
        '--bsc',
        action='store_true',
        help='if given, inference is performed on models trained on bsc data.',
    )

    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    # set working dir to the upper folder
    abspath = os.path.abspath(sys.argv[0])
    dname = os.path.dirname(abspath)
    dname = os.path.dirname(dname)
    os.chdir(dname)

    test_set_sns = args.unique_sns

    if args.emtec:
        train_path = SCANDL_MODULE_TRAIN_PATH_EMTEC
        inf_path = SCANDL_MODULE_INF_PATH_EMTEC
    elif args.bsc:
        train_path = SCANDL_MODULE_TRAIN_PATH_BSC
        inf_path = SCANDL_MODULE_INF_PATH_BSC
    else:
        train_path = SCANDL_MODULE_TRAIN_PATH
        inf_path = SCANDL_MODULE_INF_PATH

    if args.setting == 'cross_dataset':
        out_dir = os.path.join(inf_path, args.setting)
        model_path = os.path.join(train_path, args.setting, args.checkpoint_name)
    else:
        out_dir = os.path.join(inf_path, args.setting, f'fold-{args.fold}')
        model_path = os.path.join(train_path, args.setting, f'fold-{args.fold}', args.checkpoint_name)

    COMMAND = f'python -m torch.distributed.launch --nproc_per_node={args.no_gpus} --master_port={22233 + int(args.seed)} --use_env -m scripts.sp_sample_seq2seq ' \
                f'--model_path {model_path} --step {args.step} ' \
                f'--batch_size {args.bsz} --seed2 {args.seed} --split {args.split} ' \
                f'--out_dir {out_dir} --top_p {args.top_p} --clamp_first {args.clamp_first} ' \
                f'--test_set_sns {test_set_sns} --atten_vis {args.atten_vis} --notes {args.notes} ' \
                f'--tsne_vis {args.tsne_vis} --sp_vis {args.sp_vis} --no_inst {args.no_inst} ' \
                f'--atten_vis_sp {args.atten_vis_sp} --load_ids {args.load_ids} --load_test_data {args.load_test_data} ' \
                f'--setting {args.setting} --fold {args.fold}'
    print(COMMAND)

    os.system(COMMAND)
    print('#' * 30, 'decoding finished...')


if __name__ == '__main__':
    main()
