#!/bin/bash

# Define parameters
SEED=62
SPLIT="test"
NO_GPUS=1
BSZ=16

# Run commands with setting "reader" and folds 0-4
for FOLD in {0..4}; do
    python -m scandl_fixdur.scandl_module.scripts.sp_run_decode \
        --seed $SEED \
        --split $SPLIT \
        --no_gpus $NO_GPUS \
        --bsz $BSZ \
        --setting reader \
        --fold $FOLD
done

# Run commands with setting "sentence" and folds 0-4
for FOLD in {0..4}; do
    python -m scandl_fixdur.scandl_module.scripts.sp_run_decode \
        --seed $SEED \
        --split $SPLIT \
        --no_gpus $NO_GPUS \
        --bsz $BSZ \
        --setting sentence \
        --fold $FOLD
done

# Run commands with setting "combined" and folds 0-4
for FOLD in {0..4}; do
    python -m scandl_fixdur.scandl_module.scripts.sp_run_decode \
        --seed $SEED \
        --split $SPLIT \
        --no_gpus $NO_GPUS \
        --bsz $BSZ \
        --setting combined \
        --fold $FOLD
done

# Run final command with setting "cross_dataset"
python -m scandl_fixdur.scandl_module.scripts.sp_run_decode \
    --seed $SEED \
    --split $SPLIT \
    --no_gpus $NO_GPUS \
    --bsz $BSZ \
    --setting cross_dataset \
    --fold 0
