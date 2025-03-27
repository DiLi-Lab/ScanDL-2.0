#!/bin/bash

# Define common parameters
SEED=62
SPLIT="test"
NO_GPUS=1
BSZ=12
EMTEC="--emtec"

# Run for "combined" setting
for FOLD in {0..4}; do
    python -m scandl_fixdur.scandl_module.scripts.sp_run_decode \
        --seed $SEED \
        --split $SPLIT \
        --no_gpus $NO_GPUS \
        --bsz $BSZ \
        --setting combined \
        --fold $FOLD \
        $EMTEC
done

# Run for "reader" setting
for FOLD in {0..4}; do
    python -m scandl_fixdur.scandl_module.scripts.sp_run_decode \
        --seed $SEED \
        --split $SPLIT \
        --no_gpus $NO_GPUS \
        --bsz $BSZ \
        --setting reader \
        --fold $FOLD \
        $EMTEC
done
