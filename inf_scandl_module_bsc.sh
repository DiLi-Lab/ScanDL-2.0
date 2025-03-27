#!/bin/bash

# Define common parameters
SEED=61
SPLIT="test"
NO_GPUS=1
BSZ=12
BSC="--bsc"

# Run for "combined" setting
echo "Running combined setting..."
for FOLD in {0..4}; do
    python -m scandl_fixdur.scandl_module.scripts.sp_run_decode \
        --seed $SEED \
        --split $SPLIT \
        --no_gpus $NO_GPUS \
        --bsz $BSZ \
        --setting combined \
        --fold $FOLD \
        $BSC
done

# Run for "reader" setting
echo "Running reader setting..."
for FOLD in {0..4}; do
    python -m scandl_fixdur.scandl_module.scripts.sp_run_decode \
        --seed $SEED \
        --split $SPLIT \
        --no_gpus $NO_GPUS \
        --bsz $BSZ \
        --setting reader \
        --fold $FOLD \
        $BSC
done