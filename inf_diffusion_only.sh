#!/bin/bash



python -m diffusion_only.scripts.sp_run_decode_hp \
    --seed 60 \
    --split test \
    --cv \
    --no_gpus 1 \
    --step 4000 \
    --bsz 6 