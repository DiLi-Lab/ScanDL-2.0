#!/bin/bash


# train on EMTeC for the paragraph-level ScanDL 2.0
python -m scandl2_pkg.fix_dur_module.train_seq2seq \
    --max-length 352 \
    --num-epochs 600 \
    --num-heads 12 \
    --num-layers 12 \
    --num-linear 8 \
    --bsz 48 \
    --dropout 0.5 \
    --sp-pad-token 351 \
    --use-attention-mask \
    --data emtec


# train on CELER for the sentence-level ScanDL 2.0
python -m scandl2_pkg.fix_dur_module.train_seq2seq \
    --max-length 128 \
    --num-epochs 400 \
    --num-heads 12 \
    --num-layers 12 \
    --num-linear 8 \
    --bsz 128 \
    --dropout 0.5 \
    --sp-pad-token 127 \
    --use-attention-mask \
    --data celer 