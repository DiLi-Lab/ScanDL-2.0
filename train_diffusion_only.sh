#!/bin/bash


python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=12233 \
    --use_env diffusion_only/scripts/sp_run_train_hp.py \
    --corpus celer \
    --inference cv \
    --num_transformer_heads 2 \
    --num_transformer_layers 16 \
    --hidden_dim 256 \
    --noise_schedule linear \
    --schedule_sampler fixstep \
    --bsz 64 \
    --learning_steps 120000 \
    --log_interval 1000 \
    --eval_interval 1000 \
    --save_interval 10000 \
    --data_split_criterion reader \
    --diff_steps 4000 
