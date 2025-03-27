#!/bin/bash



# train on EMTeC for the paragraph-level ScanDL 2.0
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=12234 \
    --use_env -m scandl2_pkg.scandl_module.scripts.sp_run_train \
    --corpus emtec \
    --inference cv \
    --load_train_data processed_data_all_emtec \
    --num_transformer_heads 8 \
    --num_transformer_layers 12 \
    --hidden_dim 256 \
    --noise_schedule sqrt \
    --learning_steps 80000 \
    --log_interval 500 \
    --eval_interval 500 \
    --save_interval 5000 \
    --seq_len 352 \
    --hidden_t_dim 352



# train on CELER for the sentence-level ScanDL 2.0
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=12234 \
    --use_env -m scandl2_pkg.scandl_module.scripts.sp_run_train \
    --corpus celer \
    --inference cv \
    --load_train_data processed_data_all_celer \
    --num_transformer_heads 8 \
    --num_transformer_layers 12 \
    --hidden_dim 256 \
    --noise_schedule sqrt \
    --learning_steps 80000 \
    --log_interval 500 \
    --eval_interval 500 \
    --save_interval 5000 \
    --seq_len 128 \
    --hidden_t_dim 128
