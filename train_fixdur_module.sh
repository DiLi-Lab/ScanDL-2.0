#!/bin/bash

# Common training parameters
script_module="scandl_fixdur.fix_dur_module.train_seq2seq"
common_args=(
    --max-length 128
    --num-epochs 400
    --num-heads 12
    --num-layers 12
    --num-linear 8
    --bsz 128
    --dropout 0.5
    --sp-pad-token 127
    --use-attention-mask
)

# List of settings to iterate over
settings=("reader" "sentence" "combined" "cross_dataset")

# Function to run training with a specified setting
run_training () {
    local setting=$1
    echo "Running training with setting=${setting}"
    python -m $script_module "${common_args[@]}" --setting $setting
}

# Execute training jobs for each setting
for setting in "${settings[@]}"; do
    run_training "$setting"
done

echo "All training jobs completed."
