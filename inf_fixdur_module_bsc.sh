#!/bin/bash

# Script module
script_module="scandl_fixdur.fix_dur_module.inf_seq2seq"

# List of settings for inference
settings=(
    "reader" 
    #"combined"
    )

# Function to run inference with a specified setting
run_inference () {
    local setting=$1
    echo "Running inference with setting=${setting}"
    python -m $script_module --setting $setting --bsc
}

# Execute inference jobs for each setting
for setting in "${settings[@]}"; do
    run_inference "$setting"
done

echo "All inference jobs completed."