#!/bin/bash



# Common training parameters
nproc_per_node=3
master_port_base=12234  # Starting port number to avoid conflicts
script_module="scandl_fixdur.scandl_module.scripts.sp_run_train"
common_args=(
    --corpus celer
    --inference cv
    --load_train_data processed_data
    --num_transformer_heads 8
    --num_transformer_layers 12
    --hidden_dim 256
    --noise_schedule sqrt
    --learning_steps 80000
    --log_interval 500
    --eval_interval 500
    --save_interval 5000
)

# Training with different data split criteria
declare -A split_criteria=(
    ["sentence"]="cv"
    ["reader"]="cv"
    ["combined"]="cv"
    ["scanpath"]="zuco"  # Special case with inference mode 'zuco' and an extra note
)

# Function to run the training with specified criteria and additional args
run_training () {
    local split_criterion=$1
    local inference_mode=$2
    local extra_args=("${@:3}")
    local master_port=$((master_port_base++))

    echo "Running training with data_split_criterion=${split_criterion} and inference=${inference_mode}"

    python -m torch.distributed.launch \
        --nproc_per_node=$nproc_per_node \
        --master_port=$master_port \
        --use_env -m $script_module \
        "${common_args[@]}" \
        --inference $inference_mode \
        --data_split_criterion $split_criterion \
        "${extra_args[@]}"
}

# Execute training jobs
for criterion in "${!split_criteria[@]}"; do
    if [[ $criterion == "scanpath" ]]; then
        # Special case with cross-dataset note
        run_training "$criterion" "${split_criteria[$criterion]}" --notes cross_dataset
    else
        run_training "$criterion" "${split_criteria[$criterion]}"
    fi
done

echo "All training jobs completed."
