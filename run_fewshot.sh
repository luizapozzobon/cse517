#!/bin/bash
datasets=(
    'sst2'
    'mrpc'
    'rte'
    'mnli'
    'qnli'
    'snli'
)

# Loop through each main dataset and run the script
for dataset in "${datasets[@]}"
do
    python stableprompt_tc.py --dataset $dataset --epoch 100 --update_term 5 --cache_dir ./cache
done

# --agent_model google/gemma-1.1-2b-it --target_model google/gemma-1.1-2b-it
