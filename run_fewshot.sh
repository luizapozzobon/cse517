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
    python stableprompt_tc.py --target_model google/gemma-1.1-7b-it --agent_model google/gemma-1.1-7b-it --dataset $dataset --epoch 100 --update_term 5 --cache_dir ./cache
done
