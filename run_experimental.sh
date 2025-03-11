#!/bin/bash
datasets=(
    'sst2'
)

# Loop through each main dataset and run the script
for dataset in "${datasets[@]}"
do
    deepspeed --num_gpus=4 stableprompt_tc_grpo.py --dataset $dataset --epoch 512 --update_term 1
done