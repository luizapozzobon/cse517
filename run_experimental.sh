#!/bin/bash
# dataset='sst2'

# alias deepspeed="python -m deepspeed.launcher.launch"
#!/bin/bash


datasets=(
    'sst2'
)
# deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset 'sst2' --epoch 32   --lr 5e-6 --beta 1e-6 --agent_model  meta-llama/Llama-3.2-3B-Instruct

for dataset in "${datasets[@]}"
do
    deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset $dataset  --epoch 32   --lr 1e-6 --beta 1e-5 --agent_model  "google/gemma-7b-it"
    deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset $dataset  --epoch 32   --lr 1e-6 --beta 1e-5 --agent_model  "meta-llama/Llama-3.2-3B-Instruct"
    deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset $dataset  --epoch 32   --lr 1e-6 --beta 1e-5 --agent_model  "meta-llama/Llama-3.2-1B-Instruct"
done



#deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset $dataset --epoch 32   --lr 1e-6 --beta 1e-4 --agent_model  meta-llama/Llama-3.2-3B-Instruct

#deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset $dataset --epoch 32  --lr 5e-6 --beta 1e-5 --agent_model  meta-llama/Llama-3.2-3B-Instruct
#deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset $dataset --epoch 32   --lr 5e-6 --beta 1e-4 --agent_model  meta-llama/Llama-3.2-3B-Instruct

#$deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset $dataset --epoch 32   --lr 1e-6 --beta 1e-5 --agent_model  meta-llama/Llama-3.2-1B-Instruct
#$deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset $dataset --epoch 32  --lr 1e-6 --beta 1e-4 --agent_model  meta-llama/Llama-3.2-1B-Instruct

#$deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset $dataset --epoch 32   --lr 5e-6 --beta 1e-5 --agent_model  meta-llama/Llama-3.2-1B-Instruct
#$deepspeed --num_gpus=8 stableprompt_tc_grpo.py --dataset $dataset --epoch 32   --lr 5e-6 --beta 1e-4 --agent_model  meta-llama/Llama-3.2-1B-Instruct

