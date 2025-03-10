#!/bin/bash
#SBATCH --job-name=stableprompt_fewshot
#SBATCH --output=logs/%j_output.log
#SBATCH --gres=gpu:4          # Request 2 GPUs
#SBATCH --cpus-per-task=32      # Adjust based on workload
#SBATCH --mem=64G              # Adjust memory
#SBATCH --time=48:00:00        # Set job duration
#SBATCH --partition=gpu-l40        # Adjust if needed for your cluster
#SBATCH --account=zlab

# Activate virtual environment if needed
conda activate stableprompt

# Run the Bash script that executes Python scripts sequentially
bash run_fewshot.sh
