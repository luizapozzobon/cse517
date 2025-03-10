#!/bin/bash
#SBATCH --job-name=stableprompt_II
#SBATCH --output=logs/%j_output.log
#SBATCH --gres=gpu:4          # Request 2 GPUs
#SBATCH --cpus-per-task=32      # Adjust based on workload
#SBATCH --mem=64G              # Adjust memory
#SBATCH --time=24:00:00        # Set job duration
#SBATCH --partition=ckpt-all
#SBATCH --account=ark
#SBATCH --requeue
#SBATCH --nodes=1

#SBATCH --constraint=a100|l40|l40s|a40

# Activate virtual environment if needed
conda activate stableprompt

# Run the Bash script that executes Python scripts sequentially
bash run_II.sh
