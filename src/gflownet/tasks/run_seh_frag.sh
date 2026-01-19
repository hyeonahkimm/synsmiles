#!/bin/bash
#SBATCH --job-name=frag-seh-aux2
#SBATCH --error=log/job_frag_seh_aux2_error.txt
#SBATCH --output=log/job_frag_seh_aux2_output.txt
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=unkillable

module load python/3.10
module load cuda/12.4.1/cudnn  # Match your PyTorch version

# Activate venv (created *after* loading the above python module)
source ~/scratch/envs/rxnflow/bin/activate

python seh_synth.py --wandb_run_name seh-hb3-aux3-e2-sum-replay
