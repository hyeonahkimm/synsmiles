#!/bin/bash
#SBATCH --job-name=aldh-aux
#SBATCH --error=log/job_aldh_aux_error.txt
#SBATCH --output=log/job_aldh_aux_output.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=main

module load python/3.10
module load cuda/12.4.1/cudnn  # Match your PyTorch version

# Activate venv (created *after* loading the above python module)
source ~/scratch/envs/rxnflow/bin/activate

export LD_LIBRARY_PATH=/home/mila/k/kimh/.conda/pkgs/libboost-1.85.0-hba137d9_2/lib:$LD_LIBRARY_PATH

python train.py --oracle vina --vina_receptor ALDH1 --beta 25 --neg_coefficient 0.001 --use_retrosynthesis --retro_env stock_hb --max_retro_steps 3 --filter_unsynthesizable --aux_loss relative_logp --run_name hb3-beta25-aux-e3 --seed 0 --wandb online
