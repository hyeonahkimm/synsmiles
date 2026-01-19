#!/bin/bash
#SBATCH --job-name=seh-real-rs
#SBATCH --error=log/job_seh_real_rs_error.txt
#SBATCH --output=log/job_seh_real_rs_output.txt
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=unkillable

module load python/3.10
module load cuda/12.4.1/cudnn  # Match your PyTorch version

# Activate venv (created *after* loading the above python module)
source ~/scratch/envs/rxnflow/bin/activate

# python train.py --oracle SEH --beta 50 --n_steps 5000 --batch_size 64 --n_replay 1 --replay_batch_size 64 --n_warmup_steps 100 --sa_threshold 3 --filter_unsynthesizable --aux_loss relative_logp --sigmoid_alpha 1.0 --neg_coefficient 0.001 --init_z 0 --lr_z 0.001 --rtb --buffer_size 6400 --seed 0 --use_retrosynthesis --retro_env stock_hb --max_retro_steps 3 --run_name hb3-beta50-sampling-relative-logp-e3 --replace_sampling --wandb online
python train.py --oracle SEH --beta 25 --n_steps 5000 --batch_size 64 --n_replay 1 --replay_batch_size 64 --n_warmup_steps 100 --sa_threshold 3 --init_z 0 --lr_z 0.001 --rtb --buffer_size 6400 --reshape_reward --wandb online --run_name real3-beta25-rs-sampling --seed 0 --use_retrosynthesis --retro_env stock --max_retro_steps 3 --replace_sampling
# python train.py --oracle SEH --beta 25 --n_steps 5000 --batch_size 64 --n_replay 1 --replay_batch_size 64 --n_warmup_steps 100 --sa_threshold 3 --init_z 0 --lr_z 0.001 --rtb --buffer_size 6400 --reshape_reward --wandb online --run_name hb3-beta25-rs-sampling --seed 1 --use_retrosynthesis --retro_env stock_hb --max_retro_steps 3 --replace_sampling
