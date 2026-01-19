#!/bin/bash
#SBATCH --job-name=pmo-short
#SBATCH --error=log/job_pmo_short_error.txt
#SBATCH --output=log/job_pmo_short_output.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable

module load python/3.10
module load cuda/12.4.1/cudnn  # Match your PyTorch version

# Activate venv (created *after* loading the above python module)
source ~/scratch/envs/rxnflow/bin/activate

oracle_array=(
        'osimertinib_mpo' 'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo' \
        'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop')

# Run all tasks in the background
CUDA_VISIBLE_DEVICES=0 python run.py smiles_gfn --oracle jnk3 --wandb online --run_name beta50_x2_rs --config_default hparams_rs.yaml --seed 0 &
CUDA_VISIBLE_DEVICES=1 python run.py smiles_gfn --oracle jnk3 --wandb online --run_name beta50_x2_rs --config_default hparams_rs.yaml --seed 1 &
CUDA_VISIBLE_DEVICES=2 python run.py smiles_gfn --oracle jnk3 --wandb online --run_name beta50_x2_rs --config_default hparams_rs.yaml --seed 2 &
# CUDA_VISIBLE_DEVICES=3 python run.py smiles_gfn --oracle median1 --wandb online --run_name rs_aux_x2_e3_debugged --config_default hparams_aux_rs2.yaml --seed 2 &

# CRITICAL: Wait for all background jobs to finish
wait