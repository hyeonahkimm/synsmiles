#!/bin/bash
#SBATCH --job-name=pmo-mut
#SBATCH --error=log/job_pmo_mut_error.txt
#SBATCH --output=log/job_pmo_mut_output.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=long

module load python/3.10
module load cuda/12.4.1/cudnn  # Match your PyTorch version

# Activate venv (created *after* loading the above python module)
source ~/scratch/envs/rxnflow/bin/activate


oracle_array=('jnk3' 'drd2' 'gsk3b' )

for seed in {0..2}
do
for oralce in "${oracle_array[@]}"
do
python run.py smiles_gfn --oracle $oralce --wandb online --run_name real3_aux_x2_e3 --seed $seed --config_default hparams_default.yaml
done
done
