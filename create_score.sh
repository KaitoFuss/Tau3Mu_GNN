#!/bin/bash
# 
#SBATCH -p gpu
#SBATCH --account=t3
#SBATCH --job-name=sc_bkg
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -o /work/adcorel/SVJ_GNN/logs/%x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e /work/adcorel/SVJ_GNN/logs/%x-%j.err    # replace default slurm-SLURM_JOB_ID.err 

# each node has local /scratch space to be used during job run
mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)
/work/adcorel/miniforge3/envs/gnn/bin/python score_per_file_example.py \
  --root /pnfs/psi.ch/cms/trivcat/store/user/adcorel/riccdata/data_ds \
  --output_folder /work/adcorel/SVJ_GNN/output/puppi_optunabest_newdata_with_CHAT_vs51/ \
  --model_name puppi_optunabest_newdata_with_CHAT_vs51.pt \
  --arch arch_optuna

cleaning of temporal working dir when job was completed:
rm  -rf /scratch/$USER/${SLURM_JOB_ID}