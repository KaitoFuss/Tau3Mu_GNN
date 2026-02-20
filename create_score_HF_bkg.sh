#!/bin/bash
# 
#SBATCH -p gpu
#SBATCH --account=t3
#SBATCH --job-name=sc_HF_bkg
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err    # replace default slurm-SLURM_JOB_ID.err 

# each node has local /scratch space to be used during job run
mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)
/work/kfuss/miniforge3/envs/Tau3Mu/bin/python ScoreCreation.py \
  --root /pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF \
  --output_folder /work/kfuss/SVJ_GNN/output/early_stop_fix_HF_BDT_vs1/ \
  --model_name early_stop_fix_HF_BDT_vs1.pt \
  --arch arch6 \
  --tau_channel HF \
  --gnn_mode BDT

cleaning of temporal working dir when job was completed:
rm  -rf /scratch/$USER/${SLURM_JOB_ID}