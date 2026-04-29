#!/bin/bash
# 
#SBATCH --account=t3
#SBATCH --job-name=Graph_W
#SBATCH --mem=128000M     
#SBATCH --time 03:00:00
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err    # replace default slurm-SLURM_JOB_ID.err 

# each node has local /scratch space to be used during job run
mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)
source /work/kfuss/miniforge3/etc/profile.d/conda.sh
conda activate Tau3Mu

python -m tau3mu_gnn.scripts.GraphCreation --graph_type knn --k 10 --tau_channel W

# cleaning of temporal working dir when job was completed:
rm  -rf /scratch/$USER/${SLURM_JOB_ID}