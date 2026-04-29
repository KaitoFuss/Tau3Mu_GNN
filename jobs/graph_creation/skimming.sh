#!/bin/bash
#SBATCH --job-name=skimmer
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err    # replace default slurm-SLURM_JOB_ID.err 
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

INPUT='/work/kfuss/SVJ_GNN/src/tau3mu_gnn/data/signal_W/signal_skimmed_no_resonances.root'
OUTPUT='/work/kfuss/SVJ_GNN/src/tau3mu_gnn/data/signal_W/selected_signal_skimmed_no_resonances.root'

source /work/kfuss/miniforge3/bin/activate Tau3Mu
/work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.utils.skimmer \
  --input "${INPUT}" \
  --output "${OUTPUT}"