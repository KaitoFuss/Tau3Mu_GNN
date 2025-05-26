#!/bin/bash
# 
#SBATCH -p wn
#SBATCH --account=t3
#SBATCH --job-name=sc_bkg  
#SBATCH --mem=32000M      
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err    # replace default slurm-SLURM_JOB_ID.err 

# each node has local /scratch space to be used during job run
mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)
/work/kfuss/miniforge3/envs/gnn/bin/python score_per_file_example.py \
  --file /work/kfuss/SVJ_GNN/data/background/data_skimmed_no_resonances.root \
  --output_folder /work/kfuss/SVJ_GNN/output/knn_new_vs4/ \
  --model_name knn_new_vs4.pt \
  --arch arch6

# cleaning of temporal working dir when job was completed:
rm  -rf /scratch/$USER/${SLURM_JOB_ID}