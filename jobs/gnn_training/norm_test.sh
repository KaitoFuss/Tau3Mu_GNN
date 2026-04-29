#!/bin/bash
#
#SBATCH --job-name=norm_test
#SBATCH -p short               # to access gpu resources
#SBATCH --account=t3                                           
#SBATCH --mem=64000M                      # memory (per job) maybe 8000M later 
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out  # Specify the output log file location
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err  # Specify the error log file location

# each node has local /scratch space to be used during job run
mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)
/work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.scripts.Train_TauMuNet --epochs 10 --batch 128 --name normalisation_tests --model skimmed_dzd0_cone_size_0.5_knn_10 --lr 0.0016 --disco 4.0 --arch final_layer2 --tau_channel HF --gnn_mode final_layer --remark ""

# cleaning of temporal working dir when job was completed:
rm  -rf /scratch/$USER/${SLURM_JOB_ID}