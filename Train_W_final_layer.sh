#!/bin/bash
#
#SBATCH --job-name=tau3mu_W_final_layer
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=5                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=64000M                      # memory (per job) maybe 8000M later 
#SBATCH --time=0-12:00                   # time  in format DD-HH:MM
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out  # Specify the output log file location
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err  # Specify the error log file location

# each node has local /scratch space to be used during job run
mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)
/work/kfuss/miniforge3/envs/Tau3Mu/bin/python Train_TauMuNet.py --epochs 1000 --batch 128 --name correct_disco_10 --model skimmed_dzd0_cone_size_0.5_knn_10 --lr 0.0016 --disco 10.0 --arch arch_optuna --tau_channel W --gnn_mode final_layer --remark ""

# cleaning of temporal working dir when job was completed:
rm  -rf /scratch/$USER/${SLURM_JOB_ID}