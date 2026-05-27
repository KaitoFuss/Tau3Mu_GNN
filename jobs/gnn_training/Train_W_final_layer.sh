#!/bin/bash
#
#SBATCH --job-name=tau3mu_W_final_layer
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=5                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=150000M                      # memory (per job) maybe 8000M later 
#SBATCH --time 3-00:00:00
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out  # Specify the output log file location
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err  # Specify the error log file location

# each node has local /scratch space to be used during job run
mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

source /work/kfuss/miniforge3/bin/activate Tau3Mu

# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)
python -m tau3mu_gnn.scripts.Train_TauMuNet \
    --epochs 1 \
    --batch 128 \
    --name test_pt_ft \
    --model skimmed_with_bdt_iso_dzd0_cone_size_0.5_knn_10_delete_muons \
    --lr 0.0016 \
    --disco 0.0 \
    --arch fl8_sd4 \
    --tau_channel W \
    --gnn_mode final_layer \
    --unsculpting_mode delete_muons \
    --remark "ft_0.8_cut" \
    --finetune \
    --finetune_lr 1e-4 \
    --finetune_epochs 1 \
    --bdt_threshold 0.8



# cleaning of temporal working dir when job was completed:
rm  -rf /scratch/$USER/${SLURM_JOB_ID}