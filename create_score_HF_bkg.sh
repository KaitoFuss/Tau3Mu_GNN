#!/bin/bash
# 
#SBATCH -p standard
#SBATCH --account=t3
#SBATCH --job-name=sc_HF_bkg
#SBATCH --mem=32G
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err    # replace default slurm-SLURM_JOB_ID.err 

module purge
module load gcc/12.2.0

source /work/kfuss/miniforge3/bin/activate Tau3Mu

# each node has local /scratch space to be used during job run
mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

LAMBDA=1.0
CHANNEL=HF
GNN_MODE=BDT
PLOT_MODE=tau_hist
MODEL=correct_disco_1_HF_"$GNN_MODE"_vs1

# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)
/work/kfuss/miniforge3/envs/Tau3Mu/bin/python ScoreCreation.py \
  --root /pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF \
  --output_folder /work/kfuss/SVJ_GNN/output/"$MODEL"/ \
  --model_name "$MODEL".pt \
  --arch arch_optuna \
  --tau_channel $CHANNEL \
  --gnn_mode $GNN_MODE

/work/kfuss/miniforge3/envs/Tau3Mu/bin/python eval_utils/ModelEval.py --output_path /work/kfuss/SVJ_GNN/output/"$MODEL"/ --title "\$\lambda = "$LAMBDA"\$" --tau_channel "$CHANNEL" --gnn_mode "$GNN_MODE" --plot_mode $PLOT_MODE

cleaning of temporal working dir when job was completed:
rm  -rf /scratch/$USER/${SLURM_JOB_ID}