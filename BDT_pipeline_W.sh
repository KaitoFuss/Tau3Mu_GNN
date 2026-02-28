#!/bin/bash
# 
#SBATCH -p standard
#SBATCH --account=t3
#SBATCH --job-name=sc_W_bkg
#SBATCH --mem=32G
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err    # replace default slurm-SLURM_JOB_ID.err 

MODEL=adrian_config_W_BDT_vs1
LAMBDA=0.625
CHANNEL=W
GNN_MODE=BDT
PLOT_MODE=tau_hist_after

python bdt/Train_XGB_HF.py --model "$MODEL"

python utils/BDT_ScoreProducer.py --model "$MODEL" \
    --classifier_gnn /work/kfuss/SVJ_GNN/output/"$MODEL"/GNN_classifiers_24feb2026_v0.pck \
    --classifier_iso /work/kfuss/SVJ_GNN/output/"$MODEL"/ISO_classifiers_24feb2026_v0.pck \
    --tau_channel $CHANNEL

/work/kfuss/miniforge3/envs/Tau3Mu/bin/python eval_utils/ModelEval.py --output_path /work/kfuss/SVJ_GNN/output/"$MODEL"/ --title "\$\lambda = "$LAMBDA"\$" --tau_channel "$CHANNEL" --gnn_mode "$GNN_MODE" --plot_mode $PLOT_MODE
