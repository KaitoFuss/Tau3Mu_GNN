#!/bin/bash
# 
#SBATCH -p standard
#SBATCH --account=t3
#SBATCH --job-name=BDT_W
#SBATCH --mem=64G
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err    # replace default slurm-SLURM_JOB_ID.err 

set -e

# each node has local /scratch space to be used during job run
JOB_SCRATCH=/scratch/$USER/$SLURM_JOB_ID
mkdir -p "$JOB_SCRATCH"
export TMPDIR="$JOB_SCRATCH"

echo "Using scratch: $JOB_SCRATCH"

source /work/kfuss/miniforge3/bin/activate Tau3Mu

LAMBDA=1.0
CHANNEL=W
GNN_MODE=BDT
MODEL=benchmark_W_"$GNN_MODE"_vs1
STORAGE_OUTPUT=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/

# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.bdt_training.Train_XGB_W --model "$MODEL"

/work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.bdt_training.BDT_ScoreProducer --model "$MODEL" \
    --classifier_iso /work/kfuss/SVJ_GNN/output/"$MODEL"/ISO_classifiers_24feb2026_v0.pck \
    --tau_channel $CHANNEL \
    --storage_folder $JOB_SCRATCH

# PLOT_MODE=tau_hist_after
# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --output_path /work/kfuss/SVJ_GNN/output/"$MODEL"/ --title "\$\lambda = "$LAMBDA"\$" --tau_channel "$CHANNEL" --gnn_mode "$GNN_MODE" --plot_mode $PLOT_MODE

# PLOT_MODE=tau_hist_after_fpr
# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --output_path /work/kfuss/SVJ_GNN/output/"$MODEL"/ --title "\$\lambda = "$LAMBDA"\$" --tau_channel "$CHANNEL" --gnn_mode "$GNN_MODE" --plot_mode $PLOT_MODE

echo "Copying results to /storage..."
xrdfs t3dcachedb03.psi.ch:1094 mkdir -p ${STORAGE_OUTPUT}${MODEL} 2>/dev/null || true
for file in "$JOB_SCRATCH/$MODEL"/*; do
    filename=$(basename "$file")
    echo "Removing existing $filename in $MODEL from storage if present..."
    xrdfs t3dcachedb03.psi.ch:1094 rm ${STORAGE_OUTPUT}${MODEL}/${filename} 2>/dev/null || true
    echo "Copying $filename in $MODEL to storage..."
    xrdcp --force "$file" \
        root://t3dcachedb03.psi.ch:1094/${STORAGE_OUTPUT}${MODEL}/${filename}
done

echo "Cleaning scratch..."
rm -rf "$JOB_SCRATCH"