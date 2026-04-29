#!/bin/bash
# 
#SBATCH -p long
#SBATCH --account=t3
#SBATCH --job-name=sc_HF_sig
#SBATCH --mem=64G
#SBATCH --time 2-00:00:00
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err    # replace default slurm-SLURM_JOB_ID.err 

voms-proxy-info --exists --valid 12:00
if [ $? -ne 0 ]; then
    echo "ERROR: No valid proxy found. Run voms-proxy-init --voms cms --valid 192:00 before submitting."
    exit 1
fi

module purge
module load gcc/12.2.0

source /work/kfuss/miniforge3/bin/activate Tau3Mu

# each node has local /scratch space to be used during job run
JOB_SCRATCH=/scratch/$USER/$SLURM_JOB_ID
mkdir -p "$JOB_SCRATCH"
export TMPDIR="$JOB_SCRATCH"

echo "Using scratch: $JOB_SCRATCH"

CHANNEL=HF
GNN_MODE=final_layer
MAX_EVENTS_TO_PROCESS=""

# build the optional argument
if [ -n "$MAX_EVENTS_TO_PROCESS" ]; then
    MAX_EVENTS_ARG="--max_events_to_process $MAX_EVENTS_TO_PROCESS"
else
    MAX_EVENTS_ARG=""
fi

# Add multiple models here
MODELS=(
    "new_dropout_disco4_HF_fl_10_sd_3_vs1"
    "new_dropout_disco4_HF_fl_11_sd_1_vs1"
    "new_dropout_disco4_HF_fl_12_sd_9_vs1"
    # "scale_down_disco4_HF_${GNN_MODE}_vs4"
    # "scale_down_disco4_HF_${GNN_MODE}_vs5"
    # "scale_down_disco4_HF_${GNN_MODE}_vs6"
    # "benchmark_HF_${GNN_MODE}_vs1"
    # "final_layer_archs_disco3_HF_${GNN_MODE}_vs2"
    # "final_layer_archs_disco3_HF_${GNN_MODE}_vs4"
    # "final_layer_archs_disco4_HF_${GNN_MODE}_vs3"
    # "final_layer_archs_disco3_HF_${GNN_MODE}_vs1"  # add more as needed
)
ARCHS=(
    "random"
    "random"
    "random"
    # "scale_down4"
    # "scale_down5"
    # "scale_down6"
    # "arch_optuna"
    # "final_layer2"
    # "final_layer4"
    # "final_layer3"
    # "$ARCH"  # one per model
)

# Build --model_names and --archs arguments
MODEL_NAMES_ARGS=""
ARCHS_ARGS=""
for MODEL in "${MODELS[@]}"; do
    MODEL_NAMES_ARGS="$MODEL_NAMES_ARGS $MODEL"
    mkdir -p "$JOB_SCRATCH/$MODEL"
done
for ARCH_ENTRY in "${ARCHS[@]}"; do
    ARCHS_ARGS="$ARCHS_ARGS $ARCH_ENTRY"
done

# SCRATCH_OUTPUT="$JOB_SCRATCH"/"$MODEL"
STORAGE_OUTPUT=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_HF/

# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)
/work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.scripts.ScoreCreation \
  --root /pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_HF \
  --output_folder /work/kfuss/SVJ_GNN/output/ \
  --model_names $MODEL_NAMES_ARGS \
  --archs $ARCHS_ARGS \
  --tau_channel $CHANNEL \
  --gnn_mode $GNN_MODE \
  --storage_folder $JOB_SCRATCH \
  $MAX_EVENTS_ARG

echo "Copying results to /storage..."

for MODEL in "${MODELS[@]}"; do
    echo "Removing existing $MODEL from storage if present..."
    xrdfs t3dcachedb03.psi.ch:1094 rm -r ${STORAGE_OUTPUT}${MODEL} 2>/dev/null || true
    echo "Copying $MODEL to storage..."
    xrdcp -r --force "$JOB_SCRATCH/$MODEL" \
        root://t3dcachedb03.psi.ch:1094/${STORAGE_OUTPUT}
done

echo "Cleaning scratch..."
rm -rf "$JOB_SCRATCH"