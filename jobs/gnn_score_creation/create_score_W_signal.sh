#!/bin/bash
# 
#SBATCH -p long
#SBATCH --account=t3
#SBATCH --job-name=sc_W_sig
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

CHANNEL=W
# GNN_MODE=final_layer
GNN_MODE=final_layer
MAX_EVENTS_TO_PROCESS=""
UNSCULPTING_MODE=delete_muons
BDT_PRESELECTION_CUT_ENABLED=true
BDT_PRESELECTION_CUT_VALUE="0.7"

if [ "$BDT_PRESELECTION_CUT_ENABLED" = true ] && [ -n "$BDT_PRESELECTION_CUT_VALUE" ]; then
    BDT_PRESELECTION_ARG="--bdt_preselection_cut $BDT_PRESELECTION_CUT_VALUE"
    ROOT_DIR=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/bdt_full
else
    BDT_PRESELECTION_ARG=""
    ROOT_DIR=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W
fi

# build the optional argument
if [ -n "$MAX_EVENTS_TO_PROCESS" ]; then
    MAX_EVENTS_ARG="--max_events_to_process $MAX_EVENTS_TO_PROCESS"
else
    MAX_EVENTS_ARG=""
fi

# Add multiple models here
MODELS=(
    "pt_ft_W_fl_8_sd_4_GINE_delete_muons_small_ft_bdt_cut_0.7_vs1"
    # "600k_disco0_permuted_W_fl_8_sd_4_GINE_vs1"
    # "new_hope_W_fl_8_sd_4_GINE_permutation_disco1_vs4"
    # "new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2"
    # "new_hope_W_fl_8_sd_4_GINE_delete_muons_BDT_cut_0.7_vs1"
    # "new_dropout_disco0_permuted_W_fl_8_sd_4_vs1"
    # "new_dropout_disco10_W_fl_8_sd_4_vs1"
    # "new_dropout_disco1_permuted_W_fl_8_sd_4_vs1"
    # "new_dropout_disco0_permuted_transformer_W_fl_8_sd_4_vs1"
    # "new_dropout_disco0_permuted_GAT_W_fl_8_sd_4_vs1"
    # "new_dropout_disco4_W_fl_6_sd_3_vs1"
    # "new_dropout_disco4_W_fl_7_sd_5_vs1"
    # "new_dropout_disco4_W_fl_7_sd_16_vs1"
    # "new_dropout_disco1_W_fl_11_sd_12_vs1"
    # "limit_disco1_W_fl_7_sd_6_vs1"
    # "new_dropout_disco1_W_fl_1_sd_8_vs1"
    # "new_dropout_disco1_W_fl_2_sd_13_vs1"
    # "new_dropout_disco1_W_fl_7_sd_10_vs1"
    # "new_dropout_disco1_W_fl_8_sd_4_vs1"
    # "new_dropout_disco1_W_fl_9_sd_9_vs1"
    # "scale_down_disco1_W_${GNN_MODE}_vs4"
    # "scale_down_disco1_W_${GNN_MODE}_vs5"
    # "scale_down_disco1_W_${GNN_MODE}_vs6"
    # "scale_down_disco1_W_${GNN_MODE}_vs3"
    # "benchmark_W_${GNN_MODE}_vs1"
    # "final_layer_archs_disco2_W_${GNN_MODE}_vs2"
    # "final_layer_archs_disco2_W_${GNN_MODE}_vs4"
    # "final_layer_archs_disco4_W_${GNN_MODE}_vs4"
    # "final_layer_archs_disco1_W_${GNN_MODE}_vs5"
    # "final_layer_archs_disco3_W_${GNN_MODE}_vs1"  # add more as needed
)
ARCHS=(
    "random"
    # "random"
    # "random"
    # "random"
    # "random"
    # "random"
    # "random"
    # "random"
    # "arch_optuna"
    # "scale_down4"
    # "scale_down5"
    # "scale_down6"
    # "scale_down3"
    # "final_layer4"
    # "final_layer4"
    # "final_layer5"
    # "$ARCH"  # one per model
)
CONV_TYPE=(
  "GINE"
  # "Transformer"
  # "GAT"
)

# Build --model_names and --archs arguments
MODEL_NAMES_ARGS=""
ARCHS_ARGS=""
CONV_TYPES_ARGS=""
for MODEL in "${MODELS[@]}"; do
    MODEL_NAMES_ARGS="$MODEL_NAMES_ARGS $MODEL"
    mkdir -p "$JOB_SCRATCH/$MODEL"
done
for ARCH_ENTRY in "${ARCHS[@]}"; do
    ARCHS_ARGS="$ARCHS_ARGS $ARCH_ENTRY"
done
for CONV_TYPE_ENTRY in "${CONV_TYPE[@]}"; do
    CONV_TYPES_ARGS="$CONV_TYPES_ARGS $CONV_TYPE_ENTRY"
done

# SCRATCH_OUTPUT="$JOB_SCRATCH"/"$MODEL"
STORAGE_OUTPUT=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/

# python program script.py should use CUDA_VISIBLE_DEVICES variable (*NOT* hardcoded GPU's numbers)
/work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.scripts.ScoreCreation \
  --root $ROOT_DIR \
  --output_folder /work/kfuss/SVJ_GNN/output/ \
  --model_names $MODEL_NAMES_ARGS \
  --archs $ARCHS_ARGS \
  --tau_channel $CHANNEL \
  --gnn_mode $GNN_MODE \
  --storage_folder $JOB_SCRATCH \
  --conv_types $CONV_TYPES_ARGS \
  --unsculpting_mode $UNSCULPTING_MODE \
  $MAX_EVENTS_ARG \
  $BDT_PRESELECTION_ARG


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