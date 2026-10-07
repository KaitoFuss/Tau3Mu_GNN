#!/bin/bash
CHANNEL=HF
GNN_MODE=final_layer

MODELS=(
    "final_layer_archs_disco2_HF_${GNN_MODE}_vs2"
    "final_layer_archs_disco2_HF_${GNN_MODE}_vs5"
    "final_layer_archs_disco3_HF_${GNN_MODE}_vs4"
)
ARCHS=(
    "final_layer2"
    "final_layer5"
    "final_layer4"
)
LAMBDAS=(
  2.0
  2.0
  3.0
)

JOB_SCRATCH=work/kfuss/SVJ_GNN/tmp/sc_HF_bkg_local
mkdir -p "$JOB_SCRATCH"
export TMPDIR="$JOB_SCRATCH"

MODEL_NAMES_ARGS=""
ARCHS_ARGS=""
for MODEL in "${MODELS[@]}"; do
    MODEL_NAMES_ARGS="$MODEL_NAMES_ARGS $MODEL"
    mkdir -p "$JOB_SCRATCH/$MODEL"
done
for ARCH_ENTRY in "${ARCHS[@]}"; do
    ARCHS_ARGS="$ARCHS_ARGS $ARCH_ENTRY"
done

STORAGE_OUTPUT=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/

python -m tau3mu_gnn.scripts.ScoreCreation \
  --root /pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF \
  --output_folder /work/kfuss/SVJ_GNN/output/ \
  --model_names $MODEL_NAMES_ARGS \
  --archs $ARCHS_ARGS \
  --tau_channel $CHANNEL \
  --gnn_mode $GNN_MODE \
  --storage_folder $JOB_SCRATCH

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  LAMBDA="${LAMBDAS[$i]}"
  for PLOT_MODE in tau_hist_fpr tau_hist; do
    python -m tau3mu_gnn.model_eval.ModelEval \
      --output_path /work/kfuss/SVJ_GNN/output/ \
      --title "\$\lambda = ${LAMBDA}\$" \
      --tau_channel "$CHANNEL" \
      --gnn_mode "$GNN_MODE" \
      --plot_mode $PLOT_MODE
  done
done

echo "Copying results to storage..."
xrdcp -r --force "$JOB_SCRATCH" \
  root://t3dcachedb03.psi.ch:1094/${STORAGE_OUTPUT}

echo "Cleaning scratch..."
rm -rf "$JOB_SCRATCH"