#!/bin/bash

# LAMBDA=10.0
# CHANNEL=HF
# GNN_MODE=BDT
# PLOT_MODE=tau_hist_fpr
# MODEL=correct_disco_10_HF_${GNN_MODE}_vs1

# source /work/kfuss/miniforge3/bin/activate Tau3Mu

# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval \
#   --output_path /work/kfuss/SVJ_GNN/output/${MODEL}/ \
#   --title "\$\lambda = ${LAMBDA}\$" \
#   --tau_channel ${CHANNEL} \
#   --gnn_mode ${GNN_MODE} \
#   --plot_mode ${PLOT_MODE}

# PLOT_MODE=tau_hist_after_fpr

# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval \
#   --output_path /work/kfuss/SVJ_GNN/output/${MODEL}/ \
#   --title "\$\lambda = ${LAMBDA}\$" \
#   --tau_channel ${CHANNEL} \
#   --gnn_mode ${GNN_MODE} \
#   --plot_mode ${PLOT_MODE}

LAMBDA=4.0
CHANNEL=HF
PLOT_MODE=tau_hist_fpr
GNN_MODE=final_layer
ARCH=final_layer3
MODEL=final_layer_archs_disco4_${CHANNEL}_${GNN_MODE}_vs3
VIEW=both
# MODEL=correct_disco_1_${CHANNEL}_${GNN_MODE}_vs1

source /work/kfuss/miniforge3/bin/activate Tau3Mu

/work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval \
  --output_path /work/kfuss/SVJ_GNN/output/${MODEL}/ \
  --title "\$\lambda = ${LAMBDA}\$" \
  --tau_channel ${CHANNEL} \
  --gnn_mode ${ARCH} \
  --plot_mode ${PLOT_MODE} \
  --view ${VIEW}