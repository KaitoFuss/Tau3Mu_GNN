#!/bin/bash

source /work/kfuss/miniforge3/bin/activate Tau3Mu

/work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval \
    --plot_mode score_cut \
    --model scale_down_disco1_W_final_layer_vs5 \
    --tau_channel W \
    --cut 12.2 \