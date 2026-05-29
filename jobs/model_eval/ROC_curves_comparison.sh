#!/bin/bash
set -e

# vanilla plot
# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --plot_mode $PLOT_MODE --sculpt_degree "$SCULPT_DEGREE" --gnn_archs_location $GNN_FINAL_LAYER $GNN_EMBED_ALWAYS $GNN_EMBED_ONCE --tau_channel $CHANNEL --lambdas $LAM_FINALY_LAYER $LAM_EMBED_ALWAYS $LAM_EMBED_ONCE $LAM_BDT --bdt_location $BDT_LOCATION

# also show gnn for bdt
# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --plot_mode $PLOT_MODE --sculpt_degree "$SCULPT_DEGREE" --gnn_archs_location $GNN_FINAL_LAYER $GNN_EMBED_ALWAYS $GNN_EMBED_ONCE $GNN_BDT $GNN_BDT1 --tau_channel $CHANNEL --lambdas $LAM_FINALY_LAYER $LAM_EMBED_ALWAYS $LAM_EMBED_ONCE $LAM_BDT $LAM_BDT  --bdt_location $BDT_LOCATION

# skip bdt as not existing
# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --plot_mode $PLOT_MODE --sculpt_degree "$SCULPT_DEGREE" --gnn_archs_location $GNN_FINAL_LAYER $GNN_EMBED_ALWAYS $GNN_EMBED_ONCE $GNN_BDT --tau_channel $CHANNEL --lambdas $LAM_FINALY_LAYER $LAM_EMBED_ALWAYS $LAM_EMBED_ONCE $LAM_BDT


# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --plot_mode $PLOT_MODE --sculpt_degree "$SCULPT_DEGREE" --gnn_archs_location $GNN_BDT --tau_channel $CHANNEL --lambdas $LAM_BDT  --bdt_location $BDT_LOCATION

# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --plot_mode $PLOT_MODE --sculpt_degree "$SCULPT_DEGREE" --gnn_archs_location $GNN_EMBED_ALWAYS $GNN_BDT1 --tau_channel $CHANNEL --lambdas $LAM_FINALY_LAYER $LAM_BDT  --bdt_location $BDT_LOCATION

# one model vs BDT

CHANNEL=W
SCULPT_DEGREE=fine_tune_cut
PLOT_MODE=ROC_comp_full
BDT_CUT=true

ARCH1=fl_8_sd_4_GINE_delete_muons_ft_bdt_cut_0.6_vs1
GNN_FINAL_LAYER1=/work/kfuss/SVJ_GNN/output/pt_ft_"$CHANNEL"_$ARCH1/
LAM_FINALY_LAYER1="0.0,BDT_cut_0.6"

ARCH2=fl_8_sd_4_GINE_delete_muons_ft_bdt_cut_0.7_vs1
GNN_FINAL_LAYER2=/work/kfuss/SVJ_GNN/output/pt_ft_"$CHANNEL"_$ARCH2/
LAM_FINALY_LAYER2="0.0,BDT_cut_0.7"

ARCH3=fl_8_sd_4_GINE_delete_muons_ft_bdt_cut_0.8_vs1
GNN_FINAL_LAYER3=/work/kfuss/SVJ_GNN/output/pt_ft_"$CHANNEL"_$ARCH3/
LAM_FINALY_LAYER3="0.0,BDT_cut_0.8"

# ARCH4=fl_8_sd_4_vs1
# GNN_FINAL_LAYER4=/work/kfuss/SVJ_GNN/output/new_dropout_disco1_permuted_"$CHANNEL"_$ARCH4/
# LAM_FINALY_LAYER4="1.0,permuted,small"

ARCH4=fl_8_sd_4_GINE_delete_muons_ft_bdt_cut_0.75_vs1
GNN_FINAL_LAYER4=/work/kfuss/SVJ_GNN/output/pt_ft_"$CHANNEL"_$ARCH4/
LAM_FINALY_LAYER4="0.0,BDT_cut_0.8"

ARCH5=fl_8_sd_4_GINE_delete_muons_small_ft_bdt_cut_0.7_vs1
GNN_FINAL_LAYER5=/work/kfuss/SVJ_GNN/output/pt_ft_"$CHANNEL"_$ARCH5/
LAM_FINALY_LAYER5="0.0,BDT_cut_0.7_small"

BDT_LOCATION=/work/kfuss/SVJ_GNN/output/benchmark_"$CHANNEL"_BDT_vs1/
LAM_BDT=5.0

# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --plot_mode $PLOT_MODE --sculpt_degree "$SCULPT_DEGREE" --gnn_archs_location $GNN_FINAL_LAYER1 $GNN_FINAL_LAYER2 $GNN_FINAL_LAYER3 $GNN_FINAL_LAYER4 $BDT_LOCATION --tau_channel $CHANNEL --lambdas $LAM_FINALY_LAYER1 $LAM_FINALY_LAYER2 $LAM_FINALY_LAYER3 $LAM_FINALY_LAYER4 $LAM_BDT  --bdt_location $BDT_LOCATION --archs $ARCH1 $ARCH2 $ARCH3 $ARCH4
# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --plot_mode $PLOT_MODE --sculpt_degree "$SCULPT_DEGREE" --gnn_archs_location $GNN_FINAL_LAYER1 $GNN_FINAL_LAYER2 $GNN_FINAL_LAYER3 --tau_channel $CHANNEL --lambdas $LAM_FINALY_LAYER1 $LAM_FINALY_LAYER2 $LAM_FINALY_LAYER3 --archs $ARCH1 $ARCH2 $ARCH3
/work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --plot_mode $PLOT_MODE --sculpt_degree "$SCULPT_DEGREE" --gnn_archs_location $GNN_FINAL_LAYER1 $GNN_FINAL_LAYER2 $GNN_FINAL_LAYER3 $GNN_FINAL_LAYER4 $BDT_LOCATION --tau_channel $CHANNEL --lambdas $LAM_FINALY_LAYER1 $LAM_FINALY_LAYER2 $LAM_FINALY_LAYER3 $LAM_FINALY_LAYER4 $LAM_BDT  --archs $ARCH1 $ARCH2 $ARCH3 $ARCH4 --bdt_location $BDT_LOCATION --bdt_cut $BDT_CUT
# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --plot_mode $PLOT_MODE --sculpt_degree "$SCULPT_DEGREE" --gnn_archs_location $BDT_LOCATION --tau_channel $CHANNEL --lambdas $LAM_BDT  --bdt_location $BDT_LOCATION
# /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m tau3mu_gnn.model_eval.ModelEval --plot_mode $PLOT_MODE --sculpt_degree "$SCULPT_DEGREE" --gnn_archs_location $GNN_FINAL_LAYER1 $GNN_FINAL_LAYER3 $BDT_LOCATION --tau_channel $CHANNEL --lambdas $LAM_FINALY_LAYER1 $LAM_FINALY_LAYER3 $LAM_BDT  --archs $ARCH1 $ARCH3 --bdt_location $BDT_LOCATION --bdt_cut $BDT_CUT