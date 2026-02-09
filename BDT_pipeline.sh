#!/bin/bash
set -e

python bdt/Train_XGB.py --model puppi_optunabest_newdata_with_CHAT_vs46_copy

python utils/BDT_ScoreProducer.py --model puppi_optunabest_newdata_with_CHAT_vs46_copy \
    --classifier_gnn /work/adcorel/SVJ_GNN/output/puppi_optunabest_newdata_with_CHAT_vs46_copy/GNN_classifiers_4february2026_v0.pck \
    --classifier_iso /work/adcorel/SVJ_GNN/output/puppi_optunabest_newdata_with_CHAT_vs46_copy/ISO_classifiers_4february2026_v0.pck 

python utils/ModelEval.py \
  --root_path /work/adcorel/SVJ_GNN/output/puppi_optunabest_newdata_with_CHAT_vs46_copy/ \
  --title "\$\lambda = 5.0\$, kNN \$k = 10\$, with puppi, 10 MeV per bin"
