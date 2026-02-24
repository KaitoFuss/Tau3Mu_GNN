#!/bin/bash
set -e

python bdt/Train_XGB_HF.py --model early_stop_fix_HF_BDT_vs1

python utils/BDT_ScoreProducer.py --model early_stop_fix_HF_BDT_vs1 \
    --classifier_gnn /work/kfuss/SVJ_GNN/output/early_stop_fix_HF_BDT_vs1/GNN_classifiers_24feb2026_v0.pck \
    --classifier_iso /work/kfuss/SVJ_GNN/output/early_stop_fix_HF_BDT_vs1/ISO_classifiers_24feb2026_v0.pck \
    --tau_channel HF

python eval_utils/ModelEval.py \
  --root_path /work/kfuss/SVJ_GNN/output/early_stop_fix_HF_BDT_vs1/ \
  --title "\$\lambda = 0.625\$, kNN \$k = 10\$, 10 MeV per bin"
