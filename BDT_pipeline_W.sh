#!/bin/bash
set -e

MODEL=early_stop_fix_W_BDT_vs1

python bdt/Train_XGB_W.py --model "$MODEL"

python utils/BDT_ScoreProducer.py --model "$MODEL" \
    --classifier_gnn /work/kfuss/SVJ_GNN/output/"$MODEL"/GNN_classifiers_24feb2026_v0.pck \
    --classifier_iso /work/kfuss/SVJ_GNN/output/"$MODEL"/ISO_classifiers_24feb2026_v0.pck \
    --tau_channel W

python eval_utils/ModelEval.py \
  --root_path /work/kfuss/SVJ_GNN/output/"$MODEL"/ \
  --title "\$\lambda = 0.625\$, kNN \$k = 10\$, 10 MeV per bin"
