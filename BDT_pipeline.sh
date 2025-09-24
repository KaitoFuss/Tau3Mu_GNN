set -e

python bdt/Train_XGB.py --model knn_new_vs7

python utils/BDT_ScoreProducer.py --model knn_new_vs7 \
    --classifier_gnn /work/kfuss/SVJ_GNN/output/knn_new_vs7/GNN_classifiers_21may2025_v0.pck \
    --classifier_iso /work/kfuss/SVJ_GNN/output/knn_new_vs7/ISO_classifiers_21may2025_v0.pck 

python utils/ModelEval.py \
  --root_path /work/kfuss/SVJ_GNN/output/knn_new_vs7/ \
  --title "\$\lambda = 0.5\$, kNN \$k = 10\$, 10 MeV per bin"
