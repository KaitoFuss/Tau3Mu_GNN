#!/bin/bash
set -e

source /eos/user/k/kfuss/miniforge3/bin/activate
conda activate /eos/user/k/kfuss/envs/Tau3Mu

cd /eos/user/k/kfuss/SVJ_GNN

python -m tau3mu_gnn.scripts.Train_TauMuNet \
    --epochs 1000 \
    --batch 128 \
    --name 600k_disco0_deleted \
    --model skimmed_600k_dzd0_cone_size_0.5_knn_10_delete_muons \
    --lr 0.0016 \
    --disco 0.0 \
    --arch fl8_sd4 \
    --tau_channel HF \
    --gnn_mode final_layer \
    --remark "" \
    --data_base /eos/user/k/kfuss/tau3mu_gnn_round2/data \
