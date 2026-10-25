from tau3mu_gnn.gnn_training.NormalizerUtils import OnTheFlyNormalizer
from tau3mu_gnn.graph.graphDataLoader import GraphDataLoader
from tau3mu_gnn.architectures.TauMuNet import TauMuNet
from tqdm import tqdm 

import time
import torch
import argparse
import os
import pickle
import numpy as np
import json
import random

import tau3mu_gnn.gnn_training.EvaluationUtils as ueval
import tau3mu_gnn.gnn_training.TrainingUtils as utrain

# import linecache
from tau3mu_gnn.architectures.GNN_archs import get_arch_configs, get_arch_configs_random
# import tracemalloc


# import mdmm

# maximum number of threads in order to not break t3
if "t3ui" in os.uname()[1]:
	torch.set_num_threads(8)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




def main(epochs, batch_size, learning_rate, disco_reg, name, model_type, params, tau_channel, gnn_mode, conv_type, data_base, finetune=False, finetune_lr=1e-4, finetune_epochs=10, bdt_threshold=0.8,
         experiment=None, patience=50, lr_scheduler_patience=10, dataset_fraction=(0.5, 0.5),
         windows=None, seed=None, device=None):
    """Train the model and evaluate it.

    Args:
        epochs(int): number of epochs to train for
        batch_size(int): size of each batch
        learning_rate(float):
        disco_reg(float): regularisation param
        name(str): name of the training
        model_type (str): Type of the model: which graph constructions params were used; eg. cone_size_0.5_knn_3, dzd0_cone_size_0.5_fully
        params : architecture parameters
        include_puppi : whether we include puppi weights or not
        tau_channel(str): tau channel (W or HF)
        gnn_mode(str): mode of GNN. Can be BDT, embed, final_layer
        conv_type(str: type of convolution in GNN (GINE, GAT, Transformer support)
        data_base(str): path to pt graphs, default is T3: /pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data
        experiment: Comet experiment (None: no logging)
        patience(int): early stopping patience (the best epoch is always kept)
        lr_scheduler_patience(int): ReduceLROnPlateau patience
        dataset_fraction: class mix (signal, background)
        windows: {"train": (lo, hi), "validation": (lo, hi), "test": (lo, hi)}; None: 0-0.6, 0.6-0.7, 0.7-1
        seed(int): torch, numpy, random seed (None: not set)
        device: torch device (None: cuda if available, else cpu)
    Returns: summary dict (best_epoch, epochs_run, early_stopped, best_validation_loss, auc_test)
        
    """

    start = time.time()
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    windows = windows or {"train": (0., 0.6), "validation": (0.6, 0.7), "test": (0.7, 1.)}

    ###--------- Prepare Model, Paths to input graphs  -----------------------------------------------------------

    # create output directory and get path to it
    output_path = utrain.output_directory(name)

    # Directory for the graphs
    base_directory_signal = os.path.join(data_base, f"signal_{tau_channel}/")
    base_directory_bkg    = os.path.join(data_base, f"background_{tau_channel}/")

    prefix = "W_decay" if tau_channel == "W" else "HF_decay"

    datasets_signal = [[base_directory_signal, f"{prefix}_signal_{model_type}"]]
    datasets_bkg    = [[base_directory_bkg,    f"{prefix}_background_{model_type}"]]

    # List: [paths to the "processed" directory containting the .pt files with the graph, name of the .pt file]

    # datasets_W_signal = [    
    #     [base_directory_W_signal + "winter24", f"W_decay_signal_winter24_{model_type}"],
        # [base_directory_W_signal + "summer24", f"W_decay_signal_summer24_{model_type}"],
    # ]

    # datasets_bkg = [
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024D-PromptReco-v1_MINIAOD/" + "250430_100827/0000", "background_" + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024D-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024F-PromptReco-v1_MINIAOD/" + "250430_100852/0000", "background_" + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024F-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024I-PromptReco-v1_MINIAOD/" + "250430_100917/0000", "background_" + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024I-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024E-PromptReco-v1_MINIAOD/" + "250430_100835/0000", "background_" + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024E-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024G-PromptReco-v1_MINIAOD/" + "250430_100900/0000", "background_" + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024G-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024I-PromptReco-v2_MINIAOD/" + "250430_100925/0000", "background_" + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024I-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024C-PromptReco-v1_MINIAOD/" + "250430_100819/0000", "background_" + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024C-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024E-PromptReco-v2_MINIAOD/" + "250430_100843/0000", "background_" + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024E-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024H-PromptReco-v1_MINIAOD/" + "250430_100908/0000", "background_" + "ParkingDoubleMuonLowMass0/_ParkingDoubleMuonLowMass0_Run2024H-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],

    #     [base_directory_bkg + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024D-PromptReco-v1_MINIAOD/" + "250430_100949/0000", "background_" + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024D-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024F-PromptReco-v1_MINIAOD/" + "250430_101014/0000", "background_" + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024F-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024I-PromptReco-v1_MINIAOD/" + "250430_101038/0000", "background_" + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024I-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024B-PromptReco-v1_MINIAOD/" + "250430_100933/0000", "background_" + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024B-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024E-PromptReco-v1_MINIAOD/" + "250430_100957/0000", "background_" + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024E-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024G-PromptReco-v1_MINIAOD/" + "250430_101022/0000", "background_" + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024G-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024I-PromptReco-v2_MINIAOD/" + "250430_101046/0000", "background_" + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024I-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024C-PromptReco-v1_MINIAOD/" + "250430_100941/0000", "background_" + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024C-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024E-PromptReco-v2_MINIAOD/" + "250430_101005/0000", "background_" + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024E-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024H-PromptReco-v1_MINIAOD/" + "250430_101030/0000", "background_" + "ParkingDoubleMuonLowMass1/_ParkingDoubleMuonLowMass1_Run2024H-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
        
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024D-PromptReco-v1_MINIAOD/" + "250430_101111/0000", "background_" + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024D-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024F-PromptReco-v1_MINIAOD/" + "250430_101135/0000", "background_" + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024F-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024I-PromptReco-v1_MINIAOD/" + "250430_101159/0000", "background_" + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024I-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024B-PromptReco-v1_MINIAOD/" + "250430_101054/0000", "background_" + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024B-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024E-PromptReco-v1_MINIAOD/" + "250430_101119/0000", "background_" + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024E-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024G-PromptReco-v1_MINIAOD/" + "250430_101143/0000", "background_" + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024G-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024I-PromptReco-v2_MINIAOD/" + "250430_101207/0000", "background_" + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024I-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024C-PromptReco-v1_MINIAOD/" + "250430_101102/0000", "background_" + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024C-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024E-PromptReco-v2_MINIAOD/" + "250430_101127/0000", "background_" + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024E-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024H-PromptReco-v1_MINIAOD/" + "250430_101151/0000", "background_" + "ParkingDoubleMuonLowMass2/_ParkingDoubleMuonLowMass2_Run2024H-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
        
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024D-PromptReco-v1_MINIAOD/" + "250430_101232/0000", "background_" + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024D-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024F-PromptReco-v1_MINIAOD/" + "250430_101256/0000", "background_" + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024F-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024I-PromptReco-v1_MINIAOD/" + "250430_101320/0000", "background_" + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024I-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024B-PromptReco-v1_MINIAOD/" + "250430_101215/0000", "background_" + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024B-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024E-PromptReco-v1_MINIAOD/" + "250430_101240/0000", "background_" + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024E-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024G-PromptReco-v1_MINIAOD/" + "250430_101304/0000", "background_" + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024G-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024I-PromptReco-v2_MINIAOD/" + "250430_101328/0000", "background_" + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024I-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024C-PromptReco-v1_MINIAOD/" + "250430_101223/0000", "background_" + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024C-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024E-PromptReco-v2_MINIAOD/" + "250430_101248/0000", "background_" + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024E-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024H-PromptReco-v1_MINIAOD/" + "250430_101312/0000", "background_" + "ParkingDoubleMuonLowMass3/_ParkingDoubleMuonLowMass3_Run2024H-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],

    #     [base_directory_bkg + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024D-PromptReco-v1_MINIAOD/" + "250430_101352/0000", "background_" + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024D-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024F-PromptReco-v1_MINIAOD/" + "250430_101416/0000", "background_" + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024F-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024I-PromptReco-v1_MINIAOD/" + "250430_101443/0000", "background_" + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024I-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024B-PromptReco-v1_MINIAOD/" + "250430_101336/0000", "background_" + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024B-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024E-PromptReco-v1_MINIAOD/" + "250430_101400/0000", "background_" + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024E-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024G-PromptReco-v1_MINIAOD/" + "250430_101424/0000", "background_" + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024G-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024I-PromptReco-v2_MINIAOD/" + "250430_101451/0000", "background_" + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024I-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024C-PromptReco-v1_MINIAOD/" + "250430_101344/0000", "background_" + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024C-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024E-PromptReco-v2_MINIAOD/" + "250430_101408/0000", "background_" + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024E-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024H-PromptReco-v1_MINIAOD/" + "250430_101433/0000", "background_" + "ParkingDoubleMuonLowMass4/_ParkingDoubleMuonLowMass4_Run2024H-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
        
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024B-PromptReco-v1_MINIAOD/" + "250430_101459/0000", "background_" + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024B-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024D-PromptReco-v1_MINIAOD/" + "250430_101515/0000", "background_" + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024D-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024E-PromptReco-v2_MINIAOD/" + "250430_101531/0000", "background_" + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024E-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024G-PromptReco-v1_MINIAOD/" + "250430_101547/0000", "background_" + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024G-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024I-PromptReco-v1_MINIAOD/" + "250430_101603/0000", "background_" + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024I-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024C-PromptReco-v1_MINIAOD/" + "250430_101507/0000", "background_" + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024C-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024E-PromptReco-v1_MINIAOD/" + "250430_101523/0000", "background_" + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024E-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024F-PromptReco-v1_MINIAOD/" + "250430_101539/0000", "background_" + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024F-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024H-PromptReco-v1_MINIAOD/" + "250430_101555/0000", "background_" + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024H-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024I-PromptReco-v2_MINIAOD/" + "250430_101611/0000", "background_" + "ParkingDoubleMuonLowMass5/_ParkingDoubleMuonLowMass5_Run2024I-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
        
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024B-PromptReco-v1_MINIAOD/" + "250430_101619/0000", "background_" + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024B-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024D-PromptReco-v1_MINIAOD/" + "250430_101635/0000", "background_" + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024D-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024E-PromptReco-v2_MINIAOD/" + "250430_101652/0000", "background_" + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024E-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024G-PromptReco-v1_MINIAOD/" + "250430_101711/0000", "background_" + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024G-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024I-PromptReco-v1_MINIAOD/" + "250430_101727/0000", "background_" + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024I-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024C-PromptReco-v1_MINIAOD/" + "250430_101627/0000", "background_" + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024C-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024E-PromptReco-v1_MINIAOD/" + "250430_101643/0000", "background_" + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024E-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024F-PromptReco-v1_MINIAOD/" + "250430_101700/0000", "background_" + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024F-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024H-PromptReco-v1_MINIAOD/" + "250430_101719/0000", "background_" + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024H-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024I-PromptReco-v2_MINIAOD/" + "250430_101735/0000", "background_" + "ParkingDoubleMuonLowMass6/_ParkingDoubleMuonLowMass6_Run2024I-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],

    #     [base_directory_bkg + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024B-PromptReco-v1_MINIAOD/" + "250430_101743/0000", "background_" + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024B-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024D-PromptReco-v1_MINIAOD/" + "250430_101759/0000", "background_" + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024D-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024E-PromptReco-v2_MINIAOD/" + "250430_101815/0000", "background_" + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024E-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024G-PromptReco-v1_MINIAOD/" + "250430_101831/0000", "background_" + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024G-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024I-PromptReco-v1_MINIAOD/" + "250430_101847/0000", "background_" + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024I-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024C-PromptReco-v1_MINIAOD/" + "250430_101751/0000", "background_" + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024C-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024E-PromptReco-v1_MINIAOD/" + "250430_101807/0000", "background_" + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024E-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024F-PromptReco-v1_MINIAOD/" + "250430_101823/0000", "background_" + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024F-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024H-PromptReco-v1_MINIAOD/" + "250430_101839/0000", "background_" + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024H-PromptReco-v1_MINIAOD/".replace("/", "") + "_" + model_type],
    #     [base_directory_bkg + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024I-PromptReco-v2_MINIAOD/" + "250430_101855/0000", "background_" + "ParkingDoubleMuonLowMass7/_ParkingDoubleMuonLowMass7_Run2024I-PromptReco-v2_MINIAOD/".replace("/", "") + "_" + model_type],
    # ]


    # base_directory_W_signal = "data/signal_W" 
    # base_directory_bkg = "data/background" 

    # datasets_W_signal = [
    #     [base_directory_bkg, f"background_{model_type}"],
    # ]
    # datasets_bkg = [
    #     [base_directory_W_signal, f"W_decay_signal_{model_type}"]
    # ]

    datasets = datasets_signal + datasets_bkg
    # create on the fly normalizer and save the mean and std 
    means_x, stds_x, means_edge, stds_edge, means_graph, stds_graph = utrain.Normalize(datasets, model_type, tau_channel)

    # normalizer = OnTheFlyNormalizer(["x"], means, stds)

    means_dict = {
        "x": means_x,
        "edge_attr": means_edge,
        "graph_level_features": means_graph,
    }

    stds_dict = {
        "x": stds_x,
        "edge_attr": stds_edge,
        "graph_level_features": stds_graph,
    }

    normalizer = OnTheFlyNormalizer(["x", "edge_attr", "graph_level_features"], means_dict, stds_dict)

    normalize_dict = {
        'means_x': means_x,
        'stds_x': stds_x,
        'means_edge': means_edge,
        'stds_edge': stds_edge,
        'means_graph': means_graph,
        'stds_graph': stds_graph,
    }
    
    with open(os.path.join(output_path, "Normalization.pkl"), 'wb') as f:
        pickle.dump(normalize_dict, f)
    print(f"Normalization saved in {os.path.join(output_path, 'Normalization.pkl')}")

    ###--------- Prepare GraphDataLoader  -----------------------------------------------------------
    ## Specify the datasetfractions to train on and then create the dataloader for signal and background for training and validation
    
    df = list(dataset_fraction) # class mix: signal, background (default 50 / 50)

    # Create the data loaders for training and validation
    dataloader_training = GraphDataLoader(
        datasets,
        window_to_load=[list(windows["train"]) for _ in range(len(datasets))],
        dataset_fraction=df,
        batch_size=batch_size,
        normalizer=normalizer,
        drop_last=True,
    )

    
    dataloader_validation = GraphDataLoader(
        datasets,
        window_to_load=[list(windows["validation"]) for _ in range(len(datasets))],
        dataset_fraction=df,
        batch_size=batch_size,
        normalizer=normalizer,
        drop_last=True,
    )


    ###--------- Create the Model -----------------------------------------------------------
    # Input parameters for the Model
    num_node_features, num_edge_features, num_graph_level_features = utrain.get_model_parameters(model_type, tau_channel)
    
    # Create the model
    model = TauMuNet(
        conv_params=params["conv_params"],
        dropout=params["dropout"],
        fc_params=params["fc_params"],
        num_node_features=num_node_features, 
        num_edge_features=num_edge_features,
        num_graph_level_features=num_graph_level_features,
        use_fusion=params["use_fusion"],
        gnn_mode=gnn_mode,
        scale_down_params=params["fusion_scale_down_params"],
        conv_type=conv_type,
    )

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience = lr_scheduler_patience)

    # Load model on gpu if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    model = model.to(device)
    
    # Define the loss
    loss_criterion = torch.nn.BCELoss()
    


    ###--------- Train the network -----------------------------------------------------------

    training_losses, validation_losses, info = train_and_validate(model, epochs, dataloader_training, dataloader_validation, loss_criterion, device, optimizer, disco_reg, name, scheduler, tau_channel, gnn_mode, experiment=experiment, patience=patience)
    print(f"==> Training completed in {time.time()-start:.2f} seconds")

    print("==> Saving losses...")
    utrain.save_loss(training_losses, validation_losses, name)
    
    print("Traininig finished")

    ###--------- Fine-tune the network -----------------------------------------------------------

    if finetune:
        start_ft = time.time()
        print("==> Fine-tuning on BDT-selected events...")

        import shutil
        shutil.copy(
            os.path.join(output_path, name + ".pt"),
            os.path.join(output_path, name + "_pretrained.pt")
        )
        print("==> Saved pre-trained checkpoint as _pretrained.pt")
        
        dataloader_finetune = GraphDataLoader(
            datasets,
            window_to_load=[[0., 0.6] for _ in range(len(datasets))],
            dataset_fraction=df,
            batch_size=batch_size,
            normalizer=normalizer,
            drop_last=True,
            score_threshold=bdt_threshold,
        )

        # count surviving events before committing
        n_bkg = sum(1 for d in dataloader_finetune.dataset if d.y == 0)
        n_sig = sum(1 for d in dataloader_finetune.dataset if d.y == 1)
        print(f"Fine-tune sample: {n_sig} signal, {n_bkg} background")

        # reload best model checkpoint (early stopping may have saved it)
        checkpoint = torch.load(os.path.join(output_path, name + ".pt"))
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer_ft = torch.optim.Adam(model.parameters(), lr=finetune_lr)
        # no scheduler for fine-tuning — too few epochs for plateau detection to be meaningful

        training_losses, validation_losses, info = train_and_validate(
            model, finetune_epochs, dataloader_finetune, dataloader_validation,
            loss_criterion, device, optimizer_ft, disco_reg, name, 
            scheduler=None, tau_channel=tau_channel, gnn_mode=gnn_mode, finetune=finetune, experiment=experiment
        )
        print(f"==> Fine-tune completed in {time.time()-start_ft:.2f} seconds")

        print("==> Saving losses...")
        utrain.save_loss(training_losses, validation_losses, name, finetune=finetune)

    ###--------- Evaluate the model -----------------------------------------------------------

    print("==> Evaluate Model...")
    start_eva = time.time()

    # Run test dataset through model and save output in a dict 
    evaluation_dict = ueval.Evaluate_model(datasets_signal, datasets_bkg, batch_size, normalizer, model, device, tau_channel, disco_reg, window_to_load=list(windows["test"]))

    # Compute roc and auc, safe them also in pickels
    roc, auc = ueval.roc_auc(evaluation_dict, name, tau_channel, gnn_mode)
    ueval.plot_auc_simple(auc, name)
    ueval.plot_roc(roc, auc, name)
    # ueval.plot_scores(evaluation_dict,name)
    # ueval.plot_sic(roc, name)
    ueval.plot_dnn(evaluation_dict, name) # plots GNN scores of bkg and signal
    # ueval.plot_sic_small(roc,name)
    # ueval.plot_dnn(evaluation_dict,name)
    # ueval.plot_dnn_2(evaluation_dict, name)
    ueval.plot_sculpting(evaluation_dict, name)

    print(f"==> Evaluation completed in {time.time()-start_eva:.2f} seconds")

    aucs = {key: float(value) for key, value in auc.items()}   # roc_auc: {evaluation key: AUC}
    return {**info, "auc_test": next(iter(aucs.values())) if len(aucs) == 1 else aucs}

def comet_settings(tau_channel):
    """Comet workspace and API key for the old CLI: configs/training_defaults_<channel>.yaml, else the W file
    (one Comet account for all channels; no key in the code)."""
    import yaml
    from tau3mu_gnn.pipelines.common.config import REPO_ROOT
    path = REPO_ROOT / "configs" / f"training_defaults_{tau_channel}.yaml"
    if not path.is_file():
        path = REPO_ROOT / "configs" / "training_defaults_W.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["comet"]


class _NoExperiment:
    """Stand-in when no Comet experiment is given."""

    def log_metric(self, *args, **kwargs):
        pass


def train_and_validate(model, epochs, dataloader_training, dataloader_validation, loss_criterion, device, optimizer, disco_reg, name, scheduler, tau_channel, gnn_mode, finetune=False, experiment=None, patience=50):
    """Train and validate model.

    Args:
        model: model to train
        epochs(int): number of epochs to train for
        dataloader_training(GraphDataLoader): dataloader for training
        dataloader_validation(GraphDataLoader): dataloader for validation
        loss_criterion:
        device: GPU or CPU
        optimizer: ADAM
        disco_reg(float): regularisation param
        name: name of output directory 


    Returns: training_losses(List), validation_losses(List), info(dict: best_epoch, epochs_run,
        early_stopped, best_validation_loss). The model holds the best epoch afterwards; <name>.pt too.
        
    """
    experiment = experiment if experiment is not None else _NoExperiment()

    training_losses = []
    validation_losses = []
    validation_discos = []
    validation_full_losses = []

    # initalize early stopping class
    if not finetune:
        early_stop = utrain.early_stop(patience=patience)
    else:
        early_stop = utrain.early_stop(patience=20)

    # get output_path
    output_path = utrain.output_directory(name)
    global_step = 0  # optimizer steps so far: the Comet x-axis (`step`); `epoch` is logged as well

    for epoch in range(0,epochs):
        #training
        batch_counter = 0
        bce_per_batch = 0
        disco_per_batch = 0
        loss_per_batch = 0
        
        if finetune:
            lambda_eff = disco_reg
        else:
            warmup = 20
            lambda_eff = disco_reg * min(1.0, epoch / warmup) # we keep lambda for the first warmup epoch so we do in the beginnng large steps into the right directions



        for batch, data in enumerate(dataloader_training):
            data.x = torch.nan_to_num(data.x)
            model.train() # set into training modus for backpropagation
            data = data.to(device) # send data to ram of gpu!
            target = data.y # this is my label
            optimizer.zero_grad() # make sure gradients are empty
            out = model(data)

            bce_loss = loss_criterion(out[:, 0], target) # calc mean bce_loss over the batch

            score = out[:, 0].cpu()  # keep as torch tensor with grad
            score = score.detach().numpy()
            mass  = torch.nan_to_num(data.fitted_tau_mass).view(-1) # I dont know why we are doing these shinanigans. data.fitted_tau_mass should be enough I think..?

            normedweight = torch.ones_like(out[:, 0]) 
            mask = (target == 0) # we penalise disco just on bkg graphs. Disco on signal is inherently large but we dont want to penalise this
            if mask.sum() > 1:
                disco_loss = utrain.distance_corr(out[:,0][mask], mass[mask], normedweight[mask])
            else:
                disco_loss = 0.0 * out[:,0].sum()  # keep graph alive

            loss_ = bce_loss + lambda_eff * disco_loss
            
            loss_per_batch += loss_.item()
            bce_per_batch += bce_loss.item()
            disco_per_batch += disco_loss.item()

            loss_.backward()
            optimizer.step()
            global_step += 1
            batch_counter += 1
            
            del loss_, score, out
            torch.cuda.empty_cache()

        # calculate loss per epoch and log it
        epoch_loss_training = loss_per_batch/batch_counter
        epoch_bce = bce_per_batch/batch_counter
        epoch_disco = disco_per_batch/batch_counter
        training_losses.append(epoch_loss_training)
        experiment.log_metric(f"Full training loss", epoch_loss_training, step=global_step, epoch=epoch)
        experiment.log_metric(f"BCE training loss", epoch_bce, step=global_step, epoch=epoch)
        experiment.log_metric(f"Disco training", epoch_disco, step=global_step, epoch=epoch)


        # validation
        batch_counter = 0
        loss_per_batch = 0
        disco_per_batch = 0
        bce_per_batch = 0

        with torch.no_grad():
            for batch, data in enumerate(dataloader_validation):
                model.eval()
                data = data.to(device)
                target = data.y
                #evaluation_dict = ueval.Evaluate_model(data[0], data[1], batch_size, normalizer, model, device, window_to_load=[0.7,0.8])

                out = model(data)

                bce_loss = loss_criterion(out[:, 0], target)

                score = out[:, 0].cpu()  # keep as torch tensor
                score = score.detach().numpy()
                mass  = torch.nan_to_num(data.fitted_tau_mass).view(-1)

                normedweight = torch.ones_like(out[:, 0]) 
                mask = (target == 0)
                batch_disco = 0
                if mask.sum() > 1:
                    disco_loss = utrain.distance_corr(out[:,0][mask], mass[mask], normedweight[mask])
                else:
                    disco_loss = 0.0 * out[:,0].sum()  # keep graph alive
                
                loss_ = bce_loss + disco_reg * disco_loss

                loss_per_batch += loss_.item()
                bce_per_batch += bce_loss.item()
                disco_per_batch += disco_loss.item()

                batch_counter += 1

        #log the auc
        #auc_validation = auc_per_batch/batch_counter
        #experiment.log_metric("AUC", auc_validation, epoch=epoch)

        del loss_, score, out
        torch.cuda.empty_cache()

        
        # calculate loss per epoch and log it
        epoch_bce_loss_validation = bce_per_batch/batch_counter
        validation_losses.append(epoch_bce_loss_validation)
        experiment.log_metric(f"BCE validation loss", epoch_bce_loss_validation, step=global_step, epoch=epoch)

        epoch_disco_validation = disco_per_batch/batch_counter
        validation_discos.append(epoch_disco_validation)
        experiment.log_metric(f"Disco validation", epoch_disco_validation, step=global_step, epoch=epoch)

        epoch_loss_validation = loss_per_batch/batch_counter
        validation_full_losses.append(epoch_loss_validation)
        experiment.log_metric(f"Full validation validation", epoch_loss_validation, step=global_step, epoch=epoch)

        if not np.isfinite(epoch_loss_validation):  # NaN would block early stopping and the checkpoint
            raise FloatingPointError(f"validation loss is {epoch_loss_validation} in epoch {epoch} "
                                     f"(BCE {epoch_bce_loss_validation}, DisCo {epoch_disco_validation})")

        if scheduler is not None:
            scheduler.step(epoch_loss_validation)

        ## check for early stop
        early_stop(epoch_loss_validation,model,epoch,output_path,name)
        if early_stop.early_stop:
            break
        ## log the average loss as well
        experiment.log_metric(f"Average full validation loss", early_stop.avg_loss, step=global_step, epoch=epoch) # over last 10 epochs 



    # Always keep the best epoch (Kaito, 2026-10-23): early_stop saved it to <name>.pt; load it back
    # so that the evaluation uses it. The old code overwrote <name>.pt with the last epoch when all
    # epochs ran.
    best_idx = early_stop.best_epoch
    checkpoint = torch.load(os.path.join(output_path, name + ".pt"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_disco_loss = {
        "last_validation_disco": validation_discos[best_idx],
        "last_val_loss": validation_losses[best_idx],
        "last_val_full_loss": early_stop.best_loss,
        "lambda": disco_reg,
        "tau_channel": tau_channel,
        "gnn_mode": gnn_mode,
    }

    print("Model saved to ", output_path + "/" + name + ".pt")
    
    suffix = ""
    if finetune:
        suffix = "_finetune"
    with open(os.path.join(output_path, name + f"_disco_loss{suffix}.pkl"), "wb") as f:
        pickle.dump(val_disco_loss, f)

    info = {"best_epoch": int(early_stop.best_epoch), "epochs_run": epoch + 1,
            "early_stopped": bool(early_stop.early_stop), "best_validation_loss": float(early_stop.best_loss)}
    return training_losses, validation_losses, info


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default="arch1",
        help="Choose model architecture"
        )
    
    parser.add_argument(
        "--epochs",
        default=5,
        help="Number of epochs to train on",
        type=int,
        )
    
    parser.add_argument(
        "--batch",
        default=64,
        help="Batch size",
        type=int,
        )
    
    parser.add_argument(
        "--name",
        default="TauMuNet",
        help="Descriptive name of the training to save the losses and the model",
        type=str,
        )

    parser.add_argument(
        "--model",
        default="dzd0_cone_size_0.5_fully",
        help="Which model to train/ which graph construction was used",
        type=str,
        )

    parser.add_argument(
        "--lr",
        default=0.001,
        help="Learning rate of the model",
        type=float,
        )
    
    parser.add_argument(
        "--disco",
        default=0.01,
        help="Regularisation using DisCo of output and tau mass",
        type=float,
        )
    
    parser.add_argument(
    "--tau_channel",
    default="W",
    type=str,
    help="tau production channel"
        )
    
    parser.add_argument(
        "--gnn_mode",
        default="final_layer",
        type=str,
        help="GNN architecture specification. Either GNN into BDT, graph features at embedding level or graph features at final layer"
        )

    parser.add_argument(
        "--remark",
        default="",
        type=str,
        help="additional info for particular training"
        )
    
    parser.add_argument(
        "--conv_type",
        default="GINE",
        type=str,
        help="GNN convolutional layer specification"
        )
    
    parser.add_argument(
        "--unsculpting_mode",
        default="",
        type=str,
        help="Type of unsculpting mode"
        )
    
    parser.add_argument(
        "--data_base",
        default="/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data",
        type=str,
        help="Base directory for signal and background graphs"
        )
    
    parser.add_argument(
        "--finetune", 
        action="store_true", 
        help="Enable fine-tuning phase"
    )

    parser.add_argument(
        "--finetune_lr", 
        default=1e-4, 
        type=float
    )

    parser.add_argument(
        "--finetune_epochs", 
        default=10, 
        type=int
    )

    parser.add_argument(
        "--bdt_threshold", 
        default=0.8, 
        type=float
    )
    

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch
    learning_rate = args.lr
    name = args.name
    model_type = args.model
    disco_reg = args.disco
    arch = args.arch
    tau_channel = args.tau_channel
    gnn_mode = args.gnn_mode
    remark = args.remark
    conv_type = args.conv_type
    unsculpting_mode = args.unsculpting_mode
    finetune = args.finetune
    finetune_lr = args.finetune_lr
    finetune_epochs = args.finetune_epochs
    bdt_threshold = args.bdt_threshold

    assert tau_channel in ["W", "HF"], f"Invalid tau_channel: {tau_channel}"
    assert gnn_mode in ["BDT", "embed_once", "embed_always", "final_layer"]
    assert conv_type in ["GINE", "GAT", "Transformer"]
    assert unsculpting_mode in ["", "delete_muons", "permutation", "split_pt"]


    print("==> Running TauMuNet ")

    from comet_ml import Experiment
    comet = comet_settings(tau_channel)
    experiment = Experiment(api_key=comet["api_key"], project_name=name, workspace=comet["workspace"])
    
    if arch == "random":
        model_params, arch_descr = get_arch_configs_random(tau_channel, disco_reg)
    else:
        model_params, arch_descr = get_arch_configs(arch, tau_channel, disco_reg)

    # overwriting names (adding versionnumber) such that no old files get overwritten
    name = utrain.generate_name_with_versionnumber(name, os.getcwd(), tau_channel, arch_descr, unsculpting_mode, conv_type=conv_type, remark=remark)
    output_path = utrain.output_directory(name)

    with open(os.path.join(output_path, "arch_config.json"), "w") as f:
        json.dump(model_params, f)
    
    #log hyperparameters in the experiment
    hyper_params = {
        "architecture": arch,
        "learning_rate": learning_rate,
        "regularisation": disco_reg,
        "epochs": epochs,
        "Model" : model_type,
        "tau_channel" : tau_channel,
        "gnn_mode": gnn_mode,
        "conv_type": conv_type,
        "unsculpting_mode": unsculpting_mode,
        "remark": remark,
    }

    if finetune:
        hyper_params["finetune_lr"] = finetune_lr
        hyper_params["finetune_epochs"] = finetune_epochs
        hyper_params["bdt_threshold"] = bdt_threshold

    experiment.log_parameters(hyper_params)
    experiment.set_name(name)

    main(epochs, batch_size, learning_rate, disco_reg, name, model_type, model_params, tau_channel, gnn_mode, conv_type, args.data_base, finetune=finetune, finetune_lr=finetune_lr, finetune_epochs=finetune_epochs, bdt_threshold=bdt_threshold, experiment=experiment)

