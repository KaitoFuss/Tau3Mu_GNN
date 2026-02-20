from comet_ml import Experiment

from utils.NormalizerUtils import OnTheFlyNormalizer
from data_loading.graphDataLoader import GraphDataLoader
from architectures.TauMuNet import TauMuNet
from tqdm import tqdm 

import time
import torch
import argparse
import os
import pickle
import numpy as np

import utils.EvaluationUtils as ueval
import utils.TrainingUtils as utrain

# import linecache
from utils.ConeGraphProducer import ConeGraphProducer
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




def main(epochs, batch_size, learning_rate, disco_reg, fusion, name, model_type, params, tau_channel, gnn_mode):
    """Train the model and evaluate it.

    Args:
        epochs(int): number of epochs to train for
        batch_size(int): size of each batch
        learning_rate(float):
        disco_reg(float): regularisation param
        fusion(bool): fusion toggle
        name(str): name of the training
        model_type (str): Type of the model: which graph constructions params were used; eg. cone_size_0.5_knn_3, dzd0_cone_size_0.5_fully
        params : architecture parameters
        include_puppi : whether we include puppi weights or not
        tau_channel(str): tau channel (W or HF)
        gnn_mode(str): mode of GNN. Can be BDT, embed, final_layer
    Returns: -
        
    """

    start = time.time()

    ###--------- Prepare Model, Paths to input graphs  -----------------------------------------------------------

    # create output directory and get path to it
    output_path = utrain.output_directory(name)

    # Directory for the graphs
    if tau_channel == "W":
        base_directory_signal = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/"
        base_directory_bkg = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/"

        datasets_signal = [
            [base_directory_signal, f"W_decay_signal_{model_type}"],
        ]

        datasets_bkg = [
            [base_directory_bkg, f"W_decay_background_{model_type}"],
        ]

    elif tau_channel == "HF":
        base_directory_signal = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_HF/"
        base_directory_bkg = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/"

        datasets_signal = [
            [base_directory_signal, f"HF_decay_signal_{model_type}"],
        ]

        datasets_bkg = [
            [base_directory_bkg, f"HF_decay_background_{model_type}"],
        ]

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
    means_x, stds_x, means_edge, stds_edge, means_graph, stds_graph = utrain.Normalize(datasets)

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
    
    df = [0.5, 0.5] # Train on 50% background, and 50 % signal

    # Create the data loaders for training and validation
    dataloader_training = GraphDataLoader(
        datasets,
        window_to_load=[[0., 0.6] for _ in range(len(datasets))],
        dataset_fraction=df,
        batch_size=batch_size,
        normalizer=normalizer,
        drop_last=True,
    )

    
    dataloader_validation = GraphDataLoader(
        datasets,
        window_to_load=[[0.6, 0.7] for _ in range(len(datasets))],
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
        dropout=0.2633831214702671,
        fc_params=params["fc_params"],
        num_node_features=num_node_features, 
        num_edge_features=num_edge_features,
        num_graph_level_features=num_graph_level_features,
        use_fusion=fusion,
        gnn_mode=gnn_mode,
    )

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience = 10)

    # Load model on gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    model = model.to(device)
    
    # Define the loss
    loss_criterion = torch.nn.BCELoss()
    


    ###--------- Train the network -----------------------------------------------------------

    training_losses, validation_losses = train_and_validate(model, epochs, dataloader_training, dataloader_validation, loss_criterion, device, optimizer, disco_reg, name, scheduler = scheduler)
    print(f"==> Training completed in {time.time()-start:.2f} seconds")

    print("==> Saving losses...")
    utrain.save_loss(training_losses, validation_losses, name)
    
    print("Traininig finished")

     ###--------- Evaluate the model -----------------------------------------------------------

    print("==> Evaluate Model...")
    start_eva = time.time()

    # Run test dataset through model and save output in a dict 
    evaluation_dict = ueval.Evaluate_model(datasets_signal, datasets_bkg, batch_size, normalizer, model, device, tau_channel, window_to_load=[0.7,1])

    # Compute roc and auc, safe them also in pickels
    roc, auc = ueval.roc_auc(evaluation_dict, name, tau_channel)
    ueval.plot_auc_simple(auc, name)
    ueval.plot_roc(roc, auc, name)
    # ueval.plot_scores(evaluation_dict,name)
    # ueval.plot_sic(roc, name)
    ueval.plot_dnn(evaluation_dict, name) # plots GNN scores of bkg and signal
    # ueval.plot_sic_small(roc,name)
    # ueval.plot_dnn(evaluation_dict,name)
    # ueval.plot_dnn_2(evaluation_dict, name)

    print(f"==> Evaluation completed in {time.time()-start_eva:.2f} seconds")

    return auc

def train_and_validate(model, epochs, dataloader_training, dataloader_validation, loss_criterion, device, optimizer, disco_reg, name, scheduler):
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


    Returns: training_losses(List), validation_losses(List)
        
    """

    training_losses = []
    validation_losses = []
    validation_discos = []


    # initalize early stopping class
    early_stop = utrain.early_stop()

    # get output_path
    output_path = utrain.output_directory(name)
    

    for epoch in range(0,epochs):
        #training
        batch_counter = 0
        bce_per_batch = 0
        disco_per_batch = 0
        loss_per_batch = 0
        
        warmup = 20  # 30
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
            batch_counter += 1
            
            del loss_, score, out
            torch.cuda.empty_cache()

        # calculate loss per epoch and log it
        epoch_loss_training = loss_per_batch/batch_counter
        epoch_bce = bce_per_batch/batch_counter
        epoch_disco = disco_per_batch/batch_counter
        training_losses.append(epoch_loss_training)
        experiment.log_metric("Full training loss", epoch_loss_training, epoch=epoch)
        experiment.log_metric("BCE training loss", epoch_bce, epoch=epoch)
        experiment.log_metric("Disco training", epoch_disco, epoch=epoch)


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
                
                loss_ = bce_loss + lambda_eff * disco_loss

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
        experiment.log_metric("BCE validation loss", epoch_bce_loss_validation, epoch=epoch)

        epoch_disco_validation = disco_per_batch/batch_counter
        validation_discos.append(epoch_disco_validation)
        experiment.log_metric("Disco validation", epoch_disco_validation, epoch=epoch)

        epoch_loss_validation = loss_per_batch/batch_counter
        scheduler.step(epoch_loss_validation)

        ## check for early stop
        early_stop(epoch_bce_loss_validation,model,epoch,output_path,name)
        if early_stop.early_stop:
            break
        ## log the average loss as well
        experiment.log_metric("Average BCE validation loss", early_stop.avg_loss, epoch=epoch) # over last 10 epochs 



    # Safe output Model only if early stopping was not done
    if not early_stop.early_stop:
        print("==> Saving Model")      
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(output_path, name + ".pt"))
        
        # Safe val_disco and val_loss for disco_param
        val_disco_loss = {
            "last_validation_disco": validation_discos[-1],
            "last_val_loss": validation_losses[-1],
            "lambda": disco_reg,
        }
    else:
        # Save val_disco and val_loss for disco_param
        best_idx = np.where(np.array(validation_losses) == early_stop.best_loss)[0][-1]
        val_disco_loss = {
            "last_validation_disco": validation_discos[best_idx],
            "last_val_loss": early_stop.best_loss,
            "lambda": disco_reg,
        }

    print("Model saved to ", output_path + "/" + name + ".pt")

    with open(os.path.join(output_path, name + "_disco_loss.pkl"), "wb") as f:
        pickle.dump(val_disco_loss, f)

       

    return training_losses, validation_losses



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        choices=[
            "arch1", "arch2", "arch3", "arch4",
            "arch5", "arch6", "arch7", "arch8", "arch_optuna"
        ],
        default="arch1",
        help="Choose model architecture"
)


    arch_configs = {
    "arch1": {
        "conv_params": [256, 128, 64, 32],
        "fc_params": [(32, 0.8)],
    },
    "arch2": {
        "conv_params": [256, 256, 128, 128, 64, 64, 32, 32],
        "fc_params": [(32, 0.8)],
    },
    "arch3": {
        "conv_params": [512, 256, 128, 64, 32],
        "fc_params": [(32, 0.8)],
    },
    "arch4": {
        "conv_params": [512, 256, 128, 64, 32, 16, 8],
        "fc_params": [(32, 0.8)],
    },
    "arch5": {
        "conv_params": [256, 128, 64, 32],
        "fc_params": [(64, 0.8)],
    },
    "arch6": {
        "conv_params": [256, 256, 128, 128, 64, 64, 32, 32],
        "fc_params": [(64, 0.8)],
    },
    "arch7": {
        "conv_params": [512, 256, 128, 64, 32],
        "fc_params": [(64, 0.8)],
    },
    "arch8": {
        "conv_params": [512, 256, 128, 64, 32, 16, 8],
        "fc_params": [(64, 0.8)],
    },
    "arch_optuna" : {
        "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
        "fc_params" : [(51, 0.8)] 
    }
}

    
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
        "--fusion",
        default=True,
        help="bool for using fusion in GNN",
        type=str2bool,
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
    

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch
    learning_rate = args.lr
    name = args.name
    model_type = args.model
    disco_reg = args.disco
    fusion = args.fusion
    arch = args.arch
    tau_channel = args.tau_channel
    gnn_mode = args.gnn_mode

    assert tau_channel in ["W", "HF"], f"Invalid tau_channel: {tau_channel}"
    assert gnn_mode in ["BDT", "embed_once", "embed_always", "final_layer"]


    print("==> Running TauMuNet ")

    experiment = Experiment(
        api_key="89gjTpDNiLT6Pnu5RjMpO4PID",
        project_name=name,
        workspace="kaitofuss"
        )
    
    # overwriting names (adding versionnumber) such that no old files get overwritten
    name = utrain.generate_name_with_versionnumber(name, os.getcwd(), gnn_mode, tau_channel)
    utrain.output_directory(name)
    
    #log hyperparameters in the experiment
    hyper_params = {
        "architecture": arch,
        "learning_rate": learning_rate,
        "regularisation": disco_reg,
        "epochs": epochs,
        "Model" : model_type,
        "tau_channel" : tau_channel,
        "gnn_mode": gnn_mode,
        "remark": "",
    }

    experiment.log_parameters(hyper_params)
    experiment.set_name(name)

    
    main(epochs, batch_size, learning_rate, disco_reg, fusion, name, model_type, arch_configs[arch], tau_channel, gnn_mode)

