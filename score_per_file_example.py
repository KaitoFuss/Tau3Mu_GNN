
from utils.ScoreProducer import ScoreProducer
import torch
import pickle
from utils.PlotUtils import plot_histogram_root
import os
import numpy as np
from eval_utils.ModelEval import CPU_Unpickler, load_pkl
from architectures.TauMuNet import TauMuNet

import argparse
import ROOT

arch_configs = {
    "arch5": {
        "conv_params": [256, 128, 64, 32],
        "fc_params": [(64, 0.8)],
        "use_fusion" : True,
    },
    "arch6": {
        "conv_params": [256, 256, 128, 64, 64, 32, 32],
        "fc_params": [(64, 0.8)],
        "use_fusion" : True,
    },
    "arch7": {
        "conv_params": [512, 256, 128, 128, 64, 32],
        "fc_params": [(64, 0.8)],
        "use_fusion" : True,
    },
    "arch8": {
        "conv_params": [512, 256, 128, 64, 32, 16, 8],
        "fc_params": [(64, 0.8)],
        "use_fusion" : True,
    },
    "arch_optuna" : {
        "conv_params" : [1004, int(1004/2**1), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
        "fc_params" : [(51, 0.8)],
        "use_fusion": True, 
    },
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='Folder to read the data from')
    parser.add_argument('--output_folder', type=str, required=True, help="Output to store the files with the scores")
    parser.add_argument('--model_name', type=str, required=True, help="Name of GNN model")
    parser.add_argument('--arch', type=str, required=True, help="Architecture of GNN model")

    args = parser.parse_args()

    input_file = args.root
    output_folder = args.output_folder
    model_name = args.model_name
    arch = args.arch


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = arch_configs[arch]
    model =  TauMuNet(
        conv_params=params["conv_params"],
        dropout=0.2633831214702671,
        fc_params=params["fc_params"],
        use_fusion=params["use_fusion"],
    )
    model.load_state_dict(torch.load(output_folder + model_name, map_location=device)['model_state_dict'])
    # Load the normalizer
    normalizer = load_pkl(output_folder, "Normalization.pkl")[0]


    print("Starting the ScoreProducer")


    if arch == "arch_optuna":
        ScoreProducer(root=input_file,
                    edge_connection_mode="knn",
                    delta_r_threshold=None,
                    k_knn=10,
                    max_events_to_process=25000,
                    save_edge_attributes=True,
                    verbose=True,
                    slimmed=True,
                    shuffle_bool=True,
                    use_displacement=True,
                    model=model,
                    device=device,
                    normalizer=normalizer,
                    output_folder=output_folder
                    )