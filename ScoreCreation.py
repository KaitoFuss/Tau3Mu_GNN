
from utils.ScoreProducer import ScoreProducer
import torch
import pickle
from utils.PlotUtils import plot_histogram_root
import os
import numpy as np
from eval_utils.ModelEval import CPU_Unpickler, load_pkl
from architectures.TauMuNet import TauMuNet
from utils.TrainingUtils import get_model_parameters

import argparse
import ROOT

arch_configs = {
    "arch1": {
        "conv_params": [256, 128, 64, 32],
        "fc_params": [(32, 0.8)],
        "use_fusion": True,
    },
    "arch2": {
        "conv_params": [256, 256, 128, 128, 64, 64, 32, 32],
        "fc_params": [(32, 0.8)],
        "use_fusion": True,
    },
    "arch3": {
        "conv_params": [512, 256, 128, 64, 32],
        "fc_params": [(32, 0.8)],
        "use_fusion": True,
    },
    "arch4": {
        "conv_params": [512, 256, 128, 64, 32, 16, 8],
        "fc_params": [(32, 0.8)],
        "use_fusion": True,
    },
    "arch5": {
        "conv_params": [256, 128, 64, 32],
        "fc_params": [(64, 0.8)],
        "use_fusion": True,
    },
    "arch6": {
        "conv_params": [256, 256, 128, 128, 64, 64, 32, 32],
        "fc_params": [(64, 0.8)],
        "use_fusion": True,
    },
    "arch7": {
        "conv_params": [512, 256, 128, 64, 32],
        "fc_params": [(64, 0.8)],
        "use_fusion": True,
    },
    "arch8": {
        "conv_params": [512, 256, 128, 64, 32, 16, 8],
        "fc_params": [(64, 0.8)],
        "use_fusion": True,
    },
    "arch_optuna" : {
        "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
        "fc_params" : [(51, 0.8)] ,
        "use_fusion": True,
    }
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='Folder to read the data from')
    parser.add_argument('--output_folder', type=str, required=True, help="Output to store the files with the scores")
    parser.add_argument('--model_name', type=str, required=True, help="Name of GNN model")
    parser.add_argument('--arch', type=str, required=True, help="Architecture of GNN model")
    parser.add_argument('--tau_channel', type=str, required=True, help="Tau production channel")
    parser.add_argument('--gnn_mode', type=str, default="BDT", help="GNN Mode")
    parser.add_argument('--model', type=str, default="skimmed_dzd0_cone_size_0.5_knn_10", help="Which model to train/ which graph construction was used")

    args = parser.parse_args()

    input_file = args.root
    output_folder = args.output_folder
    model_name = args.model_name
    arch = args.arch
    tau_channel = args.tau_channel
    gnn_mode = args.gnn_mode
    model_type = args.model

    num_node_features, num_edge_features, num_graph_level_features = get_model_parameters(model_type, tau_channel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = arch_configs[arch]
    model =  TauMuNet(
        conv_params=params["conv_params"],
        dropout=0.2633831214702671,
        fc_params=params["fc_params"],
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_level_features=num_graph_level_features,
        use_fusion=params["use_fusion"],
        gnn_mode=gnn_mode,
    )
    model.load_state_dict(torch.load(output_folder + model_name, map_location=device)['model_state_dict'])
    # Load the normalizer
    normalizer = load_pkl(output_folder, "Normalization.pkl")[0]


    print("Starting the ScoreProducer")


    ScoreProducer(root=input_file,
                edge_connection_mode="knn",
                delta_r_threshold=None,
                k_knn=10,
                max_events_to_process=100000,
                save_edge_attributes=True,
                verbose=True,
                tau_channel=tau_channel,
                slimmed=True,
                shuffle_bool=True,
                use_displacement=True,
                model=model,
                device=device,
                normalizer=normalizer,
                output_folder=output_folder
            )