
from utils.ScoreProducer import ScoreProducer
import torch
import pickle
from utils.PlotUtils import plot_histogram_root
import os
import numpy as np
from utils.ModelEval import CPU_Unpickler, load_pkl
from architectures.TauMuNet import TauMuNet

import argparse

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
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str, required=True, help='File to read the data from')
    parser.add_argument('--output_folder', type=str, required=True, help="Output to store the files with the scores")

    args = parser.parse_args()

    input_file = args.file
    output_folder = args.output_folder


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = arch_configs["arch6"]
    model =  TauMuNet(
        conv_params=params["conv_params"],
        dropout=0.6,
        fc_params=params["fc_params"],
        use_fusion=params["use_fusion"],
    )
    model.load_state_dict(torch.load(output_folder + "knn_new_vs11.pt", map_location=device)['model_state_dict'])
    # Load the normalizer
    normalizer = load_pkl(output_folder, "Normalization.pkl")[0]


    print("Starting the ScoreProducer")

    ScoreProducer(root_file=input_file,
                edge_connection_mode="knn",
                delta_r_threshold=None,
                k_knn=10,
                max_events_to_process=500000,
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
    
    

