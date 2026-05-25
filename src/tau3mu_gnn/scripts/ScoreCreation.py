from tau3mu_gnn.gnn_training.ScoreProducer import ScoreProducer
import torch
import json
import os
import numpy as np
from tau3mu_gnn.model_eval.ModelEval import CPU_Unpickler, load_pkl
from tau3mu_gnn.architectures.TauMuNet import TauMuNet
from tau3mu_gnn.gnn_training.TrainingUtils import get_model_parameters

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='Folder to read the data from')
    parser.add_argument('--output_folder', type=str, required=True, help="Output to store the files with the scores")
    parser.add_argument('--model_names', type=str, required=True, help="Name of GNN model", nargs='+')
    parser.add_argument('--archs', type=str, required=True, help="Architecture of GNN model", nargs='+')
    parser.add_argument('--tau_channel', type=str, required=True, help="Tau production channel")
    parser.add_argument('--gnn_mode', type=str, default="BDT", help="GNN Mode")
    parser.add_argument('--model', type=str, default="skimmed_dzd0_cone_size_0.5_knn_10", help="Which model to train/ which graph construction was used")
    parser.add_argument('--storage_folder', type=str, required=True, help="Output to store the root files with the scores")
    parser.add_argument('--max_events_to_process', type=int, default=None, help="max_events_to_process")
    parser.add_argument('--conv_types', type=str, required=True, help="Conv layer types", nargs='+')
    parser.add_argument("--unsculpting_mode", type=str, default=None, help="Unsculting mode. Please choose 'permutation', 'split_pt_uniformly', 'delete_muons', None")
    parser.add_argument("--bdt_preselection_cut", type=float, default=None, help="BDT preselection score cut")

    args = parser.parse_args()

    input_file = args.root
    output_folder = args.output_folder
    model_names = args.model_names
    archs = args.archs
    tau_channel = args.tau_channel
    gnn_mode = args.gnn_mode
    model_type = args.model
    storage_folder = args.storage_folder
    max_events_to_process = args.max_events_to_process
    conv_types = args.conv_types
    unsculpting_mode = args.unsculpting_mode
    bdt_preselection_cut = args.bdt_preselection_cut

    num_node_features, num_edge_features, num_graph_level_features = get_model_parameters(model_type, tau_channel)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(model_names) == len(archs) == len(conv_types), f"Number of model names ({len(model_names)}) must match number of architectures ({len(archs)}). Got model_names={model_names}, archs={archs}"
    
    output_folders = []
    models = []
    normalizers = []
    for i, model_name in enumerate(model_names):
        model_direc = output_folder + model_name + "/"
        with open(os.path.join(model_direc, "arch_config.json")) as f:
            params = json.load(f)

        model =  TauMuNet(
            conv_params=params["conv_params"],
            dropout=params["dropout"],
            fc_params=params["fc_params"],
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_graph_level_features=num_graph_level_features,
            use_fusion=params["use_fusion"],
            gnn_mode=gnn_mode,
            scoring=True, 
            scale_down_params=params["fusion_scale_down_params"],
            conv_type=conv_types[i],
        )
        model.load_state_dict(torch.load(model_direc + model_name + ".pt", map_location=device)['model_state_dict'])
        # Load the normalizer
        normalizer = load_pkl(model_direc, "Normalization.pkl")[0]

        models.append(model)
        normalizers.append(normalizer)
        output_folders.append(model_direc)


    print("Starting the ScoreProducer")

    # max_events_to_process = 100000

    if max_events_to_process is None:
        shuffle = False
    else:
        shuffle = True

    ScoreProducer(root=input_file,
                edge_connection_mode="knn",
                delta_r_threshold=None,
                k_knn=10,
                max_events_to_process=max_events_to_process,
                save_edge_attributes=True,
                verbose=True,
                tau_channel=tau_channel,
                slimmed=True,
                shuffle_bool=shuffle,
                use_displacement=True,
                models=models,
                device=device,
                normalizers=normalizers,
                output_folders=output_folders,
                storage_folder=storage_folder,
                model_names=model_names,
                unsculpting_mode=unsculpting_mode,
                bdt_preselection_cut=bdt_preselection_cut
            )