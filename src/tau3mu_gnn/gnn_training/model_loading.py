"""Load a trained TauMuNet and its feature normalizer."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from tau3mu_gnn.architectures.TauMuNet import TauMuNet
from tau3mu_gnn.gnn_training.TrainingUtils import get_model_parameters
from tau3mu_gnn.model_eval.ModelEval import CPU_Unpickler


def load_model(model_dir, model_name, conv_type, gnn_mode, graph_type, tau_channel, device):
    """Return (model in eval mode, normalizer dict) from <model_dir>/<model_name>/."""
    folder = Path(model_dir) / model_name
    with open(folder / "arch_config.json") as f:
        params = json.load(f)
    num_node_features, num_edge_features, num_graph_level_features = get_model_parameters(graph_type, tau_channel)
    model = TauMuNet(
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
        conv_type=conv_type,
    )
    model.load_state_dict(torch.load(folder / f"{model_name}.pt", map_location=device)["model_state_dict"])
    model.eval()
    # Same unpickler as ModelEval.load_pkl, which the old ScoreCreation used.
    with open(folder / "Normalization.pkl", "rb") as f:
        normalizer = CPU_Unpickler(f).load()
    return model, normalizer
