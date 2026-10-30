import pytest
import torch

from conftest import REPO_ROOT

MODEL = "W_raw_Nano_AOD_fixed_W_fl_8_sd_4_GINE_delete_muons"
ARGS = dict(conv_type="GINE", gnn_mode="final_layer",
            graph_type="skimmed_dzd0_cone_size_0.5_knn_10", tau_channel="W", device=torch.device("cpu"))


def test_loads_frozen_w_model():
    from tau3mu_gnn.architectures.TauMuNet import TauMuNet
    from tau3mu_gnn.gnn_training.model_loading import load_model

    model, normalizer = load_model(REPO_ROOT / "models/W/gnn", MODEL, **ARGS)
    assert isinstance(model, TauMuNet)
    assert not model.training
    assert {"means_x", "stds_x", "means_edge", "stds_edge", "means_graph", "stds_graph"} <= set(normalizer)


def test_missing_model_names_path(tmp_path):
    from tau3mu_gnn.gnn_training.model_loading import load_model

    with pytest.raises(FileNotFoundError, match="no_such_model"):
        load_model(tmp_path, "no_such_model", **ARGS)
