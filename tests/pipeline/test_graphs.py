import awkward as ak
import numpy as np
import pytest
import torch

from conftest import write_tree
from test_gnn_scores import PF_GOOD, gnn_events
from tau3mu_gnn.gnn_training.gnn_inputs import graph_level_branches
from tau3mu_gnn.graph.ConeGraphProducer import ConeGraphProducer

P = "Selected_TauTo3Mu_"


def training_events(cands, bdt, masses=None, npvs=None, pf=None, first_event=10):
    """gnn_events plus the branches only the training graphs read (PV_npvs, iso_*).
    masses: fitted mass per candidate (default 1.70 + 0.01 * j)."""
    data = gnn_events(cands, bdt, pf=pf, first_event=first_event)
    n = len(cands)
    if masses is not None:
        data[f"{P}fitted_mass"] = ak.Array(masses)
    data["PV_npvs"] = np.asarray(npvs if npvs is not None else [30] * n, dtype=np.uint8)
    for iso in ("ptChargedFromPV", "ptPhotons", "ptChargedFromPU"):
        data[f"{P}iso_{iso}"] = ak.Array([[0.1] * len(c) for c in cands])
    return data


_count = [0]


def _build(tmp_path, data, **kw):
    """Write data into a fresh folder and build the graphs with the pipeline recipe (kw overrides)."""
    _count[0] += 1
    root = tmp_path / f"g{_count[0]}"
    write_tree(root / "input.root", data)
    args = dict(cone_size=0.5, edge_connection_mode="knn", k_knn=10, slimmed=True, use_displacement=True,
                unsculpting_mode="delete_muons", tau_channel="W", graph_level_branches=graph_level_branches("W"),
                save_benchmark_score=True, save_edge_attributes=True, use_relative_angles=True,
                first_candidate_only=True, bdt_cut=0.8, label=1.)
    args.update(kw)
    return list(ConeGraphProducer(str(root), output_dataset_name="t", **args))


def _pts(graphs):
    """Fitted pt of each graph (first graph-level feature)."""
    return sorted(round(float(g.graph_level_features[0, 0]), 4) for g in graphs)


def test_bdt_cut_inclusive(tmp_path):
    g = _build(tmp_path, training_events([[10.0], [20.0], [30.0], []], bdt=[0.79, 0.8, 0.95, -1]))
    assert _pts(g) == [20.0, 30.0]
    assert all(float(x.bdt_iso) >= 0.8 for x in g)


def test_background_candidate0_rule(tmp_path):
    data = training_events([[10.0, 11.0], [20.0, 21.0], [30.0]], bdt=[0.9] * 3,
                           masses=[[1.65, 1.78], [1.78, 1.65], [1.90]])
    g = _build(tmp_path, data, mask=True, label=0.)
    assert _pts(g) == [10.0, 30.0]
    assert sorted(round(float(x.fitted_tau_mass), 2) for x in g) == [1.65, 1.90]


def test_sideband_uses_config_values(tmp_path):
    data = training_events([[10.0]], bdt=[0.9], masses=[[1.75]])
    with pytest.raises(ValueError, match="no graphs"):
        _build(tmp_path, data, mask=True, label=0.)
    assert len(_build(tmp_path, data, mask=True, label=0., tau_mass=1.80, sideband_half_width=0.02)) == 1


def test_npvs_weight_clipped(tmp_path):
    npz = tmp_path / "w.npz"
    np.savez(npz, weight_per_bin=np.arange(100.0), bins=np.linspace(0, 100, 101))
    data = training_events([[10.0], [20.0], [30.0]], bdt=[0.9] * 3, npvs=[30, 100, 150])
    g = _build(tmp_path, data, weights="npvs_rescale", npvs_weights_file=str(npz))
    assert {round(float(x.graph_level_features[0, 0])): float(x.w) for x in g} == {10: 30.0, 20: 99.0, 30: 99.0}


def test_background_weight_one(tmp_path):
    data = training_events([[10.0], [20.0]], bdt=[0.9] * 2, masses=[[1.65], [1.90]])
    assert all(float(x.w) == 1.0 for x in _build(tmp_path, data, mask=True, label=0.))


def test_event_limit_seeded(tmp_path):
    data = training_events([[float(10 + i)] for i in range(20)], bdt=[0.9] * 20)
    a = _pts(_build(tmp_path, data, max_events_to_process=5, shuffle_bool=True, shuffle_seed=1))
    b = _pts(_build(tmp_path, data, max_events_to_process=5, shuffle_bool=True, shuffle_seed=1))
    c = _pts(_build(tmp_path, data, max_events_to_process=5, shuffle_bool=True, shuffle_seed=2))
    assert len(a) == 5 and a == b and a != c
    assert len(_build(tmp_path, data, max_events_to_process=50, shuffle_bool=True, shuffle_seed=1)) == 20


def test_defaults_unchanged(tmp_path):
    data = training_events([[10.0, 20.0]], bdt=[0.9])
    g = _build(tmp_path, data, first_candidate_only=False, bdt_cut=None)
    assert _pts(g) == [10.0, 20.0]
