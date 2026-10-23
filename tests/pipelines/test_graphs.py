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
                bdt_preselection_cut=0.8, label=1.)
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


def test_no_cut_one_graph_per_candidate(tmp_path):
    data = training_events([[10.0, 20.0]], bdt=[0.9])
    g = _build(tmp_path, data, bdt_preselection_cut=None)
    assert _pts(g) == [10.0, 20.0]


class RecordingGNN(torch.nn.Module):
    """Keeps every graph it is called with; score = 0."""

    def __init__(self):
        super().__init__()
        self.graphs = []

    def forward(self, data):
        self.graphs.append(data.clone())
        return torch.zeros((1, 1))


def _rich_events(n, rng):
    """n signal events (one candidate, distinct pt) with varied PF candidates: three on the muons
    (removed by delete_muons), 15 inside the cone, 2 outside; varied pt, charge, pdgId, d0, dz."""
    from test_gnn_scores import MU_ETA, PF
    data = training_events([[10.0 + 5 * i] for i in range(n)], bdt=[0.9] * n)
    etas, phis = [], []
    for _ in range(n):
        inside = rng.uniform(-0.3, 0.3, (15, 2))
        etas.append([MU_ETA[1], MU_ETA[2], MU_ETA[3], *inside[:, 0], 2.5, -2.5])
        phis.append([0.0, 0.0, 0.0, *inside[:, 1], 1.0, -1.0])
    n_pf = len(etas[0])
    data[f"{PF}eta"], data[f"{PF}phi"] = ak.Array(etas), ak.Array(phis)
    data[f"{PF}pt"] = ak.Array([list(rng.uniform(0.5, 20, n_pf)) for _ in range(n)])
    data[f"{PF}charge"] = ak.Array([list(rng.choice([-1, 0, 1], n_pf)) for _ in range(n)])
    data[f"{PF}pdgId"] = ak.Array([list(rng.choice([211, -211, 22, 130], n_pf)) for _ in range(n)])
    for name in ("d0", "dz"):
        data[f"{PF}{name}"] = ak.Array([list(rng.normal(0, 0.05, n_pf)) for _ in range(n)])
    data[f"{PF}mass"] = ak.Array([list(rng.uniform(0, 0.5, n_pf)) for _ in range(n)])
    data[f"{PF}puppiWeight"] = ak.Array([list(rng.uniform(0, 1, n_pf)) for _ in range(n)])
    return data


def test_training_graph_equals_inference_graph(tmp_path):
    from tau3mu_gnn.gnn_training.ScoreProducer import score_events
    data = _rich_events(5, np.random.default_rng(7))
    training = _build(tmp_path, data, weights=None)
    infile = write_tree(tmp_path / "inference.root", data)
    rec = RecordingGNN()
    score_events([str(infile)], {"rec": (rec, None)}, {"rec": {str(infile): str(tmp_path / "out.root")}},
                 {"rec": tmp_path / "s.npz"}, "W", "delete_muons", 0.8, device=torch.device("cpu"))
    assert len(training) == len(rec.graphs) == 5

    def key(g):
        return round(float(g.graph_level_features[0, 0]), 4)

    inference = {key(g): g for g in rec.graphs}
    for t in training:
        i = inference[key(t)]
        assert torch.equal(t.x.float(), i.x.float())
        assert torch.equal(t.graph_level_features.float(), i.graph_level_features.float())
        edges = lambda g: sorted(zip(g.edge_index.T.tolist(), g.edge_attr.float().tolist()))  # noqa: E731
        assert edges(t) == edges(i)
        assert t.x.shape[0] > 11   # knn with k = 10 really selects


NAME = "skimmed_dzd0_cone_size_0.5_knn_10_delete_muons"


def _stage_inputs(tmp_path):
    sig = write_tree(tmp_path / "in" / "sig.root", training_events([[10.0], [20.0]], bdt=[0.9, 0.9], npvs=[30, 40]))
    bkg = write_tree(tmp_path / "in" / "bkg.root",
                     training_events([[30.0], [40.0]], bdt=[0.9, 0.9], masses=[[1.65], [1.90]]))
    npz = tmp_path / "in" / "signal_W_npvs_weights.npz"
    np.savez(npz, weight_per_bin=np.arange(100.0), bins=np.linspace(0, 100, 101))
    return sig, bkg, npz


def test_stage_layout_and_names(tmp_path, load_training, training_user):
    from tau3mu_gnn.pipelines.training.stages import graphs as stage
    cfg = load_training(training_user)
    sig, bkg, npz = _stage_inputs(tmp_path)
    out = tmp_path / "graphs"
    ps = stage.run(cfg, "signal", sig, out, npvs_file=npz)
    pb = stage.run(cfg, "background", bkg, out)
    assert stage.model_type(cfg) == NAME
    assert ps == out / "signal_W" / "processed" / f"W_decay_signal_{NAME}.pt" and ps.is_file()
    assert pb == out / "background_W" / "processed" / f"W_decay_background_{NAME}.pt" and pb.is_file()
    s = list(ConeGraphProducer(str(out / "signal_W"), output_dataset_name=f"W_decay_signal_{NAME}"))
    b = list(ConeGraphProducer(str(out / "background_W"), output_dataset_name=f"W_decay_background_{NAME}"))
    assert sorted(float(g.w) for g in s) == [30.0, 40.0] and all(float(g.y) == 1 for g in s)
    assert all(float(g.w) == 1.0 and float(g.y) == 0 for g in b) and len(b) == 2
    assert not (out / "signal_W" / "input.root").exists() and not (out / "background_W" / "input.root").exists()
    assert sorted(p.name for p in (out / "signal_W").iterdir()) == ["processed"]


def test_stage_signal_needs_npvs_background_refuses_it(tmp_path, load_training, training_user):
    from tau3mu_gnn.pipelines.training.stages import graphs as stage
    cfg = load_training(training_user)
    sig, bkg, npz = _stage_inputs(tmp_path)
    with pytest.raises(ValueError, match="npvs"):
        stage.run(cfg, "signal", sig, tmp_path / "a")
    with pytest.raises(ValueError, match="signal only"):
        stage.run(cfg, "background", bkg, tmp_path / "b", npvs_file=npz)


def test_stage_missing_branch(tmp_path, load_training, training_user):
    from tau3mu_gnn.pipelines.common.branches import BranchError
    from tau3mu_gnn.pipelines.training.stages import graphs as stage
    data = training_events([[30.0]], bdt=[0.9], masses=[[1.65]])
    data.pop("bdt_ISO")
    bkg = write_tree(tmp_path / "nobdt.root", data)
    with pytest.raises(BranchError, match="nobdt.root"):
        stage.run(load_training(training_user), "background", bkg, tmp_path / "g")
    assert not list((tmp_path / "g").rglob("*.pt")) if (tmp_path / "g").exists() else True


def test_stage_refuses_existing_folder(tmp_path, load_training, training_user):
    from tau3mu_gnn.pipelines.training.stages import graphs as stage
    cfg = load_training(training_user)
    _, bkg, _ = _stage_inputs(tmp_path)
    stage.run(cfg, "background", bkg, tmp_path / "g")
    with pytest.raises(FileExistsError):
        stage.run(cfg, "background", bkg, tmp_path / "g")


def _set(data, name, values):
    data[name] = ak.Array(values)


def test_training_graph_equals_inference_graph_edge_cases(tmp_path):
    """Event 0: two candidates (candidate 0 must be used: other pt, eta, phi than candidate 1).
    Event 1: muon 1 at DeltaR 0.6 (cone grows). Event 2: tau and PF candidates near phi = +-pi."""
    from test_gnn_scores import PF
    from tau3mu_gnn.gnn_training.ScoreProducer import score_events
    rng = np.random.default_rng(11)
    data = _rich_events(3, rng)
    cands = [[10.0, 50.0], [15.0], [20.0]]
    one = training_events(cands, bdt=[0.9] * 3)   # per-candidate branches with 2 entries for event 0
    for name, arr in one.items():
        if name.startswith(P) or name.startswith("TauPlusMET_") or name == "nSelectedTauTo3Mu":
            data[name] = arr
    _set(data, f"{P}fitted_pt", cands)
    _set(data, f"{P}fitted_eta", [[0.0, 0.3], [0.0], [0.0]])
    _set(data, f"{P}fitted_phi", [[0.0, -0.2], [0.0], [3.1]])
    _set(data, f"{P}mu1_eta", [[0.01, 0.31], [0.6], [0.01]])
    for i in (1, 2, 3):
        _set(data, f"{P}mu{i}_phi", [[0.0, -0.2], [0.0], [3.1]])
    phis = [list(x) for x in ak.to_list(data[f"{PF}phi"])]
    etas = [list(x) for x in ak.to_list(data[f"{PF}eta"])]
    etas[1][0] = 0.6                                        # PF candidate on the far muon
    etas[1][3], phis[1][3] = 0.55, 0.0                      # inside only because the cone grows to 0.6
    etas[1][4], phis[1][4] = -0.52, 0.1                     # inside only because the cone grows to 0.6
    phis[2] = [3.1, 3.1, 3.1] + [((p + 3.1 + np.pi) % (2 * np.pi)) - np.pi for p in phis[2][3:]]  # wraps at +-pi
    _set(data, f"{PF}phi", phis)
    _set(data, f"{PF}eta", etas)

    training = _build(tmp_path, data, weights=None)
    infile = write_tree(tmp_path / "inference.root", data)
    rec = RecordingGNN()
    score_events([str(infile)], {"rec": (rec, None)}, {"rec": {str(infile): str(tmp_path / "out.root")}},
                 {"rec": tmp_path / "s.npz"}, "W", "delete_muons", 0.8, device=torch.device("cpu"))
    assert len(training) == len(rec.graphs) == 3
    inference = {round(float(g.graph_level_features[0, 0]), 4): g for g in rec.graphs}
    assert sorted(inference) == [10.0, 15.0, 20.0]          # candidate 0 of event 0 in both
    for t in training:
        i = inference[round(float(t.graph_level_features[0, 0]), 4)]
        assert torch.equal(t.x.float(), i.x.float())
        assert torch.equal(t.graph_level_features.float(), i.graph_level_features.float())
        edges = lambda g: sorted(zip(g.edge_index.T.tolist(), g.edge_attr.float().tolist()))  # noqa: E731
        assert edges(t) == edges(i)


def test_graph_creation_cli_takes_a_cut_value():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-c",
                          "import sys; sys.argv=['x','--graph_type','knn','--k','10','--bdt_preselection_cut','0.8','--help'];"
                          "import tau3mu_gnn.scripts.GraphCreation"], capture_output=True, text=True)
    assert "--bdt_preselection_cut BDT_PRESELECTION_CUT" in out.stdout
