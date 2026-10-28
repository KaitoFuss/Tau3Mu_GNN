"""The training chain on toy data: bdt_train -> bdt_infer (new classifier) -> npvs -> graphs."""
import pickle

import awkward as ak
import numpy as np
import pytest
import uproot

from conftest import write_tree
from tau3mu_gnn.bdt_training.bdt_inputs import input_branch_names
from test_bdt_scores import PT, _write_classifier, _events
from test_graphs import training_events

P = "Selected_TauTo3Mu_"


def test_bdt_infer_uses_given_classifier(tmp_path, load_training, training_user):
    from tau3mu_gnn.pipelines.training.stages import bdt_infer
    cfg = load_training(training_user)
    clf = _write_classifier(tmp_path / "my.pck")
    sig = write_tree(tmp_path / "s.root", _events([[100.0], [300.0]]))
    bkg = write_tree(tmp_path / "b.root", _events([[50.0]], first_event=20))
    out = bdt_infer.run(cfg, {"signal": sig, "background": bkg}, clf, tmp_path / "2_bdt")
    scores = uproot.open(out["signal"])["Events"]["bdt_ISO"].array(library="np")
    np.testing.assert_allclose(sorted(scores), [0.1, 0.3])   # toy classifier: pt / 1000


def _chain_events(n, rng, signal, first_event, channel="W"):
    """Events with every BDT and graph branch of channel. fitted_pt overlaps between the classes."""
    pts = [[float(rng.normal(45 if signal else 30, 10))] for _ in range(n)]
    data = training_events(pts, bdt=[0.0] * n, masses=[[1.78 if signal else 1.65]] * n, first_event=first_event)
    data.pop("bdt_ISO")                       # made by bdt_infer
    if channel == "W":
        data["tau3mu_mask"] = ak.Array([[True] for _ in range(n)])  # stage 1 mask, one candidate per event
    if channel == "HF":                       # muon collection (3 muons), indexed by the candidate's mu*_idx
        for name in [k for k in data if k.startswith("TauPlusMET_")]:
            data.pop(name)
        for i in (1, 2, 3):
            data[f"{P}mu{i}_idx"] = ak.values_astype(ak.Array([[i - 1]] * n), np.int32)
        data[f"{P}charge"] = ak.values_astype(ak.Array([[1]] * n), np.int32)
        data["Muon_pt"] = ak.Array([[float(abs(rng.normal(6 if signal else 4, 1))) for _ in range(3)] for _ in range(n)])
        data["Muon_tkRelIso"] = ak.Array([[float(abs(rng.normal(0.2, 0.1))) for _ in range(3)] for _ in range(n)])
        data["Muon_isGlobal"] = ak.Array([[bool(rng.random() < 0.8) for _ in range(3)] for _ in range(n)])
    for name in sorted(input_branch_names(channel)):
        if name not in data:
            data[name] = ak.Array([[float(abs(rng.normal(1.0, 0.3))) + 0.5] for _ in range(n)])
    data[f"{P}fitted_pt"] = ak.Array(pts)
    return data


@pytest.mark.parametrize("channel", ["W", "HF"])
def test_chain_bdt_train_infer_graphs(tmp_path, load_training, training_user, channel):
    from tau3mu_gnn.pipelines.training.stages import bdt_infer, bdt_train, graphs, npvs
    load_training.defaults["bdt_training"].update(n_estimators=30, kfold=2, early_stopping_rounds=5,
                                                  learning_rate=0.3)
    training_user["graphs"]["bdt_preselection_cut"] = 0.5
    training_user["channel"] = channel
    cfg = load_training(training_user)
    rng = np.random.default_rng(3)
    skim = {s: write_tree(tmp_path / "1_skim" / f"{s}.root", _chain_events(120, rng, s == "signal", first, channel))
            for s, first in (("signal", 0), ("background", 1000))}
    trained = bdt_train.run(cfg, skim, tmp_path / "bdt_train")
    scored = bdt_infer.run(cfg, skim, trained["classifier"], tmp_path / "2_bdt")
    weights = npvs.run(cfg, skim, tmp_path / "npvs")["weights"]
    pt_files = {s: graphs.run(cfg, s, scored[s], tmp_path / "graphs", npvs_file=weights if s == "signal" else None)
                for s in graphs.SAMPLES}
    from tau3mu_gnn.graph.ConeGraphProducer import ConeGraphProducer
    for sample, path in pt_files.items():
        ds = list(ConeGraphProducer(str(path.parent.parent), output_dataset_name=path.stem))
        assert ds and all(float(g.bdt_iso) >= 0.5 for g in ds), sample
