import pickle

import awkward as ak
import numpy as np
import pytest

from conftest import write_tree
from tau3mu_gnn.bdt_training import train_bdt
from tau3mu_gnn.bdt_training.bdt_inputs import input_branch_names

MASS = "Selected_TauTo3Mu_fitted_mass"
PT = "Selected_TauTo3Mu_fitted_pt"


def _toy(path, n, rng, signal, masses=None, first_event=0):
    """n events, one candidate each (masses: list of per-event candidate lists overrides).
    Every BDT input branch; fitted_pt separates signal (~50) from background (~10)."""
    masses = masses or [[1.78 if signal else 1.65]] * n
    data = {}
    for name in sorted(input_branch_names("W")):
        if name == MASS:
            data[name] = ak.Array(masses)
        elif name.startswith(("Selected_TauTo3Mu_", "TauPlusMET_")):
            centre = (50.0 if signal else 10.0) if name == PT else 1.0
            data[name] = ak.Array([list(np.abs(rng.normal(centre, 2.0, len(m))) + 0.5) for m in masses])  # > 0: no inf in chi2/ndof
        else:
            data[name] = np.ones(len(masses), dtype=np.uint32)
    data["event"] = np.arange(first_event, first_event + len(masses), dtype=np.uint64)
    return write_tree(path, data)


@pytest.fixture
def params(load_training, training_user):
    load_training.defaults["bdt_training"].update(n_estimators=30, kfold=2, early_stopping_rounds=5, learning_rate=0.3)
    return load_training(training_user).bdt_training


def test_sideband_filter():
    ev = ak.Array({MASS: [[1.71], [1.60, 1.79], [1.95], [1.62, 1.90]]})
    assert train_bdt.sideband_filter(1.78, 0.08)(ev).tolist() == [False, False, True, True]


def test_load_training_data_sideband_and_sampling(tmp_path):
    rng = np.random.default_rng(0)
    sig = _toy(tmp_path / "s.root", 50, rng, True)
    bkg = _toy(tmp_path / "b.root", 0, rng, False,
               masses=[[1.65]] * 30 + [[1.78]] * 10 + [[1.65, 1.80]] * 5, first_event=100)
    data, feats = train_bdt.load_training_data([sig], [bkg], "Events", "W", 1.78, 0.08, 40, 42)
    assert (data.target == 1).sum() == 40 and (data.target == 0).sum() == 30   # 40 = cap, 30 = sideband only
    assert feats[-1] == "relative_iso" and "deltaPhi(TauPlusMET_Puppi_phi, Selected_TauTo3Mu_fitted_phi)" in feats
    again, _ = train_bdt.load_training_data([sig], [bkg], "Events", "W", 1.78, 0.08, 40, 42)
    assert data.equals(again)   # same seed, same sample


def test_empty_class_raises(tmp_path):
    rng = np.random.default_rng(0)
    sig = _toy(tmp_path / "s.root", 20, rng, True)
    bkg = _toy(tmp_path / "b.root", 0, rng, False, masses=[[1.78]] * 20)   # nothing in the sideband
    with pytest.raises(ValueError, match="background"):
        train_bdt.load_training_data([sig], [bkg], "Events", "W", 1.78, 0.08, 100, 42)


def test_too_few_events_for_kfold(tmp_path, params):
    rng = np.random.default_rng(0)
    data, feats = train_bdt.load_training_data(
        [_toy(tmp_path / "s.root", 1, rng, True)], [_toy(tmp_path / "b.root", 40, rng, False)],
        "Events", "W", 1.78, 0.08, 100, 42)
    with pytest.raises(ValueError, match="kfold"):
        train_bdt.train_iso_bdt(data, feats, params)


def test_train_then_stage2_scores(tmp_path, params):
    from tau3mu_gnn.bdt_training.BDT_ScoreProducer import score_events
    rng = np.random.default_rng(1)
    sig = _toy(tmp_path / "s.root", 300, rng, True)
    bkg = _toy(tmp_path / "b.root", 300, rng, False, first_event=1000)
    data, feats = train_bdt.load_training_data([sig], [bkg], "Events", "W", 1.78, 0.08, 100_000, 42)
    classifiers, summary, test = train_bdt.train_iso_bdt(data, feats, params)
    assert sorted(classifiers) == [0, 1] and summary["auc_test"] > 0.95
    assert summary["n_signal"] == summary["n_background"] == 300
    clf = tmp_path / "ISO.pck"
    clf.write_bytes(pickle.dumps(classifiers))
    score_events({str(sig): str(tmp_path / "s_out.root")}, {str(bkg): str(tmp_path / "b_out.root")},
                 clf, "W", tmp_path / "p.pkl")
    p = pickle.loads((tmp_path / "p.pkl").read_bytes())
    s, b = np.asarray(p["ISO_score_sig"]), np.asarray(p["ISO_score_bkg"])
    assert 0 <= b.min() and s.max() <= 1 and s.mean() > b.mean() + 0.5


def test_plot_training_writes_files(tmp_path, params):
    rng = np.random.default_rng(2)
    data, feats = train_bdt.load_training_data(
        [_toy(tmp_path / "s.root", 100, rng, True)], [_toy(tmp_path / "b.root", 100, rng, False)],
        "Events", "W", 1.78, 0.08, 100_000, 42)
    classifiers, summary, test = train_bdt.train_iso_bdt(data, feats, params)
    paths = train_bdt.plot_training(summary, test, classifiers, feats, tmp_path / "plots")
    assert sorted(p.name for p in paths) == ["logloss_fold_1.png", "logloss_fold_2.png", "roc.png"]
    assert all(p.is_file() for p in paths)
