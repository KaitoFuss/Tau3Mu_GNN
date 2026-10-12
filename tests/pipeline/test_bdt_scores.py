import pickle

import awkward as ak
import numpy as np
import pytest
import uproot

from conftest import write_tree
from tau3mu_gnn.bdt_training.bdt_inputs import input_branch_names
from tau3mu_gnn.pipeline.branches import BranchError
from tau3mu_gnn.pipeline.stages import bdt as bdt_stage

PT = "Selected_TauTo3Mu_fitted_pt"


class ToyClassifier:
    """Signal probability = first-candidate fitted pt / 1000."""

    def predict_proba(self, x):
        p = x[PT].to_numpy(dtype=float) / 1000
        return np.column_stack([1 - p, p])


def _write_classifier(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({0: ToyClassifier(), 1: ToyClassifier()}, f)
    return path


def _events(pts, first_event=10, gnn_score=False, drop=()):
    """One event per entry of pts (list of candidate pts). Every BDT branch, plus an extra branch."""
    n = len(pts)
    data = {}
    for name in sorted(input_branch_names("W")):
        if name in drop:
            continue
        if name == PT:
            data[name] = ak.Array(pts)
        elif name.startswith("Selected_TauTo3Mu_"):
            data[name] = ak.Array([[1.0 + j for j in range(len(ev))] for ev in pts])
        elif name.startswith("TauPlusMET_"):
            data[name] = ak.Array([[0.5 + j for j in range(len(ev))] for ev in pts])
        else:
            data[name] = np.ones(n, dtype=np.uint32)
    data["event"] = np.arange(first_event, first_event + n, dtype=np.uint64)
    data["extra"] = np.arange(n, dtype=np.float64)
    if gnn_score:
        data["GNN_score"] = np.full(n, 0.7)
    return data


# event 0: two candidates (first one must be used), event 1: one, event 2: none
SIG_PTS = [[100.0, 900.0], [300.0], []]
BKG_PTS = [[50.0], [], [700.0, 20.0]]


def _run(tmp_path, load, user, sig_data=None, bkg_data=None):
    cfg = load(user)
    _write_classifier(cfg.bdt.classifier)  # replaces the placeholder file of fake_repo
    sig = write_tree(tmp_path / "signal.root", sig_data or _events(SIG_PTS))
    bkg = write_tree(tmp_path / "background.root", bkg_data or _events(BKG_PTS, first_event=20, gnn_score=True))
    out_dir = tmp_path / "2_bdt"
    return bdt_stage.run(cfg, {"signal": sig, "background": bkg}, out_dir), out_dir


def test_stage_output_names(tmp_path, load, user):
    outputs, out_dir = _run(tmp_path, load, user)
    assert outputs == {
        "signal": out_dir / "signal_with_bdt_scores.root",
        "background": out_dir / "background_with_bdt_scores.root",
        "scores_pickle": out_dir / "tau_score_corr.pkl",
    }
    assert all(p.is_file() for p in outputs.values())
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(p.name for p in outputs.values())  # no temp files left


def test_first_candidate_scored_and_no_candidate_gets_minus_one(tmp_path, load, user):
    outputs, _ = _run(tmp_path, load, user)
    sig = uproot.open(outputs["signal"])["Events"]
    bkg = uproot.open(outputs["background"])["Events"]
    np.testing.assert_array_equal(sig["bdt_ISO"].array(library="np"), np.float32([0.1, 0.3, -1]))
    np.testing.assert_array_equal(bkg["bdt_ISO"].array(library="np"), np.float32([0.05, -1, 0.7]))


def test_all_events_and_branches_kept_gnn_score_dropped(tmp_path, load, user):
    outputs, _ = _run(tmp_path, load, user)
    for sample, in_data in [("signal", _events(SIG_PTS)), ("background", _events(BKG_PTS, 20, gnn_score=True))]:
        tree = uproot.open(outputs[sample])["Events"]
        assert tree.num_entries == 3
        expected = (set(in_data) - {"GNN_score"}) | {"bdt_ISO"}
        assert {k for k in tree.keys() if not k.startswith("n")} == expected
        np.testing.assert_array_equal(tree["event"].array(library="np"), in_data["event"])
        np.testing.assert_array_equal(tree["extra"].array(library="np"), in_data["extra"])


def test_scores_pickle(tmp_path, load, user):
    outputs, _ = _run(tmp_path, load, user)
    with open(outputs["scores_pickle"], "rb") as f:
        out = pickle.load(f)
    assert set(out) == {"ISO_score_sig", "ISO_score_bkg", "tau_mass"}
    np.testing.assert_allclose(np.asarray(out["ISO_score_sig"]), [0.1, 0.3])
    np.testing.assert_allclose(np.asarray(out["ISO_score_bkg"]), [0.05, 0.7])
    np.testing.assert_array_equal(np.asarray(out["tau_mass"]), [1.0, 1.0])  # first-candidate mass


def test_missing_branch_crashes_before_any_work(tmp_path, load, user):
    sig = _events(SIG_PTS, drop={"Selected_TauTo3Mu_vtx_Ndof", "TauPlusMET_Puppi_phi"})
    with pytest.raises(BranchError) as excinfo:
        _run(tmp_path, load, user, sig_data=sig)
    problems = "\n".join(excinfo.value.problems)
    assert "Selected_TauTo3Mu_vtx_Ndof" in problems and "TauPlusMET_Puppi_phi" in problems
    assert "signal.root" in problems
    assert not (tmp_path / "2_bdt").exists()


def test_several_files_per_sample(tmp_path):
    from tau3mu_gnn.bdt_training.BDT_ScoreProducer import score_events

    clf = _write_classifier(tmp_path / "clf.pck")
    sig = write_tree(tmp_path / "s.root", _events(SIG_PTS))
    bkg = {str(write_tree(tmp_path / f"b{i}.root", _events(BKG_PTS, first_event=20 + 10 * i))): str(tmp_path / f"b{i}_out.root")
           for i in (0, 1)}
    score_events({str(sig): str(tmp_path / "s_out.root")}, bkg, clf, "W", tmp_path / "scores.pkl")
    for out in bkg.values():
        np.testing.assert_array_equal(uproot.open(out)["Events"]["bdt_ISO"].array(library="np"),
                                      np.float32([0.05, -1, 0.7]))
    with open(tmp_path / "scores.pkl", "rb") as f:
        assert len(pickle.load(f)["ISO_score_bkg"]) == 4


def test_old_cli_keeps_old_paths(monkeypatch, tmp_path):
    from tau3mu_gnn.bdt_training import BDT_ScoreProducer as producer

    calls = []
    monkeypatch.setattr(producer, "score_events", lambda *a: calls.append(a))
    producer.main(["--model", "m", "--classifier_iso", "clf.pck", "--tau_channel", "W",
                   "--storage_folder", str(tmp_path)])
    base = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data"
    assert calls == [(
        {f"{base}/signal_W/m/selected_signal_skimmed_no_resonances_with_scores.root":
         f"{tmp_path}/m/selected_signal_skimmed_no_resonances_with_scores_bdt_full.root"},
        {f"{base}/background_W/m/selected_data_skimmed_no_resonances_with_scores.root":
         f"{tmp_path}/m/selected_data_skimmed_no_resonances_with_scores_bdt_full.root"},
        "clf.pck", "W", "/work/kfuss/SVJ_GNN/output/m/tau_score_corr.pkl",
    )]
