import os

import awkward as ak
import numpy as np
import pytest
import uproot

from tau3mu_gnn.pipeline import branches as b
from tau3mu_gnn.pipeline.stages import skim as skim_stage
from tau3mu_gnn.utils.skimmer import skim

REQ_W = b.required_branches("W")

# Per event: candidates (dZmu12, dZmu13, dZmu23, cos_alpha). Cuts: |dZ| < 0.5, cos > 0.5 (strict).
CANDIDATES = [
    [(0.1, -0.2, 0.3, 0.9), (0.5, 0.0, 0.0, 0.9), (-0.49, 0.49, 0.0, 0.51)],  # pass, dZ edge, pass
    [(0.0, 0.0, 0.0, 0.5), (0.0, -0.5, 0.0, 0.9)],                             # cos edge, dZ edge
    [],                                                                        # no candidate
]
EXPECTED_MASK = [[True, False, True], [False, False], []]


def _toy_events():
    cols = {name: [[c[i] for c in ev] for ev in CANDIDATES]
            for i, name in enumerate(["dZmu12", "dZmu13", "dZmu23", "CosAlpha2D_LxyP3mu"])}
    data = {}
    for name in REQ_W:
        short = name.removeprefix("Slimmed_TauTo3Mu_")
        if name.startswith("Slimmed_TauTo3Mu_"):
            data[name] = ak.Array(cols.get(short, [[float(j) for j in range(len(ev))] for ev in CANDIDATES]))
        elif name.startswith(("Slimmed_PFCands_", "TauPlusMET_")):
            data[name] = ak.Array([[1.0, 2.0], [3.0], [4.0]])
        else:
            data[name] = np.array([1, 2, 3], dtype=np.int64)
    data["event"] = np.array([10, 11, 12], dtype=np.int64)
    # A Slimmed_TauTo3Mu branch outside the required list must be selected too.
    data["Slimmed_TauTo3Mu_extra"] = ak.Array([[7.0, 8.0, 9.0], [7.0, 8.0], []])
    return data


def _write(path, data):
    with uproot.recreate(path) as f:
        f["Events"] = data
    return path


def _skim(tmp_path, data, branch_map=None, **kw):
    src = _write(tmp_path / "in.root", data)
    out = tmp_path / "out.root"
    skim([str(src)], str(out), "Events", branch_map or {}, dz_max=0.5, cos_alpha_2d_min=0.5, **kw)
    return uproot.open(out)["Events"]


def test_mask_cut_edges_and_counts(tmp_path):
    tree = _skim(tmp_path, _toy_events())
    assert tree.num_entries == 3  # no event is removed
    assert tree["tau3mu_mask"].array().tolist() == EXPECTED_MASK
    assert tree["nSelectedTauTo3Mu"].array().tolist() == [2, 0, 0]
    assert tree["Selected_TauTo3Mu_dZmu12"].array().tolist() == [[0.1, -0.49], [], []]
    assert tree["Selected_TauTo3Mu_extra"].array().tolist() == [[7.0, 9.0], [], []]


def test_every_slimmed_tau_branch_gets_a_selected_copy(tmp_path):
    keys = set(_skim(tmp_path, _toy_events()).keys())
    slimmed = {k for k in keys if k.startswith("Slimmed_TauTo3Mu_")}
    assert {k.replace("Slimmed", "Selected", 1) for k in slimmed} <= keys
    assert set(_toy_events()) <= keys  # input branches are kept


def test_cut_values_come_from_arguments(tmp_path):
    src = _write(tmp_path / "in.root", _toy_events())
    out = tmp_path / "out.root"
    skim([str(src)], str(out), "Events", {}, dz_max=0.55, cos_alpha_2d_min=0.95)
    assert uproot.open(out)["Events"]["nSelectedTauTo3Mu"].array().tolist() == [0, 0, 0]


def test_branch_map_applied_and_user_name_dropped(tmp_path):
    data = _toy_events()
    data["MyDz12"] = data.pop("Slimmed_TauTo3Mu_dZmu12")
    tree = _skim(tmp_path, data, {"Slimmed_TauTo3Mu_dZmu12": "MyDz12"})
    keys = set(tree.keys())
    assert "MyDz12" not in keys
    assert tree["Slimmed_TauTo3Mu_dZmu12"].array().tolist()[0] == [0.1, 0.5, -0.49]
    assert tree["Selected_TauTo3Mu_dZmu12"].array().tolist() == [[0.1, -0.49], [], []]


def test_max_events(tmp_path):
    assert _skim(tmp_path, _toy_events(), max_events=2).num_entries == 2


# --- stage -------------------------------------------------------------------

def test_stage_writes_one_file_per_sample(load, user, tmp_path):
    sig = _write(tmp_path / "sig.root", _toy_events())
    _write(tmp_path / "bkg_1.root", _toy_events())
    _write(tmp_path / "bkg_2.root", _toy_events())
    user["inputs"]["signal"] = [str(sig)]
    user["inputs"]["background"] = [str(tmp_path / "bkg_*.root")]
    out_dir = tmp_path / "1_skim"
    outputs = skim_stage.run(load(user), out_dir)
    assert outputs == {"signal": out_dir / "signal.root", "background": out_dir / "background.root"}
    assert uproot.open(outputs["signal"])["Events"].num_entries == 3
    assert uproot.open(outputs["background"])["Events"].num_entries == 6


def test_stage_checks_inputs_before_any_work(load, user, tmp_path):
    data = _toy_events()
    del data["run"]
    sig = _write(tmp_path / "sig.root", data)
    user["inputs"]["signal"] = [str(sig)]
    user["inputs"]["background"] = [str(sig)]
    out_dir = tmp_path / "1_skim"
    with pytest.raises(b.BranchError, match="run"):
        skim_stage.run(load(user), out_dir)
    assert not out_dir.exists()


# --- regression vs. the old skimmer output -------------------------------------

OLD_IN = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/old_data/signal_skimmed_no_resonances.root"
OLD_OUT = "/work/kfuss/SVJ_GNN/src/tau3mu_gnn/data/signal_W/selected_signal_skimmed_no_resonances.root"
KEY = ["run", "luminosityBlock", "event"]


def test_regression_vs_old_signal_output(tmp_path):
    if not (os.path.exists(OLD_IN) and os.path.exists(OLD_OUT)):
        pytest.skip("old W signal input/output not reachable on this machine")
    n = 2000
    out = tmp_path / "new.root"
    skim([OLD_IN], str(out), "Events", {}, dz_max=0.5, cos_alpha_2d_min=0.5, max_events=n)
    new = uproot.open(out)["Events"]
    old = uproot.open(OLD_OUT)["Events"]
    compare = sorted(k for k in new.keys() if k.startswith("Selected_")) + ["nSelectedTauTo3Mu", "tau3mu_mask"]
    assert len(compare) > 90

    new_arr = new.arrays(KEY + compare, library="ak")
    old_key = old.arrays(KEY, library="np")
    index = {k: i for i, k in enumerate(zip(*(old_key[c] for c in KEY)))}
    rows = np.array([index[k] for k in zip(*(ak.to_numpy(new_arr[c]) for c in KEY))])
    assert len(rows) == n

    order = np.argsort(rows)
    start, stop = int(rows.min()), int(rows.max()) + 1
    old_arr = old.arrays(compare, entry_start=start, entry_stop=stop, library="ak")[rows[order] - start]
    for name in compare:
        a, o = new_arr[name][order], old_arr[name]
        if a.ndim > 1:
            assert ak.num(a, axis=1).tolist() == ak.num(o, axis=1).tolist(), name
        np.testing.assert_array_equal(ak.to_numpy(ak.ravel(a)), ak.to_numpy(ak.ravel(o)), err_msg=name)
