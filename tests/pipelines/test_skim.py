import os

import awkward as ak
import numpy as np
import pytest
import uproot

from tau3mu_gnn.pipelines.common import branches as b
from tau3mu_gnn.pipelines.common.stages import skim as skim_stage
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


def _skim(tmp_path, data):
    src = _write(tmp_path / "in.root", data)
    out = tmp_path / "out.root"
    skim([str(src)], str(out), "Events", dz_max=0.5, cos_alpha_2d_min=0.5)
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
    skim([str(src)], str(out), "Events", dz_max=0.55, cos_alpha_2d_min=0.95)
    assert uproot.open(out)["Events"]["nSelectedTauTo3Mu"].array().tolist() == [0, 0, 0]


# --- stage -------------------------------------------------------------------

def test_stage_writes_one_file_per_sample(load, user, tmp_path):
    sig = _write(tmp_path / "sig.root", _toy_events())
    bkg = [str(_write(tmp_path / f"bkg_{i}.root", _toy_events())) for i in (1, 2)]
    out_dir = tmp_path / "1_skim"
    outputs = skim_stage.run(load(user), {"signal": [str(sig)], "background": bkg}, out_dir)
    assert outputs == {"signal": out_dir / "signal.root", "background": out_dir / "background.root"}
    assert uproot.open(outputs["signal"])["Events"].num_entries == 3
    assert uproot.open(outputs["background"])["Events"].num_entries == 6


# --- regression vs. the old skimmer output -------------------------------------

OLD_IN = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/old_data/signal_skimmed_no_resonances.root"
OLD_OUT = "/work/kfuss/SVJ_GNN/src/tau3mu_gnn/data/signal_W/selected_signal_skimmed_no_resonances.root"
KEY = ["run", "luminosityBlock", "event"]


def test_regression_vs_old_signal_output(tmp_path):
    if not (os.path.exists(OLD_IN) and os.path.exists(OLD_OUT)):
        pytest.skip("old W signal input/output not reachable on this machine")
    import ROOT

    n = 2000
    small = tmp_path / "first_events.root"
    ROOT.DisableImplicitMT()  # RDataFrame.Range needs a single thread
    ROOT.RDataFrame("Events", OLD_IN).Range(n).Snapshot("Events", str(small))
    out = tmp_path / "new.root"
    skim([str(small)], str(out), "Events", dz_max=0.5, cos_alpha_2d_min=0.5)
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


# --- old CLI ---------------------------------------------------------------------

def _cli(monkeypatch, argv):
    from tau3mu_gnn.utils import skimmer

    calls = []
    monkeypatch.setattr(skimmer, "skim", lambda *a, **kw: calls.append((a, kw)))
    skimmer.main(argv)
    return calls


def test_cli_without_config_uses_default_cuts(monkeypatch, tmp_path):
    for name in ("a_1.root", "a_2.root"):
        (tmp_path / name).write_bytes(b"x")
    calls = _cli(monkeypatch, ["--input", str(tmp_path / "a_*.root"), "--output", "out.root"])
    assert calls == [(([str(tmp_path / "a_1.root"), str(tmp_path / "a_2.root")], "out.root", "Events"),
                      {"dz_max": 0.5, "cos_alpha_2d_min": 0.5})]


def test_cli_with_config_reads_cuts_and_tree(monkeypatch, fake_repo, user, tmp_path):
    from conftest import TOY_DEFAULTS, write_yaml
    from tau3mu_gnn.pipelines.analysis import config
    from tau3mu_gnn.utils import skimmer

    write_yaml(fake_repo / "configs" / "analysis_defaults_W.yaml",
               {**TOY_DEFAULTS, "skim": {"dz_max": 0.3, "cos_alpha_2d_min": 0.7}})
    user["inputs"]["tree_name"] = "MyTree"
    write_yaml(fake_repo / "user.yaml", user)
    real_load = config.load_config
    monkeypatch.setattr(skimmer, "load_config", lambda p: real_load(p, defaults_dir=fake_repo / "configs"))
    (tmp_path / "in.root").write_bytes(b"x")
    calls = _cli(monkeypatch, ["--config", str(fake_repo / "user.yaml"),
                               "--input", str(tmp_path / "in.root"), "--output", "out.root"])
    assert calls == [(([str(tmp_path / "in.root")], "out.root", "MyTree"),
                      {"dz_max": 0.3, "cos_alpha_2d_min": 0.7})]
