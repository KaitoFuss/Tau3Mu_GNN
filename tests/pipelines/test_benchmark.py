import numpy as np
import pytest
import uproot

from conftest import write_tree
from test_limit import SIG_WEIGHT, needs_combine, toy_stage3_file  # noqa: F401


def _bdt_file(path, n, signal, seed):
    """toy_stage3_file with bdt_ISO ~ U(0, 1) (10 % at -1 = no candidate) instead of GNN_score."""
    src = toy_stage3_file(path.with_name("src_" + path.name), n, signal, seed)
    with uproot.open(src) as f:
        arr = f["Events"].arrays(library="ak")
    rng = np.random.default_rng(seed + 10)
    score = rng.uniform(0, 1, len(arr)).astype(np.float32)
    score[rng.uniform(size=len(arr)) < 0.1] = -1
    data = {k: arr[k] for k in arr.fields if k != "GNN_score"}
    data["bdt_ISO"] = score
    return write_tree(path, data)


@pytest.fixture
def bench(load_training, training_user, tmp_path):
    training_user["benchmark"]["bdt_score_cut"] = [0.5, 0.8]
    training_user["limit"]["signal_weight"] = SIG_WEIGHT
    cfg = load_training(training_user)
    inputs = {"signal": _bdt_file(tmp_path / "in" / "signal_with_bdt_scores.root", 4000, True, 1),
              "background": _bdt_file(tmp_path / "in" / "background_with_bdt_scores.root", 40000, False, 2)}
    return cfg, inputs


def test_benchmark_rejects_unknown_cut(bench, tmp_path):
    from tau3mu_gnn.pipelines.training.stages import benchmark
    cfg, inputs = bench
    with pytest.raises(ValueError, match="benchmark.bdt_score_cut"):
        benchmark.run(cfg, inputs, tmp_path / "out", 0.7)


def test_benchmark_missing_bdt_iso(bench, tmp_path):
    from tau3mu_gnn.pipelines.common.branches import BranchError
    from tau3mu_gnn.pipelines.training.stages import benchmark
    cfg, inputs = bench
    bad = toy_stage3_file(tmp_path / "gnn_only.root", 100, False, 3)
    with pytest.raises(BranchError, match="gnn_only.root"):
        benchmark.run(cfg, {**inputs, "background": bad}, tmp_path / "out", 0.5)
    assert not (tmp_path / "out" / "cut_0.5" / "datacard.txt").exists()


@needs_combine
def test_benchmark_scan_end_to_end(bench, tmp_path):
    import json
    from tau3mu_gnn.pipelines.training.stages import benchmark
    cfg, inputs = bench
    s = benchmark.run_scan(cfg, inputs, tmp_path / "out")
    data = json.loads((tmp_path / "out" / "limits.json").read_text())
    assert [r["cut"] for r in data["rows"]] == [0.5, 0.8] and s["best_cut"] in (0.5, 0.8)
    assert np.isfinite(s["best_limit"])
    with uproot.open(tmp_path / "out" / "cut_0.5" / "temp_mc_selected.root") as f:
        n_sel = f["Events"].num_entries
    with uproot.open(inputs["signal"]) as f:
        expected = int((f["Events"]["bdt_ISO"].array(library="np") > 0.5).sum())
    assert n_sel == expected
