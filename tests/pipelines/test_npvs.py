import numpy as np
import pytest

from tau3mu_gnn.utils import CorrectPileupDist as cpd

UNIFORM = np.linspace(0, 100, 101)


def _old_weights(sig, bkg):
    """Verbatim copy of the weight part of the old create_weight_array (reference)."""
    var_bins = [0, 10, 15, 18, 20] + list(np.linspace(21, 60, 40, dtype=int)) + [65, 100]
    hs, _ = np.histogram(sig, bins=var_bins)
    hb, _ = np.histogram(bkg, bins=var_bins)
    hs, hb = hs / np.sum(hs), hb / np.sum(hb)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.true_divide(hb, hs)
        w[~np.isfinite(w)] = 0
    centers = 0.5 * (UNIFORM[:-1] + UNIFORM[1:])
    out = np.zeros(100)
    vb = np.array(var_bins)
    for i in range(len(vb) - 1):
        out[(centers >= vb[i]) & (centers < vb[i + 1])] = w[i]
    return out


def test_matches_old_code():
    rng = np.random.default_rng(1)
    sig, bkg = rng.poisson(30, 50_000), rng.poisson(25, 80_000)
    w = cpd.npvs_weights(sig, bkg, cpd.OLD_VAR_BINS, UNIFORM)
    np.testing.assert_array_equal(w, _old_weights(sig, bkg))
    assert w.shape == (100,)


def test_known_values():
    # one var bin [0, 50) and one [50, 100): signal 3:1, background 1:1
    w = cpd.npvs_weights(np.array([10, 10, 10, 60]), np.array([10, 60]), [0, 50, 100], UNIFORM)
    assert w[0] == pytest.approx(0.5 / 0.75) and w[99] == pytest.approx(0.5 / 0.25)


def test_empty_signal_bin_gives_zero():
    w = cpd.npvs_weights(np.array([10, 10]), np.array([10, 60]), [0, 50, 100], UNIFORM)
    assert w[99] == 0 and np.all(np.isfinite(w))


def test_values_outside_bins_ignored():
    w = cpd.npvs_weights(np.array([10, 150]), np.array([10, 200]), [0, 50, 100], UNIFORM)
    assert w[0] == 1.0


@pytest.mark.parametrize("sig, bkg, sample", [
    (np.array([], dtype=int), np.array([10]), "signal"),
    (np.array([10]), np.array([], dtype=int), "background"),
    (np.array([150]), np.array([10]), "signal"),     # nothing inside the bins
])
def test_empty_sample_raises(sig, bkg, sample):
    with pytest.raises(ValueError, match=sample):
        cpd.npvs_weights(sig, bkg, [0, 50, 100], UNIFORM)


def test_run_writes_outputs(tmp_path, load_training, training_user):
    from conftest import write_tree
    from tau3mu_gnn.pipelines.training.stages import npvs as stage
    rng = np.random.default_rng(2)
    sig = write_tree(tmp_path / "sig.root", {"PV_npvs": rng.poisson(30, 2000).astype(np.uint8)})
    bkg = write_tree(tmp_path / "bkg.root", {"PV_npvs": rng.poisson(25, 3000).astype(np.uint8)})
    cfg = load_training(training_user)
    out = stage.run(cfg, {"signal": sig, "background": bkg}, tmp_path / "out")
    assert set(out) == {"weights", "plot_npvs", "plot_weights"} and all(p.is_file() for p in out.values())
    d = np.load(out["weights"])
    assert d["weight_per_bin"].shape == (100,) and d["bins"].shape == (101,)
    np.testing.assert_array_equal(d["weight_per_bin"], cpd.npvs_weights(
        cpd.read_npvs([sig]), cpd.read_npvs([bkg]), cfg.npvs.var_bins, np.linspace(*cfg.npvs.uniform_bins)))


def test_missing_branch_raises(tmp_path, load_training, training_user):
    from conftest import write_tree
    from tau3mu_gnn.pipelines.common.branches import BranchError
    from tau3mu_gnn.pipelines.training.stages import npvs as stage
    sig = write_tree(tmp_path / "sig.root", {"PV_npvs": np.array([10], dtype=np.uint8)})
    bkg = write_tree(tmp_path / "bkg.root", {"other": np.array([1.0])})
    with pytest.raises(BranchError, match="bkg.root.*PV_npvs"):
        stage.run(load_training(training_user), {"signal": sig, "background": bkg}, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_signal_event_weights_clip_out_of_range():
    w = np.arange(100, dtype=float)                      # weight of uniform bin i = i
    npvs = np.array([0, 50, 99, 100, 255], dtype=np.uint8)
    np.testing.assert_array_equal(cpd.signal_event_weights(npvs, w), [0, 50, 99, 99, 99])


def test_stage_output_names_say_signal(tmp_path, load_training, training_user):
    from conftest import write_tree
    from tau3mu_gnn.pipelines.training.stages import npvs as stage
    sig = write_tree(tmp_path / "sig.root", {"PV_npvs": np.array([10, 30, 30], dtype=np.uint8)})
    bkg = write_tree(tmp_path / "bkg.root", {"PV_npvs": np.array([10, 10, 30], dtype=np.uint8)})
    out = stage.run(load_training(training_user), {"signal": sig, "background": bkg}, tmp_path / "out")
    assert {p.name for p in out.values()} == {
        "signal_W_npvs_weights.npz", "signal_W_npvs.png", "signal_W_npvs_weights.png"}
