import json
import shutil
import subprocess
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from conftest import write_tree

SIG_WEIGHT = 0.5


def toy_stage3_file(path, n, signal, seed):
    """As stage 3 writes it: GNN_score (-1 = not scored) and the jagged fitted mass (first candidate used).
    signal: mass ~ N(1.78, 0.02); else falling exponential in 1.55-2.05. GNN_score ~ U(-1, 6)."""
    rng = np.random.default_rng(seed)
    if signal:
        m = rng.normal(1.78, 0.02, n)
    else:
        m = 1.55 + rng.exponential(0.4, n)
        m = m[m < 2.05]
    n = len(m)
    second = rng.uniform(1.5, 2.1, n)  # a second candidate that must be ignored
    data = {
        "GNN_score": rng.uniform(-1, 6, n).astype(np.float32),
        "Selected_TauTo3Mu_fitted_mass": ak.Array([[a, b] for a, b in zip(m, second)]),
        "run": np.ones(n, dtype=np.uint32),
    }
    return write_tree(path, data)


@pytest.fixture
def toy_inputs(tmp_path):
    sig = toy_stage3_file(tmp_path / "in" / "signal_with_gnn_scores.root", 4000, True, 1)
    bkg = toy_stage3_file(tmp_path / "in" / "background_with_gnn_scores.root", 40000, False, 2)
    return sig, bkg


def expected_counts(path, cut, window=None):
    import uproot
    with uproot.open(path) as f:
        t = f["Events"].arrays(["GNN_score", "Selected_TauTo3Mu_fitted_mass"])
    m = ak.to_numpy(t["Selected_TauTo3Mu_fitted_mass"][:, 0])
    keep = ak.to_numpy(t["GNN_score"]) > cut
    if window:
        keep &= np.abs(m - window[0]) > window[1]
    return keep, m


def test_make_datacard_writes_files_and_counts(toy_inputs, tmp_path):
    from tau3mu_gnn.analysis.datacard_creation_GNN_permuted import make_datacard

    sig, bkg = toy_inputs
    out = tmp_path / "card"
    res = make_datacard([sig], [bkg], 3.9, SIG_WEIGHT, out)

    for name in ("temp_mc_selected.root", "temp_data_selected.root", "mass.pdf", "datacard.root", "datacard.txt"):
        assert (out / name).is_file(), name
    assert res.datacard == out / "datacard.txt"

    keep_s, m_s = expected_counts(sig, 3.9)
    keep_b, _ = expected_counts(bkg, 3.9, window=(1.78, 0.06))
    assert res.n_signal == keep_s.sum()
    assert res.n_data == keep_b.sum()
    # RooDataSet keeps only mass in the RooRealVar range 1.6-2.0; each entry has weight SIG_WEIGHT
    in_range = keep_s & (m_s > 1.6) & (m_s < 2.0)
    assert res.signal_rate == pytest.approx(SIG_WEIGHT * in_range.sum(), rel=1e-6)


def test_datacard_text_matches_result(toy_inputs, tmp_path):
    from tau3mu_gnn.analysis.datacard_creation_GNN_permuted import make_datacard

    res = make_datacard([toy_inputs[0]], [toy_inputs[1]], 3.9, SIG_WEIGHT, tmp_path / "card")
    text = res.datacard.read_text()
    assert "observation       -1" in text                                  # blinded
    assert f"rate                                    {res.signal_rate:.4f}        {res.nbkg:.4f}" in text
    assert f"slope      param   {res.slope:.4f} {res.slope_err:.4f}" in text
    assert "shapes signal        inclusive       datacard.root w:sig" in text


def test_data_tree_is_blinded(toy_inputs, tmp_path):
    import uproot
    from tau3mu_gnn.analysis.datacard_creation_GNN_permuted import make_datacard

    out = tmp_path / "card"
    make_datacard([toy_inputs[0]], [toy_inputs[1]], 0.0, SIG_WEIGHT, out)
    with uproot.open(out / "temp_data_selected.root") as f:
        mass = f["Events"]["mass"].array(library="np")
    assert len(mass) > 0
    assert np.all(np.abs(mass - 1.78) > 0.06)


def test_old_cli_builds_old_paths(monkeypatch, tmp_path):
    from tau3mu_gnn.analysis import datacard_creation_GNN_permuted as dc

    calls = {}

    def fake(sig, bkg, cut, weight, out_dir, **kw):
        calls.update(sig=sig, bkg=bkg, cut=cut, weight=weight, out_dir=out_dir, **kw)
        (Path(out_dir) / "mass.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(dc, "make_datacard", fake)
    monkeypatch.setattr(dc, "OLD_OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output" / "m").mkdir(parents=True)

    dc.main(["--model", "m", "--channel", "W", "--cut", "3.9", "--file_suffix", "_with_scores_bdt_full_with_scores"])

    base = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data"
    assert calls["sig"] == [f"{base}/signal_W/m/selected_signal_skimmed_no_resonances_with_scores_bdt_full_with_scores.root"]
    assert calls["bkg"] == [f"{base}/background_W/m/selected_data_skimmed_no_resonances_with_scores_bdt_full_with_scores.root"]
    assert calls["cut"] == 3.9 and calls["weight"] == 0.000019 and calls["out_dir"] == "."
    assert (tmp_path / "output" / "m" / "mass_3.9_with_bdt_cut.pdf").is_file()


needs_combine = pytest.mark.skipif(
    shutil.which("combine") is None and not (Path(sys.executable).parent / "combine").is_file(),
    reason="CMS Combine not installed")


def test_combine_executable_next_to_python(monkeypatch, tmp_path):
    from tau3mu_gnn.analysis import combine as cb

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "combine").write_text("#!/bin/sh\n")
    (fake_bin / "combine").chmod(0o755)
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(cb.sys, "executable", str(fake_bin / "python"))
    assert cb.combine_executable() == fake_bin / "combine"


def test_combine_executable_missing(monkeypatch, tmp_path):
    from tau3mu_gnn.analysis import combine as cb

    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(cb.sys, "executable", str(tmp_path / "python"))
    with pytest.raises(FileNotFoundError, match="combine"):
        cb.combine_executable()


def test_run_combine_failure_raises_and_keeps_log(monkeypatch, tmp_path):
    from tau3mu_gnn.analysis import combine as cb

    fake = tmp_path / "combine"
    fake.write_text("#!/bin/sh\necho 'bad datacard' >&2\nexit 3\n")
    fake.chmod(0o755)
    monkeypatch.setattr(cb, "combine_executable", lambda: fake)
    with pytest.raises(subprocess.CalledProcessError):
        cb.run_combine(tmp_path, 0.9)
    assert "bad datacard" in (tmp_path / "combine_fitdiagnostics.log").read_text()


@needs_combine
def test_run_combine_on_toy_card(toy_inputs, tmp_path):
    from tau3mu_gnn.analysis.combine import ASYMPTOTIC_OUTPUT, run_combine
    from tau3mu_gnn.analysis.datacard_creation_GNN_permuted import make_datacard

    out = tmp_path / "card"
    make_datacard([toy_inputs[0]], [toy_inputs[1]], 3.9, SIG_WEIGHT, out)
    result = run_combine(out, 0.9)
    assert result == out / ASYMPTOTIC_OUTPUT and result.is_file()
    assert (out / "combine_asymptotic.log").read_text()
