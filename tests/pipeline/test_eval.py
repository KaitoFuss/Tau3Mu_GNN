import pickle

import numpy as np
import pandas as pd
import pytest

from tau3mu_gnn.model_eval import ModelEval as me

RNG = np.random.default_rng(1)


def write_npz(path, scores, tau_mass):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scores=scores, tau_mass=tau_mass)
    return path


def toy_npz(dir_, suffix=""):
    """GNN logits: signal ~ N(3, 1), background ~ N(0, 2) with flat tau mass; one NaN in each."""
    sig = np.append(RNG.normal(3, 1, 2000), np.nan)
    bkg = np.append(RNG.normal(0, 2, 20000), np.nan)
    s = write_npz(dir_ / f"raw_scores_and_tau_mass_signal{suffix}.npz", sig, np.full(sig.size, 1.78))
    b = write_npz(dir_ / f"raw_scores_and_tau_mass_background{suffix}.npz", bkg, RNG.uniform(1.4, 2.5, bkg.size))
    return s, b


def toy_pickle(path, extra_gnn_keys=False):
    """As stage 2 writes it: pandas Series. extra_gnn_keys: as the old (pre step 5) pickle."""
    data = {"ISO_score_sig": pd.Series(RNG.uniform(0.5, 1.0, 3000)),
            "ISO_score_bkg": pd.Series(RNG.uniform(0.0, 1.0, 30000)),
            "tau_mass": pd.Series(RNG.uniform(1.4, 2.5, 30000))}
    if extra_gnn_keys:
        data["GNN_score_sig"] = pd.Series(RNG.uniform(0, 1, 3000))
        data["GNN_score_bkg"] = pd.Series(RNG.uniform(0, 1, 30000))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def test_load_gnn_npz_drops_nan_only_with_signal(tmp_path):
    s, b = toy_npz(tmp_path)
    scores, mass, sig = me.load_gnn_npz(b, s)
    assert scores.size == mass.size == 20000 and sig.size == 2000
    scores, mass, sig = me.load_gnn_npz(b)
    assert scores.size == mass.size == 20001 and sig is None


@pytest.mark.parametrize("plot_mode, view, bdt_cut, name", [
    ("tau_hist_fpr", "normalized", True, "final_layer_tau_hist_fpr_normalized_with_bdt_cut.png"),
    ("tau_hist_fpr", "non_normalized", False, "final_layer_tau_hist_fpr.png"),
    ("tau_hist", "non_normalized", True, "final_layer_tau_hist_with_bdt_cut.png"),
])
def test_old_cli_tau_hist_names(tmp_path, plot_mode, view, bdt_cut, name):
    toy_npz(tmp_path, "_with_bdt_cut" if bdt_cut else "")
    me.tau_score_hist(str(tmp_path) + "/", "final_layer", "t", "W", plot_mode, view, bdt_cut)
    assert (tmp_path / name).is_file()


FPR_TARGETS = [5e-2, 1e-2, 5e-3, 1e-3]


@pytest.mark.parametrize("extra_gnn_keys", [False, True])
def test_roc_and_sic(tmp_path, extra_gnn_keys):
    s, b = toy_npz(tmp_path)
    pkl = toy_pickle(tmp_path / "bdt" / "tau_score_corr.pkl", extra_gnn_keys)
    curves = me.plot_roc_full([("GNN toy", s, b)], pkl, tmp_path / "roc.png", FPR_TARGETS, "t")
    assert curves[3] == ["GNN toy", "BDT ISO"]
    me.plot_sic(*curves, tmp_path / "sic.png", "t")
    assert (tmp_path / "roc.png").is_file() and (tmp_path / "sic.png").is_file()


@pytest.mark.parametrize("cut", [0.5, 0.9])
def test_iso_cut(tmp_path, cut):
    s, b = toy_npz(tmp_path)
    pkl = toy_pickle(tmp_path / "bdt" / "tau_score_corr.pkl")
    _, _, threshs, _ = me.plot_roc_full([("GNN toy", s, b)], pkl, tmp_path / "roc.png", FPR_TARGETS, "t",
                                        iso_cut=cut)
    iso = threshs[-1]
    assert iso[np.isfinite(iso)].min() > cut
