"""Stage 4: model evaluation plots from the stage 3 npz files and the stage 2 pickle.

Writes tau_hist_fpr.png, tau_hist_fpr_normalized.png (background tau mass for GNN score cuts at fixed
FPR), tau_hist.png (fixed cuts on sigmoid(score)), roc.png and sic.png (GNN vs ISO BDT, background
from the mass sidebands).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np

from tau3mu_gnn.pipelines.analysis.config import Config

log = logging.getLogger(__name__)

INPUTS = ("signal_scores", "background_scores", "scores_pickle")
PLOTS = ("tau_hist_fpr", "tau_hist_fpr_normalized", "tau_hist", "roc", "sic")
FPR_TARGETS = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
NPZ_KEYS = ("scores", "tau_mass")
PICKLE_KEYS = ("ISO_score_sig", "ISO_score_bkg", "tau_mass")


def check_inputs(inputs: Mapping[str, str | Path]) -> None:
    """Raise FileNotFoundError (all missing files) or ValueError (missing key) before any plot."""
    from tau3mu_gnn.model_eval.ModelEval import load_pickle

    missing = [str(inputs[k]) for k in INPUTS if not Path(inputs[k]).is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 4 inputs missing: {missing}")
    for key in ("signal_scores", "background_scores"):
        with np.load(inputs[key]) as npz:
            _check_keys(inputs[key], npz.files, NPZ_KEYS)
    _check_keys(inputs["scores_pickle"], load_pickle(inputs["scores_pickle"]).keys(), PICKLE_KEYS)


def _check_keys(path, found, expected):
    absent = [k for k in expected if k not in found]
    if absent:
        raise ValueError(f"{path} has no {absent} (has {sorted(found)})")


def run(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict[str, Path]:
    """Plot inputs ({signal_scores, background_scores}: stage 3 npz, scores_pickle: stage 2) into out_dir.

    Returns plot name (PLOTS) -> PNG file.
    """
    check_inputs(inputs)
    from tau3mu_gnn.model_eval import ModelEval as me

    g = cfg.gnn
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: out_dir / f"{name}.png" for name in PLOTS}
    sig, bkg = inputs["signal_scores"], inputs["background_scores"]
    log.info("stage 4 eval: %s -> %s", {k: str(inputs[k]) for k in INPUTS}, out_dir)

    for name, normalize in (("tau_hist_fpr", False), ("tau_hist_fpr_normalized", True)):
        me.plot_tau_hist(*me.load_gnn_npz(bkg, sig), outputs[name], "GNN", g.gnn_mode, g.model_name,
                         cfg.channel, normalize)
    me.plot_tau_hist(*me.load_gnn_npz(bkg), outputs["tau_hist"], "GNN", g.gnn_mode, g.model_name,
                     cfg.channel, False)
    curves = me.plot_roc_full([(f"GNN {g.model_name}", sig, bkg)], inputs["scores_pickle"], outputs["roc"],
                              FPR_TARGETS, f"ROC: GNN vs BDT ISO, {cfg.channel} channel",
                              iso_cut=g.bdt_preselection_cut)
    me.plot_sic(*curves, outputs["sic"], f"SIC: GNN vs BDT ISO, {cfg.channel} channel")
    return outputs
