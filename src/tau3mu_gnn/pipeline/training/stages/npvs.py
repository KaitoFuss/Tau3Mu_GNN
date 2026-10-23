"""Training stage npvs: PV_npvs weights for the SIGNAL graphs only.

The weights reshape the signal PV_npvs distribution to the background one. Only the signal graphs
use them (graph weight = signal_event_weights(PV_npvs, weight_per_bin)); background graphs keep
weight 1, and GNN inference uses no weights. Reads all events of the stage 1 outputs of both
samples (no BDT cut, no sideband): background is the target shape. Histograms on npvs.var_bins,
weights mapped onto np.linspace(*npvs.uniform_bins) (utils/CorrectPileupDist.py).
Writes signal_<channel>_npvs_weights.npz (weight_per_bin, bins) and two plots.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from tau3mu_gnn.pipeline.branches import check_stage_inputs  # noqa: E402
from tau3mu_gnn.pipeline.training.config import TrainingConfig  # noqa: E402
from tau3mu_gnn.utils.CorrectPileupDist import npvs_weights, plot_npvs, read_npvs  # noqa: E402

log = logging.getLogger(__name__)

OUTPUTS = {"weights": "signal_{channel}_npvs_weights.npz", "plot_npvs": "signal_{channel}_npvs.png",
           "plot_weights": "signal_{channel}_npvs_weights.png"}   # name as the old graph code reads it


def run(cfg: TrainingConfig, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict[str, Path]:
    """inputs: {"signal": file, "background": file} (stage 1). Returns {key of OUTPUTS: path}."""
    tree = cfg.inputs.tree_name
    check_stage_inputs(inputs, tree, ["PV_npvs"], "npvs")
    npvs = {}
    for sample in ("signal", "background"):
        npvs[sample] = read_npvs([inputs[sample]], tree_name=tree)
        log.info("npvs: %s, %d events from %s", sample, len(npvs[sample]), inputs[sample])
    uniform_bins = np.linspace(*cfg.npvs.uniform_bins)
    weights = npvs_weights(npvs["signal"], npvs["background"], cfg.npvs.var_bins, uniform_bins)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {key: out_dir / name.format(channel=cfg.channel) for key, name in OUTPUTS.items()}
    np.savez(outputs["weights"], weight_per_bin=weights, bins=uniform_bins)
    plot_npvs(npvs["signal"], npvs["background"], weights, uniform_bins, f"{cfg.channel} channel",
              outputs["plot_npvs"], outputs["plot_weights"])
    return outputs
