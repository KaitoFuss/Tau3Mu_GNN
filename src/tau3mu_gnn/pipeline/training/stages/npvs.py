"""Training stage npvs: PV_npvs weights (signal -> background shape) for the signal graphs.

Reads all events of the stage 1 outputs (no BDT cut, no sideband). Histograms on npvs.var_bins,
weights mapped onto np.linspace(*npvs.uniform_bins) (utils/CorrectPileupDist.py).
Writes npvs_weights.npz (weight_per_bin, bins) and two plots.
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

OUTPUTS = {"weights": "npvs_weights.npz", "plot_npvs": "npvs.png", "plot_weights": "npvs_weights.png"}


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
    outputs = {key: out_dir / name for key, name in OUTPUTS.items()}
    np.savez(outputs["weights"], weight_per_bin=weights, bins=uniform_bins)
    plot_npvs(npvs["signal"], npvs["background"], weights, uniform_bins, f"{cfg.channel} channel",
              outputs["plot_npvs"], outputs["plot_weights"])
    return outputs
