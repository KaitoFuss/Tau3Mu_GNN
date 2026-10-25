"""Training stage benchmark: BDT-only limit, a cut scan on bdt_ISO (the number the GNN must beat).

Same datacard, Combine run and read-out as stage 5 (stages/limit.run_cut with score_branch="bdt_ISO"; the
old analysis/datacard_creation_ISO.py was the GNN datacard script with bdt_ISO). Inputs: the bdt_infer
outputs. One run() per cut of benchmark.bdt_score_cut into <out_dir>/cut_<cut>/, then summarize() writes
<out_dir>/limits.{json,txt,png} (best cut = lowest limit). run_scan() does both in one process.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Mapping

from tau3mu_gnn.analysis.combine import combine_executable
from tau3mu_gnn.pipelines.common.stages import limit
from tau3mu_gnn.pipelines.training.config import TrainingConfig

log = logging.getLogger(__name__)

SCORE_BRANCH = "bdt_ISO"


def run(cfg: TrainingConfig, inputs: Mapping[str, str | Path], out_dir: str | Path, cut: float) -> dict:
    """Datacard, Combine and limit for bdt_ISO > cut. inputs: sample -> bdt_infer ROOT file."""
    if cut not in cfg.benchmark.bdt_score_cut:
        raise ValueError(f"cut {cut} is not in benchmark.bdt_score_cut {list(cfg.benchmark.bdt_score_cut)}")
    lim = cfg.limit
    return limit.run_cut(inputs, out_dir, cut, score_branch=SCORE_BRANCH, signal_weight=lim.signal_weight,
                         tau_mass=lim.tau_mass, mass_window=lim.mass_window, fit_range=lim.fit_range,
                         cl=lim.cl, tree_name=cfg.inputs.tree_name)


def summarize(cfg: TrainingConfig, out_dir: str | Path, write_dir: str | Path | None = None) -> dict:
    """Table of the benchmark cut scan (cuts of benchmark.bdt_score_cut)."""
    return limit.summarize_cuts(cfg.benchmark.bdt_score_cut, cfg.limit.cl, out_dir, write_dir)


def run_scan(cfg: TrainingConfig, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict:
    """All benchmark cuts in this process, then summarize. A cut whose Combine fails is a missing row."""
    limit.check_inputs(inputs, cfg.inputs.tree_name, SCORE_BRANCH)
    combine_executable()
    for cut in cfg.benchmark.bdt_score_cut:
        try:
            run(cfg, inputs, out_dir, cut)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            log.error("benchmark: cut %s failed: %s", cut, e)
    return summarize(cfg, out_dir)
