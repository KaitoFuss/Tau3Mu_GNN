"""Training stage bdt_infer: stage 2 (pipeline/stages/bdt.py) with the classifier that bdt_train wrote.

Same outputs as stage 2: <sample>_with_bdt_scores.root (all events and branches, plus bdt_ISO) and
tau_score_corr.pkl. The graph stage reads the ROOT files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from tau3mu_gnn.pipeline.stages import bdt
from tau3mu_gnn.pipeline.training.config import TrainingConfig


def run(cfg: TrainingConfig, inputs: Mapping[str, str | Path], classifier: str | Path,
        out_dir: str | Path) -> dict[str, Path]:
    """inputs: {"signal": file, "background": file} (stage 1); classifier: the bdt_train pickle."""
    return bdt.score(inputs, out_dir, classifier, cfg.channel, cfg.inputs.tree_name)
