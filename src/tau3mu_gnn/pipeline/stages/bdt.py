"""Stage 2: BDT inference (ISO classifier only) on the stage 1 outputs.

Writes <sample>_with_bdt_scores.root (all events and branches, plus bdt_ISO) and
tau_score_corr.pkl (scores for the model evaluation).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import uproot

from tau3mu_gnn.bdt_training.bdt_inputs import input_branch_names
from tau3mu_gnn.pipeline.branches import BranchError
from tau3mu_gnn.pipeline.config import Config

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")
PICKLE_NAME = "tau_score_corr.pkl"


def check_inputs(inputs: Mapping[str, str | Path], tree_name: str, channel: str) -> None:
    """Raise BranchError listing every BDT branch missing in any input file. Reads branch names only."""
    required = sorted(input_branch_names(channel))
    problems = []
    for sample in SAMPLES:
        path = str(inputs[sample])
        try:
            with uproot.open(path) as f:
                keys = set(f[tree_name].keys())
        except FileNotFoundError:
            problems.append(f"{path}: file not found")
            continue
        except KeyError:
            problems.append(f"{path}: no tree '{tree_name}'")
            continue
        missing = [name for name in required if name not in keys]
        if missing:
            problems.append(f"{path}: missing BDT input branches: {', '.join(missing)}")
    if problems:
        raise BranchError(problems)


def run(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict[str, Path]:
    """Score inputs (sample -> stage 1 output file) into out_dir.

    Returns {'signal': ..., 'background': ..., 'scores_pickle': ...}.
    """
    check_inputs(inputs, cfg.inputs.tree_name, cfg.channel)
    from tau3mu_gnn.bdt_training.BDT_ScoreProducer import score_events

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {sample: out_dir / f"{sample}_with_bdt_scores.root" for sample in SAMPLES}
    outputs["scores_pickle"] = out_dir / PICKLE_NAME
    log.info("stage 2 BDT: %s -> %s", {s: str(inputs[s]) for s in SAMPLES}, out_dir)
    score_events({str(inputs["signal"]): str(outputs["signal"])},
                 {str(inputs["background"]): str(outputs["background"])},
                 cfg.bdt.classifier, cfg.channel, outputs["scores_pickle"], cfg.inputs.tree_name)
    return outputs
