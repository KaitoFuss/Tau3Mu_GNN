"""Stage 2: BDT inference (ISO classifier only) on the stage 1 outputs.

Writes <sample>_with_bdt_scores.root (all events and branches, plus bdt_ISO) and
tau_score_corr.pkl (scores for the model evaluation).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from tau3mu_gnn.bdt_training.bdt_inputs import input_branch_names
from tau3mu_gnn.pipelines.common.branches import check_stage_inputs
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type hints only: common code does not import the analysis package at run time
    from tau3mu_gnn.pipelines.analysis.config import Config
from tau3mu_gnn.utils.score_merge import check_unique_keys

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")
PICKLE_NAME = "tau_score_corr.pkl"


def check_inputs(inputs: Mapping[str, str | Path], tree_name: str, channel: str) -> None:
    """Raise BranchError listing every BDT branch missing in any input file, or DuplicateEventError if two
    events of a file have the same (run, luminosityBlock, event). Reads branch names and the key branches only."""
    check_stage_inputs({s: inputs[s] for s in SAMPLES}, tree_name, input_branch_names(channel), "BDT")
    for sample in SAMPLES:
        check_unique_keys(inputs[sample], tree_name)


def run(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict[str, Path]:
    """Score inputs (sample -> stage 1 output file) into out_dir with the classifier of the config.

    Returns {'signal': ..., 'background': ..., 'scores_pickle': ...}.
    """
    return score(inputs, out_dir, cfg.bdt.classifier, cfg.channel, cfg.inputs.tree_name)


def score(inputs: Mapping[str, str | Path], out_dir: str | Path, classifier: str | Path, channel: str,
          tree_name: str) -> dict[str, Path]:
    """Stage 2 with an explicit classifier (analysis: frozen pickle; training: the bdt_train output)."""
    check_inputs(inputs, tree_name, channel)
    from tau3mu_gnn.bdt_training.BDT_ScoreProducer import score_events

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {sample: out_dir / f"{sample}_with_bdt_scores.root" for sample in SAMPLES}
    outputs["scores_pickle"] = out_dir / PICKLE_NAME
    log.info("stage 2 BDT (%s): %s -> %s", classifier, {s: str(inputs[s]) for s in SAMPLES}, out_dir)
    score_events({str(inputs["signal"]): str(outputs["signal"])},
                 {str(inputs["background"]): str(outputs["background"])},
                 classifier, channel, outputs["scores_pickle"], tree_name)
    return outputs
