"""Stage 3: GNN inference on the stage 2 outputs (events with BDT ISO score >= gnn.bdt_preselection_cut).

Writes <sample>_with_gnn_scores.root (all events and branches, plus GNN_score; -1 = not scored)
and raw_scores_and_tau_mass_<sample>.npz (scores and tau mass of the scored events, for the model evaluation).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from tau3mu_gnn.gnn_training.gnn_inputs import input_branch_names
from tau3mu_gnn.pipeline.branches import check_stage_inputs
from tau3mu_gnn.pipeline.config import Config
from tau3mu_gnn.utils.score_merge import check_unique_keys

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")


def check_inputs(inputs: Mapping[str, str | Path], tree_name: str, channel: str) -> None:
    """Raise BranchError if a GNN input branch is missing, or GNN_score is already there.
    Raise DuplicateEventError if two events of a file have the same (run, luminosityBlock, event)."""
    check_stage_inputs({s: inputs[s] for s in SAMPLES}, tree_name, input_branch_names(channel), "GNN",
                       forbidden=["GNN_score"])
    for sample in SAMPLES:
        check_unique_keys(inputs[sample], tree_name)


def run(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict[str, Path]:
    """Score inputs (sample -> stage 2 output file) into out_dir.

    Returns {'signal', 'background', 'signal_scores', 'background_scores'} -> output file.
    """
    check_inputs(inputs, cfg.inputs.tree_name, cfg.channel)
    import torch

    from tau3mu_gnn.gnn_training import model_loading
    from tau3mu_gnn.gnn_training.ScoreProducer import score_events

    g = cfg.gnn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {g.model_name: model_loading.load_model(g.model_dir, g.model_name, g.conv_type, g.gnn_mode,
                                                     g.graph_type, cfg.channel, device)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for sample in SAMPLES:
        outputs[sample] = out_dir / f"{sample}_with_gnn_scores.root"
        outputs[f"{sample}_scores"] = out_dir / f"raw_scores_and_tau_mass_{sample}.npz"
        log.info("stage 3 GNN: %s -> %s", inputs[sample], outputs[sample])
        in_file = str(inputs[sample])
        score_events([in_file], models, {g.model_name: {in_file: str(outputs[sample])}},
                     {g.model_name: outputs[f"{sample}_scores"]}, cfg.channel, g.unsculpting_mode,
                     g.bdt_preselection_cut, tree_name=cfg.inputs.tree_name, device=device)
    return outputs
