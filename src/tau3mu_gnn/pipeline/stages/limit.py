"""Stage 5: datacard + CMS Combine + limit read-out for one GNN score cut, on the stage 3 outputs.

Writes <out_dir>/cut_<cut>/: the datacard files (datacard.txt/.root, mass.pdf, temp trees), the Combine
outputs and logs, and limit.json ({"cut", "limit", "limitErr"}: median expected limit at CL = limit.cl).
Step 8b runs one call per cut of limit.gnn_score_cut and writes the table.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

from tau3mu_gnn.analysis.combine import combine_executable, run_combine
from tau3mu_gnn.analysis.readout_limit import read_limit
from tau3mu_gnn.pipeline.branches import check_stage_inputs
from tau3mu_gnn.pipeline.config import Config

log = logging.getLogger(__name__)

INPUTS = ("signal", "background")
REQUIRED_BRANCHES = ("GNN_score", "Selected_TauTo3Mu_fitted_mass")


def cut_dir_name(cut: float) -> str:
    return f"cut_{cut}"


def check_inputs(inputs: Mapping[str, str | Path], tree_name: str) -> None:
    """Raise BranchError if a stage 3 output is missing or has no GNN_score / fitted mass."""
    check_stage_inputs({s: inputs[s] for s in INPUTS}, tree_name, REQUIRED_BRANCHES, "limit")


def run(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path, cut: float) -> dict:
    """Datacard, Combine and limit for GNN_score > cut. inputs: sample -> stage 3 ROOT file.

    Returns {"cut", "limit", "limitErr", "dir", "datacard", "limit_json"}.
    """
    lim = cfg.limit
    if cut not in lim.gnn_score_cut:
        raise ValueError(f"cut {cut} is not in limit.gnn_score_cut {list(lim.gnn_score_cut)}")
    check_inputs(inputs, cfg.inputs.tree_name)
    combine_executable()  # fail before the datacard work if Combine is missing
    from tau3mu_gnn.analysis.datacard_creation_GNN_permuted import make_datacard

    card_dir = Path(out_dir) / cut_dir_name(cut)
    log.info("stage 5 limit: cut %s, %s -> %s", cut, {s: str(inputs[s]) for s in INPUTS}, card_dir)
    card = make_datacard([inputs["signal"]], [inputs["background"]], cut, lim.signal_weight, card_dir,
                         tau_mass=lim.tau_mass, mass_window=lim.mass_window, fit_range=lim.fit_range,
                         tree_name=cfg.inputs.tree_name)
    limits = read_limit(run_combine(card_dir, lim.cl))
    if not limits:
        raise RuntimeError(f"Combine wrote no median expected limit; see {card_dir / 'combine_asymptotic.log'} "
                           f"and {card_dir / 'combine_fitdiagnostics.log'}")
    result = {"cut": cut, **limits}
    limit_json = card_dir / "limit.json"
    limit_json.write_text(json.dumps(result, indent=2) + "\n")
    return {**result, "dir": card_dir, "datacard": card.datacard, "limit_json": limit_json}
