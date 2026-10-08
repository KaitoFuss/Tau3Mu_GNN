"""Stage 1: skim (dZ + cos alpha mask per tau candidate, Slimmed_* -> Selected_*).

Reads inputs.signal / inputs.background from the config and applies the branch map.
When stage 0 is implemented, this stage must read stage 0's output instead and not map again.
"""
from __future__ import annotations

import logging
from pathlib import Path

from tau3mu_gnn.pipeline.branches import check_input_branches
from tau3mu_gnn.pipeline.config import Config
from tau3mu_gnn.utils.skimmer import skim

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")


def run(cfg: Config, out_dir: str | Path) -> dict[str, Path]:
    """Skim each sample into out_dir/<sample>.root. Checks all inputs before any work."""
    inputs = cfg.inputs
    files = {sample: check_input_branches(getattr(inputs, sample), inputs.tree_name, cfg.branch_map, cfg.channel)
             for sample in SAMPLES}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for sample in SAMPLES:
        outputs[sample] = out_dir / f"{sample}.root"
        log.info("stage 1 skim: %s (%d file(s)) -> %s", sample, len(files[sample]), outputs[sample])
        skim(files[sample], str(outputs[sample]), inputs.tree_name, cfg.branch_map,
             dz_max=cfg.skim.dz_max, cos_alpha_2d_min=cfg.skim.cos_alpha_2d_min)
    return outputs
