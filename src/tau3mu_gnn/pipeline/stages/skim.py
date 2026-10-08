"""Stage 1: skim (dZ + cos alpha mask per tau candidate, Slimmed_* -> Selected_*).

Reads the files returned by stage 0. They already have analysis branch names.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

from tau3mu_gnn.pipeline.config import Config
from tau3mu_gnn.utils.skimmer import skim

log = logging.getLogger(__name__)


def run(cfg: Config, inputs: Mapping[str, Sequence[str]], out_dir: str | Path) -> dict[str, Path]:
    """Skim each sample of inputs (sample -> files, from stage 0) into out_dir/<sample>.root."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for sample, files in inputs.items():
        outputs[sample] = out_dir / f"{sample}.root"
        log.info("stage 1 skim: %s (%d file(s)) -> %s", sample, len(files), outputs[sample])
        skim(files, str(outputs[sample]), cfg.inputs.tree_name,
             dz_max=cfg.skim.dz_max, cos_alpha_2d_min=cfg.skim.cos_alpha_2d_min)
    return outputs
