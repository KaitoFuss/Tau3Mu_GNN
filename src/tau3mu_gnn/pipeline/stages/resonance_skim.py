"""Stage 0: resonance skim (raw NanoAOD -> *_skimmed_no_resonances.root) and branch map.

The resonance skim itself is a placeholder until Kaito provides the code.
This stage checks the inputs and applies the branch map, so later stages only
see analysis names. With an empty branch map the inputs are passed on, no copy.
"""
from __future__ import annotations

import logging
from pathlib import Path

from tau3mu_gnn.pipeline.branches import check_input_branches, rename_columns
from tau3mu_gnn.pipeline.config import Config

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")


class StageNotImplementedError(Exception):
    pass


def run(cfg: Config, out_dir: str | Path) -> dict[str, list[str]]:
    """Return the input files of stage 1 per sample. Checks all inputs before any work."""
    inputs = cfg.inputs
    if not inputs.already_resonance_skimmed:
        raise StageNotImplementedError(
            "Stage 0 (resonance skim) is not implemented yet. "
            "Use resonance-skimmed inputs and set inputs.already_resonance_skimmed: true in your config."
        )
    files = {sample: check_input_branches(getattr(inputs, sample), inputs.tree_name, cfg.branch_map, cfg.channel)
             for sample in SAMPLES}
    if all(user == analysis for analysis, user in cfg.branch_map.items()):
        log.info("stage 0: inputs already resonance-skimmed, empty branch map: inputs passed on, no copy")
        return files

    import ROOT

    ROOT.EnableImplicitMT()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renamed = {}
    for sample in SAMPLES:
        output = out_dir / f"{sample}.root"
        log.info("stage 0: branch map %s (%d file(s)) -> %s", sample, len(files[sample]), output)
        df, columns = rename_columns(ROOT.RDataFrame(inputs.tree_name, files[sample]), cfg.branch_map)
        df.Snapshot(inputs.tree_name, str(output), columns)
        renamed[sample] = [str(output)]
    return renamed
