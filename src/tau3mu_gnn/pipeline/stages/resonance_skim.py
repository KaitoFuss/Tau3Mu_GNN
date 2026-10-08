"""Stage 0: branch map, then resonance skim (raw NanoAOD -> *_skimmed_no_resonances.root).

Order: check inputs -> rename (branch map) -> resonance skim -> write, in one RDataFrame pass.
The resonance skim therefore works on analysis names. It is a placeholder until Kaito
provides the code. Resonance-skimmed inputs with an empty branch map are passed on, no copy.
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


def _resonance_skim(df):
    raise StageNotImplementedError(
        "Stage 0 (resonance skim) is not implemented yet. "
        "Use resonance-skimmed inputs and set inputs.already_resonance_skimmed: true in your config."
    )


def run(cfg: Config, out_dir: str | Path) -> dict[str, list[str]]:
    """Return the input files of stage 1 per sample. Checks all inputs before any work."""
    inputs = cfg.inputs
    files = {sample: check_input_branches(getattr(inputs, sample), inputs.tree_name, cfg.branch_map, cfg.channel)
             for sample in SAMPLES}
    identity_map = all(user == analysis for analysis, user in cfg.branch_map.items())
    if inputs.already_resonance_skimmed and identity_map:
        log.info("stage 0: inputs already resonance-skimmed, empty branch map: inputs passed on, no copy")
        return files

    import ROOT

    ROOT.EnableImplicitMT()
    frames = {}
    for sample in SAMPLES:
        df, columns = rename_columns(ROOT.RDataFrame(inputs.tree_name, files[sample]), cfg.branch_map)
        if not inputs.already_resonance_skimmed:
            df = _resonance_skim(df)
        frames[sample] = df, columns

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for sample, (df, columns) in frames.items():
        output = out_dir / f"{sample}.root"
        log.info("stage 0: %s (%d file(s)) -> %s", sample, len(files[sample]), output)
        df.Snapshot(inputs.tree_name, str(output), columns)
        outputs[sample] = [str(output)]
    return outputs
