"""Stage 0: resonance skim (raw NanoAOD -> *_skimmed_no_resonances.root).

Placeholder until Kaito provides the code. With resonance-skimmed inputs the
stage is skipped and stage 1 reads inputs.signal / inputs.background directly.
"""
from __future__ import annotations

import logging

from tau3mu_gnn.pipeline.config import Config

log = logging.getLogger(__name__)


class StageNotImplementedError(Exception):
    pass


def run(cfg: Config) -> None:
    if cfg.inputs.already_resonance_skimmed:
        log.info("stage 0 skipped: inputs are already resonance-skimmed")
        return None
    raise StageNotImplementedError(
        "Stage 0 (resonance skim) is not implemented yet. "
        "Use resonance-skimmed inputs and set inputs.already_resonance_skimmed: true in your config."
    )
