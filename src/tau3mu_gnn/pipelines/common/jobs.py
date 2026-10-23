"""Helpers shared by the job bodies of both pipelines (analysis/jobs.py, training/jobs.py).

Each job works in a fresh scratch folder under $TMPDIR (the job script points TMPDIR to /scratch), then
copies its outputs into the phase folder with storage.put. Upstream files are read through
storage.local_path (dCache: the /pnfs path).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from tau3mu_gnn.pipelines.common import storage

RESONANCE_DIR, SKIM_DIR = "0_resonance_skim", "1_skim"   # stage 0 / stage 1 outputs in a phase folder


def _scratch(job: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"{job}_"))


def _name(path) -> str:
    return storage.join(path).rsplit("/", 1)[-1]


def _read(folder, rel) -> Path:
    return storage.local_path(storage.join(folder, rel))
