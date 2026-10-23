"""Versions of phases and debug runs: <output_dir>/<phase>_vsN, <output_dir>/debug/<stage>_vsN.

Every phase run makes a new version; nothing is overwritten. A version is complete when its manifest exists.
"""
from __future__ import annotations

import logging
import re
from typing import Mapping

import yaml

from tau3mu_gnn.pipelines.common import storage
from tau3mu_gnn.pipelines.common.manifest import MANIFEST, UpstreamError

log = logging.getLogger(__name__)

CONFIG_USED = "config_used.yaml"


def _number(name: str, prefix: str) -> int | None:
    m = re.fullmatch(rf"{re.escape(prefix)}_vs([1-9]\d*)", name)
    return int(m.group(1)) if m else None


def versions(parent, prefix: str) -> list[str]:
    found = [(n, name) for name in storage.listdir(parent) if (n := _number(name, prefix)) is not None]
    return [name for _, name in sorted(found)]


def next_version(parent, prefix: str) -> str:
    existing = versions(parent, prefix)
    return f"{prefix}_vs{_number(existing[-1], prefix) + 1 if existing else 1}"


def is_complete(parent, name: str) -> bool:
    return storage.exists(storage.join(parent, name, MANIFEST))


def create_version(cfg, prefix: str, parent=None, *, settings: dict) -> str:
    """Make the next <prefix>_vsN folder in parent (default: the run folder) with config_used.yaml
    (settings: the config snapshot of the pipeline, analysis/manifest.py or training/manifest.py)."""
    parent = parent if parent is not None else cfg.output_dir
    path = storage.join(parent, next_version(parent, prefix))
    storage.makedirs(path)
    storage.write_text(storage.join(path, CONFIG_USED), yaml.safe_dump(settings, sort_keys=False))
    log.info("new version %s", path)
    return path


def pick_version(cfg, phase: str, from_: str | None = None, complete: bool = True) -> str:
    """Full path of a version of `phase`: from_ if given, else the highest one. With complete=True only
    versions with a manifest count (failed or running ones are skipped with a warning)."""
    base = cfg.output_dir
    if from_ is not None:
        if _number(from_, phase) is None:
            raise UpstreamError(f"--from {from_!r}: needs a {phase}_vsN version")
        if not storage.exists(storage.join(base, from_)):
            raise UpstreamError(f"--from {from_!r}: not found in {base}")
        if complete and not is_complete(base, from_):
            raise UpstreamError(f"--from {from_!r}: no manifest (failed or still running)")
        return storage.join(base, from_)
    names = versions(base, phase)
    pool = [n for n in names if is_complete(base, n)] if complete else names
    if not pool:
        kind = "complete " if complete else ""
        raise UpstreamError(f"no {kind}{phase} version in {base}; run {phase.replace('_', '-')} first")
    newer = names[names.index(pool[-1]) + 1:]
    if newer:
        log.warning("%s have no manifest (failed or still running); using %s", ", ".join(newer), pool[-1])
    return storage.join(base, pool[-1])


def pick_upstream(cfg, phase: str, from_: str | None, upstream_map: Mapping[str, str | None]) -> str:
    """Full path of the upstream version for `phase` (complete only). Raise UpstreamError if there is none."""
    up = upstream_map[phase]
    if up is None:
        raise ValueError(f"{phase} has no upstream phase")
    return pick_version(cfg, up, from_)
