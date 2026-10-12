"""Versions of phases and debug runs: <output_dir>/<phase>_vsN, <output_dir>/debug/<stage>_vsN.

Every phase run makes a new version; nothing is overwritten. A version is complete when its manifest exists.
"""
from __future__ import annotations

import logging
import re

import yaml

from tau3mu_gnn.pipeline import storage
from tau3mu_gnn.pipeline.config import Config
from tau3mu_gnn.pipeline.manifest import MANIFEST, UPSTREAM, UpstreamError, config_dict

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


def create_version(cfg: Config, prefix: str, parent=None) -> str:
    """Make the next <prefix>_vsN folder in parent (default: the run folder) with config_used.yaml."""
    parent = parent if parent is not None else cfg.output_dir
    path = storage.join(parent, next_version(parent, prefix))
    storage.makedirs(path)
    storage.write_text(storage.join(path, CONFIG_USED), yaml.safe_dump(config_dict(cfg), sort_keys=False))
    log.info("new version %s", path)
    return path


def pick_upstream(cfg: Config, phase: str, from_: str | None = None) -> str:
    """Full path of the upstream version for `phase`: from_ if given (must be complete), else the highest
    complete version. Raise UpstreamError if there is none."""
    up = UPSTREAM[phase]
    if up is None:
        raise ValueError(f"{phase} has no upstream phase")
    base = cfg.output_dir
    if from_ is not None:
        if _number(from_, up) is None:
            raise UpstreamError(f"--from {from_!r}: {phase} needs a {up}_vsN version")
        if not storage.exists(storage.join(base, from_)):
            raise UpstreamError(f"--from {from_!r}: not found in {base}")
        if not is_complete(base, from_):
            raise UpstreamError(f"--from {from_!r}: no manifest (failed or still running)")
        return storage.join(base, from_)
    names = versions(base, up)
    complete = [n for n in names if is_complete(base, n)]
    if not complete:
        raise UpstreamError(f"no complete {up} version in {base}; run {up.replace('_', '-')} first")
    newer = names[names.index(complete[-1]) + 1:]
    if newer:
        log.warning("%s have no manifest (failed or still running); using %s", ", ".join(newer), complete[-1])
    return storage.join(base, complete[-1])
