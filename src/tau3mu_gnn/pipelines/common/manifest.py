"""Phase manifests: what a phase wrote, from which upstream version, with which settings.

<phase>_vsN/manifest.json is written by the last job of a phase, only after all its jobs succeeded.
The first job(s) of the next phase run check_upstream on it. Generic: the phase maps and settings of each
pipeline are in analysis/manifest.py and training/manifest.py.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

from tau3mu_gnn.pipelines.common import storage
from tau3mu_gnn.pipelines.common.config import REPO_ROOT

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
JOBS_FILE = "jobs.json"
class UpstreamError(Exception):
    pass


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flat(v, key + "."))
        elif not (isinstance(v, dict) and not v):
            out[key] = v
    return out


def settings_diff(old: dict, new: dict) -> list[str]:
    """Lines '<dotted.key>: <old> -> <new>' for every leaf that differs (both sides JSON-normalized)."""
    a, b = _flat(json.loads(json.dumps(old))), _flat(json.loads(json.dumps(new)))
    show = lambda d, k: repr(d[k]) if k in d else "(not set)"  # noqa: E731
    return [f"{k}: {show(a, k)} -> {show(b, k)}" for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)
            or (k in a) != (k in b)]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(cfg, phase_dir, phase: str, upstream: str | None, outputs: dict[str, str], missing=(), *,
                   settings: dict, pipeline: str) -> dict:
    """Write <phase_dir>/manifest.json. outputs: key -> path relative to phase_dir. settings: the phase
    settings of the pipeline (analysis/manifest.py, training/manifest.py). pipeline: analysis or training
    (both have bdt and gnn phases)."""
    jobs_file = storage.join(phase_dir, JOBS_FILE)
    data = {
        "pipeline": pipeline,
        "phase": phase,
        "version": storage.join(phase_dir).rsplit("/", 1)[-1],
        "upstream": upstream,
        "outputs": dict(outputs),
        "settings": settings,
        "missing": list(missing),
        "git_commit": _git_commit(),
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "slurm_jobs": json.loads(storage.read_text(jobs_file)) if storage.exists(jobs_file) else {},
    }
    storage.write_text(storage.join(phase_dir, MANIFEST), json.dumps(data, indent=2) + "\n")
    return data


def read_manifest(phase_dir) -> dict:
    path = storage.join(phase_dir, MANIFEST)
    if not storage.exists(path):
        raise UpstreamError(f"{phase_dir} has no {MANIFEST}: the phase failed, is still running, "
                            f"or this is not a phase folder")
    try:
        return json.loads(storage.read_text(path))
    except ValueError as e:
        raise UpstreamError(f"{path} is not valid JSON: {e}") from e


def check_upstream(cfg, upstream_dir, phase: str, upstream_map: Mapping[str, str | None],
                   settings_of: Callable[[object, str], dict], pipeline: str) -> dict:
    """Run by the first job(s) of `phase`. Raise UpstreamError if the upstream manifest or one of its
    output files is missing, or the folder is of the wrong pipeline or phase. Warn if its settings differ.
    A manifest without 'pipeline' (written before it existed) passes the pipeline check."""
    expected = upstream_map[phase]
    m = read_manifest(upstream_dir)
    if m.get("pipeline", pipeline) != pipeline:
        raise UpstreamError(f"{upstream_dir} is a folder of the {m['pipeline']} pipeline; the {pipeline} "
                            f"phase {phase} needs a {expected} folder of the {pipeline} pipeline")
    if m.get("phase") != expected:
        raise UpstreamError(f"{upstream_dir} is a {m.get('phase')} folder; {phase} needs a {expected} folder")
    missing = [rel for rel in m["outputs"].values() if not storage.exists(storage.join(upstream_dir, rel))]
    if missing:
        raise UpstreamError(f"{upstream_dir}: files listed in {MANIFEST} are missing: {missing}")
    diff = settings_diff(m["settings"], settings_of(cfg, expected))
    if diff:
        log.warning("upstream %s was made with other settings than the current config:\n  %s",
                    m["version"], "\n  ".join(diff))
    return m
