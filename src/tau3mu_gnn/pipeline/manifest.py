"""Phase manifests: what a phase wrote, from which upstream version, with which settings.

<phase>_vsN/manifest.json is written by the last job of a phase, only after all its jobs succeeded.
The first job(s) of the next phase run check_upstream on it.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

from tau3mu_gnn.pipeline import storage
from tau3mu_gnn.pipeline.config import REPO_ROOT, Config

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
JOBS_FILE = "jobs.json"
PHASES = ("phase_a", "phase_b", "phase_c")
UPSTREAM = {"phase_a": None, "phase_b": "phase_a", "phase_c": "phase_b"}
PHASE_SECTIONS = {
    "phase_a": ("channel", "inputs", "branch_map", "resonance_skim", "skim", "bdt"),
    "phase_b": ("channel", "inputs", "branch_map", "resonance_skim", "skim", "bdt", "gnn"),
    "phase_c": ("channel", "inputs", "branch_map", "resonance_skim", "skim", "bdt", "gnn", "limit"),
}


class UpstreamError(Exception):
    pass


def config_dict(cfg: Config) -> dict:
    """The loaded config (user file + defaults) as plain data. Repo files relative to the repo root."""
    repo = cfg.defaults_file.parent.parent

    def rel(p):
        try:
            return Path(p).relative_to(repo).as_posix()
        except ValueError:
            return str(p)

    g, lim, inp, rs = cfg.gnn, cfg.limit, cfg.inputs, cfg.resonance_skim
    return {
        "analysis_name": cfg.analysis_name,
        "channel": cfg.channel,
        "paths": {"output_base": cfg.paths.output_base, "log_dir": cfg.paths.log_dir,
                  "conda_env": cfg.paths.conda_env},
        "inputs": {"signal": list(inp.signal), "background": list(inp.background), "tree_name": inp.tree_name,
                   "already_resonance_skimmed": inp.already_resonance_skimmed},
        "branch_map": dict(cfg.branch_map),
        "resonance_skim": {"vtx_chi2_max": rs.vtx_chi2_max, "mass_range": list(rs.mass_range),
                           "veto_masses": list(rs.veto_masses), "veto_half_width": rs.veto_half_width,
                           "pf_dr_max": rs.pf_dr_max},
        "skim": {"dz_max": cfg.skim.dz_max, "cos_alpha_2d_min": cfg.skim.cos_alpha_2d_min},
        "bdt": {"classifier": rel(cfg.bdt.classifier)},
        "gnn": {"model_dir": rel(g.model_dir), "model_name": g.model_name, "conv_type": g.conv_type,
                "gnn_mode": g.gnn_mode, "unsculpting_mode": g.unsculpting_mode, "graph_type": g.graph_type,
                "bdt_preselection_cut": g.bdt_preselection_cut},
        "limit": {"signal_weight": lim.signal_weight, "gnn_score_cut": list(lim.gnn_score_cut),
                  "tau_mass": lim.tau_mass, "mass_window": lim.mass_window, "fit_range": list(lim.fit_range),
                  "cl": lim.cl},
        "slurm": {"account": cfg.slurm.account,
                  "jobs": {name: {"partition": j.partition, "mem": j.mem, "time": j.time, "cpus": j.cpus}
                           for name, j in cfg.slurm.jobs.items()}},
    }


def phase_settings(cfg: Config, phase: str) -> dict:
    d = config_dict(cfg)
    return {key: d[key] for key in PHASE_SECTIONS[phase]}


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


def write_manifest(cfg: Config, phase_dir, phase: str, upstream: str | None, outputs: dict[str, str],
                   missing=(), settings: dict | None = None) -> dict:
    """Write <phase_dir>/manifest.json. outputs: key -> path relative to phase_dir.
    settings default: phase_settings(cfg, phase) (the analysis sections)."""
    jobs_file = storage.join(phase_dir, JOBS_FILE)
    data = {
        "phase": phase,
        "version": storage.join(phase_dir).rsplit("/", 1)[-1],
        "upstream": upstream,
        "outputs": dict(outputs),
        "settings": phase_settings(cfg, phase) if settings is None else settings,
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


def check_upstream(cfg: Config, upstream_dir, phase: str, upstream_map: Mapping[str, str | None] = UPSTREAM,
                   settings_of: Callable[[Config, str], dict] = phase_settings) -> dict:
    """Run by the first job(s) of `phase`. Raise UpstreamError if the upstream manifest or one of its
    output files is missing, or the folder is of the wrong phase. Warn if its settings differ."""
    expected = upstream_map[phase]
    m = read_manifest(upstream_dir)
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
