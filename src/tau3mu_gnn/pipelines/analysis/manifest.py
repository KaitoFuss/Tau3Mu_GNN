"""Versions and manifests of the analysis phases: bdt, gnn, limit.

Thin wrappers over common/runs.py and common/manifest.py with the analysis phase maps and settings.
"""
from __future__ import annotations

from pathlib import Path

from tau3mu_gnn.pipelines.analysis.config import Config
from tau3mu_gnn.pipelines.common import manifest as _manifest
from tau3mu_gnn.pipelines.common import runs as _runs

PIPELINE = "analysis"
PHASES = ("bdt", "gnn", "limit")
UPSTREAM = {"bdt": None, "gnn": "bdt", "limit": "gnn"}
PHASE_SECTIONS = {
    "bdt": ("channel", "inputs", "branch_map", "resonance_skim", "skim", "models", "bdt"),
    "gnn": ("channel", "inputs", "branch_map", "resonance_skim", "skim", "models", "bdt", "gnn"),
    "limit": ("channel", "inputs", "branch_map", "resonance_skim", "skim", "models", "bdt", "gnn", "limit"),
}


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
        "models": cfg.models,   # frozen or the training gnn_vsN the models come from
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
        "scheduler": cfg.scheduler,
        "htcondor": {"jobs": {name: {"mem": j.mem, "time": j.time, "cpus": j.cpus, "gpus": j.gpus,
                                     "requirements": j.requirements, "disk": j.disk}
                              for name, j in cfg.htcondor.jobs.items()}},
    }


def phase_settings(cfg: Config, phase: str) -> dict:
    d = config_dict(cfg)
    return {key: d[key] for key in PHASE_SECTIONS[phase]}


def phase_settings(cfg: Config, phase: str) -> dict:
    d = config_dict(cfg)
    return {key: d[key] for key in PHASE_SECTIONS[phase]}


def create_version(cfg: Config, prefix: str, parent=None) -> str:
    return _runs.create_version(cfg, prefix, parent=parent, settings=config_dict(cfg))


def write_manifest(cfg: Config, phase_dir, phase: str, upstream: str | None, outputs: dict[str, str],
                   missing=()) -> dict:
    return _manifest.write_manifest(cfg, phase_dir, phase, upstream, outputs, missing,
                                    settings=phase_settings(cfg, phase), pipeline=PIPELINE)


def check_upstream(cfg: Config, upstream_dir, phase: str) -> dict:
    return _manifest.check_upstream(cfg, upstream_dir, phase, UPSTREAM, phase_settings, PIPELINE)


def pick_upstream(cfg: Config, phase: str, from_: str | None = None) -> str:
    return _runs.pick_upstream(cfg, phase, from_, UPSTREAM)
