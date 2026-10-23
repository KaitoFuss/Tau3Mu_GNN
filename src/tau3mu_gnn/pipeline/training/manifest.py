"""Versions and manifests of the training phases: bdt, gnn, benchmark.

Thin wrappers over pipeline.runs / pipeline.manifest with the training phase maps and settings.
"""
from __future__ import annotations


from tau3mu_gnn.pipeline import manifest as _manifest
from tau3mu_gnn.pipeline import runs as _runs
from tau3mu_gnn.pipeline.training.config import TrainingConfig

PHASES = ("bdt", "gnn", "benchmark")
UPSTREAM = {"bdt": None, "gnn": "bdt", "benchmark": "bdt"}
_BDT = ("channel", "inputs", "branch_map", "resonance_skim", "skim", "sideband_half_width", "bdt_training")
PHASE_SECTIONS = {
    "bdt": _BDT,
    "gnn": _BDT + ("npvs", "graphs", "gnn_training"),
    "benchmark": _BDT + ("benchmark", "limit"),
}
HIDDEN = "***"


def config_dict(cfg: TrainingConfig) -> dict:
    """The loaded training config as plain data. Repo files relative to the repo root; comet.api_key hidden."""
    inp, rs, b, g, n, t = cfg.inputs, cfg.resonance_skim, cfg.bdt_training, cfg.graphs, cfg.npvs, cfg.gnn_training
    lim = cfg.limit
    return {
        "training_name": cfg.training_name,
        "channel": cfg.channel,
        "paths": {"output_base": cfg.paths.output_base, "log_dir": str(cfg.paths.log_dir),
                  "conda_env": None if cfg.paths.conda_env is None else str(cfg.paths.conda_env)},
        "inputs": {"signal": [str(x) for x in inp.signal], "background": [str(x) for x in inp.background],
                   "tree_name": inp.tree_name, "already_resonance_skimmed": inp.already_resonance_skimmed},
        "branch_map": dict(cfg.branch_map),
        "resonance_skim": {"vtx_chi2_max": rs.vtx_chi2_max, "mass_range": list(rs.mass_range),
                           "veto_masses": list(rs.veto_masses), "veto_half_width": rs.veto_half_width,
                           "pf_dr_max": rs.pf_dr_max},
        "skim": {"dz_max": cfg.skim.dz_max, "cos_alpha_2d_min": cfg.skim.cos_alpha_2d_min},
        "sideband_half_width": cfg.sideband_half_width,
        "bdt_training": {"max_events_per_class": b.max_events_per_class, "max_depth": b.max_depth,
                         "learning_rate": b.learning_rate, "n_estimators": b.n_estimators,
                         "early_stopping_rounds": b.early_stopping_rounds, "subsample": b.subsample,
                         "colsample_bytree": b.colsample_bytree, "gamma": b.gamma, "reg_alpha": b.reg_alpha,
                         "reg_lambda": b.reg_lambda, "min_child_weight_fraction": b.min_child_weight_fraction,
                         "kfold": b.kfold, "test_size": b.test_size, "split_seed": b.split_seed,
                         "sample_seed": b.sample_seed},
        "npvs": {"var_bins": list(n.var_bins), "uniform_bins": list(n.uniform_bins)},
        "graphs": {"bdt_preselection_cut": g.bdt_preselection_cut, "max_events": g.max_events,
                   "connection": g.connection, "k": g.k, "cone_size": g.cone_size,
                   "use_displacement": g.use_displacement, "unsculpting_mode": g.unsculpting_mode,
                   "shuffle_seed": g.shuffle_seed},
        "gnn_training": {"conv_type": t.conv_type, "gnn_mode": t.gnn_mode, "arch": t.arch,
                         "learning_rate": t.learning_rate, "batch_size": t.batch_size, "disco": t.disco,
                         "epochs": t.epochs, "early_stopping_patience": t.early_stopping_patience,
                         "lr_scheduler_patience": t.lr_scheduler_patience,
                         "class_fraction": list(t.class_fraction),
                         "windows": {k: list(v) for k, v in t.windows.items()}, "seed": t.seed},
        "comet": {"workspace": cfg.comet.workspace, "api_key": HIDDEN},
        "benchmark": {"bdt_score_cut": list(cfg.benchmark.bdt_score_cut)},
        "limit": {"signal_weight": lim.signal_weight, "tau_mass": lim.tau_mass, "mass_window": lim.mass_window,
                  "fit_range": list(lim.fit_range), "cl": lim.cl},
        "slurm": {"account": cfg.slurm.account,
                  "jobs": {name: {"partition": j.partition, "mem": j.mem, "time": j.time, "cpus": j.cpus,
                                  "account": j.account, "gres": j.gres}
                           for name, j in cfg.slurm.jobs.items()}},
    }


def phase_settings(cfg: TrainingConfig, phase: str) -> dict:
    d = config_dict(cfg)
    return {key: d[key] for key in PHASE_SECTIONS[phase]}


def create_version(cfg: TrainingConfig, phase: str) -> str:
    return _runs.create_version(cfg, phase, settings=config_dict(cfg))


def write_manifest(cfg: TrainingConfig, phase_dir, phase: str, upstream: str | None, outputs: dict[str, str],
                   missing=()) -> dict:
    return _manifest.write_manifest(cfg, phase_dir, phase, upstream, outputs, missing,
                                    settings=phase_settings(cfg, phase))


def check_upstream(cfg: TrainingConfig, upstream_dir, phase: str) -> dict:
    return _manifest.check_upstream(cfg, upstream_dir, phase, UPSTREAM, phase_settings)


def pick_upstream(cfg: TrainingConfig, phase: str, from_: str | None = None) -> str:
    return _runs.pick_upstream(cfg, phase, from_, UPSTREAM)
