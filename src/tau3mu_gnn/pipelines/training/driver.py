"""Training phase submission (step 12g): pick the bdt upstream, make the next version folder, write and submit
the job scripts. Same flow as the analysis driver (analysis/driver.py); the job bodies are in training/jobs.py.

Phases: bdt (skim -> bdt_train -> bdt_infer), gnn (npvs -> graphs_signal; graphs_background; both -> gnn_train),
benchmark (benchmark_limit_<cut> per cut -> benchmark_summary, afterany). gnn and benchmark read a bdt version.
"""
from __future__ import annotations

from tau3mu_gnn.pipelines.common.driver import Submission, _submit
from tau3mu_gnn.pipelines.common.slurm import TRAINING_MODULE, Job, JobEnv, SubmitError, chain_hours, check_proxy
from tau3mu_gnn.pipelines.training.config import TrainingConfig
from tau3mu_gnn.pipelines.training.manifest import PHASES, PIPELINE, create_version, pick_upstream

MODULE = TRAINING_MODULE
LAST_JOB = {"bdt": "bdt_infer", "gnn": "gnn_train", "benchmark": "benchmark_summary"}   # writes the manifest


def _job(name, resources, args, **kw) -> Job:
    return Job(name, resources, tuple(args), module=MODULE, **kw)


def _phase_jobs(cfg: TrainingConfig, phase: str, version_dir: str, up: str | None,
                after_ids: tuple[str, ...] = ()) -> list[Job]:
    """after_ids: SLURM IDs the first jobs of the phase wait for (`all`: bdt_infer)."""
    b = ("--config", str(cfg.user_file), "--phase-dir", version_dir)
    if phase == "bdt":
        return [_job("skim", "skim", ("skim", *b), after_ids=after_ids),
                _job("bdt_train", "bdt_train", ("bdt_train", *b), after=("skim",)),
                _job("bdt_infer", "bdt_infer", ("bdt_infer", *b), after=("bdt_train",))]
    u = ("--upstream", up)
    if phase == "gnn":
        # graphs_background does not wait for npvs: the npvs weights are for signal only
        return [_job("npvs", "npvs", ("npvs", *b, *u), after_ids=after_ids),
                _job("graphs_signal", "graphs_signal", ("graphs", *b, *u, "--sample", "signal"), after=("npvs",)),
                _job("graphs_background", "graphs_background", ("graphs", *b, *u, "--sample", "background"),
                     after_ids=after_ids),
                _job("gnn_train", "gnn_train", ("gnn_train", *b, *u), after=("graphs_signal", "graphs_background"))]
    limits = [_job(f"benchmark_limit_{c}", "benchmark_limit", ("benchmark_limit", *b, *u, "--cut", repr(c)),
                   after_ids=after_ids) for c in cfg.benchmark.bdt_score_cut]
    return limits + [_job("benchmark_summary", "benchmark_summary", ("benchmark_summary", *b, *u),
                          after=tuple(j.name for j in limits), dependency="afterany", after_ids=after_ids)]


def _submit_phase(cfg, phase, up, dry_run, env, proxy=True, after_ids=()) -> Submission:
    return _submit(cfg, lambda v: _phase_jobs(cfg, phase, v, up, after_ids), phase, cfg.output_dir, (), up,
                   dry_run, env, proxy=proxy, create=lambda: create_version(cfg, phase), pipeline=PIPELINE)


def submit_phase(cfg: TrainingConfig, phase: str, from_: str | None = None, dry_run: bool = False,
                 env: JobEnv | None = None) -> Submission:
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; one of {list(PHASES)}")
    if phase == "bdt" and from_ is not None:
        raise ValueError("bdt has no upstream; --from is not allowed")
    env = env if env is not None else JobEnv.from_config(cfg)
    up = pick_upstream(cfg, phase, from_) if phase != "bdt" else None
    return _submit_phase(cfg, phase, up, dry_run, env)


def submit_all(cfg: TrainingConfig, dry_run: bool = False, env: JobEnv | None = None) -> list[Submission]:
    """Submit bdt, then gnn and benchmark on the new bdt version (it has no manifest yet). The first jobs of
    gnn and benchmark wait (afterok) for bdt_infer; their upstream check runs when they start."""
    env = env if env is not None else JobEnv.from_config(cfg)
    hours = {p: chain_hours(cfg, _phase_jobs(cfg, p, "<version>", "<upstream>")) for p in PHASES}
    check_proxy(cfg, hours["bdt"] + max(hours["gnn"], hours["benchmark"]))
    bdt = _submit_phase(cfg, "bdt", None, dry_run, env, proxy=False)
    subs = [bdt]
    for phase in ("gnn", "benchmark"):
        try:
            s = _submit_phase(cfg, phase, bdt.version_dir, dry_run, env, proxy=False,
                              after_ids=(bdt.job_ids[LAST_JOB["bdt"]],))
        except SubmitError as e:
            done = [i for prev in subs for i in prev.job_ids.values()] + list(e.submitted.values())
            raise SubmitError(f"{e} Earlier phases are already submitted ({', '.join(x.version_dir for x in subs)});"
                              f" to cancel everything: scancel {' '.join(done)}") from e
        subs.append(s)
    return subs
