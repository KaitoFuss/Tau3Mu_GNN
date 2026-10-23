"""Phase and single-stage submission (step 9b): pick the upstream version, make the next version folder,
write and submit the job scripts. The job bodies are in jobs.py; the SLURM details in slurm.py."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tau3mu_gnn.pipeline import runs, storage
from tau3mu_gnn.pipeline.config import Config
from tau3mu_gnn.pipeline.manifest import JOBS_FILE, read_manifest
from tau3mu_gnn.pipeline.slurm import Job, JobEnv, SubmitError, chain_hours, check_proxy, submit_jobs

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")
STAGES = ("skim", "bdt", "gnn", "eval", "limit", "summary")
PHASES = ("phase_a", "phase_b", "phase_c")
LAST_JOB = {"phase_a": "bdt", "phase_b": "eval"}   # writes the manifest; the next phase of `all` waits for it


@dataclass(frozen=True)
class Submission:
    version_dir: str
    log_folder: Path
    job_ids: dict[str, str]
    upstream: str | None


def _base(cfg: Config, version_dir: str) -> tuple[str, ...]:
    return ("--config", str(cfg.user_file), "--phase-dir", version_dir)


def _phase_jobs(cfg: Config, phase: str, version_dir: str, up: str | None, after_ids: tuple[str, ...] = ()) \
        -> list[Job]:
    """after_ids: SLURM IDs the first jobs of the phase wait for (`all`: the last job of the phase before)."""
    b = _base(cfg, version_dir)
    if phase == "phase_a":
        return [Job("skim", "skim", ("skim", *b), after_ids=after_ids),
                Job("bdt", "bdt", ("bdt", *b), after=("skim",))]
    if phase == "phase_b":
        gnn = [Job(f"gnn_{s}", f"gnn_{s}", ("gnn", *b, "--upstream", up, "--sample", s), after_ids=after_ids)
               for s in SAMPLES]
        return gnn + [Job("eval", "eval", ("eval", *b, "--upstream", up), after=tuple(j.name for j in gnn))]
    limits = [Job(f"limit_{c}", "limit", ("limit", *b, "--upstream", up, "--cut", repr(c)), after_ids=after_ids)
              for c in cfg.limit.gnn_score_cut]
    return limits + [Job("summary", "summary", ("summary", *b, "--upstream", up),
                         after=tuple(j.name for j in limits), dependency="afterany", after_ids=after_ids)]


def _submit(cfg, jobs_for, prefix, parent, log_parts, upstream, dry_run, env, proxy=True) -> Submission:
    """jobs_for(version_dir) -> job list. Shared tail of submit_phase / submit_stage / submit_all (steps 3-7)."""
    if proxy:
        check_proxy(cfg, chain_hours(cfg, jobs_for("<version>")))
    name = runs.next_version(parent, prefix)
    version_dir = storage.join(parent, name) if dry_run else runs.create_version(cfg, prefix, parent=parent)
    name = version_dir.rsplit("/", 1)[-1]
    log_folder = Path(cfg.paths.log_dir, cfg.run_name, *log_parts, name)
    ids = submit_jobs(cfg, jobs_for(version_dir), version_dir, log_folder, env, dry_run=dry_run)
    if not dry_run:
        storage.write_text(storage.join(version_dir, JOBS_FILE), json.dumps(ids, indent=2) + "\n")
    log.info("%s: %s, logs %s, jobs %s", name, version_dir, log_folder, ids)
    return Submission(version_dir, log_folder, ids, upstream)


def submit_phase(cfg: Config, phase: str, from_: str | None = None, dry_run: bool = False,
                 env: JobEnv | None = None) -> Submission:
    env = env if env is not None else JobEnv.from_config(cfg)
    up = runs.pick_upstream(cfg, phase, from_) if phase != "phase_a" else None
    if phase == "phase_a" and from_ is not None:
        raise ValueError("phase-a has no upstream; --from is not allowed")
    return _submit(cfg, lambda v: _phase_jobs(cfg, phase, v, up), phase, cfg.output_dir, (), up, dry_run, env)


def submit_stage(cfg: Config, stage: str, from_: str | None = None, cut: float | None = None,
                 sample: str | None = None, dry_run: bool = False, env: JobEnv | None = None) -> Submission:
    env = env if env is not None else JobEnv.from_config(cfg)
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; one of {list(STAGES)}")
    up = src = None
    if stage == "bdt":
        src = runs.pick_version(cfg, "phase_a", from_)
    elif stage == "gnn":
        if sample not in SAMPLES:
            raise ValueError(f"stage gnn needs --sample signal or background, got {sample!r}")
        up = runs.pick_version(cfg, "phase_a", from_)
    elif stage == "eval":
        src = runs.pick_version(cfg, "phase_b", from_)
        up = storage.join(cfg.output_dir, read_manifest(src)["upstream"])
    elif stage == "limit":
        if cut not in cfg.limit.gnn_score_cut:
            raise ValueError(f"cut {cut} is not in limit.gnn_score_cut {list(cfg.limit.gnn_score_cut)}")
        up = runs.pick_version(cfg, "phase_b", from_)
    elif stage == "summary":
        src = runs.pick_version(cfg, "phase_c", from_, complete=False)
    elif from_ is not None:
        raise ValueError("stage skim reads the config inputs; --from is not allowed")

    def jobs_for(version_dir):
        args = [stage, *_base(cfg, version_dir)]
        if up is not None:
            args += ["--upstream", up]
        if sample is not None and stage == "gnn":
            args += ["--sample", sample]
        if stage == "limit":
            args += ["--cut", repr(cut)]
        if src is not None:
            args += ["--source", src]
        if stage in ("bdt", "eval", "summary"):
            args += ["--no-manifest"]
        name = {"gnn": f"gnn_{sample}", "limit": f"limit_{cut}"}.get(stage, stage)
        resources = {"gnn": f"gnn_{sample}", "limit": "limit"}.get(stage, stage)
        return [Job(name, resources, tuple(args))]

    debug = storage.join(cfg.output_dir, "debug")
    return _submit(cfg, jobs_for, stage, debug, ("debug",), up or src, dry_run, env)


def submit_all(cfg: Config, dry_run: bool = False, env: JobEnv | None = None) -> list[Submission]:
    """Submit phase A, B and C at once. B uses the new A version, C the new B version (not the newest complete
    one: they have no manifest yet). The first jobs of B and C wait (afterok) for the last job of the phase
    before; their upstream check runs when they start. If a job fails, SLURM cancels the jobs after it."""
    env = env if env is not None else JobEnv.from_config(cfg)
    check_proxy(cfg, sum(chain_hours(cfg, _phase_jobs(cfg, p, "<version>", "<upstream>")) for p in PHASES))
    subs: list[Submission] = []
    up, after = None, ()
    for phase in PHASES:
        try:
            s = _submit(cfg, lambda v, phase=phase, up=up, after=after: _phase_jobs(cfg, phase, v, up, after),
                        phase, cfg.output_dir, (), up, dry_run, env, proxy=False)
        except SubmitError as e:
            if not subs:
                raise
            done = [i for prev in subs for i in prev.job_ids.values()] + list(e.submitted.values())
            raise SubmitError(f"{e} Earlier phases are already submitted ({', '.join(x.version_dir for x in subs)});"
                              f" to cancel everything: scancel {' '.join(done)}") from e
        subs.append(s)
        up, after = s.version_dir, (s.job_ids[LAST_JOB[phase]],) if phase in LAST_JOB else ()
    return subs
