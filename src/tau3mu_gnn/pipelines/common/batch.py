"""Batch layer shared by SLURM (slurm.py, PSI T3) and HTCondor (htcondor.py, lxplus).

The stages and job bodies know nothing about the batch system. The drivers describe jobs (Job) and hand them to
a Batch (make_batch): Batch.add() per phase, then Batch.finish() once. SLURM submits in add(); HTCondor writes
the job files in add() and submits one DAG over all phases in finish().
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import tau3mu_gnn
from tau3mu_gnn.pipelines.common import storage

if TYPE_CHECKING:
    from tau3mu_gnn.pipelines.analysis.config import Config
    from tau3mu_gnn.pipelines.training.config import TrainingConfig

PROXY_HINT = "voms-proxy-init --voms cms --valid 192:00"
TRAINING_MODULE = "tau3mu_gnn.pipelines.training.jobs"


class SubmitError(Exception):
    def __init__(self, message: str, submitted: Mapping[str, str] | None = None):
        super().__init__(message)
        self.submitted = dict(submitted or {})   # job name -> ID of the jobs submitted before the failure


class ProxyError(SubmitError):
    pass


class EnvError(SubmitError):
    pass


@dataclass(frozen=True)
class Job:
    name: str                      # e.g. "gnn_signal", "limit_3.9"
    resources: str                 # key of the job table (cfg.slurm.jobs / cfg.htcondor.jobs)
    args: tuple[str, ...]          # arguments of `python -m <module>`
    after: tuple[str, ...] = ()    # names of jobs this one waits for
    dependency: str = "afterok"    # afterok | afterany
    after_ids: tuple[str, ...] = ()  # IDs from an earlier Batch.add() of the same batch (afterok), e.g. the last job of phase bdt
    module: str = "tau3mu_gnn.pipelines.analysis.jobs"   # `python -m` module of the job command


@dataclass(frozen=True)
class JobEnv:
    conda_base: Path
    env_prefix: Path
    python: Path
    src_dir: Path                  # folder that contains the tau3mu_gnn package used for submitting

    @classmethod
    def from_config(cls, cfg: Config, environ: Mapping[str, str] = os.environ) -> "JobEnv":
        """The env in paths.conda_env (<base>/envs/<name> or the base env itself), else the active env.
        An env outside its conda base (lxplus: /eos/.../envs/Tau3Mu, base /eos/.../miniforge3) takes the base of
        the active conda (CONDA_EXE)."""
        if cfg.paths.conda_env is None:
            return cls.from_environment(environ)
        prefix = Path(cfg.paths.conda_env)
        base = prefix.parents[1] if prefix.parent.name == "envs" else prefix
        if not (base / "bin" / "activate").is_file() and environ.get("CONDA_EXE"):
            base = Path(environ["CONDA_EXE"]).parents[1]
        missing = [str(f) for f in (prefix / "bin" / "python", base / "bin" / "activate") if not f.is_file()]
        if missing:
            raise EnvError(f"paths.conda_env {prefix} is not a conda env: missing {', '.join(missing)}")
        return cls(conda_base=base, env_prefix=prefix, python=prefix / "bin" / "python",
                   src_dir=Path(tau3mu_gnn.__file__).resolve().parents[1])

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] = os.environ) -> "JobEnv":
        prefix, exe = environ.get("CONDA_PREFIX"), environ.get("CONDA_EXE")
        if not prefix or not exe:
            raise EnvError("no active conda env (CONDA_PREFIX / CONDA_EXE not set): "
                           "activate the conda env first (conda activate Tau3Mu) or set paths.conda_env")
        return cls(conda_base=Path(exe).parents[1], env_prefix=Path(prefix), python=Path(sys.executable),
                   src_dir=Path(tau3mu_gnn.__file__).resolve().parents[1])


def resources(cfg: "Config | TrainingConfig") -> Mapping:
    """Job table of the scheduler of cfg: name -> SlurmJob or HTCondorJob (both have mem, time, cpus)."""
    return cfg.htcondor.jobs if cfg.scheduler == "htcondor" else cfg.slurm.jobs


def time_hours(time: str) -> float:
    days, _, hms = time.rpartition("-")
    h, m, s = (int(x) for x in hms.split(":"))
    return (int(days) if days else 0) * 24 + h + m / 60 + s / 3600


def chain_hours(cfg: Config, jobs: Sequence[Job]) -> float:
    """Hours of the longest path through the `after` dependencies (the jobs are in submit order)."""
    table = resources(cfg)
    end: dict[str, float] = {}
    for j in jobs:
        end[j.name] = max((end[a] for a in j.after), default=0.0) + time_hours(table[j.resources].time)
    return max(end.values(), default=0.0)


SHORT_JOB = {"gnn_signal": "gnnS", "gnn_background": "gnnB", "summary": "sum",
             "bdt_train": "bdtT", "bdt_infer": "bdtI", "graphs_signal": "grS", "graphs_background": "grB",
             "gnn_train": "gnnT", "benchmark_summary": "bsum"}
VERSION_CODES = {"bdt": "BDT", "gnn": "GNN", "limit": "LIM", "benchmark": "BEN"}   # phase version prefixes


def job_name(run_name: str, version: str, job: str, debug: bool = False) -> str:
    """Short job first, so the 8-character NAME column of squeue shows it:
    gnn_vs3 + gnn_signal -> GNN3.gnnS.<run_name>; gnn_vs2 + gnn_train -> GNN2.gnnT.<run_name>
    (VERSION_CODES: bdt / gnn / limit / benchmark -> BDT / GNN / LIM / BEN); debug runs (debug/<stage>_vsN) get D."""
    stem, n = version.rsplit("_vs", 1)
    code = "D" if debug else VERSION_CODES.get(stem, "D")
    short = SHORT_JOB.get(job) or job.replace("benchmark_limit_", "bl", 1).replace("limit_", "lim", 1)
    return f"{code}{n}.{short}.{run_name}"


def check_dependencies(jobs: Sequence[Job]) -> None:
    names = [j.name for j in jobs]
    for i, j in enumerate(jobs):
        unknown = [a for a in j.after if a not in names[:i]]
        if unknown:
            raise ValueError(f"job {j.name} waits for {unknown}, which are not submitted before it")


def check_proxy(cfg: Config, hours: float, environ: Mapping[str, str] = os.environ) -> None:
    """dCache output needs a VOMS proxy that the jobs can read and that is valid for `hours`
    (the longest job chain of the submission, see chain_hours). EOS and local output need none."""
    if not storage.needs_proxy(cfg.output_dir):
        return
    proxy = environ.get("X509_USER_PROXY")
    if not proxy:
        raise ProxyError("X509_USER_PROXY is not set; the jobs cannot write to dCache. Set it to a file in your "
                         f"home folder (e.g. export X509_USER_PROXY=$HOME/.x509up_u$(id -u)), then: {PROXY_HINT}")
    if proxy.startswith("/tmp") and cfg.scheduler == "slurm":   # HTCondor sends the proxy with the job
        raise ProxyError(f"X509_USER_PROXY={proxy} is under /tmp, which the nodes cannot see. "
                         f"Put it in your home folder, then: {PROXY_HINT}")
    hours = math.ceil(hours)
    rc = subprocess.run(["voms-proxy-info", "--exists", "--valid", f"{hours}:00"], env={**os.environ, **environ},
                        capture_output=True).returncode
    if rc != 0:
        raise ProxyError(f"no VOMS proxy valid for {hours} h (the longest job chain of this submission). "
                         f"Run: {PROXY_HINT}")


class Batch:
    """add(): one phase (or one debug stage); returns job name -> ID. The IDs are valid as after_ids of the jobs
    of a later add() of the same batch. finish(): submits what is not submitted yet; returns
    version_dir -> {job name -> final ID} for every add().

    One subclass per batch system, in its module: slurm.SlurmBatch, htcondor.HTCondorBatch. They submit at
    different times because the batch systems link jobs differently: sbatch returns a job ID at once and later
    jobs wait for that ID, so SLURM submits in add(); DAGMan needs the whole DAG (all phases) in one submit,
    so HTCondor writes the job files in add() and submits in finish()."""

    def add(self, cfg, jobs: Sequence[Job], version_dir: str, log_folder: Path, env: JobEnv,
            dry_run: bool = False) -> dict[str, str]:
        raise NotImplementedError

    def finish(self, dry_run: bool = False) -> dict[str, dict[str, str]]:
        raise NotImplementedError


def make_batch(cfg) -> Batch:
    if cfg.scheduler == "htcondor":
        from tau3mu_gnn.pipelines.common.htcondor import HTCondorBatch
        return HTCondorBatch()
    from tau3mu_gnn.pipelines.common.slurm import SlurmBatch
    return SlurmBatch()
