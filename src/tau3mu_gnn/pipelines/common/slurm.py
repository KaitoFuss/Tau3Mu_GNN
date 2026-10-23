"""SLURM layer: one bash script per job, submitted with sbatch (step 9b).

The stages and job bodies know nothing about SLURM. A later HTCondor layer can replace this module.
"""
from __future__ import annotations

import math
import os
import shlex
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
    resources: str                 # key of cfg.slurm.jobs
    args: tuple[str, ...]          # arguments of `python -m <module>`
    after: tuple[str, ...] = ()    # names of jobs this one waits for
    dependency: str = "afterok"    # afterok | afterany
    after_ids: tuple[str, ...] = ()  # IDs of jobs submitted before (afterok), e.g. the last job of phase A
    module: str = "tau3mu_gnn.pipelines.analysis.jobs"   # `python -m` module of the job command


@dataclass(frozen=True)
class JobEnv:
    conda_base: Path
    env_prefix: Path
    python: Path
    src_dir: Path                  # folder that contains the tau3mu_gnn package used for submitting

    @classmethod
    def from_config(cls, cfg: Config, environ: Mapping[str, str] = os.environ) -> "JobEnv":
        """The env in paths.conda_env (<base>/envs/<name> or the base env itself), else the active env."""
        if cfg.paths.conda_env is None:
            return cls.from_environment(environ)
        prefix = Path(cfg.paths.conda_env)
        base = prefix.parents[1] if prefix.parent.name == "envs" else prefix
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


def time_hours(time: str) -> float:
    days, _, hms = time.rpartition("-")
    h, m, s = (int(x) for x in hms.split(":"))
    return (int(days) if days else 0) * 24 + h + m / 60 + s / 3600


def chain_hours(cfg: Config, jobs: Sequence[Job]) -> float:
    """Hours of the longest path through the `after` dependencies (the jobs are in submit order)."""
    end: dict[str, float] = {}
    for j in jobs:
        end[j.name] = max((end[a] for a in j.after), default=0.0) + time_hours(cfg.slurm.jobs[j.resources].time)
    return max(end.values(), default=0.0)


SHORT_JOB = {"gnn_signal": "gnnS", "gnn_background": "gnnB", "summary": "sum",
             "bdt_train": "bdtT", "bdt_infer": "bdtI", "graphs_signal": "grS", "graphs_background": "grB",
             "gnn_train": "gnnT", "benchmark_summary": "bsum"}
VERSION_CODES = {"bdt": "BDT", "gnn": "GNN", "benchmark": "BEN"}   # training pipeline version prefixes


TRAINING_MODULE = "tau3mu_gnn.pipelines.training.jobs"


def slurm_job_name(run_name: str, version: str, job: str, codes: Mapping[str, str] = {}) -> str:
    """Short job first, so the 8-character NAME column of squeue shows it:
    phase_b_vs3 + gnn_signal -> B3.gnnS.<run_name>; gnn_vs2 + gnn_train -> GNN2.gnnT.<run_name>
    (codes=VERSION_CODES, training jobs only: bdt / gnn / benchmark -> BDT / GNN / BEN); debug runs (<stage>_vsN) get D."""
    stem, n = version.rsplit("_vs", 1)
    code = stem[-1].upper() if stem.startswith("phase_") else codes.get(stem, "D")
    short = SHORT_JOB.get(job) or job.replace("benchmark_limit_", "bl", 1).replace("limit_", "lim", 1)
    return f"{code}{n}.{short}.{run_name}"


def render_script(cfg: "Config | TrainingConfig", job: Job, version_dir: str, log_folder: Path, env: JobEnv,
                  scratch_root: str = "/scratch/$USER") -> str:
    res = cfg.slurm.jobs[job.resources]
    version = storage.join(version_dir).rsplit("/", 1)[-1]
    q = shlex.quote
    out = f"{log_folder}/{job.name}-%j.out"
    err = f"{log_folder}/{job.name}-%j.err"
    logs_dest = storage.join(version_dir, "logs")
    copy_log = (f'{q(str(env.python))} -c "import sys; from tau3mu_gnn.pipelines.common import storage; '
                f'storage.put(sys.argv[1], sys.argv[2])" "$f" {q(logs_dest)}/"$(basename "$f")" || true')
    args = " ".join(q(a) for a in job.args)
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm_job_name(cfg.run_name, version, job.name, VERSION_CODES if job.module == TRAINING_MODULE else {})}",
        f"#SBATCH --account={res.account or cfg.slurm.account}",
        f"#SBATCH --partition={res.partition}",
        f"#SBATCH --mem={res.mem}",
        f"#SBATCH --time={res.time}",
        f"#SBATCH --cpus-per-task={res.cpus}",
        *([f"#SBATCH --gres={res.gres}"] if res.gres else []),
        f"#SBATCH --output={out}",
        f"#SBATCH --error={err}",
        "set -eo pipefail",
        f"source {q(str(env.conda_base / 'bin' / 'activate'))} {q(str(env.env_prefix))}",
        f'export PYTHONPATH={q(str(env.src_dir))}${{PYTHONPATH:+:$PYTHONPATH}}',
        f'JOB_SCRATCH="{scratch_root}/$SLURM_JOB_ID"',
        'mkdir -p "$JOB_SCRATCH"',
        'export TMPDIR="$JOB_SCRATCH"',
        "cleanup() {",
        f'    for f in {q(str(log_folder))}/{q(job.name)}-"$SLURM_JOB_ID".out '
        f'{q(str(log_folder))}/{q(job.name)}-"$SLURM_JOB_ID".err; do',
        f'        [ -f "$f" ] && {copy_log}',
        "    done",
        '    rm -rf "$JOB_SCRATCH"',
        "}",
        "trap cleanup EXIT",
        'cd "$JOB_SCRATCH"',
        f"{q(str(env.python))} -m {job.module} {args}",
    ]) + "\n"


def check_proxy(cfg: Config, hours: float, environ: Mapping[str, str] = os.environ) -> None:
    """dCache output needs a VOMS proxy that the nodes can read and that is valid for `hours`
    (the longest job chain of the submission, see chain_hours)."""
    if not storage.is_remote(cfg.output_dir):
        return
    proxy = environ.get("X509_USER_PROXY")
    if not proxy:
        raise ProxyError("X509_USER_PROXY is not set; the jobs cannot write to dCache. Set it to a file in your "
                         f"home folder (e.g. export X509_USER_PROXY=$HOME/.x509up_u$(id -u)), then: {PROXY_HINT}")
    if proxy.startswith("/tmp"):
        raise ProxyError(f"X509_USER_PROXY={proxy} is under /tmp, which the nodes cannot see. "
                         f"Put it in your home folder, then: {PROXY_HINT}")
    hours = math.ceil(hours)
    rc = subprocess.run(["voms-proxy-info", "--exists", "--valid", f"{hours}:00"], env={**os.environ, **environ},
                        capture_output=True).returncode
    if rc != 0:
        raise ProxyError(f"no VOMS proxy valid for {hours} h (the longest job chain of this submission). "
                         f"Run: {PROXY_HINT}")


def submit_jobs(cfg: Config, jobs: Sequence[Job], version_dir: str, log_folder: Path, env: JobEnv,
                dry_run: bool = False) -> dict[str, str]:
    """Write one script per job into log_folder and submit them in order (or print them with dry_run).
    Returns job name -> SLURM job ID ('DRYRUN-<job name>' in a dry run)."""
    log_folder = Path(log_folder)
    log_folder.mkdir(parents=True, exist_ok=True)
    names = [j.name for j in jobs]
    for i, j in enumerate(jobs):
        unknown = [a for a in j.after if a not in names[:i]]
        if unknown:
            raise ValueError(f"job {j.name} waits for {unknown}, which are not submitted before it")
    ids: dict[str, str] = {}
    for job in jobs:
        script = log_folder / f"{job.name}.sh"
        script.write_text(render_script(cfg, job, version_dir, log_folder, env))
        cmd = ["sbatch", "--parsable"]
        deps = ([f"{job.dependency}:" + ":".join(ids[a] for a in job.after)] if job.after else []) + \
               (["afterok:" + ":".join(job.after_ids)] if job.after_ids else [])
        if deps:
            # without this a job whose dependency can never be met stays PENDING forever
            cmd += [f"--dependency={','.join(deps)}", "--kill-on-invalid-dep=yes"]
        cmd.append(str(script))
        if dry_run:
            print(" ".join(cmd))
            ids[job.name] = f"DRYRUN-{job.name}"
            continue
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            done = " ".join(ids.values())
            hint = f" Already submitted: {ids}; cancel them with: scancel {done}" if ids else ""
            raise SubmitError(f"sbatch failed for job {job.name}: {res.stderr.strip()}.{hint}", ids)
        ids[job.name] = res.stdout.strip().split(";")[0]
    return ids
