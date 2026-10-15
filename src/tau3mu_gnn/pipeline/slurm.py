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
from typing import Mapping, Sequence

import tau3mu_gnn
from tau3mu_gnn.pipeline import storage
from tau3mu_gnn.pipeline.config import Config

PROXY_HINT = "voms-proxy-init --voms cms --valid 192:00"


class SubmitError(Exception):
    pass


class ProxyError(SubmitError):
    pass


class EnvError(SubmitError):
    pass


@dataclass(frozen=True)
class Job:
    name: str                      # e.g. "gnn_signal", "limit_3.9"
    resources: str                 # key of cfg.slurm.jobs
    args: tuple[str, ...]          # arguments of `python -m tau3mu_gnn.pipeline.jobs`
    after: tuple[str, ...] = ()    # names of jobs this one waits for
    dependency: str = "afterok"    # afterok | afterany


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


def render_script(cfg: Config, job: Job, version_dir: str, log_folder: Path, env: JobEnv,
                  scratch_root: str = "/scratch/$USER") -> str:
    res = cfg.slurm.jobs[job.resources]
    version = storage.join(version_dir).rsplit("/", 1)[-1]
    q = shlex.quote
    out = f"{log_folder}/{job.name}-%j.out"
    err = f"{log_folder}/{job.name}-%j.err"
    logs_dest = storage.join(version_dir, "logs")
    copy_log = (f'{q(str(env.python))} -c "import sys; from tau3mu_gnn.pipeline import storage; '
                f'storage.put(sys.argv[1], sys.argv[2])" "$f" {q(logs_dest)}/"$(basename "$f")" || true')
    args = " ".join(q(a) for a in job.args)
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={cfg.analysis_name}.{version}.{job.name}",
        f"#SBATCH --account={cfg.slurm.account}",
        f"#SBATCH --partition={res.partition}",
        f"#SBATCH --mem={res.mem}",
        f"#SBATCH --time={res.time}",
        f"#SBATCH --cpus-per-task={res.cpus}",
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
        f"{q(str(env.python))} -m tau3mu_gnn.pipeline.jobs {args}",
    ]) + "\n"


def check_proxy(cfg: Config, jobs: Sequence[Job], environ: Mapping[str, str] = os.environ) -> None:
    """dCache output needs a VOMS proxy that the nodes can read and that outlives the longest job."""
    if not storage.is_remote(cfg.output_dir):
        return
    proxy = environ.get("X509_USER_PROXY")
    if not proxy:
        raise ProxyError("X509_USER_PROXY is not set; the jobs cannot write to dCache. Set it to a file in your "
                         f"home folder (e.g. export X509_USER_PROXY=$HOME/.x509up_u$(id -u)), then: {PROXY_HINT}")
    if proxy.startswith("/tmp"):
        raise ProxyError(f"X509_USER_PROXY={proxy} is under /tmp, which the nodes cannot see. "
                         f"Put it in your home folder, then: {PROXY_HINT}")
    hours = math.ceil(max(time_hours(cfg.slurm.jobs[j.resources].time) for j in jobs))
    rc = subprocess.run(["voms-proxy-info", "--exists", "--valid", f"{hours}:00"], env={**os.environ, **environ},
                        capture_output=True).returncode
    if rc != 0:
        raise ProxyError(f"no VOMS proxy valid for {hours} h (the longest job of this submission). Run: {PROXY_HINT}")


def submit_jobs(cfg: Config, jobs: Sequence[Job], version_dir: str, log_folder: Path, env: JobEnv,
                dry_run: bool = False) -> dict[str, str]:
    """Write one script per job into log_folder and submit them in order (or print them with dry_run).
    Returns job name -> SLURM job ID ('DRYRUN-<n>' in a dry run)."""
    log_folder = Path(log_folder)
    log_folder.mkdir(parents=True, exist_ok=True)
    names = [j.name for j in jobs]
    for i, j in enumerate(jobs):
        unknown = [a for a in j.after if a not in names[:i]]
        if unknown:
            raise ValueError(f"job {j.name} waits for {unknown}, which are not submitted before it")
    ids: dict[str, str] = {}
    for n, job in enumerate(jobs, start=1):
        script = log_folder / f"{job.name}.sh"
        script.write_text(render_script(cfg, job, version_dir, log_folder, env))
        cmd = ["sbatch", "--parsable"]
        if job.after:
            cmd.append(f"--dependency={job.dependency}:" + ":".join(ids[a] for a in job.after))
        cmd.append(str(script))
        if dry_run:
            print(" ".join(cmd))
            ids[job.name] = f"DRYRUN-{n}"
            continue
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            done = " ".join(ids.values())
            hint = f" Already submitted: {ids}; cancel them with: scancel {done}" if ids else ""
            raise SubmitError(f"sbatch failed for job {job.name}: {res.stderr.strip()}.{hint}")
        ids[job.name] = res.stdout.strip().split(";")[0]
    return ids
