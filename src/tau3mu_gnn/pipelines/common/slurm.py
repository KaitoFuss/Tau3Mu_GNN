"""SLURM layer (PSI T3): one bash script per job, submitted with sbatch (step 9b).

The stages and job bodies know nothing about SLURM. The parts shared with the HTCondor layer (htcondor.py) are in
batch.py; they are imported here too, so `from ...common.slurm import Job` still works.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from tau3mu_gnn.pipelines.common import storage
from tau3mu_gnn.pipelines.common.batch import (  # noqa: F401  (re-exported)
    PROXY_HINT, SHORT_JOB, TRAINING_MODULE, VERSION_CODES, EnvError, Job, JobEnv, ProxyError, SubmitError,
    chain_hours, check_dependencies, check_proxy, time_hours,
)
from tau3mu_gnn.pipelines.common.batch import job_name as slurm_job_name

if TYPE_CHECKING:
    from tau3mu_gnn.pipelines.analysis.config import Config
    from tau3mu_gnn.pipelines.training.config import TrainingConfig


def render_script(cfg: "Config | TrainingConfig", job: Job, version_dir: str, log_folder: Path, env: JobEnv,
                  scratch_root: str = "/scratch/$USER") -> str:
    res = cfg.slurm.jobs[job.resources]
    parent, version = storage.join(version_dir).rsplit("/", 2)[-2:]
    q = shlex.quote
    out = f"{log_folder}/{job.name}-%j.out"
    err = f"{log_folder}/{job.name}-%j.err"
    logs_dest = storage.join(version_dir, "logs")
    copy_log = (f'{q(str(env.python))} -c "import sys; from tau3mu_gnn.pipelines.common import storage; '
                f'storage.put(sys.argv[1], sys.argv[2])" "$f" {q(logs_dest)}/"$(basename "$f")" || true')
    args = " ".join(q(a) for a in job.args)
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm_job_name(cfg.run_name, version, job.name, debug=parent == 'debug')}",
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


def submit_jobs(cfg: Config, jobs: Sequence[Job], version_dir: str, log_folder: Path, env: JobEnv,
                dry_run: bool = False) -> dict[str, str]:
    """Write one script per job into log_folder and submit them in order (or print them with dry_run).
    Returns job name -> SLURM job ID ('DRYRUN-<job name>' in a dry run)."""
    log_folder = Path(log_folder)
    log_folder.mkdir(parents=True, exist_ok=True)
    check_dependencies(jobs)
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
