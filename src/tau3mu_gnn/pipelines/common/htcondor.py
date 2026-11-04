"""HTCondor layer (lxplus): one bash script + one submit file per job, all jobs of one tau3mu-* call in one DAG.
Shared parts (Job, JobEnv, errors, proxy check) are in batch.py.

DAGMan has no afterany: a job that an afterany job waits for gets a POST script that always succeeds, so its
children run also if it failed (the summary jobs read whatever limits exist).
HTCondor writes the .out / .err files into the log folder only at the end of a job; the script also writes the
output of the job command into the job scratch folder and copies it next to the results (<version>/logs/).
"""
from __future__ import annotations

import logging
import re
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from tau3mu_gnn.pipelines.common import storage
from tau3mu_gnn.pipelines.common.batch import Batch, Job, JobEnv, SubmitError, check_dependencies, time_hours

if TYPE_CHECKING:
    from tau3mu_gnn.pipelines.analysis.config import Config
    from tau3mu_gnn.pipelines.training.config import TrainingConfig

log = logging.getLogger(__name__)

DAG_FILE = "jobs.dag"
AFS_MIN_FREE_MB = 200          # warn below this free AFS quota in the log folder
_MEM_UNIT_MB = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
_CLUSTER_RE = re.compile(r"submitted to cluster (\d+)")


def memory_mb(mem: str) -> int:
    """64G -> 65536. No unit: MB (as in SLURM)."""
    unit = mem[-1] if mem[-1].isalpha() else "M"
    number = int(mem[:-1] if mem[-1].isalpha() else mem)
    return max(1, round(number * _MEM_UNIT_MB[unit]))


def node_name(version_dir: str, job: str) -> str:
    """DAG node name, unique over all phases of one DAG: limit_vs2 + limit_3.9 -> limit_vs2-limit_3_9."""
    version = storage.join(version_dir).rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{version}-{job}")


def _no_space(path: Path) -> str:
    if re.search(r"\s", str(path)):
        raise ValueError(f"HTCondor submit files do not allow spaces in paths: {path}")
    return str(path)


def render_script(cfg: "Config | TrainingConfig", job: Job, version_dir: str, env: JobEnv) -> str:
    q = shlex.quote
    logs_dest = storage.join(version_dir, "logs")
    copy_log = (f'{q(str(env.python))} -c "import sys; from tau3mu_gnn.pipelines.common import storage; '
                f'storage.put(sys.argv[1], sys.argv[2])" "$f" {q(logs_dest)}/"$(basename "$f")" || true')
    args = " ".join(q(a) for a in job.args)
    return "\n".join([
        "#!/bin/bash",
        f"# HTCondor job {job.name} of {version_dir}",
        "set -eo pipefail",
        'export USER="${USER:-$(id -un)}"   # the config paths may use $USER; HTCondor does not always set it',
        'JOB_ID="${TAU3MU_JOB_ID:-local}"',
        'JOB_SCRATCH="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}/tau3mu_$JOB_ID"',
        'mkdir -p "$JOB_SCRATCH/work" "$JOB_SCRATCH/logs"',
        f'OUT="$JOB_SCRATCH/logs/{job.name}-$JOB_ID.out"',
        f'ERR="$JOB_SCRATCH/logs/{job.name}-$JOB_ID.err"',
        "cleanup() {",
        '    [ -f "$OUT" ] && cat "$OUT"',
        '    [ -f "$ERR" ] && cat "$ERR" >&2',
        '    for f in "$OUT" "$ERR"; do',
        f'        [ -f "$f" ] && {copy_log}',
        "    done",
        '    rm -rf "$JOB_SCRATCH"',
        "}",
        "trap cleanup EXIT",
        f"source {q(str(env.conda_base / 'bin' / 'activate'))} {q(str(env.env_prefix))}",
        f'export PYTHONPATH={q(str(env.src_dir))}${{PYTHONPATH:+:$PYTHONPATH}}',
        'export TMPDIR="$JOB_SCRATCH/work"',
        'cd "$JOB_SCRATCH/work"',
        f'{q(str(env.python))} -m {job.module} {args} >"$OUT" 2>"$ERR"',
    ]) + "\n"


def render_submit(cfg: "Config | TrainingConfig", job: Job, log_folder: Path) -> str:
    res = cfg.htcondor.jobs[job.resources]
    folder = _no_space(log_folder)
    lines = [
        "universe       = vanilla",
        f"executable     = {folder}/{job.name}.sh",
        f"output         = {folder}/{job.name}-$(ClusterId).out",
        f"error          = {folder}/{job.name}-$(ClusterId).err",
        f"log            = {folder}/{job.name}.log",
        'environment    = "TAU3MU_JOB_ID=$(ClusterId)"',
        f"request_cpus   = {res.cpus}",
        f"request_memory = {memory_mb(res.mem)}",
        f"+MaxRuntime    = {round(time_hours(res.time) * 3600)}",
    ]
    if res.gpus:
        lines.append(f"request_gpus   = {res.gpus}")
    if res.requirements:
        lines.append(f"requirements   = {res.requirements}")
    if storage.needs_proxy(cfg.output_dir):
        lines.append("use_x509userproxy = true")
    return "\n".join(lines + ["queue"]) + "\n"


def render_dag(parts: Sequence[tuple[str, Sequence[Job], Path]]) -> str:
    """parts: (version_dir, jobs, log_folder) per Batch.add(). after_ids are node names of earlier parts."""
    jobs, edges, post = [], [], set()
    for version_dir, part_jobs, log_folder in parts:
        for j in part_jobs:
            node = node_name(version_dir, j.name)
            jobs.append(f"JOB {node} {_no_space(log_folder)}/{j.name}.sub")
            parents = [node_name(version_dir, a) for a in j.after]
            if j.dependency == "afterany":
                post.update(parents)
            parents += list(j.after_ids)
            if parents:
                edges.append(f"PARENT {' '.join(parents)} CHILD {node}")
    scripts = [f"SCRIPT POST {n} /bin/true" for n in sorted(post)]   # afterany: the parent always counts as done
    return "\n".join(jobs + edges + scripts) + "\n"


def afs_free_mb(path: Path) -> float | None:
    """Free AFS quota (MB) of the volume of path; None if not on AFS or `fs` fails."""
    if not str(path).startswith("/afs/"):
        return None
    try:
        out = subprocess.run(["fs", "listquota", str(path)], capture_output=True, text=True, check=True).stdout
        quota, used = (int(x) for x in out.splitlines()[1].split()[1:3])   # KB
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return None
    return (quota - used) / 1024


class HTCondorBatch(Batch):
    """Writes the job files in add(); finish() submits one DAG over all add() calls."""

    def __init__(self):
        self._parts: list[tuple[str, list[Job], Path]] = []
        self._run_name = ""

    def add(self, cfg, jobs, version_dir, log_folder, env, dry_run=False):
        log_folder = Path(log_folder)
        check_dependencies(jobs)
        log_folder.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            script = log_folder / f"{job.name}.sh"
            script.write_text(render_script(cfg, job, version_dir, env))
            script.chmod(0o755)
            (log_folder / f"{job.name}.sub").write_text(render_submit(cfg, job, log_folder))
        self._parts.append((version_dir, list(jobs), log_folder))
        self._run_name = cfg.run_name
        return {j.name: node_name(version_dir, j.name) for j in jobs}

    def finish(self, dry_run=False):
        if not self._parts:
            return {}
        dag = self._parts[0][2] / DAG_FILE
        dag.write_text(render_dag(self._parts))
        versions = "+".join(storage.join(v).rsplit("/", 1)[-1] for v, _, _ in self._parts)
        cmd = ["condor_submit_dag", "-batch-name", f"{self._run_name}.{versions}", str(dag)]
        if dry_run:
            print(" ".join(shlex.quote(c) for c in cmd))
            return {v: {j.name: f"DRYRUN-{j.name}" for j in jobs} for v, jobs, _ in self._parts}
        free = afs_free_mb(dag.parent)
        if free is not None and free < AFS_MIN_FREE_MB:
            log.warning("only %.0f MB free on AFS in %s: the job logs may not fit (delete old logs; they are "
                        "also next to the results, <version>/logs/)", free, dag.parent)
        # cwd: the job's start folder (Iwd) is the folder of condor_submit_dag; the CERN schedds refuse /eos there
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=dag.parent)
        m = _CLUSTER_RE.search(res.stdout)
        if res.returncode != 0 or not m:
            raise SubmitError(f"condor_submit_dag failed for {dag}: {(res.stderr or res.stdout).strip()}")
        cluster = m.group(1)
        return {v: {j.name: f"{cluster}:{node_name(v, j.name)}" for j in jobs} for v, jobs, _ in self._parts}
