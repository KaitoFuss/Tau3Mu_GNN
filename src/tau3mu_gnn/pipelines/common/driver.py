"""Submission shared by both pipelines: make the next version folder, write and submit the job scripts.
The phase layouts are in analysis/driver.py and training/driver.py; the batch details in batch.py, slurm.py and
htcondor.py."""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tau3mu_gnn.pipelines.common import runs, storage
from tau3mu_gnn.pipelines.common.batch import Batch, chain_hours, check_proxy, make_batch
from tau3mu_gnn.pipelines.common.manifest import JOBS_FILE

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Submission:
    version_dir: str
    log_folder: Path
    job_ids: dict[str, str]
    upstream: str | None


def _submit(cfg, jobs_for, prefix, parent, log_parts, upstream, dry_run, env, proxy=True, *, create,
            pipeline: str, batch: Batch | None = None) -> Submission:
    """jobs_for(version_dir) -> job list. Shared tail of the submit functions of both pipelines.
    create(): makes the version folder (the pipeline's create_version, with its settings snapshot).
    pipeline: 'analysis' or 'training'; the logs go to <log_dir>/<pipeline>/<run_name>/..., so a training and an
    analysis with the same name do not share a log folder.
    batch: given (submit_all): only add the jobs; the caller runs finish(). None: a new batch, finished here."""
    if proxy:
        check_proxy(cfg, chain_hours(cfg, jobs_for("<version>")))
    name = runs.next_version(parent, prefix)
    if dry_run:
        version_dir = storage.join(parent, name)
    else:
        version_dir = create()
    name = version_dir.rsplit("/", 1)[-1]
    log_folder = Path(cfg.paths.log_dir, pipeline, cfg.run_name, *log_parts, name)
    own = batch is None
    batch = make_batch(cfg) if own else batch
    ids = batch.add(cfg, jobs_for(version_dir), version_dir, log_folder, env, dry_run=dry_run)
    sub = Submission(version_dir, log_folder, ids, upstream)
    return finish(batch, [sub], dry_run)[0] if own else sub


def finish(batch: Batch, subs: Sequence[Submission], dry_run: bool) -> list[Submission]:
    """Submit what the batch has not submitted yet; write <version>/jobs.json with the final job IDs."""
    final = batch.finish(dry_run)
    out = [dataclasses.replace(s, job_ids=final.get(s.version_dir, s.job_ids)) for s in subs]
    if not dry_run:
        write_jobs(out)
    for s in out:
        log.info("%s: logs %s, jobs %s", s.version_dir, s.log_folder, s.job_ids)
    return out


def write_jobs(subs: Sequence[Submission]) -> None:
    for s in subs:
        storage.write_text(storage.join(s.version_dir, JOBS_FILE), json.dumps(s.job_ids, indent=2) + "\n")
