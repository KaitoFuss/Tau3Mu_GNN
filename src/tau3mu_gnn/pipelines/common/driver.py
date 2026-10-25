"""Submission shared by both pipelines: make the next version folder, write and submit the job scripts.
The phase layouts are in analysis/driver.py and training/driver.py; the SLURM details in slurm.py."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tau3mu_gnn.pipelines.common import runs, storage
from tau3mu_gnn.pipelines.common.manifest import JOBS_FILE
from tau3mu_gnn.pipelines.common.slurm import chain_hours, check_proxy, submit_jobs

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Submission:
    version_dir: str
    log_folder: Path
    job_ids: dict[str, str]
    upstream: str | None


def _submit(cfg, jobs_for, prefix, parent, log_parts, upstream, dry_run, env, proxy=True, *, create) \
        -> Submission:
    """jobs_for(version_dir) -> job list. Shared tail of the submit functions of both pipelines.
    create(): makes the version folder (the pipeline's create_version, with its settings snapshot)."""
    if proxy:
        check_proxy(cfg, chain_hours(cfg, jobs_for("<version>")))
    name = runs.next_version(parent, prefix)
    if dry_run:
        version_dir = storage.join(parent, name)
    else:
        version_dir = create()
    name = version_dir.rsplit("/", 1)[-1]
    log_folder = Path(cfg.paths.log_dir, cfg.run_name, *log_parts, name)
    ids = submit_jobs(cfg, jobs_for(version_dir), version_dir, log_folder, env, dry_run=dry_run)
    if not dry_run:
        storage.write_text(storage.join(version_dir, JOBS_FILE), json.dumps(ids, indent=2) + "\n")
    log.info("%s: %s, logs %s, jobs %s", name, version_dir, log_folder, ids)
    return Submission(version_dir, log_folder, ids, upstream)
