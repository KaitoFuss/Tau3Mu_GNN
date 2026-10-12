# Mini Step 9b — SLURM Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The phase commands. `python -m tau3mu_gnn.pipeline.cli phase-a|phase-b|phase-c|stage ...` checks config, upstream and proxy, makes the next version folder, writes one bash job script per job, and submits them with `sbatch --parsable` and dependencies (or prints them with `--dry-run`). Plus the inputs for the small-input check on T3 (D9), which Kaito queues.

**Architecture:**
- `pipeline/slurm.py`: `Job` (name, resources key, job-CLI args, dependencies), `JobEnv` (conda base, env prefix, python, src dir; read from the submitting shell), `render_script`, `check_proxy`, `submit_jobs` (sbatch, dependency strings, dry-run IDs, error listing already submitted IDs).
- `pipeline/driver.py`: job lists per phase and per single stage (spec sec. 6), version folder + log folder, `submit_phase`, `submit_stage`; writes `jobs.json` into the version folder after submission.
- `pipeline/cli.py`: argparse front end; prints version folder, log folder, job IDs; known errors → message + exit 1.
- Small changes: `runs.pick_version` (any phase, complete or not; `pick_upstream` uses it); `jobs.py` items carried from 9a (log lines, `eval` checks its upstream phase, `summary` needs `--upstream` only with a manifest).

**Tech Stack:** Python 3.11 (Tau3Mu env), bash, SLURM 25.11 (`sbatch --parsable`), pytest with fake `sbatch` / `voms-proxy-info` executables. No new dependency.

**Spec:** `docs/superpowers/specs/2026-10-12-step9-driver-design.md` (sec. 1 CLI, 4 versions, 6 jobs, 7 job script + submission, 10 errors, 11 tests "9b"). Decisions added 2026-10-12 (Kaito): **E1** the job script activates the conda env of the submitting shell (`source <conda base>/bin/activate <env prefix>`); **E2** the small-input check uses 20k signal / 1M background events, cuts `[2.0, 3.9]`.

## Global Constraints

- Test command: `$PYTEST` = `PYTHONPATH=/work/kfuss/SVJ_GNN/.claude/worktrees/step9b-slurm/src /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m pytest tests/pipeline`. `$PYTEST/x.py` = same with `tests/pipeline/x.py`.
- Branch `pipeline-step9b-slurm`, from `pipeline-step9a-driver-core` (PR #20, open). Worktree `/work/kfuss/SVJ_GNN/.claude/worktrees/step9b-slurm`. The PR is stacked on #20.
- Git in this worktree: `/usr/bin/git` only. One git command per Bash call.
- **Never call the real `sbatch` in tests or by hand.** Tests put a fake `sbatch` first on `PATH`. Real submissions: Kaito only (memory rule: prepare, tell, Kaito queues).
- No new dependency.
- Job names in `slurm.jobs` (frozen defaults): `skim, bdt, gnn_signal, gnn_background, eval, limit, summary`.
- Phase jobs (spec sec. 6): A = `skim` → `bdt` (`afterok`); B = `gnn_signal`, `gnn_background` → `eval` (`afterok` both); C = `limit_<cut>` per cut → `summary` (`afterany` all). Only `bdt`, `eval`, `summary` write the manifest.
- SLURM job name: `<analysis_name>.<version name>.<job name>` (e.g. `toy_run.phase_b_vs2.gnn_signal`). Script file: `<log folder>/<job name>.sh`. Logs: `<log folder>/<job name>-%j.out|.err`. Log folder: `<paths.log_dir>/<analysis_name>/<version name>` (debug: `<paths.log_dir>/<analysis_name>/debug/<stage>_vsN`).
- Job script (spec sec. 7 + E1): `set -eo pipefail` (no `-u`: conda activate scripts read unset variables); `source <conda base>/bin/activate <env prefix>`; `export PYTHONPATH=<src dir of the submitting tau3mu_gnn>`; scratch `/scratch/$USER/$SLURM_JOB_ID`, `TMPDIR` there; `trap cleanup EXIT` copies the job's own `.out`/`.err` to `<version dir>/logs/` (best effort) and removes scratch; `cd` scratch; `<python> -m tau3mu_gnn.pipeline.jobs <args>`. All substituted values shell-quoted (`shlex.quote`).
- Conda env: from the submitting shell: `CONDA_PREFIX` (env prefix) and `CONDA_EXE` (conda base = `Path(CONDA_EXE).parents[1]`). Missing → error at submit: "activate the conda env first (conda activate Tau3Mu)".
- Proxy check only if `output_base` is `root://`: `X509_USER_PROXY` set and not under `/tmp`; `voms-proxy-info --exists --valid <H>:00` with H = ceil(longest `time` of the jobs to submit, in hours). Failure → error naming `voms-proxy-init --voms cms --valid 192:00`.
- `--dry-run`: all checks run; the version name is computed but **no** version folder and no `config_used.yaml` are made; job scripts are written to the log folder; the `sbatch` commands are printed with placeholder IDs `DRYRUN-1`, `DRYRUN-2`, ...; nothing is submitted.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **`sbatch` fails for the 2nd of 3 jobs** → error that lists the IDs already submitted (so the user can `scancel` them); no `jobs.json` claim of success. Test in Task 3.
2. **A submitted job's dependency string** for phase C with 3 cuts → `afterany:<id1>:<id2>:<id3>`, not `afterok`, and no dependency on any other job. Test in Task 4.
3. **Values with spaces or `$`** in paths (e.g. a config path `/work/x y/cfg.yaml`) → the script still passes them as one argument (shell quoting). Test in Task 3.
4. **No active conda env at submit** (`CONDA_PREFIX` unset) → clear error before any folder or script is made. Test in Task 3 / Task 4.
5. **`stage limit --cut 4.5` with 4.5 not in `limit.gnn_score_cut`** → error at submit, before any job. Test in Task 4.

## File Structure

| File | Responsibility |
|---|---|
| `src/tau3mu_gnn/pipeline/runs.py` (modify) | `pick_version` |
| `src/tau3mu_gnn/pipeline/jobs.py` (modify) | log lines; `eval` upstream phase check; `summary` `--upstream` optional with `--no-manifest` |
| `src/tau3mu_gnn/pipeline/slurm.py` (create) | `Job`, `JobEnv`, `time_hours`, `render_script`, `check_proxy`, `submit_jobs`, errors |
| `src/tau3mu_gnn/pipeline/driver.py` (create) | phase / stage job lists, `submit_phase`, `submit_stage` |
| `src/tau3mu_gnn/pipeline/cli.py` (create) | argparse front end |
| `tests/pipeline/test_runs.py`, `test_jobs.py` (modify); `test_slurm.py`, `test_driver.py`, `test_cli.py` (create) | tests |
| `docs/wrapper_roadmap.md` (modify) | 9b done, E1/E2 |
| `regression/step9b/` in `/work/kfuss/SVJ_GNN/.claude/worktrees/pipeline-wrapper/` (create, **not committed**) | small inputs job, small config, run instructions |

---

### Task 1: `runs.pick_version` and the 9a carry-overs in `jobs.py`

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/runs.py`, `src/tau3mu_gnn/pipeline/jobs.py`
- Test: `tests/pipeline/test_runs.py`, `tests/pipeline/test_jobs.py`

**Interfaces:**
- Produces: `runs.pick_version(cfg, phase, from_=None, complete=True) -> str` (full path); `pick_upstream(cfg, phase, from_=None)` = `pick_version(cfg, UPSTREAM[phase], from_)` (same messages as before). `jobs.main`: `summary` needs `--upstream` only without `--no-manifest`; `jobs.summary(cfg, phase_dir, upstream_dir=None, source_dir=None, manifest=True)` (raise `ValueError` if `manifest` and `upstream_dir is None`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/test_runs.py`:

```python
def test_pick_version_any_phase(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipeline.manifest import UpstreamError
    from tau3mu_gnn.pipeline.runs import pick_version

    user["paths"]["output_base"] = str(tmp_path)
    cfg = load(user)
    mk(tmp_path / "toy_run", "phase_c_vs1", "phase_c_vs2", complete=("phase_c_vs1",))
    assert pick_version(cfg, "phase_c") == f"{tmp_path}/toy_run/phase_c_vs1"
    assert pick_version(cfg, "phase_c", complete=False) == f"{tmp_path}/toy_run/phase_c_vs2"
    assert pick_version(cfg, "phase_c", from_="phase_c_vs2", complete=False) == f"{tmp_path}/toy_run/phase_c_vs2"
    with pytest.raises(UpstreamError, match="no manifest"):
        pick_version(cfg, "phase_c", from_="phase_c_vs2")
    with pytest.raises(UpstreamError, match="phase-b"):
        pick_version(cfg, "phase_b")
```

Append to `tests/pipeline/test_jobs.py`:

```python
def test_eval_job_checks_upstream_phase(cfg, scratch, fake_stages):
    """Debug 'stage eval' with --upstream pointing at a phase B folder: clear error, not KeyError."""
    from tau3mu_gnn.pipeline import jobs
    from tau3mu_gnn.pipeline.manifest import UpstreamError

    run_phase_a(cfg)
    b = run_phase_b(cfg)
    with pytest.raises(UpstreamError, match="phase_a"):
        jobs.evaluation(cfg, b, b, source_dir=b, manifest=False)


def test_summary_debug_without_upstream(cfg, scratch, fake_stages, monkeypatch):
    from tau3mu_gnn.pipeline import jobs, runs

    run_phase_a(cfg)
    run_phase_b(cfg)
    up = runs.pick_upstream(cfg, "phase_c")
    c = runs.create_version(cfg, "phase_c")
    for cut in cfg.limit.gnn_score_cut:
        jobs.limit(cfg, c, up, cut)
    d = runs.create_version(cfg, "summary", parent=f"{cfg.output_dir}/debug")
    monkeypatch.setattr(jobs, "load_config", lambda path: cfg)
    jobs.main(["summary", "--config", "x.yaml", "--phase-dir", d, "--source", c, "--no-manifest"])
    assert (Path(d) / "limits.json").is_file() and not (Path(d) / "manifest.json").exists()
    with pytest.raises(SystemExit):
        jobs.main(["summary", "--config", "x.yaml", "--phase-dir", d])        # manifest needs --upstream


def test_job_logs_what_it_does(cfg, scratch, fake_stages, caplog):
    import logging

    caplog.set_level(logging.INFO)
    a = run_phase_a(cfg)
    assert "job skim" in caplog.text and "job bdt" in caplog.text and a in caplog.text
```

- [ ] **Step 2: Run, expect FAIL** — `$PYTEST/test_runs.py $PYTEST/test_jobs.py -q -k "pick_version or upstream_phase or debug_without or logs_what"` (ImportError `pick_version`; KeyError instead of UpstreamError; SystemExit for the debug summary; no log text).

- [ ] **Step 3: Implement**

`runs.py` — replace `pick_upstream` with:

```python
def pick_version(cfg: Config, phase: str, from_: str | None = None, complete: bool = True) -> str:
    """Full path of a version of `phase`: from_ if given, else the highest one. With complete=True only
    versions with a manifest count (failed or running ones are skipped with a warning)."""
    base = cfg.output_dir
    if from_ is not None:
        if _number(from_, phase) is None:
            raise UpstreamError(f"--from {from_!r}: needs a {phase}_vsN version")
        if not storage.exists(storage.join(base, from_)):
            raise UpstreamError(f"--from {from_!r}: not found in {base}")
        if complete and not is_complete(base, from_):
            raise UpstreamError(f"--from {from_!r}: no manifest (failed or still running)")
        return storage.join(base, from_)
    names = versions(base, phase)
    pool = [n for n in names if is_complete(base, n)] if complete else names
    if not pool:
        kind = "complete " if complete else ""
        raise UpstreamError(f"no {kind}{phase} version in {base}; run {phase.replace('_', '-')} first")
    newer = names[names.index(pool[-1]) + 1:]
    if newer:
        log.warning("%s have no manifest (failed or still running); using %s", ", ".join(newer), pool[-1])
    return storage.join(base, pool[-1])


def pick_upstream(cfg: Config, phase: str, from_: str | None = None) -> str:
    """Full path of the upstream version for `phase` (complete only). Raise UpstreamError if there is none."""
    up = UPSTREAM[phase]
    if up is None:
        raise ValueError(f"{phase} has no upstream phase")
    return pick_version(cfg, up, from_)
```

`jobs.py`:
- `evaluation`: replace `up = read_manifest(upstream_dir)` with `up = check_upstream(cfg, upstream_dir, "phase_b")` (checks phase, files, settings — the phase A version may have been checked by the GNN jobs already; checking again is cheap and catches a wrong `--upstream` in debug runs).
- `summary` signature `def summary(cfg, phase_dir, upstream_dir=None, source_dir=None, manifest=True) -> None:`, first line of the body: `if manifest and upstream_dir is None: raise ValueError("summary with a manifest needs the upstream phase B folder")`.
- `main`: replace the `need` dict entry `"summary": ("upstream",)` by computing after parsing: `if a.job == "summary" and not a.no_manifest: need = ("upstream",)`.
- Log lines (module logger `log`): at the start of each job function `log.info("job %s: %s", "<job>", phase_dir)` plus the relevant upstream / sample / cut (e.g. `log.info("job gnn: %s, sample %s, upstream %s", phase_dir, sample, upstream_dir)`); before each `storage.put(...)`: `log.info("copy %s -> %s", <src>, <dest>)`.

- [ ] **Step 4: Run `$PYTEST/test_runs.py $PYTEST/test_jobs.py -q`, expect PASS** (old tests unchanged and passing)

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/runs.py src/tau3mu_gnn/pipeline/jobs.py tests/pipeline/test_runs.py tests/pipeline/test_jobs.py
/usr/bin/git commit -m "feat(pipeline): pick_version; jobs log lines, eval checks upstream, debug summary (step 9b)"
```

---

### Task 2: `slurm.py` — job scripts and proxy check

**Files:**
- Create: `src/tau3mu_gnn/pipeline/slurm.py`
- Test: `tests/pipeline/test_slurm.py`

**Interfaces:**
- Consumes: `Config.slurm` (`account`, `jobs[name]` → `SlurmJob(partition, mem, time, cpus)`), `Config.analysis_name`, `Config.user_file`, `storage.join`, `storage.is_remote`.
- Produces:
  - `class SubmitError(Exception)`, `class ProxyError(SubmitError)`, `class EnvError(SubmitError)`
  - `@dataclass(frozen=True) class Job: name: str; resources: str; args: tuple[str, ...]; after: tuple[str, ...] = (); dependency: str = "afterok"`
  - `@dataclass(frozen=True) class JobEnv: conda_base: Path; env_prefix: Path; python: Path; src_dir: Path`
  - `JobEnv.from_environment(environ=os.environ) -> JobEnv` (raises `EnvError`)
  - `time_hours(time: str) -> float` (`"1-00:00:00"` → 24.0, `"0-01:30:00"` → 1.5, `"06:00:00"` → 6.0)
  - `render_script(cfg, job, version_dir: str, log_folder: Path, env: JobEnv, scratch_root: str = "/scratch/$USER") -> str`
  - `check_proxy(cfg, jobs, environ=os.environ) -> None` (raises `ProxyError`; no-op for a local `output_base`)

- [ ] **Step 1: Write the failing tests** — `tests/pipeline/test_slurm.py`

```python
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ENV = {"CONDA_PREFIX": "/work/u/miniforge3/envs/Tau3Mu", "CONDA_EXE": "/work/u/miniforge3/bin/conda"}


def test_time_hours():
    from tau3mu_gnn.pipeline.slurm import time_hours

    assert time_hours("1-00:00:00") == 24.0
    assert time_hours("0-01:30:00") == 1.5
    assert time_hours("06:00:00") == 6.0
    assert time_hours("2-12:00:00") == 60.0


def test_job_env_from_environment():
    import sys

    from tau3mu_gnn.pipeline.slurm import JobEnv

    env = JobEnv.from_environment(ENV)
    assert env.env_prefix == Path("/work/u/miniforge3/envs/Tau3Mu")
    assert env.conda_base == Path("/work/u/miniforge3")
    assert env.python == Path(sys.executable)
    assert (env.src_dir / "tau3mu_gnn" / "pipeline" / "slurm.py").is_file()


@pytest.mark.parametrize("missing", ["CONDA_PREFIX", "CONDA_EXE"])
def test_job_env_needs_active_env(missing):
    from tau3mu_gnn.pipeline.slurm import EnvError, JobEnv

    env = dict(ENV)
    del env[missing]
    with pytest.raises(EnvError, match="conda activate"):
        JobEnv.from_environment(env)


def render(load, user, tmp_path, args=("skim",), user_file=None):
    from tau3mu_gnn.pipeline.slurm import Job, JobEnv, render_script

    cfg = load(user)
    job = Job(name="gnn_signal", resources="gnn_signal", args=tuple(args))
    env = JobEnv(Path("/c/base"), Path("/c/base/envs/T"), Path("/c/base/envs/T/bin/python"), Path("/repo/src"))
    return render_script(cfg, job, "/out/toy_run/phase_b_vs2", tmp_path / "logs", env)


def test_render_script_header_and_body(load, user, tmp_path):
    text = render(load, user, tmp_path, args=("gnn", "--sample", "signal"))
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH --job-name=toy_run.phase_b_vs2.gnn_signal" in lines
    assert "#SBATCH --account=t3" in lines and "#SBATCH --partition=standard" in lines
    assert "#SBATCH --mem=4G" in lines and "#SBATCH --time=0-01:00:00" in lines
    assert "#SBATCH --cpus-per-task=1" in lines
    assert f"#SBATCH --output={tmp_path / 'logs'}/gnn_signal-%j.out" in lines
    assert f"#SBATCH --error={tmp_path / 'logs'}/gnn_signal-%j.err" in lines
    assert "set -eo pipefail" in lines
    assert "source /c/base/bin/activate /c/base/envs/T" in lines
    assert "trap cleanup EXIT" in lines
    assert lines[-1] == "/c/base/envs/T/bin/python -m tau3mu_gnn.pipeline.jobs gnn --sample signal"
    assert "/out/toy_run/phase_b_vs2/logs" in text            # log copy target


def test_render_script_quotes_values(load, user, tmp_path):
    text = render(load, user, tmp_path, args=("skim", "--config", "/work/x y/c$fg.yaml"))
    last = text.splitlines()[-1]
    assert shlex.split(last)[-1] == "/work/x y/c$fg.yaml"


def test_render_script_is_valid_bash(load, user, tmp_path):
    path = tmp_path / "job.sh"
    path.write_text(render(load, user, tmp_path))
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.fixture
def fake_voms(tmp_path, monkeypatch):
    """voms-proxy-info that logs its arguments and exits with $FAKE_VOMS_RC (default 0)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "voms.log"
    exe = bin_dir / "voms-proxy-info"
    exe.write_text(f'#!/bin/sh\necho "$*" >> {log}\nexit ${{FAKE_VOMS_RC:-0}}\n')
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def remote_cfg(load, user):
    user["paths"]["output_base"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/x"
    return load(user)


def jobs_with(*resources):
    from tau3mu_gnn.pipeline.slurm import Job

    return [Job(name=r, resources=r, args=("skim",)) for r in resources]


def test_proxy_check_hours_from_longest_job(load, user, fake_voms, monkeypatch, fake_repo):
    import copy

    import yaml
    from conftest import TOY_DEFAULTS
    from tau3mu_gnn.pipeline.slurm import check_proxy

    d = copy.deepcopy(TOY_DEFAULTS)
    d["slurm"]["jobs"]["gnn_background"]["time"] = "2-00:00:00"
    (fake_repo / "configs" / "analysis_defaults_W.yaml").write_text(yaml.safe_dump(d))
    check_proxy(remote_cfg(load, user), jobs_with("gnn_signal", "gnn_background"),
                {"X509_USER_PROXY": "/t3home/u/.x509up_u1"})
    assert fake_voms.read_text().split() == ["--exists", "--valid", "48:00"]


def test_proxy_check_rounds_up(load, user, fake_voms, monkeypatch, fake_repo):
    import copy

    import yaml
    from conftest import TOY_DEFAULTS
    from tau3mu_gnn.pipeline.slurm import check_proxy

    d = copy.deepcopy(TOY_DEFAULTS)
    d["slurm"]["jobs"]["limit"]["time"] = "0-01:30:00"
    (fake_repo / "configs" / "analysis_defaults_W.yaml").write_text(yaml.safe_dump(d))
    check_proxy(remote_cfg(load, user), jobs_with("limit"), {"X509_USER_PROXY": "/t3home/u/.x509up_u1"})
    assert fake_voms.read_text().split()[-1] == "2:00"


def test_proxy_invalid(load, user, fake_voms, monkeypatch):
    from tau3mu_gnn.pipeline.slurm import ProxyError, check_proxy

    monkeypatch.setenv("FAKE_VOMS_RC", "1")
    with pytest.raises(ProxyError, match="voms-proxy-init --voms cms --valid 192:00"):
        check_proxy(remote_cfg(load, user), jobs_with("skim"), {"X509_USER_PROXY": "/t3home/u/.x509up_u1"})


@pytest.mark.parametrize("environ, match", [({}, "X509_USER_PROXY"),
                                            ({"X509_USER_PROXY": "/tmp/x509up_u1"}, "/tmp")])
def test_proxy_location(load, user, fake_voms, environ, match):
    from tau3mu_gnn.pipeline.slurm import ProxyError, check_proxy

    with pytest.raises(ProxyError, match=match):
        check_proxy(remote_cfg(load, user), jobs_with("skim"), environ)


def test_proxy_not_needed_for_local_output(load, user, fake_voms):
    from tau3mu_gnn.pipeline.slurm import check_proxy

    check_proxy(load(user), jobs_with("skim"), {})
    assert not fake_voms.exists()
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: tau3mu_gnn.pipeline.slurm`) — `$PYTEST/test_slurm.py -q`

- [ ] **Step 3: Implement** `src/tau3mu_gnn/pipeline/slurm.py` (first part; `submit_jobs` is Task 3)

```python
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
    def from_environment(cls, environ: Mapping[str, str] = os.environ) -> "JobEnv":
        prefix, exe = environ.get("CONDA_PREFIX"), environ.get("CONDA_EXE")
        if not prefix or not exe:
            raise EnvError("no active conda env (CONDA_PREFIX / CONDA_EXE not set): "
                           "activate the conda env first (conda activate Tau3Mu)")
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
```

Note: the last script line must be exactly `<python> -m tau3mu_gnn.pipeline.jobs <args>` (test). `shlex.quote` leaves `/c/base/envs/T/bin/python` unquoted.

- [ ] **Step 4: Run `$PYTEST/test_slurm.py -q`, expect PASS**

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/slurm.py tests/pipeline/test_slurm.py
/usr/bin/git commit -m "feat(pipeline): SLURM job scripts and proxy check (step 9b)"
```

---

### Task 3: `slurm.submit_jobs` + a real run of a generated script

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/slurm.py`
- Test: `tests/pipeline/test_slurm.py`

**Interfaces:**
- Consumes: Task 2 (`Job`, `JobEnv`, `render_script`, `SubmitError`).
- Produces: `submit_jobs(cfg, jobs, version_dir, log_folder, env, dry_run=False) -> dict[str, str]` (job name → SLURM ID, or `DRYRUN-<n>`). Writes `<log_folder>/<job name>.sh` for every job (also in dry-run). Jobs are submitted in list order; a job's `after` names must appear earlier in the list. Dependency option: `--dependency=<dependency>:<id1>:<id2>...`. Dry-run prints one line per job: `sbatch --parsable [--dependency=...] <script>`.

- [ ] **Step 1: Write the failing tests** (append to `tests/pipeline/test_slurm.py`)

```python
@pytest.fixture
def fake_sbatch(tmp_path, monkeypatch):
    """sbatch that logs its arguments and prints increasing IDs. FAKE_SBATCH_FAIL_AT=n: call n fails."""
    bin_dir = tmp_path / "sbin"
    bin_dir.mkdir()
    log, counter = tmp_path / "sbatch.log", tmp_path / "sbatch.n"
    exe = bin_dir / "sbatch"
    exe.write_text(f"""#!/bin/sh
n=$(( $(cat {counter} 2>/dev/null || echo 0) + 1 )); echo $n > {counter}
if [ "$n" = "${{FAKE_SBATCH_FAIL_AT:-0}}" ]; then echo "sbatch: error: invalid partition" >&2; exit 1; fi
echo "$*" >> {log}
echo $((1000 + n))
""")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def three_jobs():
    from tau3mu_gnn.pipeline.slurm import Job

    return [Job("limit_2.0", "limit", ("limit", "--cut", "2.0")),
            Job("limit_3.9", "limit", ("limit", "--cut", "3.9")),
            Job("summary", "summary", ("summary",), after=("limit_2.0", "limit_3.9"), dependency="afterany")]


def test_submit_jobs_dependencies(load, user, tmp_path, fake_sbatch):
    from tau3mu_gnn.pipeline.slurm import JobEnv, submit_jobs

    ids = submit_jobs(load(user), three_jobs(), "/out/toy_run/phase_c_vs1", tmp_path / "logs", JobEnv.from_environment(ENV))
    assert ids == {"limit_2.0": "1001", "limit_3.9": "1002", "summary": "1003"}
    calls = fake_sbatch.read_text().splitlines()
    assert calls[0] == f"--parsable {tmp_path / 'logs' / 'limit_2.0.sh'}"
    assert calls[2] == f"--parsable --dependency=afterany:1001:1002 {tmp_path / 'logs' / 'summary.sh'}"
    assert (tmp_path / "logs" / "summary.sh").read_text().startswith("#!/bin/bash")


def test_submit_jobs_dry_run(load, user, tmp_path, fake_sbatch, capsys):
    from tau3mu_gnn.pipeline.slurm import JobEnv, submit_jobs

    ids = submit_jobs(load(user), three_jobs(), "/out/toy_run/phase_c_vs1", tmp_path / "logs",
                      JobEnv.from_environment(ENV), dry_run=True)
    assert ids == {"limit_2.0": "DRYRUN-1", "limit_3.9": "DRYRUN-2", "summary": "DRYRUN-3"}
    assert not fake_sbatch.exists()
    out = capsys.readouterr().out
    assert f"sbatch --parsable --dependency=afterany:DRYRUN-1:DRYRUN-2 {tmp_path / 'logs' / 'summary.sh'}" in out
    assert (tmp_path / "logs" / "limit_3.9.sh").is_file()


def test_submit_jobs_failure_lists_submitted(load, user, tmp_path, fake_sbatch, monkeypatch):
    from tau3mu_gnn.pipeline.slurm import JobEnv, SubmitError, submit_jobs

    monkeypatch.setenv("FAKE_SBATCH_FAIL_AT", "2")
    with pytest.raises(SubmitError) as e:
        submit_jobs(load(user), three_jobs(), "/out/toy_run/phase_c_vs1", tmp_path / "logs", JobEnv.from_environment(ENV))
    msg = str(e.value)
    assert "limit_3.9" in msg and "invalid partition" in msg and "scancel 1001" in msg


def test_submit_jobs_unknown_dependency(load, user, tmp_path, fake_sbatch):
    from tau3mu_gnn.pipeline.slurm import Job, JobEnv, submit_jobs

    with pytest.raises(ValueError, match="skim"):
        submit_jobs(load(user), [Job("bdt", "bdt", ("bdt",), after=("skim",))], "/o/v", tmp_path,
                    JobEnv.from_environment(ENV))


def test_generated_script_runs(tmp_path, monkeypatch):
    """Run a generated script with bash (no SLURM): conda activation, scratch, job CLI, log copy, cleanup.
    Uses the real env of this shell and the real repo defaults; the job is a debug 'summary'."""
    import json

    from tau3mu_gnn.pipeline.config import load_config
    from tau3mu_gnn.pipeline.slurm import Job, JobEnv, render_script

    if not (os.environ.get("CONDA_PREFIX") and os.environ.get("CONDA_EXE")):
        pytest.skip("needs an active conda env")
    cfg_file = tmp_path / "user.yaml"
    cfg_file.write_text(f"""analysis_name: script_test
channel: W
paths: {{output_base: {tmp_path / 'out'}, log_dir: {tmp_path / 'logs_root'}}}
inputs: {{signal: [/data/s.root], background: [/data/b.root], already_resonance_skimmed: true}}
limit: {{signal_weight: 1.0e-5, gnn_score_cut: [3.9]}}
""")
    cfg = load_config(cfg_file)
    src = tmp_path / "out" / "script_test" / "phase_c_vs1"
    (src / "cut_3.9").mkdir(parents=True)
    (src / "cut_3.9" / "limit.json").write_text(json.dumps({"cut": 3.9, "limit": 1.25, "limitErr": 0.0}))
    version_dir = str(tmp_path / "out" / "script_test" / "debug" / "summary_vs1")
    log_folder = tmp_path / "logs"
    log_folder.mkdir()
    job = Job("summary", "summary", ("summary", "--config", str(cfg_file), "--phase-dir", version_dir,
                                     "--source", str(src), "--no-manifest"))
    script = tmp_path / "summary.sh"
    script.write_text(render_script(cfg, job, version_dir, log_folder, JobEnv.from_environment(),
                                    scratch_root=str(tmp_path / "scratch")))
    out = log_folder / "summary-77.out"
    with open(out, "w") as fo:
        subprocess.run(["bash", str(script)], env={**os.environ, "SLURM_JOB_ID": "77"}, stdout=fo,
                       stderr=subprocess.STDOUT, check=True, timeout=300)
    assert json.loads((Path(version_dir) / "limits.json").read_text())["best_cut"] == 3.9
    assert (Path(version_dir) / "logs" / "summary-77.out").is_file()            # log copied by the trap
    assert not (tmp_path / "scratch" / "77").exists()                             # scratch removed
```

- [ ] **Step 2: Run, expect FAIL** (`ImportError: submit_jobs`; the script test fails only if the script is wrong — it may pass already; if so, note it in the report) — `$PYTEST/test_slurm.py -q`

- [ ] **Step 3: Implement** (append to `slurm.py`)

```python
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
```

(`--parsable` prints `<id>` or `<id>;<cluster>`.)

- [ ] **Step 4: Run `$PYTEST/test_slurm.py -q`, expect PASS** (the script test runs if the test shell has an active conda env; run the suite from an activated `Tau3Mu` shell: `source /work/kfuss/miniforge3/bin/activate Tau3Mu` before `$PYTEST`, and say in the report whether it ran)

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/slurm.py tests/pipeline/test_slurm.py
/usr/bin/git commit -m "feat(pipeline): submit jobs with sbatch dependencies, dry run (step 9b)"
```

---

### Task 4: `driver.py` — phase and stage submission

**Files:**
- Create: `src/tau3mu_gnn/pipeline/driver.py`
- Test: `tests/pipeline/test_driver.py`

**Interfaces:**
- Consumes: `runs.next_version`, `runs.create_version`, `runs.pick_version`, `runs.pick_upstream`, `manifest.read_manifest`, `manifest.JOBS_FILE`, `storage.join`, `storage.write_text`, `slurm.Job`, `slurm.JobEnv`, `slurm.check_proxy`, `slurm.submit_jobs`.
- Produces:
  - `STAGES = ("skim", "bdt", "gnn", "eval", "limit", "summary")`
  - `@dataclass(frozen=True) class Submission: version_dir: str; log_folder: Path; job_ids: dict[str, str]; upstream: str | None`
  - `submit_phase(cfg, phase, from_=None, dry_run=False, env=None) -> Submission` (`env` default `JobEnv.from_environment()`)
  - `submit_stage(cfg, stage, from_=None, cut=None, sample=None, dry_run=False, env=None) -> Submission`

Order inside both functions: (1) `env` (EnvError early); (2) upstream choice (`pick_upstream` / `pick_version`) and argument checks; (3) job list; (4) `check_proxy`; (5) version: `dry_run` → name from `next_version`, no folder; else `create_version`; (6) `submit_jobs`; (7) not dry-run → `storage.write_text(<version>/jobs.json, json.dumps(ids))`.

Job lists (`base = ["--config", str(cfg.user_file), "--phase-dir", version_dir]`):
- `phase_a`: `Job("skim", "skim", ("skim", *base))`, `Job("bdt", "bdt", ("bdt", *base), after=("skim",))`
- `phase_b` (upstream A): `Job(f"gnn_{s}", f"gnn_{s}", ("gnn", *base, "--upstream", A, "--sample", s))` for s in signal, background; `Job("eval", "eval", ("eval", *base, "--upstream", A), after=("gnn_signal", "gnn_background"))`
- `phase_c` (upstream B): `Job(f"limit_{cut}", "limit", ("limit", *base, "--upstream", B, "--cut", repr(cut)))` per cut in config order; `Job("summary", "summary", ("summary", *base, "--upstream", B), after=<all limit names>, dependency="afterany")`
- stages (version `debug/<stage>_vsN`, parent `storage.join(cfg.output_dir, "debug")`, no manifest):
  - `skim`: `("skim", *base)`, resources `skim`
  - `bdt` (`A = pick_version(cfg, "phase_a", from_)`): `("bdt", *base, "--source", A, "--no-manifest")`
  - `gnn` (needs `sample` in signal/background, else ValueError; `A = pick_version(cfg, "phase_a", from_)`): `("gnn", *base, "--upstream", A, "--sample", sample)`, resources `gnn_<sample>`
  - `eval` (`B = pick_version(cfg, "phase_b", from_)`; `A = storage.join(cfg.output_dir, read_manifest(B)["upstream"])`): `("eval", *base, "--upstream", A, "--source", B, "--no-manifest")`
  - `limit` (needs `cut` in `cfg.limit.gnn_score_cut`, else ValueError naming the cut; `B = pick_version(cfg, "phase_b", from_)`): `("limit", *base, "--upstream", B, "--cut", repr(cut))`
  - `summary` (`C = pick_version(cfg, "phase_c", from_, complete=False)`): `("summary", *base, "--source", C, "--no-manifest")`
  - The stage job's name is the stage name, except `gnn` → `gnn_<sample>` and `limit` → `limit_<cut>`.
- Log folder: `Path(cfg.paths.log_dir) / cfg.analysis_name / <version name>`; debug: `... / "debug" / <stage>_vsN`.

- [ ] **Step 1: Write the failing tests** — `tests/pipeline/test_driver.py`

```python
import json
import os
from pathlib import Path

import pytest

from test_slurm import ENV, fake_sbatch  # noqa: F401  (fixture reused)


@pytest.fixture
def env():
    from tau3mu_gnn.pipeline.slurm import JobEnv

    return JobEnv.from_environment(ENV)


@pytest.fixture
def cfg(load, user, tmp_path):
    user["paths"]["output_base"] = str(tmp_path / "out")
    user["paths"]["log_dir"] = str(tmp_path / "logs")
    user["limit"]["gnn_score_cut"] = [2.0, 3.9, 4.5]
    return load(user)


def complete(cfg, name, upstream=None):
    d = Path(cfg.output_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"phase": name.rsplit("_vs", 1)[0], "upstream": upstream}))
    return str(d)


def sbatch_calls(log):
    return log.read_text().splitlines()


def test_phase_a(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_phase

    s = submit_phase(cfg, "phase_a", env=env)
    assert s.version_dir == f"{cfg.output_dir}/phase_a_vs1" and s.upstream is None
    assert s.job_ids == {"skim": "1001", "bdt": "1002"}
    assert Path(s.log_folder) == Path(cfg.paths.log_dir) / "toy_run" / "phase_a_vs1"
    calls = sbatch_calls(fake_sbatch)
    assert calls[1].startswith("--parsable --dependency=afterok:1001 ")
    assert json.loads((Path(s.version_dir) / "jobs.json").read_text()) == {"skim": "1001", "bdt": "1002"}
    assert (Path(s.version_dir) / "config_used.yaml").is_file()
    script = (s.log_folder / "bdt.sh").read_text().splitlines()[-1]
    assert f"bdt --config {cfg.user_file} --phase-dir {s.version_dir}" in script


def test_phase_b_uses_latest_complete_a(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_phase

    a1 = complete(cfg, "phase_a_vs1")
    Path(cfg.output_dir, "phase_a_vs2").mkdir()                       # failed run, no manifest
    s = submit_phase(cfg, "phase_b", env=env)
    assert s.upstream == a1 and s.version_dir.endswith("phase_b_vs1")
    calls = sbatch_calls(fake_sbatch)
    assert len(calls) == 3 and calls[2].startswith("--parsable --dependency=afterok:1001:1002 ")
    gnn_bkg = (s.log_folder / "gnn_background.sh").read_text().splitlines()[-1]
    assert gnn_bkg.endswith(f"--upstream {a1} --sample background")


def test_phase_c_one_job_per_cut_afterany(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_phase

    b1 = complete(cfg, "phase_b_vs1", upstream="phase_a_vs1")
    s = submit_phase(cfg, "phase_c", env=env)
    assert list(s.job_ids) == ["limit_2.0", "limit_3.9", "limit_4.5", "summary"]
    calls = sbatch_calls(fake_sbatch)
    assert [c.startswith("--parsable /") for c in calls[:3]] == [True] * 3       # limits: no dependency
    assert calls[3].startswith("--parsable --dependency=afterany:1001:1002:1003 ")
    assert (s.log_folder / "limit_3.9.sh").read_text().splitlines()[-1].endswith(f"--upstream {b1} --cut 3.9")


def test_phase_without_upstream_fails_before_anything(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_phase
    from tau3mu_gnn.pipeline.manifest import UpstreamError

    with pytest.raises(UpstreamError, match="phase-a"):
        submit_phase(cfg, "phase_b", env=env)
    assert not fake_sbatch.exists() and not Path(cfg.paths.log_dir).exists()


def test_dry_run_makes_no_version(cfg, env, fake_sbatch, capsys):
    from tau3mu_gnn.pipeline.driver import submit_phase

    s = submit_phase(cfg, "phase_a", dry_run=True, env=env)
    assert s.version_dir.endswith("phase_a_vs1") and s.job_ids == {"skim": "DRYRUN-1", "bdt": "DRYRUN-2"}
    assert not Path(cfg.output_dir).exists() and not fake_sbatch.exists()
    assert (s.log_folder / "skim.sh").is_file() and "sbatch --parsable" in capsys.readouterr().out


def test_no_conda_env_fails_first(cfg, fake_sbatch, monkeypatch):
    from tau3mu_gnn.pipeline.driver import submit_phase
    from tau3mu_gnn.pipeline.slurm import EnvError

    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    with pytest.raises(EnvError):
        submit_phase(cfg, "phase_a")
    assert not Path(cfg.output_dir).exists() and not Path(cfg.paths.log_dir).exists()


def test_stage_eval_debug(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_stage

    a1 = complete(cfg, "phase_a_vs1")
    b1 = complete(cfg, "phase_b_vs1", upstream="phase_a_vs1")
    s = submit_stage(cfg, "eval", env=env)
    assert s.version_dir == f"{cfg.output_dir}/debug/eval_vs1"
    assert Path(s.log_folder) == Path(cfg.paths.log_dir) / "toy_run" / "debug" / "eval_vs1"
    last = (s.log_folder / "eval.sh").read_text().splitlines()[-1]
    assert last.endswith(f"--upstream {a1} --source {b1} --no-manifest")


def test_stage_limit_cut_must_be_in_config(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_stage

    complete(cfg, "phase_b_vs1", upstream="phase_a_vs1")
    with pytest.raises(ValueError, match="5.0"):
        submit_stage(cfg, "limit", cut=5.0, env=env)
    assert not fake_sbatch.exists()
    s = submit_stage(cfg, "limit", cut=3.9, env=env)
    assert list(s.job_ids) == ["limit_3.9"]


def test_stage_gnn_needs_sample(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_stage

    complete(cfg, "phase_a_vs1")
    with pytest.raises(ValueError, match="sample"):
        submit_stage(cfg, "gnn", env=env)
    s = submit_stage(cfg, "gnn", sample="signal", env=env)
    assert list(s.job_ids) == ["gnn_signal"]


def test_stage_summary_uses_incomplete_phase_c(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_stage

    Path(cfg.output_dir, "phase_c_vs1").mkdir(parents=True)          # summary failed: no manifest
    s = submit_stage(cfg, "summary", env=env)
    last = (s.log_folder / "summary.sh").read_text().splitlines()[-1]
    assert last.endswith(f"--source {cfg.output_dir}/phase_c_vs1 --no-manifest")
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: tau3mu_gnn.pipeline.driver`) — `$PYTEST/test_driver.py -q`

- [ ] **Step 3: Implement** `src/tau3mu_gnn/pipeline/driver.py`

```python
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
from tau3mu_gnn.pipeline.slurm import Job, JobEnv, check_proxy, submit_jobs

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")
STAGES = ("skim", "bdt", "gnn", "eval", "limit", "summary")


@dataclass(frozen=True)
class Submission:
    version_dir: str
    log_folder: Path
    job_ids: dict[str, str]
    upstream: str | None


def _base(cfg: Config, version_dir: str) -> tuple[str, ...]:
    return ("--config", str(cfg.user_file), "--phase-dir", version_dir)


def _phase_jobs(cfg: Config, phase: str, version_dir: str, up: str | None) -> list[Job]:
    b = _base(cfg, version_dir)
    if phase == "phase_a":
        return [Job("skim", "skim", ("skim", *b)), Job("bdt", "bdt", ("bdt", *b), after=("skim",))]
    if phase == "phase_b":
        gnn = [Job(f"gnn_{s}", f"gnn_{s}", ("gnn", *b, "--upstream", up, "--sample", s)) for s in SAMPLES]
        return gnn + [Job("eval", "eval", ("eval", *b, "--upstream", up), after=tuple(j.name for j in gnn))]
    limits = [Job(f"limit_{c}", "limit", ("limit", *b, "--upstream", up, "--cut", repr(c)))
              for c in cfg.limit.gnn_score_cut]
    return limits + [Job("summary", "summary", ("summary", *b, "--upstream", up),
                         after=tuple(j.name for j in limits), dependency="afterany")]


def _submit(cfg, jobs_for, prefix, parent, log_parts, upstream, dry_run, env) -> Submission:
    """jobs_for(version_dir) -> job list. Shared tail of submit_phase / submit_stage (steps 3-7)."""
    probe = jobs_for("<version>")
    check_proxy(cfg, probe)
    name = runs.next_version(parent, prefix)
    version_dir = storage.join(parent, name) if dry_run else runs.create_version(cfg, prefix, parent=parent)
    name = version_dir.rsplit("/", 1)[-1]
    log_folder = Path(cfg.paths.log_dir, cfg.analysis_name, *log_parts, name)
    ids = submit_jobs(cfg, jobs_for(version_dir), version_dir, log_folder, env, dry_run=dry_run)
    if not dry_run:
        storage.write_text(storage.join(version_dir, JOBS_FILE), json.dumps(ids, indent=2) + "\n")
    log.info("%s: %s, logs %s, jobs %s", name, version_dir, log_folder, ids)
    return Submission(version_dir, log_folder, ids, upstream)


def submit_phase(cfg: Config, phase: str, from_: str | None = None, dry_run: bool = False,
                 env: JobEnv | None = None) -> Submission:
    env = env if env is not None else JobEnv.from_environment()
    up = runs.pick_upstream(cfg, phase, from_) if phase != "phase_a" else None
    if phase == "phase_a" and from_ is not None:
        raise ValueError("phase-a has no upstream; --from is not allowed")
    return _submit(cfg, lambda v: _phase_jobs(cfg, phase, v, up), phase, cfg.output_dir, (), up, dry_run, env)


def submit_stage(cfg: Config, stage: str, from_: str | None = None, cut: float | None = None,
                 sample: str | None = None, dry_run: bool = False, env: JobEnv | None = None) -> Submission:
    env = env if env is not None else JobEnv.from_environment()
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
```

Note: `stage eval` args order must match the test: `--upstream A --source B --no-manifest` (the code above adds `--upstream` before `--source`). `stage limit`: `--upstream B --cut 3.9`.

- [ ] **Step 4: Run `$PYTEST/test_driver.py -q`, expect PASS**

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/driver.py tests/pipeline/test_driver.py
/usr/bin/git commit -m "feat(pipeline): submit phases and single stages (step 9b)"
```

---

### Task 5: `cli.py`

**Files:**
- Create: `src/tau3mu_gnn/pipeline/cli.py`
- Test: `tests/pipeline/test_cli.py`

**Interfaces:**
- Consumes: `load_config`, `ConfigError`, `driver.submit_phase`, `driver.submit_stage`, `driver.STAGES`, `UpstreamError`, `SubmitError`.
- Produces: `main(argv=None) -> int` (0 ok, 1 known error); `python -m tau3mu_gnn.pipeline.cli`.

- [ ] **Step 1: Write the failing tests** — `tests/pipeline/test_cli.py`

```python
import pytest

from test_slurm import ENV, fake_sbatch  # noqa: F401


@pytest.fixture
def cli_env(load, user, tmp_path, monkeypatch):
    from tau3mu_gnn.pipeline import cli

    user["paths"]["output_base"] = str(tmp_path / "out")
    user["paths"]["log_dir"] = str(tmp_path / "logs")
    cfg = load(user)
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)       # toy config uses the fake repo defaults
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    return cfg


def test_phase_a_prints_summary(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipeline import cli

    assert cli.main(["phase-a", "--config", "my.yaml"]) == 0
    out = capsys.readouterr().out
    assert "phase_a_vs1" in out and "skim" in out and "1001" in out and "logs" in out


def test_dry_run(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipeline import cli

    assert cli.main(["phase-a", "--config", "my.yaml", "--dry-run"]) == 0
    assert "DRYRUN-1" in capsys.readouterr().out and not fake_sbatch.exists()


def test_known_error_exit_1(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipeline import cli

    assert cli.main(["phase-b", "--config", "my.yaml"]) == 1
    err = capsys.readouterr().err
    assert "phase-a" in err and "Traceback" not in err


def test_stage_args(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipeline import cli

    assert cli.main(["stage", "limit", "--config", "my.yaml", "--cut", "7"]) == 1
    assert "not in limit.gnn_score_cut" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["stage", "nope", "--config", "my.yaml"])


def test_config_error_exit_1(tmp_path, capsys):
    from tau3mu_gnn.pipeline import cli

    assert cli.main(["phase-a", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert "cannot read config file" in capsys.readouterr().err
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError`) — `$PYTEST/test_cli.py -q`

- [ ] **Step 3: Implement** `src/tau3mu_gnn/pipeline/cli.py`

```python
"""tau3mu pipeline driver: submit the SLURM jobs of one phase or one stage.

  python -m tau3mu_gnn.pipeline.cli phase-a --config my.yaml [--dry-run]
  python -m tau3mu_gnn.pipeline.cli phase-b --config my.yaml [--from phase_a_vsN] [--dry-run]
  python -m tau3mu_gnn.pipeline.cli phase-c --config my.yaml [--from phase_b_vsN] [--dry-run]
  python -m tau3mu_gnn.pipeline.cli stage <skim|bdt|gnn|eval|limit|summary> --config my.yaml
        [--from <phase>_vsN] [--cut X] [--sample signal|background] [--dry-run]

Run it in the activated conda env (conda activate Tau3Mu); the jobs use the same env.
"""
from __future__ import annotations

import argparse
import logging
import sys

from tau3mu_gnn.pipeline.config import ConfigError, load_config
from tau3mu_gnn.pipeline.driver import STAGES, submit_phase, submit_stage
from tau3mu_gnn.pipeline.manifest import UpstreamError
from tau3mu_gnn.pipeline.slurm import SubmitError


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tau3mu_gnn.pipeline.cli", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("phase-a", "phase-b", "phase-c", "stage"):
        s = sub.add_parser(name)
        if name == "stage":
            s.add_argument("stage", choices=STAGES)
            s.add_argument("--cut", type=float, help="GNN score cut (stage limit)")
            s.add_argument("--sample", choices=("signal", "background"), help="sample (stage gnn)")
        s.add_argument("--config", required=True, help="your user config file")
        if name != "phase-a":
            s.add_argument("--from", dest="from_", metavar="VERSION", help="upstream version, e.g. phase_a_vs2")
        s.add_argument("--dry-run", action="store_true", help="write the job scripts and print the sbatch "
                                                              "commands; submit nothing")
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_config(a.config)
        if a.command == "stage":
            s = submit_stage(cfg, a.stage, a.from_, a.cut, a.sample, dry_run=a.dry_run)
        else:
            s = submit_phase(cfg, a.command.replace("-", "_"), getattr(a, "from_", None), dry_run=a.dry_run)
    except (ConfigError, UpstreamError, SubmitError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(("DRY RUN, nothing submitted\n" if a.dry_run else "") +
          f"results: {s.version_dir}\n" +
          (f"upstream: {s.upstream}\n" if s.upstream else "") +
          f"logs:    {s.log_folder}\n" +
          "\n".join(f"  {name:<16} {job_id}" for name, job_id in s.job_ids.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full suite (from an activated Tau3Mu shell) `$PYTEST -q`, expect PASS**

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/cli.py tests/pipeline/test_cli.py
/usr/bin/git commit -m "feat(pipeline): driver CLI phase-a/b/c and stage (step 9b)"
```

---

### Task 6: Roadmap + small-input check files (not committed)

**Files:**
- Modify: `docs/wrapper_roadmap.md`
- Create (not committed): `/work/kfuss/SVJ_GNN/.claude/worktrees/pipeline-wrapper/regression/step9b/make_small_inputs.py`, `make_small_inputs.sh`, `small_W.yaml`, `README.md`

- [ ] **Step 1: Roadmap**
  - Step 9b line: prefix ✅ and append `Done: slurm.py (job scripts, sbatch --parsable + dependencies, --dry-run, proxy check), driver.py, cli.py (python -m tau3mu_gnn.pipeline.cli). Jobs activate the conda env of the submitting shell (E1). Real check: small-input run (pending, Kaito queues).`
  - Decision log row (2026-10-12): `Step 9b decisions (Kaito): E1 the job script activates the conda env of the submitting shell (CONDA_PREFIX, CONDA_EXE; error at submit if none). E2 small-input check: 20k signal / 1M background, cuts [2.0, 3.9]. Also: --dry-run makes no version folder; debug 'stage summary' may use an incomplete phase C version.`
  - Commit: `/usr/bin/git add docs/wrapper_roadmap.md docs/superpowers/plans/2026-10-12-step9b-slurm.md` then `/usr/bin/git commit -m "docs(roadmap): step 9b"`.

- [ ] **Step 2: Small-input files** (folder `/work/kfuss/SVJ_GNN/.claude/worktrees/pipeline-wrapper/regression/step9b/`, never committed)

`make_small_inputs.py`:

```python
"""First N events of Kaito's W pre-skim files -> /work/kfuss/tau3mu_small_inputs/ (step 9b check)."""
import sys

import ROOT

SRC = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data"
OUT = "/work/kfuss/tau3mu_small_inputs"
for sample, path, n in [("signal", f"{SRC}/signal_W/old_data/signal_skimmed_no_resonances.root", 20_000),
                        ("background", f"{SRC}/background_W/old_data/data_skimmed_no_resonances.root", 1_000_000)]:
    df = ROOT.RDataFrame("Events", path).Range(n)
    df.Snapshot("Events", f"{OUT}/{sample}_small.root")
    print(sample, ROOT.RDataFrame("Events", f"{OUT}/{sample}_small.root").Count().GetValue(), file=sys.stderr)
```

`make_small_inputs.sh` (SLURM, standard, 16G, 2 h, logs to `/work/kfuss/SVJ_GNN/logs`, activates `Tau3Mu`, `mkdir -p /work/kfuss/tau3mu_small_inputs`, runs the script).

`small_W.yaml`:

```yaml
analysis_name: small_W_9b
channel: W
paths:
  output_base: root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/$USER/tau3mu_pipeline_test
  log_dir: /work/$USER/tau3mu_logs
inputs:
  signal: [/work/kfuss/tau3mu_small_inputs/signal_small.root]
  background: [/work/kfuss/tau3mu_small_inputs/background_small.root]
  already_resonance_skimmed: true
branch_map: {}
limit:
  signal_weight: 0.000019
  gnn_score_cut: [2.0, 3.9]
```

`README.md`: the exact commands for Kaito, in order:
1. `sbatch make_small_inputs.sh` (wait).
2. `conda activate Tau3Mu`; `cd /work/kfuss/SVJ_GNN/.claude/worktrees/step9b-slurm`; `export PYTHONPATH=$PWD/src` (use this branch's code); proxy `voms-proxy-init --voms cms --valid 192:00`.
3. `python -m tau3mu_gnn.pipeline.cli phase-a --config <...>/small_W.yaml --dry-run`, read the scripts, then without `--dry-run`.
4. After phase A: `phase-b`, then `phase-c` (same pattern).
5. Pass criteria (spec D9): all jobs COMPLETED (`sacct -j <ids>`); `phase_a_vs1`, `phase_b_vs1`, `phase_c_vs1` with `manifest.json`; `limits.json` in `phase_c_vs1`; `logs/` copies in each version; optional: a second `phase-b` gives `phase_b_vs2` and `phase-c` then reports it as upstream.

- [ ] **Step 3:** Run `$PYTEST -q` (activated shell), expect PASS. Do **not** submit anything.
