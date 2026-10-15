# Mini Step 9a — Driver Core (no SLURM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Everything the SLURM layer (9b) needs, without SLURM: config keys `paths.log_dir` + frozen `slurm:` section, a storage helper (dCache read via `/pnfs`, write via `xrdcp`; or local), phase versions `<phase>_vsN`, manifests with upstream checks, and one Python function per job (the body a SLURM job will run), with a `python -m tau3mu_gnn.pipeline.jobs` entry point.

**Architecture:**
- `pipeline/storage.py`: path helpers for `root://host:port//pnfs/...` and local paths. Reads go through the local / `/pnfs` path; writes go through `xrdfs` / `xrdcp` (remote) or `shutil` (local).
- `pipeline/manifest.py`: config snapshot, per-phase settings, `write_manifest`, `read_manifest`, `check_upstream` (missing manifest / files → `UpstreamError`; changed settings → warning).
- `pipeline/runs.py`: versions `<prefix>_vsN`, `create_version` (folder + `config_used.yaml`), `pick_upstream` (highest complete, or `--from`).
- `pipeline/jobs.py`: job bodies `skim`, `bdt`, `gnn`, `evaluation`, `limit`, `summary`; each works in a fresh scratch folder under `$TMPDIR` and copies its outputs into the phase folder; the last job of a phase writes the manifest. `main(argv)` = the CLI the 9b job scripts call.
- Small stage changes: stage 3 `samples` argument (one sample per job) + `output_files` helper; `summarize(..., write_dir=)` (read `cut_*/` from dCache, write the table to scratch).

**Tech Stack:** Python 3.11 (Tau3Mu env), PyYAML, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-10-12-step9-driver-design.md` (this plan = 9a: sec. 3–6, 8, 9, 11 "9a"; sec. 7 and the CLI are 9b).

## Global Constraints

- Test command: `$PYTEST` = `PYTHONPATH=/work/kfuss/SVJ_GNN/.claude/worktrees/step9-driver/src /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m pytest tests/pipeline`. `$PYTEST/x.py` = same with `tests/pipeline/x.py`.
- Branch `pipeline-step9a-driver-core`, from `pipeline-step8b-cut-scan` (PR #19). Worktree `/work/kfuss/SVJ_GNN/.claude/worktrees/step9-driver`. The PR is stacked on #19.
- Git in this worktree: `/usr/bin/git` only. One git command per Bash call.
- **No SLURM in 9a**: nothing calls `sbatch`; no phase runs on the login node with real data (spec D3). Tests use toy files and a local `output_base` (`tmp_path`).
- No new dependency (roadmap). Config rules stay: missing required key → error; unknown key → error; a frozen key in the user file → error.
- Phase names: `phase_a`, `phase_b`, `phase_c`. Version folder: `<phase>_vs<N>`, N ≥ 1, highest N + 1 (numeric, `_vs10` > `_vs9`).
- Stage folders: `0_resonance_skim`, `1_skim`, `2_bdt`, `3_gnn`, `4_eval`; phase C writes `cut_<cut>/` and `limits.{json,txt,png}` directly into the phase folder (spec sec. 3).
- Manifest file `manifest.json`; config snapshot `config_used.yaml`; job-ID file `jobs.json` (written by 9b; `write_manifest` copies it in if present).
- Phase settings recorded (spec sec. 5): `phase_a`: `channel, inputs, branch_map, skim, bdt`; `phase_b`: + `gnn`; `phase_c`: + `limit`.
- Settings difference → **warning** only. Missing manifest / output file → `UpstreamError`.
- SLURM job keys (spec sec. 8): `skim, bdt, gnn_signal, gnn_background, eval, limit, summary`; each `{partition, mem, time, cpus}`; plus `slurm.account`. Kaito's values (2026-10-12):
  `skim {standard, 128G, "0-03:00:00", 4}`, `bdt {standard, 64G, "0-06:00:00", 1}`, `gnn_signal {long, 64G, "1-00:00:00", 1}`, `gnn_background {long, 64G, "2-00:00:00", 1}`, `eval {standard, 32G, "0-01:00:00", 1}`, `limit {standard, 128G, "0-01:30:00", 1}`, `summary {short, 4G, "0-00:30:00", 1}`, `account: t3`.
- `paths.log_dir`: required, `$USER` / `~` expanded, must not start with `root://` or `/pnfs`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Malformed dCache URL** (`root://host/pnfs/x` with one slash, or `root://host`) → `ValueError` naming the URL, not a silent wrong `/pnfs` path. Test in Task 2.
2. **A failed / running phase version** (folder without manifest) → `pick_upstream` skips it with a warning naming it; `--from` that version → `UpstreamError`. Test in Task 4.
3. **`--from` of the wrong phase** (`phase_b_vs1` given to phase B) or a manifest of the wrong phase → `UpstreamError`, not a run on wrong files. Test in Task 3 and Task 4.
4. **Combine fails in a limit job** → the job fails, but `cut_<cut>/` (datacard + Combine logs) is still copied to the phase folder for debugging. Test in Task 6.
5. **Unchanged config, re-loaded** (tuples vs JSON lists, absolute vs repo-relative model paths) → **no** stale warning. Test in Task 3.

## File Structure

| File | Responsibility |
|---|---|
| `src/tau3mu_gnn/pipeline/config.py` (modify) | `paths.log_dir`, `slurm.account`, `slurm.jobs`; `SlurmJob`, `SlurmConfig` |
| `configs/analysis_defaults_W.yaml`, `configs/example_W.yaml` (modify) | new keys |
| `src/tau3mu_gnn/pipeline/storage.py` (create) | `is_remote`, `local_path`, `join`, `makedirs`, `put`, `write_text`, `read_text`, `exists`, `listdir` |
| `src/tau3mu_gnn/pipeline/manifest.py` (create) | `UpstreamError`, `config_dict`, `phase_settings`, `settings_diff`, `write_manifest`, `read_manifest`, `check_upstream` |
| `src/tau3mu_gnn/pipeline/runs.py` (create) | `versions`, `next_version`, `is_complete`, `create_version`, `pick_upstream` |
| `src/tau3mu_gnn/pipeline/stages/gnn.py`, `limit.py` (modify) | `samples`, `output_files`; `summarize(write_dir=)` |
| `src/tau3mu_gnn/pipeline/jobs.py` (create) | job bodies + `main(argv)` |
| `tests/pipeline/conftest.py` (modify) | toy `slurm` + `log_dir` |
| `tests/pipeline/test_load_config.py`, `test_real_configs.py` (modify), `test_storage.py`, `test_manifest.py`, `test_runs.py`, `test_jobs.py` (create), `test_gnn_scores.py`, `test_cut_scan.py` (modify) | tests |
| `docs/wrapper_roadmap.md` (modify) | spec sec. 9 updates |

---

### Task 1: Config — `paths.log_dir` and the frozen `slurm:` section

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/config.py`
- Modify: `configs/analysis_defaults_W.yaml`, `configs/example_W.yaml`
- Modify: `tests/pipeline/conftest.py` (`TOY_DEFAULTS`, `TOY_USER`)
- Test: `tests/pipeline/test_load_config.py`, `tests/pipeline/test_real_configs.py`

**Interfaces:**
- Produces: `PathsConfig.log_dir: str`; `SlurmJob(partition: str, mem: str, time: str, cpus: int)`; `SlurmConfig(account: str, jobs: Mapping[str, SlurmJob])`; `Config.slurm: SlurmConfig`; `SLURM_JOBS = ("skim", "bdt", "gnn_signal", "gnn_background", "eval", "limit", "summary")`.

- [ ] **Step 1: Update the toy fixtures** in `tests/pipeline/conftest.py`

Add to `TOY_DEFAULTS` (after `"limit"`):

```python
    "slurm": {
        "account": "t3",
        "jobs": {name: {"partition": "standard", "mem": "4G", "time": "0-01:00:00", "cpus": 1}
                 for name in ("skim", "bdt", "gnn_signal", "gnn_background", "eval", "limit", "summary")},
    },
```

In `TOY_USER`, change `"paths"` to `{"output_base": "/tmp/toy_out", "log_dir": "/tmp/toy_logs"}`.

- [ ] **Step 2: Write the failing tests** (append to `tests/pipeline/test_load_config.py`)

```python
def test_log_dir_and_slurm(load, user):
    cfg = load(user)
    assert cfg.paths.log_dir == "/tmp/toy_logs"
    assert cfg.slurm.account == "t3"
    assert set(cfg.slurm.jobs) == {"skim", "bdt", "gnn_signal", "gnn_background", "eval", "limit", "summary"}
    job = cfg.slurm.jobs["limit"]
    assert (job.partition, job.mem, job.time, job.cpus) == ("standard", "4G", "0-01:00:00", 1)


def test_log_dir_required(load, user):
    from tau3mu_gnn.pipeline.config import ConfigError

    del user["paths"]["log_dir"]
    with pytest.raises(ConfigError, match="paths.log_dir"):
        load(user)


@pytest.mark.parametrize("bad", ["root://t3dcachedb03.psi.ch:1094//pnfs/x", "/pnfs/psi.ch/x"])
def test_log_dir_must_be_local(load, user, bad):
    from tau3mu_gnn.pipeline.config import ConfigError

    user["paths"]["log_dir"] = bad
    with pytest.raises(ConfigError, match="log_dir"):
        load(user)


def test_slurm_in_user_file_is_frozen(load, user):
    from tau3mu_gnn.pipeline.config import ConfigError

    user["slurm"] = {"account": "t3"}
    with pytest.raises(ConfigError, match="frozen"):
        load(user)


def _defaults_with(fake_repo, change):
    import copy

    import yaml
    from conftest import TOY_DEFAULTS

    d = copy.deepcopy(TOY_DEFAULTS)
    change(d["slurm"]["jobs"])
    (fake_repo / "configs" / "analysis_defaults_W.yaml").write_text(yaml.safe_dump(d))


@pytest.mark.parametrize("change, match", [
    (lambda j: j.pop("eval"), "missing job 'eval'"),
    (lambda j: j.update(gnn_bkg=dict(j["bdt"])), "unknown job 'gnn_bkg'"),
    (lambda j: j["bdt"].update(memory="4G"), "unknown key 'memory'"),
    (lambda j: j["bdt"].pop("time"), "missing key 'time'"),
    (lambda j: j["bdt"].update(mem="64 GB"), "mem"),
    (lambda j: j["bdt"].update(time="7 days"), "time"),
    (lambda j: j["bdt"].update(cpus=0), "cpus"),
    (lambda j: j["bdt"].update(cpus=True), "cpus"),
])
def test_slurm_jobs_validation(load, user, fake_repo, change, match):
    from tau3mu_gnn.pipeline.config import ConfigError

    _defaults_with(fake_repo, change)
    with pytest.raises(ConfigError, match=match):
        load(user)
```

In `tests/pipeline/test_real_configs.py`, append to `test_example_w_matches_production_settings`:

```python
    assert cfg.paths.log_dir == "/work/tester/tau3mu_logs"
    jobs = cfg.slurm.jobs
    assert cfg.slurm.account == "t3"
    assert (jobs["skim"].partition, jobs["skim"].mem, jobs["skim"].time, jobs["skim"].cpus) == \
        ("standard", "128G", "0-03:00:00", 4)
    assert (jobs["gnn_background"].partition, jobs["gnn_background"].time) == ("long", "2-00:00:00")
    assert jobs["limit"].time == "0-01:30:00" and jobs["summary"].partition == "short"
```

- [ ] **Step 3: Run, expect FAIL** (`$PYTEST/test_load_config.py $PYTEST/test_real_configs.py -q`: unknown key `paths.log_dir` / `slurm.*`)

- [ ] **Step 4: Implement** in `src/tau3mu_gnn/pipeline/config.py`

After `_user_path_list`:

```python
def _local_path(v):
    v = _user_path(v)
    if v.startswith("root://") or v.startswith("/pnfs"):
        raise ValueError(f"must be a local disk path (SLURM cannot write logs to dCache), got {v!r}")
    return v


SLURM_JOBS = ("skim", "bdt", "gnn_signal", "gnn_background", "eval", "limit", "summary")
_SLURM_KEYS = ("partition", "mem", "time", "cpus")
_MEM_RE = re.compile(r"^\d+[KMGT]?$")
_TIME_RE = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")


def _slurm_jobs(v):
    if not isinstance(v, dict):
        raise ValueError(f"must be a mapping job -> {{{', '.join(_SLURM_KEYS)}}}, got {v!r}")
    reasons = [f"missing job '{j}'" for j in SLURM_JOBS if j not in v]
    reasons += [f"unknown job '{j}'" for j in v if j not in SLURM_JOBS]
    jobs = {}
    for name in SLURM_JOBS:
        spec = v.get(name)
        if spec is None:
            continue
        if not isinstance(spec, dict):
            reasons.append(f"job '{name}' must be a mapping, got {spec!r}")
            continue
        reasons += [f"job '{name}': missing key '{k}'" for k in _SLURM_KEYS if k not in spec]
        reasons += [f"job '{name}': unknown key '{k}'" for k in spec if k not in _SLURM_KEYS]
        if not set(_SLURM_KEYS) <= set(spec):
            continue
        partition, mem, time, cpus = (spec[k] for k in _SLURM_KEYS)
        if not isinstance(partition, str) or not partition.strip():
            reasons.append(f"job '{name}': partition must be a non-empty string, got {partition!r}")
        if not isinstance(mem, str) or not _MEM_RE.match(mem):
            reasons.append(f"job '{name}': mem must look like 64G, got {mem!r}")
        if not isinstance(time, str) or not _TIME_RE.match(time):
            reasons.append(f"job '{name}': time must look like 1-00:00:00 or 06:00:00, got {time!r}")
        if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
            reasons.append(f"job '{name}': cpus must be an integer >= 1, got {cpus!r}")
        jobs[name] = SlurmJob(partition, mem, time, cpus)
    if reasons:
        raise ValueError("; ".join(reasons))
    return MappingProxyType(jobs)
```

Field tables:
- `USER_FIELDS`: add `"paths.log_dir": _Field(_local_path),` after `"paths.output_base"`.
- `DEFAULTS_FIELDS`: add `"slurm.account": _Field(_non_empty_str),` and `"slurm.jobs": _Field(_slurm_jobs),`.
- `_LEAF_KEYS = {"branch_map", "slurm.jobs"}`.

Dataclasses (with the other config dataclasses; `_slurm_jobs` uses `SlurmJob` only at call time):

```python
@dataclass(frozen=True)
class PathsConfig:
    output_base: str
    log_dir: str


@dataclass(frozen=True)
class SlurmJob:
    partition: str
    mem: str
    time: str        # D-HH:MM:SS or HH:MM:SS
    cpus: int


@dataclass(frozen=True)
class SlurmConfig:
    account: str
    jobs: Mapping[str, SlurmJob]
```

`Config`: add field `slurm: SlurmConfig` after `limit`. In `load_config`: `paths=PathsConfig(output_base=user["paths.output_base"], log_dir=user["paths.log_dir"]),` and after `limit=...`: `slurm=SlurmConfig(account=defaults["slurm.account"], jobs=defaults["slurm.jobs"]),`.

`configs/analysis_defaults_W.yaml`, append:

```yaml

slurm:                         # SLURM resources per job (step 9). Times: first values (Kaito, 2026-10-12);
  account: t3                  # raise them here if jobs hit the limit.
  jobs:
    skim:           {partition: standard, mem: 128G, time: "0-03:00:00", cpus: 4}
    bdt:            {partition: standard, mem: 64G,  time: "0-06:00:00", cpus: 1}
    gnn_signal:     {partition: long,     mem: 64G,  time: "1-00:00:00", cpus: 1}
    gnn_background: {partition: long,     mem: 64G,  time: "2-00:00:00", cpus: 1}
    eval:           {partition: standard, mem: 32G,  time: "0-01:00:00", cpus: 1}
    limit:          {partition: standard, mem: 128G, time: "0-01:30:00", cpus: 1}
    summary:        {partition: short,    mem: 4G,   time: "0-00:30:00", cpus: 1}
```

`configs/example_W.yaml`, in `paths:` after `output_base`:

```yaml
  # Local disk folder for the job scripts and SLURM logs (SLURM cannot write to dCache).
  # Logs are also copied next to the results (<phase>_vsN/logs/).
  log_dir: /work/$USER/tau3mu_logs
```

- [ ] **Step 5: Run the full suite `$PYTEST -q`, expect PASS.** If another test builds `PathsConfig(...)` or `Config(...)` by hand (grep `PathsConfig(` / `Config(` in `tests/`), add `log_dir=` / `slurm=` there.

- [ ] **Step 6: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/config.py configs/analysis_defaults_W.yaml configs/example_W.yaml tests/pipeline/conftest.py tests/pipeline/test_load_config.py tests/pipeline/test_real_configs.py
/usr/bin/git commit -m "feat(config): paths.log_dir and frozen slurm section (step 9a)"
```

---

### Task 2: `storage.py`

**Files:**
- Create: `src/tau3mu_gnn/pipeline/storage.py`
- Test: `tests/pipeline/test_storage.py`

**Interfaces:**
- Produces (all paths `str | Path`; URLs stay `str`):
  - `is_remote(path) -> bool` (starts with `root://`)
  - `local_path(path) -> Path` (`root://h:p//pnfs/x` → `/pnfs/x`; local unchanged; malformed URL → `ValueError`)
  - `join(base, *parts) -> str` (string join, keeps `root://h//...`)
  - `makedirs(path) -> None`
  - `put(src, dest) -> None` (local file or folder tree → dest file or folder; overwrites)
  - `write_text(dest, text) -> None`, `read_text(path) -> str`, `exists(path) -> bool`, `listdir(path) -> list[str]` (sorted names; `[]` if missing)

- [ ] **Step 1: Write the failing tests** — `tests/pipeline/test_storage.py`

```python
import os
from pathlib import Path

import pytest

URL = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/user/x/run"


def test_local_path_maps_xrootd_to_pnfs():
    from tau3mu_gnn.pipeline import storage

    assert storage.is_remote(URL) and not storage.is_remote("/work/x")
    assert storage.local_path(URL) == Path("/pnfs/psi.ch/cms/user/x/run")
    assert storage.local_path("/work/x/y") == Path("/work/x/y")


@pytest.mark.parametrize("bad", ["root://host/pnfs/x", "root://host", "root://host//"])
def test_malformed_url_raises(bad):
    from tau3mu_gnn.pipeline import storage

    with pytest.raises(ValueError, match="root://"):
        storage.local_path(bad)


def test_join_keeps_url():
    from tau3mu_gnn.pipeline import storage

    assert storage.join(URL + "/", "phase_a_vs1", "/1_skim/") == URL + "/phase_a_vs1/1_skim"
    assert storage.join("/work/x", "a", "b.txt") == "/work/x/a/b.txt"


def test_local_put_copies_tree_and_overwrites(tmp_path):
    from tau3mu_gnn.pipeline import storage

    src = tmp_path / "scratch"
    (src / "1_skim").mkdir(parents=True)
    (src / "1_skim" / "signal.root").write_text("new")
    (src / "top.txt").write_text("t")
    dest = tmp_path / "out" / "phase_a_vs1"
    (dest / "1_skim").mkdir(parents=True)
    (dest / "1_skim" / "signal.root").write_text("old")

    storage.put(src, dest)

    assert (dest / "1_skim" / "signal.root").read_text() == "new"
    assert (dest / "top.txt").read_text() == "t"
    assert storage.listdir(dest) == ["1_skim", "top.txt"]
    assert storage.listdir(tmp_path / "nothing") == []


def test_local_write_read_exists(tmp_path):
    from tau3mu_gnn.pipeline import storage

    f = tmp_path / "a" / "b.json"
    storage.write_text(f, "{}")
    assert storage.exists(f) and storage.read_text(f) == "{}"


def test_put_missing_source_raises(tmp_path):
    from tau3mu_gnn.pipeline import storage

    with pytest.raises(FileNotFoundError):
        storage.put(tmp_path / "missing", tmp_path / "dest")


@pytest.fixture
def fake_xrootd(tmp_path, monkeypatch):
    """xrdfs / xrdcp on PATH that only log their arguments (one line per call)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "xrd.log"
    for name in ("xrdfs", "xrdcp"):
        exe = bin_dir / name
        exe.write_text(f'#!/bin/sh\necho "{name} $*" >> {log}\n')
        exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def test_remote_put_uses_xrdfs_and_xrdcp(tmp_path, fake_xrootd):
    from tau3mu_gnn.pipeline import storage

    src = tmp_path / "scratch"
    (src / "2_bdt").mkdir(parents=True)
    (src / "2_bdt" / "a.root").write_text("a")
    (src / "2_bdt" / "b.pkl").write_text("b")

    storage.put(src, URL + "/phase_a_vs1")

    calls = fake_xrootd.read_text().splitlines()
    host = "t3dcachedb03.psi.ch:1094"
    assert calls.count(f"xrdfs {host} mkdir -p /pnfs/psi.ch/cms/user/x/run/phase_a_vs1/2_bdt") == 1
    assert f"xrdcp --force --nopbar {src / '2_bdt' / 'a.root'} {URL}/phase_a_vs1/2_bdt/a.root" in calls
    assert f"xrdcp --force --nopbar {src / '2_bdt' / 'b.pkl'} {URL}/phase_a_vs1/2_bdt/b.pkl" in calls


def test_remote_makedirs_and_write_text(fake_xrootd):
    from tau3mu_gnn.pipeline import storage

    storage.makedirs(URL + "/phase_b_vs2")
    storage.write_text(URL + "/phase_b_vs2/manifest.json", "{}")
    calls = fake_xrootd.read_text().splitlines()
    assert calls[0] == "xrdfs t3dcachedb03.psi.ch:1094 mkdir -p /pnfs/psi.ch/cms/user/x/run/phase_b_vs2"
    assert calls[-1].startswith("xrdcp --force --nopbar ") and calls[-1].endswith(
        f" {URL}/phase_b_vs2/manifest.json")


def test_remote_command_failure_raises(tmp_path, monkeypatch):
    import subprocess

    from tau3mu_gnn.pipeline import storage

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "xrdfs").write_text("#!/bin/sh\nexit 52\n")
    (bin_dir / "xrdfs").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(subprocess.CalledProcessError):
        storage.makedirs(URL + "/x")
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: tau3mu_gnn.pipeline.storage`) — `$PYTEST/test_storage.py -q`

- [ ] **Step 3: Implement** `src/tau3mu_gnn/pipeline/storage.py`

```python
"""Run-folder storage: dCache (root://host[:port]//pnfs/...) or a local disk path.

dCache is read through its /pnfs NFS mount and written with xrdfs / xrdcp (needs a valid VOMS proxy).
A local path uses plain file operations. Stages never call this; the job bodies (jobs.py) do.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_URL_RE = re.compile(r"^root://([^/]+)/(/[^/].*)$")


def is_remote(path: str | Path) -> bool:
    return str(path).startswith("root://")


def _split(url: str) -> tuple[str, str]:
    m = _URL_RE.match(url)
    if not m:
        raise ValueError(f"not an xrootd URL of the form root://host[:port]//path: {url!r}")
    return m.group(1), m.group(2).rstrip("/") or "/"


def local_path(path: str | Path) -> Path:
    """Path to read from: the /pnfs path of a root:// URL, else the path itself."""
    path = str(path)
    return Path(_split(path)[1]) if is_remote(path) else Path(path)


def join(base: str | Path, *parts: str | Path) -> str:
    """String join, so root://host//... URLs stay intact."""
    pieces = [str(base).rstrip("/")] + [str(p).strip("/") for p in parts if str(p).strip("/")]
    return "/".join(pieces)


def makedirs(path: str | Path) -> None:
    path = str(path)
    if is_remote(path):
        host, p = _split(path)
        _run(["xrdfs", host, "mkdir", "-p", p])
    else:
        Path(path).mkdir(parents=True, exist_ok=True)


def put(src: str | Path, dest: str | Path) -> None:
    """Copy a local file (to the file path dest) or a folder tree (into the folder dest). Overwrites."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"nothing to copy: {src}")
    if src.is_file():
        pairs = [(src, str(dest))]
    else:
        pairs = [(f, join(dest, f.relative_to(src).as_posix())) for f in sorted(src.rglob("*")) if f.is_file()]
    if is_remote(dest):
        host = _split(str(dest))[0]
        for folder in sorted({str(Path(_split(t)[1]).parent) for _, t in pairs}):
            _run(["xrdfs", host, "mkdir", "-p", folder])
        for f, target in pairs:
            _run(["xrdcp", "--force", "--nopbar", str(f), target])
    else:
        for f, target in pairs:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)


def write_text(dest: str | Path, text: str) -> None:
    if is_remote(dest):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / Path(_split(str(dest))[1]).name
            f.write_text(text)
            put(f, dest)
    else:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text(text)


def read_text(path: str | Path) -> str:
    return local_path(path).read_text()


def exists(path: str | Path) -> bool:
    return local_path(path).exists()


def listdir(path: str | Path) -> list[str]:
    p = local_path(path)
    return sorted(x.name for x in p.iterdir()) if p.is_dir() else []


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)
```

Note on `_URL_RE`: `(/[^/].*)` requires the path part to start with exactly one `/` after the host's `/`, i.e. `root://host//pnfs/...`. `root://host//` gives no match → `ValueError` (Review Focus 1).

- [ ] **Step 4: Run `$PYTEST/test_storage.py -q`, expect PASS**

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/storage.py tests/pipeline/test_storage.py
/usr/bin/git commit -m "feat(pipeline): storage helper, dCache read via /pnfs, write via xrdcp (step 9a)"
```

---

### Task 3: `manifest.py`

**Files:**
- Create: `src/tau3mu_gnn/pipeline/manifest.py`
- Test: `tests/pipeline/test_manifest.py`

**Interfaces:**
- Consumes: `storage.join/exists/read_text/write_text` (Task 2); `Config` with `slurm`, `paths.log_dir` (Task 1).
- Produces:
  - `MANIFEST = "manifest.json"`, `JOBS_FILE = "jobs.json"`, `PHASES = ("phase_a", "phase_b", "phase_c")`, `UPSTREAM = {"phase_a": None, "phase_b": "phase_a", "phase_c": "phase_b"}`
  - `class UpstreamError(Exception)`
  - `config_dict(cfg) -> dict` (JSON/YAML-safe; repo files relative to the repo root)
  - `phase_settings(cfg, phase) -> dict`
  - `settings_diff(old: dict, new: dict) -> list[str]` (lines `"<dotted.key>: <old> -> <new>"`, sorted)
  - `write_manifest(cfg, phase_dir, phase, upstream, outputs, missing=()) -> dict` (`upstream`: version name or `None`; `outputs`: key → path relative to `phase_dir`)
  - `read_manifest(phase_dir) -> dict`
  - `check_upstream(cfg, upstream_dir, phase) -> dict` (`phase` = the phase that is starting)

- [ ] **Step 1: Write the failing tests** — `tests/pipeline/test_manifest.py`

```python
import json

import pytest


def test_config_dict_is_plain_and_repo_relative(load, user):
    import yaml
    from tau3mu_gnn.pipeline.manifest import config_dict

    d = config_dict(load(user))
    assert d["bdt"]["classifier"] == "models/W/bdt/toy.pck"
    assert d["gnn"]["model_dir"] == "models/W/gnn"
    assert d["limit"]["gnn_score_cut"] == [3.9] and d["limit"]["fit_range"] == [1.62, 1.98]
    assert d["slurm"]["jobs"]["limit"] == {"partition": "standard", "mem": "4G", "time": "0-01:00:00", "cpus": 1}
    assert d["paths"] == {"output_base": "/tmp/toy_out", "log_dir": "/tmp/toy_logs"}
    assert yaml.safe_load(yaml.safe_dump(d)) == d and json.loads(json.dumps(d)) == d


def test_phase_settings_sections(load, user):
    from tau3mu_gnn.pipeline.manifest import phase_settings

    cfg = load(user)
    assert list(phase_settings(cfg, "phase_a")) == ["channel", "inputs", "branch_map", "skim", "bdt"]
    assert list(phase_settings(cfg, "phase_b"))[-1] == "gnn"
    assert list(phase_settings(cfg, "phase_c"))[-2:] == ["gnn", "limit"]


def test_settings_diff_dotted():
    from tau3mu_gnn.pipeline.manifest import settings_diff

    old = {"skim": {"dz_max": 0.5}, "inputs": {"signal": ["a.root"]}, "branch_map": {}}
    new = {"skim": {"dz_max": 0.4}, "inputs": {"signal": ["a.root"]}, "branch_map": {"x": "y"}}
    assert settings_diff(old, new) == ["branch_map.x: (not set) -> 'y'", "skim.dz_max: 0.5 -> 0.4"]
    assert settings_diff(old, old) == []


def test_write_and_read_manifest(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import read_manifest, write_manifest

    cfg = load(user)
    phase_dir = tmp_path / "toy_run" / "phase_b_vs2"
    (phase_dir).mkdir(parents=True)
    (phase_dir / "jobs.json").write_text(json.dumps({"eval": "125"}))
    data = write_manifest(cfg, phase_dir, "phase_b", "phase_a_vs1", {"gnn_signal": "3_gnn/s.root"})
    assert read_manifest(phase_dir) == data
    assert data["phase"] == "phase_b" and data["version"] == "phase_b_vs2" and data["upstream"] == "phase_a_vs1"
    assert data["outputs"] == {"gnn_signal": "3_gnn/s.root"} and data["missing"] == []
    assert data["slurm_jobs"] == {"eval": "125"}
    assert set(data["settings"]) == {"channel", "inputs", "branch_map", "skim", "bdt", "gnn"}
    assert data["git_commit"] and data["created"]


def test_read_manifest_missing(tmp_path):
    from tau3mu_gnn.pipeline.manifest import UpstreamError, read_manifest

    with pytest.raises(UpstreamError, match="phase_a_vs3"):
        read_manifest(tmp_path / "phase_a_vs3")


def upstream_a(cfg, tmp_path, files=("1_skim/signal.root",)):
    from tau3mu_gnn.pipeline.manifest import write_manifest

    d = tmp_path / "toy_run" / "phase_a_vs1"
    for rel in files:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("x")
    write_manifest(cfg, d, "phase_a", None, {"skim_signal": "1_skim/signal.root",
                                              "bdt_signal": "2_bdt/signal_with_bdt_scores.root"})
    return d


def test_check_upstream_missing_files(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import UpstreamError, check_upstream

    cfg = load(user)
    d = upstream_a(cfg, tmp_path)                      # 2_bdt file not written
    with pytest.raises(UpstreamError, match="2_bdt/signal_with_bdt_scores.root"):
        check_upstream(cfg, d, "phase_b")


def test_check_upstream_wrong_phase(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import UpstreamError, check_upstream

    cfg = load(user)
    d = upstream_a(cfg, tmp_path, files=("1_skim/signal.root", "2_bdt/signal_with_bdt_scores.root"))
    with pytest.raises(UpstreamError, match="phase_b"):
        check_upstream(cfg, d, "phase_c")              # phase C needs a phase_b folder


def test_check_upstream_unchanged_config_no_warning(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipeline.manifest import check_upstream

    files = ("1_skim/signal.root", "2_bdt/signal_with_bdt_scores.root")
    d = upstream_a(load(user), tmp_path, files=files)
    m = check_upstream(load(user), d, "phase_b")       # re-loaded config: tuples, absolute paths
    assert m["version"] == "phase_a_vs1"
    assert "other settings" not in caplog.text


def test_check_upstream_changed_settings_warns(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipeline.manifest import check_upstream

    files = ("1_skim/signal.root", "2_bdt/signal_with_bdt_scores.root")
    d = upstream_a(load(user), tmp_path, files=files)
    user["inputs"]["signal"] = ["/data/other_sig.root"]
    check_upstream(load(user), d, "phase_b")
    assert "other settings" in caplog.text and "inputs.signal" in caplog.text
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError`) — `$PYTEST/test_manifest.py -q`

- [ ] **Step 3: Implement** `src/tau3mu_gnn/pipeline/manifest.py`

```python
"""Phase manifests: what a phase wrote, from which upstream version, with which settings.

<phase>_vsN/manifest.json is written by the last job of a phase, only after all its jobs succeeded.
The first job(s) of the next phase run check_upstream on it.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from tau3mu_gnn.pipeline import storage
from tau3mu_gnn.pipeline.config import REPO_ROOT, Config

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
JOBS_FILE = "jobs.json"
PHASES = ("phase_a", "phase_b", "phase_c")
UPSTREAM = {"phase_a": None, "phase_b": "phase_a", "phase_c": "phase_b"}
PHASE_SECTIONS = {
    "phase_a": ("channel", "inputs", "branch_map", "skim", "bdt"),
    "phase_b": ("channel", "inputs", "branch_map", "skim", "bdt", "gnn"),
    "phase_c": ("channel", "inputs", "branch_map", "skim", "bdt", "gnn", "limit"),
}


class UpstreamError(Exception):
    pass


def config_dict(cfg: Config) -> dict:
    """The loaded config (user file + defaults) as plain data. Repo files relative to the repo root."""
    repo = cfg.defaults_file.parent.parent

    def rel(p):
        try:
            return Path(p).relative_to(repo).as_posix()
        except ValueError:
            return str(p)

    g, lim, inp = cfg.gnn, cfg.limit, cfg.inputs
    return {
        "analysis_name": cfg.analysis_name,
        "channel": cfg.channel,
        "paths": {"output_base": cfg.paths.output_base, "log_dir": cfg.paths.log_dir},
        "inputs": {"signal": list(inp.signal), "background": list(inp.background), "tree_name": inp.tree_name,
                   "already_resonance_skimmed": inp.already_resonance_skimmed},
        "branch_map": dict(cfg.branch_map),
        "skim": {"dz_max": cfg.skim.dz_max, "cos_alpha_2d_min": cfg.skim.cos_alpha_2d_min},
        "bdt": {"classifier": rel(cfg.bdt.classifier)},
        "gnn": {"model_dir": rel(g.model_dir), "model_name": g.model_name, "conv_type": g.conv_type,
                "gnn_mode": g.gnn_mode, "unsculpting_mode": g.unsculpting_mode, "graph_type": g.graph_type,
                "bdt_preselection_cut": g.bdt_preselection_cut},
        "limit": {"signal_weight": lim.signal_weight, "gnn_score_cut": list(lim.gnn_score_cut),
                  "tau_mass": lim.tau_mass, "mass_window": lim.mass_window, "fit_range": list(lim.fit_range),
                  "cl": lim.cl},
        "slurm": {"account": cfg.slurm.account,
                  "jobs": {name: {"partition": j.partition, "mem": j.mem, "time": j.time, "cpus": j.cpus}
                           for name, j in cfg.slurm.jobs.items()}},
    }


def phase_settings(cfg: Config, phase: str) -> dict:
    d = config_dict(cfg)
    return {key: d[key] for key in PHASE_SECTIONS[phase]}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flat(v, key + "."))
        elif not (isinstance(v, dict) and not v):
            out[key] = v
    return out


def settings_diff(old: dict, new: dict) -> list[str]:
    """Lines '<dotted.key>: <old> -> <new>' for every leaf that differs (both sides JSON-normalized)."""
    a, b = _flat(json.loads(json.dumps(old))), _flat(json.loads(json.dumps(new)))
    show = lambda d, k: repr(d[k]) if k in d else "(not set)"  # noqa: E731
    return [f"{k}: {show(a, k)} -> {show(b, k)}" for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)
            or (k in a) != (k in b)]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(cfg: Config, phase_dir, phase: str, upstream: str | None, outputs: dict[str, str],
                   missing=()) -> dict:
    """Write <phase_dir>/manifest.json. outputs: key -> path relative to phase_dir."""
    jobs_file = storage.join(phase_dir, JOBS_FILE)
    data = {
        "phase": phase,
        "version": storage.join(phase_dir).rsplit("/", 1)[-1],
        "upstream": upstream,
        "outputs": dict(outputs),
        "settings": phase_settings(cfg, phase),
        "missing": list(missing),
        "git_commit": _git_commit(),
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "slurm_jobs": json.loads(storage.read_text(jobs_file)) if storage.exists(jobs_file) else {},
    }
    storage.write_text(storage.join(phase_dir, MANIFEST), json.dumps(data, indent=2) + "\n")
    return data


def read_manifest(phase_dir) -> dict:
    path = storage.join(phase_dir, MANIFEST)
    if not storage.exists(path):
        raise UpstreamError(f"{phase_dir} has no {MANIFEST}: the phase failed, is still running, "
                            f"or this is not a phase folder")
    try:
        return json.loads(storage.read_text(path))
    except ValueError as e:
        raise UpstreamError(f"{path} is not valid JSON: {e}") from e


def check_upstream(cfg: Config, upstream_dir, phase: str) -> dict:
    """Run by the first job(s) of `phase`. Raise UpstreamError if the upstream manifest or one of its
    output files is missing, or the folder is of the wrong phase. Warn if its settings differ."""
    expected = UPSTREAM[phase]
    m = read_manifest(upstream_dir)
    if m.get("phase") != expected:
        raise UpstreamError(f"{upstream_dir} is a {m.get('phase')} folder; {phase} needs a {expected} folder")
    missing = [rel for rel in m["outputs"].values() if not storage.exists(storage.join(upstream_dir, rel))]
    if missing:
        raise UpstreamError(f"{upstream_dir}: files listed in {MANIFEST} are missing: {missing}")
    diff = settings_diff(m["settings"], phase_settings(cfg, expected))
    if diff:
        log.warning("upstream %s was made with other settings than the current config:\n  %s",
                    m["version"], "\n  ".join(diff))
    return m
```

- [ ] **Step 4: Run `$PYTEST/test_manifest.py -q`, expect PASS**

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/manifest.py tests/pipeline/test_manifest.py
/usr/bin/git commit -m "feat(pipeline): phase manifest, upstream check, stale-settings warning (step 9a)"
```

---

### Task 4: `runs.py` — versions and upstream choice

**Files:**
- Create: `src/tau3mu_gnn/pipeline/runs.py`
- Test: `tests/pipeline/test_runs.py`

**Interfaces:**
- Consumes: `storage` (Task 2); `MANIFEST`, `UPSTREAM`, `UpstreamError`, `config_dict` (Task 3).
- Produces:
  - `versions(parent, prefix) -> list[str]` (names `<prefix>_vsN` in `parent`, sorted by N)
  - `next_version(parent, prefix) -> str`
  - `is_complete(parent, name) -> bool`
  - `create_version(cfg, prefix, parent=None) -> str` (full path; `parent` default `cfg.output_dir`; writes `config_used.yaml`)
  - `pick_upstream(cfg, phase, from_=None) -> str` (full path of the upstream version folder)
  - `CONFIG_USED = "config_used.yaml"`

- [ ] **Step 1: Write the failing tests** — `tests/pipeline/test_runs.py`

```python
import pytest


def mk(parent, *names, complete=()):
    for n in names:
        (parent / n).mkdir(parents=True, exist_ok=True)
        if n in complete:
            (parent / n / "manifest.json").write_text("{}")


def test_next_version_numeric(tmp_path):
    from tau3mu_gnn.pipeline.runs import next_version, versions

    assert next_version(tmp_path, "phase_a") == "phase_a_vs1"
    mk(tmp_path, "phase_a_vs1", "phase_a_vs9", "phase_a_vs10", "phase_b_vs4", "phase_a_vs2_old", "debug")
    assert versions(tmp_path, "phase_a") == ["phase_a_vs1", "phase_a_vs9", "phase_a_vs10"]
    assert next_version(tmp_path, "phase_a") == "phase_a_vs11"


def test_create_version_writes_config_used(load, user, tmp_path):
    import yaml
    from tau3mu_gnn.pipeline.runs import create_version

    user["paths"]["output_base"] = str(tmp_path / "out")
    cfg = load(user)
    first = create_version(cfg, "phase_a")
    second = create_version(cfg, "phase_a")
    assert first.endswith("/toy_run/phase_a_vs1") and second.endswith("/toy_run/phase_a_vs2")
    used = yaml.safe_load((tmp_path / "out" / "toy_run" / "phase_a_vs1" / "config_used.yaml").read_text())
    assert used["analysis_name"] == "toy_run" and used["bdt"]["classifier"] == "models/W/bdt/toy.pck"
    debug = create_version(cfg, "eval", parent=f"{cfg.output_dir}/debug")
    assert debug.endswith("/toy_run/debug/eval_vs1")


def test_pick_upstream_highest_complete_and_note(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipeline.runs import pick_upstream

    user["paths"]["output_base"] = str(tmp_path)
    cfg = load(user)
    mk(tmp_path / "toy_run", "phase_a_vs1", "phase_a_vs2", "phase_a_vs3", complete=("phase_a_vs1", "phase_a_vs2"))
    assert pick_upstream(cfg, "phase_b") == f"{tmp_path}/toy_run/phase_a_vs2"
    assert "phase_a_vs3" in caplog.text


def test_pick_upstream_none_complete(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import UpstreamError
    from tau3mu_gnn.pipeline.runs import pick_upstream

    user["paths"]["output_base"] = str(tmp_path)
    cfg = load(user)
    mk(tmp_path / "toy_run", "phase_a_vs1")
    with pytest.raises(UpstreamError, match="phase-a"):
        pick_upstream(cfg, "phase_b")


def test_pick_upstream_from(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import UpstreamError
    from tau3mu_gnn.pipeline.runs import pick_upstream

    user["paths"]["output_base"] = str(tmp_path)
    cfg = load(user)
    mk(tmp_path / "toy_run", "phase_a_vs1", "phase_a_vs2", "phase_b_vs1", complete=("phase_a_vs1", "phase_b_vs1"))
    assert pick_upstream(cfg, "phase_b", from_="phase_a_vs1") == f"{tmp_path}/toy_run/phase_a_vs1"
    with pytest.raises(UpstreamError, match="no manifest"):
        pick_upstream(cfg, "phase_b", from_="phase_a_vs2")       # incomplete
    with pytest.raises(UpstreamError, match="phase_a"):
        pick_upstream(cfg, "phase_b", from_="phase_b_vs1")       # wrong phase
    with pytest.raises(UpstreamError, match="not found"):
        pick_upstream(cfg, "phase_b", from_="phase_a_vs7")


def test_pick_upstream_phase_a_has_none(load, user):
    from tau3mu_gnn.pipeline.runs import pick_upstream

    with pytest.raises(ValueError, match="phase_a"):
        pick_upstream(load(user), "phase_a")
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError`) — `$PYTEST/test_runs.py -q`

- [ ] **Step 3: Implement** `src/tau3mu_gnn/pipeline/runs.py`

```python
"""Versions of phases and debug runs: <output_dir>/<phase>_vsN, <output_dir>/debug/<stage>_vsN.

Every phase run makes a new version; nothing is overwritten. A version is complete when its manifest exists.
"""
from __future__ import annotations

import logging
import re

import yaml

from tau3mu_gnn.pipeline import storage
from tau3mu_gnn.pipeline.config import Config
from tau3mu_gnn.pipeline.manifest import MANIFEST, UPSTREAM, UpstreamError, config_dict

log = logging.getLogger(__name__)

CONFIG_USED = "config_used.yaml"


def _number(name: str, prefix: str) -> int | None:
    m = re.fullmatch(rf"{re.escape(prefix)}_vs([1-9]\d*)", name)
    return int(m.group(1)) if m else None


def versions(parent, prefix: str) -> list[str]:
    found = [(n, name) for name in storage.listdir(parent) if (n := _number(name, prefix)) is not None]
    return [name for _, name in sorted(found)]


def next_version(parent, prefix: str) -> str:
    existing = versions(parent, prefix)
    return f"{prefix}_vs{_number(existing[-1], prefix) + 1 if existing else 1}"


def is_complete(parent, name: str) -> bool:
    return storage.exists(storage.join(parent, name, MANIFEST))


def create_version(cfg: Config, prefix: str, parent=None) -> str:
    """Make the next <prefix>_vsN folder in parent (default: the run folder) with config_used.yaml."""
    parent = parent if parent is not None else cfg.output_dir
    path = storage.join(parent, next_version(parent, prefix))
    storage.makedirs(path)
    storage.write_text(storage.join(path, CONFIG_USED), yaml.safe_dump(config_dict(cfg), sort_keys=False))
    log.info("new version %s", path)
    return path


def pick_upstream(cfg: Config, phase: str, from_: str | None = None) -> str:
    """Full path of the upstream version for `phase`: from_ if given (must be complete), else the highest
    complete version. Raise UpstreamError if there is none."""
    up = UPSTREAM[phase]
    if up is None:
        raise ValueError(f"{phase} has no upstream phase")
    base = cfg.output_dir
    if from_ is not None:
        if _number(from_, up) is None:
            raise UpstreamError(f"--from {from_!r}: {phase} needs a {up}_vsN version")
        if not storage.exists(storage.join(base, from_)):
            raise UpstreamError(f"--from {from_!r}: not found in {base}")
        if not is_complete(base, from_):
            raise UpstreamError(f"--from {from_!r}: no manifest (failed or still running)")
        return storage.join(base, from_)
    names = versions(base, up)
    complete = [n for n in names if is_complete(base, n)]
    if not complete:
        raise UpstreamError(f"no complete {up} version in {base}; run {up.replace('_', '-')} first")
    newer = names[names.index(complete[-1]) + 1:]
    if newer:
        log.warning("%s have no manifest (failed or still running); using %s", ", ".join(newer), complete[-1])
    return storage.join(base, complete[-1])
```

- [ ] **Step 4: Run `$PYTEST/test_runs.py -q`, expect PASS**

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/runs.py tests/pipeline/test_runs.py
/usr/bin/git commit -m "feat(pipeline): phase versions and upstream choice (step 9a)"
```

---

### Task 5: Stage changes — stage 3 per sample, `summarize(write_dir=)`

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/stages/gnn.py`, `src/tau3mu_gnn/pipeline/stages/limit.py`
- Test: `tests/pipeline/test_gnn_scores.py`, `tests/pipeline/test_cut_scan.py`

**Interfaces:**
- Produces:
  - `gnn.output_files(out_dir, sample) -> {sample: Path, f"{sample}_scores": Path}` (used by `run` and by `jobs.py`)
  - `gnn.check_inputs(inputs, tree_name, channel, samples=SAMPLES)`, `gnn.run(cfg, inputs, out_dir, samples=SAMPLES)`: only the given samples are checked, read and written; `inputs` needs only those keys.
  - `limit.summarize(cfg, out_dir, write_dir=None)`: reads `out_dir/cut_*/limit.json`, writes and clears `limits.*` in `write_dir` (default `out_dir`).

- [ ] **Step 1: Write the failing tests**

In `tests/pipeline/test_gnn_scores.py`, change the helper `_stage` signature to `def _stage(tmp_path, monkeypatch, load, user, sig=None, bkg=None, samples=None):` and its last line to:

```python
    out_dir = tmp_path / "3_gnn"
    if samples is None:
        return gnn_stage.run(cfg, inputs, out_dir), out_dir, cfg, seen
    return gnn_stage.run(cfg, {s: inputs[s] for s in samples}, out_dir, samples=samples), out_dir, cfg, seen
```

Append:

```python
def test_stage_one_sample(tmp_path, monkeypatch, load, user):
    outputs, out_dir, *_ = _stage(tmp_path, monkeypatch, load, user, samples=("background",))
    assert outputs == {"background": out_dir / "background_with_gnn_scores.root",
                       "background_scores": out_dir / "raw_scores_and_tau_mass_background.npz"}
    assert sorted(p.name for p in out_dir.iterdir()) == ["background_with_gnn_scores.root",
                                                         "raw_scores_and_tau_mass_background.npz"]


def test_output_files(tmp_path):
    from tau3mu_gnn.pipeline.stages import gnn as gnn_stage

    assert gnn_stage.output_files(tmp_path, "signal") == {
        "signal": tmp_path / "signal_with_gnn_scores.root",
        "signal_scores": tmp_path / "raw_scores_and_tau_mass_signal.npz"}
```

In `tests/pipeline/test_cut_scan.py`, append:

```python
def test_summary_write_dir(load, user, tmp_path):
    """Phase C job: cut_*/ on dCache (read via /pnfs), table written to scratch."""
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    write_limit(tmp_path / "phase", 3.9, 1.3)
    scratch = tmp_path / "scratch"
    s = limit.summarize(cfg, tmp_path / "phase", write_dir=scratch)
    assert s["files"]["json"] == scratch / "limits.json" and (scratch / "limits.png").is_file()
    assert not (tmp_path / "phase" / "limits.json").exists()
```

- [ ] **Step 2: Run, expect FAIL** — `$PYTEST/test_gnn_scores.py $PYTEST/test_cut_scan.py -q -k "one_sample or output_files or write_dir"` (`TypeError: unexpected keyword 'samples'` / `'write_dir'`, `AttributeError: output_files`)

- [ ] **Step 3: Implement**

`gnn.py`:

```python
def output_files(out_dir: str | Path, sample: str) -> dict[str, Path]:
    out_dir = Path(out_dir)
    return {sample: out_dir / f"{sample}_with_gnn_scores.root",
            f"{sample}_scores": out_dir / f"raw_scores_and_tau_mass_{sample}.npz"}


def check_inputs(inputs: Mapping[str, str | Path], tree_name: str, channel: str,
                 samples: Sequence[str] = SAMPLES) -> None:
    """Raise BranchError if a GNN input branch is missing, or GNN_score is already there.
    Raise DuplicateEventError if two events of a file have the same (run, luminosityBlock, event)."""
    check_stage_inputs({s: inputs[s] for s in samples}, tree_name, input_branch_names(channel), "GNN",
                       forbidden=["GNN_score"])
    for sample in samples:
        check_unique_keys(inputs[sample], tree_name)


def run(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path,
        samples: Sequence[str] = SAMPLES) -> dict[str, Path]:
    """Score inputs (sample -> stage 2 output file) into out_dir. samples: which samples (one per SLURM job).

    Returns {<sample>, <sample>_scores} -> output file for each sample.
    """
    check_inputs(inputs, cfg.inputs.tree_name, cfg.channel, samples)
    ...  # unchanged up to the loop
    outputs = {}
    for sample in samples:
        files = output_files(out_dir, sample)
        outputs.update(files)
        log.info("stage 3 GNN: %s -> %s", inputs[sample], files[sample])
        in_file = str(inputs[sample])
        score_events([in_file], models, {g.model_name: {in_file: str(files[sample])}},
                     {g.model_name: files[f"{sample}_scores"]}, cfg.channel, g.unsculpting_mode,
                     g.bdt_preselection_cut, tree_name=cfg.inputs.tree_name, device=device)
    return outputs
```

Add `Sequence` to the `typing` import. Keep the existing code between `check_inputs(...)` and the loop (torch, model loading, `out_dir.mkdir`).

`limit.py`, `summarize`:

```python
def summarize(cfg: Config, out_dir: str | Path, write_dir: str | Path | None = None) -> dict:
    """Table of the cut scan: reads <out_dir>/cut_<cut>/limit.json for each cut of limit.gnn_score_cut.

    A cut without a finite limit is a missing row (warning). Best cut = lowest limit, the smaller cut on a tie.
    Writes limits.json, limits.txt, limits.png into write_dir (default out_dir). Raises RuntimeError if no cut
    has a limit.
    """
    out_dir = Path(out_dir)
    write_dir = Path(write_dir) if write_dir is not None else out_dir
    write_dir.mkdir(parents=True, exist_ok=True)
    files = {k: write_dir / name for k, name in SUMMARY_FILES.items()}
    ...  # rest unchanged
```

- [ ] **Step 4: Run `$PYTEST/test_gnn_scores.py $PYTEST/test_cut_scan.py -q`, expect PASS**

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/stages/gnn.py src/tau3mu_gnn/pipeline/stages/limit.py tests/pipeline/test_gnn_scores.py tests/pipeline/test_cut_scan.py
/usr/bin/git commit -m "feat(stages): GNN stage per sample, summarize write_dir (step 9a)"
```

---

### Task 6: `jobs.py` — job bodies and the job CLI

**Files:**
- Create: `src/tau3mu_gnn/pipeline/jobs.py`
- Test: `tests/pipeline/test_jobs.py`

**Interfaces:**
- Consumes: Tasks 1–5: `storage.*`, `write_manifest`, `read_manifest`, `check_upstream`, `gnn.output_files`, `gnn.run(samples=)`, `limit.summarize(write_dir=)`, `limit.cut_dir_name`, `limit.SUMMARY_FILES`, `evaluation.PLOTS`, stage `run` functions.
- Produces (for 9b):
  - Folder names `RESONANCE_DIR="0_resonance_skim"`, `SKIM_DIR="1_skim"`, `BDT_DIR="2_bdt"`, `GNN_DIR="3_gnn"`, `EVAL_DIR="4_eval"`; `JOBS = ("skim", "bdt", "gnn", "eval", "limit", "summary")`.
  - `skim(cfg, phase_dir) -> None`
  - `bdt(cfg, phase_dir, source_dir=None, manifest=True) -> None` (manifest `phase_a`; `source_dir`: folder with `1_skim/`, default `phase_dir`)
  - `gnn(cfg, phase_dir, upstream_dir, sample) -> None` (runs `check_upstream(..., "phase_b")`)
  - `evaluation(cfg, phase_dir, upstream_dir, source_dir=None, manifest=True) -> None` (manifest `phase_b`)
  - `limit(cfg, phase_dir, upstream_dir, cut) -> None` (runs `check_upstream(..., "phase_c")`; copies `cut_<cut>/` also on failure)
  - `summary(cfg, phase_dir, upstream_dir, source_dir=None, manifest=True) -> None` (manifest `phase_c` with `missing`)
  - Manifest output keys: phase A `skim_signal, skim_background, bdt_signal, bdt_background, scores_pickle`; phase B `gnn_signal, gnn_background, signal_scores, background_scores, eval_<plot>`; phase C `limits_json, limits_txt, limits_png, limit_<cut>`.
  - `main(argv=None)`: `python -m tau3mu_gnn.pipeline.jobs <job> --config F --phase-dir D [--upstream D] [--source D] [--cut X] [--sample S] [--no-manifest]`

- [ ] **Step 1: Write the failing tests** — `tests/pipeline/test_jobs.py`

```python
"""Job bodies on toy data. Stages are faked (they have their own tests), except summarize and, with
Combine installed, the real limit stage."""
import json
import shutil
from pathlib import Path

import pytest

from test_limit import needs_combine, toy_inputs  # noqa: F401  (fixture reused)

SAMPLES = ("signal", "background")


@pytest.fixture
def cfg(load, user, tmp_path):
    user["paths"]["output_base"] = str(tmp_path / "out")
    user["limit"]["gnn_score_cut"] = [2.0, 3.9]
    user["limit"]["signal_weight"] = 0.5
    return load(user)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    d = tmp_path / "scratch"
    d.mkdir()
    monkeypatch.setenv("TMPDIR", str(d))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)       # re-read TMPDIR
    return d


@pytest.fixture
def fake_stages(monkeypatch):
    """Fake stage run functions: write small files with the real names, record their inputs."""
    from tau3mu_gnn.pipeline import jobs

    seen = {}

    def write(p, text="x"):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(text)
        return Path(p)

    def resonance(cfg, out_dir):
        seen["resonance"] = Path(out_dir)
        return {s: list(getattr(cfg.inputs, s)) for s in SAMPLES}

    def skim(cfg, inputs, out_dir):
        seen["skim"] = dict(inputs)
        return {s: write(Path(out_dir) / f"{s}.root") for s in inputs}

    def bdt(cfg, inputs, out_dir):
        seen["bdt"] = {s: Path(inputs[s]).read_text() for s in SAMPLES}   # inputs must be readable
        out = {s: write(Path(out_dir) / f"{s}_with_bdt_scores.root") for s in SAMPLES}
        out["scores_pickle"] = write(Path(out_dir) / "tau_score_corr.pkl")
        return out

    def gnn(cfg, inputs, out_dir, samples=SAMPLES):
        seen.setdefault("gnn", []).append(dict(inputs))
        out = {}
        for s in samples:
            for key, path in jobs.gnn_stage.output_files(out_dir, s).items():
                out[key] = write(path)
        return out

    def evaluation(cfg, inputs, out_dir):
        seen["eval"] = {k: Path(v).exists() for k, v in inputs.items()}
        return {name: write(Path(out_dir) / f"{name}.png") for name in jobs.evaluation_stage.PLOTS}

    def limit(cfg, inputs, out_dir, cut):
        card = Path(out_dir) / jobs.limit_stage.cut_dir_name(cut)
        write(card / "combine_asymptotic.log", "log")
        if cut == seen.get("fail_cut"):
            raise RuntimeError("Combine wrote no median expected limit")
        write(card / "limit.json", json.dumps({"cut": cut, "limit": 1.0 + abs(cut - 3.9), "limitErr": 0.0}))
        return {}

    monkeypatch.setattr(jobs.resonance_skim, "run", resonance)
    monkeypatch.setattr(jobs.skim_stage, "run", skim)
    monkeypatch.setattr(jobs.bdt_stage, "run", bdt)
    monkeypatch.setattr(jobs.gnn_stage, "run", gnn)
    monkeypatch.setattr(jobs.evaluation_stage, "run", evaluation)
    monkeypatch.setattr(jobs.limit_stage, "run", limit)
    return seen


def run_phase_a(cfg):
    from tau3mu_gnn.pipeline import jobs, runs

    a = runs.create_version(cfg, "phase_a")
    jobs.skim(cfg, a)
    jobs.bdt(cfg, a)
    return a


def run_phase_b(cfg):
    from tau3mu_gnn.pipeline import jobs, runs

    up = runs.pick_upstream(cfg, "phase_b")
    b = runs.create_version(cfg, "phase_b")
    for s in SAMPLES:
        jobs.gnn(cfg, b, up, s)
    jobs.evaluation(cfg, b, up)
    return b


def test_phase_a(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipeline.manifest import read_manifest

    a = Path(run_phase_a(cfg))
    m = read_manifest(a)
    assert m["phase"] == "phase_a" and m["version"] == "phase_a_vs1" and m["upstream"] is None
    assert m["outputs"] == {"skim_signal": "1_skim/signal.root", "skim_background": "1_skim/background.root",
                            "bdt_signal": "2_bdt/signal_with_bdt_scores.root",
                            "bdt_background": "2_bdt/background_with_bdt_scores.root",
                            "scores_pickle": "2_bdt/tau_score_corr.pkl"}
    assert all((a / rel).is_file() for rel in m["outputs"].values())
    assert (a / "config_used.yaml").is_file()
    assert fake_stages["bdt"] == {"signal": "x", "background": "x"}      # read from the phase folder


def test_phase_b_and_c(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipeline import jobs, runs
    from tau3mu_gnn.pipeline.manifest import read_manifest

    run_phase_a(cfg)
    b = Path(run_phase_b(cfg))
    mb = read_manifest(b)
    assert mb["upstream"] == "phase_a_vs1"
    assert {"gnn_signal", "gnn_background", "signal_scores", "background_scores", "eval_roc"} <= set(mb["outputs"])
    assert all((b / rel).is_file() for rel in mb["outputs"].values())
    assert [list(i) for i in fake_stages["gnn"]] == [["signal"], ["background"]]   # one sample per job
    assert all(fake_stages["eval"].values())

    up = runs.pick_upstream(cfg, "phase_c")
    c = runs.create_version(cfg, "phase_c")
    for cut in cfg.limit.gnn_score_cut:
        jobs.limit(cfg, c, up, cut)
    jobs.summary(cfg, c, up)
    mc = read_manifest(c)
    assert mc["upstream"] == "phase_b_vs1" and mc["missing"] == []
    assert mc["outputs"]["limit_3.9"] == "cut_3.9/limit.json" and mc["outputs"]["limits_json"] == "limits.json"
    assert json.loads((Path(c) / "limits.json").read_text())["best_cut"] == 3.9


def test_rerun_phase_b_gives_new_version(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipeline import runs

    run_phase_a(cfg)
    run_phase_b(cfg)
    b2 = run_phase_b(cfg)
    assert b2.endswith("phase_b_vs2")
    assert runs.pick_upstream(cfg, "phase_c") == b2


def test_failed_limit_still_copies_cut_folder(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipeline import jobs, runs
    from tau3mu_gnn.pipeline.manifest import read_manifest

    run_phase_a(cfg)
    run_phase_b(cfg)
    up = runs.pick_upstream(cfg, "phase_c")
    c = runs.create_version(cfg, "phase_c")
    fake_stages["fail_cut"] = 2.0
    with pytest.raises(RuntimeError):
        jobs.limit(cfg, c, up, 2.0)
    assert (Path(c) / "cut_2.0" / "combine_asymptotic.log").read_text() == "log"
    jobs.limit(cfg, c, up, 3.9)
    jobs.summary(cfg, c, up)                                  # afterany: runs anyway
    assert read_manifest(c)["missing"] == [2.0]


def test_gnn_job_stops_on_incomplete_upstream(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipeline import jobs, runs
    from tau3mu_gnn.pipeline.manifest import UpstreamError

    a = runs.create_version(cfg, "phase_a")
    jobs.skim(cfg, a)                                         # bdt job never ran: no manifest
    b = runs.create_version(cfg, "phase_b")
    with pytest.raises(UpstreamError, match="manifest"):
        jobs.gnn(cfg, b, a, "signal")
    assert "gnn" not in fake_stages


def test_debug_bdt_reads_source_no_manifest(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipeline import jobs, runs

    a = run_phase_a(cfg)
    d = runs.create_version(cfg, "bdt", parent=f"{cfg.output_dir}/debug")
    jobs.bdt(cfg, d, source_dir=a, manifest=False)
    assert (Path(d) / "2_bdt" / "tau_score_corr.pkl").is_file()
    assert not (Path(d) / "manifest.json").exists()


def test_main_dispatch(cfg, scratch, fake_stages, monkeypatch):
    from tau3mu_gnn.pipeline import jobs, runs

    monkeypatch.setattr(jobs, "load_config", lambda path: cfg)   # toy config needs the fake repo defaults
    a = runs.create_version(cfg, "phase_a")
    jobs.main(["skim", "--config", str(cfg.user_file), "--phase-dir", a])
    assert (Path(a) / "1_skim" / "signal.root").is_file()
    with pytest.raises(SystemExit):
        jobs.main(["gnn", "--config", str(cfg.user_file), "--phase-dir", a])   # needs --upstream, --sample


@needs_combine
def test_limit_and_summary_jobs_real_stage(cfg, scratch, toy_inputs, tmp_path):
    """Real stage 5 + Combine on the toy stage 3 files, as phase C jobs."""
    from tau3mu_gnn.pipeline import jobs, runs
    from tau3mu_gnn.pipeline.manifest import read_manifest, write_manifest

    b = Path(cfg.output_dir) / "phase_b_vs1"
    (b / "3_gnn").mkdir(parents=True)
    for s, f in zip(SAMPLES, toy_inputs):
        shutil.copy(f, b / "3_gnn" / f"{s}_with_gnn_scores.root")
    write_manifest(cfg, b, "phase_b", "phase_a_vs1",
                   {f"gnn_{s}": f"3_gnn/{s}_with_gnn_scores.root" for s in SAMPLES})
    c = runs.create_version(cfg, "phase_c")
    for cut in cfg.limit.gnn_score_cut:
        jobs.limit(cfg, c, str(b), cut)
    jobs.summary(cfg, c, str(b))
    m = read_manifest(c)
    assert m["missing"] == [] and set(m["outputs"]) >= {"limit_2.0", "limit_3.9", "limits_json"}
    assert (Path(c) / "cut_3.9" / "datacard.txt").is_file()
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: tau3mu_gnn.pipeline.jobs`) — `$PYTEST/test_jobs.py -q`

- [ ] **Step 3: Implement** `src/tau3mu_gnn/pipeline/jobs.py`

```python
"""Job bodies: what one SLURM job runs (step 9). The generated job scripts (step 9b) call main().

Each job works in a fresh scratch folder under $TMPDIR (the job script points TMPDIR to /scratch), then
copies its outputs into the phase folder with storage.put. Upstream files are read through
storage.local_path (dCache: the /pnfs path). The last job of a phase writes the manifest.
"""
from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from tau3mu_gnn.pipeline import storage
from tau3mu_gnn.pipeline.config import load_config
from tau3mu_gnn.pipeline.manifest import check_upstream, read_manifest, write_manifest
from tau3mu_gnn.pipeline.stages import bdt as bdt_stage
from tau3mu_gnn.pipeline.stages import evaluation as evaluation_stage
from tau3mu_gnn.pipeline.stages import gnn as gnn_stage
from tau3mu_gnn.pipeline.stages import limit as limit_stage
from tau3mu_gnn.pipeline.stages import resonance_skim
from tau3mu_gnn.pipeline.stages import skim as skim_stage

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")
RESONANCE_DIR, SKIM_DIR, BDT_DIR, GNN_DIR, EVAL_DIR = "0_resonance_skim", "1_skim", "2_bdt", "3_gnn", "4_eval"
JOBS = ("skim", "bdt", "gnn", "eval", "limit", "summary")


def _scratch(job: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"{job}_"))


def _name(path) -> str:
    return storage.join(path).rsplit("/", 1)[-1]


def _read(folder, rel) -> Path:
    return storage.local_path(storage.join(folder, rel))


def skim(cfg, phase_dir) -> None:
    """Phase A, job 1: stage 0 (resonance skim / branch map) + stage 1 (skim)."""
    scratch = _scratch("skim")
    files = resonance_skim.run(cfg, scratch / RESONANCE_DIR)
    skim_stage.run(cfg, files, scratch / SKIM_DIR)
    storage.put(scratch, phase_dir)


def bdt(cfg, phase_dir, source_dir=None, manifest=True) -> None:
    """Phase A, job 2: stage 2 on <source_dir>/1_skim/ (default: this phase). Writes the phase A manifest."""
    src = source_dir if source_dir is not None else phase_dir
    inputs = {s: _read(src, f"{SKIM_DIR}/{s}.root") for s in SAMPLES}
    scratch = _scratch("bdt")
    out = bdt_stage.run(cfg, inputs, scratch / BDT_DIR)
    storage.put(scratch, phase_dir)
    if manifest:
        outputs = {f"skim_{s}": f"{SKIM_DIR}/{s}.root" for s in SAMPLES}
        outputs.update({f"bdt_{s}": Path(out[s]).relative_to(scratch).as_posix() for s in SAMPLES})
        outputs["scores_pickle"] = Path(out["scores_pickle"]).relative_to(scratch).as_posix()
        write_manifest(cfg, phase_dir, "phase_a", None, outputs)


def gnn(cfg, phase_dir, upstream_dir, sample: str) -> None:
    """Phase B: stage 3 for one sample. First job of the phase: checks the phase A version."""
    m = check_upstream(cfg, upstream_dir, "phase_b")
    scratch = _scratch(f"gnn_{sample}")
    gnn_stage.run(cfg, {sample: _read(upstream_dir, m["outputs"][f"bdt_{sample}"])}, scratch / GNN_DIR,
                  samples=(sample,))
    storage.put(scratch, phase_dir)


def evaluation(cfg, phase_dir, upstream_dir, source_dir=None, manifest=True) -> None:
    """Phase B, last job: stage 4 on <source_dir>/3_gnn/ (default: this phase) + the phase A pickle.
    Writes the phase B manifest."""
    up = read_manifest(upstream_dir)
    src = source_dir if source_dir is not None else phase_dir
    # stage 3 file names, relative to the phase folder
    gnn_files = {k: f"{GNN_DIR}/{Path(v).name}" for s in SAMPLES for k, v in gnn_stage.output_files(".", s).items()}
    inputs = {f"{s}_scores": _read(src, gnn_files[f"{s}_scores"]) for s in SAMPLES}
    inputs["scores_pickle"] = _read(upstream_dir, up["outputs"]["scores_pickle"])
    scratch = _scratch("eval")
    evaluation_stage.run(cfg, inputs, scratch / EVAL_DIR)
    storage.put(scratch, phase_dir)
    if manifest:
        outputs = {f"gnn_{s}": gnn_files[s] for s in SAMPLES}
        outputs.update({f"{s}_scores": gnn_files[f"{s}_scores"] for s in SAMPLES})
        outputs.update({f"eval_{p}": f"{EVAL_DIR}/{p}.png" for p in evaluation_stage.PLOTS})
        write_manifest(cfg, phase_dir, "phase_b", _name(upstream_dir), outputs)


def limit(cfg, phase_dir, upstream_dir, cut: float) -> None:
    """Phase C: stage 5 for one cut. Checks the phase B version. cut_<cut>/ is copied also if the stage fails."""
    m = check_upstream(cfg, upstream_dir, "phase_c")
    inputs = {s: _read(upstream_dir, m["outputs"][f"gnn_{s}"]) for s in SAMPLES}
    scratch = _scratch("limit")
    card = scratch / limit_stage.cut_dir_name(cut)
    try:
        limit_stage.run(cfg, inputs, scratch, cut)
    finally:
        if card.exists():
            storage.put(card, storage.join(phase_dir, card.name))


def summary(cfg, phase_dir, upstream_dir, source_dir=None, manifest=True) -> None:
    """Phase C, last job (afterany): limits.* from <source_dir>/cut_*/ (default: this phase).
    Writes the phase C manifest, with the cuts that have no limit in 'missing'."""
    src = source_dir if source_dir is not None else phase_dir
    scratch = _scratch("summary")
    s = limit_stage.summarize(cfg, storage.local_path(src), write_dir=scratch)
    storage.put(scratch, phase_dir)
    if manifest:
        outputs = {f"limits_{k}": name for k, name in limit_stage.SUMMARY_FILES.items()}
        outputs.update({f"limit_{r['cut']}": f"{limit_stage.cut_dir_name(r['cut'])}/limit.json"
                        for r in s["rows"] if r["limit"] is not None})
        write_manifest(cfg, phase_dir, "phase_c", _name(upstream_dir), outputs, missing=s["missing"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tau3mu_gnn.pipeline.jobs",
                                description="Run one pipeline job (called by the generated SLURM job scripts).")
    p.add_argument("job", choices=JOBS)
    p.add_argument("--config", required=True, help="user config file")
    p.add_argument("--phase-dir", required=True, help="output folder: <phase>_vsN or debug/<stage>_vsN")
    p.add_argument("--upstream", help="upstream phase version folder (gnn, eval, limit, summary)")
    p.add_argument("--source", help="read same-phase inputs from this folder (debug runs)")
    p.add_argument("--cut", type=float, help="GNN score cut (limit)")
    p.add_argument("--sample", choices=SAMPLES, help="sample (gnn)")
    p.add_argument("--no-manifest", action="store_true", help="do not write a manifest (debug runs)")
    a = p.parse_args(argv)
    need = {"gnn": ("upstream", "sample"), "eval": ("upstream",), "limit": ("upstream", "cut"),
            "summary": ("upstream",)}.get(a.job, ())
    absent = [f"--{n}" for n in need if getattr(a, n) is None]
    if absent:
        p.error(f"job {a.job} needs {' '.join(absent)}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    manifest = not a.no_manifest
    if a.job == "skim":
        skim(cfg, a.phase_dir)
    elif a.job == "bdt":
        bdt(cfg, a.phase_dir, a.source, manifest)
    elif a.job == "gnn":
        gnn(cfg, a.phase_dir, a.upstream, a.sample)
    elif a.job == "eval":
        evaluation(cfg, a.phase_dir, a.upstream, a.source, manifest)
    elif a.job == "limit":
        limit(cfg, a.phase_dir, a.upstream, a.cut)
    else:
        summary(cfg, a.phase_dir, a.upstream, a.source, manifest)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run `$PYTEST/test_jobs.py -q`, expect PASS** (the `needs_combine` test runs in the Tau3Mu env)

- [ ] **Step 5: Run the full suite `$PYTEST -q`, expect PASS**

- [ ] **Step 6: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/jobs.py tests/pipeline/test_jobs.py
/usr/bin/git commit -m "feat(pipeline): job bodies and job CLI (step 9a)"
```

---

### Task 7: Roadmap

**Files:**
- Modify: `docs/wrapper_roadmap.md`

- [ ] **Step 1: Edit** (spec sec. 9)
  - Sec. 5 output layout block → replace with the layout of spec sec. 3 (results tree + `<paths.log_dir>` tree).
  - Sec. 5 bullet "Storage": add "Read through `/pnfs`, write with `xrdcp` (`pipeline/storage.py`). A local `output_base` also works."
  - Sec. 5.1 config block: in `paths:` replace the comment `# run folder = <output_base>/<analysis_name>_vsN (version added in step 9)` with `log_dir: /work/$USER/tau3mu_logs   # job scripts + SLURM logs (local disk)`; remove the `slurm:` block there and add below the block: "`slurm:` (account + per-job partition / mem / time / cpus) is in the frozen defaults file (step 9, D7)."
  - Sec. 5.1 "Split into two files": move `slurm` from the user list to the defaults list; add `paths.log_dir` to the user list.
  - Sec. 5.2 code layout: add `jobs.py          # job bodies, python -m tau3mu_gnn.pipeline.jobs` and `runs.py          # phase versions <phase>_vsN, upstream choice`.
  - Step 9 line → replace with:
    `9. Driver, spec docs/superpowers/specs/2026-10-12-step9-driver-design.md. 9a ✅ (no SLURM): config `paths.log_dir` + frozen `slurm:`; `storage.py`, `manifest.py`, `runs.py`, `jobs.py`; stage 3 per sample; `summarize(write_dir=)`. Every phase run makes a new version `<phase>_vsN`; single-stage runs in `debug/<stage>_vsN`.`
    `9b. SLURM layer: job script template, `slurm.py` (`sbatch --parsable`, dependencies, `--dry-run`), proxy check, `cli.py` (phase-a/b/c, stage). Check: small-input run A → B → C on T3 (Kaito queues).`
  - Decision log, new rows (date 2026-10-12): one row "Step 9 design (Kaito), spec `docs/superpowers/specs/2026-10-12-step9-driver-design.md`: D1 split 9a/9b; D2 dCache read via /pnfs, write via xrdcp; D3 no local phase runs; D4 every phase run → new `<phase>_vsN` (supersedes run folder `_vsN` and 'cut scans in the same run'); D5 single-stage runs in `debug/`; D6 `paths.log_dir` + log copy next to results; D7 slurm resources in the frozen defaults file (supersedes `slurm` in the user file); D8 job scripts + `sbatch --parsable`; D9 9b check = small-input run." and one row "Step 9a done: ..." summarising the files.

- [ ] **Step 2: Run `$PYTEST -q`, expect PASS**

- [ ] **Step 3: Commit**

```bash
/usr/bin/git add docs/wrapper_roadmap.md docs/superpowers/plans/2026-10-12-step9a-driver-core.md
/usr/bin/git commit -m "docs(roadmap): step 9 design, 9a done"
```
