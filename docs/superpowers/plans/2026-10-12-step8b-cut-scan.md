# Mini Step 8b — Cut Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A cut scan over `limit.gnn_score_cut`: one stage 5 call per cut (already step 8), then a summary that writes the table `limits.json` / `limits.txt` and the plot `limits.png`, with the best cut marked. A local loop `run_scan` runs all cuts in one process. The SLURM fan-out (one job per cut + a final summary job) is step 9.

**Architecture:**
- `pipeline/config.py`: `limit.gnn_score_cut` rejects duplicate values (two cuts would share one `cut_<cut>/` folder).
- `pipeline/stages/limit.py`:
  - `run(cfg, inputs, out_dir, cut)` (step 8) now deletes an old `cut_<cut>/limit.json` before the work. A failed rerun then leaves no stale limit.
  - New `summarize(cfg, out_dir)`: reads `<out_dir>/cut_<cut>/limit.json` for each cut in the config. A cut without a usable limit is a `missing` row (warning). Best cut = lowest limit. Writes `<out_dir>/limits.json`, `limits.txt`, `limits.png`.
  - New `run_scan(cfg, inputs, out_dir)`: checks inputs and Combine once, runs `run` for each cut, logs and skips a cut whose Combine fails, then calls `summarize`.

**Tech Stack:** Python 3.11 (Tau3Mu env), matplotlib (`matplotlib.figure.Figure`, no pyplot), pytest.

**Spec:** `docs/wrapper_roadmap.md` sec. 6 step 8b, sec. 5 (output layout: `phase_c/cut_<value>/`, `phase_c/limits.json` "cut → limit table, best cut marked"), decision log 2026-10-07 Q6. Decisions from Kaito (2026-10-12) in Global Constraints.

## Global Constraints

- Test command: `$PYTEST` = `PYTHONPATH=/work/kfuss/SVJ_GNN/.claude/worktrees/step8b-cut-scan/src /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m pytest tests/pipeline`
- Branch `pipeline-step8b-cut-scan`, from `pipeline-step8-limit` (160fdf1, PR #18 open). Worktree `/work/kfuss/SVJ_GNN/.claude/worktrees/step8b-cut-scan`. The PR is stacked on #18.
- Git in this worktree: `/usr/bin/git` only. One git command per Bash call.
- **No physics change.** `make_datacard`, `run_combine`, `read_limit` stay as they are. The per-cut limit is the step 8 code.
- **Missing cut** (decision 2026-10-12): a cut without `limit.json` → row with `"limit": null`, listed in `missing`, one warning. Best cut from the cuts with a limit. `RuntimeError` only if no cut has a limit. (Step 9 will start the summary job with `afterany`.)
- **Outputs** (decision 2026-10-12), all in `out_dir` (= `phase_c/` later):
  - `limits.json`: `{"cl", "best_cut", "best_limit", "missing": [cut, ...], "rows": [{"cut", "limit", "limitErr", "best"}, ...]}`. Rows sorted by cut.
  - `limits.txt`: plain table for people, best row marked with `*`.
  - `limits.png`: median expected limit vs cut, best cut marked.
- Best cut on a tie: the smaller cut (rows sorted by cut, first minimum).
- A non-finite limit (NaN / inf from a failed fit) counts as missing.
- A `limit.json` that is not valid JSON, lacks a key, or has another `cut` than its folder → `ValueError` naming the file. (This is a broken file, not a failed job.)
- `cut_*` folders for cuts not in the config are ignored (old scans in the same run folder).
- **Regression** (decision 2026-10-12): tests only. No SLURM job.
- Plot (dataviz rules): one series, so no legend box; best cut gets a direct text label in ink color, not the series color; line 2 px, markers ≥ 8 px; light grid; missing cuts as gray dotted vertical lines. Use `matplotlib.figure.Figure` (no pyplot, no display backend needed on batch nodes).
- The stage knows nothing about SLURM (sec. 5).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Rerun of one cut that now fails** (old `limit.json` from an earlier run is still in `cut_<cut>/`) → the table must not show the old limit. `run` deletes `limit.json` first. Test in Task 4.
2. **Combine gives NaN** → must not be the best cut (`min` with NaN gives nonsense); row is `missing`. Test in Task 2.
3. **Old `cut_*` folders** from a previous config in the same `phase_c/` → not in the table. Test in Task 2.
4. **Broken or foreign `limit.json`** (bad JSON, other cut) → `ValueError` with the path, not a silent `missing`. Test in Task 2.
5. **Duplicate cuts** (`[3.9, 3.9]`) → config error at load time. Test in Task 1.

## File Structure

| File | Responsibility |
|---|---|
| `src/tau3mu_gnn/pipeline/config.py` (modify `_number_list`) | reject duplicate cuts |
| `src/tau3mu_gnn/pipeline/stages/limit.py` (modify) | stale `limit.json` removal in `run`; `summarize`, `run_scan`, table + plot helpers |
| `tests/pipeline/test_config_parsers.py` (modify) | duplicate test |
| `tests/pipeline/test_cut_scan.py` (create) | summary, plot, scan loop |
| `docs/wrapper_roadmap.md` (modify) | step 8b done, decision log |

---

### Task 1: Reject duplicate cuts

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/config.py:93-97`
- Test: `tests/pipeline/test_config_parsers.py:80-86`

**Interfaces:**
- Produces: `_number_list(v) -> tuple[float, ...]`, raises `ValueError` with `"duplicate"` in the message.

- [ ] **Step 1: Write the failing test** (append after `test_number_list_scalar_and_list`)

```python
def test_number_list_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        c._number_list([3.9, 4.5, 3.9])
    with pytest.raises(ValueError, match="duplicate"):
        c._number_list([4, 4.0])          # same cut, same cut_4.0/ folder
```

- [ ] **Step 2: Run it, expect FAIL** (no error raised)

Run: `$PYTEST/test_config_parsers.py::test_number_list_rejects_duplicates -v`

- [ ] **Step 3: Implement**

```python
def _number_list(v):
    items = v if isinstance(v, list) else [v]
    if not items:
        raise ValueError("must be a number or a non-empty list of numbers")
    numbers = tuple(_number(i) for i in items)
    duplicates = sorted({x for x in numbers if numbers.count(x) > 1})
    if duplicates:
        raise ValueError(f"has duplicate values {duplicates}")
    return numbers
```

- [ ] **Step 4: Run `$PYTEST/test_config_parsers.py $PYTEST/test_load_config.py -q`, expect PASS**

(`$PYTEST/x.py` means: the test command with `tests/pipeline/x.py` instead of `tests/pipeline`.)

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/config.py tests/pipeline/test_config_parsers.py
/usr/bin/git commit -m "fix(config): reject duplicate gnn_score_cut values"
```

---

### Task 2: `summarize` with `limits.json` and `limits.txt`

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/stages/limit.py`
- Create: `tests/pipeline/test_cut_scan.py`

**Interfaces:**
- Consumes: `cut_dir_name(cut) -> "cut_<cut>"` (step 8), `cfg.limit.gnn_score_cut: tuple[float, ...]`, `cfg.limit.cl: float`.
- Produces:
  - `summarize(cfg, out_dir) -> dict` with keys `cl`, `best_cut`, `best_limit`, `missing`, `rows`, `files` (`{"json", "txt", "png"}` → `Path`).
  - `SUMMARY_FILES = {"json": "limits.json", "txt": "limits.txt", "png": "limits.png"}`.
  - Task 3 adds `_plot(summary, path)`. In this task `summarize` must already call `_plot`; add a stub `_plot` that does nothing, Task 3 replaces it.

- [ ] **Step 1: Write the failing tests** — new file `tests/pipeline/test_cut_scan.py`

```python
import json
import math

import pytest


def scan_cfg(load, user, cuts):
    user["limit"]["gnn_score_cut"] = list(cuts)
    return load(user)


def write_limit(out_dir, cut, limit, folder_cut=None, err=0.0):
    """As stage 5 writes it: <out_dir>/cut_<cut>/limit.json."""
    d = out_dir / f"cut_{folder_cut if folder_cut is not None else cut}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "limit.json").write_text(json.dumps({"cut": cut, "limit": limit, "limitErr": err}))
    return d / "limit.json"


def test_summary_best_cut_and_files(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [4.5, 2.0, 3.9])          # config order is not sorted
    for cut, lim in [(2.0, 1.7), (3.9, 1.2890625), (4.5, 1.5)]:
        write_limit(tmp_path, cut, lim)

    s = limit.summarize(cfg, tmp_path)

    assert s["best_cut"] == 3.9 and s["best_limit"] == 1.2890625 and s["missing"] == []
    assert [r["cut"] for r in s["rows"]] == [2.0, 3.9, 4.5]
    assert [r["best"] for r in s["rows"]] == [False, True, False]
    assert s["files"] == {k: tmp_path / v for k, v in limit.SUMMARY_FILES.items()}
    saved = json.loads((tmp_path / "limits.json").read_text())
    assert saved == {"cl": 0.9, "best_cut": 3.9, "best_limit": 1.2890625, "missing": [],
                     "rows": [{"cut": 2.0, "limit": 1.7, "limitErr": 0.0, "best": False},
                              {"cut": 3.9, "limit": 1.2890625, "limitErr": 0.0, "best": True},
                              {"cut": 4.5, "limit": 1.5, "limitErr": 0.0, "best": False}]}


def test_summary_text_table(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9, 4.5])
    write_limit(tmp_path, 2.0, 1.7)
    write_limit(tmp_path, 3.9, 1.2890625)
    limit.summarize(cfg, tmp_path)
    lines = (tmp_path / "limits.txt").read_text().splitlines()
    assert lines[0] == "# median expected limit at CL 0.9; * = best cut"
    assert lines[1].split() == ["cut", "limit"]
    assert lines[2].split() == ["2.0", "1.7"]
    assert lines[3].split() == ["*", "3.9", "1.2890625"]
    assert lines[4].split() == ["4.5", "missing"]


def test_summary_missing_cut_warns(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9])
    write_limit(tmp_path, 3.9, 1.3)
    (tmp_path / "cut_2.0").mkdir()                       # job ran, Combine failed, no limit.json
    s = limit.summarize(cfg, tmp_path)
    assert s["missing"] == [2.0] and s["best_cut"] == 3.9
    assert s["rows"][0] == {"cut": 2.0, "limit": None, "limitErr": None, "best": False}
    assert "2.0" in caplog.text


def test_summary_nan_limit_is_missing(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9])
    write_limit(tmp_path, 2.0, math.nan)
    write_limit(tmp_path, 3.9, 1.3)
    s = limit.summarize(cfg, tmp_path)
    assert s["best_cut"] == 3.9 and s["missing"] == [2.0]


def test_summary_tie_takes_smaller_cut(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [4.5, 3.9])
    write_limit(tmp_path, 3.9, 1.3)
    write_limit(tmp_path, 4.5, 1.3)
    assert limit.summarize(cfg, tmp_path)["best_cut"] == 3.9


def test_summary_ignores_cuts_not_in_config(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    write_limit(tmp_path, 3.9, 1.3)
    write_limit(tmp_path, 5.0, 0.1)                      # old scan in the same phase_c/
    s = limit.summarize(cfg, tmp_path)
    assert [r["cut"] for r in s["rows"]] == [3.9] and s["best_cut"] == 3.9


def test_summary_no_limit_at_all_raises(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9])
    with pytest.raises(RuntimeError, match="2.0"):
        limit.summarize(cfg, tmp_path)
    assert not (tmp_path / "limits.json").exists()


def test_summary_broken_json_raises(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    (tmp_path / "cut_3.9").mkdir()
    (tmp_path / "cut_3.9" / "limit.json").write_text("{not json")
    with pytest.raises(ValueError, match="cut_3.9"):
        limit.summarize(cfg, tmp_path)


def test_summary_json_of_other_cut_raises(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    write_limit(tmp_path, 4.5, 1.3, folder_cut=3.9)
    with pytest.raises(ValueError, match="4.5"):
        limit.summarize(cfg, tmp_path)
```

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: ... has no attribute 'summarize'`)

Run: `$PYTEST/test_cut_scan.py -q`

- [ ] **Step 3: Implement** in `src/tau3mu_gnn/pipeline/stages/limit.py`

Add `import math` to the imports. Add after `INPUTS` / `REQUIRED_BRANCHES`:

```python
SUMMARY_FILES = {"json": "limits.json", "txt": "limits.txt", "png": "limits.png"}
```

Add at the end of the file:

```python
def summarize(cfg: Config, out_dir: str | Path) -> dict:
    """Table of the cut scan: reads <out_dir>/cut_<cut>/limit.json for each cut of limit.gnn_score_cut.

    A cut without a finite limit is a missing row (warning). Best cut = lowest limit, the smaller cut on a tie.
    Writes <out_dir>/limits.json, limits.txt, limits.png. Raises RuntimeError if no cut has a limit.
    """
    out_dir = Path(out_dir)
    rows = [_read_cut(out_dir, cut) for cut in sorted(cfg.limit.gnn_score_cut)]
    found = [r for r in rows if r["limit"] is not None]
    if not found:
        raise RuntimeError(f"no limit for any cut {sorted(cfg.limit.gnn_score_cut)} in {out_dir}; "
                           f"see the cut_<cut>/combine_*.log files")
    missing = [r["cut"] for r in rows if r["limit"] is None]
    if missing:
        log.warning("cut scan: no limit for cut(s) %s; see the cut_<cut>/combine_*.log files", missing)
    best = min(found, key=lambda r: r["limit"])
    for r in rows:
        r["best"] = r is best
    summary = {"cl": cfg.limit.cl, "best_cut": best["cut"], "best_limit": best["limit"],
               "missing": missing, "rows": rows}
    files = {k: out_dir / name for k, name in SUMMARY_FILES.items()}
    files["json"].write_text(json.dumps(summary, indent=2) + "\n")
    files["txt"].write_text(_table(summary))
    _plot(summary, files["png"])
    log.info("cut scan: best cut %s, limit %s -> %s", best["cut"], best["limit"], files["json"])
    return {**summary, "files": files}


def _read_cut(out_dir: Path, cut: float) -> dict:
    """One row from cut_<cut>/limit.json. No file or a non-finite limit -> limit None."""
    path = out_dir / cut_dir_name(cut) / "limit.json"
    if not path.is_file():
        return {"cut": cut, "limit": None, "limitErr": None}
    try:
        data = json.loads(path.read_text())
        row = {"cut": float(data["cut"]), "limit": float(data["limit"]), "limitErr": float(data["limitErr"])}
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path} is not a stage 5 limit.json: {e!r}") from e
    if row["cut"] != cut:
        raise ValueError(f"{path} is for cut {row['cut']}, expected {cut}")
    if not math.isfinite(row["limit"]):
        return {"cut": cut, "limit": None, "limitErr": None}
    return row


def _table(summary: dict) -> str:
    lines = [f"# median expected limit at CL {summary['cl']}; * = best cut", f"  {'cut':<10} limit"]
    for r in summary["rows"]:
        value = "missing" if r["limit"] is None else repr(r["limit"])
        lines.append(f"{'*' if r['best'] else ' '} {r['cut']!r:<10} {value}")
    return "\n".join(lines) + "\n"


def _plot(summary: dict, png: Path) -> None:
    """Task 3."""
```

- [ ] **Step 4: Run `$PYTEST/test_cut_scan.py $PYTEST/test_limit.py -q`, expect PASS**

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/stages/limit.py tests/pipeline/test_cut_scan.py
/usr/bin/git commit -m "feat(limit): cut scan summary table, best cut (step 8b)"
```

---

### Task 3: `limits.png`

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/stages/limit.py` (replace the `_plot` stub)
- Test: `tests/pipeline/test_cut_scan.py`

**Interfaces:**
- Consumes: the `summary` dict of Task 2 (`rows`, `best_cut`, `best_limit`, `missing`, `cl`).
- Produces: `_plot(summary, png) -> None`, writes a PNG.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_summary_plot(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9, 4.5])
    write_limit(tmp_path, 2.0, 1.7)
    write_limit(tmp_path, 3.9, 1.3)                      # 4.5 missing: drawn as a dotted line
    limit.summarize(cfg, tmp_path)
    png = (tmp_path / "limits.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000


def test_summary_plot_single_cut(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [3.9])                     # default config: one cut, still a valid plot
    write_limit(tmp_path, 3.9, 1.2890625)
    limit.summarize(cfg, tmp_path)
    assert (tmp_path / "limits.png").read_bytes()[:4] == b"\x89PNG"
```

- [ ] **Step 2: Run, expect FAIL** (stub writes no file → `FileNotFoundError`)

Run: `$PYTEST/test_cut_scan.py -k plot -q`

- [ ] **Step 3: Implement** (replace the stub)

```python
def _plot(summary: dict, png: Path) -> None:
    """Median expected limit vs GNN score cut. Best cut: star + text label. Missing cuts: gray dotted lines."""
    from matplotlib.figure import Figure  # no pyplot: no display backend needed on batch nodes

    found = [r for r in summary["rows"] if r["limit"] is not None]
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.plot([r["cut"] for r in found], [r["limit"] for r in found], "-o", color="#2a6fdb", linewidth=2,
            markersize=8)
    ax.plot([summary["best_cut"]], [summary["best_limit"]], "*", color="#2a6fdb", markersize=18,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"best: cut {summary['best_cut']}, limit {summary['best_limit']:.4g}",
                (summary["best_cut"], summary["best_limit"]), textcoords="offset points", xytext=(0, -22),
                ha="center", color="#222222", fontsize=9)
    for cut in summary["missing"]:
        ax.axvline(cut, color="#9a9a9a", linestyle=":", linewidth=1)
        ax.annotate("missing", (cut, 1), xycoords=("data", "axes fraction"), textcoords="offset points",
                    xytext=(3, -12), color="#6b6b6b", fontsize=8)
    ax.set_xlabel("GNN score cut")
    ax.set_ylabel(f"median expected limit ({summary['cl']:.0%} CL)")
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.1, y=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
```

- [ ] **Step 4: Run `$PYTEST/test_cut_scan.py -q`, expect PASS**

- [ ] **Step 5: Look at the plot.** Write a toy plot and open it with the Read tool (it shows images):

```bash
PYTHONPATH=src /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -c "
from pathlib import Path
from tau3mu_gnn.pipeline.stages.limit import _plot
rows=[{'cut':c,'limit':l,'limitErr':0.0,'best':c==3.9} for c,l in [(2.0,1.7),(3.0,1.45),(3.9,1.2890625),(4.5,1.5),(5.0,None)]]
_plot({'cl':0.9,'best_cut':3.9,'best_limit':1.2890625,'missing':[5.0],'rows':rows}, Path('/tmp/claude-828/-work-kfuss-SVJ-GNN/30324398-c75d-4a56-baff-e2157e9e63c0/scratchpad/limits_toy.png'))"
```

Check: no label overlap, star visible, "missing" label readable, axis labels not cut off. Fix and rerun if not.

- [ ] **Step 6: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/stages/limit.py tests/pipeline/test_cut_scan.py
/usr/bin/git commit -m "feat(limit): cut scan plot limits.png"
```

---

### Task 4: No stale `limit.json`; local `run_scan`

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/stages/limit.py` (`run`, new `run_scan`, module docstring)
- Test: `tests/pipeline/test_cut_scan.py`

**Interfaces:**
- Consumes: `run(cfg, inputs, out_dir, cut)`, `check_inputs(inputs, tree_name)`, `combine_executable()` (step 8), `summarize` (Task 2).
- Produces: `run_scan(cfg, inputs, out_dir) -> dict` (the `summarize` result). Step 9 does not use it; step 9 calls `run` per job and `summarize` in the final job.

- [ ] **Step 1: Write the failing tests** (append)

```python
import subprocess
import sys
from pathlib import Path

from test_limit import needs_combine, toy_inputs  # noqa: F401  (fixture reused)


def test_rerun_removes_old_limit_json(load, user, toy_inputs, monkeypatch, tmp_path):
    """A failed rerun must not leave the limit of the earlier run in the table."""
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    old = write_limit(tmp_path / "out", 3.9, 1.0)
    monkeypatch.setattr(limit, "combine_executable", lambda: Path("/bin/true"))
    monkeypatch.setattr(limit, "run_combine", lambda d, cl: Path(d) / "x.root")
    monkeypatch.setattr(limit, "read_limit", lambda p: {})
    with pytest.raises(RuntimeError):
        limit.run(cfg, {"signal": toy_inputs[0], "background": toy_inputs[1]}, tmp_path / "out", 3.9)
    assert not old.exists()


def fake_run_factory(failing):
    calls = []

    def fake_run(cfg, inputs, out_dir, cut):
        calls.append(cut)
        if cut in failing:
            raise subprocess.CalledProcessError(1, ["combine"])
        write_limit(Path(out_dir), cut, 1.0 + abs(cut - 3.9))
    return fake_run, calls


def test_run_scan_all_cuts_then_summary(load, user, monkeypatch, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9, 4.5])
    fake_run, calls = fake_run_factory(failing={4.5})
    monkeypatch.setattr(limit, "check_inputs", lambda inputs, tree: None)
    monkeypatch.setattr(limit, "combine_executable", lambda: Path("/bin/true"))
    monkeypatch.setattr(limit, "run", fake_run)

    s = limit.run_scan(cfg, {"signal": "s.root", "background": "b.root"}, tmp_path)

    assert calls == [2.0, 3.9, 4.5]                      # a failed cut does not stop the scan
    assert s["best_cut"] == 3.9 and s["missing"] == [4.5]
    assert (tmp_path / "limits.json").is_file()


def test_run_scan_no_combine_fails_before_any_cut(load, user, monkeypatch, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    def missing():
        raise FileNotFoundError("CMS Combine not found")

    fake_run, calls = fake_run_factory(failing=set())
    monkeypatch.setattr(limit, "check_inputs", lambda inputs, tree: None)
    monkeypatch.setattr(limit, "combine_executable", missing)
    monkeypatch.setattr(limit, "run", fake_run)
    with pytest.raises(FileNotFoundError, match="Combine"):
        limit.run_scan(scan_cfg(load, user, [2.0, 3.9]), {"signal": "s", "background": "b"}, tmp_path / "out")
    assert calls == [] and not (tmp_path / "out").exists()


def test_run_scan_other_errors_stop_the_scan(load, user, monkeypatch, tmp_path):
    """Only a Combine failure is a 'missing' cut; a bug (e.g. ValueError) must not be hidden."""
    from tau3mu_gnn.pipeline.stages import limit

    def broken(cfg, inputs, out_dir, cut):
        raise ValueError("bug")

    monkeypatch.setattr(limit, "check_inputs", lambda inputs, tree: None)
    monkeypatch.setattr(limit, "combine_executable", lambda: Path("/bin/true"))
    monkeypatch.setattr(limit, "run", broken)
    with pytest.raises(ValueError, match="bug"):
        limit.run_scan(scan_cfg(load, user, [3.9]), {"signal": "s", "background": "b"}, tmp_path)


@needs_combine
def test_run_scan_end_to_end(load, user, toy_inputs, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    user["limit"]["signal_weight"] = 0.5
    cfg = scan_cfg(load, user, [2.0, 3.9])
    s = limit.run_scan(cfg, {"signal": toy_inputs[0], "background": toy_inputs[1]}, tmp_path / "phase_c")
    for cut in (2.0, 3.9):
        per_cut = json.loads((tmp_path / "phase_c" / f"cut_{cut}" / "limit.json").read_text())
        row = next(r for r in s["rows"] if r["cut"] == cut)
        assert row["limit"] == per_cut["limit"]
    for name in limit.SUMMARY_FILES.values():
        assert (tmp_path / "phase_c" / name).is_file(), name
```

Note: `from test_limit import ... toy_inputs` works because `tests/pipeline` is on `sys.path` (conftest imports work the same way). If pytest warns about a fixture import, move `toy_stage3_file` + the `toy_inputs` fixture to `conftest.py` instead and import nothing.

- [ ] **Step 2: Run, expect FAIL** (`test_rerun_removes_old_limit_json`: old file still there; `run_scan` missing)

Run: `$PYTEST/test_cut_scan.py -q`

- [ ] **Step 3: Implement**

Add `import subprocess` to the imports.

In `run`, after `combine_executable()` and the `card_dir = ...` line, before `log.info(...)`:

```python
    (card_dir / "limit.json").unlink(missing_ok=True)  # a failed rerun must not leave the old limit
```

Add before `summarize`:

```python
def run_scan(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict:
    """All cuts of limit.gnn_score_cut one after the other in this process, then summarize. For local runs;
    the driver (step 9) sends one job per cut and a final summarize job instead.

    Missing inputs or a missing Combine stop before the first cut. A cut whose Combine fails (non-zero exit or
    no median limit) is logged and becomes a missing row. Other errors stop the scan.
    """
    check_inputs(inputs, cfg.inputs.tree_name)
    combine_executable()
    for cut in cfg.limit.gnn_score_cut:
        try:
            run(cfg, inputs, out_dir, cut)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            log.error("cut scan: cut %s failed: %s", cut, e)
    return summarize(cfg, out_dir)
```

Module docstring: replace the line `Step 8b runs one call per cut of limit.gnn_score_cut and writes the table.` with:

```
Cut scan (step 8b): one run() per cut of limit.gnn_score_cut, then summarize() writes <out_dir>/limits.json,
limits.txt and limits.png (best cut = lowest limit). run_scan() does both in one process.
```

- [ ] **Step 4: Run the full suite `$PYTEST -q`, expect PASS** (164 + new tests; the Combine tests run in the Tau3Mu env)

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/stages/limit.py tests/pipeline/test_cut_scan.py
/usr/bin/git commit -m "feat(limit): local run_scan, drop stale limit.json on rerun"
```

---

### Task 5: Roadmap

**Files:**
- Modify: `docs/wrapper_roadmap.md` (step list line for 8b, sec. 5 layout, decision log)

- [ ] **Step 1: Edit**
  - Step list, line `8b. Cut scan: ...`: prefix `✅` and append: `Done: `summarize` (`limits.json` / `limits.txt` / `limits.png` in `phase_c/`), `run_scan` (local loop). Missing cut = `missing` row, best from the rest. SLURM fan-out (one job per cut, summary job with `afterany`) → step 9.`
  - Sec. 5 layout, line `└── limits.json ...`: replace with three lines `limits.json`, `limits.txt`, `limits.png` (`# cut → limit table + plot, best cut marked`).
  - Step 9 line: append `Phase C: one job per cut (stage 5 `run`), then `summarize` with `--dependency=afterany`.`
  - Decision log, new row `| 2026-10-12 |`: `Step 8b decisions (Kaito): (1) missing cut → row marked, best from the rest; raise only if no cut has a limit (summary job uses afterany in step 9). (2) Outputs limits.json + limits.txt + limits.png. (3) Regression: tests only (no physics change). Also: duplicate cuts rejected in the config; stage 5 removes an old limit.json before a rerun; NaN limit = missing; tie → smaller cut.`

- [ ] **Step 2: Run `$PYTEST -q`, expect PASS**

- [ ] **Step 3: Commit**

```bash
/usr/bin/git add docs/wrapper_roadmap.md docs/superpowers/plans/2026-10-12-step8b-cut-scan.md
/usr/bin/git commit -m "docs(roadmap): step 8b done"
```
