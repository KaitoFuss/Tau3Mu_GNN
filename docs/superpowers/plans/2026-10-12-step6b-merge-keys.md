# Mini Step 6b — Merge Scores on (run, luminosityBlock, event) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The BDT (stage 2) and GNN (stage 3) write their score back to each event by `(run, luminosityBlock, event)`, not by `(run, event)`. Duplicate keys are an error. A regression job proves that only the 52 W signal events with a duplicate `(run, event)` change.

**Architecture:** One new shared module `utils/score_merge.py`. It has three parts:
- `scores_in_entry_order(...)`: a pandas left join on the three keys. It gives one score per input entry, in entry order.
- `write_with_score(...)`: the ROOT copy loop (`CloneTree(0)` + `Fill`) that is in both producers today, but it takes the ready score array. No temp `_scores_only.root` file, no `BuildIndex`, no `iterrows`.
- `check_unique_keys(...)`: reads only the three key branches of a file and raises `DuplicateEventError`.

Both `_write_with_scores` functions become thin calls to this module. Stages 2 and 3 call `check_unique_keys` in `check_inputs`, so a bad file fails before hours of work. The output file format does not change: same branches, same `/F` score branch, same entry order.

**Tech Stack:** Python 3.11 (Tau3Mu env), pandas 3.0, uproot 5.7, PyROOT 6.34, pytest.

**Spec:** `docs/wrapper_roadmap.md` sec. 6 step 6b, and E2. Measured facts: W signal (`old_data/signal_skimmed_no_resonances.root`, 801,732 entries, run = 1) has 26 `(run, event)` pairs (52 entries) that differ only in lumi. `(run, lumi, event)` is unique. W data (10,855,317 entries) has no duplicates.

## Global Constraints

- Python: `/work/kfuss/miniforge3/envs/Tau3Mu/bin/python`. Test command (`$PYTEST`): `PYTHONPATH=/work/kfuss/SVJ_GNN/.claude/worktrees/pipeline-wrapper/src /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m pytest tests/pipeline`
- Branch `pipeline-step6b-merge-keys`, from `main` after the merge of PR #15 (`4a65986`).
- Git in this worktree: `/usr/bin/git` only. One git command per Bash call.
- Importing ROOT on the login node prints a "cannot extract standard library include paths" error. It is noise.
- Merge key: `KEYS = ["run", "luminosityBlock", "event"]`, in this order.
- Events without a score get `-1.` (unchanged). A real `NaN` score stays `NaN` (do **not** use `fillna`).
- Score branch type stays `/F` (`bdt_ISO/F`, `GNN_score/F`).
- BDT: the input `GNN_score` branch is still dropped from the output (unchanged).
- Do not touch the scoring math or which events are scored.
- `gnn_training/Parallel_ScoreProducer.py`: no file imports it. Do **not** change it in this step (roadmap E2 says: fix or delete later).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Regression jobs: write the script, tell Kaito, Kaito runs `sbatch`. Never submit yourself.

## Review Focus

1. **Two events with the same `(run, event)` but a different lumi** → each event gets its own score. Test in Task 2 (BDT) and Task 3 (GNN).
2. **Input file with two entries that have the same `(run, lumi, event)`** → `DuplicateEventError` from the stage `check_inputs`, before any output file exists. Test in Task 4.
3. **A scored event with a `NaN` score** → `NaN` stays `NaN` in the output, it does not become `-1`. Test in Task 1.
4. **An event without a score** → `-1`. Test in Task 1 (helper) and the existing stage tests.
5. **Key dtypes differ** (input `run` is `uint32`, the score table has `int64` from pandas) → still matches. Test in Task 1.

## File Structure

| File | Responsibility |
|---|---|
| `src/tau3mu_gnn/utils/score_merge.py` (create) | `KEYS`, `DuplicateEventError`, `check_unique`, `check_unique_keys`, `scores_in_entry_order`, `write_with_score` |
| `src/tau3mu_gnn/bdt_training/BDT_ScoreProducer.py` (modify) | `_write_with_scores` uses `score_merge` |
| `src/tau3mu_gnn/gnn_training/ScoreProducer.py` (modify) | `_write_with_scores` uses `score_merge` |
| `src/tau3mu_gnn/pipeline/stages/bdt.py`, `gnn.py` (modify) | `check_inputs` also calls `check_unique_keys` |
| `tests/pipeline/test_score_merge.py` (create) | helper tests (pandas + uproot, ROOT only for the writer) |
| `tests/pipeline/test_bdt_scores.py`, `test_gnn_scores.py` (modify) | duplicate `(run, event)` test, duplicate key test |
| `regression/step6b/run_regression_6b.sh`, `compare_6b.py` (create, not committed) | old vs new on W subsets that contain all 52 events |
| `docs/wrapper_roadmap.md` (modify) | step 6b done + decision log |

---

### Task 1: `utils/score_merge.py`

**Files:**
- Create: `src/tau3mu_gnn/utils/score_merge.py`
- Test: `tests/pipeline/test_score_merge.py`

**Interfaces:**
- Produces:
  - `KEYS: list[str]`
  - `class DuplicateEventError(ValueError)`
  - `check_unique(keys: pd.DataFrame, what: str) -> None`
  - `check_unique_keys(path: str | Path, tree_name: str) -> None`
  - `scores_in_entry_order(entry_keys: pd.DataFrame, score_keys: pd.DataFrame, scores, default: float = -1.0) -> np.ndarray` (float64, one value per row of `entry_keys`)
  - `write_with_score(in_file, out_file, tree_name: str, branch: str, values: np.ndarray, drop: Sequence[str] = ()) -> None`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
import pytest
import uproot

from conftest import write_tree
from tau3mu_gnn.utils.score_merge import (DuplicateEventError, check_unique_keys, scores_in_entry_order,
                                          write_with_score)


def _keys(run, lumi, event, dtype=np.uint32):
    return pd.DataFrame({"run": np.array(run, dtype=dtype), "luminosityBlock": np.array(lumi, dtype=dtype),
                         "event": np.array(event, dtype=np.uint64)})


def test_lumi_separates_events_with_same_run_and_event():
    entries = _keys([1, 1, 1], [5, 6, 5], [10, 10, 11])
    scores = _keys([1, 1, 1], [6, 5, 5], [10, 10, 11])
    np.testing.assert_array_equal(scores_in_entry_order(entries, scores, [0.6, 0.5, 0.7]), [0.5, 0.6, 0.7])


def test_missing_score_gets_default_and_nan_stays_nan():
    entries = _keys([1, 1, 1], [1, 1, 1], [1, 2, 3])
    scores = _keys([1, 1], [1, 1], [3, 1])
    out = scores_in_entry_order(entries, scores, [np.nan, 0.2])
    assert out[0] == 0.2 and out[1] == -1.0 and np.isnan(out[2])


def test_key_dtypes_may_differ():
    entries = _keys([1], [2], [3])
    scores = _keys([1], [2], [3], dtype=np.int64)
    np.testing.assert_array_equal(scores_in_entry_order(entries, scores, [0.9]), [0.9])


@pytest.mark.parametrize("dup_in", ["entries", "scores"])
def test_duplicate_keys_raise(dup_in):
    unique, dup = _keys([1, 1], [1, 1], [1, 2]), _keys([1, 1], [1, 1], [1, 1])
    entries, scores = (dup, unique) if dup_in == "entries" else (unique, dup)
    with pytest.raises(DuplicateEventError, match="run, luminosityBlock, event"):
        scores_in_entry_order(entries, scores, [0.1, 0.2])


def test_check_unique_keys_reads_file(tmp_path):
    ok = write_tree(tmp_path / "ok.root", {"run": np.ones(2, np.uint32), "luminosityBlock": np.array([1, 2], np.uint32),
                                           "event": np.array([7, 7], np.uint64)})
    check_unique_keys(ok, "Events")
    bad = write_tree(tmp_path / "bad.root", {"run": np.ones(2, np.uint32), "luminosityBlock": np.ones(2, np.uint32),
                                             "event": np.array([7, 7], np.uint64)})
    with pytest.raises(DuplicateEventError, match="bad.root"):
        check_unique_keys(bad, "Events")


def test_write_with_score_keeps_entries_and_branches(tmp_path):
    src = write_tree(tmp_path / "in.root", {"x": np.arange(3.0), "drop_me": np.zeros(3)}, tree="T")
    out = tmp_path / "sub" / "out.root"
    write_with_score(src, out, "T", "s", np.array([0.5, -1.0, np.nan]), drop=["drop_me", "not_there"])
    tree = uproot.open(out)["T"]
    assert set(tree.keys()) == {"x", "s"}
    assert tree["s"].typename == "float"
    np.testing.assert_array_equal(tree["x"].array(library="np"), [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(tree["s"].array(library="np"), np.float32([0.5, -1.0, np.nan]))


def test_write_with_score_wrong_length(tmp_path):
    src = write_tree(tmp_path / "in.root", {"x": np.arange(3.0)})
    with pytest.raises(ValueError, match="3 entries"):
        write_with_score(src, tmp_path / "out.root", "Events", "s", np.zeros(2))
```

Note: `assert_array_equal` treats `NaN == NaN` as equal.

- [ ] **Step 2: Run the tests, expect FAIL** (`ModuleNotFoundError: tau3mu_gnn.utils.score_merge`)

Run: `$PYTEST tests/pipeline/test_score_merge.py -v`

- [ ] **Step 3: Write the module**

```python
"""Write a score back to its event, matched by (run, luminosityBlock, event).

Shared by the BDT (stage 2) and GNN (stage 3) score producers. (run, event) alone is not unique:
in MC the run is always 1, and events from different lumi blocks can have the same event number.
"""
from __future__ import annotations

import os
from array import array
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import uproot

KEYS = ["run", "luminosityBlock", "event"]


class DuplicateEventError(ValueError):
    """Two entries have the same (run, luminosityBlock, event)."""


def check_unique(keys: pd.DataFrame, what: str) -> None:
    dup = keys.duplicated(KEYS, keep=False)
    if dup.any():
        examples = keys.loc[dup, KEYS].drop_duplicates().head(5).to_dict("records")
        raise DuplicateEventError(f"{what}: {int(dup.sum())} entries share (run, luminosityBlock, event) "
                                  f"with another entry, e.g. {examples}")


def check_unique_keys(path: str | Path, tree_name: str) -> None:
    """Raise DuplicateEventError if two entries of path:tree_name have the same key. Reads the key branches only."""
    with uproot.open(path) as f:
        keys = f[tree_name].arrays(KEYS, library="pd")
    check_unique(keys, str(path))


def scores_in_entry_order(entry_keys: pd.DataFrame, score_keys: pd.DataFrame, scores,
                          default: float = -1.0) -> np.ndarray:
    """Score of each row of entry_keys, in the same order; default where there is no score."""
    check_unique(entry_keys, "input tree")
    check_unique(score_keys, "scores")
    left = entry_keys[KEYS].astype("uint64").reset_index(drop=True)
    right = score_keys[KEYS].astype("uint64").reset_index(drop=True)
    right["_score"] = np.asarray(scores, dtype=np.float64)
    merged = left.merge(right, on=KEYS, how="left", sort=False, indicator=True, validate="one_to_one")
    return np.where(merged["_merge"] == "both", merged["_score"], default)


def write_with_score(in_file, out_file, tree_name: str, branch: str, values: np.ndarray,
                     drop: Sequence[str] = ()) -> None:
    """Copy every entry and branch of in_file:tree_name (except drop) to out_file, plus branch/F = values."""
    import ROOT

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    fin = ROOT.TFile.Open(str(in_file))
    t1 = fin.Get(tree_name)
    n = t1.GetEntries()
    if n != len(values):
        fin.Close()
        raise ValueError(f"{in_file} has {n} entries, but {len(values)} values for {branch}")
    dropped = [name for name in drop if t1.GetBranch(name)]
    for name in dropped:
        t1.SetBranchStatus(name, 0)

    fout = ROOT.TFile.Open(str(out_file), "RECREATE")
    t_out = t1.CloneTree(0)  # empty tree
    for name in dropped:
        t1.SetBranchStatus(name, 1)
    buf = array("f", [0.])
    t_out.Branch(branch, buf, f"{branch}/F")
    for i in range(n):
        t1.GetEntry(i)
        buf[0] = values[i]
        t_out.Fill()
    t_out.Write()
    fout.Close()
    fin.Close()
```

- [ ] **Step 4: Run the tests, expect PASS**

Run: `$PYTEST tests/pipeline/test_score_merge.py -v`

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/utils/score_merge.py tests/pipeline/test_score_merge.py
/usr/bin/git commit -m "feat(scores): shared merge on (run, lumi, event)"
```

---

### Task 2: BDT writes scores with `score_merge`

**Files:**
- Modify: `src/tau3mu_gnn/bdt_training/BDT_ScoreProducer.py` (`_load` docstring `:33`, `_write_with_scores` `:141-234`)
- Test: `tests/pipeline/test_bdt_scores.py`

**Interfaces:**
- Consumes: `KEYS`, `scores_in_entry_order`, `write_with_score` from Task 1.
- Produces: `_write_with_scores(df, in_file, out_file, tree_name)` (same signature).

- [ ] **Step 1: Write the failing test** (append to `tests/pipeline/test_bdt_scores.py`)

```python
def test_same_run_and_event_different_lumi(tmp_path, load, user):
    """Signal MC: run is always 1 and event numbers repeat across lumi blocks."""
    sig = _events([[100.0], [300.0], [500.0]])
    sig["luminosityBlock"] = np.array([1, 2, 1], dtype=np.uint32)
    sig["event"] = np.array([10, 10, 11], dtype=np.uint64)
    outputs, _ = _run(tmp_path, load, user, sig_data=sig)
    scores = uproot.open(outputs["signal"])["Events"]["bdt_ISO"].array(library="np")
    np.testing.assert_allclose(scores, [0.1, 0.3, 0.5], rtol=1e-6)
```

- [ ] **Step 2: Run it, expect FAIL** (old code gives both events with `event = 10` the same score)

Run: `$PYTEST tests/pipeline/test_bdt_scores.py::test_same_run_and_event_different_lumi -v`

- [ ] **Step 3: Replace `_write_with_scores`**

```python
def _write_with_scores(df, in_file, out_file, tree_name):
    """Copy in_file to out_file (without GNN_score) and add bdt_ISO, matched by (run, luminosityBlock, event)."""
    print(f"Processing {in_file}")
    with uproot.open(in_file) as f:
        entry_keys = f[tree_name].arrays(KEYS, library="pd")
    scores = scores_in_entry_order(entry_keys, df[KEYS], df["bdt_ISO"])
    write_with_score(in_file, out_file, tree_name, "bdt_ISO", scores, drop=["GNN_score"])
    print(f"Wrote in total {int(np.sum(scores != -1.))} valid BDT scores in to ROOT file {out_file}")
```

Imports: add `from tau3mu_gnn.utils.score_merge import KEYS, scores_in_entry_order, write_with_score`. Remove `from array import array` if nothing else uses it. In `_load` change the docstring to "One row per event (first candidate), so the (run, luminosityBlock, event) keys in _write_with_scores are unique."

Note: `df` holds the rows of **all** files of the sample. Per file, rows of other files simply do not match (as before). Keys of different files that are equal raise `DuplicateEventError` (new, wanted).

- [ ] **Step 4: Run all BDT tests, expect PASS**

Run: `$PYTEST tests/pipeline/test_bdt_scores.py -v`

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/bdt_training/BDT_ScoreProducer.py tests/pipeline/test_bdt_scores.py
/usr/bin/git commit -m "fix(bdt): merge bdt_ISO on (run, lumi, event)"
```

---

### Task 3: GNN writes scores with `score_merge`

**Files:**
- Modify: `src/tau3mu_gnn/gnn_training/ScoreProducer.py` (`_write_with_scores` `:311-~390`)
- Test: `tests/pipeline/test_gnn_scores.py`

**Interfaces:**
- Consumes: `KEYS`, `scores_in_entry_order`, `write_with_score` from Task 1.
- Produces: `_write_with_scores(score_dict, in_file, out_file, tree_name)` (same signature; `score_dict` is `(run, lumi, event) -> score`, as today).

- [ ] **Step 1: Write the failing test** (append to `tests/pipeline/test_gnn_scores.py`)

```python
def test_same_run_and_event_different_lumi(tmp_path):
    data = gnn_events([[100.0], [300.0], [500.0]], [0.9, 0.9, 0.9])
    data["luminosityBlock"] = np.array([1, 2, 1], dtype=np.uint32)
    data["event"] = np.array([10, 10, 11], dtype=np.uint64)
    (out,), npz = _score(tmp_path, [data])
    np.testing.assert_array_equal(uproot.open(out)["Events"]["GNN_score"].array(library="np"),
                                  np.float32([100.0, 300.0, 500.0]))
    np.testing.assert_allclose(npz["scores"], [100.0, 300.0, 500.0])
```

- [ ] **Step 2: Run it, expect FAIL**

Run: `$PYTEST tests/pipeline/test_gnn_scores.py::test_same_run_and_event_different_lumi -v`

- [ ] **Step 3: Replace `_write_with_scores`**

```python
def _write_with_scores(score_dict, in_file, out_file, tree_name):
    """Copy in_file to out_file and add GNN_score from score_dict ((run, lumi, event) -> score)."""
    with uproot.open(in_file) as f:
        entry_keys = f[tree_name].arrays(KEYS, library="pd")
    score_keys = pd.DataFrame(list(score_dict), columns=KEYS)
    scores = scores_in_entry_order(entry_keys, score_keys, list(score_dict.values()))
    write_with_score(in_file, out_file, tree_name, "GNN_score", scores)
    print(f"Wrote in total {int(np.sum(scores != -1.))} valid GNN Scores in to ROOT file {out_file}")
```

`pd.DataFrame(list(score_dict), columns=KEYS)` with an empty dict gives an empty frame with the right columns. Add `import pandas as pd` and `from tau3mu_gnn.utils.score_merge import KEYS, scores_in_entry_order, write_with_score`. Remove imports that are now unused **only** if nothing else in the file uses them (`array` is used elsewhere? check with grep before removing).

- [ ] **Step 4: Run all GNN tests, expect PASS**

Run: `$PYTEST tests/pipeline/test_gnn_scores.py -v`

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/gnn_training/ScoreProducer.py tests/pipeline/test_gnn_scores.py
/usr/bin/git commit -m "fix(gnn): merge GNN_score on (run, lumi, event)"
```

---

### Task 4: Stages 2 and 3 check the keys first

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/stages/bdt.py` (`check_inputs`), `src/tau3mu_gnn/pipeline/stages/gnn.py` (`check_inputs`)
- Test: `tests/pipeline/test_bdt_scores.py`, `tests/pipeline/test_gnn_scores.py`

**Interfaces:**
- Consumes: `check_unique_keys(path, tree_name)` from Task 1.

- [ ] **Step 1: Write the failing tests**

In `test_bdt_scores.py`:

```python
def test_duplicate_event_keys_fail_before_output(tmp_path, load, user):
    from tau3mu_gnn.utils.score_merge import DuplicateEventError

    sig = _events(SIG_PTS)
    sig["event"] = np.array([10, 10, 11], dtype=np.uint64)  # lumi is 1 everywhere
    with pytest.raises(DuplicateEventError, match="signal.root"):
        _run(tmp_path, load, user, sig_data=sig)
    assert not (tmp_path / "2_bdt").exists()
```

In `test_gnn_scores.py`, copy the pattern of `test_missing_branch_fails_before_output` (`:215`): input data from `gnn_events(CANDS, BDT, PF_ETAS)` with `data["event"][:2] = 10`, expect `DuplicateEventError` from `gnn.run`, and the stage `out_dir` must not exist.

- [ ] **Step 2: Run them, expect FAIL**

- [ ] **Step 3: Add the check** at the end of both `check_inputs` functions (after the branch check, so a missing key branch gives the clear `BranchError` first):

```python
    for sample in SAMPLES:
        check_unique_keys(inputs[sample], tree_name)
```

Import: `from tau3mu_gnn.utils.score_merge import check_unique_keys`. Docstrings: add "or two events have the same (run, luminosityBlock, event)".

- [ ] **Step 4: Run all tests, expect PASS**

Run: `$PYTEST -v`

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/stages/bdt.py src/tau3mu_gnn/pipeline/stages/gnn.py tests/pipeline/test_bdt_scores.py tests/pipeline/test_gnn_scores.py
/usr/bin/git commit -m "feat(pipeline): stages 2 and 3 fail early on duplicate event keys"
```

---

### Task 5: Regression job (Kaito queues it)

**Files (not committed, `regression/` is untracked):** `regression/step6b/run_regression_6b.sh`, `regression/step6b/compare_6b.py`. Pattern: `regression/step6/run_regression_6.sh`.

**What it proves:** old code (`main` at `4a65986`) vs new worktree, BDT stage and GNN stage, same inputs. Only events in a duplicate `(run, event)` group may change, and the new scores are consistent with the pickle / npz.

- [ ] **Step 1: Inputs (subsets, so the job is short)**
  - BDT input: `$D/signal_W/benchmark_W_BDT_vs1/selected_signal_skimmed_no_resonances_with_scores.root` and `$D/background_W/benchmark_W_BDT_vs1/selected_data_skimmed_no_resonances_with_scores.root` (same as step 5b).
  - GNN input: `$D/signal_W/bdt_full/selected_signal_skimmed_no_resonances_with_scores_bdt_full.root` and `$D/background_W/bdt_full/selected_data_skimmed_no_resonances_with_scores_bdt_full.root` (same as step 6).
  - Signal subset per file: the first 5,000 entries **plus** every entry whose `(run, event)` is in a duplicate group (found with uproot on the key branches; expect 52). Write with `TTree::CopyTree` on an entry list (`ROOT.TEntryList`), keep the entry order.
  - Background subset: the first 200,000 entries.
  - The GNN input already has `GNN_score` → stage 3 refuses it. Drop it in the subset copy (`SetBranchStatus("GNN_score", 0)` before `CopyTree`). Same for the BDT input? No: stage 2 drops it itself.
- [ ] **Step 2: Run** old and new code from frozen copies (`git archive 4a65986 src configs` and `cp -r $WT/src $WT/configs $WT/models`), each through the stage functions: `bdt.run(cfg, {...}, out)` then `gnn.run(cfg, {...}, out)` on the GNN subsets. `cfg = load_config(<copy>/configs/example_W.yaml)`. Old stage 2/3 `check_inputs` has no key check, so it runs on the same inputs. SBATCH: `-p standard --account=t3 --mem=32G --time=6:00:00`, logs in `regression/step6b/logs/`, scratch `/scratch/$USER/$SLURM_JOB_ID` (trap removes it).
- [ ] **Step 3: `compare_6b.py`** per stage and sample:
  - same entries, same branch set; every branch except the score equal (one branch at a time; float: equal or both NaN).
  - `dup` = entries in a duplicate `(run, event)` group. Score equal on all entries **not** in `dup`.
  - **Old is wrong, new is right:** in each `dup` group, old gives every entry the same score. New: BDT `bdt_ISO[entry] == pickle ISO_score_{sig,bkg}[entry]` for every scored entry (the pickle Series index is the entry number; one input file per sample). GNN: `GNN_score[GNN_score != -1]` equals the npz `scores`, in order.
  - Report counts: `dup` entries, changed entries (must all be in `dup`), consistency checks. `RESULT: PASS` or `FAIL`. Write `regression/step6b/report_${SLURM_JOB_ID}.txt`.
- [ ] **Step 4: Smoke test locally** with `SIG_N=200 BKG_N=2000 SCRATCH_BASE=/tmp/claude-...` (as for step 6), then tell Kaito the `sbatch` command.

---

### Task 6: Roadmap

- [ ] **Step 1:** In `docs/wrapper_roadmap.md` sec. 6, mark 6b `✅` **only after the Task 5 regression passes**. Decision-log rows:
  - `| 2026-10-12 | Step 6b: shared utils/score_merge.py. BDT and GNN merge scores on (run, luminosityBlock, event) with a pandas join (no BuildIndex, no temp file, no iterrows); the ROOT copy loop stays. Stages 2 and 3 fail early (DuplicateEventError) if two input events have the same key. Regression job <id>: ... |`
  - E2: the join part is done in 6b. What is left: the per-event PyROOT copy loop. Update the E2 text.
- [ ] **Step 2: Commit**

```bash
/usr/bin/git add docs/wrapper_roadmap.md docs/superpowers/plans/2026-10-12-step6b-merge-keys.md
/usr/bin/git commit -m "docs(roadmap): step 6b done"
```
