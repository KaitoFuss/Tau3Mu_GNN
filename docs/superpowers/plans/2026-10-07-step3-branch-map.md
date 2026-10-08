# Mini Step 3 — Required Branches + Branch Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define the raw input branches the W pipeline needs, check a user's input files against them (after the user's branch map), and provide an RDataFrame helper that renames the user's branches to the analysis names.

**Architecture:** One module `src/tau3mu_gnn/pipeline/branches.py`. A shared `COMMON` tuple plus `PER_CHANNEL[channel]` tuples give `required_branches(channel)`. `check_input_branches` validates the map and reads only branch **names** of every input file with uproot (no event loop), collecting all problems into one `BranchError`. `rename_columns` adds `Define(analysis_name, user_name)` to an RDataFrame and returns the column list for `Snapshot` without the user's original names. Nothing calls these yet; step 4 (skimmer) and step 9 (driver) will.

**Tech Stack:** Python 3.11 (Tau3Mu env), uproot 5.7.1, awkward, ROOT 6.34.04 (RDataFrame), pytest 9.1.1.

**Spec:** `docs/wrapper_roadmap.md` (sec. 5 "Branch map" + "Required branches", sec. 6 step 3, decision log).

## Global Constraints

- Python: `/work/kfuss/miniforge3/envs/Tau3Mu/bin/python`. Test command (`$PYTEST`): `PYTHONPATH=/work/kfuss/SVJ_GNN/.claude/worktrees/pipeline-wrapper/src /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m pytest`
- Git in this worktree: `/usr/bin/git` only. One git command per Bash call. No heredocs / `source` (worktree guard). Edit files with Edit/Write.
- Importing ROOT on the login node prints a "cannot extract standard library include paths" error block. It is noise: RDataFrame works. Do not try to fix it. Import ROOT **inside** the test functions that need it, never at module level.
- Branch-map direction: **analysis (Kaito) name → user name**, as in `Config.branch_map` (a read-only mapping).
- Required raw branches (38 for W), all checked present in Kaito's real W files on 2026-10-07:
  - COMMON (26): `Slimmed_TauTo3Mu_{dZmu12,dZmu13,dZmu23,CosAlpha2D_LxyP3mu}`, `Slimmed_TauTo3Mu_fitted_{pt,eta,phi,mass}`, `Slimmed_TauTo3Mu_mu{1,2,3}_{eta,phi}`, `Slimmed_PFCands_{pt,eta,phi,mass,pdgId,charge,puppiWeight,d0,dz}`, `run`, `luminosityBlock`, `event`
  - W (12): `Slimmed_TauTo3Mu_{sigLxy_3muVtxBS,vtx_chi2,vtx_Ndof,iso_ptChargedFromPV,iso_ptPhotons,iso_ptChargedFromPU}`, `TauPlusMET_{Tau_Puppi_mT,PuppiMET_pt,Puppi_pt,Puppi_phi,PuppiMETminPz,PuppiMETmaxPz}`
- Missing required branch → error listing **all** missing branches of **all** files. No fallbacks.
- Unknown channel → error (no silent W check for HF).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Glob that matches nothing** (`/pnfs/.../*.root` typo) — expect an error naming the pattern, not "0 files checked, OK". Test in Task 2.
2. **Typo in a map key** (`Slimmed_PFCands_ptt: PFCand_pt`) — expect "unknown analysis branch" + hint, not a silently ignored entry. Test in Task 2.
3. **Unreadable file or wrong tree name** — expect one line per file in the error, not a traceback that stops the other files' checks. Tests in Task 2.
4. **Map target that is itself a required analysis name** (`Slimmed_TauTo3Mu_fitted_pt: Slimmed_TauTo3Mu_fitted_eta`) — `Define` would fail later with an obscure ROOT error; expect a clear map error now. Test in Task 2.
5. **Renamed column must keep its values and the user's original must not be written twice** — test with a real RDataFrame `Snapshot` in Task 3.
6. **Per-candidate branches with inconsistent lengths, or stored flat** — the skimmer mask would fail later with an obscure error; expect a clear structure error naming file, branch and first bad entry. Tests in Task 4. Families are checked **separately**: `TauPlusMET_*` vs `Slimmed_TauTo3Mu_*` already differ in 0.64% of Kaito's signal events, so no cross-family check.

## File Structure

| File | Responsibility |
|---|---|
| `src/tau3mu_gnn/pipeline/branches.py` (create) | `COMMON`, `PER_CHANNEL`, `BranchError`, `required_branches`, `check_input_branches`, `rename_columns` |
| `tests/pipeline/conftest.py` (modify) | `make_root_file` helper |
| `tests/pipeline/test_branches.py` (create) | all tests of this step |
| `docs/wrapper_roadmap.md` (modify) | correct PF branch family, record findings, mark step 3 done |

---

### Task 1: Required-branch lists

**Files:**
- Create: `src/tau3mu_gnn/pipeline/branches.py`
- Test: `tests/pipeline/test_branches.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `COMMON: tuple[str, ...]`, `PER_CHANNEL: dict[str, tuple[str, ...]]`, `BranchError(problems: list[str])` with `.problems`, `required_branches(channel: str) -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests `tests/pipeline/test_branches.py`**

```python
import os

import pytest
import uproot

from tau3mu_gnn.pipeline import branches as b

W_FILES = [
    "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/old_data/signal_skimmed_no_resonances.root",
    "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/old_data/data_skimmed_no_resonances.root",
]


def test_required_branches_w_is_common_plus_w():
    req = b.required_branches("W")
    assert len(req) == 38
    assert len(set(req)) == 38
    assert req[: len(b.COMMON)] == b.COMMON
    assert "Slimmed_PFCands_pt" in req
    assert "TauPlusMET_Puppi_phi" in req
    assert "PFCands_pt" not in req


def test_required_branches_unknown_channel():
    with pytest.raises(b.BranchError, match="no required-branch list for channel 'HF'"):
        b.required_branches("HF")


def test_branch_error_lists_all_problems():
    err = b.BranchError(["one", "two"])
    assert err.problems == ["one", "two"]
    assert "one" in str(err) and "two" in str(err)


@pytest.mark.parametrize("path", W_FILES)
def test_real_w_files_have_all_required_branches(path):
    if not os.path.exists(path):
        pytest.skip(f"not reachable on this machine: {path}")
    keys = set(uproot.open(path)["Events"].keys())
    assert [x for x in b.required_branches("W") if x not in keys] == []
```

- [ ] **Step 2: Run, expect FAIL**

Run: `$PYTEST tests/pipeline/test_branches.py -q`
Expected: collection error `cannot import name 'branches' from 'tau3mu_gnn.pipeline'`.

- [ ] **Step 3: Create `src/tau3mu_gnn/pipeline/branches.py`**

```python
"""Required raw input branches and the user's branch map.

Names here are the analysis names in the raw input files (before stage 1).
The user config's branch_map maps analysis name -> name in the user's files.
"""
from __future__ import annotations

_TAU = "Slimmed_TauTo3Mu_"
_PF = "Slimmed_PFCands_"

# Read by every channel.
COMMON: tuple[str, ...] = (
    # stage 1 skim cut
    *(f"{_TAU}{s}" for s in ("dZmu12", "dZmu13", "dZmu23", "CosAlpha2D_LxyP3mu")),
    # tau candidate: choice, cone centre, mass fit
    *(f"{_TAU}fitted_{s}" for s in ("pt", "eta", "phi", "mass")),
    # muon directions: GNN cone and delete_muons
    *(f"{_TAU}mu{i}_{s}" for i in (1, 2, 3) for s in ("eta", "phi")),
    # GNN node features
    *(f"{_PF}{s}" for s in ("pt", "eta", "phi", "mass", "pdgId", "charge", "puppiWeight", "d0", "dz")),
    # event id: merging scores back into the trees
    "run", "luminosityBlock", "event",
)

# BDT ISO features and GNN graph-level features differ per channel.
PER_CHANNEL: dict[str, tuple[str, ...]] = {
    "W": (
        *(f"{_TAU}{s}" for s in ("sigLxy_3muVtxBS", "vtx_chi2", "vtx_Ndof",
                                 "iso_ptChargedFromPV", "iso_ptPhotons", "iso_ptChargedFromPU")),
        *(f"TauPlusMET_{s}" for s in ("Tau_Puppi_mT", "PuppiMET_pt", "Puppi_pt", "Puppi_phi",
                                      "PuppiMETminPz", "PuppiMETmaxPz")),
    ),
}


class BranchError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("Input branch problems:\n" + "\n".join(f"  - {p}" for p in self.problems))


def required_branches(channel: str) -> tuple[str, ...]:
    if channel not in PER_CHANNEL:
        raise BranchError([f"no required-branch list for channel {channel!r} (known: {sorted(PER_CHANNEL)})"])
    return COMMON + PER_CHANNEL[channel]
```

- [ ] **Step 4: Run, expect PASS**

Run: `$PYTEST tests/pipeline/test_branches.py -q`
Expected: 5 passed (the 2 real-file tests pass on T3; they skip where `/pnfs` is not mounted).

- [ ] **Step 5: Commit** (one command per call)

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/branches.py tests/pipeline/test_branches.py
/usr/bin/git commit -m "feat(pipeline): required raw input branches per channel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `check_input_branches` (map validation + per-file name check)

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/branches.py` (append)
- Modify: `tests/pipeline/conftest.py` (append helper)
- Test: `tests/pipeline/test_branches.py` (append)

**Interfaces:**
- Consumes: `required_branches`, `BranchError` (Task 1).
- Produces: `check_input_branches(patterns: Sequence[str], tree_name: str, branch_map: Mapping[str, str], channel: str) -> list[str]` — returns the expanded, ordered file list; raises `BranchError` with every problem. Helper in conftest: `make_root_file(path: Path, names: Iterable[str], tree: str = "Events") -> Path` (writes one flat float branch per name, 2 entries).

- [ ] **Step 1: Append helper to `tests/pipeline/conftest.py`**

Add `import numpy as np` and `import uproot` to the imports at the top. Append:

```python
def make_root_file(path, names, tree="Events"):
    """Write a tiny ROOT file with one flat float branch per name (2 entries)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as f:
        f[tree] = {name: np.array([1.0, 2.0]) for name in names}
    return path
```

- [ ] **Step 2: Append failing tests to `tests/pipeline/test_branches.py`**

Add `from conftest import make_root_file` to the imports. Append:

```python
REQ_W = b.required_branches("W")


def problems_of(excinfo):
    return "\n".join(excinfo.value.problems)


def test_check_passes_with_identity_map(tmp_path):
    f = make_root_file(tmp_path / "sig.root", REQ_W)
    assert b.check_input_branches([str(f)], "Events", {}, "W") == [str(f)]


def test_check_passes_with_renamed_branches(tmp_path):
    names = [n for n in REQ_W if n != "Slimmed_PFCands_pt"] + ["PFCand_pt"]
    f = make_root_file(tmp_path / "sig.root", names)
    assert b.check_input_branches([str(f)], "Events", {"Slimmed_PFCands_pt": "PFCand_pt"}, "W") == [str(f)]


def test_check_expands_globs_sorted(tmp_path):
    make_root_file(tmp_path / "bkg_2.root", REQ_W)
    make_root_file(tmp_path / "bkg_1.root", REQ_W)
    files = b.check_input_branches([str(tmp_path / "bkg_*.root")], "Events", {}, "W")
    assert files == [str(tmp_path / "bkg_1.root"), str(tmp_path / "bkg_2.root")]


def test_check_lists_all_missing_branches_of_all_files(tmp_path):
    f1 = make_root_file(tmp_path / "a.root", [n for n in REQ_W if n not in ("run", "event")])
    f2 = make_root_file(tmp_path / "b.root", [n for n in REQ_W if n != "TauPlusMET_Puppi_pt"])
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f1), str(f2)], "Events", {}, "W")
    text = problems_of(e)
    assert f"{f1}: 2 required branch(es) missing" in text
    assert "run" in text and "event" in text
    assert f"{f2}: 1 required branch(es) missing" in text
    assert "TauPlusMET_Puppi_pt" in text


def test_missing_mapped_branch_names_both_names(tmp_path):
    f = make_root_file(tmp_path / "a.root", [n for n in REQ_W if n != "Slimmed_PFCands_pt"])
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f)], "Events", {"Slimmed_PFCands_pt": "PFCand_pt"}, "W")
    assert "PFCand_pt (analysis name Slimmed_PFCands_pt)" in problems_of(e)


def test_glob_matching_nothing_is_error(tmp_path):
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(tmp_path / "nothing_*.root")], "Events", {}, "W")
    assert "matches no file" in problems_of(e)


def test_missing_file_and_wrong_tree_reported_per_file(tmp_path):
    good = make_root_file(tmp_path / "good.root", REQ_W)
    wrong_tree = make_root_file(tmp_path / "tree.root", REQ_W, tree="Other")
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(tmp_path / "gone.root"), str(wrong_tree), str(good)], "Events", {}, "W")
    text = problems_of(e)
    assert "gone.root: file not found" in text
    assert "tree.root: no tree 'Events'" in text
    assert "good.root" not in text


def test_unknown_map_key_with_hint(tmp_path):
    f = make_root_file(tmp_path / "a.root", REQ_W)
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f)], "Events", {"Slimmed_PFCands_ptt": "PFCand_pt"}, "W")
    text = problems_of(e)
    assert "unknown analysis branch 'Slimmed_PFCands_ptt'" in text
    assert "did you mean 'Slimmed_PFCands_pt'" in text


def test_two_keys_same_target(tmp_path):
    f = make_root_file(tmp_path / "a.root", REQ_W)
    bad = {"Slimmed_PFCands_pt": "pt_x", "Slimmed_PFCands_eta": "pt_x"}
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f)], "Events", bad, "W")
    assert "both map to 'pt_x'" in problems_of(e)


def test_target_is_other_required_name(tmp_path):
    f = make_root_file(tmp_path / "a.root", REQ_W)
    bad = {"Slimmed_TauTo3Mu_fitted_pt": "Slimmed_TauTo3Mu_fitted_eta"}
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f)], "Events", bad, "W")
    assert "is itself a required analysis branch" in problems_of(e)


def test_identity_entry_is_allowed(tmp_path):
    f = make_root_file(tmp_path / "a.root", REQ_W)
    assert b.check_input_branches([str(f)], "Events", {"run": "run"}, "W") == [str(f)]
```

- [ ] **Step 3: Run, expect FAIL**

Run: `$PYTEST tests/pipeline/test_branches.py -q`
Expected: the 11 new tests FAIL with `AttributeError: module ... has no attribute 'check_input_branches'`; the 5 Task 1 tests pass.

- [ ] **Step 4: Implement — append to `src/tau3mu_gnn/pipeline/branches.py`**

Add to the imports at the top (below `from __future__ import annotations`):

```python
import difflib
import glob
from typing import Mapping, Sequence

import uproot
```

Append:

```python
def _check_map(branch_map: Mapping[str, str], required: tuple[str, ...]) -> list[str]:
    problems = []
    required_set = set(required)
    seen_targets: dict[str, str] = {}
    for key in sorted(branch_map):
        target = branch_map[key]
        if key not in required_set:
            hint = difflib.get_close_matches(key, required, n=1)
            problems.append(f"branch_map: unknown analysis branch '{key}'"
                            + (f" (did you mean '{hint[0]}'?)" if hint else ""))
        if target in seen_targets:
            problems.append(f"branch_map: '{seen_targets[target]}' and '{key}' both map to '{target}'")
        else:
            seen_targets[target] = key
        if target != key and target in required_set:
            problems.append(f"branch_map: '{key}' maps to '{target}', which is itself a required analysis branch name")
    return problems


def _expand(patterns: Sequence[str], problems: list[str]) -> list[str]:
    files = []
    for pattern in patterns:
        if any(c in pattern for c in "*?["):
            matches = sorted(glob.glob(pattern))
            if not matches:
                problems.append(f"input pattern matches no file: {pattern}")
            files.extend(matches)
        else:
            files.append(pattern)
    return files


def check_input_branches(patterns: Sequence[str], tree_name: str,
                         branch_map: Mapping[str, str], channel: str) -> list[str]:
    """Check that every input file has every required branch after the branch map.

    Reads branch names only (no events). Returns the expanded file list.
    Raises BranchError listing every problem of every file.
    """
    required = required_branches(channel)
    problems = _check_map(branch_map, required)
    files = _expand(patterns, problems)
    for path in files:
        try:
            with uproot.open(path) as f:
                keys = set(f[tree_name].keys())
        except FileNotFoundError:
            problems.append(f"{path}: file not found")
            continue
        except KeyError:
            problems.append(f"{path}: no tree '{tree_name}'")
            continue
        except Exception as e:  # corrupt file, not ROOT, xrootd error, ...
            problems.append(f"{path}: cannot open ({type(e).__name__}: {e})")
            continue
        missing = [name for name in required if branch_map.get(name, name) not in keys]
        if missing:
            lines = [f"      {branch_map[n]} (analysis name {n})" if branch_map.get(n, n) != n else f"      {n}"
                     for n in missing]
            problems.append(f"{path}: {len(missing)} required branch(es) missing:\n" + "\n".join(lines))
    if problems:
        raise BranchError(problems)
    return files
```

- [ ] **Step 5: Run, expect PASS**

Run: `$PYTEST tests/pipeline -q`
Expected: all passed (58 old + 16 in test_branches.py = 74; real-file tests may skip off T3).

- [ ] **Step 6: Commit** (one command per call)

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/branches.py tests/pipeline/conftest.py tests/pipeline/test_branches.py
/usr/bin/git commit -m "feat(pipeline): check input files for required branches after branch map

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `rename_columns` (RDataFrame) + roadmap

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/branches.py` (append)
- Test: `tests/pipeline/test_branches.py` (append)
- Modify: `docs/wrapper_roadmap.md`

**Interfaces:**
- Consumes: nothing from earlier tasks (independent helper), but uses the same map direction.
- Produces: `rename_columns(df, branch_map: Mapping[str, str]) -> tuple[df, list[str]]` — `df` with `Define(analysis_name, user_name)` for every non-identity entry; the list is every column of the new `df` except the user's original (renamed) names, ready for `df.Snapshot(tree, file, columns)`.

- [ ] **Step 1: Append failing tests to `tests/pipeline/test_branches.py`**

Add `import awkward as ak` and `import numpy as np` to the imports. Append:

```python
def _tiny_jagged_file(path):
    with uproot.recreate(path) as f:
        f["Events"] = {"UserPt": ak.Array([[1.0, 2.0], [3.0]]), "run": np.array([1, 2], dtype=np.uint32)}
    return path


def test_rename_columns_snapshot_has_analysis_name_only(tmp_path):
    import ROOT

    src = _tiny_jagged_file(tmp_path / "in.root")
    df = ROOT.RDataFrame("Events", str(src))
    df, columns = b.rename_columns(df, {"Slimmed_TauTo3Mu_fitted_pt": "UserPt"})
    assert "UserPt" not in columns
    assert "Slimmed_TauTo3Mu_fitted_pt" in columns and "run" in columns
    out = tmp_path / "out.root"
    df.Snapshot("Events", str(out), columns)
    tree = uproot.open(out)["Events"]
    assert "UserPt" not in tree.keys()
    assert tree["Slimmed_TauTo3Mu_fitted_pt"].array().tolist() == [[1.0, 2.0], [3.0]]
    assert tree["run"].array().tolist() == [1, 2]


def test_rename_columns_identity_and_empty_map_change_nothing(tmp_path):
    import ROOT

    src = _tiny_jagged_file(tmp_path / "in.root")
    df = ROOT.RDataFrame("Events", str(src))
    before = sorted(str(c) for c in df.GetColumnNames())
    for mapping in ({}, {"run": "run"}):
        _, columns = b.rename_columns(df, mapping)
        assert sorted(columns) == before
```

- [ ] **Step 2: Run, expect FAIL**

Run: `$PYTEST tests/pipeline/test_branches.py -q -k rename`
Expected: 2 FAILED with `AttributeError: ... has no attribute 'rename_columns'` (plus the harmless ROOT header noise on stderr).

- [ ] **Step 3: Implement — append to `src/tau3mu_gnn/pipeline/branches.py`**

```python
def rename_columns(df, branch_map: Mapping[str, str]):
    """Define each analysis name from the user's name on an RDataFrame.

    Returns (df, columns): columns are all columns of the new df except the
    user's original names that were renamed, ready for df.Snapshot(tree, file, columns).
    """
    renames = {analysis: user for analysis, user in branch_map.items() if analysis != user}
    for analysis in sorted(renames):
        df = df.Define(analysis, renames[analysis])
    originals = set(renames.values())
    columns = [str(c) for c in df.GetColumnNames() if str(c) not in originals]
    return df, columns
```

- [ ] **Step 4: Run, expect PASS**

Run: `$PYTEST tests/pipeline -q`
Expected: all passed (76; real-file tests may skip off T3).

- [ ] **Step 5: Update `docs/wrapper_roadmap.md`** (Edit tool)

- In sec. 5, bullet **Branch map**: replace ``(`Slimmed_*`, `PFCands_*`, `TauPlusMET_*`, `Muon_*`, `PV_npvs`, `run`, `luminosityBlock`, `event`)`` with ``(`Slimmed_TauTo3Mu_*`, `Slimmed_PFCands_*`, `TauPlusMET_*`, `run`, `luminosityBlock`, `event`; list in `pipeline/branches.py`)``.
- In sec. 5, bullet **Required branches**: replace `(~60 branches)` with `(38 for W: \`COMMON\` + \`PER_CHANNEL["W"]\` in \`pipeline/branches.py\`)`.
- Sec. 6: change `3. Branch map:` to `3. ✅ Branch map:`.
- Sec. 6 step 5 line: append ` **Caution:** the BDT drops rows with NaN in *any loaded* column (\`dropna\`); loading fewer branches can change which events get \`bdt_ISO\` — check against the reference limit. Add a test that the BDT feature list ⊆ \`required_branches("W")\`.`
- Sec. 6 step 6 line: append ` Add a test that the GNN branch list ⊆ \`required_branches("W")\`.`
- Sec. 6 step 7 line: append ` **Note:** \`ROC_comp_full\` reads \`GNN_score_{sig,bkg}\` from \`tau_score_corr.pkl\` (GNN-input BDT, dropped in step 5) — adapt it to ISO only.`
- Sec. 8, add rows:
  - `| 2026-10-07 | Step 3 done: \`pipeline/branches.py\` (\`required_branches\`, \`check_input_branches\`, \`rename_columns\`). 38 required W branches, all present in Kaito's W old_data files. HF list added in step 12. PF candidates are \`Slimmed_PFCands_*\` (not \`PFCands_*\`). |`
  - `| 2026-10-07 | Flagged to Kaito (outside wrapper scope): \`TauPlusMET_*\` has one entry per **unskimmed** candidate, but \`ScoreProducer.py:519/523\` indexes it with the **skimmed** candidate mask; lengths match \`nSlimmedTauTo3Mu\` in only 99.36% of signal events. |`
  - `| 2026-10-07 | On the T3 login node \`gcc/12.2.0\` is not available; importing ROOT prints a header error but RDataFrame works. Tests import ROOT only inside the tests that need it. |`

- [ ] **Step 6: Commit** (one command per call)

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/branches.py tests/pipeline/test_branches.py docs/wrapper_roadmap.md
/usr/bin/git commit -m "feat(pipeline): RDataFrame rename helper for the branch map

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `check_input_structure` (per-family length consistency)

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/branches.py` (append)
- Test: `tests/pipeline/test_branches.py` (append)
- Modify: `docs/wrapper_roadmap.md` (one decision-log row)

**Interfaces:**
- Consumes: `required_branches`, `BranchError` (Task 1). Takes the file list **returned by** `check_input_branches` (Task 2), so all names exist.
- Produces: `FAMILIES: tuple[str, ...] = ("Slimmed_TauTo3Mu_", "Slimmed_PFCands_", "TauPlusMET_")`; `check_input_structure(files: Sequence[str], tree_name: str, branch_map: Mapping[str, str], channel: str, n_events: int = 1000) -> None` — raises `BranchError` with every problem.

- [ ] **Step 1: Append failing tests to `tests/pipeline/test_branches.py`**

```python
def _structured_file(path, overrides=None):
    """All required W branches; families jagged with consistent lengths per event (2 events)."""
    lengths = {"Slimmed_TauTo3Mu_": [2, 1], "Slimmed_PFCands_": [3, 4], "TauPlusMET_": [2, 1]}
    data = {}
    for name in REQ_W:
        prefix = next((p for p in lengths if name.startswith(p)), None)
        if prefix is None:
            data[name] = np.array([1, 2], dtype=np.int64)
        else:
            data[name] = ak.Array([[1.0] * n for n in lengths[prefix]])
    data.update(overrides or {})
    with uproot.recreate(path) as f:
        f["Events"] = data
    return path


def test_structure_ok(tmp_path):
    f = _structured_file(tmp_path / "ok.root")
    b.check_input_structure([str(f)], "Events", {}, "W")


def test_structure_length_mismatch_in_family(tmp_path):
    f = _structured_file(tmp_path / "bad.root",
                         {"Slimmed_TauTo3Mu_vtx_chi2": ak.Array([[1.0, 1.0], [1.0, 1.0]])})
    with pytest.raises(b.BranchError) as e:
        b.check_input_structure([str(f)], "Events", {}, "W")
    text = problems_of(e)
    assert "bad.root" in text and "Slimmed_TauTo3Mu_" in text
    assert "Slimmed_TauTo3Mu_vtx_chi2" in text
    assert "entry 1" in text


def test_structure_flat_branch_in_family(tmp_path):
    f = _structured_file(tmp_path / "flat.root", {"Slimmed_PFCands_pt": np.array([1.0, 2.0])})
    with pytest.raises(b.BranchError) as e:
        b.check_input_structure([str(f)], "Events", {}, "W")
    assert "Slimmed_PFCands_pt is not a per-event list" in problems_of(e)


def test_structure_families_not_compared_with_each_other(tmp_path):
    taumet = {n: ak.Array([[1.0], [1.0, 1.0, 1.0]]) for n in REQ_W if n.startswith("TauPlusMET_")}
    f = _structured_file(tmp_path / "cross.root", taumet)
    b.check_input_structure([str(f)], "Events", {}, "W")


def test_structure_uses_branch_map(tmp_path):
    f = _structured_file(tmp_path / "mapped.root",
                         {"UserChi2": ak.Array([[1.0, 1.0], [1.0]])})
    b.check_input_structure([str(f)], "Events", {"Slimmed_TauTo3Mu_vtx_chi2": "UserChi2"}, "W")


@pytest.mark.parametrize("path", W_FILES)
def test_real_w_files_structure(path):
    if not os.path.exists(path):
        pytest.skip(f"not reachable on this machine: {path}")
    b.check_input_structure([path], "Events", {}, "W")
```

Note: `test_structure_uses_branch_map` keeps the original `Slimmed_TauTo3Mu_vtx_chi2` in the file too; the check must read `UserChi2` (the mapped name) — its lengths `[2, 1]` match the family, so it passes.

- [ ] **Step 2: Run, expect FAIL**

Run: `$PYTEST tests/pipeline/test_branches.py -q -k structure`
Expected: 7 FAILED with `AttributeError: ... has no attribute 'check_input_structure'`.

- [ ] **Step 3: Implement — append to `src/tau3mu_gnn/pipeline/branches.py`**

Add `import awkward as ak` and `import numpy as np` to the imports. Append:

```python
# Branches with one entry per tau candidate / PF candidate / TauPlusMET object.
# Lengths are compared inside a family only: TauPlusMET_* has one entry per
# unskimmed candidate and can differ from Slimmed_TauTo3Mu_* (seen in Kaito's signal).
FAMILIES: tuple[str, ...] = ("Slimmed_TauTo3Mu_", "Slimmed_PFCands_", "TauPlusMET_")


def check_input_structure(files: Sequence[str], tree_name: str, branch_map: Mapping[str, str],
                          channel: str, n_events: int = 1000) -> None:
    """Check per-event list lengths agree within each branch family (first n_events per file).

    Expects files that already passed check_input_branches. Raises BranchError listing every problem.
    """
    required = required_branches(channel)
    problems = []
    for path in files:
        tree = uproot.open(path)[tree_name]
        for family in FAMILIES:
            members = [n for n in required if n.startswith(family)]
            if not members:
                continue
            user_names = [branch_map.get(n, n) for n in members]
            arrays = tree.arrays(user_names, entry_stop=n_events, library="ak")
            reference = None
            for analysis, user in zip(members, user_names):
                label = analysis if user == analysis else f"{user} (analysis name {analysis})"
                try:
                    counts = np.asarray(ak.num(arrays[user], axis=1))
                except Exception:
                    problems.append(f"{path}: {label} is not a per-event list ({family}* branches must be)")
                    continue
                if reference is None:
                    reference = (label, counts)
                    continue
                bad = np.nonzero(counts != reference[1])[0]
                if bad.size:
                    i = int(bad[0])
                    problems.append(f"{path}: {family}* branches differ in length: {reference[0]} has "
                                    f"{reference[1][i]} but {label} has {counts[i]} (first at entry {i})")
    if problems:
        raise BranchError(problems)
```

- [ ] **Step 4: Run, expect PASS**

Run: `$PYTEST tests/pipeline -q`
Expected: all passed (82; real-file tests may skip off T3).

- [ ] **Step 5: Roadmap row** (Edit tool) — add to sec. 8:

`| 2026-10-07 | Structure check \`check_input_structure\`: within each family (\`Slimmed_TauTo3Mu_\`, \`Slimmed_PFCands_\`, \`TauPlusMET_\`) all required branches must be per-event lists of equal length (first 1000 events per file). No cross-family check. Raw count branches (\`nSlimmedTauTo3Mu\`, ...) are not required: the code reads only \`nSelectedTauTo3Mu\`, which the skimmer defines. |`

- [ ] **Step 6: Commit** (one command per call)

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/branches.py tests/pipeline/test_branches.py docs/wrapper_roadmap.md
/usr/bin/git commit -m "feat(pipeline): per-family structure check of input branches

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
