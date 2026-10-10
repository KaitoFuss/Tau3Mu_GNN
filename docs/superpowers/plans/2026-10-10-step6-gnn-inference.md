# Mini Step 6 — GNN Inference Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GNN scoring a function with explicit input → output files (still several models in one pass), make it score the **first** candidate (as the BDT and the limit do), wire it as pipeline stage 3 (`run(cfg, inputs, out_dir)`), and prove with a regression job that the only differences to the old `ScoreCreation` script are in multi-candidate events.

**Architecture:** Same pattern as step 5b (BDT). The branch list moves out of `ScoreProducer` into `gnn_training/gnn_inputs.py` (testable without torch). Model loading moves out of `scripts/ScoreCreation.py` into `gnn_training/model_loading.py`. `ScoreProducer` stays multi-model (each graph is built once and scored by every model) and takes `inputs`, `models: {name: (model, normalizer)}`, `outputs: {name: {input: output}}`, `npz_paths: {name: path}` and `tree_name` instead of a folder + output folders; a function `score_events(...)` wraps it. The graph building and scoring math (`build_cone`, `connect_knn`, ...) does **not** change. One behavior change, in its own commit: the scored candidate is the first one, not the highest-pt one. The old CLI stays as a thin wrapper, so the old job scripts in `jobs/gnn_score_creation/` keep working. New stage `pipeline/stages/gnn.py` checks every input branch first, then loads the frozen model from the config and scores both samples.

**Tech Stack:** Python 3.11 (Tau3Mu env), PyTorch + torch_geometric, uproot 5.7, awkward, PyROOT 6.34, pytest.

**Spec:** `docs/wrapper_roadmap.md` (sec. 6 step 6, decision log). Pattern to copy: step 5b (`bdt_training/BDT_ScoreProducer.py`, `pipeline/stages/bdt.py`, `tests/pipeline/test_bdt_scores.py`).

## Global Constraints

- Python: `/work/kfuss/miniforge3/envs/Tau3Mu/bin/python`. Test command (`$PYTEST`): `PYTHONPATH=/work/kfuss/SVJ_GNN/.claude/worktrees/pipeline-wrapper/src /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m pytest`
- Branch `pipeline-step6-gnn`, stacked on `pipeline-step5b-bdt-config` (PR #13).
- Git in this worktree: `/usr/bin/git` only. One git command per Bash call.
- Importing ROOT on the login node prints a "cannot extract standard library include paths" error. It is noise. Do not fix it.
- **One behavior change only:** the GNN scores the **first** candidate (`nt = 0`), not the highest-pt one (`np.argmax(pt)`, `ScoreProducer.py:507`). Decided by Kaito 2026-10-10: `bdt_ISO` is the first candidate's score and the limit uses `Selected_TauTo3Mu_fitted_mass[0]`, so all three now use the same candidate. Training built graphs for every candidate, so this is not a new input type for the model. Own commit (Task 3 Step 9).
- Otherwise do not touch the scoring math: `create_score` math, `build_cone`, `connect_*`, `get_muon_idx`, `calc_delta_R_sq`, `deltaphi`.
- `PV_npvs` is **not** a required branch (Kaito 2026-10-10: only needed for training weights). The GNN loads it only for `weights == "npvs_rescale"`.
- Known behaviors that stay as they are:
  - `TauPlusMET_*` is indexed with the skimmed candidate mask (E1 in the roadmap).
  - Graph settings are fixed for the frozen model: `k_knn=10`, `cone_size=0.5`, `use_displacement=True`, `edge_connection_mode="knn"`. `gnn.graph_type` in the config is used only for the feature counts (`get_model_parameters`), as in the old script.
- No `max_events` argument in the stage (PR 10 review). `score_events` keeps `max_events_to_process` only for the old CLI.
- Stage output names: `<out_dir>/{signal,background}_with_gnn_scores.root` and `<out_dir>/raw_scores_and_tau_mass_{signal,background}.npz` (no `_with_bdt_cut` suffix; step 7 adapts `ModelEval`).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Regression jobs: write the script, tell Kaito, Kaito runs `sbatch`. Never submit yourself.

## Review Focus

1. **Input file already has `GNN_score`** (rerun on a stage 3 output, or an old file) → `CloneTree` + `Branch("GNN_score")` would write two branches with the same name. Expect a clear `BranchError` before any work. Test in Task 4.
2. **Event without a candidate, or below the BDT cut, or with ≤1 PF candidate in the cone** → event kept, `GNN_score = -1`, not in the npz. Test in Task 3.
3. **Multi-candidate event where the first candidate is not the highest-pt one** → the first candidate is scored (same as BDT and limit). Test in Task 3.
4. **Tree name not `Events`** → read and write use `inputs.tree_name`. Test in Task 3.
5. **Wrong `model_name` / missing model file** → `FileNotFoundError` with the path, before any event is read. Test in Task 2.

## File Structure

| File | Responsibility |
|---|---|
| `src/tau3mu_gnn/bdt_training/bdt_inputs.py` (modify) | add `expression_names(expressions)`; `input_branch_names` uses it |
| `src/tau3mu_gnn/gnn_training/gnn_inputs.py` (create) | `graph_level_branches`, `gnn_branches`, `input_branch_names`, `STAGE_MADE` |
| `src/tau3mu_gnn/gnn_training/model_loading.py` (create) | `load_model(...)` → `(model, normalizer)` |
| `src/tau3mu_gnn/gnn_training/ScoreProducer.py` (modify) | single model; `files`, `npz_path`, `tree_name`; `_write_with_scores`; `score_events` |
| `src/tau3mu_gnn/scripts/ScoreCreation.py` (modify) | thin old CLI over `load_model` + `score_events` |
| `src/tau3mu_gnn/pipeline/branches.py` (modify) | `check_stage_inputs(...)` shared by stages 2 and 3 |
| `src/tau3mu_gnn/pipeline/stages/bdt.py` (modify) | `check_inputs` calls `check_stage_inputs` |
| `src/tau3mu_gnn/pipeline/stages/gnn.py` (create) | stage 3 `check_inputs`, `run` |
| `tests/pipeline/conftest.py` (modify) | `write_tree(path, data, tree)` (moved from `test_bdt_scores.py`) |
| `tests/pipeline/test_gnn_inputs.py` (create) | branch list tests |
| `tests/pipeline/test_gnn_model_loading.py` (create) | real frozen model loads |
| `tests/pipeline/test_gnn_scores.py` (create) | toy events + toy model: `score_events` and stage 3 |
| `regression/step6/run_regression_6.sh`, `regression/step6/compare_6.py` (create, not committed: `regression/` is untracked) | old vs new on a W subset |
| `docs/wrapper_roadmap.md` (modify) | step 6 done + decisions |

---

### Task 1: GNN branch list in its own module

**Files:**
- Modify: `src/tau3mu_gnn/bdt_training/bdt_inputs.py` (end of file)
- Create: `src/tau3mu_gnn/gnn_training/gnn_inputs.py`
- Modify: `src/tau3mu_gnn/gnn_training/ScoreProducer.py:409-451` (graph-level list) and `:782-837` (load list)
- Test: `tests/pipeline/test_gnn_inputs.py`

**Interfaces:**
- Produces: `expression_names(expressions: Iterable[str]) -> set[str]` (bdt_inputs);
  `graph_level_branches(tau_channel: str, prefix: str = "Selected_") -> list[str]`;
  `gnn_branches(tau_channel: str, slimmed: bool = True, bdt_preselection: bool = True, use_displacement: bool = True, weights: str | None = None, save_benchmark_score: bool = False) -> list[str]`;
  `input_branch_names(tau_channel: str) -> set[str]` (branches the stage-3 defaults read);
  `STAGE_MADE = frozenset({"nSelectedTauTo3Mu", "bdt_ISO"})` (made by stages 1 and 2, not raw inputs).

Behavior change (output-neutral): `PV_npvs` is loaded only for `weights == "npvs_rescale"`; the three `iso_*` branches only for `save_benchmark_score`. Both are used only there. `PV_npvs` is not in `required_branches("W")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_gnn_inputs.py
import re

import pytest

from tau3mu_gnn.gnn_training.gnn_inputs import (STAGE_MADE, gnn_branches, graph_level_branches,
                                                 input_branch_names)
from tau3mu_gnn.gnn_training.TrainingUtils import get_model_parameters
from tau3mu_gnn.pipeline.branches import required_branches

P = "Selected_TauTo3Mu_"
PF = "Slimmed_PFCands_"


def test_w_load_list_is_old_list_without_unused_branches():
    # Old list of ScoreProducer.get_all_scores (W, slimmed, BDT cut, displacement), minus
    # PV_npvs (npvs_rescale only) and the iso_* branches (save_benchmark_score only).
    expected = [
        *(f"{PF}{s}" for s in ("pt", "eta", "phi", "pdgId", "mass", "charge", "puppiWeight")),
        "nSelectedTauTo3Mu",
        *(f"{P}mu{i}_{s}" for i in (1, 2, 3) for s in ("eta", "phi")),
        *(f"{P}fitted_{s}" for s in ("mass", "eta", "phi", "pt")),
        "run", "luminosityBlock", "event",
        "bdt_ISO",
        *graph_level_branches("W"),
        "TauPlusMET_Puppi_phi",
        f"{PF}d0", f"{PF}dz",
    ]
    assert gnn_branches("W") == expected


def test_optional_branches():
    assert "bdt_ISO" not in gnn_branches("W", bdt_preselection=False)
    assert "PV_npvs" in gnn_branches("W", weights="npvs_rescale")
    assert f"{P}iso_ptPhotons" in gnn_branches("W", save_benchmark_score=True)
    assert f"{PF}d0" not in gnn_branches("W", use_displacement=False)
    assert "PFCands_pt" in gnn_branches("W", slimmed=False)


@pytest.mark.parametrize("channel", ["W", "HF"])
def test_graph_level_count_matches_model(channel):
    # W adds one computed feature (deltaPhi) in ScoreProducer.process_event
    n_computed = 1 if channel == "W" else 0
    n_graph = get_model_parameters("skimmed_dzd0_cone_size_0.5_knn_10", channel)[2]
    assert len(graph_level_branches(channel)) + n_computed == n_graph


def test_w_gnn_branches_are_required_inputs():
    raw = {re.sub(r"^Selected_TauTo3Mu", "Slimmed_TauTo3Mu", n) for n in input_branch_names("W") - STAGE_MADE}
    missing = raw - set(required_branches("W"))
    assert not missing, f"GNN reads branches not in required_branches('W'): {sorted(missing)}"


def test_input_branch_names_splits_expressions():
    names = input_branch_names("W")
    assert {f"{P}vtx_chi2", f"{P}vtx_Ndof", "bdt_ISO", "nSelectedTauTo3Mu"} <= names
    assert not any("*" in n or "[" in n for n in names)


def test_unknown_channel():
    with pytest.raises(ValueError, match="no GNN branch list"):
        gnn_branches("X")
```

- [ ] **Step 2: Run, expect FAIL** — `$PYTEST tests/pipeline/test_gnn_inputs.py -v` → `ModuleNotFoundError: tau3mu_gnn.gnn_training.gnn_inputs`.

- [ ] **Step 3: Add `expression_names` to `bdt_inputs.py`** (replace the body of `input_branch_names`):

```python
def expression_names(expressions) -> set[str]:
    """Branch names that uproot expressions read, e.g.
    'Muon_pt[Selected_TauTo3Mu_mu3_idx]' -> {'Muon_pt', 'Selected_TauTo3Mu_mu3_idx'}."""
    return {name for expr in expressions for name in re.findall(r"[A-Za-z_]\w*", expr)}


def input_branch_names(tau_channel: str, prefix: str = "Selected_") -> set[str]:
    """Branch names that the loaded expressions read."""
    _, branches = bdt_branches(tau_channel, prefix)
    return expression_names(branches)
```

- [ ] **Step 4: Create `gnn_inputs.py`.** Copy the two `graph_level_branches` lists **verbatim** (with their comments) from `ScoreProducer.py:411-451`:

```python
"""Branches that ScoreProducer loads, per channel."""
from __future__ import annotations

from tau3mu_gnn.bdt_training.bdt_inputs import expression_names

# Made by the pipeline (stage 1 skim, stage 2 BDT), not raw input branches.
STAGE_MADE = frozenset({"nSelectedTauTo3Mu", "bdt_ISO"})


def graph_level_branches(tau_channel: str, prefix: str = "Selected_") -> list[str]:
    """uproot expressions of the graph-level features, in model input order. New list on every call."""
    if tau_channel == "W":
        return [ ...W list from ScoreProducer.py:412-430, unchanged... ]
    if tau_channel == "HF":
        return [ ...HF list from ScoreProducer.py:432-451, unchanged... ]
    raise ValueError(f"no GNN branch list for channel {tau_channel!r} (known: HF, W)")


def gnn_branches(tau_channel: str, slimmed: bool = True, bdt_preselection: bool = True,
                 use_displacement: bool = True, weights: str | None = None,
                 save_benchmark_score: bool = False) -> list[str]:
    """Every expression ScoreProducer loads from the input file."""
    prefix = "Selected_" if slimmed else ""
    n_prefix = "Selected" if slimmed else ""
    old_prefix = "Slimmed_" if slimmed else ""
    graph_level = graph_level_branches(tau_channel, prefix)  # raises for unknown channel
    branches = [
        f"{old_prefix}PFCands_pt",
        f"{old_prefix}PFCands_eta",
        f"{old_prefix}PFCands_phi",
        f"{old_prefix}PFCands_pdgId",
        f"{old_prefix}PFCands_mass",
        f"{old_prefix}PFCands_charge",
        f"{old_prefix}PFCands_puppiWeight",
        f"n{n_prefix}TauTo3Mu",
        f"{prefix}TauTo3Mu_mu1_eta",
        f"{prefix}TauTo3Mu_mu1_phi",
        f"{prefix}TauTo3Mu_mu2_eta",
        f"{prefix}TauTo3Mu_mu2_phi",
        f"{prefix}TauTo3Mu_mu3_eta",
        f"{prefix}TauTo3Mu_mu3_phi",
        f"{prefix}TauTo3Mu_fitted_mass",
        f"{prefix}TauTo3Mu_fitted_eta",
        f"{prefix}TauTo3Mu_fitted_phi",
        f"{prefix}TauTo3Mu_fitted_pt",
    ]
    if weights == "npvs_rescale":
        branches.append("PV_npvs")
    if save_benchmark_score:
        branches += [f"{prefix}TauTo3Mu_iso_ptChargedFromPV",
                     f"{prefix}TauTo3Mu_iso_ptPhotons",
                     f"{prefix}TauTo3Mu_iso_ptChargedFromPU"]
    branches += ["run", "luminosityBlock", "event"]
    if bdt_preselection:
        branches.append("bdt_ISO")
    branches += graph_level
    if tau_channel == "W":
        branches.append("TauPlusMET_Puppi_phi")
    if use_displacement:
        branches += [f"{old_prefix}PFCands_d0", f"{old_prefix}PFCands_dz"]
    return branches


def input_branch_names(tau_channel: str) -> set[str]:
    """Branch names that stage 3 reads (pipeline settings: slimmed, BDT cut, displacement)."""
    return expression_names(gnn_branches(tau_channel))
```

- [ ] **Step 5: Use it in `ScoreProducer.py`.**
  - Add `from tau3mu_gnn.gnn_training.gnn_inputs import gnn_branches, graph_level_branches` to the imports.
  - Replace the `if self.tau_channel == "W": self.graph_level_branches = [...] elif ...HF...` block (`:411-451`) with `self.graph_level_branches = graph_level_branches(self.tau_channel, prefix)`.
  - Replace `branches_to_load = [...]` up to and including the `use_displacement` append (`:785-837`) with:

```python
        branches_to_load = gnn_branches(self.tau_channel, self.slimmed, self.bdt_preselection_cut is not None,
                                        self.use_displacement, self.weights, self.save_benchmark_score)
```
  Keep the `prefix = ...` lines above it only if still used in that method (`self.mask` uses `prefix`: keep).

- [ ] **Step 6: Run tests** — `$PYTEST tests/pipeline -q` → all pass (101 old + 6 new).

- [ ] **Step 7: Commit** — `feat(gnn): branch list in gnn_inputs, load PV_npvs/iso only when used`.

---

### Task 2: `load_model` for the frozen model

**Files:**
- Create: `src/tau3mu_gnn/gnn_training/model_loading.py`
- Test: `tests/pipeline/test_gnn_model_loading.py`

**Interfaces:**
- Produces: `load_model(model_dir: str | Path, model_name: str, conv_type: str, gnn_mode: str, graph_type: str, tau_channel: str, device) -> tuple[TauMuNet, dict]`. Reads `<model_dir>/<model_name>/{arch_config.json, <model_name>.pt, Normalization.pkl}`. Model is in `eval()` mode on `device`'s map location.

- [ ] **Step 1: Write the failing tests** (use the real frozen model in the repo)

```python
# tests/pipeline/test_gnn_model_loading.py
import pytest
import torch

from conftest import REPO_ROOT

MODEL = "new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2"
ARGS = dict(conv_type="GINE", gnn_mode="final_layer",
            graph_type="skimmed_dzd0_cone_size_0.5_knn_10", tau_channel="W", device=torch.device("cpu"))


def test_loads_frozen_w_model():
    from tau3mu_gnn.architectures.TauMuNet import TauMuNet
    from tau3mu_gnn.gnn_training.model_loading import load_model

    model, normalizer = load_model(REPO_ROOT / "models/W/gnn", MODEL, **ARGS)
    assert isinstance(model, TauMuNet)
    assert not model.training
    assert {"means_x", "stds_x", "means_edge", "stds_edge", "means_graph", "stds_graph"} <= set(normalizer)


def test_missing_model_names_path(tmp_path):
    from tau3mu_gnn.gnn_training.model_loading import load_model

    with pytest.raises(FileNotFoundError, match="no_such_model"):
        load_model(tmp_path, "no_such_model", **ARGS)
```

- [ ] **Step 2: Run, expect FAIL** — `ModuleNotFoundError: ...model_loading`.

- [ ] **Step 3: Implement** (logic moved from `scripts/ScoreCreation.py:49-75`)

```python
"""Load a trained TauMuNet and its feature normalizer."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch

from tau3mu_gnn.architectures.TauMuNet import TauMuNet
from tau3mu_gnn.gnn_training.TrainingUtils import get_model_parameters
from tau3mu_gnn.model_eval.ModelEval import CPU_Unpickler


def load_model(model_dir, model_name, conv_type, gnn_mode, graph_type, tau_channel, device):
    """Return (model in eval mode, normalizer dict) from <model_dir>/<model_name>/."""
    folder = Path(model_dir) / model_name
    with open(folder / "arch_config.json") as f:
        params = json.load(f)
    num_node_features, num_edge_features, num_graph_level_features = get_model_parameters(graph_type, tau_channel)
    model = TauMuNet(
        conv_params=params["conv_params"],
        dropout=params["dropout"],
        fc_params=params["fc_params"],
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_level_features=num_graph_level_features,
        use_fusion=params["use_fusion"],
        gnn_mode=gnn_mode,
        scoring=True,
        scale_down_params=params["fusion_scale_down_params"],
        conv_type=conv_type,
    )
    model.load_state_dict(torch.load(folder / f"{model_name}.pt", map_location=device)["model_state_dict"])
    model.eval()
    with open(folder / "Normalization.pkl", "rb") as f:
        normalizer = CPU_Unpickler(f).load()
    return model, normalizer
```

`CPU_Unpickler` is what `ModelEval.load_pkl` uses today (`ModelEval.py:27`), so the normalizer is identical.

- [ ] **Step 4: Run tests** — `$PYTEST tests/pipeline/test_gnn_model_loading.py -v` → PASS.

- [ ] **Step 5: Commit** — `feat(gnn): load_model for a trained TauMuNet`.

---

### Task 3: `score_events`: files in, files out

**Files:**
- Modify: `src/tau3mu_gnn/gnn_training/ScoreProducer.py` (class `ScoreProducer`: `__init__`, `process_event` end, `get_all_scores`, `create_score`, `process`, `raw_file_names`; new `_write_with_scores`, `score_events`)
- Modify: `src/tau3mu_gnn/scripts/ScoreCreation.py` (thin CLI)
- Modify: `tests/pipeline/conftest.py` (add `write_tree`), `tests/pipeline/test_bdt_scores.py` (use it)
- Test: `tests/pipeline/test_gnn_scores.py`

**Interfaces:**
- Consumes: `gnn_branches`, `graph_level_branches` (Task 1); `load_model` (Task 2, old CLI only).
- Produces:
```python
def score_events(inputs: Sequence[str], models: Mapping[str, tuple], outputs: Mapping[str, Mapping[str, str]],
                 npz_paths: Mapping[str, str | Path], tau_channel: str, unsculpting_mode: str | None,
                 bdt_preselection_cut: float | None, tree_name: str = "Events", device=None,
                 max_events_to_process: int | None = None) -> None
```
  `models`: model name → `(model, normalizer)`. Each input file is read once; each graph is built once and scored by every model. For each model `name` and input `f`: writes `outputs[name][f]` = copy of `f` (all entries, all branches) + `GNN_score` (float32; `-1` for events without a score). Writes `npz_paths[name]` with `scores`, `tau_mass` of all scored events, files concatenated in `inputs` order. `normalizer=None` means no normalization (already supported by `create_score`). `outputs` and `npz_paths` must have exactly the keys of `models` (else `ValueError`).
- `conftest.write_tree(path, data, tree="Events")` (moved from `test_bdt_scores._write`).

- [ ] **Step 1: Move the tree writer to conftest.** Add to `tests/pipeline/conftest.py`:

```python
def write_tree(path, data, tree="Events"):
    """Write data (name -> numpy / awkward array) as a TTree.
    mktree, because f[tree] = data writes an RNTuple in uproot 5.7."""
    import awkward as ak

    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as f:
        f.mktree(tree, {k: ak.type(v) if isinstance(v, ak.Array) else v.dtype for k, v in data.items()})
        f[tree].extend(data)
    return path
```
  In `test_bdt_scores.py` delete `_write` and use `from conftest import write_tree` (replace every `_write(` with `write_tree(`). Run `$PYTEST tests/pipeline/test_bdt_scores.py -q` → pass.

- [ ] **Step 2: Write the failing tests**

```python
# tests/pipeline/test_gnn_scores.py
import awkward as ak
import numpy as np
import pytest
import torch
import uproot

from conftest import write_tree
from tau3mu_gnn.gnn_training.gnn_inputs import input_branch_names

P = "Selected_TauTo3Mu_"
PF = "Slimmed_PFCands_"
MU_ETA = {1: 0.01, 2: 0.02, 3: -0.01}
# PF candidate etas (phi = 0). The first three sit on the muons (removed by delete_muons).
PF_GOOD = [0.01, 0.02, -0.01, 0.1, 0.15, -0.1, 0.2, 3.0]   # 4 left in the 0.5 cone
PF_EMPTY = [0.01, 0.02, -0.01, 3.0]                        # 0 left -> event skipped


class ToyGNN(torch.nn.Module):
    """Score = first graph-level feature = fitted pt of the scored candidate."""

    def forward(self, data):
        return data.graph_level_features[:, :1]


def gnn_events(cands, bdt, pf=None, first_event=10):
    """One event per entry of cands (list of candidate pts). bdt: bdt_ISO per event.
    pf: PF candidate etas per event (default PF_GOOD). Every stage 3 input branch, plus 'extra'."""
    n = len(cands)
    pf = pf or [PF_GOOD] * n

    def per_cand(f):
        return ak.Array([[f(j) for j in range(len(ev))] for ev in cands])

    def per_pf(f, dtype=float):
        return ak.Array([[dtype(f(k, eta)) for k, eta in enumerate(ev)] for ev in pf])

    data = {}
    for name in sorted(input_branch_names("W")):
        if name in ("run", "luminosityBlock"):
            data[name] = np.ones(n, dtype=np.uint32)
        elif name == "event":
            data[name] = np.arange(first_event, first_event + n, dtype=np.uint64)
        elif name == "nSelectedTauTo3Mu":
            data[name] = np.array([len(c) for c in cands], dtype=np.int32)
        elif name == "bdt_ISO":
            data[name] = np.asarray(bdt, dtype=np.float32)
        elif name == f"{P}fitted_pt":
            data[name] = ak.Array(cands)
        elif name == f"{P}fitted_mass":
            data[name] = per_cand(lambda j: 1.70 + 0.01 * j)
        elif name in (f"{P}fitted_eta", f"{P}fitted_phi") or name.endswith("_phi") and name.startswith(P):
            data[name] = per_cand(lambda j: 0.0)
        elif name.startswith(P) and name.endswith("_eta"):  # mu{i}_eta
            i = int(name[len(P) + 2])
            data[name] = per_cand(lambda j: MU_ETA[i])
        elif name == f"{PF}eta":
            data[name] = ak.Array(pf)
        elif name in (f"{PF}pdgId", f"{PF}charge"):
            data[name] = per_pf(lambda k, eta: 211 if name.endswith("pdgId") else 1 - 2 * (k % 2), int)
        elif name.startswith(PF):
            value = {"pt": lambda k, eta: 1.0 + k, "mass": lambda k, eta: 0.1,
                     "puppiWeight": lambda k, eta: 1.0}.get(name[len(PF):], lambda k, eta: 0.0)
            data[name] = per_pf(value)
        else:  # other Selected_TauTo3Mu_* and TauPlusMET_* (per candidate)
            data[name] = per_cand(lambda j: 1.0 + j)
    data["extra"] = np.arange(n, dtype=np.float64)
    return data


def _score(tmp_path, files_data, tree="Events", cut=0.8, models=None):
    """Score with each model (default: one ToyGNN 'm'). Returns (output files of the first model, its npz)."""
    from tau3mu_gnn.gnn_training.ScoreProducer import score_events

    models = models or {"m": (ToyGNN(), None)}
    inputs = [str(write_tree(tmp_path / f"in_{i}.root", data, tree)) for i, data in enumerate(files_data)]
    outputs = {name: {f: str(tmp_path / name / f"out_{i}.root") for i, f in enumerate(inputs)} for name in models}
    npz_paths = {name: tmp_path / name / "scores.npz" for name in models}
    score_events(inputs, models, outputs, npz_paths, "W", "delete_muons", cut, tree_name=tree,
                 device=torch.device("cpu"))
    first = next(iter(models))
    return list(outputs[first].values()), np.load(npz_paths[first])


# event 0: two candidates (first one, pt 100, is scored; not the highest-pt 900), 1: one, 2: none,
# 3: BDT below cut, 4: no PF candidate left in the cone
CANDS = [[100.0, 900.0], [300.0], [], [400.0], [500.0]]
BDT = [0.9, 0.95, 0.9, 0.5, 0.9]
PF_ETAS = [PF_GOOD, PF_GOOD, PF_GOOD, PF_GOOD, PF_EMPTY]
EXPECTED = [100.0, 300.0, -1.0, -1.0, -1.0]


def test_gnn_score_per_event(tmp_path):
    (out,), npz = _score(tmp_path, [gnn_events(CANDS, BDT, PF_ETAS)])
    tree = uproot.open(out)["Events"]
    np.testing.assert_array_equal(tree["GNN_score"].array(library="np"), np.float32(EXPECTED))
    np.testing.assert_allclose(npz["scores"], [100.0, 300.0])
    np.testing.assert_allclose(npz["tau_mass"], [1.70, 1.70], rtol=1e-6)


def test_all_events_and_branches_kept(tmp_path):
    data = gnn_events(CANDS, BDT, PF_ETAS)
    (out,), _ = _score(tmp_path, [data])
    tree = uproot.open(out)["Events"]
    assert set(tree.keys()) == set(data) | {"GNN_score"}
    np.testing.assert_array_equal(tree["extra"].array(library="np"), data["extra"])
    np.testing.assert_array_equal(tree["event"].array(library="np"), data["event"])


def test_custom_tree_name(tmp_path):
    (out,), _ = _score(tmp_path, [gnn_events(CANDS, BDT, PF_ETAS)], tree="Tree3mu")
    assert "GNN_score" in uproot.open(out)["Tree3mu"].keys()


def test_several_files_in_order(tmp_path):
    a = gnn_events([[300.0]], [0.9])
    b = gnn_events([[700.0], [200.0]], [0.9, 0.9], first_event=50)
    (out_a, out_b), npz = _score(tmp_path, [a, b])
    np.testing.assert_allclose(npz["scores"], [300.0, 700.0, 200.0])
    np.testing.assert_array_equal(uproot.open(out_b)["Events"]["GNN_score"].array(library="np"),
                                  np.float32([700.0, 200.0]))


class TwiceGNN(ToyGNN):
    def forward(self, data):
        return 2 * super().forward(data)


def test_several_models_one_pass(tmp_path):
    from tau3mu_gnn.gnn_training.ScoreProducer import score_events

    inp = str(write_tree(tmp_path / "in.root", gnn_events(CANDS, BDT, PF_ETAS)))
    models = {"a": (ToyGNN(), None), "b": (TwiceGNN(), None)}
    outputs = {n: {inp: str(tmp_path / n / "out.root")} for n in models}
    npz_paths = {n: tmp_path / n / "s.npz" for n in models}
    score_events([inp], models, outputs, npz_paths, "W", "delete_muons", 0.8, device=torch.device("cpu"))
    b = uproot.open(outputs["b"][inp])["Events"]["GNN_score"].array(library="np")
    np.testing.assert_array_equal(b, np.float32([200.0, 600.0, -1.0, -1.0, -1.0]))
    np.testing.assert_allclose(np.load(npz_paths["a"])["scores"], [100.0, 300.0])


def test_outputs_must_match_models(tmp_path):
    from tau3mu_gnn.gnn_training.ScoreProducer import score_events

    with pytest.raises(ValueError, match="outputs"):
        score_events(["x.root"], {"a": (ToyGNN(), None)}, {"b": {}}, {"a": tmp_path / "s.npz"},
                     "W", "delete_muons", 0.8)


def test_no_cut_scores_low_bdt_event(tmp_path):
    (out,), _ = _score(tmp_path, [gnn_events([[400.0]], [0.5])], cut=None)
    assert uproot.open(out)["Events"]["GNN_score"].array(library="np")[0] == 400.0
```

  Note on the toy branch filler: check `sorted(input_branch_names("W"))` once (print it) and make sure every name goes to the right `elif`: muon `mu{i}_phi` and `fitted_phi` → 0, `mu{i}_eta` → `MU_ETA[i]`, `vtx_Ndof` must be non-zero (it is: `1.0 + j`). Fix the filler, not the production code, if a name falls into the wrong branch.

- [ ] **Step 3: Run, expect FAIL** — `ImportError: cannot import name 'score_events'`.

  Until Step 9 (first candidate), `test_gnn_score_per_event`, `test_several_models_one_pass` and `test_stage_uses_config_cut` expect the first candidate and fail on event 0 (old code scores 900). That is expected. All other tests must pass after Step 5.

- [ ] **Step 4: Refactor `ScoreProducer`** (edit in place; do not reformat untouched code). The class keeps its multi-model loops (`self.model_names`, `self.models`, `self.normalizers`); only the I/O changes.
  1. `__init__`: replace params `root`, `models`, `normalizers`, `output_folders`, `storage_folder`, `model_names` with `inputs`, `models` (mapping name → `(model, normalizer)`), `outputs`, `npz_paths`, `tree_name="Events"` (keep all other params and defaults). Set:
```python
        if set(outputs) != set(models) or set(npz_paths) != set(models):
            raise ValueError(f"outputs ({sorted(outputs)}) and npz_paths ({sorted(npz_paths)}) "
                             f"must have the model names as keys ({sorted(models)})")
        self.inputs = list(inputs)
        self.model_names = list(models)
        self.models = [models[n][0] for n in self.model_names]
        self.normalizers = [models[n][1] for n in self.model_names]
        self.outputs = {n: dict(outputs[n]) for n in self.model_names}
        self.npz_paths = {n: Path(npz_paths[n]) for n in self.model_names}
        self.tree_name = tree_name
```
  Delete the old length check and the `output_folders` makedirs (`:386-391`).
  2. `get_all_scores`: `npvs_weights = np.load(os.path.join(os.path.dirname(file_name), "npvs_weights.npz"))["weight_per_bin"]` (was `self.root + "/npvs_weights.npz"`; `root` was the input folder). Iterate `f"{file_name}:{self.tree_name}"` (was `:Events`).
  3. Delete `raw_file_names`. Replace `process` with:
```python
    def process(self):
        all_scores = {name: [] for name in self.model_names}
        all_tau_masses = []
        for in_file in self.inputs:
            print(f"Processing {in_file}")
            scores, tau_masses, score_dicts = self.get_all_scores(in_file)
            all_tau_masses += ak.to_list(tau_masses)
            for name in self.model_names:
                all_scores[name] += ak.to_list(scores[name])
                _write_with_scores(score_dicts[name], in_file, self.outputs[name][in_file], self.tree_name)

        for name in self.model_names:
            npz_path = self.npz_paths[name]
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, scores=all_scores[name], tau_mass=all_tau_masses)
            print(f"Raw scores and tau masses saved to: {npz_path}")
```
  4. Module-level `_write_with_scores(score_dict, in_file, out_file, tree_name)`: the body of the old per-model loop (`:960-1041`), with `"Events"` → `tree_name` (3 places), `root_path` → `str(out_file)`, temp file `str(out_file).removesuffix(".root") + "_scores_only.root"`, `os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)` first, `score_dict_all_models[model_name]` → `score_dict`. Keep `-1.` default, `BuildIndex("run", "event")`, `GNN_score/F`.
  5. Module-level `score_events` (signature in Interfaces):
```python
def score_events(inputs, models, outputs, npz_paths, tau_channel, unsculpting_mode, bdt_preselection_cut,
                 tree_name="Events", device=None, max_events_to_process=None):
    """Score the first candidate of each event with every model. See the module docstring for outputs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ScoreProducer(inputs=inputs, models=models, outputs=outputs, npz_paths=npz_paths, tree_name=tree_name,
                  edge_connection_mode="knn", delta_r_threshold=None, k_knn=10,
                  max_events_to_process=max_events_to_process, save_edge_attributes=True, verbose=True,
                  tau_channel=tau_channel, slimmed=True, shuffle_bool=max_events_to_process is not None,
                  use_displacement=True, device=device, unsculpting_mode=unsculpting_mode,
                  bdt_preselection_cut=bdt_preselection_cut)
```
  (These are exactly the values `ScoreCreation.py:87-105` passes today.) Add a short module docstring like `BDT_ScoreProducer.py`'s, and `from pathlib import Path`, `from typing import Mapping, Sequence` if missing.

- [ ] **Step 5: Run tests** — `$PYTEST tests/pipeline/test_gnn_scores.py -v` → all PASS except the three first-candidate tests (fail on event 0 only: got 900, expected 100). For any other failure, read the toy data first (Step 2 note), then the refactor.

- [ ] **Step 6: Thin old CLI.** Rewrite `scripts/ScoreCreation.py` (same arguments and defaults as today, all kept):

```python
"""Old entry point: score every .root file of --root with each model; outputs as before."""
import argparse
import os

import torch

from tau3mu_gnn.gnn_training.model_loading import load_model
from tau3mu_gnn.gnn_training.ScoreProducer import score_events


def main(argv=None):
    parser = argparse.ArgumentParser()
    # ... the same 13 add_argument lines as today, unchanged ...
    args = parser.parse_args(argv)

    assert len(args.model_names) == len(args.archs) == len(args.conv_types), (
        f"Number of model names ({len(args.model_names)}) must match number of architectures "
        f"({len(args.archs)}). Got model_names={args.model_names}, archs={args.archs}")
    if "signal" in args.root:
        mode = "signal"
    elif "background" in args.root:
        mode = "background"
    else:
        raise ValueError(f"--root must contain 'signal' or 'background': {args.root}")
    suffix = "" if args.bdt_preselection_cut is None else "_with_bdt_cut"
    in_files = [os.path.join(args.root, f) for f in os.listdir(args.root) if f.endswith(".root")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"BDT Selections cut: {args.bdt_preselection_cut}")

    models, outputs, npz_paths = {}, {}, {}
    for model_name, conv_type in zip(args.model_names, args.conv_types):
        models[model_name] = load_model(args.output_folder, model_name, conv_type, args.gnn_mode,
                                        args.model, args.tau_channel, device)
        storage = os.path.join(args.storage_folder, model_name)
        outputs[model_name] = {f: os.path.join(storage, os.path.splitext(os.path.basename(f))[0] + "_with_scores.root")
                               for f in in_files}
        npz_paths[model_name] = os.path.join(args.output_folder, model_name,
                                             f"raw_scores_and_tau_mass_{mode}{suffix}.npz")
    score_events(in_files, models, outputs, npz_paths, args.tau_channel, args.unsculpting_mode,
                 args.bdt_preselection_cut, device=device, max_events_to_process=args.max_events_to_process)


if __name__ == "__main__":
    main()
```
  Same output paths as before. Several models are still scored in one pass.

- [ ] **Step 7: CLI test** (append to `test_gnn_scores.py`): monkeypatch `load_model` and check the old output paths.

```python
def test_old_cli_paths(tmp_path, monkeypatch):
    import tau3mu_gnn.scripts.ScoreCreation as cli

    monkeypatch.setattr(cli, "load_model", lambda *a, **k: (ToyGNN(), None))
    root = tmp_path / "signal_W" / "bdt_full"
    write_tree(root / "sig.root", gnn_events(CANDS, BDT, PF_ETAS))
    out_folder, storage = tmp_path / "output", tmp_path / "scratch"
    cli.main(["--root", str(root), "--output_folder", f"{out_folder}/", "--model_names", "m1",
              "--archs", "random", "--tau_channel", "W", "--gnn_mode", "final_layer",
              "--storage_folder", str(storage), "--conv_types", "GINE",
              "--unsculpting_mode", "delete_muons", "--bdt_preselection_cut", "0.8"])
    assert (storage / "m1" / "sig_with_scores.root").exists()
    assert (out_folder / "m1" / "raw_scores_and_tau_mass_signal_with_bdt_cut.npz").exists()
```

- [ ] **Step 8: Commit the refactor** (the three first-candidate tests still fail; all others pass) — `refactor(gnn): score_events with explicit input and output files (step 6)`.

- [ ] **Step 9: Score the first candidate.** In `process_event` (`ScoreProducer.py:507-509`) replace

```python
            nt = np.argmax(event[f"{prefix}{self.tau_collection_name}_pt"][tau_mask])
            if n_tau > 1:
                print(f"WARNING: using {nt}-th tau cand")
```
  with
```python
            nt = 0  # first candidate, as the BDT (bdt_ISO) and the limit (fitted_mass[0]) use
```
  Do not change anything else in `process_event`.

- [ ] **Step 10: Run all tests** — `$PYTEST tests/pipeline -q` → all pass.

- [ ] **Step 11: Commit** — `fix(gnn): score the first candidate, as BDT and limit do`.

---

### Task 4: Stage 3 `pipeline/stages/gnn.py`

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/branches.py` (add `check_stage_inputs`)
- Modify: `src/tau3mu_gnn/pipeline/stages/bdt.py` (`check_inputs` uses it)
- Create: `src/tau3mu_gnn/pipeline/stages/gnn.py`
- Test: `tests/pipeline/test_gnn_scores.py` (stage tests)

**Interfaces:**
- Consumes: stage 2 output `{"signal": Path, "background": Path, ...}` (extra keys such as `scores_pickle` ignored); `input_branch_names` (Task 1); `load_model` (Task 2); `score_events` (Task 3).
- Produces: `check_stage_inputs(inputs: Mapping[str, str | Path], tree_name: str, required: Iterable[str], label: str, forbidden: Iterable[str] = ()) -> None` (raises `BranchError`);
  `gnn.run(cfg, inputs, out_dir) -> {"signal": Path, "background": Path, "signal_scores": Path, "background_scores": Path}` with files `<sample>_with_gnn_scores.root`, `raw_scores_and_tau_mass_<sample>.npz`.

- [ ] **Step 1: Move the check to `branches.py`.** Copy the body of `bdt.check_inputs` into:

```python
def check_stage_inputs(inputs: Mapping[str, str | Path], tree_name: str, required: Iterable[str],
                       label: str, forbidden: Iterable[str] = ()) -> None:
    """Raise BranchError listing, for every sample file in inputs, each missing required branch
    and each forbidden branch that is present. Reads branch names only."""
    required = sorted(required)
    problems = []
    for sample, path in inputs.items():
        path = str(path)
        try:
            with uproot.open(path) as f:
                keys = set(f[tree_name].keys())
        except FileNotFoundError:
            problems.append(f"{path}: file not found")
            continue
        except KeyError:
            problems.append(f"{path}: no tree '{tree_name}'")
            continue
        missing = [name for name in required if name not in keys]
        if missing:
            problems.append(f"{path}: missing {label} input branches: {', '.join(missing)}")
        present = [name for name in sorted(forbidden) if name in keys]
        if present:
            problems.append(f"{path}: has branches that {label} writes: {', '.join(present)}")
    if problems:
        raise BranchError(problems)
```
  `bdt.check_inputs(inputs, tree_name, channel)` becomes `check_stage_inputs({s: inputs[s] for s in SAMPLES}, tree_name, input_branch_names(channel), "BDT")`. Run `$PYTEST tests/pipeline/test_bdt_scores.py -q` → pass (message text unchanged: `missing BDT input branches`).

- [ ] **Step 2: Write the failing stage tests** (append to `test_gnn_scores.py`):

```python
from tau3mu_gnn.pipeline.branches import BranchError
from tau3mu_gnn.pipeline.stages import gnn as gnn_stage


def _stage(tmp_path, monkeypatch, load, user, sig=None, bkg=None):
    import tau3mu_gnn.gnn_training.model_loading as model_loading

    seen = {}

    def fake_load_model(model_dir, model_name, conv_type, gnn_mode, graph_type, tau_channel, device):
        seen.update(model_dir=model_dir, model_name=model_name, conv_type=conv_type, gnn_mode=gnn_mode,
                    graph_type=graph_type, tau_channel=tau_channel)
        return ToyGNN(), None

    monkeypatch.setattr(model_loading, "load_model", fake_load_model)
    cfg = load(user)
    inputs = {"signal": write_tree(tmp_path / "2_bdt" / "signal_with_bdt_scores.root", sig or gnn_events(CANDS, BDT, PF_ETAS)),
              "background": write_tree(tmp_path / "2_bdt" / "background_with_bdt_scores.root",
                                       bkg or gnn_events([[50.0]], [0.85], first_event=20)),
              "scores_pickle": tmp_path / "2_bdt" / "tau_score_corr.pkl"}
    out_dir = tmp_path / "3_gnn"
    return gnn_stage.run(cfg, inputs, out_dir), out_dir, cfg, seen


def test_stage_outputs(tmp_path, monkeypatch, load, user):
    outputs, out_dir, cfg, seen = _stage(tmp_path, monkeypatch, load, user)
    assert outputs == {
        "signal": out_dir / "signal_with_gnn_scores.root",
        "background": out_dir / "background_with_gnn_scores.root",
        "signal_scores": out_dir / "raw_scores_and_tau_mass_signal.npz",
        "background_scores": out_dir / "raw_scores_and_tau_mass_background.npz",
    }
    assert all(p.exists() for p in outputs.values())
    np.testing.assert_allclose(np.load(outputs["background_scores"])["scores"], [50.0])
    assert seen == dict(model_dir=cfg.gnn.model_dir, model_name=cfg.gnn.model_name, conv_type=cfg.gnn.conv_type,
                        gnn_mode=cfg.gnn.gnn_mode, graph_type=cfg.gnn.graph_type, tau_channel="W")


def test_stage_uses_config_cut(tmp_path, monkeypatch, load, user):
    # TOY_DEFAULTS cut is 0.8: event 3 (bdt 0.5) gets -1
    outputs, *_ = _stage(tmp_path, monkeypatch, load, user)
    scores = uproot.open(outputs["signal"])["Events"]["GNN_score"].array(library="np")
    np.testing.assert_array_equal(scores, np.float32(EXPECTED))


def test_missing_branch_fails_before_output(tmp_path, monkeypatch, load, user):
    sig = gnn_events(CANDS, BDT, PF_ETAS)
    del sig["bdt_ISO"], sig[f"{PF}d0"]
    with pytest.raises(BranchError) as e:
        _stage(tmp_path, monkeypatch, load, user, sig=sig)
    assert "bdt_ISO" in str(e.value) and f"{PF}d0" in str(e.value)
    assert not (tmp_path / "3_gnn").exists()


def test_existing_gnn_score_fails(tmp_path, monkeypatch, load, user):
    bkg = gnn_events([[50.0]], [0.85])
    bkg["GNN_score"] = np.full(1, 0.3)
    with pytest.raises(BranchError, match="GNN_score"):
        _stage(tmp_path, monkeypatch, load, user, bkg=bkg)
```

- [ ] **Step 3: Run, expect FAIL** — `ImportError: cannot import name 'gnn'`.

- [ ] **Step 4: Implement `pipeline/stages/gnn.py`**

```python
"""Stage 3: GNN inference on the stage 2 outputs (events with BDT ISO score > gnn.bdt_preselection_cut).

Writes <sample>_with_gnn_scores.root (all events and branches, plus GNN_score; -1 = not scored)
and raw_scores_and_tau_mass_<sample>.npz (scores and tau mass of the scored events, for the model evaluation).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from tau3mu_gnn.gnn_training.gnn_inputs import input_branch_names
from tau3mu_gnn.pipeline.branches import check_stage_inputs
from tau3mu_gnn.pipeline.config import Config

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")


def check_inputs(inputs: Mapping[str, str | Path], tree_name: str, channel: str) -> None:
    """Raise BranchError if a GNN input branch is missing, or GNN_score is already there."""
    check_stage_inputs({s: inputs[s] for s in SAMPLES}, tree_name, input_branch_names(channel), "GNN",
                       forbidden=["GNN_score"])


def run(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict[str, Path]:
    """Score inputs (sample -> stage 2 output file) into out_dir."""
    check_inputs(inputs, cfg.inputs.tree_name, cfg.channel)
    import torch

    from tau3mu_gnn.gnn_training import model_loading
    from tau3mu_gnn.gnn_training.ScoreProducer import score_events

    g = cfg.gnn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {g.model_name: model_loading.load_model(g.model_dir, g.model_name, g.conv_type, g.gnn_mode,
                                                     g.graph_type, cfg.channel, device)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for sample in SAMPLES:
        outputs[sample] = out_dir / f"{sample}_with_gnn_scores.root"
        outputs[f"{sample}_scores"] = out_dir / f"raw_scores_and_tau_mass_{sample}.npz"
        log.info("stage 3 GNN: %s -> %s", inputs[sample], outputs[sample])
        in_file = str(inputs[sample])
        score_events([in_file], models, {g.model_name: {in_file: str(outputs[sample])}},
                     {g.model_name: outputs[f"{sample}_scores"]}, cfg.channel, g.unsculpting_mode,
                     g.bdt_preselection_cut, tree_name=cfg.inputs.tree_name, device=device)
    return outputs
```
  `model_loading.load_model` (module attribute, not `from ... import load_model`) so the test's monkeypatch works.

- [ ] **Step 5: Run all tests** — `$PYTEST tests/pipeline -q` → all pass.

- [ ] **Step 6: Commit** — `feat(pipeline): GNN inference stage from config (step 6)`.

---

### Task 5: Roadmap

**Files:** Modify `docs/wrapper_roadmap.md`.

- [ ] **Step 1:** In sec. 6, mark step 6 `✅` **only after the Task 6 regression passes**. Before that, add decision-log rows (date of the work):
  - Step 6 interface: `score_events(files, model, normalizer, tau_channel, npz_path, unsculpting_mode, bdt_preselection_cut, tree_name, device, max_events_to_process)`, `load_model(...)`, stage `pipeline/stages/gnn.py` → `3_gnn/{sample}_with_gnn_scores.root`, `raw_scores_and_tau_mass_{sample}.npz` (no `_with_bdt_cut` suffix: step 7 adapts `ModelEval`). `ScoreProducer` is single-model; the old CLI loops over models.
  - GNN branch list in `gnn_training/gnn_inputs.py`; test: GNN branches ⊆ `required_branches("W")` (except stage-made `nSelectedTauTo3Mu`, `bdt_ISO`). `PV_npvs` / `iso_*` loaded only when used.
  - Stage 3 fails if an input already has `GNN_score`.
  - 2026-10-10 (Kaito): the GNN scores the **first** candidate (was highest pt), as the BDT and the limit. Multi-model scoring in one pass kept.
  - 2026-10-10 (Kaito): `PV_npvs` is not required for inference. The npvs weights (`utils/CorrectPileupDist.py`) are only for training graph weights.
  - New roadmap item (sec. 6, after step 11, before HF): **npvs weights stage** for when retraining is in the pipeline: after the skim, before the BDT; writes `npvs_weights.npz` + plots from `PV_npvs` (signal vs data). `PV_npvs` becomes required only for that stage. Note: the old weights were made before the skim, so the new ones differ a bit.
- [ ] **Step 2: Commit** — `docs(roadmap): step 6 interface and decisions`.

---

### Task 6: Regression job (Kaito queues it)

**Files (not committed, `regression/` is untracked):** `regression/step6/run_regression_6.sh`, `regression/step6/compare_6.py`.

Reference: the old CLI at commit `c6db332` (GNN code identical to `main`; scores the highest-pt candidate). New: `pipeline.stages.gnn.run` (first candidate) with the real config (`configs/example_W.yaml`) and the frozen model in `models/W/gnn`. Same input subset for both.

Inputs: the first **5,000** signal and **400,000** background events of the old `bdt_full` files (they have `bdt_ISO`, `Selected_*`, no `GNN_score`). About 85% of signal and 1.2% of background pass the 0.8 BDT cut → ≈ 4,200 + 4,800 scored events.

- [ ] **Step 1: Write `compare_6.py`.** Subcommand-free CLI: `--old-root --new-root --old-npz --new-npz --report --sample`. Checks (per sample, append to the report):
  - same entries; same branch set; every branch equal, read **one branch at a time** (background subset is large). Float compare: equal or both NaN (copy `same()` from `regression/step5b/compare_5b.py`).
  - Every branch except `GNN_score` must be equal.
  - `GNN_score`: compute `changed = argmax(Selected_TauTo3Mu_fitted_pt) != 0` per event (from the new output file; `False` for events without a candidate). **PASS** if `GNN_score` is identical on all events with `changed == False`. Print: entries `!= -1` (old, new), number of `changed` events, how many of them differ, and how many changed `-1` ↔ scored.
  - npz: the old and new npz hold one entry per scored event in event order. Rebuild the event order from the output files (`GNN_score != -1`), drop the `changed` events from both, and require `scores` and `tau_mass` equal on the rest. Print the npz lengths.
  - Last line `RESULT: PASS` / `RESULT: FAIL`, exit code 1 on FAIL.
- [ ] **Step 2: Write `run_regression_6.sh`** (pattern: `regression/step5b/run_regression_5b.sh`): `#SBATCH -p standard --account=t3 --mem=32G --time=12:00:00`, logs in `regression/step6/logs/`, all work in `/scratch/$USER/$SLURM_JOB_ID` (trap removes it). Steps:
  1. Freeze code: `/usr/bin/git -C $WT archive c6db332 src | tar -x -C $S/old`; `cp -r $WT/src $WT/configs $WT/models $S/new/`.
  2. Subsets with PyROOT `TTree::CopyTree("", "", N)` (keeps the branch layout): `$S/in/signal/sub.root` (5,000), `$S/in/background/sub.root` (400,000). The old CLI needs `signal` / `background` in the folder path; make sure `$S` itself contains neither word.
  3. Old: copy `models/W/gnn/<model>` to `$S/old_models/`; for each sample run `PYTHONPATH=$S/old/src python -m tau3mu_gnn.scripts.ScoreCreation --root $S/in/<sample> --output_folder $S/old_models/ --model_names <model> --archs random --tau_channel W --gnn_mode final_layer --storage_folder $S/old_out --conv_types GINE --unsculpting_mode delete_muons --bdt_preselection_cut 0.8` (same flags as `jobs/gnn_score_creation/create_score_W_*.sh`).
  4. New: `PYTHONPATH=$S/new/src python - <<EOF` that asserts `tau3mu_gnn.__file__` is under `$S/new/src`, loads `load_config("$S/new/configs/example_W.yaml")` (needs `$USER`), calls `gnn.run(cfg, {"signal": ..., "background": ...}, "$S/new_out")`.
  5. Compare both samples: old `$S/old_out/<model>/sub_with_scores.root` + `$S/old_models/<model>/raw_scores_and_tau_mass_<sample>_with_bdt_cut.npz` vs new `$S/new_out/<sample>_with_gnn_scores.root` + `raw_scores_and_tau_mass_<sample>.npz`; report `regression/step6/report_${SLURM_JOB_ID}.txt`.
- [ ] **Step 3:** `bash -n` the script. Run `compare_6.py` once on the login node on two tiny files from `test_gnn_scores.py` (old = new) → `PASS`.
- [ ] **Step 4: Tell Kaito** what the job does, where it writes, and the `sbatch` command. Do **not** submit. After Kaito reports the job done: read the report; on `PASS`, mark step 6 `✅` in the roadmap with the job number and the counts, commit, push, open the PR (base `pipeline-step5b-bdt-config`, `--repo KaitoFuss/SVJ_GNN`).
