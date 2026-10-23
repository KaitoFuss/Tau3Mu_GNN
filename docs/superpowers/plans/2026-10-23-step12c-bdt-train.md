# Step 12c — Stage bdt_train: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A training stage that trains the ISO BDT on the stage 1 skim outputs and writes a classifier pickle that stage 2 (`BDT_ScoreProducer.score_events`) loads unchanged.

**Architecture:** Port the ISO part of `bdt_training/Train_XGB_W.py` into a new importable module `bdt_training/train_bdt.py`. Event loading and derived features reuse the stage 2 code (`BDT_ScoreProducer._load`, `_add_features`, `bdt_inputs.bdt_branches`), so training and inference see the same columns in the same order. A thin stage `pipeline/training/stages/bdt_train.py` wires `TrainingConfig` to it. The old script stays as it is (not imported by the pipeline).

**Tech Stack:** Python 3.11, xgboost 3.2, scikit-learn 1.8, pandas, awkward, uproot, matplotlib (Agg), pytest. Tests: `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` (conda env `Tau3Mu`; plain `python3` has no xgboost).

**Spec:** `docs/superpowers/specs/2026-10-23-step12-training-pipeline-design.md` (sec. 3 `bdt_train`, sec. 4 BDT rows).

## Global Constraints

- ISO mode only: features = `bdt_branches(channel)[0]` + derived columns from `_add_features` (W: `deltaPhi(...)`, `relative_iso`). No `GNN_score`.
- One row per event = candidate 0 (`ak.firsts` in `_load`), rows with NaN dropped (`dropna`), as stage 2.
- Signal: all events (an event without a selected candidate drops out through `dropna`; the old script used `GNN_score != -1` for the same).
- Background: **sideband only**, old rule: keep an event only if **all** its `Selected_TauTo3Mu_fitted_mass` values satisfy `abs(m - tau_mass) > sideband_half_width` (`Train_XGB_W.py:215-216`). `tau_mass` = `cfg.limit.tau_mass` (1.78), half width = `cfg.sideband_half_width` (0.08).
- Sampling: `df.sample(n=min(max_events_per_class, len(df)), random_state=sample_seed)` per class; weights all 1; `target` 1 / 0.
- Split and folds: `train_test_split(data, test_size, random_state=split_seed)`; `StratifiedKFold(n_splits=kfold, random_state=split_seed, shuffle=True)` on train.
- XGBoost per fold: `max_depth, learning_rate, n_estimators, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda, early_stopping_rounds` from `cfg.bdt_training`; `min_child_weight = min_child_weight_fraction * sum(fold train weights)`; `seed = split_seed`; `eval_metric="logloss"`; `eval_set=[(X_train, y_train), (X_valid, y_valid)]`; `verbosity=1`.
- Pickle format = what stage 2 reads: `pickle.dump({fold_index: XGBClassifier, ...})`, fold indices `0..kfold-1`.
- Columns with `object` dtype are cast to `int` before training, as stage 2 does before scoring.
- Use `/usr/bin/git` in the worktree. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch `pipeline-step12c-bdt-train` is stacked on `pipeline-step12b-npvs` (PR #27); its PR targets `main` after #27 is merged.

## Review Focus

1. No background event in the sideband (or no signal event left) → `ValueError` naming the class, before any training (Task 1, `test_empty_class_raises`).
2. Fewer events in a class than `kfold` → clear `ValueError`, not an sklearn traceback (Task 1, `test_too_few_events_for_kfold`).
3. A class with fewer events than `max_events_per_class` → all of them used, no error (Task 1, `test_load_training_data_sideband_and_sampling`).
4. The trained pickle is scored by stage 2 without change, scores in [0, 1], signal above background on separable toys (Task 1, `test_train_then_stage2_scores`).
5. An input file without a BDT branch → `BranchError` naming the file before any read (Task 2, `test_stage_missing_branch_raises`).

---

### Task 1: `bdt_training/train_bdt.py`

**Files:**
- Modify: `src/tau3mu_gnn/bdt_training/BDT_ScoreProducer.py` (`_load` gets an optional event filter)
- Create: `src/tau3mu_gnn/bdt_training/train_bdt.py`
- Test: `tests/pipeline/test_bdt_train.py` (new)

**Interfaces:**
- Consumes: `bdt_inputs.bdt_branches(channel, prefix) -> (features, branches)`, `BDT_ScoreProducer._load`, `_add_features`, `PREFIX`; `TrainingConfig.bdt_training` (`BDTTrainingConfig`, 12a).
- Produces:
  - `BDT_ScoreProducer._load(files, tree_name, branches, label, event_filter=None)` — `event_filter(events: ak.Array) -> ak.Array[bool]` (one bool per event), applied to each chunk **before** `ak.firsts`. `None` = today's behaviour (stage 2 unchanged).
  - `train_bdt.sideband_filter(tau_mass: float, half_width: float, prefix: str = "Selected_") -> Callable[[ak.Array], ak.Array]` — `ak.all(abs(events[f"{prefix}TauTo3Mu_fitted_mass"] - tau_mass) > half_width, axis=1)`; the mass branch must be in the loaded branches (it is: `bdt_branches` lists it).
  - `train_bdt.load_training_data(signal_files, background_files, tree_name: str, channel: str, tau_mass: float, sideband_half_width: float, max_events_per_class: int, sample_seed: int) -> tuple[pd.DataFrame, list[str]]` — returns `(data, bdt_features)`; `data` = signal + background rows with columns `target`, `weight`, `id`, all loaded and derived columns. Raises `ValueError(f"BDT training: no {class} events ...")` for an empty class.
  - `train_bdt.train_iso_bdt(data: pd.DataFrame, bdt_features: list[str], params: BDTTrainingConfig) -> tuple[dict[int, XGBClassifier], dict[str, Any], pd.DataFrame]` — returns `(classifiers, summary, test)`; `summary` keys: `n_signal`, `n_background`, `n_train`, `n_test`, `auc_test` (mean fold score on the test split, `roc_auc_score`), `best_iteration` (list per fold), `features` (list). Raises `ValueError` if a class in the train split has fewer rows than `kfold`. Also returns per-fold `evals_result()` inside `summary["logloss"]` = `{fold: {"train": [...], "validation": [...]}}`.
  - `train_bdt.plot_training(summary: dict, test: pd.DataFrame, classifiers: dict, bdt_features: list[str], out_dir: Path) -> list[Path]` — one `logloss_fold_<i+1>.png` per fold (train vs validation, as the old script) and `roc.png` (test split, mean fold score). `matplotlib.use("Agg")` at module import.

- [ ] **Step 1: Write the failing tests** in `tests/pipeline/test_bdt_train.py`. Toy helper (module level):

```python
import pickle
import awkward as ak
import numpy as np
import pytest

from conftest import write_tree
from tau3mu_gnn.bdt_training import train_bdt
from tau3mu_gnn.bdt_training.bdt_inputs import input_branch_names

MASS = "Selected_TauTo3Mu_fitted_mass"
PT = "Selected_TauTo3Mu_fitted_pt"


def _toy(path, n, rng, signal, masses=None, first_event=0):
    """n events, one candidate each (masses: list of per-event candidate lists overrides).
    Every BDT input branch; fitted_pt separates signal (~50) from background (~10)."""
    masses = masses or [[1.78 if signal else 1.65]] * n
    data = {}
    for name in sorted(input_branch_names("W")):
        if name == MASS:
            data[name] = ak.Array(masses)
        elif name.startswith(("Selected_TauTo3Mu_", "TauPlusMET_")):
            centre = (50.0 if signal else 10.0) if name == PT else 1.0
            data[name] = ak.Array([list(np.abs(rng.normal(centre, 2.0, len(m))) + 0.5) for m in masses])  # > 0: no inf in chi2/ndof
        else:
            data[name] = np.ones(len(masses), dtype=np.uint32)
    data["event"] = np.arange(first_event, first_event + len(masses), dtype=np.uint64)
    return write_tree(path, data)


@pytest.fixture
def params(load_training, training_user):
    load_training.defaults["bdt_training"].update(n_estimators=30, kfold=2, early_stopping_rounds=5)
    return load_training(training_user).bdt_training
```

Tests:

```python
def test_sideband_filter():
    ev = ak.Array({MASS: [[1.70], [1.60, 1.79], [1.95], [1.62, 1.90]]})
    assert train_bdt.sideband_filter(1.78, 0.08)(ev).tolist() == [False, False, True, True]


def test_load_training_data_sideband_and_sampling(tmp_path):
    rng = np.random.default_rng(0)
    sig = _toy(tmp_path / "s.root", 50, rng, True)
    bkg = _toy(tmp_path / "b.root", 0, rng, False,
               masses=[[1.65]] * 30 + [[1.78]] * 10 + [[1.65, 1.80]] * 5, first_event=100)
    data, feats = train_bdt.load_training_data([sig], [bkg], "Events", "W", 1.78, 0.08, 40, 42)
    assert (data.target == 1).sum() == 40 and (data.target == 0).sum() == 30   # 40 = cap, 30 = sideband only
    assert feats[-1] == "relative_iso" and "deltaPhi(TauPlusMET_Puppi_phi, Selected_TauTo3Mu_fitted_phi)" in feats
    again, _ = train_bdt.load_training_data([sig], [bkg], "Events", "W", 1.78, 0.08, 40, 42)
    assert data.equals(again)   # same seed, same sample


def test_empty_class_raises(tmp_path):
    rng = np.random.default_rng(0)
    sig = _toy(tmp_path / "s.root", 20, rng, True)
    bkg = _toy(tmp_path / "b.root", 0, rng, False, masses=[[1.78]] * 20)   # nothing in the sideband
    with pytest.raises(ValueError, match="background"):
        train_bdt.load_training_data([sig], [bkg], "Events", "W", 1.78, 0.08, 100, 42)


def test_too_few_events_for_kfold(tmp_path, params):
    rng = np.random.default_rng(0)
    data, feats = train_bdt.load_training_data(
        [_toy(tmp_path / "s.root", 1, rng, True)], [_toy(tmp_path / "b.root", 40, rng, False)],
        "Events", "W", 1.78, 0.08, 100, 42)
    with pytest.raises(ValueError, match="kfold"):
        train_bdt.train_iso_bdt(data, feats, params)


def test_train_then_stage2_scores(tmp_path, params):
    from tau3mu_gnn.bdt_training.BDT_ScoreProducer import score_events
    rng = np.random.default_rng(1)
    sig = _toy(tmp_path / "s.root", 300, rng, True)
    bkg = _toy(tmp_path / "b.root", 300, rng, False, first_event=1000)
    data, feats = train_bdt.load_training_data([sig], [bkg], "Events", "W", 1.78, 0.08, 100_000, 42)
    classifiers, summary, test = train_bdt.train_iso_bdt(data, feats, params)
    assert sorted(classifiers) == [0, 1] and summary["auc_test"] > 0.95
    assert summary["n_signal"] == summary["n_background"] == 300
    clf = tmp_path / "ISO.pck"
    clf.write_bytes(pickle.dumps(classifiers))
    score_events({str(sig): str(tmp_path / "s_out.root")}, {str(bkg): str(tmp_path / "b_out.root")},
                 clf, "W", tmp_path / "p.pkl")
    p = pickle.loads((tmp_path / "p.pkl").read_bytes())
    s, b = np.asarray(p["ISO_score_sig"]), np.asarray(p["ISO_score_bkg"])
    assert 0 <= b.min() and s.max() <= 1 and s.mean() > b.mean() + 0.5


def test_plot_training_writes_files(tmp_path, params):
    rng = np.random.default_rng(2)
    data, feats = train_bdt.load_training_data(
        [_toy(tmp_path / "s.root", 100, rng, True)], [_toy(tmp_path / "b.root", 100, rng, False)],
        "Events", "W", 1.78, 0.08, 100_000, 42)
    classifiers, summary, test = train_bdt.train_iso_bdt(data, feats, params)
    paths = train_bdt.plot_training(summary, test, classifiers, feats, tmp_path / "plots")
    assert sorted(p.name for p in paths) == ["logloss_fold_1.png", "logloss_fold_2.png", "roc.png"]
    assert all(p.is_file() for p in paths)
```

- [ ] **Step 2: Run, expect FAIL** — `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_bdt_train.py -q` → `ImportError: cannot import name 'train_bdt'`.
- [ ] **Step 3: Implement** `_load(..., event_filter=None)` and `train_bdt.py` per the Interfaces block and the Global Constraints. Module docstring: what it ports (`Train_XGB_W.py`, ISO mode), what it reuses from stage 2.
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_bdt_train.py tests/pipeline/test_bdt_scores.py -q` → all pass (stage 2 tests unchanged).
- [ ] **Step 5: Commit** — `feat(bdt): ISO BDT training as a module (port of Train_XGB_W.py)`.

### Task 2: Stage `bdt_train`

**Files:**
- Create: `src/tau3mu_gnn/pipeline/training/stages/bdt_train.py`
- Test: `tests/pipeline/test_bdt_train.py` (extend)
- Modify: `docs/wrapper_roadmap.md` (12c ✅)

**Interfaces:**
- Consumes: Task 1 (`load_training_data`, `train_iso_bdt`, `plot_training`); `bdt_inputs.input_branch_names(channel)`; `branches.check_stage_inputs`; `TrainingConfig` fields `inputs.tree_name`, `channel`, `limit.tau_mass`, `sideband_half_width`, `bdt_training`.
- Produces:
  - `OUTPUTS = {"classifier": "ISO_classifiers.pck", "summary": "bdt_train_summary.json"}`.
  - `run(cfg: TrainingConfig, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict[str, Path]` — `inputs = {"signal": file, "background": file}` (stage 1). Checks the inputs with `check_stage_inputs(inputs, tree, input_branch_names(cfg.channel), "BDT training")` first. Writes the pickle, the summary JSON (the `summary` dict without the per-iteration `logloss` lists) and the plots into `out_dir`. Returns `{"classifier": ..., "summary": ..., "plots": [paths]}` (`plots` is a list). 12g copies `classifier` into the version folder; 12h points stage 2 at it.

- [ ] **Step 1: Write the failing tests** (append to `tests/pipeline/test_bdt_train.py`):

```python
def test_stage_writes_outputs(tmp_path, load_training, training_user):
    import json
    from tau3mu_gnn.pipeline.training.stages import bdt_train as stage
    load_training.defaults["bdt_training"].update(n_estimators=30, kfold=2, early_stopping_rounds=5)
    cfg = load_training(training_user)
    rng = np.random.default_rng(3)
    inputs = {"signal": _toy(tmp_path / "s.root", 100, rng, True),
              "background": _toy(tmp_path / "b.root", 100, rng, False, first_event=500)}
    out = stage.run(cfg, inputs, tmp_path / "bdt_train")
    assert out["classifier"].name == "ISO_classifiers.pck" and len(pickle.loads(out["classifier"].read_bytes())) == 2
    s = json.loads(out["summary"].read_text())
    assert {"n_signal", "n_background", "auc_test", "features"} <= set(s) and "logloss" not in s
    assert len(out["plots"]) == 3 and all(p.is_file() for p in out["plots"])


def test_stage_missing_branch_raises(tmp_path, load_training, training_user):
    from tau3mu_gnn.pipeline.branches import BranchError
    from tau3mu_gnn.pipeline.training.stages import bdt_train as stage
    rng = np.random.default_rng(4)
    sig = _toy(tmp_path / "s.root", 10, rng, True)
    bkg = write_tree(tmp_path / "b.root", {"event": np.arange(3, dtype=np.uint64)})
    with pytest.raises(BranchError, match="b.root"):
        stage.run(load_training(training_user), {"signal": sig, "background": bkg}, tmp_path / "o")
    assert not (tmp_path / "o" / "ISO_classifiers.pck").exists()
```

- [ ] **Step 2: Run, expect FAIL** — `ModuleNotFoundError: ...stages.bdt_train`.
- [ ] **Step 3: Implement** the stage (style of `training/stages/npvs.py`: docstring, `log.info` with event counts and test AUC).
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` → all pass.
- [ ] **Step 5: Mark 12c ✅ in `docs/wrapper_roadmap.md`** (stage file, output names, plan path; note: old `Train_XGB_W.py` unchanged, not used by the pipeline).
- [ ] **Step 6: Commit** — `feat(training): stage bdt_train (step 12c)`.

## Real-data check (optional, after Task 2, no SLURM)

Run the stage on the small raw skim (`/pnfs/.../kaito_W_raw_small/phase_a_vs1/1_skim/{signal,background}.root`) with the frozen settings into the scratchpad. Report: events per class, test AUC, best iteration per fold, run time. Then score the same files with stage 2 using the new pickle and compare the score distribution with the frozen classifier's (`phase_a_vs1/2_bdt/`). Not a test; report to Kaito.
