# Mini Step 7 — Model Evaluation Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `ModelEval` plots used in production (`tau_hist_fpr`, `tau_hist`, `ROC_comp_full` + SIC) functions with explicit input files and an explicit output file, drop the use of the removed GNN-input BDT, and wire them as pipeline stage 4 (`run(cfg, inputs, out_dir)`). The old CLI stays and gives pixel-identical plots.

**Architecture:** Inside `model_eval/ModelEval.py`, split each used plot mode into "load" and "plot":
- `load_gnn_npz(bkg_npz, sig_npz)` and `load_pickle(path)` read the files.
- `plot_tau_hist(...)` takes arrays and an output PNG path. The body is the old `tau_score_hist` body, unchanged.
- `plot_roc_full(...)` takes a list of `(label, sig_npz, bkg_npz)` and the BDT pickle path. It no longer reads `GNN_score_{sig,bkg}`: that curve was computed but never drawn (its `plt.plot` is commented out) and the SIC loop skipped it.
- `plot_sic(...)` takes the curves that `plot_roc_full` returns.

The old CLI (`__main__`, `tau_score_hist`) builds the old file names (with `_with_bdt_cut`) and calls these functions. New stage `pipeline/stages/evaluation.py` checks its three input files, then writes five PNGs into `out_dir`.

**Tech Stack:** Python 3.11 (Tau3Mu env), numpy, pandas 3.0, matplotlib 3.10, scikit-learn 1.8, pytest.

**Spec:** `docs/wrapper_roadmap.md` sec. 6 step 7 and the two notes there. Stacked on step 6b (`docs/superpowers/plans/2026-10-12-step6b-merge-keys.md`).

## Global Constraints

- Python and test command as in step 6b: `$PYTEST` = `PYTHONPATH=/work/kfuss/SVJ_GNN/.claude/worktrees/pipeline-wrapper/src /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m pytest tests/pipeline`
- Branch `pipeline-step7-eval`, stacked on `pipeline-step6b-merge-keys`.
- Git in this worktree: `/usr/bin/git` only. One git command per Bash call.
- **No change to what a plot shows.** Same `plt` calls in the same order (colors come from the color cycle, so order matters). The old CLI must give pixel-identical PNGs (Task 4).
- Keep the old CLI and its arguments. Keep the line `root_dir = "/work/kfuss/SVJ_GNN/output/comparison_plots"` in `__main__` verbatim (the regression job patches it with `sed`).
- Do not touch modes the pipeline does not use: `tau_hist_after*`, `ROC_comp`, `score_cut`, `plot_roc`, `plot_roc_bdt`, `load_and_filter_*`.
- Sideband for background in the ROC: `|m - 1.78| > 0.08` and `1.6 < |m| < 2.0` (unchanged, a module constant).
- ISO curve in the ROC: ISO scores `> iso_cut`. Old CLI: `iso_cut = 0.8`. Stage: `cfg.gnn.bdt_preselection_cut`.
- Stage output names: `<out_dir>/{tau_hist_fpr,tau_hist_fpr_normalized,tau_hist,roc,sic}.png`.
- Stage inputs (keys as returned by stages 2 and 3): `signal_scores`, `background_scores` (stage 3 npz), `scores_pickle` (stage 2 pickle).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Regression jobs: write the script, tell Kaito, Kaito runs `sbatch`. Never submit yourself.

## Review Focus

1. **Stage 2 pickle without `GNN_score_*` keys** (every new pickle) → ROC and SIC work. Test in Task 3 (the toy pickle has no such keys).
2. **Missing input file** (stage 3 failed, wrong path) → `FileNotFoundError` naming every missing file, before any plot. Test in Task 3.
3. **npz or pickle without an expected key** (old or foreign file) → `ValueError` naming the file and the key. Test in Task 3.
4. **Old pickle that still has `GNN_score_*` keys** (old CLI on old outputs) → still works, extra keys ignored. Test in Task 2.
5. **Old CLI with `--bdt_cut true`** → reads `*_with_bdt_cut.npz`, writes the old PNG names. Test in Task 1.

## File Structure

| File | Responsibility |
|---|---|
| `src/tau3mu_gnn/model_eval/ModelEval.py` (modify) | `load_gnn_npz`, `load_pickle`, `plot_tau_hist`, `plot_roc_full`, `plot_sic`; old CLI calls them |
| `src/tau3mu_gnn/pipeline/stages/evaluation.py` (create) | stage 4 `check_inputs`, `run` |
| `tests/pipeline/test_eval.py` (create) | toy npz + pickle: functions, old CLI names, stage |
| `regression/step7/run_regression_7.sh`, `compare_7.py` (create, not committed) | old CLI vs new CLI, pixel compare; stage smoke run |
| `docs/wrapper_roadmap.md` (modify) | step 7 done + decisions |

---

### Task 1: `tau_hist`: load and plot apart

**Files:**
- Modify: `src/tau3mu_gnn/model_eval/ModelEval.py:370-500` (`tau_score_hist`)
- Test: `tests/pipeline/test_eval.py`

**Interfaces:**
- Produces:
  - `load_gnn_npz(bkg_npz, sig_npz=None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]`: `(bkg scores, bkg tau mass, sig scores or None)`. If `sig_npz` is given: NaN scores removed from both (and the matching masses), as the old fpr path did. If not: nothing removed (old non-fpr path).
  - `plot_tau_hist(scores, tau_mass, scores_signal, out_png, model, mode, title, tau_channel, normalize) -> None`. `scores_signal` not `None` = FPR mode.
  - `tau_score_hist(root, mode, title, tau_channel, plot_mode, view, bdt_score_cut)`: unchanged signature (old CLI).

- [ ] **Step 1: Write the test file with toy inputs and the failing tests**

```python
import pickle

import numpy as np
import pandas as pd
import pytest

from tau3mu_gnn.model_eval import ModelEval as me

RNG = np.random.default_rng(1)


def write_npz(path, scores, tau_mass):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scores=scores, tau_mass=tau_mass)
    return path


def toy_npz(dir_, suffix=""):
    """GNN logits: signal ~ N(3, 1), background ~ N(0, 2) with flat tau mass; one NaN in each."""
    sig = np.append(RNG.normal(3, 1, 2000), np.nan)
    bkg = np.append(RNG.normal(0, 2, 20000), np.nan)
    s = write_npz(dir_ / f"raw_scores_and_tau_mass_signal{suffix}.npz", sig, np.full(sig.size, 1.78))
    b = write_npz(dir_ / f"raw_scores_and_tau_mass_background{suffix}.npz", bkg, RNG.uniform(1.4, 2.5, bkg.size))
    return s, b


def toy_pickle(path, extra_gnn_keys=False):
    """As stage 2 writes it: pandas Series. extra_gnn_keys: as the old (pre step 5) pickle."""
    data = {"ISO_score_sig": pd.Series(RNG.uniform(0.5, 1.0, 3000)),
            "ISO_score_bkg": pd.Series(RNG.uniform(0.0, 1.0, 30000)),
            "tau_mass": pd.Series(RNG.uniform(1.4, 2.5, 30000))}
    if extra_gnn_keys:
        data["GNN_score_sig"] = pd.Series(RNG.uniform(0, 1, 3000))
        data["GNN_score_bkg"] = pd.Series(RNG.uniform(0, 1, 30000))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def test_load_gnn_npz_drops_nan_only_with_signal(tmp_path):
    s, b = toy_npz(tmp_path)
    scores, mass, sig = me.load_gnn_npz(b, s)
    assert scores.size == mass.size == 20000 and sig.size == 2000
    scores, mass, sig = me.load_gnn_npz(b)
    assert scores.size == mass.size == 20001 and sig is None


@pytest.mark.parametrize("plot_mode, view, bdt_cut, name", [
    ("tau_hist_fpr", "normalized", True, "final_layer_tau_hist_fpr_normalized_with_bdt_cut.png"),
    ("tau_hist_fpr", "non_normalized", False, "final_layer_tau_hist_fpr.png"),
    ("tau_hist", "non_normalized", True, "final_layer_tau_hist_with_bdt_cut.png"),
])
def test_old_cli_tau_hist_names(tmp_path, plot_mode, view, bdt_cut, name):
    toy_npz(tmp_path, "_with_bdt_cut" if bdt_cut else "")
    me.tau_score_hist(str(tmp_path) + "/", "final_layer", "t", "W", plot_mode, view, bdt_cut)
    assert (tmp_path / name).is_file()
```

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: ... has no attribute 'load_gnn_npz'`)

Run: `$PYTEST tests/pipeline/test_eval.py -v`

- [ ] **Step 3: Split `tau_score_hist`**

```python
def load_gnn_npz(bkg_npz, sig_npz=None):
    """(background scores, background tau mass, signal scores or None) from stage 3 npz files.
    With sig_npz (FPR plots): NaN scores removed."""
    bkg = np.load(bkg_npz, allow_pickle=True)
    scores, tau_mass = bkg["scores"], bkg["tau_mass"]
    if sig_npz is None:
        return scores, tau_mass, None
    scores_signal = np.load(sig_npz, allow_pickle=True)["scores"]
    nan_mask = ~np.isnan(scores)
    return scores[nan_mask], tau_mass[nan_mask], scores_signal[~np.isnan(scores_signal)]


def tau_score_hist(root, mode, title, tau_channel, plot_mode, view, bdt_score_cut):
    """Old CLI: input files and plot name from root, plot_mode, view and bdt_score_cut."""
    normalize = view == "normalized"
    use_fpr = plot_mode.endswith("fpr")
    suffix = "_with_bdt_cut" if bdt_score_cut else ""
    if plot_mode.startswith("tau_hist") and not plot_mode.startswith("tau_hist_after"):
        model = "GNN"
        sig_npz = root + f"raw_scores_and_tau_mass_signal{suffix}.npz" if use_fpr else None
        scores, f_tau_mass, scores_signal = load_gnn_npz(root + f"raw_scores_and_tau_mass_background{suffix}.npz",
                                                         sig_npz)
    elif plot_mode.startswith("tau_hist_after"):
        model = "BDT"
        data = load_pkl(root, "tau_score_corr.pkl")[0]
        scores = data["GNN_score_bkg"]
        f_tau_mass = data["tau_mass"]
        scores_signal = data["GNN_score_sig"] if use_fpr else None
    else:
        raise ValueError(f"Unknown plot_mode: {plot_mode}")

    name_suffix = ("_normalized" if normalize else "") + suffix
    after = "_after" if plot_mode.startswith("tau_hist_after") else ""
    save_name = f"{mode}{after}_tau_hist{'_fpr' if use_fpr else ''}{name_suffix}.png"
    plot_tau_hist(scores, f_tau_mass, scores_signal, os.path.join(root, save_name), model, mode, title,
                  tau_channel, normalize)


def plot_tau_hist(scores, f_tau_mass, scores_signal, out_png, model, mode, title, tau_channel, normalize):
    """Background tau mass for several score cuts. scores_signal given: cuts at fixed FPR, else fixed cuts."""
    use_fpr = scores_signal is not None
    if use_fpr:
        labels = np.concatenate([np.ones(len(scores_signal)), np.zeros(len(scores))])
        scores_all = np.concatenate([scores_signal, scores])
        fpr, tpr, thresh_model = roc_curve(labels, scores_all)
    ...  # old body from `print("length of inferenced bkg data set is ", len(scores))` (old :411)
         # up to `plt.legend()` / the title block (old :477-480), unchanged
    plt.savefig(out_png, dpi=400)
    plt.close()
    print(f"{os.path.basename(out_png)} has been saved.")
```

The `...` is **the old code moved as is**: old lines `:411-480` (from the `print("length of inferenced ...")` to the `plt.title(...)` block). Do not change them. Delete only the old `suffix`/`save_name` block (old `:482-493`); `save_name` is now built in `tau_score_hist`. Check: the old fpr paths both computed `roc_curve` on `[signal, background]` with labels `[1, 0]` — the same as the 3 lines above.

- [ ] **Step 4: Run, expect PASS**

Run: `$PYTEST tests/pipeline/test_eval.py -v`

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/model_eval/ModelEval.py tests/pipeline/test_eval.py
/usr/bin/git commit -m "refactor(eval): tau_hist plot takes arrays and an output file"
```

---

### Task 2: ROC + SIC: explicit files, ISO only

**Files:**
- Modify: `src/tau3mu_gnn/model_eval/ModelEval.py:105-285` (`plot_roc_gnn_arch_comparison_full`, `plot_sic_full`) and `__main__` (`ROC_comp_full` branch)
- Test: `tests/pipeline/test_eval.py`

**Interfaces:**
- Produces:
  - `SIDEBAND_MASS, SIDEBAND_HALF_WIDTH, SIDEBAND_RANGE = 1.78, 0.08, (1.6, 2.0)`, `sideband_mask(tau_mass) -> np.ndarray`
  - `load_pickle(path) -> dict` (with `CPU_Unpickler`)
  - `plot_roc_full(gnn_curves: list[tuple[str, path, path]], bdt_pickle: path | None, out_png, fpr_targets, title, iso_cut=0.8) -> tuple[list, list, list, list[str]]`: `(fprs, tprs, threshs, labels)`, one entry per GNN curve, then one for the ISO BDT (label `"BDT ISO"`).
  - `plot_sic(fprs, tprs, threshs, labels, out_png, title) -> None`

- [ ] **Step 1: Write the failing tests** (append)

```python
FPR_TARGETS = [5e-2, 1e-2, 5e-3, 1e-3]


@pytest.mark.parametrize("extra_gnn_keys", [False, True])
def test_roc_and_sic(tmp_path, extra_gnn_keys):
    s, b = toy_npz(tmp_path)
    pkl = toy_pickle(tmp_path / "bdt" / "tau_score_corr.pkl", extra_gnn_keys)
    curves = me.plot_roc_full([("GNN toy", s, b)], pkl, tmp_path / "roc.png", FPR_TARGETS, "t")
    assert curves[3] == ["GNN toy", "BDT ISO"]
    me.plot_sic(*curves, tmp_path / "sic.png", "t")
    assert (tmp_path / "roc.png").is_file() and (tmp_path / "sic.png").is_file()


@pytest.mark.parametrize("cut", [0.5, 0.9])
def test_iso_cut(tmp_path, cut):
    s, b = toy_npz(tmp_path)
    pkl = toy_pickle(tmp_path / "bdt" / "tau_score_corr.pkl")
    _, _, threshs, _ = me.plot_roc_full([("GNN toy", s, b)], pkl, tmp_path / "roc.png", FPR_TARGETS, "t",
                                        iso_cut=cut)
    iso = threshs[-1]
    assert iso[np.isfinite(iso)].min() > cut
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Rewrite the two functions.** Keep every `plt` call and its order from the old code. The old loop over `gnn_data_locs[:-1]` skipped (with `continue`) entries without an `archs` label; the CLI now passes only labelled entries.

```python
SIDEBAND_MASS, SIDEBAND_HALF_WIDTH, SIDEBAND_RANGE = 1.78, 0.08, (1.6, 2.0)


def sideband_mask(tau_mass):
    return ((np.abs(tau_mass - SIDEBAND_MASS) > SIDEBAND_HALF_WIDTH)
            & (np.abs(tau_mass) > SIDEBAND_RANGE[0]) & (np.abs(tau_mass) < SIDEBAND_RANGE[1]))


def load_pickle(path):
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()


def plot_roc_full(gnn_curves, bdt_pickle, out_png, fpr_targets, title, iso_cut=0.8):
    """ROC of each GNN (label, signal npz, background npz) and of the ISO BDT (scores > iso_cut).
    Background: sideband only. Returns (fprs, tprs, threshs, labels) for plot_sic."""
    first_ROC = True
    tpr_targets, fprs, tprs, threshs, labels = [], [], [], [], []

    for label, sig_npz, bkg_npz in gnn_curves:
        gnn_sig = np.load(sig_npz, allow_pickle=True)
        gnn_bkg = np.load(bkg_npz, allow_pickle=True)
        gnn_sig_scores = gnn_sig["scores"]
        gnn_bkg_scores = gnn_bkg["scores"][sideband_mask(gnn_bkg["tau_mass"])]
        gnn_sig_scores = gnn_sig_scores[~np.isnan(gnn_sig_scores)]
        gnn_bkg_scores = gnn_bkg_scores[~np.isnan(gnn_bkg_scores)]
        print(bkg_npz)
        print(f"Size of signal dataset is {len(gnn_sig_scores)}.")
        print(f"Size of background dataset is {len(gnn_bkg_scores)}.")

        y = np.concatenate([np.ones(len(gnn_sig_scores)), np.zeros(len(gnn_bkg_scores))])
        fpr, tpr, thresh_model = roc_curve(y, np.concatenate([gnn_sig_scores, gnn_bkg_scores]))
        auc_full = auc(fpr, tpr)
        line, = plt.plot(fpr, tpr, linewidth=0.9, label=f"{label}, AUC: {auc_full:.3f}")
        fprs.append(fpr); tprs.append(tpr); threshs.append(thresh_model); labels.append(label)
        curve_color = line.get_color()
        # old loop body from `for k, fpr_cut in enumerate(fpr_targets):` (old :154-168), unchanged
        ...
        first_ROC = False

    if bdt_pickle is not None:
        data = load_pickle(bdt_pickle)
        mask = sideband_mask(data["tau_mass"])
        ISO_bkg = data["ISO_score_bkg"][mask]
        ISO_sig = data["ISO_score_sig"]
        ISO_bkg = ISO_bkg[ISO_bkg > iso_cut]
        ISO_sig = ISO_sig[ISO_sig > iso_cut]
        # old code from `ISO_scores = np.concatenate(...)` (old :198) to the end of the
        # `for i, _ in enumerate(fpr_targets):` loop (old :227), without the GNN_score lines
        # (old :180-190, :208, :211, :217-221). Then:
        labels.append("BDT ISO")

    # old plt.grid() ... plt.xlim(1e-5, 1) block (old :229-234), unchanged
    plt.title(title)
    print(f"{os.path.basename(out_png)} has been saved.")
    plt.savefig(out_png, dpi=400)
    plt.close()
    return fprs, tprs, threshs, labels


def plot_sic(fprs, tprs, threshs, labels, out_png, title):
    """SIC = TPR / sqrt(FPR) of each curve from plot_roc_full, with the maximum marked."""
    for fpr, tpr, thresh, label in zip(fprs, tprs, threshs, labels):
        sic = np.zeros_like(tpr, dtype=float)
        np.divide(tpr, np.sqrt(fpr), out=sic, where=fpr != 0)
        line, = plt.plot(tpr, sic, linewidth=0.9, label=label)
        # old code from `curve_color = line.get_color()` (old :259) to `plt.title(...)` (old :278),
        # with plt.title(title)
        ...
    print(f"{os.path.basename(out_png)} has been saved.")
    plt.savefig(out_png, dpi=400)
    plt.close()
```

The `...` are old code moved as is (line numbers refer to `ModelEval.py` at `4a65986`). Careful with the ISO block: the old code used `tpr_targets` from the first GNN curve; keep that.

Label check (pixel-identical plots): the old GNN ROC label was `f"{archs[i].replace('vs', '')} $\lambda$ = {lambdas[i]}, AUC: {auc_full:.3f}"` and the old SIC label `f"{archs[i].replace('vs', '')} $\lambda$ = {lambdas[i]}"`. So the CLI passes `label = f"{archs[i].replace('vs', '')} $\lambda$ = {lambdas[i]}"`.

Replace the old `ROC_comp_full` branch in `__main__`:

```python
    elif plot_mode == "ROC_comp_full":
        root_dir = "/work/kfuss/SVJ_GNN/output/comparison_plots"
        fpr_targets = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
        suffix = "_with_bdt_cut" if bdt_score_cut else ""
        # the last location is the BDT one; GNN locations without an archs label are not plotted (as before)
        gnn_curves = [(f"{arch.replace('vs', '')} $\\lambda$ = {lam}",
                       loc + f"raw_scores_and_tau_mass_signal{suffix}.npz",
                       loc + f"raw_scores_and_tau_mass_background{suffix}.npz")
                      for loc, arch, lam in zip(gnn_archs_location[:-1], archs or [], lambdas or [])]
        bdt_pickle = None if bdt_location is None else os.path.join(bdt_location, "tau_score_corr.pkl")
        curves = plot_roc_full(gnn_curves, bdt_pickle,
                               os.path.join(root_dir, f"GNNs_ROC_log_comparison_{tau_channel}_sculpt_{sculpt_degree}{suffix}_full.png"),
                               fpr_targets, f"ROC Curves: different GNN archs vs BDT with for {tau_channel} channel")
        plot_sic(*curves, os.path.join(root_dir, f"GNNs_SIC_comparison_{tau_channel}_sculpt_{sculpt_degree}{suffix}_full.png"),
                 f"SIC scores: different GNN archs and BDT with GNN score for {tau_channel} channel")
```

Note: the old code used `load_pkl(bdt_location, "tau_score_corr.pkl")[0]` (first file in the folder that **ends with** that name). In the real folders there is exactly one, `tau_score_corr.pkl`. Delete `plot_roc_gnn_arch_comparison_full` and `plot_sic_full` (only `__main__` used them; grep `src jobs` to be sure).

- [ ] **Step 4: Run, expect PASS**

Run: `$PYTEST tests/pipeline/test_eval.py -v`

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/model_eval/ModelEval.py tests/pipeline/test_eval.py
/usr/bin/git commit -m "refactor(eval): ROC and SIC take explicit files, ISO BDT only"
```

---

### Task 3: Stage 4 `pipeline/stages/evaluation.py`

**Files:**
- Create: `src/tau3mu_gnn/pipeline/stages/evaluation.py`
- Test: `tests/pipeline/test_eval.py`

**Interfaces:**
- Consumes: `load_gnn_npz`, `plot_tau_hist`, `plot_roc_full`, `plot_sic`, `load_pickle` (Tasks 1, 2); `Config` (`cfg.channel`, `cfg.gnn.model_name`, `cfg.gnn.gnn_mode`, `cfg.gnn.bdt_preselection_cut`).
- Produces: `INPUTS = ("signal_scores", "background_scores", "scores_pickle")`, `PLOTS = ("tau_hist_fpr", "tau_hist_fpr_normalized", "tau_hist", "roc", "sic")`, `check_inputs(inputs) -> None`, `run(cfg, inputs, out_dir) -> dict[str, Path]` (plot name → PNG).

- [ ] **Step 1: Write the failing tests** (append)

```python
from tau3mu_gnn.pipeline.stages import evaluation as eval_stage


def _stage_inputs(tmp_path):
    s, b = toy_npz(tmp_path / "3_gnn")
    return {"signal_scores": s, "background_scores": b,
            "scores_pickle": toy_pickle(tmp_path / "2_bdt" / "tau_score_corr.pkl")}


def test_stage_writes_all_plots(tmp_path, load, user):
    out_dir = tmp_path / "4_eval"
    outputs = eval_stage.run(load(user), _stage_inputs(tmp_path), out_dir)
    assert outputs == {name: out_dir / f"{name}.png" for name in eval_stage.PLOTS}
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(f"{n}.png" for n in eval_stage.PLOTS)


def test_missing_inputs_fail_before_plots(tmp_path, load, user):
    inputs = _stage_inputs(tmp_path)
    inputs["signal_scores"].unlink()
    inputs["scores_pickle"].unlink()
    with pytest.raises(FileNotFoundError) as err:
        eval_stage.run(load(user), inputs, tmp_path / "4_eval")
    assert "raw_scores_and_tau_mass_signal.npz" in str(err.value) and "tau_score_corr.pkl" in str(err.value)
    assert not (tmp_path / "4_eval").exists()


def test_npz_without_key_fails(tmp_path, load, user):
    inputs = _stage_inputs(tmp_path)
    np.savez_compressed(inputs["background_scores"], scores=np.zeros(3))
    with pytest.raises(ValueError, match="tau_mass"):
        eval_stage.run(load(user), inputs, tmp_path / "4_eval")


def test_pickle_without_key_fails(tmp_path, load, user):
    inputs = _stage_inputs(tmp_path)
    with open(inputs["scores_pickle"], "wb") as f:
        pickle.dump({"ISO_score_sig": pd.Series([0.9])}, f)
    with pytest.raises(ValueError, match="ISO_score_bkg"):
        eval_stage.run(load(user), inputs, tmp_path / "4_eval")
```

(`load` and `user` are the fixtures from `conftest.py`.)

- [ ] **Step 2: Run, expect FAIL** (`ImportError`)

- [ ] **Step 3: Write the stage**

```python
"""Stage 4: model evaluation plots from the stage 3 npz files and the stage 2 pickle.

Writes tau_hist_fpr.png, tau_hist_fpr_normalized.png (background tau mass for GNN score cuts at fixed
FPR), tau_hist.png (fixed cuts on sigmoid(score)), roc.png and sic.png (GNN vs ISO BDT, background
from the mass sidebands).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np

from tau3mu_gnn.pipeline.config import Config

log = logging.getLogger(__name__)

INPUTS = ("signal_scores", "background_scores", "scores_pickle")
PLOTS = ("tau_hist_fpr", "tau_hist_fpr_normalized", "tau_hist", "roc", "sic")
FPR_TARGETS = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
NPZ_KEYS = ("scores", "tau_mass")
PICKLE_KEYS = ("ISO_score_sig", "ISO_score_bkg", "tau_mass")


def check_inputs(inputs: Mapping[str, str | Path]) -> None:
    """Raise FileNotFoundError (all missing files) or ValueError (missing key) before any plot."""
    from tau3mu_gnn.model_eval.ModelEval import load_pickle

    missing = [str(inputs[k]) for k in INPUTS if not Path(inputs[k]).is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 4 inputs missing: {missing}")
    for key in ("signal_scores", "background_scores"):
        with np.load(inputs[key]) as npz:
            _check_keys(inputs[key], npz.files, NPZ_KEYS)
    _check_keys(inputs["scores_pickle"], load_pickle(inputs["scores_pickle"]).keys(), PICKLE_KEYS)


def _check_keys(path, found, expected):
    absent = [k for k in expected if k not in found]
    if absent:
        raise ValueError(f"{path} has no {absent} (has {sorted(found)})")


def run(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict[str, Path]:
    """Plot inputs ({signal_scores, background_scores}: stage 3 npz, scores_pickle: stage 2) into out_dir."""
    check_inputs(inputs)
    from tau3mu_gnn.model_eval import ModelEval as me

    g = cfg.gnn
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: out_dir / f"{name}.png" for name in PLOTS}
    sig, bkg = inputs["signal_scores"], inputs["background_scores"]
    log.info("stage 4 eval: %s -> %s", {k: str(inputs[k]) for k in INPUTS}, out_dir)

    for name, normalize in (("tau_hist_fpr", False), ("tau_hist_fpr_normalized", True)):
        me.plot_tau_hist(*me.load_gnn_npz(bkg, sig), outputs[name], "GNN", g.gnn_mode, g.model_name,
                         cfg.channel, normalize)
    me.plot_tau_hist(*me.load_gnn_npz(bkg), outputs["tau_hist"], "GNN", g.gnn_mode, g.model_name,
                     cfg.channel, False)
    curves = me.plot_roc_full([(f"GNN {g.model_name}", sig, bkg)], inputs["scores_pickle"], outputs["roc"],
                              FPR_TARGETS, f"ROC: GNN vs BDT ISO, {cfg.channel} channel",
                              iso_cut=g.bdt_preselection_cut)
    me.plot_sic(*curves, outputs["sic"], f"SIC: GNN vs BDT ISO, {cfg.channel} channel")
    return outputs
```

- [ ] **Step 4: Run all tests, expect PASS**

Run: `$PYTEST -v`

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/stages/evaluation.py tests/pipeline/test_eval.py
/usr/bin/git commit -m "feat(pipeline): model evaluation stage from config (step 7)"
```

---

### Task 4: Regression job (Kaito queues it)

**Files (not committed):** `regression/step7/run_regression_7.sh`, `regression/step7/compare_7.py`.

**What it proves:** the old CLI (`ModelEval.py` at the step 6b merge commit) and the new CLI give pixel-identical PNGs on the stored production inputs, and the new stage runs on the same files.

- [ ] **Step 1: Inputs** (copied to scratch, because `tau_hist` writes its PNGs next to the npz):
  - GNN: `/work/kfuss/SVJ_GNN/output/new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2/raw_scores_and_tau_mass_{signal,background}_with_bdt_cut.npz`
  - BDT: `/work/kfuss/SVJ_GNN/output/benchmark_W_BDT_vs1/tau_score_corr.pkl` (old pickle, still has `GNN_score_*`; 8,450,182 background rows).
- [ ] **Step 2: Runs** for `old` and `new` code (frozen copies, as in step 6), each in its own input copy `$S/{old,new}/gnn/` and `$S/{old,new}/bdt/`; `sed` the `root_dir = "/work/kfuss/SVJ_GNN/output/comparison_plots"` line to `$S/{old,new}/plots` and `grep` that the patch worked:
  1. `--plot_mode tau_hist_fpr --view both --bdt_cut true --gnn_mode final_layer --title '$\lambda = 0.0$' --tau_channel W --output_path $S/<v>/gnn/`
  2. `--plot_mode tau_hist --view non_normalized` (same other arguments)
  3. `--plot_mode ROC_comp_full --sculpt_degree final_ROC3 --gnn_archs_location $S/<v>/gnn/ $S/<v>/bdt/ --tau_channel W --lambdas 0.0,BDT_cut_0.8 5.0 --archs fl_8_sd_4_GINE_delete_muons_vs2 --bdt_location $S/<v>/bdt/ --bdt_cut true`
  4. New code only: `evaluation.run(cfg, {...npz copies without suffix, pickle}, $S/new/4_eval)` with `cfg = load_config(new/configs/example_W.yaml)`; expect 5 PNGs.
  SBATCH: `-p standard --account=t3 --mem=16G --time=2:00:00`.
- [ ] **Step 3: `compare_7.py`**: same set of PNG names in old and new (5 files: 3 tau hist + ROC + SIC); for each, `matplotlib.image.imread` arrays equal (shape and values). Report bytes-equal too (information only). `RESULT: PASS` if all pixel-equal and step 4 wrote 5 files. Report `regression/step7/report_${SLURM_JOB_ID}.txt`.
- [ ] **Step 4:** Smoke run locally only if it is light enough (the pickle has 8.5M rows; `roc_curve` on it takes some seconds). Then tell Kaito the `sbatch` command.

---

### Task 5: Roadmap

- [ ] **Step 1:** Sec. 6: mark step 7 `✅` **after the Task 4 regression passes**. Remove the two notes in step 7 (both are solved). Decision-log row: `| 2026-10-12 | Step 7: ModelEval split in load_gnn_npz / plot_tau_hist / plot_roc_full / plot_sic; GNN-input BDT curve removed (it was computed, never drawn). Stage 4 pipeline/stages/evaluation.py: inputs = stage 3 npz + stage 2 pickle, writes tau_hist_fpr(_normalized), tau_hist, roc, sic .png; ISO cut in the ROC = gnn.bdt_preselection_cut (">", as before). Old CLI pixel-identical (job <id>). Not refactored: tau_hist_after*, ROC_comp, score_cut. |`
- [ ] **Step 2: Commit**

```bash
/usr/bin/git add docs/wrapper_roadmap.md docs/superpowers/plans/2026-10-12-step7-model-eval.md
/usr/bin/git commit -m "docs(roadmap): step 7 done"
```
