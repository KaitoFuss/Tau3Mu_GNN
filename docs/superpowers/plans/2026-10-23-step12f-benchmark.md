# Step 12f — Stage benchmark (BDT-only limit): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A training stage that computes the BDT-only benchmark limit as a cut scan on `bdt_ISO` (one Combine run per cut, then a summary with the best cut), on the `bdt_infer` outputs.

**Architecture:** The old `analysis/datacard_creation_ISO.py` is the old GNN datacard script with `GNN_score` replaced by `bdt_ISO` (checked by diff: only that, print lines and hard-coded paths differ). So no new datacard code: `make_datacard` (step 8) gets a `score_branch` argument, the limit stage (step 8/8b) gets branch-agnostic core functions, and a thin training stage `pipeline/training/stages/benchmark.py` calls them with `bdt_ISO` and `cfg.benchmark.bdt_score_cut`. The analysis behaviour stays the same.

**Tech Stack:** Python 3.11, ROOT/RooFit (datacard), CMS Combine (env `Tau3Mu`), pytest. Tests: `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q`.

**Spec:** `docs/superpowers/specs/2026-10-23-step12-training-pipeline-design.md` (sec. 3 benchmark, sec. 4 benchmark row, sec. 5 `benchmark.bdt_score_cut`).

## Global Constraints

- Selection `bdt_ISO > cut` (the old script and stage 5's `GNN_score > cut` use `>`; spec sec. 6). `bdt_ISO = -1` (no candidate) never passes.
- Fit and limit settings from the training config: `cfg.limit` (`signal_weight` user; `tau_mass`, `mass_window`, `fit_range`, `cl` from `analysis_defaults_W.yaml`). Combine commands as stage 5 (`run_combine`).
- Cuts: `cfg.benchmark.bdt_score_cut` (user list; example 0.98–0.996 in 0.002 steps). Output layout as stage 5: `<out_dir>/cut_<cut>/` (datacard files, Combine logs, `limit.json`) and `<out_dir>/limits.{json,txt,png}` (best cut = lowest limit, smaller cut on a tie, missing cut = row marked).
- Inputs: the `bdt_infer` outputs `{signal,background}_with_bdt_scores.root` (branches `bdt_ISO`, `Selected_TauTo3Mu_fitted_mass`).
- Analysis unchanged: `stages/limit.run(cfg, inputs, out_dir, cut)`, `summarize(cfg, out_dir, write_dir)`, `run_scan(cfg, ...)` keep signatures and behaviour (existing tests unchanged).
- Use `/usr/bin/git` in the worktree. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch `pipeline-step12f-benchmark` is stacked on `pipeline-step12e-gnn-train` (PR #30).

## Review Focus

1. Same toy data with the score in `bdt_ISO` instead of `GNN_score` → identical datacard (text and counts) (Task 1, `test_score_branch_gives_same_card`).
2. Input without `bdt_ISO` → `BranchError` naming the file before any datacard work (Task 3, `test_benchmark_missing_bdt_iso`).
3. A cut not in `benchmark.bdt_score_cut` → `ValueError` (Task 3, `test_benchmark_rejects_unknown_cut`).
4. The summary reads only the benchmark cuts, not `limit.gnn_score_cut` (the training config has none) (Task 2/3, `test_summarize_cuts_explicit`).
5. Combine missing → error before the datacard work (Task 3: same check as stage 5, covered by `combine_executable()` first).

---

### Task 1: `make_datacard(score_branch=...)` and the old ISO CLI

**Files:**
- Modify: `src/tau3mu_gnn/analysis/datacard_creation_GNN_permuted.py` (`make_datacard`), `src/tau3mu_gnn/analysis/datacard_creation_ISO.py` (replace the script body by a `main(argv)` that builds the old paths and calls `make_datacard(..., score_branch="bdt_ISO")`, as step 8 did for the GNN script; keep `--model --channel --cut`)
- Test: `tests/pipeline/test_limit.py`

**Interfaces:**
- Produces: `make_datacard(sig_files, bkg_files, cut, signal_weight, out_dir, tau_mass=1.78, mass_window=0.060, fit_range=(1.62, 1.98), tree_name="Events", score_branch="GNN_score") -> DatacardResult` — selection `f"{score_branch} > {cut}"`; docstring updated.

- [ ] **Step 1: Failing tests** in `test_limit.py`:
  - `test_score_branch_gives_same_card`: copy the toy files with the branch renamed `GNN_score` → `bdt_ISO` (score values unchanged); `make_datacard(..., 3.9, ...)` on the originals and `make_datacard(..., 3.9, ..., score_branch="bdt_ISO")` on the copies → `datacard.txt` byte-identical, same `n_signal` / `n_data` fields of the result.
  - `test_iso_old_cli_builds_old_paths`: as `test_old_cli_builds_old_paths` (monkeypatch `make_datacard`, check the W paths `.../background_W/<model>/selected_signal_skimmed_no_resonances_with_scores_bdt_full.root` and the data file, `score_branch="bdt_ISO"`, cut from `--cut`).
- [ ] **Step 2: Run, expect FAIL.** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_limit.py -q`
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run, expect PASS** (all of `test_limit.py`).
- [ ] **Step 5: Commit** — `refactor(limit): make_datacard score_branch; ISO datacard script uses it`.

### Task 2: Branch-agnostic limit core

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/stages/limit.py`
- Test: `tests/pipeline/test_cut_scan.py`, `tests/pipeline/test_limit.py` (existing must pass unchanged)

**Interfaces:**
- Produces:
  - `limit.run_cut(inputs, out_dir, cut, *, score_branch: str, signal_weight: float, tau_mass: float, mass_window: float, fit_range: tuple[float, float], cl: float, tree_name: str) -> dict` — today's body of `run` after the "cut in config" check: input check with `(score_branch, "Selected_TauTo3Mu_fitted_mass")`, `combine_executable()` first, `make_datacard(..., score_branch=...)`, Combine, `limit.json`; same return dict.
  - `limit.summarize_cuts(cuts: Sequence[float], cl: float, out_dir, write_dir=None) -> dict` — today's body of `summarize` with `cuts` / `cl` explicit.
  - `run(cfg, inputs, out_dir, cut)` and `summarize(cfg, out_dir, write_dir)` become thin wrappers (GNN branch, `cfg.limit.gnn_score_cut`); `run_scan` unchanged.
- [ ] **Step 1: Failing test** `test_summarize_cuts_explicit` in `test_cut_scan.py`: `write_limit` for cuts 0.98 and 0.99 (limits 2.0, 1.5) → `limit.summarize_cuts([0.98, 0.99], 0.9, tmp_path)["best_cut"] == 0.99`, files written; a cut list without a limit file → missing row.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** (move code, no logic change).
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_limit.py tests/pipeline/test_cut_scan.py -q` → all pass.
- [ ] **Step 5: Commit** — `refactor(limit): branch-agnostic run_cut / summarize_cuts`.

### Task 3: Training stage `benchmark`

**Files:**
- Create: `src/tau3mu_gnn/pipeline/training/stages/benchmark.py`
- Test: `tests/pipeline/test_benchmark.py` (new)
- Modify: `docs/wrapper_roadmap.md` (12f ✅)

**Interfaces:**
- Consumes: Task 2 `run_cut`, `summarize_cuts`, `cut_dir_name`; `TrainingConfig` (`benchmark.bdt_score_cut`, `limit`, `inputs.tree_name`); `combine_executable`.
- Produces (12g calls these per job):
  - `SCORE_BRANCH = "bdt_ISO"`
  - `run(cfg: TrainingConfig, inputs: Mapping[str, str | Path], out_dir, cut: float) -> dict` — `ValueError` if `cut not in cfg.benchmark.bdt_score_cut`; then `run_cut(..., score_branch="bdt_ISO", ...)`.
  - `summarize(cfg: TrainingConfig, out_dir, write_dir=None) -> dict` → `summarize_cuts(cfg.benchmark.bdt_score_cut, cfg.limit.cl, ...)`.
  - `run_scan(cfg, inputs, out_dir) -> dict` — local loop over all cuts + summarize (as stage 5's `run_scan`: failed Combine of a cut = missing row).
- [ ] **Step 1: Failing tests** (`tests/pipeline/test_benchmark.py`; toy files like `toy_stage3_file` of `test_limit.py` but with `bdt_ISO ~ U(0, 1)` (and some `-1`) instead of `GNN_score`; training config with `benchmark.bdt_score_cut: [0.5, 0.8]`):
  - `test_benchmark_rejects_unknown_cut` → `ValueError`.
  - `test_benchmark_missing_bdt_iso` → `BranchError` naming the file, no `cut_*` folder content.
  - `@needs_combine test_benchmark_scan_end_to_end`: `run_scan` → `limits.json` with both cuts, a finite best limit, `cut_0.5/datacard.txt` selects `bdt_ISO > 0.5` (counts as `expected_counts` with the renamed branch).
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** (docstring style of `stages/limit.py`; `log.info` per cut).
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` → all pass.
- [ ] **Step 5: Roadmap** 12f ✅ (stage, reuse of step 8 code, plan path).
- [ ] **Step 6: Commit** — `feat(training): stage benchmark (BDT-only limit cut scan, step 12f)`.

## Real-data check (optional, no SLURM)

On the `bdt_infer` outputs of the small raw skim (frozen BDT pickle as a stand-in for the new one; `phase_a_vs1/2_bdt/*_with_bdt_scores.root`): cuts 0.98–0.996. Expected: few background events per cut → Combine may fail for high cuts (missing rows, as in the small test run). Report rows and run time. Not a test.
