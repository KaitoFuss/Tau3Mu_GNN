# Step 12g — Training driver (`tau3mu-train`): Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the training stages (12a–12f) into SLURM jobs, phases, version folders and manifests, with a GPU job for the GNN training and the command `tau3mu-train` — the same style as the analysis driver (steps 9, 10, 10b).

**Architecture:** Reuse the analysis driver machinery (`slurm.py`, `runs.py`, `manifest.py`, `storage.py`, `driver._submit`) after removing its few analysis-only assumptions (run name, job module, per-job account / gres, the settings snapshot). Add `pipeline/training/{manifest,jobs,driver,cli}.py`. The analysis behaviour is unchanged (its tests stay as they are).

**Tech Stack:** Python 3.11, SLURM (`sbatch --parsable`), dCache via `storage.py`, pytest with the existing `fake_sbatch` / `fake_voms` fixtures. Tests: `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q`.

**Spec:** `docs/superpowers/specs/2026-10-23-step12-training-pipeline-design.md` (sec. 3 chain, sec. 8 versions / layout / commands, sec. 10 SLURM / GPU). Step 9 driver spec `docs/superpowers/specs/2026-10-12-step9-driver-design.md` (same rules: new version per run, manifest, upstream check in the first job of a phase, no local phase runs).

## Global Constraints

- **Phases, versions, upstream** (spec sec. 8): `bdt` (jobs `skim` → `bdt_train` → `bdt_infer`), `gnn` (`npvs` → `graphs_signal`; `graphs_background`; both → `gnn_train`), `benchmark` (`benchmark_limit_<cut>` per cut of `benchmark.bdt_score_cut`, in parallel → `benchmark_summary` with `afterany`). Folders `<output_base>/<training_name>/{bdt,gnn,benchmark}_vsN`. Upstream: `gnn` ← `bdt`, `benchmark` ← `bdt`. **`graphs_background` does not wait for `npvs`** (npvs weights are for signal only, 12b).
- `all` = `bdt`, then `gnn` and `benchmark` both wait `afterok` on `bdt_infer` (the job that writes the `bdt` manifest), as the analysis `all` (step 10b).
- Manifest of each phase: written by its last job (`bdt_infer`, `gnn_train`, `benchmark_summary`); outputs relative to the phase folder. The first job of `gnn` / `benchmark` checks the `bdt` upstream (phase name, files exist; settings changes = warning), as `manifest.check_upstream`.
- Settings snapshot (`config_used.yaml`, manifest `settings`): from the training config; per phase the sections it depends on: `bdt`: channel, inputs, branch_map, resonance_skim, skim, sideband_half_width, bdt_training; `gnn`: + npvs, graphs, gnn_training; `benchmark`: bdt sections + benchmark, limit. Never write `comet.api_key` into any file (replace with `"***"`).
- SLURM: resources from `cfg.slurm.jobs[<job>]`; **per-job `account` and `gres`** (`SlurmJob.account` / `.gres`, 12a) → `#SBATCH --account=<job account or slurm.account>` and `#SBATCH --gres=<gres>` only when set. The analysis jobs have neither, so their scripts stay byte-identical.
- Training input check before work: stage 0 of the training checks `branches.training_required_branches(channel)` (adds `PV_npvs`; roadmap 12b/12d note).
- Graph stage output must not carry the input file: the graph folder holds a symlink `input.root` → the 10+ GB stage 2 file; `storage.put` of the folder would copy it. Remove the symlink after the build (the `.pt` loads without it: `raw_file_names` is then empty).
- No local phase runs (as step 9 D3). Single-stage debug runs (`stage <name>`) are **not** in 12g (spec sec. 13 leaves it to the builder; deferred).
- Use `/usr/bin/git` in the worktree. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never print the Comet key.
- Branch `pipeline-step12g-training-driver` is stacked on `pipeline-step12f-benchmark` (PR #32).

## Review Focus

1. Analysis job scripts unchanged byte for byte after the SLURM generalisation (Task 1, `test_analysis_script_unchanged`).
2. `gnn_train` script has `--account=gpu_gres`, `--gres=gpu:1`, partition `gpu`; no other job has `--gres` (Task 1 and Task 5, `test_gpu_job_flags`).
3. `graphs_background` has no dependency on `npvs`; `graphs_signal` has; `gnn_train` waits for both graph jobs (Task 5, `test_gnn_phase_dependencies`).
4. The Comet API key never appears in `config_used.yaml`, a manifest or a job script (Task 2 and Task 5, `test_no_comet_key_in_files`).
5. A `gnn` / `benchmark` phase started with `--from` a `bdt` version that has no manifest → clear error at submit; with a manifest whose files are missing → `UpstreamError` in the first job (Task 4 and Task 5).

---

### Task 1: SLURM layer for both pipelines

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/config.py` (`Config.run_name` property = `analysis_name`), `src/tau3mu_gnn/pipeline/training/config.py` (`TrainingConfig.run_name` = `training_name`), `src/tau3mu_gnn/pipeline/slurm.py`, `src/tau3mu_gnn/pipeline/driver.py` (`_submit`: `cfg.run_name` instead of `cfg.analysis_name`)
- Test: `tests/pipeline/test_slurm.py`

**Interfaces:**
- Produces:
  - `Job(name, resources, args, after=(), dependency="afterok", after_ids=(), module="tau3mu_gnn.pipeline.jobs")` — `module` is the `python -m` module of the job command.
  - `render_script(cfg, job, version_dir, log_folder, env, scratch_root=...)` uses `cfg.run_name`, `job.module`, `res.account or cfg.slurm.account`, and adds `#SBATCH --gres=<res.gres>` after `--cpus-per-task` only if `res.gres`.
  - `slurm_job_name(run_name, version, job)`: unchanged for `phase_*`; version prefixes `bdt` / `gnn` / `benchmark` get the codes `TB` / `TG` / `TL` (constant `VERSION_CODES`); other → `D` as today. `SHORT_JOB` gets short names for the training jobs (`bdt_train`→`bdtT`, `bdt_infer`→`bdtI`, `graphs_signal`→`grS`, `graphs_background`→`grB`, `gnn_train`→`gnnT`, `benchmark_summary`→`bsum`; `benchmark_limit_<cut>`→`bl<cut>`).
- [ ] **Step 1: Failing tests:** `test_analysis_script_unchanged` (render the analysis `skim` and `limit_3.9` scripts; compare with the strings rendered by the code before the change — capture them as expected literals in the test from the current `main` behaviour); `test_job_module_and_gpu_flags` (training toy config: `gnn_train` → `--account=gpu_gres`, `--gres=gpu:1`, `-m tau3mu_gnn.pipeline.training.jobs`; `npvs` → `--account=t3`, no `--gres`); `test_training_job_names` (`slurm_job_name("toy_train", "gnn_vs2", "gnn_train") == "TG2.gnnT.toy_train"`).
- [ ] **Step 2: Run, expect FAIL.** **Step 3: Implement.** **Step 4: Run** `tests/pipeline/test_slurm.py tests/pipeline/test_driver.py` → pass.
- [ ] **Step 5: Commit** — `feat(slurm): per-job account and gres, job module, run name for both pipelines`.

### Task 2: Versions and manifests for the training

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/runs.py` (`create_version(cfg, prefix, parent=None, settings=None)`: `settings` default `config_dict(cfg)`; `pick_version` / `pick_upstream` take `upstream_map: Mapping[str, str | None] = UPSTREAM`), `src/tau3mu_gnn/pipeline/manifest.py` (`write_manifest(..., settings=None)` default `phase_settings(cfg, phase)`; `check_upstream(cfg, upstream_dir, phase, upstream_map=UPSTREAM, settings_of=phase_settings)`)
- Create: `src/tau3mu_gnn/pipeline/training/manifest.py`
- Test: `tests/pipeline/test_training_manifest.py` (new); `tests/pipeline/test_runs.py`, `test_manifest.py` unchanged and green

**Interfaces:**
- Produces (`training/manifest.py`):
  - `PHASES = ("bdt", "gnn", "benchmark")`, `UPSTREAM = {"bdt": None, "gnn": "bdt", "benchmark": "bdt"}`, `PHASE_SECTIONS` (Global Constraints).
  - `config_dict(cfg: TrainingConfig) -> dict` — all user and frozen values as plain YAML types (paths relative to the repo root as the analysis does), `comet.api_key` → `"***"`.
  - `phase_settings(cfg, phase) -> dict`, `create_version(cfg, phase) -> str`, `write_manifest(cfg, phase_dir, phase, upstream, outputs, missing=()) -> dict`, `check_upstream(cfg, upstream_dir, phase) -> dict`, `pick_upstream(cfg, phase, from_=None) -> str` — thin wrappers over the generic functions with the training maps.
- [ ] **Step 1: Failing tests:** version folder `bdt_vs1` with `config_used.yaml` (no key string, `comet.api_key == "***"`); `pick_upstream(cfg, "gnn")` → highest complete `bdt_vsN`; `check_upstream` → `UpstreamError` for a wrong phase name / missing output file, warning on a changed `bdt_training` value; `phase_settings(cfg, "gnn")` contains `graphs` but `phase_settings(cfg, "bdt")` does not.
- [ ] **Step 2–4:** fail → implement → `tests/pipeline -q` green (analysis manifest tests unchanged).
- [ ] **Step 5: Commit** — `feat(training): versions and manifests for the training phases`.

### Task 3: Stage fixes needed by the jobs

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/branches.py` (`check_input_branches(..., raw=False, extra=())`), `src/tau3mu_gnn/pipeline/stages/resonance_skim.py` (`run(cfg, out_dir, extra_branches=())` passes `extra`), `src/tau3mu_gnn/pipeline/training/stages/graphs.py` (remove the `input.root` symlink after the build)
- Test: `tests/pipeline/test_branches.py`, `tests/pipeline/test_stage_resonance_skim.py`, `tests/pipeline/test_graphs.py`

**Interfaces:**
- Produces: `resonance_skim.run(cfg, out_dir, extra_branches: Sequence[str] = ())`; graph folder after `graphs.run` contains only `processed/`.
- [ ] **Step 1: Failing tests:** an input without `PV_npvs` passes the analysis check but fails with `extra=("PV_npvs",)` (message names the file and `PV_npvs`); after `graphs.run`, `<sample>_W/input.root` does not exist and the dataset still loads with `ConeGraphProducer(folder, name)`.
- [ ] **Step 2–4:** fail → implement → green. **Step 5: Commit** — `fix(training): stage 0 checks the training branches; graph folders without the input link`.

### Task 4: Training job functions (`training/jobs.py`)

**Files:**
- Create: `src/tau3mu_gnn/pipeline/training/jobs.py`
- Test: `tests/pipeline/test_training_jobs.py` (new; pattern of `tests/pipeline/test_jobs.py`: `scratch` fixture, monkeypatched stage `run` functions writing small files, local `output_base`)

**Interfaces:**
- Consumes: Task 2 wrappers; stages `resonance_skim`, `skim`, `bdt_train`, `bdt_infer`, `npvs`, `graphs`, `gnn_train`, `benchmark`; `storage`; `jobs._scratch` / `_read` / `_name` pattern.
- Produces: folder names `SKIM_DIR="1_skim"`, `BDT_TRAIN_DIR="bdt_train"`, `BDT_DIR="2_bdt"`, `NPVS_DIR="npvs"`, `GRAPHS_DIR="graphs"`, `GNN_TRAIN_DIR="gnn_train"` (stage 0 in `0_resonance_skim` as the analysis). Job functions (each: scratch → stage → `storage.put(scratch, phase_dir)`):
  - `skim(cfg, phase_dir)` — stage 0 (`extra_branches=TRAINING_EXTRA[channel]`) + stage 1.
  - `bdt_train(cfg, phase_dir)` — reads `<phase_dir>/1_skim/*`.
  - `bdt_infer(cfg, phase_dir)` — classifier `<phase_dir>/bdt_train/ISO_classifiers.pck`; writes the **bdt manifest** (`skim_<s>`, `bdt_<s>`, `classifier`, `bdt_train_summary`).
  - `npvs(cfg, phase_dir, upstream_dir)` — first job of `gnn`: `check_upstream(cfg, upstream_dir, "gnn")`; inputs `skim_<s>` of the upstream.
  - `graphs(cfg, phase_dir, upstream_dir, sample)` — `bdt_<sample>` of the upstream; signal: npvs file `<phase_dir>/npvs/signal_<ch>_npvs_weights.npz`; output into `<phase_dir>/graphs/`. `graphs_background` also checks the upstream (it may start first).
  - `gnn_train(cfg, phase_dir, upstream_dir)` — graphs from `<phase_dir>/graphs`; Comet on; writes the **gnn manifest** (`model_dir` = `gnn_train/<model_name>`, `model_name`, `npvs_weights`, `graphs_<s>`, `gnn_train_summary`). 12h reads `model_dir` / `model_name` from it.
  - `benchmark_limit(cfg, phase_dir, upstream_dir, cut)` — first job of `benchmark`: `check_upstream(..., "benchmark")`; inputs `bdt_<s>`; copies `cut_<cut>/` also on failure (as the analysis `limit` job).
  - `benchmark_summary(cfg, phase_dir, upstream_dir)` — `benchmark.summarize` into scratch; writes the **benchmark manifest** (`limits_json`, `limits_txt`, `limits_png`, `missing` cuts).
  - `main(argv=None)` — `python -m tau3mu_gnn.pipeline.training.jobs <job> --config --phase-dir [--upstream] [--cut] [--sample]`, loads `load_training_config`, required-argument table as `jobs.main`.
- [ ] **Step 1: Failing tests** (fake stages): phase `bdt` end to end in a local `output_base` (three job calls) → manifest with the four output keys; `gnn` with a missing upstream file → `UpstreamError` before any stage call; `graphs` signal reads the npvs file of the phase, background passes `npvs_file=None`; `benchmark_limit` copies `cut_<cut>/` when the stage raises; `main` dispatch and required-argument errors.
- [ ] **Step 2–4:** fail → implement → green. **Step 5: Commit** — `feat(training): job functions of the training phases`.

### Task 5: Training driver and `tau3mu-train`

**Files:**
- Create: `src/tau3mu_gnn/pipeline/training/driver.py`, `src/tau3mu_gnn/pipeline/training/cli.py`
- Modify: `pyproject.toml` (`[project.scripts] tau3mu-train = "tau3mu_gnn.pipeline.training.cli:main"`), `docs/wrapper_roadmap.md` (12g ✅)
- Test: `tests/pipeline/test_training_driver.py` (new; `fake_sbatch`, `ENV`, `fake_voms` from `test_slurm.py`, `complete()` pattern of `test_driver.py`)

**Interfaces:**
- Consumes: Task 1 `Job(module=...)`, `driver._submit`, `submit_jobs`, `check_proxy`, `chain_hours`; Task 2 wrappers.
- Produces: `MODULE = "tau3mu_gnn.pipeline.training.jobs"`, `LAST_JOB = {"bdt": "bdt_infer", "gnn": "gnn_train", "benchmark": "benchmark_summary"}`; `submit_phase(cfg: TrainingConfig, phase: str, from_: str | None = None, dry_run=False, env=None) -> Submission`; `submit_all(cfg, dry_run=False, env=None) -> list[Submission]` (bdt, then gnn and benchmark with `after_ids=(bdt_infer id,)`; one proxy check over the longest chain bdt + max(gnn, benchmark); on a later `SubmitError` re-raise with the scancel list as the analysis `submit_all`). CLI: `tau3mu-train {all, bdt, gnn, benchmark} --config FILE [--dry-run]`, `gnn` / `benchmark` take `--from bdt_vsN`; same output and error handling as `pipeline/cli.py`.
- [ ] **Step 1: Failing tests:** `test_gnn_phase_dependencies` (dry run: `graphs_signal` after `npvs`, `graphs_background` no dependency, `gnn_train` after both); `test_gpu_job_flags` (the written `gnn_train.sh`); `test_all_waits_for_bdt_infer` (real fake sbatch: gnn and benchmark first jobs have `--dependency=afterok:<bdt_infer id>`); `test_benchmark_jobs_per_cut` (one `benchmark_limit_<cut>` per cut, summary `afterany`); `test_no_comet_key_in_files` (after `all`, no file under the log folder or the version folders contains the toy key); `test_from_requires_complete_bdt` (`--from bdt_vs1` without manifest → error); `test_cli_dry_run` and `test_console_script` (entry point in `pyproject.toml`).
- [ ] **Step 2–4:** fail → implement → `tests/pipeline -q` green.
- [ ] **Step 5: Roadmap** 12g ✅ (commands, phases, folders, GPU job, deferred: `stage` debug runs). Note for Kaito: `pip install -e .` again for the new console script.
- [ ] **Step 6: Commit** — `feat(training): training driver and tau3mu-train (step 12g)`.

## Real check (Kaito queues, after the PR)

`tau3mu-train all --config <small training config> --dry-run`, then a real small run on T3 (the small raw inputs of `kaito_W_raw_small`): all jobs COMPLETED, the GPU job on `gpu` with `gpu_gres`. I prepare the small config; Kaito submits.
