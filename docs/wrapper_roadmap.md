# Pipeline Wrapper — Roadmap

Living document. Update it after each mini step.

## 1. Goal

Another physicist uses **one config file** and runs **three phase commands**, one after the other.
At the end he gets the **expected exclusion limit** on B(τ → 3μ).

```bash
# names are placeholders
tau3mu-run phase-a --config my_analysis.yaml   # stages 0-2
tau3mu-run phase-b --config my_analysis.yaml   # stages 3-4
tau3mu-run phase-c --config my_analysis.yaml   # stage 5

tau3mu-run all     --config my_analysis.yaml   # optional: A → B → C in one go
```

Why phases: if a setting changes in a later stage, only that phase and the phases after it need a rerun.

Stages:

0. Resonance skim (raw NanoAOD → `*_skimmed_no_resonances.root`) — code comes later from Kaito
1. Skimming (dZ + cos α mask, `Slimmed_*` → `Selected_*`)
2. BDT inference (ISO classifier only)
3. GNN inference (only events with BDT ISO score > preselection cut)
4. Model evaluation (plots)
5. Limit setting (datacard + Combine + read-out)

Phases:

| Phase | Stages | Rerun when ... changes |
|---|---|---|
| A | 0–2 | inputs, branch map, skim cuts, BDT |
| B | 3–4 | GNN settings, BDT preselection cut |
| C | 5 | GNN score cut(s), fit / Combine settings |

## 2. Fixed decisions (from Kaito)

- The GNN model is **fixed**. It is the model in the current job scripts. No training in the wrapper.
- The BDT is **fixed**: `ISO_classifiers_24feb2026_v0.pck` only. No BDT training in the wrapper.
- **Superseded by step 12 (2026-10-23):** a separate **training pipeline** (`tau3mu-train`) trains the BDT and the GNN. The analysis keeps the frozen models as the default; a new analysis key `models:` selects a training run instead. Spec: `docs/superpowers/specs/2026-10-23-step12-training-pipeline-design.md`.
- The user passes **one config file**. It holds:
  - the analysis parameters (channel, cuts, paths, ...),
  - a **branch map**: "my branch name" → "his branch name". His ROOT trees can use other names.
- Work goes in **mini steps**. No one-shot rewrite.
- Robust and easy to use is more important than fast.
- If something is unclear: **ask Kaito**.

## 3. Success criteria

- A new user with the conda env and a config file gets a limit with three phase commands (A → B → C).
- No path to `/work/kfuss`, `/pnfs/.../kfuss` or `/eos/.../kfuss` is needed in code.
- A wrong config fails **early** with a clear message (missing file, missing branch, bad value).
- With Kaito's config, the wrapper gives the **same limit** as the current job scripts (regression check).
- Each phase can be rerun alone. Each stage can also run alone (for debugging).
- A phase stops early with a clear message if the outputs of the phase before it are missing.
- A phase warns if the outputs of the phase before it came from different upstream settings (stale outputs).

## 4. Current state (as found on 2026-10-07)

### 4.1 Current "production" settings (W channel)

From `jobs/gnn_score_creation/create_score_W_*.sh` and `jobs/analysis/create_datacard_permuted.sh`:

| Parameter | Value |
|---|---|
| Channel | `W` |
| Model | `new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2` |
| Arch | `random` |
| Conv type | `GINE` |
| GNN mode | `final_layer` |
| Unsculpting mode | `delete_muons` |
| BDT preselection cut | `0.8` |
| GNN score cut (datacard) | `3.9` |
| BDT classifier (used) | `output/benchmark_W_BDT_vs1/ISO_classifiers_24feb2026_v0.pck` |
| BDT classifier (dropped) | `output/benchmark_W_BDT_vs1/GNN_classifiers_24feb2026_v0.pck` |

### 4.2 Current stages and entry points

| Stage | Job script | Python entry point |
|---|---|---|
| Skimming | `jobs/graph_creation/skimming.sh` | `tau3mu_gnn.utils.skimmer` |
| BDT inference | `jobs/bdt/BDT_pipeline_W.sh` | `tau3mu_gnn.bdt_training.BDT_ScoreProducer` |
| GNN inference | `jobs/gnn_score_creation/create_score_W_{signal,bkg}.sh` | `tau3mu_gnn.scripts.ScoreCreation` |
| Model eval | same job scripts + `jobs/model_eval/ROC_curves_comparison.sh` | `tau3mu_gnn.model_eval.ModelEval` |
| Limit | `jobs/analysis/create_datacard_permuted.sh` | `tau3mu_gnn.analysis.datacard` (was `datacard_creation_GNN_permuted`), `combine`, `tau3mu_gnn.analysis.readout_limit` |

### 4.3 Known problems for a new user

- **Hardcoded paths** in ~19 Python files (`/work/kfuss`, `/pnfs/.../kfuss`, `/eos/...`).
- **Hardcoded input file names** per channel (e.g. `selected_signal_skimmed_no_resonances_with_scores.root`, HF `part1..4`).
- **Hardcoded branch names** everywhere (`Slimmed_TauTo3Mu_*`, `Selected_TauTo3Mu_*`, `TauPlusMET_*`, `Muon_*[idx]`, `GNN_score`).
- Skimmer renames `Slimmed_*` → `Selected_*`. Later stages expect `Selected_*`.
- `BDT_ScoreProducer.py` runs **two** BDTs: one with `GNN_score` as input (`classifiers_gnn`) and one with relative isolation (`classifiers_iso`). This makes a GNN → BDT → GNN loop. Only the ISO BDT is used now.
- Many settings live only in bash variables inside SLURM scripts.
- Job scripts hardcode the conda path `/work/kfuss/miniforge3`.
- Storage copy (`xrdcp` to dCache) is baked into each job script.
- Limit setting needs CMS Combine and a VOMS proxy.

## 5. Design (agreed in chat, 2026-10-07)

- **Config**: frozen repo defaults + one short user file (see 5.1). The same user file is used for **all** phases. Each phase reads only the sections it needs.
- One Python driver with one sub-command per phase (`bdt`, `gnn`, `limit`; were `phase-a`, `phase-b`, `phase-c` until step 12k). Each stage is also callable alone.
- One thin bash wrapper that activates the env and calls the driver.
- **Hand-off between phases**: each phase run writes to a new version folder `<phase>_vsN` under `<output_base>/<analysis_name>/`, plus a small manifest (list of output files + snapshot of the config used). The next phase uses the highest complete upstream version, or the one given with `--from`. It reads the manifest, checks the files exist, and compares the upstream settings to the current config. These checks run **inside the first job of the phase** (at run time), not at submit time, because with `all` the upstream outputs do not exist yet when the job is submitted.
- Each stage is a plain Python function/CLI. It knows **nothing** about SLURM, HTCondor or dCache.
- A separate, thin **scheduler layer** submits the stages of **one phase** as SLURM jobs, with dependencies inside the phase (`sbatch --dependency=afterok:...`). Independent jobs run in parallel (e.g. signal and background GNN inference). Phases are **not** auto-chained; the user starts the next phase. Later an HTCondor layer can replace this layer without touching the stages.
- **Storage**: every job works in node-local `/scratch/$USER/$SLURM_JOB_ID`. At the end it copies its outputs with `xrdcp` to `<output_dir>` (from config, e.g. on dCache) and cleans scratch. Copy logic is **one shared helper**, not repeated per stage. Read through `/pnfs`, write with `xrdcp` (`pipeline/storage.py`). A local `output_base` also works.
- **Output layout** (everything is kept; every phase run is a new version, see step 9):

```
<output_base>/<analysis_name>/                  results (dCache or local)
├── bdt_vs1/                                     (phase_a_vs1 before step 12k)
│   ├── manifest.json
│   ├── config_used.yaml                        user file + defaults as loaded (merged view)
│   ├── 0_resonance_skim/                       only if stage 0 copies (non-empty branch map)
│   ├── 1_skim/{signal,background}.root
│   ├── 2_bdt/...                               stage 2 outputs (ROOT + pickle)
│   └── logs/<job>-<jobid>.{out,err}
├── gnn_vs1/                                     (phase_b_vs1 before step 12k)
│   ├── manifest.json, config_used.yaml, logs/
│   ├── 3_gnn/{signal,background}_with_gnn_scores.root, raw_scores_and_tau_mass_{signal,background}.npz
│   └── 4_eval/*.png
├── limit_vs1/                                   (phase_c_vs1 before step 12k)
│   ├── manifest.json, config_used.yaml, logs/
│   ├── cut_<cut>/...                           stage 5 per cut (step 8)
│   └── limits.{json,txt,png}                   summarize (step 8b)
└── debug/<stage>_vsN/                          single-stage runs, config_used.yaml + logs/, no manifest

<paths.log_dir>/<analysis_name>/<phase or debug stage>_vsN/
├── <job>.sh                                    generated job scripts
└── <job>-<jobid>.{out,err}                     SLURM logs
```

- **Branch map**: applied **once**, in stage 0 (also when the resonance skim itself is skipped), through shared helpers in `pipeline/branches.py` (`check_input_branches`, `rename_columns`). It renames his raw names → Kaito's raw names (`Slimmed_TauTo3Mu_*`, `Slimmed_PFCands_*`, `TauPlusMET_*`, `run`, `luminosityBlock`, `event`; list in `pipeline/branches.py`). All later stages run unchanged on Kaito's names. The prefix change `Slimmed_*` → `Selected_*` stays inside the skimmer.
- **Required branches**: one explicit list in code (38 for W: `COMMON` + `PER_CHANNEL["W"]` in `pipeline/branches.py`). After mapping, every required branch must exist. If one is missing: **crash loudly**, list all missing branches, do no work. No fallbacks, no computed substitutes.

### 5.1 Config

Merged view (defaults + user file). Kaito's W values shown.

```yaml
analysis_name: W_new_hope_v2
channel: W                              # user file only; selects analysis_defaults_W.yaml

paths:
  output_base: root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/$USER/tau3mu_pipeline
  log_dir: /work/$USER/SVJ_GNN/tau3mu_logs   # job scripts + SLURM logs (local disk)
  conda_env:  /work/<user>/miniforge3/envs/Tau3Mu   # optional; default: the active env at submit

inputs:
  signal:     [/pnfs/.../signal_W/*.root]
  background: [/pnfs/.../data_W/*.root]
  tree_name:  Events
  already_resonance_skimmed: true

branch_map:                             # Kaito name: user name; omitted = same name
  Slimmed_TauTo3Mu_fitted_pt: Tau3Mu_fit_pt

skim:
  dz_max: 0.5
  cos_alpha_2d_min: 0.5

bdt:
  classifier: output/benchmark_W_BDT_vs1/ISO_classifiers_24feb2026_v0.pck

gnn:
  model: new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2
  arch: random
  conv_type: GINE
  gnn_mode: final_layer
  unsculpting_mode: delete_muons
  bdt_preselection_cut: 0.8

limit:
  gnn_score_cut: [3.9]                  # one value or a list -> scan; default [3.9]
  signal_weight: <REQUIRED, no default> # user computes it from his dataset
  mass_window: 0.060
  fit_range: [1.62, 1.98]
  cl: 0.90

```

`slurm:` (account + per-job partition / mem / time / cpus) is in the frozen defaults file (step 9, D7).

**Split into two files** (decided):

- `analysis_defaults_<channel>.yaml` (in repo, frozen): `gnn.*`, `bdt.classifier`, `skim.*`, `gnn.bdt_preselection_cut`, `limit.mass_window`, `limit.fit_range`, `limit.cl`, `slurm`.
- `my_analysis.yaml` (user, short): `analysis_name`, `channel`, `paths` (incl. `paths.log_dir`), `inputs`, `branch_map`, `limit.signal_weight`, `limit.gnn_score_cut`.
- The defaults file is a **separate config file that is not meant to be changed**. There is no override mechanism. A frozen key in the user file counts as an unknown key → crash.

Config rules: missing required key → crash. Unknown key → crash (catches typos). Defaults only where marked.

### 5.2 Code layout

New code only. Old job scripts keep working.

```
src/tau3mu_gnn/pipelines/
├── common/          # shared by both pipelines
│   ├── config.py    # value parsers, field tables, YAML reading, shared config sections
│   ├── defaults.py  # the frozen analysis_defaults_<channel>.yaml (read by both pipelines)
│   ├── branches.py  # required branches, input checks, RDataFrame rename
│   ├── manifest.py  # write/read manifest, upstream check (generic)
│   ├── runs.py      # versions <phase>_vsN, upstream choice (generic)
│   ├── driver.py    # version folder + job submission (generic)
│   ├── jobs.py      # job helpers (scratch, read, names)
│   ├── storage.py   # scratch dir + xrdcp to output_dir
│   ├── slurm.py     # job scripts, sbatch, dependencies, proxy check (later: condor.py)
│   └── stages/      # resonance_skim, skim, bdt (inference), limit
├── analysis/        # tau3mu-pipeline
│   ├── config.py, manifest.py, driver.py, cli.py, models.py
│   ├── jobs.py      # job bodies, python -m tau3mu_gnn.pipelines.analysis.jobs
│   └── stages/      # gnn (inference), evaluation
└── training/        # tau3mu-train
    ├── config.py, manifest.py, driver.py, cli.py
    ├── jobs.py      # job bodies, python -m tau3mu_gnn.pipelines.training.jobs
    └── stages/      # npvs, bdt_train, bdt_infer, graphs, gnn_train, benchmark
models/W/                     # frozen model files, committed to the repo (~40 MB)
├── gnn/new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2.pt
├── gnn/Normalization.pkl
├── gnn/arch_config.json
└── bdt/ISO_classifiers_24feb2026_v0.pck
configs/
├── analysis_defaults_W.yaml
└── example_W.yaml
bin/tau3mu-run      # bash: activate env, call cli.py
```

Existing scripts (`skimmer.py`, `BDT_ScoreProducer.py`, `datacard_creation_GNN_permuted.py`, ...) run code at module level. Each is refactored into a **function with explicit arguments**. The old `argparse` entry point stays as a thin wrapper around that function.

### 5.3 Testing

1. **Unit tests (pytest)**: config, branch map, manifest. Tiny synthetic ROOT files, no real data.
2. **Stage regression**: each refactored stage on a small subset (`max_events`) must give the same output as the old script.
3. **Final regression**: full W run with Kaito's config must reproduce the reference limit.

**Reference limit** (from `output/W_limits.json`): model `new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2`, BDT preselection 0.8, GNN cut 3.9 → expected limit **1.2890625** (units as stored by `readout_limit`).

## 6. Mini-step plan

Each step is small, testable, and keeps the old job scripts working.

1. ✅ Agree on the design (this doc). Answer the open questions.
2. ✅ Config: copy frozen model files to `models/W/` (paths in defaults are relative to repo root); `analysis_defaults_W.yaml`, `example_W.yaml`, and `config.py` (load both, merge, validation: missing key / unknown key / frozen key in user file → crash). Unit tests.
3. ✅ Branch map: write the required-branch list, the check + rename helpers, and tests (identity map on Kaito's files; renamed test file; missing branch → crash).
3b. ✅ Stage "resonance skim": add an empty placeholder stage (clear "not implemented" message, or pass-through when the input is already resonance-skimmed). Fill it when Kaito gives the code.
3c. ✅ Stage "resonance skim": the real cuts (from Riccardo's `tau3mu_gnn_pf_skimmer.py` gist, plus mass cut and ω/φ veto). Cuts in the frozen defaults (`resonance_skim`). Regression vs the old skims (old cut values: exact match).
4. ✅ Stage "skim": refactor to take inputs from config. Regression check vs old output.
5. ✅ Stage "BDT inference": make `BDT_ScoreProducer` run **only** the ISO classifier. Remove the GNN classifier and the `GNN_score` input. (Own mini step, before the config refactor of this stage.) **Caution:** the BDT drops rows with NaN in *any loaded* column (`dropna`); loading fewer branches can change which events get `bdt_ISO` — check against the reference limit. Today the W BDT also loads 5 raw branches it does not use as features and that are **not** in `required_branches("W")`: `Slimmed_TauTo3Mu_charge`, `Slimmed_TauTo3Mu_mu{1,2,3}_idx`, `Muon_tkRelIso` (`BDT_ScoreProducer.py:100-127`). Decide on purpose: drop them from the load (then check the limit) or add them to the W list. Add a test that the BDT feature list ⊆ `required_branches("W")`.
5b. ✅ Stage "BDT inference": take inputs from config.
6. ✅ Stage "GNN inference": refactor `ScoreCreation`/`ScoreProducer` into a function; inputs from config. Regression check vs old output. Add a test that the GNN branch list ⊆ `required_branches("W")`.
6b. ✅ Merge scores on `(run, luminosityBlock, event)` instead of `(run, event)` in the BDT (`BDT_ScoreProducer._write_with_scores`) and GNN (`ScoreProducer._write_with_scores`) merges. Kaito's W signal has 26 duplicate `(run, event)` pairs (52 events: run = 1, different lumi, entries 15,959–624,203); `(run, lumi, event)` is unique; data has none. Today those events can get the other event's score (also in the reference limit). Add a check that `(run, lumi, event)` is unique (error, else). Regression: only these 52 signal events may change.
7. ✅ Stage "model eval": refactor the `ModelEval` modes used here (`tau_hist_fpr`, `tau_hist`, `ROC_comp_full`) into functions; inputs from config. Done in PR #17 (both old notes solved: ISO only; the stage passes explicit npz paths, no suffix).
8. ✅ Stage "limit": refactor `datacard_creation_GNN_permuted` + `readout_limit` into functions; inputs from config. Regression check: cut 3.9 → 1.2890625. Done: `pipeline/stages/limit.py` (one cut per call → `cut_<cut>/` + `limit.json`), `make_datacard` / `run_combine` / `read_limit`, old CLIs kept. Regression job 54814 on the stored inputs: 1.2890625. The step 5 open item (limit with the **new** stage 2/3 outputs) stays open → step 11.
8b. ✅ Cut scan: config `gnn_score_cut` takes one value or a list. One Combine job per cut, in parallel. A final job writes a table (cut → expected limit) and marks the best cut. Done: `summarize` (`limits.json` / `limits.txt` / `limits.png` in `phase_c/`), `run_scan` (local loop). Missing cut = `missing` row, best from the rest. SLURM fan-out (one job per cut, summary job with `afterany`) → step 9.
9. Driver, spec `docs/superpowers/specs/2026-10-12-step9-driver-design.md`. 9a ✅ (no SLURM): config `paths.log_dir` + frozen `slurm:`; `storage.py`, `manifest.py`, `runs.py`, `jobs.py`; stage 3 per sample; `summarize(write_dir=)`. Every phase run makes a new version `<phase>_vsN`; single-stage runs in `debug/<stage>_vsN`.
9b. ✅ SLURM layer: job script template, `slurm.py` (`sbatch --parsable`, dependencies, `--dry-run`), proxy check, `cli.py` (phase-a/b/c, stage). Check: small-input run A → B → C on T3 (Kaito queues). Done: slurm.py (job scripts, sbatch --parsable + dependencies, --dry-run, proxy check), driver.py, cli.py (python -m tau3mu_gnn.pipeline.cli). Jobs activate the conda env of the submitting shell (D10). Real check: small-input run A → B → C, all 8 jobs COMPLETED (2026-10-14).
10. ✅ Entry point + README section for the new user. Done: console command `tau3mu-pipeline` (`[project.scripts]`, installed by `pip install -e .`) instead of a bash wrapper; README "Analysis Pipeline (from a config file)".
10b. ✅ `all` command: submits A, B and C at once. Done (`driver.submit_all`): B gets the **new** A version as upstream (not the newest complete one: it has no manifest yet), C the new B version. The first jobs of B wait `afterok` for A's `bdt`, the jobs of C for B's `eval`. Every job with a dependency gets `--kill-on-invalid-dep=yes`. The proxy check uses the longest job chain. No `--after` flag (only `all` needs it).
11. ✅ Combine gap closed: README "CMS Combine" (standalone conda build into Tau3Mu, pinned commit `10cef0bb`). The clean-account dry run moved to step 15.
12. Training pipeline (`tau3mu-train`), spec `docs/superpowers/specs/2026-10-23-step12-training-pipeline-design.md` (supersedes the old 12 / 12a text). Chain: skim → BDT training → BDT inference → (npvs → graphs → GNN training) ∥ (benchmark BDT-only limit, cut scan). Sub-steps:
    - 12a ✅ Training config: `configs/training_defaults_W.yaml` (frozen), `configs/example_training_W.yaml` (user), `pipelines/training/config.py` (`load_training_config`). Shared stages read `analysis_defaults_W.yaml`. Plan `docs/superpowers/plans/2026-10-23-step12a-training-config.md`.
    - 12b ✅ Stage npvs (was 12a): weights for the **signal** graphs only (background graphs weight 1, inference no weights). Histograms of all skimmed signal vs all skimmed background (target shape), old variable binning. `pipelines/training/stages/npvs.py` → `signal_W_npvs_weights.npz` (`weight_per_bin`, `bins`), `signal_W_npvs.png`, `signal_W_npvs_weights.png`; per-event lookup `CorrectPileupDist.signal_event_weights`; `branches.training_required_branches` adds `PV_npvs`. Plan `docs/superpowers/plans/2026-10-23-step12b-npvs.md`.
    - 12c ✅ Stage bdt_train (ISO only, sideband background: all candidates outside ±0.08 GeV). `bdt_training/train_bdt.py` (loading + features from stage 2 code) and `pipelines/training/stages/bdt_train.py` → `ISO_classifiers.pck` (format stage 2 reads), `bdt_train_summary.json`, `logloss_fold_<i>.png`, `roc.png`. Old `Train_XGB_W.py` unchanged, not used by the pipeline. Plan `docs/superpowers/plans/2026-10-23-step12c-bdt-train.md`.
    - 12d ✅ Stage graphs: `pipelines/training/stages/graphs.py`, one call per sample, input = stage 2 output. Order: BDT cut `bdt_ISO >= cut` → sideband (background: candidate 0) → seeded random subset of `max_events` (`graphs.shuffle_seed`) → one graph per event from candidate 0. npvs weights on signal graphs only via `CorrectPileupDist.signal_event_weights` (no IndexError for `PV_npvs` ≥ 100); background weight 1. Output `<out>/<sample>_W/processed/W_decay_<sample>_skimmed_dzd0_cone_size_0.5_knn_10_delete_muons.pt` (layout of `Train_TauMuNet`). `ConeGraphProducer` got opt-in options (defaults = old behaviour). Equivalence test: training graphs = inference graphs (`tests/pipeline/test_graphs.py`). Plan `docs/superpowers/plans/2026-10-23-step12d-graphs.md`. Note: the `pair_charge` edge feature is the charge of node i, not the product (training and inference alike); changing it needs a retraining — Kaito decides later. 12g: use `branches.training_required_branches` in the training stage-0 input check.
    - 12d+ ✅ Stage bdt_infer (`pipelines/training/stages/bdt_infer.py`): stage 2 with the classifier from `bdt_train`; `stages/bdt.score(inputs, out_dir, classifier, channel, tree_name)` is shared with the analysis (`run(cfg, ...)` unchanged). Chain test `tests/pipeline/test_training_chain.py`: bdt_train → bdt_infer → npvs → graphs on toy data.
    - 12e ✅ Stage gnn_train: `pipelines/training/stages/gnn_train.py` + `gnn_training/train_gnn.py` (reuses `Train_TauMuNet.main`, made importable). Kaito, 2026-10-23: early stopping patience 50 (20 was the fine-tune value), the best epoch is always kept, frozen seed `gnn_training.seed`. Output `<out>/<model_name>/` (`<model_name>.pt`, `Normalization.pkl`, `arch_config.json` = what the analysis loads; test: `model_loading.load_model` loads it) + losses / evaluation / plots, `gnn_train_summary.json`. `model_name` = `<training_name>_<channel>_fl_<i>_sd_<j>_<conv>_<unsculpting>`. Comet key from the config (removed from `Train_TauMuNet.py`); `GNN_archs` imports on any machine, no registry write from the pipeline. Plan `docs/superpowers/plans/2026-10-23-step12e-gnn-train.md`. Old CLI side effects: `--finetune` now also keeps the best fine-tune epoch, and its result reports the fine-tune epochs; a non-finite validation loss stops the training at once (`FloatingPointError`).
    - 12f ✅ Stage benchmark: `pipelines/training/stages/benchmark.py`, BDT-only limit cut scan on `bdt_ISO > cut` (cuts `benchmark.bdt_score_cut`), on the `bdt_infer` outputs. No new datacard code: `make_datacard(score_branch=...)` and `stages/limit.run_cut` / `summarize_cuts` (shared with stage 5; analysis wrappers unchanged). The old `analysis/datacard_creation_ISO.py` (a copy of the GNN script with `bdt_ISO`) is now a thin CLI around `make_datacard`. Output as stage 5: `cut_<cut>/`, `limits.{json,txt,png}`. Plan `docs/superpowers/plans/2026-10-23-step12f-benchmark.md`. PR #32 review: `analysis/datacard_creation_GNN_permuted.py` renamed to `analysis/datacard.py` (it serves both score branches).
    - 12g ✅ Training driver: `pipelines/training/{driver,cli,jobs}.py`, command `tau3mu-train {all,bdt,gnn,benchmark} --config FILE [--dry-run]`, `gnn` / `benchmark` take `--from bdt_vsN` (default: highest complete bdt). Phases bdt (`skim` → `bdt_train` → `bdt_infer`), gnn (`npvs` → `graphs_signal`; `graphs_background` does not wait for `npvs`; both → `gnn_train`), benchmark (`benchmark_limit_<cut>` per cut → `benchmark_summary`, afterany); `all` = bdt, then gnn and benchmark wait afterok on `bdt_infer`. Folders `<output_base>/<training_name>/{bdt,gnn,benchmark}_vsN`, logs `<log_dir>/<training_name>/<phase>_vsN`. GPU job `gnn_train`: per-job `account` / `gres` (`#SBATCH --gres`). Deferred: `stage` single-stage debug runs. Note: run `pip install -e .` again to get the new console script `tau3mu-train`.
    - 12h ✅ Analysis key `models:` (`frozen` default, or a training `gnn_vsN` path): `pipelines/analysis/models.py` (`read_trained_models`) reads the gnn manifest and its bdt upstream manifest; `bdt.classifier`, `gnn.model_dir` / `model_name` (from the `model` output path), `conv_type`, `gnn_mode`, `unsculpting_mode`, `graph_type` (via `graph/naming.graph_suffix`), `bdt_preselection_cut` from the training. Config errors for a non-gnn folder, an unfinished training, missing files. `models` recorded in `config_used.yaml` and in the settings of all analysis phases. Plan `docs/superpowers/plans/2026-10-23-step12h-models-key.md`.
    - 12j ✅ BDT training plots (Kaito, 2026-10-24): 12c ported only `logloss_fold_<i>.png` and a plain log ROC. `train_bdt.plot_training` now writes all ISO plots of the old `Train_XGB_W.py`: `roc.png` / `roc_log.png` (working points 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99 marked; PR #37 review, the old script had 0.80–0.89), `overtrain.png` / `overtrain_log.png` (train vs test score, KS p-values), `feat_importance.png`, `corr_sig.png` / `corr_bkg.png` (features + score + fitted mass); plus `roc.pck` (fpr, tpr, threshold, auc; was `_roc_log*.pck`). `train_iso_bdt` also scores the train split (mean fold score, as the old script) and puts `ks_pvalue` in `bdt_train_summary.json`. Working-point fix: the old index `(thr > cut).sum()` took the point one step past the cut; now the last point with `thr > cut`. Plain matplotlib (seaborn is not a dependency). No rerun of 12i: plots for future trainings only.
    - 12k ✅ Analysis phase names (Kaito, 2026-10-24): `phase_a` / `phase_b` / `phase_c` → `bdt` / `gnn` / `limit`, as the training phases. Commands `tau3mu-pipeline {all,bdt,gnn,limit,stage}`, folders `bdt_vsN` / `gnn_vsN` / `limit_vsN`, job names `BDT<N>.` / `GNN<N>.` / `LIM<N>.` (debug: `D<N>.`, now from the `debug/` parent folder, because debug and phase folders share names like `bdt_vs1`). Every manifest gets `pipeline: analysis|training`: the upstream check and the `models:` key refuse a folder of the other pipeline (an analysis `gnn_vsN` is no model folder). A manifest without `pipeline` (written before) passes. Clean break: no alias; old `phase_*_vsN` folders are not found by `--from` (results stay readable).
    - 12l ✅ (Kaito, 2026-10-26, stacked on PR #46) Commands renamed: `tau3mu-pipeline` → `tau3mu-analysis`, `tau3mu-train` → `tau3mu-training` (no alias; run `pip install -e .` again; older entries here keep the old names). Model settings defined once: the frozen GNN settings (conv type, GNN mode, unsculpting mode, graph settings, BDT preselection cut) moved from `analysis_defaults_<channel>.yaml` to `models/W/gnn/<model_name>/model_settings.json` (layout of the training gnn manifest settings); `analysis_defaults` keeps only the model paths; one parser (`pipelines/analysis/models.py`) reads frozen and trained models, so the `INFERENCE_GRAPH` and channel checks also run for the frozen model. The training keeps its own values (`graphs.bdt_preselection_cut`, frozen training settings) as the only source for new models. `slurm.jobs.skim` 128G → 256G in all four defaults files (full raw W training, job 854069: 5 h, MaxRSS 127.99G of 128G).
    - 12i ✅ Real check (Kaito, 2026-10-25): trained on the old skimmed inputs (`kaito_W_train`), analysis with the new models (`kaito_W_retrained`). GNN best limit 1.48 at cut 3.7 (0.1 scan 3.0–5.0, `phase_c_vs2`; 0.25 scan: 1.55 at 3.75; the curve is not smooth, e.g. 2.22 at 4.3 but 1.61 at 4.5), BDT-only benchmark 2.52 at 0.99, old reference 1.29. Kaito: accepted (the GNN still beats the BDT). Stage 1 skim of the background reproduces the old `selected_data_skimmed_no_resonances.root` (all 380 branches equal, job 848645). README sections.
Order (Kaito, 2026-10-24): 13b → 13c → 13 (HF). The labels stay, so older references to "step 13" (HF) and "step 14" (lxplus) stay correct.
13b. Full W analysis from raw NanoAOD with the new skim (Kaito, 2026-10-24). The skim criteria changed compared to the old samples (step 3c: χ² < 20, 1 < m < 3 GeV, ω/φ veto ±40 MeV), so the whole chain runs again on Riccardo's raw NanoAOD: `tau3mu-train all` on the raw inputs (a training config like `example_training_W.yaml` with the inputs and `already_resonance_skimmed: false` of `example_W_raw.yaml`), then `tau3mu-pipeline all` on the raw inputs (`example_W_raw.yaml`) with `models:` = the new `gnn_vsN`. Results: new BDT and GNN, new benchmark limit, new GNN limit; compare with 12i and the old reference. Before: check `limit.signal_weight` for the new skim (open since step 3c). The skim job (stage 0 + 1) may need more resources on the full raw inputs: today `slurm.jobs.skim` = 1 day, 128G, 4 CPUs in both defaults files, from a raw test (~5 s/file, ~6300 files); the memory for the full raw run is not measured. Check `sacct` MaxRSS / elapsed after the first full run and update both defaults files. Status (2026-10-25): `signal_weight` checked by Kaito (0.000019 stays). Training running: `training/W_raw_Nano_AOD` (jobs 854069–854085, skim = 854069); analysis next: `analysis/kaito_W_raw` with `models:` = `training/W_raw_Nano_AOD/gnn_vs1`.
    - ✅ 13b result (2026-10-28): training `training/W_raw_Nano_AOD`, analysis `analysis/W_raw_Nano_AOD/limit_vs2` (config `configs/kaito_W_raw.yaml`, not committed): best GNN limit **1.48 at cut 3.75** (scan 2.0–5.0 step 0.25; it jumps: 1.60 at 3.5, 1.77 at 4.0, about 3 background events at the best cut); BDT-only benchmark **2.58 at 0.99**. The same median as 12i (1.48 at 3.7) only by Combine's 1/64 step: bands and rates differ. First analysis try (`bdt_vs1`) lost to an xrdcp timeout after 7.5 h → PR #49 (retry). `signal_weight` 0.000019 is still not checked for the new skim.
13c. After 13b: the physics fixes, each with a new limit compared with 13b (Kaito, 2026-10-24): E1 (BDT `TauPlusMET` candidate mix), E3 (datacard slope range), E4 (datacard background rate for the larger mass range), and the GNN edge feature `pair_charge` (today the charge of node i, not the product of the two charges; same in training and inference, `graph/ConeGraphProducer.py` and `gnn_training/ScoreProducer.py`). E1 and the `pair_charge` fix change the model inputs, so the BDT / GNN are retrained with `tau3mu-train` (as in 13b). Decisions (Kaito, 2026-10-25):
    - E1 and `pair_charge`: clean break, no input-version flag. Only the fixed inputs remain; models trained before (the frozen `models/W/`, `kaito_W_train`, `W_raw_Nano_AOD`) no longer match the inference inputs, and the 1.29 reference is no longer reproducible with the current code. Follow-up: replace the frozen `models/W/` with the 13c retrained models.
    - E1 ✅ (code): study (2026-10-25): raw NanoAOD has `nTauPlusMET == nTauTo3Mu` in every event (48,574 checked) and the charges match per candidate. The length mismatch comes only from the old resonance skim: it slimmed `TauTo3Mu_*` but not `TauPlusMET_*` (in 100% of the mismatch events `len(TauPlusMET) == nTauTo3Mu`). Stage 0 masks `TauPlusMET_*` with the candidates, so new runs have no mismatch (small raw test: 0 events). Left in the new pipeline: the BDT took `TauPlusMET[0]`, the GNN already `TauPlusMET[tau3mu_mask][0]`; they differ when the first Slimmed candidate fails stage 1 (small raw test: signal 1 of 30,797, data 16 of 7,726 = 0.21%).
      Fix: the W BDT loads `tau3mu_mask` and `_load` masks `TauPlusMET_*` before the first candidate (mask cast to bool: stage 1 writes int32). Small raw test: exactly the 16 events change, in the 5 per-candidate columns (`Puppi_pt`, `Puppi_phi`, `Tau_Puppi_mT`, `PuppiMETminPz`, `PuppiMETmaxPz`); all other inputs equal.
      Kaito (option A): stage 0 refuses resonance-skimmed inputs with any event `len(TauPlusMET_*) != len(Slimmed_TauTo3Mu_*)` (`check_tauplusmet_aligned`, BranchError: "skim from raw NanoAOD"). Old W signal skim: 5,055 of 801,732 events → refused. `example_training_W.yaml` now uses the raw inputs; `example_W.yaml` / `example_W_retrained.yaml` no longer run (kept as records); README step 2 copies `example_W_raw.yaml`.
    - E3 ✅: slope range ±10σ around the fitted slope (the intent of the removed `safe_range()`), `SLOPE_RANGE_SIGMA` in `analysis/datacard.py`. Limit unchanged: toy 0.0806 / 0.1133 (cuts 2.0 / 3.9); real W card (`kaito_W_retrained/phase_c_vs2/cut_3.7`, slope 0.89 ± 2.48) 1.484375 with the old and the new range (Combine resets the start value and range from the `slope param` line, see E3 below).
    - E4 ✅ (code): one mass range, 1.6–2.0, for the fit datasets and the workspace: `limit.fit_range: [1.6, 2.0]` in `analysis_defaults_W.yaml`, the RooRealVar `mass` takes `fit_range` (`MASS_VAR_RANGE` removed). The sideband fit now also uses 1.6–2.0 (was 1.62–1.98). Finding: Combine normalizes the background over the mass range of `data_obs`, and the old `data_obs` kept 1.6–2.0; so the old cards already used `nbkg` over 1.6–2.0 and the expected +12% background did not happen. Toy: 0.1133 old and new (cut 3.9); old card with `data_obs` cut to 1.62–1.98: 0.1216 (+7% ≈ √1.12, the predicted effect). Real W check (`kaito_W_retrained` stage 3, job 854105): limit 1.484375 → 1.4296875 (cut 3.7) and 1.5859375 → 1.53125 (cut 3.9), both −3.5%. Signal rate unchanged; only the wider sideband fit moves: `nbkg` 18.09 → 16.79 / 13.15 → 12.47, slope 0.89 ± 2.48 → 1.66 ± 2.22 / 1.20 ± 2.93 → 2.21 ± 2.62 (≈ 0.3σ). The sideband fit has only 12 / 9 events at these cuts, so the limit curve is noisy.
    - Order: E3 → E4 → E1 study → E1 → `pair_charge`, one PR each.
    - 13c run (Kaito, 2026-10-25): after the 13b limit, merge #41–#45, then one retrain (`tau3mu-train all` on the raw inputs) + analysis with all 4 fixes, one combined limit compared with 13b (no separate E1 / `pair_charge` trainings). E3/E4 alone: E4 real check above (−3.5%).
    - `pair_charge` ✅ (code): edge feature 1 is now q_i · q_j (`matrix_charge * matrix_charge.T`; row i of `matrix_charge` is q_i, so it was q_i) in `graph/ConeGraphProducer.py` (training graphs), `gnn_training/ScoreProducer.py` (inference) and the unused `graph/ConeGraphProducer_Dataset.py`. Test: every edge of the training and the inference graphs has `edge_attr[:, 1] == q_i * q_j`. Clean break: the `models:` check compares graph settings, not code, so a GNN trained before this fix (frozen `models/W/`, `kaito_W_train`, `W_raw_Nano_AOD`) is still accepted but gets the new edge feature; retrain (13c run).
13d. Training / analysis event split (Kaito, 2026-10-26). Today the analysis scores all events, also the ones the models were trained on. Raw W run (training `W_raw_Nano_AOD`): the BDT used 100k of 721,526 signal events (14%) and 100k of 4,288,230 sideband background events (2.3%); the GNN used 600k of 643,216 signal events after the BDT preselection (93%) and **all** 71,401 sideband background events after the preselection (100%; 60% train, 10% validation, 30% test). The limit fits exactly these sideband events after the GNN cut, so the background after the cut can be too low and the expected limit too good; the signal efficiency is also measured on training events. The old W reference (1.29) has the same bias. Idea: a flag that splits by event number, e.g. the training uses only odd `event`, the analysis only even `event` (then the analysis signal weight / luminosity changes by the kept fraction). Decide the scheme (parity, k-fold with model per fold, or a fixed fraction) before the next reference limit.
13e. Follow-ups after 13c (was 13d; renumbered 2026-10-28, 13d is the event split) (Kaito, 2026-10-25), small, one PR when 13c is merged:
    - ✅ Skim resources (done in 12l): full raw W training skim 854069: 5 h, MaxRSS 127.99G of 128G → `slurm.jobs.skim` 256G in all four defaults files.
    - Frozen models: replace `models/W/` (BDT `.pck`, GNN `.pt`, `Normalization.pkl`, `arch_config.json`) with the models of the 13c retrain, and their `model_settings.json` (12l: the model settings live with the model); the old frozen models no longer match the inputs (E1, `pair_charge`). New reference limit = the 13c limit (replaces 1.29 in sec. 8 and the README).
    - Old example configs: `example_W.yaml` and `example_W_retrained.yaml` no longer run (E1). Kaito decides: delete them, or keep them as records.
    - Unused code: `gnn_training/Parallel_ScoreProducer.py` (old `(run, event)` index, see E2) and `graph/ConeGraphProducer_Dataset.py` + `graph/graphDataLoader_chunk_sampler.py` (nothing imports them): delete, after a last grep.
13. Extend to the HF channel (was 12), spec `docs/superpowers/specs/2026-10-25-step13-hf-design.md`, plan `docs/superpowers/plans/2026-10-25-step13-hf.md`. Kaito, 2026-10-25: no trusted old HF GNN limit (the BDT preselection approach was never run for HF), so step 13 trains new HF models; the only reference is the old BDT benchmark 1.0703 at cut 0.91. Starts now, in parallel with 13b.
    - 13-HF-a (code done): `channel: HF` in both pipelines (`SUPPORTED_CHANNELS`, `branches.PER_CHANNEL["HF"]`), `analysis_defaults_HF.yaml` / `training_defaults_HF.yaml` (W values; placeholder frozen model paths, **no frozen HF models yet**, `models:` needed), `example_HF.yaml` / `example_training_HF.yaml` (old HF skimmed files, `signal_weight` 0.000152, scans 0.88–0.97 and 2.0–8.0). Fix: the HF BDT had χ²/ndof twice (12 features); now the 11 features of the old HF classifier, in its order; `TauPlusMET_Puppi_phi` no longer loaded for HF. `models:` refuses a training of another channel. Multi-file inputs already worked. Real run (Kaito queues): HF benchmark ≈ 1.07, first HF GNN limit.
    - 13-HF-a training done (2026-10-27): `training/HF_training` (BDT test AUC 0.92, GNN test AUC 0.77). Update 2026-10-27: the old benchmark 1.07 is biased. The old HF background score job (`jobs/gnn_score_creation/create_score_HF_bkg.sh`) had `MAX_EVENTS_TO_PROCESS=170000` per file, and the old `bdt_ISO` exists only for GNN-scored events: the old benchmark used a random ~46% of the background (129k of 362k events per part) but all signal (96%) and the full signal weight. At cut 0.91: old signal rate 25.20 / background 1071, new 25.20 / 2240; 1.07 / sqrt(0.46) ≈ 1.58. The new pipeline (all events) gives **1.53 at cut 0.90–0.91** (`training/HF_training/benchmark_vs1`); the fit range (1.6–2.0 vs 1.62–1.98) changes it by < 1% (job 114656). 1.53 is the HF BDT reference. The old HF GNN limits in `output/HF_limits.json` use the same capped files. GNN cut scan of `example_HF.yaml` changed to 0.0–3.0 (logit; logit 2 keeps 6% of the preselected signal, logit 3 only 0.7%).
    - 13-HF-b (on hold, Kaito): raw NanoAOD. The old HF background is not a cut selection but a part of the old W data (streams 0 and 4: 75%, 5: 17%, 7: 1%, others 0%), so the origin of `0.000152` must be cleared first; then the HF signal weight for the full raw data and the new skim. Signal sample: Summer24 `DstoTau-Tauto3Mu_Fil-3Mu_..._05may25v2` (4432 files, has all HF branches).
    - Later: frozen HF models in `models/HF/` from a chosen training.
14. (Was 13) Port to lxplus: HTCondor scheduler layer, EOS paths, Combine on lxplus. Update the README parts listed in the "move to lxplus" note (Infrastructure). Test.
15. (Very last, was the dry run of 11) Dry run on a clean account / clean checkout, on lxplus. A new user follows only the README.

**Maybe, at the very end** (Kaito decides then; not needed for the limit). E1, E3 and E4 are scheduled in step 13c; E2 stays optional:

- **E1. BDT: `TauPlusMET` candidate mix.** The BDT takes `[0]` of `Selected_TauTo3Mu_*` (candidates that pass the skim) and `[0]` of `TauPlusMET_*` (all candidates, before the skim). If the first candidate fails the skim and a later one passes, the tau features and the tau+MET features come from different candidates. Affects 5 of the 6 W BDT `TauPlusMET` branches: `Puppi_pt`, `Puppi_phi` (in `deltaPhi`), `Tau_Puppi_mT`, `PuppiMETminPz`, `PuppiMETmaxPz` (per candidate). `PuppiMET_pt` is global (same value in every entry), not affected. Measured (W): signal 75 events (0.01%), data 2,214 of the first 1M events (0.29%). Also, `len(TauPlusMET_*) == nSlimmedTauTo3Mu` in only 99.4% (signal) / 97.6% (data) of events. A fix changes the BDT inputs → maybe retrain the classifier. Same root cause as the `ScoreProducer.py` issue in the decision log (2026-10-07).
- **E2. BDT + GNN: faster score write.** The key join is done in step 6b (`utils/score_merge.py`: pandas join on `(run, luminosityBlock, event)`, no `iterrows`, no `BuildIndex`, no temp file). Left: `write_with_score` still copies every entry with a PyROOT `GetEntry`/`Fill` loop. This is slow on data (10.8M entries). Replace with a C++ / RDataFrame copy that keeps the output format (same branches and types). `gnn_training/Parallel_ScoreProducer.py` still has the old `(run, event)` index, but no script imports it: fix it or delete it. Regression: outputs identical.
- **E3. Datacard: slope range.** `make_datacard` (`analysis/datacard.py`, was `datacard_creation_GNN_permuted.py`) writes `Exponential::bkg(mass, slope[val, err, err])`: value = fitted slope, range = `[err, err]` (zero width, value outside it). The removed, unused `safe_range()` (±10σ) was probably meant. Kept in step 8 (pure refactor). Effect (2026-10-12, toy card from `test_run_scan_end_to_end`, cut 3.9): RooFit clips the value into the range, so the workspace has slope = +err (+0.128, a rising exponential). But the datacard line `slope param <val> <err>` makes Combine reset the start value and the range: FitDiagnostics fits slope = −2.835 ± 0.090; with the range fixed by hand to ±10σ the expected limit is unchanged (r < 0.1133 both). So probably harmless for the limit. Not checked on the real W card. Fix anyway (it is wrong in the workspace); then check the limit stays the same.
- **E4. Datacard: background rate for a larger mass range.** (Fixed in 13c; the limit effect below did not happen, see 13c: Combine used the `data_obs` range 1.6–2.0.) The sideband fit runs on the dataset with the `mass` range `MASS_VAR_RANGE` = 1.6–2.0, so `nbkg` = background yield in **1.6–2.0** (toy: 4062 sideband events / sideband fraction 0.589 = 6896 = `nbkg`). The workspace `mass` has the range `fit_range` = **1.62–1.98**, and Combine spreads `rate` = `nbkg` over that range. So the expected background is too high by the factor (yield 1.6–2.0) / (yield 1.62–1.98): toy 1/0.892 ≈ +12%; real size depends on the fitted slope. `--run blind` builds the Asimov data from the pre-fit values (rateParam = 1), so the expected limit is too high (conservative). Fix: write `nbkg` × fraction of the fitted exponential in `fit_range`, or use the same range for the fit dataset and the workspace. A fix changes the limit → new reference limit. Small related points: `data_obs` keeps events in 1.6–1.62 and 1.98–2.0 (outside the workspace range; no effect on the blind limit); signal events outside 1.6–2.0 are not in the signal rate (narrow Gaussian, tiny effect).

## 7. Open questions

Tracked here. Answers move to section 8.

- (none open right now)

## 7b. Waiting on Kaito

- (none open right now)
## 8. Decision log

| Date | Decision |
|---|---|
| 2026-10-07 | Roadmap created. GNN model fixed. Config file + branch map. Mini steps. |
| 2026-10-07 | Q1: Order is Skim → BDT (ISO only) → GNN (BDT ISO cut) → Eval → Limit. The BDT with `GNN_score` input is dropped. `BDT_ScoreProducer` must be changed to use only `classifiers_iso`. |
| 2026-10-07 | Q2: Build and test for W only. Config has a `channel` field. Code must not block HF. HF support is the last step. |
| 2026-10-07 | Q3: Start on PSI T3. The wrapper submits one SLURM job per stage, chained with dependencies. Port to lxplus/HTCondor is the very last step, so the scheduler stays a separate thin layer. |
| 2026-10-07 | Q4: The user starts from raw NanoAOD. A resonance-skim stage comes first. For now it is an empty placeholder. Kaito will provide the code later. |
| 2026-10-07 | Q5: BDT is fixed (ISO pickle only). No retraining. |
| 2026-10-07 | Pipeline split in 3 phases run by hand: A = stages 0–2, B = 3–4, C = 5. One shared config. Phases hand off through a fixed folder + manifest. SLURM chaining (Q3) applies only inside a phase. Supersedes "one command" in the original goal. |
| 2026-10-07 | Add an optional `all` command. It chains phases with SLURM job dependencies (`--after <JOBID>`), **not** `--wait`. Input checks of a phase run inside its first job. |
| 2026-10-07 | Q6: GNN score cut in config is one value or a list. A list gives a parallel cut scan + a summary table with the best cut. Default: 3.9. |
| 2026-10-07 | Q7: Keep **all** outputs (limits, datacards, Combine output, fit plots, sculpting plots, ROC vs BDT) under one `<output_dir>`, layout in sec. 5. |
| 2026-10-07 | Q8: Jobs run in `/scratch`, then `xrdcp` results to `<output_dir>` from the config. One shared copy helper. |
| 2026-10-07 | Branch map: rename once at the first stage of phase A via a shared helper. Map works on raw names. Missing required branch → crash with full list. `Slimmed_` → `Selected_` stays in the skimmer. |
| 2026-10-07 | `limit.signal_weight` is a **required** config value with no default. The user computes it from his own dataset (Kaito's W value 0.000019 is for Kaito's dataset only). The README must say this clearly. |
| 2026-10-07 | Config split: frozen `analysis_defaults_<channel>.yaml` in repo + short user file with tunable/required values. Frozen values change only via `overrides:` with a warning. |
| 2026-10-07 | Reference for final regression: cut 3.9, BDT preselection 0.8 → limit 1.2890625 (`output/W_limits.json`). |
| 2026-10-07 | Frozen defaults = separate config file, not meant to be changed. No `overrides:` section. Frozen key in user file → crash. (Supersedes the override part of the config-split entry.) |
| 2026-10-07 | Frozen model files (GNN `.pt`, `Normalization.pkl`, `arch_config.json`, BDT ISO `.pck`; ~40 MB) are committed to the repo under `models/W/`. No LFS. |
| 2026-10-07 | Config validation: plain Python (dataclasses + own checks). No new dependency. |
| 2026-10-07 | Step 2 done (`src/tau3mu_gnn/pipeline/config.py`, `configs/`, `models/W/`, `tests/pipeline/`). Keys added later by their own steps: `slurm.*` (step 9), `paths.conda_env` (step 10). `gnn.arch` dropped: `ScoreCreation` only uses `--archs` in a length check. `tests/` was gitignored; now only `tests/pipeline/` is tracked. Run tests with `PYTHONPATH=<repo>/src python -m pytest`. |
| 2026-10-07 | PR #7 review: `channel` lives only in the user file (it selects the defaults file); `channel` in the defaults file → unknown key. User sets `paths.output_base`; run folder is derived as `<output_base>/<analysis_name>` (`Config.output_dir`). Run versioning `_vsN` planned for step 9. |
| 2026-10-07 | Pre-skim inputs (resonance-skimmed, before stage 1) live in `/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/<sample>/old_data/` (W: `signal_skimmed_no_resonances.root`, 801,732 events; `data_skimmed_no_resonances.root`, 10,855,317 events; HF: `ds_large_signal_...`, `data_..._part1-4`). Checked: `Slimmed_TauTo3Mu_*` present, no `Selected_*`. Used in `configs/example_W.yaml`. Raw NanoAOD (before stage 0) still unknown; not needed while `already_resonance_skimmed: true`. |
| 2026-10-07 | Step 3 done: `pipeline/branches.py` (`required_branches`, `check_input_branches`, `rename_columns`). 38 required W branches, all present in Kaito's W old_data files. HF list added in step 12. PF candidates are `Slimmed_PFCands_*` (not `PFCands_*`). |
| 2026-10-07 | Flagged to Kaito (outside wrapper scope): `TauPlusMET_*` has one entry per **unskimmed** candidate, but `ScoreProducer.py:519/523` indexes it with the **skimmed** candidate mask; lengths match `nSlimmedTauTo3Mu` in only 99.36% of signal events. |
| 2026-10-07 | On the T3 login node `gcc/12.2.0` is not available; importing ROOT prints a header error but RDataFrame works. Tests import ROOT only inside the tests that need it. |
| 2026-10-07 | Structure check `check_input_structure`: within each family (`Slimmed_TauTo3Mu_`, `Slimmed_PFCands_`, `TauPlusMET_`) all required branches must be per-event lists of equal length (first 1000 events per file). No cross-family check. Raw count branches (`nSlimmedTauTo3Mu`, ...) are not required: the code reads only `nSelectedTauTo3Mu`, which the skimmer defines. Kaito's W files pass. |
| 2026-10-08 | PR 8 review: `check_input_structure` removed (YAGNI). It did not compare families, so it missed the only real length problem (`TauPlusMET_*`). Note: awkward 2.9 does **not** raise when a boolean mask is shorter than the array, so `ScoreProducer.py:522` silently picks a wrong `TauPlusMET_*` entry in those events. |
| 2026-10-08 | PR 8 review: if a file has both a mapped user branch and its analysis name, `check_input_branches` fails (probably not intended). `rename_columns` uses `Define` only. `patterns` renamed to `inputs`. |
| 2026-10-08 | Step 3b done: `pipeline/stages/resonance_skim.py` `run(cfg, out_dir)` → input files of stage 1 per sample. Order (PR 9 review): check inputs (`check_input_branches`, both samples) → rename (branch map) → resonance skim → write, in one RDataFrame pass, so the resonance skim works on analysis names. Resonance-skimmed inputs + empty (or identity) map → inputs passed on, no copy. Non-empty map → renamed copy `out_dir/{signal,background}.root` (input files merged). `already_resonance_skimmed: false` → `_resonance_skim` placeholder raises `StageNotImplementedError` (after the check, before any output). The branch map is applied here (not in stage 1), so all later stages see analysis names only. |
| 2026-10-08 | Open for the real resonance skim: the check uses the 38 required branches (names after the resonance skim). Raw inputs may need more branches (e.g. `Muon_*`); add them to the required list then, so they are checked and can be renamed. `_resonance_skim` must also extend the Snapshot column list if it defines new columns. |
| 2026-10-08 | Step 4 done: `utils/skimmer.py` → `skim(files, output, tree_name, dz_max, cos_alpha_2d_min)`; old CLI kept (`--input`, `--output` required; optional `--config` gives cuts + tree name, else cuts 0.5 and tree `Events`; inputs never come from the config, because the config has two samples and the branch map lives in stage 0). Stage `pipeline/stages/skim.py` `run(cfg, inputs, out_dir)`: `inputs` = stage 0 result (sample → files, analysis names) → `out_dir/{signal,background}.root`. Output keeps all input branches + `tau3mu_mask`, `Selected_TauTo3Mu*`, `nSelectedTauTo3Mu` (as before); no event removed; event order not kept (implicit MT). PR 10 review: no `max_events` / `n_threads` arguments. Regression: first 2000 W signal events (copied to a small file in the test) match the old output on all `Selected_*`, `nSelectedTauTo3Mu`, `tau3mu_mask` (matched by run/lumi/event). |
| 2026-10-08 | Wiring of stage 0 (decided): stages 0 and 1 run in the **same SLURM job**. The driver (step 9) calls stage 0 and passes its returned file lists to stage 1 in memory. Reason: stage 0's output location differs (pass-through → original input paths, possibly several files; copy → `0_resonance_skim/{signal,background}.root`). No path guessing, no file list on disk. Cost: stage 1 cannot be rerun alone without stage 0 (pass-through is instant; with a branch map, stage 0 redoes the copy). Stages 2-5 use fixed paths from `1_skim/` on. When stage 0 gets the real resonance skim: it writes the skimmed files with the branch map applied in the same pass. |
| 2026-10-09 | Step 5 done (PR #11): `BDT_ScoreProducer` runs only the ISO classifier; `--classifier_gnn` and `bdt_GNN` removed; `GNN_score` in the input is optional. W no longer loads `TauTo3Mu_charge` / `Muon_tkRelIso` / `mu*_idx` (HF only). |
| 2026-10-09 | Bug fixed in step 5: the background dataframe kept one row per candidate, so the `(run, event)` index that writes `bdt_ISO` back had duplicate keys and multi-candidate events got an arbitrary candidate's score. Decision (Kaito): always use the **first** candidate (as signal and training already do). No retraining needed: `Train_XGB_W/HF.py` already use the first background candidate. |
| 2026-10-09 | Step 5 regression (W, job 946596) vs the 2026-05-26 run: signal identical. Data: 64,905 of 114,629 multi-candidate events changed (first candidate now); 4 single-candidate events old `-1` → scored (they have `GNN_score = NaN`, the old `dropna` removed them). Pickle checked: one row per scored event, `tau_mass` = first candidate. **Open:** limit check vs the reference limit (1.2890625 at cut 3.9) — do it with step 8 or an old-datacard job. |
| 2026-10-09 | BDT branch lists moved to `bdt_training/bdt_inputs.py` (`bdt_branches(channel, prefix)`, lists unchanged) so tests can import them. Test: every W BDT branch (after the skim rename `Selected_` → `Slimmed_`) is in `required_branches("W")`. |
| 2026-10-10 | Step 5b done (PR #13): `score_events(signal, background, classifier, tau_channel, pickle_path, tree_name)` + `pipeline/stages/bdt.py`. Regression (W, job 952507) vs old script at 59d7296: pickle equal; signal all 415 branches equal; data `run`/`luminosityBlock`/`event`/`bdt_ISO` equal. PASS. |
| 2026-10-10 | Step 6 interface: `score_events(inputs, models, outputs, npz_paths, tau_channel, unsculpting_mode, bdt_preselection_cut, tree_name, device, max_events_to_process)` in `gnn_training/ScoreProducer.py`; `models` = name → `(model, normalizer)`, `outputs` = name → {input: output}. Several models in one pass (each graph built once). `load_model(...)` in `gnn_training/model_loading.py`. Stage `pipeline/stages/gnn.py` → `3_gnn/{sample}_with_gnn_scores.root`, `raw_scores_and_tau_mass_{sample}.npz`. Old CLI `scripts/ScoreCreation.py` is a thin wrapper (same paths). |
| 2026-10-10 | Step 6: GNN branch list in `gnn_training/gnn_inputs.py`. Test: GNN branches ⊆ `required_branches("W")` (except the stage-made `nSelectedTauTo3Mu`, `bdt_ISO`). `PV_npvs` and `iso_*` are loaded only when used (`npvs_rescale` weights, benchmark score). Stage 3 fails if an input already has `GNN_score`. Input check shared by stages 2 and 3 (`branches.check_stage_inputs`). |
| 2026-10-10 | Kaito: the GNN scores the **first** candidate (was the highest-pt one), as the BDT (`bdt_ISO`) and the limit (`fitted_mass[0]`) do. Before, cut, score and fitted mass could come from different candidates. Training built graphs for every candidate. |
| 2026-10-10 | Kaito: `PV_npvs` is not required for inference. The npvs weights are for training only → step 11b. |
| 2026-10-10 | Step 6 review: BDT and GNN merges match on `(run, event)`; Kaito's W signal has 26 duplicate pairs (52 events, unique with lumi). New step 6b (merge on `(run, lumi, event)`). None of them is in the step 6 regression subset (first 5,000 signal events). |
| 2026-10-10 | Step 6 done (PR #15). Regression job 993393: old CLI (c6db332) vs new stage 3, first 5,000 signal / 400,000 background W events of the old `bdt_full` files: all other branches equal; `GNN_score` and npz identical on all events (signal 4,217 scored, background 4,622). 91 background events (0.02%) have a first candidate that is not the highest-pt one; none passes the BDT cut, so the first-candidate change is not seen on real scored events (unit tests cover it). PASS. |
| 2026-10-10 | Step 6 vs stored production files (`.../<sample>_W/new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2/*_bdt_full_with_scores.root`, signal 2026-06-07, background 2026-08-24): new stage 3 on the first 2,000 signal / 50,000 background events gives the same scored events (1,695 / 600) and the same `bdt_ISO`; 10% of the scores differ by 1-4 float32 steps (max 4.8e-07, environment: torch/CPU/threads). Old and new code in the same environment are bit-identical (job 993393). |
| 2026-10-10 | Step 5b review follow-up (shared signal/background loader) regression job 990620: PASS (pickle incl. `ISO_score_sig` index equal). |
| 2026-10-12 | Step 6b (PR #16): shared `utils/score_merge.py`. BDT and GNN write scores matched on `(run, luminosityBlock, event)` (pandas left join; the ROOT copy loop stays; unscored `-1`, NaN stays NaN). Stages 2 and 3 fail early (`DuplicateEventError`) if two events of an input file have the same key. Local smoke regression (first 200 signal + all 52 duplicate-group entries, 2,000 background): only 26 signal entries change (one per pair), for BDT and GNN; new scores equal the pickle / npz. The old code also gave 3 unscored signal events the score of their pair. Full regression job 22137 (first 5,000 signal + all 52 duplicate-group entries, 200,000 background; BDT and GNN stages): PASS, only 26 signal entries change per stage, new scores equal the pickle / npz. |
| 2026-10-12 | Step 7 (PR #17): `ModelEval` split into `load_gnn_npz` / `plot_tau_hist` / `plot_roc_full` / `plot_sic` (explicit files). The GNN-input BDT curve is gone (it was computed, never drawn). Stage 4 `pipeline/stages/evaluation.py`: inputs = stage 3 npz + stage 2 pickle; writes `tau_hist_fpr`, `tau_hist_fpr_normalized`, `tau_hist`, `roc`, `sic` .png. ISO curve: scores `> gnn.bdt_preselection_cut` (`>` as before; stage 3 uses `>=`, a difference only for scores exactly at the cut). Not refactored: `tau_hist_after*` (fail on new pickles: no `GNN_score_*`), `ROC_comp`, `score_cut`. Local smoke regression (50k-entry subsets): old vs new CLI, all 5 PNGs byte-identical. Full regression job 22138 (stored W production inputs: GNN npz, 8,450,182-row BDT pickle): PASS, all 5 PNGs byte-identical; stage 4 writes its 5 plots. |
| 2026-10-12 | Step 8 decisions (Kaito): (1) slope range bug kept, logged as E3. (2) Regression = same stored inputs, old CLI vs new stage; the full-chain limit check (new stages 1–5) moves to step 11. (3) Temp trees (`temp_{mc,data}_selected.root`) keep all columns and stay in `cut_<cut>/`. (4) Fixed fit values are code constants in `datacard_creation_GNN_permuted.py`; only new behavior: a warning if `nbkg` reaches its upper limit 550000. |
| 2026-10-12 | Step 8: `make_datacard` (explicit files, output folder), `analysis/combine.py` (`run_combine`: the two old `combine` commands; finds `combine` next to the env python), `read_limit`. Old CLIs kept as `main()`. Stage 5 `pipeline/stages/limit.py`: one cut per call (must be in `limit.gnn_score_cut`), inputs = stage 3 ROOT files, writes `<out_dir>/cut_<cut>/` with datacard, Combine output + logs, `limit.json`. Fails before work if `combine` is missing; raises if Combine writes no median limit. Local toy regression: datacard byte-identical, limit equal. |
| 2026-10-12 | Step 8 regression job 54814 (14 min, MaxRSS 73 GB): old CLI (c4bccac) + old `combine` commands vs new stage 5, stored W inputs (`<model>/selected_*_with_scores_bdt_full_with_scores.root`), cut 3.9: PASS. `datacard.txt` byte-identical (also to the 2026-08-24 reference), limit 1.2890625 old = new = reference, temp trees and `datacard.root` workspace identical. Note: these are the old (pre step 5) score files, so the step 5 open item is not closed by this job. Note: the workspace `data_obs` has 2 entries in the fit range (44 data events pass the cut in total); the background fit runs on 2 events. |
| 2026-10-12 | Step 8b decisions (Kaito): (1) missing cut → row marked, best from the rest; raise only if no cut has a limit (summary job uses `afterany` in step 9). (2) Outputs `limits.json` + `limits.txt` + `limits.png`. (3) Regression: tests only (no physics change). Also: duplicate cuts rejected in the config; stage 5 removes an old `limit.json` before a rerun; NaN limit = missing; tie → smaller cut. |
| 2026-10-12 | Step 9 design (Kaito), spec `docs/superpowers/specs/2026-10-12-step9-driver-design.md`: D1 split 9a/9b; D2 dCache read via /pnfs, write via xrdcp; D3 no local phase runs; D4 every phase run → new `<phase>_vsN` (supersedes run folder `_vsN` and 'cut scans in the same run'); D5 single-stage runs in `debug/`; D6 `paths.log_dir` + log copy next to results; D7 slurm resources in the frozen defaults file (supersedes `slurm` in the user file); D8 job scripts + `sbatch --parsable`; D9 9b check = small-input run. |
| 2026-10-12 | Step 9a done: config `paths.log_dir` + frozen `slurm:` (account + per-job partition/mem/time/cpus) in `configs/analysis_defaults_W.yaml`. `pipeline/storage.py` (root:// <-> /pnfs, xrdcp/xrdfs or local copy). `manifest.py` (`config_dict`, `phase_settings`, write/read manifest, `check_upstream`: missing manifest or files → `UpstreamError`, changed settings → warning). `runs.py` (versions, `next_version`, `create_version` + `config_used.yaml`, `pick_upstream`: highest complete version or `--from`). `jobs.py` (skim, bdt, gnn per sample, evaluation, limit per cut, summary; `python -m tau3mu_gnn.pipeline.jobs`). Stage 3 takes `samples` and has `output_files`; `summarize(write_dir=)`. Tests: `tests/pipeline` 234 passed, incl. a real Combine run as phase C jobs on toy files. |
| 2026-10-12 | Step 9b decisions (Kaito): D10 the job script activates the conda env of the submitting shell (CONDA_PREFIX, CONDA_EXE; error at submit if none). D11 small-input check: 20k signal / 1M background, cuts [2.0, 3.9]. Also: --dry-run makes no version folder; debug 'stage summary' may use an incomplete phase C version. |
| 2026-10-14 | Step 9b check (small inputs, `small_W_9b`): jobs skim 99382, bdt 99383, gnn_signal 193445, gnn_background 193446, eval 193447, limit_2.0 199532, limit_3.9 199533, summary 199534, all COMPLETED 0:0. `phase_a_vs1`, `phase_b_vs1`, `phase_c_vs1` with manifests and logs; `limits.json`: cut 2.0 → 27.375 (best), 3.9 → 49.0. The high limits are expected: `signal_weight` 0.000019 is for the full 801,732 signal events, but the run uses 20,000 (1/40); signal rate at cut 3.9 is 0.0400 vs 1.6468 in the reference (×41), and 1.289 × 41 ≈ 53. Background fit on very few events (rate 1.60 vs 3.06). MaxRSS: skim 15.5 GB, bdt 11.2 GB, gnn_background 23.7 GB (1M events), limit 8 GB. |
| 2026-10-14 | Final regression (roadmap 5.3 item 3), full W run A → B → C with the wrapper (`final_W_reference`, cut 3.9): limit **1.2890625** = reference; `datacard.txt` byte-identical to step 8 job 54814. So the step 5 and 6b changes do not change the limit; the step 5 open item is closed. Jobs (all COMPLETED 0:0, MaxRSS / elapsed): skim 202615 90 GB / 0:36, bdt 202616 112 GB / 1:36, gnn_signal 205116 19 GB / 7:05, gnn_background 205117 64 GB (= the 64G limit) / 3:33, eval 205118 2 GB, limit_3.9 243224 73 GB / 0:06, summary 243225 1 GB. Frozen slurm memory raised: bdt 64G → 160G, gnn_background 64G → 128G. |
| 2026-10-15 | PR #20/#21 review (Kaito): limit job variable `card` → `cut_dir`. bdt job memory 160G → 128G (full W peak 107 GiB; power of 2). New optional user key `paths.conda_env` (amends D10): the jobs activate this env; not set → the active env of the submitting shell, as before. |
| 2026-10-15 | Steps 10 + 10b (Kaito): entry point = console command `tau3mu-pipeline` (not the bash wrapper of the step 9 spec; `paths.conda_env` made the env activation unnecessary). `all` without a user `--after` flag. Two gaps of the 10b plan fixed: (1) phase B/C of `all` take the new upstream folder directly, the runtime `check_upstream` does the check; (2) T3 has no `kill_invalid_depend` (`DependencyParameters = (null)`), so jobs after a failed job stay PENDING forever → `sbatch --kill-on-invalid-dep=yes` for every dependent job (also inside one phase). The proxy check uses the longest job chain (sum over the phases for `all`), not the longest job. Open for step 11: CMS Combine is not in `environment_frozen.yaml` (built by hand into the env). |
| 2026-10-21 | Step 11, Combine gap (Kaito): Combine was built by hand from a clone (`HiggsAnalysis/CombinedLimit`, gitignored) at `10cef0bb` (v10.5.1 + 6 commits), CMake into the env (`-DCMAKE_INSTALL_PYTHONDIR=lib/python3.11/site-packages -DUSE_VDT=OFF`). conda-forge `cms-combine` does not fit: for ROOT 6.34.4 + py311 the newest is 10.2.1; 11.x needs ROOT 6.38. The CMSSW recipe does not fit either (jobs do not run `cmsenv`; `run_combine` uses `combine` next to the env python). Decision: README section "CMS Combine" with the standalone conda recipe in Tau3Mu, version pinned. |
| 2026-10-21 | Kaito: reorder. GNN training (now step 12, npvs weights = sub-task 12a) comes before HF (now 13); lxplus = 14. Reason: no HF GNN with the BDT preselection exists. Checked on T3: the only HF model `600k_disco0.5_permuted_HF_fl_8_sd_4_GINE_vs1` (permuted, no BDT preselection, no fine-tuning; W: delete_muons, BDT cut 0.8, fine-tuned); no HF graphs `with_bdt_iso` / `BDT_cut` on pnfs (W has both); `data/signal_HF/bdt_full/` on pnfs empty; `GraphCreation_HF.sh` has the `--bdt_preselection_cut` line commented out. An HF ISO BDT exists (`output/benchmark_HF_BDT_vs1/ISO_classifiers_24feb2026_v0.pck`, not the W file). EOS on lxplus not checked. Older log entries keep the old step numbers. |
| 2026-10-21 | Kaito: the clean-account dry run moves to the end (step 15, on lxplus). It tests the final setup; a T3 run now would be repeated after the lxplus move. Step 11 = the Combine gap only. |
| 2026-10-21 | Step 3c (Kaito): resonance skim from Riccardo's gist `tau3mu_gnn_pf_skimmer.py` (gist ef1bcf09; it has only χ² < 100 and the PF ΔR slimming). Checked the old skims (`old_data/`, signal + background): medium ID and charge ±1 already true in raw NanoAOD; χ² < 100; m(3μ) < 3 GeV on the 4-vector sum of the muons (muon mass 0.10566, defined by the skimmer; raw has no muon mass branch), no lower cut; OS dimuon veto on the 4-vector mass, windows 0.742–0.822 and 0.980–1.060 GeV; PF ΔR < 1.0 (not 1.5 as in the gist). New frozen values (Kaito): χ² < 20 (report value; keeps 95.6 % signal, 68.7 % background after stage 1), 1 < m < 3 GeV (4-vector mass; differs from `fitted_mass` for 0.3 % of background), veto ±40 MeV around PDG ω 0.78266 and φ 1.019461, PF ΔR < 1.0. |
| 2026-10-21 | Step 3c (Kaito): `TauPlusMET_*` has one entry per tau candidate. The old skims did not mask it, so it is misaligned with `Slimmed_TauTo3Mu_*` when candidates are removed (2.4 % of old background events; the GNN reads `TauPlusMET_Puppi_phi[tau_mask][nt]`). Stage 0 now masks `TauPlusMET_*` and `nTauPlusMET` in place (same names). Raw inputs: required list = analysis names without `Slimmed_` + the stage 0 inputs (`raw_required_branches`); branch map keys are raw names. Regression: one raw signal file and one raw Run2024C data file with the old cut values match the old skims on every branch and event (`TauPlusMET_*` compared where no candidate was removed). Open: the skim job resources (3 h, 128G) are for resonance-skimmed inputs; a full raw run may need more. |
| 2026-10-23 | Step 12 design (Kaito), spec `docs/superpowers/specs/2026-10-23-step12-training-pipeline-design.md`: two pipelines (training `tau3mu-train`, analysis unchanged + `models:` key); the analysis always reruns phase A, no reuse of training outputs; both trainings on sideband background only (0.08 GeV); GNN graphs with the BDT cut at creation (as commit `01d680e`), no fine-tune; graph limit 600k, BDT 100k per class (user keys with defaults); candidate 0 only; npvs on all skimmed events; model-defining settings travel with the model; Comet ML kept, API key in `training_defaults_W.yaml`; benchmark limit after BDT inference, cut scan. T3 for now; HTCondor in step 14. |
| 2026-10-23 | Kaito: code layout `src/tau3mu_gnn/pipelines/{common,analysis,training}` (was `pipeline/` with `pipeline/training/`); shared stages in `common/stages/`; clean break, no alias for `tau3mu_gnn.pipeline`. Generic manifest / runs / driver take the phase maps and settings explicitly; `analysis/manifest.py` and `training/manifest.py` hold them. Tests in one folder `tests/pipelines/` (shared fixtures and helper imports between test files). Console scripts unchanged. Older entries keep the old paths. |
| 2026-10-24 | Kaito: new step 13b (full W chain from raw NanoAOD with the new skim: training + analysis) and step 13c after it (E1, E3, E4 and the `pair_charge` edge feature, each with a new limit). Comet metrics now logged with `step` = optimizer steps so far (training batches) and `epoch` (the default Comet charts use the step; without it they showed "No data"; one value per metric and epoch, as before). |
| 2026-10-24 | Kaito: order 13b → 13c → 13 (HF). The skim job resources (1 day, 128G) may be too small for the full raw run in 13b; check MaxRSS / elapsed after it. |
| 2026-10-24 | Kaito: step 12j, all old BDT training plots (ISO) back in `bdt_train`, for future trainings; no rerun of the 12i training. |
| 2026-10-24 | Kaito: step 12k, analysis phases `bdt` / `gnn` / `limit` (option 1 of `bdt`/`gnn`/`limit` vs `bdt_score`/`gnn_score`/`limit`); manifest key `pipeline` keeps the two pipelines apart. No alias for the old names. |
| 2026-10-25 | Kaito: 12i accepted (GNN 1.48 vs BDT-only 2.52, reference 1.29). 13c: E1 and `pair_charge` as a clean break (old models no longer match; frozen `models/W/` to be replaced by the 13c models); E1 length mismatch studied first; E3 ±10σ; E4 one range 1.6–2.0. |
| 2026-10-25 | E1 study: the `TauPlusMET` length mismatch comes only from the old resonance skim (stage 0 keeps the collections aligned). Kaito: option A, stage 0 refuses misaligned resonance-skimmed inputs (old `old_data` skims no longer run); BDT takes `TauPlusMET` of the first selected candidate. |
| 2026-10-25 | Kaito: 13c measured with one retrain with all 4 fixes (E1, E3, E4, `pair_charge`), one combined limit vs 13b. |
