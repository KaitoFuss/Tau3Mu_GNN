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
| Limit | `jobs/analysis/create_datacard_permuted.sh` | `tau3mu_gnn.analysis.datacard_creation_GNN_permuted`, `combine`, `tau3mu_gnn.analysis.readout_limit` |

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
- One Python driver with one sub-command per phase (`phase-a`, `phase-b`, `phase-c`). Each stage is also callable alone.
- One thin bash wrapper that activates the env and calls the driver.
- **Hand-off between phases**: each phase writes to a fixed folder under the config's output dir, plus a small manifest (list of output files + snapshot of the config used). The next phase reads the manifest, checks the files exist, and compares the upstream settings to the current config. These checks run **inside the first job of the phase** (at run time), not at submit time, because with `all` the upstream outputs do not exist yet when the job is submitted.
- Each stage is a plain Python function/CLI. It knows **nothing** about SLURM, HTCondor or dCache.
- A separate, thin **scheduler layer** submits the stages of **one phase** as SLURM jobs, with dependencies inside the phase (`sbatch --dependency=afterok:...`). Independent jobs run in parallel (e.g. signal and background GNN inference). Phases are **not** auto-chained; the user starts the next phase. Later an HTCondor layer can replace this layer without touching the stages.
- **Storage**: every job works in node-local `/scratch/$USER/$SLURM_JOB_ID`. At the end it copies its outputs with `xrdcp` to `<output_dir>` (from config, e.g. on dCache) and cleans scratch. Copy logic is **one shared helper**, not repeated per stage.
- **Output layout** (everything is kept, all under one dir):

```
<output_dir>/
├── config_used.yaml
├── logs/                         # all SLURM .out/.err
├── phase_a/
│   ├── manifest.json
│   ├── 0_resonance_skim/
│   ├── 1_skim/
│   └── 2_bdt/                    # ROOT files with BDT ISO score
├── phase_b/
│   ├── manifest.json
│   ├── 3_gnn/                    # ROOT files with GNN_score
│   └── 4_eval/                   # sculpting plots, ROC vs BDT
└── phase_c/
    ├── manifest.json
    ├── cut_<value>/              # datacard, Combine output, fit plots
    └── limits.json               # cut → limit table, best cut marked
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
  # run folder = <output_base>/<analysis_name>_vsN (version added in step 9)
  conda_env:  /work/<user>/miniforge3/envs/Tau3Mu

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

slurm:
  bdt:      {partition: standard, mem: 64G}
  gnn_bkg:  {partition: long, mem: 64G, time: 7-00:00:00}
  limit:    {partition: standard, mem: 128G}
```

**Split into two files** (decided):

- `analysis_defaults_<channel>.yaml` (in repo, frozen): `gnn.*`, `bdt.classifier`, `skim.*`, `gnn.bdt_preselection_cut`, `limit.mass_window`, `limit.fit_range`, `limit.cl`.
- `my_analysis.yaml` (user, short): `analysis_name`, `channel`, `paths`, `inputs`, `branch_map`, `limit.signal_weight`, `limit.gnn_score_cut`, `slurm`.
- The defaults file is a **separate config file that is not meant to be changed**. There is no override mechanism. A frozen key in the user file counts as an unknown key → crash.

Config rules: missing required key → crash. Unknown key → crash (catches typos). Defaults only where marked.

### 5.2 Code layout

New code only. Old job scripts keep working.

```
src/tau3mu_gnn/pipeline/
├── cli.py          # tau3mu-run phase-a|phase-b|phase-c|all|stage <name>
├── config.py       # load defaults + user file, merge, validate
├── branches.py     # required branches, input checks, RDataFrame rename
├── manifest.py     # write/read manifest, stale-settings check
├── storage.py      # scratch dir + xrdcp to output_dir
├── slurm.py        # submit jobs, dependencies (later: condor.py)
└── stages/         # one thin file per stage 0-5
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
4. ✅ Stage "skim": refactor to take inputs from config. Regression check vs old output.
5. ✅ Stage "BDT inference": make `BDT_ScoreProducer` run **only** the ISO classifier. Remove the GNN classifier and the `GNN_score` input. (Own mini step, before the config refactor of this stage.) **Caution:** the BDT drops rows with NaN in *any loaded* column (`dropna`); loading fewer branches can change which events get `bdt_ISO` — check against the reference limit. Today the W BDT also loads 5 raw branches it does not use as features and that are **not** in `required_branches("W")`: `Slimmed_TauTo3Mu_charge`, `Slimmed_TauTo3Mu_mu{1,2,3}_idx`, `Muon_tkRelIso` (`BDT_ScoreProducer.py:100-127`). Decide on purpose: drop them from the load (then check the limit) or add them to the W list. Add a test that the BDT feature list ⊆ `required_branches("W")`.
5b. Stage "BDT inference": take inputs from config.
6. Stage "GNN inference": refactor `ScoreCreation`/`ScoreProducer` into a function; inputs from config. Regression check vs old output. Add a test that the GNN branch list ⊆ `required_branches("W")`.
7. Stage "model eval": refactor the `ModelEval` modes used here (`tau_hist_fpr`, `tau_hist`, `ROC_comp_full`) into functions; inputs from config. **Note:** `ROC_comp_full` reads `GNN_score_{sig,bkg}` from `tau_score_corr.pkl` (GNN-input BDT, dropped in step 5) — adapt it to ISO only.
8. Stage "limit": refactor `datacard_creation_GNN_permuted` + `readout_limit` into functions; inputs from config. Regression check: cut 3.9 → 1.2890625.
8b. Cut scan: config `gnn_score_cut` takes one value or a list. One Combine job per cut, in parallel. A final job writes a table (cut → expected limit) and marks the best cut.
9. Driver: phase sub-commands (A, B, C), run-single-stage (stages 0 + 1 always together, one job), manifest write/read, input checks, stale-settings warning. **Run versioning**: phase A creates `<output_base>/<analysis_name>_vsN` (N = next free, same logic as `TrainingUtils.py` unique-name helper). Phases B and C use the latest `_vsN` by default; `--run <analysis_name>_vsN` picks another. Phase C cut scans add `cut_<value>/` folders inside the same run, no new version.
10. Bash entry point + README section for the new user.
10b. `all` command: submits A, B and C at once. Each phase command takes `--after <JOBID>` and prints the job ID of its **last** job. `all` passes A's last job ID to B and B's to C (`--dependency=afterok:<JOBID>`). If a job fails, the later jobs never start.
11. Dry run on a clean account / clean checkout.
12. Extend to the HF channel: HF config, HF features, multi-file inputs, HF signal weight. Test.
13. (Very last) Port to lxplus: HTCondor scheduler layer, EOS paths, Combine on lxplus. Test.

**Maybe, at the very end** (Kaito decides then; not needed for the limit):

- **E1. BDT: `TauPlusMET` candidate mix.** The BDT takes `[0]` of `Selected_TauTo3Mu_*` (candidates that pass the skim) and `[0]` of `TauPlusMET_*` (all candidates, before the skim). If the first candidate fails the skim and a later one passes, the tau features and the tau+MET features come from different candidates. Affects 5 of the 6 W BDT `TauPlusMET` branches: `Puppi_pt`, `Puppi_phi` (in `deltaPhi`), `Tau_Puppi_mT`, `PuppiMETminPz`, `PuppiMETmaxPz` (per candidate). `PuppiMET_pt` is global (same value in every entry), not affected. Measured (W): signal 75 events (0.01%), data 2,214 of the first 1M events (0.29%). Also, `len(TauPlusMET_*) == nSlimmedTauTo3Mu` in only 99.4% (signal) / 97.6% (data) of events. A fix changes the BDT inputs → maybe retrain the classifier. Same root cause as the `ScoreProducer.py` issue in the decision log (2026-10-07).
- **E2. BDT: vectorized merge.** `_write_with_scores` loops in Python over all events (`iterrows`, then a PyROOT loop with `GetEntryNumberWithIndex`). This is slow: the full W BDT run (job 946596) took 2 h 20 min; the loops are a large part of it (not measured alone). Replace with a vectorized join on `(run, event)`. Regression: outputs must stay identical.

## 7. Open questions

Tracked here. Answers move to section 8.

- (none open right now)

## 7b. Waiting on Kaito

- Code for the resonance skim (raw NanoAOD → `*_skimmed_no_resonances.root`).
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
