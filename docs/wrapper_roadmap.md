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

- **Branch map**: applied **once**, by the first stage that runs in phase A (stage 0, or stage 1 if stage 0 is skipped), through one shared helper (`load_with_branch_map`). It renames his raw names → Kaito's raw names (`Slimmed_*`, `PFCands_*`, `TauPlusMET_*`, `Muon_*`, `PV_npvs`, `run`, `luminosityBlock`, `event`). All later stages run unchanged on Kaito's names. The prefix change `Slimmed_*` → `Selected_*` stays inside the skimmer.
- **Required branches**: one explicit list (~60 branches) in code. After mapping, every required branch must exist. If one is missing: **crash loudly**, list all missing branches, do no work. No fallbacks, no computed substitutes.

### 5.1 Config

Merged view (defaults + user file). Kaito's W values shown.

```yaml
analysis_name: W_new_hope_v2
channel: W                              # W now, HF later

paths:
  output_dir: root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/<user>/tau3mu_run/
  repo_dir:   /work/<user>/SVJ_GNN
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
├── branches.py     # required-branch list + load_with_branch_map
├── manifest.py     # write/read manifest, stale-settings check
├── storage.py      # scratch dir + xrdcp to output_dir
├── slurm.py        # submit jobs, dependencies (later: condor.py)
└── stages/         # one thin file per stage 0-5
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
2. Config: `analysis_defaults_W.yaml`, `example_W.yaml`, and `config.py` (load both, merge, validation: missing key / unknown key / frozen key in user file → crash). Unit tests.
3. Branch map: write the required-branch list, the `load_with_branch_map` helper, and tests (identity map on Kaito's files; renamed test file; missing branch → crash).
3b. Stage "resonance skim": add an empty placeholder stage (clear "not implemented" message, or pass-through when the input is already resonance-skimmed). Fill it when Kaito gives the code.
4. Stage "skim": refactor to take inputs from config. Regression check vs old output.
5. Stage "BDT inference": make `BDT_ScoreProducer` run **only** the ISO classifier. Remove the GNN classifier and the `GNN_score` input. (Own mini step, before the config refactor of this stage.)
5b. Stage "BDT inference": take inputs from config.
6. Stage "GNN inference": refactor `ScoreCreation`/`ScoreProducer` into a function; inputs from config. Regression check vs old output.
7. Stage "model eval": refactor the `ModelEval` modes used here (`tau_hist_fpr`, `tau_hist`, `ROC_comp_full`) into functions; inputs from config.
8. Stage "limit": refactor `datacard_creation_GNN_permuted` + `readout_limit` into functions; inputs from config. Regression check: cut 3.9 → 1.2890625.
8b. Cut scan: config `gnn_score_cut` takes one value or a list. One Combine job per cut, in parallel. A final job writes a table (cut → expected limit) and marks the best cut.
9. Driver: phase sub-commands (A, B, C), run-single-stage, manifest write/read, input checks, stale-settings warning.
10. Bash entry point + README section for the new user.
10b. `all` command: submits A, B and C at once. Each phase command takes `--after <JOBID>` and prints the job ID of its **last** job. `all` passes A's last job ID to B and B's to C (`--dependency=afterok:<JOBID>`). If a job fails, the later jobs never start.
11. Dry run on a clean account / clean checkout.
12. Extend to the HF channel: HF config, HF features, multi-file inputs, HF signal weight. Test.
13. (Very last) Port to lxplus: HTCondor scheduler layer, EOS paths, Combine on lxplus. Test.

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
