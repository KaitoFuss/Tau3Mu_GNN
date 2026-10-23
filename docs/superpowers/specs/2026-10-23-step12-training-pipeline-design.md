# Step 12 — Training Pipeline: Design

Status: agreed with Kaito in chat, 2026-10-23. Not implemented yet.
Parent document: `docs/wrapper_roadmap.md` (sec. 6 step 12). Where this spec differs from the roadmap, this spec wins and the roadmap gets updated (sec. 11).

## 1. Intent

Today the analysis pipeline (`tau3mu-pipeline`, phases A → B → C) runs with **frozen** models: the ISO BDT `models/W/bdt/ISO_classifiers_24feb2026_v0.pck` and the GNN `models/W/gnn/new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2/`. Both were trained by hand, with scripts that have hard-coded paths, on the **old** skim.

The new stage 0 (PR #24: χ² < 20, 1 < m < 3 GeV, ω/φ veto ±40 MeV) changes the input data. So both models must be retrained before a new reference limit makes sense.

Goal: **two clean pipelines**.

- **Training pipeline** (new, `tau3mu-train`): inputs → skim → BDT training → BDT inference → (npvs weights → graph creation → GNN training) and, in parallel, (benchmark BDT-only limit). Output: a trained BDT + GNN, plus a benchmark limit table.
- **Analysis pipeline** (exists, `tau3mu-pipeline`): unchanged, except one new user key selects the models: the frozen ones (default) or the ones of a training run.

A new user with a new dataset can run only the analysis pipeline, with the frozen models. Retraining is optional.

## 2. Goals, non-goals, anti-goals

Goals
- One user config + one command trains both models on T3 SLURM (GPU for the GNN), with the same style as the analysis pipeline: frozen defaults file + user file, versioned output folders, manifests, `--dry-run`.
- The training reproduces Kaito's last good recipe (sec. 4). Check: train on the **old** skimmed inputs, run the analysis with the new models, the expected limit is **roughly** the reference 1.2890625 (cut 3.9). Kaito judges "roughly".
- A BDT-only benchmark limit (cut scan, best cut) comes out of every training run: the number the GNN must beat.

Non-goals
- No fine-tuning (`--finetune` in `Train_TauMuNet`): Kaito tried it, it did not help. Not ported.
- No reuse of training outputs (skim, BDT scores) by the analysis. The analysis always runs phase A again from scratch (a real analysis uses a new dataset). Not even as an option.
- No HTCondor / lxplus / EOS. That is roadmap step 14; it then also covers this pipeline.
- No hyper-parameter search (optuna code in `adrian_utils/` stays out).
- No HF channel (step 13).

Anti-goals (would make it a failure even if it runs)
- Training and analysis disagree on a model-defining setting (BDT preselection cut, graph type, k, cone, unsculpting mode, conv type, GNN mode) without an error. See sec. 7.
- A 3-day GPU job fails at the end for a reason that could be checked at the start (missing input branch, no Comet key, output folder not writable).
- Analysis settings get two sources of truth (e.g. the skim cuts copied into the training defaults and drifting apart). See sec. 5.

## 3. The chain

```
                      ┌─► npvs ─► graphs_signal ──┐
skim ─► bdt_train ─► bdt_infer                     ├─► gnn_train
                      ├─► graphs_background ───────┘    (no npvs: background weight 1)
                      └─► benchmark_limit_<cut> (one job per cut) ─► benchmark_summary
```

- `skim` = stage 0 (resonance skim, if inputs are raw) + stage 1 (skim). The **same code** as the analysis (`pipeline/stages/resonance_skim.py`, `skim.py`), the same frozen cuts.
- `bdt_train`: new. Trains the ISO BDT. Port of `bdt_training/Train_XGB_W.py`, **ISO mode only** (the script also trains a "GNN" mode that uses `GNN_score` as a feature: not ported).
- `bdt_infer`: the existing stage 2 (`pipeline/stages/bdt.py`) with the **new** classifier.
- `npvs`: new (was roadmap step 12a). Port of `utils/CorrectPileupDist.create_weight_array`: `PV_npvs` of signal vs background → `signal_<channel>_npvs_weights.npz` (`weight_per_bin`, `bins`) + plots. **Signal only** (checked 2026-10-23 in the old code): only the signal graphs get `weights="npvs_rescale"` (`GraphCreation.py:180`), background graphs get weight 1, and GNN inference uses no weights. The training samples graphs with these weights (`graphDataLoader.py:102-113`, `WeightedRandomSampler`, normalised per sample), so they reshape the signal `PV_npvs` distribution to the background one. Background is read only as the target shape. So `graphs_background` does not wait for `npvs`. Per-event lookup: `CorrectPileupDist.signal_event_weights` (`PV_npvs` clipped to the table; values ≥ 100 get the last bin's weight). Note: the old weights were made before the skim; the new ones are made after the skim, so they differ a bit.
- `graphs_signal`, `graphs_background`: new. Port of `scripts/GraphCreation.py` + `graph/ConeGraphProducer.py`. **The BDT preselection cut is applied here**: only events that pass go into the graph files (as in commit `01d680e`, which made the graphs of the frozen model). One job per sample, in parallel.
- `gnn_train`: new. Port of `scripts/Train_TauMuNet.py` without the fine-tune part. GPU job.
- `benchmark_limit_<cut>`, `benchmark_summary`: new stage, the BDT-only limit. Port of `analysis/datacard_creation_ISO.py` + the two Combine commands of `jobs/analysis/create_datacard_benchmark.sh`. A cut scan like the GNN scan (step 8b): one job per cut, a summary job writes the table and marks the best cut. Starts after `bdt_infer` (not after `bdt_train`), in parallel with the GNN branch.

## 4. The recipe to reproduce (from Kaito and the old scripts)

| Part | Value | Source |
|---|---|---|
| BDT train events | signal: all candidates; background: **sideband only**, candidate-level `abs(fitted_mass - 1.78) > 0.08`, events with all candidates in the sideband | `Train_XGB_W.py:215-216` |
| BDT sample size | at most 100,000 events per class (`sample(random_state=42)`) | `Train_XGB_W.py:238-239` |
| BDT split / folds | `train_test_split(test_size=0.4, random_state=1986)`, `StratifiedKFold(6, random_state=1986)` | `Train_XGB_W.py:374-386` |
| BDT XGBoost | `max_depth=8`, `learning_rate=0.005`, `n_estimators=1000`, `early_stopping_rounds=100`, `min_child_weight=1e-6*sum(w)`, weights: both classes the same total weight, mean 1 (Kaito, PR #28; the old script used 1) | `Train_XGB_W.py:300-420` |
| BDT features | the ISO feature list of `Train_XGB_W.py` (`modes["ISO"]`), must equal what stage 2 reads | `Train_XGB_W.py:88-` |
| npvs events | **all skimmed signal vs all skimmed background** (`1_skim`), no BDT cut, no sideband (Kaito, 2026-10-23) | |
| npvs bins | the old **variable binning** (`var_bins`) mapped onto `np.linspace(0, 100, 101)`, as `create_weight_array` does; frozen | `CorrectPileupDist.py` |
| Graph BDT cut | 0.8 (user key, sec. 6) | Kaito, commit `01d680e` |
| Graph order | BDT cut → sideband mask (background) → event limit, as in commit `01d680e` | `01d680e` |
| Graph candidate | **candidate 0 only**, the candidate that stages 2 and 3 score. Background: the event is kept only if candidate 0 is in the sideband. One graph per event. (Kaito, 2026-10-23, option a; the old code used the first sideband candidate.) | |
| Graph type | knn, k = 10, cone 0.5, `dzd0`, unsculpting `delete_muons` | `jobs/graph_creation/GraphCreation_W.sh`, `GraphCreation.py` |
| Graph background | sideband only, same 0.08 window (`ConeGraphProducer.py:628`) | Kaito: both models train on sideband background only |
| Graph-level features | the W list; one shared source with inference: `gnn_training/gnn_inputs.graph_level_branches` (the training defaults point to it, no copy) | `GraphCreation.py:143-160` |
| Graph event limit | **600,000** per sample (user key with default, sec. 5), applied last | `01d680e` (Kaito, 2026-10-23) |
| GNN | no pretraining on all graphs, no fine-tune: one training on the BDT-cut graphs (Kaito: the condor script's `--finetune` was a later test). GINE, `final_layer`, arch `fl8_sd4` (gives exactly the frozen `arch_config.json`), lr 0.0016, batch 128, disco 0, 1000 epochs, early stopping (`utrain.early_stop(patience=20)`), `ReduceLROnPlateau(patience=10)` | `jobs/condor/train_W_GINE_deleted.sh`, `Train_TauMuNet.py` |
| GNN data split | windows train 0–0.6, validation 0.6–0.7, test 0.7–1.0; class mix 50/50 (`df = [0.5, 0.5]`) | `Train_TauMuNet.py:224-240` |
| GNN outputs | model `.pt`, `Normalization.pkl`, `arch_config.json` (the files the analysis loads from `<model_dir>/<model_name>/`), loss plots, test-set outputs | `Train_TauMuNet.py`, `models/W/gnn/...` |
| Comet ML | kept. Project = training name, workspace + API key in the frozen defaults (sec. 5) | Kaito |
| Benchmark | selection `bdt_ISO > cut`, blinded window `abs(mass - 1.78) > 0.06`, Combine `FitDiagnostics` + `AsymptoticLimits -C 0.90 --run blind` | `datacard_creation_ISO.py`, `create_datacard_benchmark.sh` |

Where the old code and this table disagree, this table wins (Kaito's decisions of 2026-10-23). Other differences found while porting: report to Kaito before choosing.

Note: the BDT training keeps its own old rule (background events with **all** candidates in the sideband, `Train_XGB_W.py:215-216`). Not changed in this step.

## 5. Configs

Two pairs of files, same rules as today (frozen file not edited by users; a frozen key in the user file → error; unknown key → error).

| File | Kind | Content |
|---|---|---|
| `configs/analysis_defaults_W.yaml` | frozen, exists | resonance_skim, skim, bdt, gnn, limit, slurm (analysis jobs) |
| `configs/example_W.yaml`, `example_W_raw.yaml` | user, exist | + new optional key `models` (sec. 7) |
| `configs/training_defaults_W.yaml` | frozen, **new** | `bdt_training`, `npvs`, `graphs`, `gnn_training`, `comet`, `slurm` (training jobs) |
| `configs/example_training_W.yaml` | user, **new** | name, paths, inputs, branch map, graph BDT cut, event limits, benchmark cuts, signal weight |

**One source of truth for shared stages.** The training pipeline runs the analysis stages skim (0 + 1), BDT inference and a limit fit. Their settings (`resonance_skim`, `skim`, `limit.tau_mass / mass_window / fit_range / cl`) are **read from `analysis_defaults_W.yaml`**, not copied into the training defaults. The training loader loads both frozen files. (`bdt.classifier` and the `gnn` block of the analysis defaults are not used by the training.)

`training_defaults_W.yaml` (frozen) — sketch, the builder fixes names:

```yaml
bdt_training:   {features: [...], max_depth: 8, learning_rate: 0.005, n_estimators: 1000,
                 early_stopping_rounds: 100, kfold: 6, test_size: 0.4, seeds: {...}}
sideband_half_width: 0.08        # GeV around tau_mass; BDT + graph background
npvs:           {var_bins: [...], bins: [0, 100, 101]}   # old variable binning, mapped onto 0-100
graphs:         {graph_type: knn, k: 10, cone_size: 0.5, edge_features: dzd0,
                 unsculpting_mode: delete_muons, graph_level_features: [...]}
gnn_training:   {conv_type: GINE, gnn_mode: final_layer, arch: fl8_sd4, learning_rate: 0.0016,
                 batch_size: 128, disco: 0.0, epochs: 1000, early_stopping_patience: 20,
                 lr_scheduler_patience: 10, class_fraction: [0.5, 0.5],
                 windows: {train: [0.0, 0.6], validation: [0.6, 0.7], test: [0.7, 1.0]}}
comet:          {workspace: kaitofuss, api_key: "..."}   # Kaito: plain key here is fine
slurm:          {jobs: {skim: ..., bdt_train: ..., bdt_infer: ..., npvs: ...,
                        graphs_signal: ..., graphs_background: ..., gnn_train: ...,
                        benchmark_limit: ..., benchmark_summary: ...}}
```

The Comet API key moves from `Train_TauMuNet.py:709` (hard-coded) into this file (Kaito, 2026-10-23: "it does not need to be that secure").

`example_training_W.yaml` (user) — sketch:

```yaml
training_name: kaito_W_train       # folder name, like analysis_name
channel: W
paths: {output_base: ..., log_dir: ..., conda_env: ...}     # same keys as the analysis
inputs: {signal: [...], background: [...], tree_name: Events, already_resonance_skimmed: true}
branch_map: {}
graphs:
  bdt_preselection_cut: 0.8        # events with bdt_ISO >= cut go into the graphs
  max_events: 600000               # optional, default 600000, per sample
bdt_training:
  max_events_per_class: 100000     # optional, default 100000
benchmark:
  bdt_score_cut: [0.98, 0.986, 0.99]   # one value or a list → cut scan
limit:
  signal_weight: 0.000019          # required, as in the analysis
```

Event limits (Kaito, 2026-10-23: "can be set with defaults"): optional user keys; the default value lives with the user-key schema (like `tree_name: Events`), not in the frozen file.

## 6. Cut rule

The graph BDT cut uses **`bdt_ISO >= cut`**, the same rule as GNN inference (`gnn_training/ScoreProducer.py`). The old graph code used `>`. Candidate 0 (the one stages 2 and 3 score), see sec. 4. The benchmark keeps its old `bdt_ISO > cut`; this matches stage 5 (`GNN_score > cut`).

## 7. How the analysis uses a training run

New optional user key in the **analysis** config:

```yaml
models: frozen                     # default
# models: root://.../<training_name>/gnn_vsN     # a finished GNN training version
```

- `frozen`: today's behaviour.
- A training version: stage 2 loads the BDT classifier of that training (via the manifest chain), stage 3 loads its GNN model, normalization and arch config.
- **Model-defining settings travel with the model** (Kaito, 2026-10-23). With a training version, the analysis reads these from the training manifest, not from `analysis_defaults_W.yaml`: `gnn.bdt_preselection_cut`, `graph_type` (k, cone, dzd0, knn), `unsculpting_mode`, `conv_type`, `gnn_mode`, `model_name`. The analysis manifest records the model source and these values, so phase checks (`check_upstream`) see a model change as a setting change.
- The analysis checks at submit time that the training version exists and is complete (manifest present, model files present).

## 8. Versions, layout, commands

My call (Claude), same style as step 9. The training has three phases so that a failed 3-day GNN job can be rerun without redoing the BDT:

| Phase | Jobs | Needs |
|---|---|---|
| `bdt` | skim → bdt_train → bdt_infer | inputs |
| `gnn` | (npvs → graphs_signal) ∥ graphs_background → gnn_train | a `bdt_vsN` |
| `benchmark` | benchmark_limit_<cut> (∥) → benchmark_summary | a `bdt_vsN` |

```
<output_base>/<training_name>/
├── bdt_vsN/        manifest.json, config_used.yaml, logs/, 0_resonance_skim/, 1_skim/, bdt_train/ (classifier .pck, plots), 2_bdt/
├── gnn_vsN/        manifest.json, config_used.yaml, logs/, npvs/ (signal_W_npvs_weights.npz, plots), graphs/ (.pt), gnn_train/ (<model_name>/ model files, plots, test outputs)
├── benchmark_vsN/  manifest.json, config_used.yaml, logs/, cut_<cut>/, limits.{json,txt,png}
└── debug/<stage>_vsN/
```

Commands (new console script, like `tau3mu-pipeline`):

```
tau3mu-train all       --config my_training.yaml [--dry-run]     # bdt, then gnn and benchmark in parallel
tau3mu-train bdt       --config ... [--dry-run]
tau3mu-train gnn       --config ... [--from bdt_vsN] [--dry-run]
tau3mu-train benchmark --config ... [--from bdt_vsN] [--dry-run]
tau3mu-train stage <name> --config ... [--from ...] [--cut X] [--sample S] [--dry-run]
```

Then, by hand: put `models: <.../gnn_vsN>` in the analysis config and run `tau3mu-pipeline all`.

`all` works like the analysis `all` (step 10b): the `gnn` and `benchmark` jobs get the new `bdt_vsN` as upstream and wait `afterok` on `bdt_infer`.

## 9. Code layout

- Shared infrastructure stays in `src/tau3mu_gnn/pipeline/` (config loading helpers, `storage.py`, `manifest.py`, `runs.py`, `slurm.py`, `branches.py`). Generalise where needed (e.g. phase names, per-job SLURM account) rather than copy.
- Shared stages stay in `pipeline/stages/` (resonance skim, skim, bdt inference, limit).
- New, training only: `src/tau3mu_gnn/pipeline/training/` with `config.py` (training config dataclasses + loader), `stages/{bdt_train,npvs,graphs,gnn_train,benchmark}.py`, `jobs.py`, `driver.py`, `cli.py`.
- The old scripts (`Train_XGB_W.py`, `GraphCreation.py`, `Train_TauMuNet.py`, `CorrectPileupDist.py`, `datacard_creation_ISO.py`) get refactored into functions that take explicit inputs, as in steps 4–8. Their old CLIs stay working where cheap; hard-coded paths go.
- Required input branches: the training adds `PV_npvs` and the BDT training features to the required list **of the training only** (the analysis list does not change).

## 10. SLURM and GPU

- `gnn_train`: partition `gpu`, account **`gpu_gres`**, `--gres=gpu:1`, 5 CPUs. The analysis `slurm:` block has one `account` for all jobs; the training needs a per-job account (and `gres`). First values: mem 128G, time `3-00:00:00` (the gpu partition allows 7 days). Memory as a power of 2.
- `graphs_*`: CPU, partition `long`, first value mem 256G, time `2-00:00:00` (old `GraphCreation_W.sh`).
- Others: like the analysis jobs. Kaito raises the values after the first real run.
- Pre-checks at submit time and at the start of each job (sec. 2 anti-goals): input branches, Comet key present, output writable, proxy valid for the longest chain.

## 11. Sub-steps (roadmap update)

Each small, tested, own PR (as steps 2–11):

| Step | Content |
|---|---|
| 12a | Training config: `training_defaults_W.yaml`, `example_training_W.yaml`, loader, validation, tests. |
| 12b | Stage npvs (was 12a). |
| 12c | Stage bdt_train (ISO only). |
| 12d | Stage graphs (BDT cut at creation). |
| 12e | Stage gnn_train (no fine-tune; Comet key from config). |
| 12f | Stage benchmark limit (cut scan + summary). |
| 12g | Training driver, SLURM (GPU job), `tau3mu-train`. |
| 12h | Analysis `models:` key; model settings from the training manifest. |
| 12i | Real check (Kaito queues): train on the old skimmed inputs, analysis with the new models, limit ≈ 1.29. README section "Training pipeline". Update README "Pipeline Overview" and "Quick Start". |

Roadmap edits: sec. 1 scope (training now in the wrapper), sec. 6 step 12 → this table, the decision log entry for 2026-10-23.

## 12. Testing

- Unit tests per stage on toy ROOT files (as `tests/pipeline/`), CPU only.
- `gnn_train`: a few epochs on toy graphs, CPU, Comet disabled in tests (offline mode or a test switch).
- `bdt_train`: small toy sample; check the classifier file loads in stage 2.
- `graphs`: check the BDT cut (`>=`), the sideband mask, the event limit, and that the graph files load in the training.
- Driver: `--dry-run` job graph (dependencies, accounts, GPU flags), like the analysis driver tests.
- No exact regression vs the frozen models (training is random). The real check is 12i.

## 13. Left to the builder

- Exact key names and dataclass layout (follow `pipeline/config.py`).
- How the training loader reuses the analysis loader code.
- Plot set of each stage (keep what the old scripts make, if cheap).
- Whether `debug/` single-stage runs are in 12g or later.
