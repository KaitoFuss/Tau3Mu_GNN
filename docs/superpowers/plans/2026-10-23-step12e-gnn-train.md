# Step 12e — Stage gnn_train: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A training stage that trains the GNN on the graphs of 12d and writes a model folder the analysis loads unchanged (`<model_name>/{<model_name>.pt, Normalization.pkl, arch_config.json}`), plus losses, plots and a test-set evaluation.

**Architecture:** Make the training code in `scripts/Train_TauMuNet.py` importable (no module-global Comet experiment, no import side effects), add the three recipe decisions of Kaito (patience 50, always the best epoch, a seed), and wrap it in `gnn_training/train_gnn.py` + a thin stage `pipeline/training/stages/gnn_train.py`. The old CLI keeps working.

**Tech Stack:** Python 3.11, torch, torch_geometric 2.5.2, comet_ml 3.56 (`Experiment(..., disabled=...)`), pytest. Tests: `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` (conda env `Tau3Mu`). CPU only in tests.

**Spec:** `docs/superpowers/specs/2026-10-23-step12-training-pipeline-design.md` (sec. 3 `gnn_train`, sec. 4 GNN rows, sec. 7 model files).

## Global Constraints

- Kaito, 2026-10-23: **early stopping patience 50** (the default `utrain.early_stop()` of the normal training; 20 was the fine-tune value) → fix `gnn_training.early_stopping_patience` in `training_defaults_W.yaml` (and toy defaults). **Always keep the best epoch** (lowest validation loss) in `<model_name>.pt`, also when all epochs run (the old code then overwrote it with the last epoch). **Seed:** new frozen key `gnn_training.seed: 1986` → `torch.manual_seed`, `np.random.seed`, `random.seed` at the start of training.
- No fine-tune (not ported; the old CLI keeps its `--finetune` branch as is).
- Recipe from `cfg.gnn_training`: `conv_type`, `gnn_mode`, `arch` (`fl8_sd4`), `learning_rate`, `batch_size`, `disco`, `epochs`, `early_stopping_patience`, `lr_scheduler_patience` (`ReduceLROnPlateau`), `class_fraction` (`dataset_fraction`), `windows` (train / validation / test). Adam without weight decay, `BCELoss`, DisCo warm-up as today. `Normalize` over all graphs (as today).
- Model files = what `gnn_training/model_loading.load_model` reads: `arch_config.json` = `json.dump(config)` of the arch lookup (equals the frozen file for `fl8_sd4`), `<model_name>.pt` = `{"epoch", "model_state_dict"}`, `Normalization.pkl` = dict with `means_x, stds_x, means_edge, stds_edge, means_graph, stds_graph`.
- `model_name = f"{cfg.training_name}_{cfg.channel}_fl_<i>_sd_<j>_{conv_type}_{unsculpting_mode}"` (same pattern as the frozen name, without `_vsN`: the version folder of 12g gives the version).
- Comet: workspace and API key from `cfg.comet`; project = `cfg.training_name`. Remove the hard-coded key from `Train_TauMuNet.py`; its old CLI reads `comet.api_key` / `comet.workspace` from `configs/training_defaults_W.yaml`. Tests run with `disabled=True` (no network).
- `architectures/GNN_archs.py` must import on any machine and must not write outside the output folder: `_detect_registry_path()` returns `None` on an unknown machine (no raise at import); `get_arch_configs(..., registry_path=None)` skips the registry; `get_arch_configs_random` raises a clear error if the path is `None`. Old behaviour on T3 / lxplus unchanged.
- Test-set evaluation must work with fewer than 5000 test graphs (`EvaluationUtils.py:160` draws 5000 without replacement): use `min(5000, n)`.
- Use `/usr/bin/git` in the worktree. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never print the Comet key.
- Branch `pipeline-step12e-gnn-train` is stacked on `pipeline-step12d-graphs` (PR #29).

## Review Focus

1. All epochs run without early stop → `<model_name>.pt` holds the best epoch, not the last (Task 2, `test_best_epoch_kept`).
2. The trained folder is loaded by `model_loading.load_model` unchanged and scores a graph (Task 3, `test_stage_model_loads_in_analysis`).
3. Same seed, CPU, twice → identical `model_state_dict` (Task 2, `test_seed_reproducible_on_cpu`).
4. Fewer than 5000 test graphs → evaluation runs (Task 2, covered by every toy training).
5. Importing `GNN_archs` / `Train_TauMuNet` on an unknown host and without network → no error, no file written outside the output folder (Task 1, `test_arch_import_no_side_effects`).

---

### Task 1: Config fixes and side-effect-free architecture lookup

**Files:**
- Modify: `configs/training_defaults_W.yaml`, `tests/pipeline/conftest.py` (`TOY_TRAINING_DEFAULTS`), `src/tau3mu_gnn/pipeline/training/config.py` (`gnn_training.seed`, `GNNTrainingConfig.seed`), `src/tau3mu_gnn/architectures/GNN_archs.py`
- Test: `tests/pipeline/test_training_config.py`, `tests/pipeline/test_real_training_configs.py`, `tests/pipeline/test_gnn_train.py` (new)

**Interfaces:**
- Produces: `GNNTrainingConfig.seed: int` (frozen, `_positive_int`, 1986); `early_stopping_patience` = 50 in both YAML/toy defaults; `GNN_archs.get_arch_configs(arch, tau_channel, disco_reg, registry_path=REGISTRY_PATH)` accepts `registry_path=None`.

- [ ] **Step 1: Failing tests:** real config `cfg.gnn_training.early_stopping_patience == 50` and `cfg.gnn_training.seed == 1986`; toy config the same; `gnn_training.seed` in the user file → frozen-setting error. In `test_gnn_train.py`:

```python
def test_arch_config_equals_frozen_file():
    import json
    from tau3mu_gnn.architectures.GNN_archs import get_arch_configs
    config, ident = get_arch_configs("fl8_sd4", "W", 0.0, registry_path=None)
    frozen = json.loads((REPO_ROOT / "models/W/gnn/new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2/arch_config.json").read_text())
    assert json.loads(json.dumps(config)) == frozen and ident == ["fl_8", "sd_4"]


def test_arch_import_no_side_effects(tmp_path, monkeypatch):
    import importlib, socket
    import tau3mu_gnn.architectures.GNN_archs as archs
    monkeypatch.setattr(socket, "gethostname", lambda: "unknown-host")
    monkeypatch.setattr(archs.os.path, "isdir", lambda p: False)   # neither /eos nor /work output exists
    reloaded = importlib.reload(archs)
    assert reloaded.REGISTRY_PATH is None
    reloaded.get_arch_configs("fl8_sd4", "W", 0.0, registry_path=None)   # no file written, no raise
    importlib.reload(archs)   # restore
```

(Adapt the monkeypatch to how `_detect_registry_path` really decides; the point: unknown machine → `None`, no raise.)
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** (YAML comment: "50 = default of the normal training; 20 was the fine-tune value"; seed comment: "torch, numpy, random; GPU training is still not bit-exact").
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_training_config.py tests/pipeline/test_real_training_configs.py tests/pipeline/test_gnn_train.py -q` → pass.
- [ ] **Step 5: Commit** — `fix(training): GNN patience 50, frozen seed; arch lookup without side effects`.

### Task 2: Importable training (`gnn_training/train_gnn.py`)

**Files:**
- Modify: `src/tau3mu_gnn/scripts/Train_TauMuNet.py` (pass `experiment` and `patience` into `train_and_validate`; after training reload the best checkpoint and re-save it as `<name>.pt` — no last-epoch overwrite; remove the hard-coded key, old CLI reads `cfg comet` from `configs/training_defaults_W.yaml`), `src/tau3mu_gnn/gnn_training/EvaluationUtils.py:160-161` (`min(5000, n)`)
- Create: `src/tau3mu_gnn/gnn_training/train_gnn.py`
- Test: `tests/pipeline/test_gnn_train.py`

**Interfaces:**
- Consumes: 12d graph layout (`<data_base>/<sample>_<channel>/processed/<prefix>_<sample>_<model_type>.pt`), `stages/graphs.model_type`, `GNNTrainingConfig`, `CometConfig`.
- Produces:
  - `train_gnn.make_experiment(comet: CometConfig, project: str, disabled: bool = False)` → `comet_ml.Experiment(api_key=comet.api_key, workspace=comet.workspace, project_name=project, disabled=disabled)`.
  - `train_gnn.train(data_base: str | Path, model_type: str, tau_channel: str, params: GNNTrainingConfig, unsculpting_mode: str, model_name: str, out_dir: str | Path, experiment, device: torch.device | None = None) -> dict` — seeds, `Normalize`, loaders (windows / class_fraction from `params`), `TauMuNet` from `get_arch_configs(params.arch, tau_channel, params.disco, registry_path=None)`, Adam / BCE / scheduler / early stopping (patience from `params`), best checkpoint kept, test evaluation. Writes into `<out_dir>/<model_name>/`: `<model_name>.pt`, `Normalization.pkl`, `arch_config.json` and the old extra files (loss root/png, disco pkl, eval pkl + plots). Returns a summary: `model_dir`, `model_name`, `best_epoch`, `best_validation_loss`, `epochs_run`, `early_stopped`, `auc_test`, `n_train`, `n_validation`, `n_test`. Reuse the functions of `Train_TauMuNet.py` (no copy of the loop).

Toy graphs for the tests: build with the 12d stage (`graphs.run`) from `test_graphs` helpers: 60 signal + 60 background events with `_rich_events`-style PF content (background masses in the sideband), into `tmp_path/graphs`. A module-scoped fixture builds them once.

- [ ] **Step 1: Failing tests** (`tests/pipeline/test_gnn_train.py`), all on CPU with `make_experiment(..., disabled=True)` and `params` = toy `gnn_training` with `epochs=3`, `batch_size=16`:
  - `test_train_writes_model_files`: the three model files + `best_epoch`/`epochs_run` in the summary; `arch_config.json` equals the `fl8_sd4` config.
  - `test_best_epoch_kept`: monkeypatch the validation-loss computation (or record the per-epoch validation losses the training returns) so that epoch 1 is the best of 3; with patience 50 (no early stop) the saved `.pt` has `epoch == 1` and its weights equal the epoch-1 weights.
  - `test_seed_reproducible_on_cpu`: two trainings, same seed → identical `model_state_dict` tensors.
- [ ] **Step 2: Run, expect FAIL** (`ImportError: train_gnn`).
- [ ] **Step 3: Implement** the refactor and the module. Keep `Train_TauMuNet.py` runnable: `python -m tau3mu_gnn.scripts.Train_TauMuNet --help` works.
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_gnn_train.py -q` → pass; `PYTHONPATH=$PWD/src python -m tau3mu_gnn.scripts.Train_TauMuNet --help` → exit 0.
- [ ] **Step 5: Commit** — `feat(gnn): importable GNN training (best epoch, seed, patience from config)`.

### Task 3: Stage `gnn_train`

**Files:**
- Create: `src/tau3mu_gnn/pipeline/training/stages/gnn_train.py`
- Test: `tests/pipeline/test_gnn_train.py` (extend)
- Modify: `docs/wrapper_roadmap.md` (12e ✅)

**Interfaces:**
- Consumes: Task 2 `train`, `make_experiment`; `stages/graphs.model_type`, `dataset_name`, `SAMPLES`; `TrainingConfig`.
- Produces:
  - `model_name(cfg: TrainingConfig) -> str` (pattern in Global Constraints).
  - `run(cfg: TrainingConfig, graphs_dir: str | Path, out_dir: str | Path, device=None, comet_disabled: bool = False) -> dict` — checks before any work that both graph files exist (`FileNotFoundError` naming the missing path); writes the model folder `<out_dir>/<model_name>/` and `<out_dir>/gnn_train_summary.json`; returns the summary + `{"summary": path}`. 12g copies `out_dir`; 12h points the analysis at `<out_dir>` / `model_name`.

- [ ] **Step 1: Failing tests:**
  - `test_stage_model_loads_in_analysis`: run the stage on the toy graphs (epochs 2, comet disabled, CPU); then `model_loading.load_model(out_dir, model_name(cfg), "GINE", "final_layer", "skimmed_dzd0_cone_size_0.5_knn_10", "W", torch.device("cpu"))` returns a model in eval mode and the normalizer keys; scoring one toy graph gives a finite number.
  - `test_stage_missing_graphs`: empty `graphs_dir` → `FileNotFoundError` naming `signal_W`.
  - `test_model_name`: `model_name(cfg) == "toy_train_W_fl_8_sd_4_GINE_delete_muons"`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the stage (docstring style of the other training stages; `log.info` with best epoch, epochs run, test AUC).
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` → all pass.
- [ ] **Step 5: Roadmap** 12e ✅ (files, model folder layout, the three decisions, plan path).
- [ ] **Step 6: Commit** — `feat(training): stage gnn_train (step 12e)`.

## Real-data check (optional, after Task 3, no SLURM)

On the graphs of the 12d real-data check (2000 signal / 136 background, scratchpad): 3 epochs on CPU (or `qgpu` if Kaito queues it), Comet disabled. Report epoch time, best epoch, and that `load_model` loads the result. Not a test; report to Kaito.
