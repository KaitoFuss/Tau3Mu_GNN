# Step 12h — Analysis key `models:`: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The analysis can use the BDT and GNN of a training run instead of the frozen models: a new optional user key `models:` (`frozen` or the path of a `gnn_vsN` training version). The model files and the model-defining settings then come from the training manifests.

**Architecture:** Resolve the key in `pipeline/config.py:load_config` into the existing `Config` fields (`bdt.classifier`, `gnn.model_dir`, `gnn.model_name`, `gnn.conv_type`, `gnn.gnn_mode`, `gnn.unsculpting_mode`, `gnn.graph_type`, `gnn.bdt_preselection_cut`). Stages, jobs and the driver keep reading those fields, so they need no change. The resolution reads the `gnn` manifest and its `bdt` upstream manifest (sibling folder) through `storage`. The analysis manifest records the source.

**Tech Stack:** Python 3.11, PyYAML, pytest. Tests: `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` (env `Tau3Mu`).

**Spec:** `docs/superpowers/specs/2026-10-23-step12-training-pipeline-design.md` sec. 7 ("How the analysis uses a training run"; Kaito: model-defining settings travel with the model). Manifests of the training (12g): `gnn` outputs `model` (`gnn_train/<model_name>/<model_name>.pt`), `normalization`, `arch_config`, ...; upstream `bdt_vsN` with output `classifier`; `settings` hold the training config sections (`graphs`, `gnn_training`, ...).

## Global Constraints

- User key `models`: optional, default `frozen`. Any other value = path of a training **`gnn_vsN`** folder (absolute path, `/pnfs/...`, or `root://host//pnfs/...`; `$USER` expanded like the other paths). It is not a frozen key and does not change the frozen defaults file.
- With a training version, the values come from the training manifests, not from `analysis_defaults_W.yaml`:
  - `bdt.classifier` = `<training>/<bdt upstream>/<bdt manifest outputs.classifier>` (bdt upstream = `gnn` manifest `upstream`, a sibling folder of the `gnn_vsN`)
  - `gnn.model_dir` = folder that holds `<model_name>/` (from `gnn` manifest `outputs.model` = `gnn_train/<model_name>/<model_name>.pt` → model_dir `<gnn_vsN>/gnn_train`, model_name = stem of the file)
  - `gnn.conv_type`, `gnn.gnn_mode` from `settings.gnn_training`; `gnn.unsculpting_mode` from `settings.graphs.unsculpting_mode`; `gnn.bdt_preselection_cut` from `settings.graphs.bdt_preselection_cut`; `gnn.graph_type` = `"skimmed" + graph/naming.graph_suffix(use_displacement, cone_size, connection, k=k, with_unsculpting=False)` from `settings.graphs`.
- Checks at load time (one `ConfigError` with all problems, as today): the folder name matches `gnn_vs<N>`; its `manifest.json` exists and has `phase == "gnn"`; the bdt upstream manifest exists and has `phase == "bdt"`; every model file exists (`model`, `normalization`, `arch_config`, `classifier`). Files are read with `storage.local_path` (the `/pnfs` path for `root://`), as everywhere else.
- `frozen`: today's behaviour unchanged (frozen model files checked, values from the defaults file).
- Analysis manifest / `config_used.yaml`: `config_dict` adds `"models"` (the user value as written, `frozen` or the path); the analysis `PHASE_SECTIONS` of all three phases include `"models"` (a model change is a setting change for every phase). The resolved `bdt` / `gnn` values are already in the snapshot.
- Analysis job scripts unchanged byte for byte with `models: frozen` (the existing baseline test must stay green).
- Use `/usr/bin/git` in the worktree. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never print the Comet key.
- Branch `pipeline-step12h-models-key` is stacked on `pipeline-step12g-training-driver` (PR #33).

## Review Focus

1. `models:` pointing at a `bdt_vsN` (or `benchmark_vsN`) folder → `ConfigError` naming the folder and saying a `gnn_vsN` is needed (Task 1, `test_models_wrong_phase`).
2. A `gnn_vsN` without `manifest.json` (training not finished) → `ConfigError`, not a crash in phase B (Task 1, `test_models_incomplete_training`).
3. A model file listed in the manifest but missing → `ConfigError` naming the file (Task 1, `test_models_missing_file`).
4. `root://` path → files resolved through `storage.local_path` (Task 1, `test_models_root_url`).
5. Two analyses, frozen vs training models → different `phase_settings` (the upstream check warns on a model change) (Task 2, `test_models_in_settings`).

---

### Task 1: Resolve `models:` in the analysis config

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/config.py` (`USER_FIELDS["models"]`, a parser, `load_config`), `configs/example_W.yaml` and `configs/example_W_raw.yaml` (commented `models:` line with an explanation)
- Create: `src/tau3mu_gnn/pipeline/models.py`
- Test: `tests/pipeline/test_models_key.py` (new)

**Interfaces:**
- Produces:
  - `models.TrainedModels` (frozen dataclass): `source: str` (the gnn_vsN path as given), `classifier: Path`, `model_dir: Path`, `model_name: str`, `conv_type: str`, `gnn_mode: str`, `unsculpting_mode: str | None`, `graph_type: str`, `bdt_preselection_cut: float`.
  - `models.read_trained_models(gnn_version: str, problems: list[str]) -> TrainedModels | None` — all checks of the Global Constraints; appends problems and returns `None` on any problem.
  - `Config.models: str` (`"frozen"` or the path as given) — new field (default for direct construction: `"frozen"`).
  - `load_config`: with a training version, `bdt` / `gnn` fields from `TrainedModels`; the frozen-model-file check runs only for `frozen`.
- [ ] **Step 1: Failing tests.** Helper `make_training(tmp_path, *, phase="gnn", complete=True, drop=None)` writes a toy training folder `<tmp>/out/toy_train/{bdt_vs1,gnn_vs1}` with `manifest.json` files (bdt: `phase: bdt`, `outputs.classifier: bdt_train/ISO_classifiers.pck`; gnn: `phase: gnn`, `upstream: bdt_vs1`, `outputs.model/normalization/arch_config` under `gnn_train/toy_train_W_fl_8_sd_4_GINE_delete_muons/`, `settings.graphs` with `bdt_preselection_cut: 0.75`, `connection: knn`, `k: 10`, `cone_size: 0.5`, `use_displacement: true`, `unsculpting_mode: delete_muons`, `settings.gnn_training` with `conv_type: GINE`, `gnn_mode: final_layer`) and the listed files (toy bytes). Tests:
  - `test_models_frozen_default`: no key → `cfg.models == "frozen"`, values as today.
  - `test_models_training_version`: `models: <gnn_vs1>` → `cfg.bdt.classifier == <bdt_vs1>/bdt_train/ISO_classifiers.pck`, `cfg.gnn.model_dir == <gnn_vs1>/gnn_train`, `model_name == "toy_train_W_fl_8_sd_4_GINE_delete_muons"`, `bdt_preselection_cut == 0.75` (not the frozen 0.8), `graph_type == "skimmed_dzd0_cone_size_0.5_knn_10"`, `unsculpting_mode == "delete_muons"`.
  - `test_models_wrong_phase`, `test_models_incomplete_training`, `test_models_missing_file` (Review Focus 1–3; message names the path).
  - `test_models_root_url`: monkeypatch `storage.local_path` to map a fake `root://host//pnfs/x/...` onto `tmp_path` → resolves; `cfg.gnn.model_dir` is the local (`/pnfs`-style) path.
  - `test_models_loadable_by_gnn_stage`: with a training version, `model_loading.load_model(cfg.gnn.model_dir, cfg.gnn.model_name, ...)` finds the three files (monkeypatch the model construction if needed; the point is the path layout).
- [ ] **Step 2: Run, expect FAIL.** **Step 3: Implement.** **Step 4: Run** `tests/pipeline -q` → green (incl. `test_analysis_script_unchanged`).
- [ ] **Step 5: Commit** — `feat(config): analysis key models: frozen or a training gnn_vsN`.

### Task 2: Record the model source; docs

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/manifest.py` (`config_dict` adds `"models": cfg.models`; `PHASE_SECTIONS` of `phase_a`, `phase_b`, `phase_c` include `"models"`), `README.md` (short subsection under the analysis config: the `models:` key, what it reads, that the settings travel with the model), `docs/wrapper_roadmap.md` (12h ✅)
- Test: `tests/pipeline/test_manifest.py` (extend), `tests/pipeline/test_models_key.py` (extend)
- [ ] **Step 1: Failing tests:** `config_used.yaml` of a frozen analysis has `models: frozen`; `test_models_in_settings`: `phase_settings(cfg_frozen, "phase_b") != phase_settings(cfg_trained, "phase_b")` and the diff names `models` and `gnn.model_dir`.
- [ ] **Step 2–4:** fail → implement → `tests/pipeline -q` green.
- [ ] **Step 5: Commit** — `feat(manifest): record the model source; README and roadmap (step 12h)`.
