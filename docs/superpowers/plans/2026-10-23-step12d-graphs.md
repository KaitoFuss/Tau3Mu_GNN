# Step 12d — Stage graphs: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A training stage that builds the GNN training graphs of one sample (signal or background) from the stage 2 output (`bdt_ISO` present), with the BDT cut at creation, in the file layout `Train_TauMuNet` reads.

**Architecture:** Keep `graph/ConeGraphProducer.py` (PyG `InMemoryDataset`) as the graph builder (Kaito, 2026-10-23: option 1). Add opt-in constructor options for the pipeline recipe; their defaults keep today's behaviour for the old scripts. Guard training/inference drift with an **equivalence test**: the same toy events built by `ConeGraphProducer` (training) and by `ScoreProducer` (inference, stage 3) must give identical node, edge and graph-level features. A thin stage `pipeline/training/stages/graphs.py` prepares the folder (`<out>/<sample>_<channel>/` with the input file linked in) and runs the producer.

**Tech Stack:** Python 3.11, torch, torch_geometric 2.5.2, awkward, uproot, numpy, pytest. Tests: `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` (conda env `Tau3Mu`).

**Spec:** `docs/superpowers/specs/2026-10-23-step12-training-pipeline-design.md` (sec. 3 `graphs_*`, sec. 4 graph rows, sec. 6 cut rule). Roadmap 12d note (npvs lookup, `PV_npvs` ≥ 100).

## Global Constraints

- Order per sample (as commit `01d680e`, rule `>=` per spec sec. 6): **BDT cut** `bdt_ISO >= graphs.bdt_preselection_cut` on the event → **sideband** (background only) → **event limit** `graphs.max_events` as a random subset (seeded) → build graphs.
- **Candidate 0 only**, one graph per event (the candidate stages 2 and 3 score). Background: the event is kept only if candidate 0 is in the sideband `abs(m0 - tau_mass) > sideband_half_width` (Kaito, option a). `tau_mass` = `cfg.limit.tau_mass`, half width = `cfg.sideband_half_width` (today hard-coded 1.78 / 0.08 in the producer).
- **npvs weights on signal graphs only**, per event via `CorrectPileupDist.signal_event_weights(PV_npvs, weight_per_bin)` (clips `PV_npvs` ≥ 100; old code raised IndexError). Background graphs: `w = 1.` (old behaviour).
- Graph settings from `cfg.graphs`: `connection` knn, `k` 10, `cone_size` 0.5, `use_displacement` true, `unsculpting_mode` delete_muons; graph-level features from `gnn_inputs.graph_level_branches(channel)` (W deltaPhi appended by the producer, as today).
- Stored per graph (what training and evaluation read): `x`, `edge_index`, `edge_attr`, `graph_level_features`, `y`, `w`, `fitted_tau_mass`, `non_permuted_tau_{mass,pT,eta,phi}`, `bdt_iso`, `benchmark_score` (save_benchmark_score=True as the old W calls).
- File layout = what `Train_TauMuNet` / `GraphDataLoader` read: `<out_dir>/<sample>_<channel>/processed/<name>.pt`, `name = f"{prefix}_{sample}_{model_type}"`, `prefix = "W_decay"` (W) / `"HF_decay"`, `model_type = f"{cfg.graphs.graph_type}_{cfg.graphs.unsculpting_mode}"` (= `skimmed_dzd0_cone_size_0.5_knn_10_delete_muons`; `get_model_parameters` needs `dzd0` in it).
- Do **not** change inference (`ScoreProducer.py`) in this step. Do not change edge features: `pair_charge` is the charge of node i, not the product — same in training and inference; noted in the roadmap only.
- Use `/usr/bin/git` in the worktree. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch `pipeline-step12d-graphs` is stacked on `pipeline-step12c-bdt-train` (PR #28).

## Review Focus

1. Signal event with `PV_npvs` = 100 or 150 → graph built, weight = last-bin weight; no IndexError (Task 2, `test_npvs_weight_clipped`).
2. Background event whose candidate 0 is in the signal window but candidate 1 in the sideband → no graph (Task 2, `test_background_candidate0_rule`).
3. Event with `bdt_ISO` exactly at the cut → kept (`>=`); `bdt_ISO = -1` (no candidate) → dropped (Task 2, `test_bdt_cut_inclusive`).
4. Fewer events than `max_events` → all used; same seed → same subset; different seed → different subset (Task 2, `test_event_limit_seeded`).
5. Training graph of an event ≠ inference graph of the same event → test fails (Task 3, `test_training_graph_equals_inference_graph`).

---

### Task 1: Frozen shuffle seed

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/training/config.py` (`TRAINING_DEFAULTS_FIELDS`, `GraphsConfig`), `configs/training_defaults_W.yaml`, `tests/pipeline/conftest.py` (`TOY_TRAINING_DEFAULTS`)
- Test: `tests/pipeline/test_training_config.py`, `tests/pipeline/test_real_training_configs.py`

**Interfaces:**
- Produces: `GraphsConfig.shuffle_seed: int` (frozen key `graphs.shuffle_seed`, parser `_positive_int`; value `1986` in the real and toy defaults; YAML comment: "random subset of max_events; the old code had no seed").

- [ ] **Step 1: Failing tests** — in `test_training_config.py`: `assert load_training(training_user).graphs.shuffle_seed == 1986`; and `training_user["graphs"]["shuffle_seed"] = 1` → `ConfigError` matching `"'graphs.shuffle_seed' is a frozen training setting"`. In `test_real_training_configs.py` add `cfg.graphs.shuffle_seed == 1986`.
- [ ] **Step 2: Run, expect FAIL** — `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_training_config.py tests/pipeline/test_real_training_configs.py -q`.
- [ ] **Step 3: Implement** (field order: put `shuffle_seed` last in `GraphsConfig`; `graph_type` unchanged).
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `feat(training): frozen graph shuffle seed`.

### Task 2: Pipeline options in `ConeGraphProducer`

**Files:**
- Modify: `src/tau3mu_gnn/graph/ConeGraphProducer.py`
- Test: `tests/pipeline/test_graphs.py` (new)

**Interfaces:**
- Consumes: `CorrectPileupDist.signal_event_weights` (12b).
- Produces: new keyword arguments of `ConeGraphProducer.__init__` (all default to today's behaviour):
  - `bdt_cut: float | None = None` — event-level cut `bdt_ISO >= bdt_cut` right after reading, before the sideband mask and the event limit; loads `bdt_ISO`, stores `bdt_iso` per graph. Independent of the old bool `bdt_preselection_cut` (leave that as is).
  - `first_candidate_only: bool = False` — one graph per event from candidate 0; with `mask=True` the event is kept only if candidate 0 is in the sideband (instead of "any candidate"); events without a candidate are skipped (as today).
  - `tau_mass: float = 1.78`, `sideband_half_width: float = 0.08` — replace the hard-coded values of the sideband mask.
  - `npvs_weights_file: str | None = None` — with `weights="npvs_rescale"`: path of the npz to read (`None`: today's `<root>/signal_<channel>_npvs_weights.npz`). In **all** modes the per-event weight uses `signal_event_weights(event["PV_npvs"], weight_per_bin)` (as a 0-d float, same type as today's value).
  - `shuffle_seed: int | None = None` — with `shuffle_bool=True`: `np.random.default_rng(shuffle_seed).choice(...)` instead of `np.random.choice` (`None` keeps the global, unseeded call).
  - `tree_name: str = "Events"` — replaces the hard-coded `"Events"` in the reader.

Toy helper in `tests/pipeline/test_graphs.py`: reuse `gnn_events(cands, bdt, pf, first_event)` from `tests/pipeline/test_gnn_scores.py` (import it; if importing a test module is awkward, move it to `conftest.py` and import it from both — one copy). Extend each event with `PV_npvs` (uint8, default 30) and the three `Selected_TauTo3Mu_iso_*` branches (default 0.1). A helper `_build(tmp_path, events_data, **kw) -> list[Data]` writes the file into a fresh dir, runs `ConeGraphProducer(root=dir, output_dataset_name="t", cone_size=0.5, edge_connection_mode="knn", k_knn=10, slimmed=True, use_displacement=True, unsculpting_mode="delete_muons", tau_channel="W", graph_level_branches=graph_level_branches("W"), save_benchmark_score=True, save_edge_attributes=True, use_relative_angles=True, first_candidate_only=True, bdt_cut=0.8, **kw)` and returns the list of graphs. Look up the real parameter names in the constructor; keep these semantics.

- [ ] **Step 1: Failing tests** (`tests/pipeline/test_graphs.py`):
  - `test_bdt_cut_inclusive`: events with `bdt` = 0.79, 0.8, 0.95, -1 (last: no candidate) → graphs for 0.8 and 0.95 only (match by `fitted_tau_mass` or distinct fitted pt).
  - `test_background_candidate0_rule`: `mask=True`, label 0; event A candidates masses [1.65, 1.78] → kept, graph from candidate 0 (`fitted_tau_mass` 1.65); event B [1.78, 1.65] → no graph; event C [1.90] → kept.
  - `test_npvs_weight_clipped`: signal, `weights="npvs_rescale"`, npz with `weight_per_bin = np.arange(100.)`; `PV_npvs` 30, 100, 150 → `w` = 30, 99, 99.
  - `test_background_weight_one`: background (`weights=None`) → every `w == 1.`.
  - `test_event_limit_seeded`: 20 passing events, `max_events_to_process=5`, `shuffle_bool=True`: seed 1 twice → same 5 events; seed 2 → a different set; `max_events_to_process=50` → 20 graphs.
  - `test_sideband_uses_config_values`: `tau_mass=1.80, sideband_half_width=0.02`, background candidate 0 at 1.75 → kept (with defaults 1.78 / 0.08 it would be dropped).
  - `test_defaults_unchanged`: without the new options, a two-candidate signal event gives two graphs (old behaviour).
- [ ] **Step 2: Run, expect FAIL** — `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_graphs.py -q` → `TypeError: unexpected keyword argument 'bdt_cut'`.
- [ ] **Step 3: Implement** the options in `ConeGraphProducer.py` (smallest change at each place: `__init__`, `get_graphs` reader + masks, `process_event` candidate loop + weight + shuffle).
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_graphs.py tests/pipeline/test_gnn_scores.py -q` → all pass.
- [ ] **Step 5: Commit** — `feat(graphs): pipeline options in ConeGraphProducer (BDT cut, candidate 0, seed, npvs file)`.

### Task 3: Training graph = inference graph (equivalence test)

**Files:**
- Test: `tests/pipeline/test_graphs.py` (extend)

**Interfaces:**
- Consumes: Task 2 `_build`; `ScoreProducer.score_events(inputs, models, outputs, npz_paths, tau_channel, unsculpting_mode, bdt_preselection_cut, tree_name)` with `models = {"rec": (RecordingGNN(), None)}` (normalizer `None` = no normalisation).

```python
class RecordingGNN(torch.nn.Module):
    """Keeps every graph it is called with; score = 0."""
    def __init__(self):
        super().__init__()
        self.graphs = []
    def forward(self, data):
        self.graphs.append(data.clone())
        return torch.zeros((1, 1))
```

- [ ] **Step 1: Write** `test_training_graph_equals_inference_graph`: one toy signal file with ≥ 4 events that pass the cut, different fitted pt per event, several PF candidates each (some inside, some outside the cone; one per muon close to it so `delete_muons` acts; enough nodes that knn k=10 matters). Build training graphs with `_build(..., weights=None)`; score the same file with `score_events(..., unsculpting_mode="delete_muons", bdt_preselection_cut=0.8)`. Match graphs by `graph_level_features[0, 0]` (fitted pt). For each pair assert: `torch.equal(x)`, `torch.equal(graph_level_features)`, and the edge sets are equal: `sorted(zip(edge_index.T.tolist(), edge_attr.tolist()))` equal (order-free). Also assert the number of graphs is equal.
- [ ] **Step 2: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_graphs.py -q -k equals` → PASS expected (both copies agree today). Then **prove the test bites**: temporarily change one node feature in `ConeGraphProducer` (e.g. drop `puppiWeight`), run → FAIL, revert, run → PASS. Note the check in the ledger.
- [ ] **Step 3: Commit** — `test(graphs): training graphs equal inference graphs`.

### Task 4: Stage `graphs`

**Files:**
- Create: `src/tau3mu_gnn/pipeline/training/stages/graphs.py`
- Test: `tests/pipeline/test_graphs.py` (extend)
- Modify: `docs/wrapper_roadmap.md` (12d ✅; note on `pair_charge`)

**Interfaces:**
- Consumes: Task 1 `GraphsConfig.shuffle_seed`; Task 2 options; `gnn_inputs.gnn_branches`, `graph_level_branches`, `expression_names`; `branches.check_stage_inputs`; `TrainingConfig` (`channel`, `inputs.tree_name`, `graphs`, `limit.tau_mass`, `sideband_half_width`).
- Produces (12e and 12g rely on these):
  - `SAMPLES = ("signal", "background")`
  - `model_type(cfg: TrainingConfig) -> str` → `f"{cfg.graphs.graph_type}_{cfg.graphs.unsculpting_mode}"`.
  - `dataset_name(cfg: TrainingConfig, sample: str) -> str` → `f"{'W_decay' if cfg.channel == 'W' else 'HF_decay'}_{sample}_{model_type(cfg)}"`.
  - `run(cfg: TrainingConfig, sample: str, bdt_file: str | Path, out_dir: str | Path, npvs_file: str | Path | None = None) -> Path` — returns `<out_dir>/<sample>_<channel>/processed/<dataset_name>.pt`. `bdt_file` = the stage 2 output of that sample. Signal: `npvs_file` required (`ValueError` if `None`), `weights="npvs_rescale"`, `mask=False`, label 1. Background: `npvs_file` must be `None` (`ValueError` otherwise: weights are for signal only), `weights=None`, `mask=True`, label 0. Before any work: `check_stage_inputs({sample: bdt_file}, tree, required, "graphs")` with `required = expression_names(gnn_branches(channel, weights=..., save_benchmark_score=True))` (contains `bdt_ISO`; `PV_npvs` for signal only). The sample folder holds a symlink `input.root` → `bdt_file` (the producer reads every `.root` in its root folder); `out_dir/<sample>_<channel>` must not exist yet (`FileExistsError`: PyG would silently reuse an old `.pt`). `shuffle_bool=True`, `shuffle_seed=cfg.graphs.shuffle_seed`, `max_events_to_process=cfg.graphs.max_events`.

- [ ] **Step 1: Failing tests:**
  - `test_stage_layout_and_names`: run signal (toy file + toy npz) and background into the same `out_dir`; assert paths `out_dir/signal_W/processed/W_decay_signal_skimmed_dzd0_cone_size_0.5_knn_10_delete_muons.pt` and the background counterpart exist; load each with `ConeGraphProducer(str(out_dir / "signal_W"), output_dataset_name=name)` (the way `GraphDataLoader` does) and check `len > 0`, `y` values, signal `w` from the npz, background `w == 1`.
  - `test_stage_signal_needs_npvs_background_refuses_it`: `ValueError` both ways.
  - `test_stage_missing_branch`: background file without `bdt_ISO` → `BranchError` naming the file; no `.pt` written.
  - `test_stage_refuses_existing_folder`: second run into the same `out_dir` → `FileExistsError`.
- [ ] **Step 2: Run, expect FAIL** — `ModuleNotFoundError ...stages.graphs`.
- [ ] **Step 3: Implement** the stage (docstring style of `training/stages/npvs.py`; `log.info` per sample with the graph count).
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` → all pass.
- [ ] **Step 5: Roadmap** — 12d ✅ (stage file, layout, names, plan path) and one line: "`pair_charge` edge feature = charge of node i, not the product (training and inference alike); changing it needs a retraining — Kaito decides later."
- [ ] **Step 6: Commit** — `feat(training): stage graphs (step 12d)`.

## Real-data check (optional, after Task 4, no SLURM)

On the small raw test outputs (`/pnfs/.../kaito_W_raw_small/phase_a_vs1/2_bdt/{signal,background}_with_bdt_scores.root`) with the npz from the 12b check: build both samples into the scratchpad with `max_events` 2000. Report: graphs per sample, run time per 1000 graphs, peak memory (`/usr/bin/time -v`), mean nodes per graph, signal weight range. Then load both with `GraphDataLoader` as `Train_TauMuNet` does (paths only, no training). Not a test; report to Kaito.
