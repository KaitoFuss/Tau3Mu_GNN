# Step 12a — Training Config: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load and validate the two training config files (`configs/training_defaults_W.yaml` frozen, `configs/example_training_W.yaml` user) into one `TrainingConfig` object, with the shared stage settings read from `configs/analysis_defaults_W.yaml`.

**Architecture:** Reuse the analysis loader machinery in `src/tau3mu_gnn/pipeline/config.py` (parsers, `_Field` tables, `_flatten`, `_parse`, `ConfigError`). Generalise two things there (SLURM job table, frozen-key message), extract the analysis-defaults read into a helper, then add `src/tau3mu_gnn/pipeline/training/config.py`. No stage, job or SLURM code changes in 12a.

**Tech Stack:** Python 3.11, PyYAML, pytest. Tests: `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` (conda env `Tau3Mu`).

**Spec:** `docs/superpowers/specs/2026-10-23-step12-training-pipeline-design.md` (sec. 4, 5, 10, 11).

## Global Constraints

- Same rules as the analysis config: frozen file not edited by users; a frozen key in the user file → error; unknown key → error; all problems collected into one `ConfigError`.
- One source of truth for shared stages: `resonance_skim.*`, `skim.*`, `limit.tau_mass / mass_window / fit_range / cl` come from `analysis_defaults_W.yaml`; not repeated in `training_defaults_W.yaml`.
- BDT features and graph-level features are **not** in any YAML: their single source stays `bdt_training/bdt_inputs.bdt_branches` and `gnn_training/gnn_inputs.graph_level_branches`.
- Comet API key: in `training_defaults_W.yaml`, plain text (Kaito, 2026-10-23). Copy the value from `src/tau3mu_gnn/scripts/Train_TauMuNet.py:709`. Do not paste it into docs or commit messages.
- Graph event limit default **600000**; BDT events per class default **100000**; sideband half width **0.08** GeV.
- SLURM memory as a power of 2.
- Use `/usr/bin/git` in the worktree. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A frozen **training** key in the user file (e.g. `gnn_training.epochs: 5`) → error that names it a frozen training setting (owned by Task 3, test `test_frozen_training_key_in_user_file`).
2. A frozen **analysis** key in the training user file (e.g. `skim.dz_max`) → error that names it a frozen analysis setting (Task 3, `test_frozen_analysis_key_in_training_user_file`).
3. Cuts at the edge: `graphs.bdt_preselection_cut: 1.0`, a benchmark cut `0` or `1.2` → error (Task 3, `test_cut_values_must_be_inside_0_1`).
4. Event limits written as `6e5`, `600000.5`, `"600k"`, `0`, `true` → error; `600000` passes (Task 1, `test_positive_int`).
5. Training data windows that overlap, leave a gap, or leave [0, 1] → error (Task 1, `test_windows`).

---

### Task 1: New value parsers and a generic SLURM job table

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/config.py` (parsers section; `SLURM_JOBS`, `_slurm_jobs`, `SlurmJob`, `DEFAULTS_FIELDS["slurm.jobs"]`)
- Test: `tests/pipeline/test_config_parsers.py`, `tests/pipeline/test_load_config.py`

**Interfaces:**
- Produces:
  - `_positive_int(v) -> int` — rejects bool, float (also `600000.0`), str, `<= 0`.
  - `_increasing_number_list(v) -> tuple[float, ...]` — at least 2 values, strictly increasing.
  - `_linspace(v) -> tuple[float, float, int]` — list `[start, stop, num]`, `start < stop`, `num` int ≥ 2.
  - `_windows(v) -> Mapping[str, tuple[float, float]]` — mapping with exactly keys `train`, `validation`, `test`, each a `_range`; in this order they must be contiguous (`train[1] == validation[0]`, `validation[1] == test[0]`) and cover `[0, 1]` (`train[0] == 0`, `test[1] == 1`).
  - `_slurm_jobs_parser(job_names: tuple[str, ...]) -> Callable[[Any], Mapping[str, SlurmJob]]` — today's `_slurm_jobs` logic with the job list as a parameter. Each job: required keys `partition, mem, time, cpus` (as today), optional keys `account` (non-empty str) and `gres` (non-empty str, e.g. `gpu:1`).
  - `SlurmJob(partition, mem, time, cpus, account: str | None = None, gres: str | None = None)`.
  - `DEFAULTS_FIELDS["slurm.jobs"] = _Field(_slurm_jobs_parser(SLURM_JOBS))` — analysis behaviour unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/pipeline/test_config_parsers.py`:

```python
@pytest.mark.parametrize("bad", [6e5, 600000.0, 600000.5, "600k", 0, -3, True, None])
def test_positive_int_rejects(bad):
    with pytest.raises(ValueError):
        c._positive_int(bad)

def test_positive_int():
    assert c._positive_int(600000) == 600000

def test_increasing_number_list():
    assert c._increasing_number_list([0, 10, 15]) == (0.0, 10.0, 15.0)
    for bad in ([5], [0, 10, 10], [10, 0], "0,10"):
        with pytest.raises(ValueError):
            c._increasing_number_list(bad)

def test_linspace():
    assert c._linspace([0, 100, 101]) == (0.0, 100.0, 101)
    for bad in ([0, 100], [100, 0, 101], [0, 100, 1], [0, 100, 10.5]):
        with pytest.raises(ValueError):
            c._linspace(bad)

def test_windows():
    ok = {"train": [0.0, 0.6], "validation": [0.6, 0.7], "test": [0.7, 1.0]}
    assert c._windows(ok)["validation"] == (0.6, 0.7)
    for bad in ({**ok, "validation": [0.5, 0.7]},          # overlap
                {**ok, "validation": [0.65, 0.7]},         # gap
                {**ok, "test": [0.7, 1.1]},                # outside [0, 1]
                {"train": [0.0, 0.6], "test": [0.6, 1.0]}, # missing key
                {**ok, "extra": [0, 1]}):                  # unknown key
        with pytest.raises(ValueError):
            c._windows(bad)

def test_slurm_jobs_parser_optional_account_and_gres():
    parse = c._slurm_jobs_parser(("a", "b"))
    jobs = parse({"a": {"partition": "gpu", "mem": "128G", "time": "3-00:00:00", "cpus": 5,
                        "account": "gpu_gres", "gres": "gpu:1"},
                  "b": {"partition": "long", "mem": "256G", "time": "2-00:00:00", "cpus": 1}})
    assert (jobs["a"].account, jobs["a"].gres) == ("gpu_gres", "gpu:1")
    assert (jobs["b"].account, jobs["b"].gres) == (None, None)
    with pytest.raises(ValueError, match="missing job 'b'"):
        parse({"a": {"partition": "x", "mem": "1G", "time": "01:00:00", "cpus": 1}})
    with pytest.raises(ValueError, match="account must be a non-empty string"):
        parse({"a": {"partition": "x", "mem": "1G", "time": "01:00:00", "cpus": 1, "account": ""},
               "b": {"partition": "x", "mem": "1G", "time": "01:00:00", "cpus": 1}})
```

(`c` = `tau3mu_gnn.pipeline.config`; use the module's existing import style.)

- [ ] **Step 2: Run, expect FAIL** — `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_config_parsers.py -q` → `AttributeError: ... has no attribute '_positive_int'`.

- [ ] **Step 3: Implement** the parsers and `_slurm_jobs_parser` in `config.py` (error texts: `"account must be a non-empty string, got ..."`, `"gres must be a non-empty string, got ..."`, same pattern as the existing `partition` message). Remove the old `_slurm_jobs`; point `DEFAULTS_FIELDS["slurm.jobs"]` at `_slurm_jobs_parser(SLURM_JOBS)`. Extend `SlurmJob` with the two optional fields (defaults `None`).

- [ ] **Step 4: Run all config tests, expect PASS** — `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_config_parsers.py tests/pipeline/test_load_config.py tests/pipeline/test_slurm.py -q` (the existing `test_slurm_jobs_validation` must still pass unchanged).

- [ ] **Step 5: Commit** — `feat(config): positive-int, bins, windows parsers; SLURM job table per job list`.

### Task 2: Shared loader helpers

Refactor only. No behaviour change for `load_config`.

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/config.py` (`_parse`, `load_config`)
- Test: `tests/pipeline/test_load_config.py` (existing, unchanged — must stay green)

**Interfaces:**
- Produces:
  - `_parse(flat, fields, source, problems, frozen: Sequence[tuple[Mapping[str, _Field], str]] = ()) -> dict` — replaces the `frozen_fields` parameter. For a key found in a frozen table the problem text is `f"{source}: '{key}' is a {label} and must not be set in the user file"`. The analysis call passes `[(DEFAULTS_FIELDS, "frozen analysis setting (defaults file)")]`, so today's text stays byte-identical (`test_load_config.py:73`).
  - `_read_flat_user(user_path: Path, fields, frozen, problems) -> dict | None` — read + flatten + parse the user file; `None` if the file cannot be read.
  - `read_analysis_defaults(defaults_dir: Path, channel: str, problems: list[str]) -> tuple[dict[str, Any], Path] | None` — read and parse `analysis_defaults_<channel>.yaml` with `DEFAULTS_FIELDS` (no model-file check: that stays in `load_config`). `None` if the file cannot be read.
  - Builders used by both loaders: `_paths(user) -> PathsConfig`, `_inputs(user) -> InputsConfig`, `_resonance_skim(defaults) -> ResonanceSkimConfig`, `_skim(defaults) -> SkimConfig`.

- [ ] **Step 1: Refactor** `load_config` onto these helpers.
- [ ] **Step 2: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` → all pass (same count as before Task 2).
- [ ] **Step 3: Commit** — `refactor(config): shared helpers for the training loader`.

### Task 3: Training config loader

**Files:**
- Create: `src/tau3mu_gnn/pipeline/training/__init__.py` (empty), `src/tau3mu_gnn/pipeline/training/config.py`
- Create: `tests/pipeline/training/__init__.py` only if the test tree needs it (check how `tests/pipeline` is collected), `tests/pipeline/test_training_config.py`
- Modify: `tests/pipeline/conftest.py` (add `TOY_TRAINING_DEFAULTS`, `TOY_TRAINING_USER`, fixture `load_training`)

**Interfaces:**
- Consumes: Task 1 parsers, `_slurm_jobs_parser`; Task 2 `_parse`, `_read_flat_user`, `read_analysis_defaults`, builders; existing `_name`, `_channel`, `_open_unit_interval`, `_number_list`, `_positive_number`, `_number`, `_bool`, `_non_empty_str`, `_str_map`, `_range`, `_user_path_list`, `_abs_or_url`, `_local_path`, `_optional_local_path`, `ConfigError`, `PathsConfig`, `InputsConfig`, `ResonanceSkimConfig`, `SkimConfig`, `SlurmConfig`, `REPO_ROOT`.
- Produces (later sub-steps rely on these names):

```python
TRAINING_SLURM_JOBS = ("skim", "bdt_train", "bdt_infer", "npvs", "graphs_signal",
                       "graphs_background", "gnn_train", "benchmark_limit", "benchmark_summary")

@dataclass(frozen=True)
class BDTTrainingConfig:
    max_events_per_class: int          # user, default 100000
    max_depth: int; learning_rate: float; n_estimators: int; early_stopping_rounds: int
    subsample: float; colsample_bytree: float; gamma: float; reg_alpha: float; reg_lambda: float
    min_child_weight_fraction: float   # min_child_weight = fraction * sum(train weights)
    kfold: int; test_size: float; split_seed: int; sample_seed: int

@dataclass(frozen=True)
class NpvsConfig:
    var_bins: tuple[float, ...]        # histogram bins
    uniform_bins: tuple[float, float, int]   # np.linspace args of the output bins

@dataclass(frozen=True)
class GraphsConfig:
    bdt_preselection_cut: float        # user, events with bdt_ISO >= cut
    max_events: int                    # user, default 600000, per sample
    connection: str                    # "knn"
    k: int
    cone_size: float
    use_displacement: bool             # "dzd0" in the name
    unsculpting_mode: str
    @property
    def graph_type(self) -> str: ...   # "skimmed_dzd0_cone_size_0.5_knn_10" for the recipe

@dataclass(frozen=True)
class GNNTrainingConfig:
    conv_type: str; gnn_mode: str; arch: str
    learning_rate: float; batch_size: int; disco: float; epochs: int
    early_stopping_patience: int; lr_scheduler_patience: int
    class_fraction: tuple[float, float]          # signal, background
    windows: Mapping[str, tuple[float, float]]   # train / validation / test

@dataclass(frozen=True)
class CometConfig:
    workspace: str
    api_key: str = field(repr=False)

@dataclass(frozen=True)
class BenchmarkConfig:
    bdt_score_cut: tuple[float, ...]   # each in (0, 1)

@dataclass(frozen=True)
class TrainingLimitConfig:
    signal_weight: float               # user
    tau_mass: float; mass_window: float; fit_range: tuple[float, float]; cl: float   # analysis defaults

@dataclass(frozen=True)
class TrainingConfig:
    training_name: str; channel: str
    paths: PathsConfig; inputs: InputsConfig; branch_map: Mapping[str, str]
    resonance_skim: ResonanceSkimConfig; skim: SkimConfig      # from analysis defaults
    sideband_half_width: float
    bdt_training: BDTTrainingConfig; npvs: NpvsConfig; graphs: GraphsConfig
    gnn_training: GNNTrainingConfig; comet: CometConfig
    benchmark: BenchmarkConfig; limit: TrainingLimitConfig; slurm: SlurmConfig
    user_file: Path; defaults_file: Path; analysis_defaults_file: Path
    @property
    def output_dir(self) -> str: ...   # f"{output_base.rstrip('/')}/{training_name}"

def load_training_config(user_path: str | Path, defaults_dir: str | Path | None = None) -> TrainingConfig: ...
```

Field tables (flat keys; parser in brackets):

- `TRAINING_USER_FIELDS`: `training_name` [`_name`], `channel` [`_channel`], `paths.output_base` [`_abs_or_url`], `paths.log_dir` [`_local_path`], `paths.conda_env` [`_optional_local_path`, optional `None`], `inputs.signal` / `inputs.background` [`_user_path_list`], `inputs.tree_name` [optional `"Events"`], `inputs.already_resonance_skimmed` [`_bool`], `branch_map` [`_str_map`, optional], `graphs.bdt_preselection_cut` [`_open_unit_interval`, required], `graphs.max_events` [`_positive_int`, optional `600000`], `bdt_training.max_events_per_class` [`_positive_int`, optional `100000`], `benchmark.bdt_score_cut` [number list, each `_open_unit_interval`, no duplicates, required], `limit.signal_weight` [`_positive_number`, required].
- `TRAINING_DEFAULTS_FIELDS`: `sideband_half_width` [`_positive_number`]; `bdt_training.{max_depth, n_estimators, early_stopping_rounds, split_seed, sample_seed}` [`_positive_int`], `bdt_training.kfold` [`_positive_int`, ≥ 2], `bdt_training.{learning_rate, min_child_weight_fraction}` [`_positive_number`], `bdt_training.{subsample, colsample_bytree, test_size}` [`_open_unit_interval`], `bdt_training.{gamma, reg_alpha, reg_lambda}` [`_number`, ≥ 0]; `npvs.var_bins` [`_increasing_number_list`], `npvs.uniform_bins` [`_linspace`]; `graphs.connection` [one of `knn`, `fully`, `r_threshold`], `graphs.k` [`_positive_int`], `graphs.cone_size` [`_positive_number`], `graphs.use_displacement` [`_bool`], `graphs.unsculpting_mode` [`_non_empty_str`]; `gnn_training.{conv_type, gnn_mode, arch}` [`_non_empty_str`], `gnn_training.{learning_rate}` [`_positive_number`], `gnn_training.disco` [`_number`, ≥ 0], `gnn_training.{batch_size, epochs, early_stopping_patience, lr_scheduler_patience}` [`_positive_int`], `gnn_training.class_fraction` [2 numbers in (0, 1) summing to 1], `gnn_training.windows` [`_windows`, leaf key]; `comet.workspace`, `comet.api_key` [`_non_empty_str`]; `slurm.account` [`_non_empty_str`], `slurm.jobs` [`_slurm_jobs_parser(TRAINING_SLURM_JOBS)`, leaf key].
- Leaf keys (not flattened): `branch_map`, `slurm.jobs`, `gnn_training.windows`.

Loading order: user file (frozen tables = training defaults with label `"frozen training setting (training_defaults_<channel>.yaml)"` and analysis `DEFAULTS_FIELDS` with label `"frozen analysis setting (analysis_defaults_<channel>.yaml)"`), then `training_defaults_<channel>.yaml`, then `read_analysis_defaults`. Collect every problem, raise one `ConfigError`. Unused analysis keys (`bdt.classifier`, `gnn.*`, `limit.gnn_score_cut` defaults, analysis `slurm`) are read but not used.

- [ ] **Step 1: Add toy configs to `conftest.py`.** `TOY_TRAINING_DEFAULTS` = the recipe values of Task 4 (real file) with SLURM jobs at `4G`, `0-01:00:00`, 1 cpu, and `gnn_train` with `account: gpu_gres`, `gres: "gpu:1"`; `comet.api_key: "toy-key"`. `TOY_TRAINING_USER` = `training_name: toy_train`, paths/inputs as `TOY_USER`, `graphs: {bdt_preselection_cut: 0.8}`, `benchmark: {bdt_score_cut: [0.98, 0.99]}`, `limit: {signal_weight: 1.9e-5}`. Fixture `load_training(user_dict=None, raw_text=None)` (a callable object with attribute `defaults`, default `TOY_TRAINING_DEFAULTS`; also a fixture `training_user` = deep copy of `TOY_TRAINING_USER`) writes `training_defaults_W.yaml` from `load_training.defaults` into `fake_repo/configs` and calls `load_training_config(path, defaults_dir=fake_repo / "configs")`.

- [ ] **Step 2: Write the failing tests** in `tests/pipeline/test_training_config.py`:

```python
def test_loads_toy(load_training, training_user):
    cfg = load_training(training_user)
    assert cfg.training_name == "toy_train"
    assert cfg.graphs.max_events == 600000 and cfg.bdt_training.max_events_per_class == 100000
    assert cfg.graphs.bdt_preselection_cut == 0.8
    assert cfg.benchmark.bdt_score_cut == (0.98, 0.99)
    assert cfg.skim.dz_max == 0.5 and cfg.resonance_skim.vtx_chi2_max == 20   # from analysis defaults
    assert (cfg.limit.tau_mass, cfg.limit.mass_window, cfg.limit.signal_weight) == (1.78, 0.06, 1.9e-5)
    assert cfg.slurm.jobs["gnn_train"].account == "gpu_gres"
    assert cfg.output_dir == "/tmp/toy_out/toy_train"

def test_graph_type_name(load_training, training_user):
    assert load_training(training_user).graphs.graph_type == "skimmed_dzd0_cone_size_0.5_knn_10"

def test_user_overrides_event_limits(load_training, training_user):
    training_user["graphs"]["max_events"] = 1000
    training_user["bdt_training"] = {"max_events_per_class": 50}
    cfg = load_training(training_user)
    assert (cfg.graphs.max_events, cfg.bdt_training.max_events_per_class) == (1000, 50)

def test_frozen_training_key_in_user_file(load_training, training_user):
    training_user["gnn_training"] = {"epochs": 5}
    with pytest.raises(ConfigError, match=r"'gnn_training.epochs' is a frozen training setting"):
        load_training(training_user)

def test_frozen_analysis_key_in_training_user_file(load_training, training_user):
    training_user["skim"] = {"dz_max": 1.0}
    with pytest.raises(ConfigError, match=r"'skim.dz_max' is a frozen analysis setting"):
        load_training(training_user)

@pytest.mark.parametrize("change", [
    lambda u: u["graphs"].update(bdt_preselection_cut=1.0),
    lambda u: u["benchmark"].update(bdt_score_cut=[0.0, 0.99]),
    lambda u: u["benchmark"].update(bdt_score_cut=[1.2]),
    lambda u: u["benchmark"].update(bdt_score_cut=[0.99, 0.99]),
])
def test_cut_values_must_be_inside_0_1(load_training, training_user, change):
    change(training_user)
    with pytest.raises(ConfigError):
        load_training(training_user)

def test_missing_required_user_keys(load_training, training_user):
    for key in ("graphs", "benchmark", "limit"):
        training_user.pop(key)
    with pytest.raises(ConfigError) as e:
        load_training(training_user)
    text = str(e.value)
    for key in ("graphs.bdt_preselection_cut", "benchmark.bdt_score_cut", "limit.signal_weight"):
        assert f"missing required key '{key}'" in text

def test_missing_analysis_defaults_is_reported(load_training, training_user, fake_repo):
    (fake_repo / "configs" / "analysis_defaults_W.yaml").unlink()
    with pytest.raises(ConfigError, match="analysis_defaults_W.yaml"):
        load_training(training_user)

def test_bad_training_defaults_all_reported(load_training, training_user, fake_repo):
    defaults = copy.deepcopy(TOY_TRAINING_DEFAULTS)
    defaults["gnn_training"]["class_fraction"] = [0.6, 0.6]
    defaults["graphs"]["connection"] = "grid"
    load_training.defaults = defaults       # fixture writes these instead of TOY_TRAINING_DEFAULTS
    with pytest.raises(ConfigError) as e:
        load_training(training_user)
    assert "'gnn_training.class_fraction'" in str(e.value) and "'graphs.connection'" in str(e.value)

def test_comet_key_not_in_repr(load_training, training_user):
    assert "toy-key" not in repr(load_training(training_user))
```

- [ ] **Step 3: Run, expect FAIL** — `PYTHONPATH=$PWD/src python -m pytest tests/pipeline/test_training_config.py -q` → `ModuleNotFoundError: tau3mu_gnn.pipeline.training`.
- [ ] **Step 4: Implement** `src/tau3mu_gnn/pipeline/training/config.py` per the Interfaces block. Module docstring in the style of `pipeline/config.py` (which files are merged, one `ConfigError`).
- [ ] **Step 5: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` → all pass.
- [ ] **Step 6: Commit** — `feat(training): training config loader (step 12a)`.

### Task 4: Real config files, roadmap

**Files:**
- Create: `configs/training_defaults_W.yaml`, `configs/example_training_W.yaml`
- Create: `tests/pipeline/test_real_training_configs.py`
- Modify: `docs/wrapper_roadmap.md` (sec. 1 scope, sec. 6 step 12 → the sub-step table of spec sec. 11 with 12a ✅, decision log entry 2026-10-23 with Kaito's decisions from the spec)

**Interfaces:**
- Consumes: `load_training_config` (Task 3), `load_config` (analysis).

`training_defaults_W.yaml` values (the recipe, spec sec. 4; comment each block with its old source file):

| Key | Value |
|---|---|
| `sideband_half_width` | `0.08` |
| `bdt_training` | `max_depth: 8`, `learning_rate: 0.005`, `n_estimators: 1000`, `early_stopping_rounds: 100`, `subsample: 0.6`, `colsample_bytree: 0.7`, `gamma: 10`, `reg_alpha: 0.6`, `reg_lambda: 1.2`, `min_child_weight_fraction: 1.0e-6`, `kfold: 6`, `test_size: 0.4`, `split_seed: 1986`, `sample_seed: 42` (`Train_XGB_W.py:238-420`) |
| `npvs` | `var_bins: [0, 10, 15, 18, 20, 21, 22, …, 60, 65, 100]` (= `[0,10,15,18,20] + linspace(21,60,40,int) + [65,100]`, write out all 47 values), `uniform_bins: [0, 100, 101]` (`CorrectPileupDist.py:39-51`) |
| `graphs` | `connection: knn`, `k: 10`, `cone_size: 0.5`, `use_displacement: true`, `unsculpting_mode: delete_muons` |
| `gnn_training` | `conv_type: GINE`, `gnn_mode: final_layer`, `arch: fl8_sd4`, `learning_rate: 0.0016`, `batch_size: 128`, `disco: 0.0`, `epochs: 1000`, `early_stopping_patience: 20`, `lr_scheduler_patience: 10`, `class_fraction: [0.5, 0.5]`, `windows: {train: [0.0, 0.6], validation: [0.6, 0.7], test: [0.7, 1.0]}` |
| `comet` | `workspace: kaitofuss`, `api_key: <value from Train_TauMuNet.py:709>` |
| `slurm.account` | `t3` |
| `slurm.jobs` | `skim` {long, 128G, 1-00:00:00, 4}; `bdt_train` {standard, 64G, 0-06:00:00, 4}; `bdt_infer` {standard, 128G, 0-06:00:00, 1}; `npvs` {standard, 32G, 0-01:00:00, 1}; `graphs_signal`, `graphs_background` {long, 256G, 2-00:00:00, 1}; `gnn_train` {gpu, 128G, 3-00:00:00, 5, account: gpu_gres, gres: "gpu:1"}; `benchmark_limit` {standard, 128G, 0-01:30:00, 1}; `benchmark_summary` {short, 4G, 0-00:30:00, 1}. Header comment: first values, Kaito raises them after the first real run. |

`example_training_W.yaml`: header comments like `example_W.yaml`; `training_name: kaito_W_train`; `paths` and `inputs` copied from `configs/example_W.yaml` (old skimmed inputs: this is the 12i check); `graphs: {bdt_preselection_cut: 0.8}` (plus commented `max_events: 600000`); commented `bdt_training: {max_events_per_class: 100000}`; `benchmark: {bdt_score_cut: [0.98, 0.985, 0.986, 0.99]}`; `limit: {signal_weight: 0.000019}` with the same comment as `example_W.yaml`.

- [ ] **Step 1: Write the failing test** `tests/pipeline/test_real_training_configs.py`:

```python
def test_example_training_w_loads_the_recipe(monkeypatch):
    monkeypatch.setenv("USER", "kfuss")
    cfg = load_training_config(REPO_ROOT / "configs" / "example_training_W.yaml")
    assert cfg.sideband_half_width == 0.08
    assert (cfg.graphs.max_events, cfg.graphs.bdt_preselection_cut) == (600000, 0.8)
    assert (cfg.bdt_training.max_depth, cfg.bdt_training.n_estimators, cfg.bdt_training.kfold) == (8, 1000, 6)
    assert (cfg.gnn_training.learning_rate, cfg.gnn_training.batch_size, cfg.gnn_training.epochs) == (0.0016, 128, 1000)
    assert cfg.gnn_training.arch == "fl8_sd4" and cfg.gnn_training.conv_type == "GINE"
    assert len(cfg.npvs.var_bins) == 47 and cfg.npvs.var_bins[-2:] == (65.0, 100.0)
    j = cfg.slurm.jobs["gnn_train"]
    assert (j.partition, j.account, j.gres) == ("gpu", "gpu_gres", "gpu:1")
    assert cfg.comet.api_key  # present, value not asserted

def test_training_graph_type_matches_analysis_defaults(monkeypatch):
    monkeypatch.setenv("USER", "kfuss")
    t = load_training_config(REPO_ROOT / "configs" / "example_training_W.yaml")
    a = load_config(REPO_ROOT / "configs" / "example_W.yaml")
    assert t.graphs.graph_type == a.gnn.graph_type
    assert t.graphs.unsculpting_mode == a.gnn.unsculpting_mode
    assert (t.gnn_training.conv_type, t.gnn_training.gnn_mode) == (a.gnn.conv_type, a.gnn.gnn_mode)
    assert t.skim == a.skim and t.resonance_skim == a.resonance_skim

def test_training_inputs_match_example_w(monkeypatch):
    monkeypatch.setenv("USER", "kfuss")
    t = load_training_config(REPO_ROOT / "configs" / "example_training_W.yaml")
    a = load_config(REPO_ROOT / "configs" / "example_W.yaml")
    assert t.inputs == a.inputs and t.paths == a.paths
```

- [ ] **Step 2: Run, expect FAIL** (files missing).
- [ ] **Step 3: Write both YAML files** per the tables above.
- [ ] **Step 4: Run** `PYTHONPATH=$PWD/src python -m pytest tests/pipeline -q` → all pass.
- [ ] **Step 5: Update the roadmap** (sec. 1, sec. 6, decision log) as listed under Files.
- [ ] **Step 6: Commit** — `feat(configs): training_defaults_W.yaml + example_training_W.yaml; roadmap step 12`.

---

## Later sub-steps (outline; each gets its own plan when it starts)

| Step | Content | Key interfaces |
|---|---|---|
| 12b | Stage `npvs`: `pipeline/training/stages/npvs.py` `run(cfg: TrainingConfig, inputs: {sample: path to 1_skim file}, out_dir) -> Path` (`npvs_weights.npz` with `weight_per_bin`, `bins`, + plots). Refactor `utils/CorrectPileupDist.create_weight_array` to take `var_bins`, `uniform_bins` as arguments. `PV_npvs` added to the training required branches. | `NpvsConfig` |
| 12c | Stage `bdt_train`: ISO mode of `Train_XGB_W.py` as a function; features from `bdt_inputs.bdt_branches("W")`; background candidates all in the sideband; writes `ISO_classifiers_<tag>.pck` that stage 2 loads unchanged. | `BDTTrainingConfig`, `sideband_half_width` |
| 12d | Stage `graphs`: one call per sample; BDT cut `bdt_ISO >= cut` → sideband (background, candidate 0) → event limit; candidate 0 only; npvs weights on signal; graph-level features from `gnn_inputs.graph_level_branches`. Writes the `.pt` file the training loads. | `GraphsConfig`, `sideband_half_width` |
| 12e | Stage `gnn_train`: `Train_TauMuNet.main` without fine-tune, settings from `GNNTrainingConfig`, Comet key from `CometConfig`; output folder `<model_name>/` with `.pt`, `Normalization.pkl`, `arch_config.json`. CPU test with Comet offline. | `GNNTrainingConfig`, `CometConfig` |
| 12f | Stage `benchmark`: `datacard_creation_ISO` as a function (selection `bdt_ISO > cut`, blind window from `limit.mass_window`), one cut per call, summary like `limit.summarize`. | `BenchmarkConfig`, `TrainingLimitConfig` |
| 12g | Training driver: phases `bdt`, `gnn`, `benchmark`; versions `bdt_vsN`, `gnn_vsN`, `benchmark_vsN`; manifests; `slurm.py` renders per-job `account` / `gres`; console script `tau3mu-train`. | `TRAINING_SLURM_JOBS`, `SlurmJob.account/gres` |
| 12h | Analysis key `models: frozen | <gnn_vsN>`; model files and model-defining settings from the training manifest; recorded in the analysis manifest. | `Config` |
| 12i | Real check (Kaito queues) on the old skimmed inputs; README "Training pipeline", "Pipeline Overview", "Quick Start". | |
