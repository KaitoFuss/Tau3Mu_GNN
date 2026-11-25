"""Load and validate the training pipeline config.

Three YAML files are merged:
- configs/training_defaults_<channel>.yaml : frozen training settings, not meant to be edited
- configs/analysis_defaults_<channel>.yaml : frozen analysis settings; the training reads the
                                             shared stages from it (resonance_skim, skim, limit fit)
- the user file                            : paths, inputs, branch map, graph BDT cut

All problems are collected and raised together as one ConfigError.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from tau3mu_gnn.pipelines.common.config import (
    REPO_ROOT, ConfigError, HTCondorConfig, InputsConfig, PathsConfig, ResonanceSkimConfig, SkimConfig,
    SlurmConfig, _abs_or_url, _bool, _channel, _Field, _htcondor_jobs_parser, _increasing_number_list, _inputs,
    _LEAF_KEYS, _linspace, _local_path, _name, _non_empty_str, _number, _open_unit_interval,
    _optional_local_path, _parse, _paths, _positive_int, _positive_number, _read_yaml,
    _resonance_skim, _scheduler, _skim, _slurm_jobs_parser, _str_map, _unsculpting_mode, _user_path_list, _windows,
)
from tau3mu_gnn.pipelines.common.defaults import DEFAULTS_FIELDS, read_analysis_defaults

TRAINING_SLURM_JOBS = ("skim", "bdt_train", "bdt_infer", "npvs", "graphs_signal",   # also the HTCondor jobs
                       "graphs_background", "gnn_train")
GRAPH_CONNECTIONS = ("knn", "fully")   # r_threshold needs a delta_r setting first


# ---------------------------------------------------------------- value parsers

def _non_negative(v):
    x = _number(v)
    if x < 0:
        raise ValueError(f"must be >= 0, got {v!r}")
    return x


def _kfold(v):
    n = _positive_int(v)
    if n < 2:
        raise ValueError(f"must be >= 2, got {v!r}")
    return n


def _connection(v):
    v = _non_empty_str(v)
    if v not in GRAPH_CONNECTIONS:
        raise ValueError(f"must be one of {list(GRAPH_CONNECTIONS)}, got {v!r}")
    return v


def _class_fraction(v):
    if not isinstance(v, list) or len(v) != 2:
        raise ValueError(f"must be a list [signal, background], got {v!r}")
    sig, bkg = (_open_unit_interval(x) for x in v)
    if abs(sig + bkg - 1) > 1e-9:
        raise ValueError(f"must sum to 1, got {v!r}")
    return (sig, bkg)


# ---------------------------------------------------------------- field tables

TRAINING_USER_FIELDS: dict[str, _Field] = {
    "training_name": _Field(_name),
    "channel": _Field(_channel),
    "paths.output_base": _Field(_abs_or_url),
    "paths.log_dir": _Field(_local_path),
    "paths.conda_env": _Field(_optional_local_path, required=False, default=None),
    "scheduler": _Field(_scheduler, required=False, default="htcondor"),
    "inputs.signal": _Field(_user_path_list),
    "inputs.background": _Field(_user_path_list),
    "inputs.tree_name": _Field(_non_empty_str, required=False, default="Events"),
    "inputs.already_resonance_skimmed": _Field(_bool),
    "branch_map": _Field(_str_map, required=False, default=None),
    "graphs.bdt_preselection_cut": _Field(_open_unit_interval),
    "graphs.max_events": _Field(_positive_int, required=False, default=600000),
    "bdt_training.max_events_per_class": _Field(_positive_int, required=False, default=100000),
}

TRAINING_DEFAULTS_FIELDS: dict[str, _Field] = {
    "sideband_half_width": _Field(_positive_number),
    "bdt_training.max_depth": _Field(_positive_int),
    "bdt_training.learning_rate": _Field(_positive_number),
    "bdt_training.n_estimators": _Field(_positive_int),
    "bdt_training.early_stopping_rounds": _Field(_positive_int),
    "bdt_training.subsample": _Field(_open_unit_interval),
    "bdt_training.colsample_bytree": _Field(_open_unit_interval),
    "bdt_training.gamma": _Field(_non_negative),
    "bdt_training.reg_alpha": _Field(_non_negative),
    "bdt_training.reg_lambda": _Field(_non_negative),
    "bdt_training.min_child_weight_fraction": _Field(_positive_number),
    "bdt_training.kfold": _Field(_kfold),
    "bdt_training.test_size": _Field(_open_unit_interval),
    "bdt_training.split_seed": _Field(_positive_int),
    "bdt_training.sample_seed": _Field(_positive_int),
    "npvs.var_bins": _Field(_increasing_number_list),
    "npvs.uniform_bins": _Field(_linspace),
    "graphs.connection": _Field(_connection),
    "graphs.k": _Field(_positive_int),
    "graphs.cone_size": _Field(_positive_number),
    "graphs.use_displacement": _Field(_bool),
    "graphs.unsculpting_mode": _Field(_unsculpting_mode),
    "graphs.shuffle_seed": _Field(_positive_int),
    "gnn_training.conv_type": _Field(_non_empty_str),
    "gnn_training.gnn_mode": _Field(_non_empty_str),
    "gnn_training.arch": _Field(_non_empty_str),
    "gnn_training.learning_rate": _Field(_positive_number),
    "gnn_training.batch_size": _Field(_positive_int),
    "gnn_training.disco": _Field(_non_negative),
    "gnn_training.epochs": _Field(_positive_int),
    "gnn_training.early_stopping_patience": _Field(_positive_int),
    "gnn_training.lr_scheduler_patience": _Field(_positive_int),
    "gnn_training.class_fraction": _Field(_class_fraction),
    "gnn_training.windows": _Field(_windows),
    "gnn_training.seed": _Field(_positive_int),
    "comet.workspace": _Field(_non_empty_str),
    "comet.api_key": _Field(_non_empty_str),
    "slurm.account": _Field(_non_empty_str),
    "slurm.jobs": _Field(_slurm_jobs_parser(TRAINING_SLURM_JOBS)),
    "htcondor.jobs": _Field(_htcondor_jobs_parser(TRAINING_SLURM_JOBS)),
}

_TRAINING_LEAF_KEYS = _LEAF_KEYS | {"gnn_training.windows"}


# ---------------------------------------------------------------- config objects

@dataclass(frozen=True)
class BDTTrainingConfig:
    max_events_per_class: int
    max_depth: int
    learning_rate: float
    n_estimators: int
    early_stopping_rounds: int
    subsample: float
    colsample_bytree: float
    gamma: float
    reg_alpha: float
    reg_lambda: float
    min_child_weight_fraction: float   # min_child_weight = fraction * sum(train weights)
    kfold: int
    test_size: float
    split_seed: int
    sample_seed: int


@dataclass(frozen=True)
class NpvsConfig:
    var_bins: tuple[float, ...]               # histogram bins
    uniform_bins: tuple[float, float, int]    # np.linspace args of the output bins


@dataclass(frozen=True)
class GraphsConfig:
    bdt_preselection_cut: float   # events with bdt_ISO >= cut go into the graphs
    max_events: int               # per sample
    connection: str
    k: int
    cone_size: float
    use_displacement: bool        # "dzd0" in the name
    unsculpting_mode: str | None  # None: no unsculpting
    shuffle_seed: int             # random subset of max_events

    @property
    def graph_type(self) -> str:
        """Graph name as in the analysis gnn.graph_type, e.g. skimmed_dzd0_cone_size_0.5_knn_10."""
        from tau3mu_gnn.graph.naming import graph_suffix
        return "skimmed" + graph_suffix(self.use_displacement, self.cone_size, self.connection, k=self.k,
                                        with_unsculpting=False)


@dataclass(frozen=True)
class GNNTrainingConfig:
    conv_type: str
    gnn_mode: str
    arch: str
    learning_rate: float
    batch_size: int
    disco: float
    epochs: int
    early_stopping_patience: int
    lr_scheduler_patience: int
    class_fraction: tuple[float, float]           # signal, background
    windows: Mapping[str, tuple[float, float]]    # train / validation / test
    seed: int                                     # torch, numpy, random


@dataclass(frozen=True)
class CometConfig:
    workspace: str
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class TrainingLimitConfig:
    """The mass fit settings from analysis_defaults_<channel>.yaml (tau_mass: BDT sideband, graph mass windows)."""
    tau_mass: float
    mass_window: float
    fit_range: tuple[float, float]
    cl: float


@dataclass(frozen=True)
class TrainingConfig:
    training_name: str
    channel: str
    paths: PathsConfig
    inputs: InputsConfig
    branch_map: Mapping[str, str]
    resonance_skim: ResonanceSkimConfig
    skim: SkimConfig
    sideband_half_width: float
    bdt_training: BDTTrainingConfig
    npvs: NpvsConfig
    graphs: GraphsConfig
    gnn_training: GNNTrainingConfig
    comet: CometConfig
    limit: TrainingLimitConfig
    slurm: SlurmConfig
    htcondor: HTCondorConfig
    user_file: Path
    defaults_file: Path
    analysis_defaults_file: Path
    scheduler: str = "htcondor"   # htcondor (lxplus) or slurm (PSI T3)

    @property
    def output_dir(self) -> str:
        """Run folder: <output_base>/<training_name>. String join, so xrootd URLs stay intact."""
        return f"{self.paths.output_base.rstrip('/')}/{self.training_name}"

    @property
    def run_name(self) -> str:
        """Name shared with the analysis config (batch job names, log folder)."""
        return self.training_name


# ---------------------------------------------------------------- loading

def _flatten_training(data: dict, fields: dict[str, _Field], prefix: str = "") -> dict[str, Any]:
    """_flatten that keeps gnn_training.windows as one value."""
    flat = {}
    for key, value in data.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict) and full not in _TRAINING_LEAF_KEYS:
            flat.update(_flatten_training(value, fields, full + "."))
        elif value is None and any(f.startswith(full + ".") for f in fields):
            continue
        else:
            flat[full] = value
    return flat


def load_training_config(user_path: str | Path, defaults_dir: str | Path | None = None) -> TrainingConfig:
    """Read the training user config, the frozen training defaults and the frozen analysis defaults
    for its channel. Raise ConfigError on any problem."""
    user_path = Path(user_path)
    defaults_dir = Path(defaults_dir) if defaults_dir is not None else REPO_ROOT / "configs"
    problems: list[str] = []

    raw_user = _read_yaml(user_path, problems)
    if raw_user is None:
        raise ConfigError(problems)
    channel = raw_user.get("channel") if isinstance(raw_user.get("channel"), str) else "<channel>"
    frozen = [(TRAINING_DEFAULTS_FIELDS, f"frozen training setting (training_defaults_{channel}.yaml)"),
              (DEFAULTS_FIELDS, f"frozen analysis setting (analysis_defaults_{channel}.yaml)")]
    flat_user = _flatten_training(raw_user, TRAINING_USER_FIELDS)
    user = _parse(flat_user, TRAINING_USER_FIELDS, f"user config {user_path}", problems, frozen=frozen)
    if "channel" not in user:
        raise ConfigError(problems)

    defaults_path = defaults_dir / f"training_defaults_{user['channel']}.yaml"
    raw_defaults = _read_yaml(defaults_path, problems)
    defaults = {} if raw_defaults is None else _parse(
        _flatten_training(raw_defaults, TRAINING_DEFAULTS_FIELDS), TRAINING_DEFAULTS_FIELDS,
        f"defaults {defaults_path}", problems)
    read = read_analysis_defaults(defaults_dir, user["channel"], problems)
    if problems or read is None or raw_defaults is None:
        raise ConfigError(problems)
    analysis, analysis_path = read

    def section(prefix: str, source: dict) -> dict:
        return {k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)}

    return TrainingConfig(
        training_name=user["training_name"],
        channel=user["channel"],
        paths=_paths(user),
        inputs=_inputs(user),
        branch_map=user["branch_map"],
        resonance_skim=_resonance_skim(analysis),
        skim=_skim(analysis),
        sideband_half_width=defaults["sideband_half_width"],
        bdt_training=BDTTrainingConfig(max_events_per_class=user["bdt_training.max_events_per_class"],
                                       **section("bdt_training.", defaults)),
        npvs=NpvsConfig(**section("npvs.", defaults)),
        graphs=GraphsConfig(bdt_preselection_cut=user["graphs.bdt_preselection_cut"],
                            max_events=user["graphs.max_events"], **section("graphs.", defaults)),
        gnn_training=GNNTrainingConfig(**section("gnn_training.", defaults)),
        comet=CometConfig(**section("comet.", defaults)),
        limit=TrainingLimitConfig(tau_mass=analysis["limit.tau_mass"],
                                  mass_window=analysis["limit.mass_window"], fit_range=analysis["limit.fit_range"],
                                  cl=analysis["limit.cl"]),
        slurm=SlurmConfig(account=defaults["slurm.account"], jobs=defaults["slurm.jobs"]),
        htcondor=HTCondorConfig(jobs=defaults["htcondor.jobs"]),
        user_file=user_path.resolve(),
        defaults_file=defaults_path.resolve(),
        analysis_defaults_file=analysis_path.resolve(),
        scheduler=user["scheduler"],
    )
