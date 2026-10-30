"""Load and validate the analysis config.

Two YAML files are merged:
- configs/analysis_defaults_<channel>.yaml : frozen analysis settings, not meant to be edited
- the user file                            : paths, inputs, branch map, tunable limit values, models

All problems are collected and raised together as one ConfigError.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tau3mu_gnn.pipelines.common.config import (
    REPO_ROOT, ConfigError, InputsConfig, PathsConfig, ResonanceSkimConfig, SkimConfig, SlurmConfig, _abs_or_url,
    _bool, _channel, _Field, _inputs, _local_path, _name, _non_empty_str, _number_list, _optional_local_path,
    _paths, _positive_number, _read_flat_user, _resonance_skim, _skim, _str_map, _user_path_list,
)
from tau3mu_gnn.pipelines.common.defaults import DEFAULTS_FIELDS, read_analysis_defaults

def _models(v):
    """'frozen' or the path of a training gnn_vsN folder (absolute or root://)."""
    v = _non_empty_str(v)
    return v if v == "frozen" else _abs_or_url(v)


# Keys the user file may contain.
USER_FIELDS: dict[str, _Field] = {
    "analysis_name": _Field(_name),
    "channel": _Field(_channel),
    "paths.output_base": _Field(_abs_or_url),
    "paths.log_dir": _Field(_local_path),
    "paths.conda_env": _Field(_optional_local_path, required=False, default=None),
    "inputs.signal": _Field(_user_path_list),
    "inputs.background": _Field(_user_path_list),
    "inputs.tree_name": _Field(_non_empty_str, required=False, default="Events"),
    "inputs.already_resonance_skimmed": _Field(_bool),
    "branch_map": _Field(_str_map, required=False, default=None),
    "limit.signal_weight": _Field(_positive_number),
    "limit.gnn_score_cut": _Field(_number_list, required=False, default=[3.5]),
    "models": _Field(_models, required=False, default="frozen"),
}


@dataclass(frozen=True)
class BDTConfig:
    classifier: Path


@dataclass(frozen=True)
class GNNConfig:
    model_dir: Path
    model_name: str
    conv_type: str
    gnn_mode: str
    unsculpting_mode: str | None    # None: no unsculpting
    graph_type: str
    bdt_preselection_cut: float


@dataclass(frozen=True)
class LimitConfig:
    signal_weight: float
    gnn_score_cut: tuple[float, ...]
    tau_mass: float
    mass_window: float
    fit_range: tuple[float, float]
    cl: float


@dataclass(frozen=True)
class Config:
    analysis_name: str
    channel: str
    paths: PathsConfig
    inputs: InputsConfig
    branch_map: Mapping[str, str]
    resonance_skim: ResonanceSkimConfig
    skim: SkimConfig
    bdt: BDTConfig
    gnn: GNNConfig
    limit: LimitConfig
    slurm: SlurmConfig
    user_file: Path
    defaults_file: Path
    models: str = "frozen"   # or the path of a training gnn_vsN (the models then come from its manifests)

    @property
    def output_dir(self) -> str:
        """Run folder: <output_base>/<analysis_name>. String join, so xrootd URLs stay intact."""
        return f"{self.paths.output_base.rstrip('/')}/{self.analysis_name}"

    @property
    def run_name(self) -> str:
        """Name shared with the training config (SLURM job names, log folder)."""
        return self.analysis_name


def load_config(user_path: str | Path, defaults_dir: str | Path | None = None) -> Config:
    """Read the user config and the frozen defaults for its channel. Raise ConfigError on any problem."""
    user_path = Path(user_path)
    defaults_dir = Path(defaults_dir) if defaults_dir is not None else REPO_ROOT / "configs"
    repo_root = defaults_dir.parent
    problems: list[str] = []

    user = _read_flat_user(user_path, USER_FIELDS,
                           [(DEFAULTS_FIELDS, "frozen analysis setting (defaults file)")], problems)
    if user is None or "channel" not in user:
        raise ConfigError(problems)

    read = read_analysis_defaults(defaults_dir, user["channel"], problems)
    if read is None:
        raise ConfigError(problems)
    defaults, defaults_path = read

    from tau3mu_gnn.pipelines.analysis.models import read_frozen_models, read_trained_models
    models = None
    if user.get("models", "frozen") != "frozen":
        models = read_trained_models(user["models"], user["channel"], problems)
    elif not problems:
        models = read_frozen_models(repo_root / defaults["bdt.classifier"], repo_root / defaults["gnn.model_dir"],
                                    defaults["gnn.model_name"], user["channel"], problems)

    if problems:
        raise ConfigError(problems)

    return Config(
        analysis_name=user["analysis_name"],
        channel=user["channel"],
        paths=_paths(user),
        inputs=_inputs(user),
        branch_map=user["branch_map"],
        resonance_skim=_resonance_skim(defaults),
        skim=_skim(defaults),
        bdt=BDTConfig(classifier=models.classifier),
        gnn=GNNConfig(
            model_dir=models.model_dir,
            model_name=models.model_name,
            conv_type=models.conv_type,
            gnn_mode=models.gnn_mode,
            unsculpting_mode=models.unsculpting_mode,
            graph_type=models.graph_type,
            bdt_preselection_cut=models.bdt_preselection_cut,
        ),
        limit=LimitConfig(
            signal_weight=user["limit.signal_weight"],
            gnn_score_cut=user["limit.gnn_score_cut"],
            tau_mass=defaults["limit.tau_mass"],
            mass_window=defaults["limit.mass_window"],
            fit_range=defaults["limit.fit_range"],
            cl=defaults["limit.cl"],
        ),
        slurm=SlurmConfig(account=defaults["slurm.account"], jobs=defaults["slurm.jobs"]),
        user_file=user_path.resolve(),
        defaults_file=defaults_path.resolve(),
        models=user["models"],
    )
