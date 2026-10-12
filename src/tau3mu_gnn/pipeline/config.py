"""Load and validate the pipeline config.

Two YAML files are merged:
- configs/analysis_defaults_<channel>.yaml : frozen analysis settings, not meant to be edited
- the user file                            : paths, inputs, branch map, tunable limit values

All problems are collected and raised together as one ConfigError.
"""
from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import yaml

SUPPORTED_CHANNELS = ("W",)
REPO_ROOT = Path(__file__).resolve().parents[3]

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_UNRESOLVED_VAR_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


class ConfigError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("Invalid config:\n" + "\n".join(f"  - {p}" for p in self.problems))


# ---------------------------------------------------------------- value parsers
# Each parser returns the parsed value or raises ValueError with a short reason.

def _non_empty_str(v):
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"must be a non-empty string, got {v!r}")
    return v


def _name(v):
    v = _non_empty_str(v)
    if not _NAME_RE.match(v):
        raise ValueError(f"may only contain letters, digits, '_', '-', '.', got {v!r}")
    return v


def _channel(v):
    v = _non_empty_str(v)
    if v not in SUPPORTED_CHANNELS:
        raise ValueError(f"must be one of {list(SUPPORTED_CHANNELS)}, got {v!r}")
    return v


def _number(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        hint = ""
        if isinstance(v, str) and re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+", v.strip()):
            # PyYAML only reads a float with a dot in the mantissa and a signed exponent.
            mantissa, exponent = re.split("[eE]", v.strip())
            if "." not in mantissa:
                mantissa += ".0"
            mantissa = re.sub(r"^([-+]?)\.", r"\g<1>0.", mantissa)
            if exponent[0] not in "+-":
                exponent = "+" + exponent
            hint = f" (YAML reads this as text; write it as {mantissa}e{exponent})"
        raise ValueError(f"must be a number, got {v!r}{hint}")
    return float(v)


def _positive_number(v):
    x = _number(v)
    if x <= 0:
        raise ValueError(f"must be > 0, got {v!r}")
    return x


def _open_unit_interval(v):
    x = _number(v)
    if not 0 < x < 1:
        raise ValueError(f"must be between 0 and 1 (exclusive), got {v!r}")
    return x


def _bool(v):
    if not isinstance(v, bool):
        raise ValueError(f"must be true or false, got {v!r}")
    return v


def _number_list(v):
    items = v if isinstance(v, list) else [v]
    if not items:
        raise ValueError("must be a number or a non-empty list of numbers")
    numbers = tuple(_number(i) for i in items)
    duplicates = sorted({x for x in numbers if numbers.count(x) > 1})
    if duplicates:
        raise ValueError(f"has duplicate values {duplicates}")
    return numbers


def _range(v):
    if not isinstance(v, list) or len(v) != 2:
        raise ValueError(f"must be a list [low, high], got {v!r}")
    lo, hi = _number(v[0]), _number(v[1])
    if lo >= hi:
        raise ValueError(f"low must be < high, got {v!r}")
    return (lo, hi)


def _str_map(v):
    if v is None:
        return MappingProxyType({})
    if not isinstance(v, dict):
        raise ValueError(f"must be a mapping 'name: name', got {v!r}")
    for key, val in v.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(val, str) or not val.strip():
            raise ValueError(f"entry {key!r}: {val!r} must map a non-empty string to a non-empty string")
    return MappingProxyType(dict(v))


def _user_path(v):
    v = os.path.expandvars(os.path.expanduser(_non_empty_str(v)))
    unresolved = _UNRESOLVED_VAR_RE.search(v)
    if unresolved:
        raise ValueError(f"environment variable ${unresolved.group(1)} is not set (in {v!r})")
    return v


def _user_path_list(v):
    items = [v] if isinstance(v, str) else v
    if not isinstance(items, list) or not items:
        raise ValueError(f"must be a path or a non-empty list of paths, got {v!r}")
    return tuple(_user_path(i) for i in items)


def _local_path(v):
    v = _user_path(v)
    if v.startswith("root://") or v.startswith("/pnfs"):
        raise ValueError(f"must be a local disk path (SLURM cannot write logs to dCache), got {v!r}")
    return v


SLURM_JOBS = ("skim", "bdt", "gnn_signal", "gnn_background", "eval", "limit", "summary")
_SLURM_KEYS = ("partition", "mem", "time", "cpus")
_MEM_RE = re.compile(r"^\d+[KMGT]?$")
_TIME_RE = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")


def _slurm_jobs(v):
    if not isinstance(v, dict):
        raise ValueError(f"must be a mapping job -> {{{', '.join(_SLURM_KEYS)}}}, got {v!r}")
    reasons = [f"missing job '{j}'" for j in SLURM_JOBS if j not in v]
    reasons += [f"unknown job '{j}'" for j in v if j not in SLURM_JOBS]
    jobs = {}
    for name in SLURM_JOBS:
        spec = v.get(name)
        if spec is None:
            continue
        if not isinstance(spec, dict):
            reasons.append(f"job '{name}' must be a mapping, got {spec!r}")
            continue
        reasons += [f"job '{name}': missing key '{k}'" for k in _SLURM_KEYS if k not in spec]
        reasons += [f"job '{name}': unknown key '{k}'" for k in spec if k not in _SLURM_KEYS]
        if not set(_SLURM_KEYS) <= set(spec):
            continue
        partition, mem, time, cpus = (spec[k] for k in _SLURM_KEYS)
        if not isinstance(partition, str) or not partition.strip():
            reasons.append(f"job '{name}': partition must be a non-empty string, got {partition!r}")
        if not isinstance(mem, str) or not _MEM_RE.match(mem):
            reasons.append(f"job '{name}': mem must look like 64G, got {mem!r}")
        if not isinstance(time, str) or not _TIME_RE.match(time):
            reasons.append(f"job '{name}': time must look like 1-00:00:00 or 06:00:00, got {time!r}")
        if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
            reasons.append(f"job '{name}': cpus must be an integer >= 1, got {cpus!r}")
        jobs[name] = SlurmJob(partition, mem, time, cpus)
    if reasons:
        raise ValueError("; ".join(reasons))
    return MappingProxyType(jobs)


# ---------------------------------------------------------------- field tables

@dataclass(frozen=True)
class _Field:
    parse: Callable[[Any], Any]
    required: bool = True
    default: Any = None


# Keys the user file may contain.
USER_FIELDS: dict[str, _Field] = {
    "analysis_name": _Field(_name),
    "channel": _Field(_channel),
    "paths.output_base": _Field(_user_path),
    "paths.log_dir": _Field(_local_path),
    "inputs.signal": _Field(_user_path_list),
    "inputs.background": _Field(_user_path_list),
    "inputs.tree_name": _Field(_non_empty_str, required=False, default="Events"),
    "inputs.already_resonance_skimmed": _Field(_bool),
    "branch_map": _Field(_str_map, required=False, default=None),
    "limit.signal_weight": _Field(_positive_number),
    "limit.gnn_score_cut": _Field(_number_list, required=False, default=[3.9]),
}

# Keys of the frozen defaults file. Paths are relative to the repo root.
# The channel is not repeated here: the user file selects analysis_defaults_<channel>.yaml.
DEFAULTS_FIELDS: dict[str, _Field] = {
    "skim.dz_max": _Field(_positive_number),
    "skim.cos_alpha_2d_min": _Field(_number),
    "bdt.classifier": _Field(_non_empty_str),
    "gnn.model_dir": _Field(_non_empty_str),
    "gnn.model_name": _Field(_name),
    "gnn.conv_type": _Field(_non_empty_str),
    "gnn.gnn_mode": _Field(_non_empty_str),
    "gnn.unsculpting_mode": _Field(_non_empty_str),
    "gnn.graph_type": _Field(_non_empty_str),
    "gnn.bdt_preselection_cut": _Field(_open_unit_interval),
    "limit.tau_mass": _Field(_positive_number),
    "limit.mass_window": _Field(_positive_number),
    "limit.fit_range": _Field(_range),
    "limit.cl": _Field(_open_unit_interval),
    "slurm.account": _Field(_non_empty_str),
    "slurm.jobs": _Field(_slurm_jobs),
}

# Keys whose value is a mapping but must not be flattened.
_LEAF_KEYS = {"branch_map", "slurm.jobs"}


# ---------------------------------------------------------------- config objects

@dataclass(frozen=True)
class PathsConfig:
    output_base: str
    log_dir: str


@dataclass(frozen=True)
class InputsConfig:
    signal: tuple[str, ...]
    background: tuple[str, ...]
    tree_name: str
    already_resonance_skimmed: bool


@dataclass(frozen=True)
class SkimConfig:
    dz_max: float
    cos_alpha_2d_min: float


@dataclass(frozen=True)
class BDTConfig:
    classifier: Path


@dataclass(frozen=True)
class GNNConfig:
    model_dir: Path
    model_name: str
    conv_type: str
    gnn_mode: str
    unsculpting_mode: str
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
class SlurmJob:
    partition: str
    mem: str
    time: str        # D-HH:MM:SS or HH:MM:SS
    cpus: int


@dataclass(frozen=True)
class SlurmConfig:
    account: str
    jobs: Mapping[str, SlurmJob]


@dataclass(frozen=True)
class Config:
    analysis_name: str
    channel: str
    paths: PathsConfig
    inputs: InputsConfig
    branch_map: Mapping[str, str]
    skim: SkimConfig
    bdt: BDTConfig
    gnn: GNNConfig
    limit: LimitConfig
    slurm: SlurmConfig
    user_file: Path
    defaults_file: Path

    @property
    def output_dir(self) -> str:
        """Run folder: <output_base>/<analysis_name>. String join, so xrootd URLs stay intact."""
        return f"{self.paths.output_base.rstrip('/')}/{self.analysis_name}"


# ---------------------------------------------------------------- loading

class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects repeated keys instead of keeping the last value."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep)


def _read_yaml(path: Path, problems: list[str]) -> dict | None:
    try:
        text = path.read_text()
    except OSError as e:
        problems.append(f"cannot read config file {path}: {e.strerror}")
        return None
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as e:
        problems.append(f"{path}: not valid YAML: {e}")
        return None
    if data is None:
        return {}
    if not isinstance(data, dict):
        problems.append(f"{path}: top level must be a mapping (key: value), got {type(data).__name__}")
        return None
    return data


def _flatten(data: dict, fields: dict[str, _Field], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in data.items():
        full = f"{prefix}{key}"
        is_section = any(f.startswith(full + ".") for f in fields)
        if isinstance(value, dict) and full not in _LEAF_KEYS:
            flat.update(_flatten(value, fields, full + "."))
        elif value is None and is_section:
            continue  # empty section header: its keys are reported as missing
        else:
            flat[full] = value
    return flat


def _parse(flat: dict, fields: dict[str, _Field], source: str, problems: list[str],
           frozen_fields: dict[str, _Field] | None = None) -> dict[str, Any]:
    for key in sorted(flat):
        if key in fields:
            continue
        if frozen_fields and key in frozen_fields:
            problems.append(f"{source}: '{key}' is a frozen analysis setting (defaults file) "
                            f"and must not be set in the user file")
        else:
            hint = difflib.get_close_matches(key, list(fields), n=1)
            problems.append(f"{source}: unknown key '{key}'"
                            + (f" (did you mean '{hint[0]}'?)" if hint else ""))
    values = {}
    for key, field in fields.items():
        if key in flat:
            raw = flat[key]
        elif field.required:
            problems.append(f"{source}: missing required key '{key}'")
            continue
        else:
            raw = field.default
        try:
            values[key] = field.parse(raw)
        except ValueError as e:
            problems.append(f"{source}: '{key}' {e}")
    return values


def load_config(user_path: str | Path, defaults_dir: str | Path | None = None) -> Config:
    """Read the user config and the frozen defaults for its channel. Raise ConfigError on any problem."""
    user_path = Path(user_path)
    defaults_dir = Path(defaults_dir) if defaults_dir is not None else REPO_ROOT / "configs"
    repo_root = defaults_dir.parent
    problems: list[str] = []

    user_raw = _read_yaml(user_path, problems)
    if user_raw is None:
        raise ConfigError(problems)
    user = _parse(_flatten(user_raw, USER_FIELDS), USER_FIELDS, f"user config {user_path}",
                  problems, frozen_fields=DEFAULTS_FIELDS)
    if "channel" not in user:
        raise ConfigError(problems)

    defaults_path = defaults_dir / f"analysis_defaults_{user['channel']}.yaml"
    defaults_raw = _read_yaml(defaults_path, problems)
    if defaults_raw is None:
        raise ConfigError(problems)
    source = f"defaults {defaults_path}"
    defaults = _parse(_flatten(defaults_raw, DEFAULTS_FIELDS), DEFAULTS_FIELDS, source, problems)

    if {"bdt.classifier", "gnn.model_dir", "gnn.model_name"} <= defaults.keys():
        name = defaults["gnn.model_name"]
        model_dir = repo_root / defaults["gnn.model_dir"]
        for path in [repo_root / defaults["bdt.classifier"],
                     model_dir / name / f"{name}.pt",
                     model_dir / name / "Normalization.pkl",
                     model_dir / name / "arch_config.json"]:
            if not path.is_file():
                problems.append(f"frozen model file not found: {path}")

    if problems:
        raise ConfigError(problems)

    return Config(
        analysis_name=user["analysis_name"],
        channel=user["channel"],
        paths=PathsConfig(output_base=user["paths.output_base"], log_dir=user["paths.log_dir"]),
        inputs=InputsConfig(
            signal=user["inputs.signal"],
            background=user["inputs.background"],
            tree_name=user["inputs.tree_name"],
            already_resonance_skimmed=user["inputs.already_resonance_skimmed"],
        ),
        branch_map=user["branch_map"],
        skim=SkimConfig(dz_max=defaults["skim.dz_max"], cos_alpha_2d_min=defaults["skim.cos_alpha_2d_min"]),
        bdt=BDTConfig(classifier=repo_root / defaults["bdt.classifier"]),
        gnn=GNNConfig(
            model_dir=repo_root / defaults["gnn.model_dir"],
            model_name=defaults["gnn.model_name"],
            conv_type=defaults["gnn.conv_type"],
            gnn_mode=defaults["gnn.gnn_mode"],
            unsculpting_mode=defaults["gnn.unsculpting_mode"],
            graph_type=defaults["gnn.graph_type"],
            bdt_preselection_cut=defaults["gnn.bdt_preselection_cut"],
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
    )
