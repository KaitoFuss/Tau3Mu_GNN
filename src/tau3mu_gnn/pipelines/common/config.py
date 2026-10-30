"""Config building blocks shared by both pipelines: value parsers, field tables, YAML reading,
and the config sections both pipelines use (paths, inputs, resonance skim, skim, SLURM, HTCondor).

The analysis config is in pipelines/analysis/config.py, the training config in
pipelines/training/config.py. All problems are collected and raised together as one ConfigError.
"""
from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import yaml

from tau3mu_gnn.pipelines.common import storage

SUPPORTED_CHANNELS = ("W", "HF")
REPO_ROOT = Path(__file__).resolve().parents[4]

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


def _positive_int(v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"must be a whole number, got {v!r}")
    if v <= 0:
        raise ValueError(f"must be > 0, got {v!r}")
    return v


def _increasing_number_list(v):
    if not isinstance(v, list) or len(v) < 2:
        raise ValueError(f"must be a list of at least 2 numbers, got {v!r}")
    numbers = tuple(_number(i) for i in v)
    if any(a >= b for a, b in zip(numbers, numbers[1:])):
        raise ValueError(f"must be strictly increasing, got {v!r}")
    return numbers


def _linspace(v):
    if not isinstance(v, list) or len(v) != 3:
        raise ValueError(f"must be a list [start, stop, num], got {v!r}")
    start, stop = _range(v[:2])
    num = v[2]
    if isinstance(num, bool) or not isinstance(num, int) or num < 2:
        raise ValueError(f"num must be a whole number >= 2, got {num!r}")
    return (start, stop, num)


_WINDOWS = ("train", "validation", "test")


def _windows(v):
    if not isinstance(v, dict) or set(v) != set(_WINDOWS):
        raise ValueError(f"must be a mapping with exactly the keys {list(_WINDOWS)}, got {v!r}")
    windows = {name: _range(v[name]) for name in _WINDOWS}
    edges = [windows[n] for n in _WINDOWS]
    if edges[0][0] != 0 or edges[-1][1] != 1 or any(a[1] != b[0] for a, b in zip(edges, edges[1:])):
        raise ValueError(f"train, validation, test must follow each other without gap or overlap "
                         f"and cover [0, 1], got {v!r}")
    return MappingProxyType(windows)


UNSCULPTING_MODES = ("delete_muons", "none")


def _unsculpting_mode(v):
    """'delete_muons' or 'none' (-> None): the modes GNN inference (ScoreProducer) reproduces.
    permutation / split_pt_uniformly change only the training graphs, so inference would differ."""
    if v not in UNSCULPTING_MODES:
        raise ValueError(f"must be one of {list(UNSCULPTING_MODES)} (GNN inference applies only "
                         f"delete_muons; other modes would give training graphs that inference does not "
                         f"reproduce), got {v!r}")
    return None if v == "none" else v


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


def _abs_or_url(v):
    """Absolute path, or an xrootd URL storage accepts. A relative path would be resolved inside
    the job scratch dir (jobs cd there), and a mistyped URL would pass for a relative path."""
    v = _user_path(v)
    if v.startswith("/"):
        return v
    if storage.is_remote(v):
        try:
            storage.local_path(v)
            return v
        except ValueError:
            pass
    raise ValueError(f"must be an absolute path or root://host[:port]//path, got {v!r}")


def _user_path_list(v):
    items = [v] if isinstance(v, str) else v
    if not isinstance(items, list) or not items:
        raise ValueError(f"must be a path or a non-empty list of paths, got {v!r}")
    return tuple(_abs_or_url(i) for i in items)


def _local_path(v):
    v = _user_path(v)
    if v.startswith("root://") or v.startswith("/pnfs"):
        raise ValueError(f"must be a local disk path (the batch system cannot write logs to dCache), got {v!r}")
    if not v.startswith("/"):
        raise ValueError(f"must be an absolute path, got {v!r}")
    return v


def _optional_local_path(v):
    return None if v is None else _local_path(v)


_SLURM_KEYS = ("partition", "mem", "time", "cpus")
_SLURM_OPTIONAL_KEYS = ("account", "gres")
_MEM_RE = re.compile(r"^\d+[KMGT]?$")
_TIME_RE = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")


def _slurm_jobs_parser(job_names: tuple[str, ...]):
    """Parser of a SLURM job table with exactly the jobs job_names."""
    def parse(v):
        if not isinstance(v, dict):
            raise ValueError(f"must be a mapping job -> {{{', '.join(_SLURM_KEYS)}}}, got {v!r}")
        reasons = [f"missing job '{j}'" for j in job_names if j not in v]
        reasons += [f"unknown job '{j}'" for j in v if j not in job_names]
        jobs = {}
        for name in job_names:
            if name not in v:
                continue
            spec = v[name]
            if not isinstance(spec, dict):
                reasons.append(f"job '{name}' must be a mapping, got {spec!r}")
                continue
            reasons += [f"job '{name}': missing key '{k}'" for k in _SLURM_KEYS if k not in spec]
            reasons += [f"job '{name}': unknown key '{k}'" for k in spec
                        if k not in _SLURM_KEYS + _SLURM_OPTIONAL_KEYS]
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
            for key in _SLURM_OPTIONAL_KEYS:
                if key in spec and (not isinstance(spec[key], str) or not spec[key].strip()):
                    reasons.append(f"job '{name}': {key} must be a non-empty string, got {spec[key]!r}")
            jobs[name] = SlurmJob(partition, mem, time, cpus, spec.get("account"), spec.get("gres"))
        if reasons:
            raise ValueError("; ".join(reasons))
        return MappingProxyType(jobs)
    return parse


SCHEDULERS = ("htcondor", "slurm")


def _scheduler(v):
    v = _non_empty_str(v)
    if v not in SCHEDULERS:
        raise ValueError(f"must be one of {list(SCHEDULERS)}, got {v!r}")
    return v


_HTCONDOR_KEYS = ("mem", "time", "cpus")
_HTCONDOR_OPTIONAL_KEYS = ("gpus", "requirements")


def _htcondor_jobs_parser(job_names: tuple[str, ...]):
    """Parser of an HTCondor job table with exactly the jobs job_names. mem and time use the SLURM formats
    (64G, 1-00:00:00): the HTCondor layer turns them into request_memory (MB) and +MaxRuntime (s)."""
    def parse(v):
        if not isinstance(v, dict):
            raise ValueError(f"must be a mapping job -> {{{', '.join(_HTCONDOR_KEYS)}}}, got {v!r}")
        reasons = [f"missing job '{j}'" for j in job_names if j not in v]
        reasons += [f"unknown job '{j}'" for j in v if j not in job_names]
        jobs = {}
        for name in job_names:
            if name not in v:
                continue
            spec = v[name]
            if not isinstance(spec, dict):
                reasons.append(f"job '{name}' must be a mapping, got {spec!r}")
                continue
            reasons += [f"job '{name}': missing key '{k}'" for k in _HTCONDOR_KEYS if k not in spec]
            reasons += [f"job '{name}': unknown key '{k}'" for k in spec
                        if k not in _HTCONDOR_KEYS + _HTCONDOR_OPTIONAL_KEYS]
            if not set(_HTCONDOR_KEYS) <= set(spec):
                continue
            mem, time, cpus = (spec[k] for k in _HTCONDOR_KEYS)
            gpus, requirements = spec.get("gpus", 0), spec.get("requirements")
            if not isinstance(mem, str) or not _MEM_RE.match(mem):
                reasons.append(f"job '{name}': mem must look like 64G, got {mem!r}")
            if not isinstance(time, str) or not _TIME_RE.match(time):
                reasons.append(f"job '{name}': time must look like 1-00:00:00 or 06:00:00, got {time!r}")
            if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
                reasons.append(f"job '{name}': cpus must be an integer >= 1, got {cpus!r}")
            if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 0:
                reasons.append(f"job '{name}': gpus must be an integer >= 0, got {gpus!r}")
            if requirements is not None and (not isinstance(requirements, str) or not requirements.strip()):
                reasons.append(f"job '{name}': requirements must be a non-empty string, got {requirements!r}")
            jobs[name] = HTCondorJob(mem, time, cpus, gpus, requirements)
        if reasons:
            raise ValueError("; ".join(reasons))
        return MappingProxyType(jobs)
    return parse


# ---------------------------------------------------------------- field tables

@dataclass(frozen=True)
class _Field:
    parse: Callable[[Any], Any]
    required: bool = True
    default: Any = None


# Keys whose value is a mapping but must not be flattened.
_LEAF_KEYS = {"branch_map", "slurm.jobs", "htcondor.jobs"}


# ---------------------------------------------------------------- config objects

@dataclass(frozen=True)
class PathsConfig:
    output_base: str
    log_dir: str
    conda_env: str | None = None   # env the jobs activate; None: the active env of the submitting shell


@dataclass(frozen=True)
class InputsConfig:
    signal: tuple[str, ...]
    background: tuple[str, ...]
    tree_name: str
    already_resonance_skimmed: bool


@dataclass(frozen=True)
class ResonanceSkimConfig:
    vtx_chi2_max: float
    mass_range: tuple[float, float]     # GeV, on the 3-muon 4-vector mass
    veto_masses: tuple[float, ...]      # GeV, vetoed opposite-sign dimuon masses
    veto_half_width: float              # GeV
    pf_dr_max: float                    # PF candidates kept within this DeltaR of a tau candidate


@dataclass(frozen=True)
class SkimConfig:
    dz_max: float
    cos_alpha_2d_min: float


@dataclass(frozen=True)
class SlurmJob:
    partition: str
    mem: str
    time: str        # D-HH:MM:SS or HH:MM:SS
    cpus: int
    account: str | None = None   # None: slurm.account
    gres: str | None = None      # e.g. gpu:1


@dataclass(frozen=True)
class SlurmConfig:
    account: str
    jobs: Mapping[str, SlurmJob]


@dataclass(frozen=True)
class HTCondorJob:
    mem: str         # like SlurmJob.mem, e.g. 128G
    time: str        # like SlurmJob.time; becomes +MaxRuntime
    cpus: int
    gpus: int = 0
    requirements: str | None = None   # HTCondor requirements expression, e.g. for the GPU type


@dataclass(frozen=True)
class HTCondorConfig:
    jobs: Mapping[str, HTCondorJob]


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
           frozen: Sequence[tuple[Mapping[str, _Field], str]] = ()) -> dict[str, Any]:
    """frozen: (field table, label) pairs; a user key found in a table is reported with its label."""
    for key in sorted(flat):
        if key in fields:
            continue
        label = next((lbl for table, lbl in frozen if key in table), None)
        if label:
            problems.append(f"{source}: '{key}' is a {label} and must not be set in the user file")
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


def _read_flat_user(user_path: Path, fields: dict[str, _Field],
                    frozen: Sequence[tuple[Mapping[str, _Field], str]], problems: list[str]) -> dict | None:
    """Read, flatten and parse the user file. None if it cannot be read."""
    raw = _read_yaml(user_path, problems)
    if raw is None:
        return None
    return _parse(_flatten(raw, fields), fields, f"user config {user_path}", problems, frozen=frozen)


def _paths(user: dict) -> PathsConfig:
    return PathsConfig(output_base=user["paths.output_base"], log_dir=user["paths.log_dir"],
                       conda_env=user["paths.conda_env"])


def _inputs(user: dict) -> InputsConfig:
    return InputsConfig(
        signal=user["inputs.signal"],
        background=user["inputs.background"],
        tree_name=user["inputs.tree_name"],
        already_resonance_skimmed=user["inputs.already_resonance_skimmed"],
    )


def _resonance_skim(defaults: dict) -> ResonanceSkimConfig:
    return ResonanceSkimConfig(
        vtx_chi2_max=defaults["resonance_skim.vtx_chi2_max"],
        mass_range=defaults["resonance_skim.mass_range"],
        veto_masses=defaults["resonance_skim.veto_masses"],
        veto_half_width=defaults["resonance_skim.veto_half_width"],
        pf_dr_max=defaults["resonance_skim.pf_dr_max"],
    )


def _skim(defaults: dict) -> SkimConfig:
    return SkimConfig(dz_max=defaults["skim.dz_max"], cos_alpha_2d_min=defaults["skim.cos_alpha_2d_min"])


