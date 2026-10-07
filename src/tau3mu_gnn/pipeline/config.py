"""Load and validate the pipeline config.

Two YAML files are merged:
- configs/analysis_defaults_<channel>.yaml : frozen analysis settings, not meant to be edited
- the user file                            : paths, inputs, branch map, tunable limit values

All problems are collected and raised together as one ConfigError.
"""
from __future__ import annotations

import os
import re
from types import MappingProxyType

SUPPORTED_CHANNELS = ("W",)

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
        if isinstance(v, str) and re.fullmatch(r"[-+]?\d+[eE][-+]?\d+", v.strip()):
            mantissa, exponent = re.split("[eE]", v.strip())
            hint = f" (YAML reads this as text; write it as {mantissa}.0e{exponent})"
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
    return tuple(_number(i) for i in items)


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
