# Mini Step 2 — Config Files + Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load and validate the two pipeline config files (frozen defaults + user file) into one read-only config object, with the frozen model files shipped in the repo.

**Architecture:** One module `src/tau3mu_gnn/pipeline/config.py`. It reads the user YAML, picks `configs/analysis_defaults_<channel>.yaml`, flattens both to dotted keys, checks each key against a field table (parser + required/default), collects **all** problems, and raises one `ConfigError` or returns nested frozen dataclasses. Frozen model files are copied byte-for-byte into `models/W/` and pinned by SHA-256 in a test.

**Tech Stack:** Python 3 (Tau3Mu conda env), PyYAML (already installed), pytest 9.1.1 (installed 2026-10-07), stdlib `dataclasses`.

**Spec:** `docs/wrapper_roadmap.md` (sections 2, 5, 5.1, 5.2, 5.3, 6 step 2, decision log).

## Global Constraints

- Python interpreter: `/work/kfuss/miniforge3/envs/Tau3Mu/bin/python`.
- Run all tests from the worktree root with `PYTHONPATH=$PWD/src` so the worktree code is imported, not the editable install in the main checkout.
- Git in this worktree: always call `/usr/bin/git` (the RTK `git` rewrite is blocked by the worktree guard). Files whose content mentions git: edit with Edit/Write, not shell heredocs.
- Validation in plain Python (dataclasses + own checks). No new dependency (no pydantic, no jsonschema).
- Supported channels now: `W` only. Code must not hardcode W elsewhere (defaults file is chosen by channel name).
- Frozen defaults file is not meant to be edited. No `overrides:` section. A frozen key in the user file → error.
- `limit.signal_weight` is required, no default, must be > 0.
- `limit.gnn_score_cut`: one number or a list; stored as a tuple; default `(3.9,)`.
- Unknown key → error. Missing required key → error. All errors reported at once.
- Not in this step: input file existence, branch existence, SLURM settings, conda path (later steps add their own keys).
- Commit messages end with: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Review Focus

1. **Empty section header** (`branch_map:` or `inputs:` with nothing under it → YAML `None`) — expect the same message as a missing key / empty map, not a crash. Test in Task 3 (`test_empty_branch_map_header_is_empty_map`).
2. **Typo in a key** (`signal_weigth`) — expect "unknown key" plus a "did you mean" hint. Test in Task 3.
3. **Unset environment variable in a path** (`$SCRATCH_DIR/out` when not set) — expect a clear error, not a literal `$SCRATCH_DIR` folder. Test in Task 2.
4. **YAML type surprises** (`signal_weight: "1e-5"` quoted → string; `already_resonance_skimmed: yes` → bool true in YAML 1.1; `true` given to a number field) — strings and bools must be rejected for numbers. Tests in Task 2.
5. **Testing the wrong checkout** (editable install points at the main repo) — a guard in `conftest.py` fails fast if `tau3mu_gnn` is not imported from this worktree. Task 1.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` (modify) | add pytest `testpaths` so old scripts in `tests/` are not collected |
| `models/W/gnn/new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2/{new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2.pt, Normalization.pkl, arch_config.json}` (create, copy) | frozen GNN |
| `models/W/bdt/ISO_classifiers_24feb2026_v0.pck` (create, copy) | frozen BDT |
| `src/tau3mu_gnn/pipeline/__init__.py` (create) | package marker |
| `src/tau3mu_gnn/pipeline/config.py` (create) | parsers, field tables, `load_config`, dataclasses |
| `configs/analysis_defaults_W.yaml` (create) | frozen W settings |
| `configs/example_W.yaml` (create) | user template with Kaito's values |
| `tests/pipeline/conftest.py` (create) | import guard + fake-repo fixtures |
| `tests/pipeline/test_model_files.py` (create) | SHA-256 pins |
| `tests/pipeline/test_config_parsers.py` (create) | unit tests of parsers |
| `tests/pipeline/test_load_config.py` (create) | `load_config` behaviour on a fake repo |
| `tests/pipeline/test_real_configs.py` (create) | the shipped config files load and match production |
| `docs/wrapper_roadmap.md` (modify) | mark step 2 done, note keys deferred to later steps |

Test command used below (call it `$PYTEST`):

```bash
PYTHONPATH=$PWD/src /work/kfuss/miniforge3/envs/Tau3Mu/bin/python -m pytest
```

---

### Task 1: Test setup + frozen model files

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/pipeline/conftest.py`, `tests/pipeline/test_model_files.py`
- Create (copies): files under `models/W/`

**Interfaces:**
- Consumes: nothing.
- Produces: `models/W/gnn/<model_name>/...`, `models/W/bdt/ISO_classifiers_24feb2026_v0.pck`; `REPO_ROOT` constant in `tests/pipeline/conftest.py` (`Path` of the worktree root).

- [ ] **Step 1: Add pytest config to `pyproject.toml`** (append at the end)

```toml

[tool.pytest.ini_options]
testpaths = ["tests/pipeline"]
```

- [ ] **Step 2: Write `tests/pipeline/conftest.py`**

```python
from pathlib import Path

import tau3mu_gnn

REPO_ROOT = Path(__file__).resolve().parents[2]

# Fail fast if the editable install (main checkout) is imported instead of this tree.
_imported_from = Path(tau3mu_gnn.__file__).resolve().parents[2]
if _imported_from != REPO_ROOT:
    raise RuntimeError(
        f"tau3mu_gnn imported from {_imported_from}, expected {REPO_ROOT}. "
        f"Run with PYTHONPATH={REPO_ROOT / 'src'}"
    )
```

- [ ] **Step 3: Write the failing test `tests/pipeline/test_model_files.py`**

```python
import hashlib

import pytest

from conftest import REPO_ROOT

MODEL = "new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2"

# SHA-256 of the originals in /work/kfuss/SVJ_GNN/output/, taken 2026-10-07.
EXPECTED = {
    f"models/W/gnn/{MODEL}/{MODEL}.pt": "b6991427c7e07cf01b8e5384dc6704bb151bbff3f94093ef4babecd8f2de3c1e",
    f"models/W/gnn/{MODEL}/Normalization.pkl": "ffcf2239032b68ab222669dfb7010d3ad17511bdf2d5acb41669acd64395d57e",
    f"models/W/gnn/{MODEL}/arch_config.json": "bf5e2e1762b6a577853dad4150b1ae3e5f41efab0f0ba65ef8da5775e6a081bd",
    "models/W/bdt/ISO_classifiers_24feb2026_v0.pck": "2aa4de47344a372ca0575c017fe48b8c1f222cc50dad353aeb459d0578003395",
}


@pytest.mark.parametrize("rel_path, sha", EXPECTED.items())
def test_frozen_model_file_is_exact_copy(rel_path, sha):
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"missing frozen model file: {rel_path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
```

- [ ] **Step 4: Run it, expect FAIL**

Run: `$PYTEST tests/pipeline/test_model_files.py -v`
Expected: 4 FAILED with `missing frozen model file`.

- [ ] **Step 5: Copy the files**

```bash
SRC=/work/kfuss/SVJ_GNN/output
M=new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2
mkdir -p models/W/gnn/$M models/W/bdt
cp $SRC/$M/$M.pt $SRC/$M/Normalization.pkl $SRC/$M/arch_config.json models/W/gnn/$M/
cp $SRC/benchmark_W_BDT_vs1/ISO_classifiers_24feb2026_v0.pck models/W/bdt/
```

- [ ] **Step 6: Run it, expect PASS**

Run: `$PYTEST tests/pipeline/test_model_files.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
/usr/bin/git add pyproject.toml tests/pipeline/conftest.py tests/pipeline/test_model_files.py models/W
/usr/bin/git commit -m "feat(pipeline): ship frozen W model files and pytest setup

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Value parsers

**Files:**
- Create: `src/tau3mu_gnn/pipeline/__init__.py` (empty), `src/tau3mu_gnn/pipeline/config.py`
- Test: `tests/pipeline/test_config_parsers.py`

**Interfaces:**
- Consumes: nothing.
- Produces (in `tau3mu_gnn.pipeline.config`): `ConfigError(problems: list[str])` with attribute `.problems`; `SUPPORTED_CHANNELS = ("W",)`; parsers, each `(value) -> parsed` and raising `ValueError(message)` on bad input: `_non_empty_str`, `_name`, `_channel`, `_number`, `_positive_number`, `_open_unit_interval`, `_bool`, `_number_list -> tuple[float, ...]`, `_range -> tuple[float, float]`, `_str_map -> MappingProxyType[str, str]`, `_user_path -> str`, `_user_path_list -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests `tests/pipeline/test_config_parsers.py`**

```python
from types import MappingProxyType

import pytest

from tau3mu_gnn.pipeline import config as c


def test_config_error_lists_all_problems():
    err = c.ConfigError(["a is bad", "b is bad"])
    assert err.problems == ["a is bad", "b is bad"]
    assert "a is bad" in str(err) and "b is bad" in str(err)


@pytest.mark.parametrize("bad", ["", "   ", None, 3])
def test_non_empty_str_rejects(bad):
    with pytest.raises(ValueError):
        c._non_empty_str(bad)


def test_name_rejects_slash_and_space():
    assert c._name("kaito_W-ref.v1") == "kaito_W-ref.v1"
    for bad in ["a/b", "a b"]:
        with pytest.raises(ValueError, match="letters, digits"):
            c._name(bad)


def test_channel_only_w():
    assert c._channel("W") == "W"
    with pytest.raises(ValueError, match=r"\['W'\]"):
        c._channel("HF")


@pytest.mark.parametrize("bad", [True, False, "1e-5", None, [1]])
def test_number_rejects_non_numbers(bad):
    with pytest.raises(ValueError, match="must be a number"):
        c._number(bad)


def test_number_hint_for_yaml_exponent_text():
    # PyYAML reads `1e-5` (no dot) as the string "1e-5"
    with pytest.raises(ValueError, match=r"write it as 1\.0e-5"):
        c._number("1e-5")


def test_number_accepts_int_and_float():
    assert c._number(3) == 3.0
    assert c._number(1.9e-5) == 1.9e-5


@pytest.mark.parametrize("bad", [0, -1.0])
def test_positive_number_rejects(bad):
    with pytest.raises(ValueError, match="> 0"):
        c._positive_number(bad)


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.1])
def test_open_unit_interval_rejects(bad):
    with pytest.raises(ValueError, match="between 0 and 1"):
        c._open_unit_interval(bad)


def test_bool_rejects_strings_and_numbers():
    assert c._bool(True) is True
    for bad in ["true", 1, None]:
        with pytest.raises(ValueError, match="true or false"):
            c._bool(bad)


def test_number_list_scalar_and_list():
    assert c._number_list(3.9) == (3.9,)
    assert c._number_list([3.5, 4]) == (3.5, 4.0)
    with pytest.raises(ValueError):
        c._number_list([])
    with pytest.raises(ValueError):
        c._number_list([3.5, "x"])


def test_range():
    assert c._range([1.62, 1.98]) == (1.62, 1.98)
    for bad in [[1.98, 1.62], [1.0], "1,2"]:
        with pytest.raises(ValueError):
            c._range(bad)


def test_str_map():
    m = c._str_map({"PFCands_pt": "PFCand_pt"})
    assert isinstance(m, MappingProxyType)
    assert dict(m) == {"PFCands_pt": "PFCand_pt"}
    assert dict(c._str_map(None)) == {}
    with pytest.raises(ValueError, match="PFCands_pt"):
        c._str_map({"PFCands_pt": 3})
    with pytest.raises(ValueError):
        c._str_map(["a", "b"])


def test_user_path_expands_home_and_vars(monkeypatch):
    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("TAU3MU_TEST_DIR", "/data/x")
    assert c._user_path("~/out") == "/home/tester/out"
    assert c._user_path("$TAU3MU_TEST_DIR/f.root") == "/data/x/f.root"


def test_user_path_unset_var_is_error(monkeypatch):
    monkeypatch.delenv("TAU3MU_UNSET_VAR", raising=False)
    with pytest.raises(ValueError, match="TAU3MU_UNSET_VAR"):
        c._user_path("$TAU3MU_UNSET_VAR/out")


def test_user_path_keeps_xrootd_url():
    url = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/x"
    assert c._user_path(url) == url


def test_user_path_list_single_string_and_list():
    assert c._user_path_list("/a.root") == ("/a.root",)
    assert c._user_path_list(["/a.root", "/b*.root"]) == ("/a.root", "/b*.root")
    with pytest.raises(ValueError):
        c._user_path_list([])
```

- [ ] **Step 2: Run, expect FAIL**

Run: `$PYTEST tests/pipeline/test_config_parsers.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'tau3mu_gnn.pipeline'`.

- [ ] **Step 3: Create `src/tau3mu_gnn/pipeline/__init__.py` (empty) and `src/tau3mu_gnn/pipeline/config.py`**

```python
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
```

- [ ] **Step 4: Run, expect PASS**

Run: `$PYTEST tests/pipeline/test_config_parsers.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/__init__.py src/tau3mu_gnn/pipeline/config.py tests/pipeline/test_config_parsers.py
/usr/bin/git commit -m "feat(pipeline): config value parsers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `load_config` (merge, key checks, model-file checks, dataclasses)

**Files:**
- Modify: `src/tau3mu_gnn/pipeline/config.py` (append)
- Modify: `tests/pipeline/conftest.py` (append fixtures)
- Test: `tests/pipeline/test_load_config.py`

**Interfaces:**
- Consumes: parsers and `ConfigError` from Task 2.
- Produces:
  - `REPO_ROOT: Path` (repo root, derived from the module location).
  - `load_config(user_path: str | Path, defaults_dir: str | Path | None = None) -> Config`. `defaults_dir` defaults to `REPO_ROOT / "configs"`; repo-relative paths in the defaults file resolve against `Path(defaults_dir).parent`.
  - Frozen dataclasses:
    - `Config(analysis_name: str, channel: str, paths: PathsConfig, inputs: InputsConfig, branch_map: Mapping[str, str], skim: SkimConfig, bdt: BDTConfig, gnn: GNNConfig, limit: LimitConfig, user_file: Path, defaults_file: Path)`
    - `PathsConfig(output_dir: str)`
    - `InputsConfig(signal: tuple[str, ...], background: tuple[str, ...], tree_name: str, already_resonance_skimmed: bool)`
    - `SkimConfig(dz_max: float, cos_alpha_2d_min: float)`
    - `BDTConfig(classifier: Path)`  (absolute)
    - `GNNConfig(model_dir: Path, model_name: str, conv_type: str, gnn_mode: str, unsculpting_mode: str, graph_type: str, bdt_preselection_cut: float)`  (`model_dir` absolute; model files at `model_dir / model_name / ...`)
    - `LimitConfig(signal_weight: float, gnn_score_cut: tuple[float, ...], tau_mass: float, mass_window: float, fit_range: tuple[float, float], cl: float)`

- [ ] **Step 1: Add fixtures to `tests/pipeline/conftest.py`**

Add these imports at the top of the file (next to the existing ones):

```python
import copy

import pytest
import yaml
```

Append at the end of the file:

```python
TOY_MODEL = "toy_model"

TOY_DEFAULTS = {
    "channel": "W",
    "skim": {"dz_max": 0.5, "cos_alpha_2d_min": 0.5},
    "bdt": {"classifier": "models/W/bdt/toy.pck"},
    "gnn": {
        "model_dir": "models/W/gnn",
        "model_name": TOY_MODEL,
        "conv_type": "GINE",
        "gnn_mode": "final_layer",
        "unsculpting_mode": "delete_muons",
        "graph_type": "skimmed_dzd0_cone_size_0.5_knn_10",
        "bdt_preselection_cut": 0.8,
    },
    "limit": {"tau_mass": 1.78, "mass_window": 0.06, "fit_range": [1.62, 1.98], "cl": 0.9},
}

TOY_USER = {
    "analysis_name": "toy_run",
    "channel": "W",
    "paths": {"output_dir": "/tmp/toy_out"},
    "inputs": {
        "signal": ["/data/sig.root"],
        "background": ["/data/bkg_*.root"],
        "already_resonance_skimmed": True,
    },
    "limit": {"signal_weight": 1.9e-5},
}


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


@pytest.fixture
def fake_repo(tmp_path):
    write_yaml(tmp_path / "configs" / "analysis_defaults_W.yaml", TOY_DEFAULTS)
    for rel in [
        "models/W/bdt/toy.pck",
        f"models/W/gnn/{TOY_MODEL}/{TOY_MODEL}.pt",
        f"models/W/gnn/{TOY_MODEL}/Normalization.pkl",
        f"models/W/gnn/{TOY_MODEL}/arch_config.json",
    ]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return tmp_path


@pytest.fixture
def user():
    return copy.deepcopy(TOY_USER)


@pytest.fixture
def load(fake_repo):
    from tau3mu_gnn.pipeline.config import load_config

    def _load(user_dict=None, raw_text=None):
        path = fake_repo / "user.yaml"
        if raw_text is not None:
            path.write_text(raw_text)
        else:
            write_yaml(path, user_dict)
        return load_config(path, defaults_dir=fake_repo / "configs")

    return _load
```

- [ ] **Step 2: Write the failing tests `tests/pipeline/test_load_config.py`**

```python
import dataclasses

import pytest

from conftest import TOY_MODEL
from tau3mu_gnn.pipeline.config import ConfigError, load_config


def problems_of(excinfo):
    return "\n".join(excinfo.value.problems)


def test_valid_config_loads(load, user, fake_repo):
    cfg = load(user)
    assert cfg.analysis_name == "toy_run"
    assert cfg.channel == "W"
    assert cfg.paths.output_dir == "/tmp/toy_out"
    assert cfg.inputs.signal == ("/data/sig.root",)
    assert cfg.inputs.background == ("/data/bkg_*.root",)
    assert cfg.inputs.already_resonance_skimmed is True
    assert cfg.limit.signal_weight == 1.9e-5
    assert cfg.limit.fit_range == (1.62, 1.98)
    assert cfg.gnn.model_name == TOY_MODEL
    assert cfg.gnn.model_dir == fake_repo / "models/W/gnn"
    assert cfg.bdt.classifier == fake_repo / "models/W/bdt/toy.pck"
    assert cfg.defaults_file == fake_repo / "configs/analysis_defaults_W.yaml"


def test_defaults_applied(load, user):
    cfg = load(user)
    assert cfg.inputs.tree_name == "Events"
    assert cfg.limit.gnn_score_cut == (3.9,)
    assert dict(cfg.branch_map) == {}


def test_score_cut_scalar_and_list(load, user):
    user["limit"]["gnn_score_cut"] = 3.75
    assert load(user).limit.gnn_score_cut == (3.75,)
    user["limit"]["gnn_score_cut"] = [3.5, 3.9]
    assert load(user).limit.gnn_score_cut == (3.5, 3.9)


def test_config_is_read_only(load, user):
    user["branch_map"] = {"PFCands_pt": "PFCand_pt"}
    cfg = load(user)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.limit.signal_weight = 1.0
    with pytest.raises(TypeError):
        cfg.branch_map["x"] = "y"


def test_missing_required_key(load, user):
    del user["limit"]["signal_weight"]
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "missing required key 'limit.signal_weight'" in problems_of(e)


def test_unknown_key_with_hint(load, user):
    user["limit"]["signal_weigth"] = user["limit"].pop("signal_weight")
    with pytest.raises(ConfigError) as e:
        load(user)
    text = problems_of(e)
    assert "unknown key 'limit.signal_weigth'" in text
    assert "did you mean 'limit.signal_weight'" in text


def test_frozen_key_in_user_file(load, user):
    user["gnn"] = {"model_name": "other_model"}
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "'gnn.model_name' is a frozen analysis setting" in problems_of(e)


def test_all_problems_reported_at_once(load, user):
    del user["analysis_name"]
    user["limit"]["signal_weight"] = 0
    user["inputs"]["already_resonance_skimmed"] = "yes please"
    with pytest.raises(ConfigError) as e:
        load(user)
    assert len(e.value.problems) == 3


def test_unsupported_channel_stops_before_defaults(load, user):
    user["channel"] = "HF"
    with pytest.raises(ConfigError) as e:
        load(user)
    text = problems_of(e)
    assert "must be one of ['W']" in text
    assert "analysis_defaults_HF" not in text


def test_empty_branch_map_header_is_empty_map(load):
    raw = """
analysis_name: toy_run
channel: W
paths:
  output_dir: /tmp/toy_out
inputs:
  signal: [/data/sig.root]
  background: [/data/bkg.root]
  already_resonance_skimmed: true
branch_map:
limit:
  signal_weight: 1.9e-5
"""
    assert dict(load(raw_text=raw).branch_map) == {}


def test_empty_section_header_reports_missing_keys(load):
    raw = """
analysis_name: toy_run
channel: W
paths:
inputs:
  signal: [/data/sig.root]
  background: [/data/bkg.root]
  already_resonance_skimmed: true
limit:
  signal_weight: 1.9e-5
"""
    with pytest.raises(ConfigError) as e:
        load(raw_text=raw)
    assert "missing required key 'paths.output_dir'" in problems_of(e)
    assert "unknown key" not in problems_of(e)


def test_missing_user_file(fake_repo):
    with pytest.raises(ConfigError) as e:
        load_config(fake_repo / "nope.yaml", defaults_dir=fake_repo / "configs")
    assert "cannot read config file" in problems_of(e)


def test_invalid_yaml(load):
    with pytest.raises(ConfigError) as e:
        load(raw_text="analysis_name: [unclosed\n")
    assert "not valid YAML" in problems_of(e)


def test_top_level_must_be_mapping(load):
    with pytest.raises(ConfigError) as e:
        load(raw_text="- a\n- b\n")
    assert "top level must be a mapping" in problems_of(e)


def test_missing_frozen_model_file(load, user, fake_repo):
    (fake_repo / f"models/W/gnn/{TOY_MODEL}/Normalization.pkl").unlink()
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "frozen model file not found" in problems_of(e)
    assert "Normalization.pkl" in problems_of(e)


def test_defaults_channel_must_match(load, user, fake_repo):
    path = fake_repo / "configs/analysis_defaults_W.yaml"
    path.write_text(path.read_text().replace("channel: W", "channel: HF"))
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "channel" in problems_of(e)
```

- [ ] **Step 3: Run, expect FAIL**

Run: `$PYTEST tests/pipeline/test_load_config.py -v`
Expected: FAIL / ImportError `cannot import name 'load_config'`.

- [ ] **Step 4: Append to `src/tau3mu_gnn/pipeline/config.py`**

Add to the imports at the top of the file:

```python
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
```

Add below `SUPPORTED_CHANNELS`:

```python
REPO_ROOT = Path(__file__).resolve().parents[3]
```

Append at the end of the file:

```python
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
    "paths.output_dir": _Field(_user_path),
    "inputs.signal": _Field(_user_path_list),
    "inputs.background": _Field(_user_path_list),
    "inputs.tree_name": _Field(_non_empty_str, required=False, default="Events"),
    "inputs.already_resonance_skimmed": _Field(_bool),
    "branch_map": _Field(_str_map, required=False, default=None),
    "limit.signal_weight": _Field(_positive_number),
    "limit.gnn_score_cut": _Field(_number_list, required=False, default=[3.9]),
}

# Keys of the frozen defaults file. Paths are relative to the repo root.
DEFAULTS_FIELDS: dict[str, _Field] = {
    "channel": _Field(_channel),
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
}

# Keys whose value is a mapping but must not be flattened.
_LEAF_KEYS = {"branch_map"}


# ---------------------------------------------------------------- config objects

@dataclass(frozen=True)
class PathsConfig:
    output_dir: str


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
    user_file: Path
    defaults_file: Path


# ---------------------------------------------------------------- loading

def _read_yaml(path: Path, problems: list[str]) -> dict | None:
    try:
        text = path.read_text()
    except OSError as e:
        problems.append(f"cannot read config file {path}: {e.strerror}")
        return None
    try:
        data = yaml.safe_load(text)
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
    if "channel" in defaults and defaults["channel"] != user["channel"]:
        problems.append(f"{source}: 'channel' is {defaults['channel']!r} but the user config asks for {user['channel']!r}")

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
        paths=PathsConfig(output_dir=user["paths.output_dir"]),
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
        user_file=user_path.resolve(),
        defaults_file=defaults_path.resolve(),
    )
```

- [ ] **Step 5: Run, expect PASS**

Run: `$PYTEST tests/pipeline -v`
Expected: all passed (parsers + load_config + model files).

- [ ] **Step 6: Commit**

```bash
/usr/bin/git add src/tau3mu_gnn/pipeline/config.py tests/pipeline/conftest.py tests/pipeline/test_load_config.py
/usr/bin/git commit -m "feat(pipeline): load_config merges user file with frozen defaults

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Real config files + roadmap update

**Files:**
- Create: `configs/analysis_defaults_W.yaml`, `configs/example_W.yaml`
- Test: `tests/pipeline/test_real_configs.py`
- Modify: `docs/wrapper_roadmap.md`

**Interfaces:**
- Consumes: `load_config`, `REPO_ROOT` from Task 3.
- Produces: the two config files later steps read.

- [ ] **Step 1: Write the failing test `tests/pipeline/test_real_configs.py`**

```python
from tau3mu_gnn.pipeline.config import REPO_ROOT, load_config


def test_example_w_matches_production_settings():
    cfg = load_config(REPO_ROOT / "configs" / "example_W.yaml")
    assert cfg.channel == "W"
    assert cfg.gnn.model_name == "new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2"
    assert cfg.gnn.conv_type == "GINE"
    assert cfg.gnn.gnn_mode == "final_layer"
    assert cfg.gnn.unsculpting_mode == "delete_muons"
    assert cfg.gnn.graph_type == "skimmed_dzd0_cone_size_0.5_knn_10"
    assert cfg.gnn.bdt_preselection_cut == 0.8
    assert cfg.bdt.classifier == REPO_ROOT / "models/W/bdt/ISO_classifiers_24feb2026_v0.pck"
    assert cfg.skim.dz_max == 0.5
    assert cfg.skim.cos_alpha_2d_min == 0.5
    assert cfg.limit.gnn_score_cut == (3.9,)
    assert cfg.limit.signal_weight == 0.000019
    assert cfg.limit.tau_mass == 1.78
    assert cfg.limit.mass_window == 0.060
    assert cfg.limit.fit_range == (1.62, 1.98)
    assert cfg.limit.cl == 0.90
```

- [ ] **Step 2: Run, expect FAIL**

Run: `$PYTEST tests/pipeline/test_real_configs.py -v`
Expected: FAIL with `ConfigError ... cannot read config file .../configs/example_W.yaml`.

- [ ] **Step 3: Create `configs/analysis_defaults_W.yaml`**

```yaml
# Frozen analysis settings for the W channel.
# NOT meant to be edited. Put your own settings in your user config (see example_W.yaml).
# Paths are relative to the repository root.

channel: W

skim:                          # stage 1, values from tau3mu_gnn/utils/skimmer.py
  dz_max: 0.5                  # |dZ(mu_i, mu_j)| < dz_max for all three pairs
  cos_alpha_2d_min: 0.5        # CosAlpha2D_LxyP3mu > cos_alpha_2d_min

bdt:                           # stage 2, ISO BDT only
  classifier: models/W/bdt/ISO_classifiers_24feb2026_v0.pck

gnn:                           # stage 3, fixed model
  model_dir: models/W/gnn      # files in <model_dir>/<model_name>/
  model_name: new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2
  conv_type: GINE
  gnn_mode: final_layer
  unsculpting_mode: delete_muons
  graph_type: skimmed_dzd0_cone_size_0.5_knn_10
  bdt_preselection_cut: 0.8    # run the GNN only on events with BDT ISO score > this

limit:                         # stage 5, mass fit + Combine
  tau_mass: 1.78               # GeV
  mass_window: 0.060           # GeV, half-width of the blinded signal window
  fit_range: [1.62, 1.98]      # GeV
  cl: 0.90                     # confidence level of the expected limit
```

- [ ] **Step 4: Create `configs/example_W.yaml`**

```yaml
# Example user config (W channel). Copy this file and edit it.
# Frozen analysis settings live in analysis_defaults_W.yaml and must NOT be set here.

analysis_name: kaito_W_reference           # letters, digits, _ - . only; used as folder name

channel: W

paths:
  # All outputs go here. Jobs write to /scratch first and then xrdcp to this location.
  output_dir: root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/$USER/tau3mu_pipeline/kaito_W_reference

inputs:
  # Globs are allowed. A single path is also fine.
  signal:
    - /work/kfuss/SVJ_GNN/src/tau3mu_gnn/data/signal_W/signal_skimmed_no_resonances.root
  background:
    - /pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/data_skimmed_no_resonances.root
  tree_name: Events                        # optional, default: Events
  already_resonance_skimmed: true          # true: skip stage 0 (resonance skim)

# Map from the analysis branch names (left) to the names in YOUR ntuples (right).
# Only list branches whose name differs. Leave empty if your names are the same.
branch_map: {}
#   Slimmed_TauTo3Mu_fitted_pt: Tau3Mu_fit_pt

limit:
  # REQUIRED. Weight that scales the signal MC to your dataset (lumi, cross section, reference BR).
  # Compute it for YOUR dataset. 0.000019 is valid only for Kaito's W dataset.
  signal_weight: 0.000019
  # One cut or a list of cuts on the GNN score. A list runs a cut scan. Default: 3.9
  gnn_score_cut: 3.9
```

- [ ] **Step 5: Run, expect PASS**

Run: `$PYTEST tests/pipeline -v`
Expected: all passed.

- [ ] **Step 6: Update `docs/wrapper_roadmap.md`** (use the Edit tool)

- In section 6, change line `2. Config: ...` to start with `2. ✅ Config: ...`.
- In section 7b "Waiting on Kaito", add a bullet:
  `- Location of the W input files **before** stage 1 (\`*_skimmed_no_resonances.root\` without \`selected_\`). Only the \`selected_*\` files exist now. \`configs/example_W.yaml\` uses the paths from \`jobs/graph_creation/skimming.sh\` until this is answered.`
- In section 8, add rows:
  `| 2026-10-07 | Step 2 done. Keys added later by their own steps: \`slurm.*\` (step 9), \`paths.conda_env\` (step 10). \`gnn.arch\` dropped: \`ScoreCreation\` only uses \`--archs\` in a length check. |`

- [ ] **Step 7: Commit**

```bash
/usr/bin/git add configs/analysis_defaults_W.yaml configs/example_W.yaml tests/pipeline/test_real_configs.py docs/wrapper_roadmap.md
/usr/bin/git commit -m "feat(pipeline): W defaults and example user config

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
