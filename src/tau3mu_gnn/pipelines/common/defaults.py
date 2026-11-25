"""The frozen analysis_defaults_<channel>.yaml, shared by both pipelines.

The analysis reads all of it; the training reads the shared stage settings from it (resonance skim,
skim, limit fit) and its own frozen file for the rest (pipelines/training/config.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tau3mu_gnn.pipelines.common.config import (
    _Field, _flatten, _name, _non_empty_str, _number, _number_list, _open_unit_interval, _parse,
    _positive_number, _range, _read_yaml, _slurm_jobs_parser, _htcondor_jobs_parser,
)

SLURM_JOBS = ("skim", "bdt", "gnn_signal", "gnn_background", "eval", "limit", "summary",   # also the HTCondor jobs
              "benchmark_limit", "benchmark_summary")

# Keys of the frozen defaults file. Paths are relative to the repo root.
# The channel is not repeated here: the user file selects analysis_defaults_<channel>.yaml.
DEFAULTS_FIELDS: dict[str, _Field] = {
    "resonance_skim.vtx_chi2_max": _Field(_positive_number),
    "resonance_skim.mass_range": _Field(_range),
    "resonance_skim.veto_masses": _Field(_number_list),
    "resonance_skim.veto_half_width": _Field(_positive_number),
    "resonance_skim.pf_dr_max": _Field(_positive_number),
    "skim.dz_max": _Field(_positive_number),
    "skim.cos_alpha_2d_min": _Field(_number),
    "bdt.classifier": _Field(_non_empty_str),
    "gnn.model_dir": _Field(_non_empty_str),
    "gnn.model_name": _Field(_name),   # model settings: <gnn.model_dir>/<gnn.model_name>/model_settings.json
    "limit.tau_mass": _Field(_positive_number),
    "limit.mass_window": _Field(_positive_number),
    "limit.fit_range": _Field(_range),
    "limit.cl": _Field(_open_unit_interval),
    "slurm.account": _Field(_non_empty_str),
    "slurm.jobs": _Field(_slurm_jobs_parser(SLURM_JOBS)),
    "htcondor.jobs": _Field(_htcondor_jobs_parser(SLURM_JOBS)),
}


def read_analysis_defaults(defaults_dir: Path, channel: str,
                           problems: list[str]) -> tuple[dict[str, Any], Path] | None:
    """Read and parse analysis_defaults_<channel>.yaml. None if it cannot be read."""
    path = defaults_dir / f"analysis_defaults_{channel}.yaml"
    raw = _read_yaml(path, problems)
    if raw is None:
        return None
    return _parse(_flatten(raw, DEFAULTS_FIELDS), DEFAULTS_FIELDS, f"defaults {path}", problems), path
