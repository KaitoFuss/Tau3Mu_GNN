"""Stage 5: datacard + CMS Combine + limit read-out for one GNN score cut, on the stage 3 outputs.

Writes <out_dir>/cut_<cut>/: the datacard files (datacard.txt/.root, mass.pdf, temp trees), the Combine
outputs and logs, and limit.json ({"cut", "limit", "limitErr"}: median expected limit at CL = limit.cl).
Step 8b runs one call per cut of limit.gnn_score_cut and writes the table.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Mapping

from tau3mu_gnn.analysis.combine import combine_executable, run_combine
from tau3mu_gnn.analysis.readout_limit import read_limit
from tau3mu_gnn.pipeline.branches import check_stage_inputs
from tau3mu_gnn.pipeline.config import Config

log = logging.getLogger(__name__)

INPUTS = ("signal", "background")
REQUIRED_BRANCHES = ("GNN_score", "Selected_TauTo3Mu_fitted_mass")
SUMMARY_FILES = {"json": "limits.json", "txt": "limits.txt", "png": "limits.png"}


def cut_dir_name(cut: float) -> str:
    return f"cut_{cut}"


def check_inputs(inputs: Mapping[str, str | Path], tree_name: str) -> None:
    """Raise BranchError if a stage 3 output is missing or has no GNN_score / fitted mass."""
    check_stage_inputs({s: inputs[s] for s in INPUTS}, tree_name, REQUIRED_BRANCHES, "limit")


def run(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path, cut: float) -> dict:
    """Datacard, Combine and limit for GNN_score > cut. inputs: sample -> stage 3 ROOT file.

    Returns {"cut", "limit", "limitErr", "dir", "datacard", "limit_json"}.
    """
    lim = cfg.limit
    if cut not in lim.gnn_score_cut:
        raise ValueError(f"cut {cut} is not in limit.gnn_score_cut {list(lim.gnn_score_cut)}")
    check_inputs(inputs, cfg.inputs.tree_name)
    combine_executable()  # fail before the datacard work if Combine is missing
    from tau3mu_gnn.analysis.datacard_creation_GNN_permuted import make_datacard

    card_dir = Path(out_dir) / cut_dir_name(cut)
    log.info("stage 5 limit: cut %s, %s -> %s", cut, {s: str(inputs[s]) for s in INPUTS}, card_dir)
    card = make_datacard([inputs["signal"]], [inputs["background"]], cut, lim.signal_weight, card_dir,
                         tau_mass=lim.tau_mass, mass_window=lim.mass_window, fit_range=lim.fit_range,
                         tree_name=cfg.inputs.tree_name)
    limits = read_limit(run_combine(card_dir, lim.cl))
    if not limits:
        raise RuntimeError(f"Combine wrote no median expected limit; see {card_dir / 'combine_asymptotic.log'} "
                           f"and {card_dir / 'combine_fitdiagnostics.log'}")
    result = {"cut": cut, **limits}
    limit_json = card_dir / "limit.json"
    limit_json.write_text(json.dumps(result, indent=2) + "\n")
    return {**result, "dir": card_dir, "datacard": card.datacard, "limit_json": limit_json}


def summarize(cfg: Config, out_dir: str | Path) -> dict:
    """Table of the cut scan: reads <out_dir>/cut_<cut>/limit.json for each cut of limit.gnn_score_cut.

    A cut without a finite limit is a missing row (warning). Best cut = lowest limit, the smaller cut on a tie.
    Writes <out_dir>/limits.json, limits.txt, limits.png. Raises RuntimeError if no cut has a limit.
    """
    out_dir = Path(out_dir)
    rows = [_read_cut(out_dir, cut) for cut in sorted(cfg.limit.gnn_score_cut)]
    found = [r for r in rows if r["limit"] is not None]
    if not found:
        raise RuntimeError(f"no limit for any cut {sorted(cfg.limit.gnn_score_cut)} in {out_dir}; "
                           f"see the cut_<cut>/combine_*.log files")
    missing = [r["cut"] for r in rows if r["limit"] is None]
    if missing:
        log.warning("cut scan: no limit for cut(s) %s; see the cut_<cut>/combine_*.log files", missing)
    best = min(found, key=lambda r: r["limit"])
    for r in rows:
        r["best"] = r is best
    summary = {"cl": cfg.limit.cl, "best_cut": best["cut"], "best_limit": best["limit"],
               "missing": missing, "rows": rows}
    files = {k: out_dir / name for k, name in SUMMARY_FILES.items()}
    files["json"].write_text(json.dumps(summary, indent=2) + "\n")
    files["txt"].write_text(_table(summary))
    _plot(summary, files["png"])
    log.info("cut scan: best cut %s, limit %s -> %s", best["cut"], best["limit"], files["json"])
    return {**summary, "files": files}


def _read_cut(out_dir: Path, cut: float) -> dict:
    """One row from cut_<cut>/limit.json. No file or a non-finite limit -> limit None."""
    path = out_dir / cut_dir_name(cut) / "limit.json"
    if not path.is_file():
        return {"cut": cut, "limit": None, "limitErr": None}
    try:
        data = json.loads(path.read_text())
        row = {"cut": float(data["cut"]), "limit": float(data["limit"]), "limitErr": float(data["limitErr"])}
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path} is not a stage 5 limit.json: {e!r}") from e
    if row["cut"] != cut:
        raise ValueError(f"{path} is for cut {row['cut']}, expected {cut}")
    if not math.isfinite(row["limit"]):
        return {"cut": cut, "limit": None, "limitErr": None}
    return row


def _table(summary: dict) -> str:
    lines = [f"# median expected limit at CL {summary['cl']}; * = best cut", f"  {'cut':<10} limit"]
    for r in summary["rows"]:
        value = "missing" if r["limit"] is None else repr(r["limit"])
        lines.append(f"{'*' if r['best'] else ' '} {r['cut']!r:<10} {value}")
    return "\n".join(lines) + "\n"


def _plot(summary: dict, png: Path) -> None:
    """Median expected limit vs GNN score cut. Best cut: star + text label. Missing cuts: gray dotted lines."""
    from matplotlib.figure import Figure  # no pyplot: no display backend needed on batch nodes

    found = [r for r in summary["rows"] if r["limit"] is not None]
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.plot([r["cut"] for r in found], [r["limit"] for r in found], "-o", color="#2a6fdb", linewidth=2,
            markersize=8)
    ax.plot([summary["best_cut"]], [summary["best_limit"]], "*", color="#2a6fdb", markersize=18,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"best: cut {summary['best_cut']}, limit {summary['best_limit']:.4g}",
                (summary["best_cut"], summary["best_limit"]), textcoords="offset points", xytext=(0, -22),
                ha="center", color="#222222", fontsize=9)
    for cut in summary["missing"]:
        ax.axvline(cut, color="#9a9a9a", linestyle=":", linewidth=1)
        ax.annotate("missing", (cut, 1), xycoords=("data", "axes fraction"), textcoords="offset points",
                    xytext=(3, -12), color="#6b6b6b", fontsize=8)
    ax.set_xlabel("GNN score cut")
    ax.set_ylabel(f"median expected limit ({summary['cl']:.0%} CL)")
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.1, y=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
