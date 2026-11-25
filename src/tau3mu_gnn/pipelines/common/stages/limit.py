"""Stage 5: datacard + CMS Combine + limit read-out for one GNN score cut, on the stage 3 outputs.

Writes <out_dir>/cut_<cut>/: the datacard files (datacard.txt/.root, mass.pdf, temp trees), the Combine
outputs and logs, and limit.json ({"cut", "limit", "limitErr"}: median expected limit at CL = limit.cl).
Cut scan: one run() per cut of limit.gnn_score_cut, then summarize() writes <out_dir>/limits_gnn.json,
limits_gnn.txt and limits_gnn.png (best cut = lowest limit). run_scan() does both in one process.
The BDT-only benchmark (analysis/stages/benchmark.py) uses the same code with the tail "benchmark".
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Mapping

from tau3mu_gnn.analysis.combine import combine_executable, run_combine
from tau3mu_gnn.analysis.readout_limit import read_limit
from tau3mu_gnn.pipelines.common.branches import check_stage_inputs
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type hints only: common code does not import the analysis package at run time
    from tau3mu_gnn.pipelines.analysis.config import Config

log = logging.getLogger(__name__)

INPUTS = ("signal", "background")
REQUIRED_BRANCHES = ("GNN_score", "Selected_TauTo3Mu_fitted_mass")
TAIL = "gnn"   # summary file names: limits_<tail>.{json,txt,png}; the benchmark uses "benchmark"
SCORE_LABELS = {"GNN_score": "GNN score cut", "bdt_ISO": "BDT ISO score cut"}


def summary_files(tail: str) -> dict[str, str]:
    """Summary file names of a cut scan: {"json", "txt", "png"} -> limits_<tail>.<ext>."""
    return {ext: f"limits_{tail}.{ext}" for ext in ("json", "txt", "png")}


def cut_dir_name(cut: float) -> str:
    return f"cut_{cut}"


def check_inputs(inputs: Mapping[str, str | Path], tree_name: str, score_branch: str = "GNN_score") -> None:
    """Raise BranchError if an input is missing or has no score branch / fitted mass."""
    check_stage_inputs({s: inputs[s] for s in INPUTS}, tree_name,
                       (score_branch, "Selected_TauTo3Mu_fitted_mass"), "limit")


def run(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path, cut: float) -> dict:
    """Datacard, Combine and limit for GNN_score > cut. inputs: sample -> stage 3 ROOT file.

    Returns {"cut", "limit", "limitErr", "dir", "datacard", "limit_json"}.
    """
    lim = cfg.limit
    if cut not in lim.gnn_score_cut:
        raise ValueError(f"cut {cut} is not in limit.gnn_score_cut {list(lim.gnn_score_cut)}")
    return run_cut(inputs, out_dir, cut, score_branch="GNN_score", signal_weight=lim.signal_weight,
                   tau_mass=lim.tau_mass, mass_window=lim.mass_window, fit_range=lim.fit_range, cl=lim.cl,
                   tree_name=cfg.inputs.tree_name)


def run_cut(inputs: Mapping[str, str | Path], out_dir: str | Path, cut: float, *, score_branch: str,
            signal_weight: float, tau_mass: float, mass_window: float, fit_range: tuple[float, float],
            cl: float, tree_name: str) -> dict:
    """Datacard, Combine and limit for <score_branch> > cut into <out_dir>/cut_<cut>/ (stage 5 and the
    BDT-only benchmark). Returns {"cut", "limit", "limitErr", "dir", "datacard", "limit_json"}."""
    card_dir = Path(out_dir) / cut_dir_name(cut)
    (card_dir / "limit.json").unlink(missing_ok=True)  # a failed rerun must not leave the old limit
    check_inputs(inputs, tree_name, score_branch)
    combine_executable()  # fail before the datacard work if Combine is missing
    from tau3mu_gnn.analysis.datacard import make_datacard

    log.info("limit: %s > %s, %s -> %s", score_branch, cut, {s: str(inputs[s]) for s in INPUTS}, card_dir)
    card = make_datacard([inputs["signal"]], [inputs["background"]], cut, signal_weight, card_dir,
                         tau_mass=tau_mass, mass_window=mass_window, fit_range=fit_range,
                         tree_name=tree_name, score_branch=score_branch)
    limits = read_limit(run_combine(card_dir, cl))
    if not limits:
        raise RuntimeError(f"Combine wrote no median expected limit; see {card_dir / 'combine_asymptotic.log'} "
                           f"and {card_dir / 'combine_fitdiagnostics.log'}")
    result = {"cut": cut, **limits}
    limit_json = card_dir / "limit.json"
    limit_json.write_text(json.dumps(result, indent=2) + "\n")
    return {**result, "dir": card_dir, "datacard": card.datacard, "limit_json": limit_json}


def run_scan(cfg: Config, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict:
    """All cuts of limit.gnn_score_cut one after the other in this process, then summarize. For local runs;
    the driver sends one job per cut and a final summarize job instead.

    Missing inputs or a missing Combine stop before the first cut. A cut whose Combine fails (non-zero exit or
    no median limit) is logged and becomes a missing row. Other errors stop the scan.
    """
    check_inputs(inputs, cfg.inputs.tree_name)
    combine_executable()
    for cut in cfg.limit.gnn_score_cut:
        try:
            run(cfg, inputs, out_dir, cut)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            log.error("cut scan: cut %s failed: %s", cut, e)
    return summarize(cfg, out_dir)


def summarize(cfg: Config, out_dir: str | Path, write_dir: str | Path | None = None) -> dict:
    """Table of the GNN cut scan (cuts of limit.gnn_score_cut) -> limits_gnn.*; see summarize_cuts."""
    return summarize_cuts(cfg.limit.gnn_score_cut, cfg.limit.cl, out_dir, write_dir, tail=TAIL,
                          score_branch="GNN_score")


def summarize_cuts(cuts, cl: float, out_dir: str | Path, write_dir: str | Path | None = None, *, tail: str,
                   score_branch: str) -> dict:
    """Table of a cut scan: reads <out_dir>/cut_<cut>/limit.json for each cut of cuts.

    A cut without a finite limit is a missing row (warning). Best cut = lowest limit, the smaller cut on a tie.
    Writes limits_<tail>.json, .txt, .png into write_dir (default out_dir). Raises RuntimeError if no cut
    has a limit.
    """
    out_dir = Path(out_dir)
    write_dir = Path(write_dir) if write_dir is not None else out_dir
    write_dir.mkdir(parents=True, exist_ok=True)
    files = {k: write_dir / name for k, name in summary_files(tail).items()}
    for f in files.values():
        f.unlink(missing_ok=True)  # a failed summary must not leave the table of an earlier scan
    rows = [_read_cut(out_dir, cut) for cut in sorted(cuts)]
    found = [r for r in rows if r["limit"] is not None]
    if not found:
        raise RuntimeError(f"no limit for any cut {sorted(cuts)} in {out_dir}; "
                           f"see the cut_<cut>/combine_*.log files")
    missing = [r["cut"] for r in rows if r["limit"] is None]
    if missing:
        log.warning("cut scan: no limit for cut(s) %s; see the cut_<cut>/combine_*.log files", missing)
    best = min(found, key=lambda r: r["limit"])
    for r in rows:
        r["best"] = r is best
    summary = {"cl": cl, "score_branch": score_branch, "best_cut": best["cut"], "best_limit": best["limit"],
               "missing": missing, "rows": rows}
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
    lines = [f"# median expected limit at CL {summary['cl']}, cut on {summary['score_branch']}; * = best cut",
             f"  {'cut':<10} limit"]
    for r in summary["rows"]:
        value = "missing" if r["limit"] is None else repr(r["limit"])
        lines.append(f"{'*' if r['best'] else ' '} {r['cut']!r:<10} {value}")
    return "\n".join(lines) + "\n"


def _plot(summary: dict, png: Path) -> None:
    """Median expected limit vs score cut. Best cut: star + text label. Missing cuts: gray dotted lines."""
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
    ax.set_xlabel(SCORE_LABELS.get(summary["score_branch"], f"{summary['score_branch']} cut"))
    ax.set_ylabel(f"median expected limit ({summary['cl']:.0%} CL)")
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.1, y=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
