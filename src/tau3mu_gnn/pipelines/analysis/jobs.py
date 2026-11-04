"""Analysis job bodies: what one batch job runs. The generated job scripts call main().

Each job works in a fresh scratch folder under $TMPDIR (the job script points TMPDIR to the job scratch folder), then
copies its outputs into the phase folder with storage.put. Upstream files are read through
storage.local_path (dCache: the /pnfs path). The last job of a phase writes the manifest.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tau3mu_gnn.pipelines.analysis.config import load_config
from tau3mu_gnn.pipelines.analysis.manifest import check_upstream, write_manifest
from tau3mu_gnn.pipelines.common import storage
from tau3mu_gnn.pipelines.common.jobs import RESONANCE_DIR, SKIM_DIR, _name, _read, _scratch
from tau3mu_gnn.pipelines.common.stages import bdt as bdt_stage
from tau3mu_gnn.pipelines.analysis.stages import evaluation as evaluation_stage
from tau3mu_gnn.pipelines.analysis.stages import gnn as gnn_stage
from tau3mu_gnn.pipelines.common.stages import limit as limit_stage
from tau3mu_gnn.pipelines.common.stages import resonance_skim
from tau3mu_gnn.pipelines.common.stages import skim as skim_stage

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")
BDT_DIR, GNN_DIR, EVAL_DIR = "2_bdt", "3_gnn", "4_eval"
JOBS = ("skim", "bdt", "gnn", "eval", "limit", "summary")


def skim(cfg, phase_dir) -> None:
    """Phase A, job 1: stage 0 (resonance skim / branch map) + stage 1 (skim)."""
    log.info("job skim: %s", phase_dir)
    scratch = _scratch("skim")
    files = resonance_skim.run(cfg, scratch / RESONANCE_DIR)
    skim_stage.run(cfg, files, scratch / SKIM_DIR)
    log.info("copy %s -> %s", scratch, phase_dir)
    storage.put(scratch, phase_dir)


def bdt(cfg, phase_dir, source_dir=None, manifest=True) -> None:
    """Phase bdt, job 2: stage 2 on <source_dir>/1_skim/ (default: this phase). Writes the bdt manifest."""
    src = source_dir if source_dir is not None else phase_dir
    log.info("job bdt: %s, source %s", phase_dir, src)
    inputs = {s: _read(src, f"{SKIM_DIR}/{s}.root") for s in SAMPLES}
    scratch = _scratch("bdt")
    out = bdt_stage.run(cfg, inputs, scratch / BDT_DIR)
    log.info("copy %s -> %s", scratch, phase_dir)
    storage.put(scratch, phase_dir)
    if manifest:
        outputs = {f"skim_{s}": f"{SKIM_DIR}/{s}.root" for s in SAMPLES}
        outputs.update({f"bdt_{s}": Path(out[s]).relative_to(scratch).as_posix() for s in SAMPLES})
        outputs["scores_pickle"] = Path(out["scores_pickle"]).relative_to(scratch).as_posix()
        write_manifest(cfg, phase_dir, "bdt", None, outputs)


def gnn(cfg, phase_dir, upstream_dir, sample: str) -> None:
    """Phase gnn: stage 3 for one sample. First job of the phase: checks the bdt version."""
    log.info("job gnn: %s, sample %s, upstream %s", phase_dir, sample, upstream_dir)
    m = check_upstream(cfg, upstream_dir, "gnn")
    scratch = _scratch(f"gnn_{sample}")
    gnn_stage.run(cfg, {sample: _read(upstream_dir, m["outputs"][f"bdt_{sample}"])}, scratch / GNN_DIR,
                  samples=(sample,))
    log.info("copy %s -> %s", scratch, phase_dir)
    storage.put(scratch, phase_dir)


def evaluation(cfg, phase_dir, upstream_dir, source_dir=None, manifest=True) -> None:
    """Phase gnn, last job: stage 4 on <source_dir>/3_gnn/ (default: this phase) + the bdt pickle.
    Writes the gnn manifest."""
    src = source_dir if source_dir is not None else phase_dir
    log.info("job eval: %s, source %s, upstream %s", phase_dir, src, upstream_dir)
    up = check_upstream(cfg, upstream_dir, "gnn")
    # stage 3 file names, relative to the phase folder
    gnn_files = {k: f"{GNN_DIR}/{Path(v).name}" for s in SAMPLES for k, v in gnn_stage.output_files(".", s).items()}
    inputs = {f"{s}_scores": _read(src, gnn_files[f"{s}_scores"]) for s in SAMPLES}
    inputs["scores_pickle"] = _read(upstream_dir, up["outputs"]["scores_pickle"])
    scratch = _scratch("eval")
    evaluation_stage.run(cfg, inputs, scratch / EVAL_DIR)
    log.info("copy %s -> %s", scratch, phase_dir)
    storage.put(scratch, phase_dir)
    if manifest:
        outputs = {f"gnn_{s}": gnn_files[s] for s in SAMPLES}
        outputs.update({f"{s}_scores": gnn_files[f"{s}_scores"] for s in SAMPLES})
        outputs.update({f"eval_{p}": f"{EVAL_DIR}/{p}.png" for p in evaluation_stage.PLOTS})
        write_manifest(cfg, phase_dir, "gnn", _name(upstream_dir), outputs)


def limit(cfg, phase_dir, upstream_dir, cut: float) -> None:
    """Phase limit: stage 5 for one cut. Checks the gnn version. cut_<cut>/ is copied also if the stage fails."""
    log.info("job limit: %s, cut %s, upstream %s", phase_dir, cut, upstream_dir)
    m = check_upstream(cfg, upstream_dir, "limit")
    inputs = {s: _read(upstream_dir, m["outputs"][f"gnn_{s}"]) for s in SAMPLES}
    scratch = _scratch("limit")
    cut_dir = scratch / limit_stage.cut_dir_name(cut)
    try:
        limit_stage.run(cfg, inputs, scratch, cut)
    finally:
        if cut_dir.exists():
            dest = storage.join(phase_dir, cut_dir.name)
            log.info("copy %s -> %s", cut_dir, dest)
            storage.put(cut_dir, dest)


def summary(cfg, phase_dir, upstream_dir=None, source_dir=None, manifest=True) -> None:
    """Phase limit, last job (afterany): limits.* from <source_dir>/cut_*/ (default: this phase).
    Writes the limit manifest, with the cuts that have no limit in 'missing'."""
    if manifest and upstream_dir is None:
        raise ValueError("summary with a manifest needs the upstream gnn folder")
    src = source_dir if source_dir is not None else phase_dir
    log.info("job summary: %s, source %s, upstream %s", phase_dir, src, upstream_dir)
    scratch = _scratch("summary")
    s = limit_stage.summarize(cfg, storage.local_path(src), write_dir=scratch)
    log.info("copy %s -> %s", scratch, phase_dir)
    storage.put(scratch, phase_dir)
    if manifest:
        outputs = {f"limits_{k}": name for k, name in limit_stage.SUMMARY_FILES.items()}
        outputs.update({f"limit_{r['cut']}": f"{limit_stage.cut_dir_name(r['cut'])}/limit.json"
                        for r in s["rows"] if r["limit"] is not None})
        write_manifest(cfg, phase_dir, "limit", _name(upstream_dir), outputs, missing=s["missing"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tau3mu_gnn.pipelines.analysis.jobs",
                                description="Run one pipeline job (called by the generated batch job scripts).")
    p.add_argument("job", choices=JOBS)
    p.add_argument("--config", required=True, help="user config file")
    p.add_argument("--phase-dir", required=True, help="output folder: <phase>_vsN or debug/<stage>_vsN")
    p.add_argument("--upstream", help="upstream phase version folder (gnn, eval, limit, summary)")
    p.add_argument("--source", help="read same-phase inputs from this folder (debug runs)")
    p.add_argument("--cut", type=float, help="GNN score cut (limit)")
    p.add_argument("--sample", choices=SAMPLES, help="sample (gnn)")
    p.add_argument("--no-manifest", action="store_true", help="do not write a manifest (debug runs)")
    a = p.parse_args(argv)
    need = {"gnn": ("upstream", "sample"), "eval": ("upstream",), "limit": ("upstream", "cut")}.get(a.job, ())
    if a.job == "summary" and not a.no_manifest:
        need = ("upstream",)
    absent = [f"--{n}" for n in need if getattr(a, n) is None]
    if absent:
        p.error(f"job {a.job} needs {' '.join(absent)}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config)
    manifest = not a.no_manifest
    if a.job == "skim":
        skim(cfg, a.phase_dir)
    elif a.job == "bdt":
        bdt(cfg, a.phase_dir, a.source, manifest)
    elif a.job == "gnn":
        gnn(cfg, a.phase_dir, a.upstream, a.sample)
    elif a.job == "eval":
        evaluation(cfg, a.phase_dir, a.upstream, a.source, manifest)
    elif a.job == "limit":
        limit(cfg, a.phase_dir, a.upstream, a.cut)
    else:
        summary(cfg, a.phase_dir, a.upstream, a.source, manifest)


if __name__ == "__main__":
    main()
