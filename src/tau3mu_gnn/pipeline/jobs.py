"""Job bodies: what one SLURM job runs (step 9). The generated job scripts (step 9b) call main().

Each job works in a fresh scratch folder under $TMPDIR (the job script points TMPDIR to /scratch), then
copies its outputs into the phase folder with storage.put. Upstream files are read through
storage.local_path (dCache: the /pnfs path). The last job of a phase writes the manifest.
"""
from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from tau3mu_gnn.pipeline import storage
from tau3mu_gnn.pipeline.config import load_config
from tau3mu_gnn.pipeline.manifest import check_upstream, read_manifest, write_manifest
from tau3mu_gnn.pipeline.stages import bdt as bdt_stage
from tau3mu_gnn.pipeline.stages import evaluation as evaluation_stage
from tau3mu_gnn.pipeline.stages import gnn as gnn_stage
from tau3mu_gnn.pipeline.stages import limit as limit_stage
from tau3mu_gnn.pipeline.stages import resonance_skim
from tau3mu_gnn.pipeline.stages import skim as skim_stage

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")
RESONANCE_DIR, SKIM_DIR, BDT_DIR, GNN_DIR, EVAL_DIR = "0_resonance_skim", "1_skim", "2_bdt", "3_gnn", "4_eval"
JOBS = ("skim", "bdt", "gnn", "eval", "limit", "summary")


def _scratch(job: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"{job}_"))


def _name(path) -> str:
    return storage.join(path).rsplit("/", 1)[-1]


def _read(folder, rel) -> Path:
    return storage.local_path(storage.join(folder, rel))


def skim(cfg, phase_dir) -> None:
    """Phase A, job 1: stage 0 (resonance skim / branch map) + stage 1 (skim)."""
    scratch = _scratch("skim")
    files = resonance_skim.run(cfg, scratch / RESONANCE_DIR)
    skim_stage.run(cfg, files, scratch / SKIM_DIR)
    storage.put(scratch, phase_dir)


def bdt(cfg, phase_dir, source_dir=None, manifest=True) -> None:
    """Phase A, job 2: stage 2 on <source_dir>/1_skim/ (default: this phase). Writes the phase A manifest."""
    src = source_dir if source_dir is not None else phase_dir
    inputs = {s: _read(src, f"{SKIM_DIR}/{s}.root") for s in SAMPLES}
    scratch = _scratch("bdt")
    out = bdt_stage.run(cfg, inputs, scratch / BDT_DIR)
    storage.put(scratch, phase_dir)
    if manifest:
        outputs = {f"skim_{s}": f"{SKIM_DIR}/{s}.root" for s in SAMPLES}
        outputs.update({f"bdt_{s}": Path(out[s]).relative_to(scratch).as_posix() for s in SAMPLES})
        outputs["scores_pickle"] = Path(out["scores_pickle"]).relative_to(scratch).as_posix()
        write_manifest(cfg, phase_dir, "phase_a", None, outputs)


def gnn(cfg, phase_dir, upstream_dir, sample: str) -> None:
    """Phase B: stage 3 for one sample. First job of the phase: checks the phase A version."""
    m = check_upstream(cfg, upstream_dir, "phase_b")
    scratch = _scratch(f"gnn_{sample}")
    gnn_stage.run(cfg, {sample: _read(upstream_dir, m["outputs"][f"bdt_{sample}"])}, scratch / GNN_DIR,
                  samples=(sample,))
    storage.put(scratch, phase_dir)


def evaluation(cfg, phase_dir, upstream_dir, source_dir=None, manifest=True) -> None:
    """Phase B, last job: stage 4 on <source_dir>/3_gnn/ (default: this phase) + the phase A pickle.
    Writes the phase B manifest."""
    up = read_manifest(upstream_dir)
    src = source_dir if source_dir is not None else phase_dir
    # stage 3 file names, relative to the phase folder
    gnn_files = {k: f"{GNN_DIR}/{Path(v).name}" for s in SAMPLES for k, v in gnn_stage.output_files(".", s).items()}
    inputs = {f"{s}_scores": _read(src, gnn_files[f"{s}_scores"]) for s in SAMPLES}
    inputs["scores_pickle"] = _read(upstream_dir, up["outputs"]["scores_pickle"])
    scratch = _scratch("eval")
    evaluation_stage.run(cfg, inputs, scratch / EVAL_DIR)
    storage.put(scratch, phase_dir)
    if manifest:
        outputs = {f"gnn_{s}": gnn_files[s] for s in SAMPLES}
        outputs.update({f"{s}_scores": gnn_files[f"{s}_scores"] for s in SAMPLES})
        outputs.update({f"eval_{p}": f"{EVAL_DIR}/{p}.png" for p in evaluation_stage.PLOTS})
        write_manifest(cfg, phase_dir, "phase_b", _name(upstream_dir), outputs)


def limit(cfg, phase_dir, upstream_dir, cut: float) -> None:
    """Phase C: stage 5 for one cut. Checks the phase B version. cut_<cut>/ is copied also if the stage fails."""
    m = check_upstream(cfg, upstream_dir, "phase_c")
    inputs = {s: _read(upstream_dir, m["outputs"][f"gnn_{s}"]) for s in SAMPLES}
    scratch = _scratch("limit")
    cut_dir = scratch / limit_stage.cut_dir_name(cut)
    try:
        limit_stage.run(cfg, inputs, scratch, cut)
    finally:
        if cut_dir.exists():
            storage.put(cut_dir, storage.join(phase_dir, cut_dir.name))


def summary(cfg, phase_dir, upstream_dir, source_dir=None, manifest=True) -> None:
    """Phase C, last job (afterany): limits.* from <source_dir>/cut_*/ (default: this phase).
    Writes the phase C manifest, with the cuts that have no limit in 'missing'."""
    src = source_dir if source_dir is not None else phase_dir
    scratch = _scratch("summary")
    s = limit_stage.summarize(cfg, storage.local_path(src), write_dir=scratch)
    storage.put(scratch, phase_dir)
    if manifest:
        outputs = {f"limits_{k}": name for k, name in limit_stage.SUMMARY_FILES.items()}
        outputs.update({f"limit_{r['cut']}": f"{limit_stage.cut_dir_name(r['cut'])}/limit.json"
                        for r in s["rows"] if r["limit"] is not None})
        write_manifest(cfg, phase_dir, "phase_c", _name(upstream_dir), outputs, missing=s["missing"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tau3mu_gnn.pipeline.jobs",
                                description="Run one pipeline job (called by the generated SLURM job scripts).")
    p.add_argument("job", choices=JOBS)
    p.add_argument("--config", required=True, help="user config file")
    p.add_argument("--phase-dir", required=True, help="output folder: <phase>_vsN or debug/<stage>_vsN")
    p.add_argument("--upstream", help="upstream phase version folder (gnn, eval, limit, summary)")
    p.add_argument("--source", help="read same-phase inputs from this folder (debug runs)")
    p.add_argument("--cut", type=float, help="GNN score cut (limit)")
    p.add_argument("--sample", choices=SAMPLES, help="sample (gnn)")
    p.add_argument("--no-manifest", action="store_true", help="do not write a manifest (debug runs)")
    a = p.parse_args(argv)
    need = {"gnn": ("upstream", "sample"), "eval": ("upstream",), "limit": ("upstream", "cut"),
            "summary": ("upstream",)}.get(a.job, ())
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
