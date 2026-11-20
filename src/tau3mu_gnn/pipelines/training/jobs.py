"""Job bodies of the training phases. The generated job scripts call main().

Same pattern as the analysis jobs (analysis/jobs.py): each job works in a fresh scratch folder under $TMPDIR,
then copies its outputs into the phase folder with storage.put. Upstream files are read through
storage.local_path. The last job of a phase writes the manifest (bdt_infer, gnn_train).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from tau3mu_gnn.pipelines.common import batch, storage
from tau3mu_gnn.pipelines.common.branches import TRAINING_EXTRA
from tau3mu_gnn.pipelines.common.jobs import RESONANCE_DIR, SKIM_DIR, _name, _read, _scratch
from tau3mu_gnn.pipelines.common.stages import resonance_skim
from tau3mu_gnn.pipelines.common.stages import skim as skim_stage
from tau3mu_gnn.pipelines.training.config import load_training_config
from tau3mu_gnn.pipelines.training.manifest import check_upstream, write_manifest
from tau3mu_gnn.pipelines.training.stages import bdt_infer as bdt_infer_stage
from tau3mu_gnn.pipelines.training.stages import bdt_train as bdt_train_stage
from tau3mu_gnn.pipelines.training.stages import gnn_train as gnn_train_stage
from tau3mu_gnn.pipelines.training.stages import graphs as graphs_stage
from tau3mu_gnn.pipelines.training.stages import npvs as npvs_stage

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")
BDT_TRAIN_DIR, BDT_DIR, NPVS_DIR, GRAPHS_DIR, GNN_TRAIN_DIR = "bdt_train", "2_bdt", "npvs", "graphs", "gnn_train"
JOBS = ("skim", "bdt_train", "bdt_infer", "npvs", "graphs", "gnn_train")
NEEDS = {"npvs": ("upstream",), "graphs": ("upstream", "sample"), "gnn_train": ("upstream",)}


def _rel(path, scratch: Path) -> str:
    return Path(path).relative_to(scratch).as_posix()


def _put(scratch: Path, phase_dir) -> None:
    log.info("copy %s -> %s", scratch, phase_dir)
    storage.put(scratch, phase_dir)


def _check_proxy(cfg, resources: str) -> None:
    """The dCache proxy must outlive this job (its own batch time limit): storage.put runs at the end.
    Raises batch.ProxyError at once, before any work. No-op for a local or EOS output_base."""
    batch.check_proxy(cfg, batch.time_hours(batch.resources(cfg)[resources].time))


def skim(cfg, phase_dir) -> None:
    """bdt, job 1: stage 0 (with the training branches) + stage 1."""
    _check_proxy(cfg, "skim")
    log.info("job skim: %s", phase_dir)
    scratch = _scratch("skim")
    files = resonance_skim.run(cfg, scratch / RESONANCE_DIR, extra_branches=TRAINING_EXTRA.get(cfg.channel, ()))
    skim_stage.run(cfg, files, scratch / SKIM_DIR)
    _put(scratch, phase_dir)


def _skim_inputs(phase_dir) -> dict[str, Path]:
    return {s: _read(phase_dir, f"{SKIM_DIR}/{s}.root") for s in SAMPLES}


def bdt_train(cfg, phase_dir) -> None:
    """bdt, job 2: ISO BDT training on <phase_dir>/1_skim/."""
    _check_proxy(cfg, "bdt_train")
    log.info("job bdt_train: %s", phase_dir)
    scratch = _scratch("bdt_train")
    bdt_train_stage.run(cfg, _skim_inputs(phase_dir), scratch / BDT_TRAIN_DIR)
    _put(scratch, phase_dir)


def bdt_infer(cfg, phase_dir) -> None:
    """bdt, last job: stage 2 with the bdt_train classifier. Writes the bdt manifest."""
    _check_proxy(cfg, "bdt_infer")
    log.info("job bdt_infer: %s", phase_dir)
    classifier = f"{BDT_TRAIN_DIR}/{bdt_train_stage.OUTPUTS['classifier']}"
    scratch = _scratch("bdt_infer")
    out = bdt_infer_stage.run(cfg, _skim_inputs(phase_dir), _read(phase_dir, classifier), scratch / BDT_DIR)
    _put(scratch, phase_dir)
    outputs = {f"skim_{s}": f"{SKIM_DIR}/{s}.root" for s in SAMPLES}
    outputs.update({f"bdt_{s}": _rel(out[s], scratch) for s in SAMPLES})
    outputs["classifier"] = classifier
    outputs["bdt_train_summary"] = f"{BDT_TRAIN_DIR}/{bdt_train_stage.OUTPUTS['summary']}"
    write_manifest(cfg, phase_dir, "bdt", None, outputs)


def _npvs_file(cfg) -> str:
    return f"{NPVS_DIR}/{npvs_stage.OUTPUTS['weights'].format(channel=cfg.channel)}"


def npvs(cfg, phase_dir, upstream_dir) -> None:
    """gnn, first job: npvs weights from the upstream stage 1 files. Checks the bdt version."""
    _check_proxy(cfg, "npvs")
    log.info("job npvs: %s, upstream %s", phase_dir, upstream_dir)
    m = check_upstream(cfg, upstream_dir, "gnn")
    inputs = {s: _read(upstream_dir, m["outputs"][f"skim_{s}"]) for s in SAMPLES}
    scratch = _scratch("npvs")
    npvs_stage.run(cfg, inputs, scratch / NPVS_DIR)
    _put(scratch, phase_dir)


def graphs(cfg, phase_dir, upstream_dir, sample: str) -> None:
    """gnn: graphs of one sample. Signal reads the npvs weights of this phase; background has none and
    may start before npvs, so both check the bdt version."""
    _check_proxy(cfg, f"graphs_{sample}")
    log.info("job graphs: %s, sample %s, upstream %s", phase_dir, sample, upstream_dir)
    m = check_upstream(cfg, upstream_dir, "gnn")
    npvs_file = _read(phase_dir, _npvs_file(cfg)) if sample == "signal" else None
    scratch = _scratch(f"graphs_{sample}")
    graphs_stage.run(cfg, sample, _read(upstream_dir, m["outputs"][f"bdt_{sample}"]), scratch / GRAPHS_DIR,
                     npvs_file=npvs_file)
    _put(scratch, phase_dir)


def gnn_train(cfg, phase_dir, upstream_dir) -> None:
    """gnn, last job: GNN training on <phase_dir>/graphs/ (Comet on). Writes the gnn manifest (files only:
    the model folder and name follow from the 'model' path)."""
    _check_proxy(cfg, "gnn_train")
    log.info("job gnn_train: %s, upstream %s", phase_dir, upstream_dir)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:  # allowed, but much slower: say so in the log
        log.warning("gnn_train: no CUDA device visible on this node (check the GPU request of the batch job); "
                    "training on CPU")
        device = torch.device("cpu")
    scratch = _scratch("gnn_train")
    out = gnn_train_stage.run(cfg, _read(phase_dir, GRAPHS_DIR), scratch / GNN_TRAIN_DIR,
                              device=device, comet_disabled=False)
    _put(scratch, phase_dir)
    model_dir = _rel(out["model_dir"], scratch)
    outputs = {"model": f"{model_dir}/{out['model_name']}.pt", "normalization": f"{model_dir}/Normalization.pkl",
               "arch_config": f"{model_dir}/arch_config.json", "npvs_weights": _npvs_file(cfg)}
    outputs.update({f"graphs_{s}": f"{GRAPHS_DIR}/{s}_{cfg.channel}/processed/"
                                   f"{graphs_stage.dataset_name(cfg, s)}.pt" for s in SAMPLES})
    outputs["gnn_train_summary"] = _rel(out["summary"], scratch)
    write_manifest(cfg, phase_dir, "gnn", _name(upstream_dir), outputs)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tau3mu_gnn.pipelines.training.jobs",
                                description="Run one training job (called by the generated batch job scripts).")
    p.add_argument("job", choices=JOBS)
    p.add_argument("--config", required=True, help="training user config file")
    p.add_argument("--phase-dir", required=True, help="output folder: <phase>_vsN")
    p.add_argument("--upstream", help="upstream bdt version folder (gnn jobs)")
    p.add_argument("--sample", choices=SAMPLES, help="sample (graphs)")
    a = p.parse_args(argv)
    absent = [f"--{n}" for n in NEEDS.get(a.job, ()) if getattr(a, n) is None]
    if absent:
        p.error(f"job {a.job} needs {' '.join(absent)}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_training_config(a.config)
    if a.job in ("skim", "bdt_train", "bdt_infer"):
        {"skim": skim, "bdt_train": bdt_train, "bdt_infer": bdt_infer}[a.job](cfg, a.phase_dir)
    elif a.job in ("npvs", "gnn_train"):
        {"npvs": npvs, "gnn_train": gnn_train}[a.job](cfg, a.phase_dir, a.upstream)
    else:
        graphs(cfg, a.phase_dir, a.upstream, a.sample)

if __name__ == "__main__":
    main()
