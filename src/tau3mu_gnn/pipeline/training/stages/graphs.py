"""Training stage graphs: GNN training graphs of one sample (graph/ConeGraphProducer.py).

Input: the stage 2 output of the sample (bdt_ISO present). Per sample, in this order:
BDT cut bdt_ISO >= graphs.bdt_preselection_cut -> sideband (background: candidate 0 outside
tau_mass +- sideband_half_width) -> random subset of graphs.max_events (seeded) -> one graph per
event from candidate 0. npvs weights on the signal graphs only; background graphs weight 1.
Writes <out_dir>/<sample>_<channel>/processed/<dataset_name>.pt, the layout Train_TauMuNet reads
(data_base = out_dir); the input.root link is removed after the build. Training and inference build identical graphs (tests/pipeline/test_graphs.py).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from tau3mu_gnn.bdt_training.bdt_inputs import expression_names
from tau3mu_gnn.gnn_training.gnn_inputs import gnn_branches, graph_level_branches
from tau3mu_gnn.graph.naming import graph_suffix
from tau3mu_gnn.pipeline.branches import check_stage_inputs
from tau3mu_gnn.pipeline.training.config import TrainingConfig

log = logging.getLogger(__name__)

SAMPLES = ("signal", "background")


def model_type(cfg: TrainingConfig) -> str:
    """Graph name part after '<prefix>_<sample>_', e.g. skimmed_dzd0_cone_size_0.5_knn_10_delete_muons."""
    g = cfg.graphs
    return "skimmed" + graph_suffix(g.use_displacement, g.cone_size, g.connection, k=g.k,
                                    unsculpting_mode=g.unsculpting_mode)


def dataset_name(cfg: TrainingConfig, sample: str) -> str:
    prefix = "W_decay" if cfg.channel == "W" else "HF_decay"
    return f"{prefix}_{sample}_{model_type(cfg)}"


def run(cfg: TrainingConfig, sample: str, bdt_file: str | Path, out_dir: str | Path,
        npvs_file: str | Path | None = None) -> Path:
    """Build the graphs of sample from bdt_file. Signal needs npvs_file; background must not get one.
    Returns the .pt file."""
    if sample not in SAMPLES:
        raise ValueError(f"graphs: sample must be one of {SAMPLES}, got {sample!r}")
    signal = sample == "signal"
    if signal and npvs_file is None:
        raise ValueError("graphs: the signal graphs need the npvs weights file (npvs_file)")
    if not signal and npvs_file is not None:
        raise ValueError("graphs: npvs weights are for signal only, not for background")
    weights = "npvs_rescale" if signal else None
    tree = cfg.inputs.tree_name
    required = expression_names(gnn_branches(cfg.channel, weights=weights, save_benchmark_score=True))
    check_stage_inputs({sample: bdt_file}, tree, required, "graphs")

    folder = Path(out_dir) / f"{sample}_{cfg.channel}"
    folder.mkdir(parents=True)  # FileExistsError: PyG would silently reuse an old .pt
    os.symlink(Path(bdt_file).resolve(), folder / "input.root")  # the producer reads every .root in its folder
    name = dataset_name(cfg, sample)
    g = cfg.graphs
    try:
        from tau3mu_gnn.graph.ConeGraphProducer import ConeGraphProducer

        dataset = ConeGraphProducer(
            str(folder), output_dataset_name=name, cone_size=g.cone_size, edge_connection_mode=g.connection,
            k_knn=g.k if g.connection == "knn" else None, max_events_to_process=g.max_events,
            use_relative_angles=True, save_edge_attributes=True, save_benchmark_score=True, weights=weights,
            mask=not signal, label=1. if signal else 0., tau_channel=cfg.channel,
            graph_level_branches=graph_level_branches(cfg.channel), slimmed=True, shuffle_bool=True,
            use_displacement=g.use_displacement, unsculpting_mode=g.unsculpting_mode,
            bdt_preselection_cut=g.bdt_preselection_cut, tau_mass=cfg.limit.tau_mass,
            sideband_half_width=cfg.sideband_half_width,
            npvs_weights_file=str(npvs_file) if signal else None, shuffle_seed=g.shuffle_seed, tree_name=tree)
    finally:
        (folder / "input.root").unlink(missing_ok=True)  # storage.put of the folder must not copy the stage 2 file
    log.info("graphs: %s, %d graphs -> %s", sample, len(dataset), folder / "processed" / f"{name}.pt")
    return folder / "processed" / f"{name}.pt"
