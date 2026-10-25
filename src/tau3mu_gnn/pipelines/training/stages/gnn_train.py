"""Training stage gnn_train: GNN training on the graphs of the graph stage (gnn_training/train_gnn.py).

Input: graphs_dir = the out_dir of the graph stage (<graphs_dir>/<sample>_<channel>/processed/*.pt).
Writes <out_dir>/<model_name>/ with <model_name>.pt (best epoch), Normalization.pkl, arch_config.json
(what the analysis loads) plus losses, evaluation and plots, and <out_dir>/gnn_train_summary.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from tau3mu_gnn.pipelines.training.config import TrainingConfig
from tau3mu_gnn.pipelines.training.stages.graphs import SAMPLES, dataset_name, model_type

log = logging.getLogger(__name__)


def model_name(cfg: TrainingConfig) -> str:
    """e.g. kaito_W_train_W_fl_8_sd_4_GINE_delete_muons (frozen-model pattern without _vsN)."""
    fl, sd = cfg.gnn_training.arch.lower().split("_")
    unsculpting = f"_{cfg.graphs.unsculpting_mode}" if cfg.graphs.unsculpting_mode else ""
    return (f"{cfg.training_name}_{cfg.channel}_fl_{fl[2:]}_sd_{sd[2:]}_{cfg.gnn_training.conv_type}"
            f"{unsculpting}")


def run(cfg: TrainingConfig, graphs_dir: str | Path, out_dir: str | Path, device=None,
        comet_disabled: bool = False) -> dict:
    """Train the GNN. Returns the training summary plus {"summary": path of the JSON}."""
    graphs_dir = Path(graphs_dir)
    for sample in SAMPLES:
        path = graphs_dir / f"{sample}_{cfg.channel}" / "processed" / f"{dataset_name(cfg, sample)}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"gnn_train: graph file not found: {path}")
    from tau3mu_gnn.gnn_training import train_gnn

    name = model_name(cfg)
    experiment = train_gnn.make_experiment(cfg.comet, cfg.training_name, disabled=comet_disabled)
    out_dir = Path(out_dir)
    summary = train_gnn.train(graphs_dir, model_type(cfg), cfg.channel, cfg.gnn_training,
                              cfg.graphs.unsculpting_mode, name, out_dir, experiment, device=device)
    log.info("gnn_train: %s, best epoch %d of %d (early stop: %s), test AUC %s", name, summary["best_epoch"],
             summary["epochs_run"], summary["early_stopped"], summary["auc_test"])
    path = out_dir / "gnn_train_summary.json"
    path.write_text(json.dumps({**summary, "model_dir": str(summary["model_dir"])}, indent=2))
    return {**summary, "summary": path}
