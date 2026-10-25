"""Training stage bdt_train: ISO BDT on the stage 1 outputs (bdt_training/train_bdt.py).

Signal: all events. Background: sideband only (all candidates abs(m - tau_mass) > sideband_half_width).
Writes the classifier pickle that stage 2 reads unchanged, a summary JSON, the ROC data and plots.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Mapping

from tau3mu_gnn.bdt_training.bdt_inputs import input_branch_names
from tau3mu_gnn.pipelines.common.branches import check_stage_inputs
from tau3mu_gnn.pipelines.training.config import TrainingConfig

log = logging.getLogger(__name__)

OUTPUTS = {"classifier": "ISO_classifiers.pck", "summary": "bdt_train_summary.json", "roc": "roc.pck"}


def run(cfg: TrainingConfig, inputs: Mapping[str, str | Path], out_dir: str | Path) -> dict:
    """inputs: {"signal": file, "background": file} (stage 1).
    Returns {"classifier": path, "summary": path, "roc": path, "plots": [paths]}."""
    tree = cfg.inputs.tree_name
    check_stage_inputs(inputs, tree, input_branch_names(cfg.channel), "BDT training")
    from tau3mu_gnn.bdt_training import train_bdt

    p = cfg.bdt_training
    data, features = train_bdt.load_training_data(
        [inputs["signal"]], [inputs["background"]], tree, cfg.channel, cfg.limit.tau_mass,
        cfg.sideband_half_width, p.max_events_per_class, p.sample_seed)
    log.info("bdt_train: %d signal, %d sideband background events",
             (data.target == 1).sum(), (data.target == 0).sum())
    classifiers, summary, train, test = train_bdt.train_iso_bdt(data, features, p)
    log.info("bdt_train: test AUC %.4f, best iteration per fold %s, KS p-value train vs test %s",
             summary["auc_test"], summary["best_iteration"], summary["ks_pvalue"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {key: out_dir / name for key, name in OUTPUTS.items()}
    with open(outputs["classifier"], "wb") as f:
        pickle.dump(classifiers, f)
    outputs["summary"].write_text(json.dumps({k: v for k, v in summary.items() if k != "logloss"}, indent=2))
    with open(outputs["roc"], "wb") as f:
        pickle.dump(train_bdt.roc_data(test), f)
    outputs["plots"] = train_bdt.plot_training(summary, train, test, classifiers, features, out_dir)
    return outputs
