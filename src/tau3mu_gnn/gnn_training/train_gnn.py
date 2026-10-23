"""GNN training for the pipeline (scripts/Train_TauMuNet.py without fine-tune).

Settings come from GNNTrainingConfig: architecture lookup without the registry side effect, patience,
seed, class mix, data windows. Kaito, 2026-10-23: the best epoch is always kept; the seed is set.
Writes <out_dir>/<model_name>/ with the files the analysis loads (<model_name>.pt, Normalization.pkl,
arch_config.json) and the old extra files (losses, evaluation, plots).
"""
from __future__ import annotations

import json
from pathlib import Path


def make_experiment(comet, project: str, disabled: bool = False):
    """Comet experiment with workspace and key from CometConfig. disabled: no network, nothing logged."""
    from comet_ml import Experiment

    return Experiment(api_key=comet.api_key, workspace=comet.workspace, project_name=project, disabled=disabled)


def train(data_base: str | Path, model_type: str, tau_channel: str, params, unsculpting_mode: str | None,
          model_name: str, out_dir: str | Path, experiment, device=None) -> dict:
    """Train on <data_base>/<sample>_<channel>/processed/<prefix>_<sample>_<model_type>.pt.
    params: GNNTrainingConfig. Returns a summary (model_dir, model_name, best_epoch, epochs_run,
    early_stopped, best_validation_loss, auc_test)."""
    import tau3mu_gnn.gnn_training.TrainingUtils as utrain
    from tau3mu_gnn.architectures.GNN_archs import get_arch_configs
    from tau3mu_gnn.scripts import Train_TauMuNet

    config, _ = get_arch_configs(params.arch, tau_channel, params.disco, registry_path=None)
    out_dir = Path(out_dir)
    with utrain.output_base(out_dir):
        model_dir = Path(utrain.output_directory(model_name))
        (model_dir / "arch_config.json").write_text(json.dumps(config))
        experiment.log_parameters({"architecture": params.arch, "learning_rate": params.learning_rate,
                                   "regularisation": params.disco, "epochs": params.epochs, "Model": model_type,
                                   "tau_channel": tau_channel, "gnn_mode": params.gnn_mode,
                                   "conv_type": params.conv_type, "unsculpting_mode": unsculpting_mode,
                                   "seed": params.seed})
        experiment.set_name(model_name)
        summary = Train_TauMuNet.main(
            params.epochs, params.batch_size, params.learning_rate, params.disco, model_name, model_type, config,
            tau_channel, params.gnn_mode, params.conv_type, str(data_base), experiment=experiment,
            patience=params.early_stopping_patience, lr_scheduler_patience=params.lr_scheduler_patience,
            dataset_fraction=params.class_fraction, windows=params.windows, seed=params.seed, device=device)
    return {"model_dir": model_dir, "model_name": model_name, **summary}
