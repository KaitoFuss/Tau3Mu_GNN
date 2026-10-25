import json

from tau3mu_gnn.pipelines.common.config import REPO_ROOT


def test_arch_config_equals_frozen_file():
    from tau3mu_gnn.architectures.GNN_archs import get_arch_configs
    config, ident = get_arch_configs("fl8_sd4", "W", 0.0, registry_path=None)
    frozen = json.loads((REPO_ROOT / "models/W/gnn/new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2/arch_config.json").read_text())
    assert json.loads(json.dumps(config)) == frozen and ident == ["fl_8", "sd_4"]


def test_arch_import_no_side_effects(monkeypatch):
    import importlib
    import os
    import tau3mu_gnn.architectures.GNN_archs as archs
    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: False if p in ("/eos/user/k/kfuss/", "/work/kfuss/") else real_exists(p))
    try:
        reloaded = importlib.reload(archs)          # unknown machine: no raise
        assert reloaded.REGISTRY_PATH is None
        reloaded.get_arch_configs("fl8_sd4", "W", 0.0)   # default registry path None: nothing written
    finally:
        monkeypatch.undo()
        importlib.reload(archs)


import numpy as np
import pytest
import torch

from test_graphs import training_events


def _varied(n, rng, signal, first_event):
    """Toy events with every feature varying (Normalize divides by the std; constant features -> NaN)."""
    import awkward as ak
    from test_graphs import _rich_events
    data = _rich_events(n, rng)
    data["event"] = np.arange(first_event, first_event + n, dtype=np.uint64)
    for name in list(data):
        if (name.startswith(("Selected_TauTo3Mu_", "TauPlusMET_")) and not name.endswith(("_eta", "_phi", "_mass"))
                and "iso_" not in name):
            data[name] = ak.Array([[float(abs(rng.normal(1.0, 0.4))) + 0.3] for _ in range(n)])
    data["Selected_TauTo3Mu_fitted_pt"] = ak.Array([[float(rng.normal(45 if signal else 30, 8))] for _ in range(n)])
    # varied masses (constant masses make the DisCo term 0/0 = NaN); background in the sideband
    data["Selected_TauTo3Mu_fitted_mass"] = ak.Array(
        [[float(rng.normal(1.78, 0.01) if signal else rng.choice([rng.uniform(1.62, 1.69), rng.uniform(1.87, 1.98)]))]
         for _ in range(n)])
    data["TauPlusMET_Puppi_phi"] = ak.Array([[float(rng.uniform(-3, 3))] for _ in range(n)])
    return data


def _toy_graphs(tmp_path, cfg, n=60):
    """n signal + n background events through the 12d graph stage -> data_base for the training."""
    from conftest import write_tree
    from tau3mu_gnn.pipelines.training.stages import graphs
    rng = np.random.default_rng(5)
    sig = write_tree(tmp_path / "in" / "s.root", _varied(n, rng, True, 0))
    bkg = write_tree(tmp_path / "in" / "b.root", _varied(n, rng, False, 1000))
    npz = tmp_path / "in" / "w.npz"
    np.savez(npz, weight_per_bin=np.ones(100), bins=np.linspace(0, 100, 101))
    out = tmp_path / "graphs"
    graphs.run(cfg, "signal", sig, out, npvs_file=npz)
    graphs.run(cfg, "background", bkg, out)
    return out


@pytest.fixture
def small(load_training, training_user):
    load_training.defaults["gnn_training"].update(epochs=3, batch_size=4)
    return load_training(training_user)


def _train(tmp_path, cfg, data_base, out="model"):
    from tau3mu_gnn.gnn_training import train_gnn
    from tau3mu_gnn.pipelines.training.stages.graphs import model_type
    exp = train_gnn.make_experiment(cfg.comet, "test", disabled=True)
    return train_gnn.train(data_base, model_type(cfg), "W", cfg.gnn_training, cfg.graphs.unsculpting_mode,
                           "toy_model", tmp_path / out, exp, device=torch.device("cpu"))


def test_train_writes_model_files(tmp_path, small):
    import json
    from tau3mu_gnn.architectures.GNN_archs import get_arch_configs
    s = _train(tmp_path, small, _toy_graphs(tmp_path, small))
    folder = tmp_path / "model" / "toy_model"
    assert s["model_dir"] == folder
    for f in ("toy_model.pt", "Normalization.pkl", "arch_config.json"):
        assert (folder / f).is_file(), f
    assert json.loads((folder / "arch_config.json").read_text()) == json.loads(
        json.dumps(get_arch_configs("fl8_sd4", "W", 0.0, registry_path=None)[0]))
    assert s["epochs_run"] == 3 and 0 <= s["best_epoch"] < 3 and "auc_test" in s
    assert not (tmp_path / "output").exists()        # nothing in cwd/output


def test_best_epoch_kept(tmp_path, small, monkeypatch):
    import tau3mu_gnn.gnn_training.TrainingUtils as utrain
    fake = [0.9, 0.1, 0.5]                           # epoch 1 is the best of 3; no early stop

    class FakeStop(utrain.early_stop):
        def __call__(self, validation_loss, model, epoch, output_path, name):
            super().__call__(fake[epoch], model, epoch, output_path, name)
            if epoch == 1:
                FakeStop.best_state = {k: v.clone() for k, v in model.state_dict().items()}

    monkeypatch.setattr(utrain, "early_stop", FakeStop)
    s = _train(tmp_path, small, _toy_graphs(tmp_path, small))
    saved = torch.load(tmp_path / "model" / "toy_model" / "toy_model.pt", map_location="cpu")
    assert s["best_epoch"] == 1 and saved["epoch"] == 1 and s["early_stopped"] is False
    assert all(torch.equal(saved["model_state_dict"][k], v) for k, v in FakeStop.best_state.items())


def test_seed_reproducible_on_cpu(tmp_path, small):
    data_base = _toy_graphs(tmp_path, small)
    _train(tmp_path, small, data_base, out="a")
    _train(tmp_path, small, data_base, out="b")
    a = torch.load(tmp_path / "a" / "toy_model" / "toy_model.pt", map_location="cpu")["model_state_dict"]
    b = torch.load(tmp_path / "b" / "toy_model" / "toy_model.pt", map_location="cpu")["model_state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_model_name(load_training, training_user):
    from tau3mu_gnn.pipelines.training.stages.gnn_train import model_name
    assert model_name(load_training(training_user)) == "toy_train_W_fl_8_sd_4_GINE_delete_muons"


def test_stage_missing_graphs(tmp_path, small):
    from tau3mu_gnn.pipelines.training.stages import gnn_train
    with pytest.raises(FileNotFoundError, match="signal_W"):
        gnn_train.run(small, tmp_path / "empty", tmp_path / "out", device=torch.device("cpu"), comet_disabled=True)


def test_stage_model_loads_in_analysis(tmp_path, load_training, training_user):
    import json
    from tau3mu_gnn.gnn_training import model_loading
    from tau3mu_gnn.graph.ConeGraphProducer import ConeGraphProducer
    from tau3mu_gnn.pipelines.training.stages import gnn_train
    load_training.defaults["gnn_training"].update(epochs=2, batch_size=4)
    cfg = load_training(training_user)
    data_base = _toy_graphs(tmp_path, cfg)
    out = gnn_train.run(cfg, data_base, tmp_path / "gnn_train", device=torch.device("cpu"), comet_disabled=True)
    name = gnn_train.model_name(cfg)
    assert json.loads(out["summary"].read_text())["model_name"] == name
    model, normalizer = model_loading.load_model(tmp_path / "gnn_train", name, "GINE", "final_layer",
                                                 "skimmed_dzd0_cone_size_0.5_knn_10", "W", torch.device("cpu"))
    assert not model.training and {"means_x", "stds_graph"} <= set(normalizer)
    from torch_geometric.data import Batch
    g = ConeGraphProducer(str(data_base / "signal_W"), next((data_base / "signal_W" / "processed").glob("W_decay*.pt")).stem)[0]
    with torch.no_grad():
        score = model(Batch.from_data_list([g]))
    assert torch.isfinite(score).all()


def test_nan_validation_loss_stops_at_once(tmp_path, small, monkeypatch):
    import tau3mu_gnn.gnn_training.TrainingUtils as utrain
    real = utrain.distance_corr

    def nan_in_validation(*args, **kwargs):          # NaN only in the validation pass (no grad)
        return torch.tensor(float("nan")) if not torch.is_grad_enabled() else real(*args, **kwargs)

    monkeypatch.setattr(utrain, "distance_corr", nan_in_validation)
    with pytest.raises(FloatingPointError, match="validation loss"):
        _train(tmp_path, small, _toy_graphs(tmp_path, small))


def test_old_cli_comet_settings_for_any_channel():
    from tau3mu_gnn.scripts.Train_TauMuNet import comet_settings
    for channel in ("W", "HF"):
        c = comet_settings(channel)
        assert c["workspace"] and c["api_key"]


class RecordingExperiment:
    """Records the Comet calls of the training."""

    def __init__(self):
        self.metrics = []

    def log_metric(self, name, value, step=None, epoch=None, **kw):
        self.metrics.append((name, step, epoch))

    def log_parameters(self, *a, **kw):
        pass

    def set_name(self, *a, **kw):
        pass


def test_metrics_logged_with_step(tmp_path, small):
    """Comet charts use the step as x-axis (without it: 'No data'). step = optimizer steps so far (training
    batches), epoch = the epoch; one value per metric and epoch, logged at the end of the epoch."""
    from tau3mu_gnn.gnn_training import train_gnn
    from tau3mu_gnn.pipelines.training.stages.graphs import model_type
    exp = RecordingExperiment()
    train_gnn.train(_toy_graphs(tmp_path, small), model_type(small), "W", small.gnn_training,
                    small.graphs.unsculpting_mode, "toy_model", tmp_path / "m", exp, device=torch.device("cpu"))
    assert exp.metrics and all(isinstance(step, int) for _, step, _ in exp.metrics)
    assert {e for _, _, e in exp.metrics} == {0, 1, 2}
    step_of = {epoch: step for _, step, epoch in exp.metrics}
    assert all(len({s for _, s, e in exp.metrics if e == ep}) == 1 for ep in step_of)   # one step per epoch
    per_epoch = step_of[0]                                    # batches per epoch (same every epoch)
    assert per_epoch > 1 and step_of == {0: per_epoch, 1: 2 * per_epoch, 2: 3 * per_epoch}
