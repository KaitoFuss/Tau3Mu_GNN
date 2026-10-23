"""Analysis key models: frozen (default) or a training gnn_vsN (step 12h)."""
import json
import shutil

import pytest

from tau3mu_gnn.pipeline.config import REPO_ROOT, ConfigError

NAME = "toy_train_W_fl_8_sd_4_GINE_delete_muons"
FROZEN = REPO_ROOT / "models" / "W" / "gnn" / "new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2"
GRAPHS = {"bdt_preselection_cut": 0.75, "max_events": 600000, "connection": "knn", "k": 10, "cone_size": 0.5,
          "use_displacement": True, "unsculpting_mode": "delete_muons", "shuffle_seed": 1986}
GNN_TRAINING = {"conv_type": "GINE", "gnn_mode": "final_layer", "arch": "fl8_sd4"}


def make_training(tmp_path, *, gnn_phase="gnn", complete=True, drop=None):
    """<tmp>/out/toy_train/{bdt_vs1,gnn_vs1} with manifests and model files (copies of the frozen model,
    so load_model works). drop: output key whose file is not written. Returns the gnn_vs1 path."""
    base = tmp_path / "out" / "toy_train"
    bdt, gnn = base / "bdt_vs1", base / "gnn_vs1"
    (bdt / "bdt_train").mkdir(parents=True)
    (bdt / "bdt_train" / "ISO_classifiers.pck").write_bytes(b"x")
    (bdt / "manifest.json").write_text(json.dumps(
        {"phase": "bdt", "upstream": None, "outputs": {"classifier": "bdt_train/ISO_classifiers.pck"}}))
    folder = gnn / "gnn_train" / NAME
    folder.mkdir(parents=True)
    files = {"model": f"{NAME}.pt", "normalization": "Normalization.pkl", "arch_config": "arch_config.json"}
    frozen = {"model": FROZEN / f"{FROZEN.name}.pt", "normalization": FROZEN / "Normalization.pkl",
              "arch_config": FROZEN / "arch_config.json"}
    for key, fname in files.items():
        if key != drop:
            shutil.copyfile(frozen[key], folder / fname)
    if complete:
        (gnn / "manifest.json").write_text(json.dumps({
            "phase": gnn_phase, "upstream": "bdt_vs1",
            "outputs": {k: f"gnn_train/{NAME}/{f}" for k, f in files.items()},
            "settings": {"graphs": GRAPHS, "gnn_training": GNN_TRAINING}}))
    return gnn


def test_models_frozen_default(load, user):
    cfg = load(user)
    assert cfg.models == "frozen" and cfg.gnn.model_name == "toy_model" and cfg.gnn.bdt_preselection_cut == 0.8


def test_models_training_version(load, user, tmp_path):
    gnn = make_training(tmp_path)
    user["models"] = str(gnn)
    cfg = load(user)
    assert cfg.models == str(gnn)
    assert cfg.bdt.classifier == gnn.parent / "bdt_vs1" / "bdt_train" / "ISO_classifiers.pck"
    assert (cfg.gnn.model_dir, cfg.gnn.model_name) == (gnn / "gnn_train", NAME)
    assert cfg.gnn.bdt_preselection_cut == 0.75
    assert cfg.gnn.graph_type == "skimmed_dzd0_cone_size_0.5_knn_10"
    assert (cfg.gnn.conv_type, cfg.gnn.gnn_mode, cfg.gnn.unsculpting_mode) == ("GINE", "final_layer", "delete_muons")


def test_models_wrong_phase(load, user, tmp_path):
    gnn = make_training(tmp_path)
    user["models"] = str(gnn.parent / "bdt_vs1")
    with pytest.raises(ConfigError, match="gnn_vs"):
        load(user)


def test_models_wrong_manifest_phase(load, user, tmp_path):
    user["models"] = str(make_training(tmp_path, gnn_phase="benchmark"))
    with pytest.raises(ConfigError, match="phase"):
        load(user)


def test_models_incomplete_training(load, user, tmp_path):
    gnn = make_training(tmp_path, complete=False)
    user["models"] = str(gnn)
    with pytest.raises(ConfigError, match="manifest"):
        load(user)


def test_models_missing_file(load, user, tmp_path):
    user["models"] = str(make_training(tmp_path, drop="normalization"))
    with pytest.raises(ConfigError, match="Normalization.pkl"):
        load(user)


def test_models_root_url(load, user, tmp_path, monkeypatch):
    from tau3mu_gnn.pipeline import storage
    gnn = make_training(tmp_path)
    url = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/toy/out/toy_train/gnn_vs1"
    real = storage.local_path

    def fake(path):
        path = str(path)
        if path.startswith("root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/toy/"):
            return tmp_path / path.split("/cms/toy/", 1)[1]
        return real(path)

    monkeypatch.setattr(storage, "local_path", fake)
    user["models"] = url
    cfg = load(user)
    assert cfg.gnn.model_dir == gnn / "gnn_train" and cfg.models == url


def test_models_loadable_by_gnn_stage(load, user, tmp_path):
    import torch
    from tau3mu_gnn.gnn_training import model_loading
    user["models"] = str(make_training(tmp_path))
    g = load(user).gnn
    model, normalizer = model_loading.load_model(g.model_dir, g.model_name, g.conv_type, g.gnn_mode,
                                                 g.graph_type, "W", torch.device("cpu"))
    assert not model.training and "means_x" in normalizer


def test_config_used_records_models(load, user, tmp_path):
    import yaml
    from tau3mu_gnn.pipeline import runs
    user["paths"]["output_base"] = str(tmp_path / "res")
    cfg = load(user)
    version = runs.create_version(cfg, "phase_a")
    assert yaml.safe_load((tmp_path / "res" / "toy_run" / "phase_a_vs1" / "config_used.yaml").read_text())["models"] \
        == "frozen" and version.endswith("phase_a_vs1")


def test_models_in_settings(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import phase_settings, settings_diff
    frozen = load(user)
    user["models"] = str(make_training(tmp_path))
    trained = load(user)
    for phase in ("phase_a", "phase_b", "phase_c"):
        assert phase_settings(frozen, phase)["models"] == "frozen"
    diff = " ".join(settings_diff(phase_settings(frozen, "phase_b"), phase_settings(trained, "phase_b")))
    assert "models" in diff and "gnn.model_dir" in diff and "bdt.classifier" in diff
