import json

import pytest
import yaml


@pytest.fixture
def cfg(load_training, training_user, tmp_path):
    training_user["paths"]["output_base"] = str(tmp_path / "out")
    return load_training(training_user)


def make_bdt(cfg, files=("bdt/model.json",), manifest=True):
    from tau3mu_gnn.pipelines.training import manifest as tm

    path = tm.create_version(cfg, "bdt")
    from pathlib import Path
    for f in files:
        (Path(path) / f).parent.mkdir(parents=True, exist_ok=True)
        (Path(path) / f).write_text("x")
    if manifest:
        tm.write_manifest(cfg, path, "bdt", None, {"model": files[0]})
    return path


def test_create_version_writes_config_without_key(cfg):
    from pathlib import Path
    from tau3mu_gnn.pipelines.training.manifest import create_version

    path = create_version(cfg, "bdt")
    assert path.endswith("/toy_train/bdt_vs1")
    text = (Path(path) / "config_used.yaml").read_text()
    assert "toy-key" not in text
    d = yaml.safe_load(text)
    assert d["comet"] == {"workspace": "toy_ws", "api_key": "***"}
    assert d["gnn_training"]["windows"]["train"] == [0.0, 0.6]
    assert create_version(cfg, "bdt").endswith("bdt_vs2")


def test_config_dict_is_plain(cfg):
    from tau3mu_gnn.pipelines.training.manifest import config_dict

    d = config_dict(cfg)
    assert yaml.safe_load(yaml.safe_dump(d)) == d and json.loads(json.dumps(d)) == d
    assert d["slurm"]["jobs"]["gnn_train"]["gres"] == "gpu:1"


def test_phase_settings_sections(cfg):
    from tau3mu_gnn.pipelines.training.manifest import phase_settings

    assert "graphs" not in phase_settings(cfg, "bdt") and "bdt_training" in phase_settings(cfg, "bdt")
    assert {"npvs", "graphs", "gnn_training"} <= set(phase_settings(cfg, "gnn"))
    assert {"benchmark", "limit"} <= set(phase_settings(cfg, "benchmark"))
    assert "graphs" not in phase_settings(cfg, "benchmark")
    assert "comet" not in phase_settings(cfg, "gnn")


def test_pick_upstream_highest_complete(cfg):
    from pathlib import Path
    from tau3mu_gnn.pipelines.common.runs import UpstreamError
    from tau3mu_gnn.pipelines.training.manifest import pick_upstream

    with pytest.raises(UpstreamError):
        pick_upstream(cfg, "gnn")
    make_bdt(cfg)
    second = make_bdt(cfg)
    make_bdt(cfg, manifest=False)
    assert pick_upstream(cfg, "gnn") == second
    assert pick_upstream(cfg, "benchmark", "bdt_vs1").endswith("bdt_vs1")
    with pytest.raises(ValueError):
        pick_upstream(cfg, "bdt")


def test_manifest_roundtrip_and_check_upstream(cfg, load_training, training_user, caplog):
    training_user["paths"]["output_base"] = cfg.paths.output_base
    import logging
    from pathlib import Path
    from tau3mu_gnn.pipelines.common.manifest import UpstreamError, read_manifest
    from tau3mu_gnn.pipelines.training.manifest import check_upstream, write_manifest

    bdt = make_bdt(cfg)
    m = read_manifest(bdt)
    assert m["phase"] == "bdt" and m["version"] == "bdt_vs1" and "graphs" not in m["settings"]
    assert "toy-key" not in (Path(bdt) / "manifest.json").read_text()
    assert check_upstream(cfg, bdt, "gnn")["phase"] == "bdt"

    gnn = Path(cfg.output_dir) / "gnn_vs1"
    gnn.mkdir()
    write_manifest(cfg, gnn, "gnn", "bdt_vs1", {})
    with pytest.raises(UpstreamError, match="gnn folder"):
        check_upstream(cfg, gnn, "benchmark")

    (Path(bdt) / "bdt/model.json").unlink()
    with pytest.raises(UpstreamError, match="missing"):
        check_upstream(cfg, bdt, "gnn")
    (Path(bdt) / "bdt/model.json").write_text("x")

    load_training.defaults["bdt_training"]["max_depth"] = 3
    changed = load_training(training_user)
    with caplog.at_level(logging.WARNING):
        check_upstream(changed, bdt, "benchmark")
    assert "bdt_training.max_depth: 8 -> 3" in caplog.text
