"""Frozen models carry their own settings: <model_dir>/<model_name>/model_settings.json (same layout as the
settings of a training gnn manifest). analysis_defaults only gives the paths."""
import json

import pytest
import yaml

from conftest import TOY_MODEL, TOY_MODEL_SETTINGS
from tau3mu_gnn.pipelines.common.config import ConfigError


def _settings_path(fake_repo):
    return fake_repo / "models" / "W" / "gnn" / TOY_MODEL / "model_settings.json"


def _write_settings(fake_repo, **changes):
    s = json.loads(json.dumps(TOY_MODEL_SETTINGS))
    for key, value in changes.items():
        section, name = key.split(".")
        s[section][name] = value
    _settings_path(fake_repo).write_text(json.dumps(s))


def _problems(e):
    return "\n".join(e.value.problems)


def test_frozen_settings_come_from_the_model_folder(load, user, fake_repo):
    _write_settings(fake_repo, **{"graphs.bdt_preselection_cut": 0.7})
    gnn = load(user).gnn
    assert gnn.bdt_preselection_cut == 0.7
    assert (gnn.conv_type, gnn.gnn_mode, gnn.unsculpting_mode) == ("GINE", "final_layer", "delete_muons")
    assert gnn.graph_type == "skimmed_dzd0_cone_size_0.5_knn_10"


def test_frozen_settings_file_missing(load, user, fake_repo):
    _settings_path(fake_repo).unlink()
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "model_settings.json" in _problems(e)


def test_frozen_settings_graph_must_match_inference(load, user, fake_repo):
    _write_settings(fake_repo, **{"graphs.k": 5})
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "graphs.k = 5" in _problems(e)


def test_frozen_settings_channel_must_match(load, user, fake_repo):
    s = json.loads(json.dumps(TOY_MODEL_SETTINGS))
    s["channel"] = "HF"
    _settings_path(fake_repo).write_text(json.dumps(s))
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "trained for channel 'HF', this analysis has channel 'W'" in _problems(e)


def test_frozen_unsculpting_mode_only_what_inference_reproduces(load, user, fake_repo):
    _write_settings(fake_repo, **{"graphs.unsculpting_mode": "permutation"})
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "unsculpting_mode" in _problems(e)


@pytest.mark.parametrize("key", ["conv_type", "gnn_mode", "unsculpting_mode", "graph_type", "bdt_preselection_cut"])
def test_defaults_no_longer_take_model_settings(load, user, fake_repo, key):
    path = fake_repo / "configs" / "analysis_defaults_W.yaml"
    d = yaml.safe_load(path.read_text())
    d["gnn"][key] = "x"
    path.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError) as e:
        load(user)
    assert f"gnn.{key}" in _problems(e)


def test_real_frozen_w_model_settings():
    from tau3mu_gnn.pipelines.common.config import REPO_ROOT
    s = json.loads((REPO_ROOT / "models/W/gnn/W_raw_Nano_AOD_fixed_W_fl_8_sd_4_GINE_delete_muons/model_settings.json").read_text())
    assert s == {"channel": "W",
                 "graphs": {"bdt_preselection_cut": 0.8, "connection": "knn", "k": 10, "cone_size": 0.5,
                            "use_displacement": True, "unsculpting_mode": "delete_muons"},
                 "gnn_training": {"conv_type": "GINE", "gnn_mode": "final_layer"}}
