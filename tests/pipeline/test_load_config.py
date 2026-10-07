import dataclasses

import pytest

from conftest import TOY_MODEL
from tau3mu_gnn.pipeline.config import ConfigError, load_config


def problems_of(excinfo):
    return "\n".join(excinfo.value.problems)


def test_valid_config_loads(load, user, fake_repo):
    cfg = load(user)
    assert cfg.analysis_name == "toy_run"
    assert cfg.channel == "W"
    assert cfg.paths.output_dir == "/tmp/toy_out"
    assert cfg.inputs.signal == ("/data/sig.root",)
    assert cfg.inputs.background == ("/data/bkg_*.root",)
    assert cfg.inputs.already_resonance_skimmed is True
    assert cfg.limit.signal_weight == 1.9e-5
    assert cfg.limit.fit_range == (1.62, 1.98)
    assert cfg.gnn.model_name == TOY_MODEL
    assert cfg.gnn.model_dir == fake_repo / "models/W/gnn"
    assert cfg.bdt.classifier == fake_repo / "models/W/bdt/toy.pck"
    assert cfg.defaults_file == fake_repo / "configs/analysis_defaults_W.yaml"


def test_defaults_applied(load, user):
    cfg = load(user)
    assert cfg.inputs.tree_name == "Events"
    assert cfg.limit.gnn_score_cut == (3.9,)
    assert dict(cfg.branch_map) == {}


def test_score_cut_scalar_and_list(load, user):
    user["limit"]["gnn_score_cut"] = 3.75
    assert load(user).limit.gnn_score_cut == (3.75,)
    user["limit"]["gnn_score_cut"] = [3.5, 3.9]
    assert load(user).limit.gnn_score_cut == (3.5, 3.9)


def test_config_is_read_only(load, user):
    user["branch_map"] = {"PFCands_pt": "PFCand_pt"}
    cfg = load(user)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.limit.signal_weight = 1.0
    with pytest.raises(TypeError):
        cfg.branch_map["x"] = "y"


def test_missing_required_key(load, user):
    del user["limit"]["signal_weight"]
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "missing required key 'limit.signal_weight'" in problems_of(e)


def test_unknown_key_with_hint(load, user):
    user["limit"]["signal_weigth"] = user["limit"].pop("signal_weight")
    with pytest.raises(ConfigError) as e:
        load(user)
    text = problems_of(e)
    assert "unknown key 'limit.signal_weigth'" in text
    assert "did you mean 'limit.signal_weight'" in text


def test_frozen_key_in_user_file(load, user):
    user["gnn"] = {"model_name": "other_model"}
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "'gnn.model_name' is a frozen analysis setting" in problems_of(e)


def test_all_problems_reported_at_once(load, user):
    del user["analysis_name"]
    user["limit"]["signal_weight"] = 0
    user["inputs"]["already_resonance_skimmed"] = "yes please"
    with pytest.raises(ConfigError) as e:
        load(user)
    assert len(e.value.problems) == 3


def test_unsupported_channel_stops_before_defaults(load, user):
    user["channel"] = "HF"
    with pytest.raises(ConfigError) as e:
        load(user)
    text = problems_of(e)
    assert "must be one of ['W']" in text
    assert "analysis_defaults_HF" not in text


def test_empty_branch_map_header_is_empty_map(load):
    raw = """
analysis_name: toy_run
channel: W
paths:
  output_dir: /tmp/toy_out
inputs:
  signal: [/data/sig.root]
  background: [/data/bkg.root]
  already_resonance_skimmed: true
branch_map:
limit:
  signal_weight: 1.9e-5
"""
    assert dict(load(raw_text=raw).branch_map) == {}


def test_empty_section_header_reports_missing_keys(load):
    raw = """
analysis_name: toy_run
channel: W
paths:
inputs:
  signal: [/data/sig.root]
  background: [/data/bkg.root]
  already_resonance_skimmed: true
limit:
  signal_weight: 1.9e-5
"""
    with pytest.raises(ConfigError) as e:
        load(raw_text=raw)
    assert "missing required key 'paths.output_dir'" in problems_of(e)
    assert "unknown key" not in problems_of(e)


def test_missing_user_file(fake_repo):
    with pytest.raises(ConfigError) as e:
        load_config(fake_repo / "nope.yaml", defaults_dir=fake_repo / "configs")
    assert "cannot read config file" in problems_of(e)


def test_invalid_yaml(load):
    with pytest.raises(ConfigError) as e:
        load(raw_text="analysis_name: [unclosed\n")
    assert "not valid YAML" in problems_of(e)


def test_top_level_must_be_mapping(load):
    with pytest.raises(ConfigError) as e:
        load(raw_text="- a\n- b\n")
    assert "top level must be a mapping" in problems_of(e)


def test_missing_frozen_model_file(load, user, fake_repo):
    (fake_repo / f"models/W/gnn/{TOY_MODEL}/Normalization.pkl").unlink()
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "frozen model file not found" in problems_of(e)
    assert "Normalization.pkl" in problems_of(e)


def test_defaults_channel_must_match(load, user, fake_repo):
    path = fake_repo / "configs/analysis_defaults_W.yaml"
    path.write_text(path.read_text().replace("channel: W", "channel: HF"))
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "channel" in problems_of(e)
