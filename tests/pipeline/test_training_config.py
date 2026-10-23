import copy

import pytest

from conftest import TOY_TRAINING_DEFAULTS
from tau3mu_gnn.pipeline.config import ConfigError


def test_loads_toy(load_training, training_user):
    cfg = load_training(training_user)
    assert cfg.training_name == "toy_train"
    assert cfg.graphs.max_events == 600000 and cfg.bdt_training.max_events_per_class == 100000
    assert cfg.graphs.bdt_preselection_cut == 0.8
    assert cfg.benchmark.bdt_score_cut == (0.98, 0.99)
    assert cfg.skim.dz_max == 0.5 and cfg.resonance_skim.vtx_chi2_max == 20   # from analysis defaults
    assert (cfg.limit.tau_mass, cfg.limit.mass_window, cfg.limit.signal_weight) == (1.78, 0.06, 1.9e-5)
    assert cfg.slurm.jobs["gnn_train"].account == "gpu_gres"
    assert cfg.output_dir == "/tmp/toy_out/toy_train"


def test_graph_type_name(load_training, training_user):
    assert load_training(training_user).graphs.graph_type == "skimmed_dzd0_cone_size_0.5_knn_10"


def test_user_overrides_event_limits(load_training, training_user):
    training_user["graphs"]["max_events"] = 1000
    training_user["bdt_training"] = {"max_events_per_class": 50}
    cfg = load_training(training_user)
    assert (cfg.graphs.max_events, cfg.bdt_training.max_events_per_class) == (1000, 50)


def test_frozen_training_key_in_user_file(load_training, training_user):
    training_user["gnn_training"] = {"epochs": 5}
    with pytest.raises(ConfigError, match=r"'gnn_training.epochs' is a frozen training setting"):
        load_training(training_user)


def test_frozen_analysis_key_in_training_user_file(load_training, training_user):
    training_user["skim"] = {"dz_max": 1.0}
    with pytest.raises(ConfigError, match=r"'skim.dz_max' is a frozen analysis setting"):
        load_training(training_user)


@pytest.mark.parametrize("change", [
    lambda u: u["graphs"].update(bdt_preselection_cut=1.0),
    lambda u: u["benchmark"].update(bdt_score_cut=[0.0, 0.99]),
    lambda u: u["benchmark"].update(bdt_score_cut=[1.2]),
    lambda u: u["benchmark"].update(bdt_score_cut=[0.99, 0.99]),
])
def test_cut_values_must_be_inside_0_1(load_training, training_user, change):
    change(training_user)
    with pytest.raises(ConfigError):
        load_training(training_user)


def test_missing_required_user_keys(load_training, training_user):
    for key in ("graphs", "benchmark", "limit"):
        training_user.pop(key)
    with pytest.raises(ConfigError) as e:
        load_training(training_user)
    text = str(e.value)
    for key in ("graphs.bdt_preselection_cut", "benchmark.bdt_score_cut", "limit.signal_weight"):
        assert f"missing required key '{key}'" in text


def test_missing_analysis_defaults_is_reported(load_training, training_user, fake_repo):
    (fake_repo / "configs" / "analysis_defaults_W.yaml").unlink()
    with pytest.raises(ConfigError, match="analysis_defaults_W.yaml"):
        load_training(training_user)


def test_bad_training_defaults_all_reported(load_training, training_user):
    defaults = copy.deepcopy(TOY_TRAINING_DEFAULTS)
    defaults["gnn_training"]["class_fraction"] = [0.6, 0.6]
    defaults["graphs"]["connection"] = "grid"
    load_training.defaults = defaults
    with pytest.raises(ConfigError) as e:
        load_training(training_user)
    assert "'gnn_training.class_fraction'" in str(e.value) and "'graphs.connection'" in str(e.value)


def test_kfold_needs_two_folds(load_training, training_user):
    load_training.defaults["bdt_training"]["kfold"] = 1
    with pytest.raises(ConfigError, match="bdt_training.kfold"):
        load_training(training_user)


def test_comet_key_not_in_repr(load_training, training_user):
    assert "toy-key" not in repr(load_training(training_user))
