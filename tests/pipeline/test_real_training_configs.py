from tau3mu_gnn.pipeline.config import REPO_ROOT, load_config
from tau3mu_gnn.pipeline.training.config import load_training_config


def _load(monkeypatch):
    monkeypatch.setenv("USER", "kfuss")
    return (load_training_config(REPO_ROOT / "configs" / "example_training_W.yaml"),
            load_config(REPO_ROOT / "configs" / "example_W.yaml"))


def test_example_training_w_loads_the_recipe(monkeypatch):
    cfg, _ = _load(monkeypatch)
    assert cfg.sideband_half_width == 0.08
    assert (cfg.graphs.max_events, cfg.graphs.bdt_preselection_cut) == (600000, 0.8)
    assert cfg.graphs.shuffle_seed == 1986
    assert (cfg.bdt_training.max_depth, cfg.bdt_training.n_estimators, cfg.bdt_training.kfold) == (8, 1000, 6)
    assert cfg.bdt_training.max_events_per_class == 100000
    assert (cfg.gnn_training.learning_rate, cfg.gnn_training.batch_size, cfg.gnn_training.epochs) == (0.0016, 128, 1000)
    assert cfg.gnn_training.arch == "fl8_sd4" and cfg.gnn_training.conv_type == "GINE"
    assert len(cfg.npvs.var_bins) == 47 and cfg.npvs.var_bins[-2:] == (65.0, 100.0)
    j = cfg.slurm.jobs["gnn_train"]
    assert (j.partition, j.account, j.gres) == ("gpu", "gpu_gres", "gpu:1")
    assert cfg.comet.api_key  # present, value not asserted


def test_training_graph_type_matches_analysis_defaults(monkeypatch):
    t, a = _load(monkeypatch)
    assert t.graphs.graph_type == a.gnn.graph_type
    assert t.graphs.unsculpting_mode == a.gnn.unsculpting_mode
    assert (t.gnn_training.conv_type, t.gnn_training.gnn_mode) == (a.gnn.conv_type, a.gnn.gnn_mode)
    assert t.graphs.bdt_preselection_cut == a.gnn.bdt_preselection_cut
    assert t.skim == a.skim and t.resonance_skim == a.resonance_skim


def test_training_inputs_match_example_w(monkeypatch):
    t, a = _load(monkeypatch)
    assert t.inputs == a.inputs and t.paths == a.paths
