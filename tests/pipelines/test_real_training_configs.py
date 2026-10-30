from tau3mu_gnn.pipelines.common.config import REPO_ROOT
from tau3mu_gnn.pipelines.analysis.config import load_config
from tau3mu_gnn.pipelines.training.config import load_training_config


def _load(monkeypatch):
    monkeypatch.setenv("USER", "kfuss")
    return (load_training_config(REPO_ROOT / "configs" / "example_training_W.yaml"),
            load_config(REPO_ROOT / "configs" / "example_W_raw.yaml"))  # the runnable analysis example (E1)


def test_example_training_w_loads_the_recipe(monkeypatch):
    cfg, _ = _load(monkeypatch)
    assert cfg.sideband_half_width == 0.08
    assert (cfg.graphs.max_events, cfg.graphs.bdt_preselection_cut) == (600000, 0.8)
    assert cfg.graphs.shuffle_seed == 1986
    assert (cfg.bdt_training.max_depth, cfg.bdt_training.n_estimators, cfg.bdt_training.kfold) == (8, 1000, 6)
    assert cfg.bdt_training.max_events_per_class == 100000
    assert (cfg.gnn_training.learning_rate, cfg.gnn_training.batch_size, cfg.gnn_training.epochs) == (0.0016, 128, 1000)
    assert cfg.gnn_training.arch == "fl8_sd4" and cfg.gnn_training.conv_type == "GINE"
    assert (cfg.gnn_training.early_stopping_patience, cfg.gnn_training.seed) == (50, 1986)
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


HF_BENCHMARK_CUTS = [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]


def test_hf_training_defaults_equal_w():
    import yaml
    w, hf = (yaml.safe_load((REPO_ROOT / "configs" / f"training_defaults_{c}.yaml").read_text()) for c in ("W", "HF"))
    assert hf == w


def test_example_training_hf(monkeypatch):
    monkeypatch.setenv("USER", "kfuss")
    hf = load_training_config(REPO_ROOT / "configs" / "example_training_HF.yaml")
    w = load_training_config(REPO_ROOT / "configs" / "example_training_W.yaml")
    assert hf.channel == "HF" and hf.training_name == "HF_training"
    assert hf.paths.output_base.endswith("/tau3mu_pipeline/training")
    assert [str(p) for p in hf.inputs.signal] == [
        "/eos/user/k/kfuss/tau3mu_inputs/kfuss/tau3mu_gnn_round2/data/signal_HF/old_data/"
        "ds_large_signal_skimmed_no_resonances.root"]
    assert [str(p) for p in hf.inputs.background] == [
        "/eos/user/k/kfuss/tau3mu_inputs/kfuss/tau3mu_gnn_round2/data/background_HF/old_data/"
        "data_skimmed_no_resonances_part*.root"]
    assert hf.inputs.already_resonance_skimmed is True
    assert hf.limit.signal_weight == 0.000152
    assert list(hf.benchmark.bdt_score_cut) == HF_BENCHMARK_CUTS
    for section in ("gnn_training", "graphs", "bdt_training", "npvs", "resonance_skim", "skim"):
        assert getattr(hf, section) == getattr(w, section), section


def test_skim_job_memory_256g(monkeypatch):
    # raw W training (job 854069): MaxRSS 127.99G with 128G requested
    cfg, _ = _load(monkeypatch)
    assert cfg.slurm.jobs["skim"].mem == "256G"
