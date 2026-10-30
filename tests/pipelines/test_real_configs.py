import pytest

from tau3mu_gnn.pipelines.common.config import REPO_ROOT
from tau3mu_gnn.pipelines.analysis.config import load_config


def test_example_w_raw_matches_production_settings(monkeypatch):
    monkeypatch.setenv("USER", "tester")
    cfg = load_config(REPO_ROOT / "configs" / "example_W_raw.yaml")
    assert cfg.channel == "W"
    assert cfg.inputs.already_resonance_skimmed is False
    assert all("/manzoni/tau3mu_gnn/" in p for p in cfg.inputs.signal + cfg.inputs.background)
    assert cfg.output_dir == ("root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/"
                              "tester/tau3mu_pipeline/kaito_W_raw")
    assert cfg.gnn.model_name == "W_raw_Nano_AOD_fixed_W_fl_8_sd_4_GINE_delete_muons"
    assert cfg.gnn.conv_type == "GINE"
    assert cfg.gnn.gnn_mode == "final_layer"
    assert cfg.gnn.unsculpting_mode == "delete_muons"
    assert cfg.gnn.graph_type == "skimmed_dzd0_cone_size_0.5_knn_10"
    assert cfg.gnn.bdt_preselection_cut == 0.8
    assert cfg.bdt.classifier == REPO_ROOT / "models/W/bdt/ISO_classifiers_W_raw_Nano_AOD_fixed.pck"
    rs = cfg.resonance_skim
    assert (rs.vtx_chi2_max, rs.mass_range, rs.veto_masses, rs.veto_half_width, rs.pf_dr_max) == \
        (20, (1.0, 3.0), (0.78266, 1.019461), 0.040, 1.0)
    assert cfg.skim.dz_max == 0.5
    assert cfg.skim.cos_alpha_2d_min == 0.5
    assert cfg.limit.gnn_score_cut == (3.5,)
    assert cfg.limit.signal_weight == 0.000019
    assert cfg.limit.tau_mass == 1.78
    assert cfg.limit.mass_window == 0.060
    assert cfg.limit.fit_range == (1.6, 2.0)
    assert cfg.limit.cl == 0.90
    assert cfg.paths.log_dir == "/work/tester/SVJ_GNN/tau3mu_logs"
    jobs = cfg.slurm.jobs
    assert cfg.slurm.account == "t3"
    assert (jobs["skim"].partition, jobs["skim"].mem, jobs["skim"].time, jobs["skim"].cpus) == \
        ("long", "256G", "1-00:00:00", 4)
    assert (jobs["gnn_background"].partition, jobs["gnn_background"].time) == ("long", "2-00:00:00")
    assert jobs["limit"].time == "0-01:30:00" and jobs["summary"].partition == "short"


HF_GNN_CUTS = [0.25 * i for i in range(13)]   # 0.0 to 3.0, step 0.25 (HF GNN: above logit 3 almost no signal)


def _yaml(name):
    import yaml
    return yaml.safe_load((REPO_ROOT / "configs" / name).read_text())


def test_hf_analysis_defaults_differ_from_w_only_in_models():
    w, hf = _yaml("analysis_defaults_W.yaml"), _yaml("analysis_defaults_HF.yaml")
    assert hf["gnn"].pop("model_dir") == "models/HF/gnn" and w["gnn"].pop("model_dir") == "models/W/gnn"
    assert hf["bdt"].pop("classifier").startswith("models/HF/") and w["bdt"].pop("classifier")
    assert hf["gnn"].pop("model_name") != w["gnn"].pop("model_name")
    assert hf["limit"]["fit_range"] == [1.6, 2.0]   # E4: one mass range for fit and workspace
    assert hf == w


def test_example_hf_needs_models(tmp_path, monkeypatch):
    import yaml
    from tau3mu_gnn.pipelines.common.config import ConfigError

    monkeypatch.setenv("USER", "tester")
    user = _yaml("example_HF.yaml")
    user.pop("models")
    path = tmp_path / "hf.yaml"
    path.write_text(yaml.safe_dump(user))
    with pytest.raises(ConfigError) as e:
        load_config(path, defaults_dir=REPO_ROOT / "configs")
    text = "\n".join(e.value.problems)
    assert "frozen model file not found" in text and "models/HF/" in text


def test_example_hf_cut_scan_and_inputs():
    hf = _yaml("example_HF.yaml")
    assert hf["channel"] == "HF"
    assert hf["limit"]["gnn_score_cut"] == HF_GNN_CUTS
    assert hf["limit"]["signal_weight"] == 0.000152
    assert hf["inputs"] == _yaml("example_training_HF.yaml")["inputs"]
    assert hf["analysis_name"] == "HF_analysis"
    assert hf["paths"]["output_base"].endswith("/tau3mu_pipeline/analysis")
    assert hf["models"].endswith("/tau3mu_pipeline/training/HF_training/gnn_vs1")
