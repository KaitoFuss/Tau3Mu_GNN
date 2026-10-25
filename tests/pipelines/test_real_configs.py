from tau3mu_gnn.pipelines.common.config import REPO_ROOT
from tau3mu_gnn.pipelines.analysis.config import load_config


def test_example_w_matches_production_settings(monkeypatch):
    monkeypatch.setenv("USER", "tester")
    cfg = load_config(REPO_ROOT / "configs" / "example_W.yaml")
    assert cfg.channel == "W"
    assert cfg.output_dir == ("root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/"
                              "tester/tau3mu_pipeline/kaito_W_reference")
    assert cfg.gnn.model_name == "new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2"
    assert cfg.gnn.conv_type == "GINE"
    assert cfg.gnn.gnn_mode == "final_layer"
    assert cfg.gnn.unsculpting_mode == "delete_muons"
    assert cfg.gnn.graph_type == "skimmed_dzd0_cone_size_0.5_knn_10"
    assert cfg.gnn.bdt_preselection_cut == 0.8
    assert cfg.bdt.classifier == REPO_ROOT / "models/W/bdt/ISO_classifiers_24feb2026_v0.pck"
    rs = cfg.resonance_skim
    assert (rs.vtx_chi2_max, rs.mass_range, rs.veto_masses, rs.veto_half_width, rs.pf_dr_max) == \
        (20, (1.0, 3.0), (0.78266, 1.019461), 0.040, 1.0)
    assert cfg.skim.dz_max == 0.5
    assert cfg.skim.cos_alpha_2d_min == 0.5
    assert cfg.limit.gnn_score_cut == (3.9,)
    assert cfg.limit.signal_weight == 0.000019
    assert cfg.limit.tau_mass == 1.78
    assert cfg.limit.mass_window == 0.060
    assert cfg.limit.fit_range == (1.62, 1.98)
    assert cfg.limit.cl == 0.90
    assert cfg.paths.log_dir == "/work/tester/SVJ_GNN/tau3mu_logs"
    jobs = cfg.slurm.jobs
    assert cfg.slurm.account == "t3"
    assert (jobs["skim"].partition, jobs["skim"].mem, jobs["skim"].time, jobs["skim"].cpus) == \
        ("long", "128G", "1-00:00:00", 4)
    assert (jobs["gnn_background"].partition, jobs["gnn_background"].time) == ("long", "2-00:00:00")
    assert jobs["limit"].time == "0-01:30:00" and jobs["summary"].partition == "short"


def test_example_w_raw_differs_only_in_inputs(monkeypatch):
    from tau3mu_gnn.pipelines.analysis.manifest import config_dict

    monkeypatch.setenv("USER", "tester")
    skimmed = config_dict(load_config(REPO_ROOT / "configs" / "example_W.yaml"))
    raw = config_dict(load_config(REPO_ROOT / "configs" / "example_W_raw.yaml"))
    assert raw["inputs"]["already_resonance_skimmed"] is False
    assert all("/manzoni/tau3mu_gnn/" in p for p in raw["inputs"]["signal"] + raw["inputs"]["background"])
    assert raw["analysis_name"] != skimmed["analysis_name"]
    differ = sorted(k for k in raw if raw[k] != skimmed[k])
    assert differ == ["analysis_name", "inputs"]
