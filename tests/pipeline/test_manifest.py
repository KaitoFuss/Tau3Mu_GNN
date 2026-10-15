import json

import pytest


def test_config_dict_is_plain_and_repo_relative(load, user):
    import yaml
    from tau3mu_gnn.pipeline.manifest import config_dict

    d = config_dict(load(user))
    assert d["bdt"]["classifier"] == "models/W/bdt/toy.pck"
    assert d["gnn"]["model_dir"] == "models/W/gnn"
    assert d["limit"]["gnn_score_cut"] == [3.9] and d["limit"]["fit_range"] == [1.62, 1.98]
    assert d["slurm"]["jobs"]["limit"] == {"partition": "standard", "mem": "4G", "time": "0-01:00:00", "cpus": 1}
    assert d["paths"] == {"output_base": "/tmp/toy_out", "log_dir": "/tmp/toy_logs", "conda_env": None}
    assert yaml.safe_load(yaml.safe_dump(d)) == d and json.loads(json.dumps(d)) == d


def test_phase_settings_sections(load, user):
    from tau3mu_gnn.pipeline.manifest import phase_settings

    cfg = load(user)
    assert list(phase_settings(cfg, "phase_a")) == ["channel", "inputs", "branch_map", "skim", "bdt"]
    assert list(phase_settings(cfg, "phase_b"))[-1] == "gnn"
    assert list(phase_settings(cfg, "phase_c"))[-2:] == ["gnn", "limit"]


def test_settings_diff_dotted():
    from tau3mu_gnn.pipeline.manifest import settings_diff

    old = {"skim": {"dz_max": 0.5}, "inputs": {"signal": ["a.root"]}, "branch_map": {}}
    new = {"skim": {"dz_max": 0.4}, "inputs": {"signal": ["a.root"]}, "branch_map": {"x": "y"}}
    assert settings_diff(old, new) == ["branch_map.x: (not set) -> 'y'", "skim.dz_max: 0.5 -> 0.4"]
    assert settings_diff(old, old) == []


def test_write_and_read_manifest(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import read_manifest, write_manifest

    cfg = load(user)
    phase_dir = tmp_path / "toy_run" / "phase_b_vs2"
    (phase_dir).mkdir(parents=True)
    (phase_dir / "jobs.json").write_text(json.dumps({"eval": "125"}))
    data = write_manifest(cfg, phase_dir, "phase_b", "phase_a_vs1", {"gnn_signal": "3_gnn/s.root"})
    assert read_manifest(phase_dir) == data
    assert data["phase"] == "phase_b" and data["version"] == "phase_b_vs2" and data["upstream"] == "phase_a_vs1"
    assert data["outputs"] == {"gnn_signal": "3_gnn/s.root"} and data["missing"] == []
    assert data["slurm_jobs"] == {"eval": "125"}
    assert set(data["settings"]) == {"channel", "inputs", "branch_map", "skim", "bdt", "gnn"}
    assert data["git_commit"] and data["created"]


def test_read_manifest_missing(tmp_path):
    from tau3mu_gnn.pipeline.manifest import UpstreamError, read_manifest

    with pytest.raises(UpstreamError, match="phase_a_vs3"):
        read_manifest(tmp_path / "phase_a_vs3")


def upstream_a(cfg, tmp_path, files=("1_skim/signal.root",)):
    from tau3mu_gnn.pipeline.manifest import write_manifest

    d = tmp_path / "toy_run" / "phase_a_vs1"
    for rel in files:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("x")
    write_manifest(cfg, d, "phase_a", None, {"skim_signal": "1_skim/signal.root",
                                              "bdt_signal": "2_bdt/signal_with_bdt_scores.root"})
    return d


def test_check_upstream_missing_files(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import UpstreamError, check_upstream

    cfg = load(user)
    d = upstream_a(cfg, tmp_path)                      # 2_bdt file not written
    with pytest.raises(UpstreamError, match="2_bdt/signal_with_bdt_scores.root"):
        check_upstream(cfg, d, "phase_b")


def test_check_upstream_wrong_phase(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import UpstreamError, check_upstream

    cfg = load(user)
    d = upstream_a(cfg, tmp_path, files=("1_skim/signal.root", "2_bdt/signal_with_bdt_scores.root"))
    with pytest.raises(UpstreamError, match="phase_b"):
        check_upstream(cfg, d, "phase_c")              # phase C needs a phase_b folder


def test_check_upstream_unchanged_config_no_warning(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipeline.manifest import check_upstream

    files = ("1_skim/signal.root", "2_bdt/signal_with_bdt_scores.root")
    d = upstream_a(load(user), tmp_path, files=files)
    m = check_upstream(load(user), d, "phase_b")       # re-loaded config: tuples, absolute paths
    assert m["version"] == "phase_a_vs1"
    assert "other settings" not in caplog.text


def test_check_upstream_changed_settings_warns(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipeline.manifest import check_upstream

    files = ("1_skim/signal.root", "2_bdt/signal_with_bdt_scores.root")
    d = upstream_a(load(user), tmp_path, files=files)
    user["inputs"]["signal"] = ["/data/other_sig.root"]
    check_upstream(load(user), d, "phase_b")
    assert "other settings" in caplog.text and "inputs.signal" in caplog.text
