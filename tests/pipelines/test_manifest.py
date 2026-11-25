import json

import pytest


def test_config_dict_is_plain_and_repo_relative(load, user):
    import yaml
    from tau3mu_gnn.pipelines.analysis.manifest import config_dict

    d = config_dict(load(user))
    assert d["bdt"]["classifier"] == "models/W/bdt/toy.pck"
    assert d["gnn"]["model_dir"] == "models/W/gnn"
    assert d["limit"]["gnn_score_cut"] == [3.5] and d["limit"]["fit_range"] == [1.6, 2.0]
    assert d["slurm"]["jobs"]["limit"] == {"partition": "standard", "mem": "4G", "time": "0-01:00:00", "cpus": 1}
    assert d["paths"] == {"output_base": "/tmp/toy_out", "log_dir": "/tmp/toy_logs", "conda_env": None}
    assert yaml.safe_load(yaml.safe_dump(d)) == d and json.loads(json.dumps(d)) == d


def test_phase_settings_sections(load, user):
    from tau3mu_gnn.pipelines.analysis.manifest import phase_settings

    cfg = load(user)
    assert list(phase_settings(cfg, "bdt")) == ["channel", "inputs", "branch_map", "resonance_skim", "skim",
                                                    "models", "bdt"]
    assert list(phase_settings(cfg, "gnn"))[-1] == "gnn"
    assert list(phase_settings(cfg, "limit"))[-2:] == ["gnn", "limit"]
    benchmark = list(phase_settings(cfg, "benchmark"))
    assert benchmark[-2:] == ["benchmark", "limit"] and "gnn" not in benchmark


def test_settings_diff_dotted():
    from tau3mu_gnn.pipelines.common.manifest import settings_diff

    old = {"skim": {"dz_max": 0.5}, "inputs": {"signal": ["a.root"]}, "branch_map": {}}
    new = {"skim": {"dz_max": 0.4}, "inputs": {"signal": ["a.root"]}, "branch_map": {"x": "y"}}
    assert settings_diff(old, new) == ["branch_map.x: (not set) -> 'y'", "skim.dz_max: 0.5 -> 0.4"]
    assert settings_diff(old, old) == []


def test_write_and_read_manifest(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.manifest import read_manifest
    from tau3mu_gnn.pipelines.analysis.manifest import write_manifest

    cfg = load(user)
    phase_dir = tmp_path / "toy_run" / "gnn_vs2"
    (phase_dir).mkdir(parents=True)
    (phase_dir / "jobs.json").write_text(json.dumps({"eval": "125"}))
    data = write_manifest(cfg, phase_dir, "gnn", "bdt_vs1", {"gnn_signal": "3_gnn/s.root"})
    assert read_manifest(phase_dir) == data
    assert data["pipeline"] == "analysis" and data["phase"] == "gnn" and data["version"] == "gnn_vs2"
    assert data["upstream"] == "bdt_vs1"
    assert data["outputs"] == {"gnn_signal": "3_gnn/s.root"} and data["missing"] == []
    assert data["slurm_jobs"] == {"eval": "125"}
    assert set(data["settings"]) == {"channel", "inputs", "branch_map", "resonance_skim", "skim", "models", "bdt", "gnn"}
    assert data["git_commit"] and data["created"]


def test_read_manifest_missing(tmp_path):
    from tau3mu_gnn.pipelines.common.manifest import UpstreamError, read_manifest

    with pytest.raises(UpstreamError, match="bdt_vs3"):
        read_manifest(tmp_path / "bdt_vs3")


def upstream_bdt(cfg, tmp_path, files=("1_skim/signal.root",)):
    from tau3mu_gnn.pipelines.analysis.manifest import write_manifest

    d = tmp_path / "toy_run" / "bdt_vs1"
    for rel in files:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("x")
    write_manifest(cfg, d, "bdt", None, {"skim_signal": "1_skim/signal.root",
                                              "bdt_signal": "2_bdt/signal_with_bdt_scores.root"})
    return d


def test_check_upstream_missing_files(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.manifest import UpstreamError
    from tau3mu_gnn.pipelines.analysis.manifest import check_upstream

    cfg = load(user)
    d = upstream_bdt(cfg, tmp_path)                      # 2_bdt file not written
    with pytest.raises(UpstreamError, match="2_bdt/signal_with_bdt_scores.root"):
        check_upstream(cfg, d, "gnn")


def test_check_upstream_wrong_phase(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.manifest import UpstreamError
    from tau3mu_gnn.pipelines.analysis.manifest import check_upstream

    cfg = load(user)
    d = upstream_bdt(cfg, tmp_path, files=("1_skim/signal.root", "2_bdt/signal_with_bdt_scores.root"))
    with pytest.raises(UpstreamError, match="gnn"):
        check_upstream(cfg, d, "limit")              # phase limit needs a gnn folder


@pytest.mark.parametrize("pipeline, ok", [("training", False), ("analysis", True), (None, True)])
def test_check_upstream_pipeline(load, user, tmp_path, pipeline, ok):
    """Both pipelines have a bdt phase: a training bdt folder is no analysis upstream. No 'pipeline' key
    (manifest written before it existed): accepted."""
    from tau3mu_gnn.pipelines.common.manifest import UpstreamError
    from tau3mu_gnn.pipelines.analysis.manifest import check_upstream

    cfg = load(user)
    d = upstream_bdt(cfg, tmp_path, files=("1_skim/signal.root", "2_bdt/signal_with_bdt_scores.root"))
    m = json.loads((d / "manifest.json").read_text())
    m.pop("pipeline") if pipeline is None else m.update(pipeline=pipeline)
    (d / "manifest.json").write_text(json.dumps(m))
    if ok:
        check_upstream(cfg, d, "gnn")
    else:
        with pytest.raises(UpstreamError, match="training pipeline"):
            check_upstream(cfg, d, "gnn")


def test_check_upstream_unchanged_config_no_warning(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipelines.analysis.manifest import check_upstream

    files = ("1_skim/signal.root", "2_bdt/signal_with_bdt_scores.root")
    d = upstream_bdt(load(user), tmp_path, files=files)
    m = check_upstream(load(user), d, "gnn")       # re-loaded config: tuples, absolute paths
    assert m["version"] == "bdt_vs1"
    assert "other settings" not in caplog.text


def test_check_upstream_changed_settings_warns(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipelines.analysis.manifest import check_upstream

    files = ("1_skim/signal.root", "2_bdt/signal_with_bdt_scores.root")
    d = upstream_bdt(load(user), tmp_path, files=files)
    user["inputs"]["signal"] = ["/data/other_sig.root"]
    check_upstream(load(user), d, "gnn")
    assert "other settings" in caplog.text and "inputs.signal" in caplog.text
