import pytest


def mk(parent, *names, complete=()):
    for n in names:
        (parent / n).mkdir(parents=True, exist_ok=True)
        if n in complete:
            (parent / n / "manifest.json").write_text("{}")


def test_next_version_numeric(tmp_path):
    from tau3mu_gnn.pipeline.runs import next_version, versions

    assert next_version(tmp_path, "phase_a") == "phase_a_vs1"
    mk(tmp_path, "phase_a_vs1", "phase_a_vs9", "phase_a_vs10", "phase_b_vs4", "phase_a_vs2_old", "debug")
    assert versions(tmp_path, "phase_a") == ["phase_a_vs1", "phase_a_vs9", "phase_a_vs10"]
    assert next_version(tmp_path, "phase_a") == "phase_a_vs11"


def test_create_version_writes_config_used(load, user, tmp_path):
    import yaml
    from tau3mu_gnn.pipeline.runs import create_version

    user["paths"]["output_base"] = str(tmp_path / "out")
    cfg = load(user)
    first = create_version(cfg, "phase_a")
    second = create_version(cfg, "phase_a")
    assert first.endswith("/toy_run/phase_a_vs1") and second.endswith("/toy_run/phase_a_vs2")
    used = yaml.safe_load((tmp_path / "out" / "toy_run" / "phase_a_vs1" / "config_used.yaml").read_text())
    assert used["analysis_name"] == "toy_run" and used["bdt"]["classifier"] == "models/W/bdt/toy.pck"
    debug = create_version(cfg, "eval", parent=f"{cfg.output_dir}/debug")
    assert debug.endswith("/toy_run/debug/eval_vs1")


def test_pick_upstream_highest_complete_and_note(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipeline.runs import pick_upstream

    user["paths"]["output_base"] = str(tmp_path)
    cfg = load(user)
    mk(tmp_path / "toy_run", "phase_a_vs1", "phase_a_vs2", "phase_a_vs3", complete=("phase_a_vs1", "phase_a_vs2"))
    assert pick_upstream(cfg, "phase_b") == f"{tmp_path}/toy_run/phase_a_vs2"
    assert "phase_a_vs3" in caplog.text


def test_pick_upstream_none_complete(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import UpstreamError
    from tau3mu_gnn.pipeline.runs import pick_upstream

    user["paths"]["output_base"] = str(tmp_path)
    cfg = load(user)
    mk(tmp_path / "toy_run", "phase_a_vs1")
    with pytest.raises(UpstreamError, match="phase-a"):
        pick_upstream(cfg, "phase_b")


def test_pick_upstream_from(load, user, tmp_path):
    from tau3mu_gnn.pipeline.manifest import UpstreamError
    from tau3mu_gnn.pipeline.runs import pick_upstream

    user["paths"]["output_base"] = str(tmp_path)
    cfg = load(user)
    mk(tmp_path / "toy_run", "phase_a_vs1", "phase_a_vs2", "phase_b_vs1", complete=("phase_a_vs1", "phase_b_vs1"))
    assert pick_upstream(cfg, "phase_b", from_="phase_a_vs1") == f"{tmp_path}/toy_run/phase_a_vs1"
    with pytest.raises(UpstreamError, match="no manifest"):
        pick_upstream(cfg, "phase_b", from_="phase_a_vs2")       # incomplete
    with pytest.raises(UpstreamError, match="phase_a"):
        pick_upstream(cfg, "phase_b", from_="phase_b_vs1")       # wrong phase
    with pytest.raises(UpstreamError, match="not found"):
        pick_upstream(cfg, "phase_b", from_="phase_a_vs7")


def test_pick_upstream_phase_a_has_none(load, user):
    from tau3mu_gnn.pipeline.runs import pick_upstream

    with pytest.raises(ValueError, match="phase_a"):
        pick_upstream(load(user), "phase_a")
