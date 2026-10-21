import json
import os
from pathlib import Path

import pytest

from test_slurm import ENV, fake_sbatch  # noqa: F401  (fixture reused)


@pytest.fixture
def env():
    from tau3mu_gnn.pipeline.slurm import JobEnv

    return JobEnv.from_environment(ENV)


@pytest.fixture
def cfg(load, user, tmp_path):
    user["paths"]["output_base"] = str(tmp_path / "out")
    user["paths"]["log_dir"] = str(tmp_path / "logs")
    user["limit"]["gnn_score_cut"] = [2.0, 3.9, 4.5]
    return load(user)


def complete(cfg, name, upstream=None):
    d = Path(cfg.output_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"phase": name.rsplit("_vs", 1)[0], "upstream": upstream}))
    return str(d)


def sbatch_calls(log):
    return log.read_text().splitlines()


def test_phase_a(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_phase

    s = submit_phase(cfg, "phase_a", env=env)
    assert s.version_dir == f"{cfg.output_dir}/phase_a_vs1" and s.upstream is None
    assert s.job_ids == {"skim": "1001", "bdt": "1002"}
    assert Path(s.log_folder) == Path(cfg.paths.log_dir) / "toy_run" / "phase_a_vs1"
    calls = sbatch_calls(fake_sbatch)
    assert calls[1].startswith("--parsable --dependency=afterok:1001 ")
    assert json.loads((Path(s.version_dir) / "jobs.json").read_text()) == {"skim": "1001", "bdt": "1002"}
    assert (Path(s.version_dir) / "config_used.yaml").is_file()
    script = (s.log_folder / "bdt.sh").read_text().splitlines()[-1]
    assert f"bdt --config {cfg.user_file} --phase-dir {s.version_dir}" in script


def test_phase_b_uses_latest_complete_a(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_phase

    a1 = complete(cfg, "phase_a_vs1")
    Path(cfg.output_dir, "phase_a_vs2").mkdir()                       # failed run, no manifest
    s = submit_phase(cfg, "phase_b", env=env)
    assert s.upstream == a1 and s.version_dir.endswith("phase_b_vs1")
    calls = sbatch_calls(fake_sbatch)
    assert len(calls) == 3 and calls[2].startswith("--parsable --dependency=afterok:1001:1002 ")
    gnn_bkg = (s.log_folder / "gnn_background.sh").read_text().splitlines()[-1]
    assert gnn_bkg.endswith(f"--upstream {a1} --sample background")


def test_phase_c_one_job_per_cut_afterany(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_phase

    b1 = complete(cfg, "phase_b_vs1", upstream="phase_a_vs1")
    s = submit_phase(cfg, "phase_c", env=env)
    assert list(s.job_ids) == ["limit_2.0", "limit_3.9", "limit_4.5", "summary"]
    calls = sbatch_calls(fake_sbatch)
    assert [c.startswith("--parsable /") for c in calls[:3]] == [True] * 3       # limits: no dependency
    assert calls[3].startswith("--parsable --dependency=afterany:1001:1002:1003 ")
    assert (s.log_folder / "limit_3.9.sh").read_text().splitlines()[-1].endswith(f"--upstream {b1} --cut 3.9")


def test_phase_without_upstream_fails_before_anything(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_phase
    from tau3mu_gnn.pipeline.manifest import UpstreamError

    with pytest.raises(UpstreamError, match="phase-a"):
        submit_phase(cfg, "phase_b", env=env)
    assert not fake_sbatch.exists() and not Path(cfg.paths.log_dir).exists()


def test_dry_run_makes_no_version(cfg, env, fake_sbatch, capsys):
    from tau3mu_gnn.pipeline.driver import submit_phase

    s = submit_phase(cfg, "phase_a", dry_run=True, env=env)
    assert s.version_dir.endswith("phase_a_vs1") and s.job_ids == {"skim": "DRYRUN-skim", "bdt": "DRYRUN-bdt"}
    assert not Path(cfg.output_dir).exists() and not fake_sbatch.exists()
    assert (s.log_folder / "skim.sh").is_file() and "sbatch --parsable" in capsys.readouterr().out


def test_no_conda_env_fails_first(cfg, fake_sbatch, monkeypatch):
    from tau3mu_gnn.pipeline.driver import submit_phase
    from tau3mu_gnn.pipeline.slurm import EnvError

    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    with pytest.raises(EnvError):
        submit_phase(cfg, "phase_a")
    assert not Path(cfg.output_dir).exists() and not Path(cfg.paths.log_dir).exists()


def test_stage_eval_debug(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_stage

    a1 = complete(cfg, "phase_a_vs1")
    b1 = complete(cfg, "phase_b_vs1", upstream="phase_a_vs1")
    s = submit_stage(cfg, "eval", env=env)
    assert s.version_dir == f"{cfg.output_dir}/debug/eval_vs1"
    assert Path(s.log_folder) == Path(cfg.paths.log_dir) / "toy_run" / "debug" / "eval_vs1"
    last = (s.log_folder / "eval.sh").read_text().splitlines()[-1]
    assert last.endswith(f"--upstream {a1} --source {b1} --no-manifest")


def test_stage_limit_cut_must_be_in_config(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_stage

    complete(cfg, "phase_b_vs1", upstream="phase_a_vs1")
    with pytest.raises(ValueError, match="5.0"):
        submit_stage(cfg, "limit", cut=5.0, env=env)
    assert not fake_sbatch.exists()
    s = submit_stage(cfg, "limit", cut=3.9, env=env)
    assert list(s.job_ids) == ["limit_3.9"]


def test_stage_gnn_needs_sample(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_stage

    complete(cfg, "phase_a_vs1")
    with pytest.raises(ValueError, match="sample"):
        submit_stage(cfg, "gnn", env=env)
    s = submit_stage(cfg, "gnn", sample="signal", env=env)
    assert list(s.job_ids) == ["gnn_signal"]


def test_stage_summary_uses_incomplete_phase_c(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_stage

    Path(cfg.output_dir, "phase_c_vs1").mkdir(parents=True)          # summary failed: no manifest
    s = submit_stage(cfg, "summary", env=env)
    last = (s.log_folder / "summary.sh").read_text().splitlines()[-1]
    assert last.endswith(f"--source {cfg.output_dir}/phase_c_vs1 --no-manifest")


def test_conda_env_from_config_needs_no_active_env(load, user, tmp_path, fake_sbatch, monkeypatch):
    from test_slurm import fake_conda

    from tau3mu_gnn.pipeline.driver import submit_phase

    _, prefix = fake_conda(tmp_path)
    user["paths"].update(output_base=str(tmp_path / "out"), log_dir=str(tmp_path / "logs"), conda_env=str(prefix))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    s = submit_phase(load(user), "phase_a")
    assert (s.log_folder / "skim.sh").read_text().splitlines()[-1].startswith(f"{prefix / 'bin' / 'python'} -m ")


def test_all_chains_the_three_phases(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_all

    a, b, c = submit_all(cfg, env=env)
    assert [s.version_dir.rsplit("/", 1)[-1] for s in (a, b, c)] == ["phase_a_vs1", "phase_b_vs1", "phase_c_vs1"]
    assert (a.upstream, b.upstream, c.upstream) == (None, a.version_dir, b.version_dir)   # A has no manifest yet
    assert a.job_ids == {"skim": "1001", "bdt": "1002"}
    assert b.job_ids == {"gnn_signal": "1003", "gnn_background": "1004", "eval": "1005"}
    calls = sbatch_calls(fake_sbatch)
    for i in (2, 3):                                                   # both GNN jobs wait for A's bdt job
        assert calls[i].startswith("--parsable --dependency=afterok:1002 --kill-on-invalid-dep=yes ")
    for i in (5, 6, 7):                                                # every limit job waits for B's eval job
        assert calls[i].startswith("--parsable --dependency=afterok:1005 --kill-on-invalid-dep=yes ")
    assert calls[8].startswith("--parsable --dependency=afterany:1006:1007:1008,afterok:1005 ")
    assert (b.log_folder / "gnn_signal.sh").read_text().splitlines()[-1].endswith(
        f"--upstream {a.version_dir} --sample signal")
    for s in (a, b, c):
        assert json.loads((Path(s.version_dir) / "jobs.json").read_text()) == s.job_ids


def test_all_uses_its_own_phase_a_not_an_older_one(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipeline.driver import submit_all

    complete(cfg, "phase_a_vs1")
    a, b, _ = submit_all(cfg, env=env)
    assert a.version_dir.endswith("phase_a_vs2") and b.upstream == a.version_dir


def test_all_dry_run(cfg, env, fake_sbatch, capsys):
    from tau3mu_gnn.pipeline.driver import submit_all

    a, b, c = submit_all(cfg, dry_run=True, env=env)
    assert f"--dependency=afterok:DRYRUN-bdt --kill-on-invalid-dep=yes {b.log_folder}/gnn_signal.sh" \
        in capsys.readouterr().out
    assert b.upstream == a.version_dir and a.version_dir.endswith("phase_a_vs1")
    assert c.job_ids["summary"] == "DRYRUN-summary"
    assert (c.log_folder / "summary.sh").is_file()
    assert not Path(cfg.output_dir).exists() and not fake_sbatch.exists()


def test_all_proxy_covers_the_whole_chain(cfg, env, fake_sbatch, monkeypatch):
    from tau3mu_gnn.pipeline import driver

    hours = []
    monkeypatch.setattr(driver, "check_proxy", lambda cfg, h: hours.append(h))
    driver.submit_all(cfg, env=env)
    assert hours == [6.0]                       # toy: every job 1 h; A 2 h + B 2 h + C 2 h


def test_all_sbatch_failure_lists_earlier_phases(cfg, env, fake_sbatch, monkeypatch):
    from tau3mu_gnn.pipeline.driver import submit_all
    from tau3mu_gnn.pipeline.slurm import SubmitError

    monkeypatch.setenv("FAKE_SBATCH_FAIL_AT", "4")                  # gnn_background of phase B
    with pytest.raises(SubmitError) as e:
        submit_all(cfg, env=env)
    assert "scancel 1001 1002 1003" in str(e.value)
