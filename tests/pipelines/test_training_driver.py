import json
from pathlib import Path

import pytest

from test_slurm import ENV, fake_sbatch  # noqa: F401  (fixture reused)


@pytest.fixture
def env():
    from tau3mu_gnn.pipelines.common.slurm import JobEnv

    return JobEnv.from_environment(ENV)


@pytest.fixture
def cfg(load_training, training_user, tmp_path):
    training_user["paths"]["output_base"] = str(tmp_path / "out")
    training_user["paths"]["log_dir"] = str(tmp_path / "logs")
    return load_training(training_user)


def complete(cfg, name, upstream=None):
    d = Path(cfg.output_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"phase": name.rsplit("_vs", 1)[0], "upstream": upstream}))
    return str(d)


def sbatch_calls(log):
    return log.read_text().splitlines()


def last_line(sub, job):
    return (sub.log_folder / f"{job}.sh").read_text().splitlines()[-1]


def test_bdt_phase(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipelines.training.driver import MODULE, submit_phase

    s = submit_phase(cfg, "bdt", env=env)
    assert s.version_dir == f"{cfg.output_dir}/bdt_vs1" and s.upstream is None
    assert s.job_ids == {"skim": "1001", "bdt_train": "1002", "bdt_infer": "1003"}
    assert Path(s.log_folder) == Path(cfg.paths.log_dir) / "toy_train" / "bdt_vs1"
    calls = sbatch_calls(fake_sbatch)
    assert calls[1].startswith("--parsable --dependency=afterok:1001 ")
    assert calls[2].startswith("--parsable --dependency=afterok:1002 ")
    assert json.loads((Path(s.version_dir) / "jobs.json").read_text()) == s.job_ids
    used = (Path(s.version_dir) / "config_used.yaml").read_text()
    assert "training_name: toy_train" in used and "bdt_score_cut" in used     # training settings, not analysis
    assert last_line(s, "bdt_infer").endswith(
        f"-m {MODULE} bdt_infer --config {cfg.user_file} --phase-dir {s.version_dir}")
    assert "#SBATCH --job-name=BDT1.bdtI.toy_train" in (s.log_folder / "bdt_infer.sh").read_text()


def test_bdt_has_no_from(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipelines.training.driver import submit_phase

    with pytest.raises(ValueError, match="--from"):
        submit_phase(cfg, "bdt", "bdt_vs1", env=env)
    assert not fake_sbatch.exists()


def test_gnn_phase_dependencies(cfg, env, fake_sbatch, capsys):
    from tau3mu_gnn.pipelines.training.driver import submit_phase

    b1 = complete(cfg, "bdt_vs1")
    s = submit_phase(cfg, "gnn", dry_run=True, env=env)
    assert s.upstream == b1 and s.version_dir == f"{cfg.output_dir}/gnn_vs1"
    assert list(s.job_ids) == ["npvs", "graphs_signal", "graphs_background", "gnn_train"]
    cmds = {Path(c.split()[-1]).stem: c for c in capsys.readouterr().out.splitlines() if c.startswith("sbatch")}
    assert "--dependency" not in cmds["npvs"]
    assert "--dependency=afterok:DRYRUN-npvs " in cmds["graphs_signal"]
    assert "--dependency" not in cmds["graphs_background"]
    assert "--dependency=afterok:DRYRUN-graphs_signal:DRYRUN-graphs_background " in cmds["gnn_train"]
    assert last_line(s, "graphs_signal").endswith(f"--upstream {b1} --sample signal")
    assert last_line(s, "graphs_background").endswith(f"--upstream {b1} --sample background")
    assert last_line(s, "npvs").endswith(f"--upstream {b1}")
    assert last_line(s, "gnn_train").endswith(f"--upstream {b1}")
    assert not fake_sbatch.exists() and not Path(s.version_dir).exists()


def test_gpu_job_flags(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipelines.training.driver import submit_phase

    complete(cfg, "bdt_vs1")
    s = submit_phase(cfg, "gnn", env=env)
    gpu = (s.log_folder / "gnn_train.sh").read_text().splitlines()
    assert "#SBATCH --partition=gpu" in gpu and "#SBATCH --account=gpu_gres" in gpu
    assert "#SBATCH --gres=gpu:1" in gpu and "#SBATCH --job-name=GNN1.gnnT.toy_train" in gpu
    cpu = (s.log_folder / "npvs.sh").read_text()
    assert "--gres" not in cpu and "#SBATCH --account=t3\n" in cpu


def test_benchmark_jobs_per_cut(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipelines.training.driver import submit_phase

    b1 = complete(cfg, "bdt_vs1")
    s = submit_phase(cfg, "benchmark", env=env)
    assert list(s.job_ids) == ["benchmark_limit_0.98", "benchmark_limit_0.99", "benchmark_summary"]
    calls = sbatch_calls(fake_sbatch)
    assert [c.startswith("--parsable /") for c in calls[:2]] == [True, True]
    assert calls[2].startswith("--parsable --dependency=afterany:1001:1002 ")
    assert last_line(s, "benchmark_limit_0.99").endswith(f"--upstream {b1} --cut 0.99")
    assert last_line(s, "benchmark_summary").endswith(f"--upstream {b1}")


def test_from_requires_complete_bdt(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipelines.common.manifest import UpstreamError
    from tau3mu_gnn.pipelines.training.driver import submit_phase

    Path(cfg.output_dir, "bdt_vs1").mkdir(parents=True)                # failed run, no manifest
    with pytest.raises(UpstreamError, match="no manifest"):
        submit_phase(cfg, "gnn", "bdt_vs1", env=env)
    with pytest.raises(UpstreamError):
        submit_phase(cfg, "benchmark", env=env)
    assert not fake_sbatch.exists()


def test_from_picks_given_bdt(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipelines.training.driver import submit_phase

    b1 = complete(cfg, "bdt_vs1")
    complete(cfg, "bdt_vs2")
    assert submit_phase(cfg, "benchmark", "bdt_vs1", env=env).upstream == b1


def test_all_waits_for_bdt_infer(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipelines.training.driver import submit_all

    bdt, gnn, ben = submit_all(cfg, env=env)
    assert [s.version_dir.rsplit("/", 1)[-1] for s in (bdt, gnn, ben)] == ["bdt_vs1", "gnn_vs1", "benchmark_vs1"]
    assert gnn.upstream == ben.upstream == bdt.version_dir
    infer = bdt.job_ids["bdt_infer"]
    assert infer == "1003"
    calls = dict(zip([n for s in (bdt, gnn, ben) for n in s.job_ids], sbatch_calls(fake_sbatch)))
    for first in ("npvs", "graphs_background", "benchmark_limit_0.98", "benchmark_limit_0.99"):
        assert calls[first].startswith(f"--parsable --dependency=afterok:{infer} "), first
    assert calls["graphs_signal"].startswith(f"--parsable --dependency=afterok:{gnn.job_ids['npvs']} ")
    assert last_line(gnn, "npvs").endswith(f"--upstream {bdt.version_dir}")


def test_all_proxy_uses_longest_chain(cfg, env, fake_sbatch, monkeypatch):
    from tau3mu_gnn.pipelines.training import driver

    monkeypatch.setattr(driver, "check_proxy", lambda c, hours: calls.append(hours))
    calls = []
    driver.submit_all(cfg, dry_run=True, env=env)
    assert calls == [3.0 + 3.0]       # bdt 3 jobs + gnn npvs -> graphs_signal -> gnn_train (1 h each)


def test_all_failure_lists_scancel(cfg, env, fake_sbatch, monkeypatch):
    from tau3mu_gnn.pipelines.common.slurm import SubmitError
    from tau3mu_gnn.pipelines.training.driver import submit_all

    monkeypatch.setenv("FAKE_SBATCH_FAIL_AT", "5")
    with pytest.raises(SubmitError, match="scancel 1001 1002 1003 1004"):
        submit_all(cfg, env=env)


def test_no_comet_key_in_files(cfg, env, fake_sbatch):
    from tau3mu_gnn.pipelines.training.driver import submit_all

    subs = submit_all(cfg, env=env)
    files = [f for root in (Path(cfg.paths.log_dir), Path(cfg.output_dir)) for f in root.rglob("*") if f.is_file()]
    assert len(files) > 10 and any(f.name == "config_used.yaml" for f in files)
    assert not [f for f in files if "toy-key" in f.read_text()]
    assert len(subs) == 3


@pytest.fixture
def cli_env(cfg, monkeypatch):
    from tau3mu_gnn.pipelines.training import cli

    monkeypatch.setattr(cli, "load_training_config", lambda path: cfg)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    return cfg


def test_cli_dry_run(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipelines.training import cli

    assert cli.main(["all", "--config", "t.yaml", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("DRY RUN") == 1 and "DRYRUN-gnn_train" in out and "DRYRUN-benchmark_summary" in out
    assert all(f"{p}_vs1" in out for p in ("bdt", "gnn", "benchmark")) and not fake_sbatch.exists()


def test_cli_phase_and_errors(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipelines.training import cli

    assert cli.main(["gnn", "--config", "t.yaml"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ") and "Traceback" not in err
    b1 = complete(cli_env, "bdt_vs1")
    assert cli.main(["benchmark", "--config", "t.yaml", "--from", "bdt_vs1"]) == 0
    out = capsys.readouterr().out
    assert f"upstream: {b1}" in out and "benchmark_summary" in out and "1003" in out
    for argv in (["bdt", "--config", "t.yaml", "--from", "bdt_vs1"], ["all", "--config", "t.yaml", "--from", "x"],
                 ["stage", "skim", "--config", "t.yaml"]):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_cli_config_error(tmp_path, capsys):
    from tau3mu_gnn.pipelines.training import cli

    assert cli.main(["bdt", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert capsys.readouterr().err.startswith("ERROR: ")


def test_console_script():
    import tomllib

    from conftest import REPO_ROOT

    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert "tau3mu-train" not in scripts and scripts["tau3mu-training"] == "tau3mu_gnn.pipelines.training.cli:main"
