import pytest

from test_slurm import ENV, fake_sbatch  # noqa: F401


@pytest.fixture
def cli_env(load, user, tmp_path, monkeypatch):
    from tau3mu_gnn.pipelines.analysis import cli

    user["paths"]["output_base"] = str(tmp_path / "out")
    user["paths"]["log_dir"] = str(tmp_path / "logs")
    cfg = load(user)
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)       # toy config uses the fake repo defaults
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    return cfg


def test_bdt_phase_prints_summary(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipelines.analysis import cli

    assert cli.main(["bdt", "--config", "my.yaml"]) == 0
    out = capsys.readouterr().out
    assert "bdt_vs1" in out and "skim" in out and "1001" in out and "logs" in out


def test_dry_run(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipelines.analysis import cli

    assert cli.main(["bdt", "--config", "my.yaml", "--dry-run"]) == 0
    assert "DRYRUN-skim" in capsys.readouterr().out and not fake_sbatch.exists()


def test_known_error_exit_1(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipelines.analysis import cli

    assert cli.main(["gnn", "--config", "my.yaml"]) == 1
    err = capsys.readouterr().err
    assert "bdt" in err and "Traceback" not in err


def test_stage_args(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipelines.analysis import cli

    assert cli.main(["stage", "limit", "--config", "my.yaml", "--cut", "7"]) == 1
    assert "not in limit.gnn_score_cut" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["stage", "nope", "--config", "my.yaml"])


def test_config_error_exit_1(tmp_path, capsys):
    from tau3mu_gnn.pipelines.analysis import cli

    assert cli.main(["bdt", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert "cannot read config file" in capsys.readouterr().err


def test_all(cli_env, fake_sbatch, capsys):
    from tau3mu_gnn.pipelines.analysis import cli

    assert cli.main(["all", "--config", "my.yaml"]) == 0
    out = capsys.readouterr().out
    assert all(f"{p}_vs1" in out for p in ("bdt", "benchmark", "gnn", "limit")) and f"  {'summary':<22} 1010" in out
    assert f"  {'benchmark_summary':<22} 1005" in out
    with pytest.raises(SystemExit):
        cli.main(["all", "--config", "my.yaml", "--from", "bdt_vs1"])


def test_console_script():
    import tomllib

    from conftest import REPO_ROOT

    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert "tau3mu-pipeline" not in scripts and scripts["tau3mu-analysis"] == "tau3mu_gnn.pipelines.analysis.cli:main"
