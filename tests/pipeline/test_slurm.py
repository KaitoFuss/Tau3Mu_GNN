import os
import shlex
import subprocess
from pathlib import Path

import pytest

ENV = {"CONDA_PREFIX": "/work/u/miniforge3/envs/Tau3Mu", "CONDA_EXE": "/work/u/miniforge3/bin/conda"}


def test_time_hours():
    from tau3mu_gnn.pipeline.slurm import time_hours

    assert time_hours("1-00:00:00") == 24.0
    assert time_hours("0-01:30:00") == 1.5
    assert time_hours("06:00:00") == 6.0
    assert time_hours("2-12:00:00") == 60.0


def test_job_env_from_environment():
    import sys

    from tau3mu_gnn.pipeline.slurm import JobEnv

    env = JobEnv.from_environment(ENV)
    assert env.env_prefix == Path("/work/u/miniforge3/envs/Tau3Mu")
    assert env.conda_base == Path("/work/u/miniforge3")
    assert env.python == Path(sys.executable)
    assert (env.src_dir / "tau3mu_gnn" / "pipeline" / "slurm.py").is_file()


@pytest.mark.parametrize("missing", ["CONDA_PREFIX", "CONDA_EXE"])
def test_job_env_needs_active_env(missing):
    from tau3mu_gnn.pipeline.slurm import EnvError, JobEnv

    env = dict(ENV)
    del env[missing]
    with pytest.raises(EnvError, match="conda activate"):
        JobEnv.from_environment(env)


def render(load, user, tmp_path, args=("skim",), user_file=None):
    from tau3mu_gnn.pipeline.slurm import Job, JobEnv, render_script

    cfg = load(user)
    job = Job(name="gnn_signal", resources="gnn_signal", args=tuple(args))
    env = JobEnv(Path("/c/base"), Path("/c/base/envs/T"), Path("/c/base/envs/T/bin/python"), Path("/repo/src"))
    return render_script(cfg, job, "/out/toy_run/phase_b_vs2", tmp_path / "logs", env)


def test_render_script_header_and_body(load, user, tmp_path):
    text = render(load, user, tmp_path, args=("gnn", "--sample", "signal"))
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH --job-name=toy_run.phase_b_vs2.gnn_signal" in lines
    assert "#SBATCH --account=t3" in lines and "#SBATCH --partition=standard" in lines
    assert "#SBATCH --mem=4G" in lines and "#SBATCH --time=0-01:00:00" in lines
    assert "#SBATCH --cpus-per-task=1" in lines
    assert f"#SBATCH --output={tmp_path / 'logs'}/gnn_signal-%j.out" in lines
    assert f"#SBATCH --error={tmp_path / 'logs'}/gnn_signal-%j.err" in lines
    assert "set -eo pipefail" in lines
    assert "source /c/base/bin/activate /c/base/envs/T" in lines
    assert "trap cleanup EXIT" in lines
    assert lines[-1] == "/c/base/envs/T/bin/python -m tau3mu_gnn.pipeline.jobs gnn --sample signal"
    assert "/out/toy_run/phase_b_vs2/logs" in text            # log copy target


def test_render_script_quotes_values(load, user, tmp_path):
    text = render(load, user, tmp_path, args=("skim", "--config", "/work/x y/c$fg.yaml"))
    last = text.splitlines()[-1]
    assert shlex.split(last)[-1] == "/work/x y/c$fg.yaml"


def test_render_script_is_valid_bash(load, user, tmp_path):
    path = tmp_path / "job.sh"
    path.write_text(render(load, user, tmp_path))
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.fixture
def fake_voms(tmp_path, monkeypatch):
    """voms-proxy-info that logs its arguments and exits with $FAKE_VOMS_RC (default 0)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "voms.log"
    exe = bin_dir / "voms-proxy-info"
    exe.write_text(f'#!/bin/sh\necho "$*" >> {log}\nexit ${{FAKE_VOMS_RC:-0}}\n')
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def remote_cfg(load, user):
    user["paths"]["output_base"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/x"
    return load(user)


def jobs_with(*resources):
    from tau3mu_gnn.pipeline.slurm import Job

    return [Job(name=r, resources=r, args=("skim",)) for r in resources]


def test_proxy_check_hours_from_longest_job(load, user, fake_voms, monkeypatch, fake_repo):
    import copy

    import yaml
    from conftest import TOY_DEFAULTS
    from tau3mu_gnn.pipeline.slurm import check_proxy

    d = copy.deepcopy(TOY_DEFAULTS)
    d["slurm"]["jobs"]["gnn_background"]["time"] = "2-00:00:00"
    (fake_repo / "configs" / "analysis_defaults_W.yaml").write_text(yaml.safe_dump(d))
    check_proxy(remote_cfg(load, user), jobs_with("gnn_signal", "gnn_background"),
                {"X509_USER_PROXY": "/t3home/u/.x509up_u1"})
    assert fake_voms.read_text().split() == ["--exists", "--valid", "48:00"]


def test_proxy_check_rounds_up(load, user, fake_voms, monkeypatch, fake_repo):
    import copy

    import yaml
    from conftest import TOY_DEFAULTS
    from tau3mu_gnn.pipeline.slurm import check_proxy

    d = copy.deepcopy(TOY_DEFAULTS)
    d["slurm"]["jobs"]["limit"]["time"] = "0-01:30:00"
    (fake_repo / "configs" / "analysis_defaults_W.yaml").write_text(yaml.safe_dump(d))
    check_proxy(remote_cfg(load, user), jobs_with("limit"), {"X509_USER_PROXY": "/t3home/u/.x509up_u1"})
    assert fake_voms.read_text().split()[-1] == "2:00"


def test_proxy_invalid(load, user, fake_voms, monkeypatch):
    from tau3mu_gnn.pipeline.slurm import ProxyError, check_proxy

    monkeypatch.setenv("FAKE_VOMS_RC", "1")
    with pytest.raises(ProxyError, match="voms-proxy-init --voms cms --valid 192:00"):
        check_proxy(remote_cfg(load, user), jobs_with("skim"), {"X509_USER_PROXY": "/t3home/u/.x509up_u1"})


@pytest.mark.parametrize("environ, match", [({}, "X509_USER_PROXY"),
                                            ({"X509_USER_PROXY": "/tmp/x509up_u1"}, "/tmp")])
def test_proxy_location(load, user, fake_voms, environ, match):
    from tau3mu_gnn.pipeline.slurm import ProxyError, check_proxy

    with pytest.raises(ProxyError, match=match):
        check_proxy(remote_cfg(load, user), jobs_with("skim"), environ)


def test_proxy_not_needed_for_local_output(load, user, fake_voms):
    from tau3mu_gnn.pipeline.slurm import check_proxy

    check_proxy(load(user), jobs_with("skim"), {})
    assert not fake_voms.exists()


@pytest.fixture
def fake_sbatch(tmp_path, monkeypatch):
    """sbatch that logs its arguments and prints increasing IDs. FAKE_SBATCH_FAIL_AT=n: call n fails."""
    bin_dir = tmp_path / "sbin"
    bin_dir.mkdir()
    log, counter = tmp_path / "sbatch.log", tmp_path / "sbatch.n"
    exe = bin_dir / "sbatch"
    exe.write_text(f"""#!/bin/sh
n=$(( $(cat {counter} 2>/dev/null || echo 0) + 1 )); echo $n > {counter}
if [ "$n" = "${{FAKE_SBATCH_FAIL_AT:-0}}" ]; then echo "sbatch: error: invalid partition" >&2; exit 1; fi
echo "$*" >> {log}
echo $((1000 + n))
""")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def three_jobs():
    from tau3mu_gnn.pipeline.slurm import Job

    return [Job("limit_2.0", "limit", ("limit", "--cut", "2.0")),
            Job("limit_3.9", "limit", ("limit", "--cut", "3.9")),
            Job("summary", "summary", ("summary",), after=("limit_2.0", "limit_3.9"), dependency="afterany")]


def test_submit_jobs_dependencies(load, user, tmp_path, fake_sbatch):
    from tau3mu_gnn.pipeline.slurm import JobEnv, submit_jobs

    ids = submit_jobs(load(user), three_jobs(), "/out/toy_run/phase_c_vs1", tmp_path / "logs", JobEnv.from_environment(ENV))
    assert ids == {"limit_2.0": "1001", "limit_3.9": "1002", "summary": "1003"}
    calls = fake_sbatch.read_text().splitlines()
    assert calls[0] == f"--parsable {tmp_path / 'logs' / 'limit_2.0.sh'}"
    assert calls[2] == f"--parsable --dependency=afterany:1001:1002 {tmp_path / 'logs' / 'summary.sh'}"
    assert (tmp_path / "logs" / "summary.sh").read_text().startswith("#!/bin/bash")


def test_submit_jobs_dry_run(load, user, tmp_path, fake_sbatch, capsys):
    from tau3mu_gnn.pipeline.slurm import JobEnv, submit_jobs

    ids = submit_jobs(load(user), three_jobs(), "/out/toy_run/phase_c_vs1", tmp_path / "logs",
                      JobEnv.from_environment(ENV), dry_run=True)
    assert ids == {"limit_2.0": "DRYRUN-1", "limit_3.9": "DRYRUN-2", "summary": "DRYRUN-3"}
    assert not fake_sbatch.exists()
    out = capsys.readouterr().out
    assert f"sbatch --parsable --dependency=afterany:DRYRUN-1:DRYRUN-2 {tmp_path / 'logs' / 'summary.sh'}" in out
    assert (tmp_path / "logs" / "limit_3.9.sh").is_file()


def test_submit_jobs_failure_lists_submitted(load, user, tmp_path, fake_sbatch, monkeypatch):
    from tau3mu_gnn.pipeline.slurm import JobEnv, SubmitError, submit_jobs

    monkeypatch.setenv("FAKE_SBATCH_FAIL_AT", "2")
    with pytest.raises(SubmitError) as e:
        submit_jobs(load(user), three_jobs(), "/out/toy_run/phase_c_vs1", tmp_path / "logs", JobEnv.from_environment(ENV))
    msg = str(e.value)
    assert "limit_3.9" in msg and "invalid partition" in msg and "scancel 1001" in msg


def test_submit_jobs_unknown_dependency(load, user, tmp_path, fake_sbatch):
    from tau3mu_gnn.pipeline.slurm import Job, JobEnv, submit_jobs

    with pytest.raises(ValueError, match="skim"):
        submit_jobs(load(user), [Job("bdt", "bdt", ("bdt",), after=("skim",))], "/o/v", tmp_path,
                    JobEnv.from_environment(ENV))


def test_generated_script_runs(tmp_path, monkeypatch):
    """Run a generated script with bash (no SLURM): conda activation, scratch, job CLI, log copy, cleanup.
    Uses the real env of this shell and the real repo defaults; the job is a debug 'summary'."""
    import json

    from tau3mu_gnn.pipeline.config import load_config
    from tau3mu_gnn.pipeline.slurm import Job, JobEnv, render_script

    if not (os.environ.get("CONDA_PREFIX") and os.environ.get("CONDA_EXE")):
        pytest.skip("needs an active conda env")
    cfg_file = tmp_path / "user.yaml"
    cfg_file.write_text(f"""analysis_name: script_test
channel: W
paths: {{output_base: {tmp_path / 'out'}, log_dir: {tmp_path / 'logs_root'}}}
inputs: {{signal: [/data/s.root], background: [/data/b.root], already_resonance_skimmed: true}}
limit: {{signal_weight: 1.0e-5, gnn_score_cut: [3.9]}}
""")
    cfg = load_config(cfg_file)
    src = tmp_path / "out" / "script_test" / "phase_c_vs1"
    (src / "cut_3.9").mkdir(parents=True)
    (src / "cut_3.9" / "limit.json").write_text(json.dumps({"cut": 3.9, "limit": 1.25, "limitErr": 0.0}))
    version_dir = str(tmp_path / "out" / "script_test" / "debug" / "summary_vs1")
    log_folder = tmp_path / "logs"
    log_folder.mkdir()
    job = Job("summary", "summary", ("summary", "--config", str(cfg_file), "--phase-dir", version_dir,
                                     "--source", str(src), "--no-manifest"))
    script = tmp_path / "summary.sh"
    script.write_text(render_script(cfg, job, version_dir, log_folder, JobEnv.from_environment(),
                                    scratch_root=str(tmp_path / "scratch")))
    out = log_folder / "summary-77.out"
    with open(out, "w") as fo:
        subprocess.run(["bash", str(script)], env={**os.environ, "SLURM_JOB_ID": "77"}, stdout=fo,
                       stderr=subprocess.STDOUT, check=True, timeout=300)
    assert json.loads((Path(version_dir) / "limits.json").read_text())["best_cut"] == 3.9
    assert (Path(version_dir) / "logs" / "summary-77.out").is_file()            # log copied by the trap
    assert not (tmp_path / "scratch" / "77").exists()                             # scratch removed
