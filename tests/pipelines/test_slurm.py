import os
import shlex
import subprocess
from pathlib import Path

import pytest

ENV = {"CONDA_PREFIX": "/work/u/miniforge3/envs/Tau3Mu", "CONDA_EXE": "/work/u/miniforge3/bin/conda"}


def test_time_hours():
    from tau3mu_gnn.pipelines.common.slurm import time_hours

    assert time_hours("1-00:00:00") == 24.0
    assert time_hours("0-01:30:00") == 1.5
    assert time_hours("06:00:00") == 6.0
    assert time_hours("2-12:00:00") == 60.0


def test_job_env_from_environment():
    import sys

    from tau3mu_gnn.pipelines.common.slurm import JobEnv

    env = JobEnv.from_environment(ENV)
    assert env.env_prefix == Path("/work/u/miniforge3/envs/Tau3Mu")
    assert env.conda_base == Path("/work/u/miniforge3")
    assert env.python == Path(sys.executable)
    assert (env.src_dir / "tau3mu_gnn" / "pipelines" / "common" / "slurm.py").is_file()


@pytest.mark.parametrize("missing", ["CONDA_PREFIX", "CONDA_EXE"])
def test_job_env_needs_active_env(missing):
    from tau3mu_gnn.pipelines.common.slurm import EnvError, JobEnv

    env = dict(ENV)
    del env[missing]
    with pytest.raises(EnvError, match="conda activate"):
        JobEnv.from_environment(env)


def fake_conda(root):
    """Conda base at root/base with an env T: base/bin/activate, base/envs/T/bin/python."""
    base, prefix = root / "base", root / "base" / "envs" / "T"
    for f in (base / "bin" / "activate", base / "bin" / "python", prefix / "bin" / "python"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("")
    return base, prefix


def test_job_env_from_config(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.slurm import JobEnv

    base, prefix = fake_conda(tmp_path)
    user["paths"]["conda_env"] = str(prefix)
    env = JobEnv.from_config(load(user), environ={})                     # no active env needed
    assert (env.conda_base, env.env_prefix, env.python) == (base, prefix, prefix / "bin" / "python")
    assert (env.src_dir / "tau3mu_gnn" / "pipelines" / "common" / "slurm.py").is_file()


def test_job_env_from_config_base_env(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.slurm import JobEnv

    base, _ = fake_conda(tmp_path)
    user["paths"]["conda_env"] = str(base)
    env = JobEnv.from_config(load(user), environ={})
    assert (env.conda_base, env.env_prefix, env.python) == (base, base, base / "bin" / "python")


def test_job_env_from_config_not_a_conda_env(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.slurm import EnvError, JobEnv

    user["paths"]["conda_env"] = str(tmp_path / "nothing")
    with pytest.raises(EnvError, match="paths.conda_env"):
        JobEnv.from_config(load(user), environ={})


def test_job_env_from_config_falls_back_to_active_env(load, user):
    from tau3mu_gnn.pipelines.common.slurm import JobEnv

    env = JobEnv.from_config(load(user), environ=ENV)
    assert env.env_prefix == Path(ENV["CONDA_PREFIX"])


def render(load, user, tmp_path, args=("skim",), user_file=None):
    from tau3mu_gnn.pipelines.common.slurm import Job, JobEnv, render_script

    cfg = load(user)
    job = Job(name="gnn_signal", resources="gnn_signal", args=tuple(args))
    env = JobEnv(Path("/c/base"), Path("/c/base/envs/T"), Path("/c/base/envs/T/bin/python"), Path("/repo/src"))
    return render_script(cfg, job, "/out/toy_run/phase_b_vs2", tmp_path / "logs", env)


@pytest.mark.parametrize("version, job, expected", [
    ("phase_a_vs1", "skim", "A1.skim.run"),
    ("phase_a_vs1", "bdt", "A1.bdt.run"),
    ("phase_b_vs3", "gnn_signal", "B3.gnnS.run"),
    ("phase_b_vs3", "gnn_background", "B3.gnnB.run"),
    ("phase_b_vs3", "eval", "B3.eval.run"),
    ("phase_c_vs12", "limit_3.9", "C12.lim3.9.run"),
    ("phase_c_vs12", "summary", "C12.sum.run"),
    ("gnn_vs2", "gnn_signal", "D2.gnnS.run"),
    ("limit_vs1", "limit_2.0", "D1.lim2.0.run"),
])
def test_slurm_job_name(version, job, expected):
    from tau3mu_gnn.pipelines.common.slurm import slurm_job_name

    assert slurm_job_name("run", version, job) == expected


def test_render_script_header_and_body(load, user, tmp_path):
    text = render(load, user, tmp_path, args=("gnn", "--sample", "signal"))
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH --job-name=B2.gnnS.toy_run" in lines
    assert "#SBATCH --account=t3" in lines and "#SBATCH --partition=standard" in lines
    assert "#SBATCH --mem=4G" in lines and "#SBATCH --time=0-01:00:00" in lines
    assert "#SBATCH --cpus-per-task=1" in lines
    assert f"#SBATCH --output={tmp_path / 'logs'}/gnn_signal-%j.out" in lines
    assert f"#SBATCH --error={tmp_path / 'logs'}/gnn_signal-%j.err" in lines
    assert "set -eo pipefail" in lines
    assert "source /c/base/bin/activate /c/base/envs/T" in lines
    assert "trap cleanup EXIT" in lines
    assert lines[-1] == "/c/base/envs/T/bin/python -m tau3mu_gnn.pipelines.analysis.jobs gnn --sample signal"
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


def test_chain_hours_is_longest_dependency_path(load, user, fake_repo):
    import copy

    import yaml
    from conftest import TOY_DEFAULTS
    from tau3mu_gnn.pipelines.common.slurm import Job, chain_hours

    d = copy.deepcopy(TOY_DEFAULTS)
    d["slurm"]["jobs"]["gnn_background"]["time"] = "2-00:00:00"
    (fake_repo / "configs" / "analysis_defaults_W.yaml").write_text(yaml.safe_dump(d))
    cfg = load(user)
    phase_b = [Job("gnn_signal", "gnn_signal", ()), Job("gnn_background", "gnn_background", ()),
               Job("eval", "eval", (), after=("gnn_signal", "gnn_background"))]
    assert chain_hours(cfg, phase_b) == 49.0                       # gnn_background 48 h, then eval 1 h
    assert chain_hours(cfg, [Job("skim", "skim", ()), Job("bdt", "bdt", (), after=("skim",))]) == 2.0
    assert chain_hours(cfg, [Job("limit_2.0", "limit", ()), Job("limit_3.9", "limit", ())]) == 1.0


def test_proxy_check_hours(load, user, fake_voms):
    from tau3mu_gnn.pipelines.common.slurm import check_proxy

    check_proxy(remote_cfg(load, user), 48, {"X509_USER_PROXY": "/t3home/u/.x509up_u1"})
    assert fake_voms.read_text().split() == ["--exists", "--valid", "48:00"]


def test_proxy_check_rounds_up(load, user, fake_voms):
    from tau3mu_gnn.pipelines.common.slurm import check_proxy

    check_proxy(remote_cfg(load, user), 1.5, {"X509_USER_PROXY": "/t3home/u/.x509up_u1"})
    assert fake_voms.read_text().split()[-1] == "2:00"


def test_proxy_invalid(load, user, fake_voms, monkeypatch):
    from tau3mu_gnn.pipelines.common.slurm import ProxyError, check_proxy

    monkeypatch.setenv("FAKE_VOMS_RC", "1")
    with pytest.raises(ProxyError, match="voms-proxy-init --voms cms --valid 192:00"):
        check_proxy(remote_cfg(load, user), 1, {"X509_USER_PROXY": "/t3home/u/.x509up_u1"})


@pytest.mark.parametrize("environ, match", [({}, "X509_USER_PROXY"),
                                            ({"X509_USER_PROXY": "/tmp/x509up_u1"}, "/tmp")])
def test_proxy_location(load, user, fake_voms, environ, match):
    from tau3mu_gnn.pipelines.common.slurm import ProxyError, check_proxy

    with pytest.raises(ProxyError, match=match):
        check_proxy(remote_cfg(load, user), 1, environ)


def test_proxy_not_needed_for_local_output(load, user, fake_voms):
    from tau3mu_gnn.pipelines.common.slurm import check_proxy

    check_proxy(load(user), 1, {})
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
    from tau3mu_gnn.pipelines.common.slurm import Job

    return [Job("limit_2.0", "limit", ("limit", "--cut", "2.0")),
            Job("limit_3.9", "limit", ("limit", "--cut", "3.9")),
            Job("summary", "summary", ("summary",), after=("limit_2.0", "limit_3.9"), dependency="afterany")]


def test_submit_jobs_dependencies(load, user, tmp_path, fake_sbatch):
    from tau3mu_gnn.pipelines.common.slurm import JobEnv, submit_jobs

    ids = submit_jobs(load(user), three_jobs(), "/out/toy_run/phase_c_vs1", tmp_path / "logs", JobEnv.from_environment(ENV))
    assert ids == {"limit_2.0": "1001", "limit_3.9": "1002", "summary": "1003"}
    calls = fake_sbatch.read_text().splitlines()
    assert calls[0] == f"--parsable {tmp_path / 'logs' / 'limit_2.0.sh'}"
    assert calls[2] == (f"--parsable --dependency=afterany:1001:1002 --kill-on-invalid-dep=yes "
                        f"{tmp_path / 'logs' / 'summary.sh'}")
    assert (tmp_path / "logs" / "summary.sh").read_text().startswith("#!/bin/bash")


def test_submit_jobs_dry_run(load, user, tmp_path, fake_sbatch, capsys):
    from tau3mu_gnn.pipelines.common.slurm import JobEnv, submit_jobs

    ids = submit_jobs(load(user), three_jobs(), "/out/toy_run/phase_c_vs1", tmp_path / "logs",
                      JobEnv.from_environment(ENV), dry_run=True)
    assert ids == {"limit_2.0": "DRYRUN-limit_2.0", "limit_3.9": "DRYRUN-limit_3.9", "summary": "DRYRUN-summary"}
    assert not fake_sbatch.exists()
    out = capsys.readouterr().out
    assert (f"sbatch --parsable --dependency=afterany:DRYRUN-limit_2.0:DRYRUN-limit_3.9 --kill-on-invalid-dep=yes "
            f"{tmp_path / 'logs' / 'summary.sh'}") in out
    assert (tmp_path / "logs" / "limit_3.9.sh").is_file()


def test_submit_jobs_failure_lists_submitted(load, user, tmp_path, fake_sbatch, monkeypatch):
    from tau3mu_gnn.pipelines.common.slurm import JobEnv, SubmitError, submit_jobs

    monkeypatch.setenv("FAKE_SBATCH_FAIL_AT", "2")
    with pytest.raises(SubmitError) as e:
        submit_jobs(load(user), three_jobs(), "/out/toy_run/phase_c_vs1", tmp_path / "logs", JobEnv.from_environment(ENV))
    msg = str(e.value)
    assert "limit_3.9" in msg and "invalid partition" in msg and "scancel 1001" in msg


def test_submit_jobs_unknown_dependency(load, user, tmp_path, fake_sbatch):
    from tau3mu_gnn.pipelines.common.slurm import Job, JobEnv, submit_jobs

    with pytest.raises(ValueError, match="skim"):
        submit_jobs(load(user), [Job("bdt", "bdt", ("bdt",), after=("skim",))], "/o/v", tmp_path,
                    JobEnv.from_environment(ENV))


def test_generated_script_runs(tmp_path, monkeypatch):
    """Run a generated script with bash (no SLURM): conda activation, scratch, job CLI, log copy, cleanup.
    Uses the real env of this shell and the real repo defaults; the job is a debug 'summary'."""
    import json

    from tau3mu_gnn.pipelines.analysis.config import load_config
    from tau3mu_gnn.pipelines.common.slurm import Job, JobEnv, render_script

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


def test_submit_jobs_waits_for_earlier_submission(load, user, tmp_path, fake_sbatch):
    from tau3mu_gnn.pipelines.common.slurm import Job, JobEnv, submit_jobs

    jobs = [Job("limit_2.0", "limit", ("limit",), after_ids=("77",)),
            Job("summary", "summary", ("summary",), after=("limit_2.0",), dependency="afterany", after_ids=("77",))]
    submit_jobs(load(user), jobs, "/out/toy_run/phase_c_vs1", tmp_path / "logs", JobEnv.from_environment(ENV))
    calls = fake_sbatch.read_text().splitlines()
    assert calls[0].startswith("--parsable --dependency=afterok:77 --kill-on-invalid-dep=yes ")
    assert calls[1].startswith("--parsable --dependency=afterany:1001,afterok:77 --kill-on-invalid-dep=yes ")


def _env():
    from tau3mu_gnn.pipelines.common.slurm import JobEnv

    return JobEnv(Path("/c/base"), Path("/c/base/envs/T"), Path("/c/base/envs/T/bin/python"), Path("/repo/src"))


@pytest.mark.parametrize("name, resources, args", [
    ("skim", "skim", ("skim", "--x", "a b")),
    ("limit_3.9", "limit", ("limit", "--cut", "3.9")),
])
def test_analysis_script_unchanged(load, user, name, resources, args):
    """Expected text rendered by the code before step 12g (tests/pipelines/expected/, module path updated for the pipelines/ layout)."""
    from tau3mu_gnn.pipelines.common.slurm import Job, render_script

    text = render_script(load(user), Job(name, resources, args), "/out/toy_run/phase_a_vs1",
                         Path("/logs/toy_run/phase_a_vs1"), _env())
    assert text == (Path(__file__).parent / "expected" / f"script_{name}.txt").read_text()


def test_job_module_and_gpu_flags(load_training, training_user, tmp_path):
    from tau3mu_gnn.pipelines.common.slurm import Job, render_script

    cfg = load_training(training_user)
    module = "tau3mu_gnn.pipelines.training.jobs"
    gpu = render_script(cfg, Job("gnn_train", "gnn_train", ("gnn_train",), module=module),
                        "/out/toy_train/gnn_vs2", tmp_path / "logs", _env()).splitlines()
    assert "#SBATCH --account=gpu_gres" in gpu and "#SBATCH --gres=gpu:1" in gpu
    assert gpu.index("#SBATCH --gres=gpu:1") == gpu.index("#SBATCH --cpus-per-task=1") + 1
    assert "#SBATCH --job-name=GNN2.gnnT.toy_train" in gpu
    assert gpu[-1] == f"/c/base/envs/T/bin/python -m {module} gnn_train"
    cpu = render_script(cfg, Job("npvs", "npvs", ("npvs",), module=module),
                        "/out/toy_train/gnn_vs2", tmp_path / "logs", _env())
    assert "#SBATCH --account=t3" in cpu and "--gres" not in cpu


@pytest.mark.parametrize("version, job, expected", [
    ("gnn_vs2", "gnn_train", "GNN2.gnnT.toy_train"),
    ("bdt_vs1", "bdt_train", "BDT1.bdtT.toy_train"),
    ("bdt_vs1", "bdt_infer", "BDT1.bdtI.toy_train"),
    ("gnn_vs3", "graphs_signal", "GNN3.grS.toy_train"),
    ("gnn_vs3", "graphs_background", "GNN3.grB.toy_train"),
    ("benchmark_vs1", "benchmark_limit_3.9", "BEN1.bl3.9.toy_train"),
    ("benchmark_vs1", "benchmark_summary", "BEN1.bsum.toy_train"),
])
def test_training_job_names(version, job, expected):
    from tau3mu_gnn.pipelines.common.slurm import VERSION_CODES, slurm_job_name

    assert slurm_job_name("toy_train", version, job, VERSION_CODES) == expected


def test_analysis_debug_job_keeps_code_d(load, user, tmp_path):
    """bdt_vs2 is an analysis debug folder (D); only training jobs get BDT."""
    from tau3mu_gnn.pipelines.common.slurm import Job, render_script

    cfg = load(user)
    for module, code in (("tau3mu_gnn.pipelines.analysis.jobs", "D2"), ("tau3mu_gnn.pipelines.training.jobs", "BDT2")):
        text = render_script(cfg, Job("bdt", "skim", ("bdt",), module=module), "/out/toy_run/bdt_vs2",
                             tmp_path / "logs", _env())
        assert f"#SBATCH --job-name={code}.bdt.toy_run" in text.splitlines()
