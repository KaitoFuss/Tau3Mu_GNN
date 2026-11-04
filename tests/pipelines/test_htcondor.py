import json
import os
import subprocess
from pathlib import Path

import pytest

from test_slurm import ENV

EOS = "root://eosuser.cern.ch//eos/user/t/toy/tau3mu_pipeline"
DCACHE = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/x"


def _env():
    from tau3mu_gnn.pipelines.common.batch import JobEnv

    return JobEnv(Path("/c/base"), Path("/c/base/envs/T"), Path("/c/base/envs/T/bin/python"), Path("/repo/src"))


@pytest.fixture
def condor_user(user, tmp_path):
    user["scheduler"] = "htcondor"
    user["paths"]["output_base"] = str(tmp_path / "out")
    user["paths"]["log_dir"] = str(tmp_path / "logs")
    user["limit"]["gnn_score_cut"] = [2.0, 3.9]
    return user


@pytest.fixture
def fake_dag(tmp_path, monkeypatch):
    """condor_submit_dag that logs its arguments and the DAG text. FAKE_DAG_FAIL=1: it fails."""
    bin_dir = tmp_path / "cbin"
    bin_dir.mkdir()
    log = tmp_path / "dag.log"
    exe = bin_dir / "condor_submit_dag"
    exe.write_text(f"""#!/bin/sh
if [ "${{FAKE_DAG_FAIL:-0}}" = 1 ]; then echo "ERROR: no schedd" >&2; exit 1; fi
echo "$*" >> {log}
pwd > {log}.cwd
for a in "$@"; do last=$a; done
cat "$last" >> {log}
echo "Submitting job(s)."
echo "1 job(s) submitted to cluster 4242."
""")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


@pytest.mark.parametrize("mem, mb", [("4G", 4096), ("128G", 131072), ("512M", 512), ("2048", 2048),
                                     ("1T", 1048576), ("1024K", 1)])
def test_memory_mb(mem, mb):
    from tau3mu_gnn.pipelines.common.htcondor import memory_mb

    assert memory_mb(mem) == mb


def test_node_name_is_unique_and_safe():
    from tau3mu_gnn.pipelines.common.htcondor import node_name

    assert node_name("/out/run/limit_vs2", "limit_3.9") == "limit_vs2-limit_3_9"
    assert node_name("root://h//eos/run/bdt_vs1", "skim") == "bdt_vs1-skim"


@pytest.mark.parametrize("path, needs", [
    (DCACHE, True), (EOS, False), ("root://eoscms.cern.ch//eos/cms/store/x", False), ("/tmp/out", False),
])
def test_needs_proxy(path, needs):
    from tau3mu_gnn.pipelines.common import storage

    assert storage.needs_proxy(path) is needs


# ---------------------------------------------------------------- config

def test_scheduler_default_is_htcondor(load, user):
    del user["scheduler"]
    cfg = load(user)
    assert cfg.scheduler == "htcondor"
    assert cfg.htcondor.jobs["limit"].mem == "4G" and cfg.htcondor.jobs["limit"].gpus == 0


def test_scheduler_must_be_known(load, user):
    from tau3mu_gnn.pipelines.common.config import ConfigError

    user["scheduler"] = "pbs"
    with pytest.raises(ConfigError, match="'scheduler' must be one of"):
        load(user)


def test_htcondor_table_errors():
    from tau3mu_gnn.pipelines.common.config import _htcondor_jobs_parser

    parse = _htcondor_jobs_parser(("a", "b"))
    with pytest.raises(ValueError) as e:
        parse({"a": {"mem": "4G", "time": "1:00", "cpus": 1, "gpus": -1, "partition": "x"},
               "c": {"mem": "4G", "time": "01:00:00", "cpus": 1}})
    msg = str(e.value)
    for part in ("missing job 'b'", "unknown job 'c'", "job 'a': unknown key 'partition'",
                 "job 'a': time must look like", "job 'a': gpus must be an integer >= 0"):
        assert part in msg


def test_training_gpu_job(load_training, training_user):
    training_user["scheduler"] = "htcondor"
    j = load_training(training_user).htcondor.jobs["gnn_train"]
    assert (j.gpus, j.requirements) == (1, "(GPUs_Capability >= 7.0)")


# ---------------------------------------------------------------- rendering

def test_render_submit_cpu_job(load, condor_user, tmp_path):
    from tau3mu_gnn.pipelines.common.batch import Job
    from tau3mu_gnn.pipelines.common.htcondor import render_submit

    lines = render_submit(load(condor_user), Job("limit_3.9", "limit", ("limit",)), tmp_path / "L").splitlines()
    assert lines[0] == "universe       = vanilla"
    assert f"executable     = {tmp_path}/L/limit_3.9.sh" in lines
    assert f"output         = {tmp_path}/L/limit_3.9-$(ClusterId).out" in lines
    assert "request_memory = 4096" in lines and "request_cpus   = 1" in lines
    assert "+MaxRuntime    = 3600" in lines
    assert lines[-1] == "queue"
    text = "\n".join(lines)
    assert "request_gpus" not in text and "requirements" not in text and "x509" not in text


def test_render_submit_gpu_job(load_training, training_user, tmp_path):
    from tau3mu_gnn.pipelines.common.batch import TRAINING_MODULE, Job
    from tau3mu_gnn.pipelines.common.htcondor import render_submit

    training_user["scheduler"] = "htcondor"
    cfg = load_training(training_user)
    lines = render_submit(cfg, Job("gnn_train", "gnn_train", ("gnn_train",), module=TRAINING_MODULE),
                          tmp_path).splitlines()
    assert "request_gpus   = 1" in lines
    assert "requirements   = (GPUs_Capability >= 7.0)" in lines


@pytest.mark.parametrize("output_base, proxy", [(DCACHE, True), (EOS, False)])
def test_render_submit_sends_proxy_only_for_dcache(load, condor_user, tmp_path, output_base, proxy):
    from tau3mu_gnn.pipelines.common.batch import Job
    from tau3mu_gnn.pipelines.common.htcondor import render_submit

    condor_user["paths"]["output_base"] = output_base
    text = render_submit(load(condor_user), Job("skim", "skim", ("skim",)), tmp_path)
    assert ("use_x509userproxy = true" in text) is proxy


def test_render_submit_refuses_spaces(load, condor_user, tmp_path):
    from tau3mu_gnn.pipelines.common.batch import Job
    from tau3mu_gnn.pipelines.common.htcondor import render_submit

    with pytest.raises(ValueError, match="spaces"):
        render_submit(load(condor_user), Job("skim", "skim", ("skim",)), tmp_path / "a b")


def test_render_script_is_valid_bash(load, condor_user, tmp_path):
    from tau3mu_gnn.pipelines.common.batch import Job
    from tau3mu_gnn.pipelines.common.htcondor import render_script

    text = render_script(load(condor_user), Job("skim", "skim", ("skim", "--x", "a b")), "/out/run/bdt_vs1", _env())
    path = tmp_path / "job.sh"
    path.write_text(text)
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert text.splitlines()[-1] == ("/c/base/envs/T/bin/python -m tau3mu_gnn.pipelines.analysis.jobs skim "
                                     "--x 'a b' >\"$OUT\" 2>\"$ERR\"")
    assert "/out/run/bdt_vs1/logs" in text


def test_render_dag_dependencies():
    from tau3mu_gnn.pipelines.common.batch import Job
    from tau3mu_gnn.pipelines.common.htcondor import render_dag

    gnn = [Job("gnn_signal", "gnn_signal", ()), Job("eval", "eval", (), after=("gnn_signal",))]
    limit = [Job("limit_2.0", "limit", (), after_ids=("gnn_vs1-eval",)),
             Job("summary", "summary", (), after=("limit_2.0",), dependency="afterany", after_ids=("gnn_vs1-eval",))]
    lines = render_dag([("/o/gnn_vs1", gnn, Path("/l/gnn_vs1")), ("/o/limit_vs1", limit, Path("/l/limit_vs1"))])
    assert lines.splitlines() == [
        "JOB gnn_vs1-gnn_signal /l/gnn_vs1/gnn_signal.sub",
        "JOB gnn_vs1-eval /l/gnn_vs1/eval.sub",
        "JOB limit_vs1-limit_2_0 /l/limit_vs1/limit_2.0.sub",
        "JOB limit_vs1-summary /l/limit_vs1/summary.sub",
        "PARENT gnn_vs1-gnn_signal CHILD gnn_vs1-eval",
        "PARENT gnn_vs1-eval CHILD limit_vs1-limit_2_0",
        "PARENT limit_vs1-limit_2_0 gnn_vs1-eval CHILD limit_vs1-summary",
        "SCRIPT POST limit_vs1-limit_2_0 /bin/true",
    ]


# ---------------------------------------------------------------- submission

def test_bdt_phase_one_dag(load, condor_user, fake_dag, tmp_path):
    from tau3mu_gnn.pipelines.analysis.driver import submit_phase
    from tau3mu_gnn.pipelines.common.batch import JobEnv

    s = submit_phase(load(condor_user), "bdt", env=JobEnv.from_environment(ENV))
    assert s.job_ids == {"skim": "4242:bdt_vs1-skim", "bdt": "4242:bdt_vs1-bdt"}
    assert json.loads((Path(s.version_dir) / "jobs.json").read_text()) == s.job_ids
    calls = fake_dag.read_text().splitlines()
    assert calls[0] == f"-batch-name toy_run.bdt_vs1 {s.log_folder}/jobs.dag"
    assert "PARENT bdt_vs1-skim CHILD bdt_vs1-bdt" in calls
    assert (s.log_folder / "skim.sub").is_file() and os.access(s.log_folder / "skim.sh", os.X_OK)


def test_all_is_one_dag_over_three_phases(load, condor_user, fake_dag):
    from tau3mu_gnn.pipelines.analysis.driver import submit_all
    from tau3mu_gnn.pipelines.common.batch import JobEnv

    subs = submit_all(load(condor_user), env=JobEnv.from_environment(ENV))
    calls = fake_dag.read_text().splitlines()
    assert sum(c.startswith("-batch-name") for c in calls) == 1
    assert calls[0].startswith("-batch-name toy_run.bdt_vs1+gnn_vs1+limit_vs1 ")
    assert "PARENT bdt_vs1-bdt CHILD gnn_vs1-gnn_signal" in calls
    assert "PARENT gnn_vs1-eval CHILD limit_vs1-limit_2_0" in calls
    assert "SCRIPT POST limit_vs1-limit_3_9 /bin/true" in calls
    for s in subs:
        ids = json.loads((Path(s.version_dir) / "jobs.json").read_text())
        assert ids == s.job_ids and all(i.startswith("4242:") for i in ids.values())


def test_training_all_is_one_dag(load_training, training_user, tmp_path, fake_dag):
    from tau3mu_gnn.pipelines.common.batch import JobEnv
    from tau3mu_gnn.pipelines.training.driver import submit_all

    training_user["scheduler"] = "htcondor"
    training_user["paths"]["output_base"] = str(tmp_path / "out")
    training_user["paths"]["log_dir"] = str(tmp_path / "logs")
    submit_all(load_training(training_user), env=JobEnv.from_environment(ENV))
    calls = fake_dag.read_text().splitlines()
    assert calls[0].startswith("-batch-name toy_train.bdt_vs1+gnn_vs1+benchmark_vs1 ")
    assert "PARENT bdt_vs1-bdt_infer CHILD gnn_vs1-npvs" in calls
    assert "PARENT bdt_vs1-bdt_infer CHILD benchmark_vs1-benchmark_limit_0_98" in calls


def test_dry_run_prints_command_submits_nothing(load, condor_user, fake_dag, capsys):
    from tau3mu_gnn.pipelines.analysis.driver import submit_phase
    from tau3mu_gnn.pipelines.common.batch import JobEnv

    cfg = load(condor_user)
    s = submit_phase(cfg, "bdt", dry_run=True, env=JobEnv.from_environment(ENV))
    assert not fake_dag.exists() and not Path(cfg.output_dir).exists()
    assert "condor_submit_dag -batch-name toy_run.bdt_vs1" in capsys.readouterr().out
    assert (s.log_folder / "jobs.dag").is_file() and s.job_ids["skim"] == "DRYRUN-skim"


def test_submit_failure(load, condor_user, fake_dag, monkeypatch):
    from tau3mu_gnn.pipelines.analysis.driver import submit_phase
    from tau3mu_gnn.pipelines.common.batch import JobEnv, SubmitError

    monkeypatch.setenv("FAKE_DAG_FAIL", "1")
    with pytest.raises(SubmitError, match="no schedd"):
        submit_phase(load(condor_user), "bdt", env=JobEnv.from_environment(ENV))


def test_eos_output_needs_no_proxy(load, condor_user, fake_dag, monkeypatch, tmp_path):
    from tau3mu_gnn.pipelines.common.batch import check_proxy

    monkeypatch.delenv("X509_USER_PROXY", raising=False)
    condor_user["paths"]["output_base"] = EOS
    check_proxy(load(condor_user), 100.0)   # no ProxyError


def test_afs_free_mb_not_afs(tmp_path):
    from tau3mu_gnn.pipelines.common.htcondor import afs_free_mb

    assert afs_free_mb(tmp_path) is None


def test_generated_script_runs(tmp_path):
    """Run a generated script with bash (no HTCondor): conda activation, scratch, job CLI, log copy, cleanup,
    exit code. Uses the real env of this shell and the real repo defaults; the job is a debug 'summary'."""
    from tau3mu_gnn.pipelines.analysis.config import load_config
    from tau3mu_gnn.pipelines.common.batch import Job, JobEnv
    from tau3mu_gnn.pipelines.common.htcondor import render_script

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
    assert cfg.scheduler == "htcondor"
    src = tmp_path / "out" / "script_test" / "limit_vs1"
    (src / "cut_3.9").mkdir(parents=True)
    (src / "cut_3.9" / "limit.json").write_text(json.dumps({"cut": 3.9, "limit": 1.25, "limitErr": 0.0}))
    version_dir = str(tmp_path / "out" / "script_test" / "debug" / "summary_vs1")
    base = ("summary", "--config", str(cfg_file), "--phase-dir", version_dir, "--no-manifest")
    scratch = tmp_path / "condor_scratch"
    scratch.mkdir()
    run_env = {**os.environ, "_CONDOR_SCRATCH_DIR": str(scratch), "TAU3MU_JOB_ID": "77"}

    script = tmp_path / "summary.sh"
    script.write_text(render_script(cfg, Job("summary", "summary", (*base, "--source", str(src))), version_dir,
                                    JobEnv.from_environment()))
    res = subprocess.run(["bash", str(script)], env=run_env, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert json.loads((Path(version_dir) / "limits.json").read_text())["best_cut"] == 3.9
    assert (Path(version_dir) / "logs" / "summary-77.out").is_file()             # log copied by the trap
    assert list(scratch.iterdir()) == []                                          # scratch removed

    bad = tmp_path / "bad.sh"   # a failing job: the exit code reaches HTCondor, the log is still copied
    bad.write_text(render_script(cfg, Job("bad", "summary", (*base, "--source", str(tmp_path / "none"))),
                                 version_dir, JobEnv.from_environment()))
    res = subprocess.run(["bash", str(bad)], env=run_env, capture_output=True, text=True, timeout=300)
    assert res.returncode != 0
    assert (Path(version_dir) / "logs" / "bad-77.err").is_file()
    assert res.stderr.strip()                                                     # job stderr shown to HTCondor


@pytest.mark.parametrize("value, environ, expected", [
    ("/eos/user/$USER_INITIAL/$USER/x", {"USER": "kfuss"}, "/eos/user/k/kfuss/x"),
    ("/eos/user/${USER_INITIAL}/${USER}/x", {"USER": "abc"}, "/eos/user/a/abc/x"),
    ("/eos/user/$USER_INITIAL/x", {"USER": "kfuss", "USER_INITIAL": "z"}, "/eos/user/z/x"),   # set: wins
    ("/a/$USER_INITIALS/x", {"USER": "kfuss", "USER_INITIALS": "q"}, "/a/q/x"),            # another variable
])
def test_user_initial(monkeypatch, value, environ, expected):
    from tau3mu_gnn.pipelines.common.config import _user_path

    for k in ("USER", "USER_INITIAL", "USER_INITIALS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in environ.items():
        monkeypatch.setenv(k, v)
    assert _user_path(value) == expected


def test_user_initial_needs_user(monkeypatch):
    from tau3mu_gnn.pipelines.common.config import _user_path

    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USER_INITIAL", raising=False)
    with pytest.raises(ValueError, match=r"\$USER_INITIAL is not set"):
        _user_path("/eos/user/$USER_INITIAL/x")


def test_script_sets_user_for_the_config(load, condor_user, tmp_path):
    """The job reads the config again on the node; its paths may use $USER."""
    from tau3mu_gnn.pipelines.common.batch import Job
    from tau3mu_gnn.pipelines.common.htcondor import render_script

    text = render_script(load(condor_user), Job("skim", "skim", ("skim",)), "/out/run/bdt_vs1", _env())
    path = tmp_path / "probe.sh"
    head = text.split("JOB_ID=")[0]                      # shebang, set -eo pipefail, USER line
    path.write_text(head + 'echo "$USER"\n')
    env = {k: v for k, v in os.environ.items() if k != "USER"}
    out = subprocess.run(["bash", str(path)], env=env, capture_output=True, text=True, check=True).stdout
    assert out.strip() == subprocess.run(["id", "-un"], capture_output=True, text=True).stdout.strip()


def test_job_env_outside_conda_base(load, user, tmp_path):
    """lxplus: env /eos/.../envs/Tau3Mu, conda base /eos/.../miniforge3 (not its parent)."""
    from tau3mu_gnn.pipelines.common.batch import EnvError, JobEnv

    base, prefix = tmp_path / "miniforge3", tmp_path / "envs" / "T"
    for f in (base / "bin" / "activate", base / "bin" / "conda", prefix / "bin" / "python"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("")
    user["paths"]["conda_env"] = str(prefix)
    cfg = load(user)
    env = JobEnv.from_config(cfg, environ={"CONDA_EXE": str(base / "bin" / "conda")})
    assert (env.conda_base, env.env_prefix) == (base, prefix)
    with pytest.raises(EnvError, match="activate"):
        JobEnv.from_config(cfg, environ={})


def test_submit_runs_in_the_log_folder(load, condor_user, fake_dag, monkeypatch, tmp_path):
    """CERN schedds refuse a DAG submitted from /eos (Iwd = the current folder): submit from the log folder."""
    from tau3mu_gnn.pipelines.analysis.driver import submit_phase
    from tau3mu_gnn.pipelines.common.batch import JobEnv

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    s = submit_phase(load(condor_user), "bdt", env=JobEnv.from_environment(ENV))
    assert Path(str(fake_dag) + ".cwd").read_text().strip() == str(s.log_folder)


@pytest.mark.parametrize("scheduler, module, name", [
    ("slurm", "tau3mu_gnn.pipelines.common.slurm", "SlurmBatch"),
    ("htcondor", "tau3mu_gnn.pipelines.common.htcondor", "HTCondorBatch"),
])
def test_make_batch_per_scheduler(load, user, scheduler, module, name):
    """Each batch system has its Batch class in its own module."""
    from tau3mu_gnn.pipelines.common.batch import Batch, make_batch

    user["scheduler"] = scheduler
    b = make_batch(load(user))
    assert isinstance(b, Batch) and type(b).__module__ == module and type(b).__name__ == name
