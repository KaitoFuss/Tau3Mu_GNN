"""Job bodies on toy data. Stages are faked (they have their own tests), except summarize and, with
Combine installed, the real limit stage."""
import json
import shutil
from pathlib import Path

import pytest

from test_limit import needs_combine, toy_inputs  # noqa: F401  (fixture reused)

SAMPLES = ("signal", "background")


@pytest.fixture
def cfg(load, user, tmp_path):
    user["paths"]["output_base"] = str(tmp_path / "out")
    user["limit"]["gnn_score_cut"] = [2.0, 3.9]
    user["limit"]["signal_weight"] = 0.5
    return load(user)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    d = tmp_path / "scratch"
    d.mkdir()
    monkeypatch.setenv("TMPDIR", str(d))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)       # re-read TMPDIR
    return d


@pytest.fixture
def fake_stages(monkeypatch):
    """Fake stage run functions: write small files with the real names, record their inputs."""
    from tau3mu_gnn.pipelines.analysis import jobs

    seen = {}

    def write(p, text="x"):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(text)
        return Path(p)

    def resonance(cfg, out_dir):
        seen["resonance"] = Path(out_dir)
        return {s: list(getattr(cfg.inputs, s)) for s in SAMPLES}

    def skim(cfg, inputs, out_dir):
        seen["skim"] = dict(inputs)
        return {s: write(Path(out_dir) / f"{s}.root") for s in inputs}

    def bdt(cfg, inputs, out_dir):
        seen["bdt"] = {s: Path(inputs[s]).read_text() for s in SAMPLES}   # inputs must be readable
        out = {s: write(Path(out_dir) / f"{s}_with_bdt_scores.root") for s in SAMPLES}
        out["scores_pickle"] = write(Path(out_dir) / "tau_score_corr.pkl")
        return out

    def gnn(cfg, inputs, out_dir, samples=SAMPLES):
        seen.setdefault("gnn", []).append(dict(inputs))
        out = {}
        for s in samples:
            for key, path in jobs.gnn_stage.output_files(out_dir, s).items():
                out[key] = write(path)
        return out

    def evaluation(cfg, inputs, out_dir):
        seen["eval"] = {k: Path(v).exists() for k, v in inputs.items()}
        return {name: write(Path(out_dir) / f"{name}.png") for name in jobs.evaluation_stage.PLOTS}

    def limit(cfg, inputs, out_dir, cut):
        card = Path(out_dir) / jobs.limit_stage.cut_dir_name(cut)
        write(card / "combine_asymptotic.log", "log")
        if cut == seen.get("fail_cut"):
            raise RuntimeError("Combine wrote no median expected limit")
        write(card / "limit.json", json.dumps({"cut": cut, "limit": 1.0 + abs(cut - 3.9), "limitErr": 0.0}))
        return {}

    monkeypatch.setattr(jobs.resonance_skim, "run", resonance)
    monkeypatch.setattr(jobs.skim_stage, "run", skim)
    monkeypatch.setattr(jobs.bdt_stage, "run", bdt)
    monkeypatch.setattr(jobs.gnn_stage, "run", gnn)
    monkeypatch.setattr(jobs.evaluation_stage, "run", evaluation)
    def benchmark(cfg, inputs, out_dir, cut):
        seen.setdefault("benchmark", []).append({s: Path(inputs[s]).name for s in SAMPLES})
        card = Path(out_dir) / jobs.limit_stage.cut_dir_name(cut)
        write(card / "combine_asymptotic.log", "log")
        if cut == seen.get("fail_cut"):
            raise RuntimeError("Combine wrote no median expected limit")
        write(card / "limit.json", json.dumps({"cut": cut, "limit": 2.0 - cut, "limitErr": 0.0}))
        return {}

    monkeypatch.setattr(jobs.limit_stage, "run", limit)
    monkeypatch.setattr(jobs.benchmark_stage, "run", benchmark)
    return seen


def run_bdt_phase(cfg):
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am

    a = am.create_version(cfg, "bdt")
    jobs.skim(cfg, a)
    jobs.bdt(cfg, a)
    return a


def run_gnn_phase(cfg):
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am

    up = am.pick_upstream(cfg, "gnn")
    b = am.create_version(cfg, "gnn")
    for s in SAMPLES:
        jobs.gnn(cfg, b, up, s)
    jobs.evaluation(cfg, b, up)
    return b


def test_bdt_phase(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipelines.common.manifest import read_manifest

    a = Path(run_bdt_phase(cfg))
    m = read_manifest(a)
    assert m["phase"] == "bdt" and m["version"] == "bdt_vs1" and m["upstream"] is None
    assert m["outputs"] == {"skim_signal": "1_skim/signal.root", "skim_background": "1_skim/background.root",
                            "bdt_signal": "2_bdt/signal_with_bdt_scores.root",
                            "bdt_background": "2_bdt/background_with_bdt_scores.root",
                            "scores_pickle": "2_bdt/tau_score_corr.pkl"}
    assert all((a / rel).is_file() for rel in m["outputs"].values())
    assert (a / "config_used.yaml").is_file()
    assert fake_stages["bdt"] == {"signal": "x", "background": "x"}      # read from the phase folder


def test_gnn_and_limit_phases(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am
    from tau3mu_gnn.pipelines.common.manifest import read_manifest

    run_bdt_phase(cfg)
    b = Path(run_gnn_phase(cfg))
    mb = read_manifest(b)
    assert mb["upstream"] == "bdt_vs1"
    assert {"gnn_signal", "gnn_background", "signal_scores", "background_scores", "eval_roc"} <= set(mb["outputs"])
    assert all((b / rel).is_file() for rel in mb["outputs"].values())
    assert [list(i) for i in fake_stages["gnn"]] == [["signal"], ["background"]]   # one sample per job
    assert all(fake_stages["eval"].values())

    up = am.pick_upstream(cfg, "limit")
    c = am.create_version(cfg, "limit")
    for cut in cfg.limit.gnn_score_cut:
        jobs.limit(cfg, c, up, cut)
    jobs.summary(cfg, c, up)
    mc = read_manifest(c)
    assert mc["upstream"] == "gnn_vs1" and mc["missing"] == []
    assert mc["outputs"]["limit_3.9"] == "cut_3.9/limit.json" and mc["outputs"]["limits_json"] == "limits_gnn.json"
    assert json.loads((Path(c) / "limits_gnn.json").read_text())["best_cut"] == 3.9


def test_rerun_gnn_phase_gives_new_version(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipelines.analysis import manifest as am

    run_bdt_phase(cfg)
    run_gnn_phase(cfg)
    b2 = run_gnn_phase(cfg)
    assert b2.endswith("gnn_vs2")
    assert am.pick_upstream(cfg, "limit") == b2


def test_failed_limit_still_copies_cut_folder(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am
    from tau3mu_gnn.pipelines.common.manifest import read_manifest

    run_bdt_phase(cfg)
    run_gnn_phase(cfg)
    up = am.pick_upstream(cfg, "limit")
    c = am.create_version(cfg, "limit")
    fake_stages["fail_cut"] = 2.0
    with pytest.raises(RuntimeError):
        jobs.limit(cfg, c, up, 2.0)
    assert (Path(c) / "cut_2.0" / "combine_asymptotic.log").read_text() == "log"
    jobs.limit(cfg, c, up, 3.9)
    jobs.summary(cfg, c, up)                                  # afterany: runs anyway
    assert read_manifest(c)["missing"] == [2.0]


def test_gnn_job_stops_on_incomplete_upstream(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am
    from tau3mu_gnn.pipelines.common.manifest import UpstreamError

    a = am.create_version(cfg, "bdt")
    jobs.skim(cfg, a)                                         # bdt job never ran: no manifest
    b = am.create_version(cfg, "gnn")
    with pytest.raises(UpstreamError, match="manifest"):
        jobs.gnn(cfg, b, a, "signal")
    assert "gnn" not in fake_stages


def test_debug_bdt_reads_source_no_manifest(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am

    a = run_bdt_phase(cfg)
    d = am.create_version(cfg, "bdt", parent=f"{cfg.output_dir}/debug")
    jobs.bdt(cfg, d, source_dir=a, manifest=False)
    assert (Path(d) / "2_bdt" / "tau_score_corr.pkl").is_file()
    assert not (Path(d) / "manifest.json").exists()


def test_main_dispatch(cfg, scratch, fake_stages, monkeypatch):
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am

    monkeypatch.setattr(jobs, "load_config", lambda path: cfg)   # toy config needs the fake repo defaults
    a = am.create_version(cfg, "bdt")
    jobs.main(["skim", "--config", str(cfg.user_file), "--phase-dir", a])
    assert (Path(a) / "1_skim" / "signal.root").is_file()
    with pytest.raises(SystemExit):
        jobs.main(["gnn", "--config", str(cfg.user_file), "--phase-dir", a])   # needs --upstream, --sample


@needs_combine
def test_limit_and_summary_jobs_real_stage(cfg, scratch, toy_inputs, tmp_path):
    """Real stage 5 + Combine on the toy stage 3 files, as phase limit jobs."""
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am
    from tau3mu_gnn.pipelines.common.manifest import read_manifest
    from tau3mu_gnn.pipelines.analysis.manifest import write_manifest

    b = Path(cfg.output_dir) / "gnn_vs1"
    (b / "3_gnn").mkdir(parents=True)
    for s, f in zip(SAMPLES, toy_inputs):
        shutil.copy(f, b / "3_gnn" / f"{s}_with_gnn_scores.root")
    write_manifest(cfg, b, "gnn", "bdt_vs1",
                   {f"gnn_{s}": f"3_gnn/{s}_with_gnn_scores.root" for s in SAMPLES})
    c = am.create_version(cfg, "limit")
    for cut in cfg.limit.gnn_score_cut:
        jobs.limit(cfg, c, str(b), cut)
    jobs.summary(cfg, c, str(b))
    m = read_manifest(c)
    assert m["missing"] == [] and set(m["outputs"]) >= {"limit_2.0", "limit_3.9", "limits_json"}
    assert (Path(c) / "cut_3.9" / "datacard.txt").is_file()


def test_eval_job_checks_upstream_phase(cfg, scratch, fake_stages):
    """Debug 'stage eval' with --upstream pointing at a gnn folder: clear error, not KeyError."""
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.common.manifest import UpstreamError

    run_bdt_phase(cfg)
    b = run_gnn_phase(cfg)
    with pytest.raises(UpstreamError, match="bdt"):
        jobs.evaluation(cfg, b, b, source_dir=b, manifest=False)


def test_summary_debug_without_upstream(cfg, scratch, fake_stages, monkeypatch):
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am

    run_bdt_phase(cfg)
    run_gnn_phase(cfg)
    up = am.pick_upstream(cfg, "limit")
    c = am.create_version(cfg, "limit")
    for cut in cfg.limit.gnn_score_cut:
        jobs.limit(cfg, c, up, cut)
    d = am.create_version(cfg, "summary", parent=f"{cfg.output_dir}/debug")
    monkeypatch.setattr(jobs, "load_config", lambda path: cfg)
    jobs.main(["summary", "--config", "x.yaml", "--phase-dir", d, "--source", c, "--no-manifest"])
    assert (Path(d) / "limits_gnn.json").is_file() and not (Path(d) / "manifest.json").exists()
    with pytest.raises(SystemExit):
        jobs.main(["summary", "--config", "x.yaml", "--phase-dir", d])        # manifest needs --upstream


def test_job_logs_what_it_does(cfg, scratch, fake_stages, caplog):
    import logging

    caplog.set_level(logging.INFO)
    a = run_bdt_phase(cfg)
    assert "job skim" in caplog.text and "job bdt" in caplog.text and a in caplog.text


def test_benchmark_phase(cfg, scratch, fake_stages):
    """Benchmark reads the stage 2 files of the analysis bdt version; summary files get the tail _benchmark."""
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am
    from tau3mu_gnn.pipelines.common.manifest import read_manifest

    a = run_bdt_phase(cfg)
    b = am.create_version(cfg, "benchmark")
    fake_stages["fail_cut"] = 0.99
    jobs.benchmark_limit(cfg, b, a, 0.98)
    with pytest.raises(RuntimeError):
        jobs.benchmark_limit(cfg, b, a, 0.99)
    assert fake_stages["benchmark"][0] == {s: f"{s}_with_bdt_scores.root" for s in SAMPLES}
    assert (Path(b) / "cut_0.99" / "combine_asymptotic.log").is_file()        # copied although it failed
    jobs.benchmark_summary(cfg, b, a)
    m = read_manifest(b)
    assert m["phase"] == "benchmark" and m["upstream"] == "bdt_vs1" and m["missing"] == [0.99]
    assert m["outputs"]["limits_json"] == "limits_benchmark.json" and m["outputs"]["limit_0.98"] == "cut_0.98/limit.json"
    assert json.loads((Path(b) / "limits_benchmark.json").read_text())["best_cut"] == 0.98
    assert not (Path(b) / "limits_gnn.json").exists()


def test_benchmark_job_needs_a_bdt_version(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am
    from tau3mu_gnn.pipelines.common.manifest import UpstreamError

    run_bdt_phase(cfg)
    g = run_gnn_phase(cfg)
    with pytest.raises(UpstreamError, match="needs a bdt folder"):
        jobs.benchmark_limit(cfg, am.create_version(cfg, "benchmark"), g, 0.98)


def test_main_benchmark_dispatch(cfg, scratch, fake_stages, monkeypatch):
    from tau3mu_gnn.pipelines.analysis import jobs
    from tau3mu_gnn.pipelines.analysis import manifest as am

    monkeypatch.setattr(jobs, "load_config", lambda path: cfg)   # toy config needs the fake repo defaults

    a = run_bdt_phase(cfg)
    b = am.create_version(cfg, "benchmark")
    base = ["--config", str(cfg.user_file), "--phase-dir", b, "--upstream", a]
    jobs.main(["benchmark_limit", *base, "--cut", "0.98"])
    jobs.main(["benchmark_limit", *base, "--cut", "0.99"])
    jobs.main(["benchmark_summary", *base])
    assert (Path(b) / "manifest.json").is_file() and (Path(b) / "limits_benchmark.txt").is_file()
    with pytest.raises(SystemExit):
        jobs.main(["benchmark_limit", *base])                                   # no --cut
