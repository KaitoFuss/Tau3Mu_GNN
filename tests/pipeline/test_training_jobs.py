"""Training job bodies on toy data. Stages are faked (they have their own tests)."""
import json
from pathlib import Path

import pytest

SAMPLES = ("signal", "background")


@pytest.fixture
def cfg(load_training, training_user, tmp_path):
    training_user["paths"]["output_base"] = str(tmp_path / "out")
    training_user["paths"]["log_dir"] = str(tmp_path / "logs")
    return load_training(training_user)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    d = tmp_path / "scratch"
    d.mkdir()
    monkeypatch.setenv("TMPDIR", str(d))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)       # re-read TMPDIR
    return d


def _write(p, text="x"):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(text)
    return Path(p)


@pytest.fixture
def fake_stages(monkeypatch):
    """Fake stage run functions: write small files with the real names, record their inputs."""
    from tau3mu_gnn.pipeline.training import jobs

    seen = {"calls": []}

    def resonance(cfg, out_dir, extra_branches=()):
        seen["calls"].append("resonance")
        seen["extra_branches"] = tuple(extra_branches)
        return {s: list(getattr(cfg.inputs, s)) for s in SAMPLES}

    def skim(cfg, inputs, out_dir):
        seen["calls"].append("skim")
        return {s: _write(Path(out_dir) / f"{s}.root") for s in inputs}

    def bdt_train(cfg, inputs, out_dir):
        seen["calls"].append("bdt_train")
        seen["bdt_train"] = {s: Path(inputs[s]).read_text() for s in SAMPLES}
        return {"classifier": _write(Path(out_dir) / "ISO_classifiers.pck", "clf"),
                "summary": _write(Path(out_dir) / "bdt_train_summary.json", "{}"),
                "plots": [_write(Path(out_dir) / "roc.png")]}

    def bdt_infer(cfg, inputs, classifier, out_dir):
        seen["calls"].append("bdt_infer")
        seen["classifier"] = Path(classifier).read_text()
        out = {s: _write(Path(out_dir) / f"{s}_with_bdt_scores.root") for s in SAMPLES}
        out["scores_pickle"] = _write(Path(out_dir) / "tau_score_corr.pkl")
        return out

    def npvs(cfg, inputs, out_dir):
        seen["calls"].append("npvs")
        seen["npvs"] = {s: Path(inputs[s]).is_file() for s in SAMPLES}
        ch = cfg.channel
        return {"weights": _write(Path(out_dir) / f"signal_{ch}_npvs_weights.npz", "w"),
                "plot_npvs": _write(Path(out_dir) / f"signal_{ch}_npvs.png"),
                "plot_weights": _write(Path(out_dir) / f"signal_{ch}_npvs_weights.png")}

    def graphs(cfg, sample, bdt_file, out_dir, npvs_file=None):
        seen["calls"].append(f"graphs_{sample}")
        seen.setdefault("graphs", {})[sample] = {
            "bdt": Path(bdt_file).is_file(), "npvs": None if npvs_file is None else Path(npvs_file).read_text()}
        name = jobs.graphs_stage.dataset_name(cfg, sample)
        return _write(Path(out_dir) / f"{sample}_{cfg.channel}" / "processed" / f"{name}.pt")

    def gnn_train(cfg, graphs_dir, out_dir, device=None, comet_disabled=False):
        seen["calls"].append("gnn_train")
        seen["gnn_train"] = {"graphs_dir": Path(graphs_dir), "comet_disabled": comet_disabled}
        name = jobs.gnn_train_stage.model_name(cfg)
        model_dir = Path(out_dir) / name
        _write(model_dir / f"{name}.pt")
        return {"model_dir": model_dir, "model_name": name, "best_epoch": 3, "epochs_run": 5,
                "early_stopped": True, "auc_test": 0.9, "summary": _write(Path(out_dir) / "gnn_train_summary.json")}

    def benchmark(cfg, inputs, out_dir, cut):
        seen["calls"].append(f"benchmark_{cut}")
        card = Path(out_dir) / jobs.limit_stage.cut_dir_name(cut)
        _write(card / "combine_asymptotic.log", "log")
        if cut == seen.get("fail_cut"):
            raise RuntimeError("Combine wrote no median expected limit")
        _write(card / "limit.json", json.dumps({"cut": cut, "limit": 1.0 + cut, "limitErr": 0.0}))
        return {}

    monkeypatch.setattr(jobs.resonance_skim, "run", resonance)
    monkeypatch.setattr(jobs.skim_stage, "run", skim)
    monkeypatch.setattr(jobs.bdt_train_stage, "run", bdt_train)
    monkeypatch.setattr(jobs.bdt_infer_stage, "run", bdt_infer)
    monkeypatch.setattr(jobs.npvs_stage, "run", npvs)
    monkeypatch.setattr(jobs.graphs_stage, "run", graphs)
    monkeypatch.setattr(jobs.gnn_train_stage, "run", gnn_train)
    monkeypatch.setattr(jobs.benchmark_stage, "run", benchmark)
    return seen


def run_bdt(cfg):
    from tau3mu_gnn.pipeline.training import jobs
    from tau3mu_gnn.pipeline.training.manifest import create_version

    a = create_version(cfg, "bdt")
    jobs.skim(cfg, a)
    jobs.bdt_train(cfg, a)
    jobs.bdt_infer(cfg, a)
    return a


def test_bdt_phase(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipeline.manifest import read_manifest

    a = Path(run_bdt(cfg))
    m = read_manifest(a)
    assert m["phase"] == "bdt" and m["version"] == "bdt_vs1" and m["upstream"] is None
    assert m["outputs"] == {"skim_signal": "1_skim/signal.root", "skim_background": "1_skim/background.root",
                            "bdt_signal": "2_bdt/signal_with_bdt_scores.root",
                            "bdt_background": "2_bdt/background_with_bdt_scores.root",
                            "classifier": "bdt_train/ISO_classifiers.pck",
                            "bdt_train_summary": "bdt_train/bdt_train_summary.json"}
    assert all((a / rel).is_file() for rel in m["outputs"].values())
    assert fake_stages["extra_branches"] == ("PV_npvs",)
    assert fake_stages["bdt_train"] == {"signal": "x", "background": "x"}   # read from the phase folder
    assert fake_stages["classifier"] == "clf"
    assert "bdt_training" in m["settings"] and "npvs" not in m["settings"]


def run_gnn(cfg, up):
    from tau3mu_gnn.pipeline.training import jobs
    from tau3mu_gnn.pipeline.training.manifest import create_version

    g = create_version(cfg, "gnn")
    jobs.graphs(cfg, g, up, "background")        # may start before npvs
    jobs.npvs(cfg, g, up)
    jobs.graphs(cfg, g, up, "signal")
    jobs.gnn_train(cfg, g, up)
    return g


def test_gnn_phase(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipeline.manifest import read_manifest
    from tau3mu_gnn.pipeline.training import jobs

    up = run_bdt(cfg)
    g = Path(run_gnn(cfg, up))
    m = read_manifest(g)
    name = jobs.gnn_train_stage.model_name(cfg)
    ds = {s: jobs.graphs_stage.dataset_name(cfg, s) for s in SAMPLES}
    assert m["phase"] == "gnn" and m["upstream"] == "bdt_vs1"
    assert m["outputs"] == {"model_dir": f"gnn_train/{name}", "model_name": name,
                            "npvs_weights": "npvs/signal_W_npvs_weights.npz",
                            "graphs_signal": f"graphs/signal_W/processed/{ds['signal']}.pt",
                            "graphs_background": f"graphs/background_W/processed/{ds['background']}.pt",
                            "gnn_train_summary": "gnn_train/gnn_train_summary.json"}
    assert (g / "gnn_train" / name).is_dir()
    assert all((g / m["outputs"][k]).is_file() for k in m["outputs"] if k not in ("model_dir", "model_name"))
    assert fake_stages["graphs"]["signal"] == {"bdt": True, "npvs": "w"}       # npvs file of this phase
    assert fake_stages["graphs"]["background"] == {"bdt": True, "npvs": None}
    assert fake_stages["npvs"] == {"signal": True, "background": True}
    assert fake_stages["gnn_train"]["graphs_dir"] == g / "graphs"
    assert fake_stages["gnn_train"]["comet_disabled"] is False


@pytest.mark.parametrize("job", ["npvs", "graphs_background", "graphs_signal", "benchmark_limit"])
def test_missing_upstream_file_stops_before_stage(cfg, scratch, fake_stages, job):
    from tau3mu_gnn.pipeline.manifest import UpstreamError
    from tau3mu_gnn.pipeline.training import jobs
    from tau3mu_gnn.pipeline.training.manifest import create_version

    up = run_bdt(cfg)
    (Path(up) / "2_bdt" / "signal_with_bdt_scores.root").unlink()
    phase = "benchmark" if job == "benchmark_limit" else "gnn"
    d = create_version(cfg, phase)
    n = len(fake_stages["calls"])
    call = {"npvs": lambda: jobs.npvs(cfg, d, up),
            "graphs_background": lambda: jobs.graphs(cfg, d, up, "background"),
            "graphs_signal": lambda: jobs.graphs(cfg, d, up, "signal"),
            "benchmark_limit": lambda: jobs.benchmark_limit(cfg, d, up, 0.98)}[job]
    with pytest.raises(UpstreamError, match="missing"):
        call()
    assert fake_stages["calls"][n:] == []


def test_benchmark_phase(cfg, scratch, fake_stages):
    from tau3mu_gnn.pipeline.manifest import read_manifest
    from tau3mu_gnn.pipeline.training import jobs
    from tau3mu_gnn.pipeline.training.manifest import create_version

    up = run_bdt(cfg)
    b = create_version(cfg, "benchmark")
    fake_stages["fail_cut"] = 0.99
    jobs.benchmark_limit(cfg, b, up, 0.98)
    with pytest.raises(RuntimeError, match="Combine"):
        jobs.benchmark_limit(cfg, b, up, 0.99)
    assert (Path(b) / "cut_0.99" / "combine_asymptotic.log").is_file()      # copied although the stage failed
    jobs.benchmark_summary(cfg, b, up)
    m = read_manifest(b)
    assert m["phase"] == "benchmark" and m["upstream"] == "bdt_vs1"
    assert m["outputs"] == {"limits_json": "limits.json", "limits_txt": "limits.txt", "limits_png": "limits.png"}
    assert m["missing"] == [0.99]
    assert all((Path(b) / rel).is_file() for rel in m["outputs"].values())


def test_main_dispatch(cfg, scratch, fake_stages, monkeypatch):
    from tau3mu_gnn.pipeline.training import jobs
    from tau3mu_gnn.pipeline.training.manifest import create_version

    monkeypatch.setattr(jobs, "load_training_config", lambda path: cfg)
    a = create_version(cfg, "bdt")
    for job in ("skim", "bdt_train", "bdt_infer"):
        jobs.main([job, "--config", str(cfg.user_file), "--phase-dir", a])
    assert (Path(a) / "manifest.json").is_file()
    g = create_version(cfg, "gnn")
    jobs.main(["npvs", "--config", "x.yaml", "--phase-dir", g, "--upstream", a])
    jobs.main(["graphs", "--config", "x.yaml", "--phase-dir", g, "--upstream", a, "--sample", "signal"])
    assert fake_stages["graphs"]["signal"]["npvs"] == "w"
    b = create_version(cfg, "benchmark")
    jobs.main(["benchmark_limit", "--config", "x.yaml", "--phase-dir", b, "--upstream", a, "--cut", "0.98"])
    assert (Path(b) / "cut_0.98" / "limit.json").is_file()


@pytest.mark.parametrize("argv", [
    ["npvs"], ["graphs", "--upstream", "u"], ["graphs", "--sample", "signal"], ["gnn_train"],
    ["benchmark_limit", "--upstream", "u"], ["benchmark_summary"],
])
def test_main_required_arguments(cfg, monkeypatch, capsys, argv):
    from tau3mu_gnn.pipeline.training import jobs

    monkeypatch.setattr(jobs, "load_training_config", lambda path: pytest.fail("must not load the config"))
    with pytest.raises(SystemExit):
        jobs.main(argv[:1] + ["--config", "x.yaml", "--phase-dir", "p"] + argv[1:])
    assert f"job {argv[0]} needs" in capsys.readouterr().err
