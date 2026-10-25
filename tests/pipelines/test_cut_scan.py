import json
import math
import subprocess
from pathlib import Path

import pytest

from test_limit import needs_combine, toy_inputs  # noqa: F401  (fixture reused)


def scan_cfg(load, user, cuts):
    user["limit"]["gnn_score_cut"] = list(cuts)
    return load(user)


def write_limit(out_dir, cut, limit, folder_cut=None, err=0.0):
    """As stage 5 writes it: <out_dir>/cut_<cut>/limit.json."""
    d = out_dir / f"cut_{folder_cut if folder_cut is not None else cut}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "limit.json").write_text(json.dumps({"cut": cut, "limit": limit, "limitErr": err}))
    return d / "limit.json"


def test_summary_best_cut_and_files(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [4.5, 2.0, 3.9])          # config order is not sorted
    for cut, lim in [(2.0, 1.7), (3.9, 1.2890625), (4.5, 1.5)]:
        write_limit(tmp_path, cut, lim)

    s = limit.summarize(cfg, tmp_path)

    assert s["best_cut"] == 3.9 and s["best_limit"] == 1.2890625 and s["missing"] == []
    assert [r["cut"] for r in s["rows"]] == [2.0, 3.9, 4.5]
    assert [r["best"] for r in s["rows"]] == [False, True, False]
    assert s["files"] == {k: tmp_path / v for k, v in limit.SUMMARY_FILES.items()}
    saved = json.loads((tmp_path / "limits.json").read_text())
    assert saved == {"cl": 0.9, "best_cut": 3.9, "best_limit": 1.2890625, "missing": [],
                     "rows": [{"cut": 2.0, "limit": 1.7, "limitErr": 0.0, "best": False},
                              {"cut": 3.9, "limit": 1.2890625, "limitErr": 0.0, "best": True},
                              {"cut": 4.5, "limit": 1.5, "limitErr": 0.0, "best": False}]}


def test_summary_text_table(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9, 4.5])
    write_limit(tmp_path, 2.0, 1.7)
    write_limit(tmp_path, 3.9, 1.2890625)
    limit.summarize(cfg, tmp_path)
    lines = (tmp_path / "limits.txt").read_text().splitlines()
    assert lines[0] == "# median expected limit at CL 0.9; * = best cut"
    assert lines[1].split() == ["cut", "limit"]
    assert lines[2].split() == ["2.0", "1.7"]
    assert lines[3].split() == ["*", "3.9", "1.2890625"]
    assert lines[4].split() == ["4.5", "missing"]


def test_summary_missing_cut_warns(load, user, tmp_path, caplog):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9])
    write_limit(tmp_path, 3.9, 1.3)
    (tmp_path / "cut_2.0").mkdir()                       # job ran, Combine failed, no limit.json
    s = limit.summarize(cfg, tmp_path)
    assert s["missing"] == [2.0] and s["best_cut"] == 3.9
    assert s["rows"][0] == {"cut": 2.0, "limit": None, "limitErr": None, "best": False}
    assert "2.0" in caplog.text


def test_summary_nan_limit_is_missing(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9])
    write_limit(tmp_path, 2.0, math.nan)
    write_limit(tmp_path, 3.9, 1.3)
    s = limit.summarize(cfg, tmp_path)
    assert s["best_cut"] == 3.9 and s["missing"] == [2.0]


def test_summary_tie_takes_smaller_cut(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [4.5, 3.9])
    write_limit(tmp_path, 3.9, 1.3)
    write_limit(tmp_path, 4.5, 1.3)
    assert limit.summarize(cfg, tmp_path)["best_cut"] == 3.9


def test_summary_ignores_cuts_not_in_config(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    write_limit(tmp_path, 3.9, 1.3)
    write_limit(tmp_path, 5.0, 0.1)                      # old scan in the same limit/
    s = limit.summarize(cfg, tmp_path)
    assert [r["cut"] for r in s["rows"]] == [3.9] and s["best_cut"] == 3.9


def test_summary_no_limit_at_all_raises(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9])
    with pytest.raises(RuntimeError, match="2.0"):
        limit.summarize(cfg, tmp_path)
    assert not (tmp_path / "limits.json").exists()


def test_summary_broken_json_raises(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    (tmp_path / "cut_3.9").mkdir()
    (tmp_path / "cut_3.9" / "limit.json").write_text("{not json")
    with pytest.raises(ValueError, match="cut_3.9"):
        limit.summarize(cfg, tmp_path)


def test_summary_json_of_other_cut_raises(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    write_limit(tmp_path, 4.5, 1.3, folder_cut=3.9)
    with pytest.raises(ValueError, match="4.5"):
        limit.summarize(cfg, tmp_path)


def test_summary_plot(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9, 4.5])
    write_limit(tmp_path, 2.0, 1.7)
    write_limit(tmp_path, 3.9, 1.3)                      # 4.5 missing: drawn as a dotted line
    limit.summarize(cfg, tmp_path)
    png = (tmp_path / "limits.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000


def test_summary_plot_single_cut(load, user, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [3.9])                     # default config: one cut, still a valid plot
    write_limit(tmp_path, 3.9, 1.2890625)
    limit.summarize(cfg, tmp_path)
    assert (tmp_path / "limits.png").read_bytes()[:4] == b"\x89PNG"


def test_rerun_removes_old_limit_json(load, user, toy_inputs, monkeypatch, tmp_path):
    """A failed rerun must not leave the limit of the earlier run in the table."""
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    old = write_limit(tmp_path / "out", 3.9, 1.0)
    monkeypatch.setattr(limit, "combine_executable", lambda: Path("/bin/true"))
    monkeypatch.setattr(limit, "run_combine", lambda d, cl: Path(d) / "x.root")
    monkeypatch.setattr(limit, "read_limit", lambda p: {})
    with pytest.raises(RuntimeError):
        limit.run(cfg, {"signal": toy_inputs[0], "background": toy_inputs[1]}, tmp_path / "out", 3.9)
    assert not old.exists()


def fake_run_factory(failing):
    calls = []

    def fake_run(cfg, inputs, out_dir, cut):
        calls.append(cut)
        if cut in failing:
            raise subprocess.CalledProcessError(1, ["combine"])
        write_limit(Path(out_dir), cut, 1.0 + abs(cut - 3.9))
    return fake_run, calls


def test_run_scan_all_cuts_then_summary(load, user, monkeypatch, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9, 4.5])
    fake_run, calls = fake_run_factory(failing={4.5})
    monkeypatch.setattr(limit, "check_inputs", lambda inputs, tree: None)
    monkeypatch.setattr(limit, "combine_executable", lambda: Path("/bin/true"))
    monkeypatch.setattr(limit, "run", fake_run)

    s = limit.run_scan(cfg, {"signal": "s.root", "background": "b.root"}, tmp_path)

    assert calls == [2.0, 3.9, 4.5]                      # a failed cut does not stop the scan
    assert s["best_cut"] == 3.9 and s["missing"] == [4.5]
    assert (tmp_path / "limits.json").is_file()


def test_run_scan_no_combine_fails_before_any_cut(load, user, monkeypatch, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    def missing():
        raise FileNotFoundError("CMS Combine not found")

    fake_run, calls = fake_run_factory(failing=set())
    monkeypatch.setattr(limit, "check_inputs", lambda inputs, tree: None)
    monkeypatch.setattr(limit, "combine_executable", missing)
    monkeypatch.setattr(limit, "run", fake_run)
    with pytest.raises(FileNotFoundError, match="Combine"):
        limit.run_scan(scan_cfg(load, user, [2.0, 3.9]), {"signal": "s", "background": "b"}, tmp_path / "out")
    assert calls == [] and not (tmp_path / "out").exists()


def test_run_scan_other_errors_stop_the_scan(load, user, monkeypatch, tmp_path):
    """Only a Combine failure is a 'missing' cut; a bug (e.g. ValueError) must not be hidden."""
    from tau3mu_gnn.pipelines.common.stages import limit

    def broken(cfg, inputs, out_dir, cut):
        raise ValueError("bug")

    monkeypatch.setattr(limit, "check_inputs", lambda inputs, tree: None)
    monkeypatch.setattr(limit, "combine_executable", lambda: Path("/bin/true"))
    monkeypatch.setattr(limit, "run", broken)
    with pytest.raises(ValueError, match="bug"):
        limit.run_scan(scan_cfg(load, user, [3.9]), {"signal": "s", "background": "b"}, tmp_path)


@needs_combine
def test_run_scan_end_to_end(load, user, toy_inputs, tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit

    user["limit"]["signal_weight"] = 0.5
    cfg = scan_cfg(load, user, [2.0, 3.9])
    s = limit.run_scan(cfg, {"signal": toy_inputs[0], "background": toy_inputs[1]}, tmp_path / "limit")
    for cut in (2.0, 3.9):
        per_cut = json.loads((tmp_path / "limit" / f"cut_{cut}" / "limit.json").read_text())
        row = next(r for r in s["rows"] if r["cut"] == cut)
        assert row["limit"] == per_cut["limit"]
    for name in limit.SUMMARY_FILES.values():
        assert (tmp_path / "limit" / name).is_file(), name


def test_rerun_with_bad_inputs_removes_old_limit_json(load, user, tmp_path):
    """A rerun that fails on its inputs (before any datacard work) must not leave the old limit either."""
    from tau3mu_gnn.pipelines.common.branches import BranchError
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    old = write_limit(tmp_path / "out", 3.9, 1.0)
    with pytest.raises(BranchError, match="gone.root"):
        limit.run(cfg, {"signal": tmp_path / "gone.root", "background": tmp_path / "gone.root"}, tmp_path / "out", 3.9)
    assert not old.exists()


def test_failed_summary_removes_old_summary_files(load, user, tmp_path):
    """No cut has a limit: the limits.* files of an earlier scan must not stay and look valid."""
    from tau3mu_gnn.pipelines.common.stages import limit

    for name in limit.SUMMARY_FILES.values():
        (tmp_path / name).write_text("old")
    with pytest.raises(RuntimeError):
        limit.summarize(scan_cfg(load, user, [3.9]), tmp_path)
    for name in limit.SUMMARY_FILES.values():
        assert not (tmp_path / name).exists(), name


def test_summary_write_dir(load, user, tmp_path):
    """Phase C job: cut_*/ on dCache (read via /pnfs), table written to scratch."""
    from tau3mu_gnn.pipelines.common.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    write_limit(tmp_path / "phase", 3.9, 1.3)
    scratch = tmp_path / "scratch"
    s = limit.summarize(cfg, tmp_path / "phase", write_dir=scratch)
    assert s["files"]["json"] == scratch / "limits.json" and (scratch / "limits.png").is_file()
    assert not (tmp_path / "phase" / "limits.json").exists()


def test_summarize_cuts_explicit(tmp_path):
    from tau3mu_gnn.pipelines.common.stages import limit
    write_limit(tmp_path, 0.98, 2.0)
    write_limit(tmp_path, 0.99, 1.5)
    s = limit.summarize_cuts([0.98, 0.99], 0.9, tmp_path)
    assert s["best_cut"] == 0.99 and s["cl"] == 0.9 and all(f.is_file() for f in s["files"].values())
    s = limit.summarize_cuts([0.98, 0.99, 0.995], 0.9, tmp_path)
    assert s["missing"] == [0.995]
