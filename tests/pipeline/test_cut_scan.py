import json
import math

import pytest


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
    from tau3mu_gnn.pipeline.stages import limit

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
    from tau3mu_gnn.pipeline.stages import limit

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
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9])
    write_limit(tmp_path, 3.9, 1.3)
    (tmp_path / "cut_2.0").mkdir()                       # job ran, Combine failed, no limit.json
    s = limit.summarize(cfg, tmp_path)
    assert s["missing"] == [2.0] and s["best_cut"] == 3.9
    assert s["rows"][0] == {"cut": 2.0, "limit": None, "limitErr": None, "best": False}
    assert "2.0" in caplog.text


def test_summary_nan_limit_is_missing(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9])
    write_limit(tmp_path, 2.0, math.nan)
    write_limit(tmp_path, 3.9, 1.3)
    s = limit.summarize(cfg, tmp_path)
    assert s["best_cut"] == 3.9 and s["missing"] == [2.0]


def test_summary_tie_takes_smaller_cut(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [4.5, 3.9])
    write_limit(tmp_path, 3.9, 1.3)
    write_limit(tmp_path, 4.5, 1.3)
    assert limit.summarize(cfg, tmp_path)["best_cut"] == 3.9


def test_summary_ignores_cuts_not_in_config(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    write_limit(tmp_path, 3.9, 1.3)
    write_limit(tmp_path, 5.0, 0.1)                      # old scan in the same phase_c/
    s = limit.summarize(cfg, tmp_path)
    assert [r["cut"] for r in s["rows"]] == [3.9] and s["best_cut"] == 3.9


def test_summary_no_limit_at_all_raises(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9])
    with pytest.raises(RuntimeError, match="2.0"):
        limit.summarize(cfg, tmp_path)
    assert not (tmp_path / "limits.json").exists()


def test_summary_broken_json_raises(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    (tmp_path / "cut_3.9").mkdir()
    (tmp_path / "cut_3.9" / "limit.json").write_text("{not json")
    with pytest.raises(ValueError, match="cut_3.9"):
        limit.summarize(cfg, tmp_path)


def test_summary_json_of_other_cut_raises(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [3.9])
    write_limit(tmp_path, 4.5, 1.3, folder_cut=3.9)
    with pytest.raises(ValueError, match="4.5"):
        limit.summarize(cfg, tmp_path)


def test_summary_plot(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [2.0, 3.9, 4.5])
    write_limit(tmp_path, 2.0, 1.7)
    write_limit(tmp_path, 3.9, 1.3)                      # 4.5 missing: drawn as a dotted line
    limit.summarize(cfg, tmp_path)
    png = (tmp_path / "limits.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000


def test_summary_plot_single_cut(load, user, tmp_path):
    from tau3mu_gnn.pipeline.stages import limit

    cfg = scan_cfg(load, user, [3.9])                     # default config: one cut, still a valid plot
    write_limit(tmp_path, 3.9, 1.2890625)
    limit.summarize(cfg, tmp_path)
    assert (tmp_path / "limits.png").read_bytes()[:4] == b"\x89PNG"
