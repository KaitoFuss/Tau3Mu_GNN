import os
from pathlib import Path

import pytest

URL = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/user/x/run"


def test_local_path_maps_xrootd_to_pnfs():
    from tau3mu_gnn.pipelines.common import storage

    assert storage.is_remote(URL) and not storage.is_remote("/work/x")
    assert storage.local_path(URL) == Path("/pnfs/psi.ch/cms/user/x/run")
    assert storage.local_path("/work/x/y") == Path("/work/x/y")


@pytest.mark.parametrize("bad", ["root://host/pnfs/x", "root://host", "root://host//"])
def test_malformed_url_raises(bad):
    from tau3mu_gnn.pipelines.common import storage

    with pytest.raises(ValueError, match="root://"):
        storage.local_path(bad)


def test_join_keeps_url():
    from tau3mu_gnn.pipelines.common import storage

    assert storage.join(URL + "/", "bdt_vs1", "/1_skim/") == URL + "/bdt_vs1/1_skim"
    assert storage.join("/work/x", "a", "b.txt") == "/work/x/a/b.txt"


def test_local_put_copies_tree_and_overwrites(tmp_path):
    from tau3mu_gnn.pipelines.common import storage

    src = tmp_path / "scratch"
    (src / "1_skim").mkdir(parents=True)
    (src / "1_skim" / "signal.root").write_text("new")
    (src / "top.txt").write_text("t")
    dest = tmp_path / "out" / "bdt_vs1"
    (dest / "1_skim").mkdir(parents=True)
    (dest / "1_skim" / "signal.root").write_text("old")

    storage.put(src, dest)

    assert (dest / "1_skim" / "signal.root").read_text() == "new"
    assert (dest / "top.txt").read_text() == "t"
    assert storage.listdir(dest) == ["1_skim", "top.txt"]
    assert storage.listdir(tmp_path / "nothing") == []


def test_local_write_read_exists(tmp_path):
    from tau3mu_gnn.pipelines.common import storage

    f = tmp_path / "a" / "b.json"
    storage.write_text(f, "{}")
    assert storage.exists(f) and storage.read_text(f) == "{}"


def test_put_missing_source_raises(tmp_path):
    from tau3mu_gnn.pipelines.common import storage

    with pytest.raises(FileNotFoundError):
        storage.put(tmp_path / "missing", tmp_path / "dest")


@pytest.fixture
def fake_xrootd(tmp_path, monkeypatch):
    """xrdfs / xrdcp on PATH that only log their arguments (one line per call)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "xrd.log"
    for name in ("xrdfs", "xrdcp"):
        exe = bin_dir / name
        exe.write_text(f'#!/bin/sh\necho "{name} $*" >> {log}\n')
        exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def test_remote_put_uses_xrdfs_and_xrdcp(tmp_path, fake_xrootd):
    from tau3mu_gnn.pipelines.common import storage

    src = tmp_path / "scratch"
    (src / "2_bdt").mkdir(parents=True)
    (src / "2_bdt" / "a.root").write_text("a")
    (src / "2_bdt" / "b.pkl").write_text("b")

    storage.put(src, URL + "/bdt_vs1")

    calls = fake_xrootd.read_text().splitlines()
    host = "t3dcachedb03.psi.ch:1094"
    assert calls.count(f"xrdfs {host} mkdir -p /pnfs/psi.ch/cms/user/x/run/bdt_vs1/2_bdt") == 1
    assert f"xrdcp --force --nopbar {src / '2_bdt' / 'a.root'} {URL}/bdt_vs1/2_bdt/a.root" in calls
    assert f"xrdcp --force --nopbar {src / '2_bdt' / 'b.pkl'} {URL}/bdt_vs1/2_bdt/b.pkl" in calls


def test_remote_makedirs_and_write_text(fake_xrootd):
    from tau3mu_gnn.pipelines.common import storage

    storage.makedirs(URL + "/gnn_vs2")
    storage.write_text(URL + "/gnn_vs2/manifest.json", "{}")
    calls = fake_xrootd.read_text().splitlines()
    assert calls[0] == "xrdfs t3dcachedb03.psi.ch:1094 mkdir -p /pnfs/psi.ch/cms/user/x/run/gnn_vs2"
    assert calls[-1].startswith("xrdcp --force --nopbar ") and calls[-1].endswith(
        f" {URL}/gnn_vs2/manifest.json")


def _flaky(tmp_path, monkeypatch, name, n_fail):
    """name on PATH: the first n_fail calls exit 51 (xrdcp socket timeout), later calls succeed.
    Every call is logged. time.sleep is recorded instead of waiting. Returns (log, sleeps)."""
    from tau3mu_gnn.pipelines.common import storage

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log, count = tmp_path / f"{name}.log", tmp_path / f"{name}.count"
    count.write_text("0")
    (bin_dir / name).write_text(
        f'#!/bin/sh\necho "{name} $*" >> {log}\nn=$(cat {count}); echo $((n + 1)) > {count}\n'
        f'[ "$n" -lt {n_fail} ] && exit 51\nexit 0\n')
    (bin_dir / name).chmod(0o755)
    for other in ("xrdfs", "xrdcp"):
        if other != name and not (bin_dir / other).exists():
            (bin_dir / other).write_text("#!/bin/sh\nexit 0\n")
            (bin_dir / other).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    sleeps = []
    monkeypatch.setattr(storage, "_sleep", sleeps.append)
    return log, sleeps


def test_remote_copy_retries_after_a_timeout(tmp_path, monkeypatch):
    # raw W analysis skim (job 960697): one xrdcp socket timeout lost a 7.5 h job
    from tau3mu_gnn.pipelines.common import storage

    log, sleeps = _flaky(tmp_path, monkeypatch, "xrdcp", n_fail=2)
    src = tmp_path / "a.root"
    src.write_text("a")
    storage.put(src, URL + "/bdt_vs1/a.root")
    assert len(log.read_text().splitlines()) == 3
    assert sleeps == [60, 120]


def test_remote_command_failure_raises_after_all_attempts(tmp_path, monkeypatch):
    import subprocess

    from tau3mu_gnn.pipelines.common import storage

    log, sleeps = _flaky(tmp_path, monkeypatch, "xrdfs", n_fail=99)
    with pytest.raises(subprocess.CalledProcessError):
        storage.makedirs(URL + "/x")
    assert len(log.read_text().splitlines()) == storage.ATTEMPTS == 3
    assert sleeps == [60, 120]
