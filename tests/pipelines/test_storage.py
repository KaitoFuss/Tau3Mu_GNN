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

    assert storage.join(URL + "/", "phase_a_vs1", "/1_skim/") == URL + "/phase_a_vs1/1_skim"
    assert storage.join("/work/x", "a", "b.txt") == "/work/x/a/b.txt"


def test_local_put_copies_tree_and_overwrites(tmp_path):
    from tau3mu_gnn.pipelines.common import storage

    src = tmp_path / "scratch"
    (src / "1_skim").mkdir(parents=True)
    (src / "1_skim" / "signal.root").write_text("new")
    (src / "top.txt").write_text("t")
    dest = tmp_path / "out" / "phase_a_vs1"
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

    storage.put(src, URL + "/phase_a_vs1")

    calls = fake_xrootd.read_text().splitlines()
    host = "t3dcachedb03.psi.ch:1094"
    assert calls.count(f"xrdfs {host} mkdir -p /pnfs/psi.ch/cms/user/x/run/phase_a_vs1/2_bdt") == 1
    assert f"xrdcp --force --nopbar {src / '2_bdt' / 'a.root'} {URL}/phase_a_vs1/2_bdt/a.root" in calls
    assert f"xrdcp --force --nopbar {src / '2_bdt' / 'b.pkl'} {URL}/phase_a_vs1/2_bdt/b.pkl" in calls


def test_remote_makedirs_and_write_text(fake_xrootd):
    from tau3mu_gnn.pipelines.common import storage

    storage.makedirs(URL + "/phase_b_vs2")
    storage.write_text(URL + "/phase_b_vs2/manifest.json", "{}")
    calls = fake_xrootd.read_text().splitlines()
    assert calls[0] == "xrdfs t3dcachedb03.psi.ch:1094 mkdir -p /pnfs/psi.ch/cms/user/x/run/phase_b_vs2"
    assert calls[-1].startswith("xrdcp --force --nopbar ") and calls[-1].endswith(
        f" {URL}/phase_b_vs2/manifest.json")


def test_remote_command_failure_raises(tmp_path, monkeypatch):
    import subprocess

    from tau3mu_gnn.pipelines.common import storage

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "xrdfs").write_text("#!/bin/sh\nexit 52\n")
    (bin_dir / "xrdfs").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(subprocess.CalledProcessError):
        storage.makedirs(URL + "/x")
