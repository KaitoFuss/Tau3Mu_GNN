"""Run-folder storage: dCache (root://host[:port]//pnfs/...) or a local disk path.

dCache is read through its /pnfs NFS mount and written with xrdfs / xrdcp (needs a valid VOMS proxy).
A local path uses plain file operations. Stages never call this; the job bodies (jobs.py) do.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_URL_RE = re.compile(r"^root://([^/]+)/(/[^/].*)$")


def is_remote(path: str | Path) -> bool:
    return str(path).startswith("root://")


def _split(url: str) -> tuple[str, str]:
    m = _URL_RE.match(url)
    if not m:
        raise ValueError(f"not an xrootd URL of the form root://host[:port]//path: {url!r}")
    return m.group(1), m.group(2).rstrip("/") or "/"


def local_path(path: str | Path) -> Path:
    """Path to read from: the /pnfs path of a root:// URL, else the path itself."""
    path = str(path)
    return Path(_split(path)[1]) if is_remote(path) else Path(path)


def join(base: str | Path, *parts: str | Path) -> str:
    """String join, so root://host//... URLs stay intact."""
    pieces = [str(base).rstrip("/")] + [str(p).strip("/") for p in parts if str(p).strip("/")]
    return "/".join(pieces)


def makedirs(path: str | Path) -> None:
    path = str(path)
    if is_remote(path):
        host, p = _split(path)
        _run(["xrdfs", host, "mkdir", "-p", p])
    else:
        Path(path).mkdir(parents=True, exist_ok=True)


def put(src: str | Path, dest: str | Path) -> None:
    """Copy a local file (to the file path dest) or a folder tree (into the folder dest). Overwrites."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"nothing to copy: {src}")
    if src.is_file():
        pairs = [(src, str(dest))]
    else:
        pairs = [(f, join(dest, f.relative_to(src).as_posix())) for f in sorted(src.rglob("*")) if f.is_file()]
    if is_remote(dest):
        host = _split(str(dest))[0]
        for folder in sorted({str(Path(_split(t)[1]).parent) for _, t in pairs}):
            _run(["xrdfs", host, "mkdir", "-p", folder])
        for f, target in pairs:
            _run(["xrdcp", "--force", "--nopbar", str(f), target])
    else:
        for f, target in pairs:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)


def write_text(dest: str | Path, text: str) -> None:
    if is_remote(dest):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / Path(_split(str(dest))[1]).name
            f.write_text(text)
            put(f, dest)
    else:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text(text)


def read_text(path: str | Path) -> str:
    return local_path(path).read_text()


def exists(path: str | Path) -> bool:
    return local_path(path).exists()


def listdir(path: str | Path) -> list[str]:
    p = local_path(path)
    return sorted(x.name for x in p.iterdir()) if p.is_dir() else []


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)
