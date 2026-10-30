"""Run-folder storage: dCache (root://host[:port]//pnfs/...), EOS (root://eosuser.cern.ch//eos/...) or a local disk path.

A root:// URL is read through the mount of its path (/pnfs NFS on the T3, /eos FUSE on lxplus) and written with
xrdfs / xrdcp. dCache needs a valid VOMS proxy; EOS uses the Kerberos token of the job (needs_proxy).
Every xrdfs / xrdcp call is tried ATTEMPTS times (pause RETRY_WAIT_S, then twice that, ...): one storage
timeout must not lose a long job (raw W analysis skim, job 960697: xrdcp socket timeout after 7.5 h).
A local path uses plain file operations. Stages never call this; the job bodies (jobs.py) do.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

_URL_RE = re.compile(r"^root://([^/]+)/(/[^/].*)$")
ATTEMPTS = 3
RETRY_WAIT_S = 60   # pause after the first failure; doubles after each further failure

log = logging.getLogger(__name__)
_sleep = time.sleep   # tests replace this, not time.sleep of the whole process


def is_remote(path: str | Path) -> bool:
    return str(path).startswith("root://")


def needs_proxy(path: str | Path) -> bool:
    """True for a dCache URL. EOS (root://eos*.cern.ch) uses Kerberos; a local path needs nothing."""
    if not is_remote(path):
        return False
    host = _split(str(path))[0].split(":")[0]
    return not (host.startswith("eos") and host.endswith(".cern.ch"))


def _split(url: str) -> tuple[str, str]:
    m = _URL_RE.match(url)
    if not m:
        raise ValueError(f"not an xrootd URL of the form root://host[:port]//path: {url!r}")
    return m.group(1), m.group(2).rstrip("/") or "/"


def local_path(path: str | Path) -> Path:
    """Path to read from: the mounted path (/pnfs/..., /eos/...) of a root:// URL, else the path itself."""
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
    """Run an xrdfs / xrdcp command; on failure try again (ATTEMPTS in total). xrdcp --force overwrites
    a partly written target, so a repeat is safe."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            if attempt == ATTEMPTS:
                raise
            wait = RETRY_WAIT_S * 2 ** (attempt - 1)
            log.warning("%s failed (exit %s), attempt %d of %d; next try in %d s",
                        " ".join(cmd), e.returncode, attempt, ATTEMPTS, wait)
            _sleep(wait)
