"""Run CMS Combine on a datacard folder: the two commands of jobs/analysis/create_datacard_permuted.sh."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ASYMPTOTIC_OUTPUT = "higgsCombineTest.AsymptoticLimits.mH120.root"


def combine_executable() -> Path:
    """`combine` on PATH, else next to the running python (env not activated). FileNotFoundError if neither."""
    found = shutil.which("combine")
    if found:
        return Path(found)
    beside = Path(sys.executable).parent / "combine"
    if beside.is_file():
        return beside
    raise FileNotFoundError(f"CMS Combine not found: no 'combine' on PATH and no {beside}")


def run_combine(card_dir: str | Path, cl: float) -> Path:
    """Run FitDiagnostics and the blind AsymptoticLimits (CL = cl) on card_dir/datacard.txt, in card_dir.

    Output of each command goes to card_dir/combine_{fitdiagnostics,asymptotic}.log.
    Raise CalledProcessError if a command fails. Returns card_dir / ASYMPTOTIC_OUTPUT.
    """
    card_dir = Path(card_dir)
    exe = combine_executable()
    env = {**os.environ, "PATH": f"{exe.parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    for log_name, args in (("combine_fitdiagnostics.log", ["-M", "FitDiagnostics", "datacard.txt"]),
                           ("combine_asymptotic.log", ["-C", f"{cl}", "--run", "blind", "-M", "AsymptoticLimits",
                                                       "datacard.txt"])):
        with open(card_dir / log_name, "w") as log:
            subprocess.run([str(exe), *args], cwd=card_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
    return card_dir / ASYMPTOTIC_OUTPUT
