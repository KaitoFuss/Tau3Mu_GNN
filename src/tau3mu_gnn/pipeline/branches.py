"""Required raw input branches and the user's branch map.

Names here are the analysis names in the raw input files (before stage 1).
The user config's branch_map maps analysis name -> name in the user's files.
"""
from __future__ import annotations

import difflib
import glob
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import uproot

_TAU = "Slimmed_TauTo3Mu_"
_PF = "Slimmed_PFCands_"

# Read by every channel.
COMMON: tuple[str, ...] = (
    # stage 1 skim cut
    *(f"{_TAU}{s}" for s in ("dZmu12", "dZmu13", "dZmu23", "CosAlpha2D_LxyP3mu")),
    # tau candidate: choice, cone centre, mass fit
    *(f"{_TAU}fitted_{s}" for s in ("pt", "eta", "phi", "mass")),
    # muon directions: GNN cone and delete_muons
    *(f"{_TAU}mu{i}_{s}" for i in (1, 2, 3) for s in ("eta", "phi")),
    # GNN node features
    *(f"{_PF}{s}" for s in ("pt", "eta", "phi", "mass", "pdgId", "charge", "puppiWeight", "d0", "dz")),
    # event id: merging scores back into the trees
    "run", "luminosityBlock", "event",
)

# BDT ISO features and GNN graph-level features differ per channel.
PER_CHANNEL: dict[str, tuple[str, ...]] = {
    "W": (
        *(f"{_TAU}{s}" for s in ("sigLxy_3muVtxBS", "vtx_chi2", "vtx_Ndof",
                                 "iso_ptChargedFromPV", "iso_ptPhotons", "iso_ptChargedFromPU")),
        *(f"TauPlusMET_{s}" for s in ("Tau_Puppi_mT", "PuppiMET_pt", "Puppi_pt", "Puppi_phi",
                                      "PuppiMETminPz", "PuppiMETmaxPz")),
    ),
}


class BranchError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("Input branch problems:\n" + "\n".join(f"  - {p}" for p in self.problems))


def required_branches(channel: str) -> tuple[str, ...]:
    if channel not in PER_CHANNEL:
        raise BranchError([f"no required-branch list for channel {channel!r} (known: {sorted(PER_CHANNEL)})"])
    return COMMON + PER_CHANNEL[channel]


def _check_map(branch_map: Mapping[str, str], required: tuple[str, ...]) -> list[str]:
    problems = []
    required_set = set(required)
    seen_targets: dict[str, str] = {}
    for key in sorted(branch_map):
        target = branch_map[key]
        if key not in required_set:
            hint = difflib.get_close_matches(key, required, n=1)
            problems.append(f"branch_map: unknown analysis branch '{key}'"
                            + (f" (did you mean '{hint[0]}'?)" if hint else ""))
        if target in seen_targets:
            problems.append(f"branch_map: '{seen_targets[target]}' and '{key}' both map to '{target}'")
        else:
            seen_targets[target] = key
        if target != key and target in required_set:
            problems.append(f"branch_map: '{key}' maps to '{target}', which is itself a required analysis branch name")
    return problems


def _expand(inputs: Sequence[str], problems: list[str]) -> list[str]:
    files = []
    for entry in inputs:
        if any(c in entry for c in "*?["):
            matches = sorted(glob.glob(entry))
            if not matches:
                problems.append(f"input pattern matches no file: {entry}")
            files.extend(matches)
        else:
            files.append(entry)
    return list(dict.fromkeys(files))  # overlapping globs must not process a file twice


def check_input_branches(inputs: Sequence[str], tree_name: str,
                         branch_map: Mapping[str, str], channel: str) -> list[str]:
    """Check that every input file has every required branch after the branch map.

    inputs: config input entries (inputs.signal / inputs.background); each is a path or a glob.
    A file that holds both a mapped user name and its analysis name is an error.
    Reads branch names only (no events). Returns the expanded file list.
    Raises BranchError listing every problem of every file.
    """
    required = required_branches(channel)
    problems = _check_map(branch_map, required)
    files = _expand(inputs, problems)
    for path in files:
        try:
            with uproot.open(path) as f:
                keys = set(f[tree_name].keys())
        except FileNotFoundError:
            problems.append(f"{path}: file not found")
            continue
        except KeyError:
            problems.append(f"{path}: no tree '{tree_name}'")
            continue
        except Exception as e:  # corrupt file, not ROOT, xrootd error, ...
            problems.append(f"{path}: cannot open ({type(e).__name__}: {e})")
            continue
        missing = [name for name in required if branch_map.get(name, name) not in keys]
        if missing:
            lines = [f"      {branch_map[n]} (analysis name {n})" if branch_map.get(n, n) != n else f"      {n}"
                     for n in missing]
            problems.append(f"{path}: {len(missing)} required branch(es) missing:\n" + "\n".join(lines))
        clashes = sorted(n for n, user in branch_map.items() if user != n and n in keys)
        if clashes:
            lines = [f"      {n} (mapped from {branch_map[n]})" for n in clashes]
            problems.append(f"{path}: branch_map renames onto name(s) the file already has:\n" + "\n".join(lines))
    if problems:
        raise BranchError(problems)
    return files


def rename_columns(df, branch_map: Mapping[str, str]):
    """Define each analysis name from the user's name on an RDataFrame.

    Returns (df, columns): columns are all columns of the new df except the
    user's original names that were renamed, ready for df.Snapshot(tree, file, columns).
    """
    renames = {analysis: user for analysis, user in branch_map.items() if analysis != user}
    for analysis in sorted(renames):
        # Define fails if the column exists; check_input_branches reports that case first.
        df = df.Define(analysis, renames[analysis])
    originals = set(renames.values())
    columns = [str(c) for c in df.GetColumnNames() if str(c) not in originals]
    return df, columns


def check_stage_inputs(inputs: Mapping[str, str | Path], tree_name: str, required: Iterable[str],
                       label: str, forbidden: Iterable[str] = ()) -> None:
    """Raise BranchError listing, for every file in inputs (sample -> file), each missing required
    branch and each forbidden branch that is present. Reads branch names only."""
    required = sorted(required)
    problems = []
    for path in inputs.values():
        path = str(path)
        try:
            with uproot.open(path) as f:
                keys = set(f[tree_name].keys())
        except FileNotFoundError:
            problems.append(f"{path}: file not found")
            continue
        except KeyError:
            problems.append(f"{path}: no tree '{tree_name}'")
            continue
        missing = [name for name in required if name not in keys]
        if missing:
            problems.append(f"{path}: missing {label} input branches: {', '.join(missing)}")
        present = [name for name in sorted(forbidden) if name in keys]
        if present:
            problems.append(f"{path}: has branches that {label} writes: {', '.join(present)}")
    if problems:
        raise BranchError(problems)
