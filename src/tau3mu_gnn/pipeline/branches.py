"""Required raw input branches and the user's branch map.

Names here are the analysis names in the raw input files (before stage 1).
The user config's branch_map maps analysis name -> name in the user's files.
"""
from __future__ import annotations

import difflib
import glob
from typing import Mapping, Sequence

import awkward as ak
import numpy as np
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


def _expand(patterns: Sequence[str], problems: list[str]) -> list[str]:
    files = []
    for pattern in patterns:
        if any(c in pattern for c in "*?["):
            matches = sorted(glob.glob(pattern))
            if not matches:
                problems.append(f"input pattern matches no file: {pattern}")
            files.extend(matches)
        else:
            files.append(pattern)
    return list(dict.fromkeys(files))  # overlapping patterns must not process a file twice


def check_input_branches(patterns: Sequence[str], tree_name: str,
                         branch_map: Mapping[str, str], channel: str) -> list[str]:
    """Check that every input file has every required branch after the branch map.

    Reads branch names only (no events). Returns the expanded file list.
    Raises BranchError listing every problem of every file.
    """
    required = required_branches(channel)
    problems = _check_map(branch_map, required)
    files = _expand(patterns, problems)
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
    if problems:
        raise BranchError(problems)
    return files


def rename_columns(df, branch_map: Mapping[str, str]):
    """Define each analysis name from the user's name on an RDataFrame.

    Returns (df, columns): columns are all columns of the new df except the
    user's original names that were renamed, ready for df.Snapshot(tree, file, columns).
    """
    renames = {analysis: user for analysis, user in branch_map.items() if analysis != user}
    existing = {str(c) for c in df.GetColumnNames()}
    for analysis in sorted(renames):
        # The user's file may also hold the analysis name; the mapped branch wins.
        if analysis in existing:
            df = df.Redefine(analysis, renames[analysis])
        else:
            df = df.Define(analysis, renames[analysis])
    originals = set(renames.values())
    columns = [str(c) for c in df.GetColumnNames() if str(c) not in originals]
    return df, columns


# Branches with one entry per tau candidate / PF candidate / TauPlusMET object.
# Lengths are compared inside a family only: TauPlusMET_* has one entry per
# unskimmed candidate and can differ from Slimmed_TauTo3Mu_* (seen in Kaito's signal).
FAMILIES: tuple[str, ...] = ("Slimmed_TauTo3Mu_", "Slimmed_PFCands_", "TauPlusMET_")


def check_input_structure(files: Sequence[str], tree_name: str, branch_map: Mapping[str, str],
                          channel: str, n_events: int = 1000) -> None:
    """Check per-event list lengths agree within each branch family (first n_events per file).

    Expects files that already passed check_input_branches. Raises BranchError listing every problem.
    """
    required = required_branches(channel)
    problems = []
    for path in files:
        tree = uproot.open(path)[tree_name]
        for family in FAMILIES:
            members = [n for n in required if n.startswith(family)]
            if not members:
                continue
            user_names = [branch_map.get(n, n) for n in members]
            arrays = tree.arrays(user_names, entry_stop=n_events, library="ak")
            reference = None
            for analysis, user in zip(members, user_names):
                label = analysis if user == analysis else f"{user} (analysis name {analysis})"
                try:
                    counts = np.asarray(ak.num(arrays[user], axis=1))
                except Exception:
                    problems.append(f"{path}: {label} is not a per-event list ({family}* branches must be)")
                    continue
                if reference is None:
                    reference = (label, counts)
                    continue
                bad = np.nonzero(counts != reference[1])[0]
                if bad.size:
                    i = int(bad[0])
                    problems.append(f"{path}: {family}* branches differ in length: {reference[0]} has "
                                    f"{reference[1][i]} but {label} has {counts[i]} (first at entry {i})")
    if problems:
        raise BranchError(problems)
