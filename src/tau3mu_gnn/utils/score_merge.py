"""Write a score back to its event, matched by (run, luminosityBlock, event).

Shared by the BDT (stage 2) and GNN (stage 3) score producers. (run, event) alone is not unique:
in MC the run is always 1, and events from different lumi blocks can have the same event number.
"""
from __future__ import annotations

import os
from array import array
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import uproot

KEYS = ["run", "luminosityBlock", "event"]


class DuplicateEventError(ValueError):
    """Two entries have the same (run, luminosityBlock, event)."""


def check_unique(keys: pd.DataFrame, what: str) -> None:
    dup = keys.duplicated(KEYS, keep=False)
    if dup.any():
        examples = keys.loc[dup, KEYS].drop_duplicates().head(5).to_dict("records")
        raise DuplicateEventError(f"{what}: {int(dup.sum())} entries share (run, luminosityBlock, event) "
                                  f"with another entry, e.g. {examples}")


def check_unique_keys(path: str | Path, tree_name: str) -> None:
    """Raise DuplicateEventError if two entries of path:tree_name have the same key. Reads the key branches only."""
    with uproot.open(path) as f:
        keys = f[tree_name].arrays(KEYS, library="pd")
    check_unique(keys, str(path))


def scores_in_entry_order(entry_keys: pd.DataFrame, score_keys: pd.DataFrame, scores,
                          default: float = -1.0) -> np.ndarray:
    """Score of each row of entry_keys, in the same order; default where there is no score."""
    check_unique(entry_keys, "input tree")
    check_unique(score_keys, "scores")
    left = entry_keys[KEYS].astype("uint64").reset_index(drop=True)
    right = score_keys[KEYS].astype("uint64").reset_index(drop=True)
    right["_score"] = np.asarray(scores, dtype=np.float64)
    merged = left.merge(right, on=KEYS, how="left", sort=False, indicator=True, validate="one_to_one")
    return np.where(merged["_merge"] == "both", merged["_score"], default)


def write_with_score(in_file, out_file, tree_name: str, branch: str, values: np.ndarray,
                     drop: Sequence[str] = ()) -> None:
    """Copy every entry and branch of in_file:tree_name (except drop) to out_file, plus branch/F = values."""
    import ROOT

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    fin = ROOT.TFile.Open(str(in_file))
    t1 = fin.Get(tree_name)
    n = t1.GetEntries()
    if n != len(values):
        fin.Close()
        raise ValueError(f"{in_file} has {n} entries, but {len(values)} values for {branch}")
    dropped = [name for name in drop if t1.GetBranch(name)]
    for name in dropped:
        t1.SetBranchStatus(name, 0)

    fout = ROOT.TFile.Open(str(out_file), "RECREATE")
    t_out = t1.CloneTree(0)  # empty tree
    for name in dropped:
        t1.SetBranchStatus(name, 1)
    buf = array("f", [0.])
    t_out.Branch(branch, buf, f"{branch}/F")
    for i in range(n):
        t1.GetEntry(i)
        buf[0] = values[i]
        t_out.Fill()
    t_out.Write()
    fout.Close()
    fin.Close()
