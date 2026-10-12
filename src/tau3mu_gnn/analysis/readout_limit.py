"""Read the median expected limit from the Combine AsymptoticLimits output.

read_limit() is the function (used by pipeline stage 5); main() is the old CLI (updates <channel>_limits.json).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import ROOT

from tau3mu_gnn.analysis.combine import ASYMPTOTIC_OUTPUT

OLD_OUTPUT_ROOT = "/work/kfuss/SVJ_GNN/output"


def read_limit(combine_root: str | Path) -> dict:
    """{"limit", "limitErr"} of the median expected limit (quantile 0.5), or {} if the file has none."""
    file = ROOT.TFile(str(combine_root))
    tree = file.Get("limit")

    limits = {}
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        if round(tree.quantileExpected, 3) == 0.5:
            limits = {"limit": tree.limit, "limitErr": tree.limitErr}
    file.Close()
    return limits


def main(argv: Sequence[str] | None = None) -> None:
    """Old CLI: add the limit of <path>/higgsCombineTest.AsymptoticLimits.mH120.root to <channel>_limits.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--cut", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--fine_tune", default="")
    args = parser.parse_args(argv)

    MODEL = args.model
    CHANNEL = args.channel
    CUT = args.cut
    FINE_TUNE = args.fine_tune

    if FINE_TUNE:
        MODEL = f"{MODEL}_{FINE_TUNE}"

    limits = read_limit(f"{args.path}/{ASYMPTOTIC_OUTPUT}")  # string join keeps root:// URLs intact

    json_path = f"{OLD_OUTPUT_ROOT}/{CHANNEL}_limits.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            result = json.load(f)
    else:
        result = {}

    # Update nested structure
    if CHANNEL not in result:
        result[CHANNEL] = {}
    if MODEL not in result[CHANNEL]:
        result[CHANNEL][MODEL] = {}
    result[CHANNEL][MODEL][CUT] = limits

    for ch in result:
        for model in result[ch]:
            result[ch][model] = dict(sorted(result[ch][model].items(), key=lambda x: float(x[0])))

        def model_sort_key(model_name):
            is_benchmark = 0 if "benchmark" in model_name.lower() else 1
            best_limit = min(
                (v.get("limit", float("inf")) for v in result[ch][model_name].values()),
                default=float("inf")
            )
            return (is_benchmark, best_limit)

        result[ch] = dict(sorted(result[ch].items(), key=lambda x: model_sort_key(x[0])))

    with open(json_path, "w") as f:
        json.dump(result, f, indent=3)

    print(json.dumps({CHANNEL: {MODEL: {CUT: limits}}}, indent=3))


if __name__ == "__main__":
    main()
