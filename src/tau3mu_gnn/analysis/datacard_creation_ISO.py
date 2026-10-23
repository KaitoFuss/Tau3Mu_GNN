"""BDT-only (ISO) datacard: make_datacard of datacard_creation_GNN_permuted.py with score_branch="bdt_ISO".

The old script was a copy of the GNN datacard script with GNN_score replaced by bdt_ISO. main() keeps its
CLI (--model --channel --cut) and its hard-coded input files; outputs go into the current directory, and
mass.pdf is copied to <OLD_OUTPUT_ROOT>/<model>/mass_<cut>.pdf as the old script saved it.
"""
from __future__ import annotations

import argparse
import shutil
from typing import Sequence

from tau3mu_gnn.analysis.datacard_creation_GNN_permuted import (OLD_DATA_ROOT, OLD_OUTPUT_ROOT,
                                                                OLD_SIGNAL_WEIGHT, make_datacard)


def main(argv: Sequence[str] | None = None) -> None:
    """Old CLI: hard-coded input files per model and channel, outputs into the current directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--cut", required=True)
    args = parser.parse_args(argv)

    model, channel = args.model, args.channel
    # The old script read the background_W/<model>/ folder for both channels.
    base = f"{OLD_DATA_ROOT}/background_W/{model}"
    if channel == "W":
        sig = [f"{base}/selected_signal_skimmed_no_resonances_with_scores_bdt_full.root"]
        bkg = [f"{base}/selected_data_skimmed_no_resonances_with_scores_bdt_full.root"]
    elif channel == "HF":
        sig = [f"{base}/selected_ds_large_signal_skimmed_no_resonances_with_scores_with_scores_bdt.root"]
        bkg = [f"{base}/selected_data_skimmed_no_resonances_part{i}_with_scores_bdt.root" for i in range(1, 5)]
    else:
        parser.error(f"unknown channel {channel!r} (W or HF)")

    make_datacard(sig, bkg, float(args.cut), OLD_SIGNAL_WEIGHT[channel], ".", score_branch="bdt_ISO")
    target = f"{OLD_OUTPUT_ROOT}/{model}/mass_{args.cut}.pdf"   # old name: raw --cut string, no suffix
    try:
        shutil.copyfile("mass.pdf", target)
    except OSError as e:  # the old ROOT SaveAs only printed an error; the job goes on
        print(f"Warning: could not copy mass.pdf to {target}: {e}")


if __name__ == "__main__":
    main()
