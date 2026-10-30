#!/usr/bin/env python3
"""Step 14: list the T3 files that move to EOS. Run on T3 (needs the /pnfs mount), not on lxplus.

    python3 jobs/migration/t3_list_files.py                 # sizes per group, writes filelist.tsv
    python3 jobs/migration/t3_list_files.py --out my.tsv W_signal_raw HF_signal_old

filelist.tsv: one line per file, <group> <bytes> <T3 path> <EOS path>, tab separated.
Copy it to lxplus and give it to copy_to_eos.sh.

Path rule (Kaito, 2026-10-30), with Y = the path after /pnfs/psi.ch/cms/trivcat/store/user/:
- raw NanoAOD (manzoni/...)  -> /eos/cms/store/group/phys_bphys/kfuss/tau3mu_inputs/Y  (B-physics group space:
  copied once, never deleted; the e-group may write but not delete)
- kfuss/tau3mu_pipeline/X    -> /eos/user/k/kfuss/tau3mu_pipeline/X
- anything else              -> /eos/user/k/kfuss/tau3mu_inputs/Y
So a config glob changes only in its prefix.
"""
import argparse
import glob
import os
import sys

T3_USER = "/pnfs/psi.ch/cms/trivcat/store/user/"
PIPELINE = T3_USER + "kfuss/tau3mu_pipeline/"
EOS_HOME = "/eos/user/k/kfuss/"
EOS_GROUP = "/eos/cms/store/group/phys_bphys/kfuss/"
RAW = T3_USER + "manzoni/tau3mu_gnn/"
OLD = T3_USER + "kfuss/tau3mu_gnn_round2/data/"

# globs from configs/example_*.yaml; results: the T3 reference runs (roadmap step 14 handoff)
GROUPS = {
    "W_signal_raw": [RAW + "WtoTauNu-Tauto3Mu_TuneCP5_13p6TeV_pythia8/05may25v2__WtoTauNu-Tauto3Mu_TuneCP5_13p6TeV_"
                           "pythia8_RunIII2024Summer24MiniAOD-140X_mcRun3_2024_realistic_v26-v2_MINIAODSIM/"
                           "250506_093818/*/*.root"],
    "W_background_raw": [RAW + "ParkingDoubleMuonLowMass*/_ParkingDoubleMuonLowMass*_Run2024*/*/*/*.root"],
    "HF_signal_old": [OLD + "signal_HF/old_data/ds_large_signal_skimmed_no_resonances.root"],
    "HF_background_old": [OLD + "background_HF/old_data/data_skimmed_no_resonances_part*.root"],
    "results_W": [PIPELINE + "training/W_raw_Nano_AOD_fixed/**", PIPELINE + "analysis/W_raw_Nano_AOD_fixed/**"],
    "results_HF": [PIPELINE + "training/HF_training_fixed/**", PIPELINE + "analysis/HF_analysis_fixed/**"],
}


def eos_path(t3: str) -> str:
    if t3.startswith(PIPELINE):
        return EOS_HOME + "tau3mu_pipeline/" + t3[len(PIPELINE):]
    if t3.startswith(RAW):
        return EOS_GROUP + "tau3mu_inputs/" + t3[len(T3_USER):]
    return EOS_HOME + "tau3mu_inputs/" + t3[len(T3_USER):]


def gb(n: int) -> str:
    return f"{n / 1e9:10.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="*", help=f"default: all ({', '.join(GROUPS)})")
    ap.add_argument("--out", default="filelist.tsv")
    a = ap.parse_args()
    unknown = [g for g in a.groups if g not in GROUPS]
    if unknown:
        sys.exit(f"unknown group(s) {unknown}; known: {list(GROUPS)}")
    total = 0
    with open(a.out, "w") as out:
        for g in a.groups or GROUPS:
            files = sorted({f for pat in GROUPS[g] for f in glob.glob(pat, recursive=True) if os.path.isfile(f)})
            if not files:
                print(f"WARNING {g}: no files match {GROUPS[g]}", file=sys.stderr)
            size = 0
            for f in files:
                n = os.path.getsize(f)
                size += n
                out.write(f"{g}\t{n}\t{f}\t{eos_path(f)}\n")
            total += size
            print(f"{g:20s} {len(files):6d} files {gb(size)}")
    print(f"{'total':20s} {'':6s}       {gb(total)}\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
