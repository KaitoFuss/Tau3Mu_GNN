import glob
import argparse
from typing import Sequence

from tau3mu_gnn.pipelines.analysis.config import load_config


def skim(files: Sequence[str], output: str, tree_name: str, dz_max: float, cos_alpha_2d_min: float) -> None:
    """Select tau candidates with |dZmu12|, |dZmu13|, |dZmu23| < dz_max and cos alpha > cos_alpha_2d_min.

    Every Slimmed_TauTo3Mu* branch gets a masked copy Selected_TauTo3Mu*. No event is removed.
    Expects analysis branch names (the branch map is applied in stage 0).
    """
    import ROOT

    # Enable implicit multi-threading for better performance
    ROOT.EnableImplicitMT()

    # Create RDataFrame from the list of NanoAOD files
    df = ROOT.RDataFrame(tree_name, list(files))

    df = df.Define(
        "tau3mu_mask",
        f"abs(Slimmed_TauTo3Mu_dZmu12) < {dz_max!r} && "
        f"abs(Slimmed_TauTo3Mu_dZmu13) < {dz_max!r} && "
        f"abs(Slimmed_TauTo3Mu_dZmu23) < {dz_max!r} && "
        f"Slimmed_TauTo3Mu_CosAlpha2D_LxyP3mu > {cos_alpha_2d_min!r}"
    )

    # Automatically find branches
    tau3mu_branches = [str(b) for b in df.GetColumnNames() if str(b).startswith("Slimmed_TauTo3Mu")]
    for br in tau3mu_branches:
        new_br = br.replace("Slimmed", "Selected")
        df = df.Define(new_br, f"{br}[tau3mu_mask]")

    df = df.Define("nSelectedTauTo3Mu", "Selected_TauTo3Mu_dZmu12.size()")

    df.Snapshot(tree_name, output)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="pipeline config; gives the skim cuts and tree name (default: cuts 0.5, tree Events)")
    args = parser.parse_args(argv)

    tree_name, dz_max, cos_alpha_2d_min = "Events", 0.5, 0.5
    if args.config:
        cfg = load_config(args.config)
        tree_name, dz_max, cos_alpha_2d_min = cfg.inputs.tree_name, cfg.skim.dz_max, cfg.skim.cos_alpha_2d_min

    files = sorted(glob.glob(args.input))
    print(f"Found {len(files)} input files: {files}")
    skim(files, args.output, tree_name, dz_max=dz_max, cos_alpha_2d_min=cos_alpha_2d_min)


if __name__ == "__main__":
    main()
