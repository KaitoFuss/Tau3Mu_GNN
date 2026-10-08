import glob
import argparse
from typing import Mapping, Sequence

from tau3mu_gnn.pipeline.branches import rename_columns


def skim(files: Sequence[str], output: str, tree_name: str, branch_map: Mapping[str, str],
         dz_max: float, cos_alpha_2d_min: float, n_threads: int = 0, max_events: int | None = None) -> None:
    """Select tau candidates with |dZmu12|, |dZmu13|, |dZmu23| < dz_max and cos alpha > cos_alpha_2d_min.

    Every Slimmed_TauTo3Mu* branch gets a masked copy Selected_TauTo3Mu*. No event is removed.
    branch_map (analysis name -> user name) is applied first; the user's names are not written.
    n_threads: 0 = all cores. max_events (for tests) runs single-threaded, as RDataFrame.Range needs that.
    """
    import ROOT

    if max_events is None:
        ROOT.EnableImplicitMT(n_threads)
    else:
        ROOT.DisableImplicitMT()

    # Create RDataFrame from the list of NanoAOD files
    df = ROOT.RDataFrame(tree_name, list(files))
    if max_events is not None:
        df = df.Range(max_events)
    df, columns = rename_columns(df, branch_map)

    df = df.Define(
        "tau3mu_mask",
        f"abs(Slimmed_TauTo3Mu_dZmu12) < {dz_max!r} && "
        f"abs(Slimmed_TauTo3Mu_dZmu13) < {dz_max!r} && "
        f"abs(Slimmed_TauTo3Mu_dZmu23) < {dz_max!r} && "
        f"Slimmed_TauTo3Mu_CosAlpha2D_LxyP3mu > {cos_alpha_2d_min!r}"
    )
    columns.append("tau3mu_mask")

    # Automatically find branches
    tau3mu_branches = [c for c in columns if c.startswith("Slimmed_TauTo3Mu")]
    for br in tau3mu_branches:
        new_br = br.replace("Slimmed", "Selected")
        df = df.Define(new_br, f"{br}[tau3mu_mask]")
        columns.append(new_br)

    df = df.Define("nSelectedTauTo3Mu", "Selected_TauTo3Mu_dZmu12.size()")
    columns.append("nSelectedTauTo3Mu")

    df.Snapshot(tree_name, output, columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    files = glob.glob(args.input)
    print(f"Found {len(files)} input files: {files}")
    skim(files, args.output, "Events", {}, dz_max=0.5, cos_alpha_2d_min=0.5)
