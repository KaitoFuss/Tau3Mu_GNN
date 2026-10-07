import ROOT
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

# Enable implicit multi-threading for better performance
ROOT.EnableImplicitMT()

# Find ROOT files
# Make sure these files exist in the specified path
# files = glob.glob('/pnfs/psi.ch/cms/trivcat/store/user/manzoni/tau3mu_gnn/WtoTauNu-Tauto3Mu_TuneCP5_13p6TeV_pythia8/05may25v2__WtoTauNu-Tauto3Mu_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24MiniAOD-140X_mcRun3_2024_realistic_v26-v2_MINIAODSIM/250506_093818/0000/*.root')
# files = glob.glob('/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data_skimmed_no_resonances_part*.root')
# files = glob.glob('/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/data_skimmed_no_resonances.root')
file_name = args.input
files = glob.glob(file_name)

print(f"Found {len(files)} input files: {files}")

# Define the output file name
output_file = args.output

# Create RDataFrame from the list of NanoAOD files
df = ROOT.RDataFrame("Events", files)

df = df.Define(
    "tau3mu_mask",
    "abs(Slimmed_TauTo3Mu_dZmu12) < 0.5 && "
    "abs(Slimmed_TauTo3Mu_dZmu13) < 0.5 && "
    "abs(Slimmed_TauTo3Mu_dZmu23) < 0.5 && "
    "Slimmed_TauTo3Mu_CosAlpha2D_LxyP3mu > 0.5"
)

# Automatically find branches
all_branches = df.GetColumnNames()
tau3mu_branches = [str(b) for b in all_branches if str(b).startswith("Slimmed_TauTo3Mu")]
for br in tau3mu_branches:
    new_br = br.replace("Slimmed","Selected")
    df = df.Define(new_br, f"{br}[tau3mu_mask]")    

df = df.Define("nSelectedTauTo3Mu", "Selected_TauTo3Mu_dZmu12.size()")




# # --- Step 1: Define the boolean mask for the TauTo3Mu vtx_chi2 cut ---
# df = df.Define("dz12_mask", "abs(Slimmed_TauTo3Mu_dZmu12)<0.5")
# df = df.Define("dz13_mask", "abs(Slimmed_TauTo3Mu_dZmu13)<0.5")
# df = df.Define("dz23_mask", "abs(Slimmed_TauTo3Mu_dZmu23)<0.5")
# 
# # --- Step 2: Filter events based on the TauTo3Mu mask (Skimming) ---
# # Keep events where at least one TauTo3Mu candidate passed the chi2 cut.
# df = df.Filter("Slimmed_TauTo3Mu_dZmu12[(dz12_mask)].size() > 0", "null")
# df = df.Filter("Slimmed_TauTo3Mu_dZmu13[(dz13_mask)].size() > 0", "null")
# df = df.Filter("Slimmed_TauTo3Mu_dZmu23[(dz23_mask)].size() > 0", "null")
# 
# df = df.Define("cos2d_mask", "Slimmed_TauTo3Mu_CosAlpha2D_LxyP3mu>0.5")
# df = df.Filter("Slimmed_TauTo3Mu_CosAlpha2D_LxyP3mu[(cos2d_mask)].size() > 0", "null")


df.Snapshot("Events", output_file)













