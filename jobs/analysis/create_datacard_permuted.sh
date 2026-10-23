#!/bin/bash
# 
#SBATCH -p standard
#SBATCH --account=t3
#SBATCH --job-name=datacard
#SBATCH --mem=128G
#SBATCH -o /work/kfuss/SVJ_GNN/logs/%x-%j.out    # replace default slurm-SLURM_JOB_ID.out; %x is a job-name (or script name when there is no job-name)
#SBATCH -e /work/kfuss/SVJ_GNN/logs/%x-%j.err    # replace default slurm-SLURM_JOB_ID.err 

set -e

voms-proxy-info --exists --valid 12:00
if [ $? -ne 0 ]; then
    echo "ERROR: No valid proxy found. Run voms-proxy-init --voms cms --valid 192:00 before submitting."
    exit 1
fi

source /work/kfuss/miniforge3/bin/activate Tau3Mu
#############################################
# Config
#############################################

CHANNEL=W
CONV_TYPE=GINE
UNSCULPTING_MODE=delete_muons
MODEL="new_hope_${CHANNEL}_fl_8_sd_4_GINE_delete_muons_vs2"
# MODEL="pt_ft_${CHANNEL}_fl_8_sd_4_${CONV_TYPE}_${UNSCULPTING_MODE}_small_ft_bdt_cut_0.7_vs1"
# MODEL="new_hope_W_fl_8_sd_4_GINE_delete_muons_vs1"
CUT=3.9
BDT_PRESELECTION_CUT="0.8"  # set to "" to disable
FINE_TUNE="" # "", with_bdt_cut, without_bdt_cut

if [ -n "$BDT_PRESELECTION_CUT" ]; then
    FILE_SUFFIX="_with_scores_bdt_full_with_scores"
    if [ -n "$FINE_TUNE" ]; then
      OUTPUT=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/datacard_tests/${MODEL}/with_bdt_cut/${CUT}
    else
      OUTPUT=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/datacard_tests/${MODEL}/${CUT}
    fi
else
    FILE_SUFFIX="_with_scores"
    if [ -n "$FINE_TUNE" ]; then
      OUTPUT=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/datacard_tests/${MODEL}/without_bdt_cut/${CUT}
    else
      OUTPUT=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/datacard_tests/${MODEL}/${CUT}
    fi
fi

JOB_SCRATCH=/scratch/$USER/$SLURM_JOB_ID
mkdir -p "$JOB_SCRATCH"
export TMPDIR="$JOB_SCRATCH"

echo "Using scratch: $JOB_SCRATCH"

#############################################
# Move to working directory
#############################################

cd $JOB_SCRATCH

echo "Running datacard test in: $JOB_SCRATCH"

#############################################
# Run datacard script
#############################################

python -m tau3mu_gnn.analysis.datacard \
  --model "$MODEL" \
  --channel "$CHANNEL" \
  --cut "$CUT" \
  --file_suffix "$FILE_SUFFIX"

combine -M FitDiagnostics datacard.txt 
combine -C 0.90 --run blind -M AsymptoticLimits datacard.txt

#############################################
# Collect outputs
#############################################

echo "Copying outputs to storage..."

echo "Removing existing $MODEL datacard from storage if present..."
xrdfs t3dcachedb03.psi.ch:1094 rm -r ${OUTPUT} 2>/dev/null || true
xrdfs t3dcachedb03.psi.ch:1094 mkdir -p ${OUTPUT} 2>/dev/null || true
echo "Copying $MODEL datacard to storage..."
for file in $JOB_SCRATCH/*; do
    xrdcp --force "$file" root://t3dcachedb03.psi.ch:1094/${OUTPUT}/$(basename $file)
done

#############################################
# Save Limit in JSON
#############################################

python -m tau3mu_gnn.analysis.readout_limit \
  --model "$MODEL" \
  --channel "$CHANNEL" \
  --cut "$CUT" \
  --path "$OUTPUT" \
  --fine_tune "$FINE_TUNE"

#############################################
# Cleanup
#############################################

echo "Cleaning scratch..."
rm -rf "$JOB_SCRATCH"