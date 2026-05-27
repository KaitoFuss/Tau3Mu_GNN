#!/bin/bash
# 
#SBATCH -p standard
#SBATCH --account=t3
#SBATCH --job-name=bkg_fit
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
# MODEL="new_dropout_disco2_permuted_${CHANNEL}_fl_12_sd_15_vs1"
MODEL="new_dropout_disco10_${CHANNEL}_fl_8_sd_4_vs1"
CUT=4.5
MAX_POLY_ORDER=13


OUTPUT=/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/datacard_tests/${MODEL}/bkg_fit_${CUT}

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

python -m tau3mu_gnn.analysis.bkg_fit_params \
  --model "$MODEL" \
  --channel "$CHANNEL" \
  --cut "$CUT" \
  --max_poly_order "$MAX_POLY_ORDER"

#############################################
# Collect outputs
#############################################

echo "Copying outputs to storage..."

echo "Removing existing $MODEL datacard from storage if present..."
xrdfs t3dcachedb03.psi.ch:1094 rm -r ${JOB_SCRATCH}${MODEL} 2>/dev/null || true
echo "Copying $MODEL datacard to storage..."
xrdcp -r --force $JOB_SCRATCH root://t3dcachedb03.psi.ch:1094/${OUTPUT}

#############################################
# Cleanup
#############################################

echo "Cleaning scratch..."
rm -rf "$JOB_SCRATCH"