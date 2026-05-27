# tau3mu GNN

A Graph Neural Network pipeline for searching the lepton-flavour-violating decay **τ → 3μ** with CMS data. The GNN is intended to outperform the existing BDT baseline, with the physics goal of setting an expected upper limit on the branching ratio B(τ → 3μ) using CMS Combine.

---

## Table of Contents

- [Infrastructure](#infrastructure)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
  - [Step 0 — Skimming](#step-0--skimming)
  - [Step 1 — Graph Creation (T3)](#step-1--graph-creation-t3)
  - [Step 2 — Copy Graphs to lxplus](#step-2--copy-graphs-to-lxplus)
  - [Step 3 — GNN Training (lxplus)](#step-3--gnn-training-lxplus)
  - [Step 4 — Copy Model back to T3](#step-4--copy-model-back-to-t3)
  - [Step 5 — GNN Inference (T3)](#step-5--gnn-inference-t3)
  - [Step 6 — Model Evaluation (T3)](#step-6--model-evaluation-t3)
  - [Step 7 — Limit Setting (T3)](#step-7--limit-setting-t3)
- [Key Concepts](#key-concepts)
  - [Channels](#channels)
  - [Graph Construction](#graph-construction)
  - [Unsculpting / Mass Decorrelation](#unsculpting--mass-decorrelation)
  - [GNN Architecture (TauMuNet)](#gnn-architecture-taumunet)
  - [Graph Feature Integration Modes](#graph-feature-integration-modes)
  - [DisCo Regularisation](#disco-regularisation)
  - [Fine-Tuning](#fine-tuning)
- [BDT Baseline](#bdt-baseline)
- [Data Storage and Transfer](#data-storage-and-transfer)
- [Job Submission](#job-submission)
- [Contact](#contact)

---

## Infrastructure

Work is split across two clusters depending on the task:

| Task | Cluster | Scheduler |
|---|---|---|
| Graph creation, GNN inference, limit-setting | **PSI T3** | SLURM (`sbatch`) |
| GNN training | **CERN lxplus** | HTCondor (`condor_submit`) |

> Paths in job scripts are hardcoded for user `kfuss`. Adjust all occurrences of `/work/kfuss/`, `/pnfs/psi.ch/cms/trivcat/store/user/kfuss/` and `/eos/user/k/kfuss/` when running under a different account.

---

## Repository Structure

```
tau3mu_gnn/
├── environment.yaml              # Conda environment spec
├── pyproject.toml                # Package definition (src layout)
├── jobs/                         # Cluster job scripts
│   ├── graph_creation/           # SLURM: skimming and graph creation
│   ├── gnn_training/             # SLURM: GNN training on T3 (for quick tests)
│   ├── condor/                   # HTCondor: GNN training on lxplus
│   ├── bdt/                      # SLURM: BDT training and score production
│   ├── gnn_score_creation/       # SLURM: GNN inference on T3
│   ├── model_eval/               # SLURM: ROC comparisons and evaluations
│   └── analysis/                 # SLURM: datacard creation and limit setting
└── src/tau3mu_gnn/
    ├── scripts/                  # Top-level entry points
    │   ├── GraphCreation.py      # Graph creation entry point
    │   ├── Train_TauMuNet.py     # GNN training entry point
    │   └── ScoreCreation.py      # GNN inference entry point
    ├── graph/
    │   ├── ConeGraphProducer.py          # Core graph building logic
    │   └── graphDataLoader.py            # Data loading / batching
    ├── architectures/
    │   ├── TauMuNet.py           # GNN model definition
    │   └── GNN_archs.py          # Pre-defined architecture configs
    ├── gnn_training/
    │   ├── TrainingUtils.py      # Normalisation, early stopping, helpers
    │   ├── EvaluationUtils.py    # ROC/AUC evaluation, plotting
    │   ├── NormalizerUtils.py    # On-the-fly feature normaliser
    │   └── ScoreProducer.py      # Inference on ROOT files
    ├── bdt_training/
    │   ├── Train_XGB_W.py        # Train XGBoost BDT (W channel)
    │   ├── Train_XGB_HF.py       # Train XGBoost BDT (HF channel)
    │   └── BDT_ScoreProducer.py  # Score production for BDT
    ├── model_eval/
    │   └── ModelEval.py          # Plotting: ROC, sculpting, score distributions
    ├── analysis/
    │   ├── datacard_creation_GNN_permuted.py      # Generate CMS Combine datacards (GNN)
    │   ├── datacard_creation_ISO.py      # Generate CMS Combine datacards (ISO/BDT baseline)
    │   └── readout_limit.py              # Parse Combine output and save limits to JSON
    └── utils/
        ├── skimmer.py            # Pre-selection skimmer
        └── ...
```

---

## Installation

All steps use the `Tau3Mu` conda environment. Install it with:

```bash
conda env create -f environment.yaml
conda activate Tau3Mu

# Additional packages not in the yaml:
conda install pyg=2.5.2 -c pyg
conda install root
pip install comet_ml
pip install seaborn
pip install xgboost
pip install scikit-learn

# Install this package in editable mode (from repo root):
pip install -e .
```

> Use `mamba` as a drop-in replacement for `conda` for faster dependency resolution (if you want).

The package follows a `src` layout — all modules live under `src/tau3mu_gnn/` and are imported as `tau3mu_gnn.*`. After `pip install -e .`, all scripts can be run with `python -m tau3mu_gnn.scripts.<script_name>`.

---

## Pipeline Overview

The full pipeline proceeds in the following order. Each step has a corresponding job script under `jobs/`.

### Step 0 — Skimming (OPTIONAL)

Before graph creation, raw ROOT files can optionally be pre-selected (fiducial cuts, resonance vetoes, etc.) using the skimmer. Most of these steps were done by supervisor:

```bash
# jobs/graph_creation/skimming.sh
python -m tau3mu_gnn.utils.skimmer \
  --input  <path/to/raw.root> \
  --output <path/to/skimmed.root>
```

### Step 1 — Graph Creation (T3)

Reads skimmed ROOT files, applies selection cuts, and builds PyTorch Geometric graph objects (`.pt` files). Separate graphs are created for signal and background, and for each channel.

```bash
# SLURM: jobs/graph_creation/GraphCreation_W.sh
# SLURM: jobs/graph_creation/GraphCreation_HF.sh

python -m tau3mu_gnn.scripts.GraphCreation \
  --graph_type   knn          # knn | r_threshold | fully
  --k            10           # number of nearest neighbours (knn only)
  --tau_channel  W            # W | HF
  --unsculpting_mode <mode>   # see §Key Concepts
  [--bdt_preselection_cut]    # save bdt benchmark score to graph for furhter use downstream (root file needs to have been inferenced in order to do so)
```

The output dataset name is constructed automatically from the parameters and written into `data/signal_<channel>/processed/` and `data/background_<channel>/processed/`.

**Node features** include: particle kinematics (pT, η, φ), displacement (dz, d0), PUPPI weights, and charge.  
**Edge features** include: relative angles (ΔR, Δη, Δφ) and pairwise pT.  
**Graph-level features** include: τ candidate kinematics, MET, vertex quality, displacement significance (previously BDT input features).

### Step 2 — Copy Graphs to lxplus

Transfer the processed `.pt` graph files from T3 to lxplus for training.

> All `xrdcp` calls require a valid CMS VOMS proxy: `voms-proxy-init --voms cms --valid 192:00`

```bash
# Run on T3
xrdcp -r root://t3se01.psi.ch//pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/processed \
    /eos/user/k/kfuss/tau3mu_gnn_round2/data/background_W/
# Repeat for signal and for HF channel as needed
```

### Step 3 — GNN Training (lxplus)

Training runs on lxplus GPU nodes via HTCondor. The training entry point is `Train_TauMuNet.py`.

```bash
# HTCondor: jobs/condor/train_W_GINE.sub  (submit file)
#           jobs/condor/train_W_GINE_deleted.sh  (executable)

python -m tau3mu_gnn.scripts.Train_TauMuNet \
  --epochs        1000 \
  --batch         128 \
  --name          <run_name> \
  --model         <graph_identifier>   \  # must match the .pt filename suffix from Step 1
  --lr            0.0016 \
  --disco         <float>  \             # DisCo regularisation strength (0 = off)
  --arch          <arch_name> \          # architecture config from GNN_archs.py
  --tau_channel   W \                    # W | HF
  --gnn_mode      <mode> \               # see §Key Concepts
  --conv_type     <type> \               # GINE | GAT | Transformer
  --unsculpting_mode <mode> \
  --data_base     /eos/user/k/kfuss/tau3mu_gnn_round2/data \
  [--finetune] \
  [--finetune_lr <float>] \
  [--finetune_epochs <int>] \
  [--bdt_threshold <float>]
```

Training uses:
- **Loss**: Binary cross-entropy (+ optional DisCo term on background)
- **Optimiser**: Adam with `ReduceLROnPlateau` scheduler
- **Early stopping**: enabled by default; saves the best checkpoint
- **Experiment tracking**: CometML (project logged automatically)
- **Outputs** saved in `output/<run_name>/`: model checkpoint (`.pt`), normalisation parameters (`Normalization.pkl`), loss curves, ROC plots, sculpting check plots, architecture config (`arch_config.json`)

At the end of training, the model is automatically evaluated on the held-out test set (events in the 70–100% window) and plots are saved.

### Step 4 — Copy Model back to T3

```bash
# On lxplus — push trained model to T3 storage
xrdcp -r /eos/user/k/kfuss/SVJ_GNN/output/<MODEL_NAME>/ \
    root://t3se01.psi.ch//pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/gnn_from_lxplus/

# On T3 — move into working directory
mv /pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/gnn_from_lxplus/<MODEL_NAME> \
    /work/kfuss/SVJ_GNN/output/
```

### Step 5 — GNN Inference (T3)

Runs a trained model on ROOT files and writes a new ROOT file containing the GNN score for each event.

```bash
# SLURM: jobs/gnn_score_creation/create_score_W_bkg.sh
#        jobs/gnn_score_creation/create_score_W_signal.sh

python -m tau3mu_gnn.scripts.ScoreCreation \
  --root           <path/to/root/files> \
  --output_folder  /work/kfuss/SVJ_GNN/output/ \
  --model_names    <model_name1> [<model_name2> ...] \
  --archs          <arch1> [<arch2> ...] \
  --conv_types     <type1> [<type2> ...] \
  --tau_channel    W \
  --gnn_mode       <mode> \
  --storage_folder <scratch_dir> \
  --unsculpting_mode <mode> \
  [--bdt_preselection_cut <float>] \
  [--max_events_to_process <int>]
```

The output ROOT files (containing the `GNN_score` branch) are written to `--storage_folder`. After score production, `ModelEval` is called automatically within the job script to produce tau mass histograms at fixed FPR, which are used to check for mass sculpting.

### Step 6 — Model Evaluation (T3) (OPTIONAL)

ROC curve comparisons across multiple models (and against the BDT baseline) can be produced with:

```bash
# jobs/model_eval/ROC_curves_comparison.sh

python -m tau3mu_gnn.model_eval.ModelEval \
  --plot_mode       ROC_comp_full \
  --tau_channel     W \
  --gnn_archs_location  <path/to/model1/> [<path/to/model2/> ...] \
  --lambdas         <label1> [<label2> ...] \
  --archs           <arch1> [<arch2> ...] \
  --bdt_location    <path/to/bdt_model/>
```

Other `--plot_mode` options: `tau_hist`, `tau_hist_fpr`, `score_cut`.

### Step 7 — Limit Setting (T3)

Limit setting requires a valid CMS VOMS proxy and CMS Combine to be available.

**7a — Generate datacard and run Combine:**

```bash
# jobs/analysis/create_datacard.sh
python -m tau3mu_gnn.analysis.datacard_creation_GNN \
  --model   <model_name> \
  --channel W \
  --cut     <score_cut>

# Run Combine (must be in the datacard directory)
combine -M FitDiagnostics datacard.txt
combine -C 0.90 --run blind -M AsymptoticLimits datacard.txt
```

The signal is modelled as a Gaussian at m(τ) = 1.78 GeV, background as a falling exponential. The analysis is run blinded (signal mass window excluded from the fit).

**7b — Parse and save the limit:**

```bash
python -m tau3mu_gnn.analysis.readout_limit \
  --model   <model_name> \
  --channel W \
  --cut     <score_cut> \
  --path    <path/to/combine/output/>
```

The median expected limit is extracted and appended to `/work/kfuss/SVJ_GNN/output/<channel>_limits.json`, sorted by model performance.

---

## Key Concepts

### Channels

Two τ production channels are supported throughout the pipeline:

- **W channel** (`--tau_channel W`): τ from W → τν decay
- **HF channel** (`--tau_channel HF`): τ from heavy-flavour decays

Graph-level features, signal weights, and some file naming conventions differ between channels. The channel must be specified consistently at every pipeline step.

### Graph Construction

Each event is represented as a graph over particle-flow (PF) candidates within a cone of ΔR < 0.5 around the τ candidate.

**Edge connection modes** (`--graph_type`):

| Mode | Description |
|---|---|
| `knn` | Each node connects to its k nearest neighbours in (η, φ) space |
| `r_threshold` | All pairs within ΔR < δ are connected |
| `fully` | All nodes connected to all others |

### Unsculpting / Mass Decorrelation

A key challenge is preventing the GNN from sculpting the τ invariant mass distribution, which would bias the background model used in the limit fit. Several strategies are implemented via `--unsculpting_mode`:

| Mode | Description |
|---|---|
| `delete_muons` | Removes the 3 signal muons from PF candidates before graph construction |
| `permutation` | Permutes muon-to-PF-candidate assignments |
| `split_pt_uniformly` | Replaces muon pT with a fixed value drawn from a uniform distribution |
| *(None)* | No unsculpting applied |

DisCo regularisation (see below) can also be used independently or in combination with these modes.

### GNN Architecture (TauMuNet)

The model (`TauMuNet`) is a message-passing GNN with the following structure:

1. Configurable number of graph convolution layers (the convolution type is selectable)
2. Feature fusion: concatenated outputs from all convolution layers are projected into a shared representation
3. Optional scale-down fully-connected layers
4. Global mean pooling over nodes
5. Graph-level feature injection (position depends on `gnn_mode`)
6. Final fully-connected classification head (with sigmoid output, logits are preferred for stability)

**Convolution types** (`--conv_type`):

| Type | Description |
|---|---|
| `GINE` | GIN with edge features (Graph Isomorphism Network with Edges) |
| `GAT` | Graph Attention Network |
| `Transformer` | Transformer-based graph convolution |

Architecture hyperparameters (number of layers, hidden dimensions, dropout) are defined in `GNN_archs.py` and selected via `--arch`. A `random` option draws a configuration from a predefined random grid.

### Graph Feature Integration Modes

Graph-level features (τ kinematics, MET, vertex variables) can be injected into the GNN at different points, controlled by `--gnn_mode`:

| Mode | Description |
|---|---|
| `BDT` | No graph-level features injected into the GNN; used as input to an XGBoost BDT post-classifier |
| `embed_once` | Graph-level features concatenated to node features at the input layer only |
| `embed_always` | Graph-level features concatenated to node features at every convolution layer |
| `final_layer` | Graph-level features concatenated after global pooling, before the classification head |

### DisCo Regularisation

A distance-correlation (DisCo) penalty term between the GNN output score and the reconstructed τ mass is added to the training loss on background events:

```
L = BCE + λ · DisCo(score, m_τ)
```

The regularisation strength is set via `--disco` (λ = 0 disables it). A linear warm-up over the first 20 epochs is applied. The DisCo penalty is only evaluated on background events; signal events are masked out.

### Fine-Tuning

After the main training phase, a second fine-tuning phase can be run on a subset of events selected by a BDT ISO score threshold:

```bash
--finetune
--finetune_lr      <float>   # learning rate for fine-tuning
--finetune_epochs  <int>     # number of fine-tuning epochs
--bdt_threshold    <float>   # minimum BDT ISO score for event inclusion
```

The pre-fine-tune checkpoint is saved separately as `<name>_pretrained.pt`. Fine-tuning uses a higher early-stopping patience to allow adaptation on the smaller dataset.

---

## BDT Baseline

An XGBoost BDT serves as the reference baseline classifier. There is legacy code that trains two BDTs: one taking a `GNN_Score` as input and one using `rel_iso`. This was the initial approach before moving to an end-to-end GNN. It can still be trained and used to produce scores independently:

```bash
# Train (W channel)
# jobs/bdt/BDT_pipeline_W.sh
python -m tau3mu_gnn.bdt_training.Train_XGB_W --model <model_name>

# Score production
python -m tau3mu_gnn.bdt_training.BDT_ScoreProducer \
  --model          <model_name> \
  --classifier_gnn <path/to/GNN_classifiers.pck> \
  --classifier_iso <path/to/ISO_classifiers.pck> \
  --tau_channel    W \
  --storage_folder <scratch_dir>
```

The BDT ISO score can also be saved as a graph-level feature during graph creation (`--bdt_preselection_cut`), for event pre-selection during fine-tuning.

---

## Data Storage and Transfer

Processed data lives on the T3 CMS storage system and is accessed via `xrootd`. All jobs that read from or write to `/pnfs/` require a valid VOMS proxy:

```bash
voms-proxy-init --voms cms --valid 192:00
```

Data paths on T3 storage:
```
/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/
├── data/
│   ├── signal_W/          # initial ROOT file, signal graphs and scored ROOT files
│   ├── background_W/
│   ├── signal_HF/
│   └── background_HF/
├── gnn_from_lxplus/       # staging area for models copied from lxplus
└── datacard_tests/        # Combine datacards and output
```

Combine outputs and limit results are written to `/work/kfuss/SVJ_GNN/output/`.

---

## Job Submission

### T3 (SLURM)

Standard jobs (CPU):
```bash
sbatch jobs/graph_creation/GraphCreation_W.sh
sbatch jobs/gnn_score_creation/create_score_W_bkg.sh
sbatch jobs/analysis/create_datacard.sh
```

GPU training jobs on T3 (for quick tests):
```bash
sbatch jobs/gnn_training/Train_W_final_layer.sh
```

Log files are written to `/work/kfuss/SVJ_GNN/logs/`.

### lxplus (HTCondor)

For full-scale training, submit via HTCondor on lxplus:
```bash
condor_submit jobs/condor/train_W_GINE.sub
```

The `.sub` file points to an executable `.sh` script (stored on AFS). Logs go to `/afs/cern.ch/user/k/kfuss/condor_logs/`.

HTCondor jobs request a GPU with compute capability ≥ 7.0 and ≥ 30 GB VRAM.

---

## Contact

kaito.fuss@gmail.com
