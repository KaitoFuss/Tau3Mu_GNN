# Tau3Mu GNN

A Graph Neural Network (GNN) pipeline that searches for the decay **τ → 3μ** in CMS data.

- This decay violates lepton flavour.
- The GNN must be better than the current BDT baseline.
- The physics goal: an expected upper limit on the branching ratio B(τ → 3μ), set with CMS Combine.

---

## Table of Contents

- [Infrastructure](#infrastructure)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Analysis Pipeline (tau3mu-analysis)](#analysis-pipeline-tau3mu-analysis)
- [Training Pipeline (tau3mu-training)](#training-pipeline-tau3mu-training)
- [HF Channel](#hf-channel)
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

The work uses two clusters:

| Task | Cluster | Scheduler |
|---|---|---|
| Graph creation, GNN inference, limit-setting | **PSI T3** | SLURM (`sbatch`) |
| GNN training | **CERN lxplus** | HTCondor (`condor_submit`) |

> **Other account?** Some paths in the job scripts and the Python scripts are hardcoded for the user `kfuss`. Change all of these paths:
> - `/work/kfuss/`
> - `/pnfs/psi.ch/cms/trivcat/store/user/kfuss/`
> - `/eos/user/k/kfuss/`

> **This will change: move to lxplus.** A later step moves all work to lxplus (HTCondor and EOS). Then the split between T3 and lxplus stops. These README parts describe the T3 setup now and will change:
> - [Infrastructure](#infrastructure): the clusters, the schedulers and the hardcoded paths.
> - [CMS Combine](#cms-combine): the build on lxplus.
> - [Analysis Pipeline](#analysis-pipeline-tau3mu-analysis): SLURM, `sbatch`, `squeue`, `sacct`, the dCache `root://` output, the VOMS proxy, `paths.log_dir` on `/work`, the `slurm.jobs` resources.
> - [Data Storage and Transfer](#data-storage-and-transfer): `/pnfs/`, `/work/`, the data flow.
> - [Job Submission](#job-submission): SLURM on T3.

---

## Repository Structure

```
tau3mu_gnn/
├── environment.yaml              # Conda environment spec (unpinned)
├── environment_frozen.yaml       # Pinned conda environment (reproducible)
├── requirements_frozen.txt       # Pinned pip packages (reproducible)
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
    │   ├── EvaluationUtils.py    # ROC/AUC evaluation of test dataset
    │   ├── NormalizerUtils.py    # On-the-fly feature normaliser
    │   └── ScoreProducer.py      # Inference on ROOT files
    ├── bdt_training/
    │   ├── Train_XGB_W.py        # Train XGBoost BDT (W channel)
    │   ├── Train_XGB_HF.py       # Train XGBoost BDT (HF channel)
    │   └── BDT_ScoreProducer.py  # Score production for BDT
    ├── model_eval/
    │   └── ModelEval.py          # Plotting: ROC, sculpting, score distributions
    ├── pipelines/                # tau3mu-analysis and tau3mu-training
    │   ├── common/               # shared: config parts, storage, SLURM, manifests, shared stages
    │   ├── analysis/             # analysis pipeline (phases bdt, gnn, limit)
    │   └── training/             # training pipeline (phases bdt, gnn, benchmark)
    ├── analysis/
    │   ├── datacard.py                            # CMS Combine datacards (GNN score or BDT ISO)
    │   ├── datacard_creation_ISO.py      # Generate CMS Combine datacards (ISO/BDT baseline)
    │   └── readout_limit.py              # Parse Combine output and save limits to JSON
    └── utils/
        ├── skimmer.py            # Pre-selection skimmer
        └── ...
```

---

## Installation

All steps use the conda env `Tau3Mu`. Make it and activate it:

```bash
conda env create -f environment_frozen.yaml
conda activate Tau3Mu
pip install -r requirements_frozen.txt
pip install -e .
```

> **Faster:** you can use `mamba` instead of `conda`. It finds the package versions faster.

How the package works:

- It has a `src` layout. All modules are in `src/tau3mu_gnn/`.
- You import them as `tau3mu_gnn.*`.
- After `pip install -e .`, run a script with `python -m tau3mu_gnn.scripts.<script_name>`.

### CMS Combine

The limit step needs CMS Combine **in the `Tau3Mu` env** (`<env>/bin/combine`).

- Combine is not a conda package of the env. Build it one time from source.
- Use the [standalone conda recipe](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/#standalone-compilation-with-conda-cmake-based) of Combine.
- Do not make a new env. Tau3Mu already has all build dependencies: ROOT 6.34, gsl, boost-cpp, vdt, eigen, tbb, cmake, ninja.
- Run the commands in the repo folder. Git ignores `HiggsAnalysis/`.

```bash
conda activate Tau3Mu
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git checkout 10cef0bb      # v10.5.1 + 6 commits: the version of the reference limit
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
      -DCMAKE_INSTALL_PYTHONDIR=lib/python3.11/site-packages -DUSE_VDT=OFF
cmake --build build -j8
cmake --install build
which combine              # must be $CONDA_PREFIX/bin/combine
```

> **Do not** use the CMSSW recipe (`cmsrel`, `scram`). The jobs do not run `cmsenv`. They look for `combine` next to the env Python.
>
> **Do not** use a different Combine version. The limit can change.

---

## Analysis Pipeline (tau3mu-analysis)

This pipeline runs the frozen W analysis on **your** input files. You need only one config file.

- Stages: resonance skim (raw NanoAOD only) → skim → BDT → GNN → evaluation → limit.
- The models are fixed. The pipeline does not train.
- All jobs run on the PSI T3 with SLURM.

### 1. Set up (once)

1. **Install the env and the package.** Follow [Installation](#installation).
   - `pip install -e .` also installs the command `tau3mu-analysis`.
   - Did you install the package before this command existed, or before the rename (2026-10-26: `tau3mu-pipeline` → `tau3mu-analysis`, `tau3mu-train` → `tau3mu-training`)? Then run `pip install -e .` again. The old names are gone.
2. **Build CMS Combine into the env.** Follow [CMS Combine](#cms-combine). Combine is **not** in `environment_frozen.yaml`.
3. **Make a VOMS proxy.** You need it only if your outputs go to dCache.
   - The proxy must be in your **home folder**. The nodes cannot see `/tmp`.

   ```bash
   echo 'export X509_USER_PROXY=$HOME/.x509up_u$(id -u)' >> ~/.bashrc    # once: puts this line in your ~/.bashrc
   source ~/.bashrc
   voms-proxy-init --voms cms --valid 192:00                               # every 8 days
   ```

   - Before it submits, the command checks the proxy. The proxy must be valid for the full job chain.
   - Proxy not valid long enough? The command stops with an error.

### 2. Write your config

```bash
cp configs/example_W_raw.yaml my_W.yaml
```

Edit `my_W.yaml`. It starts from raw NanoAOD (stage 0 runs). The comments in the file explain each key.

| Key | What it is |
|---|---|
| `analysis_name` | The name of your run folder. |
| `paths.output_base` | Where the results go. A `root://` dCache URL or a local absolute path. |
| `paths.log_dir` | A local folder for the job scripts and the SLURM logs. |
| `paths.conda_env` | Optional. The env that the jobs use. Not set: the env that is active when you submit. |
| `inputs.signal`, `inputs.background` | Your ROOT files. Globs are allowed. |
| `inputs.already_resonance_skimmed` | `false`: your files are raw NanoAOD. Stage 0 applies the resonance skim. `true`: your files are already resonance-skimmed. Stage 0 is skipped. See [Skimming cuts](#skimming-cuts). |
| `branch_map` | Set it only if your branch names are different. For raw NanoAOD, use the raw names as keys (`TauTo3Mu_*`, `PFCands_*`). |
| `limit.signal_weight` | **Required.** It has no default. See below. |
| `limit.gnn_score_cut` | One cut, or a list for a cut scan. A cut scan runs one Combine job per cut. |
| `models` | Optional. `frozen` (default): the frozen BDT and GNN in `models/W/` (step 13c training `W_raw_Nano_AOD_fixed`; reference: expected limit 1.414 at GNN cut 3.5). Or the path of a finished training GNN phase (`.../<training_name>/gnn_vsN`, from `tau3mu-training`). See below. |

> **`limit.signal_weight`:** the weight that scales the signal MC to the dataset (lumi, cross section, reference BR). The example values (W 0.000019, HF 0.000152) are not yet the physical values: they only scale the signal so that the limits have a comparable size. For a physics result, compute the value for **your** dataset.

> **`models`:** with a training `gnn_vsN` path, the analysis takes from the training manifests: the BDT classifier (of its `bdt_vsN` upstream), the GNN model files, and the settings that belong to the model: BDT preselection cut, graph type, unsculpting mode, conv type and GNN mode. These replace the frozen values. The analysis also has `gnn_vsN` folders (GNN scores): use the one of `tau3mu-training`, under `<training_name>/`. The config stops with an error if the folder is not a training `gnn_vsN` (an analysis `gnn_vsN` is refused), its training is not finished (no `manifest.json`), or a model file is missing. The result manifests record the model source (`models`), so a model change shows as a setting change.

> **Frozen settings:** `configs/analysis_defaults_W.yaml` has the frozen analysis settings: skimming cuts, model paths, mass window, SLURM resources. Do not set them in your file. If you do, the command stops with an error.
>
> **Model settings** (BDT preselection cut, graph type, conv type, GNN mode, unsculpting mode) are defined once, with the model: for the frozen model in `models/W/gnn/<model_name>/model_settings.json`, for a trained model in the manifest of its training `gnn_vsN` (`models:`).

#### Skimming cuts

All cuts apply to each tau candidate. Values are in `configs/analysis_defaults_W.yaml` (`resonance_skim`, `skim`).

| Stage | Cut | Value | Notes |
|---|---|---|---|
| NanoAOD production | 3 muons, medium ID | — | Already true in the raw NanoAOD. The pipeline does not apply it again. |
| NanoAOD production | Total charge | ±1 | Already true in the raw NanoAOD. |
| 0 (resonance skim) | 3-muon vertex χ² | < 20 | `TauTo3Mu_vtx_chi2` |
| 0 | 3-muon mass | 1 < m < 3 GeV | Sum of the three muon 4-vectors (muon mass 0.10566 GeV), not `fitted_mass`. |
| 0 | ω(782) veto | \|m(μμ) − 0.78266\| ≥ 40 MeV | Opposite-sign pairs only, 4-vector mass. |
| 0 | φ(1020) veto | \|m(μμ) − 1.019461\| ≥ 40 MeV | Opposite-sign pairs only, 4-vector mass. |
| 0 | Events | ≥ 1 candidate passes | Other events are removed. |
| 0 | PF candidates kept | ΔR < 1.0 to a kept candidate | Slimming only. The GNN cone is 0.5. |
| 1 (skim) | \|dZ(μi, μj)\| | < 0.5 | All three pairs. |
| 1 | cos α (2D) | > 0.5 | `CosAlpha2D_LxyP3mu` |
| 1 | Events | — | No event is removed. Candidates that fail go out of `Selected_*`. |

Stage 0 output:

- `TauTo3Mu_*` → `Slimmed_TauTo3Mu_*` (passing candidates only). Masks: `Slimmed_TauTo3Mu_pass_*_mask`.
- `TauPlusMET_*` and `nTauPlusMET`: same names, masked like the candidates. So they stay aligned with `Slimmed_TauTo3Mu_*`.
- `PFCands_*` → `Slimmed_PFCands_*`.

> **The old skims are different.** The files in `.../tau3mu_gnn_round2/data/*/old_data/` used: χ² < 100, m < 3 GeV (no lower cut), veto centres 0.782 and 1.020 GeV. Their `TauPlusMET_*` are not masked, so in some events (signal 0.6%, data 2.4%) `TauPlusMET_*` has more entries than `Slimmed_TauTo3Mu_*` and the entry of a candidate cannot be found. Stage 0 therefore refuses resonance-skimmed inputs where any event has `len(TauPlusMET_*) != len(Slimmed_TauTo3Mu_*)`: start from raw NanoAOD. Resonance-skimmed files written by stage 0 pass.

### 3. Submit

Do a dry run first. Then submit.

```bash
tau3mu-analysis all --config my_W.yaml --dry-run    # write the job scripts, print the sbatch commands, submit nothing
tau3mu-analysis all --config my_W.yaml              # submit the phases bdt, gnn and limit at once
```

| Phase | Jobs | Stages |
|---|---|---|
| `bdt` | `skim`, `bdt` | resonance skim (skipped if `already_resonance_skimmed: true`), skim, BDT |
| `gnn` | `gnn_signal`, `gnn_background`, `eval` | GNN scores, evaluation plots |
| `limit` | `limit_<cut>` (one per cut), `summary` | datacard + Combine per cut, table of limits |

How `all` connects the phases (SLURM dependencies):

- `gnn` starts when the `bdt` job of phase `bdt` succeeds.
- `limit` starts when the `eval` job of phase `gnn` succeeds.
- If a job fails, SLURM cancels the jobs after it.

**One phase at a time.** You can also run each phase alone.

- A later phase uses the newest **complete** version of the phase before it.
- To use a different version, give it with `--from`.

```bash
tau3mu-analysis bdt   --config my_W.yaml
tau3mu-analysis gnn   --config my_W.yaml [--from bdt_vs2]
tau3mu-analysis limit --config my_W.yaml [--from gnn_vs1]
```

**One stage, for debugging.** The output goes to `debug/`. It has no manifest.

```bash
tau3mu-analysis stage <skim|bdt|gnn|eval|limit|summary> --config my_W.yaml [--from ...] [--sample signal|background] [--cut X]
```

**Help.** `tau3mu-analysis <command> --help` shows all options. `python -m tau3mu_gnn.pipelines.analysis.cli` is the same command (training: `python -m tau3mu_gnn.pipelines.training.cli`).

### 4. Results

Each phase run makes a **new version folder**. The pipeline does not overwrite a folder.

```
<output_base>/<analysis_name>/
├── bdt_vs1/               0_resonance_skim/, 1_skim/, 2_bdt/
├── gnn_vs1/               3_gnn/, 4_eval/ (tau mass, ROC and SIC plots)
├── limit_vs1/             cut_<cut>/ (datacard, Combine logs, limit.json), limits.json, limits.txt, limits.png
└── debug/<stage>_vsN/     single-stage runs
```

Each version folder also has these files:

- `config_used.yaml`
- `jobs.json`: the SLURM job IDs.
- `logs/`: a copy of the SLURM logs.
- `manifest.json`: only when the phase succeeded.

> **A version is complete only if `manifest.json` exists.**

- **The limit:** `limit_vsN/limits.txt` has the expected limit and the best cut.
- **Local copy:** `<log_dir>/analysis/<analysis_name>/<phase>_vsN/` also has the job scripts and the SLURM logs.

### 5. When a job fails

1. **Find the job.**
   - `squeue -u $USER` shows the jobs that wait or run.
   - `sacct -j <ID>` shows the state of one job.
   - Job names have the format `<phase><version>.<job>.<analysis_name>`, for example `GNN1.gnnS.my_W`.
     - Phase: `BDT`, `GNN`, `LIM`, or `D` for a debug stage. Version: the `N` of `_vsN`.
     - Job: `skim`, `bdt`, `gnnS` (gnn_signal), `gnnB` (gnn_background), `eval`, `lim<cut>`, `sum` (summary).
2. **Read the log** of the failed job: `<log_dir>/analysis/<analysis_name>/<phase>_vsN/<job>-<ID>.err`.
3. **Fix the problem. Then submit the phase again.**
   - This makes the next version, for example `gnn_vs2`.
   - The failed version stays. It has no manifest.
4. **Out of memory or time?** Increase the value for that job in `configs/analysis_defaults_W.yaml` (`slurm.jobs`).

---

## Training Pipeline (tau3mu-training)

Trains a new BDT and GNN on your inputs, and computes a BDT-only benchmark limit. You need it only if you want new models. Without it, the analysis uses the frozen models.

### 1. Write the training config

```bash
cp configs/example_training_W.yaml my_training_W.yaml
```

The comments in the file explain each key. The frozen training settings (BDT and GNN recipe, graph settings, Comet ML, SLURM resources) are in `configs/training_defaults_W.yaml`. The shared settings (skim cuts, limit fit) come from `configs/analysis_defaults_W.yaml`.

| Key | What it is |
|---|---|
| `training_name` | The name of the training folder. |
| `paths`, `inputs`, `branch_map` | As in the analysis config. |
| `graphs.bdt_preselection_cut` | **Required.** Only events with `bdt_ISO >= cut` go into the GNN training graphs. |
| `graphs.max_events`, `bdt_training.max_events_per_class` | Optional. Defaults 600000 and 100000. |
| `benchmark.bdt_score_cut` | **Required.** One cut or a list on `bdt_ISO` for the BDT-only benchmark limit. |
| `limit.signal_weight` | **Required.** As in the analysis config. |

### 2. Submit

```bash
voms-proxy-init --voms cms --valid 192:00     # the longest job chain is about 157 h
tau3mu-training all --config my_training_W.yaml --dry-run
tau3mu-training all --config my_training_W.yaml
```

`all` submits three phases. You can also submit one phase: `tau3mu-training bdt|gnn|benchmark --config ...` (`gnn` and `benchmark` take `--from bdt_vsN`).

| Phase | Jobs | Needs |
|---|---|---|
| `bdt` | skim → BDT training → BDT inference | your inputs |
| `gnn` | npvs weights → signal graphs; background graphs → GNN training (GPU) | a `bdt_vsN` |
| `benchmark` | one BDT-only limit per cut → summary | a `bdt_vsN` |

The jobs use the code of the folder you submit from. Do not change that folder while the jobs run. The GNN training runs on the `gpu` partition. Without a visible GPU it trains on CPU and writes a warning to its log. Each job checks at its start that the proxy is valid for its own time limit.

### 3. Results

```
<output_base>/<training_name>/
├── bdt_vsN/         1_skim/, bdt_train/ (ISO_classifiers.pck, plots), 2_bdt/
├── gnn_vsN/         npvs/, graphs/, gnn_train/<model_name>/ (model, Normalization.pkl, arch_config.json)
└── benchmark_vsN/   cut_<cut>/, limits.json, limits.txt, limits.png   (BDT-only limit, best cut)
```

The job scripts and the SLURM logs: `<log_dir>/training/<training_name>/<phase>_vsN/`.

Job names: `BDT<N>`, `GNN<N>`, `BEN<N>` plus the job, for example `GNN1.gnnT.my_training`.

### 4. Use the new models in the analysis

Set `models` in your analysis config to the `gnn_vsN` folder, then run the analysis as usual (`tau3mu-analysis all`):

```yaml
models: root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/<you>/tau3mu_pipeline/<training_name>/gnn_vs1
```

The analysis then uses the new BDT and GNN, and the settings that belong to them (see [`models`](#2-write-your-config)). Compare `limit_vsN/limits.txt` with the benchmark `benchmark_vsN/limits.txt`: the GNN limit should be lower than the BDT-only limit.

---

## HF Channel

Both pipelines also run the HF channel (Ds → τ → 3μ). Set `channel: HF` in the config: it selects `configs/analysis_defaults_HF.yaml` and `configs/training_defaults_HF.yaml`. Design: `docs/superpowers/specs/2026-10-25-step13-hf-design.md`.

What differs from W:

- BDT and GNN graph-level features: the HF lists in `bdt_training/bdt_inputs.py` and `gnn_training/gnn_inputs.py` (μ3 pT, isGlobal of the 3 muons, max `Muon_tkRelIso` for the BDT, no MET features). The inputs need the `Muon_*` collection and `TauTo3Mu_mu{1,2,3}_idx`.
- Signal weight and cut scans: set in your config (examples: `0.000152` for Kaito's old HF dataset, BDT benchmark scan 0.88–0.97, GNN scan 2.0–8.0).
- **No frozen HF models yet.** An HF analysis needs `models:` set to an HF training `gnn_vsN`. Without it, the config check stops with `frozen model file not found: .../models/HF/...`. A W training is refused for an HF analysis (and the other way round).
- All other settings (skim cuts, BDT hyper-parameters, graphs, GNN training, limit fit, SLURM resources) are the W values.

Run (example files with Kaito's old HF skimmed inputs):

```bash
tau3mu-training all --config configs/example_training_HF.yaml
# when the training is done: set models: in your copy of example_HF.yaml to the new gnn_vsN
tau3mu-analysis all --config configs/example_HF.yaml
```

---

## Key Concepts

### Channels

The pipeline supports two τ production channels:

- **W channel** (`--tau_channel W`): τ from W → τν decay.
- **HF channel** (`--tau_channel HF`): τ from heavy-flavour decays.

These are different for each channel:

- the graph-level features
- the signal weights
- some file names

> **Use the same channel in every pipeline step.**

### Graph Construction

Each event is a graph.

- The nodes are the particle-flow (PF) candidates.
- Only candidates in a cone of ΔR < 0.5 around the τ candidate are used.

**Edge connection modes** (`--graph_type`):

| Mode | Description |
|---|---|
| `knn` | Each node connects to its k nearest neighbours in (η, φ) space |
| `r_threshold` | All pairs within ΔR < δ are connected |
| `fully` | All nodes connected to all others |

### Unsculpting / Mass Decorrelation

**The problem:** the GNN can sculpt the τ invariant mass distribution. This gives a bias to the background model of the limit fit.

**The solutions:** `--unsculpting_mode` selects one strategy:

| Mode | Description |
|---|---|
| `delete_muons` | Removes the 3 signal muons from PF candidates before graph construction |
| `permutation` | Permutes muon-to-PF-candidate assignments |
| `split_pt_uniformly` | Replaces muon pT with a fixed value drawn from a uniform distribution |
| *(None)* | No unsculpting applied |

You can also use DisCo regularisation (see below). Use it alone or together with these modes.

> **Name difference:**
> - `GraphCreation.py` and `ConeGraphProducer.py` use `split_pt_uniformly`.
> - `Train_TauMuNet.py` uses `split_pt`.
> - Use the name of the script that you call.

### GNN Architecture (TauMuNet)

The model `TauMuNet` is a message-passing GNN. Its structure:

1. **Graph convolution layers.** You select the number of layers and the convolution type.
2. **Feature fusion.** The outputs of all convolution layers are concatenated. Then they are projected into one shared representation.
3. **Scale-down layers.** Optional fully-connected layers.
4. **Global mean pooling** over the nodes.
5. **Graph-level features** go in. The position depends on `gnn_mode`.
6. **Classification head.** Fully-connected, with sigmoid output. Logits are better for stability.

**Convolution types** (`--conv_type`):

| Type | Description |
|---|---|
| `GINE` | GIN with edge features (Graph Isomorphism Network with Edges) |
| `GAT` | Graph Attention Network |
| `Transformer` | Transformer-based graph convolution |

**Hyperparameters:**

- `GNN_archs.py` defines the number of layers, the hidden dimensions and the dropout.
- `--arch` selects one configuration.
- `--arch random` takes a configuration from a predefined random grid.

### Graph Feature Integration Modes

The graph-level features are τ kinematics, MET and vertex variables. `--gnn_mode` sets where they go into the GNN:

| Mode | Description |
|---|---|
| `BDT` | No graph-level features injected into the GNN; used as input to an XGBoost BDT post-classifier |
| `embed_once` | Graph-level features concatenated to node features at the input layer only |
| `embed_always` | Graph-level features concatenated to node features at every convolution layer |
| `final_layer` | Graph-level features concatenated after global pooling, before the classification head |

### DisCo Regularisation

The training loss can have a distance-correlation (DisCo) penalty.

- The penalty is between the GNN output score and the reconstructed τ mass.
- It is used only on background events. Signal events are masked out.

```
L = BCE + λ · DisCo(score, m_τ)
```

- `--disco` sets the strength λ. λ = 0 turns it off.
- λ increases linearly over the first 20 epochs (warm-up).

### Fine-Tuning

After the main training, you can do a second training phase: fine-tuning.

- It uses only the events above a BDT ISO score threshold.

```bash
--finetune
--finetune_lr      <float>   # learning rate for fine-tuning
--finetune_epochs  <int>     # number of fine-tuning epochs
--bdt_threshold    <float>   # minimum BDT ISO score for event inclusion
```

- The checkpoint before fine-tuning is saved separately, as `<name>_pretrained.pt`.
- Fine-tuning has a higher early-stopping patience. The smaller dataset needs more time to adapt.

---

## BDT Baseline

**The primary baseline is the ISO BDT** (input: `rel_iso`). The analysis also uses this classifier now.

- An old variant uses `GNN_score` as input. It is no longer the focus.
- You can train and score both variants:

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

Graph creation can also save the BDT ISO score as a graph-level feature (`--bdt_preselection_cut`). Fine-tuning then uses it to pre-select events.

---

## Data Storage and Transfer

The processed data is on the T3 CMS storage. You access it with `xrootd`.

> **Proxy:** all jobs that read from or write to `/pnfs/` need a valid VOMS proxy:

```bash
voms-proxy-init --voms cms --valid 192:00
```

Data paths on the T3 storage:

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

The Combine outputs and the limit results go to:

- `/work/kfuss/SVJ_GNN/output/`
- `/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/<MODEL_NAME>`

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

The log files go to `/work/kfuss/SVJ_GNN/logs/`.

### lxplus (HTCondor)

For full-size training, submit with HTCondor on lxplus:

```bash
condor_submit jobs/condor/train_W_GINE.sub
```

- The `.sub` file points to an executable `.sh` script. The script is on AFS.
- The logs go to `/afs/cern.ch/user/k/kfuss/condor_logs/`.
- HTCondor jobs ask for one GPU with:
  - compute capability ≥ 7.0
  - ≥ 30 GB VRAM

---

## Contact

kaito.fuss@gmail.com
