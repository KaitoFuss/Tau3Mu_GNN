# Tau3Mu GNN

A Graph Neural Network (GNN) pipeline that searches for the decay **τ → 3μ** in CMS data.

- This decay violates lepton flavour.
- The GNN must be better than the current BDT baseline.
- The physics goal: an expected upper limit on the branching ratio B(τ → 3μ), set with CMS Combine.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Infrastructure](#infrastructure)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [CMS Combine](#cms-combine)
  - [Check Your Setup](#check-your-setup)
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
- [BDT Baseline (old scripts)](#bdt-baseline-old-scripts)
- [Data Storage and Transfer](#data-storage-and-transfer)
- [Batch Systems](#batch-systems)
- [Contact](#contact)

---

## Quick Start

lxplus, W channel, from flat NTuples to a limit (Chiara Basile's [NTuplizer](https://github.com/BasChiara/Tau3muNANO/tree/CMSSW_14_0_16) with the addition of PFCands retention was used in this example):

Needs miniforge on EOS (see [Installation](#installation)).

```bash
cd /eos/user/${USER:0:1}/$USER                                                           # EOS, not AFS (AFS home: 2 GB)
git clone git@github.com:KaitoFuss/Tau3Mu_GNN.git && cd Tau3Mu_GNN
conda env create -f environment_frozen.yaml -p /eos/user/${USER:0:1}/$USER/envs/Tau3Mu   # once, see Installation
conda activate /eos/user/${USER:0:1}/$USER/envs/Tau3Mu
pip install -r requirements_frozen.txt && pip install -e .                               # once
# once: build CMS Combine into the env, see CMS Combine
cp configs/example_W_raw.yaml my_W.yaml                                                  # edit analysis_name
tau3mu-analysis all --config my_W.yaml --dry-run
tau3mu-analysis all --config my_W.yaml
condor_q                                                                                 # watch
cat /eos/user/${USER:0:1}/$USER/tau3mu_pipeline/analysis/<analysis_name>/limit_vs1/limits_gnn.txt            # GNN limit
cat /eos/user/${USER:0:1}/$USER/tau3mu_pipeline/analysis/<analysis_name>/benchmark_vs1/limits_benchmark.txt  # BDT-only limit
```

First time? Do [Check Your Setup](#check-your-setup) before a full run.

---

## Infrastructure

All work runs on **CERN lxplus**:

| What | Where |
|---|---|
| Batch jobs | HTCondor (`condor_submit_dag`), the default (`scheduler: htcondor`) |
| Inputs | EOS: raw W NanoAOD in the B-physics group space, the old HF skims in Kaito's EOS space (see [Data Storage](#data-storage-and-transfer)) |
| Outputs | Your EOS user space, `/eos/user/<initial>/$USER/tau3mu_pipeline` |
| Job scripts and HTCondor logs | Your AFS home, `/afs/cern.ch/user/<initial>/$USER/tau3mu_logs` |

> **PSI T3 (SLURM) still works.** Set `scheduler: slurm` in your config and use T3 paths (dCache `output_base`, `log_dir` on `/work`). See [Batch Systems](#batch-systems).

> **`$USER_INITIAL`:** config paths can use `$USER_INITIAL`, the first letter of `$USER` (`/eos/user/$USER_INITIAL/$USER` = `/eos/user/k/kfuss`).

> **Other account?** The example configs use `$USER` and `$USER_INITIAL` for your output and log folders. The old manual job scripts in `jobs/` still have hardcoded `kfuss` paths.

> **Warning: data paths may not be readable for you.** The inputs, the HF models and the reference results are in Kaito's EOS spaces (`/eos/user/k/kfuss/...`, `/eos/cms/store/group/phys_bphys/kfuss/...`). Kaito's user space is not shared by default. No read access: ask Kaito, or point `inputs` and `models` to your own copies.

---

## Repository Structure

```
tau3mu_gnn/
├── environment.yaml              # Conda environment spec (unpinned)
├── environment_frozen.yaml       # Pinned conda environment (reproducible)
├── requirements_frozen.txt       # Pinned pip packages (reproducible)
├── pyproject.toml                # Package definition (src layout)
├── configs/                      # Pipeline configs: example_*.yaml (copy + edit), *_defaults_*.yaml (frozen)
├── models/                       # Frozen models: models/W/bdt/, models/W/gnn/<model_name>/
├── tests/pipelines/              # Tests of the pipelines (pytest)
├── jobs/                         # Cluster job scripts (old manual workflow; the pipelines write their own)
│   ├── migration/                # Copy T3 files to EOS (t3_list_files.py, copy_to_eos.sh)
│   ├── graph_creation/           # SLURM: skimming and graph creation
│   ├── gnn_training/             # SLURM: GNN training on T3 (for quick tests)
│   ├── condor/                   # HTCondor: GNN training on lxplus
│   ├── bdt/                      # SLURM: BDT training and score production
│   ├── gnn_score_creation/       # SLURM: GNN inference on T3
│   ├── model_eval/               # SLURM: ROC comparisons and evaluations
│   └── analysis/                 # SLURM: datacard creation and limit setting
└── src/tau3mu_gnn/
    ├── pipelines/                # tau3mu-analysis and tau3mu-training (start here)
    │   ├── common/               # shared: config parts, storage, batch (HTCondor, SLURM), manifests, shared stages
    │   ├── analysis/             # analysis pipeline (phases bdt, gnn, limit, benchmark)
    │   └── training/             # training pipeline (phases bdt, gnn)
    ├── scripts/                  # Old manual entry points (the pipelines use the code, not these scripts)
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

> **lxplus:** put miniforge and the env on EOS, not on AFS (AFS home is only 2 GB).
> - Miniforge: install it into `/eos/user/<initial>/$USER/miniforge3` ([installer](https://github.com/conda-forge/miniforge)).
> - Env: `conda env create` puts the env into the miniforge folder. For another place, add `-p`: `conda env create -f environment_frozen.yaml -p /eos/user/<initial>/$USER/envs/Tau3Mu`. Activate it by path.
> - The jobs activate the same env from EOS.
> - Tests: `pip install pytest`, then `python -m pytest tests` (about 13 min on EOS).

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
- Tested on lxplus. Activate the env first: ROOT needs the compiler of the env.

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

### Check Your Setup

Run the limit on the GNN scores of the W reference run. Small (3 limit jobs, about 1.5 h). Known result. Tests HTCondor, the env, Combine, reading and writing EOS.

Needs read access to the reference run (see the warning in [Infrastructure](#infrastructure)).

```bash
REF=/eos/user/k/kfuss/tau3mu_pipeline/analysis/W_raw_Nano_AOD_fixed
MY=/eos/user/${USER:0:1}/$USER/tau3mu_pipeline/analysis/W_raw_Nano_AOD_fixed
mkdir -p $MY && cp -r $REF/gnn_vs1 $MY/       # GNN scores, about 42 GB (Kaito: REF = MY, skip this)
cp configs/example_W_raw.yaml check_W.yaml
```

Edit `check_W.yaml`:

```yaml
analysis_name: W_raw_Nano_AOD_fixed
paths:
  output_base: root://eosuser.cern.ch//eos/user/$USER_INITIAL/$USER/tau3mu_pipeline/analysis
limit:
  gnn_score_cut: [3.0, 3.25, 3.5]
```

```bash
tau3mu-analysis limit --config check_W.yaml --from gnn_vs1
cat $MY/limit_vsN/limits_gnn.txt             # N: the version the command printed (results: ...)
```

Expected (median expected limit at CL 0.9):

| Cut | Limit |
|---|---|
| 3.0 | 1.43359375 |
| 3.25 | 1.4296875 |
| 3.5 | 1.4140625 (best) |

A warning that the settings differ from `gnn_vs1` is normal: the reference run used other input paths.

---

## Analysis Pipeline (tau3mu-analysis)

This pipeline runs the frozen W analysis on **your** input files. You need only one config file.

- Stages: resonance skim (raw NanoAOD only) → skim → BDT → GNN → evaluation → limit.
- The models are fixed. The pipeline does not train.
- All jobs run on lxplus with HTCondor (PSI T3 with SLURM: `scheduler: slurm`, see [Batch Systems](#batch-systems)).

### 1. Set up (once)

1. **Install the env and the package.** Follow [Installation](#installation).
   - `pip install -e .` also installs the commands `tau3mu-analysis` and `tau3mu-training`.
   - Command not found? Run `pip install -e .` again.
2. **Build CMS Combine into the env.** Follow [CMS Combine](#cms-combine). Combine is **not** in `environment_frozen.yaml`.
3. **lxplus with EOS output: nothing to do.** EOS uses your Kerberos ticket (you get it when you log in); HTCondor gives it to the jobs.
4. **A VOMS proxy** is needed only if your outputs go to dCache (T3).
   - With SLURM, the proxy must be in your **home folder**. The nodes cannot see `/tmp`. HTCondor sends the proxy with the job.

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
| `scheduler` | Optional. `htcondor` (default, lxplus) or `slurm` (PSI T3). |
| `paths.output_base` | Where the results go. A `root://` URL (EOS `root://eosuser.cern.ch//eos/user/...`, or dCache) or a local absolute path. |
| `paths.log_dir` | A local folder for the job scripts and the batch logs. On lxplus: AFS (HTCondor cannot write its logs to EOS). |
| `paths.conda_env` | Optional. The env that the jobs use. Not set: the env that is active when you submit. |
| `inputs.signal`, `inputs.background` | Your ROOT files. Globs are allowed. |
| `inputs.already_resonance_skimmed` | `false`: your files are raw NanoAOD. Stage 0 applies the resonance skim. `true`: your files are already resonance-skimmed. Stage 0 is skipped. See [Skimming cuts](#skimming-cuts). |
| `branch_map` | Set it only if your branch names are different. For raw NanoAOD, use the raw names as keys (`TauTo3Mu_*`, `PFCands_*`). |
| `limit.signal_weight` | **Required.** It has no default. See below. |
| `limit.gnn_score_cut` | One cut, or a list for a cut scan, on the GNN logit. A cut scan runs one Combine job per cut. Each cut must be > -1 (see below). |
| `benchmark.bdt_score_cut` | **Required.** One cut or a list on `bdt_ISO`, each in (0, 1), for the BDT-only benchmark limit. |
| `models` | Optional. `frozen` (default): the frozen BDT and GNN in `models/W/` (training `W_raw_Nano_AOD_fixed`; reference: expected limit 1.414 at GNN cut 3.5). Or the path of a finished training GNN phase (`.../<training_name>/gnn_vsN`, from `tau3mu-training`). See below. |

> **`limit.signal_weight`:** the weight that scales the signal MC to the dataset (lumi, cross section, reference BR). The example values (W 0.000019, HF 0.000152) are not yet the physical values: they only scale the signal so that the limits have a comparable size. For a physics result, compute the value for **your** dataset.

> **`limit.gnn_score_cut` > -1:** the GNN scores only the events with `bdt_ISO >= bdt_preselection_cut`. The other events stay in the file with `GNN_score = -1`. A cut at -1 or lower would let them into the limit, so the config check refuses it.

> **`benchmark`:** the BDT-only limit (the limit the GNN must beat). It cuts on `bdt_ISO` in the stage 2 files of the same `bdt_vsN` that the GNN reads. So both limits use the same events, `signal_weight` and mass fit, and you can compare them.

> **`models`:** with a training `gnn_vsN` path, the analysis takes from the training manifests: the BDT classifier (of its `bdt_vsN` upstream), the GNN model files, and the settings that belong to the model: BDT preselection cut, graph type, unsculpting mode, conv type and GNN mode. These replace the frozen values. The analysis also has `gnn_vsN` folders (GNN scores): use the one of `tau3mu-training`, under `<training_name>/`. The config stops with an error if the folder is not a training `gnn_vsN` (an analysis `gnn_vsN` is refused), its training is not finished (no `manifest.json`), or a model file is missing. The result manifests record the model source (`models`), so a model change shows as a setting change.

> **Frozen settings:** `configs/analysis_defaults_W.yaml` has the frozen analysis settings: skimming cuts, model paths, mass window, batch resources (`htcondor.jobs`, `slurm.jobs`). Do not set them in your file. If you do, the command stops with an error.
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
tau3mu-analysis all --config my_W.yaml --dry-run    # write the job files, print the submit command, submit nothing
tau3mu-analysis all --config my_W.yaml              # submit the phases bdt, gnn, limit and benchmark at once
```

| Phase | Jobs | Stages |
|---|---|---|
| `bdt` | `skim`, `bdt` | resonance skim (skipped if `already_resonance_skimmed: true`), skim, BDT |
| `gnn` | `gnn_signal`, `gnn_background`, `eval` | GNN scores, evaluation plots |
| `limit` | `limit_<cut>` (one per cut), `summary` | GNN limit: datacard + Combine per cut on `GNN_score`, table of limits |
| `benchmark` | `benchmark_limit_<cut>` (one per cut), `benchmark_summary` | BDT-only limit: datacard + Combine per cut on `bdt_ISO`, table of limits |

How `all` connects the phases (HTCondor: one DAG over the four phases; SLURM: job dependencies):

- `gnn` and `benchmark` start when the `bdt` job of phase `bdt` succeeds. They run in parallel.
- `limit` starts when the `eval` job of phase `gnn` succeeds.
- If a job fails, the jobs after it do not run.
- `summary` and `benchmark_summary` run also if some cut jobs fail. With HTCondor the DAG then still shows as done: check `limits_gnn.txt` and `limits_benchmark.txt` for missing cuts.

**One phase at a time.** You can also run each phase alone.

- A later phase uses the newest **complete** version of the phase before it.
- To use a different version, give it with `--from`.

```bash
tau3mu-analysis bdt       --config my_W.yaml
tau3mu-analysis gnn       --config my_W.yaml [--from bdt_vs2]
tau3mu-analysis limit     --config my_W.yaml [--from gnn_vs1]
tau3mu-analysis benchmark --config my_W.yaml [--from bdt_vs2]
```

**One stage, for debugging.** The output goes to `debug/`. It has no manifest.

```bash
tau3mu-analysis stage <skim|bdt|gnn|eval|limit|summary|benchmark_limit|benchmark_summary> --config my_W.yaml [--from ...] [--sample signal|background] [--cut X]
```

**Help.** `tau3mu-analysis <command> --help` shows all options. `python -m tau3mu_gnn.pipelines.analysis.cli` is the same command (training: `python -m tau3mu_gnn.pipelines.training.cli`).

### 4. Results

Each phase run makes a **new version folder**. The pipeline does not overwrite a folder.

```
<output_base>/<analysis_name>/
├── bdt_vs1/               0_resonance_skim/, 1_skim/, 2_bdt/
├── gnn_vs1/               3_gnn/, 4_eval/ (tau mass, ROC and SIC plots)
├── limit_vs1/             cut_<cut>/ (datacard, Combine logs, limit.json), limits_gnn.{json,txt,png}        (GNN limit)
├── benchmark_vs1/         cut_<cut>/ (datacard, Combine logs, limit.json), limits_benchmark.{json,txt,png}   (BDT-only limit)
└── debug/<stage>_vsN/     single-stage runs
```

Each version folder also has these files:

- `config_used.yaml`
- `jobs.json`: the job IDs (HTCondor: `<DAG cluster>:<node>`; SLURM: the job ID).
- `logs/`: a copy of the job logs.
- `manifest.json`: only when the phase succeeded.

> **A version is complete only if `manifest.json` exists.**

- **The limits:** `limit_vsN/limits_gnn.txt` (GNN) and `benchmark_vsN/limits_benchmark.txt` (BDT only) have the expected limit and the best cut. The GNN limit should be lower. Compare only versions of the same `bdt_vsN` (see `upstream` in the manifests).
- **Local copy:** `<log_dir>/analysis/<analysis_name>/<phase>_vsN/` also has the job scripts and the batch logs. With HTCondor, the DAG (`jobs.dag` and the DAGMan files) is in the folder of the first phase.

### 5. When a job fails

1. **Find the job.**
   - HTCondor: `condor_q` shows the DAGs that run (batch name `<analysis_name>.<versions>`, e.g. `my_W.bdt_vs1+gnn_vs1+limit_vs1`); `condor_q -dag -nobatch` shows each job; `condor_history -limit 20` shows finished jobs.
   - SLURM: `squeue -u $USER`, `sacct -j <ID>`. Job names: `<phase><version>.<job>.<analysis_name>`, for example `GNN1.gnnS.my_W` (phase `BDT`, `GNN`, `LIM`, `BEN`, or `D` for a debug stage; job `skim`, `bdt`, `gnnS`, `gnnB`, `eval`, `lim<cut>`, `sum`, `bl<cut>`, `bsum`).
2. **Read the log** of the failed job: `<log_dir>/analysis/<analysis_name>/<phase>_vsN/<job>-<ID>.err` (also in `<phase>_vsN/logs/` next to the results). HTCondor: `jobs.dag.dagman.out` in the folder of the first phase says which node failed.
3. **Fix the problem. Then submit the phase again.**
   - This makes the next version, for example `gnn_vs2`.
   - The failed version stays. It has no manifest.
4. **Out of memory or time?** Increase the value for that job in `configs/analysis_defaults_W.yaml` (`htcondor.jobs`, or `slurm.jobs` on T3). Job memory: powers of 2 (64G, 128G, 256G). HTCondor job out of scratch disk? Increase `disk`.

---

## Training Pipeline (tau3mu-training)

Trains a new BDT and GNN on your inputs. You need it only if you want new models. Without it, the analysis uses the frozen models. The BDT-only benchmark limit is a phase of the analysis (same events as the GNN limit).

### 1. Write the training config

```bash
cp configs/example_training_W.yaml my_training_W.yaml
```

The comments in the file explain each key. The frozen training settings (BDT and GNN recipe, graph settings, Comet ML, batch resources) are in `configs/training_defaults_W.yaml`.

> **Comet ML:** the defaults files have Kaito's Comet workspace and API key (`comet:`). Your GNN training logs into that workspace. Own workspace: edit `comet:` in `configs/training_defaults_<channel>.yaml` of your checkout. Do not share the key outside the group. The shared settings (skim cuts, limit fit) come from `configs/analysis_defaults_W.yaml`.

| Key | What it is |
|---|---|
| `training_name` | The name of the training folder. |
| `paths`, `inputs`, `branch_map` | As in the analysis config. |
| `graphs.bdt_preselection_cut` | **Required.** Only events with `bdt_ISO >= cut` go into the GNN training graphs. |
| `graphs.max_events`, `bdt_training.max_events_per_class` | Optional. Defaults 600000 and 100000. |

### 2. Submit

```bash
# only for dCache output (T3): voms-proxy-init --voms cms --valid 192:00   (the longest job chain is about 157 h)
tau3mu-training all --config my_training_W.yaml --dry-run
tau3mu-training all --config my_training_W.yaml
```

`all` submits two phases. You can also submit one phase: `tau3mu-training bdt|gnn --config ...` (`gnn` takes `--from bdt_vsN`).

| Phase | Jobs | Needs |
|---|---|---|
| `bdt` | skim → BDT training → BDT inference | your inputs |
| `gnn` | npvs weights → signal graphs; background graphs → GNN training (GPU) | a `bdt_vsN` |

The jobs use the code of the folder you submit from. Do not change that folder while the jobs run. The GNN training asks for one GPU (HTCondor: `request_gpus = 1`, compute capability ≥ 7.0, ≥ 30 GB GPU memory; SLURM: the `gpu` partition). Without a visible GPU it trains on CPU and writes a warning to its log. With dCache output, each job checks at its start that the proxy is valid for its own time limit.

### 3. Results

```
<output_base>/<training_name>/
├── bdt_vsN/         1_skim/, bdt_train/ (ISO_classifiers.pck, plots), 2_bdt/
└── gnn_vsN/         npvs/, graphs/, gnn_train/<model_name>/ (model, Normalization.pkl, arch_config.json)
```

The job scripts and the batch logs: `<log_dir>/training/<training_name>/<phase>_vsN/`.

HTCondor batch name: `<training_name>.<versions>`. SLURM job names: `BDT<N>`, `GNN<N>` plus the job, for example `GNN1.gnnT.my_training`.

### 4. Use the new models in the analysis

Set `models` in your analysis config to the `gnn_vsN` folder, then run the analysis as usual (`tau3mu-analysis all`):

```yaml
models: root://eosuser.cern.ch//eos/user/<initial>/<you>/tau3mu_pipeline/training/<training_name>/gnn_vs1
```

The analysis then uses the new BDT and GNN, and the settings that belong to them (see [`models`](#2-write-your-config)). Compare `limit_vsN/limits_gnn.txt` with `benchmark_vsN/limits_benchmark.txt` of the same analysis: the GNN limit should be lower than the BDT-only limit.

---

## HF Channel

Both pipelines also run the HF channel (Ds → τ → 3μ). Set `channel: HF` in the config: it selects `configs/analysis_defaults_HF.yaml` and `configs/training_defaults_HF.yaml`.

What differs from W:

- BDT and GNN graph-level features: the HF lists in `bdt_training/bdt_inputs.py` and `gnn_training/gnn_inputs.py` (μ3 pT, isGlobal of the 3 muons, max `Muon_tkRelIso` for the BDT, no MET features). The inputs need the `Muon_*` collection and `TauTo3Mu_mu{1,2,3}_idx`.
- Signal weight and cut scans: set in your config (examples: `0.000152` for Kaito's old HF dataset, BDT benchmark scan 0.88–0.97, GNN scan 0.0–3.0 on the GNN logit).
- **No frozen HF models yet.** An HF analysis needs `models:` set to an HF training `gnn_vsN` (`example_HF.yaml`: Kaito's `HF_training_fixed/gnn_vs1`). Without it, the config check stops with `frozen model file not found: .../models/HF/...`. A W training is refused for an HF analysis (and the other way round).
- All other settings (skim cuts, BDT hyper-parameters, graphs, GNN training, limit fit, batch resources) are the W values.

Run (example files with Kaito's old HF skimmed inputs):

```bash
tau3mu-training all --config configs/example_training_HF.yaml
# when the training is done: set models: in your copy of example_HF.yaml to the new gnn_vsN
tau3mu-analysis all --config configs/example_HF.yaml
```

---

## Key Concepts

> **Pipeline settings vs old scripts.** The pipelines set these options in the frozen `configs/training_defaults_<channel>.yaml` (key names below). Change them there, not in your user config. The `--flags` belong to the old manual scripts (`scripts/Train_TauMuNet.py`, `scripts/GraphCreation.py`).

### Channels

The pipeline supports two τ production channels:

- **W channel** (`channel: W`): τ from W → τν decay.
- **HF channel** (`channel: HF`): τ from heavy-flavour decays. See [HF Channel](#hf-channel).

These are different for each channel:

- the graph-level features
- the signal weights
- some file names

> **Use the same channel in every pipeline step.**

### Graph Construction

Each event is a graph.

- The nodes are the particle-flow (PF) candidates.
- Only candidates in a cone of ΔR < 0.5 around the τ candidate are used.

**Edge connection modes** (`graphs.connection`; old scripts: `--graph_type`). Cone: `graphs.cone_size`, neighbours: `graphs.k`.

| Mode | Description | Pipeline |
|---|---|---|
| `knn` | Each node connects to its k nearest neighbours in (η, φ) space | yes (frozen: k = 10) |
| `r_threshold` | All pairs within ΔR < δ are connected | no (old scripts only) |
| `fully` | All nodes connected to all others | yes |

### Unsculpting / Mass Decorrelation

**The problem:** the GNN can sculpt the τ invariant mass distribution. This gives a bias to the background model of the limit fit.

**The solutions:** `graphs.unsculpting_mode` (old scripts: `--unsculpting_mode`) selects one strategy:

| Mode | Description | Pipeline |
|---|---|---|
| `delete_muons` | Removes the 3 signal muons from PF candidates before graph construction | yes (frozen) |
| `permutation` | Permutes muon-to-PF-candidate assignments | no |
| `split_pt_uniformly` | Replaces muon pT with a fixed value drawn from a uniform distribution | no |
| `none` | No unsculpting applied | yes |

The pipeline refuses `permutation` and `split_pt_uniformly`: GNN inference cannot apply them, so training and inference graphs would differ.

You can also use DisCo regularisation (see below). Use it alone or together with these modes.

> **Old scripts, name difference:** `GraphCreation.py` and `ConeGraphProducer.py` use `split_pt_uniformly`, `Train_TauMuNet.py` uses `split_pt`.

### GNN Architecture (TauMuNet)

The model `TauMuNet` is a message-passing GNN. Its structure:

1. **Graph convolution layers.** You select the number of layers and the convolution type.
2. **Feature fusion.** The outputs of all convolution layers are concatenated. Then they are projected into one shared representation.
3. **Scale-down layers.** Optional fully-connected layers.
4. **Global mean pooling** over the nodes.
5. **Graph-level features** go in. The position depends on `gnn_mode`.
6. **Classification head.** Fully-connected, with sigmoid output. Logits are better for stability.

**Convolution types** (`gnn_training.conv_type`, frozen: `GINE`; old scripts: `--conv_type`):

| Type | Description |
|---|---|
| `GINE` | GIN with edge features (Graph Isomorphism Network with Edges) |
| `GAT` | Graph Attention Network |
| `Transformer` | Transformer-based graph convolution |

**Hyperparameters:**

- `GNN_archs.py` defines the number of layers, the hidden dimensions and the dropout.
- `gnn_training.arch` selects one configuration (frozen: `fl8_sd4`).
- Old scripts only: `--arch random` takes a configuration from a predefined random grid.

### Graph Feature Integration Modes

The graph-level features are τ kinematics, MET and vertex variables. `gnn_training.gnn_mode` (frozen: `final_layer`; old scripts: `--gnn_mode`) sets where they go into the GNN:

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

- `gnn_training.disco` (old scripts: `--disco`) sets the strength λ. λ = 0 turns it off (frozen: 0).
- λ increases linearly over the first 20 epochs (warm-up).

### Fine-Tuning

**Old scripts only.** The training pipeline does no fine-tuning.

After the main training, `Train_TauMuNet.py` can do a second training phase: fine-tuning.

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

## BDT Baseline (old scripts)

**The baseline is the ISO BDT** (input: `rel_iso`). Both pipelines train or use it themselves:

- `tau3mu-training`, phase `bdt`: trains the ISO BDT.
- `tau3mu-analysis`, phase `bdt`: BDT scores with the frozen (or `models:`) classifier. Phase `benchmark`: the BDT-only limit.

The old manual scripts still exist. You do not need them for the pipelines:

```bash
python -m tau3mu_gnn.bdt_training.Train_XGB_W --model <model_name>          # train (W channel)
python -m tau3mu_gnn.bdt_training.BDT_ScoreProducer \
  --model <model_name> --classifier_iso <path/to/ISO_classifiers.pck> \
  --classifier_gnn <path/to/GNN_classifiers.pck> --tau_channel W --storage_folder <scratch_dir>
```

- An old variant uses `GNN_score` as BDT input. It is no longer the focus.

---

## Data Storage and Transfer

The data is on EOS. The files are copies of the T3 files (`/pnfs/psi.ch/cms/trivcat/store/user/...`):

| Data | On EOS |
|---|---|
| Raw W NanoAOD (Riccardo, signal + 8 ParkingDoubleMuonLowMass streams, 1.1 TB) | `/eos/cms/store/group/phys_bphys/kfuss/tau3mu_inputs/manzoni/tau3mu_gnn/` |
| Old HF skims (Kaito) | `/eos/user/k/kfuss/tau3mu_inputs/kfuss/tau3mu_gnn_round2/data/{signal,background}_HF/old_data/` |
| T3 pipeline results (`W_raw_Nano_AOD_fixed`, `HF_training_fixed`, `HF_analysis_fixed`; without the big `bdt_vs1` ROOT files) | `/eos/user/k/kfuss/tau3mu_pipeline/{analysis,training}/` |

> **Group space:** the e-group `cms-eos-pag-bphysics` may write in `/eos/cms/store/group/phys_bphys/`, but not delete. Put there only data that is copied once.

> **Your outputs** go to your EOS user space (1 TB). Delete old versions you do not need.

**Copy more files from T3** (scripts in `jobs/migration/`):

```bash
# on T3 (needs the /pnfs mount): sizes per group, file list
python3 jobs/migration/t3_list_files.py [group ...]
scp filelist.tsv <you>@lxplus.cern.ch:<folder on EOS>
# on lxplus (grid proxy to read T3, Kerberos to write EOS), in tmux:
voms-proxy-init --voms cms --valid 192:00
jobs/migration/copy_to_eos.sh filelist.tsv [group ...]
```

- `copy_to_eos.sh` reads from the public T3 door `t3se01.psi.ch` with xrootd 5.8.3 from CVMFS (the lxplus `xrdcp` 5.9.8 fails the T3 login).
- It checks each file (adler32), skips files that are done, and writes failed files to `copy_failed.tsv` (run it again with that list).

---

## Batch Systems

The pipelines write and submit the jobs; you do not write job scripts.

| | HTCondor (lxplus, default) | SLURM (PSI T3) |
|---|---|---|
| Config | `scheduler: htcondor` | `scheduler: slurm` |
| Resources | `htcondor.jobs` in the defaults files (`mem`, `time`, `cpus`; optional `gpus`, `requirements`, `disk`) | `slurm.jobs` (`partition`, `mem`, `time`, `cpus`; optional `account`, `gres`) |
| One submit | one DAG (`condor_submit_dag`); `all` = one DAG over all phases | one `sbatch` per job, with dependencies |
| Watch | `condor_q`, `condor_q -dag -nobatch`, `condor_history` | `squeue -u $USER`, `sacct -j <ID>` |
| Cancel | `condor_rm <DAG cluster>` (the number before `:` in `jobs.json`) | `scancel <IDs>` |
| Output storage | EOS (Kerberos) | dCache (VOMS proxy) |

> **Old job scripts:** `jobs/` still has the scripts of the old manual workflow (`jobs/condor/*.sub` for the old GPU trainings, `jobs/*/*.sh` for T3). The pipelines do not use them.

---

## Contact

kaito.fuss@gmail.com
