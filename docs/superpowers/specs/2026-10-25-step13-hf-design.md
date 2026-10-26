# Step 13 — HF Channel: Design

Status: agreed with Kaito in chat, 2026-10-25. Not implemented yet.
Parent document: `docs/wrapper_roadmap.md` (sec. 6 step 13). Where this spec differs from the roadmap, this spec wins and the roadmap gets updated (sec. 9).

## 1. Intent

Both pipelines (`tau3mu-train`, `tau3mu-pipeline`) support only the W channel today (`SUPPORTED_CHANNELS = ("W",)` in `pipelines/common/config.py`). Step 13 adds the HF channel (Ds → τ → 3μ signal, same `ParkingDoubleMuonLowMass` data).

There is **no trusted HF GNN result**. Kaito never ran the HF GNN with the BDT preselection approach. The old HF GNN numbers in `output/HF_limits.json` are not reliable. The only HF reference is the BDT-only benchmark: `benchmark_HF_BDT_vs1`, expected limit **1.0703 at cut 0.91** (`output/HF_limits.json`).

So step 13 does **not** reproduce an old HF GNN limit. It trains new HF models with the pipeline and makes the first HF GNN limit.

Step 13 runs after 13b/13c in the roadmap order, but it does not depend on them. Kaito starts it now, in parallel with the 13b W run.

## 2. Two runs

**13-HF-a: old skimmed files** (like step 12i for W).
- Inputs: the old HF resonance-skimmed files, `already_resonance_skimmed: true`:
  - signal: `/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_HF/old_data/ds_large_signal_skimmed_no_resonances.root` (500,000 entries)
  - background: `/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/old_data/data_skimmed_no_resonances_part{1,2,3,4}.root` (1,447,376 entries in total)
- `tau3mu-train all` → new HF BDT, new HF GNN, BDT-only benchmark cut scan.
- `tau3mu-pipeline all` with `models:` = the new HF `gnn_vsN` → HF GNN limit cut scan.
- Signal weight: `0.000152` (the old HF value). Its origin is not known (no code in the repo computes it). 13-HF-a uses it as it is.

Finding (2026-10-25, scratchpad scripts `hf_subset.py`, `hf_stream.py`): the old HF background is **not** a cut selection. It is a part of the old W data file (10,855,317 entries):
- All HF events (run, lumi, event) are in the W file. Same run range 378985–386951 (2024).
- Every cut fraction is the same in HF and W: 2 μ pT > 3 + 3rd tracker μ pT > 1.2 (98.98% both), 1.60 < m < 2.02 (18.7% both), τ window (92.87% both), the `HLT_*` bits (±0.1%). So the thesis cuts (old 2017/2018 text) do not make the subset, and neither file has the mass cut or the τ window applied.
- 4 raw Run2024C files per parking stream, fraction of the W events that is also in HF: streams 0 and 4: 75%, stream 5: 17%, stream 7: 1%, streams 1, 2, 3, 6: 0%. So HF = W data for only some streams / files (guess: HF skim jobs ran on fewer files, or some failed).
- Consequence: `0.000152` is correct only if it was computed for this smaller luminosity. Kaito clears where the data and the weight come from before 13-HF-b.

**13-HF-b: raw NanoAOD** (like step 13b for W). Starts after 13-HF-a looks good. **On hold:** Kaito first clears the data source and the signal weight (see the finding above). Not planned in detail now.
- Signal: Riccardo's Summer24 Ds sample, same production tag as the W raw signal:
  `/pnfs/psi.ch/cms/trivcat/store/user/manzoni/tau3mu_gnn/DstoTau-Tauto3Mu_Fil-3Mu_TuneCP5_13p6TeV_pythia8-evtgen/05may25v2__DstoTau-Tauto3Mu_Fil-3Mu_TuneCP5_13p6TeV_pythia8-evtgen_RunIII2024Summer24MiniAOD-140X_mcRun3_2024_realistic_v26-v2_MINIAODSIM/*/*/*.root` (4432 files, 317 GB). Kaito confirms the sample before the run. (The other Ds folders there are older Winter24 productions or other versions.)
- Background: the same raw data glob as `configs/example_W_raw.yaml`.
- `already_resonance_skimmed: false`, then the same two commands.
- Signal weight: must be computed again for the new skim (same open item as W in step 13b).

## 3. Goals, non-goals, anti-goals

Goals
- `channel: HF` works in both pipelines with the same commands, folders, manifests and checks as W.
- HF uses the W frozen recipe everywhere, except where HF physics needs something else (sec. 5).
- 13-HF-a gives: an HF BDT benchmark limit roughly equal to 1.07 (Kaito judges "roughly"; same input files and recipe as the old benchmark, but a new training), and the first HF GNN limit.

Non-goals
- No frozen HF models now. `analysis_defaults_HF.yaml` has placeholder frozen paths under `models/HF/` (sec. 6). After 13-HF-b, Kaito chooses a training and copies it into `models/HF/`. That is a later step.
- No new code to make `models:` required for HF. Without `models:`, the existing check stops the run at load time ("frozen model file not found").
- No HF hyper-parameter search. No change to the W channel behaviour.
- No combined W + HF limit.
- No HTCondor / lxplus (step 14).

Anti-goals
- A W run gives a different result after this step (W outputs must stay identical).
- An HF BDT feature list that differs between training and inference.
- An HF job runs for hours and then fails on a missing branch that a start check could find.

## 4. Code changes

1. `pipelines/common/config.py`: `SUPPORTED_CHANNELS = ("W", "HF")`.
2. `pipelines/common/branches.py`: `PER_CHANNEL["HF"]` and `TRAINING_EXTRA["HF"] = ("PV_npvs",)`. The HF list holds every raw branch that the HF BDT and the HF GNN read, in analysis names (`Slimmed_*` before stage 1): `Slimmed_TauTo3Mu_{sigLxy_3muVtxBS, vtx_chi2, vtx_Ndof, charge, mu1_idx, mu2_idx, mu3_idx}`, `Muon_pt`, `Muon_isGlobal`, `Muon_tkRelIso`, and the iso branches that `bdt_branches` loads for all channels (`iso_ptChargedFromPV`, `iso_ptPhotons`, `iso_ptChargedFromPU`). The builder derives the exact list from `bdt_branches("HF")` and `gnn_branches("HF")`. `raw_required_branches("HF")` must also work: `Muon_*` has no `Slimmed_` prefix, so the prefix strip leaves it unchanged; check it.
3. `bdt_training/bdt_inputs.py` — **bug fix**. `vtx_chi2*(vtx_Ndof**-1)` is in the HF feature list (Kaito added it in commit `74c94b6`) **and** `BDT_ScoreProducer._add_features` appends it again for HF. Result today: 12 features with one duplicate. The old HF classifier (`output/benchmark_HF_BDT_vs1/ISO_classifiers_24feb2026_v0.pck`) has 11 unique features. Fix (builder's call where): one copy only. Add a test: HF BDT features have no duplicates and, as a set, equal the 11 feature names of the old HF classifier.
4. `bdt_training/bdt_inputs.py` — drop `TauPlusMET_Puppi_phi` from the HF load (my call, Kaito agreed in chat). HF does not use it, and `dropna` on loaded columns can remove events because of it. W is unchanged.
5. New config files:
   - `configs/analysis_defaults_HF.yaml`: copy of the W file. Same `resonance_skim`, `skim`, `limit`, `slurm` values. `bdt` / `gnn` keys: placeholder paths `models/HF/bdt/...`, `models/HF/gnn/...` (files do not exist), with the same model settings as W (`conv_type: GINE`, `gnn_mode: final_layer`, `unsculpting_mode: delete_muons`, `graph_type: skimmed_dzd0_cone_size_0.5_knn_10`, `bdt_preselection_cut: 0.8`). A comment says: "no frozen HF models yet; set `models:` in the user config".
   - `configs/training_defaults_HF.yaml`: copy of the W file, all values the same (sec. 5).
   - `configs/example_training_HF.yaml` and `configs/example_HF.yaml`: 13-HF-a inputs (sec. 2), `signal_weight: 0.000152`, the HF cut scans (sec. 5).
   - Raw examples (`example_HF_raw.yaml`, training equivalent) come with 13-HF-b.
6. Tests, as for W: HF BDT branch list ⊆ `required_branches("HF")`, HF GNN branch list ⊆ `required_branches("HF")`, config load of the HF examples, a training chain test on toy HF data if the W chain test can take a channel parameter.
7. README: one short section "HF channel" (what differs from W, no frozen models, `models:` needed).

What already works for HF (checked 2026-10-25, no change needed):
- Multi-file inputs: `inputs.signal` / `background` take lists and globs; stage 1 merges all files of a sample into one `<sample>.root`.
- Stage 0 keeps all non-`TauTo3Mu_*` / non-`PFCands_*` columns as they are, so `Muon_*` and `mu*_idx` stay correct for HF.
- The old HF files have the HF branches (`Muon_tkRelIso`, `Muon_isGlobal`, `Muon_pt`, `Slimmed_TauTo3Mu_mu*_idx`, `Slimmed_TauTo3Mu_charge`, `PV_npvs`, `Slimmed_PFCands_d0/dz`).
- Graph and score code have HF paths (`ConeGraphProducer`, `ScoreProducer`; HF has no derived graph-level feature). `gnn_inputs.graph_level_branches("HF")` = the old `GraphCreation.py` HF list (10 features; max tkRelIso is a BDT feature only).
- `stages/graphs.py` names HF graphs `HF_decay_*`.
- The datacard code has no channel-specific part (only the old CLI's `OLD_SIGNAL_WEIGHT`).

## 5. HF settings vs W

| Setting | HF value | Why |
|---|---|---|
| Resonance skim, skim, limit fit (`tau_mass`, `mass_window`, `fit_range`, `cl`) | same as W | same τ mass, same data |
| BDT hyper-parameters, sideband, sample sizes, seeds | same as W | the old `Train_XGB_HF.py` uses the same values as `Train_XGB_W.py` |
| BDT features | the HF list (11): μ3 pT, 3× isGlobal, 3× dZ, sigLxy, cos α, χ²/ndof, max tkRelIso | old `Train_XGB_HF.py` |
| GNN graph-level features | the HF list (10): the BDT list without max tkRelIso | old `GraphCreation.py` |
| Graphs, GNN training (knn 10, cone 0.5, dzd0, `delete_muons`, GINE, `final_layer`, `fl8_sd4`, lr, batch, disco 0, patience) | same as W | Kaito: use the W frozen recipe. The old HF graphs used `split_pt_uniformly`; not taken. |
| `graphs.bdt_preselection_cut` | 0.8 (as W) | Kaito's default; revisit after 13-HF-a if the HF BDT score shape needs it |
| npvs bins | same as W | same data, Summer24 MC |
| `benchmark.bdt_score_cut` | scan 0.88 – 0.97 (my call) | old HF best cut 0.91; W's 0.98–0.996 would miss it |
| `gnn_score_cut` (analysis) | wide scan (my call; builder picks the grid, e.g. 2.0 – 8.0 step 0.5) | no HF reference |
| `limit.signal_weight` | 0.000152 (13-HF-a) | old HF value, full HF dataset (Kaito) |
| SLURM resources | same as W | raise after the first HF run (`sacct` MaxRSS / elapsed) |

## 6. Frozen models

There are no frozen HF models. The analysis loader checks the frozen files only when `models` is `frozen` (`pipelines/analysis/config.py:118-129`). So:
- `channel: HF` without `models:` → `ConfigError: frozen model file not found: …/models/HF/…`. Good enough (Kaito).
- `channel: HF` with `models: <HF gnn_vsN>` → works. The `models:` key already refuses an analysis folder and an unfinished training. The builder checks that it also refuses a **W** training for an HF analysis (the manifest has `channel`); if not, add that check.

## 7. Checks

- W regression: the W unit tests pass; W configs load as before; a W dry run (`--dry-run`) gives the same job scripts as before.
- HF unit tests (sec. 4 item 6).
- HF dry runs of `tau3mu-train all` and `tau3mu-pipeline all` with the example configs.
- Real run 13-HF-a (Kaito queues the SLURM jobs): HF benchmark ≈ 1.07; first HF GNN limit; Kaito judges.

## 8. Open items / later

- 13-HF-b (on hold, Kaito): where the old HF data comes from; how `0.000152` was computed (luminosity, cross section, generated events); then the HF signal weight for the full raw data and the new skim; confirm the Ds raw sample; measure the skim job on 4432 raw signal files (W measure: ~5 s/file). The raw background is the full W data (~7.5× the old HF data), so HF jobs need W-size resources.
- Step 13c fixes also touch HF: E3, E4 (datacard) and `pair_charge` (graphs) are channel-independent. E1 (`TauPlusMET` mix) is W only.
- Frozen HF models in `models/HF/` after a chosen training.

## 9. Roadmap update

Step 13 in `docs/wrapper_roadmap.md`: point to this spec; replace "HF signal weight" with the two runs (sec. 2); note "no frozen HF models yet".

## Calls made by Claude (Kaito: check these)

- Fix of the duplicate χ²/ndof feature: one copy (where: builder).
- Drop `TauPlusMET_Puppi_phi` from the HF BDT load.
- HF benchmark scan 0.88 – 0.97; wide GNN cut scan.
- The Summer24 `05may25v2` Ds sample for 13-HF-b.
