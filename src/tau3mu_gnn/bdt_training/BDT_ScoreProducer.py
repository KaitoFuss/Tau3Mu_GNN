"""ISO BDT scores for the W / HF channel (stage 2).

score_events() reads the input files, scores the first tau candidate of each event with the
frozen ISO classifier, and writes for each input file a copy with the branch bdt_ISO
(-1 for events without a score). It also writes tau_score_corr.pkl for the model evaluation.
"""
import argparse
import os
import pickle
from array import array
from pathlib import Path
from typing import Mapping

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from tau3mu_gnn.bdt_training.bdt_inputs import bdt_branches

PREFIX = "Selected_"


def deltaphi(phi1, phi2):
    delta_phi_internal = np.abs(phi1 - phi2)
    delta_phi_external = 2*np.pi - np.abs(phi1 - phi2)
    delta_phi = np.minimum(delta_phi_internal, delta_phi_external)

    return delta_phi


def _load_signal(files, tree_name, branches):
    all_chunks_sig = []
    print("Going to load signal events")
    for file in files:
        for events in uproot.iterate(f"{file}:{tree_name}", expressions=branches, library="ak", step_size="100 MB"):
            new_fields = {}
            for branch in events.fields:
                if events[branch].ndim > 1 and ak.any(ak.num(events[branch], axis=1) > 1):
                    new_fields[branch] = ak.firsts(events[branch])
                else:
                    new_fields[branch] = events[branch]

            events_masked = ak.zip(new_fields)
            all_chunks_sig.append(events_masked)
    sig = ak.concatenate(all_chunks_sig)
    sig = ak.to_dataframe(sig)
    sig = sig.dropna()
    print(sig.shape)
    print(sig.columns)
    return sig


def _load_background(files, tree_name, branches):
    all_chunks_bkg = []
    print("Going to load bkg events")
    for file in files:
        for events in uproot.iterate(f"{file}:{tree_name}", expressions=branches, library="ak", step_size="100 MB"):
            new_fields = {}
            for branch in events.fields:
                if events[branch].ndim > 1 and ak.any(ak.num(events[branch], axis=1) > 1):
                    new_fields[branch] = ak.firsts(events[branch])
                else:
                    new_fields[branch] = events[branch]

            events_stripped = ak.zip(new_fields)
            # one row per event (first candidate), so the (run, event) index in _write_with_scores is unique
            df_chunk = ak.to_dataframe(events_stripped)
            all_chunks_bkg.append(df_chunk)

    bkg = pd.concat(all_chunks_bkg, ignore_index=True)
    bkg = bkg.dropna()
    print(bkg.shape)
    print(bkg.columns)
    return bkg


def _add_features(sig, bkg, features, tau_channel, prefix):
    """Add the derived feature columns to sig and bkg. Return the BDT feature list."""
    if tau_channel == "HF":
        sig[f'Muon_pt_{prefix}TauTo3Mu_mu3_idx'] = sig[f'Muon_pt[{prefix}TauTo3Mu_mu3_idx]']
        bkg[f'Muon_pt_{prefix}TauTo3Mu_mu3_idx'] = bkg[f'Muon_pt[{prefix}TauTo3Mu_mu3_idx]']

        features += [f'Muon_pt_{prefix}TauTo3Mu_mu3_idx']
        features.remove(f'Muon_pt[{prefix}TauTo3Mu_mu3_idx]')

        # sig['Muon_mediumId_Slimmed_TauTo3Mu_mu1_idx'] = sig['Muon_mediumId[Slimmed_TauTo3Mu_mu1_idx]']

        # features += ['Muon_mediumId_Slimmed_TauTo3Mu_mu1_idx']
        # features.remove('Muon_mediumId[Slimmed_TauTo3Mu_mu1_idx]')

        # sig['Muon_mediumId_Slimmed_TauTo3Mu_mu2_idx'] = sig['Muon_mediumId[Slimmed_TauTo3Mu_mu2_idx]']

        # features += ['Muon_mediumId_Slimmed_TauTo3Mu_mu2_idx']
        # features.remove('Muon_mediumId[Slimmed_TauTo3Mu_mu2_idx]')

        # sig['Muon_mediumId_Slimmed_TauTo3Mu_mu3_idx'] = sig['Muon_mediumId[Slimmed_TauTo3Mu_mu3_idx]']

        # features += ['Muon_mediumId_Slimmed_TauTo3Mu_mu3_idx']
        # features.remove('Muon_mediumId[Slimmed_TauTo3Mu_mu3_idx]')

        sig[f'Muon_isGlobal_{prefix}TauTo3Mu_mu1_idx'] = sig[f'Muon_isGlobal[{prefix}TauTo3Mu_mu1_idx]']
        bkg[f'Muon_isGlobal_{prefix}TauTo3Mu_mu1_idx'] = bkg[f'Muon_isGlobal[{prefix}TauTo3Mu_mu1_idx]']

        features += [f'Muon_isGlobal_{prefix}TauTo3Mu_mu1_idx']
        features.remove(f'Muon_isGlobal[{prefix}TauTo3Mu_mu1_idx]')

        sig[f'Muon_isGlobal_{prefix}TauTo3Mu_mu2_idx'] = sig[f'Muon_isGlobal[{prefix}TauTo3Mu_mu2_idx]']
        bkg[f'Muon_isGlobal_{prefix}TauTo3Mu_mu2_idx'] = bkg[f'Muon_isGlobal[{prefix}TauTo3Mu_mu2_idx]']

        features += [f'Muon_isGlobal_{prefix}TauTo3Mu_mu2_idx']
        features.remove(f'Muon_isGlobal[{prefix}TauTo3Mu_mu2_idx]')

        sig[f'Muon_isGlobal_{prefix}TauTo3Mu_mu3_idx'] = sig[f'Muon_isGlobal[{prefix}TauTo3Mu_mu3_idx]']
        bkg[f'Muon_isGlobal_{prefix}TauTo3Mu_mu3_idx'] = bkg[f'Muon_isGlobal[{prefix}TauTo3Mu_mu3_idx]']

        features += [f'Muon_isGlobal_{prefix}TauTo3Mu_mu3_idx']
        features.remove(f'Muon_isGlobal[{prefix}TauTo3Mu_mu3_idx]')

        sig[f'{prefix}TauTo3Mu_vtx_chi2*({prefix}TauTo3Mu_vtx_Ndof**-1)'] = sig[f'{prefix}TauTo3Mu_vtx_chi2']/sig[f'{prefix}TauTo3Mu_vtx_Ndof']
        bkg[f'{prefix}TauTo3Mu_vtx_chi2*({prefix}TauTo3Mu_vtx_Ndof**-1)'] = bkg[f'{prefix}TauTo3Mu_vtx_chi2']/bkg[f'{prefix}TauTo3Mu_vtx_Ndof']

        features += [f'{prefix}TauTo3Mu_vtx_chi2*({prefix}TauTo3Mu_vtx_Ndof**-1)']

        sig[f'max(Muon_tkRelIso_{prefix}TauTo3Mu_mu1_idx Muon_tkRelIso_{prefix}TauTo3Mu_mu2_idx Muon_tkRelIso_{prefix}TauTo3Mu_mu3_idx)'] = np.max(
            np.stack([
                sig[f'Muon_tkRelIso[{prefix}TauTo3Mu_mu1_idx]'].values,
                sig[f'Muon_tkRelIso[{prefix}TauTo3Mu_mu2_idx]'].values,
                sig[f'Muon_tkRelIso[{prefix}TauTo3Mu_mu3_idx]'].values
            ], axis=0),
            axis=0
        )

        bkg[f'max(Muon_tkRelIso_{prefix}TauTo3Mu_mu1_idx Muon_tkRelIso_{prefix}TauTo3Mu_mu2_idx Muon_tkRelIso_{prefix}TauTo3Mu_mu3_idx)'] = np.max(
            np.stack([
                bkg[f'Muon_tkRelIso[{prefix}TauTo3Mu_mu1_idx]'].values,
                bkg[f'Muon_tkRelIso[{prefix}TauTo3Mu_mu2_idx]'].values,
                bkg[f'Muon_tkRelIso[{prefix}TauTo3Mu_mu3_idx]'].values
            ], axis=0),
            axis=0
        )

    elif tau_channel == "W":
        bkg[f'deltaPhi(TauPlusMET_Puppi_phi, {prefix}TauTo3Mu_fitted_phi)'] = deltaphi(bkg['TauPlusMET_Puppi_phi'], bkg[f'{prefix}TauTo3Mu_fitted_phi'])
        sig[f'deltaPhi(TauPlusMET_Puppi_phi, {prefix}TauTo3Mu_fitted_phi)'] = deltaphi(sig['TauPlusMET_Puppi_phi'], sig[f'{prefix}TauTo3Mu_fitted_phi'])

        features += [f'deltaPhi(TauPlusMET_Puppi_phi, {prefix}TauTo3Mu_fitted_phi)']

        # bkg['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)'] = bkg['Slimmed_TauTo3Mu_fitted_pt'] / bkg['TauPlusMET_PuppiMET_pt']

        # features += ['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)']

        abs_iso_bkg = bkg[f"{prefix}TauTo3Mu_iso_ptChargedFromPV"] + np.maximum(0., bkg[f"{prefix}TauTo3Mu_iso_ptPhotons"] - 0.2 * bkg[f"{prefix}TauTo3Mu_iso_ptChargedFromPU"])
        bkg['relative_iso'] = abs_iso_bkg / bkg[f"{prefix}TauTo3Mu_fitted_pt"]

        abs_iso_sig = sig[f"{prefix}TauTo3Mu_iso_ptChargedFromPV"] + np.maximum(0., sig[f"{prefix}TauTo3Mu_iso_ptPhotons"] - 0.2 * sig[f"{prefix}TauTo3Mu_iso_ptChargedFromPU"])
        sig['relative_iso'] = abs_iso_sig / sig[f"{prefix}TauTo3Mu_fitted_pt"]

    if tau_channel == "HF":
        return features + [f'max(Muon_tkRelIso_{prefix}TauTo3Mu_mu1_idx Muon_tkRelIso_{prefix}TauTo3Mu_mu2_idx Muon_tkRelIso_{prefix}TauTo3Mu_mu3_idx)']
    return features + ['relative_iso']


def _write_with_scores(df, in_file, out_file, tree_name):
    """Copy in_file to out_file (without GNN_score) and add bdt_ISO, matched by (run, event)."""
    import ROOT

    print(f"Processing {in_file}")
    root_path = str(out_file)
    root_path_sc_only = root_path.removesuffix(".root") + "_scores_only.root"

    runs = []
    lumis = []
    events = []
    scores_ISO = []

    for _, row in df.iterrows():
        runs.append(row["run"])
        lumis.append(row["luminosityBlock"])
        events.append(row["event"])
        scores_ISO.append(row["bdt_ISO"])

    runs = np.array(runs, dtype=np.uint32)
    lumis = np.array(lumis, dtype=np.uint32)
    events = np.array(events, dtype=np.uint64)
    scores_ISO = np.array(scores_ISO, dtype=np.float64)

    scores_to_save = {
        "run": runs,
        "luminosityBlock": lumis,
        "event": events,
        "bdt_ISO": scores_ISO,
    }

    branches = {k: v.dtype for k, v in scores_to_save.items()}

    print(f"Saving scores only to {root_path_sc_only}")

    with uproot.recreate(root_path_sc_only, compression=uproot.ZLIB(4)) as f:
        # Create a new TTree in the file
        tree = f.mktree(tree_name, branches)
        tree.extend(scores_to_save)  # Fill the tree with data

    print(f"Merging scores with events {root_path}")

    fin = ROOT.TFile.Open(in_file)
    t1 = fin.Get(tree_name)
    has_gnn_score = bool(t1.GetBranch("GNN_score"))  # old inputs only; kept out of the output
    if has_gnn_score:
        t1.SetBranchStatus("GNN_score", 0)

    friend_file = ROOT.TFile.Open(root_path_sc_only)
    t2 = friend_file.Get(tree_name)

    t2.BuildIndex("run", "event")

    # Write merged tree
    fout = ROOT.TFile.Open(root_path, "RECREATE")
    t_merged = t1.CloneTree(0) # empty tree
    if has_gnn_score:
        t1.SetBranchStatus("GNN_score", 1)

    # Prepare buffer for bdt_ISO
    iso_score = array('f', [0.])

    t_merged.Branch("bdt_ISO", iso_score, "bdt_ISO/F")

    count = 0
    # Loop over t1
    for i in range(t1.GetEntries()):
        t1.GetEntry(i)

        # entry2 = t2.GetEntryNumberWithIndex(t1.run, t1.event)
        entry2 = t2.GetEntryNumberWithIndex(t1.run, t1.event)

        if entry2 >= 0:
            t2.GetEntry(entry2) 
            iso_score[0] = t2.bdt_ISO
            count += 1
        else:
            iso_score[0] = -1.

        t_merged.Fill()

    print(f"Wrote in total {count} valid BDT scores in to ROOT file")
    t_merged.Write()
    fout.Close()

    fin.Close()
    friend_file.Close()

    # print(f"ROOT file merging time: {end - start:.3f}")
    print(f"Merged score and event root files to {root_path}")

    os.remove(root_path_sc_only)

    print(f"Deleted temporary file {root_path_sc_only}")


def score_events(signal: Mapping[str, str], background: Mapping[str, str], classifier: str | Path,
                 tau_channel: str, pickle_path: str | Path, tree_name: str = "Events") -> None:
    """signal / background map each input file to its output file.

    Writes the output files and pickle_path (keys ISO_score_sig, ISO_score_bkg, tau_mass).
    """
    prefix = PREFIX
    features, branches = bdt_branches(tau_channel, prefix)
    sig = _load_signal(list(signal), tree_name, branches)
    bkg = _load_background(list(background), tree_name, branches)
    bdt_features = _add_features(sig, bkg, features, tau_channel, prefix)

    with open(classifier, "rb") as f:
        classifiers = pickle.load(f)

    mode = "ISO"
    for col in bdt_features:
        if sig[col].dtype == object:
            sig[col] = sig[col].astype(int)
            bkg[col] = bkg[col].astype(int)

    n_class = len(classifiers)
    bkg  ['bdt_' + mode] = np.zeros_like(bkg.Selected_TauTo3Mu_fitted_mass)
    sig  ['bdt_' + mode] = np.zeros_like(sig.Selected_TauTo3Mu_fitted_mass)

    for i, iclas in classifiers.items():
        print(f'evaluating the {i}-th classifier')
        bkg  ['bdt_' + mode] += iclas.predict_proba(bkg[bdt_features])[:, 1] / n_class
        sig  ['bdt_' + mode] += iclas.predict_proba(sig[bdt_features])[:, 1] / n_class

    output = {
        "ISO_score_bkg": bkg.bdt_ISO,
        "ISO_score_sig": sig.bdt_ISO,
        "tau_mass": bkg.Selected_TauTo3Mu_fitted_mass,
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(output, f)

    for in_file, out_file in signal.items():
        _write_with_scores(sig, in_file, out_file, tree_name)
    for in_file, out_file in background.items():
        _write_with_scores(bkg, in_file, out_file, tree_name)


def _old_inputs(model, tau_channel):
    if tau_channel == "HF":
        signals = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_HF/" + model + "/selected_ds_large_signal_skimmed_no_resonances_with_scores.root"]
        backgrounds = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/selected_data_skimmed_no_resonances_part1_with_scores.root", 
                    "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/selected_data_skimmed_no_resonances_part2_with_scores.root", 
                    "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/selected_data_skimmed_no_resonances_part3_with_scores.root", 
                    "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/selected_data_skimmed_no_resonances_part4_with_scores.root"]

    elif tau_channel == "W":
        signals = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/" + model + "/selected_signal_skimmed_no_resonances_with_scores.root"]
        backgrounds = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/" + model + "/selected_data_skimmed_no_resonances_with_scores.root"]
    return signals, backgrounds


def main(argv=None):
    """Old entry point: hard-coded input files per model, outputs as before."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name of GNN")
    parser.add_argument("--classifier_iso", type=str, required=True, help="Path to classifier ISO")
    parser.add_argument("--tau_channel", type=str, required=True, help="Tau prduction channel")
    parser.add_argument('--storage_folder', type=str, required=True, help="Output to store the root files with the scores")
    args = parser.parse_args(argv)

    signals, backgrounds = _old_inputs(args.model, args.tau_channel)
    model_storage = os.path.join(args.storage_folder, args.model)
    os.makedirs(model_storage, exist_ok=True)

    def out_name(file):
        name_without_ext = os.path.splitext(os.path.basename(file))[0]
        return os.path.join(model_storage, f"{name_without_ext}_bdt_full.root")

    model_path = "/work/kfuss/SVJ_GNN/output/" + args.model + "/"
    score_events({signals[0]: out_name(signals[0])}, {f: out_name(f) for f in backgrounds},
                 args.classifier_iso, args.tau_channel, model_path + "tau_score_corr.pkl")


if __name__ == "__main__":
    main()
