# import pandas
import uproot
import awkward as ak
import argparse
import pickle
import numpy  as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
# import matplotlib as mpl ; mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import ROOT
from array import array
# sns.set(style="whitegrid", font_scale=2)
sns.set_theme(style="white")


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name of GNN")
parser.add_argument("--classifier_gnn", type=str, required=True, help="Path to classifier GNN")
parser.add_argument("--classifier_iso", type=str, required=True, help="Path to classifier ISO")
parser.add_argument("--tau_channel", type=str, required=True, help="Tau prduction channel")
parser.add_argument('--storage_folder', type=str, required=True, help="Output to store the root files with the scores")
args = parser.parse_args()


model = args.model
tau_channel = args.tau_channel
storage_folder = args.storage_folder

model_path = "/work/kfuss/SVJ_GNN/output/" + model + "/"

prefix = "Selected_"

if tau_channel == "HF":
    signals = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_HF/" + model + "/selected_ds_large_signal_skimmed_no_resonances_with_scores.root"]
    backgrounds = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/selected_data_skimmed_no_resonances_part1_with_scores.root", 
                "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/selected_data_skimmed_no_resonances_part2_with_scores.root", 
                "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/selected_data_skimmed_no_resonances_part3_with_scores.root", 
                "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/selected_data_skimmed_no_resonances_part4_with_scores.root"]

elif tau_channel == "W":
    signals = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/" + model + "/selected_signal_skimmed_no_resonances_with_scores.root"]
    backgrounds = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/" + model + "/selected_data_skimmed_no_resonances_with_scores.root"]

classifier_file_gnn = args.classifier_gnn
classifier_file_iso = args.classifier_iso



##########################################################################################
tag = '24feb2026_v0'
##########################################################################################

if tau_channel == "HF":
    features = [
        f'Muon_pt[{prefix}TauTo3Mu_mu3_idx]',
                
        # 'Muon_mediumId[Slimmed_TauTo3Mu_mu1_idx]', 
        # 'Muon_mediumId[Slimmed_TauTo3Mu_mu2_idx]',
        # 'Muon_mediumId[Slimmed_TauTo3Mu_mu3_idx]',

        f'Muon_isGlobal[{prefix}TauTo3Mu_mu1_idx]', 
        f'Muon_isGlobal[{prefix}TauTo3Mu_mu2_idx]', 
        f'Muon_isGlobal[{prefix}TauTo3Mu_mu3_idx]', 

        f'{prefix}TauTo3Mu_dZmu12', 
        f'{prefix}TauTo3Mu_dZmu13', 
        f'{prefix}TauTo3Mu_dZmu23', 

        f'{prefix}TauTo3Mu_sigLxy_3muVtxBS', 
        f'{prefix}TauTo3Mu_CosAlpha2D_LxyP3mu', 
                
    ]

elif tau_channel == "W":
    features = [
        f'{prefix}TauTo3Mu_fitted_pt',
        'TauPlusMET_Tau_Puppi_mT',
        #'cand_refit_tau_dBetaIsoCone0p8strength0p2_rel',
        #'deltaPhi(TauPlusMET_Puppi_phi[0], Slimmed_TauTo3Mu_fitted_phi[0])',
        'TauPlusMET_PuppiMET_pt',
        #'Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)', # only for >= v4
    #     'cand_refit_tau_pt/cand_refit_met_pt', # only for >= v2
    #     'cand_refit_dRtauMuonMax',
        'TauPlusMET_Puppi_pt',
        'TauPlusMET_PuppiMETminPz',
        'TauPlusMET_PuppiMETmaxPz',
        f'{prefix}TauTo3Mu_dZmu12', 
        f'{prefix}TauTo3Mu_dZmu13', 
        f'{prefix}TauTo3Mu_dZmu23',
        f'{prefix}TauTo3Mu_sigLxy_3muVtxBS',
        f'{prefix}TauTo3Mu_vtx_chi2*({prefix}TauTo3Mu_vtx_Ndof**-1)',
        f'{prefix}TauTo3Mu_CosAlpha2D_LxyP3mu',
    ]


branches = features + [

    f'{prefix}TauTo3Mu_vtx_chi2', 
    f'{prefix}TauTo3Mu_vtx_Ndof',

    f'{prefix}TauTo3Mu_charge',
    f'{prefix}TauTo3Mu_fitted_eta',
    f'{prefix}TauTo3Mu_fitted_mass',
    'TauPlusMET_Puppi_phi',
    f'{prefix}TauTo3Mu_fitted_phi',
    'GNN_score',
    f'{prefix}TauTo3Mu_iso_ptChargedFromPV',
    f'{prefix}TauTo3Mu_iso_ptPhotons',
    f'{prefix}TauTo3Mu_iso_ptChargedFromPU',
    f'Muon_tkRelIso[{prefix}TauTo3Mu_mu1_idx]',
    f'Muon_tkRelIso[{prefix}TauTo3Mu_mu2_idx]',
    f'Muon_tkRelIso[{prefix}TauTo3Mu_mu3_idx]',
    # 'mu1_refit_muonid_soft', 'mu1_refit_muonid_loose', 'mu1_refit_muonid_medium', 'mu1_refit_muonid_tight',
    # 'mu2_refit_muonid_soft', 'mu2_refit_muonid_loose', 'mu2_refit_muonid_medium', 'mu2_refit_muonid_tight',
    # 'mu3_refit_muonid_soft', 'mu3_refit_muonid_loose', 'mu3_refit_muonid_medium', 'mu3_refit_muonid_tight',
    "run",
    "luminosityBlock",
    "event",
]

all_chunks_sig = []
print("Going to load signal events")
for events in uproot.iterate(f"{signals[0]}:Events", expressions=branches, library="ak", step_size="100 MB"):
    score_mask = events["GNN_score"] != -1
    events = events[score_mask]
    new_fields = {}
    for branch in events.fields:
        if events[branch].ndim > 1 and ak.any(ak.num(events[branch], axis=1) > 1):
            new_fields[branch] = ak.firsts(events[branch])
        else:
            new_fields[branch] = events[branch]

    events_masked = ak.zip(new_fields)
    all_chunks_sig.append(events_masked)
sig = ak.concatenate(all_chunks_sig)
# print(ak.type(sig))
sig = ak.to_dataframe(sig)
sig = sig.dropna()
print(sig.shape)
print(sig.columns)

all_chunks_bkg = []
print("Going to load bkg events")
for i in range(len(backgrounds)):
    for events in uproot.iterate(f"{backgrounds[i]}:Events", expressions=branches, library="ak", step_size="100 MB"):
        # bkg_mask = abs(events["Slimmed_TauTo3Mu_fitted_mass"] - 1.78) >= 0.00
        # events = events[ak.any(bkg_mask, axis=1)]
        score_mask = events["GNN_score"] != -1
        events_masked = events[score_mask]

        new_fields = {}
        for branch in events_masked.fields:
            if events_masked[branch].ndim > 1 and ak.any(ak.num(events_masked[branch], axis=1) > 1):
                new_fields[branch] = ak.firsts(events_masked[branch])
            else:
                new_fields[branch] = events_masked[branch]

        events_stripped = ak.zip(new_fields)
        df_chunk = ak.to_dataframe(events_masked)
        all_chunks_bkg.append(df_chunk)

bkg = pd.concat(all_chunks_bkg, ignore_index=True)
# print(ak.type(bkg))
nan_rows = bkg[bkg.isna().any(axis=1)]
bkg = bkg.dropna()
print(bkg.shape)
print(bkg.columns)

# sig = sig.sample(n=min(100_000, len(sig)), random_state=42).reset_index(drop=True)
# bkg = bkg.sample(n=min(100_000, len(bkg)), random_state=42).reset_index(drop=True)

# print(sig.shape)

##########################################################################################
## DEFINE TARGETS
##########################################################################################

bkg['target'] = np.zeros(bkg.shape[0], dtype=int)
sig['target'] = np.ones(sig.shape[0], dtype=int)


def deltaphi(phi1, phi2):
    delta_phi_internal = np.abs(phi1 - phi2)
    delta_phi_external = 2*np.pi - np.abs(phi1 - phi2)
    delta_phi = np.minimum(delta_phi_internal, delta_phi_external)

    return delta_phi

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



# features += ['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)']

# sig['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)'] = sig['Slimmed_TauTo3Mu_fitted_pt'] / sig['TauPlusMET_PuppiMET_pt']

# features += ['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)']


with open(classifier_file_gnn, "rb") as f:
    classifiers_gnn = pickle.load(f)

with open(classifier_file_iso, "rb") as f:
    classifiers_iso = pickle.load(f)

output = {}
for mode in ["GNN", "ISO"]:
    print(f"now mode {mode}")
    if mode == "GNN":
        bdt_features = features + ['GNN_score']
        classifiers = classifiers_gnn
    elif mode == "ISO":
        if tau_channel == "HF":
            bdt_features = features + [f'max(Muon_tkRelIso_{prefix}TauTo3Mu_mu1_idx Muon_tkRelIso_{prefix}TauTo3Mu_mu2_idx Muon_tkRelIso_{prefix}TauTo3Mu_mu3_idx)']
        elif tau_channel == "W":
            bdt_features = features + ['relative_iso']
        classifiers = classifiers_iso

    n_class = len(classifiers)
    bkg  ['bdt_' + mode] = np.zeros_like(bkg.Selected_TauTo3Mu_fitted_mass)
    sig  ['bdt_' + mode] = np.zeros_like(sig.Selected_TauTo3Mu_fitted_mass)


    for i, iclas in classifiers.items():
        print(f'evaluating the {i}-th classifier')
        bkg  ['bdt_fold%d_' %i + mode] = iclas.predict_proba(bkg[bdt_features])[:, 1]
        bkg  ['bdt_' + mode] += iclas.predict_proba(bkg[bdt_features])[:, 1] / n_class
        sig  ['bdt_fold%d_' %i + mode] = iclas.predict_proba(sig[bdt_features])[:, 1]
        sig  ['bdt_' + mode] += iclas.predict_proba(sig[bdt_features])[:, 1] / n_class

    if mode == "GNN":
        output["GNN_score_bkg"] = bkg.bdt_GNN
        output["GNN_score_sig"] = sig.bdt_GNN

    elif mode == "ISO":
        output["ISO_score_bkg"] = bkg.bdt_ISO
        output["ISO_score_sig"] = sig.bdt_ISO


    output["tau_mass"] = bkg.Selected_TauTo3Mu_fitted_mass
    
pkl_file = open(model_path + 'tau_score_corr.pkl', 'wb')
pickle.dump(output, pkl_file)
pkl_file.close()

for file in signals:
    print(f"Processing {file}")
    base_name = os.path.basename(file)   # only filename not full path      
    name_without_ext = os.path.splitext(base_name)[0] # drop .root
    model_storage = os.path.join(storage_folder, model)
    os.makedirs(model_storage, exist_ok=True)
    root_path_sc_only = os.path.join(model_storage, f"{name_without_ext}_with_scores_only.root")
    root_path = os.path.join(model_storage, f"{name_without_ext}_with_scores_bdt.root")

    runs = []
    lumis = []
    events = []
    scores_GNN = []
    scores_ISO = []

    for _, row in sig.iterrows():
        runs.append(row["run"])
        lumis.append(row["luminosityBlock"])
        events.append(row["event"])
        scores_GNN.append(row["bdt_GNN"])
        scores_ISO.append(row["bdt_ISO"])
    
    runs = np.array(runs, dtype=np.uint32)
    lumis = np.array(lumis, dtype=np.uint32)
    events = np.array(events, dtype=np.uint64)
    scores_GNN = np.array(scores_GNN, dtype=np.float64)
    scores_ISO = np.array(scores_ISO, dtype=np.float64)

    scores_to_save = {
        "run": runs,
        "luminosityBlock": lumis,
        "event": events,
        "bdt_GNN": scores_GNN,
        "bdt_ISO": scores_ISO,
    }

    branches = {k: v.dtype for k, v in scores_to_save.items()}

    print(f"Saving scores only to {root_path_sc_only}")

    with uproot.recreate(root_path_sc_only, compression=uproot.ZLIB(4)) as f:
        # Create a new TTree in the file
        tree = f.mktree("Events", branches)
        tree.extend(scores_to_save)  # Fill the tree with data

    print(f"Merging scores with events {root_path}")

    fin = ROOT.TFile.Open(file)
    t1 = fin.Get("Events")

    friend_file = ROOT.TFile.Open(root_path_sc_only)
    t2 = friend_file.Get("Events")

    t2.BuildIndex("run", "event")

    # Write merged tree
    fout = ROOT.TFile.Open(root_path, "RECREATE")
    t_merged = t1.CloneTree(0) # empty tree

    # Prepare buffer for GNN_score
    gnn_score = array('f', [0.])
    iso_score = array('f', [0.])
    
    t_merged.Branch("bdt_GNN", gnn_score, "bdt_GNN/F")
    t_merged.Branch("bdt_ISO", iso_score, "bdt_ISO/F")

    count = 0
    # Loop over t1
    for i in range(t1.GetEntries()):
        t1.GetEntry(i)

        # entry2 = t2.GetEntryNumberWithIndex(t1.run, t1.event)
        entry2 = t2.GetEntryNumberWithIndex(t1.run, t1.event)

        if entry2 >= 0:
            t2.GetEntry(entry2) 
            gnn_score[0] = t2.bdt_GNN
            iso_score[0] = t2.bdt_ISO
            count += 1
        else:
            gnn_score[0] = -1.  # default if no match
            iso_score[0] = -1.

        t_merged.Fill()

    print(f"Wrote in total {count} valid GNN Scores in to ROOT file")
    t_merged.Write()
    fout.Close()

    fin.Close()
    friend_file.Close()

    # print(f"ROOT file merging time: {end - start:.3f}")
    print(f"Merged score and event root files to {root_path}")
    
    os.remove(root_path_sc_only)

    print(f"Deleted temporary file {root_path_sc_only}")


for file in backgrounds:
    print(f"Processing {file}")
    base_name = os.path.basename(file)   # only filename not full path      
    name_without_ext = os.path.splitext(base_name)[0] # drop .root
    model_storage = os.path.join(storage_folder, model)
    os.makedirs(model_storage, exist_ok=True)
    root_path_sc_only = os.path.join(model_storage, f"{name_without_ext}_with_scores_only.root")
    root_path = os.path.join(model_storage, f"{name_without_ext}_bdt.root")

    runs = []
    lumis = []
    events = []
    scores_GNN = []
    scores_ISO = []

    for _, row in bkg.iterrows():
        runs.append(row["run"])
        lumis.append(row["luminosityBlock"])
        events.append(row["event"])
        scores_GNN.append(row["bdt_GNN"])
        scores_ISO.append(row["bdt_ISO"])
    
    runs = np.array(runs, dtype=np.uint32)
    lumis = np.array(lumis, dtype=np.uint32)
    events = np.array(events, dtype=np.uint64)
    scores_GNN = np.array(scores_GNN, dtype=np.float64)
    scores_ISO = np.array(scores_ISO, dtype=np.float64)

    scores_to_save = {
        "run": runs,
        "luminosityBlock": lumis,
        "event": events,
        "bdt_GNN": scores_GNN,
        "bdt_ISO": scores_ISO,
    }

    branches = {k: v.dtype for k, v in scores_to_save.items()}

    print(f"Saving scores only to {root_path_sc_only}")

    with uproot.recreate(root_path_sc_only, compression=uproot.ZLIB(4)) as f:
        # Create a new TTree in the file
        tree = f.mktree("Events", branches)
        tree.extend(scores_to_save)  # Fill the tree with data

    print(f"Merging scores with events {root_path}")

    fin = ROOT.TFile.Open(file)
    t1 = fin.Get("Events")

    friend_file = ROOT.TFile.Open(root_path_sc_only)
    t2 = friend_file.Get("Events")

    t2.BuildIndex("run", "event")

    # Write merged tree
    fout = ROOT.TFile.Open(root_path, "RECREATE")
    t_merged = t1.CloneTree(0) # empty tree

    # Prepare buffer for GNN_score
    gnn_score = array('f', [0.])
    iso_score = array('f', [0.])
    
    t_merged.Branch("bdt_GNN", gnn_score, "bdt_GNN/F")
    t_merged.Branch("bdt_ISO", iso_score, "bdt_ISO/F")

    count = 0
    # Loop over t1
    for i in range(t1.GetEntries()):
        t1.GetEntry(i)

        # entry2 = t2.GetEntryNumberWithIndex(t1.run, t1.event)
        entry2 = t2.GetEntryNumberWithIndex(t1.run, t1.event)

        if entry2 >= 0:
            t2.GetEntry(entry2) 
            gnn_score[0] = t2.bdt_GNN
            iso_score[0] = t2.bdt_ISO
            count += 1
        else:
            gnn_score[0] = -1.  # default if no match
            iso_score[0] = -1.

        t_merged.Fill()

    print(f"Wrote in total {count} valid GNN Scores in to ROOT file")
    t_merged.Write()
    fout.Close()

    fin.Close()
    friend_file.Close()

    # print(f"ROOT file merging time: {end - start:.3f}")
    print(f"Merged score and event root files to {root_path}")
    
    os.remove(root_path_sc_only)

    print(f"Deleted temporary file {root_path_sc_only}")