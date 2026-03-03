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
# sns.set(style="whitegrid", font_scale=2)
sns.set_theme(style="white")

import xgboost
from xgboost import XGBClassifier, plot_importance

from sklearn.metrics         import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from scipy.stats import ks_2samp

from collections import OrderedDict, defaultdict
from itertools import product


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name of GNN")
parser.add_argument("--classifier_gnn", type=str, required=True, help="Path to classifier GNN")
parser.add_argument("--classifier_iso", type=str, required=True, help="Path to classifier ISO")
parser.add_argument("--tau_channel", type=str, required=True, help="Tau prduction channel")
args = parser.parse_args()


model = args.model
tau_channel = args.tau_channel

model_path = "/work/kfuss/SVJ_GNN/output/" + model + "/"

if tau_channel == "HF":
    signals = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_HF/" + model + "/ds_large_signal_skimmed_no_resonances_with_scores.root"]
    backgrounds = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/data_skimmed_no_resonances_part1_with_scores.root", 
                "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/data_skimmed_no_resonances_part2_with_scores.root", 
                "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/data_skimmed_no_resonances_part3_with_scores.root", 
                "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/" + model + "/data_skimmed_no_resonances_part4_with_scores.root"]

elif tau_channel == "W":
    signals = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/" + model + "/signal_skimmed_no_resonances_with_scores.root"]
    backgrounds = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/" + model + "/data_skimmed_no_resonances_with_scores.root"]

classifier_file_gnn = args.classifier_gnn
classifier_file_iso = args.classifier_iso



##########################################################################################
tag = '24feb2026_v0'
##########################################################################################

if tau_channel == "HF":
    features = [
        'Muon_pt[Slimmed_TauTo3Mu_mu3_idx]',
                
        # 'Muon_mediumId[Slimmed_TauTo3Mu_mu1_idx]', 
        # 'Muon_mediumId[Slimmed_TauTo3Mu_mu2_idx]',
        # 'Muon_mediumId[Slimmed_TauTo3Mu_mu3_idx]',

        'Muon_isGlobal[Slimmed_TauTo3Mu_mu1_idx]', 
        'Muon_isGlobal[Slimmed_TauTo3Mu_mu2_idx]', 
        'Muon_isGlobal[Slimmed_TauTo3Mu_mu3_idx]', 

        'Slimmed_TauTo3Mu_dZmu12', 
        'Slimmed_TauTo3Mu_dZmu13', 
        'Slimmed_TauTo3Mu_dZmu23', 

        'Slimmed_TauTo3Mu_sigLxy_3muVtxBS', 
        'Slimmed_TauTo3Mu_CosAlpha2D_LxyP3mu', 
                
    ]

elif tau_channel == "W":
    features = [
        'Slimmed_TauTo3Mu_fitted_pt',
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
        'Slimmed_TauTo3Mu_dZmu12', 
        'Slimmed_TauTo3Mu_dZmu13', 
        'Slimmed_TauTo3Mu_dZmu23',
        'Slimmed_TauTo3Mu_sigLxy_3muVtxBS',
        'Slimmed_TauTo3Mu_vtx_chi2*(Slimmed_TauTo3Mu_vtx_Ndof**-1)',
        'Slimmed_TauTo3Mu_CosAlpha2D_LxyP3mu',
    ]


branches = features + [

    'Slimmed_TauTo3Mu_vtx_chi2', 
    'Slimmed_TauTo3Mu_vtx_Ndof',

    'Slimmed_TauTo3Mu_charge',
    'Slimmed_TauTo3Mu_fitted_eta',
    'Slimmed_TauTo3Mu_fitted_mass',
    'TauPlusMET_Puppi_phi',
    'Slimmed_TauTo3Mu_fitted_phi',
    'GNN_score',
    'Slimmed_TauTo3Mu_iso_ptChargedFromPV',
    'Slimmed_TauTo3Mu_iso_ptPhotons',
    'Slimmed_TauTo3Mu_iso_ptChargedFromPU',
    'Muon_tkRelIso[Slimmed_TauTo3Mu_mu1_idx]',
    'Muon_tkRelIso[Slimmed_TauTo3Mu_mu2_idx]',
    'Muon_tkRelIso[Slimmed_TauTo3Mu_mu3_idx]',
    # 'mu1_refit_muonid_soft', 'mu1_refit_muonid_loose', 'mu1_refit_muonid_medium', 'mu1_refit_muonid_tight',
    # 'mu2_refit_muonid_soft', 'mu2_refit_muonid_loose', 'mu2_refit_muonid_medium', 'mu2_refit_muonid_tight',
    # 'mu3_refit_muonid_soft', 'mu3_refit_muonid_loose', 'mu3_refit_muonid_medium', 'mu3_refit_muonid_tight',
]

# all_chunks_sig = []

# print("Going to load signal events")
# for events in uproot.iterate(f"{signals[0]}:Events", expressions=branches, library="ak", step_size="100 MB"):
#     score_mask = events["GNN_Score"] != -1
#     events = events[score_mask]
#     new_fields = {}
#     for branch in events.fields:
#         if events[branch].ndim > 1 and ak.any(ak.num(events[branch], axis=1) > 1):
#             new_fields[branch] = ak.firsts(events[branch])
#         else:
#             new_fields[branch] = events[branch]

#     events_masked = ak.zip(new_fields)
#     all_chunks_sig.append(events_masked)
# sig = ak.concatenate(all_chunks_sig)
# # print(ak.type(sig))
# sig = ak.to_dataframe(sig)
# print(sig.shape)

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
# sig['target'] = np.ones(sig.shape[0], dtype=int)


def deltaphi(phi1, phi2):
    delta_phi_internal = np.abs(phi1 - phi2)
    delta_phi_external = 2*np.pi - np.abs(phi1 - phi2)
    delta_phi = np.minimum(delta_phi_internal, delta_phi_external)

    return delta_phi

# sig['Muon_pt_Slimmed_TauTo3Mu_mu3_idx'] = sig['Muon_pt[Slimmed_TauTo3Mu_mu3_idx]']

if tau_channel == "HF":
    bkg['Muon_pt_Slimmed_TauTo3Mu_mu3_idx'] = bkg['Muon_pt[Slimmed_TauTo3Mu_mu3_idx]']

    features += ['Muon_pt_Slimmed_TauTo3Mu_mu3_idx']
    features.remove('Muon_pt[Slimmed_TauTo3Mu_mu3_idx]')

    # sig['Muon_mediumId_Slimmed_TauTo3Mu_mu1_idx'] = sig['Muon_mediumId[Slimmed_TauTo3Mu_mu1_idx]']

    # features += ['Muon_mediumId_Slimmed_TauTo3Mu_mu1_idx']
    # features.remove('Muon_mediumId[Slimmed_TauTo3Mu_mu1_idx]')

    # sig['Muon_mediumId_Slimmed_TauTo3Mu_mu2_idx'] = sig['Muon_mediumId[Slimmed_TauTo3Mu_mu2_idx]']

    # features += ['Muon_mediumId_Slimmed_TauTo3Mu_mu2_idx']
    # features.remove('Muon_mediumId[Slimmed_TauTo3Mu_mu2_idx]')

    # sig['Muon_mediumId_Slimmed_TauTo3Mu_mu3_idx'] = sig['Muon_mediumId[Slimmed_TauTo3Mu_mu3_idx]']

    # features += ['Muon_mediumId_Slimmed_TauTo3Mu_mu3_idx']
    # features.remove('Muon_mediumId[Slimmed_TauTo3Mu_mu3_idx]')

    # sig['Muon_isGlobal_Slimmed_TauTo3Mu_mu1_idx'] = sig['Muon_isGlobal[Slimmed_TauTo3Mu_mu1_idx]']
    bkg['Muon_isGlobal_Slimmed_TauTo3Mu_mu1_idx'] = bkg['Muon_isGlobal[Slimmed_TauTo3Mu_mu1_idx]']

    features += ['Muon_isGlobal_Slimmed_TauTo3Mu_mu1_idx']
    features.remove('Muon_isGlobal[Slimmed_TauTo3Mu_mu1_idx]')

    # sig['Muon_isGlobal_Slimmed_TauTo3Mu_mu2_idx'] = sig['Muon_isGlobal[Slimmed_TauTo3Mu_mu2_idx]']
    bkg['Muon_isGlobal_Slimmed_TauTo3Mu_mu2_idx'] = bkg['Muon_isGlobal[Slimmed_TauTo3Mu_mu2_idx]']

    features += ['Muon_isGlobal_Slimmed_TauTo3Mu_mu2_idx']
    features.remove('Muon_isGlobal[Slimmed_TauTo3Mu_mu2_idx]')

    # sig['Muon_isGlobal_Slimmed_TauTo3Mu_mu3_idx'] = sig['Muon_isGlobal[Slimmed_TauTo3Mu_mu3_idx]']
    bkg['Muon_isGlobal_Slimmed_TauTo3Mu_mu3_idx'] = bkg['Muon_isGlobal[Slimmed_TauTo3Mu_mu3_idx]']

    features += ['Muon_isGlobal_Slimmed_TauTo3Mu_mu3_idx']
    features.remove('Muon_isGlobal[Slimmed_TauTo3Mu_mu3_idx]')

    # sig['Slimmed_TauTo3Mu_vtx_chi2*(Slimmed_TauTo3Mu_vtx_Ndof**-1)'] = sig['Slimmed_TauTo3Mu_vtx_chi2']/sig['Slimmed_TauTo3Mu_vtx_Ndof']
    bkg['Slimmed_TauTo3Mu_vtx_chi2*(Slimmed_TauTo3Mu_vtx_Ndof**-1)'] = bkg['Slimmed_TauTo3Mu_vtx_chi2']/bkg['Slimmed_TauTo3Mu_vtx_Ndof']

    features += ['Slimmed_TauTo3Mu_vtx_chi2*(Slimmed_TauTo3Mu_vtx_Ndof**-1)']

    # sig['max(Muon_tkRelIso_Slimmed_TauTo3Mu_mu1_idx Muon_tkRelIso_Slimmed_TauTo3Mu_mu2_idx Muon_tkRelIso_Slimmed_TauTo3Mu_mu3_idx)'] = np.max(
    #     np.stack([
    #         sig['Muon_tkRelIso[Slimmed_TauTo3Mu_mu1_idx]'].values,
    #         sig['Muon_tkRelIso[Slimmed_TauTo3Mu_mu2_idx]'].values,
    #         sig['Muon_tkRelIso[Slimmed_TauTo3Mu_mu3_idx]'].values
    #     ], axis=0),
    #     axis=0
    # )

    bkg['max(Muon_tkRelIso_Slimmed_TauTo3Mu_mu1_idx Muon_tkRelIso_Slimmed_TauTo3Mu_mu2_idx Muon_tkRelIso_Slimmed_TauTo3Mu_mu3_idx)'] = np.max(
        np.stack([
            bkg['Muon_tkRelIso[Slimmed_TauTo3Mu_mu1_idx]'].values,
            bkg['Muon_tkRelIso[Slimmed_TauTo3Mu_mu2_idx]'].values,
            bkg['Muon_tkRelIso[Slimmed_TauTo3Mu_mu3_idx]'].values
        ], axis=0),
        axis=0
    )

elif tau_channel == "W":
    bkg['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)'] = deltaphi(bkg['TauPlusMET_Puppi_phi'], bkg['Slimmed_TauTo3Mu_fitted_phi'])

    features += ['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)']

    # bkg['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)'] = bkg['Slimmed_TauTo3Mu_fitted_pt'] / bkg['TauPlusMET_PuppiMET_pt']

    # features += ['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)']

    abs_iso_bkg = bkg["Slimmed_TauTo3Mu_iso_ptChargedFromPV"] + np.maximum(0., bkg["Slimmed_TauTo3Mu_iso_ptPhotons"] - 0.2 * bkg["Slimmed_TauTo3Mu_iso_ptChargedFromPU"])
    bkg['relative_iso'] = abs_iso_bkg / bkg["Slimmed_TauTo3Mu_fitted_pt"]

# sig['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)'] = deltaphi(sig['TauPlusMET_Puppi_phi'], sig['Slimmed_TauTo3Mu_fitted_phi'])

# features += ['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)']

# sig['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)'] = sig['Slimmed_TauTo3Mu_fitted_pt'] / sig['TauPlusMET_PuppiMET_pt']

# features += ['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)']

# abs_iso_sig = sig["Slimmed_TauTo3Mu_iso_ptChargedFromPV"] + np.maximum(0., sig["Slimmed_TauTo3Mu_iso_ptPhotons"] - 0.2 * sig["Slimmed_TauTo3Mu_iso_ptChargedFromPU"])
# sig['relative_iso'] = abs_iso_sig / sig["Slimmed_TauTo3Mu_fitted_pt"]


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
            bdt_features = features + ['max(Muon_tkRelIso_Slimmed_TauTo3Mu_mu1_idx Muon_tkRelIso_Slimmed_TauTo3Mu_mu2_idx Muon_tkRelIso_Slimmed_TauTo3Mu_mu3_idx)']
        elif tau_channel == "W":
            bdt_features = features + ['relative_iso']
        classifiers = classifiers_iso

    n_class = len(classifiers)
    bkg  ['bdt_' + mode] = np.zeros_like(bkg.Slimmed_TauTo3Mu_fitted_mass)


    for i, iclas in classifiers.items():
        print(f'evaluating the {i}-th classifier')
        bkg  ['bdt_fold%d_' %i + mode] = iclas.predict_proba(bkg[bdt_features])[:, 1]
        bkg  ['bdt_' + mode] += iclas.predict_proba(bkg[bdt_features])[:, 1] / n_class

    if mode == "GNN":
        output["GNN_score_bkg"] = bkg.bdt_GNN

    elif mode == "ISO":
        output["ISO_score_bkg"] = bkg.bdt_ISO


    output["tau_mass"] = bkg.Slimmed_TauTo3Mu_fitted_mass
    
pkl_file = open(model_path + 'tau_score_corr.pkl', 'wb')
pickle.dump(output, pkl_file)
pkl_file.close()