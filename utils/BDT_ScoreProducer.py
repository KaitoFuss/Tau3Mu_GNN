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
args = parser.parse_args()


model = args.model
model_path = "/work/kfuss/SVJ_GNN/output/" + model + "/"
backgrounds = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/" + model + "/data_skimmed_no_resonances_with_scores.root"]
classifier_file_gnn = args.classifier_gnn
classifier_file_iso = args.classifier_iso



##########################################################################################
tag = '21may2025_v0'
##########################################################################################

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
    'Slimmed_TauTo3Mu_charge',
    'Slimmed_TauTo3Mu_fitted_eta',
    'Slimmed_TauTo3Mu_fitted_mass',
    'TauPlusMET_Puppi_phi',
    'Slimmed_TauTo3Mu_fitted_phi',
    'GNN_Score',
    'Slimmed_TauTo3Mu_iso_ptChargedFromPV',
    'Slimmed_TauTo3Mu_iso_ptPhotons',
    'Slimmed_TauTo3Mu_iso_ptChargedFromPU',
    # 'mu1_refit_muonid_soft', 'mu1_refit_muonid_loose', 'mu1_refit_muonid_medium', 'mu1_refit_muonid_tight',
    # 'mu2_refit_muonid_soft', 'mu2_refit_muonid_loose', 'mu2_refit_muonid_medium', 'mu2_refit_muonid_tight',
    # 'mu3_refit_muonid_soft', 'mu3_refit_muonid_loose', 'mu3_refit_muonid_medium', 'mu3_refit_muonid_tight',
]

all_chunks_bkg = []
print("Going to load bkg events")
for events in uproot.iterate(f"{backgrounds[0]}:Events", expressions=branches, library="ak", step_size="100 MB"):
    # bkg_mask = abs(events["Slimmed_TauTo3Mu_fitted_mass"] - 1.78) > 0.08
    # events = events[ak.any(bkg_mask, axis=1)]
    score_mask = events["GNN_Score"] != -1
    events_masked = events[score_mask]

    new_fields = {}
    for branch in events_masked.fields:
        if events_masked[branch].ndim > 1 and ak.any(ak.num(events_masked[branch], axis=1) > 1):
            new_fields[branch] = ak.firsts(events_masked[branch])
        else:
            new_fields[branch] = events_masked[branch]

    events_stripped = ak.zip(new_fields)
    all_chunks_bkg.append(events_stripped)
bkg = ak.concatenate(all_chunks_bkg)
# print(ak.type(bkg))
bkg = ak.to_dataframe(bkg)
nan_rows = bkg[bkg.isna().any(axis=1)]
bkg = bkg.dropna()
print(bkg.shape)

# bkg = bkg.sample(n=min(1_000_000, len(bkg)), random_state=42).reset_index(drop=True)

# print(bkg.shape)

##########################################################################################
## DEFINE TARGETS
##########################################################################################

bkg['target'] = np.zeros(bkg.shape[0], dtype=int)


def deltaphi(phi1, phi2):
    delta_phi_internal = np.abs(phi1 - phi2)
    delta_phi_external = 2*np.pi - np.abs(phi1 - phi2)
    delta_phi = np.minimum(delta_phi_internal, delta_phi_external)

    return delta_phi

bkg['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)'] = deltaphi(bkg['TauPlusMET_Puppi_phi'], bkg['Slimmed_TauTo3Mu_fitted_phi'])

features += ['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)']

bkg['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)'] = bkg['Slimmed_TauTo3Mu_fitted_pt'] / bkg['TauPlusMET_PuppiMET_pt']

features += ['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)']

abs_iso_bkg = bkg["Slimmed_TauTo3Mu_iso_ptChargedFromPV"] + np.maximum(0., bkg["Slimmed_TauTo3Mu_iso_ptPhotons"] - 0.2 * bkg["Slimmed_TauTo3Mu_iso_ptChargedFromPU"])
bkg['relative_iso'] = abs_iso_bkg / bkg["Slimmed_TauTo3Mu_fitted_pt"]

with open(classifier_file_gnn, "rb") as f:
    classifiers_gnn = pickle.load(f)

with open(classifier_file_iso, "rb") as f:
    classifiers_iso = pickle.load(f)

output = {}
for mode in ["GNN", "ISO"]:
    print(f"now mode {mode}")
    if mode == "GNN":
        bdt_features = features + ['GNN_Score']
        classifiers = classifiers_gnn
    elif mode == "ISO":
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