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

from pdb import set_trace



##########################################################################################
# https://www.dataiku.com/learn/guide/code/python/advanced-xgboost-tuning.html


# signals     = [
#     "/work/kfuss/SVJ_GNN/output/knn_new_vs7/signal_skimmed_no_resonances_with_scores.root"
# ]

# backgrounds = [
#     "/work/kfuss/SVJ_GNN/output/knn_new_vs7/data_skimmed_no_resonances_with_scores.root"
# ]

# model = "knn_new_vs7"

parser = argparse.ArgumentParser()
# parser.add_argument("--signal", nargs='+', help="Path(s) to signal ROOT file(s)")
# parser.add_argument("--background", nargs='+', help="Path(s) to background ROOT file(s)")
parser.add_argument("--model", type=str, required=True, help="Model name of GNN")
args = parser.parse_args()


model = args.model
model_path = "/work/kfuss/SVJ_GNN/output/" + model + "/"
signals = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/" + model + "/signal_skimmed_no_resonances_with_scores.root"]
backgrounds = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/" + model + "/data_skimmed_no_resonances_with_scores.root"]

# give labels human readable names
modes = defaultdict(OrderedDict)
labels = modes["GNN"]

labels['Slimmed_TauTo3Mu_fitted_pt'                                         ] = '$\\tau$ $p_{T}$'
labels['TauPlusMET_Tau_Puppi_mT'                                            ] = '$m_{T}(\\tau, MET)$'
labels['TauPlusMET_PuppiMET_pt'                                             ] = 'MET $p_{T}$'
# labels['cand_refit_tau_pt/cand_refit_met_pt'                              ] = '$\\tau$ $p_{T}$/MET $p_{T}$' # only for >= v2
# labels['cand_refit_dRtauMuonMax'                                          ] = 'max($\Delta R(\\tau \mu_{i})$)'
labels['TauPlusMET_Puppi_pt'                                                ] = 'W $p_{T}$'
labels['TauPlusMET_PuppiMETminPz'                                           ] = '$max(ME_z^i)$'
labels['TauPlusMET_PuppiMETmaxPz'                                           ] = '$min(ME_z^i)$'
labels['Slimmed_TauTo3Mu_dZmu12'                                            ] = '$\Delta z (\mu_1, \mu_2)$'
labels['Slimmed_TauTo3Mu_dZmu13'                                            ] = '$\Delta z (\mu_1, \mu_3)$'
labels['Slimmed_TauTo3Mu_dZmu23'                                            ] = '$\Delta z (\mu_2, \mu_3)$'
labels['Slimmed_TauTo3Mu_sigLxy_3muVtxBS'                                   ] = 'SV L/$\sigma$'
labels['Slimmed_TauTo3Mu_vtx_chi2*(Slimmed_TauTo3Mu_vtx_Ndof**-1)'          ] = '$\\chi^2_{\\tau}$'
labels['Slimmed_TauTo3Mu_CosAlpha2D_LxyP3mu'                                ] = 'SV cos($\\theta_{IP}$)'
labels['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)'        ] = '$\Delta\phi(\\tau MET)$'
# labels['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)'            ] = '$\\tau$ $p_{T}$/MET $p_{T}$' # only for >= v2
# labels['mu1ID'                                                            ] = '$\mu_1$ ID'
# labels['mu2ID'                                                            ] = '$\mu_2$ ID'
# labels['mu3ID'                                                            ] = '$\mu_3$ ID'
# labels['tauEta'                                                           ] = '$|\eta_{\\tau}|$'
# labels['relative_iso'                                                     ] = '$\\tau$ iso'
labels['GNN_score'                                                          ] = 'GNN Score'
labels['bdt_GNN'                                                            ] = 'BDT'
labels['Slimmed_TauTo3Mu_fitted_mass'                                       ] = '$\\tau$ mass'

labels = modes["ISO"]

labels['Slimmed_TauTo3Mu_fitted_pt'                                         ] = '$\\tau$ $p_{T}$'
labels['TauPlusMET_Tau_Puppi_mT'                                            ] = '$m_{T}(\\tau, MET)$'
labels['TauPlusMET_PuppiMET_pt'                                             ] = 'MET $p_{T}$'
# labels['cand_refit_tau_pt/cand_refit_met_pt'                              ] = '$\\tau$ $p_{T}$/MET $p_{T}$' # only for >= v2
# labels['cand_refit_dRtauMuonMax'                                          ] = 'max($\Delta R(\\tau \mu_{i})$)'
labels['TauPlusMET_Puppi_pt'                                                ] = 'W $p_{T}$'
labels['TauPlusMET_PuppiMETminPz'                                           ] = '$max(ME_z^i)$'
labels['TauPlusMET_PuppiMETmaxPz'                                           ] = '$min(ME_z^i)$'
labels['Slimmed_TauTo3Mu_dZmu12'                                            ] = '$\Delta z (\mu_1, \mu_2)$'
labels['Slimmed_TauTo3Mu_dZmu13'                                            ] = '$\Delta z (\mu_1, \mu_3)$'
labels['Slimmed_TauTo3Mu_dZmu23'                                            ] = '$\Delta z (\mu_2, \mu_3)$'
labels['Slimmed_TauTo3Mu_sigLxy_3muVtxBS'                                   ] = 'SV L/$\sigma$'
labels['Slimmed_TauTo3Mu_vtx_chi2*(Slimmed_TauTo3Mu_vtx_Ndof**-1)'          ] = '$\chi^2_{\tau}$'
labels['Slimmed_TauTo3Mu_CosAlpha2D_LxyP3mu'                                ] = 'SV cos($\\theta_{IP}$)'
labels['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)'        ] = '$\Delta\phi(\\tau, MET)$'
# labels['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)'            ] = '$\\tau$ $p_{T}$/MET $p_{T}$' # only for >= v2
# labels['mu1ID'                                                            ] = '$\mu_1$ ID'
# labels['mu2ID'                                                            ] = '$\mu_2$ ID'
# labels['mu3ID'                                                            ] = '$\mu_3$ ID'
# labels['tauEta'                                                           ] = '$|\eta_{\\tau}|$'
labels['relative_iso'                                                       ] = '$\\tau$ iso'
# labels['GNN_Score'                                                        ] = 'GNN Score'
labels['bdt_ISO'                                                            ] = 'BDT'
labels['Slimmed_TauTo3Mu_fitted_mass'                                       ] = '$\\tau$ mass'


##########################################################################################
# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

##########################################################################################
tag = '24feb2026_v0'
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
    'GNN_score',
    'Slimmed_TauTo3Mu_iso_ptChargedFromPV',
    'Slimmed_TauTo3Mu_iso_ptPhotons',
    'Slimmed_TauTo3Mu_iso_ptChargedFromPU',
    # 'mu1_refit_muonid_soft', 'mu1_refit_muonid_loose', 'mu1_refit_muonid_medium', 'mu1_refit_muonid_tight',
    # 'mu2_refit_muonid_soft', 'mu2_refit_muonid_loose', 'mu2_refit_muonid_medium', 'mu2_refit_muonid_tight',
    # 'mu3_refit_muonid_soft', 'mu3_refit_muonid_loose', 'mu3_refit_muonid_medium', 'mu3_refit_muonid_tight',
]

##########################################################################################

# sig_selection = '(  Slimmed_TauTo3Mu_fitted_mass > 1.6 & Slimmed_TauTo3Mu_fitted_mass < 2.0                                                                                 & abs(Slimmed_TauTo3Mu_charge)==1 & GNN_Score != -1)'
# bkg_selection = '(((Slimmed_TauTo3Mu_fitted_mass > 1.6 & Slimmed_TauTo3Mu_fitted_mass < 1.72) | (Slimmed_TauTo3Mu_fitted_mass > 1.84 & Slimmed_TauTo3Mu_fitted_mass < 2.0)) & abs(Slimmed_TauTo3Mu_charge)==1 & GNN_Score != -1)' #TODO adapt masking 

# sig_selection = '(GNN_Score != -1)'
# bkg_selection = '(((Slimmed_TauTo3Mu_fitted_mass > 1.86 & Slimmed_TauTo3Mu_fitted_mass < 1.70)) & abs(Slimmed_TauTo3Mu_charge)==1 & GNN_Score != -1)'

##########################################################################################


# sig = pd.DataFrame( root_numpy.root2array(signals    , 'Events', branches  = branches + ['weight'], selection = sig_selection) )
# bkg = pd.DataFrame( root_numpy.root2array(backgrounds, 'Events', branches  = branches + ['weight'], selection = bkg_selection) )

all_chunks_sig = []

print("Going to load signal events")
for events in uproot.iterate(f"{signals[0]}:Events", expressions=branches, library="ak", step_size="100 MB"):
    print("Signal fraction != -1:", ak.mean(events["GNN_score"] != -1))
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
print(sig.shape)
print(sig["GNN_score"])

all_chunks_bkg = []
print("Going to load bkg events")
for events in uproot.iterate(f"{backgrounds[0]}:Events", expressions=branches, library="ak", step_size="100 MB"):
    bkg_mask = abs(events["Slimmed_TauTo3Mu_fitted_mass"] - 1.78) > 0.08
    events = events[ak.all(bkg_mask, axis=1)]
    score_mask = events["GNN_score"] != -1
    events_masked = events[score_mask]

    new_fields = {}
    for branch in events_masked.fields:
        arr = events_masked[branch]
        if arr.ndim > 1 and ak.any(ak.num(arr, axis=1) > 1):
            new_fields[branch] = ak.firsts(arr)
        else:
            new_fields[branch] = arr

    events_masked = ak.zip(new_fields)
    df_chunk = ak.to_dataframe(events_masked)
    all_chunks_bkg.append(df_chunk)

bkg = pd.concat(all_chunks_bkg, ignore_index=True)
nan_rows = bkg[bkg.isna().any(axis=1)]
bkg = bkg.dropna()
print(bkg.shape)
print(bkg["GNN_score"])

sig = sig.sample(n=min(100_000, len(sig)), random_state=42).reset_index(drop=True)
bkg = bkg.sample(n=min(100_000, len(bkg)), random_state=42).reset_index(drop=True)

print(sig.shape)
print(bkg.shape)

##########################################################################################
## DEFINE TARGETS
##########################################################################################

sig['target'] = np.ones (sig.shape[0], dtype=int)
bkg['target'] = np.zeros(bkg.shape[0], dtype=int)

##########################################################################################
## WEIGHT A BIT MORE EVENTS AT LOWER MASSES, AS THERE'S LESS
########################################################################################## 
# @np.vectorize
# def massWeighter(mass):
#     # slope
#     m = (1.15 - 0.85) / (1.6-2)    
#     # intercept
#     q = 2.35
#     return m * mass + q
# 
# sig['weight'] *= massWeighter(sig['cand_refit_tau_mass'])
# bkg['weight'] *= massWeighter(bkg['cand_refit_tau_mass'])

##########################################################################################
## REWEIGHT AND MAKE TAU MASS FLAT
########################################################################################## 
# sig_integral = np.sum(sig['weight'])
# bkg_integral = np.sum(bkg['weight'])

# bins = np.linspace(1.6, 2, 40)

# sig_mass_sum_weight = []
# bkg_mass_sum_weight = []

# for ibin in range(len(bins)-1):
#     m_min = bins[ibin]
#     m_max = bins[ibin+1]
    
#     sig_mass_sum_weight.append( np.sum(sig[sig.cand_refit_tau_mass>=m_min][sig.cand_refit_tau_mass<m_max]['weight']) )
#     bkg_mass_sum_weight.append( np.sum(bkg[bkg.cand_refit_tau_mass>=m_min][bkg.cand_refit_tau_mass<m_max]['weight']) )

# sig_mass_weights = np.array(sig_mass_sum_weight) / sig_integral
# bkg_mass_weights = np.array(bkg_mass_sum_weight) / bkg_integral

# @np.vectorize
# def massWeighterSig(mass):
#     bin_low = np.max(np.where(mass>=bins))
#     return sig_mass_weights[bin_low]

# @np.vectorize
# def massWeighterBkg(mass):
#     bin_low = np.max(np.where(mass>=bins))
#     return bkg_mass_weights[bin_low]
    
# sig['weight'] /= massWeighterSig(sig['cand_refit_tau_mass'])
# bkg['weight'] /= massWeighterBkg(bkg['cand_refit_tau_mass'])

# further weight adjustment
sig['weight'] = 1.
bkg['weight'] = 1.

def deltaphi(phi1, phi2):
    delta_phi_internal = np.abs(phi1 - phi2)
    delta_phi_external = 2*np.pi - np.abs(phi1 - phi2)
    delta_phi = np.minimum(delta_phi_internal, delta_phi_external)

    return delta_phi

sig['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)'] = deltaphi(sig['TauPlusMET_Puppi_phi'], sig['Slimmed_TauTo3Mu_fitted_phi'])
bkg['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)'] = deltaphi(bkg['TauPlusMET_Puppi_phi'], bkg['Slimmed_TauTo3Mu_fitted_phi'])

features += ['deltaPhi(TauPlusMET_Puppi_phi, Slimmed_TauTo3Mu_fitted_phi)']

# sig['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)'] = sig['Slimmed_TauTo3Mu_fitted_pt'] / sig['TauPlusMET_PuppiMET_pt']
# bkg['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)'] = bkg['Slimmed_TauTo3Mu_fitted_pt'] / bkg['TauPlusMET_PuppiMET_pt']

# features += ['Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)']

abs_iso_sig = sig["Slimmed_TauTo3Mu_iso_ptChargedFromPV"] + np.maximum(0., sig["Slimmed_TauTo3Mu_iso_ptPhotons"] - 0.2 * sig["Slimmed_TauTo3Mu_iso_ptChargedFromPU"])
sig['relative_iso'] = abs_iso_sig / sig["Slimmed_TauTo3Mu_fitted_pt"]

abs_iso_bkg = bkg["Slimmed_TauTo3Mu_iso_ptChargedFromPV"] + np.maximum(0., bkg["Slimmed_TauTo3Mu_iso_ptPhotons"] - 0.2 * bkg["Slimmed_TauTo3Mu_iso_ptChargedFromPU"])
bkg['relative_iso'] = abs_iso_bkg / bkg["Slimmed_TauTo3Mu_fitted_pt"]

##########################################################################################
#####   COMPACTIFY AND ADD THE (POG) MUON ID TO THE SAMPLES
##########################################################################################
# @np.vectorize
# def muID(loose, medium, tight):
# 	if   tight  > 0.5: return 3
# 	elif medium > 0.5: return 2
# 	elif loose  > 0.5: return 1
# 	else             : return 0

# for mu in [1,2,3]:
# 	name = 'mu%iID' % mu
# 	features.append(name)
# 	sig[name] = muID(
# 		sig['mu%d_refit_muonid_loose'  % mu], 
# 		sig['mu%d_refit_muonid_medium' % mu], 
# 		sig['mu%d_refit_muonid_tight'  % mu],
# 	)
# 	bkg[name] = muID(
# 		bkg['mu%d_refit_muonid_loose'  % mu], 
# 		bkg['mu%d_refit_muonid_medium' % mu], 
# 		bkg['mu%d_refit_muonid_tight'  % mu],
# 	)
	
##########################################################################################
#####   ETA BINS
##########################################################################################
# @np.vectorize
# def tauEta(eta):
# 	if   abs(eta) > 2.1 : return 7
# 	elif abs(eta) > 1.8 : return 6
# 	elif abs(eta) > 1.5 : return 5
# 	elif abs(eta) > 1.1 : return 4
# 	elif abs(eta) > 0.8 : return 3
# 	elif abs(eta) > 0.5 : return 2
# 	elif abs(eta) > 0.2 : return 1
# 	else                : return 0
    
# features.append('tauEta')
# sig['tauEta'] = tauEta(sig['cand_refit_tau_eta'])
# bkg['tauEta'] = tauEta(bkg['cand_refit_tau_eta'])

##########################################################################################
data = pd.concat([sig, bkg], ignore_index=True)
data['id'] = np.arange(len(data))

print("Splitting data")

train, test = train_test_split(data, test_size=0.4, random_state=1986)
both_models = defaultdict(OrderedDict)

##########################################################################################
for mode in modes.keys():
    print(f"starting with training with {mode} score")
    if mode == "GNN":
        bdt_features = features + ['GNN_score']
    elif mode == "ISO":
        bdt_features = features + ['relative_iso']
    
    kfold = 6
    skf = StratifiedKFold(n_splits=kfold, random_state=1986, shuffle=True)

    ##########################################################################################
    sub = pd.DataFrame()
    sub['id']     = test.id
    sub['target'] = test.target
    sub['score']  = np.zeros_like(test.id)

    classifier_file = open(model_path + mode + '_classifiers_%s.pck' % tag, 'wb')
    classifiers = both_models[mode]

    # https://www.kaggle.com/sudosudoohio/stratified-kfold-xgboost-eda-tutorial-0-281
    for i, (train_index, test_index) in enumerate(skf.split(train[bdt_features].values, train['target'].values)):
        # if i>1: break
        print('[Fold %d/%d]' % (i + 1, kfold))    
        sample_weight = train.iloc[train_index]['weight']
        X_train, X_valid = train.iloc[train_index][bdt_features], train.iloc[test_index][bdt_features]
        y_train, y_valid = train.iloc[train_index]['target'], train.iloc[test_index]['target']
        
        # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
        # and the custom metric (maximize=True tells xgb that higher metric is better)
        clf = XGBClassifier(#silent = False)
            max_depth        = 8,
            learning_rate    = 0.005, # 0.01
            n_estimators     = 1000, # 400
            verbosity        = 1,
            subsample        = 0.6,
            colsample_bytree = 0.7,
            min_child_weight = 1E-6 * np.sum(train.iloc[train_index].weight),
            gamma            = 10, # 20
            seed             = 1986,
            # scale_pos_weight = 0.5,
            reg_alpha        = 0.6,
            reg_lambda       = 1.2,
            early_stopping_rounds = 100,
            eval_metric           = 'logloss',
            # eval_metric = 'logloss',
        )

        print("y_train min/max/mean:", y_train.min(), y_train.max(), y_train.mean())

        clf.fit(
            X_train, 
            y_train,
            eval_set              = [(X_train, y_train), (X_valid, y_valid)],
            verbose               = False,
            sample_weight         = sample_weight,
        )
        evals_result = clf.evals_result()
        classifiers[i] = clf
        
        print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
        # Predict on our test data
        p_test = clf.predict_proba(test[bdt_features])[:, 1]
        sub['score'] += p_test/kfold

        # adjust the score to match 0,1
        smin = min(p_test)
        smax = max(p_test)

        sub['score_norm'] = (p_test - smin) / (smax - smin)

        print(f'round {i+1}')
        print(f'\tcross validation error      {np.sum(np.abs(sub["score_norm"] - sub["target"])) / len(sub):.5f}')
        print(f'\tcross validation signal     {np.sum(np.abs(sub[sub.target > 0.5]["score_norm"] - sub[sub.target > 0.5]["target"])) / len(sub):.5f}')
        print(f'\tcross validation background {np.sum(np.abs(sub[sub.target < 0.5]["score_norm"] - sub[sub.target < 0.5]["target"])) / len(sub):.5f}')


        smin = min(sub['score'])
        smax = max(sub['score'])
        
        sub['score_i'] = (sub['score'] - smin) / (smax - smin)

        print(f'global score as of round {i + 1}')
        print(f'\tcross validation error      {np.sum(np.abs(sub["score_i"] - sub["target"])) / len(sub):.5f}')
        print(f'\tcross validation signal     {np.sum(np.abs(sub[sub.target > 0.5]["score_i"] - sub[sub.target > 0.5]["target"])) / len(sub):.5f}')
        print(f'\tcross validation background {np.sum(np.abs(sub[sub.target < 0.5]["score_i"] - sub[sub.target < 0.5]["target"])) / len(sub):.5f}')

        epochs = len(evals_result['validation_0']['logloss'])
        x_axis = range(0, epochs)

        plt.figure()
        plt.plot(x_axis, evals_result['validation_0']['logloss'], label='Train')
        plt.plot(x_axis, evals_result['validation_1']['logloss'], label='Validation')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Fold {i+1} - Loss over Epochs ({mode})')
        plt.grid(True)
        plt.savefig(f'{model_path}logloss_fold_{i+1}_{mode}.png')  # or plt.show() to display inline
        plt.close()


    pickle.dump(classifiers, classifier_file)
    classifier_file.close()

##########################################################################################
n_class = len(classifiers)

for mode in modes.keys():

    train['bdt_' + mode] = np.zeros_like(train.Slimmed_TauTo3Mu_fitted_mass)
    test ['bdt_' + mode] = np.zeros_like(test .Slimmed_TauTo3Mu_fitted_mass)
    sig  ['bdt_' + mode] = np.zeros_like(sig.Slimmed_TauTo3Mu_fitted_mass)
    bkg  ['bdt_' + mode] = np.zeros_like(bkg.Slimmed_TauTo3Mu_fitted_mass)
    classifiers = both_models[mode]

    if mode == "GNN":
        bdt_features = features + ['GNN_score']
    elif mode == "ISO":
        bdt_features = features + ['relative_iso']

    for i, iclas in classifiers.items():
        print(f'evaluating the {i}-th classifier')
        train['bdt_fold%d_' %i + mode] = iclas.predict_proba(train[bdt_features])[:, 1]
        test ['bdt_fold%d_' %i + mode] = iclas.predict_proba(test [bdt_features])[:, 1]
        sig  ['bdt_fold%d_' %i + mode] = iclas.predict_proba(sig[bdt_features])[:, 1]
        bkg  ['bdt_fold%d_' %i + mode] = iclas.predict_proba(bkg[bdt_features])[:, 1]

        train['bdt_' + mode] += iclas.predict_proba(train[bdt_features])[:, 1] / n_class
        test ['bdt_' + mode] += iclas.predict_proba(test [bdt_features])[:, 1] / n_class
        sig  ['bdt_' + mode] += iclas.predict_proba(sig[bdt_features])[:, 1] / n_class
        bkg  ['bdt_' + mode] += iclas.predict_proba(bkg[bdt_features])[:, 1] / n_class

# print '=================================>'
# print  

# plt.clf()
# aa = sns.distplot(bkg.bdt_fold0)
# bb = sns.distplot(bkg.bdt_fold1)
# cc = sns.distplot(bkg.bdt_fold2)
# aa.set_yscale('log')
# bb.set_yscale('log')
# cc.set_yscale('log')
# plt.savefig('histo.pdf')
# exit()

##########################################################################################
#####   ROC CURVE
##########################################################################################
plt.clf()

cuts_to_display = [0.80, 0.82, 0.84, 0.86, 0.89]

xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
plt.plot(xy, xy, color='grey', linestyle='--')
plt.xlim([10**-5, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

fpr_GNN, tpr_GNN, wps_GNN = roc_curve(test.target, test.bdt_GNN, sample_weight=test.weight)
auc_GNN = roc_auc_score(test.target, test.bdt_GNN, sample_weight=test.weight)
plt.plot(fpr_GNN, tpr_GNN, label='test sample GNN', color='b')

output = {}

output["GNN_score_bkg"] = bkg.bdt_GNN
output["tau_mass"] = bkg.Slimmed_TauTo3Mu_fitted_mass
output["ISO_score_bkg"] = bkg.bdt_ISO

output["fpr_GNN"] = fpr_GNN
output["tpr_GNN"] = tpr_GNN
output["threshold_GNN"] = wps_GNN
output["auc_GNN"] = auc_GNN

wp_x = []
wp_y = []

for icut in cuts_to_display:
    idx = (wps_GNN>icut).sum()
    wp_x.append(fpr_GNN[idx])
    wp_y.append(tpr_GNN[idx])
    
plt.scatter(wp_x, wp_y)
for i, note in enumerate(cuts_to_display):
    plt.annotate(note, (wp_x[i], wp_y[i]))

fpr_ISO, tpr_ISO, wps_ISO = roc_curve(test.target, test.bdt_ISO, sample_weight=test.weight)
auc_ISO = roc_auc_score(test.target, test.bdt_ISO, sample_weight=test.weight)
plt.plot(fpr_ISO, tpr_ISO, label='test sample ISO', color='r')

output["fpr_ISO"] = fpr_ISO
output["tpr_ISO"] = tpr_ISO
output["threshold_ISO"] = wps_ISO
output["auc_ISO"] = auc_ISO

wp_x = []
wp_y = []

for icut in cuts_to_display:
    idx = (wps_ISO>icut).sum()
    wp_x.append(fpr_ISO[idx])
    wp_y.append(tpr_ISO[idx])
    
plt.scatter(wp_x, wp_y)
for i, note in enumerate(cuts_to_display):
    plt.annotate(note, (wp_x[i], wp_y[i]))

# fpr, tpr, wps = roc_curve(train.target, train.bdt, sample_weight=train.weight)
# plt.plot(fpr, tpr, label='train sample', color='r')

# wp_x = []
# wp_y = []

# for icut in cuts_to_display:
#     idx = (wps>icut).sum()
#     wp_x.append(fpr[idx])
#     wp_y.append(tpr[idx])
    
# plt.scatter(wp_x, wp_y)
# for i, note in enumerate(cuts_to_display):
#     plt.annotate(note, (wp_x[i], wp_y[i]))

# print(f"ROC AUC train: {roc_auc_score(train.target, train.bdt, sample_weight=train.weight):.5f}")
print(f"ROC AUC test GNN:  {roc_auc_score(test.target, test.bdt_GNN, sample_weight=test.weight):.5f}")
print(f"ROC AUC test ISO:  {roc_auc_score(test.target, test.bdt_ISO, sample_weight=test.weight):.5f}")

plt.legend(loc='best')
plt.grid()
plt.title('ROC')
plt.tight_layout()
plt.savefig(model_path + '_roc_%s.png' %tag, dpi=400)
plt.clf()

roc_file = open(model_path + '_roc_%s.pck' % tag, 'wb')

pickle.dump(output, roc_file)
roc_file.close()

# log ROC

xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
plt.plot(xy, xy, color='grey', linestyle='--')
plt.xlim([10**-5, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.xscale('log')

fpr_GNN, tpr_GNN, wps_GNN = roc_curve(test.target, test.bdt_GNN, sample_weight=test.weight)
auc_GNN = roc_auc_score(test.target, test.bdt_GNN, sample_weight=test.weight)
plt.plot(fpr_GNN, tpr_GNN, label='test sample GNN', color='b')

output = {}

output["GNN_score_bkg"] = bkg.bdt_GNN
output["tau_mass"] = bkg.Slimmed_TauTo3Mu_fitted_mass
output["ISO_score_bkg"] = bkg.bdt_ISO

output["fpr_GNN"] = fpr_GNN
output["tpr_GNN"] = tpr_GNN
output["threshold_GNN"] = wps_GNN
output["auc_GNN"] = auc_GNN

wp_x = []
wp_y = []

for icut in cuts_to_display:
    idx = (wps_GNN>icut).sum()
    wp_x.append(fpr_GNN[idx])
    wp_y.append(tpr_GNN[idx])
    
plt.scatter(wp_x, wp_y)
for i, note in enumerate(cuts_to_display):
    plt.annotate(note, (wp_x[i], wp_y[i]))

fpr_ISO, tpr_ISO, wps_ISO = roc_curve(test.target, test.bdt_ISO, sample_weight=test.weight)
auc_ISO = roc_auc_score(test.target, test.bdt_ISO, sample_weight=test.weight)
plt.plot(fpr_ISO, tpr_ISO, label='test sample ISO', color='r')

output["fpr_ISO"] = fpr_ISO
output["tpr_ISO"] = tpr_ISO
output["threshold_ISO"] = wps_ISO
output["auc_ISO"] = auc_ISO

wp_x = []
wp_y = []

for icut in cuts_to_display:
    idx = (wps_ISO>icut).sum()
    wp_x.append(fpr_ISO[idx])
    wp_y.append(tpr_ISO[idx])
    
plt.scatter(wp_x, wp_y)
for i, note in enumerate(cuts_to_display):
    plt.annotate(note, (wp_x[i], wp_y[i]))

# fpr, tpr, wps = roc_curve(train.target, train.bdt, sample_weight=train.weight)
# plt.plot(fpr, tpr, label='train sample', color='r')

# wp_x = []
# wp_y = []

# for icut in cuts_to_display:
#     idx = (wps>icut).sum()
#     wp_x.append(fpr[idx])
#     wp_y.append(tpr[idx])
    
# plt.scatter(wp_x, wp_y)
# for i, note in enumerate(cuts_to_display):
#     plt.annotate(note, (wp_x[i], wp_y[i]))

# print(f"ROC AUC train: {roc_auc_score(train.target, train.bdt, sample_weight=train.weight):.5f}")
print(f"ROC AUC test GNN:  {roc_auc_score(test.target, test.bdt_GNN, sample_weight=test.weight):.5f}")
print(f"ROC AUC test ISO:  {roc_auc_score(test.target, test.bdt_ISO, sample_weight=test.weight):.5f}")

plt.legend(loc='best')
plt.grid()
plt.title('ROC log')
plt.tight_layout()
plt.savefig(model_path + '_roc_log%s.png' %tag, dpi=400)
plt.clf()

roc_file = open(model_path + '_roc_log%s.pck' % tag, 'wb')

pickle.dump(output, roc_file)
roc_file.close()

##########################################################################################
#####   OVERTRAINING TEST
##########################################################################################
train_sig_GNN = train[train.target>0.5].bdt_GNN
train_bkg_GNN = train[train.target<0.5].bdt_GNN

test_sig_GNN = test[test.target>0.5].bdt_GNN
test_bkg_GNN = test[test.target<0.5].bdt_GNN

train_sig_ISO = train[train.target>0.5].bdt_ISO
train_bkg_ISO = train[train.target<0.5].bdt_ISO

test_sig_ISO = test[test.target>0.5].bdt_ISO
test_bkg_ISO = test[test.target<0.5].bdt_ISO

low  = 0
high = 1
low_high = (low,high)
bins = 50
bins_GNN = np.linspace(low, high, bins + 1)
bins_ISO = np.linspace(low, high, bins + 1)


#################################################
hist_GNN, _ = np.histogram(
    test_sig_GNN,
    bins=bins_GNN, 
#     range=low_high, 
    density=True
)

width  = (bins_GNN[1] - bins_GNN[0])
center = (bins_GNN[:-1] + bins_GNN[1:]) / 2
scale  = len(test_sig_GNN) / sum(hist_GNN)
err    = np.sqrt(hist_GNN * scale) / scale

plt.errorbar(
    center, 
    hist_GNN, 
    yerr=err, 
    fmt='o', 
    c='r', 
    label='S (test) BDT with GNN'
)

#################################################
# sns.distplot(train_sig, bins=bins, kde=False, rug=False, norm_hist=True, hist_kws={"alpha": 0.5, "color": "r"}, label='S (train)')
sns.histplot(
    train_sig_GNN,
    bins=bins_GNN,
    stat="density",         
    kde=False,
    element="step",        
    fill=True,
    alpha=0.5,
    color="r",
    label="S (train) BDT with GNN"
)

#################################################
# hist_ISO, _ = np.histogram(
#     test_sig_ISO,
#     bins=bins_ISO, 
# #     range=low_high, 
#     density=True
# )

# width  = (bins_ISO[1] - bins_ISO[0])
# center = (bins_ISO[:-1] + bins_ISO[1:]) / 2
# scale  = len(test_sig_ISO) / sum(hist_ISO)
# err    = np.sqrt(hist_ISO * scale) / scale

# plt.errorbar(
#     center, 
#     hist_ISO, 
#     yerr=err, 
#     fmt='o', 
#     c='y', 
#     label='S (test) ISO'
# )

#################################################
# sns.distplot(train_sig, bins=bins, kde=False, rug=False, norm_hist=True, hist_kws={"alpha": 0.5, "color": "r"}, label='S (train)')
# sns.histplot(
#     train_sig_ISO,
#     bins=bins_ISO,
#     stat="density",         
#     kde=False,
#     element="step",        
#     fill=True,
#     alpha=0.5,
#     color="y",
#     label="S (train) ISO"
# )

#################################################
hist_GNN, _ = np.histogram(
    test_bkg_GNN,
    bins=bins_GNN, 
#     range=low_high, 
    density=True
)

width  = (bins_GNN[1] - bins_GNN[0])
center = (bins_GNN[:-1] + bins_GNN[1:]) / 2
scale  = len(test_bkg_GNN) / sum(hist_GNN)
err    = np.sqrt(hist_GNN * scale) / scale

plt.errorbar(
    center, 
    hist_GNN, 
    yerr=err, 
    fmt='o', 
    c='b', 
    label='B (test) BDT with GNN'
)

#################################################
# sns.distplot(train_bkg, bins=bins, kde=False, rug=False, norm_hist=True, hist_kws={"alpha": 0.5, "color": "b"}, label='B (train)')
sns.histplot(
    train_bkg_GNN,
    bins=bins_GNN,
    stat="density",       
    kde=False,
    element="step",       
    fill=True,
    alpha=0.5,
    color="b",
    label="B (train) BDT with GNN"
)

#################################################
# hist_ISO, _ = np.histogram(
#     test_bkg_ISO,
#     bins=bins_ISO, 
# #     range=low_high, 
#     density=True
# )

# width  = (bins_ISO[1] - bins_ISO[0])
# center = (bins_ISO[:-1] + bins_ISO[1:]) / 2
# scale  = len(test_bkg_ISO) / sum(hist_ISO)
# err    = np.sqrt(hist_ISO * scale) / scale

# plt.errorbar(
#     center, 
#     hist_ISO, 
#     yerr=err, 
#     fmt='o', 
#     c='g', 
#     label='B (test) ISO'
# )

#################################################
# sns.distplot(train_bkg, bins=bins, kde=False, rug=False, norm_hist=True, hist_kws={"alpha": 0.5, "color": "b"}, label='B (train)')
# sns.histplot(
#     train_bkg_ISO,
#     bins=bins_ISO,
#     stat="density",       
#     kde=False,
#     element="step",       
#     fill=True,
#     alpha=0.5,
#     color="g",
#     label="B (train) ISO"
# )


#################################################
plt.xlabel('BDT output')
plt.ylabel('Count density')
plt.legend(loc='best')
ks_sig_GNN = ks_2samp(train_sig_GNN, test_sig_GNN)
ks_bkg_GNN = ks_2samp(train_bkg_GNN, test_bkg_GNN)
# ks_sig_ISO = ks_2samp(train_sig_ISO, test_sig_ISO)
# ks_bkg_ISO = ks_2samp(train_bkg_ISO, test_bkg_ISO)

plt.suptitle('KS p-values', fontsize=16, fontweight='bold', y=0.98)

plt.figtext(0.5, 0.90, 
            'GNN: sig = %.2f%%, bkg = %.2f%%' % (ks_sig_GNN.pvalue * 100, ks_bkg_GNN.pvalue * 100),
            ha='center', fontsize=12)

# plt.figtext(0.5, 0.85, 
#             'ISO: sig = %.2f%%, bkg = %.2f%%' % (ks_sig_ISO.pvalue * 100, ks_bkg_ISO.pvalue * 100),
#             ha='center', fontsize=12)

plt.subplots_adjust(top=0.88) 


# train_sig_w = np.ones_like(train_sig) * 1./len(train_sig)
# train_bkg_w = np.ones_like(train_bkg) * 1./len(train_bkg)
# test_sig_w  = np.ones_like(test_sig)  * 1./len(test_sig )
# test_bkg_w  = np.ones_like(test_bkg)  * 1./len(test_bkg )
# 
# ks_sig = ks_w2(train_sig, test_sig, train_sig_w, test_sig_w)
# ks_bkg = ks_w2(train_bkg, test_bkg, train_bkg_w, test_bkg_w)
# plt.suptitle('KS p-value: sig = %.3f%s - bkg = %.2f%s' %(ks_sig * 100., '%', ks_bkg * 100., '%'))

plt.savefig(model_path + 'overtrain_%s.png' %tag, dpi=400)

plt.yscale('log')

plt.savefig(model_path + 'overtrain_log_%s.png' %tag, dpi=400)

plt.clf()
##########################################################################################
#####   FEATURE IMPORTANCE
##########################################################################################
for mode in modes.keys():
    if mode == "GNN":
        bdt_features = features + ['GNN_score']
    elif mode == "ISO":
        bdt_features = features + ['relative_iso']

    fscores = OrderedDict(zip(bdt_features, np.zeros(len(bdt_features))))
    classifiers = both_models[mode]
    for i, iclas in classifiers.items():
        myscores = iclas.get_booster().get_fscore()
        for jj in myscores.keys():
            fscores[jj] += myscores[jj]

    totalsplits = np.sum(list(fscores.values()))

    for k, v in fscores.items():
        fscores[k] = v/totalsplits 

    plt.xlabel('relative F-score')
    plt.ylabel('feature')
    plt.title(f'Feature Importance - BDT with {mode}')

    orderedfscores = OrderedDict(sorted(fscores.items(), key=lambda x : x[1], reverse=False ))

    bars = [modes[mode][k] for k in orderedfscores.keys()]
    y_pos = np.arange(len(bars))
    
    # Create horizontal bars
    plt.barh(y_pos, orderedfscores.values())
    
    # Create names on the y-axis
    plt.yticks(y_pos, bars)

    # plot_importance(clf)
    plt.tight_layout()
    plt.savefig(model_path + mode + '_feat_importance_%s.png' %tag, dpi=400)
    plt.clf()

# 
# ##########################################################################################
# #####   OVERTRAINING SCORE
# ##########################################################################################
# plt.clf()
# 
# auc_train = clf.evals_result()['validation_0']['auc']
# auc_test  = clf.evals_result()['validation_1']['auc']
# 
# n_estimators = np.arange(len(auc_train))
# 
# plt.plot(n_estimators, auc_train, color='r', label='AUC train')
# plt.plot(n_estimators, auc_test , color='b', label='AUC test' )
# 
# plt.xlabel('# tree')
# plt.ylabel('Area Under ROC')
# 
# plt.xscale('log')
# # plt.yscale('log')
# plt.tight_layout()
# plt.grid()
# 
# # plt.xlim([1, 1000])
# # plt.ylim([0.985, 1.0])
# plt.ylim([0.95, 1.0])
# 
# plt.legend(loc='best')
# 
# plt.savefig('auc_score_%s.pdf' %tag)
# plt.clf()

##########################################################################################
#####   CORRELATION MATRIX 
##########################################################################################
for mode in modes.keys():
    if mode == "GNN":
        bdt_features = features + ['GNN_score']
    elif mode == "ISO":
        bdt_features = features + ['relative_iso']

##########################################################################################
#####   CORRELATION MATRIX SIGNAL
##########################################################################################
    # Compute the correlation matrix for the signal
    corr = sig[bdt_features + ['bdt_' + mode, 'Slimmed_TauTo3Mu_fitted_mass']].corr()
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(corr, cmap=cmap, vmax=1., vmin=-1, center=0, annot=True, fmt='.2f',
                    square=True, linewidths=.8, cbar_kws={"shrink": .8})

    # rotate axis labels
    g.set_xticklabels(modes[mode].values(), rotation='vertical')
    g.set_yticklabels(modes[mode].values(), rotation='horizontal')

    # plt.show()
    plt.title(f'linear correlation matrix - signal - BDT with {mode}')
    plt.tight_layout()
    plt.savefig(model_path + mode + '_corr_sig_%s.png' %tag, dpi=400)
    plt.clf()

##########################################################################################
#####   CORRELATION MATRIX BACKGROUND
##########################################################################################
    # Compute the correlation matrix for the signal
    corr = bkg[bdt_features + ['bdt_' + mode, 'Slimmed_TauTo3Mu_fitted_mass']].corr()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(corr, cmap=cmap, vmax=1., vmin=-1, center=0, annot=True, fmt='.2f',
                    square=True, linewidths=.8, cbar_kws={"shrink": .8})

    # rotate axis labels
    g.set_xticklabels(modes[mode].values(), rotation='vertical')
    g.set_yticklabels(modes[mode].values(), rotation='horizontal')

    # plt.show()
    plt.title(f'linear correlation matrix - background - BDT with {mode}')
    plt.tight_layout()
    plt.savefig(model_path + mode + '_corr_bkg_%s.png' %tag, dpi=400)