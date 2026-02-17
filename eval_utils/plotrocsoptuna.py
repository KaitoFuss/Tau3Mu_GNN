import matplotlib.pyplot as plt

from comet_ml import Experiment

from utils.NormalizerUtils import OnTheFlyNormalizer
from data_loading.graphDataLoader import GraphDataLoader
from architectures.TauMuNet import TauMuNet
from tqdm import tqdm 

import time
import torch
import argparse
import os
import pickle
import numpy as np
import dcor
import pandas as pd
import plotly.express as px

import utils.EvaluationUtils as ueval
import utils.TrainingUtils as utrain

import linecache
from utils.ConeGraphProducer import ConeGraphProducer
import tracemalloc

dataoptunalrplateau1 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs144/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau2 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs145/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau3 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs146/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau4 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs147/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau5 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs148/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau6 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs149/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau7 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs150/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau8 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs151/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau9 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs152/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau10 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs153/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau11 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs154/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau12 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs155/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau13 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs156/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau14 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs157/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau15 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs158/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau16 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs159/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau17 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs160/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau18 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs161/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau19 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs162/eval_ROC_AUC.pkl", 'rb'))





fprdataoptunalrplateau1 = dataoptunalrplateau1['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau2 = dataoptunalrplateau2['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau3 = dataoptunalrplateau3['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau4 = dataoptunalrplateau4['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau5 = dataoptunalrplateau5['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau6 = dataoptunalrplateau6['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau7 = dataoptunalrplateau7['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau8 = dataoptunalrplateau8['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau9 = dataoptunalrplateau9['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau10 = dataoptunalrplateau10['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau11 = dataoptunalrplateau11['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau12 = dataoptunalrplateau12['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau13 = dataoptunalrplateau13['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau14 = dataoptunalrplateau14['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau15 = dataoptunalrplateau15['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau16 = dataoptunalrplateau16['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau17 = dataoptunalrplateau17['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau18 = dataoptunalrplateau18['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau19 = dataoptunalrplateau19['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']



tprdataoptunalrplateau1 = dataoptunalrplateau1['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau2 = dataoptunalrplateau2['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau3 = dataoptunalrplateau3['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau4 = dataoptunalrplateau4['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau5 = dataoptunalrplateau5['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau6 = dataoptunalrplateau6['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau7 = dataoptunalrplateau7['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau8 = dataoptunalrplateau8['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau9 = dataoptunalrplateau9['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau10 = dataoptunalrplateau10['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau11 = dataoptunalrplateau11['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau12 = dataoptunalrplateau12['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau13 = dataoptunalrplateau13['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau14 = dataoptunalrplateau14['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau15 = dataoptunalrplateau15['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau16 = dataoptunalrplateau16['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau17 = dataoptunalrplateau17['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau18 = dataoptunalrplateau18['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau19 = dataoptunalrplateau19['ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']


"""trials = {
    -auc : 0.8254917208717507 and parameters: {'-hidden_dim': 1491, '-layers': 6, '-fcparam': 35, '-dropout': 0.2704947245808065, '-lr': 0.06478824906326734

-auc : 0.8119965487739129 and parameters: {'-hidden_dim': 937, '-layers': 7, '-fcparam': 64, '-dropout': 0.49591905438302125, '-lr': 0.07813487189444218 

-auc : 0.8417704027164967 and parameters: {'-hidden_dim': 1649, '-layers': 4, '-fcparam': 46, '-dropout': 0.1549698171972044, '-lr': 0.03418588766490392 

-auc : 0.6952061444247087 and parameters: {'-hidden_dim': 396, '-layers': 7, '-fcparam': 50, '-dropout': 0.3222105644870486, '-lr': 0.06574707250193298 

-auc : 0.8192276546846047 and parameters: {'-hidden_dim': 1470, '-layers': 4, '-fcparam': 32, '-dropout': 0.01221123722011317, '-lr': 0.005898439597867514 

-auc : 0.809154562249139 and parameters: {'-hidden_dim': 459, '-layers': 5, '-fcparam': 50, '-dropout': 0.03951836713542996, '-lr': 0.005721363735348228 

-auc : 0.8353750902021226 and parameters: {'-hidden_dim': 1405, '-layers': 6, '-fcparam': 52, '-dropout': 0.44457762335953044, '-lr': 0.01604272836984895 

-auc : 0.7949048658246002  and parameters: {'-hidden_dim': 1688, '-layers': 6, '-fcparam': 90, '-dropout': 0.3924953902553264, '-lr': 0.0012458135467871266 

-auc : 0.8190752006520537 and parameters: {'-hidden_dim': 575, '-layers': 6, '-fcparam': 35, '-dropout': 0.08804987300988919, '-lr': 0.014867817637342998 

-auc : 0.80840932887201 and parameters: {'-hidden_dim': 1882, '-layers': 6, '-fcparam': 77, '-dropout': 0.598092272553284, '-lr': 0.002841590668508556 

-auc : 0.8260851608399267 and parameters: {'-hidden_dim': 1014, '-layers': 4, '-fcparam': 71, '-dropout': 0.7882700097272095, '-lr': 0.030323138661352292 

-auc : 0.842492512350167 and parameters: {'-hidden_dim': 1407, '-layers': 5, '-fcparam': 51, '-dropout': 0.21746654252127326, '-lr': 0.023908041469768146 

-auc : 0.8446476721532956 and parameters: {'-hidden_dim': 2027, '-layers': 5, '-fcparam': 59, '-dropout': 0.19475616951990174, '-lr': 0.026460366659964506

-auc : 0.8071157245784069 and parameters: {'-hidden_dim': 2019, '-layers': 5, '-fcparam': 64, '-dropout': 0.18774924356575748, '-lr': 0.028306761580455544 

-auc : 0.866266780500973 and parameters: {'-hidden_dim': 1242, '-layers': 5, '-fcparam': 84, '-dropout': 0.23785681884696608, '-lr': 0.010409003024260908 

-auc : 0.8134446421330908 and parameters: {'-hidden_dim': 708, '-layers': 5, '-fcparam': 97, '-dropout': 0.3227523026037715, '-lr': 0.007830621831789916 

-auc : 0.7862309500861939 and parameters: {'-hidden_dim': 147, '-layers': 5, '-fcparam': 84, '-dropout': 0.1364063041685744, '-lr': 0.003009006182583328 

-auc : 0.9023312709877027 and parameters: {'-hidden_dim': 1195, '-layers': 4, '-fcparam': 2, '-dropout': 0.5701129068728891, '-lr': 0.010778564189478616 

-auc : 0.9232433629090284 and parameters: {'-hidden_dim': 1188, '-layers': 4, '-fcparam': 2, '-dropout': 0.6169119010803057, '-lr': 0.011150597540537554 

-auc : 0.814127099660456 and parameters: {'-hidden_dim': 856, '-layers': 4, '-fcparam': 75, '-dropout': 0.6498769495389107, '-lr': 0.0035198890245710063 
}"""


"""trials = {
    "try1" : {"fpr" : fprdataoptunalrplateau1, "tpr" : tprdataoptunalrplateau1},
    "try2" : {"fpr" : fprdataoptunalrplateau2, "tpr" : tprdataoptunalrplateau2},
    "try3" : {"fpr" : fprdataoptunalrplateau3, "tpr" : tprdataoptunalrplateau3},
    "try4" : {"fpr" : fprdataoptunalrplateau4, "tpr" : tprdataoptunalrplateau4},
    "try5" : {"fpr" : fprdataoptunalrplateau5, "tpr" : tprdataoptunalrplateau5},
    "try6" : {"fpr" : fprdataoptunalrplateau6, "tpr" : tprdataoptunalrplateau6},
    "try7" : {"fpr" : fprdataoptunalrplateau7, "tpr" : tprdataoptunalrplateau7},
    "try8" : {"fpr" : fprdataoptunalrplateau8, "tpr" : tprdataoptunalrplateau8},
    "try9" : {"fpr" : fprdataoptunalrplateau9, "tpr" : tprdataoptunalrplateau9},
    "try10" : {"fpr" : fprdataoptunalrplateau10, "tpr" : tprdataoptunalrplateau10},
    "try11" : {"fpr" : fprdataoptunalrplateau11, "tpr" : tprdataoptunalrplateau11},
    "try12" : {"fpr" : fprdataoptunalrplateau12, "tpr" : tprdataoptunalrplateau12},
    "try13" : {"fpr" : fprdataoptunalrplateau13, "tpr" : tprdataoptunalrplateau13},
    "try14" : {"fpr" : fprdataoptunalrplateau14, "tpr" : tprdataoptunalrplateau14},
    "try15" : {"fpr" : fprdataoptunalrplateau15, "tpr" : tprdataoptunalrplateau15},
    "try16" : {"fpr" : fprdataoptunalrplateau16, "tpr" : tprdataoptunalrplateau16},
    "try17" : {"fpr" : fprdataoptunalrplateau17, "tpr" : tprdataoptunalrplateau17},
    "try18" : {"fpr" : fprdataoptunalrplateau18, "tpr" : tprdataoptunalrplateau18},
    "try19" : {"fpr" : fprdataoptunalrplateau19, "tpr" : tprdataoptunalrplateau19},
    }
"""


trials = {'hidden_dim': [1836, 2022, 1860, 2020, 1387, 452, 1610, 986, 525, 702, 1372, 1344, 1216, 882, 139, 190, 128, 1517, 1093, 809],
          'layers': [7, 5, 5, 5, 5, 5, 7, 6, 5, 6, 4, 4, 4, 4, 4, 6, 4, 4, 5, 6],
          'fcparam': [69, 57, 73, 33, 44, 63, 84, 35, 86, 41, 49, 99, 100, 99, 52, 49, 48, 58, 41, 58], 
          'dropout': [0.031176248396436713, 0.7837305039493212, 0.13447270979736317, 0.0022730812243362485, 0.46283285785219647, 0.5742993379211198, 0.6173425894340983, 0.5596275585754378, 0.09447805006655968, 0.3732597271140292, 0.32434802736728013, 0.23066286667311553, 0.26467003625778884, 0.21565823385186508, 0.45246357483005545, 0.47494641147390704, 0.45807011850607554, 0.7277322172868161, 0.47225698033683505, 0.6433469255219473], 
          'lr': [0.005686690942393673, 0.07115593698121218, 0.036050675196428864, 0.0016714679132356532, 0.06896506507150169, 0.0967019669161748, 0.0338084501507633, 0.0039942495640962975, 0.0025496459960568704, 0.009780819456807007, 0.01891449949388893, 0.02818927147208518, 0.03310645057167968, 0.015160060084976864, 0.05452363551692726, 0.06462848598914477, 0.054398889871199924, 0.012263196605264877, 0.09609933597798662, 0.02128535464206974],
          'auc': [0.37332666201599574, 0.5483962815433714, 0.6685340682234894, 0.5540907412035829, 0.7268135703615911, 0.5092616722181466, 0.5137460812415559, 0.41013264564055263, 0.5036261786190642, 0.42328429937234097, 0.6267280422171582, 0.704858136281488, 0.5088734306739539, 0.46494764585305015, 0.7255638698510538, 0.7189929835837392, 0.44963631959800016, 0.6609950516797806, 0.5381028275522098, 0.5466284086572106],
          #"fpr": [fprdataoptunalrplateau49, fprdataoptunalrplateau50, fprdataoptunalrplateau51, fprdataoptunalrplateau52, fprdataoptunalrplateau53, fprdataoptunalrplateau54, fprdataoptunalrplateau55, fprdataoptunalrplateau56, fprdataoptunalrplateau57, fprdataoptunalrplateau58], 
          #"tpr" : [tprdataoptunalrplateau49, tprdataoptunalrplateau50, tprdataoptunalrplateau51, tprdataoptunalrplateau52, tprdataoptunalrplateau53, tprdataoptunalrplateau54, tprdataoptunalrplateau55, tprdataoptunalrplateau56, tprdataoptunalrplateau57, tprdataoptunalrplateau58],
}

df = pd.DataFrame(data = trials)
fig = px.parallel_coordinates(df, color="auc", labels={"auc": "auc",
                "hidden_dim": "Initial hidden dim", "layers": "Number of layers",
                "fcparam": "fully connected param", "dropout": "Dropout", 
                "lr" : "Learning rate"},
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             range_color=[0.4, 0.75])


output_path = utrain.output_directory("comparison/"+"Parallel_coordinates_optuna_first")
fig.write_image(output_path + "/Parallel_coordinates_optuna_first.png")


#ueval.plot_roc(trials, "comparison_optunacodelrplateau_semilogx")
