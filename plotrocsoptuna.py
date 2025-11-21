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

dataoptunalrplateau1 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs1/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau2 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs2/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau3 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs3/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau4 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs4/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau5 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs5/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau6 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs6/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau7 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs7/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau8 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs8/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau9 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs9/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau10 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs10/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau11 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs11/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau12 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs12/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau13 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs13/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau14 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs14/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau15 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs15/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau16 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs16/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau17 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs17/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau18 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs18/eval_ROC_AUC.pkl", 'rb'))
dataoptunalrplateau19 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/optunacodelrplateau_vs19/eval_ROC_AUC.pkl", 'rb'))





fprdataoptunalrplateau1 = dataoptunalrplateau1['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau2 = dataoptunalrplateau2['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau3 = dataoptunalrplateau3['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau4 = dataoptunalrplateau4['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau5 = dataoptunalrplateau5['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau6 = dataoptunalrplateau6['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau7 = dataoptunalrplateau7['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau8 = dataoptunalrplateau8['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau9 = dataoptunalrplateau9['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau10 = dataoptunalrplateau10['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau11 = dataoptunalrplateau11['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau12 = dataoptunalrplateau12['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau13 = dataoptunalrplateau13['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau14 = dataoptunalrplateau14['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau15 = dataoptunalrplateau15['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau16 = dataoptunalrplateau16['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau17 = dataoptunalrplateau17['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau18 = dataoptunalrplateau18['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fprdataoptunalrplateau19 = dataoptunalrplateau19['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']



tprdataoptunalrplateau1 = dataoptunalrplateau1['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau2 = dataoptunalrplateau2['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau3 = dataoptunalrplateau3['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau4 = dataoptunalrplateau4['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau5 = dataoptunalrplateau5['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau6 = dataoptunalrplateau6['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau7 = dataoptunalrplateau7['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau8 = dataoptunalrplateau8['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau9 = dataoptunalrplateau9['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau10 = dataoptunalrplateau10['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau11 = dataoptunalrplateau11['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau12 = dataoptunalrplateau12['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau13 = dataoptunalrplateau13['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau14 = dataoptunalrplateau14['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau15 = dataoptunalrplateau15['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau16 = dataoptunalrplateau16['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau17 = dataoptunalrplateau17['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau18 = dataoptunalrplateau18['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tprdataoptunalrplateau19 = dataoptunalrplateau19['no_puppi_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']


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


trials = {'hidden_dim': [1491, 937, 1649, 396, 1470, 459, 1405, 1688, 575, 1882, 1014, 1407, 2027, 2019, 1242, 708, 147, 1195, 1188, 856],     
          'layers': [6, 7, 4, 7, 4, 5, 6, 6, 6, 6, 4, 5, 5, 5, 5, 5, 5, 4, 4, 4],
          'fcparam': [35, 64, 46, 50, 32, 50, 52, 90, 35, 77, 71, 51, 59, 64, 84, 97, 84, 2, 2, 75], 
          'dropout': [0.2704947245808065, 0.49591905438302125, 0.1549698171972044, 0.3222105644870486, 0.01221123722011317, 0.03951836713542996, 0.44457762335953044, 0.3924953902553264, 0.08804987300988919, 0.598092272553284, 0.7882700097272095, 0.21746654252127326, 0.19475616951990174, 0.18774924356575748, 0.23785681884696608, 0.3227523026037715, 0.1364063041685744, 0.5701129068728891, 0.6169119010803057, 0.6498769495389107], 
          'lr': [0.06478824906326734, 0.07813487189444218, 0.03418588766490392, 0.06574707250193298, 0.005898439597867514, 0.005721363735348228, 0.01604272836984895, 0.0012458135467871266, 0.014867817637342998, 0.002841590668508556, 0.030323138661352292, 0.023908041469768146, 0.026460366659964506, 0.028306761580455544, 0.010409003024260908, 0.007830621831789916, 0.003009006182583328, 0.010778564189478616, 0.011150597540537554, 0.0035198890245710063],
          'auc': [0.8254917208717507, 0.8119965487739129, 0.8417704027164967, 0.6952061444247087, 0.8192276546846047, 0.809154562249139, 0.8353750902021226, 0.7949048658246002 , 0.8190752006520537, 0.80840932887201, 0.8260851608399267, 0.842492512350167, 0.8446476721532956, 0.8071157245784069, 0.866266780500973, 0.8134446421330908, 0.7862309500861939, 0.9023312709877027, 0.9232433629090284, 0.814127099660456],
          #"fpr": [fprdataoptunalrplateau49, fprdataoptunalrplateau50, fprdataoptunalrplateau51, fprdataoptunalrplateau52, fprdataoptunalrplateau53, fprdataoptunalrplateau54, fprdataoptunalrplateau55, fprdataoptunalrplateau56, fprdataoptunalrplateau57, fprdataoptunalrplateau58], 
          #"tpr" : [tprdataoptunalrplateau49, tprdataoptunalrplateau50, tprdataoptunalrplateau51, tprdataoptunalrplateau52, tprdataoptunalrplateau53, tprdataoptunalrplateau54, tprdataoptunalrplateau55, tprdataoptunalrplateau56, tprdataoptunalrplateau57, tprdataoptunalrplateau58],
}

df = pd.DataFrame(data = trials)
fig = px.parallel_coordinates(df, color="auc", labels={"auc": "auc",
                "hidden_dim": "Initial hidden dim", "layers": "Number of layers",
                "fcparam": "fully connected param", "dropout": "Dropout", 
                "lr" : "Learning rate"},
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             range_color=[0.7, 0.9])


output_path = utrain.output_directory("comparison/"+"Parallel_coordinates_optuna_first")
fig.write_image(output_path + "/Parallel_coordinates_optuna_first.png")


#ueval.plot_roc(trials, "comparison_optunacodelrplateau_semilogx")
