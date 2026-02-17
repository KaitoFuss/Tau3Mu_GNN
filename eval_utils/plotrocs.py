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

import utils.EvaluationUtils as ueval
import utils.TrainingUtils as utrain

import linecache
from utils.ConeGraphProducer import ConeGraphProducer
import tracemalloc

data8knn1 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts1_vs6/eval_ROC_AUC.pkl", 'rb'))
data8knn2 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts2_vs6/eval_ROC_AUC.pkl", 'rb'))
data8knn3 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts3_vs6/eval_ROC_AUC.pkl", 'rb'))
data8knn4 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts4_vs6/eval_ROC_AUC.pkl", 'rb'))
data8knn5 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts5_vs5/eval_ROC_AUC.pkl", 'rb'))
data8knn6 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts6_vs5/eval_ROC_AUC.pkl", 'rb'))
data8knn7 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts7_vs5/eval_ROC_AUC.pkl", 'rb'))
data8knn8 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts8_vs5/eval_ROC_AUC.pkl", 'rb'))

data9knn1 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts1_vs7/eval_ROC_AUC.pkl", 'rb'))
data9knn2 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts2_vs7/eval_ROC_AUC.pkl", 'rb'))
data9knn3 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts3_vs7/eval_ROC_AUC.pkl", 'rb'))
data9knn4 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts4_vs7/eval_ROC_AUC.pkl", 'rb'))
data9knn5 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts5_vs6/eval_ROC_AUC.pkl", 'rb'))
data9knn6 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts6_vs6/eval_ROC_AUC.pkl", 'rb'))
data9knn7 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts7_vs6/eval_ROC_AUC.pkl", 'rb'))
data9knn8 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts8_vs6/eval_ROC_AUC.pkl", 'rb'))

data10knn1 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts1_vs8/eval_ROC_AUC.pkl", 'rb'))
data10knn2 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts2_vs8/eval_ROC_AUC.pkl", 'rb'))
data10knn3 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts3_vs8/eval_ROC_AUC.pkl", 'rb'))
data10knn4 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts4_vs8/eval_ROC_AUC.pkl", 'rb'))
data10knn5 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts5_vs7/eval_ROC_AUC.pkl", 'rb'))
data10knn6 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts6_vs7/eval_ROC_AUC.pkl", 'rb'))
data10knn7 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts7_vs7/eval_ROC_AUC.pkl", 'rb'))
data10knn8 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts8_vs7/eval_ROC_AUC.pkl", 'rb'))

data11knn1 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts1_vs9/eval_ROC_AUC.pkl", 'rb'))
data11knn2 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts2_vs9/eval_ROC_AUC.pkl", 'rb'))
data11knn3 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts3_vs9/eval_ROC_AUC.pkl", 'rb'))
data11knn4 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts4_vs9/eval_ROC_AUC.pkl", 'rb'))
data11knn5 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts5_vs8/eval_ROC_AUC.pkl", 'rb'))
data11knn6 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts6_vs8/eval_ROC_AUC.pkl", 'rb'))
data11knn7 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts7_vs8/eval_ROC_AUC.pkl", 'rb'))
data11knn8 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts8_vs8/eval_ROC_AUC.pkl", 'rb'))

data12knn1 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts1_vs10/eval_ROC_AUC.pkl", 'rb'))
data12knn2 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts2_vs10/eval_ROC_AUC.pkl", 'rb'))
data12knn3 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts3_vs10/eval_ROC_AUC.pkl", 'rb'))
data12knn4 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts4_vs10/eval_ROC_AUC.pkl", 'rb'))
data12knn5 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts5_vs9/eval_ROC_AUC.pkl", 'rb'))
data12knn6 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts6_vs9/eval_ROC_AUC.pkl", 'rb'))
data12knn7 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts7_vs9/eval_ROC_AUC.pkl", 'rb'))
data12knn8 = pickle.load(open(r"/work/adcorel/SVJ_GNN/output/tryouts8_vs9/eval_ROC_AUC.pkl", 'rb'))

print(data8knn1.keys())

fpr8knn1 = data8knn1['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['fpr']
fpr8knn2 = data8knn2['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['fpr']
fpr8knn3 = data8knn3['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['fpr']
fpr8knn4 = data8knn4['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['fpr']
fpr8knn5 = data8knn5['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['fpr']
fpr8knn6 = data8knn6['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['fpr']
fpr8knn7 = data8knn7['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['fpr']
fpr8knn8 = data8knn8['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['fpr']

fpr9knn1 = data9knn1['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['fpr']
fpr9knn2 = data9knn2['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['fpr']
fpr9knn3 = data9knn3['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['fpr']
fpr9knn4 = data9knn4['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['fpr']
fpr9knn5 = data9knn5['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['fpr']
fpr9knn6 = data9knn6['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['fpr']
fpr9knn7 = data9knn7['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['fpr']
fpr9knn8 = data9knn8['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['fpr']

fpr10knn1 = data10knn1['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fpr10knn2 = data10knn2['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fpr10knn3 = data10knn3['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fpr10knn4 = data10knn4['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fpr10knn5 = data10knn5['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fpr10knn6 = data10knn6['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fpr10knn7 = data10knn7['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']
fpr10knn8 = data10knn8['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['fpr']

fpr11knn1 = data11knn1['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['fpr']
fpr11knn2 = data11knn2['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['fpr']
fpr11knn3 = data11knn3['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['fpr']
fpr11knn4 = data11knn4['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['fpr']
fpr11knn5 = data11knn5['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['fpr']
fpr11knn6 = data11knn6['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['fpr']
fpr11knn7 = data11knn7['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['fpr']
fpr11knn8 = data11knn8['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['fpr']

fpr12knn1 = data12knn1['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['fpr']
fpr12knn2 = data12knn2['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['fpr']
fpr12knn3 = data12knn3['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['fpr']
fpr12knn4 = data12knn4['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['fpr']
fpr12knn5 = data12knn5['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['fpr']
fpr12knn6 = data12knn6['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['fpr']
fpr12knn7 = data12knn7['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['fpr']
fpr12knn8 = data12knn8['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['fpr']

tpr8knn1 = data8knn1['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['tpr']
tpr8knn2 = data8knn2['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['tpr']
tpr8knn3 = data8knn3['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['tpr']
tpr8knn4 = data8knn4['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['tpr']
tpr8knn5 = data8knn5['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['tpr']
tpr8knn6 = data8knn6['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['tpr']
tpr8knn7 = data8knn7['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['tpr']
tpr8knn8 = data8knn8['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_8']['tpr']

tpr9knn1 = data9knn1['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['tpr']
tpr9knn2 = data9knn2['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['tpr']
tpr9knn3 = data9knn3['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['tpr']
tpr9knn4 = data9knn4['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['tpr']
tpr9knn5 = data9knn5['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['tpr']
tpr9knn6 = data9knn6['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['tpr']
tpr9knn7 = data9knn7['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['tpr']
tpr9knn8 = data9knn8['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_9']['tpr']

tpr10knn1 = data10knn1['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tpr10knn2 = data10knn2['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tpr10knn3 = data10knn3['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tpr10knn4 = data10knn4['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tpr10knn5 = data10knn5['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tpr10knn6 = data10knn6['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tpr10knn7 = data10knn7['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']
tpr10knn8 = data10knn8['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_10']['tpr']

tpr11knn1 = data11knn1['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['tpr']
tpr11knn2 = data11knn2['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['tpr']
tpr11knn3 = data11knn3['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['tpr']
tpr11knn4 = data11knn4['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['tpr']
tpr11knn5 = data11knn5['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['tpr']
tpr11knn6 = data11knn6['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['tpr']
tpr11knn7 = data11knn7['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['tpr']
tpr11knn8 = data11knn8['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_11']['tpr']

tpr12knn1 = data12knn1['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['tpr']
tpr12knn2 = data12knn2['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['tpr']
tpr12knn3 = data12knn3['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['tpr']
tpr12knn4 = data12knn4['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['tpr']
tpr12knn5 = data12knn5['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['tpr']
tpr12knn6 = data12knn6['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['tpr']
tpr12knn7 = data12knn7['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['tpr']
tpr12knn8 = data12knn8['no_puppi_SMALL_ds_decay_signal_signal_skimmed_dzd0_cone_size_0.5_knn_12']['tpr']

arch1 = {
    "nopuppi_knn_8" : {"fpr" : fpr8knn1, "tpr" : tpr8knn1},
    "nopuppi_knn_9" : {"fpr" : fpr9knn1, "tpr" : tpr9knn1},
    "nopuppi_knn_10" : {"fpr" : fpr10knn1, "tpr" : tpr10knn1},
    "nopuppi_knn_11" : {"fpr" : fpr11knn1, "tpr" : tpr11knn1},
    "nopuppi_knn_12" : {"fpr" : fpr12knn1, "tpr" : tpr12knn1}
}

arch2 = {
    "nopuppi_knn_8" : {"fpr" : fpr8knn2, "tpr" : tpr8knn2},
    "nopuppi_knn_9" : {"fpr" : fpr9knn2, "tpr" : tpr9knn2},
    "nopuppi_knn_10" : {"fpr" : fpr10knn2, "tpr" : tpr10knn2},
    "nopuppi_knn_11" : {"fpr" : fpr11knn2, "tpr" : tpr11knn2},
    "nopuppi_knn_12" : {"fpr" : fpr12knn2, "tpr" : tpr12knn2}
}

arch3 = {
    "nopuppi_knn_8" : {"fpr" : fpr8knn3, "tpr" : tpr8knn3},
    "nopuppi_knn_9" : {"fpr" : fpr9knn3, "tpr" : tpr9knn3},
    "nopuppi_knn_10" : {"fpr" : fpr10knn3, "tpr" : tpr10knn3},
    "nopuppi_knn_11" : {"fpr" : fpr11knn3, "tpr" : tpr11knn3},
    "nopuppi_knn_12" : {"fpr" : fpr12knn3, "tpr" : tpr12knn3}
}

arch4 = {
    "nopuppi_knn_8" : {"fpr" : fpr8knn4, "tpr" : tpr8knn4},
    "nopuppi_knn_9" : {"fpr" : fpr9knn4, "tpr" : tpr9knn4},
    "nopuppi_knn_10" : {"fpr" : fpr10knn4, "tpr" : tpr10knn4},
    "nopuppi_knn_11" : {"fpr" : fpr11knn4, "tpr" : tpr11knn4},
    "nopuppi_knn_12" : {"fpr" : fpr12knn4, "tpr" : tpr12knn4}
}

arch5 = {
    "nopuppi_knn_8" : {"fpr" : fpr8knn5, "tpr" : tpr8knn5},
    "nopuppi_knn_9" : {"fpr" : fpr9knn5, "tpr" : tpr9knn5},
    "nopuppi_knn_10" : {"fpr" : fpr10knn5, "tpr" : tpr10knn5},
    "nopuppi_knn_11" : {"fpr" : fpr11knn5, "tpr" : tpr11knn5},
    "nopuppi_knn_12" : {"fpr" : fpr12knn5, "tpr" : tpr12knn5}
}

arch6 = {
    "nopuppi_knn_8" : {"fpr" : fpr8knn6, "tpr" : tpr8knn6},
    "nopuppi_knn_9" : {"fpr" : fpr9knn6, "tpr" : tpr9knn6},
    "nopuppi_knn_10" : {"fpr" : fpr10knn6, "tpr" : tpr10knn6},
    "nopuppi_knn_11" : {"fpr" : fpr11knn6, "tpr" : tpr11knn6},
    "nopuppi_knn_12" : {"fpr" : fpr12knn6, "tpr" : tpr12knn6}
}

arch7 = {
    "nopuppi_knn_8" : {"fpr" : fpr8knn7, "tpr" : tpr8knn7},
    "nopuppi_knn_9" : {"fpr" : fpr9knn7, "tpr" : tpr9knn7},
    "nopuppi_knn_10" : {"fpr" : fpr10knn7, "tpr" : tpr10knn7},
    "nopuppi_knn_11" : {"fpr" : fpr11knn7, "tpr" : tpr11knn7},
    "nopuppi_knn_12" : {"fpr" : fpr12knn7, "tpr" : tpr12knn7}
}

arch8 = {
    "nopuppi_knn_8" : {"fpr" : fpr8knn8, "tpr" : tpr8knn8},
    "nopuppi_knn_9" : {"fpr" : fpr9knn8, "tpr" : tpr9knn8},
    "nopuppi_knn_10" : {"fpr" : fpr10knn8, "tpr" : tpr10knn8},
    "nopuppi_knn_11" : {"fpr" : fpr11knn8, "tpr" : tpr11knn8},
    "nopuppi_knn_12" : {"fpr" : fpr12knn8, "tpr" : tpr12knn8}
}

ueval.plot_roc(arch1, "comparison_arch1_semilogx")
ueval.plot_roc(arch2, "comparison_arch2_semilogx")
ueval.plot_roc(arch3, "comparison_arch3_semilogx")
ueval.plot_roc(arch4, "comparison_arch4_semilogx")
ueval.plot_roc(arch5, "comparison_arch5_semilogx")
ueval.plot_roc(arch6, "comparison_arch6_semilogx")
ueval.plot_roc(arch7, "comparison_arch7_semilogx")
ueval.plot_roc(arch8, "comparison_arch8_semilogx")




