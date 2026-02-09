# import pandas
import uproot
import awkward as ak
import argparse
import pickle
import numpy  as np
import pandas as pd
from sklearn.metrics import accuracy_score
# import matplotlib as mpl ; mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse



from sklearn.metrics         import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from scipy.stats import ks_2samp

from collections import OrderedDict, defaultdict
from itertools import product

from pdb import set_trace

signals = ["/pnfs/psi.ch/cms/trivcat/store/user/adcorel/riccdata/signal_ds/ds_large_signal_skimmed_no_resonances.root"]
data1 = ["/pnfs/psi.ch/cms/trivcat/store/user/adcorel/riccdata/data_ds/data_skimmed_no_resonances_part1.root"]
data2 = ["/pnfs/psi.ch/cms/trivcat/store/user/adcorel/riccdata/data_ds/data_skimmed_no_resonances_part2.root"]
data3 = ["/pnfs/psi.ch/cms/trivcat/store/user/adcorel/riccdata/data_ds/data_skimmed_no_resonances_part3.root"]
data4 = ["/pnfs/psi.ch/cms/trivcat/store/user/adcorel/riccdata/data_ds/data_skimmed_no_resonances_part4.root"]


branches = ['nSlimmedTauTo3Mu']

output_sig = []
output_data = []

print("Going to load signal events")
for events in uproot.iterate(f"{signals[0]}:Events", expressions=branches, library="ak", step_size="100 MB"):
    output_sig = events["nSlimmedTauTo3Mu"]

print(len(output_sig))
print(output_sig)
print(np.mean(output_sig))
relevant_sig = output_sig[output_sig != 1]
print(relevant_sig)
print(max(relevant_sig))
print(np.mean(relevant_sig))
print(len(relevant_sig))
print(len(relevant_sig)/len(output_sig))

print("Going to load data events number 1")
for events in uproot.iterate(f"{data1[0]}:Events", expressions=branches, library="ak", step_size="100 MB"):
    output_data1 = events["nSlimmedTauTo3Mu"]


print("Going to load data events number 2")
for events in uproot.iterate(f"{data2[0]}:Events", expressions=branches, library="ak", step_size="100 MB"):
    output_data2 = events["nSlimmedTauTo3Mu"]


print("Going to load data events number 3")
for events in uproot.iterate(f"{data3[0]}:Events", expressions=branches, library="ak", step_size="100 MB"):
    output_data3 = events["nSlimmedTauTo3Mu"]


print("Going to load data events number 4")
for events in uproot.iterate(f"{data4[0]}:Events", expressions=branches, library="ak", step_size="100 MB"):
    output_data4 = events["nSlimmedTauTo3Mu"]

output_data = ak.concatenate([output_data1, output_data2, output_data3, output_data4])
print(len(output_data1))
print(len(output_data))
print(output_data)
print(np.mean(output_data))
relevant_data = output_data[output_data != 1]
print(relevant_data)
print(max(relevant_data))
print(np.mean(relevant_data))
print(len(relevant_data))
print(len(relevant_data)/len(output_data))