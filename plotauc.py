import utils.EvaluationUtils as eval
import pickle

pkl_path = "output/puppi_optunabest_newdata_vs35/eval_ROC_AUC.pkl"

with open(pkl_path, "rb") as f:
    auc_dict = pickle.load(f)

auc_values = {name: d["auc"] for name, d in auc_dict.items()}


eval.plot_auc_simple(auc = auc_values, name = "puppi_optunabest_newdata_vs35", suffix = "_eval")