import os
import torch
import io
import pickle
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import argparse

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_pkl(root_dir, file_name_tail):
    data = []
    
    for file in os.listdir(root_dir):
        if file.endswith(file_name_tail):
            file_path = os.path.join(root_dir, file)
            print(file_path)
            with open(file_path, "rb") as f:
                try:
                    data.append(CPU_Unpickler(f).load())
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return data

def plot_roc(data, key, root_dir, thresholds_bench, title):
    thresh_benchmark = data[key]["treshold_benchmark"]
    fpr = data[key]["fpr"]
    tpr = data[key]["tpr"]
    fpr_bench = data[key]["fpr_benchmark"]
    tpr_bench = data[key]["tpr_benchmark"]
    thresh_model = data[key]["threshold"]

    ax = plt.gca()
    plt.plot(fpr, tpr, linewidth=0.9, label=f"AUC: {data[key]['auc']:.3f}, " + key.replace("W_decay_signal_skimmed_", ""))
    plt.plot(fpr_bench, tpr_bench, linewidth = 0.9, label=f"AUC: {data[key]['auc_benchmark']:.3f}, Benchmark")
    for threshold in thresholds_bench:
        thresh_idx = min(range(len(thresh_benchmark)), key=lambda i: abs(thresh_benchmark[i] - threshold))
        plt.scatter(fpr_bench[thresh_idx], tpr_bench[thresh_idx], s=20, label=f"Threshold bench: {-thresh_benchmark[thresh_idx]:.3f}")
        tpr_target = tpr_bench[thresh_idx]
        thresh_idx_model = min(range(len(tpr)), key=lambda i: abs(tpr[i] - tpr_target))
        plt.scatter(fpr[thresh_idx_model], tpr[thresh_idx_model], s=20,
            label=f"Threshold model={thresh_model[thresh_idx_model]:.3f}")

    plt.grid()
    plt.legend(fontsize=7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xscale('log')
    plt.xlim(1e-5,1)
    plt.title(title)
    outout_plot_name = "ROC_log.png"
    print(f"{outout_plot_name} has been saved.")
    plt.savefig(os.path.join(root_dir,outout_plot_name), dpi=400)
    plt.close()

def plot_roc_bdt(data, root_dir, thresholds_bench, title):
    thresh_benchmark = data["threshold_ISO"]
    fpr_bench = data["fpr_ISO"]
    tpr_bench = data["tpr_ISO"]
    thresh_model = data["threshold_GNN"]
    fpr = data["fpr_GNN"]
    tpr = data["tpr_GNN"]

    ax = plt.gca()
    plt.plot(fpr, tpr, linewidth=0.9, label=f"AUC: {data['auc_GNN']:.3f}, GNN")
    plt.plot(fpr_bench, tpr_bench, linewidth=0.9, label=f"AUC: {data['auc_ISO']:.3f}, ISO")

    for threshold in thresholds_bench:
        # Benchmark point
        thresh_idx = min(range(len(thresh_benchmark)), key=lambda i: abs(thresh_benchmark[i] - threshold))
        x_bench, y_bench = fpr_bench[thresh_idx], tpr_bench[thresh_idx]
        plt.scatter(x_bench, y_bench, s=20, color="C1")
        plt.text(x_bench * 1.1, y_bench, f"{thresh_benchmark[thresh_idx]:.3f}", fontsize=6, color="C1")

        # Corresponding GNN model point
        tpr_target = y_bench
        thresh_idx_model = min(range(len(tpr)), key=lambda i: abs(tpr[i] - tpr_target))
        x_model, y_model = fpr[thresh_idx_model], tpr[thresh_idx_model]
        plt.scatter(x_model, y_model, s=20, color="C0")
        plt.text(x_model * 1.1, y_model, f"{thresh_model[thresh_idx_model]:.3f}", fontsize=6, color="C0")

    plt.grid()
    plt.legend(fontsize=7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xscale('log')
    plt.xlim(1e-5, 1)
    plt.title("ROC Curve: GNN (" + title + ") vs ISO")

    output_plot_name = "BDT_ROC_log.png"
    print(f"{output_plot_name} has been saved.")
    plt.savefig(os.path.join(root_dir, output_plot_name), dpi=400)
    plt.close()

def tau_score_hist(root, pickl_name, mode, title):
    data = np.load(root + pickl_name, allow_pickle=True)
    scores = data[f"{mode}_score_bkg"]
    f_tau_mass = data["tau_mass"]

    # score_cuts = [0,0.2,0.3,0.5,0.75]
    # bins=np.linspace(1.5,2.5,200)
    score_cuts = [0.95, 0.97, 0.98, 0.99]
    bins = np.linspace(1.72, 1.85, 14)
    for cut in score_cuts:
        mask = scores > cut
        filtered_data = ak.to_numpy(ak.concatenate([f_tau_mass[mask]]))
        plt.hist(filtered_data, bins=bins, alpha=0.6, label=f"BDT Score > {cut}")
    if mode == "GNN":
        plt.title(f"Fitted Tau Mass Distribution with Different Score Cuts:\n {mode}, {title}")  
    else:
        plt.title(f"Fitted Tau Mass Distribution with Different Score Cuts:\n {mode}")
    plt.xlabel(r"$m_{\tau,\mathrm{fit}}$ [GeV]")
    plt.ylabel("Counts")
    plt.grid()
    plt.legend()
    # plt.yscale('log')
    print(f"{mode}_tau_hist.png has been saved.")
    plt.savefig(os.path.join(root, f'{mode}_tau_hist.png'), dpi=400)
    plt.close()

if __name__ == "__main__":
    # root_dir = '/work/kfuss/SVJ_GNN/output/knn_new_vs7'
    # data = load_pkl(root_dir, 'ROC_AUC.pkl')
    # plot_roc(data[0], "GNN", root_dir, [-0.2,-0.3,-0.5, -0.75, -0.9], r"$\lambda=0.5$ kNN $k=10$")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True, help="Root directory of results")
    parser.add_argument("--title", type=str, required=True, help="Plot title")

    args = parser.parse_args()
    root_path = args.root_path
    title = args.title

    data = np.load(root_path + '_roc_21may2025_v0.pck', allow_pickle=True)
    plot_roc_bdt(data, root_path, [0.5, 0.75, 0.9, 0.95, 0.97, 0.98, 0.985, 0.99], title)
    tau_score_hist(root_path, "tau_score_corr.pkl", "GNN", title)
    tau_score_hist(root_path, "tau_score_corr.pkl", "ISO", title)
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import awkward as ak

    # data = np.load("output/knn_new_vs7/raw_scores_and_tau_mass.npz")
    # scores = data["scores"]
    # f_tau_mass = data["tau_mass"]

    # score_cuts = [0,0.2,0.3,0.5,0.75]
    # bins=np.linspace(0,2,100)
    # for cut in score_cuts:
    #     mask = scores > cut
    #     filtered_data = ak.to_numpy(ak.concatenate([f_tau_mass[mask]]))
    #     plt.hist(filtered_data, bins=bins, alpha=0.6, label=f"GNN Score > {cut}")
    # plt.title(r"$m_{\tau,\text{fit}}$ distribution with different Score Cuts $\lambda=0.5$ knn $k=10$")
    # plt.xlabel("Fitted Tau Mass")
    # plt.ylabel("Counts")
    # plt.grid()
    # plt.legend()
    # #plt.xlim((1.65,1.9))
    # plt.xlim((0.25,2))
    # plt.yscale('log')
    # plt.savefig('/work/kfuss/SVJ_GNN/output/knn_new_vs7/tau_hist.png', dpi=400)