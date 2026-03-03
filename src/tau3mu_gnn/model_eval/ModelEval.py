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

def plot_roc(data, key, root_dir, thresholds_bench, title, tau_channel):
    thresh_benchmark = data[key]["treshold_benchmark"]
    fpr = data[key]["fpr"]
    tpr = data[key]["tpr"]
    fpr_bench = data[key]["fpr_benchmark"]
    tpr_bench = data[key]["tpr_benchmark"]
    thresh_model = data[key]["threshold"]

    ax = plt.gca()
    if tau_channel == "HF":
        plt.plot(fpr, tpr, linewidth=0.9, label=f"AUC: {data[key]['auc']:.3f}, " + key.replace("D_s_decay_signal_skimmed_", ""))
    elif tau_channel == "W":
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
    output_plot_name = "ROC_log.png"
    print(f"{output_plot_name} has been saved.")
    plt.savefig(os.path.join(root_dir,output_plot_name), dpi=400)
    plt.close()

def plot_roc_bdt(data, root_dir, thresholds_bench, title, tau_channel):
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


def plot_roc_gnn_arch_comparison(gnn_data_locs, bdt_data_loc, root_dir, thresholds_bench, tau_channel, sculpt_degree, lambdas):

    ax = plt.gca()
    first_ROC = True
    tpr_targets = []
    
    for i, gnn_data_loc in enumerate(gnn_data_locs):
        gnn_data = load_pkl(gnn_data_loc, "eval_ROC_AUC.pkl")
        gnn_data = next(iter(gnn_data[0].values()))
        
        fpr = gnn_data["fpr"]
        tpr = gnn_data["tpr"]
        thresh_model = gnn_data["threshold"]

        line, = plt.plot(fpr, tpr, linewidth=0.9, label=f"{gnn_data['gnn_mode']} $\lambda$ = {lambdas[i]}, AUC: {gnn_data['auc']:.3f}")
        curve_color = line.get_color()

        for i, threshold in enumerate(thresholds_bench):
            # GNN model point
            if first_ROC:
                thresh_idx_model = min(range(len(thresh_model)), key=lambda j: abs(thresh_model[j] - threshold))
                x_model, y_model = fpr[thresh_idx_model], tpr[thresh_idx_model]
                tpr_targets.append(y_model)

            # GNN model point matching
            else:
                tpr_target = tpr_targets[i]
                thresh_idx_model = min(range(len(tpr)), key=lambda j: abs(tpr[j] - tpr_target))
                x_model, y_model = fpr[thresh_idx_model], tpr[thresh_idx_model]
            
            plt.scatter(x_model, y_model, s=20, color=curve_color)
            plt.text(x_model * 1.1, y_model, f"{thresh_model[thresh_idx_model]:.3f}", fontsize=6, color=curve_color)
        
        first_ROC = False

    if bdt_data_loc is not None:
        bdt_data = np.load(bdt_data_loc + '_roc_log24feb2026_v0.pck', allow_pickle=True)
        thresh_bdt = bdt_data["threshold_GNN"]
        fpr_bdt = bdt_data["fpr_GNN"]
        tpr_bdt = bdt_data["tpr_GNN"]

        thresh_benchmark = bdt_data["threshold_ISO"]
        fpr_bench = bdt_data["fpr_ISO"]
        tpr_bench = bdt_data["tpr_ISO"]

        bdt_line, = plt.plot(fpr_bdt, tpr_bdt, linewidth=0.9, label=f"BDT GNN $\lambda$ = {lambdas[-1]}, AUC: {bdt_data['auc_GNN']:.3f}")
        bench_line, = plt.plot(fpr_bench, tpr_bench, linewidth=0.9, label=f"BDT ISO, AUC: {bdt_data['auc_ISO']:.3f}")

        bdt_color = bdt_line.get_color()
        bench_color = bench_line.get_color()

        for i, threshold in enumerate(thresholds_bench):
            tpr_target = tpr_targets[i]

            thresh_idx_model = min(range(len(tpr_bdt)), key=lambda j: abs(tpr_bdt[j] - tpr_target))
            x_model, y_model = fpr_bdt[thresh_idx_model], tpr_bdt[thresh_idx_model]
                
            plt.scatter(x_model, y_model, s=20, color=bdt_color)
            plt.text(x_model * 1.1, y_model, f"{thresh_bdt[thresh_idx_model]:.3f}", fontsize=6, color=bdt_color)

            thresh_idx_model = min(range(len(tpr_bench)), key=lambda j: abs(tpr_bench[j] - tpr_target))
            x_model, y_model = fpr_bench[thresh_idx_model], tpr_bench[thresh_idx_model]
                
            plt.scatter(x_model, y_model, s=20, color=bench_color)
            plt.text(x_model * 1.1, y_model, f"{thresh_benchmark[thresh_idx_model]:.3f}", fontsize=6, color=bench_color)

    plt.grid()
    plt.legend(fontsize=7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xscale('log')
    plt.xlim(1e-5, 1)
    plt.title(f"ROC Curves: different GNN archs vs BDT with GNN score for {tau_channel} channel")

    output_plot_name = f"GNNs_ROC_log_comparison_{tau_channel}_sculpt_{sculpt_degree}.png"
    print(f"{output_plot_name} has been saved.")
    plt.savefig(os.path.join(root_dir, output_plot_name), dpi=400)
    plt.close()

def tau_score_hist(root, pickl_name, mode, title, tau_channel):
    data = np.load(root + pickl_name, allow_pickle=True)
    scores = data["scores"]
    f_tau_mass = data["tau_mass"]
    print("data shape is", len(f_tau_mass))

    score_cuts = [0.95, 0.97, 0.98, 0.99]
    bins = np.linspace(1.40, 2.50, 110)
    for cut in score_cuts:
        mask = scores > cut
        filtered_data = ak.to_numpy(ak.concatenate([f_tau_mass[mask]]))
        plt.hist(filtered_data, bins=bins, alpha=0.6, label=f"GNN Score > {cut}")
    plt.title(f"Fitted Tau Mass Distribution with Different Score Cuts of GNN score:\n {tau_channel} channel {mode}, {title}") 
    xarray = np.linspace(1.40, 2.50, 1000)
    plt.plot(xarray, 100*np.exp(-(xarray - 1.78)**2 / (2*0.02**2)))
    plt.xlabel(r"$m_{\tau,\mathrm{fit}}$ [GeV]")
    plt.ylabel("Counts")
    plt.xlim(1.40, 2.50)
    plt.grid()
    plt.legend()
    # plt.yscale('log')
    print(f"{mode}_tau_hist.png has been saved.")
    plt.savefig(os.path.join(root, f'{mode}_tau_hist.png'), dpi=400)
    plt.close()

def tau_score_hist_after(root, pickl_name, mode, title, tau_channel):
    data = load_pkl(root, pickl_name)[0]
    scores = data["GNN_score_bkg"]
    f_tau_mass = data["tau_mass"]
    print("data shape is", len(f_tau_mass))

    score_cuts = [0.95, 0.97, 0.98, 0.99]
    bins = np.linspace(1.40, 2.50, 110)
    for cut in score_cuts:
        mask = scores > cut
        filtered_data = ak.to_numpy(ak.concatenate([f_tau_mass[mask]]))
        plt.hist(filtered_data, bins=bins, alpha=0.6, label=f"BDT Score > {cut}")
    plt.title(f"Fitted Tau Mass Distribution with Different Score Cuts of BDT score:\n {tau_channel} channel {mode}, {title}") 
    xarray = np.linspace(1.40, 2.50, 1000)
    plt.plot(xarray, 100*np.exp(-(xarray - 1.78)**2 / (2*0.02**2)))
    plt.xlabel(r"$m_{\tau,\mathrm{fit}}$ [GeV]")
    plt.ylabel("Counts")
    plt.xlim(1.40, 2.50)
    plt.grid()
    plt.legend()
    # plt.yscale('log')
    print(f"{mode}_after_tau_hist.png has been saved.")
    plt.savefig(os.path.join(root, f'{mode}_after_tau_hist.png'), dpi=400)
    plt.close()


if __name__ == "__main__":
    # root_dir = '/work/kfuss/SVJ_GNN/output/knn_new_vs7'
    # data = load_pkl(root_dir, 'ROC_AUC.pkl')
    # plot_roc(data[0], "GNN", root_dir, [-0.2,-0.3,-0.5, -0.75, -0.9], r"$\lambda=0.5$ kNN $k=10$")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=False, help="Root directory of results")
    parser.add_argument("--title", type=str, required=False, help="Plot title")
    parser.add_argument("--tau_channel", type=str, required=True, help="Tau production channel")
    parser.add_argument("--gnn_mode", type=str, required=False)
    parser.add_argument("--plot_mode", type=str, required=True)
    parser.add_argument("--sculpt_degree", required=False)
    parser.add_argument("--gnn_archs_location", required=False, nargs='+')
    parser.add_argument("--bdt_location", default=None)
    parser.add_argument("--lambdas", required=False, nargs='+')

    args = parser.parse_args()
    output_path = args.output_path
    title = args.title
    tau_channel = args.tau_channel
    gnn_mode = args.gnn_mode
    plot_mode = args.plot_mode
    sculpt_degree = args.sculpt_degree
    gnn_archs_location = args.gnn_archs_location
    bdt_location = args.bdt_location
    lambdas = args.lambdas

    if plot_mode == "tau_hist":
        tau_score_hist(output_path, "raw_scores_and_tau_mass_background.npz", gnn_mode, title, tau_channel)
    if plot_mode == "tau_hist_after":
        tau_score_hist_after(output_path, "tau_score_corr.pkl", gnn_mode, title, tau_channel)

    elif plot_mode == "ROC_comp":
        root_dir = "/work/kfuss/SVJ_GNN/output/"
        thresholds = [0.5, 0.85, 0.95, 0.99, 0.999]
        plot_roc_gnn_arch_comparison(gnn_archs_location, bdt_location, root_dir, thresholds, tau_channel, sculpt_degree, lambdas)