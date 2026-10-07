import os
import torch
import io
import pickle
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_curve, auc
import uproot

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

def plot_roc_gnn_arch_comparison_full(gnn_data_locs, bdt_data_loc, root_dir, fpr_targets, tau_channel, sculpt_degree, lambdas, archs, bdt_score_cut):

    suffix = "_with_bdt_cut" if bdt_score_cut else "" 
    ax = plt.gca()
    first_ROC = True
    tpr_targets = []
    fprs = []
    tprs = []
    threshs = []
    
    for i in range(len(gnn_data_locs)-1):
        gnn_data_loc = gnn_data_locs[i]
        gnn_sig = np.load(gnn_data_loc + f'raw_scores_and_tau_mass_signal{suffix}.npz', allow_pickle=True)
        gnn_bkg = np.load(gnn_data_loc + f'raw_scores_and_tau_mass_background{suffix}.npz', allow_pickle=True)

        gnn_sig_scores = gnn_sig["scores"]
        gnn_bkg_scores = gnn_bkg["scores"]
        
        tau_mass = gnn_bkg["tau_mass"]
        sideband_mask = (
            (np.abs(tau_mass - 1.78) > 0.08)
            & (np.abs(tau_mass) > 1.6)
            & (np.abs(tau_mass) < 2.0)
        )
        gnn_bkg_scores = gnn_bkg_scores[sideband_mask]
        
        gnn_sig_scores = gnn_sig_scores[~np.isnan(gnn_sig_scores)]
        gnn_bkg_scores = gnn_bkg_scores[~np.isnan(gnn_bkg_scores)]

        print(gnn_data_loc)
        print(f"Size of signal dataset is {len(gnn_sig_scores)}.")
        print(f"Size of background dataset is {len(gnn_bkg_scores)}.")

        labels = np.concatenate([np.ones(len(gnn_sig_scores)), np.zeros(len(gnn_bkg_scores))])
        scores_all = np.concatenate([gnn_sig_scores, gnn_bkg_scores])

        fpr, tpr, thresh_model = roc_curve(labels, scores_all)
        auc_full = auc(fpr, tpr)

        if archs and i < len(archs):
            line, = plt.plot(fpr, tpr, linewidth=0.9, label=f"{archs[i].replace('vs', '')} $\lambda$ = {lambdas[i]}, AUC: {auc_full:.3f}")
            fprs.append(fpr)
            tprs.append(tpr)
            threshs.append(thresh_model)
        else:
            continue
            # line, = plt.plot(fpr, tpr, linewidth=0.9, label=f"GNN for BDT $\lambda$ = {lambdas[i]}, AUC: {auc_full:.3f}")
        curve_color = line.get_color()

        for k, fpr_cut in enumerate(fpr_targets):
            # GNN model point
            if first_ROC:
                thresh_idx_model = np.argmin(np.abs(fpr - fpr_cut))
                x_model, y_model = fpr[thresh_idx_model], tpr[thresh_idx_model]
                tpr_targets.append(y_model)

            # GNN model point matching
            else:
                tpr_target = tpr_targets[k]
                thresh_idx_model = min(range(len(tpr)), key=lambda j: abs(tpr[j] - tpr_target))
                x_model, y_model = fpr[thresh_idx_model], tpr[thresh_idx_model]
            
            plt.scatter(x_model, y_model, s=20, color=curve_color)
            plt.text(x_model * 1.1, y_model, f"{thresh_model[thresh_idx_model]:.5f}", fontsize=6, color=curve_color)
        
        first_ROC = False

    if bdt_data_loc is not None:
        data = load_pkl(bdt_data_loc, "tau_score_corr.pkl")[0]
        tau_mass = data["tau_mass"]
        sideband_mask = (
            (np.abs(tau_mass - 1.78) > 0.08)
            & (np.abs(tau_mass) > 1.6)
            & (np.abs(tau_mass) < 2.0)
        )
        GNN_bkg = data["GNN_score_bkg"]
        GNN_sig = data["GNN_score_sig"]
        GNN_scores = np.concatenate([GNN_sig, GNN_bkg])
        GNN_labels = np.concatenate([np.ones(len(GNN_sig)), np.zeros(len(GNN_bkg))])

        fpr_bdt, tpr_bdt, thresh_bdt = roc_curve(GNN_labels, GNN_scores)
        auc_bdt = auc(fpr_bdt, tpr_bdt)

        fprs.append(fpr_bdt)
        tprs.append(tpr_bdt)
        threshs.append(thresh_bdt)

        ISO_bkg = data["ISO_score_bkg"][sideband_mask]
        ISO_sig = data["ISO_score_sig"]

        ISO_bkg = ISO_bkg[ISO_bkg > 0.8]
        ISO_sig = ISO_sig[ISO_sig > 0.8]

        ISO_scores = np.concatenate([ISO_sig, ISO_bkg])
        ISO_labels = np.concatenate([np.ones(len(ISO_sig)), np.zeros(len(ISO_bkg))])

        fpr_bench, tpr_bench, thresh_benchmark = roc_curve(ISO_labels, ISO_scores)
        auc_bench = auc(fpr_bench, tpr_bench)

        fprs.append(fpr_bench)
        tprs.append(tpr_bench)
        threshs.append(thresh_benchmark)
        
        # bdt_line, = plt.plot(fpr_bdt, tpr_bdt, linewidth=0.9, label=f"BDT $\lambda$ = {lambdas[-1]}, AUC: {auc_bdt:.3f}")
        bench_line, = plt.plot(fpr_bench, tpr_bench, linewidth=0.9, label=f"BDT ISO, AUC: {auc_bench:.3f}")

        # bdt_color = bdt_line.get_color()
        bench_color = bench_line.get_color()

        for i, _ in enumerate(fpr_targets):
            tpr_target = tpr_targets[i]

            thresh_idx_model = min(range(len(tpr_bdt)), key=lambda j: abs(tpr_bdt[j] - tpr_target))
            x_model, y_model = fpr_bdt[thresh_idx_model], tpr_bdt[thresh_idx_model]
                
            # plt.scatter(x_model, y_model, s=20, color=bdt_color)
            # plt.text(x_model * 1.1, y_model, f"{thresh_bdt[thresh_idx_model]:.5f}", fontsize=6, color=bdt_color)

            thresh_idx_model = min(range(len(tpr_bench)), key=lambda j: abs(tpr_bench[j] - tpr_target))
            x_model, y_model = fpr_bench[thresh_idx_model], tpr_bench[thresh_idx_model]
                
            plt.scatter(x_model, y_model, s=20, color=bench_color)
            plt.text(x_model * 1.1, y_model, f"{thresh_benchmark[thresh_idx_model]:.5f}", fontsize=6, color=bench_color)

    plt.grid()
    plt.legend(fontsize=7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xscale('log')
    plt.xlim(1e-5, 1)
    plt.title(f"ROC Curves: different GNN archs vs BDT with for {tau_channel} channel")

    output_plot_name = f"GNNs_ROC_log_comparison_{tau_channel}_sculpt_{sculpt_degree}{suffix}_full.png"
    print(f"{output_plot_name} has been saved.")
    plt.savefig(os.path.join(root_dir, output_plot_name), dpi=400)
    plt.close()
    
    return fprs, tprs, threshs

def plot_sic_full(fprs, tprs, threshs, root_dir, tau_channel, sculpt_degree, lambdas, archs, bdt_score_cut):
    suffix = "_with_bdt_cut" if bdt_score_cut else "" 
    bdt = True
    for i, (fpr, tpr, thresh) in enumerate(zip(fprs, tprs, threshs)):
        #calculate sic
        sic = np.zeros_like(tpr, dtype=float)
        np.divide(tpr, np.sqrt(fpr), out=sic, where=fpr != 0)
        if archs and i < len(archs):
            line, = plt.plot(tpr, sic, linewidth=0.9, label=f"{archs[i].replace('vs', '')} $\lambda$ = {lambdas[i]}")
        elif bdt:
            # line, = plt.plot(tpr, sic, linewidth=0.9, label=f"BDT $\lambda$ = {lambdas[i]}")    
            bdt = False
            continue
        else:
            line, = plt.plot(tpr, sic, linewidth=0.9, label=f"BDT ISO")    
        curve_color = line.get_color()

        # find maximum SIC and corresponding threshold
        valid_mask = fpr != 0
        sic_max_idx = np.argmax(sic[valid_mask])
        # map back to original indices
        sic_max_idx = np.where(valid_mask)[0][sic_max_idx]

        x_sic, y_sic = tpr[sic_max_idx], sic[sic_max_idx]

        plt.scatter(x_sic, y_sic, s=20, color=curve_color)
        plt.text(x_sic + 0.02, y_sic, f"{thresh[sic_max_idx]:.5f}", fontsize=6, color=curve_color)

    plt.grid()
    plt.legend(fontsize=7)
    plt.xlabel("True Positive Rate")
    plt.ylabel("SIC")
    # plt.xscale('log')
    plt.axhline(y=1, color='gray', linestyle='--')
    plt.title(f"SIC scores: different GNN archs and BDT with GNN score for {tau_channel} channel")

    output_plot_name = f"GNNs_SIC_comparison_{tau_channel}_sculpt_{sculpt_degree}{suffix}_full.png"
    print(f"{output_plot_name} has been saved.")
    plt.savefig(os.path.join(root_dir, output_plot_name), dpi=400)
    plt.close()
    
    return fprs, tprs, threshs

def plot_roc_gnn_arch_comparison(gnn_data_locs, bdt_data_loc, root_dir, fpr_targets, tau_channel, sculpt_degree, lambdas, archs):

    ax = plt.gca()
    first_ROC = True
    tpr_targets = []
    
    for i, gnn_data_loc in enumerate(gnn_data_locs):
        gnn_data = load_pkl(gnn_data_loc, "eval_ROC_AUC.pkl")
        gnn_data = next(iter(gnn_data[0].values()))

        fpr = gnn_data["fpr"]
        tpr = gnn_data["tpr"]
        thresh_model = gnn_data["threshold"]

        print(f"Size of fpr list is {len(fpr)}")
        if gnn_data['gnn_mode'] == "final_layer":
            line, = plt.plot(fpr, tpr, linewidth=0.9, label=f"GNN {archs[i].replace('vs', '')} $\lambda$ = {lambdas[i]}, AUC: {gnn_data['auc']:.3f}")
        else:
            line, = plt.plot(fpr, tpr, linewidth=0.9, label=f"GNN {gnn_data['gnn_mode']} $\lambda$ = {lambdas[i]}, AUC: {gnn_data['auc']:.3f}")
        curve_color = line.get_color()

        for k, fpr_cut in enumerate(fpr_targets):
            # GNN model point
            if first_ROC:
                thresh_idx_model = np.argmin(np.abs(fpr - fpr_cut))
                x_model, y_model = fpr[thresh_idx_model], tpr[thresh_idx_model]
                tpr_targets.append(y_model)

            # GNN model point matching
            else:
                tpr_target = tpr_targets[k]
                thresh_idx_model = min(range(len(tpr)), key=lambda j: abs(tpr[j] - tpr_target))
                x_model, y_model = fpr[thresh_idx_model], tpr[thresh_idx_model]
            
            plt.scatter(x_model, y_model, s=20, color=curve_color)
            plt.text(x_model * 1.1, y_model, f"{thresh_model[thresh_idx_model]:.5f}", fontsize=6, color=curve_color)
        
        first_ROC = False

    if bdt_data_loc is not None:
        bdt_data = np.load(bdt_data_loc + '_roc_log24feb2026_v0.pck', allow_pickle=True)
        thresh_bdt = bdt_data["threshold_GNN"]
        fpr_bdt = bdt_data["fpr_GNN"]
        tpr_bdt = bdt_data["tpr_GNN"]

        thresh_benchmark = bdt_data["threshold_ISO"]
        fpr_bench = bdt_data["fpr_ISO"]
        tpr_bench = bdt_data["tpr_ISO"]

        bdt_line, = plt.plot(fpr_bdt, tpr_bdt, linewidth=0.9, label=f"BDT $\lambda$ = {lambdas[-1]}, AUC: {bdt_data['auc_GNN']:.3f}")
        bench_line, = plt.plot(fpr_bench, tpr_bench, linewidth=0.9, label=f"BDT ISO, AUC: {bdt_data['auc_ISO']:.3f}")

        bdt_color = bdt_line.get_color()
        bench_color = bench_line.get_color()

        for i, _ in enumerate(fpr_targets):
            tpr_target = tpr_targets[i]

            thresh_idx_model = min(range(len(tpr_bdt)), key=lambda j: abs(tpr_bdt[j] - tpr_target))
            x_model, y_model = fpr_bdt[thresh_idx_model], tpr_bdt[thresh_idx_model]
                
            plt.scatter(x_model, y_model, s=20, color=bdt_color)
            plt.text(x_model * 1.1, y_model, f"{thresh_bdt[thresh_idx_model]:.5f}", fontsize=6, color=bdt_color)

            thresh_idx_model = min(range(len(tpr_bench)), key=lambda j: abs(tpr_bench[j] - tpr_target))
            x_model, y_model = fpr_bench[thresh_idx_model], tpr_bench[thresh_idx_model]
                
            plt.scatter(x_model, y_model, s=20, color=bench_color)
            plt.text(x_model * 1.1, y_model, f"{thresh_benchmark[thresh_idx_model]:.5f}", fontsize=6, color=bench_color)

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

def tau_score_hist(root, mode, title, tau_channel, plot_mode, view, bdt_score_cut):
    normalize = view == "normalized"
    use_fpr = plot_mode.endswith("fpr")
    suffix = "_with_bdt_cut" if bdt_score_cut else "" 
    if plot_mode.startswith("tau_hist") and not plot_mode.startswith("tau_hist_after"):
        model = "GNN"

        data = np.load(root + f"raw_scores_and_tau_mass_background{suffix}.npz", allow_pickle=True)
        scores = data["scores"]
        f_tau_mass = data["tau_mass"]

        if use_fpr:
            signal = np.load(root + f'raw_scores_and_tau_mass_signal{suffix}.npz', allow_pickle=True)
            scores_signal = signal["scores"]
            nan_mask = ~np.isnan(scores)
            scores = scores[nan_mask]
            f_tau_mass = f_tau_mass[nan_mask]
            scores_signal = scores_signal[~np.isnan(scores_signal)]

            labels = np.concatenate([np.ones(len(scores_signal)), np.zeros(len(scores))])
            scores_all = np.concatenate([scores_signal, scores])

            fpr, tpr, thresh_model = roc_curve(labels, scores_all)

    elif plot_mode.startswith("tau_hist_after"):
        model = "BDT"

        data = load_pkl(root, "tau_score_corr.pkl")[0]
        scores = data["GNN_score_bkg"]
        f_tau_mass = data["tau_mass"]

        if use_fpr:
            scores_signal = data["GNN_score_sig"]
            scores_all = np.concatenate([scores_signal, scores])
            labels = np.concatenate([np.ones(len(scores_signal)), np.zeros(len(scores))])
            fpr, tpr, thresh_model = roc_curve(labels, scores_all)

    else:
        raise ValueError(f"Unknown plot_mode: {plot_mode}")


    print("length of inferenced bkg data set is ", len(scores))
    if use_fpr:
        print("length of inferenced sig data set is ", len(scores_signal))
        print("lenght of fpr list is:", len(fpr))
        fpr_cuts = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        # fpr_cuts = [1e-2, 0.05, 0.1, 0.2]
        score_cuts = []
        for fpr_cut in fpr_cuts:
            cut = np.interp(fpr_cut, fpr, thresh_model)
            score_cuts.append(cut)
    else:
        if model == "GNN":
            print("WARNING: GNN scores have been passed through sigmoid as score cuts on logits is ill-balanced.")
            scores = 1 / (1 + np.exp(-scores)) 
        score_cuts = [0.95, 0.97, 0.98, 0.99]
    
    if normalize:
        bins = np.linspace(1.40, 2.50, 55)
    else:
        bins = np.linspace(1.40, 2.50, 110)
    
    xarray = np.linspace(1.40, 2.50, 1000)
    amplitudes = []
    for i, cut in enumerate(score_cuts):
        mask = scores > cut
        filtered_data = ak.to_numpy(ak.concatenate([f_tau_mass[mask]]))
        weights = np.ones(len(filtered_data)) / len(filtered_data) if normalize else None
        if use_fpr:
            tpr_at_cut = np.interp(fpr_cuts[i], fpr, tpr)
            print(f"We have now filtered {len(filtered_data)} events for an FPR of {fpr_cuts[i]:.0e}.\nWe would expect to roughly have {len(f_tau_mass) * fpr_cuts[i]} events insted.")
            n_pass = np.sum(scores > cut)
            actual_fpr = n_pass / len(scores)
            print(f"Target FPR: {fpr_cuts[i]:.0e}, threshold: {cut:.6f}, actual FPR on inference set: {actual_fpr:.2e}, ratio: {actual_fpr/fpr_cuts[i]:.2f}")
            counts, _, patches = plt.hist(filtered_data, bins=bins, alpha=0.6, label=f"{model} Score > {cut:.5f}, FPR: {fpr_cuts[i]:.0e}", histtype="step", weights=weights)
            color = patches[0].get_edgecolor()
            if not normalize:
                if tau_channel == "HF":
                    ampl = tpr_at_cut * 200
                elif tau_channel == "W":
                    ampl = tpr_at_cut * 200

                if i == 0:
                    plt.plot(xarray, ampl*np.exp(-(xarray - 1.78)**2 / (2*0.02**2)), label="Signal Gaussian (20 MeV width)", color=color, linewidth=0.8)
                else:
                    plt.plot(xarray, ampl*np.exp(-(xarray - 1.78)**2 / (2*0.02**2)), color=color, linewidth=0.8)
            elif normalize and i == len(score_cuts) - 1:
                ampl = counts.max() * 1.1
                plt.plot(xarray, ampl*np.exp(-(xarray - 1.78)**2 / (2*0.02**2)), label="Signal Gaussian (20 MeV width)", color=color, linewidth=0.8)

        else:
            counts, _, _ = plt.hist(filtered_data, bins=bins, alpha=0.6, label=f"{model} Score > {cut:.2f}", histtype="step")
            amplitudes.append(counts.max())
    
    if not use_fpr:
        ampl = sorted(amplitudes, reverse=True)[1]
        plt.plot(xarray, ampl*np.exp(-(xarray - 1.78)**2 / (2*0.02**2)), label="Signal Gaussian (20 MeV width)")
    plt.xlabel(r"$m_{\tau,\mathrm{fit}}$ [GeV]")
    if normalize:
        plt.ylabel("Density")
    else:
        plt.ylabel("Counts")
    plt.xlim(1.40, 2.50)
    plt.grid()
    plt.legend()
    # plt.yscale('log')

    if use_fpr:
        plt.title(f"Fitted Tau Mass Distribution with Different Score Cuts of {model} score \nfor fixed FPR: {tau_channel} channel {mode}, {title}") 
    else:
        plt.title(f"Fitted Tau Mass Distribution with Different Score Cuts of {model} score:\n {tau_channel} channel {mode}, {title}") 

    suffix = "_normalized" if normalize else ""
    suffix += "_with_bdt_cut" if bdt_score_cut else "" 
    if plot_mode.startswith("tau_hist_after"):
        if use_fpr:
            save_name = f'{mode}_after_tau_hist_fpr{suffix}.png'
        else:
            save_name = f'{mode}_after_tau_hist{suffix}.png'
    else:
        if use_fpr:
            save_name = f'{mode}_tau_hist_fpr{suffix}.png'
        else:
            save_name = f'{mode}_tau_hist{suffix}.png'

    plt.savefig(os.path.join(root, save_name), dpi=400)
    plt.close()

    print(f"{save_name} has been saved.")

    return


def load_and_filter_root(model, channel, score_cut):
    """
    Load signal and background data for a given model and channel,
    apply score cut and return filtered scores and masses.
    
    Args:
        model: model name string
        channel: 'W' or 'HF'
        score_cut: minimum GNN score threshold
    
    Returns:
        dict with scores and masses for signal and background
    """
    base = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data"
    
    if channel == 'W':
        # Background
        bkg_file = f"{base}/background_W/{model}/selected_data_skimmed_no_resonances_with_scores.root"
        scores_bkg = uproot.open(bkg_file)["Events"]["GNN_score"].array(library="np")
        mass_bkg = uproot.open(bkg_file)["Events"]["Selected_TauTo3Mu_fitted_mass"].array(library="ak")
        
        # Signal
        sig_file = f"{base}/signal_W/{model}/selected_signal_skimmed_no_resonances_with_scores.root"
        scores_sig = uproot.open(sig_file)["Events"]["GNN_score"].array(library="np")
        mass_sig = uproot.open(sig_file)["Events"]["Selected_TauTo3Mu_fitted_mass"].array(library="ak")

    elif channel == 'HF':
        # Background (multiple parts)
        bkg_files = [
            f"{base}/background_HF/{model}/selected_data_skimmed_no_resonances_part{i}_with_scores.root"
            for i in range(1, 5)
        ]
        scores_bkg = np.concatenate([uproot.open(f)["Events"]["GNN_score"].array(library="np") for f in bkg_files])
        mass_bkg = ak.concatenate([uproot.open(f)["Events"]["Selected_TauTo3Mu_fitted_mass"].array(library="ak") for f in bkg_files])
        
        # Signal
        sig_file = f"{base}/signal_HF/{model}/selected_ds_large_signal_skimmed_no_resonances_with_scores.root"
        scores_sig = uproot.open(sig_file)["Events"]["GNN_score"].array(library="np")
        mass_sig = uproot.open(sig_file)["Events"]["Selected_TauTo3Mu_fitted_mass"].array(library="ak")
    
    else:
        raise ValueError(f"Unknown channel: {channel}. Use 'W' or 'HF'.")

    print(f"Before cleaning: bkg={len(scores_bkg)}, sig={len(scores_sig)}")

    # Clean: remove sentinel -1 and NaN
    def clean_and_cut(scores, mass, cut, mode):
        n_before = len(scores)
        valid = (scores != -1) & ~np.isnan(scores)
        scores = scores[valid]
        mass = mass[valid]
        print(f"Removed sentinel (-1): {mode}={n_before - len(scores)}")
        print(f"Remaining: {mode}={len(scores_bkg)}")
        mask = scores > cut
        return scores[mask], mass[mask]

    scores_sig, mass_sig = clean_and_cut(scores_sig, mass_sig, score_cut, "sig")
    scores_bkg, mass_bkg = clean_and_cut(scores_bkg, mass_bkg, score_cut, "bkg")

    print(f"[{channel} | {model} | cut={score_cut}]")
    print(f"  Signal:     {len(scores_sig)} events")
    print(f"  Background: {len(scores_bkg)} events")

    return {
        'scores_sig': scores_sig,
        'scores_bkg': scores_bkg,
        'mass_sig':   mass_sig,
        'mass_bkg':   mass_bkg,
    }

def load_and_filter_npz(model, channel, score_cut):
    base = "/work/kfuss/SVJ_GNN/output"
    
    sig_data = np.load(f"{base}/{model}/raw_scores_and_tau_mass_signal.npz")
    bkg_data = np.load(f"{base}/{model}/raw_scores_and_tau_mass_background.npz")
    
    scores_sig = sig_data['scores']
    scores_bkg = bkg_data['scores']
    
    # Clean and cut
    def clean_and_cut(scores, cut):
        valid = (scores != -1) & ~np.isnan(scores)
        scores = scores[valid]
        mask = scores > cut
        return scores[mask]
    
    scores_sig = clean_and_cut(scores_sig, score_cut)
    scores_bkg = clean_and_cut(scores_bkg, score_cut)

    print(f"[{channel} | {model} | cut={score_cut}]")
    print(f"  Signal:     {len(scores_sig)} events")
    print(f"  Background: {len(scores_bkg)} events")

    return {
        'scores_sig': scores_sig,
        'scores_bkg': scores_bkg,
    }

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
    parser.add_argument("--archs", required=False, nargs='+')
    parser.add_argument("--view", default=None)
    parser.add_argument("--cut", type=float)
    parser.add_argument("--model", type=str)
    parser.add_argument("--bdt_cut", type=str)


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
    archs = args.archs
    view = args.view
    cut = args.cut
    model = args.model
    bdt_score_cut = True if args.bdt_cut == "true" else False

    if plot_mode.startswith("tau_hist"):
        if view == "both":
            tau_score_hist(output_path, gnn_mode, title, tau_channel, plot_mode, "non_normalized", bdt_score_cut)
            tau_score_hist(output_path, gnn_mode, title, tau_channel, plot_mode, "normalized", bdt_score_cut)
        else:
            tau_score_hist(output_path, gnn_mode, title, tau_channel, plot_mode, view, bdt_score_cut)

    elif plot_mode == "ROC_comp":
        root_dir = "/work/kfuss/SVJ_GNN/output/comparison_plots"
        fpr_targets = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
        plot_roc_gnn_arch_comparison(gnn_archs_location, bdt_location, root_dir, fpr_targets, tau_channel, sculpt_degree, lambdas, archs)
    
    elif plot_mode == "ROC_comp_full":
        root_dir = "/work/kfuss/SVJ_GNN/output/comparison_plots"
        fpr_targets = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
        fprs, tprs, threshs = plot_roc_gnn_arch_comparison_full(gnn_archs_location, bdt_location, root_dir, fpr_targets, tau_channel, sculpt_degree, lambdas, archs, bdt_score_cut)
        plot_sic_full(fprs, tprs, threshs, root_dir, tau_channel, sculpt_degree, lambdas, archs, bdt_score_cut)
    
    elif plot_mode == "score_cut":
        data = load_and_filter_root(model, tau_channel, cut)
        data = load_and_filter_npz(model, tau_channel, cut)