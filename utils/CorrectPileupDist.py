import uproot
import awkward as ak
import numpy as np
import os
import matplotlib.pyplot as plt


def create_weight_array(signal_filepath: str, bkg_filepath: str, output_path: str, col_name="PV_npvs", bins=np.linspace(0, 100, 101)):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read signal data
    f_signal = uproot.open(signal_filepath)
    df_signal = f_signal["Events"].arrays([col_name])
    npvs_signal = ak.to_numpy(ak.concatenate([df_signal[col_name]]))

    # Read background data
    f_bkg = uproot.open(bkg_filepath)
    df_bkg = f_bkg["Events"].arrays([col_name])
    npvs_bkg = ak.to_numpy(ak.concatenate([df_bkg[col_name]]))

    # Compute histograms
    var_bins = [0,10,15,18,20]  + list(np.linspace(21,60,40,dtype=int)) + [65,100]
    hist_signal, _ = np.histogram(npvs_signal, bins=var_bins)
    hist_bkg, _ = np.histogram(npvs_bkg, bins=var_bins)

    hist_signal_norm = hist_signal / np.sum(hist_signal)
    hist_bkg_norm = hist_bkg / np.sum(hist_bkg)

    # Compute bin-wise weights (e.g. for reweighting signal to match background)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight_per_bin = np.true_divide(hist_bkg_norm, hist_signal_norm)
        weight_per_bin[~np.isfinite(weight_per_bin)] = 0

    uniform_bins = np.linspace(0, 100, 101)
    uniform_bin_centers = 0.5 * (uniform_bins[:-1] + uniform_bins[1:])
    rescaled_weights = np.zeros_like(uniform_bins[:-1], dtype=float)  # 100 elements for 100 bins

    var_bins_np = np.array(var_bins)
    for i in range(len(var_bins_np) - 1):
        mask = (uniform_bin_centers >= var_bins_np[i]) & (uniform_bin_centers < var_bins_np[i+1])
        rescaled_weights[mask] = weight_per_bin[i]

    # Create figure and two y-axes
    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()

    # Plot histograms on primary y-axis
    ax1.hist(npvs_signal, bins=bins, alpha=0.7, color='blue', label="signal", density=True)
    ax1.hist(npvs_bkg, bins=bins, alpha=0.7, color='red', label="background", density=True)
    
    # Compute which bin each signal event falls into
    signal_bin_indices = np.digitize(npvs_signal, bins=uniform_bins) - 1  # subtract 1 to get 0-based index
    # Clip indices to be in [0, len(rescaled_weights)-1]
    signal_bin_indices = np.clip(signal_bin_indices, 0, len(rescaled_weights) - 1)
    # Create per-event weights
    event_weights = rescaled_weights[signal_bin_indices]
    ax1.hist(npvs_signal, bins=bins, weights=event_weights, alpha=0.7, color='green', label="rescaled signal", density=True)
    ax1.set_ylabel("Normalised Counts")
    ax1.set_xlabel("Number of PVs")
    ax1.set_yscale('log')
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # # Plot step on secondary y-axis
    # ax2.step(uniform_bin_centers, rescaled_weights, where='mid', color='lightblue', label="weight")
    # ax2.set_ylabel("Weight (bkg/sig)")
    # ax2.legend(loc="upper right")

    plt.title("Comparison of Number of PVs")
    output_name = output_path + "npvs"
    plt.savefig(output_name, dpi=300)
    plt.close()

    # Save weight array and bin edges
    # np.savez(output_path + "npvs_weights.npz",
    #          weight_per_bin=rescaled_weights,
    #          bins=uniform_bins)

    plt.figure()
    plt.hist(rescaled_weights, bins=np.linspace(0, 25, 26), color='blue', alpha=0.7)
    plt.xlabel("Weight (bkg/sig)")
    plt.ylabel("Counts")
    plt.grid(True)
    plt.title("Weights distribution")
    output_name = output_path + "weights"
    # plt.savefig(output_name, dpi=300)
    plt.close()

    

if __name__ == "__main__":
    signal_filepath = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/signal_W/signal_skimmed_no_resonances.root"
    bkg_filepath = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/background/data_skimmed_no_resonances.root"
    output_path = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/signal_W/"
    create_weight_array(signal_filepath, bkg_filepath, output_path)