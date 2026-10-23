import uproot
import awkward as ak
import numpy as np
import os
import matplotlib.pyplot as plt


OLD_VAR_BINS = [0, 10, 15, 18, 20] + list(np.linspace(21, 60, 40, dtype=int)) + [65, 100]


def read_npvs(files, tree_name="Events", col_name="PV_npvs"):
    """col_name of all entries of files, as one numpy array."""
    arrays = uproot.concatenate([f"{fp}:{tree_name}" for fp in files], library="ak", filter_name=[col_name])
    return ak.to_numpy(arrays[col_name])


def npvs_weights(npvs_signal, npvs_bkg, var_bins, uniform_bins):
    """Signal -> background weights of the normalised npvs histograms on var_bins, mapped onto
    uniform_bins: a uniform bin gets the weight of the variable bin that holds its centre.
    Variable bins without signal get weight 0. Raise ValueError if a sample has no entry in var_bins."""
    hist_signal, _ = np.histogram(npvs_signal, bins=var_bins)
    hist_bkg, _ = np.histogram(npvs_bkg, bins=var_bins)
    for name, hist in (("signal", hist_signal), ("background", hist_bkg)):
        if hist.sum() == 0:
            raise ValueError(f"npvs weights: {name} has no events inside the bins {var_bins[0]}-{var_bins[-1]}")

    hist_signal_norm = hist_signal / np.sum(hist_signal)
    hist_bkg_norm = hist_bkg / np.sum(hist_bkg)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight_per_bin = np.true_divide(hist_bkg_norm, hist_signal_norm)
        weight_per_bin[~np.isfinite(weight_per_bin)] = 0

    uniform_bin_centers = 0.5 * (uniform_bins[:-1] + uniform_bins[1:])
    rescaled_weights = np.zeros_like(uniform_bins[:-1], dtype=float)
    var_bins_np = np.array(var_bins)
    for i in range(len(var_bins_np) - 1):
        mask = (uniform_bin_centers >= var_bins_np[i]) & (uniform_bin_centers < var_bins_np[i+1])
        rescaled_weights[mask] = weight_per_bin[i]
    return rescaled_weights


def plot_npvs(npvs_signal, npvs_bkg, weight_per_bin, uniform_bins, title, out_npvs, out_weights):
    """Signal, background and rescaled signal npvs (log y), and the weight distribution."""
    fig, ax1 = plt.subplots()
    ax1.hist(npvs_signal, bins=uniform_bins, alpha=0.7, color='blue', label="signal", density=True)
    ax1.hist(npvs_bkg, bins=uniform_bins, alpha=0.7, color='red', label="background", density=True)
    signal_bin_indices = np.clip(np.digitize(npvs_signal, bins=uniform_bins) - 1, 0, len(weight_per_bin) - 1)
    event_weights = weight_per_bin[signal_bin_indices]
    ax1.hist(npvs_signal, bins=uniform_bins, weights=event_weights, alpha=0.7, color='green',
             label="rescaled signal", density=True)
    ax1.set_ylabel("Normalised Counts")
    ax1.set_xlabel("Number of PVs")
    ax1.set_yscale('log')
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.title(f"Comparison of Number of PVs {title}")
    plt.savefig(out_npvs, dpi=300)
    plt.close()

    plt.figure()
    plt.hist(weight_per_bin, bins=np.linspace(0, 25, 26), color='blue', alpha=0.7)
    plt.xlabel("Weight (bkg/sig)")
    plt.ylabel("Counts")
    plt.grid(True)
    plt.title(f"Weights distribution {title}")
    plt.savefig(out_weights, dpi=300)
    plt.close()


def create_weight_array(signal_filepaths: list, bkg_filepaths: list, output_path: str, tau_channel, col_name="PV_npvs", bins=np.linspace(0, 100, 101)):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    npvs_signal = read_npvs(signal_filepaths, col_name=col_name)
    npvs_bkg = read_npvs(bkg_filepaths, col_name=col_name)
    uniform_bins = np.linspace(0, 100, 101)
    rescaled_weights = npvs_weights(npvs_signal, npvs_bkg, OLD_VAR_BINS, uniform_bins)
    plot_npvs(npvs_signal, npvs_bkg, rescaled_weights, bins, f"{tau_channel} channel",
              output_path + "_npvs", output_path + "_weights")
    # Save weight array and bin edges
    np.savez(output_path + f"_npvs_weights.npz",
             weight_per_bin=rescaled_weights,
             bins=uniform_bins)


if __name__ == "__main__":
    signal_filepaths = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/signal_skimmed_no_resonances.root"]
    bkg_filepaths = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/data_skimmed_no_resonances.root"]
    output_path = "data/signal_W"
    tau_channel = "W"
    create_weight_array(signal_filepaths, bkg_filepaths, output_path, tau_channel)

    signal_filepaths = ["/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_HF/ds_large_signal_skimmed_no_resonances.root"]
    bkg_filepaths = [
        "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/data_skimmed_no_resonances_part1.root",
        "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/data_skimmed_no_resonances_part2.root",
        "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/data_skimmed_no_resonances_part3.root",
        "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_HF/data_skimmed_no_resonances_part4.root",
    ]
    output_path = "data/signal_HF"
    tau_channel = "HF"
    create_weight_array(signal_filepaths, bkg_filepaths, output_path, tau_channel)