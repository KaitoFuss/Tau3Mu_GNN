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
    hist_signal, _ = np.histogram(npvs_signal, bins=bins, density=True)
    hist_bkg, _ = np.histogram(npvs_bkg, bins=bins, density=True)

    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        weight_array = np.true_divide(hist_bkg, hist_signal)
        weight_array[~np.isfinite(weight_array)] = 0  # Set inf/nan to 0

    # Plot histogram
    plt.hist(npvs_signal, bins=bins, alpha=0.7, color='blue', label="signal", density=True)
    plt.hist(npvs_bkg, bins=bins, alpha=0.7, color='red', label="background", density=True)
    plt.xlabel("Number of PV")
    plt.ylabel("Counts")
    plt.title("Comparison of Number of PVs")
    plt.grid(True)
    plt.legend()
    output_name = output_path + "npvs"
    plt.savefig(output_name, dpi=300)
    plt.close()

    # Save weight array and bin edges
    np.savez(output_name.replace(".png", "_fraction.npz"),
             weight_array=weight_array,
             bins=bins)
    

if __name__ == "__main__":
    signal_filepath = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/signal_W/signal_skimmed_no_resonances.root"
    bkg_filepath = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/background/data_skimmed_no_resonances.root"
    output_path = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/signal_W/"
    create_weight_array(signal_filepath, bkg_filepath, output_path)