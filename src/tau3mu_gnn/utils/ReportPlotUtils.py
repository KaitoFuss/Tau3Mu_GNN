import numpy as np

def calc_delta_R(eta, phi, ref_eta, ref_phi):
    delta_eta = eta - ref_eta
    delta_phi_internal = np.abs(phi - ref_phi)
    delta_phi_external = 2*np.pi - np.abs(phi - ref_phi)
    delta_phi = np.minimum(delta_phi_internal, delta_phi_external)
    delta_R = np.sqrt(delta_eta**2 + delta_phi**2)
    return delta_R

import uproot
import matplotlib.pyplot as plt
import awkward as ak

def plot_deltaR_histogram(file_path):
    # Open the ROOT file and extract necessary branches
    f = uproot.open(file_path)
    df = f["Events"].arrays([
        "Slimmed_TauTo3Mu_fitted_eta", "Slimmed_TauTo3Mu_fitted_phi",
        "Slimmed_TauTo3Mu_mu1_eta", "Slimmed_TauTo3Mu_mu1_phi",
        "Slimmed_TauTo3Mu_mu2_eta", "Slimmed_TauTo3Mu_mu2_phi",
        "Slimmed_TauTo3Mu_mu3_eta", "Slimmed_TauTo3Mu_mu3_phi"
    ])

    # Extract the needed values
    fitted_tau_eta = df["Slimmed_TauTo3Mu_fitted_eta"]
    fitted_tau_phi = df["Slimmed_TauTo3Mu_fitted_phi"]
    mu1_eta = df["Slimmed_TauTo3Mu_mu1_eta"]
    mu1_phi = df["Slimmed_TauTo3Mu_mu1_phi"]
    mu2_eta = df["Slimmed_TauTo3Mu_mu2_eta"]
    mu2_phi = df["Slimmed_TauTo3Mu_mu2_phi"]
    mu3_eta = df["Slimmed_TauTo3Mu_mu3_eta"]
    mu3_phi = df["Slimmed_TauTo3Mu_mu3_phi"]

    # Compute delta R
    delta_R_mu1_tau = calc_delta_R(mu1_eta, mu1_phi, fitted_tau_eta, fitted_tau_phi)
    delta_R_mu2_tau = calc_delta_R(mu2_eta, mu2_phi, fitted_tau_eta, fitted_tau_phi)
    delta_R_mu3_tau = calc_delta_R(mu3_eta, mu3_phi, fitted_tau_eta, fitted_tau_phi)

    # Flatten and concatenate
    flat_mu1 = ak.flatten(delta_R_mu1_tau)
    flat_mu2 = ak.flatten(delta_R_mu2_tau)
    flat_mu3 = ak.flatten(delta_R_mu3_tau)
    all_deltas = ak.to_numpy(ak.concatenate([flat_mu1, flat_mu2, flat_mu3]))

    # Plot histogram
    fig = plt.figure()
    plt.hist(all_deltas, bins=np.linspace(0,0.5,100), alpha=0.75, color='blue')
    plt.xlabel("ΔR(mu, tau)")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.title("ΔR between muons and fitted tau")
    plt.grid(True)
    plt.savefig("/work/kfuss/SVJ_GNN/npv_histograms/deltaR.png", dpi=300) 

plot_deltaR_histogram("/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn/signal_W/signal_skimmed_no_resonances.root")