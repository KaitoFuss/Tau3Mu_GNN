import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

data = np.load("output/knn_new_vs4/raw_scores_and_tau_mass.npz")
scores = data["scores"]
f_tau_mass = data["tau_mass"]

# score_cuts = [0, 0.5, 0.9, 0.995]
score_cuts = [0.9, 0.95, 0.98, 0.99, 0.995]
bins = np.linspace(1.72, 1.85, 14)
bin_centers = 0.5 * (bins[1:] + bins[:-1])  # for errorbar x-positions
width = bins[1] - bins[0]  # bin width

plt.figure(figsize=(8, 6))

for cut in score_cuts:
    mask = scores > cut
    filtered_data = ak.to_numpy(ak.concatenate([f_tau_mass[mask]]))
    
    # Histogram
    counts, _ = np.histogram(filtered_data, bins=bins)
    bars = plt.bar(bin_centers, counts, width=width, alpha=0.5, label=f"GNN Score > {cut}", align='center')
    orign = bars.patches[0].get_facecolor()
    color = tuple(c * 0.7 if i < 3 else c for i, c in enumerate(orign))
    # Error bars
    errors = np.sqrt(counts)
    plt.errorbar(bin_centers, counts, yerr=errors, fmt='none', ecolor=color, elinewidth=0.8, capsize=1)

plt.title("Fitted Tau Mass Distribution with Different Score Cuts:\nGNN, $\\lambda=1$, kNN $k=10$, 10 MeV per bin")
plt.xlabel(r"$m_{\tau,\mathrm{fit}}$ [GeV]")
plt.ylabel("Counts")
plt.grid()
plt.legend()
# plt.yscale('log')
plt.tight_layout()
plt.savefig("/work/kfuss/SVJ_GNN/tau_histo_nosculpt_10MeV.png")