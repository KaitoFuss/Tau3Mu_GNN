import os
import torch
import io
import pickle
import matplotlib.pyplot as plt

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
    plt.xlim(0.01,1)
    plt.ylim(0.8,1)
    plt.title(title)
    outout_plot_name = "ROC_log.png"
    print(f"{outout_plot_name} has been saved.")
    plt.savefig(os.path.join(root_dir,outout_plot_name), dpi=400)
    plt.close()


if __name__ == "__main__":
    root_dir = '/work/kfuss/SVJ_GNN/output/knn_new_vs4'
    data = load_pkl(root_dir, 'ROC_AUC.pkl')
    plot_roc(data[0], "W_decay_signal_skimmed_dzd0_cone_size_0.5_knn_10", root_dir, [-0.2,-0.3,-0.5, -0.75, -0.9], r"$\lambda=1$ knn $k=10$")