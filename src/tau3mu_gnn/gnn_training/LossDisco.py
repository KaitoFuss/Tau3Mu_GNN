import os
import matplotlib.pyplot as plt
import numpy as np
from tau3mu_gnn.model_eval.ModelEval import CPU_Unpickler, load_pkl
import argparse


def extract_data_from_pickled_dict(pickled_data):
    """Extract relevant fields (loss, disco, disco_param) from pickled data"""
    losses = []
    discos = []
    disco_params = []
    
    for data in pickled_data:
            losses.append(data["last_val_loss"])
            discos.append(data["last_validation_disco"])
            disco_params.append(data["lambda"])
    
    return np.array(losses), np.array(discos), np.array(disco_params)

def plot_data(losses, discos, disco_params, output_path, channel):
    """Plot loss vs disco with color representing disco_param"""
    plt.figure()
    scatter = plt.scatter(losses, discos, c=disco_params, cmap='viridis', s=50, vmin=0, vmax=100)
    plt.colorbar(scatter, label="Disco Param")
    plt.xlabel('Validation Loss')
    plt.ylabel('Disco')
    plt.title(f'{channel} channel: Loss vs Disco (Colored by Disco Param)')
    plt.grid(True)
    outout_plot_name = f"{channel}_loss_disco.png"
    print(f"{outout_plot_name} has been saved.")
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=400)
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tau channel results")
    parser.add_argument(
        "--channel",
        type=str,
        required=True,
        help="Tau channel (W or HF)"
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default="fl_8_sd_4",
        help="eg. fl_8_sd_4"
    )

    args = parser.parse_args()
    channel = args.channel
    model_arch = args.model_arch

    root_directory = "/work/kfuss/SVJ_GNN/output/" 

    subdirs = [os.path.join(root_directory, name) for name in os.listdir(root_directory)
           if os.path.isdir(os.path.join(root_directory, name)) and "new_dropout" in name and channel in name and "fl_8_sd_4" in name]
    pickled_data = []
    for subdir in subdirs:
        pickled_data += load_pkl(subdir, "disco_loss.pkl")
    losses, discos, disco_params = extract_data_from_pickled_dict(pickled_data)

    plot_data(losses, discos, disco_params, root_directory, channel)
