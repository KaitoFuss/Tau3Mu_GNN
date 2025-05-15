import os
import matplotlib.pyplot as plt
import numpy as np
from ModelEval import CPU_Unpickler, load_pkl


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

def plot_data(losses, discos, disco_params, output_path):
    """Plot loss vs disco with color representing disco_param"""
    plt.figure()
    scatter = plt.scatter(losses, discos, c=disco_params, cmap='viridis', s=50)
    plt.colorbar(scatter, label="Disco Param")
    plt.xlabel('Validation Loss')
    plt.ylabel('Disco')
    plt.title('Loss vs Disco (Colored by Disco Param)')
    plt.grid(True)
    outout_plot_name = "loss_disco.png"
    print(f"{outout_plot_name} has been saved.")
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi=400)
    plt.close()
    

if __name__ == "__main__":
        
    root_directory = "/work/kfuss/SVJ_GNN/output/" 

    # Step 1: Load pickled data
    pickled_data = load_pkl(root_directory, "disco_loss.pkl")

    # Step 2: Extract relevant data
    losses, discos, disco_params = extract_data_from_pickled_dict(pickled_data)

    # Step 3: Plot the data
    plot_data(losses, discos, disco_params, root_directory)
