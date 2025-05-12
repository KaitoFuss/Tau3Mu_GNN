import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_pickled_data_from_subdirs(root_dir):
    """Load pickled data from the root directory and one level deeper"""
    data = []
    
    # Iterate over directories and files in root_dir, but don't go deeper than one level
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Only process directories in the first level under root_dir
        if dirpath == root_dir or os.path.dirname(dirpath) == root_dir:
            for filename in filenames:
                if filename.endswith("_disco_loss.pkl"):  # Assuming the pickled files end with .pkl
                    file_path = os.path.join(dirpath, filename)
                    with open(file_path, "rb") as f:
                        try:
                            data.append(pickle.load(f))
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
    return data

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
    

# Main
root_directory = "output/" 

# Step 1: Load pickled data
pickled_data = load_pickled_data_from_subdirs(root_directory)

# Step 2: Extract relevant data
losses, discos, disco_params = extract_data_from_pickled_dict(pickled_data)

# Step 3: Plot the data
plot_data(losses, discos, disco_params, root_directory)
