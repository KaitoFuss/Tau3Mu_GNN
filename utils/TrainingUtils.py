from utils.ConeGraphProducer import ConeGraphProducer
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import uproot
import pandas as pd



def Normalize(datasets):
    """Create means and stds for the normalization.

    Args:
        datasets (lists): List of all (dataset folder, graph_file_name) pairs

    Returns: means, stds (np.array)
        
    """

    graph_dict = {}


    for directory, key in zip(datasets[:,0], datasets[:,1]):
        graph_dict[key] = ConeGraphProducer(directory,key)

    means, stds = 0., 0.
    weights = []

    for key in graph_dict:
        for graph in graph_dict[key]:
            means += graph.x.sum(dim=0)*graph.w
            weights.append(graph.w)
    means /= sum(weights)

    for key in graph_dict:
        for graph in graph_dict[key]:   
            stds += ((graph.x - means)**2).sum(dim=0)*graph.w

    stds /= sum(weights)
    stds = torch.sqrt(stds)

    return means, stds

def get_model_parameters(model_type):
    """Get model(graph) parameters.

    Args:
        model_type: Name of model: which graph constructions params were used; eg. cone_size_0.5_knn_3, dzd0_cone_size_0.5_fully

    Returns: num_node_features(int)
        
    """
    num_node_features = 5
    num_edge_features = 4

    if "dzd0" in model_type:
        num_node_features += 2
    
    return num_node_features, num_edge_features

def output_directory(name):
    """Create output directory with the name of the training and
        return the path s.t. I can save everything in it.

    Args:
        name(str): Name of the training

    Returns: path to output directory (str)
        
    """
    output_path =  os.path.join(os.getcwd(),"output",name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"==> Created directory {output_path}")
    
    return output_path

def save_loss(training_losses, validation_losses, name):
    """Save Losses in root file and make plots of it.

    Args:
        training_losses(np.array): array of training losses
        validation_losses(np.array): array of validation losses


    Returns: 
        
    """

    output_dict = {
        "Training_loss": np.array(training_losses),
        "Validation_loss": np.array(validation_losses),
    }

    output_path = output_directory(name)

    # take output and convert each column to a list and safe it in a root file
    output_file_path = output_path + f"/{name}_loss.root"
    file = uproot.recreate(output_file_path)
    df = pd.DataFrame(output_dict,columns=output_dict.keys())
    file["Output"] = df
    file.close()
    print(f"Output saved to {output_file_path}")

    # make also the loss plot
    # Iterate over each key in roc dictionary and plot the curves
    
    x_data = range(0,len(training_losses))
    plt.plot(x_data, training_losses, color = '#457B9D', linewidth = 0.9, linestyle = "-", label="Training Loss")
    plt.plot(x_data, validation_losses,color = '#457B9D', linewidth = 0.9, linestyle = ":", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("BinaryCrossEntropy")
    plt.yscale("log")
    plt.legend()

    output_path = output_directory(name)
    outout_plot_name = name + "_loss.png"
    plt.savefig(os.path.join(output_path,outout_plot_name), dpi = 200)
    plt.close()
    print(f"{outout_plot_name} has been saved.")

 


def generate_name_with_versionnumber(base_name, directory):
    """Generate the name with a version number so that I dont' overwrite stuff.

    Args:
        base_name(str): Name of the experiment
        directory(str): current working directory

    Returns: unique_name(str): base_name with version number
        
    """
        
    version = 1
    unique_name = f"{base_name}_vs{version}"

    while os.path.exists(os.path.join(directory, "output", unique_name)):
        version += 1
        unique_name = f"{base_name}_vs{version}"

    return unique_name



class early_stop:
    """Early stops the training if validation loss does not improve """
    def __init__(self, patience=50, naverage = 10):
        self.patience = patience
        self.naverage = naverage
        self.counter = 0
        self.best_loss = float('inf') # start somewhere large
        self.current_loss = float('inf')
        self.early_stop = False
        self.best_epoch = 0
        self.loss_dict = {f"{i}": None for i in range(0, naverage)} # create dictionary s.t. I can put the last 10 validation losses in it
        self.avg_loss = float('inf')
    
    def __call__(self, validation_loss, model, epoch, output_path, name):
        if epoch == 0:
            self.best_loss = validation_loss
        self.current_loss = validation_loss
        self.calculate_avg_val_loss(validation_loss,epoch)
        self.early_stopping(epoch, model, output_path, name)

    def calculate_avg_val_loss(self, validation_loss, epoch):
        self.loss_dict[f"{epoch % self.naverage}"] = validation_loss
        length = len({k: v for k, v in self.loss_dict.items() if v is not None})
        self.avg_loss = sum(filter(None,self.loss_dict.values()))/length

    # check whether early stopping is necessary
    def early_stopping(self, epoch, model, output_path, name):
        if self.current_loss > self.best_loss:
            self.counter += 1
            print(f"Average loss = {self.avg_loss} and current loss = {self.current_loss} is larger than {self.best_loss}")
            if self.counter > self.patience:
                self.early_stop = True
                print(f"==> Early stop because of overfitting")
                print(f"Model saved for epoch {self.best_epoch}")
        elif self.current_loss <= self.best_loss:
            self.best_loss = self.current_loss
            self.counter = 0
            self.best_epoch = epoch
            print("Saving new best model")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, os.path.join(output_path, name + ".pt"))
        else:
            self.counter = 0



    # # always save the model with the lowest validation loss so far
    # def save_checkpoint(self, model, epoch, output_path, name):
    #     if self.avg_loss <= self.best_loss:
    #         self.best_epoch = epoch
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             }, os.path.join(output_path, name + ".pt"))

