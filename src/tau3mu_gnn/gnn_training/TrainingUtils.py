from tau3mu_gnn.graph.ConeGraphProducer import ConeGraphProducer
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import uproot
import pandas as pd



def Normalize(datasets, model_type, tau_channel):
    """Create means and stds for the normalization.
    Args:
        datasets (lists): List of all (dataset folder, graph_file_name) pairs
    Returns: means, stds (np.array)
        
    """
    graph_dict = {}
    for directory, key in datasets:
        print(directory, key)
        graph_dict[key] = ConeGraphProducer(directory, key)

    num_node_features, num_edge_features, num_graph_level_features = get_model_parameters(model_type, tau_channel)

    # First pass: means
    total_w_x, total_w_edge, total_w_graph = 0., 0., 0.
    means_x     = torch.zeros(num_node_features)
    means_edge  = torch.zeros(num_edge_features)
    means_graph = torch.zeros(num_graph_level_features)

    for key in graph_dict:
        for graph in graph_dict[key]:
            w = graph.w.item() if torch.is_tensor(graph.w) else float(graph.w)
            total_w_x     += w * graph.x.shape[0]
            total_w_edge  += w * graph.edge_attr.shape[0]
            total_w_graph += w
            means_x     += torch.nan_to_num(graph.x.sum(dim=0))                    * w
            means_edge  += torch.nan_to_num(graph.edge_attr.sum(dim=0))             * w
            means_graph += torch.nan_to_num(graph.graph_level_features.sum(dim=0)) * w

    means_x     /= total_w_x
    means_edge  /= total_w_edge
    means_graph /= total_w_graph

    # Second pass: stds
    total_w_x, total_w_edge, total_w_graph = 0., 0., 0.
    stds_x     = torch.zeros(num_node_features)
    stds_edge  = torch.zeros(num_edge_features)
    stds_graph = torch.zeros(num_graph_level_features)

    for key in graph_dict:
        for graph in graph_dict[key]:
            w = graph.w.item() if torch.is_tensor(graph.w) else float(graph.w)
            total_w_x     += w * graph.x.shape[0]
            total_w_edge  += w * graph.edge_attr.shape[0]
            total_w_graph += w
            stds_x     += ((torch.nan_to_num(graph.x)                      - means_x)    ** 2).sum(dim=0) * w
            stds_edge  += ((torch.nan_to_num(graph.edge_attr)               - means_edge) ** 2).sum(dim=0) * w
            stds_graph += ((torch.nan_to_num(graph.graph_level_features)    - means_graph)** 2).sum(dim=0) * w

    stds_x     = torch.sqrt(stds_x     / total_w_x)
    stds_edge  = torch.sqrt(stds_edge  / total_w_edge)
    stds_graph = torch.sqrt(stds_graph / total_w_graph)

    return means_x, stds_x, means_edge, stds_edge, means_graph, stds_graph

def distance_corr(
        var_1:torch.tensor,
        var_2:torch.tensor,
        normedweight:torch.tensor,
        power=1,
        )->torch.tensor:
    """
    Compute the distance correlation function
    between two variables.

    Args:
        var_1 (torch.tensor): The first variable.
        var_2 (torch.tensor): The second variable.
        normedweight (torch.tensor): The weight matrix.
        power (int): The power of the distance correlation.

    Returns:
        torch.tensor: The distance correlation between the two variables.
    """
    
    # Normalize the weights
    normedweight = normedweight/torch.sum(normedweight)*len(var_1)
    
    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    
    return dCorr

def get_model_parameters(model_type, tau_channel):
    """Get model(graph) parameters.

    Args:
        model_type: Name of model: which graph constructions params were used; eg. cone_size_0.5_knn_3, dzd0_cone_size_0.5_fully
        tau_channel: tau production channel

    Returns: num_node_features(int)
        
    """
    num_node_features = 6
    num_edge_features = 4
    
    if tau_channel == "W":
        num_graph_level_features = 13
    elif tau_channel == "HF":
        num_graph_level_features = 10

    if "dzd0" in model_type:
        num_node_features += 2
    
    return num_node_features, num_edge_features, num_graph_level_features

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

 


def generate_name_with_versionnumber(base_name, directory, tau_channel, unique_ident, conv_type="GINE"):
    """Generate the name with a version number so that I don't overwrite stuff.
    Args:
        base_name (str): Name of the experiment
        directory (str): current working directory
        tau_channel (str): tau channel identifier
        unique_ident (list[str]): list of identifier strings to include in name
        conv_type (str): name of Conv layer type
    Returns:
        unique_name (str): base_name with version number
    """
    ident_str = "_".join(unique_ident) if unique_ident else ""
    prefix = f"{base_name}_{tau_channel}"
    unique_name_base = f"{prefix}_{ident_str}" if ident_str else prefix
    unique_name_base = f"{unique_name_base}_{conv_type}"
    version = 1
    unique_name = f"{unique_name_base}_vs{version}"
    while os.path.exists(os.path.join(directory, "output", unique_name)):
        version += 1
        unique_name = f"{unique_name_base}_vs{version}"
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

