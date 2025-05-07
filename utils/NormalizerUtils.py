import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
import numba as nb
import vector

# Avoid nasty interplays of torch and numba
torch.multiprocessing.set_sharing_strategy('file_system')

class OnTheFlyNormalizer(BaseTransform):
    """
    Normalize the data on the fly by simple subtraction and division.

    Args:
        attrs: list of attributes to normalize
        means: means of the attributes
        stds: standard deviations of the attributes
    """
    def __init__(self, attrs: list, means: torch.tensor, stds: torch.tensor, device=None):
        self.attrs = attrs
        self.means = means.to(device) if device is not None else means
        self.stds = stds.to(device) if device is not None else stds
        
    def __call__(self, data):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value -= self.means
                value /= self.stds
                store[key] = value

        return data