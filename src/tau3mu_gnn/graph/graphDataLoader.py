from tau3mu_gnn.graph.ConeGraphProducer_Dataset import ConeGraphProducer_Dataset
from torch_geometric.loader import DataLoader
from typing import Optional, List
from tau3mu_gnn.gnn_training.NormalizerUtils import OnTheFlyNormalizer
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import numpy as np
import os
import torch
import warnings

class GraphDataLoader(DataLoader):
    """
    A custom DataLoader for loading ConeGraphProducer objects. It allows for the
    use of a normalizer to be applied to the data, and for the specification of
    a window of the data to be loaded. It also allows for the loading of multiple
    datasets at once, and concatenates them into a single dataset.

    Args:
        dataset_paths (List[List]): A list of lists, where each inner list contains
            the path to the dataset and the name of the dataset to be loaded.
        normalizer (OnTheFlyNormalizer): A normalizer to be applied to the data.
            It is registered as a per-graph ``transform`` on each dataset so that
            normalisation happens lazily at access time rather than up-front.
        window_to_load (List[List[float]]): A list of lists, where each inner list
            contains the start and end of the window to be loaded. If None, the
            entire dataset is loaded.
        batch_size (int): The batch size to be used. Negative numbers will read as len(dataset)
        dataset_fraction (List[float]): List of weights in order to have balanced datasets.
        shuffle (bool): Whether to shuffle the data.
        follow_batch (Optional[List[str]]): A list of strings specifying which
            attributes of the data object should be batched together. If None,
            the default behavior is used.
        exclude_keys (Optional[List[str]]): A list of strings specifying which
            attributes of the data object should not be batched together. If None,
            the default behavior is used.
        device (torch.device): Kept for API compatibility.  With the chunk-based
            dataset, moving data to a specific device should be done in the
            training loop (e.g. ``batch.to(device)``) rather than pre-loading
            everything onto the device.
        sample (Optional(Bool)): Wether or not to use weighted random sampling (True by default)
        drop_last (bool): Whether to drop last batch 
        **kwargs: Additional keyword arguments to be passed to the DataLoader.

    Returns:
        DataLoader: A DataLoader object for loading ConeGraphProducer objects.
    """
    def __init__(
            self,
            dataset_paths: List[List],
            normalizer: OnTheFlyNormalizer=None,
            window_to_load: List[List[float]]=None,
            batch_size: int=1,
            dataset_fraction: List[float]=None,
            shuffle: bool = False,
            permutation: Optional[dict] = None,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            device: torch.device=None,
            sample = True,
            drop_last=False,
            **kwargs,
    ):
        self.dataset_paths = dataset_paths
        self.normalizer = normalizer
        self.window_to_load = window_to_load
        self.shuffle = shuffle
        self.permutation = permutation if permutation is not None else {}
        if self.window_to_load is None:
            self.window_to_load = [[0., 1.] for _ in dataset_paths]
        self.device = device


        if device is not None:
            warnings.warn(
                "The 'device' parameter is deprecated. With the chunk-based dataset, "
                "move batches to the target device in your training loop "
                "(e.g. `batch = batch.to(device)`) instead of pre-loading everything "
                "onto the device. The parameter is kept for backwards compatibility but "
                "has no effect on the dataset itself.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Consistency checks
        assert len(self.window_to_load) == len(self.dataset_paths), "Please provide one slicing window per dataset!"
        assert (not self.permutation) or (len(self.permutation) == len(self.dataset_paths)), \
            "Mismatch between number of datasets and permutations, please check for consistency"
        
        
        datasets = []
        weights = []
        # Build the dataset
        for i, (path, name) in enumerate(dataset_paths):
            meta_file = f"{path}/processed/{name}_meta.pt"
            if not os.path.exists(meta_file):
                raise FileNotFoundError(
                    f"The requested dataset {path}/{name} has not been processed, "\
                    "please do so before attempting to load it."
                        )
            # Pass the normalizer as a per-graph transform so it is applied
            # lazily at access time instead of requiring all data in RAM.
            data = ConeGraphProducer_Dataset(path, output_dataset_name=name, transform=normalizer)
            
            if self.device is not None:
                data.data.to(self.device)
            
            if self.shuffle:
                if name not in self.permutation.keys():
                    self.permutation[name] = torch.randperm(len(data))
                data = data[self.permutation[name]]
            
            data = data[int(self.window_to_load[i][0]*len(data)):int(self.window_to_load[i][1]*len(data))]
            weights.append(data.w)

            if dataset_fraction is not None:
                weights[i] = weights[i]*dataset_fraction[i]/np.sum(dataset_fraction)/weights[i].sum()

            datasets.append(data)

        data = ConcatDataset(datasets)
        weights = torch.cat(weights).to(device) if device is not None else torch.cat(weights)
        sampler = WeightedRandomSampler(weights, len(data), replacement=True) if sample else None
        batch_size = batch_size if batch_size > 0 else len(data)

        super(GraphDataLoader, self).__init__(data, batch_size, False, follow_batch, exclude_keys, sampler=sampler, drop_last=drop_last, **kwargs)
        
    def get_permutation(self):
        return self.permutation
