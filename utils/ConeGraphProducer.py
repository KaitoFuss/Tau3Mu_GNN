import os
import numpy as np
import uproot
import torch
from torch_geometric.data import InMemoryDataset, Data
import awkward as ak
from tqdm import tqdm
import vector
import logging
from utils.LundTreeUtilities import tensor_to_tree, prune_tree, tree_to_tensor, get_lund_decomp


class ConeGraphProducer(InMemoryDataset):

    """
    Produces graphs from a root file containing jet constituents.
    The graphs are built by reclustering the constituents with the Cambridge-Aachen algorithm.
    Supports both spacial graphs and lund graphs.
    For spacial graphs, graph nodes are the constituents, while the edges encode vicinity in a given metric.
    For Lund graphs, graph nodes are pseudojets, while the edges encode the clustering history.
    For spacial graphs, the node features are the pt, eta, phi, pdgId and mass of the constituents.
    For Lund graphs, the node features are the pt, eta, phi, mass, Lund coordinates and energy fractions of the pseudojets.
    Spacial graphs can optionally have edge features, which are the invariant mass, kt distance and CA distance between constituent pairs.
    The graphs are stored in a pytorch geometric dataset and written to disk in a "processed" folder in the root file directory.
    
    Args:
        root (str): Path to the folder with (one or more) root files containing the jet constituents. All files in the folder will be processed and treated as one sample.
        output_dataset_name (str): Name of the processed output dataset
        cone_size (float): Size of cone around jet to consider pf_cand
        n_store_jets (int): Number of jets to store per event
        edge_connection_mode (str): Mode to connect nodes, should be in ["r_threshold", "knn", "fully"]
        delta_r_threshold (float): Threshold for the deltaR distance between constituents to be considered connected (spacial graphs only)
        k_knn (int): Number of nearest neighbors for knn
        n_store_cands (int): Number of constituents to store per jet (spacial graphs only)
        max_events_to_process (int): Maximum number of events to process
        use_delta_r_star (bool): Whether to use the deltaR* distance instead of the standard deltaR (TODO: implement)
        use_delta_r_star_star (bool): Whether to use the deltaR** distance instead of the standard deltaR (TODO: implement)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_dummy_values (bool): Whether to use dummy values for the features instead of the actual values
        save_edge_attributes (bool): Whether to save the edge attributes (pairwise invariant masses, Cambridge-Aachen and kt distances)
        save_n_constituents (bool): Whether to save the number of constituents in the graph
        save_event_number (bool): Whether to save the event number in the graph
        extra_label (str): Extra label to add to the processed file name)
        weights (str): What event weights to use (None, 'xsec') (TODO: implement pt flattening)
        save_jet_idx (bool): Whether to save the jet index
        mask (ak.Array): Mask to apply to the root file
        label (float): Label to add to the graph for supervised learning
        verbose (bool): Whether to print progress bars
        jet_collection (str): Name of the jet collection in the root file (default: "FatJet")
        use_displacement (bool): Whether to use the displacement features (d0, dz) in the Lund decomposition (Scouting Data has no displacement information)
    """

    def __init__(
        self,
        root,
        output_dataset_name,
        cone_size,
        n_store_jets=1,
        edge_connection_mode="fully",
        delta_r_threshold=None,
        k_knn=None,
        n_store_cands=None,
        max_events_to_process=None,
        use_delta_r_star=False,
        use_delta_r_star_star=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_dummy_values=False,
        save_edge_attributes=False,
        save_n_constituents=False,
        save_event_number=False,
        weights=None,
        save_jet_idx=False,
        mask=None,
        label:float=0.,
        verbose=False,
        jet_collection="FatJet",
        use_displacement=False  
    ):

        self.root = root
        self.output_dataset_name = output_dataset_name
        self.cone_size = cone_size
        self.n_store_jets = n_store_jets
        self.edge_connection_mode = edge_connection_mode
        self.delta_r_threshold = delta_r_threshold
        self.k_knn = k_knn
        self.n_store_cands = n_store_cands
        self.max_events_to_process = max_events_to_process
        self.use_delta_r_star = use_delta_r_star
        self.use_delta_r_star_star = use_delta_r_star_star
        self.pre_transform = pre_transform
        self.use_dummy_values = use_dummy_values
        self.save_edge_attributes = save_edge_attributes
        self.save_n_constituents = save_n_constituents
        self.save_event_number = save_event_number
        self.weights = weights
        self.save_jet_idx = save_jet_idx
        self.mask = mask
        self.label = label
        self.verbose = verbose
        self.jet_collection = jet_collection
        self.use_displacement = use_displacement 
        self.edge_connection_methods = {
            "fully": self.connect_fully,
            "knn": self.connect_knn,
            "r_threshold": self.connect_r_threshold,
        } 

        assert self.weights in ["xsec", None], "Only xsec and None are supported for the weights argument (more to be added later)"

        assert self.edge_connection_mode in self.edge_connection_methods, f"Unsupported edge_connection_mode: {self.edge_connection_mode}"

        if self.edge_connection_mode == "fully":
            assert self.delta_r_threshold is None, "delta_r_threshold is not supported for fully connected graph. Please set to None"
            assert self.k_knn is None, "k_knn is not supported for fully connected graph. Please set to None"
        elif self.edge_connection_mode == "knn":
            assert self.delta_r_threshold is None, "delta_r_threshold is not supported for knn. Please set to None"
            assert type(self.k_knn) is int, "k_knn needs to be int"
        elif self.edge_connection_mode == "r_threshold":
            assert self.k_knn is None, "k_knn is not supported for r_threshold. Please set to None"
            assert type(self.delta_r_threshold) is float, "r_threshold needs to be float"

        # If xsec weighs are requested but the number of events is hard capped, warn that an appoximation is used
        if self.weights == "xsec" and self.max_events_to_process:
            logging.warning("Using approximation for initial number of events with max number of events")

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root + "/" + f for f in os.listdir(self.root) if f.endswith(".root")]

    @property
    def processed_file_names(self):
        return [self.output_dataset_name+".pt"]
    
    def get_total_initial_events(self):
        n_initial = 0
        for f in self.raw_file_names:
            with uproot.open(f) as in_file:
                if "CutFlow" in in_file:
                    n_initial += in_file["CutFlow"]["Initial"].array()[0]
                else:
                    raise ValueError("No CutFlow tree found in file")
        return n_initial

    def process_event(self, i_start, i_stop, events, xsec, gen_events_before_selection, total_events, events_to_process):

        graphs = []

        for i_ev in tqdm(range(i_start, i_stop), disable=not self.verbose):
            n_jets = min(self.n_store_jets, events.nFatJet[i_ev])
            for nj in range(n_jets):
                event = events[i_ev]

                jet_pt = event[f"{self.jet_collection}_pt"][nj]
                jet_eta = event[f"{self.jet_collection}_eta"][nj]
                jet_phi = event[f"{self.jet_collection}_phi"][nj]
                
                pt = event["PFCands_pt"]
                eta = event["PFCands_eta"]
                phi = event["PFCands_phi"]
                pdgId = event["PFCands_pdgId"]
                mass = event["PFCands_mass"]

                # Only consider pf_cand that are inside cone
                mask = self.build_cone(eta, phi, jet_eta, jet_phi)

                pt = pt[mask]
                eta = eta[mask]
                phi = phi[mask]
                pdgId = pdgId[mask]
                mass = mass[mask]

                if self.use_displacement:
                    d0 = event["PFCands_d0"][mask]
                    dz = event["PFCands_dz"][mask]
                else:
                    d0, dz = None, None 
                jet_idx = nj

                # Order everything by pt and keep the desired number of candidates
                permutation = ak.argsort(pt, ascending=False)
                n_constituents = min(len(permutation), self.n_store_cands) if self.n_store_cands else len(permutation)
                pt = np.array(pt[permutation][:n_constituents], dtype=np.float64)
                eta = np.array(eta[permutation][:n_constituents], dtype=np.float64)
                phi = np.array(phi[permutation][:n_constituents], dtype=np.float64)

                if self.use_displacement:  
                    d0 = np.array(d0[permutation][:n_constituents], dtype=np.float64)   
                    dz = np.array(dz[permutation][:n_constituents], dtype=np.float64)   
                else:
                    d0 = np.zeros(len(pt), dtype=np.float64)   
                    dz = np.zeros(len(pt), dtype=np.float64)   

                pdgId = np.array(pdgId[permutation][:n_constituents], dtype=np.int64)
                # Clip masses to zero since for some reason sometimes they are slightly negative
                mass = np.clip(np.array(mass[permutation][:n_constituents]), a_min=0., a_max=None, dtype=np.float64)

                # If requested, use random values for features
                if self.use_dummy_values:
                    pt = np.random.random(size=(len(pt),))
                    eta = np.random.random(size=(len(eta),))
                    phi = np.random.random(size=(len(phi),))
                    pdgId = np.random.random(size=(len(pdgId),))
                    mass = np.random.random(size=(len(mass),))

                pos = [[e, p] for e, p in zip(eta, phi)]

                # Converting to np.array and subsequently to torch.tensor as suggested in torch docs for performance
                if self.use_displacement: 
                    features = torch.tensor(np.array([
                        pt,
                        eta,
                        phi,
                        pdgId,
                        mass,
                        d0,   
                        dz   
                    ]).T, dtype=torch.float)
                else:
                    features = torch.tensor(np.array([
                        pt,
                        eta,
                        phi,
                        pdgId,
                        mass
                    ]).T, dtype=torch.float)
                # Calculate edges and edge features
                matrix_eta = np.repeat(eta, len(eta)).reshape((len(eta), -1))
                matrix_phi = np.repeat(phi, len(phi)).reshape((len(phi), -1))
                matrix_pt = np.repeat(pt, len(pt)).reshape((len(pt), -1))
                matrix_mass = np.repeat(mass, len(mass)).reshape((len(mass), -1))
                delta_eta = matrix_eta - matrix_eta.T

                # Calculate delta phi accounting for circularity
                delta_phi_internal = np.abs(matrix_phi - matrix_phi.T)
                delta_phi_external = 2*np.pi - np.abs(matrix_phi - matrix_phi.T)
                delta_phi = np.minimum(delta_phi_internal, delta_phi_external)
                delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

                # build graph connections according to specified mode
                edge_index = self.edge_connection_methods[self.edge_connection_mode](delta_R)
                
                if self.use_dummy_values:
                    adjacency = np.random.binomial(1, 0.5, delta_R.shape)
                    np.fill_diagonal(adjacency, 1)

                    edge_connections = np.where( (adjacency - np.identity(adjacency.shape[0])) == 1)
                    edge_index = torch.tensor([ edge for edge in zip(edge_connections[0], edge_connections[1]) ], dtype=torch.long)

                if self.save_edge_attributes:
                    # Build pair-wise invariant masses
                    lorentz_vectors = vector.array({"pt": matrix_pt, "eta": matrix_eta, "phi": matrix_phi, "mass": matrix_mass})
                    pair_masses = (lorentz_vectors + lorentz_vectors.T).mass
                    # Only keep connected edges and take log to squeeze the distribution
                    pair_masses = np.clip(np.nan_to_num(np.log(pair_masses[edge_connections])), a_max=1e5, a_min=-1e5)

                    # Build Cambridge-Aachen and kt distances between constituents
                    R = 0.8
                    d_ca = ((delta_R**2) / (R**2))[edge_connections]
                    d_kt = (np.minimum(matrix_pt**2, matrix_pt.T**2) * (delta_R**2) / (R**2))[edge_connections]

                    edge_features = torch.tensor(np.array([
                        pair_masses,
                        d_ca,
                        d_kt,
                    ]).T, dtype=torch.float)

                    # Build the graph
                    graph = Data(
                            x=features,
                            edge_index=edge_index.t().contiguous(),
                            edge_attr=edge_features,
                            num_nodes=n_constituents,
                            num_node_features=int(features.shape[1]),
                            pos=pos,
                            y=torch.Tensor([self.label])
                        )

                else:
                    # Build the graph
                    graph = Data(
                            x=features,
                            edge_index=edge_index.t().contiguous(),
                            num_nodes=n_constituents,
                            num_node_features=int(features.shape[1]),
                            pos=pos,
                            y=torch.Tensor([self.label])
                        )
                            
                # Add the event weight
                if self.weights == "xsec":
                    if not self.max_events_to_process:
                        graph.w = torch.tensor([xsec/gen_events_before_selection], dtype=torch.float)
                    else:
                        graph.w = torch.tensor([xsec/(gen_events_before_selection * events_to_process / total_events)], dtype=torch.float)
                        
                # Add the number of constituents if requested
                if self.save_n_constituents:
                    graph.n_constituents = torch.tensor([n_constituents], dtype=torch.long)

                # Add the event number if requested
                if self.save_event_number:
                    graph.event_number = torch.tensor([i_ev], dtype=torch.long)

                # Add jet index if requested
                if self.save_jet_idx:
                    graph["jet_idx"] = torch.tensor([jet_idx], dtype=torch.int)
                        
                graphs.append(graph)

        return graphs

    def get_graphs(self, file_name):

        # Check if xsec weights are requested
        if self.weights is None:
            xsec = 1.
            gen_events_before_selection = None

        if self.weights == "xsec":
            with uproot.open(file_name) as in_file:
                if "Metadata" in in_file:
                    if "GenCrossSection" in in_file["Metadata"].keys():
                        xsec = in_file["Metadata"]["GenCrossSection"].array()[0]
                        gen_events_before_selection = self.get_total_initial_events()
                    else:
                        raise ValueError("No GenCrossSection found in Metadata")
                else:
                    raise ValueError("No Metadata tree found in file")

        branches_to_load = [
            "PFCands_pt",
            "PFCands_eta",
            "PFCands_phi",
            "PFCands_pdgId",
            "PFCands_mass",  
            "FatJetPFCands_jetIdx",
            "FatJetPFCands_pFCandsIdx",
            "nFatJet",
            "FatJet_eta",
            "FatJet_phi",
            "FatJet_pt",
        ]

        if self.use_displacement:
            branches_to_load += ["PFCands_d0", "PFCands_dz"]

        with uproot.open(f"{file_name}:Events") as in_file:
            events = in_file.arrays(branches_to_load, library="ak")


        total_events = len(events)
        if gen_events_before_selection is None:
            gen_events_before_selection = total_events
        events_to_process = min(self.max_events_to_process, total_events) if self.max_events_to_process else total_events
        graphDataset = self.process_event(
            0,
            events_to_process,
            events,
            xsec,
            gen_events_before_selection,
            total_events,
            events_to_process,
        )

        if self.pre_transform is not None:
            graphDataset = [self.pre_transform(d) for d in graphDataset]

        return graphDataset

    def process(self):
        # Read data into huge `Data` list.
        graphs = []
        for file in self.raw_file_names:
            print(f"Processing {file}")
            graphs += self.get_graphs(file)
        
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[-1])
    
    def connect_fully(self, delta_R: np.ndarray) -> torch.tensor:
        adjacency = np.ones_like(delta_R)
        edge_connections = np.where( (adjacency - np.identity(adjacency.shape[0])) == 1)
        edge_index = torch.tensor([ edge for edge in zip(edge_connections[0], edge_connections[1]) ], dtype=torch.long)
        return edge_index
        
    def connect_knn(self, delta_R: np.ndarray) -> torch.tensor:
        nn_matrix = np.argsort(delta_R, axis=1)
        k = min(self.k_knn, nn_matrix.shape[0] - 1) # ensure that k is not too large 
        knn_matrix = nn_matrix[:, 1:(k+1)] # as closest pf_cand is the pf_cand itself and we dont want self loops
        edges = [[i, j] for i in range(knn_matrix.shape[0]) for j in knn_matrix[i]]
        edges += [[j, i] for i in range(knn_matrix.shape[0]) for j in knn_matrix[i]]  # Add reverse edges to make graph undirected
        edge_index = torch.tensor(edges, dtype=torch.long) # What happens if some edges are added twice
        return edge_index


    def connect_r_threshold(self, delta_R: np.ndarray) -> torch.tensor:
        adjacency = (delta_R < self.delta_r_threshold).astype(int)
        edge_connections = np.where( (adjacency - np.identity(adjacency.shape[0])) == 1)
        edge_index = torch.tensor([ edge for edge in zip(edge_connections[0], edge_connections[1]) ], dtype=torch.long)
        return edge_index

    def build_cone(self, eta: ak.Array, phi: ak.Array, jet_eta: float, jet_phi: float) -> ak.Array:
        delta_eta = eta - jet_eta
        delta_phi_internal = np.abs(phi - jet_phi)
        delta_phi_external = 2*np.pi - np.abs(phi - jet_phi) 
        delta_phi = np.minimum(delta_phi_internal, delta_phi_external)
        delta_R_cone = np.sqrt(delta_eta**2 + delta_phi**2)

        cone_mask = (delta_R_cone < self.cone_size) # binary array whether pf_cand is inside cone or not

        return cone_mask