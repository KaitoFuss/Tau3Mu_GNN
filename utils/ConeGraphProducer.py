import os
import numpy as np
import uproot
import torch
from torch_geometric.data import InMemoryDataset, Data
import awkward as ak
from tqdm import tqdm
import vector
import logging


class ConeGraphProducer(InMemoryDataset):

    """
    Produces graphs from a root file containing jet constituents.
    The graphs are built by reclustering the constituents with the Cambridge-Aachen algorithm.
    Supports both spacial graphs.
    For spacial graphs, graph nodes are the constituents, while the edges encode vicinity in a given metric.
    For spacial graphs, the node features are the pt, eta, phi, pdgId and mass of the constituents.
    Spacial graphs can optionally have edge features, which are the invariant mass, kt distance and CA distance between constituent pairs.
    The graphs are stored in a pytorch geometric dataset and written to disk in a "processed" folder in the root file directory.
    
    Args:
        root (str): Path to the folder with (one or more) root files containing the jet constituents. All files in the folder will be processed and treated as one sample.
        output_dataset_name (str): Name of the processed output dataset
        cone_size (float): Size of cone around jet to consider pf_cand
        n_cones (int): Number of cones/ graphs to build per event. Should be 1 or 3
        edge_connection_mode (str): Mode to connect nodes, should be in ["r_threshold", "knn", "fully"]
        delta_r_threshold (float): Threshold for the deltaR distance between constituents to be considered connected (spacial graphs only)
        k_knn (int): Number of nearest neighbors for knn
        max_events_to_process (int): Maximum number of events to process
        use_relative_angles (bool): Whether to use relative angles (eta - jet_eta, phi - jet_phi) or absolute angles (eta, phi)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        save_edge_attributes (bool): Whether to save the edge attributes (pairwise invariant masses, Cambridge-Aachen and kt distances)
        save_event_number (bool): Whether to save the event number in the graph
        save_fitted_tau_mass (bool): Whether to save fitted tau mass in the graph
        extra_label (str): Extra label to add to the processed file name)
        weights (str): What event weights to use (None, 'xsec') (TODO: implement pt flattening)
        save_fitted_tau_idx (bool): Whether to save the fitted tau index
        mask (bool): Bool whether to apply a mask extracting only tau3mu cands outside vicinity of tau mass, used to get background events
        label (float): Label to add to the graph for supervised learning
        verbose (bool): Whether to print progress bars
        tau_collection_name (str): Name of the fitted tau in the root file (default: "TauTo3Mu_fitted")
        slimmed (bool): Whether data was pre skimmed and disregarded Meson resonances and bad PV matches
        use_displacement (bool): Whether to use the displacement features (d0, dz) in the Lund decomposition (Scouting Data has no displacement information)
    """

    def __init__(
        self,
        root,
        output_dataset_name,
        cone_size=0.5,
        n_cones=1,
        edge_connection_mode="fully",
        delta_r_threshold=None,
        k_knn=None,
        max_events_to_process=None,
        use_relative_angles=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        save_edge_attributes=False,
        save_event_number=False,
        save_fitted_tau_mass=True,
        weights=None,
        save_fitted_tau_idx=False,
        mask=False,
        label:float=0.,
        verbose=False,
        tau_collection_name="TauTo3Mu_fitted",
        slimmed=False,
        use_displacement=False  
    ):

        self.root = root
        self.output_dataset_name = output_dataset_name
        self.cone_size = cone_size
        self.n_cones = n_cones
        self.edge_connection_mode = edge_connection_mode
        self.delta_r_threshold = delta_r_threshold
        self.k_knn = k_knn
        self.max_events_to_process = max_events_to_process
        self.use_relative_angles = use_relative_angles
        self.pre_transform = pre_transform
        self.save_edge_attributes = save_edge_attributes
        self.save_event_number = save_event_number
        self.save_fitted_tau_mass = save_fitted_tau_mass
        self.weights = weights
        self.save_fitted_tau_idx = save_fitted_tau_idx
        self.mask = mask
        self.label = label
        self.verbose = verbose
        self.tau_collection_name = tau_collection_name
        self.slimmed = slimmed
        self.use_displacement = use_displacement 
        self.edge_connection_methods = {
            "fully": self.connect_fully,
            "knn": self.connect_knn,
            "r_threshold": self.connect_r_threshold,
        } 

        assert self.weights in ["xsec", None], "Only xsec and None are supported for the weights argument (more to be added later)"

        assert self.edge_connection_mode in self.edge_connection_methods, f"Unsupported edge_connection_mode: {self.edge_connection_mode}"

        if self.edge_connection_mode == "fully":
            if self.delta_r_threshold is not None:
                logging.warning("delta_r_threshold is not supported for fully connected graph. Please set to None")
            if self.k_knn is not None:
                logging.warning("k_knn is not supported for fully connected graph. Please set to None")

        elif self.edge_connection_mode == "knn":
            if self.delta_r_threshold is not None:
                logging.warning("delta_r_threshold is not supported for knn. Please set to None")
            if not isinstance(self.k_knn, int):
                logging.warning("k_knn needs to be int")

        elif self.edge_connection_mode == "r_threshold":
            if self.k_knn is not None:
                logging.warning("k_knn is not supported for r_threshold. Please set to None")
            if not isinstance(self.delta_r_threshold, float):
                logging.warning("r_threshold needs to be float")

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
    
    # This function is now not called because there is no metadata I think
    def get_total_initial_events(self):
        n_initial = 0
        for f in self.raw_file_names:
            with uproot.open(f) as in_file:
                if "CutFlow" in in_file:
                    n_initial += in_file["CutFlow"]["Initial"].array()[0]
                else:
                    raise ValueError("No CutFlow tree found in file")
        return n_initial

    def process_event(self, i_start, i_stop, events, xsec, gen_events_before_selection, total_events, events_to_process, bkg_mask):
        prefix = "Slimmed_" if self.slimmed else ""
        n_prefix = "Slimmed" if self.slimmed else ""
        graphs = []
        for i_ev in tqdm(range(i_start, i_stop), disable=not self.verbose):
            n_cones = self.n_cones # TODO implement option to build three cones instead of one big one if n_cones == 3; or maybe change to bool 

            event = events[i_ev]
            if self.mask:
                n_tau = event[f"n{n_prefix}TauTo3Mu"] 
                n_tau = min(n_tau, ak.sum(bkg_mask[i_ev]))
            else:
                n_tau = event[f"n{n_prefix}TauTo3Mu"] 

            pfcands_pt = event[f"{prefix}PFCands_pt"]
            pfcands_eta = event[f"{prefix}PFCands_eta"]
            pfcands_phi = event[f"{prefix}PFCands_phi"]
            pfcands_pdgId = event[f"{prefix}PFCands_pdgId"]
            pfcands_mass = event[f"{prefix}PFCands_mass"]
            pfcands_charge = event[f"{prefix}PFCands_charge"]
            
            # Iterating over all tau3mu candidates in event and creating for all a graph in its vicinity 
            for nt in range(n_tau):
                tau_mask = bkg_mask[i_ev] if self.mask else ak.ones_like(event[f"{prefix}{self.tau_collection_name}_pt"], dtype=bool)
                
                # Read data for fitted tau from event
                fitted_tau_pt = event[f"{prefix}{self.tau_collection_name}_pt"][tau_mask][nt] 
                fitted_tau_eta = event[f"{prefix}{self.tau_collection_name}_eta"][tau_mask][nt] 
                fitted_tau_phi = event[f"{prefix}{self.tau_collection_name}_phi"][tau_mask][nt] 
                fitted_tau_mass = event[f"{prefix}{self.tau_collection_name}_mass"][tau_mask][nt] 
                
                # Read data for mu1 from event
                mu1_idx = event[f"{prefix}TauTo3Mu_mu1_idx"][tau_mask][nt] 
                mu1_eta = event[f"{prefix}TauTo3Mu_mu1_eta"][tau_mask][nt] 
                mu1_phi = event[f"{prefix}TauTo3Mu_mu1_phi"][tau_mask][nt] 
                mu1_pt = event[f"{prefix}TauTo3Mu_mu1_pt"][tau_mask][nt] 
    
                # Read data for mu2 from event
                mu2_idx = event[f"{prefix}TauTo3Mu_mu2_idx"][tau_mask][nt] 
                mu2_eta = event[f"{prefix}TauTo3Mu_mu2_eta"][tau_mask][nt] 
                mu2_phi = event[f"{prefix}TauTo3Mu_mu2_phi"][tau_mask][nt] 
                mu2_pt = event[f"{prefix}TauTo3Mu_mu2_pt"][tau_mask][nt] 
                
                # Read data for mu3 from event
                mu3_idx = event[f"{prefix}TauTo3Mu_mu3_idx"][tau_mask][nt] 
                mu3_eta = event[f"{prefix}TauTo3Mu_mu3_eta"][tau_mask][nt] 
                mu3_phi = event[f"{prefix}TauTo3Mu_mu3_phi"][tau_mask][nt] 
                mu3_pt = event[f"{prefix}TauTo3Mu_mu3_pt"][tau_mask][nt] 

                # Ensuring that cone is large enough such that all 3mu are inside it; center of cone will be (fitted_tau_eta, fitted_tau_phi)
                cone_mask = self.build_cone(
                    pfcands_eta, 
                    pfcands_phi, 
                    fitted_tau_eta, 
                    fitted_tau_phi, 
                    mu1_eta, 
                    mu1_phi, 
                    mu1_idx, 
                    mu2_eta, 
                    mu2_phi, 
                    mu2_idx, 
                    mu3_eta, 
                    mu3_phi, 
                    mu3_idx
                )
                pfcands_pt_masked = pfcands_pt[cone_mask]
                pfcands_eta_masked = pfcands_eta[cone_mask]
                pfcands_phi_masked = pfcands_phi[cone_mask]
                pfcands_pdgId_masked = pfcands_pdgId[cone_mask]
                pfcands_mass_masked = pfcands_mass[cone_mask]
                pfcands_charge_masked = pfcands_charge[cone_mask]

                if self.use_displacement:
                    pfcands_d0_masked = event[f"{prefix}PFCands_d0"][cone_mask]
                    pfcands_dz_masked = event[f"{prefix}PFCands_dz"][cone_mask]
                else:
                    pfcands_d0_masked, pfcands_dz_masked = None, None 

                fitted_tau_idx = nt

                pfcands_pt_masked = np.array(pfcands_pt_masked, dtype=np.float64)
                pfcands_eta_masked = np.array(pfcands_eta_masked, dtype=np.float64)
                pfcands_phi_masked = np.array(pfcands_phi_masked, dtype=np.float64)

                if self.use_displacement:  
                    pfcands_d0_masked = np.array(pfcands_d0_masked, dtype=np.float64)   
                    pfcands_dz_masked = np.array(pfcands_dz_masked, dtype=np.float64)   
                else:
                    pfcands_d0_masked = np.zeros(len(pfcands_pt_masked), dtype=np.float64)   
                    pfcands_dz_masked = np.zeros(len(pfcands_pt_masked), dtype=np.float64)   

                if self.use_relative_angles:
                    pfcands_eta_masked = pfcands_eta_masked - fitted_tau_eta
                    # Need care to account for circularity
                    pfcands_phi_masked = (pfcands_phi_masked - fitted_tau_phi + np.pi) % (2*np.pi) - np.pi
                
                pfcands_pdgId_masked = np.array(pfcands_pdgId_masked, dtype=np.int64)
                # Clip masses to zero since for some reason sometimes they are slightly negative
                pfcands_mass_masked = np.clip(np.array(pfcands_mass_masked), a_min=0., a_max=None, dtype=np.float64)
                pfcands_charge_masked = np.array(pfcands_charge_masked, dtype=np.int64) # is i64 fine for charge? Saw that it was saved as i32 in root

                pos = [[e, p] for e, p in zip(pfcands_eta_masked, pfcands_phi_masked)]

                # Converting to np.array and subsequently to torch.tensor as suggested in torch docs for performance
                if self.use_displacement: 
                    features = torch.tensor(np.array([
                        pfcands_pt_masked,
                        pfcands_eta_masked,
                        pfcands_phi_masked,
                        pfcands_pdgId_masked,
                        pfcands_mass_masked,
                        pfcands_d0_masked,   
                        pfcands_dz_masked   
                    ]).T, dtype=torch.float)
                else:
                    features = torch.tensor(np.array([
                        pfcands_pt_masked,
                        pfcands_eta_masked,
                        pfcands_phi_masked,
                        pfcands_pdgId_masked,
                        pfcands_mass_masked
                    ]).T, dtype=torch.float)
                
                # TODO do this calc outside loop and then just mask inside 
                # Calculate edges and edge features
                matrix_eta = np.repeat(pfcands_eta_masked, len(pfcands_eta_masked)).reshape((len(pfcands_eta_masked), -1))
                matrix_phi = np.repeat(pfcands_phi_masked, len(pfcands_phi_masked)).reshape((len(pfcands_phi_masked), -1))
                matrix_pt = np.repeat(pfcands_pt_masked, len(pfcands_pt_masked)).reshape((len(pfcands_pt_masked), -1))
                matrix_mass = np.repeat(pfcands_mass_masked, len(pfcands_mass_masked)).reshape((len(pfcands_mass_masked), -1))
                matrix_charge = np.repeat(pfcands_charge_masked, len(pfcands_charge_masked)).reshape((len(pfcands_charge_masked), -1))
                matrix_delta_eta = matrix_eta - matrix_eta.T

                # Calculate delta phi accounting for circularity
                matrix_delta_phi_internal = np.abs(matrix_phi - matrix_phi.T)
                matrix_delta_phi_external = 2*np.pi - np.abs(matrix_phi - matrix_phi.T)
                matrix_delta_phi = np.minimum(matrix_delta_phi_internal, matrix_delta_phi_external)
                matrix_delta_R = np.sqrt(matrix_delta_eta**2 + matrix_delta_phi**2)

                # build graph connections according to specified mode
                edge_index = self.edge_connection_methods[self.edge_connection_mode](matrix_delta_R)
                
                if self.save_edge_attributes:
                    # Build pair-wise invariant masses
                    lorentz_vectors = vector.array({"pt": matrix_pt, "eta": matrix_eta, "phi": matrix_phi, "mass": matrix_mass})
                    pair_masses = (lorentz_vectors + lorentz_vectors.T).mass
                    # Only keep connected edges and take log to squeeze the distribution
                    pair_masses = np.array([pair_masses[i,j] for i,j in edge_index])
                    # pair_masses = np.clip(np.nan_to_num(np.log(pair_masses)), a_max=1e5, a_min=-1e5)
                    # Get pair charge
                    pair_charges = np.array([matrix_charge[i,j] for i,j in edge_index])
                    
                    # Build Cambridge-Aachen and kt distances between constituents
                    R = 0.8
                    d_ca = ((matrix_delta_R**2) / (R**2))
                    d_ca = np.array([d_ca[i,j] for i,j in edge_index])
                    d_kt = (np.minimum(matrix_pt**2, matrix_pt.T**2) * (matrix_delta_R**2) / (R**2))
                    d_kt = np.array([d_kt[i,j] for i,j in edge_index])

                    edge_features = torch.tensor(np.array([
                        pair_masses,
                        pair_charges,
                        d_ca,
                        d_kt,
                    ]).T, dtype=torch.float)

                    # Build the graph
                    graph = Data(
                            x=features,
                            edge_index=edge_index.t().contiguous(),
                            edge_attr=edge_features,
                            num_nodes=int(features.shape[0]),
                            num_node_features=int(features.shape[1]),
                            num_edge_features=int(edge_features.shape[1]),
                            pos=pos,
                            y=torch.Tensor([self.label])
                        )

                else:
                    # Build the graph
                    graph = Data(
                            x=features,
                            edge_index=edge_index.t().contiguous(),
                            num_nodes=int(features.shape[0]),
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
                elif self.weights is None:
                    graph.w = 1.
                
                # Add the event number if requested
                if self.save_event_number:
                    graph.event_number = torch.tensor([i_ev], dtype=torch.long)

                # Add tau index if requested
                if self.save_fitted_tau_idx:
                    graph.fitted_tau_index = torch.tensor([fitted_tau_idx], dtype=torch.int)

                # Add fitted_tau_mass if requested
                if self.save_fitted_tau_mass:
                    graph.fitted_tau_mass = torch.tensor([fitted_tau_mass], dtype=torch.float)
                        
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
        prefix = "Slimmed_" if self.slimmed else ""
        n_prefix = "Slimmed" if self.slimmed else ""
        branches_to_load = [
            f"n{n_prefix}PFCands",
            f"{prefix}PFCands_pt",
            f"{prefix}PFCands_eta",
            f"{prefix}PFCands_phi",
            f"{prefix}PFCands_pdgId",
            f"{prefix}PFCands_mass",  
            f"{prefix}PFCands_charge",
            f"n{n_prefix}TauTo3Mu",
            f"{prefix}TauTo3Mu_mu1_idx",
            f"{prefix}TauTo3Mu_mu1_eta",
            f"{prefix}TauTo3Mu_mu1_phi",
            f"{prefix}TauTo3Mu_mu1_pt",
            f"{prefix}TauTo3Mu_mu2_idx",
            f"{prefix}TauTo3Mu_mu2_eta",
            f"{prefix}TauTo3Mu_mu2_phi",
            f"{prefix}TauTo3Mu_mu2_pt",
            f"{prefix}TauTo3Mu_mu3_idx",
            f"{prefix}TauTo3Mu_mu3_eta",
            f"{prefix}TauTo3Mu_mu3_phi",
            f"{prefix}TauTo3Mu_mu3_pt",
            f"{prefix}TauTo3Mu_fitted_mass",
            f"{prefix}TauTo3Mu_fitted_eta",
            f"{prefix}TauTo3Mu_fitted_phi",
            f"{prefix}TauTo3Mu_fitted_pt",
        ]

        if self.use_displacement:
            branches_to_load += [f"{prefix}PFCands_d0", f"{prefix}PFCands_dz"]

        # with uproot.open(f"{file_name}:Events") as in_file:
        #     events = in_file.arrays(branches_to_load, library="ak")
        all_chunks = []
        for events in uproot.iterate(f"{file_name}:Events", expressions=branches_to_load, library="ak", step_size="100 MB"):
            # Optional: process events here if you want to avoid keeping them all in memory
            all_chunks.append(events)

        # If you really want to concatenate them all after processing:
        events = ak.concatenate(all_chunks)
        
        bkg_mask = None
        total_events = len(events)
        if self.mask:
            bkg_mask = abs(events[f"{prefix}TauTo3Mu_fitted_mass"] - 1.78) > 0.08
            # Filters events that do not have single bkg-looking event
            events = events[ak.any(bkg_mask, axis=1)] # This flattening is needed as otherwise the ak.type of bkg_mask and events would not coincide
            bkg_mask = bkg_mask[ak.any(bkg_mask, axis=1)]
            
        if gen_events_before_selection is None:
            gen_events_before_selection = total_events
        events_to_process = min(self.max_events_to_process, len(events)) if self.max_events_to_process else len(events)
        graphDataset = self.process_event(
            0,
            events_to_process,
            events,
            xsec,
            gen_events_before_selection,
            total_events,
            events_to_process,
            bkg_mask,
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
    
    def connect_fully(self, delta_R: np.ndarray) -> tuple:
        adjacency = np.ones_like(delta_R) - np.identity(delta_R.shape[0])
        edge_connections = np.where(adjacency == 1)
        edge_index = torch.tensor([ edge for edge in zip(edge_connections[0], edge_connections[1]) ], dtype=torch.long)
        return edge_index
        
    def connect_knn(self, delta_R: np.ndarray) -> tuple:
        nn_matrix = np.argsort(delta_R, axis=1)
        knn_matrix = nn_matrix[:, 1:(self.k_knn+1)] # as closest pf_cand is the pf_cand itself and we dont want self loops for now
        edge_set = set()
        for i in range(knn_matrix.shape[0]):
            for j in knn_matrix[i]:
                edge_set.add((i,j))
                edge_set.add((j,i))
         
        edge_index = torch.tensor(list(edge_set), dtype=torch.long) 
        return edge_index

    def connect_r_threshold(self, delta_R: np.ndarray) -> tuple:
        adjacency = (delta_R < self.delta_r_threshold).astype(int) - np.identity(delta_R.shape[0])
        edge_connections = np.where(adjacency == 1)
        edge_index = torch.tensor([ edge for edge in zip(edge_connections[0], edge_connections[1]) ], dtype=torch.long)
        return edge_index

    def build_cone(
            self, 
            pfcands_eta: ak.Array, 
            pfcands_phi: ak.Array, 
            fitted_tau_eta: float, 
            fitted_tau_phi: float, 
            mu1_eta: float,
            mu1_phi: float, 
            mu1_idx: int, 
            mu2_eta: float, 
            mu2_phi: float, 
            mu2_idx: int, 
            mu3_eta: float, 
            mu3_phi: float, 
            mu3_idx: int
        ) -> ak.Array:

        delta_R_pfcands_tau = self.calc_delta_R(pfcands_eta, pfcands_phi, fitted_tau_eta, fitted_tau_phi)

        delta_R_mu1_tau = self.calc_delta_R(mu1_eta, mu1_phi, fitted_tau_eta, fitted_tau_phi)
        delta_R_mu2_tau = self.calc_delta_R(mu2_eta, mu2_phi, fitted_tau_eta, fitted_tau_phi)
        delta_R_mu3_tau = self.calc_delta_R(mu3_eta, mu3_phi, fitted_tau_eta, fitted_tau_phi)
        if self.cone_size < max(delta_R_mu1_tau, delta_R_mu2_tau, delta_R_mu3_tau):
            logging.warning("given cone size is not large enough. Using farthest away tau3mu candidate delta_R as cone size instead.")
            cone_size = max(delta_R_mu1_tau, delta_R_mu2_tau, delta_R_mu3_tau)
        else:
            cone_size = self.cone_size

        cone_mask = (delta_R_pfcands_tau < cone_size) # binary ak.array whether pf_cand is inside cone or not
        
        # TODO extract tau3mu cands from pfcands to prevent them from being in the Graph, st. GNN does not just learn tau mass

        return cone_mask
    
    def calc_delta_R(self, eta, phi, ref_eta, ref_phi):
        delta_eta = eta - ref_eta
        delta_phi_internal = np.abs(phi - ref_phi)
        delta_phi_external = 2*np.pi - np.abs(phi - ref_phi)
        delta_phi = np.minimum(delta_phi_internal, delta_phi_external)
        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)
        return delta_R

        