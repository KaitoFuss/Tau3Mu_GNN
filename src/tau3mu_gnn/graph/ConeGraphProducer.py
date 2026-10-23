import os
import numpy as np

from tau3mu_gnn.utils.CorrectPileupDist import signal_event_weights
import uproot
import torch
from torch_geometric.data import InMemoryDataset, Data
import awkward as ak
from tqdm import tqdm
import vector
import logging
import numba as nb
import ROOT


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
        save_benchmark_score (bool): Whether to save score of current discrimination approach
        extra_label (str): Extra label to add to the processed file name)
        weights (str): What event weights to use (None, 'xsec', 'npvs_rescale') (TODO: implement pt flattening)
        save_fitted_tau_idx (bool): Whether to save the fitted tau index
        mask (bool): Bool whether to apply a mask extracting only tau3mu cands outside vicinity of tau mass, used to get background events
        label (float): Label to add to the graph for supervised learning
        verbose (bool): Whether to print progress bars
        tau_collection_name (str): Name of the fitted tau in the root file (default: "TauTo3Mu_fitted")
        tau_channel (str): Name of tau production channel ('W' or 'HF')
        graph_level_branches (lsit): List of branch names that will be loaded and added as graph level features, previously BDT input features
        slimmed (bool): Whether data was pre skimmed and disregarded Meson resonances and bad PV matches
        shuffle_bool (bool): Whether events should be shuffeled before processing
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
        save_benchmark_score=False,
        weights=None,
        save_fitted_tau_idx=False,
        mask=False,
        label:float=0.,
        verbose=False,
        tau_collection_name="TauTo3Mu_fitted",
        tau_channel="W",
        graph_level_branches=[],
        slimmed=False,
        shuffle_bool=False,
        use_displacement=False,
        unsculpting_mode=None,  
        bdt_preselection_cut=None,
        tau_mass=1.78,
        sideband_half_width=0.08,
        npvs_weights_file=None,
        shuffle_seed=None,
        tree_name="Events",
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
        self.save_benchmark_score = save_benchmark_score
        self.weights = weights
        self.save_fitted_tau_idx = save_fitted_tau_idx
        self.mask = mask
        self.label = label
        self.verbose = verbose
        self.tau_collection_name = tau_collection_name
        self.tau_channel = tau_channel
        self.graph_level_branches = graph_level_branches
        self.slimmed = slimmed
        self.shuffle_bool = shuffle_bool
        self.use_displacement = use_displacement 
        self.unsculpting_mode = unsculpting_mode
        self.edge_connection_methods = {
            "fully": self.connect_fully,
            "knn": self.connect_knn,
            "r_threshold": self.connect_r_threshold,
        } 
        # None: no cut, one graph per candidate. A value: event cut bdt_ISO >= value (before the sideband
        # and the event limit), one graph per event from candidate 0 (background: candidate 0 in the
        # sideband), bdt_iso stored per graph.
        self.bdt_preselection_cut = bdt_preselection_cut
        # Pipeline options (training stage graphs); the defaults keep the old behaviour.
        self.tau_mass = tau_mass
        self.sideband_half_width = sideband_half_width
        self.npvs_weights_file = npvs_weights_file        # None: <root>/signal_<channel>_npvs_weights.npz
        self.shuffle_seed = shuffle_seed                  # None: unseeded np.random
        self.tree_name = tree_name

        assert self.weights in ["xsec", None, "npvs_rescale"], "Only xsec, npvs_rescale and None are supported for the weights argument (more to be added later)"

        assert self.edge_connection_mode in self.edge_connection_methods, f"Unsupported edge_connection_mode: {self.edge_connection_mode}"

        assert self.tau_channel in ["W", "HF"], "Unsupported tau_channel. Please choose either 'W' or 'HF'"

        assert self.unsculpting_mode in ["permutation", "split_pt_uniformly", "delete_muons", None], "Unsupported unsculting mode. Please choose 'permutation', 'split_pt_uniformly', 'delete_muons', None"
        
        self.use_muon_permutation = False
        self.use_split_pt_uniformly = False
        self.delete_muons = False

        if self.unsculpting_mode == "permutation":
            self.use_muon_permutation = True
        elif self.unsculpting_mode == "split_pt_uniformly":
            self.use_split_pt_uniformly = True
        elif self.unsculpting_mode == "delete_muons":
            self.delete_muons = True

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

    def process_event(self, i_start, i_stop, events, xsec, gen_events_before_selection, total_events, events_to_process, bkg_mask, npvs_weights):
        prefix = "Selected_" if self.slimmed else ""
        n_prefix = "Selected" if self.slimmed else ""
        old_prefix = "Slimmed_" if self.slimmed else ""
        graphs = []

        # indices = range(i_start, i_stop) if not self.shuffle_bool else np.random.choice(np.arange(i_start, i_stop), size=i_stop-i_start, replace=False)
        # for i_ev in tqdm(indices, disable=not self.verbose):
        choice = np.random.choice if self.shuffle_seed is None else np.random.default_rng(self.shuffle_seed).choice
        indices = range(i_start, i_stop) if not self.shuffle_bool else choice(np.arange(i_start, len(events)), size=i_stop - i_start, replace=False)
        events = events[indices] 

        if self.mask:
            bkg_mask = bkg_mask[indices]

        for i_ev in tqdm(range(len(events)), disable=not self.verbose):
            n_cones = self.n_cones # TODO implement option to build three cones instead of one big one if n_cones == 3; or maybe change to bool 
            
            event = events[i_ev]

            if self.mask:
                n_tau = event[f"n{n_prefix}TauTo3Mu"] 
                n_tau = min(n_tau, ak.sum(bkg_mask[i_ev]))
            else:
                n_tau = event[f"n{n_prefix}TauTo3Mu"] 

            # if self.bdt_preselection_cut is not None:
            #     raw_tau_mask = bkg_mask[i_ev] if self.mask else ak.ones_like(
            #         event[f"{prefix}{self.tau_collection_name}_pt"], dtype=bool
            #     )
            #     n_tau = int(ak.sum(raw_tau_mask))
            #     n_tau = min(n_tau, 1)
            if self.bdt_preselection_cut is not None:
                n_tau = min(event[f"n{n_prefix}TauTo3Mu"], 1)
            if n_tau == 0:
                logging.warning(f"Skipping this event as selected mask deleted it")
                continue

            pfcands_pt = event[f"{old_prefix}PFCands_pt"]
            pfcands_eta = event[f"{old_prefix}PFCands_eta"]
            pfcands_phi = event[f"{old_prefix}PFCands_phi"]
            pfcands_pdgId = event[f"{old_prefix}PFCands_pdgId"]
            pfcands_mass = event[f"{old_prefix}PFCands_mass"]
            pfcands_charge = event[f"{old_prefix}PFCands_charge"]
            pfcands_puppiWeight = event[f"{old_prefix}PFCands_puppiWeight"]

            if self.bdt_preselection_cut is not None:
                bdt_iso = event["bdt_ISO"]
            
            # Iterating over all tau3mu candidates in event and creating for all a graph in its vicinity 
            for nt in range(n_tau):
                tau_mask = bkg_mask[i_ev] if self.mask else ak.ones_like(event[f"{prefix}{self.tau_collection_name}_pt"], dtype=bool)
                
                # Read data for fitted tau from event
                fitted_tau_pt = event[f"{prefix}{self.tau_collection_name}_pt"][tau_mask][nt] 
                fitted_tau_eta = event[f"{prefix}{self.tau_collection_name}_eta"][tau_mask][nt] 
                fitted_tau_phi = event[f"{prefix}{self.tau_collection_name}_phi"][tau_mask][nt] 
                fitted_tau_mass = event[f"{prefix}{self.tau_collection_name}_mass"][tau_mask][nt] 
                
                # get graph level features for this fitted tau
                graph_level_features = []
                # graph_level_features = ak.Array([event[self.graph_level_branches]])[tau_mask][nt] # this just creates an array of records

                for branch_name in self.graph_level_branches:
                    graph_level_feature = event[branch_name][tau_mask][nt]
                    graph_level_features.append(graph_level_feature)

                if self.tau_channel == "W":
                    TauPlusMET_Puppi_phi = event["TauPlusMET_Puppi_phi"][tau_mask][nt]
                    delta_phi_TauPlusMET_Puppi_fitted_tau = self.deltaphi(TauPlusMET_Puppi_phi, fitted_tau_phi)
                    graph_level_features.append(delta_phi_TauPlusMET_Puppi_fitted_tau)
                
                elif self.tau_channel == "HF":
                    pass # no calculated graph level features



                # Read data for mu1 from event
                # mu1_idx = event[f"{prefix}TauTo3Mu_mu1_idx"][tau_mask][nt] 
                mu1_eta = event[f"{prefix}TauTo3Mu_mu1_eta"][tau_mask][nt] 
                mu1_phi = event[f"{prefix}TauTo3Mu_mu1_phi"][tau_mask][nt] 
                # mu1_pt = event[f"{prefix}TauTo3Mu_mu1_pt"][tau_mask][nt] 
    
                # Read data for mu2 from event
                # mu2_idx = event[f"{prefix}TauTo3Mu_mu2_idx"][tau_mask][nt] 
                mu2_eta = event[f"{prefix}TauTo3Mu_mu2_eta"][tau_mask][nt] 
                mu2_phi = event[f"{prefix}TauTo3Mu_mu2_phi"][tau_mask][nt] 
                # mu2_pt = event[f"{prefix}TauTo3Mu_mu2_pt"][tau_mask][nt] 
                
                # Read data for mu3 from event
                # mu3_idx = event[f"{prefix}TauTo3Mu_mu3_idx"][tau_mask][nt] 
                mu3_eta = event[f"{prefix}TauTo3Mu_mu3_eta"][tau_mask][nt] 
                mu3_phi = event[f"{prefix}TauTo3Mu_mu3_phi"][tau_mask][nt] 
                # mu3_pt = event[f"{prefix}TauTo3Mu_mu3_pt"][tau_mask][nt] 

                # Ensuring that cone is large enough such that all 3mu are inside it; center of cone will be (fitted_tau_eta, fitted_tau_phi)
                cone_mask = self.build_cone(
                    pfcands_eta, 
                    pfcands_phi, 
                    fitted_tau_eta, 
                    fitted_tau_phi, 
                    mu1_eta, 
                    mu1_phi, 
                    # mu1_idx, 
                    mu2_eta, 
                    mu2_phi, 
                    # mu2_idx, 
                    mu3_eta, 
                    mu3_phi, 
                    # mu3_idx
                )
                pfcands_pt_masked = pfcands_pt[cone_mask]
                pfcands_eta_masked = pfcands_eta[cone_mask]
                pfcands_phi_masked = pfcands_phi[cone_mask]
                pfcands_pdgId_masked = pfcands_pdgId[cone_mask]
                pfcands_mass_masked = pfcands_mass[cone_mask]
                pfcands_charge_masked = pfcands_charge[cone_mask]
                pfcands_puppiWeight_masked = pfcands_puppiWeight[cone_mask]
                
                # print(f"Event {i_ev}, tau {nt}: n_pfcands_in_cone={ak.sum(cone_mask)}, delete_muons={self.delete_muons}")
                if len(pfcands_pt_masked) <= 1:
                    logging.warning(f"Empty graph for tau candidate {nt} in event {i_ev}, skipping.")
                    continue

                n_graphs = 5 if self.use_muon_permutation else 1

                # for n_graph in range(n_graphs):
                n_graph = np.random.randint(0, n_graphs)

                if self.use_displacement:
                    pfcands_d0_masked = event[f"{old_prefix}PFCands_d0"][cone_mask]
                    pfcands_dz_masked = event[f"{old_prefix}PFCands_dz"][cone_mask]
                else:
                    pfcands_d0_masked, pfcands_dz_masked = None, None 

                fitted_tau_idx = nt

                pfcands_pt_masked = np.array(pfcands_pt_masked, dtype=np.float32)
                pfcands_eta_masked = np.array(pfcands_eta_masked, dtype=np.float32)
                pfcands_phi_masked = np.array(pfcands_phi_masked, dtype=np.float32)

                if self.use_muon_permutation:
                    pfcands_pt_masked, permuted_rescaled_3muon_mass, permuted_rescaled_3muon_pT, permuted_rescaled_3muon_eta, permuted_rescaled_3muon_phi = self.permute_muons(
                        n_graph, 
                        mu1_eta, 
                        mu1_phi,
                        mu2_eta,
                        mu2_phi,
                        mu3_eta,
                        mu3_phi,
                        pfcands_pt_masked,
                        pfcands_eta_masked,
                        pfcands_phi_masked
                    )

                if self.use_split_pt_uniformly:
                    pfcands_pt_masked, split_rescaled_3muon_mass, split_rescaled_3muon_pT, split_rescaled_3muon_eta, split_rescaled_3muon_phi = self.split_pt_uniformly(
                        mu1_eta, 
                        mu1_phi,
                        mu2_eta,
                        mu2_phi,
                        mu3_eta,
                        mu3_phi,
                        pfcands_pt_masked,
                        pfcands_eta_masked,
                        pfcands_phi_masked
                    )

                if self.use_displacement:  
                    pfcands_d0_masked = np.array(pfcands_d0_masked, dtype=np.float32)   
                    pfcands_dz_masked = np.array(pfcands_dz_masked, dtype=np.float32)   
                else:
                    pfcands_d0_masked = np.zeros(len(pfcands_pt_masked), dtype=np.float32)   
                    pfcands_dz_masked = np.zeros(len(pfcands_pt_masked), dtype=np.float32)   

                if self.use_relative_angles:
                    pfcands_eta_masked = pfcands_eta_masked - fitted_tau_eta
                    # Need care to account for circularity
                    pfcands_phi_masked = (pfcands_phi_masked - fitted_tau_phi + np.pi) % (2*np.pi) - np.pi
                
                pfcands_pdgId_masked = np.array(pfcands_pdgId_masked, dtype=np.int32)
                # Clip masses to zero since for some reason sometimes they are slightly negative
                pfcands_mass_masked = np.clip(np.array(pfcands_mass_masked), a_min=0., a_max=None, dtype=np.float32)
                pfcands_charge_masked = np.array(pfcands_charge_masked, dtype=np.int32) # is i64 fine for charge? Saw that it was saved as i32 in root

                num_nodes = len(pfcands_charge_masked)
                pos = [[e, p] for e, p in zip(pfcands_eta_masked, pfcands_phi_masked)]

                # Converting to np.array and subsequently to torch.tensor as suggested in torch docs for performance
                if self.use_displacement: 
                    features = torch.tensor(np.array([
                        pfcands_pt_masked,
                        pfcands_eta_masked,
                        pfcands_phi_masked,
                        pfcands_pdgId_masked,
                        pfcands_mass_masked,
                        pfcands_puppiWeight_masked,
                        pfcands_d0_masked,   
                        pfcands_dz_masked,
                    ]).T, dtype=torch.float)
                else:
                    features = torch.tensor(np.array([
                        pfcands_pt_masked,
                        pfcands_eta_masked,
                        pfcands_phi_masked,
                        pfcands_pdgId_masked,
                        pfcands_mass_masked, 
                        pfcands_puppiWeight_masked
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
                    
                    # Build Cambridge-Aachen and kt distances between constituents and other edge_atrributes 
                    R = 0.8
                    d_ca = ((matrix_delta_R**2) / (R**2))
                    d_kt = (np.minimum(matrix_pt**2, matrix_pt.T**2) * (matrix_delta_R**2) / (R**2))
                    if num_nodes < 200:
                        # Please also doulbe check whether this indexing is okay, I also did but just to make sure...
                        np_edge_index = np.array(edge_index.t())
                        pair_masses = np.array(pair_masses[np_edge_index[0], np_edge_index[1]])
                        pair_charges = np.array(matrix_charge[np_edge_index[0], np_edge_index[1]])
                        d_ca = np.array(d_ca[np_edge_index[0], np_edge_index[1]])
                        d_kt = np.array(d_kt[np_edge_index[0], np_edge_index[1]])
                    else:
                        pair_masses = np.array([pair_masses[i,j] for i,j in edge_index])
                        pair_charges = np.array([matrix_charge[i,j] for i,j in edge_index])
                        d_ca = np.array([d_ca[i,j] for i,j in edge_index])
                        d_kt = np.array([d_kt[i,j] for i,j in edge_index])
                    
                    # Only keep connected edges and take log to squeeze the distribution
                    # pair_masses = np.clip(np.nan_to_num(np.log(pair_masses)), a_max=1e5, a_min=-1e5)

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
                elif self.weights == "npvs_rescale":
                    graph.w = signal_event_weights(np.array([event["PV_npvs"]]), npvs_weights)[0]

                graph.tau_channel = self.tau_channel

                if len(self.graph_level_branches) > 0:
                    graph_level_features = torch.tensor(graph_level_features, dtype=torch.float)
                    graph_level_features = graph_level_features.unsqueeze(0)
                    graph.graph_level_features = graph_level_features
                else:
                    graph.graph_level_features = torch.tensor([])
                
                # Add the event number if requested
                if self.save_event_number:
                    graph.event_number = torch.tensor([i_ev], dtype=torch.long)

                # Add tau index if requested
                if self.save_fitted_tau_idx:
                    graph.fitted_tau_index = torch.tensor([fitted_tau_idx], dtype=torch.int)

                # Add fitted_tau_mass if requested
                if self.save_fitted_tau_mass:
                    graph.fitted_tau_mass = torch.tensor([fitted_tau_mass], dtype=torch.float)
                if self.use_muon_permutation:
                    graph.permuted_tau_mass = torch.tensor([permuted_rescaled_3muon_mass], dtype=torch.float)
                    graph.permuted_tau_pT = torch.tensor([permuted_rescaled_3muon_pT], dtype=torch.float)
                    graph.permuted_tau_eta = torch.tensor([permuted_rescaled_3muon_eta], dtype=torch.float)
                    graph.permuted_tau_phi = torch.tensor([permuted_rescaled_3muon_phi], dtype=torch.float)
                elif self.use_split_pt_uniformly:
                    graph.permuted_tau_mass = torch.tensor([split_rescaled_3muon_mass], dtype=torch.float)
                    graph.permuted_tau_pT = torch.tensor([split_rescaled_3muon_pT], dtype=torch.float)
                    graph.permuted_tau_eta = torch.tensor([split_rescaled_3muon_eta], dtype=torch.float)
                    graph.permuted_tau_phi = torch.tensor([split_rescaled_3muon_phi], dtype=torch.float)
                else:
                    graph.non_permuted_tau_mass = torch.tensor([fitted_tau_mass], dtype=torch.float)
                    graph.non_permuted_tau_pT = torch.tensor([fitted_tau_pt], dtype=torch.float)
                    graph.non_permuted_tau_eta = torch.tensor([fitted_tau_eta], dtype=torch.float)
                    graph.non_permuted_tau_phi = torch.tensor([fitted_tau_phi], dtype=torch.float)

                if self.bdt_preselection_cut is not None:
                    graph.bdt_iso = torch.tensor([bdt_iso], dtype=torch.float)
                # Add previous isolation approach score if requested
                if self.save_benchmark_score:
                    abs_iso = event[f"{prefix}TauTo3Mu_iso_ptChargedFromPV"][tau_mask][nt]  + max(0., event[f"{prefix}TauTo3Mu_iso_ptPhotons"][tau_mask][nt] - 0.2 * event[f"{prefix}TauTo3Mu_iso_ptChargedFromPU"][tau_mask][nt])
                    rel_iso = -1 * abs_iso / fitted_tau_pt
                    graph.benchmark_score = torch.tensor([rel_iso], dtype=torch.float)
                    
                    # abs_iso_pt05 = event[f"{prefix}TauTo3Mu_iso_ptChargedFromPV_pT05"][tau_mask][nt]  + max(0., event[f"{prefix}TauTo3Mu_iso_ptPhotons_pT05"][tau_mask][nt] - 0.2 * event[f"{prefix}TauTo3Mu_iso_ptChargedFromPU_pT05"][tau_mask][nt])
                    # rel_iso_pt05 = abs_iso_pt05 / fitted_tau_pt
                    # graph.benchmark_score_pt05 = torch.tensor([rel_iso_pt05], dtype=torch.float)

                graphs.append(graph)

        return graphs

    def get_graphs(self, file_name):

        # Check if xsec weights are requested
        if self.weights is None:
            xsec = 1.
            gen_events_before_selection = None
            npvs_weights = None

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
            npvs_weights = None

        if self.weights == "npvs_rescale":
            xsec = 1.
            gen_events_before_selection = None
            npvs_file = self.npvs_weights_file or self.root + f"/signal_{self.tau_channel}_npvs_weights.npz"
            npvs_weights = np.load(npvs_file)["weight_per_bin"]

        prefix = "Selected_" if self.slimmed else ""
        n_prefix = "Selected" if self.slimmed else ""
        old_prefix = "Slimmed_" if self.slimmed else ""
        branches_to_load = [
            # f"n{n_prefix}PFCands",
            f"{old_prefix}PFCands_pt",
            f"{old_prefix}PFCands_eta",
            f"{old_prefix}PFCands_phi",
            f"{old_prefix}PFCands_pdgId",
            f"{old_prefix}PFCands_mass",  
            f"{old_prefix}PFCands_charge",
            f"{old_prefix}PFCands_puppiWeight",
            f"n{n_prefix}TauTo3Mu",
            # f"{prefix}TauTo3Mu_mu1_idx",
            f"{prefix}TauTo3Mu_mu1_eta",
            f"{prefix}TauTo3Mu_mu1_phi",
            # f"{prefix}TauTo3Mu_mu1_pt",
            # f"{prefix}TauTo3Mu_mu2_idx",
            f"{prefix}TauTo3Mu_mu2_eta",
            f"{prefix}TauTo3Mu_mu2_phi",
            # f"{prefix}TauTo3Mu_mu2_pt",
            # f"{prefix}TauTo3Mu_mu3_idx",
            f"{prefix}TauTo3Mu_mu3_eta",
            f"{prefix}TauTo3Mu_mu3_phi",
            # f"{prefix}TauTo3Mu_mu3_pt",
            f"{prefix}TauTo3Mu_fitted_mass",
            f"{prefix}TauTo3Mu_fitted_eta",
            f"{prefix}TauTo3Mu_fitted_phi",
            f"{prefix}TauTo3Mu_fitted_pt",
            "PV_npvs",
            f"{prefix}TauTo3Mu_iso_ptChargedFromPV",
            f"{prefix}TauTo3Mu_iso_ptPhotons",
            f"{prefix}TauTo3Mu_iso_ptChargedFromPU",
            # f"{prefix}TauTo3Mu_iso_ptChargedFromPV_pT05",
            # f"{prefix}TauTo3Mu_iso_ptPhotons_pT05",
            # f"{prefix}TauTo3Mu_iso_ptChargedFromPU_pT05",
        ]

        if self.bdt_preselection_cut is not None:
            branches_to_load += [f"bdt_ISO"]  # adjust branch name to whatever it's called in your ROOT files

        # self.graph_level_branches = [f"{prefix}{b}" for b in self.graph_level_features]
        branches_to_load += self.graph_level_branches

        if self.tau_channel == "W":
            branches_to_load += [
                "TauPlusMET_Puppi_phi", 
            ]
            
        elif self.tau_channel == "HF":
            pass # no additional features needed

        if self.use_displacement:
            branches_to_load += [f"{old_prefix}PFCands_d0", f"{old_prefix}PFCands_dz"] 

        # with uproot.open(f"{file_name}:Events") as in_file:
        #     events = in_file.arrays(branches_to_load, library="ak")
        all_chunks = []
        for events in uproot.iterate(f"{file_name}:{self.tree_name}", expressions=branches_to_load, library="ak", step_size="100 MB"):
            all_chunks.append(events)

        events = ak.concatenate(all_chunks)
        if self.bdt_preselection_cut is not None:
            events = events[events["bdt_ISO"] >= self.bdt_preselection_cut]
        bkg_mask = None
        total_events = len(events)
        if self.mask:
            bkg_mask = abs(events[f"{prefix}TauTo3Mu_fitted_mass"] - self.tau_mass) > self.sideband_half_width
            if self.bdt_preselection_cut is not None:
                keep = ak.fill_none(ak.firsts(bkg_mask), False)   # candidate 0 in the sideband
            else:
                keep = ak.any(bkg_mask, axis=1)   # Filters events that do not have single bkg-looking event
            events = events[keep] # This flattening is needed as otherwise the ak.type of bkg_mask and events would not coincide
            bkg_mask = bkg_mask[keep]
            
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
            npvs_weights,
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
        if not graphs:  # collate of an empty list fails with an IndexError
            raise ValueError(f"ConeGraphProducer: no graphs for {self.output_dataset_name} from {self.raw_file_names}")
        
        data, slices = self.collate(graphs)
        for key, value in data.items():
            if torch.is_tensor(value):
                if torch.isnan(value).any().values:
                    nansnum = torch.isnan(value)
                    data[key] = torch.nan_to_num(value)
                    nansnum = torch.isnan(value)
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
            # mu1_idx: int, 
            mu2_eta: float, 
            mu2_phi: float, 
            # mu2_idx: int, 
            mu3_eta: float, 
            mu3_phi: float, 
            # mu3_idx: int
        ) -> ak.Array:

        delta_R_sq_pfcands_tau = self.calc_delta_R_sq(pfcands_eta, pfcands_phi, fitted_tau_eta, fitted_tau_phi)

        delta_R_sq_mu1_tau = self.calc_delta_R_sq(mu1_eta, mu1_phi, fitted_tau_eta, fitted_tau_phi)
        delta_R_sq_mu2_tau = self.calc_delta_R_sq(mu2_eta, mu2_phi, fitted_tau_eta, fitted_tau_phi)
        delta_R_sq_mu3_tau = self.calc_delta_R_sq(mu3_eta, mu3_phi, fitted_tau_eta, fitted_tau_phi)
        if self.cone_size ** 2 < max(delta_R_sq_mu1_tau, delta_R_sq_mu2_tau, delta_R_sq_mu3_tau):
            logging.warning("given cone size is not large enough. Using farthest away tau3mu candidate delta_R as cone size instead.")
            cone_size_sq = max(delta_R_sq_mu1_tau, delta_R_sq_mu2_tau, delta_R_sq_mu3_tau)
        else:
            cone_size_sq = self.cone_size ** 2

        if self.delete_muons:
            mu_indices = self.get_muon_idx(mu1_eta, mu1_phi, mu2_eta, mu2_phi, mu3_eta, mu3_phi, pfcands_eta, pfcands_phi)
            n = len(pfcands_eta)
            mu_mask = np.zeros(n, dtype=bool)
            mu_mask[mu_indices[0]] = True
            mu_mask[mu_indices[1]] = True
            mu_mask[mu_indices[2]] = True

            non_mu_mask = ak.Array(~mu_mask)
            cone_mask = (delta_R_sq_pfcands_tau < cone_size_sq) & non_mu_mask
        else:
            cone_mask = (delta_R_sq_pfcands_tau < cone_size_sq) # binary ak.array whether pf_cand is inside cone or not

        return cone_mask
    
    def calc_delta_R_sq(self, eta, phi, ref_eta, ref_phi):
        delta_eta = eta - ref_eta
        delta_phi = self.deltaphi(phi, ref_phi)
        delta_R_sq = delta_eta**2 + delta_phi**2
        return delta_R_sq

        
    def deltaphi(self, phi1, phi2):
        delta_phi_internal = np.abs(phi1 - phi2)
        delta_phi_external = 2*np.pi - np.abs(phi1 - phi2)
        delta_phi = np.minimum(delta_phi_internal, delta_phi_external)
        return delta_phi
    
    def _sum_vec(self, vecs):
        return vecs[0] + vecs[1] + vecs[2]

    def invariant_mass(self, vecs):
        return self._sum_vec(vecs).M()

    def pt(self, vecs):
        return self._sum_vec(vecs).Pt()

    def eta(self, vecs):
        return self._sum_vec(vecs).Eta()

    def phi(self, vecs):
        return self._sum_vec(vecs).Phi()

    def get_muon_idx(self, mu1_eta, mu1_phi, mu2_eta, mu2_phi, mu3_eta, mu3_phi, pfcands_eta_masked, pfcands_phi_masked):
        delta_R_sq_mu1 = self.calc_delta_R_sq(mu1_eta, mu1_phi, pfcands_eta_masked, pfcands_phi_masked)
        delta_R_sq_mu2 = self.calc_delta_R_sq(mu2_eta, mu2_phi, pfcands_eta_masked, pfcands_phi_masked)
        delta_R_sq_mu3 = self.calc_delta_R_sq(mu3_eta, mu3_phi, pfcands_eta_masked, pfcands_phi_masked)

        idx_mu1 = np.argmin(delta_R_sq_mu1)
        idx_mu2 = np.argmin(delta_R_sq_mu2)
        idx_mu3 = np.argmin(delta_R_sq_mu3)

        mu_indices = [idx_mu1, idx_mu2, idx_mu3]
        return mu_indices

    def permute_muons(self, n_graph, mu1_eta, mu1_phi, mu2_eta, mu2_phi, mu3_eta, mu3_phi,
                  pfcands_pt_masked, pfcands_eta_masked, pfcands_phi_masked):
        PERMUTATIONS = [
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]
        MUON_MASS = 0.10566

        mu_indices = self.get_muon_idx(mu1_eta, mu1_phi, mu2_eta, mu2_phi, mu3_eta, mu3_phi, pfcands_eta_masked, pfcands_phi_masked)
        perm = PERMUTATIONS[n_graph]

        # Original vector sum pT
        v_orig = [
            ROOT.Math.PtEtaPhiMVector(pfcands_pt_masked[mu_indices[i]],
                                    pfcands_eta_masked[mu_indices[i]],
                                    pfcands_phi_masked[mu_indices[i]],
                                    MUON_MASS)
            for i in range(3)
        ]
        pt_orig = (v_orig[0] + v_orig[1] + v_orig[2]).Pt()
        m_orig = self.invariant_mass(v_orig)
        # print(f"Original pT vector sum: {pt_orig:.6f}, invariant mass: {m_orig:.4f}")
        # Apply permutation
        pfcands_pt_permuted = pfcands_pt_masked.copy()
        pfcands_pt_permuted[mu_indices[0]] = pfcands_pt_masked[mu_indices[perm[0]]]
        pfcands_pt_permuted[mu_indices[1]] = pfcands_pt_masked[mu_indices[perm[1]]]
        pfcands_pt_permuted[mu_indices[2]] = pfcands_pt_masked[mu_indices[perm[2]]]

        # Rescale to restore original vector sum pT
        v_perm = [
            ROOT.Math.PtEtaPhiMVector(pfcands_pt_permuted[mu_indices[i]],
                                    pfcands_eta_masked[mu_indices[i]],
                                    pfcands_phi_masked[mu_indices[i]],
                                    MUON_MASS)
            for i in range(3)
        ]
        pt_perm = (v_perm[0] + v_perm[1] + v_perm[2]).Pt()
        scale = pt_orig / pt_perm
        m_perm = self.invariant_mass(v_perm)
        # print(f"Unscaled pT vector sum: {pt_perm:.6f}, invariant mass: {m_perm:.4f}")
        # print(f"Resulting scale factor: {scale}")
        pfcands_pt_permuted[mu_indices[0]] *= scale
        pfcands_pt_permuted[mu_indices[1]] *= scale
        pfcands_pt_permuted[mu_indices[2]] *= scale
                
        v_rescale = [
            ROOT.Math.PtEtaPhiMVector(pfcands_pt_permuted[mu_indices[i]],
                                    pfcands_eta_masked[mu_indices[i]],
                                    pfcands_phi_masked[mu_indices[i]],
                                    MUON_MASS)
            for i in range(3)
        ]
        tau_rescale = self._sum_vec(v_rescale)
        m_rescale   = tau_rescale.M()
        pt_rescale  = tau_rescale.Pt()
        eta_rescale = tau_rescale.Eta()
        phi_rescale = tau_rescale.Phi()
        # print(f"Rescaled pT vector sum: {pt_rescale:.6f}, invariant mass: {m_rescale:.4f}")

        return pfcands_pt_permuted, m_rescale, pt_rescale, eta_rescale, phi_rescale

    def split_pt_uniformly(self, mu1_eta, mu1_phi, mu2_eta, mu2_phi, mu3_eta, mu3_phi,
                  pfcands_pt_masked, pfcands_eta_masked, pfcands_phi_masked):
        MUON_MASS = 0.10566

        mu_indices = self.get_muon_idx(mu1_eta, mu1_phi, mu2_eta, mu2_phi, mu3_eta, mu3_phi, pfcands_eta_masked, pfcands_phi_masked)

        # Original vector sum pT
        v_orig = [
            ROOT.Math.PtEtaPhiMVector(pfcands_pt_masked[mu_indices[i]],
                                    pfcands_eta_masked[mu_indices[i]],
                                    pfcands_phi_masked[mu_indices[i]],
                                    MUON_MASS)
            for i in range(3)
        ]
        pt_orig = (v_orig[0] + v_orig[1] + v_orig[2]).Pt()
        m_orig = self.invariant_mass(v_orig)
        pt_new = pt_orig / 3
        print(f"Original pT vector sum: {pt_orig:.6f}, invariant mass: {m_orig:.4f}")
        # Apply permutation
        pfcands_pt_permuted = pfcands_pt_masked.copy()
        pfcands_pt_permuted[mu_indices[0]] = pt_new
        pfcands_pt_permuted[mu_indices[1]] = pt_new
        pfcands_pt_permuted[mu_indices[2]] = pt_new

        # Rescale to restore original vector sum pT
        v_perm = [
            ROOT.Math.PtEtaPhiMVector(pfcands_pt_permuted[mu_indices[i]],
                                    pfcands_eta_masked[mu_indices[i]],
                                    pfcands_phi_masked[mu_indices[i]],
                                    MUON_MASS)
            for i in range(3)
        ]
        pt_new_sum = (v_perm[0] + v_perm[1] + v_perm[2]).Pt()
        scale = pt_orig / pt_new_sum
        m_perm = self.invariant_mass(v_perm)
        print(f"Unscaled pT vector sum: {pt_new_sum:.6f}, invariant mass: {m_perm:.4f}")
        print(f"Resulting scale factor: {scale}")
        pfcands_pt_permuted[mu_indices[0]] *= scale
        pfcands_pt_permuted[mu_indices[1]] *= scale
        pfcands_pt_permuted[mu_indices[2]] *= scale

        v_rescale = [
            ROOT.Math.PtEtaPhiMVector(pfcands_pt_permuted[mu_indices[i]],
                                    pfcands_eta_masked[mu_indices[i]],
                                    pfcands_phi_masked[mu_indices[i]],
                                    MUON_MASS)
            for i in range(3)
        ]
        tau_rescale = self._sum_vec(v_rescale)
        m_rescale   = tau_rescale.M()
        pt_rescale  = tau_rescale.Pt()
        eta_rescale = tau_rescale.Eta()
        phi_rescale = tau_rescale.Phi()
        print(f"Rescaled pT vector sum: {pt_rescale:.6f}, invariant mass: {m_rescale:.4f}")

        return pfcands_pt_permuted, m_rescale, pt_rescale, eta_rescale, phi_rescale