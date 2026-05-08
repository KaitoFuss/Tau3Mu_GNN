import collections
import os
import numpy as np
import uproot
import torch
from torch_geometric.data import Dataset, Data
import awkward as ak
from tqdm import tqdm
import vector
import logging
from utils.LundTreeUtilities import tensor_to_tree, prune_tree, tree_to_tensor, get_lund_decomp


class JetGraphProducer(Dataset):

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
        n_store_jets (int): Number of jets to store per event
        delta_r_threshold (float): Threshold for the deltaR distance between constituents to be considered connected (spacial graphs only)
        n_store_cands (int): Number of constituents to store per jet (spacial graphs only)
        max_events_to_process (int): Maximum number of events to process
        use_delta_r_star (bool): Whether to use the deltaR* distance instead of the standard deltaR (TODO: implement)
        use_delta_r_star_star (bool): Whether to use the deltaR** distance instead of the standard deltaR (TODO: implement)
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: :obj:`None`)
        use_relative_angles (bool): Whether to use relative angles (eta - jet_eta, phi - jet_phi) or absolute angles (eta, phi)
        use_dummy_values (bool): Whether to use dummy values for the features instead of the actual values
        save_edge_attributes (bool): Whether to save the edge attributes (pairwise invariant masses, Cambridge-Aachen and kt distances)
        save_n_constituents (bool): Whether to save the number of constituents in the graph
        save_event_number (bool): Whether to save the event number in the graph
        use_lund_decomp (bool): Whether to use the Lund decomposition instead of the spacial graph construction
        n_lund_vars (int): Number of Lund variables to use (3 or 5)
        save_4vectors_in_lund_tree (bool): Whether to save the 4-vectors of the constituents in the Lund tree
        kt_cut (float): kt cut to prune the tree (if None, no pruning is performed)
        extra_label (str): Extra label to add to the processed file name)
        weights (str): What event weights to use (None, 'xsec') (TODO: implement pt flattening)
        extra_obs_to_save_per_jet (list): List of extra jet-level observables to store in the graph (must be available in the root file)
        extra_obs_to_save_per_event (list): List of extra event-level observables to store in the graph (must be available in the root file)
        extra_obs_to_compute_per_event (list): List of extra event-level observables to compute and store in the graph (callable functions that take the event as input and return a float)
        extra_obs_to_load (list): List of extra observables to load from the root file (must be available in the root file)
        save_jet_idx (bool): Whether to save the jet index
        mask (ak.Array): Mask to apply to the root file
        label (float): Label to add to the graph for supervised learning
        verbose (bool): Whether to print progress bars
        input_format (str): Format of the input root files ["PFNanoAOD", "TreeMaker2"]
        jet_collection (str): Name of the jet collection in the root file (default: "FatJet")
        use_displacement (bool): Whether to use the displacement features (d0, dz) in the Lund decomposition (Scouting Data has no displacement information)
        chunk_size (int): Number of graphs to store per chunk file on disk (default: 1000).
            Larger values reduce the number of files but increase per-chunk memory use.
        n_cached_chunks (int): Number of chunk files to keep in memory at once (default: 2).
            Increase for random-access workloads; decrease to save RAM.

    Note:
        Processed datasets are stored as a metadata file ``{name}_meta.pt`` and a set of
        chunk files ``{name}_chunk_{i}.pt`` inside the ``processed/`` sub-directory.
        Datasets processed with the old ``InMemoryDataset``-based format (a single
        ``{name}.pt`` file) must be re-processed before they can be used with this class.
    """

    def __init__(
        self,
        root,
        output_dataset_name,
        n_store_jets=2,
        delta_r_threshold=0.2,
        n_store_cands=None,
        max_events_to_process=None,
        use_delta_r_star=False,
        use_delta_r_star_star=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_relative_angles=True,
        use_dummy_values=False,
        save_edge_attributes=False,
        save_n_constituents=False,
        save_event_number=False,
        use_lund_decomp=False,
        n_lund_vars=3,
        save_4vectors_in_lund_tree=False,
        save_energy_fractions=True,
        kt_cut=None,
        weights=None,
        extra_obs_to_save_per_jet=[],
        extra_obs_to_save_per_event=[],
        extra_obs_to_compute_per_event:list[callable]=[],
        extra_obs_to_load=[],
        save_jet_idx=False,
        mask=None,
        label:float=0.,
        verbose=False,
        input_format="PFNanoAOD",
        jet_collection="FatJet",
        use_displacement=False,
        chunk_size=1000,
        n_cached_chunks=2,
    ):

        self.root = root
        self.n_store_jets = n_store_jets
        self.output_dataset_name = output_dataset_name
        self.n_store_cands = n_store_cands
        self.delta_r_threshold = delta_r_threshold
        self.max_events_to_process = max_events_to_process
        self.use_delta_r_star = use_delta_r_star
        self.use_delta_r_star_star = use_delta_r_star_star
        self.pre_transform = pre_transform
        self.use_relative_angles = use_relative_angles
        self.use_dummy_values = use_dummy_values
        self.save_edge_attributes = save_edge_attributes
        self.save_n_constituents = save_n_constituents
        self.save_event_number = save_event_number
        self.use_lund_decomp = use_lund_decomp
        self.n_lund_vars = n_lund_vars
        self.save_4vectors_in_lund_tree = save_4vectors_in_lund_tree
        self.save_energy_fractions = save_energy_fractions
        self.kt_cut = kt_cut
        self.weights = weights
        self.extra_obs_to_save_per_jet = extra_obs_to_save_per_jet
        self.extra_obs_to_save_per_event = extra_obs_to_save_per_event
        self.extra_obs_to_compute_per_event = extra_obs_to_compute_per_event
        self.extra_obs_to_load = extra_obs_to_load
        self.save_jet_idx = save_jet_idx
        self.mask = mask
        self.label = label
        self.verbose = verbose
        self.input_format = input_format
        self.jet_collection = jet_collection
        self.use_displacement = use_displacement
        self.chunk_size = chunk_size
        self._n_cached_chunks = n_cached_chunks
        # OrderedDict used as a bounded LRU cache: key = chunk index, value = list of graphs
        self._chunk_cache: collections.OrderedDict = collections.OrderedDict()

        assert self.weights in ["xsec", None], "Only xsec and None are supported for the weights argument (more to be added later)"

        # Assert that no incompatible flags are used
        if use_lund_decomp:
            assert use_delta_r_star == False, "use_delta_r_star is not compatible with use_lund_decomp"
            assert use_delta_r_star_star == False, "use_delta_r_star_star is not compatible with use_lund_decomp"
            assert use_relative_angles == True, "use_relative_angles is not compatible with use_lund_decomp"
            assert use_dummy_values == False, "use_dummy_values is not compatible with use_lund_decomp"
            assert save_edge_attributes == False, "save_edge_attributes is not compatible with use_lund_decomp"
        
        else:
            assert kt_cut is None, "kt_cut is only supported for Lund tree representations"

        # If xsec weighs are requested but the number of events is hard capped, warn that an appoximation is used
        if self.weights == "xsec" and self.max_events_to_process:
            logging.warning("Using approximation for initial number of events with max number of events")

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load lightweight metadata from disk (weights, jet indices, total count).
        # No graph tensors are loaded here — those are fetched on-demand in get().
        meta_path = os.path.join(self.processed_dir, f"{self.output_dataset_name}_meta.pt")
        meta = torch.load(meta_path)
        self._n_graphs: int = meta["n_graphs"]
        self._w: torch.Tensor = meta["w"]
        self._jet_idx_all: torch.Tensor = meta.get("jet_idx")

    # ------------------------------------------------------------------
    # Properties that return per-sample values respecting the current
    # index view (set by Dataset.index_select / slicing / permutation).
    # ------------------------------------------------------------------

    @property
    def w(self) -> torch.Tensor:
        """Event weights for the graphs in the current dataset view."""
        if self._indices is None:
            return self._w
        return self._w[self._indices]

    @property
    def jet_idx(self) -> torch.Tensor:
        """Jet indices for the graphs in the current dataset view."""
        if self._jet_idx_all is None:
            raise AttributeError("jet_idx was not saved during processing (save_jet_idx=False)")
        if self._indices is None:
            return self._jet_idx_all
        return self._jet_idx_all[self._indices]

    # ------------------------------------------------------------------
    # torch_geometric.data.Dataset interface
    # ------------------------------------------------------------------

    def len(self) -> int:
        return self._n_graphs

    @property
    def raw_file_names(self):
        return [self.root + "/" + f for f in os.listdir(self.root) if f.endswith(".root")]

    @property
    def processed_file_names(self):
        # Only the metadata file is used to determine whether processing has been done.
        # The chunk files are written alongside it during process() and are discovered
        # at load time via get().
        return [f"{self.output_dataset_name}_meta.pt"]
    
    def get_total_initial_events(self):
        if self.input_format == "PFNanoAOD":
            n_initial = 0
            for f in self.raw_file_names:
                with uproot.open(f) as in_file:
                    if "CutFlow" in in_file:
                        n_initial += in_file["CutFlow"]["Initial"].array()[0]
                    else:
                        raise ValueError("No CutFlow tree found in file")
            return n_initial
        elif self.input_format == "TreeMaker2":
            raise NotImplementedError("TreeMaker2 format not implemented yet")
        else:
            raise ValueError("Invalid input format. Must be either PFNanoAOD or TreeMaker2")

    def process_event(self, i_start, i_stop, events, xsec, gen_events_before_selection, total_events, events_to_process):

        graphs = []

        for i_ev in tqdm(range(i_start, i_stop), disable=not self.verbose):
            if self.input_format == "PFNanoAOD":
                n_jets = min(self.n_store_jets, events.nFatJet[i_ev])
            elif self.input_format == "TreeMaker2":
                if events[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt"][i_ev] is None:
                    continue
                n_jets = min(self.n_store_jets, len(events[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt"][i_ev]))
            for nj in range(n_jets):
                event = events[i_ev]
                if self.input_format == "PFNanoAOD":
                    jet_pt = event[f"{self.jet_collection}_pt"][nj]
                    jet_eta = event[f"{self.jet_collection}_eta"][nj]
                    jet_phi = event[f"{self.jet_collection}_phi"][nj]
                    pf_cands_matching_filter = event[f"{self.jet_collection}PFCands_pFCandsIdx"][event[f"{self.jet_collection}PFCands_jetIdx"] == nj].to_numpy().astype(np.int32)
                    pt = event["PFCands_pt"][pf_cands_matching_filter]
                    eta = event["PFCands_eta"][pf_cands_matching_filter]
                    phi = event["PFCands_phi"][pf_cands_matching_filter]
                    pdgId = event["PFCands_pdgId"][pf_cands_matching_filter]
                    mass = event["PFCands_mass"][pf_cands_matching_filter]
                    if self.use_displacement:
                        d0 = event["PFCands_d0"][pf_cands_matching_filter]   
                        dz = event["PFCands_dz"][pf_cands_matching_filter]  
                    else:
                        d0, dz = None, None 
                    jet_idx = nj
                elif self.input_format == "TreeMaker2":
                    jet_pt = event[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt"][nj]
                    jet_eta = event[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fEta"][nj]
                    jet_phi = event[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPhi"][nj]
                    # In the TreeMaker2 format, we have to count how many constituents are matched to the first nj-1 jets
                    n_constituents_prev = sum([
                        event[f"{self.jet_collection}_constituentsIndexCounts"]
                        for i in range(nj-1)
                              ]) if nj > 0 else 0
                    cands_idx = event[f"{self.jet_collection}_constituentsIndex"][
                        n_constituents_prev:(n_constituents_prev + event[f"{self.jet_collection}_constituentsIndexCounts"][nj])
                        ]
                    pt = event["JetsConstituents/JetsConstituents.fCoordinates.fPt"][cands_idx]
                    eta = event["JetsConstituents/JetsConstituents.fCoordinates.fEta"][cands_idx]
                    phi = event["JetsConstituents/JetsConstituents.fCoordinates.fPhi"][cands_idx]
                    energy = event["JetsConstituents/JetsConstituents.fCoordinates.fE"][cands_idx]
                    mass = vector.array({"pt": pt, "eta": eta, "phi": phi, "E": energy}).mass
                    pdgId = event["JetsConstituents_PdgId"][cands_idx]

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


                if self.use_relative_angles:
                    eta = eta - jet_eta
                    # Need care to account for circularity
                    phi = (phi - jet_phi + np.pi) % (2*np.pi) - np.pi
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

                if not self.use_lund_decomp:
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
                    adjacency = (delta_R < self.delta_r_threshold).astype(int)

                    # If requested, substitute actual adjacency matrix with random values (to check for nconstituents dependency)
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
                    
                else:
                    # Initialize energy fractions
                    energy_fractions = np.array([
                        1.*(np.abs(pdgId) == 11), # electrons
                        1.*(np.abs(pdgId) == 13), # muons
                        1.*(pdgId == 22), # photons
                        1.*((np.abs(pdgId) != 11) & \
                            (np.abs(pdgId) != 13) & \
                            (np.abs(pdgId) != 22)), # hadrons
                    ])
  
                    # Get lund decomposition
                    ret = get_lund_decomp(
                        pt,
                        eta,
                        phi,
                        mass,
                        d0,   
                        dz,   
                        n_lund_vars=self.n_lund_vars,
                        save_4vectors=self.save_4vectors_in_lund_tree,
                        fractions=energy_fractions,
                        save_fractions=self.save_energy_fractions,
                        use_displacement=self.use_displacement  
                        )

                    feature_matrix, adjacency_matrix = ret

                    # Pass features to torch tensor
                    features = torch.tensor(np.array(feature_matrix).T, dtype=torch.float)
                        
                    # Initialize adjacency matrix
                    adjacency = torch.tensor(np.array(adjacency_matrix))

                    if self.kt_cut is not None:
                        # Convert to tree representation
                        root_node = tensor_to_tree(features, adjacency)
                        # Prune the tree
                        root_node, n_nodes = prune_tree(root_node, 0, self.kt_cut)
                        # Convert back to tensor representation
                        features, adjacency = tree_to_tensor(root_node, (n_nodes, features.shape[1]))
                        # TODO: deal with 1-node or empty graphs after pruning

                    # Build the graph
                    graph = Data(
                            x=features,                                   
                            edge_index=torch.tensor(adjacency, dtype=torch.long),
                            num_nodes=len(features),                            
                            num_node_features=int(features.shape[1]),
                            y=torch.Tensor([self.label])
                        )
                        
                # Add the event weight
                if self.weights == "xsec":
                    if self.input_format == "PFNanoAOD":
                        if not self.max_events_to_process:
                            graph.w = torch.tensor([xsec/gen_events_before_selection], dtype=torch.float)
                        else:
                            graph.w = torch.tensor([xsec/(gen_events_before_selection * events_to_process / total_events)], dtype=torch.float)
                    elif self.input_format == "TreeMaker2":
                        graph.w = torch.tensor(event["Weight"], dtype=torch.float)
                        
                # Add the number of constituents if requested
                if self.save_n_constituents:
                    graph.n_constituents = torch.tensor([n_constituents], dtype=torch.long)

                # Add the event number if requested
                if self.save_event_number:
                    graph.event_number = torch.tensor([i_ev], dtype=torch.long)

                # Add jet index if requested
                if self.save_jet_idx:
                    graph["jet_idx"] = torch.tensor([jet_idx], dtype=torch.int)

                # Add any extra jet-level observables if requested
                for obs in self.extra_obs_to_save_per_jet:
                    graph[obs] = torch.tensor([event[obs][nj]], dtype=torch.float)
                for obs in self.extra_obs_to_save_per_event:
                    graph[obs] = torch.tensor([event[obs]], dtype=torch.float)
                for obs in self.extra_obs_to_compute_per_event:
                    graph[obs.__name__] = torch.tensor([obs(event)], dtype=torch.float)
                        
                graphs.append(graph)

        return graphs

    def get_graphs(self, file_name):

        # Check if xsec weights are requested
        if self.weights is None:
            xsec = 1.
            gen_events_before_selection = None

        if self.input_format == "PFNanoAOD":
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
            ] + self.extra_obs_to_load

            if self.use_displacement:
                branches_to_load += ["PFCands_d0", "PFCands_dz"]
            
            for obs in self.extra_obs_to_save_per_event:
                if obs not in branches_to_load: branches_to_load.append(obs)
            for obs in self.extra_obs_to_save_per_jet:
                if obs not in branches_to_load: branches_to_load.append(obs)

            with uproot.open(f"{file_name}:Events") as in_file:
                events = in_file.arrays(branches_to_load, library="ak")

        elif self.input_format == "TreeMaker2":
            branches_to_load = [
                "JetsConstituents/JetsConstituents.fCoordinates.fPt",
                "JetsConstituents/JetsConstituents.fCoordinates.fEta",
                "JetsConstituents/JetsConstituents.fCoordinates.fPhi",
                "JetsConstituents/JetsConstituents.fCoordinates.fE",
                "JetsConstituents_PdgId",
                f"{self.jet_collection}_constituentsIndex",
                f"{self.jet_collection}_constituentsIndexCounts",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fEta",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPhi",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fE",
                "Weight",
            ] + self.extra_obs_to_load

            xsec = None
            gen_events_before_selection = None
       
            for obs in self.extra_obs_to_save_per_event:
                if obs not in branches_to_load: branches_to_load.append(obs)
            for obs in self.extra_obs_to_save_per_jet:
                if obs not in branches_to_load: branches_to_load.append(obs)

            with uproot.open(f"{file_name}:TreeMaker2/PreSelection") as in_file:
                events = in_file.arrays(branches_to_load, library="ak")

            if self.mask:
                events = ak.mask(events, ak.num(events[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt"], axis = 1) > 1)

        total_events = len(events)
        if self.input_format == "PFNanoAOD":
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

    def _iter_graphs_from_file(self, file_name):
        """Generator that yields processed :obj:`Data` objects from a single
        ROOT file, streaming events in batches of ``chunk_size`` to keep
        memory usage bounded during dataset creation.

        Unlike :meth:`get_graphs`, this method never loads the full event
        array into RAM.  ``pre_filter`` and ``pre_transform`` are applied
        to each graph before it is yielded.
        """
        if self.input_format == "PFNanoAOD":
            xsec = 1.
            gen_events_before_selection = None

            if self.weights == "xsec":
                with uproot.open(file_name) as meta_file:
                    if "Metadata" not in meta_file:
                        raise ValueError("No Metadata tree found in file")
                    if "GenCrossSection" not in meta_file["Metadata"].keys():
                        raise ValueError("No GenCrossSection found in Metadata")
                    xsec = meta_file["Metadata"]["GenCrossSection"].array()[0]
                    gen_events_before_selection = self.get_total_initial_events()

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
            ] + self.extra_obs_to_load

            if self.use_displacement:
                branches_to_load += ["PFCands_d0", "PFCands_dz"]

            for obs in self.extra_obs_to_save_per_event:
                if obs not in branches_to_load:
                    branches_to_load.append(obs)
            for obs in self.extra_obs_to_save_per_jet:
                if obs not in branches_to_load:
                    branches_to_load.append(obs)

            # Get the total number of entries without loading any event data
            with uproot.open(f"{file_name}:Events") as tree:
                total_events = tree.num_entries

            if gen_events_before_selection is None:
                gen_events_before_selection = total_events

            events_to_process = (
                min(self.max_events_to_process, total_events)
                if self.max_events_to_process else total_events
            )

            cumulative_offset = 0
            for events_batch in uproot.iterate(
                f"{file_name}:Events",
                branches_to_load,
                step_size=self.chunk_size,
                entry_stop=events_to_process,
                library="ak",
            ):
                graphs = self.process_event(
                    0,
                    len(events_batch),
                    events_batch,
                    xsec,
                    gen_events_before_selection,
                    total_events,
                    events_to_process,
                )
                if self.save_event_number:
                    for g in graphs:
                        g.event_number = g.event_number + cumulative_offset
                cumulative_offset += len(events_batch)
                if self.pre_transform is not None:
                    graphs = [self.pre_transform(g) for g in graphs]
                for g in graphs:
                    if self.pre_filter is None or self.pre_filter(g):
                        yield g

        elif self.input_format == "TreeMaker2":
            branches_to_load = [
                "JetsConstituents/JetsConstituents.fCoordinates.fPt",
                "JetsConstituents/JetsConstituents.fCoordinates.fEta",
                "JetsConstituents/JetsConstituents.fCoordinates.fPhi",
                "JetsConstituents/JetsConstituents.fCoordinates.fE",
                "JetsConstituents_PdgId",
                f"{self.jet_collection}_constituentsIndex",
                f"{self.jet_collection}_constituentsIndexCounts",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fEta",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPhi",
                f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fE",
                "Weight",
            ] + self.extra_obs_to_load

            for obs in self.extra_obs_to_save_per_event:
                if obs not in branches_to_load:
                    branches_to_load.append(obs)
            for obs in self.extra_obs_to_save_per_jet:
                if obs not in branches_to_load:
                    branches_to_load.append(obs)

            with uproot.open(f"{file_name}:TreeMaker2/PreSelection") as tree:
                total_events = tree.num_entries

            events_to_process = (
                min(self.max_events_to_process, total_events)
                if self.max_events_to_process else total_events
            )

            cumulative_offset = 0
            for events_batch in uproot.iterate(
                f"{file_name}:TreeMaker2/PreSelection",
                branches_to_load,
                step_size=self.chunk_size,
                entry_stop=events_to_process,
                library="ak",
            ):
                if self.mask:
                    events_batch = ak.mask(
                        events_batch,
                        ak.num(events_batch[f"{self.jet_collection}/{self.jet_collection}.fCoordinates.fPt"], axis=1) > 1,
                    )
                graphs = self.process_event(
                    0,
                    len(events_batch),
                    events_batch,
                    None,
                    None,
                    total_events,
                    events_to_process,
                )
                if self.save_event_number:
                    for g in graphs:
                        g.event_number = g.event_number + cumulative_offset
                cumulative_offset += len(events_batch)
                if self.pre_transform is not None:
                    graphs = [self.pre_transform(g) for g in graphs]
                for g in graphs:
                    if self.pre_filter is None or self.pre_filter(g):
                        yield g

    def get(self, idx: int) -> Data:
        """Load and return the graph at global position *idx*.

        Graphs are stored on disk in fixed-size chunk files.  The relevant
        chunk is loaded into a small in-memory LRU cache so that sequential
        and batch-level access patterns reuse already-loaded chunks without
        keeping the entire dataset in RAM.
        """
        chunk_idx = idx // self.chunk_size
        pos = idx % self.chunk_size

        if chunk_idx not in self._chunk_cache:
            chunk_path = os.path.join(
                self.processed_dir,
                f"{self.output_dataset_name}_chunk_{chunk_idx}.pt",
            )
            chunk = torch.load(chunk_path)
            # Evict the oldest entry when the cache is full
            if len(self._chunk_cache) >= self._n_cached_chunks:
                self._chunk_cache.popitem(last=False)
            self._chunk_cache[chunk_idx] = chunk
        else:
            # Move to the end to mark as most-recently used
            self._chunk_cache.move_to_end(chunk_idx)

        return self._chunk_cache[chunk_idx][pos]

    def process(self):
        # Stream graphs from every input file and write them to disk in
        # chunk files as they are produced, so that the peak RAM usage
        # during processing is proportional to chunk_size, not to the
        # total dataset size.
        buffer = []
        chunk_idx = 0
        n_graphs = 0
        w_list = []
        jet_idx_list = [] if self.save_jet_idx else None
        default_weight = torch.tensor([1.0])

        def _flush_chunk(data, idx):
            chunk_path = os.path.join(
                self.processed_dir,
                f"{self.output_dataset_name}_chunk_{idx}.pt",
            )
            torch.save(data, chunk_path)

        for file in self.raw_file_names:
            print(f"Processing {file}")
            for g in self._iter_graphs_from_file(file):
                buffer.append(g)
                w_list.append(g.w if hasattr(g, "w") else default_weight)
                if jet_idx_list is not None:
                    jet_idx_list.append(g.jet_idx)
                n_graphs += 1

                if len(buffer) == self.chunk_size:
                    _flush_chunk(buffer, chunk_idx)
                    buffer = []
                    chunk_idx += 1

        # Flush any graphs that didn't fill a complete chunk
        if buffer:
            _flush_chunk(buffer, chunk_idx)
            chunk_idx += 1

        meta = {
            "n_graphs": n_graphs,
            "n_chunks": chunk_idx,
            "chunk_size": self.chunk_size,
            "w": torch.cat(w_list),
            "jet_idx": torch.cat(jet_idx_list) if jet_idx_list is not None else None,
        }
        torch.save(
            meta,
            os.path.join(self.processed_dir, f"{self.output_dataset_name}_meta.pt"),
        )

