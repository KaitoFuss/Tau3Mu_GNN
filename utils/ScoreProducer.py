import os
import numpy as np
import uproot
import torch
from torch_geometric.data import Data
import awkward as ak
from tqdm import tqdm
import vector
import logging
from torch_geometric.utils import negative_sampling
from utils.LundTreeUtilities import tensor_to_tree, prune_tree, tree_to_tensor, get_lund_decomp
import utils.awkward_array_utilities as akUtl

def __cast_unknown_type_branches(ak_array, field):

    type_text = str(ak.type(ak_array))

    if "unknown" in type_text:
        ak_array = akUtl.as_type(ak_array, np.float64)
        
    elif "?" in type_text:
        new_type_text = type_text.split("*")[-1].replace("?", "").replace(" ", "")
        ak_array = akUtl.as_type(ak_array, getattr(np, new_type_text))
        
    return ak_array


def __prepare_array(ak_array, field):
    return ak.packed(ak.without_parameters(__cast_unknown_type_branches(ak_array, field)))


def __is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""

    t = ak.type(a)
    if isinstance(t, ak._ext.ArrayType):
        if isinstance(t.type, ak._ext.PrimitiveType):
            return True
        if isinstance(t.type, ak._ext.ListType) and isinstance(
            t.type.type, ak._ext.PrimitiveType
        ):
            return True
    return False


def __zip_composite(ak_array):
    """Zip together branches belonging to the same collection.
    
    Args:
        ak_array (ak.Array): 
    
    Returns:
        ak.Array
    """

    # Additional naming scheme to allow composite object readback
    _rename_lookup = {
        "pt": "/.fPt",
        "eta": "/.fEta",
        "phi": "/.fPhi",
        "energy": "/.fE",
        "x": "/.fX",
        "y": "/.fY",
        "z": "/.fZ",
        "j": "jerFactor",
        "o": "origIndex",
    }

    dict_ = {}
    for n in ak_array.fields:
        ak_array_ = __prepare_array(ak_array[n], n)
        if __is_rootcompat(ak_array_):
            dict_[_rename_lookup.get(n, n)] = ak_array_

    return ak.zip(dict_)


def __make_tree_maker_event_tree(events):
    """Prepare dict with simple branches and collections that uproot can write.
    
    Args:
        events (ak.Array): Events opened with the NTreeMaker schema.

    Returns:
        dict[str, ak.Array]
    """

    out = {}
    for bname in events.fields:
        if events[bname].fields:
            sub_collection = [  # Handling sub collection first
                x.replace("Counts", "")
                for x in events[bname].fields
                if x.endswith("Counts")
            ]
            if sub_collection:
                for subname in sub_collection:
                    if events[bname][subname].fields:
                        out[f"{bname}_{subname}"] = __zip_composite(
                            ak.flatten(events[bname][subname], axis=-1)
                        )
                    else:
                        out[f"{bname}_{subname}"] = ak.flatten(
                            events[bname][subname], axis=-1
                        )
            out[bname] = __zip_composite(events[bname])
        else:
            out[bname] = __prepare_array(events[bname], bname)
    return out


def write_tree_maker_root_file(output_file_name, events=None, trees={}, mode="recreate"):
    """Write events opened with the TreeMaker or NTreeMaker schema to a ROOT file.
    
    Args:
        output_file_name (str)
        events (ak.Array)
        trees (dict[str, any]): Other trees with no collections
            Keys are tree name
            Values are tree content
        mode (str): "recreate" or "update"
    """

    with getattr(uproot, mode)(output_file_name) as output_file:
        if events is not None:
            output_file["Events"] = __make_tree_maker_event_tree(events)

        for tree_name, tree in trees.items():
            output_file[tree_name] = ak.Array(tree)




def __get_keys_from_file(file_):
    """Return keys of a ROOT file open with uproot4 without suffix ;1.

    Args:
        file_ (uproot.writing.writable.WritableDirectory)

    Returns:
        list[str]
    """

    return [key.split(";")[0] for key in file_.keys()]



def __get_collection_and_variable_names(field):
    """Get collection and variable names from field name.

    Args:
        field (str)

    Returns:
        tuple(str, str)
    """

    split = field.split("_")
    if len(split) > 2:
        return split[0], "_".join(split[1:])
    if len(split) == 2:
        return split[0], split[1]
    else:
        return split[0], None


def __get_jagged_collections(ak_array):
    """Get collections in the ak array.

    Args:
       ak_array (awkward.Array): ak array with fields

    Returns:
        list[str]
    """

    collections = []
    for field in ak_array.fields:
        collection_candidate, variable = __get_collection_and_variable_names(field)
        if variable is None or collection_candidate is None: continue
        if not "var" in str(ak.type(ak_array[field])): continue
        if len(ak_array[field]) == 1: continue
        if collection_candidate not in collections:
            collections.append(collection_candidate)

    return collections
        

def __make_nano_aod_event_tree(ak_array, sort_by_name=True):
    """Get object that can be written as a TTree to a ROOT file with uproot4.

    Args:
       ak_array (awkward.Array): ak array with fields

    Returns:
        list[str]
    """

    collections = __get_jagged_collections(ak_array)

    # Book dictionaries for arrays to zip together and that does not not have to be zipped
    branches_to_zip = {collection: {} for collection in collections}
    single_branches = {}
    
    branch_to_skip = []
    
    for field in ak_array.fields:
        for n_branch in ["n" + field.split("_")[0], "n" + field[0]]:
            if n_branch in ak_array.fields:
                branch_to_skip.append(n_branch)
    # Fill in those dictionaries
    for field in ak_array.fields:
        type_ = akUtl.get_type(ak_array[field])
        if field in branch_to_skip:
            continue
        if type_ in ["str", "string"]:
            continue
        if ak.count(ak.fill_none(ak_array[field], 0), axis=None) == 0:
            continue
        
        is_collection_variable = False
        
        collection_candidate, variable = __get_collection_and_variable_names(field)
        for collection in collections:
            if collection_candidate == collection:
                branches_to_zip[collection][variable] = __cast_unknown_type_branches(ak_array[field], field)
                is_collection_variable = True

        if not is_collection_variable:
            single_branches[field] = __cast_unknown_type_branches(ak_array[field], field)

    # Zip branches and add everything in a dict
    collections_branches = {}
    for collection in branches_to_zip.keys():

        counter = {}
        for branch_name, branch in branches_to_zip[collection].items():
            size = ak.count(branch, axis=None)
            if size not in counter.keys():
                counter[size] = 1
            else:
                counter[size] += 1
        
        if len(counter.keys()) != 1:
            highest_count_key = 0
            highest_count_value = 0
            for k, v in counter.items():
                if v > highest_count_value:
                    highest_count_key = k
                    highest_count_value = v
            items = dict(branches_to_zip[collection].items()).copy()
            for branch_name, branch in items.items():
                size = ak.count(branch, axis=None)
                if size != highest_count_key:
                    
                    single_branches[f"{collection}_{branch_name}"] = branches_to_zip[collection].pop(branch_name)
 
        try:
            collections_branches[collection] = ak.zip(branches_to_zip[collection])
        except:
            print(" ")

    branches = collections_branches
    if sort_by_name:
        single_branches_names = sorted(single_branches)
    else:
        single_branches_names = single_branches.keys()
    for branch_name in single_branches_names:
        if branch_name not in branches.keys():
            branches[branch_name] = single_branches[branch_name]

    return branches


def write_nano_aod_root_file(output_file_name, events=None, trees={}, mode="recreate"):
    """Write events opened with the BaseSchema to a ROOT file.
    
    Args:
        output_file_name (str)
        events (ak.Array)
        trees (dict[str, any]): Other trees with no collections
            Keys are tree name
            Values are tree content
        mode (str): "recreate" or "update"
    """


    with getattr(uproot, mode)(output_file_name) as output_file:
        if events is not None:
            output_file["Events"] = __make_nano_aod_event_tree(events)
        for tree_name, tree in trees.items():
            output_file[tree_name] = ak.Array(tree)

class ScoreProducer:

    """
    Produces new root files with GNN losses for each jet.
    Note:
    model = GNN model to use for scoring
    normalizer = normalizer for the normalization of the features (dictionary with keys "means" and "stds")
    """


    def __init__(
        self,
        root_file,
        delta_r_threshold=0.2,
        n_store_cands=None,
        max_events_to_process=None,
        use_relative_angles=True,
        save_n_constituents=False,
        save_event_number=False,
        n_lund_vars=3,
        save_4vectors_in_lund_tree=False,
        save_energy_fractions=True,
        kt_cut=None,
        weights=None,
        save_jet_idx=False,
        mask=None,
        label:float=0.,
        verbose=False,
        input_format="PFNanoAOD",
        jet_collection="FatJet",
        use_displacement=False,
        model=None,
        normalizer=None,
        output_folder=None,
        alpha=1,
    ):

        self.root_file = root_file
        self.n_store_cands = n_store_cands
        self.delta_r_threshold = delta_r_threshold
        self.max_events_to_process = max_events_to_process
        self.use_relative_angles = use_relative_angles
        self.save_n_constituents = save_n_constituents
        self.save_event_number = save_event_number
        self.n_lund_vars = n_lund_vars
        self.save_4vectors_in_lund_tree = save_4vectors_in_lund_tree
        self.save_energy_fractions = save_energy_fractions
        self.kt_cut = kt_cut
        self.weights = weights
        self.save_jet_idx = save_jet_idx
        self.mask = mask
        self.label = label
        self.verbose = verbose
        self.input_format = input_format
        self.jet_collection = jet_collection
        self.use_displacement = use_displacement
        self.model = model
        self.normalizer = normalizer
        self.output_folder = output_folder
        self.alpha = alpha

        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)

        assert self.weights in ["xsec", None], "Only xsec and None are supported for the weights argument (more to be added later)"

        # Assert that no incompatible flags are used

        # If xsec weighs are requested but the number of events is hard capped, warn that an appoximation is used
        if self.weights == "xsec" and self.max_events_to_process:
            logging.warning("Using approximation for initial number of events with max number of events")
        self.model.eval()
        self.process()


    def process_event(self, i_start, i_stop, events, xsec, gen_events_before_selection, total_events, events_to_process, file_name):

        graphs = []

        for i_ev in tqdm(range(i_start, i_stop), disable=not self.verbose):
            if self.input_format == "PFNanoAOD":
                n_jets = events.nFatJet[i_ev]
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
                        #edge_index=torch.tensor(adjacency, dtype=torch.int32),
                        edge_index = adjacency.clone().detach(),
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
                    graph.n_constituents = torch.tensor([n_constituents], dtype=torch.int32)

                # Add the event number if requested
                if self.save_event_number:
                    graph.event_number = torch.tensor([i_ev], dtype=torch.int32)

                # Add jet index if requested
                if self.save_jet_idx:
                    graph["jet_idx"] = torch.tensor([jet_idx], dtype=torch.int32)

                # Add any extra jet-level observables if requested

                graph.event_number = torch.tensor([i_ev], dtype=torch.int32)
                graph.jet_number = torch.tensor([nj], dtype=torch.int32)
                graph.score = torch.tensor([0.], dtype=torch.float)
                graph.file_name = file_name
                        
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
                            gen_events_before_selection = in_file["CutFlow"]["Initial"].array()[0]
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
                branches_to_load += [
                    "PFCands_d0",
                    "PFCands_dz"
                ]



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
            ]

            xsec = None
            gen_events_before_selection = None
       
            for obs in self.extra_obs_to_save_per_event: ####
                if obs not in branches_to_load: branches_to_load.append(obs) ####
            for obs in self.extra_obs_to_save_per_jet: ####
                if obs not in branches_to_load: branches_to_load.append(obs) ####

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
            file_name,
        )

        return graphDataset


    def process(self):
        # Read data into huge `Data` list.
        graphs = []
        model = self.model
        model.eval()
        
        # Dictionary to hold scores
        scores = []
        graphs = self.get_graphs(self.root_file)

        # Compute scores for each graph
        for graph in tqdm(graphs, desc="Computing scores", disable=not self.verbose):
            x = graph.x
            edges = graph.edge_index
            negative_edge_index = negative_sampling(edges)
            if negative_edge_index.numel()==0:
                scores.append(float(-1.0))
            else:
                if self.normalizer is not None:
                    x = (x-self.normalizer["means"])/self.normalizer["stds"]
                
                # Calculate losses
                features_loss = model.features_loss(x, edges)
                edges_loss = model.edges_loss(
                    x,
                    edges,
                    negative_sampling(edges),
                )
                loss = edges_loss + self.alpha * features_loss

                scores.append(float(loss.detach()))

            

        # Read the original ROOT file
        with uproot.open(self.root_file) as root_file:
            events_tree = root_file["Events"]
            original_data = events_tree.arrays(library="ak")
            if "Metadata" in root_file:
                metadata = root_file["Metadata"]
                original_metadata = metadata.arrays(library="ak")
            if "CutFlow" in root_file:
                cutflow = root_file["CutFlow"]
                original_cutflow = cutflow.arrays(library="ak")
            trees = {"Metadata": original_metadata, "CutFlow": original_cutflow}
            

            scores = ak.unflatten(scores, ak.num(root_file["Events"]["FatJet_pt"].array()))

            
        # Add new branch to the data
        original_data["FatJet_GNNScore"] = ak.Array(scores)

        # Write the updated data to a new ROOT file
        output_file = (os.path.join(self.output_folder, os.path.basename(self.root_file).replace(".root","_with_scores.root")) if self.output_folder else self.root_file.replace(".root","_with_scores.root"))

        with uproot.recreate(output_file) as new_file:
            print(f"Writing to {output_file}")
            write_tree_maker_root_file(
                output_file_name=output_file,
                events=original_data,
                trees=trees,
            )