"""GNN scores for the W / HF channel (stage 3).

score_events() reads the input files once, builds one graph per event around the first tau
candidate (events with bdt_ISO >= bdt_preselection_cut only, if a cut is given) and scores it with
every model. For each model it writes a copy of each input file with the branch GNN_score
(-1 for events without a score) and an npz with the scores and tau masses of the scored events.
"""
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
from tau3mu_gnn.utils.LundTreeUtils import tensor_to_tree, prune_tree, tree_to_tensor, get_lund_decomp
import tau3mu_gnn.utils.AwkwardArrayUtils as akUtl
import matplotlib.pyplot as plt
import awkward as ak
import ROOT
import csv
import time
import copy
from pathlib import Path
import pandas as pd
from tau3mu_gnn.gnn_training.gnn_inputs import gnn_branches, graph_level_branches
from tau3mu_gnn.utils.score_merge import KEYS, scores_in_entry_order, write_with_score

def __cast_unknown_type_branches(ak_array, field):

    type_text = str(ak.type(ak_array))

    if "unknown" in type_text:
        ak_array = akUtl.as_type(ak_array, np.float64)
        
    elif "?" in type_text:
        new_type_text = type_text.split("*")[-1].replace("?", "").replace(" ", "")
        ak_array = akUtl.as_type(ak_array, getattr(np, new_type_text))
        
    return ak_array


def __prepare_array(ak_array, field):
    # return ak.packed(ak.without_parameters(__cast_unknown_type_branches(ak_array, field)))
    return ak.to_packed(ak.without_parameters(__cast_unknown_type_branches(ak_array, field)))


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

def _write_with_scores(score_dict, in_file, out_file, tree_name):
    """Copy in_file to out_file and add GNN_score from score_dict ((run, lumi, event) -> score)."""
    with uproot.open(in_file) as f:
        entry_keys = f[tree_name].arrays(KEYS, library="pd")
    score_keys = pd.DataFrame(list(score_dict), columns=KEYS)
    scores = scores_in_entry_order(entry_keys, score_keys, list(score_dict.values()))
    write_with_score(in_file, out_file, tree_name, "GNN_score", scores)
    print(f"Wrote in total {int(np.sum(scores != -1.))} valid GNN Scores in to ROOT file {out_file}")


class ScoreProducer:

    """
    Produces new root files with GNN losses for each jet.
    Note:
    model = GNN model to use for scoring
    normalizer = normalizer for the normalization of the features (dictionary with keys "means" and "stds")
    """


    def __init__(
        self,
        inputs,
        models,
        outputs,
        npz_paths,
        tree_name="Events",
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
        tau_channel="W",
        tau_collection_name="TauTo3Mu_fitted",
        slimmed=False,
        shuffle_bool=False,
        use_displacement=False,
        unsculpting_mode=None,
        device=None,
        bdt_preselection_cut=None
    ):

        if set(outputs) != set(models) or set(npz_paths) != set(models):
            raise ValueError(f"outputs ({sorted(outputs)}) and npz_paths ({sorted(npz_paths)}) "
                             f"must have the model names as keys ({sorted(models)})")
        self.inputs = list(inputs)
        self.model_names = list(models)
        self.models = [models[n][0] for n in self.model_names]
        self.normalizers = [models[n][1] for n in self.model_names]
        self.outputs = {n: dict(outputs[n]) for n in self.model_names}
        self.npz_paths = {n: Path(npz_paths[n]) for n in self.model_names}
        self.tree_name = tree_name
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
        self.tau_channel = tau_channel
        self.tau_collection_name = tau_collection_name
        self.slimmed = slimmed
        self.shuffle_bool = shuffle_bool
        self.use_displacement = use_displacement 
        self.device = device
        self.unsculpting_mode = unsculpting_mode
        self.edge_connection_methods = {
            "fully": self.connect_fully,
            "knn": self.connect_knn,
            "r_threshold": self.connect_r_threshold,
        } 
        self.bdt_preselection_cut = bdt_preselection_cut

        assert self.weights in ["xsec", None, "npvs_rescale"], "Only xsec, npvs_rescale and None are supported for the weights argument (more to be added later)"

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

        # Assert that no incompatible flags are used
        prefix = "Selected_" if self.slimmed else ""

        self.graph_level_branches = graph_level_branches(self.tau_channel, prefix)

        # If xsec weighs are requested but the number of events is hard capped, warn that an appoximation is used
        if self.weights == "xsec" and self.max_events_to_process:
            logging.warning("Using approximation for initial number of events with max number of events")
        for model in self.models:
            model.eval()
        self.process()

    def process_event(self, i_start, i_stop, events, xsec, gen_events_before_selection, total_events, events_to_process, bkg_mask, npvs_weights, score_dict_for_root):

        prefix = "Selected_" if self.slimmed else ""
        n_prefix = "Selected" if self.slimmed else ""
        old_prefix = "Slimmed_" if self.slimmed else ""

        # indices = range(i_start, i_stop) if not self.shuffle_bool else np.random.choice(np.arange(i_start, i_stop), size=i_stop-i_start, replace=False)
        # for i_ev in tqdm(indices, disable=not self.verbose):
        indices = range(i_start, i_stop) if not self.shuffle_bool else np.random.choice(np.arange(i_start, len(events)), size=i_stop - i_start, replace=False)
        events = events[indices] 
        runs = events["run"]
        lumis = events["luminosityBlock"]
        event_ids = events["event"]
        tau_masses = []
        scores_dict_all_models = {name: [] for name in self.model_names}

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
            
            if self.bdt_preselection_cut is not None and event["bdt_ISO"] < self.bdt_preselection_cut:
                continue

            pfcands_pt = event[f"{old_prefix}PFCands_pt"]
            pfcands_eta = event[f"{old_prefix}PFCands_eta"]
            pfcands_phi = event[f"{old_prefix}PFCands_phi"]
            pfcands_pdgId = event[f"{old_prefix}PFCands_pdgId"]
            pfcands_mass = event[f"{old_prefix}PFCands_mass"]
            pfcands_charge = event[f"{old_prefix}PFCands_charge"]
            pfcands_pupppiWeight = event[f"{old_prefix}PFCands_puppiWeight"]
            
            # Iterating over all tau3mu candidates in event and creating for all a graph in its vicinity 
            key = (runs[i_ev], lumis[i_ev], event_ids[i_ev])
            tau_mask = bkg_mask[i_ev] if self.mask else ak.ones_like(event[f"{prefix}{self.tau_collection_name}_pt"], dtype=bool)
            if n_tau == 0:
                # print("Skipping event as all TauCands have not been selected")
                continue
            nt = 0  # first candidate, as the BDT (bdt_ISO) and the limit (fitted_mass[0]) use
            # Read data for fitted tau from event
            fitted_tau_pt = event[f"{prefix}{self.tau_collection_name}_pt"][tau_mask][nt] 
            fitted_tau_eta = event[f"{prefix}{self.tau_collection_name}_eta"][tau_mask][nt] 
            fitted_tau_phi = event[f"{prefix}{self.tau_collection_name}_phi"][tau_mask][nt] 
            fitted_tau_mass = event[f"{prefix}{self.tau_collection_name}_mass"][tau_mask][nt] 

            graph_level_features = []

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
            pfcands_puppiWeight_masked = pfcands_pupppiWeight[cone_mask]

            if len(pfcands_pt_masked) <= 1:
                    logging.warning(f"Empty graph for tau candidate {nt} in event {i_ev}, skipping.")
                    continue

            if self.use_displacement:
                pfcands_d0_masked = event[f"{old_prefix}PFCands_d0"][cone_mask]
                pfcands_dz_masked = event[f"{old_prefix}PFCands_dz"][cone_mask]
            else:
                pfcands_d0_masked, pfcands_dz_masked = None, None 

            fitted_tau_idx = nt

            pfcands_pt_masked = np.array(pfcands_pt_masked, dtype=np.float32)
            pfcands_eta_masked = np.array(pfcands_eta_masked, dtype=np.float32)
            pfcands_phi_masked = np.array(pfcands_phi_masked, dtype=np.float32)

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
                    pfcands_puppiWeight_masked,
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
                graph.w = npvs_weights[event["PV_npvs"]]
            
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

            # Add previous isolation approach score if requested
            if self.save_benchmark_score:
                abs_iso = event[f"{prefix}TauTo3Mu_iso_ptChargedFromPV"][tau_mask][nt]  + max(0., event[f"{prefix}TauTo3Mu_iso_ptPhotons"][tau_mask][nt] - 0.2 * event[f"{prefix}TauTo3Mu_iso_ptChargedFromPU"][tau_mask][nt])
                rel_iso = -1 * abs_iso / fitted_tau_pt
                graph.benchmark_score = torch.tensor([rel_iso], dtype=torch.float)
                
                # abs_iso_pt05 = event[f"{prefix}TauTo3Mu_iso_ptChargedFromPV_pT05"][tau_mask][nt]  + max(0., event[f"{prefix}TauTo3Mu_iso_ptPhotons_pT05"][tau_mask][nt] - 0.2 * event[f"{prefix}TauTo3Mu_iso_ptChargedFromPU_pT05"][tau_mask][nt])
                # rel_iso_pt05 = abs_iso_pt05 / fitted_tau_pt
                # graph.benchmark_score_pt05 = torch.tensor([rel_iso_pt05], dtype=torch.float)
                
            score_model_dict, tau_mass = self.create_score(graph)
            for name in self.model_names:
                if key in score_dict_for_root[name]:
                    print(f"WARNING: overwriting score in {name} for {key}")
                score_dict_for_root[name][key] = score_model_dict[name][0]

                scores_dict_all_models[name].append(score_model_dict[name][0])
            tau_masses.append(tau_mass[0])

        return scores_dict_all_models, tau_masses

    def get_all_scores(self, file_name):

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
            npvs_weights = np.load(os.path.join(os.path.dirname(file_name), "npvs_weights.npz"))["weight_per_bin"]

        prefix = "Selected_" if self.slimmed else ""
        branches_to_load = gnn_branches(self.tau_channel, self.slimmed, self.bdt_preselection_cut is not None,
                                        self.use_displacement, self.weights, self.save_benchmark_score)

        all_scores_dict = {name: [] for name in self.model_names}
        all_tau_masses = []
        score_dicts = {name: {} for name in self.model_names}
        counter = 1
        for events in uproot.iterate(f"{file_name}:{self.tree_name}", expressions=branches_to_load, library="ak", step_size="2000 MB"):
            print(f"Processing bunch {counter} of {file_name}")
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
            scores_per_model, tau_mass = self.process_event(
                0,
                events_to_process,
                events,
                xsec,
                gen_events_before_selection,
                total_events,
                events_to_process,
                bkg_mask,
                npvs_weights,
                score_dicts,
            )
            # score, tau_mass = self.create_score(graphDataset)
            for name in self.model_names:
                all_scores_dict[name].append(scores_per_model[name])
            all_tau_masses.append(tau_mass)

            counter += 1

        for name in self.model_names:
            all_scores_dict[name] = ak.flatten(all_scores_dict[name])
        all_tau_masses = ak.flatten(all_tau_masses)
        
        return all_scores_dict, all_tau_masses, score_dicts

    def create_score(self, graph):
        # Dictionary to hold scores
        scores = {}
        tau_mass = graph.fitted_tau_mass

        for i, model in enumerate(self.models):
            graph_copy = graph.clone()
            device = self.device
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                # Compute scores for each graph
                graph_copy.x = graph_copy.x.float()  # Ensure float32
                # print(f"Now model {self.model_names[i]}")
                if self.normalizers[i] is not None:
                    means_x = torch.tensor(self.normalizers[i]["means_x"], dtype=torch.float, device=graph_copy.x.device)
                    stds_x = torch.tensor(self.normalizers[i]["stds_x"], dtype=torch.float, device=graph_copy.x.device)
                    # print(f"Node features before norm - mean: {graph_copy.x.mean(dim=0)}, std: {graph_copy.x.std(dim=0)}")
                    graph_copy.x = (graph_copy.x - means_x) / stds_x
                    # print(f"Feature mean and std: {means_x, stds_x}")
                    # print(f"Node features after norm - mean: {graph_copy.x.mean(dim=0)}, std: {graph_copy.x.std(dim=0)}")

                    means_edge = torch.tensor(self.normalizers[i]["means_edge"], dtype=torch.float, device=graph_copy.edge_attr.device)
                    stds_edge = torch.tensor(self.normalizers[i]["stds_edge"], dtype=torch.float, device=graph_copy.edge_attr.device)
                    # print(f"Edge features before norm - mean: {graph_copy.edge_attr.mean(dim=0)}, std: {graph_copy.edge_attr.std(dim=0)}")
                    graph_copy.edge_attr = (graph_copy.edge_attr - means_edge) / stds_edge
                    # print(f"Edge mean and std: {means_edge, stds_edge}")
                    # print(f"Edge features after norm - mean: {graph_copy.edge_attr.mean(dim=0)}, std: {graph_copy.edge_attr.std(dim=0)}")
                    
                    means_graph = torch.tensor(self.normalizers[i]["means_graph"], dtype=torch.float, device=graph_copy.graph_level_features.device)
                    stds_graph = torch.tensor(self.normalizers[i]["stds_graph"], dtype=torch.float, device=graph_copy.graph_level_features.device)
                    # print(f"Graph features before norm - mean: {graph_copy.graph_level_features.mean(dim=0)}, std: {graph_copy.graph_level_features.std(dim=0)}")
                    graph_copy.graph_level_features = (graph_copy.graph_level_features - means_graph) / stds_graph
                    # print(f"Graph feature mean and std: {means_graph, stds_graph}")
                    # print(f"Graph features after norm - mean: {graph_copy.graph_level_features.mean(dim=0)}, std: {graph_copy.graph_level_features.std(dim=0)}")
                    
                graph_copy = graph_copy.to(device)
                out = model(graph_copy)
                # print(f"model output (logit): {out[:,0]}")
                scores[self.model_names[i]] = out[:,0].cpu().detach().numpy()
        return scores, tau_mass


    def process(self):
        all_scores = {name: [] for name in self.model_names}
        all_tau_masses = []
        for in_file in self.inputs:
            print(f"Processing {in_file}")
            scores, tau_masses, score_dicts = self.get_all_scores(in_file)
            all_tau_masses += ak.to_list(tau_masses)
            for name in self.model_names:
                all_scores[name] += ak.to_list(scores[name])
                _write_with_scores(score_dicts[name], in_file, self.outputs[name][in_file], self.tree_name)

        for name in self.model_names:
            npz_path = self.npz_paths[name]
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, scores=all_scores[name], tau_mass=all_tau_masses)
            print(f"Raw scores and tau masses saved to: {npz_path}")

    
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
        delta_phi_internal = np.abs(phi - ref_phi)
        delta_phi_external = 2*np.pi - np.abs(phi - ref_phi)
        delta_phi = np.minimum(delta_phi_internal, delta_phi_external)
        delta_R_sq = delta_eta**2 + delta_phi**2
        return delta_R_sq

    def deltaphi(self, phi1, phi2):
        delta_phi_internal = np.abs(phi1 - phi2)
        delta_phi_external = 2*np.pi - np.abs(phi1 - phi2)
        delta_phi = np.minimum(delta_phi_internal, delta_phi_external)
        return delta_phi
        
    def get_muon_idx(self, mu1_eta, mu1_phi, mu2_eta, mu2_phi, mu3_eta, mu3_phi, pfcands_eta_masked, pfcands_phi_masked):
        delta_R_sq_mu1 = self.calc_delta_R_sq(mu1_eta, mu1_phi, pfcands_eta_masked, pfcands_phi_masked)
        delta_R_sq_mu2 = self.calc_delta_R_sq(mu2_eta, mu2_phi, pfcands_eta_masked, pfcands_phi_masked)
        delta_R_sq_mu3 = self.calc_delta_R_sq(mu3_eta, mu3_phi, pfcands_eta_masked, pfcands_phi_masked)

        idx_mu1 = np.argmin(delta_R_sq_mu1)
        idx_mu2 = np.argmin(delta_R_sq_mu2)
        idx_mu3 = np.argmin(delta_R_sq_mu3)

        mu_indices = [idx_mu1, idx_mu2, idx_mu3]
        return mu_indices


def score_events(inputs, models, outputs, npz_paths, tau_channel, unsculpting_mode, bdt_preselection_cut,
                 tree_name="Events", device=None, max_events_to_process=None):
    """inputs: input files. models: name -> (model, normalizer). outputs: name -> {input: output file}.
    npz_paths: name -> npz file. See the module docstring for the outputs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ScoreProducer(inputs=inputs, models=models, outputs=outputs, npz_paths=npz_paths, tree_name=tree_name,
                  edge_connection_mode="knn", delta_r_threshold=None, k_knn=10,
                  max_events_to_process=max_events_to_process, save_edge_attributes=True, verbose=True,
                  tau_channel=tau_channel, slimmed=True, shuffle_bool=max_events_to_process is not None,
                  use_displacement=True, device=device, unsculpting_mode=unsculpting_mode,
                  bdt_preselection_cut=bdt_preselection_cut)
