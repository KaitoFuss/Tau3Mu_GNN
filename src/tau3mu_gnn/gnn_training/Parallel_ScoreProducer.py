### WARNING: this has been created with one Claude prompt and is not up to date to current ScoreProducer, so use with care

import os
import numpy as np
import uproot
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import awkward as ak
from tqdm import tqdm
import vector
import logging
from array import array
import time
import copy
import ROOT
from multiprocessing import Pool
import tau3mu_gnn.utils.AwkwardArrayUtils as akUtl


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers (must be top-level for multiprocessing pickling)
# ──────────────────────────────────────────────────────────────────────────────

def _min_dphi_arr(phi_arr: np.ndarray, ref_phi: float) -> np.ndarray:
    """Vectorised minimum |Δφ| between an array and a scalar, handling wrap-around."""
    d = np.abs(phi_arr - ref_phi)
    return np.minimum(d, 2.0 * np.pi - d)


def _min_dphi_scalar(phi: float, ref_phi: float) -> float:
    """Minimum |Δφ| between two scalars, handling wrap-around."""
    d = abs(phi - ref_phi)
    return min(d, 2.0 * np.pi - d)


def _connect_fully(n: int) -> np.ndarray:
    """Return (2, E) edge index for a fully connected graph (no self-loops)."""
    src, dst = np.where(~np.eye(n, dtype=bool))
    return np.stack([src, dst], axis=0)  # shape (2, E)


def _connect_knn(delta_R: np.ndarray, k: int) -> np.ndarray:
    """Return (2, E) edge index for k-NN graph (undirected, no self-loops)."""
    nn_matrix = np.argsort(delta_R, axis=1)
    knn_matrix = nn_matrix[:, 1:k + 1]
    edge_set = set()
    for i in range(knn_matrix.shape[0]):
        for j in knn_matrix[i]:
            edge_set.add((i, int(j)))
            edge_set.add((int(j), i))
    if not edge_set:
        return np.zeros((2, 0), dtype=np.int64)
    edges = np.array(sorted(edge_set), dtype=np.int64).T  # (2, E)
    return edges


def _connect_r_threshold(delta_R: np.ndarray, threshold: float) -> np.ndarray:
    """Return (2, E) edge index for radius-threshold graph (no self-loops)."""
    adjacency = (delta_R < threshold).astype(int) - np.eye(delta_R.shape[0], dtype=int)
    src, dst = np.where(adjacency == 1)
    return np.stack([src, dst], axis=0)


def build_graph_from_event(args: tuple) -> Data:
    """
    Pure top-level function: constructs a single PyG Data object from per-event arrays.

    This function is intentionally top-level (not a method) so it can be pickled
    by multiprocessing.Pool.map.

    Parameters
    ----------
    args : tuple
        All inputs packed as a single tuple (required by pool.map).

    Returns
    -------
    torch_geometric.data.Data
    """
    (
        pfcands_pt,           # np.float32[N]
        pfcands_eta,          # np.float32[N]   — already cone-masked
        pfcands_phi,          # np.float32[N]   — already cone-masked
        pfcands_pdgId,        # np.int32[N]
        pfcands_mass,         # np.float32[N]
        pfcands_charge,       # np.int32[N]
        pfcands_puppiWeight,  # np.float32[N]
        pfcands_d0,           # np.float32[N]
        pfcands_dz,           # np.float32[N]
        fitted_tau_eta,       # float
        fitted_tau_phi,       # float
        fitted_tau_mass,      # float
        graph_level_features, # list[float]
        label,                # float
        save_edge_attributes, # bool
        use_relative_angles,  # bool
        use_displacement,     # bool
        edge_connection_mode, # str: "fully" | "knn" | "r_threshold"
        cone_size,            # float  (unused here, kept for completeness)
        delta_r_threshold,    # float | None
        k_knn,                # int | None
        tau_channel,          # str: "W" | "HF"
        run,                  # int
        lumi,                 # int
        event_id,             # int
    ) = args

    # ── Relative angles ──────────────────────────────────────────────────────
    if use_relative_angles:
        pfcands_eta = pfcands_eta - fitted_tau_eta
        pfcands_phi = (pfcands_phi - fitted_tau_phi + np.pi) % (2.0 * np.pi) - np.pi

    # ── Node features ────────────────────────────────────────────────────────
    if use_displacement:
        feat_arrays = [
            pfcands_pt,
            pfcands_eta,
            pfcands_phi,
            pfcands_pdgId.astype(np.float32),
            pfcands_mass,
            pfcands_puppiWeight,
            pfcands_d0,
            pfcands_dz,
        ]
    else:
        feat_arrays = [
            pfcands_pt,
            pfcands_eta,
            pfcands_phi,
            pfcands_pdgId.astype(np.float32),
            pfcands_mass,
            pfcands_puppiWeight,
        ]

    features = torch.tensor(np.stack(feat_arrays, axis=1), dtype=torch.float)
    n = len(pfcands_pt)

    # ── Delta-R matrix (needed for knn / r_threshold / edge attrs) ───────────
    # Only compute if needed to avoid wasting time for fully-connected
    need_delta_R = (edge_connection_mode != "fully") or save_edge_attributes
    if need_delta_R:
        d_eta = pfcands_eta[:, None] - pfcands_eta[None, :]   # (N, N)
        d_phi_raw = np.abs(pfcands_phi[:, None] - pfcands_phi[None, :])
        d_phi = np.minimum(d_phi_raw, 2.0 * np.pi - d_phi_raw)
        delta_R = np.sqrt(d_eta ** 2 + d_phi ** 2)            # (N, N)
    else:
        delta_R = None

    # ── Edge index ───────────────────────────────────────────────────────────
    if edge_connection_mode == "fully":
        edge_np = _connect_fully(n)
    elif edge_connection_mode == "knn":
        edge_np = _connect_knn(delta_R, k_knn)
    elif edge_connection_mode == "r_threshold":
        edge_np = _connect_r_threshold(delta_R, delta_r_threshold)
    else:
        raise ValueError(f"Unknown edge_connection_mode: {edge_connection_mode}")

    edge_index = torch.tensor(edge_np, dtype=torch.long)  # (2, E)

    # ── Edge attributes ──────────────────────────────────────────────────────
    edge_attr = None
    if save_edge_attributes and edge_np.shape[1] > 0:
        src = edge_np[0]
        dst = edge_np[1]

        # Pair-wise invariant mass via vector library
        lv = vector.array({
            "pt":   pfcands_pt,
            "eta":  pfcands_eta,
            "phi":  pfcands_phi,
            "mass": pfcands_mass,
        })
        pair_m = (lv[src] + lv[dst]).mass

        # Cambridge-Aachen and kt jet distances
        R = 0.8
        pt_i, pt_j = pfcands_pt[src], pfcands_pt[dst]
        dR_pairs = delta_R[src, dst]
        d_ca = dR_pairs ** 2 / R ** 2
        d_kt = np.minimum(pt_i ** 2, pt_j ** 2) * dR_pairs ** 2 / R ** 2

        pair_charge = pfcands_charge[src].astype(np.float32) * pfcands_charge[dst].astype(np.float32)

        edge_attr = torch.tensor(
            np.stack([
                pair_m.astype(np.float32),
                pair_charge,
                d_ca.astype(np.float32),
                d_kt.astype(np.float32),
            ], axis=1),
            dtype=torch.float,
        )
    elif save_edge_attributes and edge_np.shape[1] == 0:
        # Degenerate graph (single node etc.) — return zero-dim edge_attr
        n_feat = 4
        edge_attr = torch.zeros((0, n_feat), dtype=torch.float)

    # ── Assemble PyG Data ────────────────────────────────────────────────────
    graph = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n,
        y=torch.tensor([label], dtype=torch.float),
        fitted_tau_mass=torch.tensor([fitted_tau_mass], dtype=torch.float),
        graph_level_features=torch.tensor(
            graph_level_features, dtype=torch.float
        ).unsqueeze(0),   # shape (1, F_graph)
        # Keep (run, lumi, event) so we can reconstruct score_dict_for_root
        # after batched inference without having to re-derive valid indices.
        run=torch.tensor([run], dtype=torch.long),
        lumi=torch.tensor([lumi], dtype=torch.long),
        event_id=torch.tensor([event_id], dtype=torch.long),
    )
    return graph


# ──────────────────────────────────────────────────────────────────────────────
# ROOT file utilities (unchanged from original — kept here for completeness)
# ──────────────────────────────────────────────────────────────────────────────

def __cast_unknown_type_branches(ak_array, field):
    type_text = str(ak.type(ak_array))
    if "unknown" in type_text:
        ak_array = akUtl.as_type(ak_array, np.float64)
    elif "?" in type_text:
        new_type_text = type_text.split("*")[-1].replace("?", "").replace(" ", "")
        ak_array = akUtl.as_type(ak_array, getattr(np, new_type_text))
    return ak_array


def __prepare_array(ak_array, field):
    return ak.to_packed(ak.without_parameters(__cast_unknown_type_branches(ak_array, field)))


def __is_rootcompat(a):
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
    _rename_lookup = {
        "pt": "/.fPt", "eta": "/.fEta", "phi": "/.fPhi", "energy": "/.fE",
        "x": "/.fX", "y": "/.fY", "z": "/.fZ", "j": "jerFactor", "o": "origIndex",
    }
    dict_ = {}
    for n in ak_array.fields:
        ak_array_ = __prepare_array(ak_array[n], n)
        if __is_rootcompat(ak_array_):
            dict_[_rename_lookup.get(n, n)] = ak_array_
    return ak.zip(dict_)


def __make_tree_maker_event_tree(events):
    out = {}
    for bname in events.fields:
        if events[bname].fields:
            sub_collection = [
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
    with getattr(uproot, mode)(output_file_name) as output_file:
        if events is not None:
            output_file["Events"] = __make_tree_maker_event_tree(events)
        for tree_name, tree in trees.items():
            output_file[tree_name] = ak.Array(tree)


def __get_keys_from_file(file_):
    return [key.split(";")[0] for key in file_.keys()]


def __get_collection_and_variable_names(field):
    split = field.split("_")
    if len(split) > 2:
        return split[0], "_".join(split[1:])
    if len(split) == 2:
        return split[0], split[1]
    else:
        return split[0], None


def __get_jagged_collections(ak_array):
    collections = []
    for field in ak_array.fields:
        collection_candidate, variable = __get_collection_and_variable_names(field)
        if variable is None or collection_candidate is None:
            continue
        if "var" not in str(ak.type(ak_array[field])):
            continue
        if len(ak_array[field]) == 1:
            continue
        if collection_candidate not in collections:
            collections.append(collection_candidate)
    return collections


def __make_nano_aod_event_tree(ak_array, sort_by_name=True):
    collections = __get_jagged_collections(ak_array)
    branches_to_zip = {collection: {} for collection in collections}
    single_branches = {}
    branch_to_skip = []

    for field in ak_array.fields:
        for n_branch in ["n" + field.split("_")[0], "n" + field[0]]:
            if n_branch in ak_array.fields:
                branch_to_skip.append(n_branch)

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

    collections_branches = {}
    for collection in branches_to_zip.keys():
        counter = {}
        for branch_name, branch in branches_to_zip[collection].items():
            size = ak.count(branch, axis=None)
            counter[size] = counter.get(size, 0) + 1

        if len(counter) != 1:
            highest_count_key = max(counter, key=counter.get)
            items = dict(branches_to_zip[collection].items()).copy()
            for branch_name, branch in items.items():
                size = ak.count(branch, axis=None)
                if size != highest_count_key:
                    single_branches[f"{collection}_{branch_name}"] = branches_to_zip[collection].pop(branch_name)

        try:
            collections_branches[collection] = ak.zip(branches_to_zip[collection])
        except Exception:
            print(" ")

    branches = collections_branches
    single_branches_names = sorted(single_branches) if sort_by_name else single_branches.keys()
    for branch_name in single_branches_names:
        if branch_name not in branches:
            branches[branch_name] = single_branches[branch_name]

    return branches


def write_nano_aod_root_file(output_file_name, events=None, trees={}, mode="recreate"):
    with getattr(uproot, mode)(output_file_name) as output_file:
        if events is not None:
            output_file["Events"] = __make_nano_aod_event_tree(events)
        for tree_name, tree in trees.items():
            output_file[tree_name] = ak.Array(tree)


# ──────────────────────────────────────────────────────────────────────────────
# ScoreProducer
# ──────────────────────────────────────────────────────────────────────────────

class ScoreProducer:
    """
    Produces new ROOT files with GNN scores for each tau-to-3mu candidate.

    Speed improvements over v1
    --------------------------
    1. Graph construction is parallelised across CPU workers via
       multiprocessing.Pool — workers call the module-level
       ``build_graph_from_event`` function, which can be pickled.
    2. GNN inference runs in batches on GPU via PyG DataLoader instead of
       one graph at a time.
    3. ROOT score merging uses a C++-level TTree::CopyTree call rather than
       a per-entry Python loop.

    Parameters
    ----------
    n_workers : int | None
        Number of CPU workers for graph construction.
        None → use min(os.cpu_count(), 8).
        Set to 1 to disable multiprocessing (useful for debugging).
    inference_batch_size : int
        Number of graphs per GPU batch during inference.  Tune upward
        if GPU memory allows; lower if you get OOM errors.
    """

    def __init__(
        self,
        root,
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
        label: float = 0.,
        verbose=False,
        tau_channel="W",
        tau_collection_name="TauTo3Mu_fitted",
        slimmed=False,
        shuffle_bool=False,
        use_displacement=False,
        models=None,
        device=None,
        normalizers=None,
        output_folders=None,
        storage_folder=None,
        model_names=None,
        # ── new performance parameters ──
        n_workers: int = None,
        inference_batch_size: int = 256,
    ):
        self.root = root
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
        self.models = models
        self.device = device
        self.normalizers = normalizers
        self.output_folders = output_folders
        self.storage_folder = storage_folder
        self.model_names = model_names
        self.n_workers = n_workers if n_workers is not None else min(os.cpu_count() or 1, 8)
        self.inference_batch_size = inference_batch_size

        if not (len(self.models) == len(self.normalizers) == len(self.output_folders)):
            raise ValueError(
                f"models ({len(self.models)}), normalizers ({len(self.normalizers)}) "
                f"and output_folders ({len(self.output_folders)}) must have the same length"
            )

        if self.output_folders:
            for output_folder in self.output_folders:
                os.makedirs(output_folder, exist_ok=True)

        assert self.weights in ["xsec", None, "npvs_rescale"], \
            "Only xsec, npvs_rescale and None are supported for the weights argument"

        assert self.edge_connection_mode in ["fully", "knn", "r_threshold"], \
            f"Unknown edge_connection_mode: {self.edge_connection_mode}"

        # ── graph-level branch definitions ───────────────────────────────────
        prefix = "Selected_" if self.slimmed else ""

        if self.tau_channel == "W":
            self.graph_level_branches = [
                f'{prefix}TauTo3Mu_fitted_pt',
                'TauPlusMET_Tau_Puppi_mT',
                'TauPlusMET_PuppiMET_pt',
                'TauPlusMET_Puppi_pt',
                'TauPlusMET_PuppiMETminPz',
                'TauPlusMET_PuppiMETmaxPz',
                f'{prefix}TauTo3Mu_dZmu12',
                f'{prefix}TauTo3Mu_dZmu13',
                f'{prefix}TauTo3Mu_dZmu23',
                f'{prefix}TauTo3Mu_sigLxy_3muVtxBS',
                f'{prefix}TauTo3Mu_vtx_chi2*({prefix}TauTo3Mu_vtx_Ndof**-1)',
                f'{prefix}TauTo3Mu_CosAlpha2D_LxyP3mu',
            ]
        elif self.tau_channel == "HF":
            self.graph_level_branches = [
                f'Muon_pt[{prefix}TauTo3Mu_mu3_idx]',
                f'Muon_isGlobal[{prefix}TauTo3Mu_mu1_idx]',
                f'Muon_isGlobal[{prefix}TauTo3Mu_mu2_idx]',
                f'Muon_isGlobal[{prefix}TauTo3Mu_mu3_idx]',
                f'{prefix}TauTo3Mu_dZmu12',
                f'{prefix}TauTo3Mu_dZmu13',
                f'{prefix}TauTo3Mu_dZmu23',
                f'{prefix}TauTo3Mu_sigLxy_3muVtxBS',
                f'{prefix}TauTo3Mu_CosAlpha2D_LxyP3mu',
                f'{prefix}TauTo3Mu_vtx_chi2*({prefix}TauTo3Mu_vtx_Ndof**-1)',
            ]

        if self.weights == "xsec" and self.max_events_to_process:
            logging.warning("Using approximation for initial number of events with max number of events")

        for model in self.models:
            model.eval()

        self.process()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def raw_file_names(self):
        return [
            os.path.join(self.root, f)
            for f in os.listdir(self.root)
            if f.endswith(".root")
        ]

    def _prefixes(self):
        """Return the three prefix strings used throughout."""
        prefix = "Selected_" if self.slimmed else ""
        n_prefix = "Selected" if self.slimmed else ""
        old_prefix = "Slimmed_" if self.slimmed else ""
        return prefix, n_prefix, old_prefix

    def get_total_initial_events(self):
        """Sum of total initial events across all files (for xsec normalisation)."""
        total = 0
        for file in self.raw_file_names:
            with uproot.open(file) as f:
                if "Metadata" in f:
                    total += f["Metadata"]["NEvents"].array()[0]
        return total

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 — Build argument tuples for build_graph_from_event
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_args_for_chunk(
        self,
        events: ak.Array,
        bkg_mask,           # ak.Array[bool] | None
        runs: np.ndarray,
        lumis: np.ndarray,
        event_ids: np.ndarray,
    ) -> list:
        """
        Iterate over events in a chunk and build one argument tuple per valid
        candidate.  All heavy numpy work (cone masking, feature extraction) is
        done here so that ``build_graph_from_event`` (called in workers) only
        needs to do edge building and tensor creation.

        Returns a list of arg tuples consumed by build_graph_from_event.
        """
        prefix, n_prefix, old_prefix = self._prefixes()
        tau_collection = self.tau_collection_name
        args_list = []

        n_events = len(events)
        event_range = range(n_events)

        # Optional shuffle (preserve original behaviour)
        if self.shuffle_bool:
            event_range = np.random.permutation(n_events)

        # Apply max_events_to_process cap
        if self.max_events_to_process:
            event_range = list(event_range)[: self.max_events_to_process]

        for i_ev in event_range:
            ev = events[i_ev]

            # ── Tau candidate selection ───────────────────────────────────────
            n_tau = int(ev[f"n{n_prefix}TauTo3Mu"])
            if self.mask:
                n_tau = min(n_tau, int(ak.sum(bkg_mask[i_ev])))
            if n_tau == 0:
                continue

            tau_mask = (
                bkg_mask[i_ev]
                if self.mask
                else ak.ones_like(ev[f"{prefix}{tau_collection}_pt"], dtype=bool)
            )

            # Highest-pT candidate
            nt = int(np.argmax(np.array(ev[f"{prefix}{tau_collection}_pt"][tau_mask])))

            fitted_tau_pt   = float(ev[f"{prefix}{tau_collection}_pt"][tau_mask][nt])
            fitted_tau_eta  = float(ev[f"{prefix}{tau_collection}_eta"][tau_mask][nt])
            fitted_tau_phi  = float(ev[f"{prefix}{tau_collection}_phi"][tau_mask][nt])
            fitted_tau_mass = float(ev[f"{prefix}{tau_collection}_mass"][tau_mask][nt])

            # ── PF candidate arrays (full event, pre-mask) ───────────────────
            pfc_eta = np.array(ev[f"{old_prefix}PFCands_eta"], dtype=np.float32)
            pfc_phi = np.array(ev[f"{old_prefix}PFCands_phi"], dtype=np.float32)
            pfc_pt  = np.array(ev[f"{old_prefix}PFCands_pt"],  dtype=np.float32)

            # ── Cone mask ────────────────────────────────────────────────────
            dR_sq_pfcands = (
                (pfc_eta - fitted_tau_eta) ** 2
                + _min_dphi_arr(pfc_phi, fitted_tau_phi) ** 2
            )

            mu_etas = [
                float(ev[f"{prefix}TauTo3Mu_mu{k}_eta"][tau_mask][nt])
                for k in (1, 2, 3)
            ]
            mu_phis = [
                float(ev[f"{prefix}TauTo3Mu_mu{k}_phi"][tau_mask][nt])
                for k in (1, 2, 3)
            ]

            dR_sq_mus = [
                (me - fitted_tau_eta) ** 2 + _min_dphi_scalar(mp, fitted_tau_phi) ** 2
                for me, mp in zip(mu_etas, mu_phis)
            ]

            cone_sq = max(self.cone_size ** 2, *dR_sq_mus)
            if self.cone_size ** 2 < max(dR_sq_mus):
                logging.warning(
                    "Cone size too small for muon candidates — expanding to farthest muon ΔR."
                )

            cone_mask = dR_sq_pfcands < cone_sq   # bool np.ndarray

            # ── Apply cone mask ──────────────────────────────────────────────
            def _extract(branch_name, dtype):
                return np.array(ev[f"{old_prefix}PFCands_{branch_name}"], dtype=dtype)[cone_mask]

            pfc_pt_m      = pfc_pt[cone_mask]
            pfc_eta_m     = pfc_eta[cone_mask]
            pfc_phi_m     = pfc_phi[cone_mask]
            pfc_pdgId_m   = _extract("pdgId",      np.int32)
            pfc_mass_m    = np.clip(_extract("mass", np.float32), 0., None)
            pfc_charge_m  = _extract("charge",     np.int32)
            pfc_pweight_m = _extract("puppiWeight", np.float32)

            if self.use_displacement:
                pfc_d0_m = _extract("d0", np.float32)
                pfc_dz_m = _extract("dz", np.float32)
            else:
                n_pfc = int(cone_mask.sum())
                pfc_d0_m = np.zeros(n_pfc, dtype=np.float32)
                pfc_dz_m = np.zeros(n_pfc, dtype=np.float32)

            # ── Graph-level features ─────────────────────────────────────────
            gl_feats = []
            for branch_name in self.graph_level_branches:
                gl_feats.append(float(ev[branch_name][tau_mask][nt]))

            if self.tau_channel == "W":
                puppi_phi = float(ev["TauPlusMET_Puppi_phi"][tau_mask][nt])
                dphi = _min_dphi_scalar(puppi_phi, fitted_tau_phi)
                gl_feats.append(dphi)
            # HF: no additional calculated features

            # ── Pack into tuple ──────────────────────────────────────────────
            args_list.append((
                pfc_pt_m,
                pfc_eta_m,
                pfc_phi_m,
                pfc_pdgId_m,
                pfc_mass_m,
                pfc_charge_m,
                pfc_pweight_m,
                pfc_d0_m,
                pfc_dz_m,
                fitted_tau_eta,
                fitted_tau_phi,
                fitted_tau_mass,
                gl_feats,
                self.label,
                self.save_edge_attributes,
                self.use_relative_angles,
                self.use_displacement,
                self.edge_connection_mode,
                self.cone_size,
                self.delta_r_threshold,
                self.k_knn,
                self.tau_channel,
                int(runs[i_ev]),
                int(lumis[i_ev]),
                int(event_ids[i_ev]),
            ))

        return args_list

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 — Normalise a batch (in-place) using stored normaliser stats
    # ──────────────────────────────────────────────────────────────────────────

    def _normalise_batch(self, batch, model_idx: int):
        """Apply z-score normalisation to node, edge, and graph-level features."""
        norm = self.normalizers[model_idx]
        if norm is None:
            return

        dev = self.device

        means_x = torch.tensor(norm["means_x"], dtype=torch.float, device=dev)
        stds_x  = torch.tensor(norm["stds_x"],  dtype=torch.float, device=dev)
        batch.x = (batch.x - means_x) / stds_x

        if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
            means_e = torch.tensor(norm["means_edge"], dtype=torch.float, device=dev)
            stds_e  = torch.tensor(norm["stds_edge"],  dtype=torch.float, device=dev)
            batch.edge_attr = (batch.edge_attr - means_e) / stds_e

        means_g = torch.tensor(norm["means_graph"], dtype=torch.float, device=dev)
        stds_g  = torch.tensor(norm["stds_graph"],  dtype=torch.float, device=dev)
        batch.graph_level_features = (batch.graph_level_features - means_g) / stds_g

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 — Parallel graph construction + batched inference for one chunk
    # ──────────────────────────────────────────────────────────────────────────

    def _process_chunk(
        self,
        events: ak.Array,
        bkg_mask,
        runs: np.ndarray,
        lumis: np.ndarray,
        event_ids: np.ndarray,
        score_dict_for_root: dict,
    ):
        """
        Full pipeline for one uproot chunk:
          1. Build arg tuples (single-threaded, cheap).
          2. Construct graphs in parallel (multiprocessing).
          3. Run batched GPU inference for each model.
          4. Populate score_dict_for_root.

        Returns
        -------
        scores_dict : dict[model_name -> list[float]]
        tau_masses  : list[float]
        """

        # ── 1. Argument preparation ──────────────────────────────────────────
        args_list = self._prepare_args_for_chunk(events, bkg_mask, runs, lumis, event_ids)

        if not args_list:
            return {name: [] for name in self.model_names}, []

        # ── 2. Graph construction in parallel ────────────────────────────────
        if self.n_workers > 1:
            # chunksize tuned for typical graph sizes (~50-300 nodes)
            chunksize = max(1, len(args_list) // (self.n_workers * 4))
            with Pool(processes=self.n_workers) as pool:
                graphs = pool.map(build_graph_from_event, args_list, chunksize=chunksize)
        else:
            # Single-process path — useful for debugging / profiling
            graphs = [build_graph_from_event(a) for a in tqdm(args_list, disable=not self.verbose, desc="Building graphs")]

        tau_masses = [g.fitted_tau_mass.item() for g in graphs]

        # ── 3. Batched inference ─────────────────────────────────────────────
        # PyG DataLoader collates a list of Data objects into a single Batch,
        # handling edge_index offsetting automatically.
        loader = PyGDataLoader(
            graphs,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=0,   # data already in memory; extra workers waste RAM
        )

        scores_dict = {name: [] for name in self.model_names}

        for model_idx, (model, name) in enumerate(zip(self.models, self.model_names)):
            model = model.to(self.device)
            model.eval()
            model_scores = []

            with torch.no_grad():
                for batch in tqdm(loader, disable=not self.verbose, desc=f"Inference [{name}]"):
                    batch = batch.to(self.device)
                    self._normalise_batch(batch, model_idx)
                    out = model(batch)          # shape: (batch_size, 1) — logits
                    model_scores.extend(out[:, 0].cpu().tolist())

            scores_dict[name] = model_scores

        # ── 4. Populate score_dict_for_root ──────────────────────────────────
        for g_idx, graph in enumerate(graphs):
            key = (
                int(graph.run.item()),
                int(graph.lumi.item()),
                int(graph.event_id.item()),
            )
            for name in self.model_names:
                if key in score_dict_for_root[name]:
                    logging.warning(f"Overwriting score in {name} for key {key}")
                score_dict_for_root[name][key] = scores_dict[name][g_idx]

        return scores_dict, tau_masses

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 — ROOT score merging (C++-level, no per-entry Python loop)
    # ──────────────────────────────────────────────────────────────────────────

    def _merge_scores_to_root(
        self,
        file_in: str,
        scores_to_save: dict,
        root_path: str,
    ):
        """
        Merge GNN scores into the original ROOT file using a TTree friend index.

        The heavy lifting (matching run/event across trees) is done entirely in
        C++ via ROOT's TTree::BuildIndex + TTree::AddFriend + TTree::CopyTree —
        no per-entry Python loop.

        Parameters
        ----------
        file_in : str
            Path to the original (unscored) ROOT file.
        scores_to_save : dict
            Dict with keys "run", "luminosityBlock", "event", "GNN_score"
            as numpy arrays.
        root_path : str
            Output path for the merged ROOT file.
        """
        tmp_path = root_path.replace(".root", "_scores_only_tmp.root")

        # Write scores-only tree with uproot (fast, pure Python)
        with uproot.recreate(tmp_path, compression=uproot.ZLIB(4)) as f:
            f["Scores"] = {k: v for k, v in scores_to_save.items()}

        print(f"  Merging scores into {root_path} ...")
        t_start = time.time()

        fin         = ROOT.TFile.Open(file_in)
        t1          = fin.Get("Events")
        friend_file = ROOT.TFile.Open(tmp_path)
        t2          = friend_file.Get("Scores")

        # BuildIndex runs in C++ — creates an in-memory B-tree on (run, event)
        t2.BuildIndex("run", "event")
        t1.AddFriend(t2, "ScoreFriend")

        fout  = ROOT.TFile.Open(root_path, "RECREATE")
        # CopyTree copies every branch including the friend's GNN_score
        # and runs entirely in compiled ROOT code.
        t_out = t1.CopyTree("")
        t_out.Write("", ROOT.TObject.kOverwrite)
        fout.Close()

        fin.Close()
        friend_file.Close()

        os.remove(tmp_path)
        print(f"  Merge completed in {time.time() - t_start:.1f}s → {root_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5 — Per-file scoring (replaces get_all_scores)
    # ──────────────────────────────────────────────────────────────────────────

    def _score_file(self, file_name: str):
        """
        Score all events in a single ROOT file.

        Returns
        -------
        all_scores_dict : dict[model_name -> list[float]]
        all_tau_masses  : list[float]
        score_dicts     : dict[model_name -> dict[(run,lumi,event) -> float]]
        """
        prefix, n_prefix, old_prefix = self._prefixes()

        # ── Weights setup ────────────────────────────────────────────────────
        xsec = 1.
        gen_events_before_selection = None
        npvs_weights = None

        if self.weights == "xsec":
            with uproot.open(file_name) as in_file:
                if "Metadata" not in in_file:
                    raise ValueError("No Metadata tree found in file")
                if "GenCrossSection" not in in_file["Metadata"].keys():
                    raise ValueError("No GenCrossSection in Metadata")
                xsec = float(in_file["Metadata"]["GenCrossSection"].array()[0])
                gen_events_before_selection = self.get_total_initial_events()

        elif self.weights == "npvs_rescale":
            npvs_weights = np.load(os.path.join(self.root, "npvs_weights.npz"))["weight_per_bin"]

        # ── Branch list ──────────────────────────────────────────────────────
        branches_to_load = [
            f"{old_prefix}PFCands_pt",
            f"{old_prefix}PFCands_eta",
            f"{old_prefix}PFCands_phi",
            f"{old_prefix}PFCands_pdgId",
            f"{old_prefix}PFCands_mass",
            f"{old_prefix}PFCands_charge",
            f"{old_prefix}PFCands_puppiWeight",
            f"n{n_prefix}TauTo3Mu",
            f"{prefix}TauTo3Mu_mu1_eta",
            f"{prefix}TauTo3Mu_mu1_phi",
            f"{prefix}TauTo3Mu_mu2_eta",
            f"{prefix}TauTo3Mu_mu2_phi",
            f"{prefix}TauTo3Mu_mu3_eta",
            f"{prefix}TauTo3Mu_mu3_phi",
            f"{prefix}TauTo3Mu_fitted_mass",
            f"{prefix}TauTo3Mu_fitted_eta",
            f"{prefix}TauTo3Mu_fitted_phi",
            f"{prefix}TauTo3Mu_fitted_pt",
            f"{prefix}{self.tau_collection_name}_pt",
            f"{prefix}{self.tau_collection_name}_eta",
            f"{prefix}{self.tau_collection_name}_phi",
            f"{prefix}{self.tau_collection_name}_mass",
            "PV_npvs",
            f"{prefix}TauTo3Mu_iso_ptChargedFromPV",
            f"{prefix}TauTo3Mu_iso_ptPhotons",
            f"{prefix}TauTo3Mu_iso_ptChargedFromPU",
            "run",
            "luminosityBlock",
            "event",
        ] + self.graph_level_branches

        # De-duplicate in case tau_collection_name == "TauTo3Mu_fitted"
        branches_to_load = list(dict.fromkeys(branches_to_load))

        if self.tau_channel == "W":
            branches_to_load.append("TauPlusMET_Puppi_phi")

        if self.use_displacement:
            branches_to_load += [
                f"{old_prefix}PFCands_d0",
                f"{old_prefix}PFCands_dz",
            ]

        # ── Iterate over uproot chunks ────────────────────────────────────────
        all_scores_dict = {name: [] for name in self.model_names}
        all_tau_masses  = []
        score_dicts     = {name: {} for name in self.model_names}
        chunk_counter   = 0

        for events in uproot.iterate(
            f"{file_name}:Events",
            expressions=branches_to_load,
            library="ak",
            step_size="2000 MB",
        ):
            chunk_counter += 1
            print(f"  Processing chunk {chunk_counter} of {file_name}  ({len(events)} events)")

            total_events = len(events)

            # ── Background mask ───────────────────────────────────────────────
            bkg_mask = None
            if self.mask:
                bkg_mask = abs(events[f"{prefix}TauTo3Mu_fitted_mass"] - 1.78) > 0.08
                keep     = ak.any(bkg_mask, axis=1)
                events   = events[keep]
                bkg_mask = bkg_mask[keep]

            if gen_events_before_selection is None:
                gen_events_before_selection = total_events

            runs      = np.array(events["run"])
            lumis     = np.array(events["luminosityBlock"])
            event_ids = np.array(events["event"])

            # ── Core pipeline ─────────────────────────────────────────────────
            chunk_scores, chunk_masses = self._process_chunk(
                events,
                bkg_mask,
                runs,
                lumis,
                event_ids,
                score_dicts,
            )

            for name in self.model_names:
                all_scores_dict[name] += chunk_scores[name]
            all_tau_masses += chunk_masses

        return all_scores_dict, all_tau_masses, score_dicts

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6 — Top-level process loop (one file at a time)
    # ──────────────────────────────────────────────────────────────────────────

    def process(self):
        """
        Entry point called by __init__.
        Loops over all ROOT files, scores them, writes output ROOT files,
        and saves compressed .npz score arrays.
        """
        if "signal" in self.root:
            mode = "signal"
        elif "background" in self.root:
            mode = "background"
        else:
            mode = "unknown"

        # Accumulate across all files for the final .npz
        global_scores_dict  = {name: [] for name in self.model_names}
        global_tau_masses   = []

        for file in self.raw_file_names:
            print(f"\n{'='*60}")
            print(f"Processing: {file}")
            print(f"{'='*60}")

            all_scores_dict, all_tau_masses, score_dicts = self._score_file(file)

            global_tau_masses += all_tau_masses

            # ── Save per-model scored ROOT file ──────────────────────────────
            for model_name in self.model_names:
                global_scores_dict[model_name] += all_scores_dict[model_name]

                base_name        = os.path.basename(file)
                name_without_ext = os.path.splitext(base_name)[0]
                model_storage    = os.path.join(self.storage_folder, model_name)
                os.makedirs(model_storage, exist_ok=True)
                root_path        = os.path.join(model_storage, f"{name_without_ext}_with_scores.root")

                # Build numpy arrays from the (run, lumi, event) → score dict
                score_map = score_dicts[model_name]
                if score_map:
                    run_arr   = np.array([k[0] for k in score_map], dtype=np.uint32)
                    lumi_arr  = np.array([k[1] for k in score_map], dtype=np.uint32)
                    event_arr = np.array([k[2] for k in score_map], dtype=np.uint64)
                    score_arr = np.array(list(score_map.values()),   dtype=np.float32)

                    scores_to_save = {
                        "run":              run_arr,
                        "luminosityBlock":  lumi_arr,
                        "event":            event_arr,
                        "GNN_score":        score_arr,
                    }
                    self._merge_scores_to_root(file, scores_to_save, root_path)
                else:
                    logging.warning(f"No scores to save for model {model_name} in {file}")

        # ── Save global .npz arrays ───────────────────────────────────────────
        for name, output_folder in zip(self.model_names, self.output_folders):
            os.makedirs(output_folder, exist_ok=True)
            out_path = os.path.join(output_folder, f"raw_scores_and_tau_mass_{mode}.npz")
            np.savez_compressed(
                out_path,
                scores=global_scores_dict[name],
                tau_mass=global_tau_masses,
            )
            print(f"\nRaw scores saved to: {out_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy public API — kept so existing call-sites don't break
    # ──────────────────────────────────────────────────────────────────────────

    def deltaphi(self, phi1, phi2):
        """Kept for backward compatibility."""
        d = np.abs(phi1 - phi2)
        return np.minimum(d, 2.0 * np.pi - d)

    def calc_delta_R_sq(self, eta, phi, ref_eta, ref_phi):
        """Kept for backward compatibility."""
        delta_eta = eta - ref_eta
        delta_phi = _min_dphi_arr(np.asarray(phi), ref_phi)
        return delta_eta ** 2 + delta_phi ** 2

    def connect_fully(self, delta_R):
        n = delta_R.shape[0]
        edge_np = _connect_fully(n)
        return torch.tensor(edge_np.T, dtype=torch.long)   # (E, 2) to match old API

    def connect_knn(self, delta_R):
        edge_np = _connect_knn(delta_R, self.k_knn)
        return torch.tensor(edge_np.T, dtype=torch.long)

    def connect_r_threshold(self, delta_R):
        edge_np = _connect_r_threshold(delta_R, self.delta_r_threshold)
        return torch.tensor(edge_np.T, dtype=torch.long)

    def build_cone(
        self, pfcands_eta, pfcands_phi,
        fitted_tau_eta, fitted_tau_phi,
        mu1_eta, mu1_phi,
        mu2_eta, mu2_phi,
        mu3_eta, mu3_phi,
    ):
        """Kept for backward compatibility."""
        pfc_eta = np.asarray(pfcands_eta, dtype=np.float32)
        pfc_phi = np.asarray(pfcands_phi, dtype=np.float32)
        dR_sq = (pfc_eta - fitted_tau_eta) ** 2 + _min_dphi_arr(pfc_phi, fitted_tau_phi) ** 2

        dR_sq_mus = [
            (me - fitted_tau_eta) ** 2 + _min_dphi_scalar(mp, fitted_tau_phi) ** 2
            for me, mp in [(mu1_eta, mu1_phi), (mu2_eta, mu2_phi), (mu3_eta, mu3_phi)]
        ]
        cone_sq = max(self.cone_size ** 2, *dR_sq_mus)
        return dR_sq < cone_sq