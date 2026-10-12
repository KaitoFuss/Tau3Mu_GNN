"""Branches that ScoreProducer loads, per channel."""
from __future__ import annotations

from tau3mu_gnn.bdt_training.bdt_inputs import expression_names

# Made by the pipeline (stage 1 skim, stage 2 BDT), not raw input branches.
STAGE_MADE = frozenset({"nSelectedTauTo3Mu", "bdt_ISO"})


def graph_level_branches(tau_channel: str, prefix: str = "Selected_") -> list[str]:
    """uproot expressions of the graph-level features, in model input order. New list on every call."""
    if tau_channel == "W":
        return [
            f'{prefix}TauTo3Mu_fitted_pt',
            'TauPlusMET_Tau_Puppi_mT',
            #'cand_refit_tau_dBetaIsoCone0p8strength0p2_rel',
            #'deltaPhi(TauPlusMET_Puppi_phi[0], Slimmed_TauTo3Mu_fitted_phi[0])',
            'TauPlusMET_PuppiMET_pt',
            #'Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)', # only for >= v4
        #     'cand_refit_tau_pt/cand_refit_met_pt', # only for >= v2
        #     'cand_refit_dRtauMuonMax',
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
    if tau_channel == "HF":
        return [
            f'Muon_pt[{prefix}TauTo3Mu_mu3_idx]',

            # 'Muon_mediumId[Slimmed_TauTo3Mu_mu1_idx]',
            # 'Muon_mediumId[Slimmed_TauTo3Mu_mu2_idx]',
            # 'Muon_mediumId[Slimmed_TauTo3Mu_mu3_idx]',

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
    raise ValueError(f"no GNN branch list for channel {tau_channel!r} (known: HF, W)")


def gnn_branches(tau_channel: str, slimmed: bool = True, bdt_preselection: bool = True,
                 use_displacement: bool = True, weights: str | None = None,
                 save_benchmark_score: bool = False) -> list[str]:
    """Every expression ScoreProducer loads from the input file."""
    prefix = "Selected_" if slimmed else ""
    n_prefix = "Selected" if slimmed else ""
    old_prefix = "Slimmed_" if slimmed else ""
    graph_level = graph_level_branches(tau_channel, prefix)  # raises for an unknown channel
    branches = [
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
    ]
    # Only needed for the npvs training weights and the benchmark score.
    if weights == "npvs_rescale":
        branches.append("PV_npvs")
    if save_benchmark_score:
        branches += [
            f"{prefix}TauTo3Mu_iso_ptChargedFromPV",
            f"{prefix}TauTo3Mu_iso_ptPhotons",
            f"{prefix}TauTo3Mu_iso_ptChargedFromPU",
        ]
    branches += ["run", "luminosityBlock", "event"]
    if bdt_preselection:
        branches.append("bdt_ISO")
    branches += graph_level
    if tau_channel == "W":
        branches.append("TauPlusMET_Puppi_phi")
    if use_displacement:
        branches += [f"{old_prefix}PFCands_d0", f"{old_prefix}PFCands_dz"]
    return branches


def input_branch_names(tau_channel: str) -> set[str]:
    """Branch names that stage 3 reads (pipeline settings: slimmed, BDT cut, displacement)."""
    return expression_names(gnn_branches(tau_channel))
