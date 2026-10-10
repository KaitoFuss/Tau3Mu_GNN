"""Branches that BDT_ScoreProducer loads, per channel.

Kept apart from the script so tests can import them: BDT_ScoreProducer runs on import.
"""
from __future__ import annotations


def bdt_branches(tau_channel: str, prefix: str = "Selected_") -> tuple[list[str], list[str]]:
    """Return (features, branches): uproot expressions of the BDT input features, and every
    expression loaded from the input file (features first). New lists on every call."""
    if tau_channel == "HF":
        features = [
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
        ]

    elif tau_channel == "W":
        features = [
            f'{prefix}TauTo3Mu_fitted_pt',
            'TauPlusMET_Tau_Puppi_mT',
            #'cand_refit_tau_dBetaIsoCone0p8strength0p2_rel',
            #'deltaPhi(TauPlusMET_Puppi_phi[0], Slimmed_TauTo3Mu_fitted_phi[0])',
            'TauPlusMET_PuppiMET_pt',
            #'Slimmed_TauTo3Mu_fitted_pt*(TauPlusMET_PuppiMET_pt**-1)', # only for >= v4
            # 'cand_refit_tau_pt/cand_refit_met_pt', # only for >= v2
            # 'cand_refit_dRtauMuonMax',
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

    else:
        raise ValueError(f"no BDT branch list for channel {tau_channel!r} (known: HF, W)")

    branches = features + [

        f'{prefix}TauTo3Mu_vtx_chi2',
        f'{prefix}TauTo3Mu_vtx_Ndof',

        f'{prefix}TauTo3Mu_fitted_eta',
        f'{prefix}TauTo3Mu_fitted_mass',
        'TauPlusMET_Puppi_phi',
        f'{prefix}TauTo3Mu_fitted_phi',
        f'{prefix}TauTo3Mu_iso_ptChargedFromPV',
        f'{prefix}TauTo3Mu_iso_ptPhotons',
        f'{prefix}TauTo3Mu_iso_ptChargedFromPU',
        # 'mu1_refit_muonid_soft', 'mu1_refit_muonid_loose', 'mu1_refit_muonid_medium', 'mu1_refit_muonid_tight',
        # 'mu2_refit_muonid_soft', 'mu2_refit_muonid_loose', 'mu2_refit_muonid_medium', 'mu2_refit_muonid_tight',
        # 'mu3_refit_muonid_soft', 'mu3_refit_muonid_loose', 'mu3_refit_muonid_medium', 'mu3_refit_muonid_tight',
        "run",
        "luminosityBlock",
        "event",
    ]

    # Only HF uses these (ISO feature: max Muon_tkRelIso of the three muons). W does not load them:
    # BDT_ScoreProducer drops rows with NaN in any loaded column.
    if tau_channel == "HF":
        branches += [
            f'{prefix}TauTo3Mu_charge',
            f'Muon_tkRelIso[{prefix}TauTo3Mu_mu1_idx]',
            f'Muon_tkRelIso[{prefix}TauTo3Mu_mu2_idx]',
            f'Muon_tkRelIso[{prefix}TauTo3Mu_mu3_idx]',
        ]

    return features, branches
