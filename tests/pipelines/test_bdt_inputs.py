import pickle
import re
from pathlib import Path

import pandas as pd
import pytest

from tau3mu_gnn.bdt_training.BDT_ScoreProducer import _add_features
from tau3mu_gnn.bdt_training.bdt_inputs import bdt_branches, input_branch_names

from tau3mu_gnn.pipelines.common import branches as b


def test_input_branch_names_splits_expressions():
    names = input_branch_names("W")
    assert {"Selected_TauTo3Mu_vtx_chi2", "Selected_TauTo3Mu_vtx_Ndof", "run"} <= names
    assert not any("*" in n or "(" in n for n in names)
    assert {"Muon_pt", "Selected_TauTo3Mu_mu3_idx"} <= input_branch_names("HF")


def test_w_bdt_branches_are_required_inputs():
    features, branches = bdt_branches("W")
    assert set(features) <= set(branches)
    # The stage 1 skim writes Selected_TauTo3Mu* as a masked copy of Slimmed_TauTo3Mu*.
    raw = {re.sub(r"^Selected_TauTo3Mu", "Slimmed_TauTo3Mu", n) for n in input_branch_names("W")}
    assert raw - set(b.required_branches("W")) == set()


def test_unknown_channel():
    with pytest.raises(ValueError, match="no BDT branch list for channel 'X'"):
        bdt_branches("X")


OLD_HF_CLASSIFIER = Path("/work/kfuss/SVJ_GNN/output/benchmark_HF_BDT_vs1/ISO_classifiers_24feb2026_v0.pck")
OLD_HF_FEATURES = [  # feature_names of OLD_HF_CLASSIFIER, in order
    "Selected_TauTo3Mu_dZmu12", "Selected_TauTo3Mu_dZmu13", "Selected_TauTo3Mu_dZmu23",
    "Selected_TauTo3Mu_sigLxy_3muVtxBS", "Selected_TauTo3Mu_CosAlpha2D_LxyP3mu",
    "Muon_pt_Selected_TauTo3Mu_mu3_idx", "Muon_isGlobal_Selected_TauTo3Mu_mu1_idx",
    "Muon_isGlobal_Selected_TauTo3Mu_mu2_idx", "Muon_isGlobal_Selected_TauTo3Mu_mu3_idx",
    "Selected_TauTo3Mu_vtx_chi2*(Selected_TauTo3Mu_vtx_Ndof**-1)",
    "max(Muon_tkRelIso_Selected_TauTo3Mu_mu1_idx Muon_tkRelIso_Selected_TauTo3Mu_mu2_idx "
    "Muon_tkRelIso_Selected_TauTo3Mu_mu3_idx)",
]
W_BRANCHES = [  # bdt_branches("W")[1] on main, 2026-10-25
    "Selected_TauTo3Mu_fitted_pt", "TauPlusMET_Tau_Puppi_mT", "TauPlusMET_PuppiMET_pt", "TauPlusMET_Puppi_pt",
    "TauPlusMET_PuppiMETminPz", "TauPlusMET_PuppiMETmaxPz", "Selected_TauTo3Mu_dZmu12",
    "Selected_TauTo3Mu_dZmu13", "Selected_TauTo3Mu_dZmu23", "Selected_TauTo3Mu_sigLxy_3muVtxBS",
    "Selected_TauTo3Mu_vtx_chi2*(Selected_TauTo3Mu_vtx_Ndof**-1)", "Selected_TauTo3Mu_CosAlpha2D_LxyP3mu",
    "Selected_TauTo3Mu_vtx_chi2", "Selected_TauTo3Mu_vtx_Ndof", "Selected_TauTo3Mu_fitted_eta",
    "Selected_TauTo3Mu_fitted_mass", "TauPlusMET_Puppi_phi", "Selected_TauTo3Mu_fitted_phi",
    "Selected_TauTo3Mu_iso_ptChargedFromPV", "Selected_TauTo3Mu_iso_ptPhotons",
    "Selected_TauTo3Mu_iso_ptChargedFromPU", "run", "luminosityBlock", "event",
]


def test_hf_bdt_features_are_the_old_classifier_features():
    features, branches = bdt_branches("HF")
    sig, bkg = (pd.DataFrame({name: [1.0] for name in branches}) for _ in range(2))
    assert _add_features(sig, bkg, features, "HF", "Selected_") == OLD_HF_FEATURES


def test_hf_bdt_does_not_load_tauplusmet():
    assert not any(name.startswith("TauPlusMET_") for name in bdt_branches("HF")[1])


def test_w_bdt_branches_unchanged():
    assert bdt_branches("W")[1] == W_BRANCHES


def test_old_hf_classifier_feature_names():
    if not OLD_HF_CLASSIFIER.is_file():
        pytest.skip(f"not on this machine: {OLD_HF_CLASSIFIER}")
    with open(OLD_HF_CLASSIFIER, "rb") as f:
        classifiers = pickle.load(f)
    for clf in classifiers.values():
        assert clf.get_booster().feature_names == OLD_HF_FEATURES
