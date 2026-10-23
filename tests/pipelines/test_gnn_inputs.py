import re

import pytest

from tau3mu_gnn.gnn_training.gnn_inputs import (STAGE_MADE, gnn_branches, graph_level_branches,
                                                 input_branch_names)
from tau3mu_gnn.gnn_training.TrainingUtils import get_model_parameters
from tau3mu_gnn.pipelines.common.branches import required_branches

P = "Selected_TauTo3Mu_"
PF = "Slimmed_PFCands_"


def test_w_load_list_is_old_list_without_unused_branches():
    # Old list of ScoreProducer.get_all_scores (W, slimmed, BDT cut, displacement), minus
    # PV_npvs (npvs_rescale only) and the iso_* branches (save_benchmark_score only).
    expected = [
        *(f"{PF}{s}" for s in ("pt", "eta", "phi", "pdgId", "mass", "charge", "puppiWeight")),
        "nSelectedTauTo3Mu",
        *(f"{P}mu{i}_{s}" for i in (1, 2, 3) for s in ("eta", "phi")),
        *(f"{P}fitted_{s}" for s in ("mass", "eta", "phi", "pt")),
        "run", "luminosityBlock", "event",
        "bdt_ISO",
        *graph_level_branches("W"),
        "TauPlusMET_Puppi_phi",
        f"{PF}d0", f"{PF}dz",
    ]
    assert gnn_branches("W") == expected


def test_optional_branches():
    assert "bdt_ISO" not in gnn_branches("W", bdt_preselection=False)
    assert "PV_npvs" in gnn_branches("W", weights="npvs_rescale")
    assert f"{P}iso_ptPhotons" in gnn_branches("W", save_benchmark_score=True)
    assert f"{PF}d0" not in gnn_branches("W", use_displacement=False)
    assert "PFCands_pt" in gnn_branches("W", slimmed=False)


@pytest.mark.parametrize("channel", ["W", "HF"])
def test_graph_level_count_matches_model(channel):
    # W adds one computed feature (deltaPhi) in ScoreProducer.process_event
    n_computed = 1 if channel == "W" else 0
    n_graph = get_model_parameters("skimmed_dzd0_cone_size_0.5_knn_10", channel)[2]
    assert len(graph_level_branches(channel)) + n_computed == n_graph


def test_w_gnn_branches_are_required_inputs():
    raw = {re.sub(r"^Selected_TauTo3Mu", "Slimmed_TauTo3Mu", n) for n in input_branch_names("W") - STAGE_MADE}
    missing = raw - set(required_branches("W"))
    assert not missing, f"GNN reads branches not in required_branches('W'): {sorted(missing)}"


def test_input_branch_names_splits_expressions():
    names = input_branch_names("W")
    assert {f"{P}vtx_chi2", f"{P}vtx_Ndof", "bdt_ISO", "nSelectedTauTo3Mu"} <= names
    assert not any("*" in n or "[" in n for n in names)


def test_unknown_channel():
    with pytest.raises(ValueError, match="no GNN branch list"):
        gnn_branches("X")
