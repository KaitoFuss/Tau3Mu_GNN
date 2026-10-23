import re

import pytest

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
