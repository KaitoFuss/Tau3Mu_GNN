import re

import pytest

from tau3mu_gnn.bdt_training.bdt_inputs import bdt_branches
from tau3mu_gnn.pipeline import branches as b


def raw_names(expressions):
    """Raw input branch names read by uproot expressions, before the stage 1 skim.

    The skim writes Selected_TauTo3Mu* as a masked copy of Slimmed_TauTo3Mu*.
    """
    names = set()
    for expr in expressions:
        for name in re.findall(r"[A-Za-z_]\w*", expr):
            names.add(re.sub(r"^Selected_TauTo3Mu", "Slimmed_TauTo3Mu", name))
    return names


def test_raw_names_splits_expressions():
    assert raw_names(["Selected_TauTo3Mu_vtx_chi2*(Selected_TauTo3Mu_vtx_Ndof**-1)", "run"]) == {
        "Slimmed_TauTo3Mu_vtx_chi2", "Slimmed_TauTo3Mu_vtx_Ndof", "run"}


def test_w_bdt_branches_are_required_inputs():
    features, branches = bdt_branches("W")
    assert set(features) <= set(branches)
    missing = raw_names(branches) - set(b.required_branches("W"))
    assert missing == set()


def test_unknown_channel():
    with pytest.raises(ValueError, match="no BDT branch list for channel 'X'"):
        bdt_branches("X")
