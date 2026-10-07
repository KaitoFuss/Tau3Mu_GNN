import os

import pytest
import uproot

from tau3mu_gnn.pipeline import branches as b

W_FILES = [
    "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/old_data/signal_skimmed_no_resonances.root",
    "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/old_data/data_skimmed_no_resonances.root",
]


def test_required_branches_w_is_common_plus_w():
    req = b.required_branches("W")
    assert len(req) == 38
    assert len(set(req)) == 38
    assert req[: len(b.COMMON)] == b.COMMON
    assert "Slimmed_PFCands_pt" in req
    assert "TauPlusMET_Puppi_phi" in req
    assert "PFCands_pt" not in req


def test_required_branches_unknown_channel():
    with pytest.raises(b.BranchError, match="no required-branch list for channel 'HF'"):
        b.required_branches("HF")


def test_branch_error_lists_all_problems():
    err = b.BranchError(["one", "two"])
    assert err.problems == ["one", "two"]
    assert "one" in str(err) and "two" in str(err)


@pytest.mark.parametrize("path", W_FILES)
def test_real_w_files_have_all_required_branches(path):
    if not os.path.exists(path):
        pytest.skip(f"not reachable on this machine: {path}")
    keys = set(uproot.open(path)["Events"].keys())
    assert [x for x in b.required_branches("W") if x not in keys] == []
