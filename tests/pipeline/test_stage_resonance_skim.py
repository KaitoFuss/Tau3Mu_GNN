import logging

import pytest
import uproot

from conftest import make_root_file
from tau3mu_gnn.pipeline import branches as b
from tau3mu_gnn.pipeline.stages import resonance_skim

REQ_W = b.required_branches("W")


def _inputs(user, tmp_path, names=REQ_W):
    sig = make_root_file(tmp_path / "in" / "sig.root", names)
    make_root_file(tmp_path / "in" / "bkg_1.root", names)
    make_root_file(tmp_path / "in" / "bkg_2.root", names)
    user["inputs"]["signal"] = [str(sig)]
    user["inputs"]["background"] = [str(tmp_path / "in" / "bkg_*.root")]


def test_empty_map_passes_inputs_through(load, user, tmp_path, caplog):
    _inputs(user, tmp_path)
    out_dir = tmp_path / "0_resonance_skim"
    with caplog.at_level(logging.INFO):
        files = resonance_skim.run(load(user), out_dir)
    assert files == {"signal": [str(tmp_path / "in" / "sig.root")],
                     "background": [str(tmp_path / "in" / f"bkg_{i}.root") for i in (1, 2)]}
    assert "no copy" in caplog.text
    assert not out_dir.exists()


def test_branch_map_writes_renamed_copy(load, user, tmp_path):
    _inputs(user, tmp_path, [n for n in REQ_W if n != "Slimmed_PFCands_pt"] + ["PFCand_pt"])
    user["branch_map"] = {"Slimmed_PFCands_pt": "PFCand_pt"}
    out_dir = tmp_path / "0_resonance_skim"
    files = resonance_skim.run(load(user), out_dir)
    assert files == {"signal": [str(out_dir / "signal.root")], "background": [str(out_dir / "background.root")]}
    sig = uproot.open(files["signal"][0])["Events"]
    assert "PFCand_pt" not in sig.keys()
    assert sig["Slimmed_PFCands_pt"].array().tolist() == [1.0, 2.0]
    assert uproot.open(files["background"][0])["Events"].num_entries == 4  # two input files merged


def test_inputs_checked_before_any_work(load, user, tmp_path):
    _inputs(user, tmp_path, [n for n in REQ_W if n != "run"])
    user["branch_map"] = {"Slimmed_PFCands_pt": "Slimmed_PFCands_pt"}
    with pytest.raises(b.BranchError, match="run"):
        resonance_skim.run(load(user), tmp_path / "0_resonance_skim")
    assert not (tmp_path / "0_resonance_skim").exists()


@pytest.mark.parametrize("branch_map", [{}, {"Slimmed_PFCands_pt": "PFCand_pt"}])
def test_not_implemented_when_inputs_are_raw(load, user, tmp_path, branch_map):
    # Order: check -> rename -> resonance skim. The skim is not implemented: error, no output.
    names = [n for n in REQ_W if n not in branch_map] + list(branch_map.values())
    _inputs(user, tmp_path, names)
    user["branch_map"] = branch_map
    user["inputs"]["already_resonance_skimmed"] = False
    out_dir = tmp_path / "0_resonance_skim"
    with pytest.raises(resonance_skim.StageNotImplementedError) as e:
        resonance_skim.run(load(user), out_dir)
    assert "already_resonance_skimmed: true" in str(e.value)
    assert not out_dir.exists()


def test_raw_inputs_are_checked_first(load, user, tmp_path):
    _inputs(user, tmp_path, [n for n in REQ_W if n != "run"])
    user["inputs"]["already_resonance_skimmed"] = False
    with pytest.raises(b.BranchError, match="run"):
        resonance_skim.run(load(user), tmp_path / "0_resonance_skim")
