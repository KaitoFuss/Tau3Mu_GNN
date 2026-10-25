import os

import awkward as ak
import numpy as np
import pytest
import uproot

from conftest import make_root_file
from tau3mu_gnn.pipelines.common import branches as b

W_FILES = [
    "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/signal_W/old_data/signal_skimmed_no_resonances.root",
    "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/background_W/old_data/data_skimmed_no_resonances.root",
]
_HF = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data"
HF_FILES = [
    f"{_HF}/signal_HF/old_data/ds_large_signal_skimmed_no_resonances.root",
    *(f"{_HF}/background_HF/old_data/data_skimmed_no_resonances_part{i}.root" for i in (1, 2, 3, 4)),
]


def test_required_branches_w_is_common_plus_w():
    req = b.required_branches("W")
    assert len(req) == 38
    assert len(set(req)) == 38
    assert req[: len(b.COMMON)] == b.COMMON
    assert "Slimmed_PFCands_pt" in req
    assert "TauPlusMET_Puppi_phi" in req
    assert "PFCands_pt" not in req


def test_required_branches_hf_is_common_plus_hf():
    req = b.required_branches("HF")
    assert len(req) == 39 and len(set(req)) == 39
    assert req[: len(b.COMMON)] == b.COMMON
    assert {"Muon_pt", "Muon_isGlobal", "Muon_tkRelIso", "Slimmed_TauTo3Mu_mu3_idx"} <= set(req)
    assert not any(n.startswith("TauPlusMET_") for n in req)


def test_training_required_branches_hf():
    t = b.training_required_branches("HF")
    assert "PV_npvs" in t and "PV_npvs" not in b.required_branches("HF")


def test_raw_required_branches_hf():
    raw = b.raw_required_branches("HF")
    assert {"Muon_pt", "Muon_isGlobal", "Muon_tkRelIso", "TauTo3Mu_mu1_idx", "TauTo3Mu_charge"} <= set(raw)
    assert not any(n.startswith("Slimmed_") for n in raw)


@pytest.mark.parametrize("path", HF_FILES)
def test_real_hf_files_have_all_required_branches(path):
    if not os.path.exists(path):
        pytest.skip(f"not reachable on this machine: {path}")
    keys = set(uproot.open(path)["Events"].keys())
    assert [x for x in b.training_required_branches("HF") if x not in keys] == []


def test_required_branches_unknown_channel():
    with pytest.raises(b.BranchError, match="no required-branch list for channel 'XX'"):
        b.required_branches("XX")


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


REQ_W = b.required_branches("W")


def problems_of(excinfo):
    return "\n".join(excinfo.value.problems)


def test_check_passes_with_identity_map(tmp_path):
    f = make_root_file(tmp_path / "sig.root", REQ_W)
    assert b.check_input_branches([str(f)], "Events", {}, "W") == [str(f)]


def test_check_passes_with_renamed_branches(tmp_path):
    names = [n for n in REQ_W if n != "Slimmed_PFCands_pt"] + ["PFCand_pt"]
    f = make_root_file(tmp_path / "sig.root", names)
    assert b.check_input_branches([str(f)], "Events", {"Slimmed_PFCands_pt": "PFCand_pt"}, "W") == [str(f)]


def test_check_expands_globs_sorted(tmp_path):
    make_root_file(tmp_path / "bkg_2.root", REQ_W)
    make_root_file(tmp_path / "bkg_1.root", REQ_W)
    files = b.check_input_branches([str(tmp_path / "bkg_*.root")], "Events", {}, "W")
    assert files == [str(tmp_path / "bkg_1.root"), str(tmp_path / "bkg_2.root")]


def test_check_lists_all_missing_branches_of_all_files(tmp_path):
    f1 = make_root_file(tmp_path / "a.root", [n for n in REQ_W if n not in ("run", "event")])
    f2 = make_root_file(tmp_path / "b.root", [n for n in REQ_W if n != "TauPlusMET_Puppi_pt"])
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f1), str(f2)], "Events", {}, "W")
    text = problems_of(e)
    assert f"{f1}: 2 required branch(es) missing" in text
    assert "run" in text and "event" in text
    assert f"{f2}: 1 required branch(es) missing" in text
    assert "TauPlusMET_Puppi_pt" in text


def test_missing_mapped_branch_names_both_names(tmp_path):
    f = make_root_file(tmp_path / "a.root", [n for n in REQ_W if n != "Slimmed_PFCands_pt"])
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f)], "Events", {"Slimmed_PFCands_pt": "PFCand_pt"}, "W")
    assert "PFCand_pt (analysis name Slimmed_PFCands_pt)" in problems_of(e)


def test_glob_matching_nothing_is_error(tmp_path):
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(tmp_path / "nothing_*.root")], "Events", {}, "W")
    assert "matches no file" in problems_of(e)


def test_missing_file_and_wrong_tree_reported_per_file(tmp_path):
    good = make_root_file(tmp_path / "good.root", REQ_W)
    wrong_tree = make_root_file(tmp_path / "tree.root", REQ_W, tree="Other")
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(tmp_path / "gone.root"), str(wrong_tree), str(good)], "Events", {}, "W")
    text = problems_of(e)
    assert "gone.root: file not found" in text
    assert "tree.root: no tree 'Events'" in text
    assert "good.root" not in text


def test_unknown_map_key_with_hint(tmp_path):
    f = make_root_file(tmp_path / "a.root", REQ_W)
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f)], "Events", {"Slimmed_PFCands_ptt": "PFCand_pt"}, "W")
    text = problems_of(e)
    assert "unknown analysis branch 'Slimmed_PFCands_ptt'" in text
    assert "did you mean 'Slimmed_PFCands_pt'" in text


def test_two_keys_same_target(tmp_path):
    f = make_root_file(tmp_path / "a.root", REQ_W)
    bad = {"Slimmed_PFCands_pt": "pt_x", "Slimmed_PFCands_eta": "pt_x"}
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f)], "Events", bad, "W")
    assert "both map to 'pt_x'" in problems_of(e)


def test_target_is_other_required_name(tmp_path):
    f = make_root_file(tmp_path / "a.root", REQ_W)
    bad = {"Slimmed_TauTo3Mu_fitted_pt": "Slimmed_TauTo3Mu_fitted_eta"}
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f)], "Events", bad, "W")
    assert "is itself a required analysis branch" in problems_of(e)


def test_identity_entry_is_allowed(tmp_path):
    f = make_root_file(tmp_path / "a.root", REQ_W)
    assert b.check_input_branches([str(f)], "Events", {"run": "run"}, "W") == [str(f)]


def _tiny_jagged_file(path):
    with uproot.recreate(path) as f:
        f["Events"] = {"UserPt": ak.Array([[1.0, 2.0], [3.0]]), "run": np.array([1, 2], dtype=np.uint32)}
    return path


def test_rename_columns_snapshot_has_analysis_name_only(tmp_path):
    import ROOT

    src = _tiny_jagged_file(tmp_path / "in.root")
    df = ROOT.RDataFrame("Events", str(src))
    df, columns = b.rename_columns(df, {"Slimmed_TauTo3Mu_fitted_pt": "UserPt"})
    assert "UserPt" not in columns
    assert "Slimmed_TauTo3Mu_fitted_pt" in columns and "run" in columns
    out = tmp_path / "out.root"
    df.Snapshot("Events", str(out), columns)
    tree = uproot.open(out)["Events"]
    assert "UserPt" not in tree.keys()
    assert tree["Slimmed_TauTo3Mu_fitted_pt"].array().tolist() == [[1.0, 2.0], [3.0]]
    assert tree["run"].array().tolist() == [1, 2]


def test_mapped_name_already_in_file_is_error(tmp_path):
    # File has the analysis name AND the user's mapped branch: probably not intended.
    f1 = make_root_file(tmp_path / "a.root", [*REQ_W, "MyPt"])
    f2 = make_root_file(tmp_path / "b.root", [n for n in REQ_W if n != "Slimmed_TauTo3Mu_fitted_pt"] + ["MyPt"])
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f1), str(f2)], "Events", {"Slimmed_TauTo3Mu_fitted_pt": "MyPt"}, "W")
    text = problems_of(e)
    assert f"{f1}: branch_map renames onto name(s) the file already has" in text
    assert "Slimmed_TauTo3Mu_fitted_pt (mapped from MyPt)" in text
    assert str(f2) not in text


def test_check_dedupes_overlapping_globs(tmp_path):
    make_root_file(tmp_path / "x1.root", REQ_W)
    files = b.check_input_branches([str(tmp_path / "x*.root"), str(tmp_path / "x1.root")], "Events", {}, "W")
    assert files == [str(tmp_path / "x1.root")]


def test_rename_columns_identity_and_empty_map_change_nothing(tmp_path):
    import ROOT

    src = _tiny_jagged_file(tmp_path / "in.root")
    df = ROOT.RDataFrame("Events", str(src))
    before = sorted(str(c) for c in df.GetColumnNames())
    for mapping in ({}, {"run": "run"}):
        _, columns = b.rename_columns(df, mapping)
        assert sorted(columns) == before


def test_training_required_branches_add_pv_npvs():
    t = b.training_required_branches("W")
    assert "PV_npvs" in t and "PV_npvs" not in b.required_branches("W")
    assert set(b.required_branches("W")) <= set(t) and len(t) == len(set(t))


def test_extra_branches_are_required_only_when_asked(tmp_path):
    f = make_root_file(tmp_path / "a.root", REQ_W)
    assert b.check_input_branches([str(f)], "Events", {}, "W") == [str(f)]
    with pytest.raises(b.BranchError) as e:
        b.check_input_branches([str(f)], "Events", {}, "W", extra=("PV_npvs",))
    text = problems_of(e)
    assert f"{f}: 1 required branch(es) missing" in text and "PV_npvs" in text


def test_extra_branches_pass_when_present_raw_and_resonance_skimmed(tmp_path):
    f = make_root_file(tmp_path / "a.root", [*REQ_W, "PV_npvs"])
    assert b.check_input_branches([str(f)], "Events", {}, "W", extra=("PV_npvs",)) == [str(f)]
    r = make_root_file(tmp_path / "r.root", list(b.raw_required_branches("W")) + ["PV_npvs"])
    assert b.check_input_branches([str(r)], "Events", {}, "W", raw=True, extra=("PV_npvs",)) == [str(r)]
