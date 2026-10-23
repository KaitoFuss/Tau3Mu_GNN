import glob
import logging
import os

import awkward as ak
import numpy as np
import pytest
import uproot

from conftest import TOY_DEFAULTS, make_root_file, write_tree, write_yaml
from tau3mu_gnn.pipeline import branches as b
from tau3mu_gnn.pipeline.stages import resonance_skim

REQ_W = b.required_branches("W")
RAW_W = b.raw_required_branches("W")


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


def test_raw_inputs_are_checked_first(load, user, tmp_path):
    _inputs(user, tmp_path, [n for n in RAW_W if n != "TauTo3Mu_vtx_chi2"])
    user["inputs"]["already_resonance_skimmed"] = False
    with pytest.raises(b.BranchError, match="TauTo3Mu_vtx_chi2"):
        resonance_skim.run(load(user), tmp_path / "0_resonance_skim")
    assert not (tmp_path / "0_resonance_skim").exists()


def test_raw_inputs_need_raw_names(load, user, tmp_path):
    _inputs(user, tmp_path)  # resonance-skimmed names (Slimmed_*), but declared raw
    user["inputs"]["already_resonance_skimmed"] = False
    with pytest.raises(b.BranchError, match="TauTo3Mu_mu1_pt"):
        resonance_skim.run(load(user), tmp_path / "0_resonance_skim")


def test_raw_required_branches():
    assert "TauTo3Mu_fitted_pt" in RAW_W and "PFCands_eta" in RAW_W and "TauPlusMET_Puppi_phi" in RAW_W
    assert "TauTo3Mu_mu2_charge" in RAW_W and "run" in RAW_W
    assert not any(n.startswith("Slimmed_") for n in RAW_W)
    assert "TauTo3Mu_mu1_mass" not in RAW_W  # raw NanoAOD has none; stage 0 defines it
    assert len(RAW_W) == len(set(RAW_W))


# --- resonance skim on toy raw events ---------------------------------------------

MU = np.float32(resonance_skim.MUON_MASS)
CUTS = TOY_DEFAULTS["resonance_skim"]


def _p4(pt, eta, phi, m):
    pt, eta, phi, m = (np.asarray(x, dtype=np.float64) for x in (pt, eta, phi, m))
    return np.stack([np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2), pt * np.cos(phi), pt * np.sin(phi),
                     pt * np.sinh(eta)])


def _mass(*p4s):
    e, x, y, z = sum(p4s)
    return np.sqrt(np.maximum(e ** 2 - x ** 2 - y ** 2 - z ** 2, 0))


def _toy_raw(n_events=400, seed=1):
    """Raw-name events with 0-3 tau candidates, muons close together, masses ~0.3-4 GeV."""
    rng = np.random.default_rng(seed)
    n_cand = rng.integers(0, 4, n_events)
    n = int(n_cand.sum())
    f32 = lambda x: np.asarray(x, dtype=np.float32)
    eta0, phi0 = rng.uniform(-1.5, 1.5, n), rng.uniform(-np.pi, np.pi, n)
    cand = {"vtx_chi2": f32(rng.uniform(0, 40, n)), "fitted_eta": f32(eta0), "fitted_phi": f32(phi0)}
    signs = np.array([(1, 1, -1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)])
    charges = signs[rng.integers(0, len(signs), n)]
    for i in (1, 2, 3):
        cand[f"mu{i}_pt"] = f32(rng.uniform(1, 8, n))
        cand[f"mu{i}_eta"] = f32(eta0 + rng.uniform(-0.4, 0.4, n))
        cand[f"mu{i}_phi"] = f32(phi0 + rng.uniform(-0.4, 0.4, n))
        cand[f"mu{i}_charge"] = charges[:, i - 1].astype(np.int32)
    n_pf = rng.integers(0, 5, n_events)
    first = np.concatenate([[0], np.cumsum(n_cand)])
    pf_eta, pf_phi = [], []
    for ev in range(n_events):  # each PF candidate near a random tau candidate of its event
        for _ in range(n_pf[ev]):
            c = first[ev] + rng.integers(n_cand[ev]) if n_cand[ev] else None
            pf_eta.append((eta0[c] if c is not None else 0.0) + rng.uniform(-1.2, 1.2))
            pf_phi.append((phi0[c] if c is not None else 0.0) + rng.uniform(-1.2, 1.2))
    pf_eta, pf_phi = f32(pf_eta), f32(pf_phi)
    data = {}
    for name in RAW_W:
        if name.startswith("TauTo3Mu_"):
            short = name.removeprefix("TauTo3Mu_")
            flat = cand[short] if short in cand else f32(rng.uniform(0, 1, n))
            data[name] = ak.unflatten(flat, n_cand)
        elif name.startswith("PFCands_"):
            short = name.removeprefix("PFCands_")
            flat = {"eta": pf_eta, "phi": pf_phi}.get(short, f32(rng.uniform(0, 1, len(pf_eta))))
            data[name] = ak.unflatten(flat, n_pf)
        elif name.startswith("TauPlusMET_"):
            data[name] = ak.unflatten(cand["vtx_chi2"] + 1000, n_cand)  # tags its candidate
        else:
            data[name] = np.ones(n_events, dtype=np.int64)
    data["event"] = np.arange(n_events, dtype=np.int64)
    data["nTauPlusMET"] = n_cand.astype(np.int32)
    data["TauTo3Mu_extra"] = ak.unflatten(f32(np.arange(n)), n_cand)  # not in the required list
    data["Muon_pt"] = ak.unflatten(f32(rng.uniform(1, 5, 2 * n_events)), np.full(n_events, 2))
    return data


def _expected_mask(data):
    flat = {k: ak.to_numpy(ak.flatten(data[f"TauTo3Mu_{k}"])) for k in
            [f"mu{i}_{s}" for i in (1, 2, 3) for s in ("pt", "eta", "phi", "charge")] + ["vtx_chi2"]}
    p = {i: _p4(flat[f"mu{i}_pt"], flat[f"mu{i}_eta"], flat[f"mu{i}_phi"], MU) for i in (1, 2, 3)}
    m3 = _mass(p[1], p[2], p[3])
    reasons = {"chi2": flat["vtx_chi2"] >= CUTS["vtx_chi2_max"],
               "mass": ~((m3 > CUTS["mass_range"][0]) & (m3 < CUTS["mass_range"][1]))}
    veto, same_sign_near = np.zeros(len(m3), bool), np.zeros(len(m3), bool)
    for i, j in resonance_skim.PAIRS:
        m = _mass(p[i], p[j])
        near = np.any([np.abs(m - v) < CUTS["veto_half_width"] for v in CUTS["veto_masses"]], axis=0)
        os_ = flat[f"mu{i}_charge"] * flat[f"mu{j}_charge"] < 0
        veto |= near & os_
        same_sign_near |= near & ~os_
    reasons.update(veto=veto, same_sign_near=same_sign_near)
    passed = ~(reasons["chi2"] | reasons["mass"] | veto)
    return ak.unflatten(passed, ak.num(data["TauTo3Mu_vtx_chi2"])), reasons


@pytest.fixture
def toy_run(load, user, tmp_path):
    data = _toy_raw()
    write_tree(tmp_path / "in" / "sig.root", data)
    write_tree(tmp_path / "in" / "bkg.root", _toy_raw(50, seed=2))
    user["inputs"].update(signal=[str(tmp_path / "in" / "sig.root")], background=[str(tmp_path / "in" / "bkg.root")],
                          already_resonance_skimmed=False)
    files = resonance_skim.run(load(user), tmp_path / "0_resonance_skim")
    tree = uproot.open(files["signal"][0])["Events"]
    out = tree.arrays(library="ak")
    return data, out[np.argsort(ak.to_numpy(out.event))], set(tree.keys())


def test_toy_covers_every_cut(toy_run):
    data, _, _ = toy_run
    passed, reasons = _expected_mask(data)
    only = lambda r: reasons[r] & ~np.any([reasons[o] for o in ("chi2", "mass", "veto") if o != r], axis=0)
    for r in ("chi2", "mass", "veto"):
        assert only(r).sum() >= 5, r  # each cut alone removes candidates
    assert (reasons["same_sign_near"] & ~reasons["veto"] & ~reasons["chi2"] & ~reasons["mass"]).sum() >= 3
    n_pass = ak.to_numpy(ak.sum(passed, axis=1))
    n_cand = ak.to_numpy(ak.num(passed))
    assert ((n_pass > 0) & (n_pass < n_cand)).sum() >= 10  # events that lose some candidates
    assert ((n_cand > 0) & (n_pass == 0)).sum() >= 10       # events removed by the cuts


def test_toy_events_and_candidates(toy_run):
    data, out, keys = toy_run
    passed, _ = _expected_mask(data)
    kept = ak.to_numpy(ak.any(passed, axis=1))
    assert ak.to_numpy(out.event).tolist() == np.flatnonzero(kept).tolist()
    p = passed[kept]
    for short in ("vtx_chi2", "mu2_pt", "extra"):
        assert out[f"Slimmed_TauTo3Mu_{short}"].tolist() == data[f"TauTo3Mu_{short}"][kept][p].tolist(), short
    assert out.nSlimmedTauTo3Mu.tolist() == ak.sum(p, axis=1).tolist()
    for mask in ("pass_mask", "pass_chi2_mask", "pass_mass_mask", "pass_m12_mask", "pass_m13_mask", "pass_m23_mask"):
        assert ak.all(out[f"Slimmed_TauTo3Mu_{mask}"]), mask
    assert ak.all(out.Slimmed_TauTo3Mu_mu1_mass == MU)
    assert not any(k.startswith(("TauTo3Mu_", "PFCands_", "_rs_")) for k in keys)
    assert {"Muon_pt", "nTauPlusMET", "run"} <= keys  # other branches kept


def test_toy_tauplusmet_stays_aligned(toy_run):
    _, out, _ = toy_run
    assert out.TauPlusMET_Puppi_phi.tolist() == (out.Slimmed_TauTo3Mu_vtx_chi2 + 1000).tolist()
    assert out.nTauPlusMET.tolist() == out.nSlimmedTauTo3Mu.tolist()


def test_toy_pf_candidates_near_kept_taus(toy_run):
    data, out, _ = toy_run
    passed, _ = _expected_mask(data)
    kept = ak.to_numpy(ak.any(passed, axis=1))
    exp = []
    for ev in np.flatnonzero(kept):
        te = ak.to_numpy(data["TauTo3Mu_fitted_eta"][ev][passed[ev]])[None, :]
        tp = ak.to_numpy(data["TauTo3Mu_fitted_phi"][ev][passed[ev]])[None, :]
        pe = ak.to_numpy(data["PFCands_eta"][ev])[:, None]
        pp = ak.to_numpy(data["PFCands_phi"][ev])[:, None]
        dr = np.sqrt((pe - te) ** 2 + ((pp - tp + np.pi) % (2 * np.pi) - np.pi) ** 2).min(axis=1, initial=999)
        exp.append(ak.to_numpy(data["PFCands_pt"][ev])[dr < CUTS["pf_dr_max"]].tolist())
    assert out.Slimmed_PFCands_pt.tolist() == exp
    assert out.nSlimmedPFCands.tolist() == [len(e) for e in exp]
    assert 0 < sum(map(len, exp)) < ak.sum(ak.num(data["PFCands_pt"][kept]))  # some kept, some dropped


def test_branch_map_on_raw_names(load, user, tmp_path):
    data = _toy_raw(50)
    data["PFCand_pt"] = data.pop("PFCands_pt")
    for s in ("sig", "bkg"):
        write_tree(tmp_path / "in" / f"{s}.root", data)
    user["inputs"].update(signal=[str(tmp_path / "in" / "sig.root")], background=[str(tmp_path / "in" / "bkg.root")],
                          already_resonance_skimmed=False)
    user["branch_map"] = {"PFCands_pt": "PFCand_pt"}
    files = resonance_skim.run(load(user), tmp_path / "0_resonance_skim")
    keys = set(uproot.open(files["signal"][0])["Events"].keys())
    assert "Slimmed_PFCands_pt" in keys and "PFCand_pt" not in keys


# --- regression vs. the old skims (raw NanoAOD -> *_skimmed_no_resonances.root) ----

RAW_SIG = ("/pnfs/psi.ch/cms/trivcat/store/user/manzoni/tau3mu_gnn/WtoTauNu-Tauto3Mu_TuneCP5_13p6TeV_pythia8/"
           "05may25v2__WtoTauNu-Tauto3Mu_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24MiniAOD-140X_mcRun3_2024_realistic_"
           "v26-v2_MINIAODSIM/250506_093818/0000/tau3muNANO_mc_05may25v2_1.root")
RAW_BKG = ("/pnfs/psi.ch/cms/trivcat/store/user/manzoni/tau3mu_gnn/ParkingDoubleMuonLowMass0/"
           "*Run2024C*/*/0000/tau3muNANO_data_30apr2025v1_1.root")
OLD = "/pnfs/psi.ch/cms/trivcat/store/user/kfuss/tau3mu_gnn_round2/data/{}/old_data/{}_skimmed_no_resonances.root"
OLD_SIG, OLD_BKG = OLD.format("signal_W", "signal"), OLD.format("background_W", "data")
# Settings that reproduce the old skims: no lower mass cut, chi2 < 100, rounded veto masses.
OLD_CUTS = {"vtx_chi2_max": 100, "mass_range": [0.0, 3.0], "veto_masses": [0.782, 1.020],
            "veto_half_width": 0.04, "pf_dr_max": 1.0}
KEY = ["run", "luminosityBlock", "event"]


def _keys(tree):
    d = tree.arrays(KEY, library="np")
    return list(zip(*(d[c] for c in KEY)))


def _read_rows(tree, names, rows):
    """tree entries rows (sorted), in that order. Reads contiguous blocks only."""
    parts = []
    breaks = np.flatnonzero(np.diff(rows) > 1000) + 1
    for block in np.split(rows, breaks):
        lo, hi = int(block[0]), int(block[-1]) + 1
        parts.append(tree.arrays(names, entry_start=lo, entry_stop=hi, library="ak")[block - lo])
    return ak.concatenate(parts)


def test_regression_vs_old_skims(load, user, fake_repo, tmp_path):
    raw_bkg = sorted(glob.glob(RAW_BKG))
    if not (os.path.exists(RAW_SIG) and raw_bkg and os.path.exists(OLD_SIG) and os.path.exists(OLD_BKG)):
        pytest.skip("raw NanoAOD / old skims not reachable on this machine")
    write_yaml(fake_repo / "configs" / "analysis_defaults_W.yaml", {**TOY_DEFAULTS, "resonance_skim": OLD_CUTS})
    user["inputs"].update(signal=[RAW_SIG], background=raw_bkg[:1], already_resonance_skimmed=False)
    files = resonance_skim.run(load(user), tmp_path / "0_resonance_skim")

    for sample, raw, old_path in (("signal", RAW_SIG, OLD_SIG), ("background", raw_bkg[0], OLD_BKG)):
        new, old = uproot.open(files[sample][0])["Events"], uproot.open(old_path)["Events"]
        assert set(old.keys()) == set(new.keys()) - {"Slimmed_TauTo3Mu_pass_mass_mask"}, sample
        raw_keys = set(_keys(uproot.open(raw)["Events"]))
        old_index = {k: i for i, k in enumerate(_keys(old)) if k in raw_keys}
        new_keys = _keys(new)
        assert len(new_keys) > 1000 and set(new_keys) == set(old_index), sample  # same events

        names = sorted(set(old.keys()))
        rows = np.array([old_index[k] for k in new_keys])
        order = np.argsort(rows)
        a = new.arrays(names, library="ak")[order]
        o = _read_rows(old, names, rows[order])
        # TauPlusMET_* is now masked like the candidates; the old skims kept all entries.
        same = ak.to_numpy(a.nSlimmedTauTo3Mu == a.nTauTo3Mu)
        assert (~same).sum() > 0, sample
        for name in names:
            x, y = (a[name][same], o[name][same]) if name.startswith(("TauPlusMET_", "nTauPlusMET")) \
                else (a[name], o[name])
            if x.ndim > 1:
                assert ak.num(x, axis=1).tolist() == ak.num(y, axis=1).tolist(), (sample, name)
            np.testing.assert_array_equal(ak.to_numpy(ak.ravel(x)), ak.to_numpy(ak.ravel(y)), err_msg=f"{sample} {name}")


def test_extra_branches_are_checked(load, user, tmp_path):
    _inputs(user, tmp_path)
    assert resonance_skim.run(load(user), tmp_path / "0_resonance_skim")["signal"]
    with pytest.raises(b.BranchError, match="PV_npvs"):
        resonance_skim.run(load(user), tmp_path / "0_resonance_skim", extra_branches=b.TRAINING_EXTRA["W"])
    assert not (tmp_path / "0_resonance_skim").exists()
