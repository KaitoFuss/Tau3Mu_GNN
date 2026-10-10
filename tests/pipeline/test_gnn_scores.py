import awkward as ak
import numpy as np
import pytest
import torch
import uproot

from conftest import write_tree
from tau3mu_gnn.gnn_training.gnn_inputs import input_branch_names

P = "Selected_TauTo3Mu_"
PF = "Slimmed_PFCands_"
MU_ETA = {1: 0.01, 2: 0.02, 3: -0.01}
# PF candidate etas (phi = 0). The first three sit on the muons (removed by delete_muons).
PF_GOOD = [0.01, 0.02, -0.01, 0.1, 0.15, -0.1, 0.2, 3.0]   # 4 left in the 0.5 cone
PF_EMPTY = [0.01, 0.02, -0.01, 3.0]                        # 0 left -> event skipped


class ToyGNN(torch.nn.Module):
    """Score = first graph-level feature = fitted pt of the scored candidate."""

    def forward(self, data):
        return data.graph_level_features[:, :1]


class TwiceGNN(ToyGNN):
    def forward(self, data):
        return 2 * super().forward(data)


def gnn_events(cands, bdt, pf=None, first_event=10):
    """One event per entry of cands (list of candidate pts). bdt: bdt_ISO per event.
    pf: PF candidate etas per event (default PF_GOOD). Every stage 3 input branch, plus 'extra'."""
    n = len(cands)
    pf = pf or [PF_GOOD] * n

    def per_cand(f):
        return ak.Array([[f(j) for j in range(len(ev))] for ev in cands])

    def per_pf(f, dtype=float):
        return ak.Array([[dtype(f(k)) for k in range(len(ev))] for ev in pf])

    data = {}
    for name in sorted(input_branch_names("W")):
        if name in ("run", "luminosityBlock"):
            data[name] = np.ones(n, dtype=np.uint32)
        elif name == "event":
            data[name] = np.arange(first_event, first_event + n, dtype=np.uint64)
        elif name == "nSelectedTauTo3Mu":
            data[name] = np.array([len(c) for c in cands], dtype=np.int32)
        elif name == "bdt_ISO":
            data[name] = np.asarray(bdt, dtype=np.float32)
        elif name == f"{P}fitted_pt":
            data[name] = ak.Array(cands)
        elif name == f"{P}fitted_mass":
            data[name] = per_cand(lambda j: 1.70 + 0.01 * j)
        elif name.startswith(P) and name.endswith(("_eta", "_phi")):
            # tau at (0, 0); muon i at (MU_ETA[i], 0)
            i = int(name[len(P) + 2]) if name[len(P):].startswith("mu") and name.endswith("_eta") else 0
            data[name] = per_cand(lambda j: MU_ETA.get(i, 0.0))
        elif name == f"{PF}eta":
            data[name] = ak.Array(pf)
        elif name == f"{PF}pdgId":
            data[name] = per_pf(lambda k: 211, int)
        elif name == f"{PF}charge":
            data[name] = per_pf(lambda k: 1 - 2 * (k % 2), int)
        elif name.startswith(PF):
            value = {"pt": lambda k: 1.0 + k, "mass": lambda k: 0.1, "puppiWeight": lambda k: 1.0}
            data[name] = per_pf(value.get(name[len(PF):], lambda k: 0.0))
        else:  # other Selected_TauTo3Mu_* and TauPlusMET_* (per candidate)
            data[name] = per_cand(lambda j: 1.0 + j)
    data["extra"] = np.arange(n, dtype=np.float64)
    return data


def _score(tmp_path, files_data, tree="Events", cut=0.8, models=None):
    """Score with each model (default: one ToyGNN 'm'). Returns (output files of the first model, its npz)."""
    from tau3mu_gnn.gnn_training.ScoreProducer import score_events

    models = models or {"m": (ToyGNN(), None)}
    inputs = [str(write_tree(tmp_path / f"in_{i}.root", data, tree)) for i, data in enumerate(files_data)]
    outputs = {name: {f: str(tmp_path / name / f"out_{i}.root") for i, f in enumerate(inputs)} for name in models}
    npz_paths = {name: tmp_path / name / "scores.npz" for name in models}
    score_events(inputs, models, outputs, npz_paths, "W", "delete_muons", cut, tree_name=tree,
                 device=torch.device("cpu"))
    first = next(iter(models))
    return list(outputs[first].values()), np.load(npz_paths[first])


# event 0: two candidates (first one, pt 100, is scored; not the highest-pt 900), 1: one, 2: none,
# 3: BDT below cut, 4: no PF candidate left in the cone
CANDS = [[100.0, 900.0], [300.0], [], [400.0], [500.0]]
BDT = [0.9, 0.95, 0.9, 0.5, 0.9]
PF_ETAS = [PF_GOOD, PF_GOOD, PF_GOOD, PF_GOOD, PF_EMPTY]
EXPECTED = [100.0, 300.0, -1.0, -1.0, -1.0]


def test_gnn_score_per_event(tmp_path):
    (out,), npz = _score(tmp_path, [gnn_events(CANDS, BDT, PF_ETAS)])
    tree = uproot.open(out)["Events"]
    np.testing.assert_array_equal(tree["GNN_score"].array(library="np"), np.float32(EXPECTED))
    np.testing.assert_allclose(npz["scores"], [100.0, 300.0])
    np.testing.assert_allclose(npz["tau_mass"], [1.70, 1.70], rtol=1e-6)


def test_all_events_and_branches_kept(tmp_path):
    data = gnn_events(CANDS, BDT, PF_ETAS)
    (out,), _ = _score(tmp_path, [data])
    tree = uproot.open(out)["Events"]
    in_keys = set(uproot.open(tmp_path / "in_0.root")["Events"].keys())  # includes uproot's n<name> counters
    assert set(tree.keys()) == in_keys | {"GNN_score"}
    np.testing.assert_array_equal(tree["extra"].array(library="np"), data["extra"])
    np.testing.assert_array_equal(tree["event"].array(library="np"), data["event"])


def test_custom_tree_name(tmp_path):
    (out,), _ = _score(tmp_path, [gnn_events(CANDS, BDT, PF_ETAS)], tree="Tree3mu")
    assert "GNN_score" in uproot.open(out)["Tree3mu"].keys()


def test_several_files_in_order(tmp_path):
    a = gnn_events([[300.0]], [0.9])
    b = gnn_events([[700.0], [200.0]], [0.9, 0.9], first_event=50)
    (out_a, out_b), npz = _score(tmp_path, [a, b])
    np.testing.assert_allclose(npz["scores"], [300.0, 700.0, 200.0])
    np.testing.assert_array_equal(uproot.open(out_b)["Events"]["GNN_score"].array(library="np"),
                                  np.float32([700.0, 200.0]))


def test_several_models_one_pass(tmp_path):
    from tau3mu_gnn.gnn_training.ScoreProducer import score_events

    inp = str(write_tree(tmp_path / "in.root", gnn_events(CANDS, BDT, PF_ETAS)))
    models = {"a": (ToyGNN(), None), "b": (TwiceGNN(), None)}
    outputs = {n: {inp: str(tmp_path / n / "out.root")} for n in models}
    npz_paths = {n: tmp_path / n / "s.npz" for n in models}
    score_events([inp], models, outputs, npz_paths, "W", "delete_muons", 0.8, device=torch.device("cpu"))
    b = uproot.open(outputs["b"][inp])["Events"]["GNN_score"].array(library="np")
    np.testing.assert_array_equal(b, np.float32([200.0, 600.0, -1.0, -1.0, -1.0]))
    np.testing.assert_allclose(np.load(npz_paths["a"])["scores"], [100.0, 300.0])


def test_outputs_must_match_models(tmp_path):
    from tau3mu_gnn.gnn_training.ScoreProducer import score_events

    with pytest.raises(ValueError, match="outputs"):
        score_events(["x.root"], {"a": (ToyGNN(), None)}, {"b": {}}, {"a": tmp_path / "s.npz"},
                     "W", "delete_muons", 0.8)


def test_no_cut_scores_low_bdt_event(tmp_path):
    (out,), _ = _score(tmp_path, [gnn_events([[400.0]], [0.5])], cut=None)
    assert uproot.open(out)["Events"]["GNN_score"].array(library="np")[0] == 400.0


def test_old_cli_paths(tmp_path, monkeypatch):
    import tau3mu_gnn.scripts.ScoreCreation as cli

    monkeypatch.setattr(cli, "load_model", lambda *a, **k: (ToyGNN(), None))
    root = tmp_path / "signal_W" / "bdt_full"
    write_tree(root / "sig.root", gnn_events(CANDS, BDT, PF_ETAS))
    out_folder, storage = tmp_path / "output", tmp_path / "scratch"
    cli.main(["--root", str(root), "--output_folder", f"{out_folder}/", "--model_names", "m1",
              "--archs", "random", "--tau_channel", "W", "--gnn_mode", "final_layer",
              "--storage_folder", str(storage), "--conv_types", "GINE",
              "--unsculpting_mode", "delete_muons", "--bdt_preselection_cut", "0.8"])
    assert (storage / "m1" / "sig_with_scores.root").exists()
    assert (out_folder / "m1" / "raw_scores_and_tau_mass_signal_with_bdt_cut.npz").exists()
