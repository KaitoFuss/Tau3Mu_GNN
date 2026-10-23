import copy
from pathlib import Path

import numpy as np
import pytest
import uproot
import yaml

import tau3mu_gnn

REPO_ROOT = Path(__file__).resolve().parents[2]

# Fail fast if the editable install (main checkout) is imported instead of this tree.
_imported_from = Path(tau3mu_gnn.__file__).resolve().parents[2]
if _imported_from != REPO_ROOT:
    raise RuntimeError(
        f"tau3mu_gnn imported from {_imported_from}, expected {REPO_ROOT}. "
        f"Run with PYTHONPATH={REPO_ROOT / 'src'}"
    )


TOY_MODEL = "toy_model"

TOY_DEFAULTS = {
    "resonance_skim": {"vtx_chi2_max": 20, "mass_range": [1.0, 3.0], "veto_masses": [0.78266, 1.019461],
                       "veto_half_width": 0.04, "pf_dr_max": 1.0},
    "skim": {"dz_max": 0.5, "cos_alpha_2d_min": 0.5},
    "bdt": {"classifier": "models/W/bdt/toy.pck"},
    "gnn": {
        "model_dir": "models/W/gnn",
        "model_name": TOY_MODEL,
        "conv_type": "GINE",
        "gnn_mode": "final_layer",
        "unsculpting_mode": "delete_muons",
        "graph_type": "skimmed_dzd0_cone_size_0.5_knn_10",
        "bdt_preselection_cut": 0.8,
    },
    "limit": {"tau_mass": 1.78, "mass_window": 0.06, "fit_range": [1.62, 1.98], "cl": 0.9},
    "slurm": {
        "account": "t3",
        "jobs": {name: {"partition": "standard", "mem": "4G", "time": "0-01:00:00", "cpus": 1}
                 for name in ("skim", "bdt", "gnn_signal", "gnn_background", "eval", "limit", "summary")},
    },
}

TOY_USER = {
    "analysis_name": "toy_run",
    "channel": "W",
    "paths": {"output_base": "/tmp/toy_out", "log_dir": "/tmp/toy_logs"},
    "inputs": {
        "signal": ["/data/sig.root"],
        "background": ["/data/bkg_*.root"],
        "already_resonance_skimmed": True,
    },
    "limit": {"signal_weight": 1.9e-5},
}


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


@pytest.fixture
def fake_repo(tmp_path):
    write_yaml(tmp_path / "configs" / "analysis_defaults_W.yaml", TOY_DEFAULTS)
    for rel in [
        "models/W/bdt/toy.pck",
        f"models/W/gnn/{TOY_MODEL}/{TOY_MODEL}.pt",
        f"models/W/gnn/{TOY_MODEL}/Normalization.pkl",
        f"models/W/gnn/{TOY_MODEL}/arch_config.json",
    ]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return tmp_path


@pytest.fixture
def user():
    return copy.deepcopy(TOY_USER)


@pytest.fixture
def load(fake_repo):
    from tau3mu_gnn.pipeline.config import load_config

    def _load(user_dict=None, raw_text=None):
        path = fake_repo / "user.yaml"
        if raw_text is not None:
            path.write_text(raw_text)
        else:
            write_yaml(path, user_dict)
        return load_config(path, defaults_dir=fake_repo / "configs")

    return _load


def make_root_file(path, names, tree="Events"):
    """Write a tiny ROOT file with one flat float branch per name (2 entries)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as f:
        f[tree] = {name: np.array([1.0, 2.0]) for name in names}
    return path


def write_tree(path, data, tree="Events"):
    """Write data (name -> numpy / awkward array) as a TTree.
    mktree, because f[tree] = data writes an RNTuple in uproot 5.7."""
    import awkward as ak

    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as f:
        f.mktree(tree, {k: ak.type(v) if isinstance(v, ak.Array) else v.dtype for k, v in data.items()})
        f[tree].extend(data)
    return path


VAR_BINS = [0, 10, 15, 18, 20] + list(range(21, 61)) + [65, 100]

TOY_TRAINING_DEFAULTS = {
    "sideband_half_width": 0.08,
    "bdt_training": {"max_depth": 8, "learning_rate": 0.005, "n_estimators": 1000, "early_stopping_rounds": 100,
                     "subsample": 0.6, "colsample_bytree": 0.7, "gamma": 10, "reg_alpha": 0.6, "reg_lambda": 1.2,
                     "min_child_weight_fraction": 1.0e-6, "kfold": 6, "test_size": 0.4,
                     "split_seed": 1986, "sample_seed": 42},
    "npvs": {"var_bins": VAR_BINS, "uniform_bins": [0, 100, 101]},
    "graphs": {"connection": "knn", "k": 10, "cone_size": 0.5, "use_displacement": True,
               "unsculpting_mode": "delete_muons", "shuffle_seed": 1986},
    "gnn_training": {"conv_type": "GINE", "gnn_mode": "final_layer", "arch": "fl8_sd4", "learning_rate": 0.0016,
                     "batch_size": 128, "disco": 0.0, "epochs": 1000, "early_stopping_patience": 20,
                     "lr_scheduler_patience": 10, "class_fraction": [0.5, 0.5],
                     "windows": {"train": [0.0, 0.6], "validation": [0.6, 0.7], "test": [0.7, 1.0]}},
    "comet": {"workspace": "toy_ws", "api_key": "toy-key"},
    "slurm": {
        "account": "t3",
        "jobs": {**{name: {"partition": "standard", "mem": "4G", "time": "0-01:00:00", "cpus": 1}
                    for name in ("skim", "bdt_train", "bdt_infer", "npvs", "graphs_signal", "graphs_background",
                                 "benchmark_limit", "benchmark_summary")},
                 "gnn_train": {"partition": "gpu", "mem": "4G", "time": "0-01:00:00", "cpus": 1,
                               "account": "gpu_gres", "gres": "gpu:1"}},
    },
}

TOY_TRAINING_USER = {
    "training_name": "toy_train",
    "channel": "W",
    "paths": {"output_base": "/tmp/toy_out", "log_dir": "/tmp/toy_logs"},
    "inputs": {
        "signal": ["/data/sig.root"],
        "background": ["/data/bkg_*.root"],
        "already_resonance_skimmed": True,
    },
    "graphs": {"bdt_preselection_cut": 0.8},
    "benchmark": {"bdt_score_cut": [0.98, 0.99]},
    "limit": {"signal_weight": 1.9e-5},
}


@pytest.fixture
def training_user():
    return copy.deepcopy(TOY_TRAINING_USER)


class _TrainingLoader:
    """load_training(user_dict) writes .defaults as training_defaults_W.yaml, then loads."""

    def __init__(self, repo):
        self.repo = repo
        self.defaults = copy.deepcopy(TOY_TRAINING_DEFAULTS)

    def __call__(self, user_dict=None, raw_text=None):
        from tau3mu_gnn.pipeline.training.config import load_training_config

        write_yaml(self.repo / "configs" / "training_defaults_W.yaml", self.defaults)
        path = self.repo / "training_user.yaml"
        if raw_text is not None:
            path.write_text(raw_text)
        else:
            write_yaml(path, user_dict)
        return load_training_config(path, defaults_dir=self.repo / "configs")


@pytest.fixture
def load_training(fake_repo):
    return _TrainingLoader(fake_repo)
