import hashlib

import pytest

from conftest import REPO_ROOT

MODEL = "new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2"

# SHA-256 of the originals in /work/kfuss/SVJ_GNN/output/, taken 2026-10-07.
EXPECTED = {
    f"models/W/gnn/{MODEL}/{MODEL}.pt": "b6991427c7e07cf01b8e5384dc6704bb151bbff3f94093ef4babecd8f2de3c1e",
    f"models/W/gnn/{MODEL}/Normalization.pkl": "ffcf2239032b68ab222669dfb7010d3ad17511bdf2d5acb41669acd64395d57e",
    f"models/W/gnn/{MODEL}/arch_config.json": "bf5e2e1762b6a577853dad4150b1ae3e5f41efab0f0ba65ef8da5775e6a081bd",
    "models/W/bdt/ISO_classifiers_24feb2026_v0.pck": "2aa4de47344a372ca0575c017fe48b8c1f222cc50dad353aeb459d0578003395",
}


@pytest.mark.parametrize("rel_path, sha", EXPECTED.items())
def test_frozen_model_file_is_exact_copy(rel_path, sha):
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"missing frozen model file: {rel_path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
