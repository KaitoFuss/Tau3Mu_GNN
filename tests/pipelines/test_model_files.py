import hashlib

import pytest

from conftest import REPO_ROOT

MODEL = "W_raw_Nano_AOD_fixed_W_fl_8_sd_4_GINE_delete_muons"

# SHA-256 of the 13c models (training W_raw_Nano_AOD_fixed bdt_vs1 / gnn_vs1 on dCache), taken 2026-10-30.
EXPECTED = {
    f"models/W/gnn/{MODEL}/{MODEL}.pt": "fbbae1e34c5c1915cdbd774d2623129ab56059aa2960355f5cd71b8331721d1e",
    f"models/W/gnn/{MODEL}/Normalization.pkl": "4eeede21e3ab17bd4e0b64f70714975e75bc9ba0759abcd52863dd6f7a971785",
    f"models/W/gnn/{MODEL}/arch_config.json": "bf5e2e1762b6a577853dad4150b1ae3e5f41efab0f0ba65ef8da5775e6a081bd",
    "models/W/bdt/ISO_classifiers_W_raw_Nano_AOD_fixed.pck": "3005dfddf0f8c53e1874032e1125956faa964787939b0a89761be6076d261b93",
}


@pytest.mark.parametrize("rel_path, sha", EXPECTED.items())
def test_frozen_model_file_is_exact_copy(rel_path, sha):
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"missing frozen model file: {rel_path}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
