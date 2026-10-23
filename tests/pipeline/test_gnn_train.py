import json

from tau3mu_gnn.pipeline.config import REPO_ROOT


def test_arch_config_equals_frozen_file():
    from tau3mu_gnn.architectures.GNN_archs import get_arch_configs
    config, ident = get_arch_configs("fl8_sd4", "W", 0.0, registry_path=None)
    frozen = json.loads((REPO_ROOT / "models/W/gnn/new_hope_W_fl_8_sd_4_GINE_delete_muons_vs2/arch_config.json").read_text())
    assert json.loads(json.dumps(config)) == frozen and ident == ["fl_8", "sd_4"]


def test_arch_import_no_side_effects(monkeypatch):
    import importlib
    import os
    import tau3mu_gnn.architectures.GNN_archs as archs
    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: False if p in ("/eos/user/k/kfuss/", "/work/kfuss/") else real_exists(p))
    try:
        reloaded = importlib.reload(archs)          # unknown machine: no raise
        assert reloaded.REGISTRY_PATH is None
        reloaded.get_arch_configs("fl8_sd4", "W", 0.0)   # default registry path None: nothing written
    finally:
        monkeypatch.undo()
        importlib.reload(archs)
