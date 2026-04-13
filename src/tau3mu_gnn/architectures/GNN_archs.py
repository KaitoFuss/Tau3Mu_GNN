import numpy as np
import random
import os
from itertools import product

# final_layer
FC_PARAMS_OPTIONS = {
    # 2-layer
    1:  [128, 64],
    2:  [512, 64],
    3:  [1024, 128],
    # 3-layer
    4:  [256, 128, 64],
    5:  [512, 128, 64],
    6:  [512, 256, 32],
    7:  [512, 512, 64],
    8:  [1024, 256, 64],
    9:  [2056, 512, 64],
    # 4-layer
    10: [256, 128, 128, 64], ### this was the previous standard
    11: [256, 192, 128, 64],
    12: [512, 256, 128, 64],
}

# scale_down
SCALE_DOWN_PARAMS_OPTIONS = {
    # 2-layer
    1:  [64, 8],
    2:  [256, 32],
    3:  [512, 32],
    4:  [512, 64],
    # 3-layer → non-terminal
    5:  [128, 64, 16],
    6:  [128, 64, 32],
    7:  [256, 64, 8],
    8:  [256, 64, 16],
    9:  [256, 64, 32],
    10:  [1024, 128, 16],
    11:  [256, 128, 32, 8],        
    # 2-layer → terminal (width==1)
    12: [64, 1],
    13: [128, 1],
    # 3-layer → terminal
    14: [256, 64, 1],
    15: [512, 128, 32, 1],
    16: [256, 128, 64, 16],    
}

REGISTRY_PATH = "/work/kfuss/SVJ_GNN/output/"

def load_registry(path):
    if not os.path.exists(path):
        return set()
    data = np.load(path, allow_pickle=True)
    return set(data["tried_keys"].tolist())

def save_registry(path, tried_keys):
    np.savez(path, tried_keys=np.array(list(tried_keys)))

def inject_dropout(widths, dropout, terminal_dropout=0.0):
    """Attach dropout to each layer; terminal layer (width==1) gets terminal_dropout."""
    return [(w, terminal_dropout if w == 1 else dropout) for w in widths]

def sample_new_config(registry, max_attempts=10_000):
    for _ in range(max_attempts):
        fc_idx        = random.randint(1, len(FC_PARAMS_OPTIONS))
        fusion_idx    = random.randint(1, len(SCALE_DOWN_PARAMS_OPTIONS))

        fc_widths     = FC_PARAMS_OPTIONS[fc_idx]
        fusion_widths = SCALE_DOWN_PARAMS_OPTIONS[fusion_idx]

        key = repr((fc_idx, fusion_idx))
        if key not in registry:
            fc_params     = inject_dropout(fc_widths, 0.2)
            fusion_params = inject_dropout(fusion_widths, 0.2)
            return fc_params, fusion_params, key, [f"final_layer_{fc_idx}", f"scale_down_{fusion_idx}"]

    raise RuntimeError("Search space exhausted.")

def get_arch_configs_random(tau_channel, registry_path=REGISTRY_PATH):
    registry_path = registry_path + tau_channel + "_arch_registry.npz"
    registry = load_registry(registry_path)
    fc, fusion, key, unique_ident = sample_new_config(registry)

    # Register immediately to avoid race conditions across jobs
    registry.add(key)
    save_registry(registry_path, registry)

    config = {
        "conv_params": [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
        "fc_params": fc,
        "use_fusion": True,
        "dropout": 0.2633831214702671,
        "fusion_scale_down_params": fusion,
    }
    return config, unique_ident


def get_arch_configs(arch):
    arch_configs = {
        "arch1": {
            "conv_params": [256, 128, 64, 32],
            "fc_params": [(32, 0.8)],
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "arch2": {
            "conv_params": [256, 256, 128, 128, 64, 64, 32, 32],
            "fc_params": [(32, 0.8)],
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "arch3": {
            "conv_params": [512, 256, 128, 64, 32],
            "fc_params": [(32, 0.8)],
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "arch4": {
            "conv_params": [512, 256, 128, 64, 32, 16, 8],
            "fc_params": [(32, 0.8)],
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "arch5": {
            "conv_params": [256, 128, 64, 32],
            "fc_params": [(64, 0.8)],
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "arch6": {
            "conv_params": [256, 256, 128, 128, 64, 64, 32, 32],
            "fc_params": [(64, 0.8)],
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "arch7": {
            "conv_params": [512, 256, 128, 64, 32],
            "fc_params": [(64, 0.8)],
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "arch8": {
            "conv_params": [512, 256, 128, 64, 32, 16, 8],
            "fc_params": [(64, 0.8)],
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "arch_optuna" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(51, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "final_layer1" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(256, 0.8), (128, 0.8), (64, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "final_layer2" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(512, 0.8), (64, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "final_layer3" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(256, 0.8), (128, 0.8), (128, 0.8), (64, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "final_layer4" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(512, 0.8), (256, 0.8), (32, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "final_layer5" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(2056, 0.8), (512, 0.8), (64, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [],
        },
        "scale_down1" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(256, 0.8), (128, 0.8), (128, 0.8), (64, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [(256, 0.2), (64, 0.2), (1, 0.)],
        },
        "scale_down2" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(256, 0.8), (128, 0.8), (128, 0.8), (64, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [(256, 0.2), (64, 0.2), (8, 0.2)],
        },
        "scale_down3" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(256, 0.8), (128, 0.8), (128, 0.8), (64, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [(256, 0.2), (64, 0.2), (16, 0.2)],
        },
        "scale_down4" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(256, 0.8), (128, 0.8), (128, 0.8), (64, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [(256, 0.2), (64, 0.2), (32, 0.2)],
        },
        "scale_down5" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(256, 0.8), (128, 0.8), (128, 0.8), (64, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [(64, 0.2), (8, 0.2)],
        },
        "scale_down6" : {
            "conv_params" : [1004, int(1004/2), int(1004/2**2), int(1004/2**3), int(1004/2**4)],
            "fc_params" : [(256, 0.8), (128, 0.8), (128, 0.8), (64, 0.8)] ,
            "use_fusion": True,
            "dropout": 0.2633831214702671,
            "fusion_scale_down_params": [(256, 0.2), (32, 0.2)],
        },
    }

    return arch_configs[arch], [arch]