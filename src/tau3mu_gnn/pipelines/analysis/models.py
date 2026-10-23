"""Analysis key `models:` (step 12h): the BDT and GNN of a training run instead of the frozen models.

The value is the path of a training gnn_vsN folder. Its manifest (phase gnn) lists the model files; its
upstream bdt_vsN (a sibling folder) lists the BDT classifier. The model-defining settings (conv type,
GNN mode, unsculpting mode, graph type, BDT preselection cut) come from the gnn manifest settings, so they
travel with the model. Files are read through storage.local_path (the /pnfs path of a root:// URL).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tau3mu_gnn.graph.naming import graph_suffix
from tau3mu_gnn.pipelines.common import storage

FROZEN = "frozen"
_GNN_VERSION_RE = re.compile(r"^gnn_vs[1-9]\d*$")
_MODEL_FILES = ("model", "normalization", "arch_config")


@dataclass(frozen=True)
class TrainedModels:
    source: str                     # the gnn_vsN path as given
    classifier: Path
    model_dir: Path                 # holds <model_name>/
    model_name: str
    conv_type: str
    gnn_mode: str
    unsculpting_mode: str | None
    graph_type: str
    bdt_preselection_cut: float


def _manifest(folder: Path, phase: str, source: str, problems: list[str]) -> dict | None:
    path = folder / "manifest.json"
    if not path.is_file():
        problems.append(f"models: {source}: no {path.name} in {folder} (the training phase is not finished?)")
        return None
    try:
        m = json.loads(path.read_text())
    except ValueError as e:
        problems.append(f"models: {path} is not valid JSON: {e}")
        return None
    if m.get("phase") != phase:
        problems.append(f"models: {path} is a manifest of phase {m.get('phase')!r}, expected {phase!r}")
        return None
    return m


def read_trained_models(gnn_version: str, problems: list[str]) -> TrainedModels | None:
    """Resolve a training gnn_vsN folder. Appends every problem to problems; None if there is any."""
    n_before = len(problems)
    name = str(gnn_version).rstrip("/").rsplit("/", 1)[-1]
    if not _GNN_VERSION_RE.match(name):
        problems.append(f"models: {gnn_version} is not a training gnn_vs<N> folder (the GNN training "
                        f"phase holds the models), got folder {name!r}")
        return None
    gnn_dir = storage.local_path(str(gnn_version).rstrip("/"))
    gnn = _manifest(gnn_dir, "gnn", gnn_version, problems)
    if gnn is None:
        return None
    bdt_dir = gnn_dir.parent / str(gnn.get("upstream"))
    bdt = _manifest(bdt_dir, "bdt", gnn_version, problems)
    if bdt is None:
        return None
    try:
        files = {key: gnn_dir / gnn["outputs"][key] for key in _MODEL_FILES}
        files["classifier"] = bdt_dir / bdt["outputs"]["classifier"]
        graphs, training = gnn["settings"]["graphs"], gnn["settings"]["gnn_training"]
        graph_type = "skimmed" + graph_suffix(graphs["use_displacement"], graphs["cone_size"],
                                              graphs["connection"], k=graphs["k"], with_unsculpting=False)
        values = dict(conv_type=training["conv_type"], gnn_mode=training["gnn_mode"],
                      unsculpting_mode=graphs["unsculpting_mode"],
                      bdt_preselection_cut=float(graphs["bdt_preselection_cut"]))
    except (KeyError, TypeError) as e:
        problems.append(f"models: the manifests of {gnn_version} lack an entry: {e!r}")
        return None
    from tau3mu_gnn.gnn_training.gnn_inputs import INFERENCE_GRAPH
    for key, value in INFERENCE_GRAPH.items():
        if graphs.get(key) != value:
            problems.append(f"models: {gnn_version} was trained on graphs.{key} = {graphs.get(key)!r}, but GNN "
                            f"inference builds graphs with {key} = {value!r}; this model cannot be scored")
    for path in files.values():
        if not path.is_file():
            problems.append(f"models: file not found: {path}")
    if len(problems) > n_before:
        return None
    model = files["model"]
    return TrainedModels(source=str(gnn_version), classifier=files["classifier"], model_dir=model.parent.parent,
                         model_name=model.stem, graph_type=graph_type, **values)
