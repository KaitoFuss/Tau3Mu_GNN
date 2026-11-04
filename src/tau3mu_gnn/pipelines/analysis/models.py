"""The BDT and GNN of the analysis and their model-defining settings, from one place per model.

- Frozen (default): analysis_defaults gives only the paths (bdt.classifier, gnn.model_dir, gnn.model_name).
  The settings are in <model_dir>/<model_name>/model_settings.json.
- Key `models:`: the path of a training gnn_vsN folder. Its manifest (phase gnn) lists the model
  files; its upstream bdt_vsN (a sibling folder) lists the BDT classifier. The settings are the gnn manifest
  settings. Files are read through storage.local_path (the mounted path of a root:// URL).

model_settings.json has the layout of those manifest settings: {"channel", "graphs": {bdt_preselection_cut,
connection, k, cone_size, use_displacement, unsculpting_mode}, "gnn_training": {conv_type, gnn_mode}}. The
model-defining settings (conv type, GNN mode, unsculpting mode, graph type, BDT preselection cut) so travel
with the model and are written only once.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tau3mu_gnn.graph.naming import graph_suffix
from tau3mu_gnn.pipelines.common import storage

FROZEN = "frozen"
SETTINGS_FILE = "model_settings.json"
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
    if m.get("pipeline", "training") != "training":   # no 'pipeline': written before it existed
        problems.append(f"models: {path} is a manifest of the {m['pipeline']} pipeline, expected a training "
                        f"folder (tau3mu-training)")
        return None
    if m.get("phase") != phase:
        problems.append(f"models: {path} is a manifest of phase {m.get('phase')!r}, expected {phase!r}")
        return None
    return m


def read_trained_models(gnn_version: str, channel: str, problems: list[str]) -> TrainedModels | None:
    """Resolve a training gnn_vsN folder of the given channel. Appends every problem to problems; None if
    there is any."""
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
    # no settings / channel: written before, passes
    if not _channel_ok((bdt.get("settings") or {}).get("channel"), channel, f"models: {gnn_version}", problems):
        return None
    try:
        files = {key: gnn_dir / gnn["outputs"][key] for key in _MODEL_FILES}
        files["classifier"] = bdt_dir / bdt["outputs"]["classifier"]
        settings = gnn["settings"]
    except (KeyError, TypeError) as e:
        problems.append(f"models: the manifests of {gnn_version} lack an entry: {e!r}")
        return None
    values = _model_values(settings, f"models: {gnn_version}", f"the manifests of {gnn_version}", problems)
    _check_files(files.values(), "models: file not found", problems)
    if values is None or len(problems) > n_before:
        return None
    model = files["model"]
    return TrainedModels(source=str(gnn_version), classifier=files["classifier"], model_dir=model.parent.parent,
                         model_name=model.stem, **values)


def read_frozen_models(classifier: Path, model_dir: Path, model_name: str, channel: str,
                       problems: list[str]) -> TrainedModels | None:
    """The frozen models of analysis_defaults (paths) with the settings in <model_dir>/<model_name>/
    model_settings.json. Appends every problem to problems; None if there is any."""
    n_before = len(problems)
    folder = model_dir / model_name
    _check_files([classifier, folder / f"{model_name}.pt", folder / "Normalization.pkl",
                  folder / "arch_config.json"], "frozen model file not found", problems)
    path = folder / SETTINGS_FILE
    try:
        settings = json.loads(path.read_text())
    except FileNotFoundError:
        problems.append(f"frozen model settings not found: {path} (the settings of the frozen model live "
                        f"next to its files)")
        return None
    except ValueError as e:
        problems.append(f"frozen model settings: {path} is not valid JSON: {e}")
        return None
    if not isinstance(settings, dict) or not _channel_ok(settings.get("channel"), channel,
                                                         f"frozen model {folder}", problems):
        if not isinstance(settings, dict):
            problems.append(f"frozen model settings: {path} must be a JSON object")
        return None
    values = _model_values(settings, f"frozen model {folder}", str(path), problems)
    if values is None or len(problems) > n_before:
        return None
    return TrainedModels(source=FROZEN, classifier=classifier, model_dir=model_dir, model_name=model_name,
                         **values)


def _channel_ok(trained: str | None, channel: str, what: str, problems: list[str]) -> bool:
    if trained is not None and trained != channel:
        problems.append(f"{what} was trained for channel {trained!r}, this analysis has channel {channel!r}")
        return False
    return True


def _check_files(paths, message: str, problems: list[str]) -> None:
    for path in paths:
        if not Path(path).is_file():
            problems.append(f"{message}: {path}")


def _model_values(settings: dict, what: str, where: str, problems: list[str]) -> dict | None:
    """conv_type, gnn_mode, unsculpting_mode, graph_type, bdt_preselection_cut from manifest-style settings.
    Checks that GNN inference can reproduce the graphs. None (and a problem) if anything is wrong."""
    from tau3mu_gnn.gnn_training.gnn_inputs import INFERENCE_GRAPH
    from tau3mu_gnn.pipelines.common.config import _unsculpting_mode
    try:
        graphs, training = settings["graphs"], settings["gnn_training"]
        graph_type = "skimmed" + graph_suffix(graphs["use_displacement"], graphs["cone_size"],
                                              graphs["connection"], k=graphs["k"], with_unsculpting=False)
        mode = graphs["unsculpting_mode"]
        values = dict(conv_type=training["conv_type"], gnn_mode=training["gnn_mode"],
                      unsculpting_mode=None if mode is None else _unsculpting_mode(mode),
                      graph_type=graph_type, bdt_preselection_cut=float(graphs["bdt_preselection_cut"]))
    except (KeyError, TypeError) as e:
        problems.append(f"{where} lack an entry: {e!r}")
        return None
    except ValueError as e:
        problems.append(f"{what}: graphs.unsculpting_mode {e}")
        return None
    ok = True
    for key, value in INFERENCE_GRAPH.items():
        if graphs.get(key) != value:
            problems.append(f"{what} was trained on graphs.{key} = {graphs.get(key)!r}, but GNN inference "
                            f"builds graphs with {key} = {value!r}; this model cannot be scored")
            ok = False
    return values if ok else None
