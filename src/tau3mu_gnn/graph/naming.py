"""Graph file names, one source for GraphCreation.py and the training pipeline."""
from __future__ import annotations


def graph_suffix(use_displacement: bool, cone_size: float, connection: str, k: int | None = None,
                 delta_r: float | None = None, unsculpting_mode: str | None = None,
                 with_unsculpting: bool = True) -> str:
    """e.g. '_dzd0_cone_size_0.5_knn_10_delete_muons' (no unsculpting mode: '_testing').
    with_unsculpting=False: without the last part ('_dzd0_cone_size_0.5_knn_10')."""
    suffix = f"{'_dzd0' if use_displacement else ''}_cone_size_{cone_size}_{connection}"
    if connection == "knn":
        suffix += f"_{k}"
    elif connection == "r_threshold":
        suffix += f"_{delta_r}"
    if with_unsculpting:
        suffix += f"_{unsculpting_mode}" if unsculpting_mode else "_testing"
    return suffix
