import pytest

from tau3mu_gnn.graph.naming import graph_suffix


@pytest.mark.parametrize("kw, expected", [
    (dict(use_displacement=True, cone_size=0.5, connection="knn", k=10, unsculpting_mode="delete_muons"),
     "_dzd0_cone_size_0.5_knn_10_delete_muons"),
    (dict(use_displacement=False, cone_size=0.5, connection="r_threshold", delta_r=0.3, unsculpting_mode=None),
     "_cone_size_0.5_r_threshold_0.3_testing"),
    (dict(use_displacement=True, cone_size=1.0, connection="fully", unsculpting_mode="permutation"),
     "_dzd0_cone_size_1.0_fully_permutation"),
])
def test_graph_suffix_as_old_graph_creation(kw, expected):
    assert graph_suffix(**kw) == expected


def test_without_unsculpting_part():
    assert graph_suffix(True, 0.5, "knn", k=10, with_unsculpting=False) == "_dzd0_cone_size_0.5_knn_10"


def test_training_names_use_it(load_training, training_user):
    from tau3mu_gnn.pipelines.training.stages.graphs import model_type
    cfg = load_training(training_user)
    assert cfg.graphs.graph_type == "skimmed" + graph_suffix(True, 0.5, "knn", k=10, with_unsculpting=False)
    assert model_type(cfg) == "skimmed" + graph_suffix(True, 0.5, "knn", k=10, unsculpting_mode="delete_muons")
