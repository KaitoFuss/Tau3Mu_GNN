import numpy as np
import pandas as pd
import pytest
import uproot

from conftest import write_tree
from tau3mu_gnn.utils.score_merge import (DuplicateEventError, check_unique_keys, scores_in_entry_order,
                                          write_with_score)


def _keys(run, lumi, event, dtype=np.uint32):
    return pd.DataFrame({"run": np.array(run, dtype=dtype), "luminosityBlock": np.array(lumi, dtype=dtype),
                         "event": np.array(event, dtype=np.uint64)})


def test_lumi_separates_events_with_same_run_and_event():
    entries = _keys([1, 1, 1], [5, 6, 5], [10, 10, 11])
    scores = _keys([1, 1, 1], [6, 5, 5], [10, 10, 11])
    np.testing.assert_array_equal(scores_in_entry_order(entries, scores, [0.6, 0.5, 0.7]), [0.5, 0.6, 0.7])


def test_missing_score_gets_default_and_nan_stays_nan():
    entries = _keys([1, 1, 1], [1, 1, 1], [1, 2, 3])
    scores = _keys([1, 1], [1, 1], [3, 1])
    out = scores_in_entry_order(entries, scores, [np.nan, 0.2])
    assert out[0] == 0.2 and out[1] == -1.0 and np.isnan(out[2])


def test_key_dtypes_may_differ():
    entries = _keys([1], [2], [3])
    scores = _keys([1], [2], [3], dtype=np.int64)
    np.testing.assert_array_equal(scores_in_entry_order(entries, scores, [0.9]), [0.9])


@pytest.mark.parametrize("dup_in", ["entries", "scores"])
def test_duplicate_keys_raise(dup_in):
    unique, dup = _keys([1, 1], [1, 1], [1, 2]), _keys([1, 1], [1, 1], [1, 1])
    entries, scores = (dup, unique) if dup_in == "entries" else (unique, dup)
    with pytest.raises(DuplicateEventError, match="run, luminosityBlock, event"):
        scores_in_entry_order(entries, scores, [0.1, 0.2])


def test_check_unique_keys_reads_file(tmp_path):
    ok = write_tree(tmp_path / "ok.root", {"run": np.ones(2, np.uint32), "luminosityBlock": np.array([1, 2], np.uint32),
                                           "event": np.array([7, 7], np.uint64)})
    check_unique_keys(ok, "Events")
    bad = write_tree(tmp_path / "bad.root", {"run": np.ones(2, np.uint32), "luminosityBlock": np.ones(2, np.uint32),
                                             "event": np.array([7, 7], np.uint64)})
    with pytest.raises(DuplicateEventError, match="bad.root"):
        check_unique_keys(bad, "Events")


def test_write_with_score_keeps_entries_and_branches(tmp_path):
    src = write_tree(tmp_path / "in.root", {"x": np.arange(3.0), "drop_me": np.zeros(3)}, tree="T")
    out = tmp_path / "sub" / "out.root"
    write_with_score(src, out, "T", "s", np.array([0.5, -1.0, np.nan]), drop=["drop_me", "not_there"])
    tree = uproot.open(out)["T"]
    assert set(tree.keys()) == {"x", "s"}
    assert tree["s"].typename == "float"
    np.testing.assert_array_equal(tree["x"].array(library="np"), [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(tree["s"].array(library="np"), np.float32([0.5, -1.0, np.nan]))


def test_write_with_score_wrong_length(tmp_path):
    src = write_tree(tmp_path / "in.root", {"x": np.arange(3.0)})
    with pytest.raises(ValueError, match="3 entries"):
        write_with_score(src, tmp_path / "out.root", "Events", "s", np.zeros(2))
