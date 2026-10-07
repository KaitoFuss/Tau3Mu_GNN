from types import MappingProxyType

import pytest

from tau3mu_gnn.pipeline import config as c


def test_config_error_lists_all_problems():
    err = c.ConfigError(["a is bad", "b is bad"])
    assert err.problems == ["a is bad", "b is bad"]
    assert "a is bad" in str(err) and "b is bad" in str(err)


@pytest.mark.parametrize("bad", ["", "   ", None, 3])
def test_non_empty_str_rejects(bad):
    with pytest.raises(ValueError):
        c._non_empty_str(bad)


def test_name_rejects_slash_and_space():
    assert c._name("kaito_W-ref.v1") == "kaito_W-ref.v1"
    for bad in ["a/b", "a b"]:
        with pytest.raises(ValueError, match="letters, digits"):
            c._name(bad)


def test_channel_only_w():
    assert c._channel("W") == "W"
    with pytest.raises(ValueError, match=r"\['W'\]"):
        c._channel("HF")


@pytest.mark.parametrize("bad", [True, False, "1e-5", None, [1]])
def test_number_rejects_non_numbers(bad):
    with pytest.raises(ValueError, match="must be a number"):
        c._number(bad)


def test_number_hint_for_yaml_exponent_text():
    # PyYAML reads `1e-5` (no dot) as the string "1e-5"
    with pytest.raises(ValueError, match=r"write it as 1\.0e-5"):
        c._number("1e-5")


def test_number_accepts_int_and_float():
    assert c._number(3) == 3.0
    assert c._number(1.9e-5) == 1.9e-5


@pytest.mark.parametrize("bad", [0, -1.0])
def test_positive_number_rejects(bad):
    with pytest.raises(ValueError, match="> 0"):
        c._positive_number(bad)


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.1])
def test_open_unit_interval_rejects(bad):
    with pytest.raises(ValueError, match="between 0 and 1"):
        c._open_unit_interval(bad)


def test_bool_rejects_strings_and_numbers():
    assert c._bool(True) is True
    for bad in ["true", 1, None]:
        with pytest.raises(ValueError, match="true or false"):
            c._bool(bad)


def test_number_list_scalar_and_list():
    assert c._number_list(3.9) == (3.9,)
    assert c._number_list([3.5, 4]) == (3.5, 4.0)
    with pytest.raises(ValueError):
        c._number_list([])
    with pytest.raises(ValueError):
        c._number_list([3.5, "x"])


def test_range():
    assert c._range([1.62, 1.98]) == (1.62, 1.98)
    for bad in [[1.98, 1.62], [1.0], "1,2"]:
        with pytest.raises(ValueError):
            c._range(bad)


def test_str_map():
    m = c._str_map({"PFCands_pt": "PFCand_pt"})
    assert isinstance(m, MappingProxyType)
    assert dict(m) == {"PFCands_pt": "PFCand_pt"}
    assert dict(c._str_map(None)) == {}
    with pytest.raises(ValueError, match="PFCands_pt"):
        c._str_map({"PFCands_pt": 3})
    with pytest.raises(ValueError):
        c._str_map(["a", "b"])


def test_user_path_expands_home_and_vars(monkeypatch):
    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("TAU3MU_TEST_DIR", "/data/x")
    assert c._user_path("~/out") == "/home/tester/out"
    assert c._user_path("$TAU3MU_TEST_DIR/f.root") == "/data/x/f.root"


def test_user_path_unset_var_is_error(monkeypatch):
    monkeypatch.delenv("TAU3MU_UNSET_VAR", raising=False)
    with pytest.raises(ValueError, match="TAU3MU_UNSET_VAR"):
        c._user_path("$TAU3MU_UNSET_VAR/out")


def test_user_path_keeps_xrootd_url():
    url = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/x"
    assert c._user_path(url) == url


def test_user_path_list_single_string_and_list():
    assert c._user_path_list("/a.root") == ("/a.root",)
    assert c._user_path_list(["/a.root", "/b*.root"]) == ("/a.root", "/b*.root")
    with pytest.raises(ValueError):
        c._user_path_list([])
