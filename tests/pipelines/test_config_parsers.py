from types import MappingProxyType

import pytest
import yaml

from tau3mu_gnn.pipelines.common import config as c


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


@pytest.mark.parametrize("text", ["1e5", "1.0e5", "1.9E5", ".5e3", "1e-5"])
def test_number_hint_is_valid_yaml_float(text):
    # PyYAML needs a dot in the mantissa and a signed exponent; the hint must satisfy both.
    assert isinstance(yaml.safe_load(f"x: {text}")["x"], str)
    with pytest.raises(ValueError, match="write it as") as e:
        c._number(text)
    suggested = str(e.value).split("write it as ")[1].rstrip(")")
    assert yaml.safe_load(f"x: {suggested}")["x"] == float(text)


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


def test_number_list_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        c._number_list([3.9, 4.5, 3.9])
    with pytest.raises(ValueError, match="duplicate"):
        c._number_list([4, 4.0])          # same cut, same cut_4.0/ folder


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


@pytest.mark.parametrize("bad", [6e5, 600000.0, 600000.5, "600k", 0, -3, True, None])
def test_positive_int_rejects(bad):
    with pytest.raises(ValueError):
        c._positive_int(bad)


def test_positive_int():
    assert c._positive_int(600000) == 600000


def test_increasing_number_list():
    assert c._increasing_number_list([0, 10, 15]) == (0.0, 10.0, 15.0)
    for bad in ([5], [0, 10, 10], [10, 0], "0,10"):
        with pytest.raises(ValueError):
            c._increasing_number_list(bad)


def test_linspace():
    assert c._linspace([0, 100, 101]) == (0.0, 100.0, 101)
    for bad in ([0, 100], [100, 0, 101], [0, 100, 1], [0, 100, 10.5]):
        with pytest.raises(ValueError):
            c._linspace(bad)


def test_windows():
    ok = {"train": [0.0, 0.6], "validation": [0.6, 0.7], "test": [0.7, 1.0]}
    assert c._windows(ok)["validation"] == (0.6, 0.7)
    for bad in ({**ok, "validation": [0.5, 0.7]},          # overlap
                {**ok, "validation": [0.65, 0.7]},         # gap
                {**ok, "test": [0.7, 1.1]},                # outside [0, 1]
                {"train": [0.0, 0.6], "test": [0.6, 1.0]}, # missing key
                {**ok, "extra": [0, 1]}):                  # unknown key
        with pytest.raises(ValueError):
            c._windows(bad)


def test_slurm_jobs_parser_optional_account_and_gres():
    parse = c._slurm_jobs_parser(("a", "b"))
    jobs = parse({"a": {"partition": "gpu", "mem": "128G", "time": "3-00:00:00", "cpus": 5,
                        "account": "gpu_gres", "gres": "gpu:1"},
                  "b": {"partition": "long", "mem": "256G", "time": "2-00:00:00", "cpus": 1}})
    assert (jobs["a"].account, jobs["a"].gres) == ("gpu_gres", "gpu:1")
    assert (jobs["b"].account, jobs["b"].gres) == (None, None)
    with pytest.raises(ValueError, match="missing job 'b'"):
        parse({"a": {"partition": "x", "mem": "1G", "time": "01:00:00", "cpus": 1}})
    with pytest.raises(ValueError, match="account must be a non-empty string"):
        parse({"a": {"partition": "x", "mem": "1G", "time": "01:00:00", "cpus": 1, "account": ""},
               "b": {"partition": "x", "mem": "1G", "time": "01:00:00", "cpus": 1}})
