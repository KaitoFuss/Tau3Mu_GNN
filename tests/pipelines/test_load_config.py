import dataclasses

import pytest

from conftest import TOY_MODEL
from tau3mu_gnn.pipelines.common.config import ConfigError
from tau3mu_gnn.pipelines.analysis.config import load_config


def problems_of(excinfo):
    return "\n".join(excinfo.value.problems)


def test_valid_config_loads(load, user, fake_repo):
    cfg = load(user)
    assert cfg.analysis_name == "toy_run"
    assert cfg.channel == "W"
    assert cfg.paths.output_base == "/tmp/toy_out"
    assert cfg.output_dir == "/tmp/toy_out/toy_run"
    assert cfg.inputs.signal == ("/data/sig.root",)
    assert cfg.inputs.background == ("/data/bkg_*.root",)
    assert cfg.inputs.already_resonance_skimmed is True
    assert cfg.limit.signal_weight == 1.9e-5
    assert cfg.limit.fit_range == (1.6, 2.0)
    assert cfg.gnn.model_name == TOY_MODEL
    assert cfg.gnn.model_dir == fake_repo / "models/W/gnn"
    assert cfg.bdt.classifier == fake_repo / "models/W/bdt/toy.pck"
    assert cfg.defaults_file == fake_repo / "configs/analysis_defaults_W.yaml"


def test_defaults_applied(load, user):
    cfg = load(user)
    assert cfg.inputs.tree_name == "Events"
    assert cfg.limit.gnn_score_cut == (3.5,)
    assert dict(cfg.branch_map) == {}


def test_score_cut_scalar_and_list(load, user):
    user["limit"]["gnn_score_cut"] = 3.75
    assert load(user).limit.gnn_score_cut == (3.75,)
    user["limit"]["gnn_score_cut"] = [3.5, 3.9]
    assert load(user).limit.gnn_score_cut == (3.5, 3.9)


def test_benchmark_cuts_required(load, user):
    assert load(user).benchmark.bdt_score_cut == (0.98, 0.99)
    user["benchmark"]["bdt_score_cut"] = 0.95
    assert load(user).benchmark.bdt_score_cut == (0.95,)
    del user["benchmark"]
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "missing required key 'benchmark.bdt_score_cut'" in problems_of(e)


@pytest.mark.parametrize("bad", [[0.0, 0.98], [1.0], [0.98, 0.98], ["x"]])
def test_benchmark_cut_must_be_in_unit_interval(load, user, bad):
    user["benchmark"]["bdt_score_cut"] = bad
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "benchmark.bdt_score_cut" in problems_of(e)


@pytest.mark.parametrize("bad", [-1, -1.0, -2.5, [0.0, -1.0]])
def test_gnn_cut_must_be_above_not_scored(load, user, bad):
    """GNN_score = -1 marks the events the GNN did not score (below the BDT preselection): a cut at or below
    -1 would let them into the limit."""
    user["limit"]["gnn_score_cut"] = bad
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "limit.gnn_score_cut" in problems_of(e) and "must be > -1.0" in problems_of(e)


def test_gnn_cut_negative_above_not_scored_ok(load, user):
    user["limit"]["gnn_score_cut"] = [-0.99, 0.0]
    assert load(user).limit.gnn_score_cut == (-0.99, 0.0)


def test_config_is_read_only(load, user):
    user["branch_map"] = {"PFCands_pt": "PFCand_pt"}
    cfg = load(user)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.limit.signal_weight = 1.0
    with pytest.raises(TypeError):
        cfg.branch_map["x"] = "y"


def test_missing_required_key(load, user):
    del user["limit"]["signal_weight"]
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "missing required key 'limit.signal_weight'" in problems_of(e)


def test_unknown_key_with_hint(load, user):
    user["limit"]["signal_weigth"] = user["limit"].pop("signal_weight")
    with pytest.raises(ConfigError) as e:
        load(user)
    text = problems_of(e)
    assert "unknown key 'limit.signal_weigth'" in text
    assert "did you mean 'limit.signal_weight'" in text


def test_frozen_key_in_user_file(load, user):
    user["gnn"] = {"model_name": "other_model"}
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "'gnn.model_name' is a frozen analysis setting" in problems_of(e)


def test_all_problems_reported_at_once(load, user):
    del user["analysis_name"]
    user["limit"]["signal_weight"] = 0
    user["inputs"]["already_resonance_skimmed"] = "yes please"
    with pytest.raises(ConfigError) as e:
        load(user)
    assert len(e.value.problems) == 3


def test_unsupported_channel_stops_before_defaults(load, user):
    user["channel"] = "XX"
    with pytest.raises(ConfigError) as e:
        load(user)
    text = problems_of(e)
    assert "must be one of ['W', 'HF']" in text
    assert "analysis_defaults_XX" not in text


def test_empty_branch_map_header_is_empty_map(load):
    raw = """
analysis_name: toy_run
channel: W
paths:
  output_base: /tmp/toy_out
  log_dir: /tmp/toy_logs
inputs:
  signal: [/data/sig.root]
  background: [/data/bkg.root]
  already_resonance_skimmed: true
branch_map:
limit:
  signal_weight: 1.9e-5
benchmark:
  bdt_score_cut: 0.98
"""
    assert dict(load(raw_text=raw).branch_map) == {}


def test_empty_section_header_reports_missing_keys(load):
    raw = """
analysis_name: toy_run
channel: W
paths:
inputs:
  signal: [/data/sig.root]
  background: [/data/bkg.root]
  already_resonance_skimmed: true
limit:
  signal_weight: 1.9e-5
"""
    with pytest.raises(ConfigError) as e:
        load(raw_text=raw)
    assert "missing required key 'paths.output_base'" in problems_of(e)
    assert "unknown key" not in problems_of(e)


def test_duplicate_key_is_error(load):
    raw = """
analysis_name: toy_run
channel: W
paths:
  output_base: /tmp/toy_out
inputs:
  signal: [/data/sig.root]
  background: [/data/bkg.root]
  already_resonance_skimmed: true
limit:
  signal_weight: 1.9e-5
limit:
  signal_weight: 2.0
"""
    with pytest.raises(ConfigError) as e:
        load(raw_text=raw)
    assert "duplicate key 'limit'" in problems_of(e)


def test_duplicate_nested_key_is_error(load):
    raw = """
analysis_name: toy_run
channel: W
paths:
  output_base: /tmp/toy_out
inputs:
  signal: [/data/sig.root]
  background: [/data/bkg.root]
  already_resonance_skimmed: true
limit:
  signal_weight: 1.9e-5
  signal_weight: 2.0
"""
    with pytest.raises(ConfigError) as e:
        load(raw_text=raw)
    assert "duplicate key 'signal_weight'" in problems_of(e)


def test_missing_user_file(fake_repo):
    with pytest.raises(ConfigError) as e:
        load_config(fake_repo / "nope.yaml", defaults_dir=fake_repo / "configs")
    assert "cannot read config file" in problems_of(e)


def test_invalid_yaml(load):
    with pytest.raises(ConfigError) as e:
        load(raw_text="analysis_name: [unclosed\n")
    assert "not valid YAML" in problems_of(e)


def test_top_level_must_be_mapping(load):
    with pytest.raises(ConfigError) as e:
        load(raw_text="- a\n- b\n")
    assert "top level must be a mapping" in problems_of(e)


def test_missing_frozen_model_file(load, user, fake_repo):
    (fake_repo / f"models/W/gnn/{TOY_MODEL}/Normalization.pkl").unlink()
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "frozen model file not found" in problems_of(e)
    assert "Normalization.pkl" in problems_of(e)


def test_channel_not_allowed_in_defaults(load, user, fake_repo):
    # The channel is chosen once, in the user file; the defaults file name carries it.
    path = fake_repo / "configs/analysis_defaults_W.yaml"
    path.write_text("channel: W\n" + path.read_text())
    with pytest.raises(ConfigError) as e:
        load(user)
    assert "unknown key 'channel'" in problems_of(e)


@pytest.mark.parametrize("base, expected", [
    ("/tmp/toy_out/", "/tmp/toy_out/toy_run"),
    ("root://t3dcachedb03.psi.ch:1094//pnfs/x", "root://t3dcachedb03.psi.ch:1094//pnfs/x/toy_run"),
])
def test_output_dir_from_base_and_analysis_name(load, user, base, expected):
    user["paths"]["output_base"] = base
    assert load(user).output_dir == expected


def test_log_dir_and_slurm(load, user):
    cfg = load(user)
    assert cfg.paths.log_dir == "/tmp/toy_logs"
    assert cfg.slurm.account == "t3"
    assert set(cfg.slurm.jobs) == {"skim", "bdt", "gnn_signal", "gnn_background", "eval", "limit", "summary",
                                   "benchmark_limit", "benchmark_summary"}
    job = cfg.slurm.jobs["limit"]
    assert (job.partition, job.mem, job.time, job.cpus) == ("standard", "4G", "0-01:00:00", 1)


def test_log_dir_required(load, user):
    from tau3mu_gnn.pipelines.common.config import ConfigError

    del user["paths"]["log_dir"]
    with pytest.raises(ConfigError, match="paths.log_dir"):
        load(user)


@pytest.mark.parametrize("bad", ["root://t3dcachedb03.psi.ch:1094//pnfs/x", "/pnfs/psi.ch/x"])
def test_log_dir_must_be_local(load, user, bad):
    from tau3mu_gnn.pipelines.common.config import ConfigError

    user["paths"]["log_dir"] = bad
    with pytest.raises(ConfigError, match="log_dir"):
        load(user)


def test_slurm_in_user_file_is_frozen(load, user):
    from tau3mu_gnn.pipelines.common.config import ConfigError

    user["slurm"] = {"account": "t3"}
    with pytest.raises(ConfigError, match="frozen"):
        load(user)


def _defaults_with(fake_repo, change):
    import copy

    import yaml
    from conftest import TOY_DEFAULTS

    d = copy.deepcopy(TOY_DEFAULTS)
    change(d["slurm"]["jobs"])
    (fake_repo / "configs" / "analysis_defaults_W.yaml").write_text(yaml.safe_dump(d))


@pytest.mark.parametrize("change, match", [
    (lambda j: j.pop("eval"), "missing job 'eval'"),
    (lambda j: j.update(gnn_bkg=dict(j["bdt"])), "unknown job 'gnn_bkg'"),
    (lambda j: j["bdt"].update(memory="4G"), "unknown key 'memory'"),
    (lambda j: j["bdt"].pop("time"), "missing key 'time'"),
    (lambda j: j["bdt"].update(mem="64 GB"), "mem"),
    (lambda j: j["bdt"].update(time="7 days"), "time"),
    (lambda j: j["bdt"].update(cpus=0), "cpus"),
    (lambda j: j["bdt"].update(cpus=True), "cpus"),
    (lambda j: j.update(eval=None), "must be a mapping"),
])
def test_slurm_jobs_validation(load, user, fake_repo, change, match):
    from tau3mu_gnn.pipelines.common.config import ConfigError

    _defaults_with(fake_repo, change)
    with pytest.raises(ConfigError, match=match):
        load(user)


@pytest.mark.parametrize("bad", [
    "out/tau3mu", "root://host/pnfs/x", "roots://host//pnfs/x", "root:/host//pnfs/x"])
def test_output_base_must_be_absolute_or_valid_url(load, user, bad):
    user["paths"]["output_base"] = bad
    with pytest.raises(ConfigError, match="paths.output_base"):
        load(user)


@pytest.mark.parametrize("good", ["root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/x", "/work/x"])
def test_output_base_valid_forms_load(load, user, good):
    user["paths"]["output_base"] = good
    assert load(user).paths.output_base == good


@pytest.mark.parametrize("key", ["signal", "background"])
@pytest.mark.parametrize("bad", ["data/sig.root", "root://host/pnfs/x.root"])
def test_input_entries_must_be_absolute_or_valid_url(load, user, key, bad):
    user["inputs"][key] = ["/data/ok.root", bad]
    with pytest.raises(ConfigError, match=f"inputs.{key}"):
        load(user)


def test_input_valid_url_loads(load, user):
    url = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/x/*.root"
    user["inputs"]["signal"] = [url]
    assert load(user).inputs.signal == (url,)


def test_log_dir_must_be_absolute(load, user):
    user["paths"]["log_dir"] = "logs"
    with pytest.raises(ConfigError, match="paths.log_dir"):
        load(user)


def test_conda_env_optional(load, user):
    assert load(user).paths.conda_env is None
    user["paths"]["conda_env"] = "/work/u/miniforge3/envs/Tau3Mu"
    assert load(user).paths.conda_env == "/work/u/miniforge3/envs/Tau3Mu"


@pytest.mark.parametrize("bad", ["envs/Tau3Mu", "/pnfs/psi.ch/x"])
def test_conda_env_must_be_absolute_local(load, user, bad):
    user["paths"]["conda_env"] = bad
    with pytest.raises(ConfigError, match="paths.conda_env"):
        load(user)
