import logging

import pytest

from tau3mu_gnn.pipeline.stages import resonance_skim


def test_skipped_when_inputs_already_resonance_skimmed(load, user, caplog):
    cfg = load(user)
    with caplog.at_level(logging.INFO):
        assert resonance_skim.run(cfg) is None
    assert "stage 0 skipped" in caplog.text


def test_not_implemented_when_inputs_are_raw(load, user):
    user["inputs"]["already_resonance_skimmed"] = False
    cfg = load(user)
    with pytest.raises(resonance_skim.StageNotImplementedError) as e:
        resonance_skim.run(cfg)
    assert "already_resonance_skimmed: true" in str(e.value)
