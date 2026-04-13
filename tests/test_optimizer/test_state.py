"""Tests for bayesopt_human.optimizer.state module."""

import json

import pytest

from bayesopt_human.optimizer.state import (
    ModelChoiceRecord,
    OptimizationState,
    RoundRecord,
    load_state,
    record_round,
    save_state,
    state_filepath,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model_record(**overrides):
    defaults = dict(
        output_transform="none",
        ard=True,
        warping=False,
        loo_mae=0.05,
        loo_lpd=1.5,
        calibration_var=0.8,
        n_obs_when_chosen=15,
    )
    defaults.update(overrides)
    return ModelChoiceRecord(**defaults)


def _make_state(**overrides):
    defaults = dict(
        schema_version=1,
        config_name="test_fn",
        last_updated="2026-01-01T00:00:00",
        model_choice=_make_model_record(),
        rounds=[],
    )
    defaults.update(overrides)
    return OptimizationState(**defaults)


# ---------------------------------------------------------------------------
# state_filepath
# ---------------------------------------------------------------------------

class TestStateFilepath:
    def test_basic(self, tmp_path):
        fp = state_filepath(str(tmp_path), "branin")
        assert fp.name == "branin_state.json"
        assert fp.parent == tmp_path


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        state = _make_state(config_name="fn_1")
        save_state(state, str(tmp_path))

        loaded = load_state(str(tmp_path), "fn_1")
        assert loaded is not None
        assert loaded.schema_version == 1
        assert loaded.config_name == "fn_1"
        assert loaded.model_choice.output_transform == "none"
        assert loaded.model_choice.ard is True
        assert loaded.model_choice.warping is False
        assert loaded.model_choice.loo_mae == pytest.approx(0.05)
        assert loaded.model_choice.loo_lpd == pytest.approx(1.5)
        assert loaded.model_choice.calibration_var == pytest.approx(0.8)
        assert loaded.model_choice.n_obs_when_chosen == 15

    def test_roundtrip_with_rounds(self, tmp_path):
        state = _make_state(config_name="fn_2")
        record_round(
            state, n_obs=15,
            model_choice=_make_model_record(),
            candidate_source="EI",
            candidate_category="balanced",
            candidate_confidence="high",
            candidate_posterior_mean=1.5,
            candidate_posterior_std=0.2,
            candidate_expected_improvement=0.3,
            was_custom=False,
        )
        save_state(state, str(tmp_path))

        loaded = load_state(str(tmp_path), "fn_2")
        assert loaded is not None
        assert len(loaded.rounds) == 1
        r = loaded.rounds[0]
        assert r.round_number == 1
        assert r.candidate_source == "EI"
        assert r.candidate_category == "balanced"
        assert r.candidate_confidence == "high"
        assert r.candidate_was_custom is False

    def test_load_missing_file_returns_none(self, tmp_path):
        result = load_state(str(tmp_path), "nonexistent")
        assert result is None

    def test_load_corrupt_json_returns_none(self, tmp_path):
        fp = tmp_path / "bad_state.json"
        fp.write_text("this is not json!", encoding="utf-8")
        # load_state uses config_name to build filename
        bad_fp = tmp_path / "bad_state.json"
        bad_fp.rename(tmp_path / "corrupt_state.json")
        result = load_state(str(tmp_path), "corrupt")
        assert result is None

    def test_load_wrong_schema_version_returns_none(self, tmp_path):
        fp = tmp_path / "versioned_state.json"
        fp.write_text(json.dumps({
            "schema_version": 99,
            "config_name": "versioned",
        }), encoding="utf-8")
        result = load_state(str(tmp_path), "versioned")
        assert result is None

    def test_load_no_model_choice(self, tmp_path):
        state = _make_state(config_name="empty", model_choice=None)
        save_state(state, str(tmp_path))
        loaded = load_state(str(tmp_path), "empty")
        assert loaded is not None
        assert loaded.model_choice is None


# ---------------------------------------------------------------------------
# record_round
# ---------------------------------------------------------------------------

class TestRecordRound:
    def test_appends_rounds(self, tmp_path):
        state = _make_state()
        mr = _make_model_record()

        record_round(
            state, n_obs=15, model_choice=mr,
            candidate_source="EI", candidate_category="balanced",
            candidate_confidence="high",
            candidate_posterior_mean=1.0, candidate_posterior_std=0.1,
            candidate_expected_improvement=0.5, was_custom=False,
        )
        assert len(state.rounds) == 1
        assert state.rounds[0].round_number == 1

        record_round(
            state, n_obs=16, model_choice=mr,
            candidate_source="UCB", candidate_category="exploration",
            candidate_confidence="medium",
            candidate_posterior_mean=2.0, candidate_posterior_std=0.3,
            candidate_expected_improvement=0.1, was_custom=False,
        )
        assert len(state.rounds) == 2
        assert state.rounds[1].round_number == 2
        assert state.rounds[1].candidate_source == "UCB"

    def test_updates_model_choice(self):
        state = _make_state(model_choice=None)
        mr = _make_model_record(output_transform="log", ard=False)

        record_round(
            state, n_obs=10, model_choice=mr,
            candidate_source="EI", candidate_category="balanced",
            candidate_confidence="high",
            candidate_posterior_mean=1.0, candidate_posterior_std=0.1,
            candidate_expected_improvement=0.5, was_custom=False,
        )

        assert state.model_choice is not None
        assert state.model_choice.output_transform == "log"
        assert state.model_choice.ard is False

    def test_updates_last_updated(self):
        state = _make_state(last_updated="old")
        mr = _make_model_record()

        record_round(
            state, n_obs=10, model_choice=mr,
            candidate_source="EI", candidate_category="balanced",
            candidate_confidence="high",
            candidate_posterior_mean=1.0, candidate_posterior_std=0.1,
            candidate_expected_improvement=0.5, was_custom=False,
        )

        assert state.last_updated != "old"
        assert "T" in state.last_updated  # ISO format

    def test_round_record_fields_roundtrip(self, tmp_path):
        """All RoundRecord fields survive JSON serialization."""
        state = _make_state(config_name="rt")
        record_round(
            state, n_obs=20,
            model_choice=_make_model_record(warping=True),
            candidate_source="Max GP Mean",
            candidate_category="exploitation",
            candidate_confidence="low",
            candidate_posterior_mean=5.5,
            candidate_posterior_std=0.05,
            candidate_expected_improvement=-2.0,
            was_custom=False,
        )
        save_state(state, str(tmp_path))

        loaded = load_state(str(tmp_path), "rt")
        r = loaded.rounds[0]
        assert r.model_output_transform == "none"  # from _make_model_record default
        assert r.model_warping is True
        assert r.candidate_source == "Max GP Mean"
        assert r.candidate_category == "exploitation"
        assert r.candidate_confidence == "low"
        assert r.candidate_posterior_mean == pytest.approx(5.5)
        assert r.candidate_posterior_std == pytest.approx(0.05)
        assert r.candidate_expected_improvement == pytest.approx(-2.0)
        assert r.candidate_was_custom is False
