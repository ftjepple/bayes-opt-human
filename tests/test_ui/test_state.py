"""Tests for AppState invalidation logic."""

from bayesopt_human.ui.state import (
    STEP_CANDIDATE,
    STEP_DATA,
    STEP_DONE,
    STEP_MODEL,
    AppState,
)


class TestAppState:

    def test_initial_state(self):
        state = AppState()
        assert state.current_step == STEP_DATA
        assert state.data_result is None
        assert state.model_result is None
        assert state.candidate_result is None

    def test_invalidate_from_model_clears_downstream(self):
        state = AppState()
        state.data_result = "data"
        state.data_payload = "payload"
        state.model_result = "models"
        state.fitted_gp = "gp"
        state.candidate_result = "candidates"

        state.invalidate_from(STEP_MODEL)

        assert state.data_result == "data"
        assert state.data_payload == "payload"
        assert state.model_result is None
        assert state.fitted_gp is None
        assert state.candidate_result is None

    def test_invalidate_from_candidate_clears_only_candidates(self):
        state = AppState()
        state.data_result = "data"
        state.model_result = "models"
        state.fitted_gp = "gp"
        state.candidate_result = "candidates"

        state.invalidate_from(STEP_CANDIDATE)

        assert state.data_result == "data"
        assert state.model_result == "models"
        assert state.fitted_gp == "gp"
        assert state.candidate_result is None

    def test_invalidate_preserves_upstream(self):
        state = AppState()
        state.data_result = "data"
        state.data_payload = "payload"
        state.data_choice = 3

        state.invalidate_from(STEP_MODEL)

        assert state.data_result == "data"
        assert state.data_payload == "payload"
        assert state.data_choice == 3

    def test_invalidate_resets_choice_defaults(self):
        state = AppState()
        state.model_choice = 5
        state.candidate_choice = 3

        state.invalidate_from(STEP_MODEL)

        assert state.model_choice == 0
        assert state.candidate_choice == 1

    def test_invalidate_from_done_is_noop(self):
        state = AppState()
        state.model_result = "models"
        state.candidate_result = "candidates"

        state.invalidate_from(STEP_DONE)

        assert state.model_result == "models"
        assert state.candidate_result == "candidates"

    def test_reset_clears_everything(self):
        state = AppState()
        state.current_step = STEP_DONE
        state.data_result = "data"
        state.data_payload = "payload"
        state.model_result = "models"
        state.fitted_gp = "gp"
        state.candidate_result = "candidates"

        state.reset()

        assert state.current_step == STEP_DATA
        assert state.data_result is None
        assert state.data_payload is None
        assert state.model_result is None
        assert state.candidate_result is None

    def test_invalidate_clears_selected_coordinates(self):
        state = AppState()
        state.selected_coordinates = [1.0, 2.0]

        state.invalidate_from(STEP_CANDIDATE)

        assert state.selected_coordinates is None
