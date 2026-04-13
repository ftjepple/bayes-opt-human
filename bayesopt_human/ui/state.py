"""Central application state for the GUI wizard.

AppState holds all intermediate results and manages downstream invalidation
when the user navigates back to a previous step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STEP_DATA = 0
STEP_MODEL = 1
STEP_CANDIDATE = 2
STEP_DONE = 3


@dataclass
class AppState:
    """Mutable state container shared across all wizard steps."""

    current_step: int = STEP_DATA

    # Phase 1: Data
    data_result: object | None = None
    data_payload: object | None = None
    data_choice: int = 0

    # Phase 2: Model
    model_result: object | None = None
    fitted_gp: object | None = None
    model_choice: int = 0

    # Phase 3: Candidates
    candidate_result: object | None = None
    candidate_choice: int = 1

    # Final output
    selected_coordinates: object | None = None

    def invalidate_from(self, step: int) -> None:
        """Clear all results from ``step`` onwards.

        Call this when the user navigates back to ``step - 1`` so that
        downstream results are recomputed on the next forward pass.
        """
        if step <= STEP_MODEL:
            self.model_result = None
            self.fitted_gp = None
            self.model_choice = 0
        if step <= STEP_CANDIDATE:
            self.candidate_result = None
            self.candidate_choice = 1
            self.selected_coordinates = None

    def reset(self) -> None:
        """Reset all state for a new round."""
        self.current_step = STEP_DATA
        self.data_result = None
        self.data_payload = None
        self.data_choice = 0
        self.invalidate_from(STEP_MODEL)
