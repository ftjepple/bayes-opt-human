"""Data loading and selection step."""

from __future__ import annotations

import os
from typing import Any

import panel as pn

from bayesopt_human.reporting.data import (
    INDIVIDUAL_REPORTS as _INDIVIDUAL_REPORTS,
    SUMMARY_REPORTS as _SUMMARY_REPORTS,
)
from bayesopt_human.ui.capture import capture_output
from bayesopt_human.ui.state import AppState
from bayesopt_human.ui.steps.base import BaseStep

SUMMARY_REPORTS = [r.key for r in _SUMMARY_REPORTS]
INDIVIDUAL_REPORTS = [r.key for r in _INDIVIDUAL_REPORTS]


class DataStep(BaseStep):
    title = "Data"
    description = "Load observation files, inspect diagnostics, and select a dataset."

    def __init__(self, step: Any, state: AppState, data_dir: str) -> None:
        super().__init__(step, state)
        self.data_dir = data_dir
        self._choice_widget: pn.widgets.Select | None = None
        self._individual_pane = pn.Column(sizing_mode="stretch_width")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        if self.state.data_result is not None:
            return  # already loaded, keep cached view
        self._run_blocking(
            fn=lambda: self.step.get_data(self.data_dir),
            callback=self._on_data_loaded,
        )

    def on_leave_forward(self) -> None:
        """Called when user clicks Continue. Runs choose_data()."""
        choice = self._choice_widget.value if self._choice_widget else 0
        with capture_output():
            payload = self.step.choose_data(self.state.data_result, choice=choice)
        self.state.data_payload = payload
        self.state.data_choice = choice

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_data_loaded(self, result: Any, captured: Any) -> None:
        self.state.data_result = result
        entries = result.entries

        # Build selection widget
        options = {
            f"[{i}] {os.path.basename(e.filepath)} — {e.n_obs} obs, {e.n_dim} dims": i
            for i, e in enumerate(entries)
        }
        self._choice_widget = pn.widgets.Select(
            name="Select dataset",
            options=options,
            value=0,
            sizing_mode="stretch_width",
        )
        self._choice_widget.param.watch(self._on_choice_changed, "value")

        # Action panel: captured output only
        self._action_pane.objects = self.render_captured(captured)

        # Summary inner tabs — choice-independent.
        summary_inner = self._build_group_tabs(
            result, self.step.report_data, SUMMARY_REPORTS, choice=None,
        )
        # Individual inner tabs — re-built on choice change.
        self._rebuild_individual_pane(choice=0)

        self._outer_tabs = pn.Tabs(
            ("Summary", summary_inner),
            ("Individual", self._individual_pane),
            sizing_mode="stretch_width",
            dynamic=True,
        )
        self._report_pane.objects = [
            self._choice_widget,
            self._outer_tabs,
        ]

    def _on_choice_changed(self, event: Any) -> None:
        if self.state.data_result is not None:
            self.clear_report_cache()
            self._rebuild_individual_pane(choice=event.new)

    def _rebuild_individual_pane(self, choice: int) -> None:
        individual_inner = self._build_group_tabs(
            self.state.data_result,
            self.step.report_data,
            INDIVIDUAL_REPORTS,
            choice=choice,
        )
        self._individual_pane.objects = [individual_inner]
