"""Candidate generation and selection step."""

from __future__ import annotations

from typing import Any

import panel as pn

from bayesopt_human.reporting.candidates import (
    INDIVIDUAL_REPORTS as _INDIVIDUAL_REPORTS,
    SUMMARY_REPORTS as _SUMMARY_REPORTS,
)
from bayesopt_human.ui.capture import capture_output
from bayesopt_human.ui.state import AppState
from bayesopt_human.ui.steps.base import BaseStep

SUMMARY_REPORTS = [r.key for r in _SUMMARY_REPORTS]
INDIVIDUAL_REPORTS = [r.key for r in _INDIVIDUAL_REPORTS]


class CandidateStep(BaseStep):
    title = "Candidates"
    description = "Generate candidate evaluation points and select the next experiment."

    def __init__(self, step: Any, state: AppState) -> None:
        super().__init__(step, state)
        self._choice_widget: pn.widgets.Select | None = None
        self._save_widget: pn.widgets.TextInput | None = None
        self._individual_pane = pn.Column(sizing_mode="stretch_width")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        if self.state.candidate_result is not None:
            return
        data = self.state.data_payload
        fitted_gp = self.state.fitted_gp
        if data is None or fitted_gp is None:
            self._action_pane.objects = [
                pn.pane.Alert(
                    "Missing data or model. Go back to complete previous steps.",
                    alert_type="warning",
                )
            ]
            return
        self._action_pane.objects = [
            pn.pane.HTML(
                "<em>Generating candidates — this may take a few seconds...</em>"
            )
        ]
        self._run_blocking(
            fn=lambda: self.step.get_candidates(data, fitted_gp),
            callback=self._on_candidates_generated,
        )

    def on_leave_forward(self) -> None:
        choice = self._choice_widget.value if self._choice_widget else 1
        save_to = self._save_widget.value if self._save_widget else None
        save_to = save_to.strip() if save_to else None
        with capture_output():
            coords = self.step.choose_candidate(
                self.state.candidate_result,
                choice=choice,
                save_to=save_to or None,
            )
        self.state.selected_coordinates = coords
        self.state.candidate_choice = choice

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_candidates_generated(self, result: Any, captured: Any) -> None:
        self.state.candidate_result = result

        # Top recommendation
        top_rec_html = ""
        if result.recommendations:
            top = result.recommendations[0]
            top_rec_html = (
                f"<div style='padding:8px; background:#e8f4fd; "
                f"border-left:4px solid #4a90d9; margin:8px 0; border-radius:4px'>"
                f"<strong>Top recommendation:</strong> {top.source}<br>"
                f"<strong>Rationale:</strong> {top.rationale}<br>"
                f"<strong>Confidence:</strong> {top.confidence}"
                f"</div>"
            )

        # Build dropdown options. ★ marks the best candidate within its arm.
        options = {}
        for rec in result.recommendations:
            label = f"[Rank {rec.priority}] {rec.source}"
            if rec.is_arm_winner:
                label += " ★"
            if rec.priority == 1:
                label += " *"
            options[label] = rec.priority

        self._choice_widget = pn.widgets.Select(
            name="Select candidate",
            options=options,
            value=1,
            sizing_mode="stretch_width",
        )
        self._choice_widget.param.watch(self._on_choice_changed, "value")

        # Save-to widget
        self._save_widget = pn.widgets.TextInput(
            name="Save coordinates to (optional)",
            placeholder="e.g. next_point.csv",
            sizing_mode="stretch_width",
        )

        # Action panel: captured output + recommendation
        rendered = self.render_captured(captured)
        self._action_pane.objects = [
            o for o in [
                *rendered,
                pn.pane.HTML(top_rec_html, sizing_mode="stretch_width") if top_rec_html else None,
            ] if o is not None
        ]

        # Summary inner tabs — choice-independent.
        summary_inner = self._build_group_tabs(
            result, self.step.report_candidates, SUMMARY_REPORTS, choice=None,
        )
        # Individual inner tabs — re-built on choice change.
        self._rebuild_individual_pane(choice=1)

        self._outer_tabs = pn.Tabs(
            ("Summary", summary_inner),
            ("Individual", self._individual_pane),
            sizing_mode="stretch_width",
            dynamic=True,
        )
        self._report_pane.objects = [
            self._choice_widget,
            self._outer_tabs,
            self._save_widget,
        ]

    def _on_choice_changed(self, event: Any) -> None:
        if self.state.candidate_result is not None:
            self.clear_report_cache()
            self._rebuild_individual_pane(choice=event.new)

    def _rebuild_individual_pane(self, choice: int) -> None:
        individual_inner = self._build_group_tabs(
            self.state.candidate_result,
            self.step.report_candidates,
            INDIVIDUAL_REPORTS,
            choice=choice,
        )
        self._individual_pane.objects = [individual_inner]
