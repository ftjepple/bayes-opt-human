"""Model fitting and selection step."""

from __future__ import annotations

from typing import Any

import panel as pn

from bayesopt_human.reporting.models import (
    INDIVIDUAL_REPORTS as _INDIVIDUAL_REPORTS,
    SUMMARY_REPORTS as _SUMMARY_REPORTS,
)
from bayesopt_human.ui.capture import capture_output
from bayesopt_human.ui.state import AppState
from bayesopt_human.ui.steps.base import BaseStep

SUMMARY_REPORTS = [r.key for r in _SUMMARY_REPORTS]
INDIVIDUAL_REPORTS = [r.key for r in _INDIVIDUAL_REPORTS]


class ModelStep(BaseStep):
    title = "Models"
    description = "Fit GP model candidates, compare diagnostics, and select a model."

    def __init__(self, step: Any, state: AppState) -> None:
        super().__init__(step, state)
        self._choice_widget: pn.widgets.Select | None = None
        self._individual_pane = pn.Column(sizing_mode="stretch_width")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        if self.state.model_result is not None:
            return  # already computed
        data = self.state.data_payload
        if data is None:
            self._action_pane.objects = [
                pn.pane.Alert("No data selected. Go back and select data first.",
                              alert_type="warning")
            ]
            return
        self._action_pane.objects = [
            pn.pane.HTML("<em>Fitting models — this may take a few seconds...</em>")
        ]
        self._run_blocking(
            fn=lambda: self.step.get_models(data),
            callback=self._on_models_fitted,
        )

    def on_leave_forward(self) -> None:
        choice = self._choice_widget.value if self._choice_widget else 0
        with capture_output():
            fitted_gp = self.step.choose_model(
                self.state.model_result, choice=choice
            )
        self.state.fitted_gp = fitted_gp
        self.state.model_choice = choice

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_models_fitted(self, result: Any, captured: Any) -> None:
        self.state.model_result = result

        # Recommendation info
        rec_html = ""
        if result.recommendation is not None:
            rec = result.recommendation
            rec_html = (
                f"<div style='padding:8px; background:#e8f4fd; "
                f"border-left:4px solid #4a90d9; margin:8px 0; border-radius:4px'>"
                f"<strong>Recommendation ({rec.action.name}):</strong> "
                f"Model [{rec.recommended_idx}] &mdash; {rec.rationale}"
                f"</div>"
            )

        # Build dropdown options
        rec_idx = result.recommendation.recommended_idx if result.recommendation else 0
        options = {}
        for i, mc in enumerate(result.candidates):
            kernel = "WARP" if mc.warping else ("ARD" if mc.ard else "ISO")
            label = f"[{i}] {mc.output_transform} ({kernel})"
            if i == rec_idx and result.recommendation:
                label += " *"
            options[label] = i

        self._choice_widget = pn.widgets.Select(
            name="Select model",
            options=options,
            value=rec_idx,
            sizing_mode="stretch_width",
        )
        self._choice_widget.param.watch(self._on_choice_changed, "value")

        # Action panel: captured output + recommendation
        rendered = self.render_captured(captured)
        self._action_pane.objects = [
            o for o in [
                *rendered,
                pn.pane.HTML(rec_html, sizing_mode="stretch_width") if rec_html else None,
            ] if o is not None
        ]

        # Summary inner tabs — choice-independent.
        summary_inner = self._build_group_tabs(
            result, self.step.report_models, SUMMARY_REPORTS, choice=None,
        )
        # Individual inner tabs — re-built on choice change.
        self._rebuild_individual_pane(choice=rec_idx)

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
        if self.state.model_result is not None:
            self.clear_report_cache()
            self._rebuild_individual_pane(choice=event.new)

    def _rebuild_individual_pane(self, choice: int) -> None:
        individual_inner = self._build_group_tabs(
            self.state.model_result,
            self.step.report_models,
            INDIVIDUAL_REPORTS,
            choice=choice,
        )
        self._individual_pane.objects = [individual_inner]
