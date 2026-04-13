"""Main application: wizard assembly, navigation, and layout."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("agg")

import panel as pn

from bayesopt_human.optimizer.step import OptimizationStep
from bayesopt_human.ui.state import (
    STEP_CANDIDATE,
    STEP_DATA,
    STEP_DONE,
    STEP_MODEL,
    AppState,
)
from bayesopt_human.ui.steps.candidate_step import CandidateStep
from bayesopt_human.ui.steps.data_step import DataStep
from bayesopt_human.ui.steps.done_step import DoneStep
from bayesopt_human.ui.steps.model_step import ModelStep

STEP_LABELS = ["Data", "Models", "Candidates", "Done"]
NEXT_LABELS = {
    STEP_DATA: "Fit Models",
    STEP_MODEL: "Generate Candidates",
    STEP_CANDIDATE: "Finalize",
}


class BayesOptApp:
    """Wizard-style Panel application wrapping OptimizationStep.

    Parameters
    ----------
    configs : list[OptimizationConfig]
    data_dir : str
    """

    def __init__(self, configs: list, data_dir: str) -> None:
        self.configs = configs
        self.data_dir = data_dir
        self.optimization_step = OptimizationStep(list(configs))
        self.state = AppState()

        # Steps
        self.steps = {
            STEP_DATA: DataStep(self.optimization_step, self.state, data_dir),
            STEP_MODEL: ModelStep(self.optimization_step, self.state),
            STEP_CANDIDATE: CandidateStep(self.optimization_step, self.state),
            STEP_DONE: DoneStep(self.optimization_step, self.state),
        }
        self.steps[STEP_DONE].set_new_round_callback(self._new_round)

        # Navigation buttons
        self._back_btn = pn.widgets.Button(
            name="Back", button_type="light", width=120
        )
        self._next_btn = pn.widgets.Button(
            name="Fit Models", button_type="primary", width=180
        )
        self._back_btn.on_click(self._go_back)
        self._next_btn.on_click(self._go_next)

        # Step indicator
        self._step_indicator = pn.pane.HTML(
            self._render_step_indicator(), sizing_mode="stretch_width"
        )

        # Main content area
        self._content = pn.Column(sizing_mode="stretch_both")

        # Kick off
        self._update_view()
        self.steps[STEP_DATA].on_enter()

    def layout(self) -> pn.Column:
        """Return the full application layout."""
        # Fixed nav bar at the bottom of the viewport via CSS
        nav = pn.Row(
            self._back_btn,
            pn.Spacer(),
            self._next_btn,
            sizing_mode="stretch_width",
            stylesheets=[
                ":host { position: fixed; bottom: 0; left: 0; right: 0; "
                "background: white; padding: 8px 16px; "
                "border-top: 1px solid #e0e0e0; z-index: 1000; "
                "box-shadow: 0 -2px 4px rgba(0,0,0,0.1); }"
            ],
        )
        # Content flows naturally; browser handles scrolling.
        # Bottom spacer prevents content from hiding behind the fixed nav.
        return pn.Column(
            self._step_indicator,
            pn.layout.Divider(),
            self._content,
            nav,
            pn.Spacer(height=60),
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_back(self, event: object = None) -> None:
        if self.state.current_step <= STEP_DATA:
            return
        target = self.state.current_step - 1
        self.state.invalidate_from(target + 1)
        self.state.current_step = target
        self._update_view()
        self.steps[target].on_enter()

    def _go_next(self, event: object = None) -> None:
        current = self.state.current_step
        if current >= STEP_DONE:
            return

        # Run the current step's forward action (choose_*)
        step = self.steps[current]
        try:
            step.on_leave_forward()
        except Exception as exc:
            step._action_pane.objects = [
                *step._action_pane.objects,
                pn.pane.Alert(f"**Error:** {exc}", alert_type="danger"),
            ]
            return

        # Advance
        self.state.current_step = current + 1
        self._update_view()
        self.steps[self.state.current_step].on_enter()

    def _new_round(self) -> None:
        """Reset everything and start a fresh round."""
        # Create a fresh OptimizationStep (clears internal _locked_config etc.)
        self.optimization_step = OptimizationStep(list(self.configs))
        self.state = AppState()

        # Rebuild steps with the fresh instances
        self.steps = {
            STEP_DATA: DataStep(self.optimization_step, self.state, self.data_dir),
            STEP_MODEL: ModelStep(self.optimization_step, self.state),
            STEP_CANDIDATE: CandidateStep(self.optimization_step, self.state),
            STEP_DONE: DoneStep(self.optimization_step, self.state),
        }
        self.steps[STEP_DONE].set_new_round_callback(self._new_round)

        self._update_view()
        self.steps[STEP_DATA].on_enter()

    # ------------------------------------------------------------------
    # View update
    # ------------------------------------------------------------------

    def _update_view(self) -> None:
        current = self.state.current_step
        self._step_indicator.object = self._render_step_indicator()
        self._content.objects = [self.steps[current].panel()]

        # Button states
        self._back_btn.disabled = current <= STEP_DATA
        self._next_btn.disabled = current >= STEP_DONE
        self._next_btn.name = NEXT_LABELS.get(current, "Continue")
        self._next_btn.visible = current < STEP_DONE

    def _render_step_indicator(self) -> str:
        current = self.state.current_step
        parts = []
        for i, label in enumerate(STEP_LABELS):
            if i < current:
                cls = "completed"
                marker = "&#10003;"
                detail = self._step_detail(i)
                text = f"{label}: {detail}" if detail else label
            elif i == current:
                cls = "active"
                marker = "&#9679;"
                text = label
            else:
                cls = "pending"
                marker = "&#9675;"
                text = label
            parts.append(
                f"<span class='bo-step {cls}'>{marker} {text}</span>"
            )
        steps_html = " &mdash; ".join(parts)
        return (
            f"<div style='display:flex; align-items:center; gap:8px; flex-wrap:wrap; "
            f"padding:12px 0; font-family:-apple-system,BlinkMacSystemFont,"
            f"\"Segoe UI\",sans-serif; font-size:14px'>"
            f"{steps_html}</div>"
            f"<style>"
            f".bo-step {{ padding:4px 12px; border-radius:4px; font-weight:500; white-space:nowrap }}"
            f".bo-step.active {{ background:#4a90d9; color:white }}"
            f".bo-step.completed {{ color:#2ca02c }}"
            f".bo-step.pending {{ color:#999 }}"
            f"</style>"
        )

    def _step_detail(self, step_idx: int) -> str:
        """Return a short summary of the choice made at a completed step."""
        if step_idx == STEP_DATA:
            dr = self.state.data_result
            if dr and self.state.data_choice < len(dr.entries):
                entry = dr.entries[self.state.data_choice]
                return os.path.basename(entry.filepath).replace(".csv", "")
        elif step_idx == STEP_MODEL:
            mr = self.state.model_result
            if mr and self.state.model_choice < len(mr.candidates):
                mc = mr.candidates[self.state.model_choice]
                kernel = "WARP" if mc.warping else ("ARD" if mc.ard else "ISO")
                return f"{mc.output_transform} ({kernel})"
        elif step_idx == STEP_CANDIDATE:
            cr = self.state.candidate_result
            if cr and cr.recommendations:
                for rec in cr.recommendations:
                    if rec.priority == self.state.candidate_choice:
                        return rec.source
        return ""
