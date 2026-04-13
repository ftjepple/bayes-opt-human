"""Done step — summary of the optimization round."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import panel as pn

from bayesopt_human.ui.state import AppState
from bayesopt_human.ui.steps.base import BaseStep


class DoneStep(BaseStep):
    title = "Done"
    description = "Round complete. Review your selections and start a new round."

    def __init__(self, step: Any, state: AppState) -> None:
        super().__init__(step, state)
        self._new_round_btn: pn.widgets.Button | None = None
        self._on_new_round: Any = None

    def set_new_round_callback(self, callback: Any) -> None:
        self._on_new_round = callback

    def on_enter(self) -> None:
        sections: list[pn.viewable.Viewable] = []

        # Data summary
        data_result = self.state.data_result
        if data_result and self.state.data_choice < len(data_result.entries):
            entry = data_result.entries[self.state.data_choice]
            sections.append(pn.pane.HTML(
                f"<h3>Data</h3>"
                f"<p>{os.path.basename(entry.filepath)} &mdash; "
                f"{entry.n_obs} observations, {entry.n_dim} dimensions</p>"
            ))

        # Model summary
        model_result = self.state.model_result
        if model_result and self.state.model_choice < len(model_result.candidates):
            mc = model_result.candidates[self.state.model_choice]
            kernel = "WARP" if mc.warping else ("ARD" if mc.ard else "ISO")
            sections.append(pn.pane.HTML(
                f"<h3>Model</h3>"
                f"<p>Transform: {mc.output_transform}, Kernel: {kernel}<br>"
                f"LOO-MAE: {mc.loo_mae:.4f} &pm; {mc.loo_std:.4f}, "
                f"LPD: {mc.loo_lpd:+.4f}, Var(r): {mc.calibration_var:.3f}</p>"
            ))

        # Candidate summary
        coords = self.state.selected_coordinates
        cand_result = self.state.candidate_result
        if coords is not None:
            choice = self.state.candidate_choice
            source = "custom"
            if cand_result and cand_result.recommendations:
                for rec in cand_result.recommendations:
                    if rec.priority == choice:
                        source = rec.source
                        break

            if isinstance(coords, np.ndarray):
                coord_str = "-".join(f"{v:.6f}" for v in coords)
            else:
                coord_str = str(coords)

            sections.append(pn.pane.HTML(
                f"<h3>Selected Candidate</h3>"
                f"<p>Rank #{choice}: {source}<br>"
                f"Coordinates: {coord_str}</p>"
            ))

        # New round button
        self._new_round_btn = pn.widgets.Button(
            name="Start New Round",
            button_type="primary",
            sizing_mode="fixed",
            width=200,
        )
        self._new_round_btn.on_click(self._start_new_round)
        sections.append(pn.layout.Divider())
        sections.append(self._new_round_btn)

        self._action_pane.objects = sections
        self._report_pane.objects = []

    def _start_new_round(self, event: Any) -> None:
        if self._on_new_round:
            self._on_new_round()
