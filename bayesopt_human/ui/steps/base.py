"""Shared infrastructure for all wizard steps."""

from __future__ import annotations

import base64
import io
import threading
from typing import Any, Callable

import matplotlib.pyplot as plt
import panel as pn

from bayesopt_human.ui.capture import CapturedOutput, capture_output
from bayesopt_human.ui.state import AppState


class BaseStep:
    """Common behaviour shared by Data, Model, and Candidate steps.

    Parameters
    ----------
    step : OptimizationStep
        The (unmodified) orchestrator instance.
    state : AppState
        Central mutable state shared across all steps.
    """

    # Subclasses override
    title: str = ""
    description: str = ""

    def __init__(self, step: Any, state: AppState) -> None:
        self.step = step
        self.state = state

        self._action_pane = pn.Column(sizing_mode="stretch_width")
        self._report_pane = pn.Column(sizing_mode="stretch_width")
        self._loading = pn.indicators.LoadingSpinner(
            value=False, visible=False, size=40, name="Working..."
        )
        self._report_cache: dict[str, list[pn.viewable.Viewable]] = {}
        # Max observed content height across all reports in this step. Applied
        # as ``min_height`` to both the active inner pn.Tabs and the outer
        # group-selector pn.Tabs (when set), so switching tabs within a group
        # or between the Summary/Individual groups does not shrink the report
        # area and scroll the viewport up.
        self._max_tab_height: int = 0
        self._outer_tabs: pn.Tabs | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def panel(self) -> pn.Column:
        """Return the full layout for this step."""
        header = pn.pane.HTML(
            f"<h2 style='margin:0 0 4px 0'>{self.title}</h2>"
            f"<p style='margin:0; color:#666'>{self.description}</p>",
            sizing_mode="stretch_width",
        )
        return pn.Column(
            header,
            self._loading,
            self._action_pane,
            self._report_pane,
            sizing_mode="stretch_both",
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks (called by BayesOptApp)
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        """Called when the wizard navigates to this step."""

    def on_leave(self) -> None:
        """Called when the wizard navigates away (back or forward)."""

    # ------------------------------------------------------------------
    # Background execution helper
    # ------------------------------------------------------------------

    def _run_blocking(
        self,
        fn: Callable[[], Any],
        callback: Callable[[Any], None],
        *,
        capture: bool = True,
    ) -> None:
        """Run *fn* in a background thread, then call *callback* on the
        main thread with the result.

        If *capture* is True, ``fn`` is wrapped in :func:`capture_output`
        and the callback receives ``(result, captured)`` instead of just
        ``result``.
        """
        self._loading.visible = True
        self._loading.value = True

        def _worker() -> None:
            try:
                if capture:
                    with capture_output() as captured:
                        result = fn()
                    pn.state.execute(lambda: self._on_done(callback, result, captured))
                else:
                    result = fn()
                    pn.state.execute(lambda: self._on_done(callback, result, None))
            except Exception as exc:
                err = exc
                pn.state.execute(lambda: self._on_error(err))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _on_done(
        self,
        callback: Callable,
        result: Any,
        captured: CapturedOutput | None,
    ) -> None:
        self._loading.value = False
        self._loading.visible = False
        if captured is not None:
            callback(result, captured)
        else:
            callback(result)

    def _on_error(self, exc: Exception) -> None:
        self._loading.value = False
        self._loading.visible = False
        self._action_pane.objects = [
            pn.pane.Alert(f"**Error:** {exc}", alert_type="danger")
        ]

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def render_captured(captured: CapturedOutput) -> list[pn.viewable.Viewable]:
        """Convert a CapturedOutput into a list of Panel panes."""
        panes: list[pn.viewable.Viewable] = []
        if captured.text.strip():
            text = _escape_html(captured.text)
            text = _colorize_rag_markers(text)
            panes.append(
                pn.pane.HTML(
                    f"<pre style='font-size:12px; line-height:1.4; "
                    f"background:#fafafa; padding:12px; border-radius:4px; "
                    f"overflow-x:auto; white-space:pre-wrap'>"
                    f"{text}</pre>",
                    sizing_mode="stretch_width",
                )
            )
        for fig in captured.figures:
            panes.append(_fig_to_scaled_html(fig))
        return panes

    # ------------------------------------------------------------------
    # Report tab builder — one group (summary or individual) per call
    # ------------------------------------------------------------------
    #
    # Each step wraps two ``pn.Tabs`` widgets inside an outer ``pn.Tabs``
    # with "Summary" / "Individual" as the group selector. This method
    # builds one inner ``pn.Tabs`` with lazy-loaded, cached content per
    # report. Pass ``choice=None`` to build the summary group.

    def _build_group_tabs(
        self,
        result: Any,
        report_fn: Callable,
        report_names: list[str],
        *,
        choice: int | None,
    ) -> pn.Tabs:
        tabs = pn.Tabs(sizing_mode="stretch_width", dynamic=True)
        for name in report_names:
            tabs.append((
                name.replace("_", " ").title(),
                pn.Column(
                    pn.pane.HTML("<em>Loading...</em>"),
                    sizing_mode="stretch_width",
                ),
            ))

        cache_prefix = "summary" if choice is None else f"individual:{choice}"

        def on_tab_change(event: Any) -> None:
            self._load_tab(
                event.new, tabs, result, report_fn,
                report_names, cache_prefix, choice,
            )

        tabs.param.watch(on_tab_change, "active")
        # Eagerly load tab 0 so the first view is populated immediately.
        self._load_tab(0, tabs, result, report_fn, report_names, cache_prefix, choice)
        return tabs

    def _load_tab(
        self,
        idx: int,
        tabs: pn.Tabs,
        result: Any,
        report_fn: Callable,
        report_names: list[str],
        cache_prefix: str,
        choice: int | None,
    ) -> None:
        """Render the report at ``idx`` into its tab container. Cached."""
        if idx < 0 or idx >= len(report_names):
            return
        name = report_names[idx]
        cache_key = f"{cache_prefix}:{name}"

        container = tabs[idx]

        panes = self._report_cache.get(cache_key)
        if panes is not None:
            container.objects = panes
            return

        container.loading = True
        try:
            with capture_output() as captured:
                report_fn(result, report=name, choice=choice)
            panes = self.render_captured(captured)
            if not panes:
                panes = [pn.pane.HTML("<em>No output for this report.</em>")]
            container.objects = panes
            self._report_cache[cache_key] = panes
            self._sync_tab_min_height(tabs, captured)
        except Exception as exc:
            container.objects = [
                pn.pane.Alert(f"Report error: {exc}", alert_type="warning")
            ]
        finally:
            container.loading = False

    def _sync_tab_min_height(
        self, tabs: pn.Tabs, captured: CapturedOutput
    ) -> None:
        """Keep the report area at least as tall as the tallest loaded report.

        Bumps a single step-wide max (``self._max_tab_height``) and applies it
        to the firing inner tabs' children plus the outer group-selector
        ``pn.Tabs`` (if set). Switching among tabs within a group or across
        groups then preserves scroll position because the total report height
        never shrinks below what's already been seen.
        """
        height = _estimate_content_height(captured)
        if height <= self._max_tab_height:
            return
        self._max_tab_height = height
        for i in range(len(tabs)):
            tabs[i].min_height = height
        if self._outer_tabs is not None:
            self._outer_tabs.min_height = height

    def clear_report_cache(self) -> None:
        """Invalidate all cached report outputs."""
        self._report_cache.clear()


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _colorize_rag_markers(text: str) -> str:
    """Replace plain-text RAG markers with colored HTML equivalents.

    The ``rag_circle`` function in ``step.py`` emits different Unicode
    characters for red/amber/green when not running in Jupyter.  These
    render as grayscale in a ``<pre>`` block and have inconsistent
    monospace widths.  Replace them all with the same circle character
    in the appropriate colour.
    """
    text = text.replace("\u25cf", '<span style="color:#d62728">\u25cf</span>')  # red ●
    text = text.replace("\u25cd", '<span style="color:#ffbf00">\u25cf</span>')  # amber ◍ → ●
    text = text.replace("\u2022", '<span style="color:#2ca02c">\u25cf</span>')  # green • → ●
    return text


def _fig_to_scaled_html(fig: plt.Figure) -> pn.pane.HTML:
    """Render a Matplotlib figure as an ``<img>`` that scales to fit the container."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return pn.pane.HTML(
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:100%; height:auto; display:block;">',
        sizing_mode="stretch_width",
    )


def _estimate_content_height(captured: CapturedOutput) -> int:
    """Rough pixel-height estimate for captured content (text + figures)."""
    height = 0
    if captured.text.strip():
        # ~17px per line (12px font * 1.4 line-height) + 24px padding
        height += captured.text.count("\n") * 17 + 24
    for fig in captured.figures:
        # Use the figure's own height at 100 DPI
        height += int(fig.get_figheight() * 100)
    return height
