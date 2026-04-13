"""Shared report registry and dispatcher.

Each phase (data, models, candidates) exports one or more lists of ``Report``
entries. The ``report_*`` methods on ``OptimizationStep`` forward to
``run_reports`` with the appropriate list and a context object that carries
everything the per-report functions need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Report:
    """One entry in a phase's report registry.

    Attributes:
        key: The report keyword the user passes as ``report=...``.
        description: One-line help text. Shown when the user calls
            ``report_*`` without a ``report`` argument.
        fn: Callable ``(context) -> Any``. Return values are ignored — reports
            communicate via stdout and ``plt.show``. The signature keeps ``Any``
            so helpers can still return figures for internal use.
    """

    key: str
    description: str
    fn: Callable[[Any], Any]


def run_reports(
    reports: list[Report],
    report: str | None,
    context: Any,
    header: str,
    kind: str = "",
) -> list[str]:
    """Dispatch a ``report_*`` call against a registry.

    Args:
        reports: The list of available ``Report`` entries for this context.
        report: The user-supplied report keyword. ``None`` lists all available
            reports; ``"all"`` runs every report in order; otherwise the
            matching report is run.
        context: Opaque context object passed to each report function.
        header: First line printed when ``report`` is ``None``.
        kind: Inserted into the "Unknown ... report" error message (e.g.
            "summary" or "individual"). Kept as a positional for the error
            only; the existing tests match on the exact phrasing.

    Returns:
        The list of report keys that were rendered (or would be rendered, when
        ``report`` is ``None``).
    """
    by_key = {r.key: r for r in reports}

    if report is None:
        print(header)
        for r in reports:
            print(f"  - {r.key:<22} \u2014 {r.description}")
        print('Use report="all" to show all reports.')
        return [r.key for r in reports]

    if report == "all":
        for r in reports:
            r.fn(context)
        return [r.key for r in reports]

    if report in by_key:
        by_key[report].fn(context)
        return [report]

    label = f"Unknown {kind} report" if kind else "Unknown report"
    raise ValueError(
        f"{label} {report!r}. Valid: {list(by_key)}"
    )
