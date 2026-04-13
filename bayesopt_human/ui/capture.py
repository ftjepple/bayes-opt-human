"""Output capture utility for bridging the print-based API to the GUI.

Captures stdout text and collects matplotlib figures created during a
context, suppressing plt.show() so figures are retained for Panel rendering.
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass
class CapturedOutput:
    """Container for captured stdout text and matplotlib figures."""

    text: str = ""
    figures: list[plt.Figure] = field(default_factory=list)


@contextmanager
def capture_output():
    """Capture stdout and collect matplotlib figures without displaying them.

    Yields a CapturedOutput instance. After the context exits, its ``text``
    and ``figures`` attributes are populated.

    Example::

        with capture_output() as captured:
            step.get_data(data_dir)
        print(captured.text)       # all printed output
        print(captured.figures)    # list of plt.Figure objects
    """
    captured = CapturedOutput()

    old_stdout = sys.stdout
    old_show = plt.show
    figs_before = set(plt.get_fignums())

    buffer = io.StringIO()
    sys.stdout = buffer
    plt.show = lambda *_args, **_kwargs: None

    try:
        yield captured
    finally:
        sys.stdout = old_stdout
        plt.show = old_show

        captured.text = buffer.getvalue()
        new_fig_nums = sorted(set(plt.get_fignums()) - figs_before)
        captured.figures = [plt.figure(n) for n in new_fig_nums]
