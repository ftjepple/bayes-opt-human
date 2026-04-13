"""BayesOpt-Human Web GUI.

Launch a standalone web application that guides users through an
optimization round (Data -> Models -> Candidates -> Done) with
interactive reports at each step.

Usage::

    from bayesopt_human.ui import serve
    serve(configs, data_dir="./data/")

Or from the command line::

    python -m bayesopt_human.ui --config my_config.py --data-dir ./data/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bayesopt_human.config import OptimizationConfig


def serve(
    configs: list[OptimizationConfig],
    data_dir: str,
    port: int = 5006,
    show: bool = True,
) -> None:
    """Launch the BayesOpt-Human GUI as a standalone web app.

    Parameters
    ----------
    configs : list[OptimizationConfig]
        Optimization configurations (one per objective function).
    data_dir : str
        Directory containing CSV observation files.
    port : int
        Server port (default 5006).
    show : bool
        Open a browser window automatically.
    """
    import panel as pn

    from bayesopt_human.ui.app import BayesOptApp

    pn.extension(sizing_mode="stretch_width")

    def _create_app():
        app = BayesOptApp(configs, data_dir)
        return app.layout()

    pn.serve(
        {"/": _create_app},
        port=port,
        show=show,
        title="BayesOpt-Human",
    )
