"""CLI entry point: python -m bayesopt_human.ui"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BayesOpt-Human Web GUI",
        prog="python -m bayesopt_human.ui",
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to a Python file that defines CONFIGS "
            "(a list of OptimizationConfig objects)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing CSV data files.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="Port to serve on (default: 5006).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't open browser automatically.",
    )
    args = parser.parse_args()

    # Load config file
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("user_config", str(config_path))
    if spec is None or spec.loader is None:
        print(f"Cannot load config file: {config_path}", file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "CONFIGS"):
        print(
            "Config file must define CONFIGS (a list of OptimizationConfig objects).",
            file=sys.stderr,
        )
        sys.exit(1)

    from bayesopt_human.ui import serve

    serve(
        configs=mod.CONFIGS,
        data_dir=args.data_dir,
        port=args.port,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
