"""Run the planning, comparison, and ablation benchmarks."""

import argparse
from importlib import import_module
import os
import sys
from typing import Sequence

_COMMANDS = {
    "comparison": "benchmark.run_comparison",
    "planning": "benchmark.run_planning",
    "ablation": "benchmark.run_ablation",
    "ablation-two": "benchmark.run_ablation_two_variants",
    "summary": "benchmark.metrics",
}


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(_COMMANDS))
    if not args:
        parser.print_help()
        return
    selected = parser.parse_args(args[:1])
    os.environ.setdefault("ROOT_DIR", os.getcwd())
    runner = import_module(_COMMANDS[selected.command])
    original_argv = sys.argv
    try:
        sys.argv = [f"python -m benchmark {selected.command}", *args[1:]]
        runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
