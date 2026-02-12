# scripts/__main__.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional

from .plot_driver_response import (
    plot_driver_response_main,
)
from .plot_core_ablation import (
    plot_fig3_core_ablation_main,
)
from .plot_litho_parity import (
    figS1_lithology_parity_main,
)


_CMD: Dict[str, Callable[[Optional[List[str]]], None]] = {
    "plot-driver-response": plot_driver_response_main,
    "plot-core-ablation": plot_fig3_core_ablation_main,
    "plot-litho-parity": figS1_lithology_parity_main,
}


def _print_help() -> None:
    cmds = "\n".join(f"  {k}" for k in sorted(_CMD))
    msg = (
        "Usage:\n"
        "  python -m scripts <command> [args]\n\n"
        "Commands:\n"
        f"{cmds}\n\n"
        "Tip:\n"
        "  python -m scripts <command> -h\n"
    )
    print(msg)


def main(argv: Optional[List[str]] = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return

    cmd = args[0]
    rest = args[1:]

    fn = _CMD.get(cmd)
    if fn is None:
        print(f"[ERR] Unknown command: {cmd}")
        _print_help()
        raise SystemExit(2)

    fn(rest)


if __name__ == "__main__":
    main()
