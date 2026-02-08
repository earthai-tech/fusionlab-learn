# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple


def _extract_inline_scripts(html: str) -> List[str]:
    # Inline scripts only: <script> ... </script> (no src=)
    pat = re.compile(
        r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
        re.IGNORECASE | re.DOTALL,
    )
    return [m.group(1).strip("\n") for m in pat.finditer(html)]


def _write_tmp_files(name: str, html: str) -> Tuple[Path, Path]:
    td = Path(tempfile.mkdtemp(prefix=f"gp_{name}_"))
    html_path = td / f"{name}.html"
    js_path = td / f"{name}.inline.js"
    html_path.write_text(html, encoding="utf-8")

    scripts = _extract_inline_scripts(html)
    merged = []
    for i, s in enumerate(scripts):
        merged.append(f"\n/* --- inline script #{i} --- */\n")
        merged.append(s)
        merged.append("\n")
    js_path.write_text("".join(merged), encoding="utf-8")
    return html_path, js_path


def _node_check(js_path: Path) -> Tuple[int, str, str]:
    node = shutil.which("node")
    if not node:
        return (
            2,
            "",
            "Node.js not found on PATH. Install Node to run JS check.",
        )

    p = subprocess.run(
        [node, "--check", str(js_path)],
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout, p.stderr


def _print_context(js_path: Path, stderr: str, radius: int = 4) -> None:
    # Try to extract "file:line:col" from node output
    m = re.search(r":(\d+):(\d+)\b", stderr)
    if not m:
        return
    line = int(m.group(1))
    col = int(m.group(2))

    lines = js_path.read_text(encoding="utf-8").splitlines()
    lo = max(1, line - radius)
    hi = min(len(lines), line + radius)

    print("\n--- context ---")
    print(f"at {js_path}:{line}:{col}\n")
    for i in range(lo, hi + 1):
        mark = ">>" if i == line else "  "
        print(f"{mark} {i:4d} | {lines[i - 1]}")
    print("--- end context ---\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine",
        choices=["maplibre", "leaflet", "all"],
        default="all",
    )
    args = ap.parse_args()

    targets = []
    if args.engine in ("maplibre", "all"):
        targets.append("maplibre")
    if args.engine in ("leaflet", "all"):
        targets.append("leaflet")

    # Import inside main so Python syntax errors show clearly
    for eng in targets:
        if eng == "maplibre":
            from fusionlab.tools.app.geoprior.ui.map.engines.maplibre_html import (
                maplibre_html,
            )

            html = maplibre_html()
        else:
            from fusionlab.tools.app.geoprior.ui.map.engines.leaflet_html import (
                leaflet_html,
            )

            html = leaflet_html()

        html_path, js_path = _write_tmp_files(eng, html)
        code, out, err = _node_check(js_path)

        print(f"\n[{eng}] html  : {html_path}")
        print(f"[{eng}] js    : {js_path}")

        if code == 2:
            print(f"[{eng}] ERROR: {err.strip()}")
            return 2

        if code != 0:
            print(f"[{eng}] JS SYNTAX ERROR:\n{err.strip()}")
            _print_context(js_path, err)
            return 1

        print(f"[{eng}] OK (JS syntax)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# Windows:
# py scripts\check_map_html_syntax.py --engine maplibre
# py scripts\check_map_html_syntax.py --engine leaflet
# py scripts\check_map_html_syntax.py --engine all

# macOS/Linux:
# python3 scripts/check_map_html_syntax.py --engine all
  
