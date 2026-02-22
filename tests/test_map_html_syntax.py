import shutil
import subprocess
import tempfile
from pathlib import Path
import re


def _inline_js(html: str) -> str:
    scripts = re.findall(
        r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
        html,
        flags=re.I | re.S,
    )
    return "\n\n".join(scripts)


def _node_check(js: str) -> int:
    node = shutil.which("node")
    if not node:
        return 0  # skip if node not installed
    td = Path(tempfile.mkdtemp(prefix="gp_js_"))
    p = td / "check.js"
    p.write_text(js, encoding="utf-8")
    r = subprocess.run([node, "--check", str(p)], capture_output=True)
    return r.returncode


def test_maplibre_inline_js_parses():
    from fusionlab.tools.app.geoprior.ui.map.engines.maplibre_html import maplibre_html

    assert _node_check(_inline_js(maplibre_html())) == 0


def test_leaflet_inline_js_parses():
    from fusionlab.tools.app.geoprior.ui.map.engines.leaflet_html import leaflet_html

    assert _node_check(_inline_js(leaflet_html())) == 0
