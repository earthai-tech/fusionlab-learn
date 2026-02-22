# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
stage1_workspace.quicklook

Quicklook panel for Stage-1 (preprocess) workspace.

This widget is intentionally "view-only":
- it does not touch GeoPriorConfig directly
- it does not talk to GeoConfigStore
- the controller (app.py) pushes context + manifest/audit dicts

Expected data sources:
- manifest.json (dict)
- stage1_scaling_audit.json (dict, optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape as _esc
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

Json = Dict[str, Any]
PathLike = Union[str, Path]


@dataclass
class QuicklookContext:
    """Small immutable UI context for quicklook."""
    city: str = ""
    csv_path: str = ""
    runs_root: str = ""
    stage1_dir: str = ""


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _to_file_url(p: PathLike) -> str:
    """
    Convert a local path to file:// URL for QTextBrowser links.
    """
    s = _as_str(p).strip()
    if not s:
        return ""
    try:
        return QUrl.fromLocalFile(s).toString()
    except Exception:
        return ""


def _fmt_list(xs: Sequence[Any], n: int = 8) -> str:
    """
    Render list in a compact form, truncating after n items.
    """
    if not xs:
        return "-"
    items = [_esc(_as_str(v)) for v in xs[:n]]
    suffix = ""
    if len(xs) > n:
        suffix = f" ... (+{len(xs) - n})"
    return ", ".join(items) + suffix


def _get(d: Json, *keys: str, default: Any = None) -> Any:
    """
    Safe nested dict get.
    """
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def _kv_row(k: str, v: str) -> str:
    """
    Simple HTML key/value row.
    """
    kk = _esc(k)
    vv = v if v else "-"
    return (
        "<tr>"
        f"<td style='padding:2px 10px 2px 0;"
        "font-weight:600;white-space:nowrap;'>"
        f"{kk}</td>"
        f"<td style='padding:2px 0;'>{vv}</td>"
        "</tr>"
    )


class Stage1Quicklook(QWidget):
    """
    Stage-1 "Quicklook" panel.

    Public API:
    - clear()
    - set_context(...)
    - set_manifest(...)
    - set_scaling_audit(...)
    - set_status(...)
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._ctx = QuicklookContext()
        self._manifest: Optional[Json] = None
        self._audit: Optional[Json] = None
        self._status: str = ""

        self._build_ui()
        self._render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._ctx = QuicklookContext()
        self._manifest = None
        self._audit = None
        self._status = ""
        self._render()

    def set_context(
        self,
        *,
        city: str,
        csv_path: Optional[PathLike],
        runs_root: Optional[PathLike],
        stage1_dir: Optional[PathLike] = None,
    ) -> None:
        self._ctx = QuicklookContext(
            city=_as_str(city),
            csv_path=_as_str(csv_path),
            runs_root=_as_str(runs_root),
            stage1_dir=_as_str(stage1_dir),
        )
        self._render()

    def set_status(self, text: str) -> None:
        self._status = _as_str(text)
        self._render()

    def set_manifest(self, manifest: Optional[Json]) -> None:
        self._manifest = manifest if isinstance(manifest, dict) else None
        self._render()

    def set_scaling_audit(self, audit: Optional[Json]) -> None:
        self._audit = audit if isinstance(audit, dict) else None
        self._render()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.setSpacing(10)

        self.lbl_title = QLabel("Stage-1 Quicklook")
        self.lbl_title.setObjectName("stage1QuicklookTitle")

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("stage1QuicklookStatus")

        hdr.addWidget(self.lbl_title)
        hdr.addStretch(1)
        hdr.addWidget(self.lbl_status)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)

        layout.addLayout(hdr)
        layout.addWidget(self.browser, 1)

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------
    def _render(self) -> None:
        self.lbl_status.setText(self._status or "")

        html = []
        html.append("<html><body>")
        html.append(self._render_context())
        html.append(self._render_manifest_summary())
        html.append(self._render_audit_hint())
        html.append("</body></html>")

        self.browser.setHtml("".join(html))

    def _render_context(self) -> str:
        ctx = self._ctx

        city = _esc(ctx.city or "-")
        csv = _esc(ctx.csv_path or "-")
        root = _esc(ctx.runs_root or "-")
        s1 = _esc(ctx.stage1_dir or "-")

        csv_url = _to_file_url(ctx.csv_path)
        root_url = _to_file_url(ctx.runs_root)
        s1_url = _to_file_url(ctx.stage1_dir)

        def link_or_text(text: str, url: str) -> str:
            if url:
                return f"<a href='{_esc(url)}'>{text}</a>"
            return text

        csv_v = link_or_text(csv, csv_url)
        root_v = link_or_text(root, root_url)
        s1_v = link_or_text(s1, s1_url)

        rows = []
        rows.append(_kv_row("City", city))
        rows.append(_kv_row("Dataset", csv_v))
        rows.append(_kv_row("Results root", root_v))
        rows.append(_kv_row("Stage-1 dir", s1_v))

        return (
            "<h3 style='margin:6px 0 6px 0;'>Context</h3>"
            "<table>"
            + "".join(rows)
            + "</table>"
            "<hr/>"
        )

    def _render_manifest_summary(self) -> str:
        m = self._manifest
        if not isinstance(m, dict):
            return (
                "<h3 style='margin:6px 0;'>Run summary</h3>"
                "<p>No manifest loaded yet.</p>"
                "<hr/>"
            )

        schema = _esc(_as_str(m.get("schema_version", "-")))
        ts = _esc(_as_str(m.get("timestamp", "-")))
        stage = _esc(_as_str(m.get("stage", "-")))
        model = _esc(_as_str(m.get("model", "-")))
        city = _esc(_as_str(m.get("city", "-")))

        cfg = _get(m, "config", default={})
        feats = _get(cfg, "features", default={})
        cols = _get(cfg, "cols", default={})
        conv = _get(cfg, "conventions", default={})
        idxs = _get(cfg, "indices", default={})
        skw = _get(cfg, "scaling_kwargs", default={})

        t_steps = _esc(_as_str(_get(cfg, "TIME_STEPS", default="-")))
        hz = _esc(
            _as_str(_get(cfg, "FORECAST_HORIZON_YEARS", default="-"))
        )
        train_end = _esc(_as_str(_get(cfg, "TRAIN_END_YEAR", "-")))
        fstart = _esc(
            _as_str(_get(cfg, "FORECAST_START_YEAR", default="-"))
        )

        s_count = len(_get(feats, "static", default=[]) or [])
        d_count = len(_get(feats, "dynamic", default=[]) or [])
        f_count = len(_get(feats, "future", default=[]) or [])

        coord_mode = _esc(_as_str(_get(skw, "coord_mode", "-")))
        epsg = _esc(_as_str(_get(skw, "coord_epsg_used", "-")))
        norm = _esc(_as_str(_get(skw, "coords_normalized", "-")))
        t_units = _esc(_as_str(_get(conv, "time_units", "-")))

        gwl_kind = _esc(_as_str(_get(conv, "gwl_kind", "-")))
        gwl_sign = _esc(_as_str(_get(conv, "gwl_sign", "-")))

        gwl_idx = _esc(_as_str(_get(idxs, "gwl_dyn_index", "-")))
        subs_idx = _esc(_as_str(_get(idxs, "subs_dyn_index", "-")))

        raw_csv = _get(m, "artifacts", "csv", "raw", default="")
        clean_csv = _get(m, "artifacts", "csv", "clean", default="")
        scaled_csv = _get(m, "artifacts", "csv", "scaled", default="")

        def path_link(p: str) -> str:
            p = _as_str(p)
            if not p:
                return "-"
            url = _to_file_url(p)
            txt = _esc(p)
            if not url:
                return txt
            return f"<a href='{_esc(url)}'>{txt}</a>"

        rows = []
        rows.append(_kv_row("schema_version", schema))
        rows.append(_kv_row("timestamp", ts))
        rows.append(_kv_row("city", city))
        rows.append(_kv_row("model", model))
        rows.append(_kv_row("stage", stage))
        rows.append(_kv_row("TIME_STEPS", t_steps))
        rows.append(_kv_row("horizon", hz))
        rows.append(_kv_row("train_end_year", train_end))
        rows.append(_kv_row("forecast_start", fstart))
        rows.append(_kv_row("features", f"{s_count}/{d_count}/{f_count}"
                            " (static/dyn/fut)"))
        rows.append(_kv_row("gwl_kind / sign", f"{gwl_kind} / {gwl_sign}"))
        rows.append(_kv_row("indices (gwl/subs)", f"{gwl_idx} / {subs_idx}"))
        rows.append(_kv_row("time_units", t_units))
        rows.append(_kv_row("coord_mode", coord_mode))
        rows.append(_kv_row("epsg_used", epsg))
        rows.append(_kv_row("coords_normalized", norm))

        # Short feature preview
        static = _get(feats, "static", default=[]) or []
        dyn = _get(feats, "dynamic", default=[]) or []
        fut = _get(feats, "future", default=[]) or []

        f_rows = []
        f_rows.append(_kv_row("static", _fmt_list(static)))
        f_rows.append(_kv_row("dynamic", _fmt_list(dyn)))
        f_rows.append(_kv_row("future", _fmt_list(fut)))

        # Column mapping preview (important for debugging)
        col_keys = (
            "time", "lon", "lat", "x_used", "y_used",
            "subs_model", "depth_model", "head_model",
            "h_field_model", "z_surf_static",
        )
        c_rows = []
        for k in col_keys:
            v = _esc(_as_str(cols.get(k, "-")))
            c_rows.append(_kv_row(k, v))

        return (
            "<h3 style='margin:6px 0 6px 0;'>Run summary</h3>"
            "<table>" + "".join(rows) + "</table>"
            "<h4 style='margin:10px 0 6px 0;'>CSV artifacts</h4>"
            "<ul style='margin:4px 0 8px 18px;'>"
            f"<li>raw: {path_link(raw_csv)}</li>"
            f"<li>clean: {path_link(clean_csv)}</li>"
            f"<li>scaled: {path_link(scaled_csv)}</li>"
            "</ul>"
            "<h4 style='margin:10px 0 6px 0;'>Feature preview</h4>"
            "<table>" + "".join(f_rows) + "</table>"
            "<h4 style='margin:10px 0 6px 0;'>Column mapping</h4>"
            "<table>" + "".join(c_rows) + "</table>"
            "<hr/>"
        )

    def _render_audit_hint(self) -> str:
        """
        Quicklook should not duplicate the scaling tab.
        Just show whether audit exists and 2-3 highlights.
        """
        a = self._audit
        if not isinstance(a, dict):
            return (
                "<h3 style='margin:6px 0;'>Scaling audit</h3>"
                "<p>Audit not loaded.</p>"
            )

        prov = _get(a, "provenance", default={})
        rstats = _get(a, "raw_stats", default={})
        coord = _get(a, "coord_scaler", default={})

        mode = _esc(_as_str(_get(prov, "COORD_MODE", "-")))
        epsg = _esc(_as_str(_get(prov, "coord_epsg_used", "-")))
        plaus = _esc(_as_str(_get(rstats, "utm_plausibility", "-")))

        cr = _get(coord, "coord_ranges(chain_rule)", default={})
        t_rng = _esc(_as_str(_get(cr, "t", "-")))
        x_rng = _esc(_as_str(_get(cr, "x", "-")))
        y_rng = _esc(_as_str(_get(cr, "y", "-")))

        rows = []
        rows.append(_kv_row("coord_mode", mode))
        rows.append(_kv_row("epsg_used", epsg))
        rows.append(_kv_row("utm_plausibility", plaus))
        rows.append(_kv_row("coord_ranges t/x/y", f"{t_rng}, {x_rng}, {y_rng}"))

        return (
            "<h3 style='margin:6px 0 6px 0;'>Scaling audit</h3>"
            "<table>" + "".join(rows) + "</table>"
            "<p style='margin-top:8px;'>"
            "See <b>Feature scaling</b> tab for full details."
            "</p>"
        )
