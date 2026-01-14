# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
stage1_workspace.feature_scaling

Feature scaling panel for Stage-1 (preprocess) workspace.

This widget is view-only:
- no direct GeoPriorConfig access
- no GeoConfigStore mutation
- controller pushes manifest + optional scaling audit dicts

Data sources (best-effort):
- manifest.json (dict)
- stage1_scaling_audit.json (dict, optional)

The panel explains:
- feature contract (static/dynamic/future)
- scaling contract (scaled vs SI vs unscaled)
- coordinate normalization summary
- units provenance and key physics scaling kwargs
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape as _esc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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
class FeatureScalingContext:
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
    s = _as_str(p).strip()
    if not s:
        return ""
    try:
        return QUrl.fromLocalFile(s).toString()
    except Exception:
        return ""


def _link_or_text(path: str) -> str:
    p = _as_str(path)
    if not p:
        return "-"
    url = _to_file_url(p)
    txt = _esc(p)
    if not url:
        return txt
    return f"<a href='{_esc(url)}'>{txt}</a>"


def _get(d: Json, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def _kv_row(k: str, v_html: str) -> str:
    kk = _esc(k)
    vv = v_html if v_html else "-"
    return (
        "<tr>"
        "<td style='padding:2px 10px 2px 0;"
        "font-weight:600;white-space:nowrap;'>"
        f"{kk}</td>"
        f"<td style='padding:2px 0;'>{vv}</td>"
        "</tr>"
    )


def _fmt_list(xs: Sequence[Any], n: int = 10) -> str:
    if not xs:
        return "-"
    items = [_esc(_as_str(v)) for v in xs[:n]]
    suf = ""
    if len(xs) > n:
        suf = f" ... (+{len(xs) - n})"
    return ", ".join(items) + suf


def _badge(text: str, kind: str) -> str:
    """
    kind: scaled|si|unscaled|other
    """
    k = (kind or "").lower().strip()
    bg = "#455a64"
    if k == "scaled":
        bg = "#1565c0"
    elif k == "si":
        bg = "#2e7d32"
    elif k == "unscaled":
        bg = "#6d4c41"
    elif k == "other":
        bg = "#6a1b9a"

    return (
        "<span style='display:inline-block;"
        "padding:1px 8px;border-radius:10px;"
        f"background:{bg};color:white;"
        "font-weight:600;'>"
        f"{_esc(text)}</span>"
    )


class Stage1FeatureScaling(QWidget):
    """
    Stage-1 feature scaling panel.

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

        self._ctx = FeatureScalingContext()
        self._manifest: Optional[Json] = None
        self._audit: Optional[Json] = None
        self._status: str = ""

        self._build_ui()
        self._render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._ctx = FeatureScalingContext()
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
        self._ctx = FeatureScalingContext(
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

        self.lbl_title = QLabel("Stage-1 Feature scaling")
        self.lbl_title.setObjectName("stage1FeatureScalingTitle")

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("stage1FeatureScalingStatus")

        hdr.addWidget(self.lbl_title)
        hdr.addStretch(1)
        hdr.addWidget(self.lbl_status)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)

        layout.addLayout(hdr)
        layout.addWidget(self.browser, 1)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def _render(self) -> None:
        self.lbl_status.setText(self._status or "")

        html: List[str] = []
        html.append("<html><body>")
        html.append(self._render_context())
        html.append(self._render_feature_contract())
        html.append(self._render_scaling_table())
        html.append(self._render_coords_block())
        html.append(self._render_units_provenance())
        html.append(self._render_scaling_kwargs())
        html.append("</body></html>")

        self.browser.setHtml("".join(html))

    def _render_context(self) -> str:
        c = self._ctx
        rows = []
        rows.append(_kv_row("City", _esc(c.city or "-")))
        rows.append(_kv_row("Dataset", _link_or_text(c.csv_path)))
        rows.append(_kv_row("Results root", _link_or_text(c.runs_root)))
        rows.append(_kv_row("Stage-1 dir", _link_or_text(c.stage1_dir)))
        return (
            "<h3 style='margin:6px 0 6px 0;'>Context</h3>"
            "<table>"
            + "".join(rows)
            + "</table>"
            "<hr/>"
        )

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------
    def _render_feature_contract(self) -> str:
        m = self._manifest
        if not isinstance(m, dict):
            return (
                "<h3 style='margin:6px 0 6px 0;'>Feature contract</h3>"
                "<p>No manifest loaded yet.</p>"
                "<hr/>"
            )

        cfg = _get(m, "config", default={})
        feats = _get(cfg, "features", default={})
        idxs = _get(cfg, "indices", default={})

        static = _get(feats, "static", default=[]) or []
        dyn = _get(feats, "dynamic", default=[]) or []
        fut = _get(feats, "future", default=[]) or []
        gid = _get(feats, "group_id_cols", default=[]) or []

        rows = []
        rows.append(_kv_row("static (count)", _esc(str(len(static)))))
        rows.append(_kv_row("dynamic (count)", _esc(str(len(dyn)))))
        rows.append(_kv_row("future (count)", _esc(str(len(fut)))))
        rows.append(_kv_row("group_id_cols", _fmt_list(gid)))

        idx_rows = []
        for k in (
            "gwl_dyn_index",
            "subs_dyn_index",
            "gwl_dyn_name",
            "z_surf_static_index",
        ):
            idx_rows.append(_kv_row(k, _esc(_as_str(idxs.get(k, "-")))))

        return (
            "<h3 style='margin:6px 0 6px 0;'>Feature contract</h3>"
            "<table>" + "".join(rows) + "</table>"
            "<h4 style='margin:10px 0 6px 0;'>Indices</h4>"
            "<table>" + "".join(idx_rows) + "</table>"
            "<h4 style='margin:10px 0 6px 0;'>Preview</h4>"
            "<table>"
            + _kv_row("static", _fmt_list(static))
            + _kv_row("dynamic", _fmt_list(dyn))
            + _kv_row("future", _fmt_list(fut))
            + "</table>"
            "<hr/>"
        )

    def _render_scaling_table(self) -> str:
        """
        Main user-facing table:
        Feature | Role | Scaling
        """
        m = self._manifest
        if not isinstance(m, dict):
            return (
                "<h3 style='margin:6px 0 6px 0;'>Scaling split</h3>"
                "<p>No manifest loaded yet.</p>"
                "<hr/>"
            )

        cfg = _get(m, "config", default={})
        feats = _get(cfg, "features", default={})

        static = _get(feats, "static", default=[]) or []
        dyn = _get(feats, "dynamic", default=[]) or []
        fut = _get(feats, "future", default=[]) or []

        audit = self._audit if isinstance(self._audit, dict) else {}
        split = _get(audit, "feature_split", default={})

        scaled_ml = _get(
            audit,
            "scaled_ml_numeric_cols",
            default=[],
        ) or []

        dyn_split = _get(split, "dynamic_features", default={})
        sta_split = _get(split, "static_features", default={})
        fut_split = _get(split, "future_features", default={})

        dyn_scaled = _get(dyn_split, "scaled_by_main_scaler", default=[]) or []
        dyn_si = _get(dyn_split, "si_unscaled", default=[]) or []
        dyn_other = _get(dyn_split, "other_unscaled_mixed", default=[]) or []

        sta_scaled = _get(sta_split, "scaled_by_main_scaler", default=[]) or []
        sta_si = _get(sta_split, "si_unscaled", default=[]) or []
        sta_other = _get(sta_split, "other_unscaled_mixed", default=[]) or []

        fut_scaled = _get(fut_split, "scaled_by_main_scaler", default=[]) or []
        fut_si = _get(fut_split, "si_unscaled", default=[]) or []
        fut_other = _get(fut_split, "other_unscaled_mixed", default=[]) or []

        # Fallback classification if audit is absent.
        scaler_info = _get(cfg, "scaler_info", default={})
        scaled_from_manifest = set(scaler_info.keys() or ())

        def classify(name: str, role: str) -> str:
            if audit:
                return self._classify_from_audit(
                    name=name,
                    role=role,
                    dyn_scaled=dyn_scaled,
                    dyn_si=dyn_si,
                    dyn_other=dyn_other,
                    sta_scaled=sta_scaled,
                    sta_si=sta_si,
                    sta_other=sta_other,
                    fut_scaled=fut_scaled,
                    fut_si=fut_si,
                    fut_other=fut_other,
                )
            if name in scaled_from_manifest:
                return "scaled"
            if name.endswith("__si"):
                return "si"
            return "unscaled"

        rows: List[str] = []
        for role, names in (
            ("static", static),
            ("dynamic", dyn),
            ("future", fut),
        ):
            for f in names:
                name = _as_str(f)
                kind = classify(name, role)
                badge = self._scaling_badge(kind)
                rows.append(
                    "<tr>"
                    "<td style='padding:3px 10px 3px 0;"
                    "white-space:nowrap;'>"
                    f"{_esc(name)}</td>"
                    "<td style='padding:3px 10px 3px 0;'>"
                    f"{_esc(role)}</td>"
                    f"<td style='padding:3px 0;'>{badge}</td>"
                    "</tr>"
                )

        hint_rows: List[str] = []
        hint_rows.append(
            _kv_row(
                "scaled_ml_numeric_cols",
                _fmt_list(scaled_ml),
            )
        )

        return (
            "<h3 style='margin:6px 0 6px 0;'>Scaling split</h3>"
            "<table style='border-collapse:collapse;'>"
            "<tr>"
            "<th style='text-align:left;padding:3px 10px 3px 0;'>"
            "Feature</th>"
            "<th style='text-align:left;padding:3px 10px 3px 0;'>"
            "Role</th>"
            "<th style='text-align:left;padding:3px 0;'>"
            "Scaling</th>"
            "</tr>"
            + "".join(rows)
            + "</table>"
            "<h4 style='margin:10px 0 6px 0;'>Notes</h4>"
            "<table>"
            + "".join(hint_rows)
            + "</table>"
            "<p style='margin-top:8px;color:#546e7a;'>"
            "Scaled = main ML scaler. SI = kept in physical units."
            "</p>"
            "<hr/>"
        )

    def _classify_from_audit(
        self,
        *,
        name: str,
        role: str,
        dyn_scaled: Sequence[str],
        dyn_si: Sequence[str],
        dyn_other: Sequence[str],
        sta_scaled: Sequence[str],
        sta_si: Sequence[str],
        sta_other: Sequence[str],
        fut_scaled: Sequence[str],
        fut_si: Sequence[str],
        fut_other: Sequence[str],
    ) -> str:
        if role == "dynamic":
            if name in dyn_scaled:
                return "scaled"
            if name in dyn_si:
                return "si"
            if name in dyn_other:
                return "other"
            return "unscaled"

        if role == "static":
            if name in sta_scaled:
                return "scaled"
            if name in sta_si:
                return "si"
            if name in sta_other:
                return "other"
            return "unscaled"

        if role == "future":
            if name in fut_scaled:
                return "scaled"
            if name in fut_si:
                return "si"
            if name in fut_other:
                return "other"
            return "unscaled"

        return "unscaled"

    def _scaling_badge(self, kind: str) -> str:
        k = (kind or "").lower().strip()
        if k == "scaled":
            return _badge("SCALED", "scaled")
        if k == "si":
            return _badge("SI", "si")
        if k == "other":
            return _badge("MIXED", "other")
        return _badge("UNSCALED", "unscaled")

    def _render_coords_block(self) -> str:
        m = self._manifest
        if not isinstance(m, dict):
            return ""
        cfg = _get(m, "config", default={})
        skw = _get(cfg, "scaling_kwargs", default={})

        coord_mode = _esc(_as_str(_get(skw, "coord_mode", default="-")))
        epsg = _esc(_as_str(_get(skw, "coord_epsg_used", default="-")))
        deg = _esc(_as_str(_get(skw, "coords_in_degrees", default="-")))
        norm = _esc(_as_str(_get(skw, "coords_normalized", default="-")))

        # Prefer audit coord ranges when available.
        a = self._audit if isinstance(self._audit, dict) else {}
        cr = _get(a, "coord_scaler", "coord_ranges(chain_rule)", default={})
        if not cr:
            cr = _get(skw, "coord_ranges", default={}) or {}

        t_rng = _esc(_as_str(_get(cr, "t", default="-")))
        x_rng = _esc(_as_str(_get(cr, "x", default="-")))
        y_rng = _esc(_as_str(_get(cr, "y", default="-")))

        rows = []
        rows.append(_kv_row("coord_mode", coord_mode))
        rows.append(_kv_row("coord_epsg_used", epsg))
        rows.append(_kv_row("coords_in_degrees", deg))
        rows.append(_kv_row("coords_normalized", norm))
        rows.append(_kv_row("coord_ranges (t,x,y)", f"{t_rng}, {x_rng}, {y_rng}"))

        return (
            "<h3 style='margin:6px 0 6px 0;'>Coordinates</h3>"
            "<table>"
            + "".join(rows)
            + "</table>"
            "<hr/>"
        )

    def _render_units_provenance(self) -> str:
        m = self._manifest
        if not isinstance(m, dict):
            return ""
        up = _get(m, "config", "units_provenance", default={}) or {}

        if not up:
            return (
                "<h3 style='margin:6px 0 6px 0;'>Units provenance</h3>"
                "<p>No units provenance provided in manifest.</p>"
                "<hr/>"
            )

        rows = []
        for k, v in sorted(up.items(), key=lambda kv: kv[0]):
            rows.append(_kv_row(k, _esc(_as_str(v))))

        return (
            "<h3 style='margin:6px 0 6px 0;'>Units provenance</h3>"
            "<table>"
            + "".join(rows)
            + "</table>"
            "<hr/>"
        )

    def _render_scaling_kwargs(self) -> str:
        m = self._manifest
        if not isinstance(m, dict):
            return ""
        skw = _get(m, "config", "scaling_kwargs", default={}) or {}
        if not skw:
            return (
                "<h3 style='margin:6px 0 6px 0;'>Scaling kwargs</h3>"
                "<p>No scaling_kwargs found in manifest.</p>"
            )

        # Keep a curated subset for readability.
        keys = [
            # Target scaling
            "subs_scale_si",
            "subs_bias_si",
            "head_scale_si",
            "head_bias_si",
            "H_scale_si",
            "H_bias_si",
            # Conventions
            "subsidence_kind",
            "allow_subs_residual",
            # Coordinates + chain rule
            "coords_normalized",
            "coord_order",
            "coord_mode",
            "coord_src_epsg",
            "coord_target_epsg",
            "coord_epsg_used",
            "coords_in_degrees",
            # Residual units
            "cons_residual_units",
            "gw_residual_units",
            "dt_min_units",
            # Debug / stability
            "clip_global_norm",
            "scaling_error_policy",
            "debug_physics_grads",
            # Indices snapshot (useful)
            "gwl_dyn_index",
            "subs_dyn_index",
            "z_surf_static_index",
            "gwl_col",
            "z_surf_col",
            "subs_dyn_name",
        ]

        rows: List[str] = []
        for k in keys:
            if k not in skw:
                continue
            rows.append(_kv_row(k, _esc(_as_str(skw.get(k)))))

        # Also show where the main scaler lives.
        scaler_path = _get(
            m,
            "artifacts",
            "encoders",
            "main_scaler",
            default="",
        )
        if scaler_path:
            rows.append(_kv_row("main_scaler", _link_or_text(scaler_path)))

        return (
            "<h3 style='margin:6px 0 6px 0;'>Scaling kwargs</h3>"
            "<table>"
            + "".join(rows)
            + "</table>"
            "<p style='margin-top:8px;color:#546e7a;'>"
            "This section is a curated subset to keep the view readable."
            "</p>"
        )
