# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
stage1_workspace.visual_checks

Stage-1 Visual checks panel.

Scope (Stage-1 only):
- Map scatter (x/y or lon/lat) colored by subsidence or GWL
- Time series preview for 3-5 sample points (subsidence + GWL)
- Raw vs scaled comparison for scaled ML numeric cols

Notes
-----
- This widget is UI-first. It *may* read CSV artifacts
  (raw/clean/scaled) from manifest paths for convenience, but the
  controller can also push DataFrames directly.
- Heavy computations are avoided by sampling rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavToolbar,
)
from matplotlib.figure import Figure

Json = Dict[str, Any]
PathLike = Union[str, Path]


@dataclass
class Stage1VisualContext:
    city: str = ""
    stage1_dir: str = ""


@dataclass
class Stage1VisualData:
    """
    Optional data payload.

    If any DataFrame is None, we attempt to load it from manifest
    CSV artifact paths (if provided).
    """
    raw_df: Optional["pd.DataFrame"] = None
    clean_df: Optional["pd.DataFrame"] = None
    scaled_df: Optional["pd.DataFrame"] = None


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _get(d: Json, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def _safe_cols(df, cols: Sequence[str]) -> bool:
    try:
        return all(c in df.columns for c in cols)
    except Exception:
        return False


def _sample_df(df, n: int, seed: int) -> "pd.DataFrame":
    if df is None:
        return df
    if n <= 0:
        return df
    if len(df) <= n:
        return df
    return df.sample(
        n=n,
        random_state=seed,
        replace=False,
    )


def _coerce_numeric(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return s


class _PlotPanel(QWidget):
    """
    Base helper: a FigureCanvas + toolbar + status label.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavToolbar(self.canvas, self)

        self.lbl = QLabel("")
        self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.lbl)

    def set_status(self, text: str) -> None:
        self.lbl.setText(_as_str(text))

    def clear_plot(self, msg: str = "") -> None:
        self.fig.clear()
        self.canvas.draw_idle()
        self.set_status(msg)


class _MapScatterPanel(QWidget):
    """
    Map scatter plot:
    - selects x/y or lon/lat
    - colors by subsidence or GWL
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._manifest: Optional[Json] = None
        self._data = Stage1VisualData()

        self._seed = 7
        self._max_points = 40000

        self._build_ui()

    def set_payload(
        self,
        *,
        manifest: Optional[Json],
        data: Stage1VisualData,
    ) -> None:
        self._manifest = manifest if isinstance(manifest, dict) else None
        self._data = data
        # self._refresh() #  # Lazy: do not auto-refresh here

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.cmb_color = QComboBox()
        self.cmb_color.addItems(
            [
                "Subsidence",
                "GWL (depth/head)",
            ]
        )

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh)

        controls.addWidget(QLabel("Color:"))
        controls.addWidget(self.cmb_color)
        controls.addStretch(1)
        controls.addWidget(self.btn_refresh)

        self.plot = _PlotPanel()

        lay.addLayout(controls)
        lay.addWidget(self.plot, 1)

    def _refresh(self) -> None:
        if pd is None:
            self.plot.clear_plot("pandas is not available.")
            return

        df = self._data.clean_df
        if df is None:
            df = self._try_load_csv(kind="clean")
        if df is None:
            df = self._data.scaled_df
        if df is None:
            df = self._try_load_csv(kind="scaled")

        if df is None or df.empty:
            self.plot.clear_plot("No data to plot.")
            return

        m = self._manifest or {}
        cols = _get(m, "config", "cols", default={}) or {}
        feats = _get(m, "config", "features", default={}) or {}

        time_col = (
            _as_str(cols.get("time_used"))
            or _as_str(cols.get("time"))
        )

        x_col = (
            _as_str(cols.get("x_used"))
            or _as_str(cols.get("x_base"))
        )
        y_col = (
            _as_str(cols.get("y_used"))
            or _as_str(cols.get("y_base"))
        )

        lon_col = _as_str(cols.get("lon"))
        lat_col = _as_str(cols.get("lat"))

        use_xy = bool(x_col and y_col and _safe_cols(df, [x_col, y_col]))
        if not use_xy:
            use_xy = False

        if not use_xy:
            if not (lon_col and lat_col and _safe_cols(df, [lon_col, lat_col])):
                self.plot.clear_plot(
                    "Missing coordinate columns for map."
                )
                return

        if not time_col or time_col not in df.columns:
            self.plot.clear_plot("Missing time column.")
            return

        subs_col = _as_str(cols.get("subs_model")) or _as_str(
            cols.get("subs_raw")
        )

        depth_col = _as_str(cols.get("depth_model")) or _as_str(
            cols.get("depth_raw")
        )
        head_col = _as_str(cols.get("head_model")) or _as_str(
            cols.get("head_raw")
        )

        want = self.cmb_color.currentText().lower().strip()
        if "subs" in want:
            val_col = subs_col
            label = "subsidence"
        else:
            val_col = depth_col if depth_col in df.columns else head_col
            label = "gwl"

        if not val_col or val_col not in df.columns:
            self.plot.clear_plot("Missing value column for coloring.")
            return

        # last time slice, then aggregate per group_id (if possible)
        tmax = df[time_col].max()
        d0 = df[df[time_col] == tmax].copy()
        if d0.empty:
            self.plot.clear_plot("Empty last-time slice.")
            return

        gid_cols = feats.get("group_id_cols") or []
        gid_cols = [c for c in gid_cols if c in d0.columns]

        if gid_cols:
            keep = [val_col, time_col]
            if use_xy:
                keep += [x_col, y_col]
            else:
                keep += [lon_col, lat_col]
            keep += gid_cols
            keep = list(dict.fromkeys(keep))

            d0 = d0[keep]
            d0[val_col] = _coerce_numeric(d0[val_col])
            d0 = d0.dropna(subset=[val_col])

            agg = {val_col: "mean"}
            if use_xy:
                agg[x_col] = "mean"
                agg[y_col] = "mean"
            else:
                agg[lon_col] = "mean"
                agg[lat_col] = "mean"

            d0 = d0.groupby(gid_cols, as_index=False).agg(agg)

        d0 = _sample_df(d0, self._max_points, self._seed)

        self.plot.fig.clear()
        ax = self.plot.fig.add_subplot(111)

        if use_xy:
            x = _coerce_numeric(d0[x_col])
            y = _coerce_numeric(d0[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        else:
            x = _coerce_numeric(d0[lon_col])
            y = _coerce_numeric(d0[lat_col])
            ax.set_xlabel(lon_col)
            ax.set_ylabel(lat_col)

        c = _coerce_numeric(d0[val_col])

        sc = ax.scatter(x, y, c=c, s=8)
        self.plot.fig.colorbar(sc, ax=ax, shrink=0.85)

        ax.set_title(f"Map scatter colored by {label} @ {tmax}")
        ax.grid(True, alpha=0.25)

        self.plot.canvas.draw_idle()
        self.plot.set_status(
            f"Points: {len(d0)} | time={tmax} | col={val_col}"
        )

    def _try_load_csv(self, *, kind: str) -> Optional["pd.DataFrame"]:
        if pd is None:
            return None
        m = self._manifest or {}
        p = _get(m, "artifacts", "csv", kind, default="")
        p = _as_str(p).strip()
        if not p:
            return None
        try:
            return pd.read_csv(p)
        except Exception:
            return None


class _TimeSeriesPanel(QWidget):
    """
    Time series preview:
    - pick N sample points (group_id_cols)
    - plot subsidence and GWL vs time
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._manifest: Optional[Json] = None
        self._data = Stage1VisualData()

        self._seed = 13
        self._max_rows = 200000

        self._build_ui()

    def set_payload(
        self,
        *,
        manifest: Optional[Json],
        data: Stage1VisualData,
    ) -> None:
        self._manifest = manifest if isinstance(manifest, dict) else None
        self._data = data
        # self._refresh() # Lazy: do not auto-refresh here

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.sp_n = QSpinBox()
        self.sp_n.setRange(1, 10)
        self.sp_n.setValue(5)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh)

        controls.addWidget(QLabel("Samples:"))
        controls.addWidget(self.sp_n)
        controls.addStretch(1)
        controls.addWidget(self.btn_refresh)

        self.plot = _PlotPanel()

        lay.addLayout(controls)
        lay.addWidget(self.plot, 1)

    def _refresh(self) -> None:
        if pd is None:
            self.plot.clear_plot("pandas is not available.")
            return

        df = self._data.clean_df
        if df is None:
            df = self._try_load_csv(kind="clean")
        if df is None:
            df = self._data.scaled_df
        if df is None:
            df = self._try_load_csv(kind="scaled")

        if df is None or df.empty:
            self.plot.clear_plot("No data to plot.")
            return

        df = _sample_df(df, self._max_rows, self._seed)

        m = self._manifest or {}
        cols = _get(m, "config", "cols", default={}) or {}
        feats = _get(m, "config", "features", default={}) or {}

        time_col = (
            _as_str(cols.get("time"))
            or _as_str(cols.get("time_used"))
        )
        if not time_col or time_col not in df.columns:
            self.plot.clear_plot("Missing time column.")
            return

        subs_col = _as_str(cols.get("subs_model")) or _as_str(
            cols.get("subs_raw")
        )
        depth_col = _as_str(cols.get("depth_model")) or _as_str(
            cols.get("depth_raw")
        )
        head_col = _as_str(cols.get("head_model")) or _as_str(
            cols.get("head_raw")
        )

        gwl_col = depth_col if depth_col in df.columns else head_col

        if not subs_col or subs_col not in df.columns:
            self.plot.clear_plot("Missing subsidence column.")
            return
        if not gwl_col or gwl_col not in df.columns:
            self.plot.clear_plot("Missing GWL column.")
            return

        gid_cols = feats.get("group_id_cols") or []
        gid_cols = [c for c in gid_cols if c in df.columns]
        if not gid_cols:
            self.plot.clear_plot(
                "Missing group_id_cols for sampling points."
            )
            return

        # sample unique groups
        g = df[gid_cols].drop_duplicates()
        if g.empty:
            self.plot.clear_plot("No groups found.")
            return

        n = int(self.sp_n.value())
        n = max(1, min(n, len(g)))

        g = g.sample(n=n, random_state=self._seed)
        key_df = g.copy()

        # inner join to get series rows
        d0 = df.merge(key_df, on=gid_cols, how="inner")
        if d0.empty:
            self.plot.clear_plot("No rows for sampled points.")
            return

        d0[time_col] = _coerce_numeric(d0[time_col])
        d0[subs_col] = _coerce_numeric(d0[subs_col])
        d0[gwl_col] = _coerce_numeric(d0[gwl_col])

        self.plot.fig.clear()
        ax1 = self.plot.fig.add_subplot(211)
        ax2 = self.plot.fig.add_subplot(212, sharex=ax1)

        # plot each group
        for _, row in key_df.iterrows():
            mask = np.ones(len(d0), dtype=bool)
            for c in gid_cols:
                mask &= d0[c].values == row[c]
            di = d0.loc[mask].sort_values(time_col)

            lbl = ", ".join(f"{c}={row[c]}" for c in gid_cols)
            ax1.plot(di[time_col], di[subs_col], label=lbl)
            ax2.plot(di[time_col], di[gwl_col], label=lbl)

        ax1.set_title("Subsidence time series")
        ax1.set_ylabel(subs_col)
        ax1.grid(True, alpha=0.25)

        ax2.set_title("GWL time series")
        ax2.set_xlabel(time_col)
        ax2.set_ylabel(gwl_col)
        ax2.grid(True, alpha=0.25)

        # keep legend short
        ax2.legend(
            loc="upper left",
            fontsize=8,
            ncol=1,
        )

        self.plot.fig.tight_layout()
        self.plot.canvas.draw_idle()

        self.plot.set_status(
            f"Groups: {n} | rows: {len(d0)} | "
            f"subs={subs_col} | gwl={gwl_col}"
        )

    def _try_load_csv(self, *, kind: str) -> Optional["pd.DataFrame"]:
        if pd is None:
            return None
        m = self._manifest or {}
        p = _get(m, "artifacts", "csv", kind, default="")
        p = _as_str(p).strip()
        if not p:
            return None
        try:
            return pd.read_csv(p)
        except Exception:
            return None


class _RawScaledPanel(QWidget):
    """
    Raw vs scaled comparison for scaled ML numeric cols.

    It compares distributions:
    - raw CSV vs scaled CSV for a chosen feature col
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._manifest: Optional[Json] = None
        self._audit: Optional[Json] = None
        self._data = Stage1VisualData()

        self._seed = 23
        self._max_rows = 150000

        self._build_ui()

    def set_payload(
        self,
        *,
        manifest: Optional[Json],
        audit: Optional[Json],
        data: Stage1VisualData,
    ) -> None:
        self._manifest = manifest if isinstance(manifest, dict) else None
        self._audit = audit if isinstance(audit, dict) else None
        self._data = data
        # Keep list rebuilt (cheap) so UI is ready when user opens the tab
        self._rebuild_feature_list()
        # self._refresh()  # Lazy: do not auto-refresh here

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.cmb_feat = QComboBox()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh)

        controls.addWidget(QLabel("Feature:"))
        controls.addWidget(self.cmb_feat, 1)
        controls.addStretch(1)
        controls.addWidget(self.btn_refresh)

        self.plot = _PlotPanel()

        lay.addLayout(controls)
        lay.addWidget(self.plot, 1)

    def _rebuild_feature_list(self) -> None:
        self.cmb_feat.clear()

        cols = self._scaled_ml_cols()
        if not cols:
            self.cmb_feat.addItem("(no scaled ML cols)")
            self.cmb_feat.setEnabled(False)
            return

        self.cmb_feat.setEnabled(True)
        for c in cols:
            self.cmb_feat.addItem(c)

    def _scaled_ml_cols(self) -> Sequence[str]:
        a = self._audit or {}
        cols = _get(a, "scaled_ml_numeric_cols", default=None)
        if isinstance(cols, list) and cols:
            return [str(x) for x in cols]

        m = self._manifest or {}
        enc = _get(m, "artifacts", "encoders", default={}) or {}
        cols2 = enc.get("scaled_ml_numeric_cols")
        if isinstance(cols2, list) and cols2:
            return [str(x) for x in cols2]

        # fallback: manifest.config.scaler_info keys
        si = _get(m, "config", "scaler_info", default={}) or {}
        if isinstance(si, dict) and si:
            return list(sorted(si.keys()))
        return []

    def _refresh(self) -> None:
        if pd is None:
            self.plot.clear_plot("pandas is not available.")
            return

        feat = _as_str(self.cmb_feat.currentText()).strip()
        if not feat or feat.startswith("("):
            self.plot.clear_plot("No feature selected.")
            return

        raw = self._data.raw_df
        if raw is None:
            raw = self._try_load_csv(kind="raw")

        scaled = self._data.scaled_df
        if scaled is None:
            scaled = self._try_load_csv(kind="scaled")

        if raw is None or scaled is None:
            self.plot.clear_plot("Need raw and scaled CSV to compare.")
            return

        if feat not in raw.columns or feat not in scaled.columns:
            self.plot.clear_plot("Feature not found in both frames.")
            return

        raw = _sample_df(raw, self._max_rows, self._seed)
        scaled = _sample_df(scaled, self._max_rows, self._seed)

        r = _coerce_numeric(raw[feat]).dropna()
        s = _coerce_numeric(scaled[feat]).dropna()
        if r.empty or s.empty:
            self.plot.clear_plot("Empty feature vectors.")
            return

        self.plot.fig.clear()
        ax = self.plot.fig.add_subplot(111)

        ax.hist(r.values, bins=50, alpha=0.6, label="raw")
        ax.hist(s.values, bins=50, alpha=0.6, label="scaled")

        ax.set_title(f"Raw vs scaled: {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
        ax.legend()

        self.plot.fig.tight_layout()
        self.plot.canvas.draw_idle()

        self.plot.set_status(
            f"feat={feat} | raw_n={len(r)} | scaled_n={len(s)}"
        )

    def _try_load_csv(self, *, kind: str) -> Optional["pd.DataFrame"]:
        if pd is None:
            return None
        m = self._manifest or {}
        p = _get(m, "artifacts", "csv", kind, default="")
        p = _as_str(p).strip()
        if not p:
            return None
        try:
            return pd.read_csv(p)
        except Exception:
            return None


class Stage1VisualChecks(QWidget):
    """
    Stage-1 Visual checks (Map / Time series / Raw vs scaled).

    Public API
    ----------
    - clear()
    - set_context(...)
    - set_manifest(...)
    - set_scaling_audit(...)
    - set_data(...)
    - refresh_all()

    Controller usage
    ----------------
    You can push only manifest/audit (widget will load CSVs),
    or push DataFrames too (faster, avoids IO in UI thread).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._ctx = Stage1VisualContext()
        self._manifest: Optional[Json] = None
        self._audit: Optional[Json] = None
        self._data = Stage1VisualData()

        self._build_ui()
        
        # Lazy refresh flags
        self._dirty_map = True
        self._dirty_ts = True
        self._dirty_rs = True
        
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._ctx = Stage1VisualContext()
        self._manifest = None
        self._audit = None
        self._data = Stage1VisualData()

        self._dirty_map = True
        self._dirty_ts = True
        self._dirty_rs = True

        self.map_panel.plot.clear_plot("No data.")
        self.ts_panel.plot.clear_plot("No data.")
        self.rs_panel.plot.clear_plot("No data.")


    def set_context(
        self,
        *,
        city: str,
        stage1_dir: Optional[PathLike] = None,
    ) -> None:
        self._ctx = Stage1VisualContext(
            city=_as_str(city),
            stage1_dir=_as_str(stage1_dir),
        )

    # def set_manifest(self, manifest: Optional[Json]) -> None:
    #     self._manifest = manifest if isinstance(manifest, dict) else None
    #     self._push()
    
    def set_manifest(self, manifest: Optional[Json]) -> None:
        self.map_panel.plot.set_status("Ready. Open this tab or click Refresh.")
        self._manifest = manifest if isinstance(manifest, dict) else None
        self._mark_dirty(all_=True)
        self._push_payloads_only()
        
    # def set_scaling_audit(self, audit: Optional[Json]) -> None:
    #     self._audit = audit if isinstance(audit, dict) else None
    #     self._push()
    def set_scaling_audit(self, audit: Optional[Json]) -> None:
        self._audit = audit if isinstance(audit, dict) else None
        self._mark_dirty(all_=True)
        self._push_payloads_only()
        
    # def set_data(self, data: Optional[Stage1VisualData]) -> None:
    #     self._data = data if isinstance(data, Stage1VisualData) else Stage1VisualData()
    #     self._push()
    
    def set_data(self, data: Optional[Stage1VisualData]) -> None:
        self._data = data if isinstance(data, Stage1VisualData) else Stage1VisualData()
        self._mark_dirty(all_=True)
        self._push_payloads_only()
        
    # def refresh_all(self) -> None:
    #     self.map_panel._refresh()
    #     self.ts_panel._refresh()
    #     self.rs_panel._refresh()
    
    def refresh_all(self) -> None:
        # Ensure panels have the latest payloads
        self._push_payloads_only()
    
        self.map_panel._refresh()
        self.ts_panel._refresh()
        self.rs_panel._refresh()
    
        self._dirty_map = False
        self._dirty_ts = False
        self._dirty_rs = False

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.tabs = QTabWidget()

        self.map_panel = _MapScatterPanel()
        self.ts_panel = _TimeSeriesPanel()
        self.rs_panel = _RawScaledPanel()

        self.tabs.addTab(self.map_panel, "Map scatter")
        self.tabs.addTab(self.ts_panel, "Time series")
        self.tabs.addTab(self.rs_panel, "Raw vs scaled")

        lay.addWidget(self.tabs, 1)

        # top-level refresh (optional convenience)
        row = QHBoxLayout()
        row.setSpacing(8)

        self.btn_refresh_all = QPushButton("Refresh all")
        self.btn_refresh_all.clicked.connect(self.refresh_all)

        row.addStretch(1)
        row.addWidget(self.btn_refresh_all)

        lay.addLayout(row)
        
        # Refresh only when user opens a specific visual tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # def _push(self) -> None:
    #     self.map_panel.set_payload(
    #         manifest=self._manifest,
    #         data=self._data,
    #     )
    #     self.ts_panel.set_payload(
    #         manifest=self._manifest,
    #         data=self._data,
    #     )
    #     self.rs_panel.set_payload(
    #         manifest=self._manifest,
    #         audit=self._audit,
    #         data=self._data,
    #     )

    def _push(self) -> None:
        # Backward-compat if something still calls _push()
        self._push_payloads_only()


    def _mark_dirty(self, *, all_: bool = False) -> None:
        if all_:
            self._dirty_map = True
            self._dirty_ts = True
            self._dirty_rs = True
    
    def _push_payloads_only(self) -> None:
        """
        Push manifest/audit/data into panels without triggering refresh.
    
        This must be cheap and must not read CSVs or plot.
        """
        self.map_panel.set_payload(
            manifest=self._manifest,
            data=self._data,
        )
        self.ts_panel.set_payload(
            manifest=self._manifest,
            data=self._data,
        )
        self.rs_panel.set_payload(
            manifest=self._manifest,
            audit=self._audit,
            data=self._data,
        )
        
    def _on_tab_changed(self, index: int) -> None:
        # Only refresh the currently visible panel, and only if dirty.
        self._refresh_active_if_dirty(index=index)
        
    def _refresh_active_if_dirty(self, *, index: Optional[int] = None) -> None:
        if index is None:
            index = int(self.tabs.currentIndex())
    
        # No payload? Don’t attempt IO/plot.
        if not self._manifest and not (self._data.raw_df or self._data.clean_df or self._data.scaled_df):
            return
    
        if index == 0:
            if self._dirty_map:
                self.map_panel._refresh()
                self._dirty_map = False
        elif index == 1:
            if self._dirty_ts:
                self.ts_panel._refresh()
                self._dirty_ts = False
        elif index == 2:
            if self._dirty_rs:
                self.rs_panel._refresh()
                self._dirty_rs = False
