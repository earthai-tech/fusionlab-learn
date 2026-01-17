# geoprior/ui/map/analytics_panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.analytics_panel

Bottom analytics panel (D), collapsible.

Modern skeleton
---------------
- Tabs: Spatial | Sharpness | Reliability | Inspector
- Each tab: controls + matplotlib canvas
- Store-driven refresh (map.* keys)

Notes
-----
This is a skeleton with real, useful plots.
We will deepen each tab progressively.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QDoubleSpinBox,
    QCheckBox,
)



from ...config.store import GeoConfigStore
from .plot_utils import (
    HAS_MPL,
    MplPlot,
    pick_obs_col,
    safe_quantile_cols,
)


@dataclass
class MapAnaCtx:
    path: Optional[Path] = None
    df: Optional[pd.DataFrame] = None

    x: str = ""
    y: str = ""
    z: str = ""
    t: str = ""
    step: str = ""

    time_value: str = ""
    q_cols: list[tuple[float, str]] = None  # type: ignore
    obs_col: str = ""


class CollapsibleAnalyticsPanel(QFrame):
    """
    Bottom panel container for analytics tabs.
    """

    def __init__(
        self,
        *,
        store: Optional[GeoConfigStore] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("MapAnalyticsPanel")
        self.setFrameShape(QFrame.StyledPanel)

        self.store = store
        self._expanded = True

        self.tabs = QTabWidget(self)
        self._ctx = MapAnaCtx(q_cols=[])

        self._spatial = SpatialTab(parent=self)
        self._sharp = SharpnessTab(parent=self)
        self._rely = ReliabilityTab(parent=self)
        self._insp = InspectorTab(parent=self)

        self._build_ui()
        self._build_tabs()

        if self.store is not None:
            self.store.config_changed.connect(
                self._on_store_changed,
            )
            self.refresh()

    # -------------------------
    # Public API
    # -------------------------
    def expand(self) -> None:
        self._expanded = True
        self.setVisible(True)
        self.refresh()


    def collapse(self) -> None:
        self._expanded = False
        self.setVisible(False)

    def is_expanded(self) -> bool:
        return bool(self._expanded)

    def refresh(self) -> None:
        """
        Pull from store and update all tabs.
        """
        if self.store is None:
            return

        p = str(self.store.get("map.active_file", ""))
        p = p.strip()
        path = Path(p) if p else None

        x = str(self.store.get("map.x_col", "")).strip()
        y = str(self.store.get("map.y_col", "")).strip()
        z = str(self.store.get("map.value_col", "")).strip()
        if not z:
            z = str(self.store.get("map.z_col", "")).strip()

        t = str(self.store.get("map.time_col", "")).strip()
        step = str(self.store.get("map.step_col", "")).strip()
        tv = str(self.store.get("map.time_value", "")).strip()

        df = self._load_frame(
            path=path,
            x=x,
            y=y,
            z=z,
            t=t,
            step=step,
            time_value=tv,
        )

        cols = list(df.columns) if df is not None else []
        q_cols = safe_quantile_cols(cols)
        obs = pick_obs_col(cols)

        self._ctx = MapAnaCtx(
            path=path,
            df=df,
            x=x,
            y=y,
            z=z,
            t=t,
            step=step,
            time_value=tv,
            q_cols=q_cols,
            obs_col=obs,
        )

        self._spatial.set_context(self._ctx)
        self._sharp.set_context(self._ctx)
        self._rely.set_context(self._ctx)
        self._insp.set_context(self._ctx)

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        lay.addWidget(self.tabs)

        self.setStyleSheet(
            "\n".join([
                "QTabBar::tab{",
                "  padding: 6px 12px;",
                "  border-radius: 10px;",
                "  margin-right: 6px;",
                "}",
                "QTabBar::tab:selected{",
                "  font-weight: 600;",
                "}",
            ])
        )

    def _build_tabs(self) -> None:
        self.tabs.clear()
    
        self.tabs.addTab(self._spatial, "Spatial")
        self.tabs.addTab(self._sharp, "Sharpness")
        self.tabs.addTab(self._rely, "Reliability")
    
        # Single inspector instance (used by MapTab click hook)
        self.inspector = self._insp
        self.tabs.addTab(self.inspector, "Inspector")
        
        self.tabs.currentChanged.connect(lambda _: self.tabs.currentWidget().refresh())



    # -------------------------
    # Store + IO
    # -------------------------
    def _on_store_changed(self, keys) -> None:
        ks = set(keys or [])
        want = {
            "map.active_file",
            "map.x_col",
            "map.y_col",
            "map.z_col",
            "map.value_col",
            "map.time_col",
            "map.step_col",
            "map.time_value",
        }
        if ks.intersection(want):
            self.refresh()

    def _load_frame(
        self,
        *,
        path: Optional[Path],
        x: str,
        y: str,
        z: str,
        t: str,
        step: str,
        time_value: str,
    ) -> Optional[pd.DataFrame]:
        if path is None:
            return None
        if not path.exists():
            return None

        use = []
        for c in (x, y, z, t, step):
            if c and c not in use:
                use.append(c)

        # If we have no mapping yet, load a small header.
        if not use:
            try:
                df0 = pd.read_csv(path, nrows=200)
            except Exception:
                return None
            return df0

        try:
            df = pd.read_csv(path, usecols=use)
        except Exception:
            try:
                df = pd.read_csv(path)
            except Exception:
                return None

        if time_value and t and t in df.columns:
            try:
                tv = int(time_value)
                df = df[df[t] == tv]
            except Exception:
                pass

        return df


# =====================================================
# Tabs
# =====================================================

class _TabBase(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ctx = MapAnaCtx(q_cols=[])

    def set_context(self, ctx: MapAnaCtx) -> None:
        self._ctx = ctx
        self.refresh()

    def refresh(self) -> None:
        return


class SpatialTab(_TabBase):
    """
    Spatial view (2D scatter or 3D scatter).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.cmb_mode = QComboBox(self)
        self.cmb_mode.addItems(["2D", "3D"])

        self.sp_max = QSpinBox(self)
        self.sp_max.setRange(2000, 500000)
        self.sp_max.setValue(80000)

        self.plot = MplPlot(
            title="Spatial scatter",
            parent=self,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        ctrl = self._controls(self)
        root.addWidget(ctrl, 0)
        root.addWidget(self.plot, 1)

        self.cmb_mode.currentIndexChanged.connect(
            self.refresh,
        )
        self.sp_max.valueChanged.connect(
            self.refresh,
        )

        if self.plot.actions is not None:
            self.plot.actions.btn_clear.clicked.connect(
                self.plot.clear_axes,
            )
            self.plot.actions.btn_refresh.clicked.connect(
                self.refresh,
            )
            self.plot.actions.btn_save.clicked.connect(
                self._save_png,
            )

    def _controls(self, parent: QWidget) -> QWidget:
        box = QGroupBox("View", parent)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        v.addWidget(QLabel("Mode:", box), 0)
        v.addWidget(self.cmb_mode, 0)

        v.addWidget(QLabel("Max points:", box), 0)
        v.addWidget(self.sp_max, 0)

        v.addStretch(1)
        return box

    def _save_png(self) -> None:
        p, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            "spatial.png",
            "PNG (*.png)",
        )
        if not p:
            return
        self.plot.save_png(p)

    def refresh(self) -> None:
        if not HAS_MPL:
            return
        if not self.plot.is_ready():
            return

        df = self._ctx.df
        x = self._ctx.x
        y = self._ctx.y
        z = self._ctx.z

        if df is None or not len(df):
            _show_empty(
                self.plot,
                title="No data to plot",
                message="Select an active dataset to enable spatial analytics.",
                hint="Go to Map → Data panel and choose an active file.",
            )
            return

        if not (x and y and z):
            _show_empty(
                self.plot,
                title="Mapping incomplete",
                message="Select X, Y and Z columns to render the spatial scatter.",
                hint="Use the pickers in the map header (X / Y / Z).",
            )
            return
    
        if x not in df.columns or y not in df.columns or z not in df.columns:
            _show_empty(
                self.plot,
                title="Columns not found",
                message="One or more selected columns do not exist in the dataset.",
                hint="Check X/Y/Z selection and reload columns if needed.",
            )
            return
        
        # Guard: X and Y must differ (otherwise d[x] becomes 2D after selection)
        if x == y:
            _show_empty(
                self.plot,
                title="Invalid mapping",
                message="X and Y cannot be the same column.",
                hint="Pick two different columns for X and Y in the map header.",
            )
            return
        
        # Build a guaranteed-1D frame (handles duplicate column labels safely)
        sx = _pick_1d(df, x)
        sy = _pick_1d(df, y)
        sz = _pick_1d(df, z)
        
        d = pd.concat([sx, sy, sz], axis=1)
        d.columns = [x, y, z]
        d = d.dropna()
        
        nmax = int(self.sp_max.value())
        if len(d) > nmax:
            d = d.sample(n=nmax, random_state=0)
        
        # ---- plotting ----
        self.plot.clear_axes()
        assert self.plot.fig is not None
        
        mode = str(self.cmb_mode.currentText()).lower()
        
        if mode == "3d":
            self.plot.fig.clf()
            ax = self.plot.fig.add_subplot(111, projection="3d")
            ax.scatter(
                d[x].to_numpy(),
                d[y].to_numpy(),
                d[z].to_numpy(),
                s=8,
                alpha=0.85,
            )
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            self.plot.ax = ax
            self.plot.draw()
            return
        
        ax = self.plot.ax
        assert ax is not None
        
        sc = ax.scatter(
            d[x].to_numpy(),
            d[y].to_numpy(),
            c=d[z].to_numpy(),
            s=10,
            alpha=0.85,
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title("Colored by Z")
        
        self.plot.fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        self.plot.draw()



class SharpnessTab(_TabBase):
    """
    Uncertainty width vs time (if quantiles exist).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.cmb_int = QComboBox(self)
        self.cmb_int.addItems([
            "50% (q25-q75)",
            "80% (q10-q90)",
            "90% (q05-q95)",
        ])

        self.plot = MplPlot(
            title="Interval width (sharpness)",
            parent=self,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        ctrl = self._controls(self)
        root.addWidget(ctrl, 0)
        root.addWidget(self.plot, 1)

        self.cmb_int.currentIndexChanged.connect(
            self.refresh,
        )

        if self.plot.actions is not None:
            self.plot.actions.btn_clear.clicked.connect(
                self.plot.clear_axes,
            )
            self.plot.actions.btn_refresh.clicked.connect(
                self.refresh,
            )

    def _controls(self, parent: QWidget) -> QWidget:
        box = QGroupBox("Sharpness", parent)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        v.addWidget(QLabel("Interval:", box), 0)
        v.addWidget(self.cmb_int, 0)

        v.addStretch(1)
        return box

    def refresh(self) -> None:
        if not HAS_MPL:
            return
        if not self.plot.is_ready():
            return

        df = self._ctx.df
        qs = self._ctx.q_cols or []
        t = self._ctx.t



        if df is None or not len(df):
            _show_empty(
                self.plot,
                title="No data to plot",
                message="Select an active dataset to compute sharpness.",
            )
            return
    
        if not qs:
            _show_empty(
                self.plot,
                title="No quantiles found",
                message="Sharpness requires quantile columns (qXX).",
                hint="Load a probabilistic output file (q10/q50/q90...).",
            )
            return
    
        if not t or t not in df.columns:
            _show_empty(
                self.plot,
                title="Time column missing",
                message="Pick a Time column to group interval width over time.",
                hint="Set Map → Time col, then pick a time value if needed.",
            )
            return
        self.plot.clear_axes()
        ax = self.plot.ax
        assert ax is not None
        lo_q, hi_q = self._pick_interval(qs)
        
        if not (lo_q and hi_q):
            _show_empty(
                self.plot,
                title="Interval unavailable",
                message="Requested interval columns were not found.",
            )
            return
        
        d = df[[t, lo_q, hi_q]].dropna()
        
        if not len(d):
            _show_empty(
                self.plot,
                title="No rows after filtering",
                message="Nothing matched the current filters.",
            )
            return

        # Ensure that hi_q and lo_q are single columns
        d["w"] = (d[hi_q].iloc[:, 0] - d[lo_q].iloc[:, 0]).abs()  # This assumes both are DataFrames

        g = d.groupby(t)["w"].median()

        ax.plot(g.index.values, g.values)
        ax.set_xlabel(t)
        ax.set_ylabel("Median width")
        ax.set_title(f"{lo_q} .. {hi_q}")

        self.plot.draw()

    def _pick_interval(
        self,
        qs: list[tuple[float, str]],
    ) -> tuple[str, str]:
        want = str(self.cmb_int.currentText())

        mapping = {
            "50%": (0.25, 0.75),
            "80%": (0.10, 0.90),
            "90%": (0.05, 0.95),
        }
        a = b = None
        for k, (qa, qb) in mapping.items():
            if k in want:
                a, b = qa, qb

        if a is None or b is None:
            return "", ""

        lo = self._find_q(qs, a)
        hi = self._find_q(qs, b)
        return lo, hi

    def _find_q(
        self,
        qs: list[tuple[float, str]],
        q: float,
    ) -> str:
        best = ""
        err = 1e9
        for qq, col in qs:
            e = abs(float(qq) - float(q))
            if e < err:
                err = e
                best = col
        return best


class ReliabilityTab(_TabBase):
    """
    PIT + coverage if observation exists.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.ed_obs = QLineEdit(self)
        self.ed_obs.setReadOnly(True)

        self.plot = MplPlot(
            title="Reliability / PIT",
            parent=self,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        box = QGroupBox("Inputs", self)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        v.addWidget(QLabel("Obs column:", box), 0)
        v.addWidget(self.ed_obs, 0)
        v.addStretch(1)

        root.addWidget(box, 0)
        root.addWidget(self.plot, 1)

        if self.plot.actions is not None:
            self.plot.actions.btn_refresh.clicked.connect(
                self.refresh,
            )
            self.plot.actions.btn_clear.clicked.connect(
                self.plot.clear_axes,
            )

    def refresh(self) -> None:
        if not HAS_MPL:
            return
        if not self.plot.is_ready():
            return

        df = self._ctx.df
        qs = self._ctx.q_cols or []
        obs = self._ctx.obs_col or ""

        self.ed_obs.setText(obs)



        if df is None or not len(df):
            _show_empty(
                self.plot,
                title="No data to plot",
                message="Select an active dataset to compute reliability.",
            )
            return
    
        if not obs or obs not in df.columns:
            _show_empty(
                self.plot,
                title="No observation column",
                message="Reliability needs an observation/truth column.",
                hint="Provide a column like obs / truth / target (depending on your export).",
            )
            return
    
        if not qs:
            _show_empty(
                self.plot,
                title="No quantiles found",
                message="PIT/reliability requires quantile columns (qXX).",
            )
            return
        
        self.plot.clear_axes()
        ax = self.plot.ax
        assert ax is not None
        
        # PIT: fraction of quantiles below obs
        # (simple discrete PIT)
        cols = [c for _, c in qs if c in df.columns]
        d = df[[obs] + cols].dropna()
        if not len(d):
          _show_empty(
              self.plot,
              title="No valid rows",
              message="No rows had both obs and quantiles available.",
          )
          return

        y = d[obs].values
        qmat = d[cols].values
        pit = (qmat < y[:, None]).mean(axis=1)

        ax.hist(pit, bins=20)
        ax.set_title("PIT histogram")
        ax.set_xlabel("PIT")
        ax.set_ylabel("Count")

        self.plot.draw()


class InspectorTab(_TabBase):
    """
    Inspector v1:
    - Group by (X, Y)
    - Plot time series for one point
    - Draw fan (quantile bands) if available
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._pts: list[tuple[float, float]] = []
        self._pts_key: tuple[str, str, str] = ("", "", "")

        self.sp_pt = QSpinBox(self)
        self.sp_pt.setRange(0, 0)

        self.lb_xy = QLabel("-", self)
        self.lb_xy.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.btn_rebuild = QToolButton(self)
        self.btn_rebuild.setText("↻")
        self.btn_rebuild.setToolTip("Rebuild point list")
        self.btn_rebuild.setAutoRaise(True)

        self.sp_max = QSpinBox(self)
        self.sp_max.setRange(1000, 2000000)
        self.sp_max.setValue(80000)

        self.sp_tol = QDoubleSpinBox(self)
        self.sp_tol.setRange(0.0, 1e6)
        self.sp_tol.setDecimals(9)
        self.sp_tol.setSingleStep(1e-6)
        self.sp_tol.setValue(1e-6)

        self.chk_fan = QCheckBox("Fan", self)
        self.chk_fan.setChecked(True)

        self.cmb_band = QComboBox(self)
        self.cmb_band.addItems([
            "80% (q10-q90)",
            "50% (q25-q75)",
            "90% (q05-q95)",
        ])

        self.chk_mid = QCheckBox("Median line", self)
        self.chk_mid.setChecked(True)

        self.plot = MplPlot(
            title="Point time series",
            parent=self,
        )

        self._build_ui()
        self._connect()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        box = QGroupBox("Point", self)
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        r0 = QHBoxLayout()
        r0.setContentsMargins(0, 0, 0, 0)
        r0.setSpacing(6)
        r0.addWidget(QLabel("Index:", box), 0)
        r0.addWidget(self.sp_pt, 1)
        r0.addWidget(self.btn_rebuild, 0)
        v.addLayout(r0)

        v.addWidget(QLabel("X/Y:", box), 0)
        v.addWidget(self.lb_xy, 0)

        v.addWidget(QLabel("Max points:", box), 0)
        v.addWidget(self.sp_max, 0)

        v.addWidget(QLabel("Match tol:", box), 0)
        v.addWidget(self.sp_tol, 0)

        v.addSpacing(8)
        v.addWidget(self.chk_fan, 0)
        v.addWidget(self.cmb_band, 0)
        v.addWidget(self.chk_mid, 0)

        v.addStretch(1)

        root.addWidget(box, 0)
        root.addWidget(self.plot, 1)

    def _connect(self) -> None:
        self.sp_pt.valueChanged.connect(self.refresh)
        self.sp_max.valueChanged.connect(self._rebuild_pts)
        self.sp_tol.valueChanged.connect(self.refresh)
        self.chk_fan.toggled.connect(self.refresh)
        self.cmb_band.currentIndexChanged.connect(self.refresh)
        self.chk_mid.toggled.connect(self.refresh)
        self.btn_rebuild.clicked.connect(self._rebuild_pts)

        if self.plot.actions is not None:
            self.plot.actions.btn_refresh.clicked.connect(
                self.refresh,
            )
            self.plot.actions.btn_clear.clicked.connect(
                self.plot.clear_axes,
            )

    # -------------------------
    # Context
    # -------------------------
    def set_context(self, ctx: MapAnaCtx) -> None:
        self._ctx = ctx
        self._ensure_pts()
        self.refresh()

    def _ensure_pts(self) -> None:
        """
        Rebuild point list when:
        - file changed
        - x/y mapping changed
        """
        p = str(self._ctx.path or "")
        x = str(self._ctx.x or "")
        y = str(self._ctx.y or "")
        key = (p, x, y)

        if key != self._pts_key:
            self._pts_key = key
            self._rebuild_pts()

    def _rebuild_pts(self) -> None:
        self._pts = []
        self.lb_xy.setText("-")
        self.sp_pt.setRange(0, 0)
        self.sp_pt.setValue(0)

        path = self._ctx.path
        x = self._ctx.x
        y = self._ctx.y

        if path is None or not path.exists():
            return
        if not (x and y):
            return

        pts = self._scan_unique_xy(
            path=path,
            x=x,
            y=y,
            limit=int(self.sp_max.value()),
        )
        self._pts = pts

        if not self._pts:
            return

        self.sp_pt.setRange(0, len(self._pts) - 1)
        self.sp_pt.setValue(0)
        self._update_xy_label()

    def _update_xy_label(self) -> None:
        if not self._pts:
            self.lb_xy.setText("-")
            return

        i = int(self.sp_pt.value())
        i = max(0, min(i, len(self._pts) - 1))
        x0, y0 = self._pts[i]
        self.lb_xy.setText(f"{x0:.6f}, {y0:.6f}")

    # -------------------------
    # Data access
    # -------------------------
    def _scan_unique_xy(
        self,
        *,
        path: Path,
        x: str,
        y: str,
        limit: int,
    ) -> list[tuple[float, float]]:
        """
        Chunked unique scan for (x,y).

        Stops when reaching `limit`.
        """
        out: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()

        use = [x, y]
        try:
            it = pd.read_csv(
                path,
                usecols=use,
                chunksize=200000,
            )
        except Exception:
            return out

        for ch in it:
            if x not in ch.columns or y not in ch.columns:
                continue

            d = ch[[x, y]].dropna()
            if not len(d):
                continue

            d = d.drop_duplicates()
            for row in d.itertuples(index=False):
                try:
                    xx = float(row[0])
                    yy = float(row[1])
                except Exception:
                    continue
                key = (xx, yy)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
                if len(out) >= int(limit):
                    return out

        return out

    def _load_point_series(
        self,
        *,
        path: Path,
        x: str,
        y: str,
        t: str,
        z: str,
        qs: list[tuple[float, str]],
        x0: float,
        y0: float,
        tol: float,
    ) -> Optional[pd.DataFrame]:
        """
        Chunked filter for a single (x0,y0).
        """
        cols: list[str] = []
        for c in (x, y, t, z):
            if c and c not in cols:
                cols.append(c)

        for _, qc in (qs or []):
            if qc and qc not in cols:
                cols.append(qc)

        try:
            it = pd.read_csv(
                path,
                usecols=cols,
                chunksize=200000,
            )
        except Exception:
            return None

        parts: list[pd.DataFrame] = []
        for ch in it:
            if x not in ch.columns or y not in ch.columns:
                continue

            dx = (ch[x] - float(x0)).abs()
            dy = (ch[y] - float(y0)).abs()
            m = (dx <= float(tol)) & (dy <= float(tol))

            sub = ch.loc[m]
            if len(sub):
                parts.append(sub)

        if not parts:
            return None

        df = pd.concat(parts, axis=0, ignore_index=True)
        return df

    # -------------------------
    # Plot
    # -------------------------
    def refresh(self) -> None:
        if not HAS_MPL:
            return
        if not self.plot.is_ready():
            return

        self._update_xy_label()

        df0 = self._ctx.df
        path = self._ctx.path

        x = self._ctx.x
        y = self._ctx.y
        t = self._ctx.t
        z = self._ctx.z
        qs = self._ctx.q_cols or []


        if path is None or not path.exists():
            _show_empty(
                self.plot,
                title="No dataset selected",
                message="Select an active file to inspect point time series.",
                hint="Map → Data panel → choose an active dataset.",
            )
            return
        
        if not (x and y and t and z):
            _show_empty(
                self.plot,
                title="Mapping incomplete",
                message="Inspector needs X/Y + Time col + Z to plot a point series.",
                hint="Pick columns in the header (X/Y/Z) and set Time col.",
            )
            return
        
        if not self._pts:
            _show_empty(
                self.plot,
                title="No points found",
                message="Could not build the (X,Y) point list from this file.",
                hint="Try increasing Max points or verify X/Y columns.",
            )
            return

        self.plot.clear_axes()
        ax = self.plot.ax
        assert ax is not None
        
        i = int(self.sp_pt.value())
        i = max(0, min(i, len(self._pts) - 1))
        x0, y0 = self._pts[i]

        tol = float(self.sp_tol.value())
        df = self._load_point_series(
            path=path,
            x=x,
            y=y,
            t=t,
            z=z,
            qs=qs,
            x0=x0,
            y0=y0,
            tol=tol,
        )

        if df is None or not len(df):
            _show_empty(
                self.plot,
                title="No rows for this point",
                message="No rows matched the selected (X,Y) within tolerance.",
                hint="Increase Match tol or click ↻ to rebuild points.",
            )
            return

        if t not in df.columns or z not in df.columns:
            self.plot.draw()
            return

        d = df.dropna(subset=[t, z]).copy()
        if not len(d):
            self.plot.draw()
            return

        # Sort by time
        try:
            d[t] = d[t].astype(float)
        except Exception:
            pass
        d = d.sort_values(t)

        xs = d[t].values
        yz = d[z].values

        ax.plot(xs, yz, linewidth=1.5)
        ax.set_xlabel(t)
        ax.set_ylabel(z)

        title = f"({x0:.4f}, {y0:.4f})"
        ax.set_title(title)

        if self.chk_fan.isChecked():
            self._draw_fan(ax=ax, d=d, t=t, qs=qs)

        self.plot.draw()

    def _draw_fan(
        self,
        *,
        ax,
        d: pd.DataFrame,
        t: str,
        qs: list[tuple[float, str]],
    ) -> None:
        if not qs:
            return

        lo, hi = self._pick_band(qs)
        if not lo or not hi:
            return
        if lo not in d.columns or hi not in d.columns:
            return

        dd = d.dropna(subset=[t, lo, hi])
        if not len(dd):
            return

        x = dd[t].values
        y0 = dd[lo].values
        y1 = dd[hi].values

        ax.fill_between(
            x,
            y0,
            y1,
            alpha=0.25,
        )

        if self.chk_mid.isChecked():
            mid = self._find_mid(qs)
            if mid and mid in dd.columns:
                ym = dd[mid].values
                ax.plot(x, ym, linewidth=1.2)

    def _pick_band(
        self,
        qs: list[tuple[float, str]],
    ) -> tuple[str, str]:
        s = str(self.cmb_band.currentText())

        mapping = {
            "80%": (0.10, 0.90),
            "50%": (0.25, 0.75),
            "90%": (0.05, 0.95),
        }
        qa = qb = None
        for k, (a, b) in mapping.items():
            if k in s:
                qa, qb = a, b

        if qa is None or qb is None:
            return "", ""

        lo = self._find_q(qs, qa)
        hi = self._find_q(qs, qb)
        return lo, hi

    def _find_mid(
        self,
        qs: list[tuple[float, str]],
    ) -> str:
        return self._find_q(qs, 0.50)

    def _find_q(
        self,
        qs: list[tuple[float, str]],
        q: float,
    ) -> str:
        best = ""
        err = 1e9
        for qq, col in qs:
            e = abs(float(qq) - float(q))
            if e < err:
                err = e
                best = col
        return best

    def set_xy(self, x: float, y: float) -> None:
        """
        Jump to nearest point in the current _pts list.
        """
        if not self._pts:
            return
    
        try:
            x0 = float(x)
            y0 = float(y)
        except Exception:
            return
    
        best_i = 0
        best_d = float("inf")
    
        for i, (xx, yy) in enumerate(self._pts):
            try:
                dx = float(xx) - x0
                dy = float(yy) - y0
            except Exception:
                continue
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_i = i
    
        # triggers refresh via valueChanged
        self.sp_pt.setValue(int(best_i))

def _show_empty(
    plot: MplPlot,
    *,
    title: str,
    message: str,
    hint: str = "",
) -> None:
    """
    Render a clean empty-state (no axes/ticks) inside the plot area.
    """
    if not HAS_MPL:
        return
    if not plot.is_ready():
        return

    plot.clear_axes()
    ax = plot.ax
    if ax is None:
        return

    ax.set_axis_off()

    ax.text(
        0.5, 0.58, title,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=11,
        fontweight="bold",
    )

    body = message if not hint else f"{message}\n\n{hint}"
    ax.text(
        0.5, 0.40, body,
        transform=ax.transAxes,
        ha="center", va="center",
        wrap=True,
    )
    plot.draw()
    
def _pick_1d(df: pd.DataFrame, key: str) -> pd.Series:
    """
    Return a 1D Series for column key.

    If df has duplicate column labels, df.loc[:, key] returns a DataFrame;
    we keep the first occurrence to stay 1D.
    """
    col = df.loc[:, key]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col
