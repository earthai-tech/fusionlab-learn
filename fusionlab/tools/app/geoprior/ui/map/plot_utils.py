# geoprior/ui/map/plot_utils.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.plot_utils

Small matplotlib helpers for Map analytics panel.

Design goals
------------
- Safe imports (fallback if matplotlib missing)
- Reusable Qt plot widget
- Small plotting helpers (2D/3D, hist, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Iterable

try:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as _Canvas,
    )
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as _Toolbar,
    )
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    _Canvas = None  # type: ignore
    _Toolbar = None  # type: ignore
    Figure = None  # type: ignore

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


HAS_MPL = _Canvas is not None and Figure is not None


@dataclass
class PlotActionBar:
    """
    Tiny action bar for plot panels.
    """

    btn_refresh: QToolButton
    btn_save: QToolButton
    btn_clear: QToolButton


class MplPlot(QFrame):
    """
    Matplotlib plot widget for Qt.

    Provides:
    - fig/ax access
    - small action bar (refresh/save/clear)
    - safe fallback when matplotlib missing
    """

    def __init__(
        self,
        *,
        title: str = "",
        with_toolbar: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)

        self._title = str(title or "")
        self._with_tb = bool(with_toolbar)

        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar = None

        self._placeholder = None
        self.actions: Optional[PlotActionBar] = None

        self._build_ui()

    def is_ready(self) -> bool:
        return bool(HAS_MPL and self.canvas is not None)

    def clear_axes(self) -> None:
        if not self.is_ready():
            return
        assert self.ax is not None
        self.ax.clear()
        self.canvas.draw_idle()

    def save_png(self, path: str) -> None:
        if not self.is_ready():
            return
        assert self.fig is not None
        self.fig.savefig(path, dpi=160, bbox_inches="tight")

    def draw(self) -> None:
        if not self.is_ready():
            return
        self.canvas.draw_idle()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        if self._title:
            lb = QLabel(self._title, self)
            lb.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lb.setStyleSheet(
                "QLabel{font-weight:600;}"
            )
            root.addWidget(lb, 0)

        bar = self._make_actions(self)
        root.addWidget(bar, 0)

        if not HAS_MPL:
            self._placeholder = QLabel(
                "Matplotlib not available.\n"
                "Install: pip install matplotlib",
                self,
            )
            self._placeholder.setAlignment(Qt.AlignCenter)
            root.addWidget(self._placeholder, 1)
            return

        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.canvas = _Canvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        if self._with_tb and _Toolbar is not None:
            self.toolbar = _Toolbar(self.canvas, self)
            root.addWidget(self.toolbar, 0)

        root.addWidget(self.canvas, 1)

    def _make_actions(self, parent: QWidget) -> QWidget:
        w = QWidget(parent)
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        btn_refresh = QToolButton(w)
        btn_refresh.setText("↻")
        btn_refresh.setToolTip("Refresh plot")
        btn_refresh.setAutoRaise(True)

        btn_save = QToolButton(w)
        btn_save.setText("💾")
        btn_save.setToolTip("Save PNG")
        btn_save.setAutoRaise(True)

        btn_clear = QToolButton(w)
        btn_clear.setText("✖")
        btn_clear.setToolTip("Clear")
        btn_clear.setAutoRaise(True)

        row.addStretch(1)
        row.addWidget(btn_refresh)
        row.addWidget(btn_save)
        row.addWidget(btn_clear)

        self.actions = PlotActionBar(
            btn_refresh=btn_refresh,
            btn_save=btn_save,
            btn_clear=btn_clear,
        )
        return w



def safe_quantile_cols(cols: Optional[Iterable[object]]) -> list[Tuple[float, str]]:
    """
    Find quantile columns ending with `_qXX` (e.g. 'head_q50', 'subs_q10').

    Returns
    -------
    list of (q, col)
        Sorted by q ascending.
    """
    out: list[Tuple[float, str]] = []

    cols_iter = cols if cols is not None else []
    for c in cols_iter:
        s = str(c)
        lo = s.lower()
        if "_q" not in lo:
            continue

        tail = lo.split("_q")[-1]
        # allow things like q05, q5, q50
        try:
            q = float(tail) / 100.0
        except Exception:
            continue

        if 0.0 <= q <= 1.0:
            out.append((q, s))

    out.sort(key=lambda x: x[0])
    return out

def pick_obs_col(cols: Optional[Iterable[object]]) -> str:
    """
    Heuristic: try to find an observed/true column.

    Examples it may match:
    - *_obs, *_true, *_target
    - *_subsidence
    """
    cands = ["obs", "true", "target", "y", "subsidence"]

    cols_iter = cols if cols is not None else []
    low = {str(c).lower(): str(c) for c in cols_iter}

    for k in cands:
        suf = f"_{k}"
        for name, orig in low.items():
            if name.endswith(suf):
                return orig
    return ""
