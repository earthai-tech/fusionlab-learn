# geoprior/ui/map/selection_plot.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

"""
selection_plot

Small Matplotlib widget + helpers for the SelectionPanel.

- Point mode: plot (t vs z) for one id.
- Group mode: plot group trend (t vs mid) + optional p10/p90 band.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure


def _as_numeric(s: pd.Series) -> pd.Series:
    try:
        out = pd.to_numeric(s, errors="coerce")
        return out
    except Exception:
        return pd.Series([], dtype="float64")


def _pick_col(df: pd.DataFrame, want: str) -> str:
    w = str(want or "").strip()
    if w and w in df.columns:
        return w
    if "t" in df.columns and w.lower().startswith("t"):
        return "t"
    if "v" in df.columns and w.lower().startswith(("z", "v")):
        return "v"
    return w


class SelectionPlot(QWidget):
    """
    Tiny Matplotlib plot widget for SelectionPanel.

    Methods:
      - clear(msg="")
      - plot_point(df, t_col, z_col, band=(lo, hi))
      - plot_group(trend, t_col)
    """

    def __init__(
        self,
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("gpSelPlot")
        self.fig = Figure(figsize=(4.0, 2.4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.canvas, 1)

        self.clear("")

    def clear(self, msg: str = "") -> None:
        self.ax.clear()
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        m = str(msg or "").strip()
        if m:
            self.ax.text(
                0.5,
                0.5,
                m,
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw_idle()

    def plot_point(
        self,
        df: pd.DataFrame,
        *,
        t_col: str,
        z_col: str,
        band: Optional[Tuple[str, str]] = None,
    ) -> None:
        if df is None or df.empty:
            self.clear("No series")
            return

        tc = _pick_col(df, t_col)
        zc = _pick_col(df, z_col)

        if tc not in df.columns or zc not in df.columns:
            self.clear("Missing columns")
            return

        d = df[[tc, zc]].copy()
        d[tc] = pd.to_numeric(d[tc], errors="coerce")
        d[zc] = pd.to_numeric(d[zc], errors="coerce")
        d = d.dropna(subset=[tc, zc])
        if d.empty:
            self.clear("Empty series")
            return

        d = d.sort_values(tc, kind="mergesort")

        self.ax.clear()
        self.ax.grid(True, alpha=0.25)

        x = d[tc].to_numpy()
        y = d[zc].to_numpy()
        self.ax.plot(x, y)

        if band is not None:
            lo, hi = band
            lo = str(lo or "").strip()
            hi = str(hi or "").strip()
            if lo in df.columns and hi in df.columns:
                dd = df[[tc, lo, hi]].copy()
                dd[tc] = pd.to_numeric(
                    dd[tc],
                    errors="coerce",
                )
                dd[lo] = pd.to_numeric(
                    dd[lo],
                    errors="coerce",
                )
                dd[hi] = pd.to_numeric(
                    dd[hi],
                    errors="coerce",
                )
                dd = dd.dropna(subset=[tc, lo, hi])
                if not dd.empty:
                    dd = dd.sort_values(
                        tc,
                        kind="mergesort",
                    )
                    self.ax.fill_between(
                        dd[tc].to_numpy(),
                        dd[lo].to_numpy(),
                        dd[hi].to_numpy(),
                        alpha=0.18,
                    )

        self.ax.set_xlabel(str(tc))
        self.ax.set_ylabel(str(zc))
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw_idle()

    def plot_group(
        self,
        trend: pd.DataFrame,
        *,
        t_col: str,
    ) -> None:
        if trend is None or trend.empty:
            self.clear("No trend")
            return

        tc = _pick_col(trend, t_col)
        if tc not in trend.columns or "mid" not in trend.columns:
            self.clear("Missing trend cols")
            return

        d = trend[[tc, "mid"]].copy()
        d[tc] = pd.to_numeric(d[tc], errors="coerce")
        d["mid"] = pd.to_numeric(d["mid"], errors="coerce")
        d = d.dropna(subset=[tc, "mid"])
        if d.empty:
            self.clear("Empty trend")
            return

        d = d.sort_values(tc, kind="mergesort")

        self.ax.clear()
        self.ax.grid(True, alpha=0.25)

        self.ax.plot(d[tc].to_numpy(), d["mid"].to_numpy())

        # Optional spread band if present
        if "p10" in trend.columns and "p90" in trend.columns:
            dd = trend[[tc, "p10", "p90"]].copy()
            dd[tc] = pd.to_numeric(dd[tc], errors="coerce")
            dd["p10"] = pd.to_numeric(
                dd["p10"],
                errors="coerce",
            )
            dd["p90"] = pd.to_numeric(
                dd["p90"],
                errors="coerce",
            )
            dd = dd.dropna(subset=[tc, "p10", "p90"])
            if not dd.empty:
                dd = dd.sort_values(tc, kind="mergesort")
                self.ax.fill_between(
                    dd[tc].to_numpy(),
                    dd["p10"].to_numpy(),
                    dd["p90"].to_numpy(),
                    alpha=0.18,
                )

        self.ax.set_xlabel(str(tc))
        self.ax.set_ylabel("mid")
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw_idle()
