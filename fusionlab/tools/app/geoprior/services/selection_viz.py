# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

r"""Smart selection visualization for the DataTab.

Provides:
- Plot recipes (heuristics)
- Matplotlib pane
- Insights pane (plot + summary)
- Selection controller (debounced + sampled)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
    )
    from matplotlib.figure import Figure
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Matplotlib Qt backend is required for selection_viz."
    ) from exc


# ---------------------------------------------------------------------
# Plot recipes
# ---------------------------------------------------------------------
class PlotKind(str, Enum):
    NONE = "none"
    HIST_1D = "hist_1d"
    BAR_1D = "bar_1d"
    SCATTER_2D = "scatter_2d"
    LINE_2D = "line_2d"
    BOX_BY_CAT = "box_by_cat"
    HEATMAP_CATCAT = "heatmap_catcat"
    HEATMAP_CORR = "heatmap_corr"
    SUMMARY = "summary"


@dataclass(frozen=True)
class PlotRecipe:
    kind: PlotKind
    cols: Tuple[str, ...] = ()
    title: str = ""
    note: str = ""


# ---------------------------------------------------------------------
# Helpers (types, sampling, selection extraction)
# ---------------------------------------------------------------------
def _is_numeric(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(s.dtype)
    except Exception:
        return False


def _is_categorical(s: pd.Series) -> bool:
    try:
        if pd.api.types.is_bool_dtype(s.dtype):
            return True
        if pd.api.types.is_categorical_dtype(s.dtype):
            return True
        return pd.api.types.is_object_dtype(s.dtype)
    except Exception:
        return False


def _looks_like_time(name: str, s: pd.Series) -> bool:
    n = str(name).strip().lower()
    if "year" in n or "time" in n or "date" in n:
        return True
    try:
        if pd.api.types.is_datetime64_any_dtype(s.dtype):
            return True
    except Exception:
        pass
    if not _is_numeric(s):
        return False
    v = s.dropna()
    if v.empty:
        return False
    try:
        q1 = float(v.quantile(0.1))
        q9 = float(v.quantile(0.9))
    except Exception:
        return False
    if q1 >= 1800.0 and q9 <= 2200.0:
        return True
    return False


def _sample_range(
    start: int,
    stop: int,
    cap: int,
) -> List[int]:
    if cap <= 0:
        return []
    n = (stop - start) + 1
    if n <= cap:
        return list(range(start, stop + 1))
    step = max(1, n // cap)
    out = list(range(start, stop + 1, step))
    if out[-1] != stop:
        out.append(stop)
    if len(out) > cap:
        out = out[:cap]
    return out


def _dedup_sorted(vals: Iterable[int], cap: int) -> List[int]:
    out = sorted(set(int(v) for v in vals))
    if cap <= 0:
        return []
    if len(out) <= cap:
        return out
    step = max(1, len(out) // cap)
    samp = out[::step]
    if samp and samp[-1] != out[-1]:
        samp.append(out[-1])
    if len(samp) > cap:
        samp = samp[:cap]
    return samp


def _ranges_to_rows_cols(
    table: QTableView,
    *,
    row_cap: int,
    col_cap: int,
) -> Tuple[List[int], List[int]]:
    sel = table.selectionModel()
    if sel is None:
        return ([], [])
    qsel = sel.selection()
    if qsel is None:
        return ([], [])

    rows: List[int] = []
    cols: List[int] = []

    # QItemSelection is iterable over QItemSelectionRange.
    for r in qsel:
        try:
            rt = int(r.top())
            rb = int(r.bottom())
            cl = int(r.left())
            cr = int(r.right())
        except Exception:
            continue

        rows.extend(_sample_range(rt, rb, row_cap))
        cols.extend(_sample_range(cl, cr, col_cap))

    rows = _dedup_sorted(rows, row_cap)
    cols = _dedup_sorted(cols, col_cap)
    return (rows, cols)


def _df_from_table_model(
    table: QTableView,
) -> Optional[pd.DataFrame]:
    m = table.model()
    if m is None:
        return None
    for attr in ("df", "_df", "dataframe", "_dataframe"):
        try:
            v = getattr(m, attr)
        except Exception:
            v = None
        if isinstance(v, pd.DataFrame):
            return v
    return None


def _slice_df(
    df: pd.DataFrame,
    rows: List[int],
    cols: List[int],
) -> pd.DataFrame:
    if df.empty:
        return df
    if rows:
        df = df.iloc[rows, :]
    if cols:
        keep = []
        for i in cols:
            if 0 <= i < df.shape[1]:
                keep.append(df.columns[i])
        if keep:
            df = df.loc[:, keep]
    return df


# ---------------------------------------------------------------------
# Recipe inference + summary
# ---------------------------------------------------------------------
def infer_recipe(df: pd.DataFrame) -> PlotRecipe:
    if df is None or df.empty:
        return PlotRecipe(
            kind=PlotKind.NONE,
            title="No selection",
        )

    cols = list(df.columns)
    if not cols:
        return PlotRecipe(
            kind=PlotKind.NONE,
            title="No columns selected",
        )

    if len(cols) == 1:
        c0 = cols[0]
        s0 = df[c0]
        if _is_numeric(s0):
            return PlotRecipe(
                kind=PlotKind.HIST_1D,
                cols=(str(c0),),
                title=str(c0),
            )
        if _is_categorical(s0):
            return PlotRecipe(
                kind=PlotKind.BAR_1D,
                cols=(str(c0),),
                title=str(c0),
            )
        return PlotRecipe(
            kind=PlotKind.SUMMARY,
            cols=(str(c0),),
            title=str(c0),
        )

    # Prefer time-line when a time-like column exists.
    for c in cols[:3]:
        s = df[c]
        if _looks_like_time(str(c), s):
            # Time + numeric => line
            num = [x for x in cols if _is_numeric(df[x])]
            if num:
                y = str(num[0])
                return PlotRecipe(
                    kind=PlotKind.LINE_2D,
                    cols=(str(c), y),
                    title=f"{y} vs {c}",
                )

    if len(cols) == 2:
        c0, c1 = cols[0], cols[1]
        s0, s1 = df[c0], df[c1]
        if _is_numeric(s0) and _is_numeric(s1):
            return PlotRecipe(
                kind=PlotKind.SCATTER_2D,
                cols=(str(c0), str(c1)),
                title=f"{c1} vs {c0}",
            )
        if _is_categorical(s0) and _is_numeric(s1):
            return PlotRecipe(
                kind=PlotKind.BOX_BY_CAT,
                cols=(str(c0), str(c1)),
                title=f"{c1} by {c0}",
            )
        if _is_numeric(s0) and _is_categorical(s1):
            return PlotRecipe(
                kind=PlotKind.BOX_BY_CAT,
                cols=(str(c1), str(c0)),
                title=f"{c0} by {c1}",
            )
        if _is_categorical(s0) and _is_categorical(s1):
            return PlotRecipe(
                kind=PlotKind.HEATMAP_CATCAT,
                cols=(str(c0), str(c1)),
                title=f"{c1} x {c0}",
            )
        return PlotRecipe(
            kind=PlotKind.SUMMARY,
            cols=(str(c0), str(c1)),
            title="Mixed selection",
        )

    # 3+ columns: if many numeric -> corr heatmap.
    num_cols = [c for c in cols if _is_numeric(df[c])]
    if len(num_cols) >= 3:
        keep = num_cols[:12]
        return PlotRecipe(
            kind=PlotKind.HEATMAP_CORR,
            cols=tuple(str(x) for x in keep),
            title="Correlation",
            note="Top numeric columns",
        )

    return PlotRecipe(
        kind=PlotKind.SUMMARY,
        cols=tuple(str(x) for x in cols[:8]),
        title="Selection summary",
    )


def build_quick_summary(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No selection."
    nrows, ncols = df.shape
    num = [c for c in df.columns if _is_numeric(df[c])]
    cat = [c for c in df.columns if _is_categorical(df[c])]
    other = ncols - len(num) - len(cat)

    miss = 0.0
    try:
        miss = float(df.isna().mean().mean()) * 100.0
    except Exception:
        miss = 0.0

    parts = [
        f"Rows: {nrows}",
        f"Cols: {ncols}",
        f"Numeric: {len(num)}",
        f"Cat: {len(cat)}",
    ]
    if other > 0:
        parts.append(f"Other: {other}")
    parts.append(f"Missing: {miss:.1f}%")
    return " | ".join(parts)


# ---------------------------------------------------------------------
# Matplotlib widgets
# ---------------------------------------------------------------------
class MatplotlibPane(QWidget):
    """Simple Matplotlib canvas for recipes."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.fig = Figure(figsize=(5.0, 3.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.canvas, 1)

    def clear(self) -> None:
        self.fig.clear()

    def draw_recipe(
        self,
        recipe: PlotRecipe,
        df: pd.DataFrame,
    ) -> None:
        self.clear()
        ax = self.fig.add_subplot(111)

        if recipe.kind == PlotKind.NONE:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                recipe.title or "No selection",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            self.canvas.draw_idle()
            return

        if recipe.title:
            ax.set_title(recipe.title)

        try:
            self._draw_impl(ax, recipe, df)
        except Exception:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "Plot failed for this selection.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_impl(
        self,
        ax,
        recipe: PlotRecipe,
        df: pd.DataFrame,
    ) -> None:
        k = recipe.kind

        if k == PlotKind.HIST_1D:
            c = recipe.cols[0]
            s = df[c].dropna()
            ax.hist(s.values, bins=30)
            ax.set_xlabel(c)
            ax.set_ylabel("count")
            return

        if k == PlotKind.BAR_1D:
            c = recipe.cols[0]
            s = df[c].astype(str)
            vc = s.value_counts(dropna=False)
            vc = vc.head(20)
            ax.bar(vc.index.astype(str), vc.values)
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_xlabel(c)
            ax.set_ylabel("count")
            return

        if k == PlotKind.SCATTER_2D:
            x, y = recipe.cols[0], recipe.cols[1]
            d = df[[x, y]].dropna()
            ax.scatter(d[x].values, d[y].values, s=10)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            return

        if k == PlotKind.LINE_2D:
            x, y = recipe.cols[0], recipe.cols[1]
            d = df[[x, y]].dropna()
            d = d.sort_values(by=x, kind="mergesort")
            ax.plot(d[x].values, d[y].values)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            return

        if k == PlotKind.BOX_BY_CAT:
            cat, num = recipe.cols[0], recipe.cols[1]
            d = df[[cat, num]].dropna()
            d[cat] = d[cat].astype(str)
            top = d[cat].value_counts().head(12).index
            d = d[d[cat].isin(top)]
            groups = []
            labels = []
            for g in top:
                vals = d.loc[d[cat] == g, num].values
                if len(vals) > 0:
                    groups.append(vals)
                    labels.append(g)
            if not groups:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "Not enough data for boxplot.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return
            ax.boxplot(groups, labels=labels, showfliers=False)
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_xlabel(cat)
            ax.set_ylabel(num)
            return

        if k == PlotKind.HEATMAP_CATCAT:
            c0, c1 = recipe.cols[0], recipe.cols[1]
            d = df[[c0, c1]].dropna()
            d[c0] = d[c0].astype(str)
            d[c1] = d[c1].astype(str)

            a0 = d[c0].value_counts().head(20).index
            a1 = d[c1].value_counts().head(20).index
            d = d[d[c0].isin(a0) & d[c1].isin(a1)]

            tab = pd.crosstab(d[c1], d[c0])
            im = ax.imshow(tab.values, aspect="auto")
            ax.set_xticks(range(len(tab.columns)))
            ax.set_yticks(range(len(tab.index)))
            ax.set_xticklabels(tab.columns, rotation=45)
            ax.set_yticklabels(tab.index)
            self.fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xlabel(c0)
            ax.set_ylabel(c1)
            return

        if k == PlotKind.HEATMAP_CORR:
            cols = list(recipe.cols)
            d = df[cols].dropna()
            if d.empty:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "Not enough numeric rows for corr.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return
            corr = d.corr(numeric_only=True)
            im = ax.imshow(corr.values, vmin=-1.0, vmax=1.0)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.index)))
            ax.set_xticklabels(corr.columns, rotation=45)
            ax.set_yticklabels(corr.index)
            self.fig.colorbar(im, ax=ax, fraction=0.046)
            return

        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No plot for this selection.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


class SelectionInsightsPane(QFrame):
    """Plot + short summary."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.setSpacing(6)

        head = QHBoxLayout()
        self.lbl_title = QLabel("Auto-Insights")
        self.lbl_title.setStyleSheet("font-weight:600;")
        head.addWidget(self.lbl_title, 0)

        head.addStretch(1)

        self.lbl_meta = QLabel("")
        self.lbl_meta.setStyleSheet("opacity:0.8;")
        head.addWidget(self.lbl_meta, 0)

        outer.addLayout(head)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        outer.addWidget(self.lbl_summary, 0)

        self.plot = MatplotlibPane(self)
        outer.addWidget(self.plot, 1)

    def set_summary(
        self,
        text: str,
        *,
        meta: str = "",
    ) -> None:
        self.lbl_summary.setText(text or "")
        self.lbl_meta.setText(meta or "")


# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------
class SelectionVizController(QObject):
    """Debounced selection -> recipe -> plot."""

    def __init__(
        self,
        *,
        table: QTableView,
        pane: SelectionInsightsPane,
        df_getter: Optional[
            Callable[[], Optional[pd.DataFrame]]
        ] = None,
        debounce_ms: int = 150,
        max_rows: int = 5000,
        max_cols: int = 12,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)

        self._table = table
        self._pane = pane
        self._df_getter = df_getter

        self._debounce_ms = int(debounce_ms)
        self._max_rows = int(max_rows)
        self._max_cols = int(max_cols)

        self._enabled = True
        self._last_sig: Optional[Tuple[int, int]] = None

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.refresh_now)

        self._connect_selection()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            self._timer.stop()

    def set_table(self, table: QTableView) -> None:
        self._disconnect_selection()
        self._table = table
        self._connect_selection()

    def set_df_getter(
        self,
        getter: Optional[Callable[[], Optional[pd.DataFrame]]],
    ) -> None:
        self._df_getter = getter

    def schedule_refresh(self) -> None:
        if not self._enabled:
            return
        self._timer.start(self._debounce_ms)

    def refresh_now(self) -> None:
        if not self._enabled:
            return
        df = self._get_df()
        if df is None:
            self._render_empty("No data")
            return

        rows, cols = _ranges_to_rows_cols(
            self._table,
            row_cap=self._max_rows,
            col_cap=self._max_cols,
        )

        sub = _slice_df(df, rows, cols)

        sig = (len(rows), len(cols))
        if self._last_sig == sig and sub.empty:
            return
        self._last_sig = sig

        rec = infer_recipe(sub)
        summ = build_quick_summary(sub)
        meta = f"sel: {sig[0]}x{sig[1]}"

        self._pane.set_summary(summ, meta=meta)
        self._pane.plot.draw_recipe(rec, sub)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _get_df(self) -> Optional[pd.DataFrame]:
        if self._df_getter is not None:
            try:
                v = self._df_getter()
                if isinstance(v, pd.DataFrame):
                    return v
            except Exception:
                pass
        try:
            return _df_from_table_model(self._table)
        except Exception:
            return None

    def _render_empty(self, msg: str) -> None:
        self._pane.set_summary(msg, meta="")
        rec = PlotRecipe(kind=PlotKind.NONE, title=msg)
        self._pane.plot.draw_recipe(rec, pd.DataFrame())

    def _connect_selection(self) -> None:
        sel = self._table.selectionModel()
        if sel is None:
            return
        try:
            sel.selectionChanged.connect(self._on_changed)
        except Exception:
            return

    def _disconnect_selection(self) -> None:
        sel = self._table.selectionModel()
        if sel is None:
            return
        try:
            sel.selectionChanged.disconnect(self._on_changed)
        except Exception:
            return

    def _on_changed(self, *_args) -> None:
        self.schedule_refresh()
