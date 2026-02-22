# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Results & Downloads tab for GeoPrior GUI.

v3.2 UX:
- Root "Viewing: Config/Custom" chip
- Summary dashboard + small stacked bar
- Xfer table + optional matrix heatmap view
- Context menus: Copy path / Open / ZIP
- View-root browsing does NOT pollute cfg root
"""

from __future__ import annotations

import os
import zipfile
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import (
    Qt,
    QThread,
    QUrl,
    QSettings,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QDesktopServices,
    QGuiApplication,
)
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyledItemDelegate,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from .results_index import (
    CityResults,
    ResultsIndex,
    discover_results_for_root,
)
from ..styles import PRIMARY
from .icon_utils import try_icon

_HAS_MPL = False
try:
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
    )
    from matplotlib.figure import Figure

    _HAS_MPL = True
except Exception:
    FigureCanvas = None
    Figure = None


# ---------------------------------------------------------------------
# Zip worker thread
# ---------------------------------------------------------------------
class ZipWorker(QThread):
    """Background worker that zips a directory with progress."""

    progress_changed = pyqtSignal(int, int)  # done, total
    finished_ok = pyqtSignal(str)  # target path
    failed = pyqtSignal(str)  # error message

    def __init__(
        self,
        source_dir: Path,
        target_zip: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.source_dir = Path(source_dir)
        self.target_zip = Path(target_zip)

    def run(self) -> None:
        try:
            files: list[Path] = []
            for root, _, fns in os.walk(self.source_dir):
                for fn in fns:
                    files.append(Path(root) / fn)

            total = max(len(files), 1)
            done = 0

            self.target_zip.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with zipfile.ZipFile(
                self.target_zip,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as zf:
                for fp in files:
                    if self.isInterruptionRequested():
                        raise RuntimeError(
                            "Zipping cancelled by user"
                        )

                    rel = fp.relative_to(self.source_dir)
                    zf.write(str(fp), str(rel))
                    done += 1
                    self.progress_changed.emit(done, total)

            self.finished_ok.emit(str(self.target_zip))
        except Exception as e:
            self.failed.emit(str(e))


# ---------------------------------------------------------------------
# Small delegates / helpers
# ---------------------------------------------------------------------
class ElideRightDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index) -> None:
        super().initStyleOption(option, index)
        option.textElideMode = Qt.ElideRight


class _MplCanvas(QWidget):
    """Matplotlib canvas wrapper (safe fallback)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        if not _HAS_MPL:
            lab = QLabel("Matplotlib not available.")
            lab.setAlignment(Qt.AlignCenter)
            lay.addWidget(lab)
            self._ax = None
            self._fig = None
            self._canvas = None
            return

        fig = Figure(figsize=(4.2, 1.2))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        canvas.updateGeometry()

        lay.addWidget(canvas)
        self._fig = fig
        self._ax = ax
        self._canvas = canvas

    def clear(self) -> None:
        if self._ax is None:
            return
        self._ax.clear()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def ax(self):
        return self._ax

    def draw(self) -> None:
        if self._canvas is None:
            return
        self._canvas.draw_idle()


class ResultsSummaryPanel(QFrame):
    """Top dashboard: chips + stacked bar per city."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("resultsSummary")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)

        self.chip_cities = QLabel("Cities: 0")
        self.chip_runs = QLabel("Runs: 0")
        self.chip_xfer = QLabel("Xfer: 0")
        self.chip_newest = QLabel("Newest: —")

        for w in (
            self.chip_cities,
            self.chip_runs,
            self.chip_xfer,
            self.chip_newest,
        ):
            w.setObjectName("resultsChip")
            top.addWidget(w)

        top.addStretch(1)
        lay.addLayout(top)

        self.plots = QSplitter(Qt.Horizontal)
        self.plots.setChildrenCollapsible(True)
        
        self.canvas = _MplCanvas(self)
        self.plots.addWidget(self.canvas)
        
        self.sec = QFrame(self)
        self.sec.setObjectName("resultsInsights")
        sec_lay = QVBoxLayout(self.sec)
        sec_lay.setContentsMargins(0, 0, 0, 0)
        sec_lay.setSpacing(4)
        
        self.sec_hint = QLabel("Insights")
        self.sec_hint.setObjectName("resultsHint")
        sec_lay.addWidget(self.sec_hint)
        
        self.sec_canvas = _MplCanvas(self.sec)
        sec_lay.addWidget(self.sec_canvas)
        
        self.plots.addWidget(self.sec)
        self.sec.setVisible(False)
        
        lay.addWidget(self.plots)
        
        self._main_tip: dict[object, str] = {}
        self._sec_tip: dict[object, str] = {}
        
        self._ann_main = None
        self._ann_sec = None
        
        self._bind_hover(self.canvas, which="main")
        self._bind_hover(self.sec_canvas, which="sec")

    def set_secondary_visible(self, on: bool) -> None:
        self.sec.setVisible(bool(on))
    
        if on:
            self.plots.setSizes([700, 360])
        else:
            self.plots.setSizes([1, 0])

    def set_index(self, idx: Optional[ResultsIndex]) -> None:
        if idx is None:
            self._set_zero()
            return
        self._update_chips(idx)
        self._plot_city_stack(idx)
        self._plot_kind_mix(idx)

    def _set_zero(self) -> None:
        self.chip_cities.setText("Cities: 0")
        self.chip_runs.setText("Runs: 0")
        self.chip_xfer.setText("Xfer: 0")
        self.chip_newest.setText("Newest: —")
        self.canvas.clear()
        if hasattr(self, "sec_canvas"):
            self.sec_canvas.clear()
            
    def _plot_kind_mix(self, idx: ResultsIndex) -> None:
        ax = self.sec_canvas.ax()
        if ax is None:
            return
    
        cities = list(idx.cities.values())
    
        n_train = sum(len(c.train_runs) for c in cities)
        n_tune = sum(len(c.tune_runs) for c in cities)
        n_inf = sum(len(c.inference_runs) for c in cities)
        n_xfer = len(list(idx.xfer_runs))
    
        labels = ["train", "tune", "infer", "xfer"]
        vals = [n_train, n_tune, n_inf, n_xfer]
    
        keep = [(l, v) for l, v in zip(labels, vals) if v > 0]
        labels = [k[0] for k in keep]
        vals = [k[1] for k in keep]
    
        ax.clear()
        self._sec_tip.clear()
    
        if not vals:
            ax.set_axis_off()
            self.sec_canvas.draw()
            return
    
        wedges, _ = ax.pie(
            vals,
            startangle=90,
            wedgeprops={"width": 0.45},
        )
        ax.set_aspect("equal")

        # Reserve room for legend on the right
        self._apply_mpl_margins(
            self.sec_canvas,
            left=0.06,
            right=0.78,
            bottom=0.10,
            top=0.90,
        )
    
        ax.legend(
            wedges,
            [f"{l} ({v})" for l, v in zip(labels, vals)],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
            frameon=False,
        )
        ax.set_title("workflow mix", fontsize=10, pad=10)
        
        self.sec_canvas.draw()

    def _update_chips(self, idx: ResultsIndex) -> None:
        cities = list(idx.cities.values())
        n_city = len(cities)

        n_train = sum(len(c.train_runs) for c in cities)
        n_tune = sum(len(c.tune_runs) for c in cities)
        n_inf = sum(len(c.inference_runs) for c in cities)
        n_runs = n_train + n_tune + n_inf

        n_xfer = len(list(idx.xfer_runs))

        newest = self._newest_stamp(idx)

        self.chip_cities.setText(f"Cities: {n_city}")
        self.chip_runs.setText(f"Runs: {n_runs}")
        self.chip_xfer.setText(f"Xfer: {n_xfer}")
        self.chip_newest.setText(f"Newest: {newest}")

    def _newest_stamp(self, idx: ResultsIndex) -> str:
        stamps: list[str] = []
        for c in idx.cities.values():
            stamps += [r.stamp for r in c.train_runs]
            stamps += [r.stamp for r in c.tune_runs]
            stamps += [r.stamp for r in c.inference_runs]
        stamps += [r.stamp for r in idx.xfer_runs]
        if not stamps:
            return "—"
        return sorted(stamps)[-1]

    def _plot_city_stack(self, idx: ResultsIndex) -> None:
        ax = self.canvas.ax()
        if ax is None:
            return

        cities = sorted(
            idx.cities.values(),
            key=lambda x: x.city.lower(),
        )
        names = [c.city for c in cities]
        tr = [len(c.train_runs) for c in cities]
        tu = [len(c.tune_runs) for c in cities]
        inf = [len(c.inference_runs) for c in cities]

        ax.clear()
        if not names:
            ax.set_axis_off()
            self.canvas.draw()
            return

        xs = list(range(len(names)))
        c1 = ax.bar(xs, tr, label="train")
        c2 = ax.bar(xs, tu, bottom=tr, label="tune")
        
        bot = [tr[i] + tu[i] for i in range(len(xs))]
        c3 = ax.bar(xs, inf, bottom=bot, label="infer")
        
        self._main_tip.clear()
        for lab, cont in (("train", c1), ("tune", c2), ("infer", c3)):
            for i, p in enumerate(cont.patches):
                v = int(p.get_height())
                city = names[i] if i < len(names) else "?"
                self._main_tip[p] = f"{city} | {lab}: {v}"

        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=0)
        ax.set_ylabel("runs")
        ax.legend(loc="upper right", fontsize=8)
        ax.margins(x=0.02)
        ax.tick_params(axis="x", pad=6)
    
        # Ensure x labels are visible
        self._apply_mpl_margins(
            self.canvas,
            bottom=0.30,
            top=0.92,
            right=0.98,
            left=0.08,
        )
    
        self.canvas.draw()

    def _bind_hover(self, canvas: _MplCanvas, *, which: str) -> None:
        if not _HAS_MPL:
            return
        if getattr(canvas, "_canvas", None) is None:
            return
    
        c = canvas._canvas
    
        def on_mv(evt):
            self._on_hover(evt, which=which)
    
        c.mpl_connect("motion_notify_event", on_mv)
    

    def _apply_mpl_margins(
        self,
        canvas: _MplCanvas,
        *,
        left: float = 0.08,
        right: float = 0.98,
        bottom: float = 0.22,
        top: float = 0.92,
    ) -> None:
        """
        Apply consistent Matplotlib figure margins.
    
        Parameters
        ----------
        canvas:
            The _MplCanvas wrapper (main or secondary).
        left, right, bottom, top:
            Figure fractions in [0..1]. Keep right smaller (e.g. 0.78)
            when you have a legend anchored outside the axes.
        """
        fig = getattr(canvas, "_fig", None)
        if fig is None:
            return
        try:
            fig.subplots_adjust(
                left=left,
                right=right,
                bottom=bottom,
                top=top,
            )
        except Exception:
            return
    
    def _on_hover(self, evt, *, which: str) -> None:
        if evt.inaxes is None:
            self._hide_ann(which)
            return
    
        tip_map = self._main_tip if which == "main" else self._sec_tip
        if not tip_map:
            self._hide_ann(which)
            return
    
        hit = None
        for p in tip_map.keys():
            ok, _ = p.contains(evt)
            if ok:
                hit = p
                break
    
        if hit is None:
            self._hide_ann(which)
            return
    
        txt = tip_map.get(hit, "")
        ax = evt.inaxes
        ann = self._ensure_ann(ax, which=which)
    
        ann.xy = (evt.xdata, evt.ydata)
        ann.set_text(txt)
        ann.set_visible(True)
    
        if which == "main":
            self.canvas.draw()
        else:
            self.sec_canvas.draw()
    
    
    def _ensure_ann(self, ax, *, which: str):
        if which == "main":
            ann = self._ann_main
        else:
            ann = self._ann_sec
    
        if ann is None:
            ann = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec="0.5",
                    alpha=0.9,
                ),
            )
            ann.set_visible(False)
    
            if which == "main":
                self._ann_main = ann
            else:
                self._ann_sec = ann
    
        return ann
    
    
    def _hide_ann(self, which: str) -> None:
        ann = self._ann_main if which == "main" else self._ann_sec
        if ann is None:
            return
        if not ann.get_visible():
            return
        ann.set_visible(False)
    
        if which == "main":
            self.canvas.draw()
        else:
            self.sec_canvas.draw()

class XferMatrixView(QFrame):
    """Heatmap matrix: CityA x CityB => count."""

    pair_clicked = pyqtSignal(str, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("xferMatrix")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self.hint = QLabel("Hover/click a cell.")
        self.hint.setObjectName("resultsHint")
        lay.addWidget(self.hint)

        self.canvas = _MplCanvas(self)
        lay.addWidget(self.canvas)

        self._names: list[str] = []
        self._mat: list[list[int]] = []

        self._bind_mpl_events()

    def _bind_mpl_events(self) -> None:
        if not _HAS_MPL:
            return
        if self.canvas._canvas is None:
            return

        c = self.canvas._canvas
        c.mpl_connect("motion_notify_event", self._on_mv)
        c.mpl_connect("button_press_event", self._on_click)

    def set_index(self, idx: Optional[ResultsIndex]) -> None:
        if idx is None:
            self._names = []
            self._mat = []
            self.canvas.clear()
            self.hint.setText("No data.")
            return

        self._build(idx)
        self._plot()

    def _build(self, idx: ResultsIndex) -> None:
        runs = list(idx.xfer_runs)
        names = set()
        for r in runs:
            names.add(r.city_a)
            names.add(r.city_b)
        self._names = sorted(names)

        n = len(self._names)
        mat = [[0 for _ in range(n)] for _ in range(n)]
        pos = {c: i for i, c in enumerate(self._names)}

        for r in runs:
            i = pos.get(r.city_a, -1)
            j = pos.get(r.city_b, -1)
            if i >= 0 and j >= 0:
                mat[i][j] += 1

        self._mat = mat

    def _plot(self) -> None:
        ax = self.canvas.ax()
        if ax is None:
            return

        ax.clear()

        if not self._names:
            ax.set_axis_off()
            self.canvas.draw()
            return

        ax.imshow(self._mat)
        ax.set_xticks(range(len(self._names)))
        ax.set_yticks(range(len(self._names)))
        ax.set_xticklabels(self._names, rotation=45, ha="right")
        ax.set_yticklabels(self._names)

        ax.set_title("xfer runs count")
        self.canvas.draw()

    def _on_mv(self, evt) -> None:
        if evt.xdata is None or evt.ydata is None:
            return
        i = int(round(evt.ydata))
        j = int(round(evt.xdata))
        if i < 0 or j < 0:
            return
        if i >= len(self._names) or j >= len(self._names):
            return
        a = self._names[i]
        b = self._names[j]
        n = self._mat[i][j]
        self.hint.setText(f"{a} → {b}: {n} run(s)")

    def _on_click(self, evt) -> None:
        if evt.xdata is None or evt.ydata is None:
            return
        i = int(round(evt.ydata))
        j = int(round(evt.xdata))
        if i < 0 or j < 0:
            return
        if i >= len(self._names) or j >= len(self._names):
            return
        a = self._names[i]
        b = self._names[j]
        self.pair_clicked.emit(a, b)


# ---------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------
class ResultsDownloadTab(QWidget):
    """
    Browse GeoPrior results and download jobs as ZIP.

    Notes
    -----
    - "view root" can be custom without touching cfg root.
    - Refresh resets view root to cfg root.
    """

    def __init__(
        self,
        *,
        results_root: Path | str,
        get_results_root: Optional[Callable[[], Path | str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._get_results_root = get_results_root
        self._results_root = Path(results_root).expanduser().resolve()

        self._index: Optional[ResultsIndex] = None

        self._zip_worker: ZipWorker | None = None
        self._zip_progress: QProgressDialog | None = None

        self._settings = QSettings("fusionlab", "geoprior")

        self._view_root: Path | None = None
        self._root_hist: list[str] = []
        self._last_scan: str = "—"

        self._build_ui()
        self._connect_signals()

        self._root_hist = self._load_hist()
        last = self._load_last_view()
        if last is not None:
            self._view_root = last

        self.refresh_index(use_cfg_root=False)

    # ------------------------------------------------------------------
    # Styling helpers
    # ------------------------------------------------------------------
    def _table_qss(self) -> str:
        return "\n".join(
            [
                "QTableWidget{",
                "background: palette(base);",
                "border: 1px solid rgba(0,0,0,35);",
                "border-radius: 10px;",
                "selection-background-color: rgba(0,0,0,18);",
                "}",
                "QHeaderView::section{",
                "background: palette(window);",
                "padding: 6px 8px;",
                "border: none;",
                "border-bottom: 1px solid rgba(0,0,0,35);",
                "font-weight: 600;",
                "}",
                "QTableWidget::item{",
                "padding: 6px 8px;",
                "}",
                "QTableWidget::item:hover{",
                "background: rgba(0,0,0,10);",
                "}",
                "QTableWidget::item:selected{",
                f"border-left: 3px solid {PRIMARY};",
                "}",
            ]
        )

    def _icon_btn_qss(self) -> str:
        return "\n".join(
            [
                "QToolButton{",
                "border: none;",
                "padding: 4px 6px;",
                "}",
                "QToolButton:hover{",
                "background: rgba(0,0,0,12);",
                "border-radius: 10px;",
                "}",
                "QToolButton:pressed{",
                "background: rgba(0,0,0,18);",
                "border-radius: 10px;",
                "}",
            ]
        )

    def _style_table(self, t: QTableWidget) -> None:
        t.setShowGrid(False)
        t.setWordWrap(False)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.setSelectionMode(QTableWidget.SingleSelection)
        t.verticalHeader().setVisible(False)
        t.setFocusPolicy(Qt.NoFocus)
        t.setStyleSheet(self._table_qss())

    def _make_dl_btn(self, tip: str) -> QToolButton:
        btn = QToolButton(self)
        btn.setAutoRaise(True)
        btn.setToolTip(tip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(self._icon_btn_qss())
        icon = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        btn.setIcon(icon)
        return btn

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        self._build_root_row(root)
        self._build_filter_row(root)
        self._build_summary(root)
        self._build_main_split(root)

        self._style_table(self.cities_table)
        self._style_table(self.xfer_table)
        self._style_table(self.details_table)

        self.details_table.setItemDelegateForColumn(
            2,
            ElideRightDelegate(self.details_table),
        )

        self.details_table.setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self.xfer_table.setContextMenuPolicy(
            Qt.CustomContextMenu
        )

    def _build_root_row(self, root: QVBoxLayout) -> None:
        top = QHBoxLayout()
        top.setSpacing(8)

        top.addWidget(QLabel("Results root:"))

        self.root_combo = QComboBox()
        self.root_combo.setEditable(True)
        self.root_combo.setInsertPolicy(QComboBox.NoInsert)

        le = self.root_combo.lineEdit()
        if le is not None:
            le.setObjectName("resultsRootEdit")

        top.addWidget(self.root_combo, 1)

        self.root_chip = QLabel("Viewing: —")
        self.root_chip.setObjectName("resultsRootChip")
        top.addWidget(self.root_chip)

        self.scan_chip = QLabel("Last scan: —")
        self.scan_chip.setObjectName("resultsScanChip")
        top.addWidget(self.scan_chip)

        self.browse_root_btn = QPushButton("Browse…")
        top.addWidget(self.browse_root_btn)

        self.refresh_btn = QPushButton("Refresh")
        top.addWidget(self.refresh_btn)

        root.addLayout(top)

    def _build_filter_row(self, root: QVBoxLayout) -> None:
        flt = QHBoxLayout()
        flt.setSpacing(8)
    
        def _icon_or_std(
                svg: str, std: QStyle.StandardPixmap
                ):
            ico = try_icon(svg)
            if ico is None:
                ico = self.style().standardIcon(std)
            return ico
    
        self.filter_edit = QLineEdit()
        self.filter_edit.setObjectName("resultsFilter")
        self.filter_edit.setPlaceholderText(
            "Filter city / job / stamp..."
        )
        # -------------------------------------------------
        # Leading filter icon (SVG first, fallback to Qt std)
        # -------------------------------------------------
        filter_ico = _icon_or_std(
            "filter2.svg",
            QStyle.SP_FileDialogContentsView,  # decent “filter/search” fallback
        )
        act = self.filter_edit.addAction(
            filter_ico,
            QLineEdit.LeadingPosition,
        )
        act.setToolTip("Filter city / job / stamp")
        act.triggered.connect(self.filter_edit.setFocus)
    
        flt.addWidget(self.filter_edit, 1)
    
        self.clear_filter_btn = QToolButton()
        self.clear_filter_btn.setObjectName("miniAction")
        self.clear_filter_btn.setAutoRaise(True)
        self.clear_filter_btn.setToolTip("Clear filter")
        self.clear_filter_btn.setCursor(Qt.PointingHandCursor)
        self.clear_filter_btn.setStyleSheet(self._icon_btn_qss())
        self.clear_filter_btn.setIcon(
            self.style().standardIcon(QStyle.SP_DialogCloseButton)
        )
        flt.addWidget(self.clear_filter_btn)
    
        self.insights_btn = QToolButton()
        self.insights_btn.setText("Insights")
        self.insights_btn.setObjectName("miniAction")
        self.insights_btn.setAutoRaise(True)
        self.insights_btn.setCheckable(True)
        self.insights_btn.setToolTip("Toggle insights plot")
        self.insights_btn.setCursor(Qt.PointingHandCursor)
        self.insights_btn.setStyleSheet(self._icon_btn_qss())
        self.insights_btn.setIcon(
            self.style().standardIcon(QStyle.SP_MessageBoxInformation)
        )
        self.insights_btn.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
    
        flt.addWidget(self.insights_btn)
    
        root.addLayout(flt)

    def _build_summary(self, root: QVBoxLayout) -> None:
        self.summary = ResultsSummaryPanel(self)
        root.addWidget(self.summary)

    def _build_main_split(self, root: QVBoxLayout) -> None:
        main = QSplitter(Qt.Vertical)
        root.addWidget(main, 1)

        top = QSplitter(Qt.Horizontal)
        main.addWidget(top)

        # left: cities
        left_box = QGroupBox("Cities && workflows")
        left_lay = QVBoxLayout(left_box)

        self.cities_table = QTableWidget()
        self.cities_table.setObjectName("resultsTable")
        self.cities_table.setColumnCount(5)
        self.cities_table.setHorizontalHeaderLabels(
            ["City", "Artifacts", "Train", "Tune", "Inference"]
        )

        self.cities_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )
        for c in range(1, 5):
            self.cities_table.horizontalHeader().setSectionResizeMode(
                c,
                QHeaderView.ResizeToContents,
            )

        left_lay.addWidget(self.cities_table)
        top.addWidget(left_box)

        # right: xfer (table + matrix toggle)
        xfer_box = QGroupBox("Transferability runs")
        xfer_lay = QVBoxLayout(xfer_box)

        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        self.xfer_toggle = QToolButton()
        self.xfer_toggle.setObjectName("miniAction")
        self.xfer_toggle.setText("Matrix")
        self.xfer_toggle.setCheckable(True)
        self.xfer_toggle.setToolTip("Toggle table / matrix")
        hdr.addWidget(self.xfer_toggle)

        hdr.addStretch(1)
        xfer_lay.addLayout(hdr)

        self.xfer_stack = QStackedWidget()
        xfer_lay.addWidget(self.xfer_stack)

        self.xfer_table = QTableWidget()
        self.xfer_table.setObjectName("resultsTable")
        self.xfer_table.setColumnCount(4)
        self.xfer_table.setHorizontalHeaderLabels(
            ["City A", "City B", "Timestamp", "Download"]
        )
        for c in range(4):
            self.xfer_table.horizontalHeader().setSectionResizeMode(
                c,
                QHeaderView.ResizeToContents,
            )

        self.xfer_matrix = XferMatrixView(self)

        self.xfer_stack.addWidget(self.xfer_table)
        self.xfer_stack.addWidget(self.xfer_matrix)

        top.addWidget(xfer_box)

        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 1)

        # bottom: details
        details_box = QGroupBox("Details")
        dlay = QVBoxLayout(details_box)

        self.details_label = QLabel(
            "Select a city + workflow to see jobs."
        )
        self.details_label.setTextFormat(Qt.RichText)
        dlay.addWidget(self.details_label)

        self.details_table = QTableWidget()
        self.details_table.setObjectName("resultsTable")
        self.details_table.setColumnCount(4)
        self.details_table.setHorizontalHeaderLabels(
            ["Type", "Job", "Path", "Download"]
        )

        self.details_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )
        self.details_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents,
        )
        self.details_table.horizontalHeader().setSectionResizeMode(
            2,
            QHeaderView.Stretch,
        )
        self.details_table.horizontalHeader().setSectionResizeMode(
            3,
            QHeaderView.ResizeToContents,
        )

        dlay.addWidget(self.details_table)
        main.addWidget(details_box)

        main.setStretchFactor(0, 1)
        main.setStretchFactor(1, 2)

    def _connect_signals(self) -> None:
        self.browse_root_btn.clicked.connect(self._on_browse_root)

        self.refresh_btn.clicked.connect(
            lambda: self.refresh_index(use_cfg_root=True)
        )

        self.filter_edit.textChanged.connect(self._apply_filter)

        self.clear_filter_btn.clicked.connect(
            lambda: self.filter_edit.setText("")
        )

        self.cities_table.cellClicked.connect(
            self._on_cities_cell_clicked
        )

        self.root_combo.activated.connect(self._on_root_combo)

        le = self.root_combo.lineEdit()
        if le is not None:
            le.returnPressed.connect(self._on_root_entered)

        self.details_table.customContextMenuRequested.connect(
            self._on_details_menu
        )

        self.xfer_table.customContextMenuRequested.connect(
            self._on_xfer_menu
        )

        self.xfer_toggle.toggled.connect(self._on_xfer_toggle)

        self.xfer_matrix.pair_clicked.connect(
            self._on_matrix_pair_clicked
        )
        self.insights_btn.toggled.connect(
            self._on_insights_toggled
        )

    # ------------------------------------------------------------------
    # Root behavior: cfg root vs view root
    # ------------------------------------------------------------------
    def _cfg_root(self) -> Path:
        if self._get_results_root is None:
            return self._results_root
        try:
            raw = self._get_results_root()
        except Exception:
            return self._results_root
        return Path(raw).expanduser().resolve()

    def _current_root(self) -> Path:
        if self._view_root is not None:
            return self._view_root
        return self._cfg_root()
    
    def _on_insights_toggled(self, on: bool) -> None:
        self.summary.set_secondary_visible(on)

    def _on_root_combo(self, *_args) -> None:
        txt = self.root_combo.currentText().strip()
        self._apply_root_text(txt)

    def _on_root_entered(self) -> None:
        txt = self.root_combo.currentText().strip()
        self._apply_root_text(txt)

    def _apply_root_text(self, txt: str) -> None:
        if not txt:
            return
        p = Path(txt).expanduser()
        if not p.is_dir():
            QMessageBox.warning(
                self,
                "Invalid folder",
                f"Not a folder:\n{p}",
            )
            return
        self._set_view_root(p)
        self.refresh_index(use_cfg_root=False)

    def _on_browse_root(self) -> None:
        start = str(self._current_root())
        path = QFileDialog.getExistingDirectory(
            self,
            "Select folder to view",
            start,
        )
        if not path:
            return
        self._set_view_root(Path(path))
        self.refresh_index(use_cfg_root=False)

    # ------------------------------------------------------------------
    # Persist view roots
    # ------------------------------------------------------------------
    def _load_hist(self) -> list[str]:
        key = "results.view_roots"
        v = self._settings.value(key, [])
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for s in v:
            if isinstance(s, str) and s.strip():
                out.append(s.strip())
        return out

    def _save_hist(self) -> None:
        key = "results.view_roots"
        self._settings.setValue(key, self._root_hist)

    def _load_last_view(self) -> Optional[Path]:
        key = "results.last_view_root"
        v = self._settings.value(key, "")
        if not isinstance(v, str):
            return None
        p = Path(v).expanduser()
        if p.is_dir():
            return p.resolve()
        return None

    def _save_last_view(self, root: Path) -> None:
        key = "results.last_view_root"
        self._settings.setValue(key, str(root))

    def _set_view_root(self, root: Path) -> None:
        root = Path(root).expanduser().resolve()
        self._view_root = root

        s = str(root)
        if s not in self._root_hist:
            self._root_hist.insert(0, s)
            self._root_hist = self._root_hist[:12]
            self._save_hist()

        self._save_last_view(root)
        self._rebuild_root_combo(select=s)

    def _rebuild_root_combo(self, *, select: str) -> None:
        cfg = str(self._cfg_root())
        items: list[str] = [cfg]
        for s in self._root_hist:
            if s not in items:
                items.append(s)

        self.root_combo.blockSignals(True)
        self.root_combo.clear()
        self.root_combo.addItems(items)

        idx = self.root_combo.findText(select)
        if idx < 0:
            self.root_combo.insertItem(0, select)
            idx = 0

        self.root_combo.setCurrentIndex(idx)
        self.root_combo.blockSignals(False)

    def _update_root_chip(self) -> None:
        cur = self._current_root()
        cfg = self._cfg_root()
        mode = "Config" if cur == cfg else "Custom"

        self.root_chip.setText(f"Viewing: {mode}")
        self.root_chip.setProperty("mode", mode.lower())

        self.scan_chip.setText(f"Last scan: {self._last_scan}")

        self.root_chip.style().unpolish(self.root_chip)
        self.root_chip.style().polish(self.root_chip)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def _filter_table(
        self,
        t: QTableWidget,
        key: str,
        cols: list[int],
    ) -> None:
        key = (key or "").strip().lower()
        for r in range(t.rowCount()):
            if not key:
                t.setRowHidden(r, False)
                continue
            hit = False
            for c in cols:
                it = t.item(r, c)
                if it is None:
                    continue
                if key in it.text().lower():
                    hit = True
                    break
            t.setRowHidden(r, not hit)

    def _apply_filter(self, text: str) -> None:
        key = text or ""
        self._filter_table(self.cities_table, key, [0])
        self._filter_table(self.xfer_table, key, [0, 1, 2])
        self._filter_table(self.details_table, key, [0, 1, 2])

    # ------------------------------------------------------------------
    # Discovery + populate
    # ------------------------------------------------------------------
    def refresh_index(self, use_cfg_root: bool = True) -> None:
        if use_cfg_root:
            cfg = self._cfg_root()
            self._view_root = cfg
            self._root_hist = self._load_hist()
            self._rebuild_root_combo(select=str(cfg))
        else:
            if not self._root_hist:
                self._root_hist = self._load_hist()
            vr = self._view_root or self._cfg_root()
            self._rebuild_root_combo(select=str(vr))

        root = self._current_root()
        self._index = discover_results_for_root(root)

        self._last_scan = datetime.now().strftime("%H:%M:%S")
        self._update_root_chip()

        self._populate_cities_table()
        self._populate_xfer_table()
        self._clear_details()

        self.summary.set_index(self._index)
        self.xfer_matrix.set_index(self._index)

    def _populate_cities_table(self) -> None:
        self.cities_table.clearContents()
        if self._index is None:
            self.cities_table.setRowCount(0)
            return

        cities = sorted(
            self._index.cities.values(),
            key=lambda cr: cr.city.lower(),
        )
        self.cities_table.setRowCount(len(cities))

        for r, city_res in enumerate(cities):
            it_city = QTableWidgetItem(city_res.city)
            self.cities_table.setItem(r, 0, it_city)

            self._set_workflow_item(
                row=r,
                col=1,
                kind="artifacts",
                count=1 if city_res.artifacts_dir else 0,
            )
            self._set_workflow_item(
                row=r,
                col=2,
                kind="train",
                count=len(city_res.train_runs),
            )
            self._set_workflow_item(
                row=r,
                col=3,
                kind="tune",
                count=len(city_res.tune_runs),
            )
            self._set_workflow_item(
                row=r,
                col=4,
                kind="inference",
                count=len(city_res.inference_runs),
            )

        self.cities_table.resizeRowsToContents()

    def _set_workflow_item(
        self,
        *,
        row: int,
        col: int,
        kind: str,
        count: int,
    ) -> None:
        if count <= 0:
            it = QTableWidgetItem("—")
            it.setToolTip("No runs found")
            it.setData(Qt.UserRole, "")
        else:
            it = QTableWidgetItem(str(count))
            it.setToolTip("Click to browse")
            it.setData(Qt.UserRole, kind)

        it.setTextAlignment(Qt.AlignCenter)
        self.cities_table.setItem(row, col, it)

    def _populate_xfer_table(self) -> None:
        self.xfer_table.clearContents()
        if self._index is None:
            self.xfer_table.setRowCount(0)
            return

        runs = list(self._index.xfer_runs)
        self.xfer_table.setRowCount(len(runs))

        for row, r in enumerate(runs):
            base = f"{r.city_a}_to_{r.city_b}_{r.stamp}"

            it_a = QTableWidgetItem(r.city_a)
            it_b = QTableWidgetItem(r.city_b)

            it_s = QTableWidgetItem(r.stamp)
            it_s.setToolTip(str(r.run_dir))
            it_s.setData(
                Qt.UserRole,
                (str(r.run_dir), base),
            )

            self.xfer_table.setItem(row, 0, it_a)
            self.xfer_table.setItem(row, 1, it_b)
            self.xfer_table.setItem(row, 2, it_s)

            btn = self._make_dl_btn("Download ZIP")

            def on_click(_c=False, run=r, base=base) -> None:
                self._download_directory(run.run_dir, base)

            btn.clicked.connect(on_click)
            self.xfer_table.setCellWidget(row, 3, btn)

        self.xfer_table.resizeRowsToContents()

    def _on_xfer_toggle(self, on: bool) -> None:
        if on:
            self.xfer_toggle.setText("Table")
            self.xfer_stack.setCurrentIndex(1)
        else:
            self.xfer_toggle.setText("Matrix")
            self.xfer_stack.setCurrentIndex(0)

    def _on_matrix_pair_clicked(self, a: str, b: str) -> None:
        # Select first matching row in table
        for r in range(self.xfer_table.rowCount()):
            ia = self.xfer_table.item(r, 0)
            ib = self.xfer_table.item(r, 1)
            if ia is None or ib is None:
                continue
            if ia.text() == a and ib.text() == b:
                self.xfer_table.selectRow(r)
                self.xfer_table.scrollToItem(ia)
                if self.xfer_stack.currentIndex() != 0:
                    self.xfer_toggle.setChecked(False)
                break

    def _on_cities_cell_clicked(self, row: int, col: int) -> None:
        if col <= 0:
            return
        it_city = self.cities_table.item(row, 0)
        it_kind = self.cities_table.item(row, col)
        if it_city is None or it_kind is None:
            return

        kind = it_kind.data(Qt.UserRole) or ""
        if not kind:
            return

        self._show_city_details(it_city.text(), kind)

    # ------------------------------------------------------------------
    # Details view
    # ------------------------------------------------------------------
    def _clear_details(self) -> None:
        self.details_label.setText(
            "Select a city + workflow to see jobs."
        )
        self.details_table.setRowCount(0)
        self.details_table.clearContents()

    def _show_city_details(self, city: str, kind: str) -> None:
        if self._index is None:
            return
        city_res = self._index.cities.get(city)
        if city_res is None:
            return

        city_html = escape(city)

        rows: list[tuple[str, str, Path]] = []

        if kind == "artifacts":
            if city_res.artifacts_dir is not None:
                rows.append(
                    ("artifacts", "Stage-1 artifacts", city_res.artifacts_dir)
                )
        elif kind == "train":
            rows = [("train", r.stamp, r.run_dir) for r in city_res.train_runs]
        elif kind == "tune":
            rows = [("tune", r.stamp, r.run_dir) for r in city_res.tune_runs]
        elif kind == "inference":
            for r in city_res.inference_runs:
                ds = r.dataset or "?"
                lab = f"{r.stamp} (dataset={ds})"
                rows.append(("inference", lab, r.run_dir))

        self.details_table.setRowCount(len(rows))
        self.details_table.clearContents()

        if kind == "artifacts":
            title = (
                "Stage-1 artifacts for city: "
                f"<b>{city_html}</b>"
            )
        else:
            title = f"{kind.capitalize()} jobs for <b>{city_html}</b>"
        self.details_label.setText(title)

        for i, (typ, label, path) in enumerate(rows):
            base = label.replace(" ", "_")
            base = base.replace("[", "").replace("]", "")
            base = f"{city}_{typ}_{base}"

            self.details_table.setItem(i, 0, QTableWidgetItem(typ))
            self.details_table.setItem(i, 1, QTableWidgetItem(label))

            it_path = QTableWidgetItem(str(path))
            it_path.setToolTip(str(path))
            it_path.setData(Qt.UserRole, (str(path), base))
            self.details_table.setItem(i, 2, it_path)

            btn = self._make_dl_btn("Download ZIP")

            def on_click(_c=False, p=path, b=base) -> None:
                self._download_directory(p, b)

            btn.clicked.connect(on_click)
            self.details_table.setCellWidget(i, 3, btn)

        self.details_table.resizeRowsToContents()

    # ------------------------------------------------------------------
    # Context menus
    # ------------------------------------------------------------------
    def _copy_to_clip(self, text: str) -> None:
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(text)

    def _open_in_fs(self, p: Path) -> None:
        p = Path(p)
        if p.is_file():
            p = p.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _on_details_menu(self, pos) -> None:
        row = self.details_table.rowAt(pos.y())
        if row < 0:
            return
        it = self.details_table.item(row, 2)
        if it is None:
            return
        meta = it.data(Qt.UserRole)
        if not isinstance(meta, tuple) or len(meta) != 2:
            return

        path_s, base = meta
        p = Path(path_s)

        m = QMenu(self)
        a_copy = m.addAction("Copy path")
        a_open = m.addAction("Open folder")
        a_zip = m.addAction("Download ZIP")

        act = m.exec_(self.details_table.mapToGlobal(pos))
        if act == a_copy:
            self._copy_to_clip(str(p))
        elif act == a_open:
            if p.exists():
                self._open_in_fs(p)
        elif act == a_zip:
            self._download_directory(p, base)

    def _on_xfer_menu(self, pos) -> None:
        row = self.xfer_table.rowAt(pos.y())
        if row < 0:
            return
        it = self.xfer_table.item(row, 2)
        if it is None:
            return
        meta = it.data(Qt.UserRole)
        if not isinstance(meta, tuple) or len(meta) != 2:
            return

        path_s, base = meta
        p = Path(path_s)

        m = QMenu(self)
        a_copy = m.addAction("Copy path")
        a_open = m.addAction("Open folder")
        a_zip = m.addAction("Download ZIP")

        act = m.exec_(self.xfer_table.mapToGlobal(pos))
        if act == a_copy:
            self._copy_to_clip(str(p))
        elif act == a_open:
            if p.exists():
                self._open_in_fs(p)
        elif act == a_zip:
            self._download_directory(p, base)

    # ------------------------------------------------------------------
    # ZIP download
    # ------------------------------------------------------------------
    def _download_directory(self, run_dir: Path, base_label: str) -> None:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            QMessageBox.warning(
                self,
                "Missing directory",
                "The selected job directory does not exist:\n"
                f"{run_dir}",
            )
            return

        default_name = f"{base_label}.zip"
        suggested = str(Path.home() / default_name)

        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save job as ZIP",
            suggested,
            "ZIP archives (*.zip)",
        )
        if not target_path:
            return

        target = Path(target_path)
        if target.suffix.lower() != ".zip":
            target = target.with_suffix(".zip")

        progress = QProgressDialog(
            "Creating ZIP archive…",
            "Cancel",
            0,
            100,
            self,
        )
        progress.setWindowTitle("Zipping job")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setValue(0)

        worker = ZipWorker(run_dir, target, self)
        self._zip_worker = worker
        self._zip_progress = progress

        def on_progress(done: int, total: int) -> None:
            if total <= 0:
                progress.setRange(0, 0)
            else:
                progress.setRange(0, total)
                progress.setValue(done)

        def on_ok(path_str: str) -> None:
            progress.setValue(progress.maximum())
            progress.close()
            self._zip_worker = None
            self._zip_progress = None
            QMessageBox.information(
                self,
                "Download ready",
                "ZIP archive saved to:\n"
                f"{path_str}",
            )

        def on_fail(msg: str) -> None:
            progress.close()
            self._zip_worker = None
            self._zip_progress = None
            QMessageBox.critical(
                self,
                "Error while zipping",
                "Could not create archive:\n"
                f"{msg}",
            )

        def on_cancel() -> None:
            if self._zip_worker is not None:
                self._zip_worker.requestInterruption()

        worker.progress_changed.connect(on_progress)
        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        progress.canceled.connect(on_cancel)

        worker.start()
        progress.show()
