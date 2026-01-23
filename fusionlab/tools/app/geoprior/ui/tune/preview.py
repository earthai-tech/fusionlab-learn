# geoprior/ui/tune/preview.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from math import log10
from typing import Any, Dict, Optional, Tuple

from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QPainterPath
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QStyle,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..icon_utils import try_icon
from ...config.prior_schema import FieldKey
from ...config.store import GeoConfigStore

from .plan import build_plan_text, _get_fk, space_stats


__all__ = ["TunePreviewPanel"]


class TunePreviewViz(QWidget):
    """
    Small, readable mini-viz for Tune.

    Bars are normalized against reasonable reference caps so
    they are informative (not always 100%).

    - max_trials: cap at 200
    - epochs: cap at 300
    - search-space: log-scaled active/total
    """

    _TRIALS_CAP = 200.0
    _EPOCHS_CAP = 300.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._max_trials = 0
        self._epochs = 0
        self._space_a = 0
        self._space_t = 0
        self.setMinimumHeight(160)

    def set_values(
        self,
        *,
        max_trials: int,
        epochs: int,
        space_a: int,
        space_t: int,
    ) -> None:
        self._max_trials = int(max_trials)
        self._epochs = int(epochs)
        self._space_a = int(space_a)
        self._space_t = int(space_t)
        self.update()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    @staticmethod
    def _log_ratio(a: int, t: int) -> float:
        # log-scaled ratio: log10(1+a) / log10(1+t)
        aa = max(0, int(a))
        tt = max(1, int(t))
        try:
            num = log10(1.0 + float(aa))
            den = log10(1.0 + float(tt))
            if den <= 0:
                return 0.0
            return num / den
        except Exception:
            return 0.0

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())

        pad = 12.0
        x0 = pad
        x1 = w - pad
        y = pad + 16.0

        def bar(label: str, frac: float) -> None:
            nonlocal y
            frac = self._clip01(frac)

            p.setPen(QPen(self.palette().text().color()))
            p.drawText(int(x0), int(y), label)

            y2 = y + 10.0
            bw = (x1 - x0)
            bh = 10.0

            r = QRectF(x0, y2, bw, bh)
            path = QPainterPath()
            path.addRoundedRect(r, 5.0, 5.0)

            p.setPen(Qt.NoPen)
            p.setBrush(self.palette().midlight())
            p.drawPath(path)

            rf = QRectF(x0, y2, bw * frac, bh)
            pathf = QPainterPath()
            pathf.addRoundedRect(rf, 5.0, 5.0)

            p.setBrush(self.palette().highlight())
            p.drawPath(pathf)

            y = y2 + bh + 18.0

        trials_frac = float(self._max_trials) / self._TRIALS_CAP
        epochs_frac = float(self._epochs) / self._EPOCHS_CAP
        space_frac = self._log_ratio(self._space_a, self._space_t)

        bar(f"max_trials  ({self._max_trials})", trials_frac)
        bar(f"epochs  ({self._epochs})", epochs_frac)
        bar(
            f"space active  ({self._space_a}/{self._space_t})",
            space_frac,
        )


class TunePreviewPanel(QFrame):
    """
    Right-side preview panel for Tune.

    - Run plan (text) + copy
    - Mini viz (bars)
    - Tuning overview (grouped tree)
    """

    toast = pyqtSignal(str)

    # Optional actions for the overview header
    reset_space_clicked = pyqtSignal()
    export_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._plan_text = ""

        self.setObjectName("runPreviewPanel")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()
        self.refresh_from_store()

    def _std_icon(self, sp: QStyle.StandardPixmap):
        return self.style().standardIcon(sp)

    def _set_icon(
        self,
        btn: QToolButton,
        name: str,
        fallback: QStyle.StandardPixmap,
    ) -> None:
        ic = try_icon(name)
        if ic is None:
            ic = self._std_icon(fallback)
        btn.setIcon(ic)

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # -------------------------
        # Run plan header
        # -------------------------
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(8)

        t = QLabel("Run plan")
        t.setObjectName("subTitle")

        self.btn_copy = QToolButton(self)
        self.btn_copy.setAutoRaise(True)
        self.btn_copy.setToolTip("Copy run plan")
        self._set_icon(
            self.btn_copy,
            "copy.svg",
            QStyle.SP_DialogSaveButton,
        )

        hdr.addWidget(t, 0)
        hdr.addStretch(1)
        hdr.addWidget(self.btn_copy, 0)
        root.addLayout(hdr)

        self.lbl_plan = QLabel("")
        self.lbl_plan.setObjectName("runPlanText")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_plan, 0)

        self.viz = TunePreviewViz(self)
        root.addWidget(self.viz, 0)

        # -------------------------
        # Overview header
        # -------------------------
        ov_hdr = QHBoxLayout()
        ov_hdr.setContentsMargins(0, 0, 0, 0)
        ov_hdr.setSpacing(8)

        ov_t = QLabel("Tuning overview")
        ov_t.setObjectName("subTitle")

        self.btn_reset = QPushButton("Reset space", self)
        self.btn_reset.setObjectName("miniAction")
        ic = try_icon("reset.svg")
        if ic is not None:
            self.btn_reset.setIcon(ic)

        self.btn_export = QPushButton("Export…", self)
        self.btn_export.setObjectName("miniAction")
        ic = try_icon("export.svg")
        if ic is not None:
            self.btn_export.setIcon(ic)

        ov_hdr.addWidget(ov_t, 0)
        ov_hdr.addStretch(1)
        ov_hdr.addWidget(self.btn_reset, 0)
        ov_hdr.addWidget(self.btn_export, 0)
        root.addLayout(ov_hdr)

        # quick hints line (3 labels)
        hints = QHBoxLayout()
        hints.setContentsMargins(0, 0, 0, 0)
        hints.setSpacing(12)

        self.lbl_space_keys = QLabel("Keys: -")
        self.lbl_trial_hint = QLabel("Trial: -")
        self.lbl_device_hint = QLabel("Device: -")

        self.lbl_space_keys.setObjectName("sumLine")
        self.lbl_trial_hint.setObjectName("sumLine")
        self.lbl_device_hint.setObjectName("sumLine")

        hints.addWidget(self.lbl_space_keys, 0)
        hints.addWidget(self.lbl_trial_hint, 0)
        hints.addWidget(self.lbl_device_hint, 0)
        hints.addStretch(1)
        root.addLayout(hints)

        # filter
        self.ed_overview_filter = QLineEdit(self)
        self.ed_overview_filter.setPlaceholderText(
            "Filter hyperparameters…"
        )
        root.addWidget(self.ed_overview_filter, 0)

        # grouped tree
        self.tree_space_preview = QTreeWidget(self)
        self.tree_space_preview.setColumnCount(2)
        self.tree_space_preview.setHeaderLabels(
            ["Hyperparameter", "Search space"]
        )
        self.tree_space_preview.setRootIsDecorated(True)
        self.tree_space_preview.setAlternatingRowColors(True)
        self.tree_space_preview.setUniformRowHeights(True)
        self.tree_space_preview.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self.tree_space_preview.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )

        hdrw = self.tree_space_preview.header()
        hdrw.setStretchLastSection(True)
        hdrw.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdrw.setSectionResizeMode(1, QHeaderView.Stretch)

        root.addWidget(self.tree_space_preview, 1)

        # -------------------------
        # Wiring
        # -------------------------
        self.btn_copy.clicked.connect(self._on_copy)
        self.btn_reset.clicked.connect(self.reset_space_clicked.emit)
        self.btn_export.clicked.connect(self.export_clicked.emit)
        self.ed_overview_filter.textChanged.connect(
            lambda _=None: self._apply_tree_filter()
        )

    # -----------------------------------------------------------------
    # Refresh
    # -----------------------------------------------------------------
    def refresh_from_store(self) -> None:
        txt = build_plan_text(self._store)
        self._plan_text = txt
        self.lbl_plan.setText(txt)

        space = _get_fk(self._store, "tuner_search_space", {})
        a, t = space_stats(space)

        mt = int(_get_fk(self._store, "tuner_max_trials", 20))
        ep = int(_get_fk(self._store, "epochs", 0))

        self.viz.set_values(
            max_trials=mt,
            epochs=ep,
            space_a=a,
            space_t=t,
        )

        # overview header hints + tree
        if not isinstance(space, dict):
            space = {}
        self._refresh_overview(dict(space))

    # -----------------------------------------------------------------
    # Copy
    # -----------------------------------------------------------------
    def _on_copy(self) -> None:
        txt = (self._plan_text or "").strip()
        if not txt:
            return
        QApplication.clipboard().setText(txt)
        self.toast.emit("Tune plan copied.")

    # -----------------------------------------------------------------
    # Overview helpers
    # -----------------------------------------------------------------
    def _get_space(self) -> Dict[str, Any]:
        try:
            val = self._store.get_value(
                FieldKey("tuner_search_space"),
                default={},
            )
        except Exception:
            val = {}
        return dict(val) if isinstance(val, dict) else {}

    @staticmethod
    def _pretty_name(key: str) -> str:
        mapping = {
            "embed_dim": "Embedding dim",
            "hidden_units": "Hidden units",
            "lstm_units": "LSTM units",
            "attention_units": "Attention units",
            "num_heads": "Attention heads",
            "vsn_units": "VSN units",
            "dropout_rate": "Dropout",
            "pde_mode": "PDE mode(s)",
            "kappa_mode": "k mode (bar/kb)",
            "hd_factor": "HD factor",
            "learning_rate": "Learning rate",
            "lambda_cons": "Lambda (cons)",
            "lambda_gw": "Lambda (gw)",
            "lambda_prior": "Lambda (prior)",
            "lambda_smooth": "Lambda (smooth)",
            "lambda_bounds": "Lambda (bounds)",
            "lambda_offset": "Lambda offset",
            "lambda_q": "Lambda (q)",
            "memory_size": "Memory size",
            "max_window_size": "Max window size",
            "attention_levels": "Attention levels",
            "scale_pde_residuals": "Scale PDE residuals",
        }
        return mapping.get(key, key.replace("_", " ").title())

    def _fmt_value_pretty(self, v: Any) -> Tuple[str, str]:
        tip = ""

        if isinstance(v, list):
            s = ", ".join(str(x) for x in v)
            if len(s) > 60:
                tip = s
                s = s[:57] + "…"
            return s, tip

        if isinstance(v, dict):
            tip = str(v)
            t = str(v.get("type", "")).lower()

            if t in ("float", "int", "range"):
                vmin = v.get("min_value", v.get("min"))
                vmax = v.get("max_value", v.get("max"))
                step = v.get("step")
                samp = v.get("sampling")

                s = f"{vmin}–{vmax}"
                if step is not None:
                    s += f"  (Δ {step})"
                if samp:
                    s += f"  [{samp}]"
                return s, tip

            if t == "choice":
                vals = v.get("values", [])
                s = ", ".join(str(x) for x in vals)
                if len(s) > 60:
                    tip = s
                    s = s[:57] + "…"
                return s, tip

            if t in ("bool", "boolean"):
                return "True / False", tip

            return str(v), tip

        if isinstance(v, bool):
            return ("True" if v else "False"), ""

        return str(v), ""

    def _refresh_overview(self, space: Dict[str, Any]) -> None:
        try:
            keys_n = len(space)
        except Exception:
            keys_n = 0

        epochs = _get_fk(self._store, "epochs", None)
        batch = _get_fk(self._store, "batch_size", None)
        dev = _get_fk(self._store, "tf_device_mode", "auto")

        self.lbl_space_keys.setText(f"Keys: {keys_n}")
        self.lbl_trial_hint.setText(
            f"Trial: epochs={epochs}  batch={batch}"
        )
        self.lbl_device_hint.setText(f"Device: {dev}")

        self._populate_overview_tree(space)
        self._apply_tree_filter()

    def _populate_overview_tree(self, space: Dict[str, Any]) -> None:
        tree = self.tree_space_preview
        tree.setUpdatesEnabled(False)
        try:
            tree.clear()

            groups = [
                (
                    "Architecture",
                    [
                        "embed_dim",
                        "hidden_units",
                        "lstm_units",
                        "attention_units",
                        "num_heads",
                        "vsn_units",
                        "dropout_rate",
                        "attention_levels",
                    ],
                ),
                (
                    "Physics",
                    [
                        "pde_mode",
                        "kappa_mode",
                        "hd_factor",
                        "scale_pde_residuals",
                        "kappa",
                        "mv",
                    ],
                ),
                ("Optimization", ["learning_rate", "kappa_lr_mult", "mv_lr_mult"]),
                (
                    "Loss weights",
                    [
                        "lambda_cons",
                        "lambda_gw",
                        "lambda_prior",
                        "lambda_smooth",
                        "lambda_bounds",
                        "lambda_offset",
                        "lambda_q",
                        "scale_q_with_offset",
                        "scale_mv_with_offset",
                    ],
                ),
                ("Data / memory", ["max_window_size", "memory_size", "scales"]),
            ]

            used = set()
            idx = 1

            def _make_group(title: str, n: int) -> QTreeWidgetItem:
                it = QTreeWidgetItem([f"{title}  ({n})", ""])
                f = it.font(0)
                f.setBold(True)
                it.setFont(0, f)
                return it

            for title, keys in groups:
                present = [k for k in keys if k in space]
                if not present:
                    continue

                g = _make_group(title, len(present))
                tree.addTopLevelItem(g)

                for k in present:
                    used.add(k)
                    name = self._pretty_name(k)
                    disp, tip = self._fmt_value_pretty(space.get(k))
                    row = QTreeWidgetItem([f"[{idx:02d}] {name}", disp])
                    row.setData(0, Qt.UserRole, k)
                    if tip:
                        row.setToolTip(0, tip)
                        row.setToolTip(1, tip)
                    g.addChild(row)
                    idx += 1

                g.setExpanded(True)

            leftovers = [k for k in sorted(space.keys()) if k not in used]
            if leftovers:
                g = _make_group("Other", len(leftovers))
                tree.addTopLevelItem(g)
                for k in leftovers:
                    name = self._pretty_name(k)
                    disp, tip = self._fmt_value_pretty(space.get(k))
                    row = QTreeWidgetItem([f"[{idx:02d}] {name}", disp])
                    row.setData(0, Qt.UserRole, k)
                    if tip:
                        row.setToolTip(0, tip)
                        row.setToolTip(1, tip)
                    g.addChild(row)
                    idx += 1
                g.setExpanded(False)

        finally:
            tree.setUpdatesEnabled(True)

    def _apply_tree_filter(self) -> None:
        q = str(self.ed_overview_filter.text() or "").strip().lower()
        tree = self.tree_space_preview

        def _row_text(it: QTreeWidgetItem) -> str:
            return (it.text(0) + " " + it.text(1)).strip().lower()

        for i in range(tree.topLevelItemCount()):
            g = tree.topLevelItem(i)
            any_child = False

            for j in range(g.childCount()):
                ch = g.child(j)
                show = True
                if q:
                    show = q in _row_text(ch)
                ch.setHidden(not show)
                any_child = any_child or show

            # hide empty groups when filtering
            g.setHidden(bool(q) and (not any_child))
