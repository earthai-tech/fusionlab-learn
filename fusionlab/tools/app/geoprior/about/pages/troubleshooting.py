# geoprior/about/pages/troubleshooting.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import platform

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from ...config.prior_schema import FieldKey  
from ...ui.icon_utils import try_icon
from . import content
from ..widgets import make_card, wrap_scroll


def _try_store_value(win: object, key: str) -> Optional[object]:
    st = getattr(win, "store", None) or getattr(win, "_store", None)
    if st is None:
        return None

    try:
        return st.get_value(FieldKey(key), default=None)
    except Exception:
        pass

    for m in ("get_value", "get"):
        fn = getattr(st, m, None)
        if callable(fn):
            try:
                return fn(key)
            except Exception:
                pass

    return None


def _get_main_window(ctx: QWidget) -> Optional[QWidget]:
    dlg = ctx.window()
    if dlg is None:
        return None

    p = dlg.parent()
    if isinstance(p, QWidget):
        return p
    return None


def _get_results_root(win: QWidget) -> str:
    rr = _try_store_value(win, "results_root")
    if rr:
        return str(rr)

    dt = getattr(win, "data_tab", None)
    if dt is not None:
        rr2 = getattr(dt, "_results_root", None)
        if rr2:
            return str(rr2)

    rt = getattr(win, "results_tab", None)
    if rt is not None:
        rr3 = getattr(rt, "_results_root", None)
        if rr3:
            return str(rr3)

    return ""


def _get_dataset(win: QWidget) -> str:
    dt = getattr(win, "data_tab", None)
    if dt is None:
        return ""

    fn = getattr(dt, "current_csv_path", None)
    if callable(fn):
        try:
            p = fn()
            if p:
                return str(p)
        except Exception:
            pass

    p2 = getattr(dt, "_csv_path", None)
    if p2:
        return str(p2)

    return ""


def _get_manifest_and_model(win: QWidget) -> tuple[str, str]:
    it = getattr(win, "inference_tab", None)
    if it is None:
        return "", ""

    man = ""
    mdl = ""

    w_man = getattr(it, "inf_manifest_edit", None)
    if w_man is not None:
        try:
            man = w_man.text().strip()
        except Exception:
            pass

    w_mdl = getattr(it, "inf_model_edit", None)
    if w_mdl is not None:
        try:
            mdl = w_mdl.text().strip()
        except Exception:
            pass

    return man, mdl


def _tf_version_text() -> str:
    try:
        import tensorflow as tf  # type: ignore

        try:
            dev = tf.config.list_physical_devices("GPU")
            mode = "GPU" if dev else "CPU"
        except Exception:
            mode = "unknown"
        return f"{tf.__version__} ({mode})"
    except Exception:
        return "not installed"


def _collect_diag(ctx: QWidget) -> str:
    win = _get_main_window(ctx)

    rr = ""
    ds = ""
    man = ""
    mdl = ""

    city = ""
    if win is not None:
        rr = _get_results_root(win)
        ds = _get_dataset(win)
        man, mdl = _get_manifest_and_model(win)

        c = _try_store_value(win, "city")
        if c:
            city = str(c)

    lines = []
    lines.append("GeoPrior Forecaster diagnostics")
    lines.append(f"Timestamp: {datetime.utcnow().isoformat()}Z")

    if city:
        lines.append(f"City: {city}")

    lines.append(f"Results root: {rr or '(unset)'}")
    lines.append(f"Dataset: {ds or '(none)'}")
    lines.append(f"Manifest: {man or '(auto/empty)'}")
    lines.append(f"Model: {mdl or '(empty)'}")

    lines.append("")
    lines.append(f"Python: {platform.python_version()}")
    lines.append(f"TensorFlow: {_tf_version_text()}")
    lines.append(
        f"OS: {platform.system()} {platform.release()}"
    )

    return "\n".join(lines)


def build_troubleshoot_page(parent: QWidget) -> QWidget:
    inner = QWidget(parent)
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(10)

    # --- Diagnostics bar (tiny) ---
    bar = QWidget(inner)
    hb = QHBoxLayout(bar)
    hb.setContentsMargins(0, 0, 0, 0)
    hb.setSpacing(8)

    lbl = QLabel("Diagnostics", bar)
    lbl.setObjectName("aboutSectionTitle")

    status = QLabel("", bar)
    status.setObjectName("aboutFootnote")

    btn = QToolButton(bar)
    btn.setAutoRaise(True)
    btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    btn.setText("Copy diagnostics")
    btn.setToolTip(
        "Copy results root, dataset, manifest, and model "
        "paths for bug reports."
    )

    ic = try_icon("copy.svg")
    if ic is None:
        ic = btn.style().standardIcon(
            QStyle.SP_DialogApplyButton
        )
    btn.setIcon(ic)

    def _on_copy() -> None:
        txt = _collect_diag(inner)
        QApplication.clipboard().setText(txt)

        status.setText("Copied to clipboard.")
        QTimer.singleShot(
            1800,
            lambda: status.setText(""),
        )

    btn.clicked.connect(_on_copy)

    hb.addWidget(lbl, 0)
    hb.addStretch(1)
    hb.addWidget(status, 0, alignment=Qt.AlignVCenter)
    hb.addWidget(btn, 0, alignment=Qt.AlignRight)

    lay.addWidget(bar)

    # --- Cards grid ---
    host = QWidget(inner)
    grid = QGridLayout(host)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(10)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

    grid.addWidget(
        make_card(
            host,
            "Troubleshooting",
            content.TROUBLE_HERO_HTML,
        ),
        0,
        0,
        1,
        2,
    )

    grid.addWidget(
        make_card(
            host,
            "First checks",
            content.TROUBLE_FIRST_CHECKS_HTML,
        ),
        1,
        0,
        1,
        2,
    )

    grid.addWidget(
        make_card(
            host,
            "Dataset not loaded",
            content.TROUBLE_DATASET_NOT_LOADED_HTML,
        ),
        2,
        0,
    )
    grid.addWidget(
        make_card(
            host,
            "Wrong results root",
            content.TROUBLE_RESULTS_ROOT_HTML,
        ),
        2,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "Manifest mismatch",
            content.TROUBLE_MANIFEST_MISMATCH_HTML,
        ),
        3,
        0,
    )
    grid.addWidget(
        make_card(
            host,
            "Rebuild Stage-1",
            content.TROUBLE_STAGE1_REBUILD_HTML,
        ),
        3,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "Model load issues",
            content.TROUBLE_MODEL_LOAD_HTML,
        ),
        4,
        0,
    )
    grid.addWidget(
        make_card(
            host,
            "Empty map",
            content.TROUBLE_EMPTY_MAP_HTML,
        ),
        4,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "No runs discovered",
            content.TROUBLE_NO_RUNS_HTML,
        ),
        5,
        0,
    )
    grid.addWidget(
        make_card(
            host,
            "Tuning slow/unstable",
            content.TROUBLE_TUNER_HTML,
        ),
        5,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "Transferability odd",
            content.TROUBLE_XFER_HTML,
        ),
        6,
        0,
    )
    grid.addWidget(
        make_card(
            host,
            "Exports missing",
            content.TROUBLE_EXPORT_HTML,
        ),
        6,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "Performance",
            content.TROUBLE_PERF_HTML,
        ),
        7,
        0,
    )
    grid.addWidget(
        make_card(
            host,
            "Support info",
            content.TROUBLE_FOOT_HTML,
        ),
        7,
        1,
    )

    lay.addWidget(host)
    lay.addStretch(1)
    return wrap_scroll(parent, inner)
