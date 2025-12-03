# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
About / Help dialog for GeoPrior-3.0 Forecaster.

This module provides a small, self-contained ``AboutDialog`` that can be
invoked from the main GeoPrior GUI (for example from a "Help → About…"
menu item or a small "info" button in the toolbar).

The dialog shows:

* a short description of GeoPrior-3.0 and the GeoPriorSubsNet model;
* links to online documentation, source code, and the conceptor’s
  portfolio;
* copyright / licensing information;
* a "Quick help" tab with a minimal workflow reminder.

The implementation is intentionally lightweight: it only depends on
PyQt5 and the global Fusionlab stylesheet defined in
:mod:`fusionlab.tools.app.geoprior.styles`.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QWidget,
    QFrame,
    QGridLayout,
    QDialogButtonBox,
)

from .styles import FLAB_STYLE_SHEET, PRIMARY, PALETTE

import platform

try:
    import tensorflow as _tf  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency
    _tf = None

# Lightweight system-information string (evaluated once at import).
if _tf is not None:
    try:
        _tf_version = _tf.__version__
        _devices = _tf.config.list_physical_devices("GPU")
        _device_str = "GPU" if _devices else "CPU"
    except Exception:  # pragma: no cover - best-effort only
        _tf_version = "unknown"
        _device_str = "unknown"
else:
    _tf_version = "not installed"
    _device_str = "n/a"

SYSTEM_INFO_TEXT: str = (
    f"Python {platform.python_version()}, "
    f"TensorFlow {_tf_version} ({_device_str}), "
    f"OS: {platform.system()} {platform.release()}."
)

# ---------------------------------------------------------------------------
# Static metadata
# ---------------------------------------------------------------------------

APP_NAME: str = "GeoPrior-3.0 Forecaster"
APP_VERSION: str = "v3.0"

DOCS_URL: str = (
    "https://fusion-lab.readthedocs.io/en/latest/user_guide/"
    "geopriorv3_guide.html"
)
GITHUB_URL: str = "https://github.com/earthai-tech"
PORTFOLIO_URL: str = "https://lkouadio.com/"

COPYRIGHT_LINE: str = "© 2025 EarthAi-tech"

# Optional placeholder path for the conceptor avatar.  The dialog will
# simply skip the avatar if the file does not exist.  You can replace
# this with a real resource (for example a Qt resource path like
# ':/geoprior/conceptor.png') when integrating.
DEFAULT_AVATAR_PATH: str = "geoprior_conceptor.png"

# Short citation text for the main GeoPrior paper.
# You can refine authors / venue later without touching the dialog code.
CITATION_TEXT: str = """
<pre style="font-family: Consolas, 'DejaVu Sans Mono', monospace;
            font-size: 11px;">

@article{GeoPriorSubsNet2025,
  author  = {{GeoPrior-3.0 Development Team}},
  title   = {Physics-Informed Attention Learning for Divergent Land
             Subsidence Forecasting in Rapidly Urbanizing Coastal Zones},
  year    = {2025},
  note    = {Preprint. See GeoPrior-3.0 documentation for the latest
             citation details.}
}

</pre>
""".strip()


ABOUT_STYLES: str = f"""
QDialog#aboutDialog {{
    background: {PALETTE['light_bg']};
    color: {PALETTE['light_text']};
}}

QFrame#aboutHeader {{
    background: {PALETTE['light_card_bg']};
    border-radius: 12px;
    border: 1px solid {PALETTE['light_border']};
    padding: 12px;
}}

QLabel#aboutTitle {{
    font-size: 20px;
    font-weight: 600;
    color: {PALETTE['light_text_title']};
}}

QLabel#aboutTagline {{
    font-size: 12px;
    color: {PALETTE['light_text_muted']};
}}

QLabel#aboutLinks {{
    color: {PRIMARY};
}}

QLabel#aboutSectionTitle {{
    font-size: 13px;
    font-weight: 600;
    margin-top: 6px;
}}

QLabel#aboutBody {{
    font-size: 12px;
}}

QLabel#aboutFootnote {{
    font-size: 11px;
    color: {PALETTE['light_text_muted']};
}}
"""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def show_about_dialog(parent: QWidget | None = None) -> None:
    """
    Convenience helper to show the :class:`AboutDialog`.

    Use this from the main window, e.g.::

        from .about import show_about_dialog

        action = QAction("About GeoPrior…", self)
        action.triggered.connect(lambda: show_about_dialog(self))

    Parameters
    ----------
    parent :
        Parent widget or main window.  The dialog will be centred on
        this widget.
    """
    dlg = AboutDialog(parent=parent)
    dlg.exec_()


# ---------------------------------------------------------------------------
# Dialog implementation
# ---------------------------------------------------------------------------

class AboutDialog(QDialog):
    """
    Simple two-tab dialog: "About" and "Quick help".

    The dialog is intentionally small and modal.  It uses the Fusionlab
    stylesheet so it matches the rest of the GUI.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        avatar_path: Optional[str] = None,
    ) -> None:
        super().__init__(parent)

        self.setObjectName("aboutDialog")
        self.setWindowTitle("About GeoPrior-3.0 Forecaster")
        self.setModal(True)
        self.setMinimumWidth(640)

        # Style: reuse the global Fusionlab sheet and add a small
        # dialog-specific layer.
        self.setStyleSheet(FLAB_STYLE_SHEET + ABOUT_STYLES)

        main = QVBoxLayout(self)

        tabs = QTabWidget(self)
        tabs.addTab(self._build_about_tab(avatar_path), "About")
        tabs.addTab(self._build_help_tab(), "Quick help")
        main.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        main.addWidget(buttons)

    # ------------------------------------------------------------------ #
    # Tab builders
    # ------------------------------------------------------------------ #

    def _build_about_tab(self, avatar_path: Optional[str]) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Header: icon / title / tagline
        header = QFrame(w)
        header.setObjectName("aboutHeader")
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(12, 8, 12, 8)
        hbox.setSpacing(12)

        # Optional avatar on the left (conceptor photo or logo)
        avatar_path = avatar_path or DEFAULT_AVATAR_PATH
        avatar_label = QLabel(header)
        pix = QPixmap(avatar_path)
        if not pix.isNull():
            size = 72
            avatar_label.setPixmap(
                pix.scaled(
                    size,
                    size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            # Keep layout aligned even when no image is found
            avatar_label.setFixedWidth(1)
        hbox.addWidget(avatar_label, alignment=Qt.AlignTop)

        # Text block
        title_box = QVBoxLayout()
        title_lbl = QLabel(f"{APP_NAME} [{APP_VERSION}]", header)
        title_lbl.setObjectName("aboutTitle")
        title_box.addWidget(title_lbl)

        tagline = QLabel(
            "Physics-informed subsidence forecasting with "
            "GeoPriorSubsNet and multi-city transfer learning.",
            header,
        )
        tagline.setObjectName("aboutTagline")
        tagline.setWordWrap(True)
        title_box.addWidget(tagline)

        links = QLabel(
            (
                f'<a href="{DOCS_URL}">User guide</a> · '
                f'<a href="{GITHUB_URL}">Source code</a> · '
                f'<a href="{PORTFOLIO_URL}">Conceptor portfolio</a>'
            ),
            header,
        )
        links.setObjectName("aboutLinks")
        links.setTextFormat(Qt.RichText)
        links.setOpenExternalLinks(True)
        title_box.addWidget(links)

        title_box.addStretch(1)
        hbox.addLayout(title_box, stretch=1)

        layout.addWidget(header)

        # Summary grid
        grid_frame = QFrame(w)
        grid = QGridLayout(grid_frame)
        grid.setColumnStretch(1, 1)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        def _add_row(row: int, title: str, body: str) -> None:
            lbl_title = QLabel(title, grid_frame)
            lbl_title.setObjectName("aboutSectionTitle")
        
            lbl_body = QLabel(body, grid_frame)
            lbl_body.setObjectName("aboutBody")
            lbl_body.setWordWrap(True)
            # Allow copying text and clicking links (docs, portfolio, etc.)
            lbl_body.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
            )
        
            grid.addWidget(lbl_title, row, 0, alignment=Qt.AlignTop)
            grid.addWidget(lbl_body, row, 1)


        _add_row(
            0,
            "What is GeoPrior-3.0?",
            (
                "GeoPrior-3.0 is a physics-informed forecasting toolkit for "
                "urban land subsidence. It couples a deep sequence model "
                "with hydro-geomechanical priors so that predictions remain "
                "consistent with groundwater flow and consolidation physics."
            ),
        )

        _add_row(
            1,
            "Workflow in this GUI",
            (
                "The Forecaster wraps the Stage-1 data preparation, "
                "Stage-2 model training / tuning, and Stage-3 inference "
                "routines into a single interface. You can train, tune, "
                "evaluate and reuse GeoPriorSubsNet models for multiple "
                "cities from the same window."
            ),
        )

        _add_row(
            2,
            "Documentation & papers",
            (
                "A detailed user guide, model description, and experiment "
                "recipes are available in the online documentation. The "
                "scientific foundations of GeoPrior are described in the "
                "accompanying research papers; please cite them when you "
                "use this tool in publications."
            ),
        )

        _add_row(
            3,
            "How to cite GeoPrior",
            CITATION_TEXT,
        )

        _add_row(
            4,
            "License & copyright",
            (
                f"{COPYRIGHT_LINE}. GeoPrior-3.0 is distributed under a "
                "BSD-3-Clause license together with the Fusionlab library."
            ),
        )
        
        _add_row(
            5,
            "System info",
            SYSTEM_INFO_TEXT,
        )

        layout.addWidget(grid_frame)

        footnote = QLabel(
            "Tip: hover the main controls in the Train / Tune / "
            "Inference / Transferability tabs to see more detailed "
            "tooltips.",
            w,
        )
        footnote.setObjectName("aboutFootnote")
        footnote.setWordWrap(True)
        layout.addWidget(footnote)

        layout.addStretch(1)
        return w

    def _build_help_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Quick workflow reminder", w)
        title.setObjectName("aboutSectionTitle")
        layout.addWidget(title)

        body = QLabel(w)
        body.setObjectName("aboutBody")
        body.setWordWrap(True)
        body.setText(
            "<ol>"
            "<li><b>Select CSV…</b> to load a harmonised city dataset "
            "(or a custom CSV following the GeoPrior column "
            "conventions). The base file name (without <code>.csv</code>) "
            "is used as the default city / dataset name for all runs."
            "</li>"

            "<li>In the <b>Train</b> tab, choose the temporal window: "
            "<b>Train end year</b>, <b>Forecast start year</b>, "
            "<b>Forecast horizon</b>, and <b>Time steps</b> (look-back). "
            "Configure the <b>Training</b> block (epochs, batch size, "
            "learning rate) and the <b>Physics weights</b> block to "
            "control how strongly the PDE priors influence learning."
            "</li>"

            "<li>Still in <b>Train</b>, use <b>Feature config…</b>, "
            "<b>Architecture…</b>, and <b>Probabilistic…</b> to adjust "
            "which variables are used, the GeoPriorSubsNet architecture "
            "dimensions, and the quantile / probabilistic settings. "
            "Enable <b>Evaluate training metrics</b> to compute extra "
            "diagnostics and <b>Build future NPZ</b> to export a "
            "forecast-only dataset for future inference runs."
            "</li>"

            "<li>Click the green play button on the right to run "
            "<b>Stage-1</b> (sequence building) followed by "
            "<b>Stage-2</b> training. Progress and logs are shown in the "
            "console at the bottom, and all artifacts are written under "
            "the selected results root."
            "</li>"

            "<li>In the <b>Tune</b> tab, perform hyper-parameter search "
            "around an existing configuration. The left side controls "
            "architecture ranges (embedding, hidden units, attention "
            "heads, LSTM / VSN units, dropout), while the right side "
            "controls physics-related switches. "
            "Use <b>Scalars losses…</b> to edit scalar loss weights and "
            "ranges, and <b>Max trials</b> to cap the number of tuner "
            "experiments."
            "</li>"

            "<li>In the <b>Inference</b> tab, load a trained "
            "<code>.keras</code> model and a Stage-1 <b>manifest</b>, "
            "choose the dataset (train / val / test or future NPZ), and "
            "optionally apply probabilistic calibration. You can export "
            "plots and CSV outputs, and evaluate coverage / sharpness on "
            "the validation set."
            "</li>"

            "<li>In the <b>Transferability</b> tab, select a source "
            "city (A) and target city (B), choose which splits to use, "
            "and configure calibration modes. This tab lets you evaluate "
            "how a model trained on one city generalises to another, "
            "with optional rescaling of the target city to the source "
            "domain."
            "</li>"

            "<li>In the <b>Results</b> tab, browse all cities and "
            "workflows found under the current results root. You can "
            "inspect individual train / tune / inference / transfer "
            "runs and download them as ZIP archives for archiving or "
            "sharing."
            "</li>"

            "<li>The <b>Dry run</b> checkbox (top-right) prepares the "
            "configuration and decides which Stage-1 / Stage-2 steps "
            "would be executed, without actually launching long jobs. "
            "This is useful when exploring a new dataset or results "
            "root."
            "</li>"
            "</ol>"
        )
        layout.addWidget(body)

        extra = QLabel(w)
        extra.setObjectName("aboutFootnote")
        extra.setWordWrap(True)
        extra.setText(
            "For full configuration details (physics weights, censoring "
            "options, search spaces, and result folders), please refer "
            "to the online user guide."
        )
        layout.addWidget(extra)

        layout.addStretch(1)
        return w

