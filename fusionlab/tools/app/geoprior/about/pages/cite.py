# geoprior/about/pages/cite.py
# -*- coding: utf-8 -*-

from __future__ import annotations

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

from ...ui.icon_utils import try_icon
from . import content
from ..widgets import make_card, wrap_scroll


def build_cite_page(parent: QWidget) -> QWidget:
    inner = QWidget(parent)
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(10)

    # --- Copy bar ---
    bar = QWidget(inner)
    hb = QHBoxLayout(bar)
    hb.setContentsMargins(0, 0, 0, 0)
    hb.setSpacing(8)

    lbl = QLabel("Cite", bar)
    lbl.setObjectName("aboutSectionTitle")

    status = QLabel("", bar)
    status.setObjectName("aboutFootnote")

    btn_all = QToolButton(bar)
    btn_all.setAutoRaise(True)
    btn_all.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    btn_all.setText("Copy BibTeX")
    btn_all.setToolTip("Copy all BibTeX entries to the clipboard.")

    ic = try_icon("copy.svg")
    if ic is None:
        ic = btn_all.style().standardIcon(QStyle.SP_DialogApplyButton)
    btn_all.setIcon(ic)

    btn_paper = QToolButton(bar)
    btn_paper.setAutoRaise(True)
    # smaller, “secondary” feel
    btn_paper.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    btn_paper.setText("Copy paper only")
    btn_paper.setToolTip("Copy only the primary paper BibTeX entry.")
    # reuse same icon but keep it subtle
    btn_paper.setIcon(ic)

    def _flash(msg: str) -> None:
        status.setText(msg)
        QTimer.singleShot(1800, lambda: status.setText(""))

    def _on_copy_all() -> None:
        QApplication.clipboard().setText(content.CITE_COPY_TEXT)
        _flash("Copied all BibTeX entries.")

    def _on_copy_paper() -> None:
        QApplication.clipboard().setText(content.CITATION_BIBTEX_TEXT)
        _flash("Copied paper BibTeX.")

    btn_all.clicked.connect(_on_copy_all)
    btn_paper.clicked.connect(_on_copy_paper)

    hb.addWidget(lbl, 0)
    hb.addStretch(1)
    hb.addWidget(status, 0, alignment=Qt.AlignVCenter)
    hb.addWidget(btn_paper, 0, alignment=Qt.AlignRight)
    hb.addWidget(btn_all, 0, alignment=Qt.AlignRight)
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
            "Citing this work",
            content.CITE_HERO_HTML,
            show_title=False,
        ),
        0,
        0,
        1,
        2,
    )

    grid.addWidget(
        make_card(
            host,
            "When to cite",
            content.CITE_WHEN_HTML,
            show_title=False,
            min_height=120
        ),
        1,
        0,
    )
    
    grid.addWidget(
        make_card(
            host,
            "Acknowledgement",
            content.CITE_ACK_HTML,
            show_title=False,
            min_height=120
        ),
        1,
        1,
    )
    
    grid.addWidget(
        make_card(
            host,
            "Primary paper",
            content.CITATION_HTML,
            show_title=False,
        ),
        2,
        0,
        1,
        2,
    )
    
    grid.addWidget(
        make_card(
            host,
            "BibTeX",
            content.CITE_BIBTEX_HTML,
            show_title=False,
        ),
        3,
        0,
        1,
        2,
    )
    


    lay.addWidget(host)
    lay.addStretch(1)
    return wrap_scroll(parent, inner)
