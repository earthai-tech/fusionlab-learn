# geoprior/about/pages/quickstart.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from PyQt5.QtWidgets import (
    QGridLayout,
    QVBoxLayout,
    QWidget,
)

from . import content
from ..widgets import make_card, wrap_scroll


def build_quickstart_page(parent: QWidget) -> QWidget:
    inner = QWidget(parent)
    v = QVBoxLayout(inner)
    v.setContentsMargins(6, 6, 6, 6)
    v.setSpacing(10)

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
            "Quickstart",
            content.QUICKSTART_HERO_HTML,
        ),
        0,
        0,
        1,
        2,
    )

    grid.addWidget(
        make_card(host, "1) Data", content.QS_STEP1_HTML),
        1,
        0,
    )
    grid.addWidget(
        make_card(host, "2) Setup", content.QS_STEP2_HTML),
        1,
        1,
    )

    grid.addWidget(
        make_card(host, "3) Preprocess", content.QS_STEP3_HTML),
        2,
        0,
    )
    grid.addWidget(
        make_card(host, "4) Train", content.QS_STEP4_HTML),
        2,
        1,
    )

    grid.addWidget(
        make_card(host, "5) Tune", content.QS_STEP5_HTML),
        3,
        0,
    )
    grid.addWidget(
        make_card(host, "6) Inference", content.QS_STEP6_HTML),
        3,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "7) Transferability",
            content.QS_STEP7_HTML,
        ),
        4,
        0,
    )
    grid.addWidget(
        make_card(
            host,
            "Map & Results",
            content.QS_MAP_RESULTS_HTML,
        ),
        4,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "Tips",
            content.QS_TIPS_HTML,
        ),
        5,
        0,
        1,
        2,
    )

    v.addWidget(host)
    v.addStretch(1)
    return wrap_scroll(parent, inner)
