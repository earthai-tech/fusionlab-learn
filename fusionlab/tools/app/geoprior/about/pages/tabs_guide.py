# geoprior/about/pages/tabs_guide.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from PyQt5.QtWidgets import (
    QGridLayout,
    QVBoxLayout,
    QWidget,
)

from . import content
from ..widgets import make_card, wrap_scroll


def build_tabs_page(parent: QWidget) -> QWidget:
    inner = QWidget(parent)
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.setSpacing(10)

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
            "Tabs guide",
            content.TABS_GUIDE_HERO_HTML,
        ),
        0,
        0,
        1,
        2,
    )

    grid.addWidget(
        make_card(
            host,
            "Legend",
            content.TABS_GUIDE_LEGEND_HTML,
        ),
        1,
        0,
        1,
        2,
    )
    grid.addWidget(
        make_card(
            host,
            "File layout",
            content.TABS_GUIDE_FILES_HTML,
        ),
        2,
        0,
        1,
        2,
    )

    grid.addWidget(
        make_card(host, "Data", content.TAB_DATA_HTML),
        3,
        0,
    )
    grid.addWidget(
        make_card(host, "Setup", content.TAB_SETUP_HTML),
        3,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "Preprocess",
            content.TAB_PREPROCESS_HTML,
        ),
        4,
        0,
    )
    grid.addWidget(
        make_card(host, "Train", content.TAB_TRAIN_HTML),
        4,
        1,
    )

    grid.addWidget(
        make_card(host, "Tune", content.TAB_TUNE_HTML),
        5,
        0,
    )
    grid.addWidget(
        make_card(
            host,
            "Inference",
            content.TAB_INFERENCE_HTML,
        ),
        5,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "Transferability",
            content.TAB_TRANSFER_HTML,
        ),
        6,
        0,
    )
    grid.addWidget(
        make_card(host, "Results", content.TAB_RESULTS_HTML),
        6,
        1,
    )

    grid.addWidget(
        make_card(host, "Map", content.TAB_MAP_HTML),
        7,
        0,
    )
    grid.addWidget(
        make_card(host, "Tools", content.TAB_TOOLS_HTML),
        7,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "Debug order",
            content.TABS_GUIDE_FOOT_HTML,
        ),
        8,
        0,
        1,
        2,
    )

    lay.addWidget(host)
    lay.addStretch(1)
    return wrap_scroll(parent, inner)
