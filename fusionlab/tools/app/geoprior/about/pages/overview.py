# geoprior/about/pages/overview.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from PyQt5.QtWidgets import (
    QGridLayout,
    QVBoxLayout,
    QWidget,
)

from . import content
from ..widgets import make_card, wrap_scroll


def build_overview_page(parent: QWidget) -> QWidget:
    inner = QWidget(parent)
    v = QVBoxLayout(inner)
    v.setContentsMargins(6, 6, 6, 6)
    v.setSpacing(10)

    host = QWidget(inner)
    grid = QGridLayout(host)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(10)
    grid.setColumnStretch(0, 2)
    grid.setColumnStretch(1, 1)

    hero = make_card(
        host,
        "GeoPrior Forecaster",
        content.OVERVIEW_INTRO_HTML,
    )
    grid.addWidget(hero, 0, 0, 1, 2)

    grid.addWidget(
        make_card(
            host,
            "What’s new",
            content.OVERVIEW_WHATS_NEW_HTML,
        ),
        1,
        0,
    )

    grid.addWidget(
        make_card(
            host,
            "Project info",
            content.OVERVIEW_PROJECT_HTML,
            show_title=False, 
        ),
        1,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "Workflow",
            content.OVERVIEW_FLOW_HTML,
            show_title=False, 
        ),
        2,
        0,
    )

    grid.addWidget(
        make_card(
            host,
            "Add-ons",
            content.OVERVIEW_ADDONS_HTML,
            show_title=False, 
        ),
        2,
        1,
    )

    grid.addWidget(
        make_card(
            host,
            "Next",
            content.OVERVIEW_NEXT_HTML,
            show_title=False, 
        ),
        3,
        0,
        1,
        2,
    )

    v.addWidget(host)
    v.addStretch(1)
    return wrap_scroll(parent, inner)
