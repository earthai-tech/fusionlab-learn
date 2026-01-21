# geoprior/about/widgets.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QStyle,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)

from ..ui.icon_utils import try_icon  


def _icon(
    svg: Optional[str],
    fallback: QStyle.StandardPixmap,
) -> QIcon:
    ico: Optional[QIcon] = None
    if svg:
        ico = try_icon(svg)

    if ico is not None and not ico.isNull():
        return ico

    return QApplication.style().standardIcon(fallback)


def make_nav(parent: QWidget) -> QListWidget:
    nav = QListWidget(parent)
    nav.setObjectName("aboutNav")
    nav.setFixedWidth(200)
    return nav


def add_nav_item(
    nav: QListWidget,
    text: str,
    *,
    svg: Optional[str] = None,
    sp: QStyle.StandardPixmap = QStyle.SP_FileIcon,
) -> None:
    it = QListWidgetItem(text, nav)
    it.setIcon(_icon(svg, sp))


def wrap_scroll(parent: QWidget, inner: QWidget) -> QScrollArea:
    sc = QScrollArea(parent)
    sc.setWidgetResizable(True)
    sc.setFrameShape(QFrame.NoFrame)
    sc.setWidget(inner)
    return sc


def make_card(
    parent: QWidget,
    title: str,
    body_html: str,
    *,
    show_title: bool = True,
    min_height: Optional[int] = None,
) -> QFrame:
    """
    Build a modern "card" with optional title.

    Parameters
    ----------
    show_title:
        If False, the card header label is not rendered. Use this when
        body_html already contains its own heading to avoid redundancy.
    min_height:
        Optional minimum height for visual consistency in grids.
    """
    c = QFrame(parent)
    c.setObjectName("aboutCard")

    # Critical for grid layouts: avoid cards stretching vertically
    # and making text appear centered.
    c.setSizePolicy(
        QSizePolicy.Expanding,
        QSizePolicy.Maximum,
    )

    v = QVBoxLayout(c)
    v.setContentsMargins(12, 10, 12, 10)
    v.setSpacing(6)

    if show_title:
        t = QLabel(title, c)
        t.setObjectName("aboutCardTitle")
        t.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # Hug content vertically too
        t.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Maximum,
        )
        v.addWidget(t)

    b = QLabel(c)
    b.setObjectName("aboutBody")
    b.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    b.setSizePolicy(
        QSizePolicy.Expanding,
        QSizePolicy.Maximum,
    )
    b.setWordWrap(True)
    b.setTextFormat(Qt.RichText)
    b.setText(body_html)
    b.setOpenExternalLinks(True)
    b.setTextInteractionFlags(
        Qt.TextSelectableByMouse
        | Qt.LinksAccessibleByMouse
    )
    v.addWidget(b, 1)

    # Ensure content stays at the top even if the grid row grows.
    v.addStretch(1)

    if min_height is not None:
        c.setMinimumHeight(int(min_height))
        
    c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    return c



def make_header(
    parent: QWidget,
    *,
    title: str,
    tagline: str,
    chips: Optional[list[str]] = None,
    svg_icon: Optional[str] = None,
) -> QFrame:
    header = QFrame(parent)
    header.setObjectName("aboutHeader")

    h = QHBoxLayout(header)
    h.setContentsMargins(12, 10, 12, 10)
    h.setSpacing(12)

    logo = QLabel(header)
    logo.setFixedSize(56, 56)
    logo.setAlignment(Qt.AlignCenter)

    ico = _icon(svg_icon, QStyle.SP_ComputerIcon)
    logo.setPixmap(ico.pixmap(44, 44))
    h.addWidget(logo, alignment=Qt.AlignTop)

    v = QVBoxLayout()

    ttl = QLabel(title, header)
    ttl.setObjectName("aboutTitle")
    v.addWidget(ttl)

    if chips:
        row = QHBoxLayout()
        row.setSpacing(6)
        for ch in chips:
            lab = QLabel(ch, header)
            lab.setObjectName("aboutChip")
            row.addWidget(lab)
        row.addStretch(1)
        v.addLayout(row)

    sub = QLabel(tagline, header)
    sub.setObjectName("aboutTagline")
    sub.setWordWrap(True)
    v.addWidget(sub)

    h.addLayout(v, 1)
    return header
