# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards.base

Shared card building blocks for Setup UI.

CardBase provides:
- Consistent container styling (uses QWidget#card).
- Title, subtitle, badges, and a compact action row.
- A body container where each card adds its UI.

The panel owns scrolling and section mapping.
Cards can be store-aware, but don't have to be.
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CardBase(QFrame):
    """Base class for Setup cards."""

    def __init__(
        self,
        *,
        section_id: str,
        title: str,
        subtitle: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.section_id = str(section_id)

        self.setObjectName("card")
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Maximum,
        )
 
        # also helps rounded backgrounds paint correctly
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._badges: Dict[str, QLabel] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        # -------------------------
        # Header row
        # -------------------------
        head = QWidget(self)
        head.setObjectName("cardHeaderRow")
        head_l = QHBoxLayout(head)
        head_l.setContentsMargins(0, 0, 0, 0)
        head_l.setSpacing(8)


        self.lbl_title = QLabel(str(title), head)
        self.lbl_title.setObjectName("cardTitle")

        self.badges = QWidget(head)
        self.badges.setObjectName("cardBadgesRow")
        self.badges_l = QHBoxLayout(self.badges)
        self.badges_l.setContentsMargins(0, 0, 0, 0)
        self.badges_l.setSpacing(6)

        head_l.addWidget(self.lbl_title, 0)
        head_l.addWidget(self.badges, 0)
        head_l.addStretch(1)

        self.actions = QWidget(head)
        self.actions.setObjectName("cardActionsRow")
        self.actions_l = QHBoxLayout(self.actions)
        self.actions_l.setContentsMargins(0, 0, 0, 0)
        self.actions_l.setSpacing(6)

        head_l.addWidget(self.actions, 0)

        root.addWidget(head, 0)

        # Subtitle (optional)
        self.lbl_subtitle = QLabel(str(subtitle), self)
        self.lbl_subtitle.setWordWrap(True)
        self.lbl_subtitle.setVisible(bool(subtitle))
        self.lbl_subtitle.setObjectName("setupCardSubtitle")

        root.addWidget(self.lbl_subtitle, 0)

        # -------------------------
        # Body
        # -------------------------
        self.body = QWidget(self)
        self.body.setObjectName("cardBodyRoot")
        self.body_l = QVBoxLayout(self.body)
        self.body_l.setContentsMargins(0, 0, 0, 0)
        self.body_l.setSpacing(8)

        root.addWidget(self.body, 1)

        # Local, safe styles
        self.setStyleSheet(
            "\n".join(
                [
                    "QLabel#setupCardSubtitle {",
                    "  color: rgba(30,30,30,0.72);",
                    "  font-size: 11px;",
                    "}",
                    "QLabel#setupBadge {",
                    "  padding: 1px 8px;",
                    "  border-radius: 10px;",
                    "  border: 1px solid",
                    "    rgba(46,49,145,0.28);",
                    "  background: rgba(46,49,145,0.06);",
                    "  font-weight: 600;",
                    "}",
                    "QLabel#setupBadge[accent='warn'] {",
                    "  border-color: rgba(242,134,32,0.45);",
                    "  background: rgba(242,134,32,0.12);",
                    "}",
                    "QLabel#setupBadge[accent='ok'] {",
                    "  border-color: rgba(34,197,94,0.45);",
                    "  background: rgba(34,197,94,0.10);",
                    "}",
                ]
            )
        )

    # -----------------------------------------------------------------
    # Header helpers
    # -----------------------------------------------------------------
    def set_subtitle(self, text: str) -> None:
        t = str(text or "")
        self.lbl_subtitle.setText(t)
        self.lbl_subtitle.setVisible(bool(t.strip()))

    def badge(
        self,
        key: str,
        *,
        text: str,
        accent: str = "",
        tip: str = "",
    ) -> QLabel:
        """Create/update a badge in the header."""
        k = str(key)
        lab = self._badges.get(k)
        if lab is None:
            lab = QLabel(self.badges)
            lab.setObjectName("setupBadge")
            self.badges_l.addWidget(lab, 0)
            self._badges[k] = lab

        lab.setText(str(text))
        if accent:
            lab.setProperty("accent", str(accent))
        if tip:
            lab.setToolTip(str(tip))

        lab.style().unpolish(lab)
        lab.style().polish(lab)
        lab.update()
        return lab

    def add_action(
        self,
        *,
        text: str,
        tip: str = "",
        icon: Optional[QStyle.StandardPixmap] = None,
    ) -> QToolButton:
        """Add a compact action button (miniAction)."""
        btn = QToolButton(self.actions)
        btn.setObjectName("miniAction")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setText(str(text))
        if tip:
            btn.setToolTip(str(tip))

        if icon is not None:
            btn.setIcon(self.style().standardIcon(icon))
            btn.setToolButtonStyle(
                Qt.ToolButtonTextBesideIcon
            )

        self.actions_l.addWidget(btn, 0)
        return btn

    # -----------------------------------------------------------------
    # Body access
    # -----------------------------------------------------------------
    def body_layout(self) -> QVBoxLayout:
        return self.body_l
    
    def body_widget(self) -> QWidget:
        return self.body
