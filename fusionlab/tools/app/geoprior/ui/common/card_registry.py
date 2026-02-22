# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.common.card_registry

Small helper to jump from navigator -> center card.

Works with a QScrollArea that hosts the center content.
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt5.QtCore import (
    QEasingCurve,
    QObject,
    QPoint,
    QPropertyAnimation,
    QTimer,
)
from PyQt5.QtWidgets import QScrollArea, QWidget


class CardRegistry(QObject):
    def __init__(
        self,
        *,
        scroll: QScrollArea,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._scroll = scroll
        self._cards: Dict[str, QWidget] = {}
        self._anim: Optional[QPropertyAnimation] = None

    def register(self, key: str, w: QWidget) -> None:
        k = str(key or "").strip()
        if not k:
            return
        self._cards[k] = w

    def goto(
        self,
        key: str,
        *,
        animate: bool = True,
        flash: bool = True,
    ) -> None:
        w = self._cards.get(str(key or "").strip())
        if w is None:
            return

        host = self._scroll.widget()
        if host is None:
            return

        y = w.mapTo(host, QPoint(0, 0)).y()
        target = max(0, int(y) - 8)

        bar = self._scroll.verticalScrollBar()
        if not animate:
            bar.setValue(target)
        else:
            if self._anim is not None:
                try:
                    self._anim.stop()
                except Exception:
                    pass
            self._anim = QPropertyAnimation(bar, b"value")
            self._anim.setDuration(220)
            self._anim.setStartValue(bar.value())
            self._anim.setEndValue(target)
            self._anim.setEasingCurve(QEasingCurve.OutCubic)
            self._anim.start()

        if flash:
            self._flash(w)

        try:
            w.setFocus()
        except Exception:
            pass

    def _flash(self, w: QWidget) -> None:
        w.setProperty("flash", True)
        w.style().unpolish(w)
        w.style().polish(w)
        w.update()

        def _off() -> None:
            w.setProperty("flash", False)
            w.style().unpolish(w)
            w.style().polish(w)
            w.update()

        QTimer.singleShot(420, _off)
