# geoprior/ui/console/visibility.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
ConsoleVisibilityPolicy

Remembers console dock visibility per main tab.

Rules
-----
- If dock is floating: do NOT auto-apply visibility.
- If dock is docked: apply per-tab visibility memory.
- User manual show/hide is remembered per tab.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Set

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QDockWidget


SetVisibleFn = Callable[..., None]


class ConsoleVisibilityPolicy(QObject):
    """
    Per-tab console visibility memory.

    Parameters
    ----------
    dock:
        The Console dock widget.

    set_visible:
        Callable like: set_console_visible(vis, remember).
        This should be your app's method so it can update
        menu text etc. We always call it with remember=False
        from the policy.

    hidden_tabs:
        Main tabs where console is hidden by default.

    vis_by_tab:
        Optional dict to reuse your existing
        self._console_vis_by_tab mapping.
    """

    def __init__(
        self,
        dock: QDockWidget,
        *,
        set_visible: SetVisibleFn,
        hidden_tabs: Optional[Set[int]] = None,
        vis_by_tab: Optional[Dict[int, bool]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)

        self._dock = dock
        self._set_visible = set_visible

        self._hidden_tabs: Set[int] = set(hidden_tabs or set())
        self._vis_by_tab: Dict[int, bool] = (
            vis_by_tab if isinstance(vis_by_tab, dict) else {}
        )

        self._current_tab: Optional[int] = None

        self._dock.visibilityChanged.connect(
            self._on_dock_visibility_changed
        )

    # -----------------------------
    # Configuration
    # -----------------------------
    def set_hidden_tabs(self, tabs: Set[int]) -> None:
        self._hidden_tabs = set(tabs)

    def get_vis_map(self) -> Dict[int, bool]:
        return dict(self._vis_by_tab)

    def set_vis_map(self, m: Dict[int, bool]) -> None:
        if isinstance(m, dict):
            self._vis_by_tab = m
            
    def apply(self, tab_index: int) -> None:
        """
        Apply visibility for tab_index.

        No-op if dock is floating.
        """
        self._current_tab = int(tab_index)

        if self._dock.isFloating():
            return

        dvis = self.default_visible(self._current_tab)
        vis = bool(
            self._vis_by_tab.get(self._current_tab, dvis)
        )

        # Call app setter WITHOUT breaking keyword-only
        try:
            self._set_visible(vis, remember=False)
        except TypeError:
            # Back-compat for older setters
            try:
                self._set_visible(vis, False)
            except TypeError:
                self._set_visible(vis)
                
    # -----------------------------
    # Core behavior
    # -----------------------------
    def default_visible(self, tab_index: int) -> bool:
        return tab_index not in self._hidden_tabs

    # -----------------------------
    # Signal handlers
    # -----------------------------
    def _on_dock_visibility_changed(self, vis: bool) -> None:
        """
        When user shows/hides the dock manually,
        remember it for the current tab, but only
        when dock is not floating.
        """
        if self._dock.isFloating():
            return

        if self._current_tab is None:
            return

        self._vis_by_tab[int(self._current_tab)] = bool(vis)
