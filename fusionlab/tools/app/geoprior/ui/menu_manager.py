# geoprior/ui/menu_manager.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QMainWindow, QStyle, QAction
from PyQt5.QtGui import QDesktopServices


class MenuManager:
    """
    Small helper responsible for building the GeoPrior menu bar.

    It uses callbacks / attributes exposed by the main window
    (train/tune/infer/xfer handlers, tab indices, dry-run checkbox,
    icon helpers, etc.) but keeps the verbose QAction plumbing out of
    the main class.

    Parameters
    ----------
    window : QMainWindow
        The GeoPriorForecaster instance (or compatible main window).
    mode_mgr : object
        ModeManager instance, used for the global Stop action.
    docs_url : str
        URL of the online documentation.
    """

    def __init__(
        self,
        window: QMainWindow,
        mode_mgr: Any,
        docs_url: str,
    ) -> None:
        self.win = window
        self.mode_mgr = mode_mgr
        self.docs_url = docs_url

    # ------------------------------------------------------------------
    # Internal icon helpers
    # ------------------------------------------------------------------
    def _std_icon(self, sp: QStyle.StandardPixmap):
        """
        Prefer the window's _std_icon helper if available, otherwise
        fall back to style().standardIcon().
        """
        if hasattr(self.win, "_std_icon"):
            return self.win._std_icon(sp)  # type: ignore[no-any-return]
        return self.win.style().standardIcon(sp)

    def _workflow_icon(
        self,
        svg_name: str,
        fallback: QStyle.StandardPixmap,
    ):
        """
        Prefer the window's _workflow_icon helper if available,
        otherwise just use the fallback standard icon.
        """
        if hasattr(self.win, "_workflow_icon"):
            return self.win._workflow_icon(svg_name, fallback)  # type: ignore[no-any-return]
        return self._std_icon(fallback)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def build(self) -> None:
        """
        Create the menu bar and expose key QActions on the main window.

        This mirrors the previous `_build_menu_bar` implementation,
        but the logic is encapsulated here.
        """
        win = self.win
        menubar = win.menuBar()

        # ---------------------- File menu ----------------------
        file_menu = menubar.addMenu("&File")

        act_open = QAction(
            self._std_icon(QStyle.SP_DialogOpenButton),
            "Open dataset…",
            win,
        )
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(win._on_open_dataset)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_exit = QAction(
            self._std_icon(QStyle.SP_DialogCloseButton),
            "Exit",
            win,
        )
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(win.close)
        file_menu.addAction(act_exit)

        # ---------------------- Run menu -----------------------
        run_menu = menubar.addMenu("&Run")

        act_run_train = QAction(
            self._workflow_icon("train.svg", QStyle.SP_MediaPlay),
            "Run training",
            win,
        )
        act_run_train.setShortcut("F5")
        act_run_train.triggered.connect(win._on_train_clicked)
        run_menu.addAction(act_run_train)

        act_run_tune = QAction(
            self._workflow_icon("tune.svg", QStyle.SP_BrowserReload),
            "Run tuning",
            win,
        )
        act_run_tune.setShortcut("Shift+F5")
        act_run_tune.triggered.connect(win._on_tune_clicked)
        run_menu.addAction(act_run_tune)

        act_run_infer = QAction(
            self._workflow_icon("inference.svg", QStyle.SP_ArrowForward),
            "Run inference",
            win,
        )
        act_run_infer.setShortcut("Ctrl+F5")
        act_run_infer.triggered.connect(win._on_infer_clicked)
        run_menu.addAction(act_run_infer)

        act_run_xfer = QAction(
            self._workflow_icon("transfer.svg", QStyle.SP_ArrowRight),
            "Run transfer matrix",
            win,
        )
        act_run_xfer.setShortcut("Alt+F5")
        act_run_xfer.triggered.connect(win._on_xfer_clicked)
        run_menu.addAction(act_run_xfer)

        run_menu.addSeparator()

        # Global Stop, mirroring the Stop button
        act_stop = QAction(
            self._std_icon(QStyle.SP_MediaStop),
            "Stop",
            win,
        )
        act_stop.setShortcut("Esc")
        act_stop.triggered.connect(self.mode_mgr.on_stop_clicked)
        run_menu.addAction(act_stop)

        # expose for other parts of the GUI (same name as before)
        win._act_stop = act_stop  # type: ignore[attr-defined]

        run_menu.addSeparator()

        # Dry run as a global toggle, synced with the checkbox
        act_dry_run = QAction(
            self._std_icon(QStyle.SP_MessageBoxInformation),
            "Dry run",
            win,
            checkable=True,
        )
        # initial state mirrors the checkbox
        act_dry_run.setChecked(win.chk_dry_run.isChecked())
        act_dry_run.toggled.connect(win.chk_dry_run.setChecked)
        win.chk_dry_run.toggled.connect(act_dry_run.setChecked)
        run_menu.addAction(act_dry_run)

        # expose for other parts of the GUI
        win.act_dry_run = act_dry_run  # type: ignore[attr-defined]

        # ---------------------- View menu ----------------------
        view_menu = menubar.addMenu("&View")

        view_train = QAction(
            self._workflow_icon("train.svg", QStyle.SP_ComputerIcon),
            "Train tab",
            win,
        )
        view_train.setShortcut("Ctrl+1")
        view_train.triggered.connect(
            lambda: win.tabs.setCurrentIndex(win._train_tab_index)
        )
        view_menu.addAction(view_train)

        view_tune = QAction(
            self._workflow_icon("tune.svg", QStyle.SP_FileDialogDetailedView),
            "Tune tab",
            win,
        )
        view_tune.setShortcut("Ctrl+2")
        view_tune.triggered.connect(
            lambda: win.tabs.setCurrentIndex(win._tune_tab_index)
        )
        view_menu.addAction(view_tune)

        view_infer = QAction(
            self._workflow_icon("inference.svg", QStyle.SP_FileDialogListView),
            "Inference tab",
            win,
        )
        view_infer.setShortcut("Ctrl+3")
        view_infer.triggered.connect(
            lambda: win.tabs.setCurrentIndex(win._infer_tab_index)
        )
        view_menu.addAction(view_infer)

        view_xfer = QAction(
            self._workflow_icon("transfer.svg", QStyle.SP_ArrowRight),
            "Transferability tab",
            win,
        )
        view_xfer.setShortcut("Ctrl+4")
        view_xfer.triggered.connect(
            lambda: win.tabs.setCurrentIndex(win._xfer_tab_index)
        )
        view_menu.addAction(view_xfer)

        view_results = QAction(
            self._workflow_icon("results.svg", QStyle.SP_DirHomeIcon),
            "Results tab",
            win,
        )
        view_results.setShortcut("Ctrl+5")
        view_results.triggered.connect(
            lambda: win.tabs.setCurrentIndex(win._results_tab_index)
        )
        view_menu.addAction(view_results)

        # ---------------------- Help menu ----------------------
        help_menu = menubar.addMenu("&Help")

        act_docs = QAction(
            self._std_icon(QStyle.SP_DialogHelpButton),
            "Online documentation…",
            win,
        )
        act_docs.setShortcut("F1")
        act_docs.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(self.docs_url))
        )
        help_menu.addAction(act_docs)

        help_menu.addSeparator()

        act_about = QAction(
            self._std_icon(QStyle.SP_MessageBoxInformation),
            "About GeoPrior…",
            win,
        )
        # uses your existing show_about_dialog(self)
        act_about.triggered.connect(lambda: win._on_show_about())
        help_menu.addAction(act_about)

