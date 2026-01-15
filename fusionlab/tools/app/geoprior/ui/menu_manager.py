# geoprior/ui/menu_manager.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt5.QtCore import QUrl, QSignalBlocker
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QAction, QMainWindow, QStyle


class MenuManager:
    """
    Helper responsible for building the GeoPrior menu bar.

    Uses callbacks / attributes exposed by the main window
    (train/tune/infer/xfer handlers, tab indices, dry-run
    checkbox, icon helpers, etc.), while keeping the QAction
    plumbing out of the main class.

    Parameters
    ----------
    window : QMainWindow
        The GeoPriorForecaster instance (or compatible).
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
    # Icon helpers
    # ------------------------------------------------------------------
    def _std_icon(self, sp: QStyle.StandardPixmap):
        """
        Prefer window._std_icon if present; else style icon.
        """
        if hasattr(self.win, "_std_icon"):
            return self.win._std_icon(  # type: ignore[no-any-return]
                sp
            )
        return self.win.style().standardIcon(sp)

    def _workflow_icon(
        self,
        svg_name: str,
        fallback: QStyle.StandardPixmap,
    ):
        """
        Prefer window._workflow_icon if present; else std icon.
        """
        if hasattr(self.win, "_workflow_icon"):
            return self.win._workflow_icon(  # type: ignore[no-any-return]
                svg_name,
                fallback,
            )
        return self._std_icon(fallback)

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------
    def _log_warn(self, msg: str) -> None:
        if not hasattr(self.win, "log_updated"):
            return
        try:
            self.win.log_updated.emit(  # type: ignore[attr-defined]
                f"[WARN] {msg}"
            )
        except Exception:
            return

    def _goto_tab(self, tab_index_attr: str) -> Callable[[], None]:
        """
        Return a slot that switches to a stored tab index.

        Safe: if the attribute does not exist, it logs a warn.
        """

        def _fn() -> None:
            idx = getattr(self.win, tab_index_attr, None)
            tabs = getattr(self.win, "tabs", None)

            if tabs is None or not hasattr(tabs, "setCurrentIndex"):
                self._log_warn("Missing main 'tabs' widget.")
                return

            if not isinstance(idx, int):
                self._log_warn(
                    f"Missing tab index: {tab_index_attr}"
                )
                return

            try:
                tabs.setCurrentIndex(idx)
            except Exception as exc:
                self._log_warn(
                    f"Could not switch tab: {exc}"
                )

        return _fn

    def _connect_if_exists(
        self,
        act: QAction,
        handler_name: str,
    ) -> bool:
        """
        Connect act.triggered to win.<handler_name> if present.
        """
        fn = getattr(self.win, handler_name, None)
        if callable(fn):
            act.triggered.connect(fn)
            return True
        return False

    def _get_log_dock(self) -> Optional[Any]:
        """
        Best-effort lookup of a log dock widget.
        """
        candidates = (
            "log_dock",
            "dock_log",
            "dock_logs",
            "dock_log_panel",
            "logDock",
        )
        for name in candidates:
            dock = getattr(self.win, name, None)
            if dock is None:
                continue
            if hasattr(dock, "setVisible") and hasattr(
                dock,
                "isVisible",
            ):
                return dock
        return None

    # ------------------------------------------------------------------
    # Builders (small)
    # ------------------------------------------------------------------
    def _add_view_tab_action(
        self,
        view_menu: Any,
        *,
        label: str,
        svg: str,
        fallback: QStyle.StandardPixmap,
        tab_attr: str,
        shortcut: str,
    ) -> None:
        act = QAction(
            self._workflow_icon(svg, fallback),
            label,
            self.win,
        )
        act.setShortcut(shortcut)
        act.triggered.connect(self._goto_tab(tab_attr))
        view_menu.addAction(act)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def build(self) -> None:
        """
        Create the menu bar and expose key QActions on the window.
        """
        win = self.win
        menubar = win.menuBar()

        # --------------------------- File ---------------------------
        file_menu = menubar.addMenu("&File")

        act_open = QAction(
            self._std_icon(QStyle.SP_DialogOpenButton),
            "Open dataset…",
            win,
        )
        act_open.setShortcut("Ctrl+O")
        self._connect_if_exists(act_open, "_on_open_dataset")
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

        # --------------------------- Run ----------------------------
        run_menu = menubar.addMenu("&Run")

        act_run_train = QAction(
            self._workflow_icon("train.svg", QStyle.SP_MediaPlay),
            "Run training",
            win,
        )
        act_run_train.setShortcut("F5")
        self._connect_if_exists(
            act_run_train,
            "_on_train_clicked",
        )
        run_menu.addAction(act_run_train)

        act_run_tune = QAction(
            self._workflow_icon(
                "tune.svg",
                QStyle.SP_BrowserReload,
            ),
            "Run tuning",
            win,
        )
        act_run_tune.setShortcut("Shift+F5")
        self._connect_if_exists(
            act_run_tune,
            "_on_tune_clicked",
        )
        run_menu.addAction(act_run_tune)

        act_run_infer = QAction(
            self._workflow_icon(
                "inference.svg",
                QStyle.SP_ArrowForward,
            ),
            "Run inference",
            win,
        )
        act_run_infer.setShortcut("Ctrl+F5")
        self._connect_if_exists(
            act_run_infer,
            "_on_infer_clicked",
        )
        run_menu.addAction(act_run_infer)

        act_run_xfer = QAction(
            self._workflow_icon(
                "transfer.svg",
                QStyle.SP_ArrowRight,
            ),
            "Run transfer matrix",
            win,
        )
        act_run_xfer.setShortcut("Alt+F5")
        self._connect_if_exists(
            act_run_xfer,
            "_on_xfer_clicked",
        )
        run_menu.addAction(act_run_xfer)

        run_menu.addSeparator()

        act_stop = QAction(
            self._std_icon(QStyle.SP_MediaStop),
            "Stop",
            win,
        )
        act_stop.setShortcut("Esc")
        act_stop.triggered.connect(
            self.mode_mgr.on_stop_clicked
        )
        run_menu.addAction(act_stop)
        win._act_stop = act_stop  # type: ignore[attr-defined]

        run_menu.addSeparator()

        act_dry_run = QAction(
            self._std_icon(QStyle.SP_MessageBoxInformation),
            "Dry run",
            win,
            checkable=True,
        )
        if hasattr(win, "chk_dry_run"):
            act_dry_run.setChecked(
                win.chk_dry_run.isChecked()
            )
            act_dry_run.toggled.connect(
                win.chk_dry_run.setChecked
            )
            win.chk_dry_run.toggled.connect(
                act_dry_run.setChecked
            )
        run_menu.addAction(act_dry_run)
        win.act_dry_run = act_dry_run  # type: ignore[attr-defined]

        # --------------------------- View ---------------------------
        view_menu = menubar.addMenu("&View")

        # Dark mode toggle (live)
        act_dark = QAction(
            self._std_icon(QStyle.SP_TitleBarShadeButton),
            "Dark mode",
            win,
            checkable=True,
        )
        act_dark.setShortcut("Ctrl+Shift+D")

        is_dark = False
        is_dark_fn = getattr(win, "_is_dark", None)
        if callable(is_dark_fn):
            try:
                is_dark = bool(is_dark_fn())
            except Exception:
                is_dark = False
        act_dark.setChecked(is_dark)
        
        # XXX TODO: Make Dark mode visible once stable:
            
        set_dark_fn = getattr(win, "set_dark_mode", None)
        can_dark = bool(getattr(win, "enable_dark_mode", False))
        
        def _toast(msg: str) -> None:
            bar = getattr(win, "statusBar", None)
            if callable(bar):
                try:
                    bar().showMessage(msg, 5000)
                except Exception:
                    pass
            self._log_warn(msg)
        
        def _on_dark_toggled(enabled: bool) -> None:
            if enabled and not can_dark:
                _toast("Dark mode is disabled for now.")
                try:
                    with QSignalBlocker(act_dark):
                        act_dark.setChecked(False)
                except Exception:
                    act_dark.setChecked(False)
                return
        
            if callable(set_dark_fn):
                try:
                    set_dark_fn(enabled)
                except Exception as exc:
                    _toast(f"Theme switch failed: {exc}")
        
        act_dark.toggled.connect(_on_dark_toggled)
        # ------------------------------- end ----------------------------

        view_menu.addAction(act_dark)
        view_menu.addSeparator()

        # Nice addition: show/hide log dock
        act_log = QAction(
            self._std_icon(QStyle.SP_FileDialogInfoView),
            "Show log panel",
            win,
            checkable=True,
        )
        act_log.setShortcut("Ctrl+Shift+L")
        
        is_vis_fn = getattr(win, "is_console_visible", None)
        if callable(is_vis_fn):
            act_log.setChecked(bool(is_vis_fn()))
        else:
            act_log.setChecked(True)
        
        set_vis_fn = getattr(win, "set_console_visible", None)
        if callable(set_vis_fn):
            act_log.toggled.connect(set_vis_fn)
        else:
            act_log.setEnabled(False)
        
        view_menu.addAction(act_log)
        win.act_show_log = act_log  # type: ignore[attr-defined]

        view_menu.addSeparator()

        # Tab navigation (Ctrl+1..Ctrl+9 in tab order)
        self._add_view_tab_action(
            view_menu,
            label="Data tab",
            svg="data.svg",
            fallback=QStyle.SP_DirOpenIcon,
            tab_attr="_data_tab_index",
            shortcut="Ctrl+1",
        )
        self._add_view_tab_action(
            view_menu,
            label="Experiment Setup tab",
            svg="setup.svg",
            fallback=QStyle.SP_FileDialogContentsView,
            tab_attr="_setup_tab_index",
            shortcut="Ctrl+2",
        )
        self._add_view_tab_action(
            view_menu,
            label="Preprocess tab",
            svg="stage1.svg",
            fallback=QStyle.SP_DriveHDIcon,
            tab_attr="_preprocess_tab_index",
            shortcut="Ctrl+3",
        )
        self._add_view_tab_action(
            view_menu,
            label="Train tab",
            svg="train.svg",
            fallback=QStyle.SP_ComputerIcon,
            tab_attr="_train_tab_index",
            shortcut="Ctrl+4",
        )
        self._add_view_tab_action(
            view_menu,
            label="Tune tab",
            svg="tune.svg",
            fallback=QStyle.SP_FileDialogDetailedView,
            tab_attr="_tune_tab_index",
            shortcut="Ctrl+5",
        )
        self._add_view_tab_action(
            view_menu,
            label="Inference tab",
            svg="inference.svg",
            fallback=QStyle.SP_FileDialogListView,
            tab_attr="_infer_tab_index",
            shortcut="Ctrl+6",
        )
        self._add_view_tab_action(
            view_menu,
            label="Transfer tab",
            svg="transfer.svg",
            fallback=QStyle.SP_ArrowRight,
            tab_attr="_xfer_tab_index",
            shortcut="Ctrl+7",
        )
        self._add_view_tab_action(
            view_menu,
            label="Results tab",
            svg="results.svg",
            fallback=QStyle.SP_DirHomeIcon,
            tab_attr="_results_tab_index",
            shortcut="Ctrl+8",
        )
        self._add_view_tab_action(
            view_menu,
            label="Tools tab",
            svg="tools.svg",
            fallback=QStyle.SP_FileDialogInfoView,
            tab_attr="_tools_tab_index",
            shortcut="Ctrl+9",
        )

        # Expose optional actions (handy for syncing)
        win.act_dark_mode = act_dark  # type: ignore[attr-defined]
        win.act_show_log = act_log  # type: ignore[attr-defined]

        # --------------------------- Help ---------------------------
        help_menu = menubar.addMenu("&Help")

        act_docs = QAction(
            self._std_icon(QStyle.SP_DialogHelpButton),
            "Online documentation…",
            win,
        )
        act_docs.setShortcut("F1")
        act_docs.triggered.connect(
            lambda: QDesktopServices.openUrl(
                QUrl(self.docs_url)
            )
        )
        help_menu.addAction(act_docs)

        help_menu.addSeparator()

        act_about = QAction(
            self._std_icon(QStyle.SP_MessageBoxInformation),
            "About GeoPrior…",
            win,
        )
        self._connect_if_exists(act_about, "_on_show_about")
        help_menu.addAction(act_about)
