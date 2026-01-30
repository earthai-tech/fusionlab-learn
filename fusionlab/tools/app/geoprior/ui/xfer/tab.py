# geoprior/ui/xfer/tab.py

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QMainWindow, 
    QSizePolicy
)

from ...config.store import GeoConfigStore
from .keys import K_VIEW_MODE, K_MAP_EXPANDED, K_MAP_OVERLAY  
from .map.page import XferMapPage
from .map.head import XferMapHeadBar
from .map.tool_dock import XferMapToolDock
from .run.preview import XferRunPreview
from .run.navigator import XferNavigator 
from .run.run_center import XferRunCenter 
from .run.head import XferHeadBar


class XferTab(QWidget):
    run_clicked = pyqtSignal()
    view_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: Callable[[str], tuple[QWidget, QVBoxLayout]],
        make_run_button: Callable[[str], QToolButton],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._make_card = make_card
        self._make_run_button = make_run_button

        self._build_ui()
        self._wire()
        self._apply_mode_from_store()
        self._apply_map_expanded_from_store()

        self._s.config_changed.connect(self._on_store_changed)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)
    
        # ---------------------------------------------------------
        # Head
        # ---------------------------------------------------------
        self.head = XferHeadBar(
            store=self._s,
            overlay_key=K_MAP_OVERLAY,
            parent=self,
        )
        root.addWidget(self.head, 0)
    
        # ---------------------------------------------------------
        # Main stack (run/map workspaces)
        # IMPORTANT: build workspaces BEFORE bottom bar so
        # status comes from preview (consistent readiness).
        # ---------------------------------------------------------
        self._stack = QStackedWidget(self)
        root.addWidget(self._stack, 1)
    
        self.run_ws = XferRunWorkspace(
            store=self._s,
            make_card=self._make_card,
            parent=self,
        )
        self._stack.addWidget(self.run_ws)
    
        self.map_ws = XferMapWorkspace(
            store=self._s,
            parent=self,
        )
        self._stack.addWidget(self.map_ws)
    
        # --- alias stable public widget refs (important)
        self._alias_public_widgets()
    
        # ---------------------------------------------------------
        # Bottom bar: status + Run
        # ---------------------------------------------------------
        bot = QHBoxLayout()
        bot.setContentsMargins(0, 0, 0, 0)
        bot.setSpacing(10)
    
        # Status should be THE SAME object preview updates
        self.status = self.run_ws.preview.status_label()
    
        self.lbl_run = QLabel("Run:", self)
        self.lbl_run.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    
        self.btn_run_xfer = self._make_run_button(
            "Run transfer matrix"
        )
    
        bot.addWidget(self.status, 1)
        bot.addWidget(self.lbl_run, 0)
        bot.addWidget(self.btn_run_xfer, 0)
    
        root.addLayout(bot, 0)

    def _alias_public_widgets(self) -> None:
        # Keep your existing XferTab API stable
        w = self.run_ws
        self.ed_city_a = w.ed_city_a
        self.ed_city_b = w.ed_city_b
        self.ed_results_root = w.ed_results_root
        self.sp_batch = w.sp_batch
        self.chk_rescale = w.chk_rescale

        self.chk_split_train = w.chk_split_train
        self.chk_split_val = w.chk_split_val
        self.chk_split_test = w.chk_split_test

        self.chk_cal_none = w.chk_cal_none
        self.chk_cal_source = w.chk_cal_source
        self.chk_cal_target = w.chk_cal_target

        self.cmb_view_kind = w.cmb_view_kind
        self.cmb_view_split = w.cmb_view_split
        self.btn_make_view = w.btn_make_view
        self.lbl_last_out = w.lbl_last_out
        self.btn_browse_root = w.btn_browse_root

        # self.btn_run_xfer = self.head.btn_run

        # Map page stays reachable
        self.map_page = self.map_ws.map_page

    def _wire(self) -> None:
        self.btn_run_xfer.clicked.connect(self.run_clicked.emit)
        self.run_ws.view_clicked.connect(self.view_clicked.emit)

        self.head.mode_changed.connect(self._set_mode)

        self.map_ws.expand_changed.connect(self._on_map_expand)
        self.map_ws.request_open_run.connect(
            lambda: self._set_mode("run")
        )
        # self.map_ws.head.mode_changed.connect(self._set_mode)

    def _mode_from_store(self) -> str:
        raw = str(self._s.get(K_VIEW_MODE, "run") or "")
        m = raw.strip().lower()
        if m == "options":
            return "run"
        if m not in ("run", "map"):
            return "run"
        return m

    def _apply_mode_from_store(self) -> None:
        self._set_mode(self._mode_from_store(), persist=False)

    
    def _set_mode(self, mode: str, *, persist: bool = True):
        m = "map" if str(mode).lower() == "map" else "run"
    
        # keep head visible and in sync
        self.head.set_mode(m)
    
        # switch workspace
        self._stack.setCurrentIndex(1 if m == "map" else 0)
    
        # bottom run button is only meaningful in run mode
        self.lbl_run.setVisible(m == "run")
        self.btn_run_xfer.setVisible(m == "run")
    
        if persist:
            self._s.set(K_VIEW_MODE, m)
    
        if m != "map":
            self.map_ws.set_expanded(False, persist=False)
        else:
            self._apply_map_expanded_from_store()
            QTimer.singleShot(0, self.map_ws._activate_map_ui)
            QTimer.singleShot(0, self.map_ws.map_page.refresh)

    def _on_map_expand(self, on: bool) -> None:
        if self._mode_from_store() != "map":
            self.map_ws.set_expanded(False, persist=False)
            return
        self.map_ws.set_expanded(bool(on))

    def _apply_map_expanded_from_store(self) -> None:
        want = bool(self._s.get(K_MAP_EXPANDED, False))
        if self._mode_from_store() == "map":
            self.map_ws.set_expanded(want, persist=False)
        else:
            self.map_ws.set_expanded(False, persist=False)

    def _on_store_changed(self, keys: object) -> None:

        # self.head.update_chips_from_store(self._s)

        try:
            changed = set(keys or [])
        except Exception:
            changed = set()

        if K_VIEW_MODE in changed:
            self._apply_mode_from_store()

        if K_MAP_EXPANDED in changed:
            self._apply_map_expanded_from_store()

    # -------------------------------------------------
    # Public helpers
    # -------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        st = dict(self.run_ws.center.get_state())
        # advanced is store-backed already; no merge needed
        return st

    def set_last_output(self, out_dir: Optional[str]) -> None:
        txt = out_dir or "No transfer run yet."
        self.lbl_last_out.setText(txt)
        self.btn_make_view.setVisible(bool(out_dir))

    def set_has_result(self, has: bool) -> None:
        self.btn_make_view.setVisible(bool(has))
        self.btn_make_view.setEnabled(bool(has))

    def set_view_enabled(self, enabled: bool) -> None:
        self.btn_make_view.setEnabled(bool(enabled))

    def set_run_enabled(self, enabled: bool) -> None:
        self.btn_run_xfer.setEnabled(bool(enabled))
        
class XferRunWorkspace(QWidget):
    view_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: Callable[[str], tuple[QWidget, QVBoxLayout]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._make_card = make_card
        self._build_ui()
        self._wire()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal, self)
        root.addWidget(self.splitter, 1)

        self.nav = XferNavigator(
            store= self._s, 
            make_card= self._make_card, 
            parent=self 
        )
        self.splitter.addWidget(self.nav)

        self.center = XferRunCenter(
            store=self._s,
            make_card=self._make_card,
            parent=self,
        )
        self.splitter.addWidget(self.center)

        # -----------------------------------------
        # Preview card (wrap to get the visible
        # frame like inference)
        # -----------------------------------------
        prev_card, prev_body = self._make_card(
            "Run preview"
        )
        self.preview = XferRunPreview(
            store=self._s,
            parent=prev_card,
        )
        prev_body.addWidget(self.preview, 1)
        self.splitter.addWidget(prev_card)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        # alias key fields from center cards
        self._alias_from_center()

    def _alias_from_center(self) -> None:
        c = self.center
        self.ed_city_a = c.ed_city_a
        self.ed_city_b = c.ed_city_b
        self.ed_results_root = c.ed_results_root
        self.btn_browse_root = c.btn_browse_root

        self.chk_split_train = c.chk_split_train
        self.chk_split_val = c.chk_split_val
        self.chk_split_test = c.chk_split_test

        self.chk_cal_none = c.chk_cal_none
        self.chk_cal_source = c.chk_cal_source
        self.chk_cal_target = c.chk_cal_target

        self.sp_batch = c.sp_batch
        self.chk_rescale = c.chk_rescale

        self.cmb_view_kind = c.cmb_view_kind
        self.cmb_view_split = c.cmb_view_split
        self.btn_make_view = c.btn_make_view
        self.lbl_last_out = c.lbl_last_out

    def _wire(self) -> None:
        self.btn_make_view.clicked.connect(self.view_clicked.emit)
        # self.nav.item_selected.connect(self.center.scroll_to)
        self.nav.clicked.connect(self.center.goto)

class XferMapWorkspace(QWidget):
    expand_changed = pyqtSignal(bool)
    request_open_run = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = store
        self._expanded = False
        self._build_ui()
        self._wire()


    def showEvent(self, e) -> None:
        super().showEvent(e)
        QTimer.singleShot(0, self._activate_map_ui)
    
    def _activate_map_ui(self) -> None:
        # Ensure dock is visible and has height *after* show.
        if hasattr(self, "tool_dock"):
            self.tool_dock.setVisible(True)
            self.tool_dock.raise_()
            self.tool_dock.setMinimumHeight(80)
    
            tb = getattr(self, "_tb", None)
            if tb is not None:
                h = tb.sizeHint().height()
                h = max(80, int(h) + 18)
            else:
                h = 120
    
            self._mw.resizeDocks(
                [self.tool_dock],
                [h],
                Qt.Vertical,
            )
    
        # Refresh map after it is actually shown.
        try:
            self.map_page.refresh()
        except Exception:
            pass

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
    
        # Head stays as info card (no toolbar in it)
        self.head = XferMapHeadBar(parent=self)
        root.addWidget(self.head, 0)
    
        # Dock host needs a QMainWindow. Make it native so
        # docks + QtWebEngine reliably paint.
        self._mw = QMainWindow(self)
        self._mw.setObjectName("xferMapMainWin")
        self._mw.setDockNestingEnabled(True)

        root.addWidget(self._mw, 1)
    
        # Central map page (let QMainWindow parent it)
        self.map_page = XferMapPage(store=self._s, parent=None)
        self._mw.setCentralWidget(self.map_page)
    
        # head must NOT eat the whole page should be on the top
        self.head.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        
        # the dock + map host must expand
        self._mw.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self._mw.setMinimumHeight(1)

        # Take existing toolbar from the page
        tb = self.map_page.toolbar
        self._tb = tb
    
        self.map_page.take_toolbar()
        if hasattr(self.map_page, "_toolbar_w"):
            self.map_page._toolbar_w.setVisible(False)
    
        # Dock widget (dock by default)
        self.tool_dock = XferMapToolDock(store=self._s, parent=self._mw)
        self.tool_dock.set_toolbar(tb)
        self._mw.addDockWidget(
            Qt.TopDockWidgetArea,
            self.tool_dock,
        )
        self.tool_dock.setVisible(True)

        if hasattr(tb, "set_mode"):
            tb.set_mode("map")
            
        def _post_layout() -> None:
            self.tool_dock.show()
        
            if self._tb is not None:
                self._tb.show()
        
                h = self._tb.sizeHint().height()
                h = max(80, int(h) + 18)
        
                self._mw.resizeDocks(
                    [self.tool_dock],
                    [h],
                    Qt.Vertical,
                )
        
            try:
                self.map_page.refresh()
            except Exception:
                pass

        QTimer.singleShot(0, _post_layout)
        

    def _wire(self) -> None:
        tb = getattr(self, "_tb", None)
    
        if tb is not None:
            tb.request_expand.connect(
                self.expand_changed.emit
            )
    
            tb.request_mode_switch.connect(
                self._on_toolbar_mode_switch
            )
    
            # legacy wiring kept alive
            tb.request_open_options.connect(
                self.request_open_run.emit
            )
    
        self.head.open_run_clicked.connect(
            self.request_open_run.emit
        )
    
        self.map_page.request_open_options.connect(
            self.request_open_run.emit
        )

        
    def _on_head_mode(self, mode: str) -> None:
        if str(mode).strip().lower() == "run":
            self.request_open_run.emit()
                
    def _on_toolbar_mode_switch(self, tgt: str) -> None:
        m = str(tgt or "").strip().lower()
        if m == "run":
            self.request_open_run.emit()

    def set_expanded(self, on: bool, *, persist: bool = True) -> None:
        on = bool(on)
    
        # tb = self._current_toolbar()
        tb = self._tb
        if tb is not None:
            tb.set_expanded(on)
    
        if on == self._expanded:
            return
    
        self._expanded = on
    
        # Expanded == hide head only (gain vertical space)
        self.head.setVisible(not on)
    
        if persist:
            cur = bool(self._s.get(K_MAP_EXPANDED, False))
            if cur != on:
                self._s.set(K_MAP_EXPANDED, on)

