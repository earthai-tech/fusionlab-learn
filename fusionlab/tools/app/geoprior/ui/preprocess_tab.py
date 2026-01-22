# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Preprocess tab UI for GeoPrior GUI (Stage-1).

UI-only:
- builds widgets
- binds Stage-1 options to GeoConfigStore (single source of truth)
- hosts Stage1Workspace (quicklook/readiness/feature scaling/etc.)

Business logic lives in app.py (controller).
"""

from __future__ import annotations

import os
from pathlib import Path
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QToolButton,   
    QStyle, 
    QScrollArea,
    QFrame, 
    QSizePolicy      
)

from ..config.prior_schema import FieldKey
from ..config.store import GeoConfigStore
from ..config.smart_stage1 import (
    canonical_hash_cfg,
    find_stage1_for_city,
    find_stage1_for_city_root,
    pick_best_stage1_run,
    
)
from .stage1_workspace.workspace import Stage1Workspace
from .stage1_workspace.readiness import (
    CompatibilityResult,
    Stage1Scan,
)
from .stage1_workspace.run_history import Stage1RunEntry
from .stage1_workspace.visual_checks import (
    Stage1VisualData,
)

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]
MakeRunBtnFn = Callable[[str], object]
Json = Dict[str, Any]


class PreprocessTab(QWidget):
    """Stage-1 preprocessing UI tab."""

    request_open_dataset = pyqtSignal()
    request_refresh = pyqtSignal()
    request_run_stage1 = pyqtSignal()
    request_feature_cfg = pyqtSignal()
    request_open_manifest = pyqtSignal()
    request_open_stage1_dir = pyqtSignal()
    request_use_for_city = pyqtSignal()
    request_browse_results_root = pyqtSignal()

    # From workspace subpanels
    request_open_path = pyqtSignal(str)
    request_set_active_stage1 = pyqtSignal(str, str)
    request_refresh_history = pyqtSignal()
    
    request_open_city_root = pyqtSignal()

    def __init__(
        self,
        *,
        make_card: MakeCardFn,
        make_run_button: MakeRunBtnFn,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._make_card = make_card
        self._make_run_button = make_run_button

        self._store: Optional[GeoConfigStore] = None
        self._ws_stage1_dir: Optional[str] = None
        self._ws_model: str = ""
        
        # ---- Stage-1 status cache (fast tab switching) ----
        self._prep_refresh_key = None  # tuple[str, str, str]
        self._prep_cache_best = None   # dict | None
        self._prep_cache_manifest = None
        self._prep_cache_audit = None


        self._build_ui()
        self.bind_store(store)
        self._wire()
        


    # ------------------------------------------------------------------
    # Store binding
    # ------------------------------------------------------------------
    def bind_store(self, store: GeoConfigStore) -> None:
        if self._store is store:
            return

        if self._store is not None:
            try:
                self._store.config_changed.disconnect(
                    self._on_store_changed
                )
            except Exception:
                pass
            try:
                self._store.config_replaced.disconnect(
                    self._on_store_replaced
                )
            except Exception:
                pass

        self._store = store

        self._store.config_changed.connect(
            self._on_store_changed
        )
        self._store.config_replaced.connect(
            self._on_store_replaced
        )

        self._sync_ui_from_store(keys=None)

    def _on_store_replaced(self, _cfg) -> None:
        self._sync_ui_from_store(keys=None)

    def _on_store_changed(self, keys) -> None:
        self._sync_ui_from_store(keys=set(keys or []))

    def _sync_ui_from_store(
        self,
        *,
        keys: Optional[set[str]],
    ) -> None:
        st = self._store
        if st is None:
            return

        def want(k: str) -> bool:
            return keys is None or (k in keys)

        if want("results_root"):
            rr = st.get_value(
                FieldKey("results_root"),
                default=None,
            )
            txt = "" if rr is None else str(rr)
            self.ed_prep_root.setText(txt)
            self.ed_prep_root.setToolTip(txt)

        if want("city"):
            city = st.get_value(
                FieldKey("city"),
                default="",
            )
            self.lbl_prep_city.setText(
                f"City: {city or '-'}"
            )

        if want("dataset_path"):
            ds = st.get_value(
                FieldKey("dataset_path"),
                default=None,
            )
            self.lbl_prep_csv.setText(
                f"Dataset: {ds or '-'}"
            )

        # Update computed city root whenever results_root or city changes
        if want("results_root") or want("city"):
            city_root = self._compute_city_root()
            self.ed_prep_city_root.setText(city_root)
            self.ed_prep_city_root.setToolTip(city_root or "")
            self.btn_prep_open_city_root.setEnabled(bool(city_root))

        self._sync_stage1_options_from_store(keys=keys)
        self._push_context_to_workspace()

    def _sync_stage1_options_from_store(
        self,
        *,
        keys: Optional[set[str]],
    ) -> None:
        st = self._store
        if st is None:
            return

        def want(k: str) -> bool:
            return keys is None or (k in keys)

        if want("clean_stage1_dir"):
            with QSignalBlocker(self.chk_prep_clean):
                self.chk_prep_clean.setChecked(
                    bool(
                        st.get_value(
                            FieldKey("clean_stage1_dir"),
                            default=False,
                        )
                    )
                )

        if want("build_future_npz"):
            with QSignalBlocker(self.chk_prep_build_future):
                self.chk_prep_build_future.setChecked(
                    bool(
                        st.get_value(
                            FieldKey("build_future_npz"),
                            default=False,
                        )
                    )
                )

        if want("stage1_auto_reuse_if_match"):
            with QSignalBlocker(self.chk_prep_auto_reuse):
                self.chk_prep_auto_reuse.setChecked(
                    bool(
                        st.get_value(
                            FieldKey(
                                "stage1_auto_reuse_if_match"
                            ),
                            default=True,
                        )
                    )
                )

        if want("stage1_force_rebuild_if_mismatch"):
            with QSignalBlocker(
                self.chk_prep_force_rebuild
            ):
                self.chk_prep_force_rebuild.setChecked(
                    bool(
                        st.get_value(
                            FieldKey(
                                "stage1_force_rebuild_if_mismatch"
                            ),
                            default=True,
                        )
                    )
                )

    def _push_context_to_workspace(self) -> None:
        st = self._store
        if st is None:
            return

        city = st.get_value(FieldKey("city"), default="")
        ds = st.get_value(FieldKey("dataset_path"), default=None)
        rr = st.get_value(FieldKey("results_root"), default=None)
        rr = None if rr is None else str(rr)

        self.stage1_ws.set_context(
            city=str(city or ""),
            csv_path=ds,
            runs_root=rr,              # global root (tests/)
            stage1_dir=self._ws_stage1_dir,  # active run dir (train_... or stage1 run dir)
            model=self._ws_model,
        )

    def _compute_city_root(self) -> str:
        """
        Compute city root folder:
            <results_root>/<city>_<model>_stage1/

        Returns "" if missing inputs.
        """
        st = self._store
        if st is None:
            return ""

        rr = st.get_value(FieldKey("results_root"), default=None)
        city = st.get_value(FieldKey("city"), default="")

        rr = "" if rr is None else str(rr).strip()
        city = str(city or "").strip()
        if not rr or not city:
            return ""

        # Prefer workspace model if already known; fallback to store/model
        model = (self._ws_model or "").strip()
        if not model:
            # Optional: if you store model somewhere, use it here.
            # Otherwise fallback to GeoPriorSubsNet
            model = "GeoPriorSubsNet"

        base = Path(rr).expanduser()
        folder = f"{city}_{model}_stage1"
        return str(base / folder)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def stage1_options(self) -> dict:
        """
        Current Stage-1 option snapshot (from store).

        Kept for backward-compat with existing controller code.
        """
        st = self._store
        if st is None:
            return {}

        return {
            "clean_stage1_dir": bool(
                st.get_value(
                    FieldKey("clean_stage1_dir"),
                    default=False,
                )
            ),
            "build_future_npz": bool(
                st.get_value(
                    FieldKey("build_future_npz"),
                    default=False,
                )
            ),
            "stage1_auto_reuse_if_match": bool(
                st.get_value(
                    FieldKey("stage1_auto_reuse_if_match"),
                    default=True,
                )
            ),
            "stage1_force_rebuild_if_mismatch": bool(
                st.get_value(
                    FieldKey(
                        "stage1_force_rebuild_if_mismatch"
                    ),
                    default=True,
                )
            ),
        }

    def set_stage1_status(
        self,
        *,
        state_text: str,
        manifest_text: str,
    ) -> None:
        self.lbl_prep_stage1_state.setText(state_text)
        self.lbl_prep_stage1_manifest.setText(manifest_text)

    # ---- Workspace pass-throughs (controller-friendly) ----
    def set_workspace_context(
        self,
        *,
        stage1_dir: Optional[str] = None,
        model: str = "",
    ) -> None:
        self._ws_stage1_dir = stage1_dir
        self._ws_model = model or ""
        self._push_context_to_workspace()

    def clear_workspace(self) -> None:
        self.stage1_ws.clear()

    def set_workspace_manifest(self, manifest: Optional[Json]) -> None:
        self.stage1_ws.set_manifest(manifest)

    def set_workspace_scaling_audit(self, audit: Optional[Json]) -> None:
        self.stage1_ws.set_scaling_audit(audit)

    def set_workspace_readiness_payload(
        self,
        *,
        options: Optional[Dict[str, Any]] = None,
        scan: Optional[Stage1Scan] = None,
        compat: Optional[CompatibilityResult] = None,
        status: str = "",
    ) -> None:
        self.stage1_ws.set_readiness_payload(
            options=options,
            scan=scan,
            compat=compat,
            status=status,
        )

    def set_workspace_artifacts_extra(
        self,
        items: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.stage1_ws.set_artifacts_extra(items)

    def set_workspace_history_entries(
        self,
        entries: Optional[List[Stage1RunEntry]],
    ) -> None:
        self.stage1_ws.set_history_entries(entries)

    def set_workspace_visual_data(
        self,
        data: Optional[Stage1VisualData],
    ) -> None:
        self.stage1_ws.set_visual_data(data)

    def set_workspace_active_tab(self, name: str) -> None:
        self.stage1_ws.set_active_tab(name)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        page = QWidget(scroll)
        scroll.setWidget(page)

        p_layout = QVBoxLayout(page)
        p_layout.setContentsMargins(0, 0, 0, 0)
        p_layout.setSpacing(8)

        # -------------------------------------------------
        # Row 1: Results root (full width)
        # -------------------------------------------------
        # -------------------------------------------------
        # Top bar: Paths (icon-only actions, City root is primary)
        # -------------------------------------------------
        paths = QGridLayout()
        paths.setHorizontalSpacing(8)
        paths.setVerticalSpacing(0)

        lbl_city_root = QLabel("City root:")
        lbl_root = QLabel("Results root:")

        # Stable label widths (prevents jitter)
        lbl_city_root.setFixedWidth(70)
        lbl_root.setFixedWidth(85)

        # --- Fields ---
        self.ed_prep_city_root = QLineEdit()
        self.ed_prep_city_root.setReadOnly(True)
        self.ed_prep_city_root.setPlaceholderText(
            "Auto-computed from results root + city (+ model)"
        )

        self.ed_prep_root = QLineEdit()
        self.ed_prep_root.setReadOnly(True)
        self.ed_prep_root.setPlaceholderText("Select results root…")

        # --- Icon-only buttons ---
        def _icon_btn(std_icon: QStyle.StandardPixmap, tip: str) -> QToolButton:
            b = QToolButton()
            b.setObjectName("miniAction")  #  binds to styles rules
            b.setIcon(self.style().standardIcon(std_icon))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedSize(28, 28)
            return b


        self.btn_prep_open_city_root = _icon_btn(
            QStyle.SP_DirOpenIcon, "Open city folder"
        )
        self.btn_prep_open_city_root.setEnabled(False)

        self.btn_prep_browse_root = _icon_btn(
            QStyle.SP_DialogOpenButton, "Browse results root…"
        )

        self.btn_prep_refresh = _icon_btn(
            QStyle.SP_BrowserReload, "Refresh Stage-1 status"
        )

        # Accessibility (optional but good): screen readers / tests
        self.btn_prep_open_city_root.setAccessibleName("Open city folder")
        self.btn_prep_browse_root.setAccessibleName("Browse results root")
        self.btn_prep_refresh.setAccessibleName("Refresh preprocess status")

        # Layout: [City root][Open] [Results root][Browse][Refresh]
        paths.addWidget(lbl_city_root, 0, 0)
        paths.addWidget(self.ed_prep_city_root, 0, 1)
        paths.addWidget(self.btn_prep_open_city_root, 0, 2)

        paths.addWidget(lbl_root, 0, 3)
        paths.addWidget(self.ed_prep_root, 0, 4)
        paths.addWidget(self.btn_prep_browse_root, 0, 5)
        paths.addWidget(self.btn_prep_refresh, 0, 6)

        # Stretch: city root gets more space than results root
        paths.setColumnStretch(1, 6)  # city root field (big)
        paths.setColumnStretch(4, 3)  # results root field (smaller)

        p_layout.addLayout(paths)

        # -------------------------------------------------
        # Top row: 3 cards
        # -------------------------------------------------
        row = QHBoxLayout()
        row.setSpacing(10)

        # ---------------- Card 1: Inputs ----------------
        inp_card, inp_box = self._make_card(
            "Inputs (City + Dataset)"
        )

        self.lbl_prep_city = QLabel("City: -")
        self.lbl_prep_csv = QLabel("Dataset: -")

        for w in (self.lbl_prep_city, self.lbl_prep_csv):
            w.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            inp_box.addWidget(w)

        self.btn_prep_open_dataset = QPushButton(
            "Open dataset…"
        )
        self.btn_prep_feature_cfg = QPushButton(
            "Feature config…"
        )

        inp_btns = QHBoxLayout()
        inp_btns.setSpacing(8)
        inp_btns.addWidget(self.btn_prep_open_dataset)
        inp_btns.addWidget(self.btn_prep_feature_cfg)
        inp_btns.addStretch(1)
        inp_box.addLayout(inp_btns)

        row.addWidget(inp_card, 1)

        # ------------- Card 2: Stage-1 options ----------
        opt_card, opt_box = self._make_card("Stage-1 options")
        
        self.chk_prep_clean = QCheckBox("Clean Stage-1 run dir before build")
        self.chk_prep_auto_reuse = QCheckBox("Auto-reuse compatible Stage-1 run")
        self.chk_prep_force_rebuild = QCheckBox("Force rebuild if mismatch")
        self.chk_prep_build_future = QCheckBox("Build future NPZ")
        
        opt_box.addWidget(self.chk_prep_clean)
        opt_box.addWidget(self.chk_prep_auto_reuse)
        
        row_force = QWidget(opt_card)
        gl = QGridLayout(row_force)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setHorizontalSpacing(12)
        gl.setVerticalSpacing(6)
        
        gl.addWidget(self.chk_prep_force_rebuild, 0, 0)
        gl.addWidget(self.chk_prep_build_future, 0, 1)
        gl.setColumnStretch(0, 1)
        gl.setColumnStretch(1, 1)
        
        opt_box.addWidget(row_force)

        row.addWidget(opt_card, 1)

        # -------------- Card 3: Stage-1 status ----------
        stat_card, stat_box = self._make_card(
            "Stage-1 status"
        )

        self.lbl_prep_stage1_state = QLabel(
            "Stage-1: unknown"
        )
        self.lbl_prep_stage1_manifest = QLabel("Manifest: -")

        for w in (
            self.lbl_prep_stage1_state,
            self.lbl_prep_stage1_manifest,
        ):
            w.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            stat_box.addWidget(w)

        btns = QHBoxLayout()
        btns.setSpacing(8)

        self.btn_prep_open_manifest = QPushButton(
            "Open manifest"
        )
        self.btn_prep_open_stage1_dir = QPushButton(
            "Open folder"
        )
        self.btn_prep_use_for_city = QPushButton(
            "Use as default for city"
        )

        btns.addWidget(self.btn_prep_open_manifest)
        btns.addWidget(self.btn_prep_open_stage1_dir)
        btns.addWidget(self.btn_prep_use_for_city)
        btns.addStretch(1)
        stat_box.addLayout(btns)

        row.addWidget(stat_card, 1)
        p_layout.addLayout(row)

        # -------------------------------------------------
        # Workspace (expands)
        # -------------------------------------------------
        ws_card, ws_box = self._make_card(
            "Stage-1 workspace"
        )
        ws_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        
        self.stage1_ws = Stage1Workspace()
        ws_box.addWidget(self.stage1_ws, 1)
        p_layout.addWidget(ws_card, 1)
        # p_layout.addStretch(1)\

        self.stage1_ws.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        
        self.stage1_ws.tabs.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        def _dump(layout):
            for i in range(layout.count()):
                it = layout.itemAt(i)
                w = it.widget()
                s = it.spacerItem()
                print(i, "W", type(w).__name__ if w else None,
                      "S", bool(s), "stretch", layout.stretch(i))
                
        if os.getenv("GEOPRIOR_DEBUG_LAYOUT") == "1":
            _dump(ws_box)
  
        # -------------------------------------------------
        # Bottom: Run (right)
        # -------------------------------------------------
        outer.addWidget(scroll, 1)

        run_row = QHBoxLayout()
        run_row.setSpacing(10)

        self.btn_run_stage1 = self._make_run_button(
            "Run Stage-1 preprocessing"
        )
        run_row.addStretch(1)
        run_row.addWidget(self.btn_run_stage1)
        outer.addLayout(run_row)

    def _wire(self) -> None:
        # Top buttons
        self.btn_prep_open_dataset.clicked.connect(
            self.request_open_dataset.emit
        )
        self.btn_prep_feature_cfg.clicked.connect(
            self.request_feature_cfg.emit
        )

        self.btn_prep_refresh.clicked.connect(
            self.request_refresh.emit
        )
        self.btn_prep_browse_root.clicked.connect(
            self.request_browse_results_root.emit
        )

        self.btn_run_stage1.clicked.connect(
            self.request_run_stage1.emit
        )

        self.btn_prep_open_manifest.clicked.connect(
            self.request_open_manifest.emit
        )
        self.btn_prep_open_stage1_dir.clicked.connect(
            self.request_open_stage1_dir.emit
        )
        self.btn_prep_use_for_city.clicked.connect(
            self.request_use_for_city.emit
        )

        # Stage-1 options -> store
        self.chk_prep_clean.toggled.connect(
            self._on_clean_toggled
        )
        self.chk_prep_build_future.toggled.connect(
            self._on_build_future_toggled
        )
        self.chk_prep_auto_reuse.toggled.connect(
            self._on_auto_reuse_toggled
        )
        self.chk_prep_force_rebuild.toggled.connect(
            self._on_force_rebuild_toggled
        )

        # Workspace signals -> controller
        self.stage1_ws.request_open_path.connect(
            self.request_open_path.emit
        )
        self.stage1_ws.request_set_active_stage1.connect(
            self.request_set_active_stage1.emit
        )
        self.stage1_ws.request_refresh_history.connect(
            self.request_refresh_history.emit
        )
        self.btn_prep_open_city_root.clicked.connect(
            self.request_open_city_root.emit
        )


    # ------------------------------------------------------------------
    # Stage-1 options handlers
    # ------------------------------------------------------------------
    def _set_bool(self, name: str, value: bool) -> None:
        st = self._store
        if st is None:
            return
        st.set_value_by_key(FieldKey(name), bool(value))

    def _on_clean_toggled(self, v: bool) -> None:
        self._set_bool("clean_stage1_dir", v)

    def _on_build_future_toggled(self, v: bool) -> None:
        self._set_bool("build_future_npz", v)

    def _on_auto_reuse_toggled(self, v: bool) -> None:
        self._set_bool("stage1_auto_reuse_if_match", v)

    def _on_force_rebuild_toggled(self, v: bool) -> None:
        self._set_bool(
            "stage1_force_rebuild_if_mismatch",
            v,
        )

    def best_stage1_dir(self) -> Optional[str]:
        b = self._prep_cache_best or {}
        return b.get("stage1_dir")

    def best_manifest_path(self) -> Optional[str]:
        b = self._prep_cache_best or {}
        return b.get("manifest_path")

    def best_model(self) -> str:
        b = self._prep_cache_best or {}
        return str(b.get("model") or "")

    def city_root_path(self) -> str:
        return (self.ed_prep_city_root.text() or "").strip()
    
    def _set_stage1_actions(
        self,
        *,
        stage1_dir: Optional[str],
        manifest_path: Optional[str],
    ) -> None:
        has_dir = bool(stage1_dir)
        has_mf = bool(manifest_path)

        self.btn_prep_open_stage1_dir.setEnabled(has_dir)
        self.btn_prep_open_manifest.setEnabled(has_mf)
        self.btn_prep_use_for_city.setEnabled(has_dir)

        if not has_dir:
            tip = "Run Stage-1 first."
            self.btn_prep_open_stage1_dir.setToolTip(tip)
            self.btn_prep_use_for_city.setToolTip(tip)
        else:
            self.btn_prep_open_stage1_dir.setToolTip(stage1_dir or "")
            self.btn_prep_use_for_city.setToolTip(stage1_dir or "")

        if not has_mf:
            self.btn_prep_open_manifest.setToolTip(
                "Run Stage-1 to create a manifest."
            )
        else:
            self.btn_prep_open_manifest.setToolTip(manifest_path or "")

    def refresh_status(self, *, force: bool = False) -> None:
        st = self._store
        if st is None:
            return

        # --- inputs from store ---
        city = str(st.get_value(FieldKey("city"), default="") or "").strip()
        rr_raw = st.get_value(FieldKey("results_root"), default=None)
        rr = "" if rr_raw is None else str(rr_raw).strip()
        rr_path = Path(rr).expanduser() if rr else None

        # --- stage1 cfg snapshot (for match + cache key) ---
        cfg_snap = None
        try:
            # Prefer "pure" if you want to avoid NAT dependency
            cfg_snap = st.cfg.to_stage1_cfg(pure=True)
        except Exception:
            try:
                cfg_snap = st.cfg.to_stage1_config()
            except Exception:
                cfg_snap = None

        key = (rr, city.lower(), canonical_hash_cfg(cfg_snap))
        if (not force) and (key == self._prep_refresh_key):
            if self._prep_cache_best is not None:
                best = self._prep_cache_best
                self.set_stage1_status(
                    state_text=best["state"],
                    manifest_text=best["mf_text"],
                )
                self._set_stage1_actions(
                    stage1_dir=best["stage1_dir"],
                    manifest_path=best["manifest_path"],
                )
                self.set_workspace_context(
                    stage1_dir=best["stage1_dir"],
                    model=best["model"],
                )
                self.set_workspace_manifest(self._prep_cache_manifest)
                self.set_workspace_scaling_audit(self._prep_cache_audit)
                return

        self._prep_refresh_key = key

        # --- missing city/root cases ---
        if not city:
            self.set_stage1_status(
                state_text="Stage-1: (no city selected)",
                manifest_text="Manifest: -",
            )
            self._set_stage1_actions(stage1_dir=None, manifest_path=None)
            self.set_workspace_context(stage1_dir=None, model="")
            self.set_workspace_manifest(None)
            self.set_workspace_scaling_audit(None)
            self._prep_cache_best = None
            self._prep_cache_manifest = None
            self._prep_cache_audit = None
            return

        if rr_path is None:
            self.set_stage1_status(
                state_text="Stage-1: (no results root selected)",
                manifest_text="Manifest: -",
            )
            self._set_stage1_actions(stage1_dir=None, manifest_path=None)
            return

        # --- discover runs (FAST: city_root only if it exists) ---
        city_root_txt = self._compute_city_root()
        city_root = Path(city_root_txt).expanduser() if city_root_txt else None

        runs = []
        try:
            if city_root is not None and city_root.is_dir():
                runs, _ = find_stage1_for_city_root(
                    city_root=city_root,
                    current_cfg=cfg_snap,
                )
            else:
                runs, _ = find_stage1_for_city(
                    city=city,
                    results_root=rr_path,
                    current_cfg=cfg_snap,
                )
        except Exception:
            runs = []

        best = pick_best_stage1_run(runs)
        if best is None:
            self.set_stage1_status(
                state_text="Stage-1: not found for this city",
                manifest_text="Manifest: -",
            )
            self._set_stage1_actions(stage1_dir=None, manifest_path=None)
            self.set_workspace_context(stage1_dir=None, model="")
            self.set_workspace_manifest(None)
            self.set_workspace_scaling_audit(None)
            self._prep_cache_best = None
            self._prep_cache_manifest = None
            self._prep_cache_audit = None
            return

        tag = "OK" if best.is_complete else "INCOMPLETE"
        match = "MATCH" if best.config_match else "MISMATCH"

        state = (
            f"Stage-1: {tag} / {match} "
            f"(n_train={best.n_train}, n_val={best.n_val})"
        )
        mf_text = f"Manifest: {best.manifest_path}"

        stage1_dir = str(best.run_dir)
        mf_path = str(best.manifest_path)
        model = str(best.model or "GeoPriorSubsNet")

        self.set_stage1_status(state_text=state, manifest_text=mf_text)
        self._set_stage1_actions(stage1_dir=stage1_dir, manifest_path=mf_path)
        self.set_workspace_context(stage1_dir=stage1_dir, model=model)

        # --- load manifest + audit (safe) ---
        manifest = None
        audit = None
        try:
            manifest = json.loads(
                Path(best.manifest_path).read_text(encoding="utf-8")
            )
        except Exception:
            manifest = None

        audit_path = Path(best.run_dir) / "stage1_scaling_audit.json"
        if audit_path.is_file():
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except Exception:
                audit = None

        self.set_workspace_manifest(manifest)
        self.set_workspace_scaling_audit(audit)

        # --- cache for next tab entry ---
        self._prep_cache_best = dict(
            state=state,
            mf_text=mf_text,
            stage1_dir=stage1_dir,
            manifest_path=mf_path,
            model=model,
        )
        self._prep_cache_manifest = manifest
        self._prep_cache_audit = audit
