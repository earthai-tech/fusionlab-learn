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

from PyQt5.QtCore import ( 
    Qt, 
    QSignalBlocker, 
    QPoint,
    QSize, 
    pyqtSignal, 
)
from PyQt5.QtGui import (
    QColor,
    QPainter,
    QPen,
    QPixmap,
    QPalette,
    QIcon
)
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
    QSizePolicy, 
    QButtonGroup,
    QSplitter,
    QStackedWidget, 
)

from ..config.prior_schema import FieldKey
from ..config.store import GeoConfigStore
from ..config.smart_stage1 import (
    canonical_hash_cfg,
    find_stage1_for_city,
    find_stage1_for_city_root,
    pick_best_stage1_run,
    
)
from ..device_options import runtime_summary_text

from .stage1_workspace.workspace import Stage1Workspace
from .stage1_workspace.readiness import (
    CompatibilityResult,
    Stage1Scan,
)
from .stage1_workspace.run_history import Stage1RunEntry
from .stage1_workspace.visual_checks import (
    Stage1VisualData,
)
from .icon_utils import try_icon
from .kv_panel import KeyValuePanel 


MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]
MakeRunBtnFn = Callable[[str], object]
Json = Dict[str, Any]

class PreprocessTab(QWidget):
    """Stage-1 preprocessing UI tab."""
    
    _RUNTIME_KEYS = {
        "backend",
        "engine",
        "framework",
        "dl_backend",
        "nn_backend",
        "trainer_backend",
        "compute_backend",
        "tf_device_mode",
        "tf_intra_threads",
        "tf_inter_threads",
        "tf_gpu_allow_growth",
        "tf_gpu_memory_limit_mb",
    }
        
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
        self._refresh_preview_only()
        self._refresh_runtime_snapshot(keys=keys)
        self._update_recap()

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
    # View mode: Build / Inspect
    # ------------------------------------------------------------------
    def _set_view_mode(self, mode: str) -> None:
        mode = (mode or "").strip().lower()
        if mode not in ("build", "inspect"):
            mode = "build"

        self._view_mode = mode

        if mode == "build":
            self.btn_mode_build.setChecked(True)
            self._stack.setCurrentIndex(0)
        else:
            self.btn_mode_inspect.setChecked(True)
            self._stack.setCurrentIndex(1)

    def _goto_build_card(self, card: QWidget) -> None:
        self._set_view_mode("build")
        try:
            self._build_scroll.ensureWidgetVisible(card, 0, 20)
        except Exception:
            pass

    def _goto_inspect_tab(self, tab_name: str) -> None:
        self._set_view_mode("inspect")
        try:
            self.stage1_ws.set_active_tab(tab_name)
        except Exception:
            pass

    def _update_run_preview(
        self,
        *,
        decision: str,
        reason: str,
        stage1_dir: Optional[str],
        manifest_path: Optional[str],
    ) -> None:
        dec = (decision or "").strip()
        rsn = (reason or "").strip()

        self.lbl_prep_decision.setText(dec or "-")
        self.lbl_prep_reason.setText(rsn or "")

        s1 = "" if stage1_dir is None else str(stage1_dir)
        mf = "" if manifest_path is None else str(manifest_path)

        self.lbl_prep_best_dir.setText(s1 or "-")
        self.lbl_prep_best_mf.setText(mf or "-")

        self.btn_prep_preview_open_dir.setEnabled(bool(s1))
        self.btn_prep_preview_open_mf.setEnabled(bool(mf))
        self._set_reco_chip(dec)
        self._apply_run_button_state(dec)
        
        if "FOUND (MATCH)" in dec:
            self.btn_prep_use_for_city.setDefault(True)
        else:
            self.btn_prep_use_for_city.setDefault(False)
            
        self._update_recap()

        
    # --------------------------------------------------------------
    # Decision logic helpers
    # --------------------------------------------------------------
    def _opt_bool(
        self,
        key: str,
        default: bool,
    ) -> bool:
        st = self._store
        if st is None:
            return bool(default)

        return bool(
            st.get_value(
                FieldKey(key),
                default=default,
            )
        )

    def _compute_preview_decision(
        self,
        *,
        has_best: bool,
        is_complete: bool,
        config_match: bool,
    ) -> tuple[str, str]:
        auto_reuse = self._opt_bool(
            "stage1_auto_reuse_if_match",
            True,
        )
        force_rb = self._opt_bool(
            "stage1_force_rebuild_if_mismatch",
            True,
        )
        clean = self._opt_bool(
            "clean_stage1_dir",
            False,
        )
        future = self._opt_bool(
            "build_future_npz",
            False,
        )

        if not has_best:
            dec = "BUILD"
            rsn = (
                "No Stage-1 run found. "
                "Press Run to build Stage-1."
            )
            return dec, rsn

        if not is_complete:
            dec = "BUILD"
            rsn = (
                "Found a Stage-1 run, but it is incomplete. "
                "Press Run to rebuild outputs."
            )
        elif config_match:
            if auto_reuse:
                dec = "REUSE"
                rsn = (
                    "Compatible Stage-1 run found. "
                    "Auto-reuse is ON."
                )
            else:
                dec = "FOUND (MATCH) • MANUAL"
                rsn = (
                    "Compatible Stage-1 run found, "
                    "but auto-reuse is OFF. "
                    "Click 'Use as default for city' "
                    "to reuse it, or press Run to rebuild."
                )
        else:
            if force_rb:
                dec = "REBUILD"
                rsn = (
                    "Stage-1 run found, but config mismatches. "
                    "Force-rebuild is ON. "
                    "Press Run to rebuild with current config."
                )
            else:
                dec = "FOUND (MISMATCH) • MANUAL"
                rsn = (
                    "Stage-1 run found, but config mismatches. "
                    "Force-rebuild is OFF. "
                    "You may reuse it manually "
                    "(not recommended), or press Run to rebuild."
                )

        extras: list[str] = []
        if dec in ("BUILD", "REBUILD"):
            if clean:
                extras.append("Clean is ON.")
            if future:
                extras.append("Future NPZ is ON.")

        if extras:
            rsn = f"{rsn} " + " ".join(extras)

        return dec, rsn

    def _refresh_preview_only(self) -> None:
        st = self._store
        if st is None:
            return

        city = str(
            st.get_value(
                FieldKey("city"),
                default="",
            )
            or ""
        ).strip()

        rr_raw = st.get_value(
            FieldKey("results_root"),
            default=None,
        )
        rr = "" if rr_raw is None else str(rr_raw).strip()

        if not city:
            self._update_run_preview(
                decision="WAITING",
                reason="No city selected.",
                stage1_dir=None,
                manifest_path=None,
            )
            return

        if not rr:
            self._update_run_preview(
                decision="WAITING",
                reason="No results root selected.",
                stage1_dir=None,
                manifest_path=None,
            )
            return

        b = self._prep_cache_best or {}
        has_best = bool(b.get("stage1_dir"))

        dec, rsn = self._compute_preview_decision(
            has_best=has_best,
            is_complete=bool(b.get("is_complete", False)),
            config_match=bool(b.get("config_match", False)),
        )

        self._update_run_preview(
            decision=dec,
            reason=rsn,
            stage1_dir=b.get("stage1_dir"),
            manifest_path=b.get("manifest_path"),
        )

        if self._prep_cache_best is not None:
            self._prep_cache_best["decision"] = dec
            self._prep_cache_best["reason"] = rsn

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._view_mode = "build"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # ==============================================================
        # Main split: [A+E] | [B + (C/D or Inspect)]
        # ==============================================================
        main = QSplitter(Qt.Horizontal, self)
        main.setChildrenCollapsible(False)

        # -------------------------
        # Left: Navigator (A) + Info (E)
        # -------------------------
        left = QWidget(self)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(8)

        nav_card, nav_box = self._make_card("Setup checklist")

        def _nav_btn(
            text: str,
            svg_name: str,
            fallback_std: QStyle.StandardPixmap,
            *,
            mini: bool = False,
        ) -> QPushButton:
            b = QPushButton(text, self)
        
            ico = try_icon(svg_name) if svg_name else None
            if (ico is None) or ico.isNull():
                ico = self.style().standardIcon(fallback_std)
        
            b.setIcon(ico)
            b.setIconSize(QSize(16, 16))
            b.setCursor(Qt.PointingHandCursor)
            b.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Fixed,
            )
            b.setLayoutDirection(Qt.LeftToRight)
        
            if mini:
                # QPushButton#miniAction
                b.setObjectName("miniAction")
            else:
                b.setObjectName("navBtn")
        
            return b
        
        # --- icon + button (fallback if svg missing)
        self.nav_inputs = _nav_btn(
            "Inputs",
            "inputs.svg",
            QStyle.SP_FileIcon,
        )
        self.nav_policy = _nav_btn(
            "Stage-1 policy",
            "policy.svg",
            QStyle.SP_MessageBoxInformation,
        )
        self.nav_status = _nav_btn(
            "Stage-1 status",
            "status",
            QStyle.SP_DialogApplyButton,
        )
        
        # --- miniAction look (we'll style later)
        self.nav_readiness = _nav_btn(
            "Readiness",
            "readiness.svg",
            QStyle.SP_DialogYesButton,
            mini=True,
        )
        self.nav_inspect = _nav_btn(
            "Inspect",
            "inspect.svg",
            QStyle.SP_FileDialogContentsView,
            mini=True,
        )
        self.nav_history = _nav_btn(
            "Run history",
            "history.svg",
            QStyle.SP_BrowserReload,
            mini=True,
        )
        self.nav_artifacts = _nav_btn(
            "Artifacts",
            "artifacts.svg",
            QStyle.SP_DirIcon,
            mini=True,
        )
        nav_box.addWidget(self.nav_inputs)
        nav_box.addWidget(self.nav_policy)
        nav_box.addWidget(self.nav_status)
        nav_box.addSpacing(6)
        nav_box.addWidget(self.nav_readiness)
        nav_box.addWidget(self.nav_inspect)
        nav_box.addSpacing(6)
        nav_box.addWidget(self.nav_history)
        nav_box.addWidget(self.nav_artifacts)
        nav_box.addStretch(1)

        left_lay.addWidget(nav_card, 0)

        info_card, info_box = self._make_card("Computer details")
        
        self.comp_panel = KeyValuePanel(self, max_rows=12, compact=True)
        info_box.addWidget(self.comp_panel, 0)
        info_box.addStretch(1)
        
        left_lay.addWidget(info_card, 1)

        # Make left stable but not fixed
        left.setMinimumWidth(230)
        left.setSizePolicy(
            QSizePolicy.Minimum,
            QSizePolicy.Expanding,
        )

        main.addWidget(left)

        # -------------------------
        # Right: Head (B) + stack
        # -------------------------
        right = QWidget(self)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        # ==============================================================
        # Head bar (B)
        # ==============================================================
        head = QWidget(self)
        head_lay = QHBoxLayout(head)
        head_lay.setContentsMargins(0, 0, 0, 0)
        head_lay.setSpacing(8)

        # ---- Segmented switch: Build | Inspect
        self.btn_mode_build = QToolButton(self)
        self.btn_mode_build.setText("Build")
        self.btn_mode_build.setCheckable(True)
        self.btn_mode_build.setAutoRaise(True)
        self.btn_mode_build.setCursor(Qt.PointingHandCursor)

        self.btn_mode_inspect = QToolButton(self)
        self.btn_mode_inspect.setText("Inspect")
        self.btn_mode_inspect.setCheckable(True)
        self.btn_mode_inspect.setAutoRaise(True)
        self.btn_mode_inspect.setCursor(Qt.PointingHandCursor)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self.btn_mode_build)
        self._mode_group.addButton(self.btn_mode_inspect)

        self.btn_mode_build.setChecked(True)

        head_lay.addWidget(self.btn_mode_build, 0)
        head_lay.addWidget(self.btn_mode_inspect, 0)
        head_lay.addSpacing(10)

        # ---- Path controls
        def _icon_btn(
            svg_name: str,
            fallback_std: QStyle.StandardPixmap,
            tip: str,
        ) -> QToolButton:
            b = QToolButton(self)
            ico = try_icon(svg_name) if svg_name else None
            if (ico is None) or ico.isNull():
                ico = self.style().standardIcon(fallback_std)
        
            b.setIcon(ico)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedSize(28, 28)
            return b

        head_lay.addWidget(QLabel("City root:"), 0)

        self.ed_prep_city_root = QLineEdit(self)
        self.ed_prep_city_root.setReadOnly(True)
        self.ed_prep_city_root.setPlaceholderText(
            "Auto-computed from results root + city (+ model)"
        )

        self.btn_prep_open_city_root = _icon_btn(
            "folder_open.svg",
            QStyle.SP_DirOpenIcon,
            "Open city folder",
        )
        self.btn_prep_open_city_root.setEnabled(False)

        head_lay.addWidget(self.ed_prep_city_root, 1)
        head_lay.addWidget(self.btn_prep_open_city_root, 0)

        head_lay.addSpacing(10)
        head_lay.addWidget(QLabel("Results root:"), 0)

        self.ed_prep_root = QLineEdit(self)
        self.ed_prep_root.setReadOnly(True)
        self.ed_prep_root.setPlaceholderText(
            "Select results root…"
        )

        self.btn_prep_browse_root = _icon_btn(
            "folder_search.svg",
            QStyle.SP_DialogOpenButton,
            "Browse results root…",
        )
        
        self.btn_prep_refresh = _icon_btn(
            "refresh.svg",
            QStyle.SP_BrowserReload,
            "Refresh Stage-1 status",
        )

        head_lay.addWidget(self.ed_prep_root, 1)
        head_lay.addWidget(self.btn_prep_browse_root, 0)
        head_lay.addWidget(self.btn_prep_refresh, 0)

        right_lay.addWidget(head, 0)

        # ==============================================================
        # Stack: Build page (C/D) | Inspect page (workspace)
        # ==============================================================
        self._stack = QStackedWidget(self)

        # -------------------------
        # Build page: [C | D]
        # -------------------------
        build_page = QWidget(self)
        build_lay = QVBoxLayout(build_page)
        build_lay.setContentsMargins(0, 0, 0, 0)
        build_lay.setSpacing(0)

        build_split = QSplitter(Qt.Horizontal, self)
        build_split.setChildrenCollapsible(False)

        # C: cards (scroll)
        self._build_scroll = QScrollArea(self)
        self._build_scroll.setWidgetResizable(True)
        self._build_scroll.setFrameShape(QFrame.NoFrame)

        build_body = QWidget(self._build_scroll)
        self._build_scroll.setWidget(build_body)

        c_lay = QVBoxLayout(build_body)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(10)

        # ---- Card: Inputs
        self.card_inputs, inp_box = self._make_card(
            "Inputs (City + Dataset)"
        )
        self.lbl_prep_city = QLabel("City: -", self)
        self.lbl_prep_csv = QLabel("Dataset: -", self)

        for w in (self.lbl_prep_city, self.lbl_prep_csv):
            w.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            inp_box.addWidget(w)

        self.btn_prep_open_dataset = QPushButton(
            "Open dataset…",
            self,
        )
        self.btn_prep_feature_cfg = QPushButton(
            "Feature config…",
            self,
        )

        inp_btns = QHBoxLayout()
        inp_btns.setContentsMargins(0, 0, 0, 0)
        inp_btns.setSpacing(8)
        inp_btns.addWidget(self.btn_prep_open_dataset)
        inp_btns.addWidget(self.btn_prep_feature_cfg)
        inp_btns.addStretch(1)
        inp_box.addLayout(inp_btns)

        c_lay.addWidget(self.card_inputs, 0)

        # ---- Card: Stage-1 policy
        self.card_policy, opt_box = self._make_card(
            "Stage-1 policy"
        )

        self.chk_prep_clean = QCheckBox(
            "Clean Stage-1 run dir before build",
            self,
        )
        self.chk_prep_auto_reuse = QCheckBox(
            "Auto-reuse compatible Stage-1 run",
            self,
        )
        self.chk_prep_force_rebuild = QCheckBox(
            "Force rebuild if mismatch",
            self,
        )
        self.chk_prep_build_future = QCheckBox(
            "Build future NPZ",
            self,
        )

        opt_box.addWidget(self.chk_prep_clean)
        opt_box.addWidget(self.chk_prep_auto_reuse)

        row_force = QWidget(self.card_policy)
        gl = QGridLayout(row_force)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setHorizontalSpacing(12)
        gl.setVerticalSpacing(6)

        gl.addWidget(self.chk_prep_force_rebuild, 0, 0)
        gl.addWidget(self.chk_prep_build_future, 0, 1)
        gl.setColumnStretch(0, 1)
        gl.setColumnStretch(1, 1)

        opt_box.addWidget(row_force)

        c_lay.addWidget(self.card_policy, 0)

        # ---- Card: Stage-1 status
        self.card_status, stat_box = self._make_card(
            "Stage-1 status"
        )

        self.lbl_prep_stage1_state = QLabel(
            "Stage-1: unknown",
            self,
        )
        self.lbl_prep_stage1_manifest = QLabel(
            "Manifest: -",
            self,
        )

        for w in (
            self.lbl_prep_stage1_state,
            self.lbl_prep_stage1_manifest,
        ):
            w.setTextInteractionFlags(
                Qt.TextSelectableByMouse
            )
            stat_box.addWidget(w)

        btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(8)

        self.btn_prep_open_manifest = QPushButton(
            "Open manifest",
            self,
        )
        self.btn_prep_open_stage1_dir = QPushButton(
            "Open folder",
            self,
        )
        self.btn_prep_use_for_city = QPushButton(
            "Use as default for city",
            self,
        )

        btns.addWidget(self.btn_prep_open_manifest)
        btns.addWidget(self.btn_prep_open_stage1_dir)
        btns.addWidget(self.btn_prep_use_for_city)
        btns.addStretch(1)
        stat_box.addLayout(btns)

        c_lay.addWidget(self.card_status, 0)
        c_lay.addStretch(1)

        # D: run preview (right)
        d_card, d_box = self._make_card("Run preview")
        
        d_box.addWidget(QLabel("Decision:", self))
        
        dec_row = QWidget(self)
        dec_lay = QHBoxLayout(dec_row)
        dec_lay.setContentsMargins(0, 0, 0, 0)
        dec_lay.setSpacing(8)
        
        self.lbl_prep_decision = QLabel("-", self)
        self.lbl_prep_decision.setObjectName("bigValue")
        
        self.lbl_prep_chip = QLabel("", self)
        self.lbl_prep_chip.setObjectName("chip")
        self.lbl_prep_chip.setVisible(False)
        self.lbl_prep_chip.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
        self.lbl_prep_chip.setStyleSheet(
            "padding:2px 8px;"
            "border-radius:10px;"
            "background:rgba(120,120,120,40);"
            "font-size:11px;"
        )
        
        dec_lay.addWidget(self.lbl_prep_decision, 1)
        dec_lay.addWidget(self.lbl_prep_chip, 0)
        
        d_box.addWidget(dec_row)

        self.lbl_prep_reason = QLabel("", self)
        self.lbl_prep_reason.setWordWrap(True)
        d_box.addWidget(self.lbl_prep_reason)

        d_box.addSpacing(8)
        d_box.addWidget(QLabel("Best Stage-1 dir:", self))
        self.lbl_prep_best_dir = QLabel("-", self)
        self.lbl_prep_best_dir.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        d_box.addWidget(self.lbl_prep_best_dir)

        d_box.addWidget(QLabel("Manifest:", self))
        self.lbl_prep_best_mf = QLabel("-", self)
        self.lbl_prep_best_mf.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        d_box.addWidget(self.lbl_prep_best_mf)

        d_btns = QHBoxLayout()
        d_btns.setContentsMargins(0, 0, 0, 0)
        d_btns.setSpacing(8)

        self.btn_prep_preview_open_dir = QPushButton(
            "Open folder",
            self,
        )
        self.btn_prep_preview_open_mf = QPushButton(
            "Open manifest",
            self,
        )
        self.btn_prep_preview_readiness = QPushButton(
            "View readiness",
            self,
        )

        d_btns.addWidget(self.btn_prep_preview_open_dir)
        d_btns.addWidget(self.btn_prep_preview_open_mf)
        d_btns.addWidget(self.btn_prep_preview_readiness)
        d_btns.addStretch(1)
        d_box.addLayout(d_btns)
        d_box.addSpacing(10)
        d_box.addWidget(QLabel("Recap:", self))
        
        self.prep_recap = KeyValuePanel(self, max_rows=14, compact=True)
        d_box.addWidget(self.prep_recap, 1)  # <-- eats the vertical slack

        d_card.setMinimumWidth(320)

        build_split.addWidget(self._build_scroll)
        build_split.addWidget(d_card)
        build_split.setStretchFactor(0, 7)
        build_split.setStretchFactor(1, 3)

        build_lay.addWidget(build_split, 1)

        self._stack.addWidget(build_page)

        # -------------------------
        # Inspect page: workspace
        # -------------------------
        inspect_page = QWidget(self)
        insp_lay = QVBoxLayout(inspect_page)
        insp_lay.setContentsMargins(0, 0, 0, 0)
        insp_lay.setSpacing(0)

        self.stage1_ws = Stage1Workspace(self)
        self.stage1_ws.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        insp_lay.addWidget(self.stage1_ws, 1)

        self._stack.addWidget(inspect_page)

        right_lay.addWidget(self._stack, 1)

        main.addWidget(right)

        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)

        outer.addWidget(main, 1)

        # ==============================================================
        # Bottom run row (kept like Train)
        # ==============================================================
        run_row = QHBoxLayout()
        run_row.setSpacing(10)

        self.btn_run_stage1 = self._make_run_button(
            "Run Stage-1 preprocessing"
        )
        self._run_icon_filled = self.btn_run_stage1.icon()
        self._run_icon_hollow = self._make_local_play_icon(
            diameter=26,
            hollow=True,
        )
        run_row.addStretch(1)
        run_row.addWidget(self.btn_run_stage1)
        outer.addLayout(run_row)

        # default mode
        self._set_view_mode("build")

    def _update_recap(self) -> None:
        st = self._store
        if st is None:
            return
    
        city = str(st.get_value(FieldKey("city"), default="") or "").strip()
        ds = st.get_value(FieldKey("dataset_path"), default=None)
        rr = st.get_value(FieldKey("results_root"), default=None)
    
        rows = [
            ("City", city or "-"),
            ("Dataset", str(ds or "-")),
            ("Results root", str(rr or "-")),
            ("City root", self.city_root_path() or "-"),
            ("Decision", (self.lbl_prep_decision.text() or "-")),
            ("Best Stage-1 dir", (self.best_stage1_dir() or "-")),
            ("Manifest", (self.best_manifest_path() or "-")),
            ("Auto-reuse", "ON" if self._opt_bool("stage1_auto_reuse_if_match", True) else "OFF"),
            ("Force rebuild", "ON" if self._opt_bool("stage1_force_rebuild_if_mismatch", True) else "OFF"),
            ("Clean", "ON" if self._opt_bool("clean_stage1_dir", False) else "OFF"),
            ("Future NPZ", "ON" if self._opt_bool("build_future_npz", False) else "OFF"),
        ]
        self.prep_recap.set_rows(rows)

    def _make_local_play_icon(
        self,
        *,
        diameter: int = 26,
        hollow: bool = False,
    ):
        pix = QPixmap(diameter, diameter)
        pix.fill(Qt.transparent)

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)

        color = self.palette().color(QPalette.Highlight)
        if not color.isValid():
            color = QColor(0, 180, 0)

        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.NoBrush)

        radius = diameter // 2 - 2
        center = QPoint(diameter // 2, diameter // 2)
        painter.drawEllipse(center, radius, radius)

        tri_w = diameter // 3
        tri_h = diameter // 2
        x0 = diameter // 2 - tri_w // 3
        y0 = diameter // 2 - tri_h // 2

        p1 = QPoint(x0, y0)
        p2 = QPoint(x0, y0 + tri_h)
        p3 = QPoint(x0 + tri_w, diameter // 2)

        if hollow:
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.NoBrush)
        else:
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)

        painter.drawPolygon(p1, p2, p3)
        painter.end()

        return QIcon(pix)
    
    def _set_reco_chip(self, decision: str) -> None:
        d = (decision or "").strip()

        txt = ""
        tip = ""

        if d == "REUSE":
            txt = "Ready"
            tip = "Stage-1 is compatible; reuse it."
        elif d.startswith("WAITING"):
            txt = "Select inputs"
            tip = "Choose city + results root first."
        elif d == "BUILD":
            txt = "Run Stage-1"
            tip = "Build Stage-1 outputs."
        elif d == "REBUILD":
            txt = "Run rebuild"
            tip = "Rebuild Stage-1 for current config."
        elif "FOUND (MATCH)" in d:
            txt = "Use default"
            tip = "Click 'Use as default for city' to reuse."
        elif "FOUND (MISMATCH)" in d:
            txt = "Rebuild advised"
            tip = "Mismatch found; rebuilding is recommended."
        else:
            txt = "Review"
            tip = "Check readiness details."

        self.lbl_prep_chip.setText(txt)
        self.lbl_prep_chip.setToolTip(tip)
        self.lbl_prep_chip.setVisible(bool(txt))

    def _apply_run_button_state(self, decision: str) -> None:
        d = (decision or "").strip()

        disable = (d == "REUSE") or d.startswith("WAITING")
        if disable:
            self.btn_run_stage1.setEnabled(False)
            self.btn_run_stage1.setIcon(self._run_icon_hollow)
            if d == "REUSE":
                self.btn_run_stage1.setToolTip(
                    "Stage-1 already compatible (reuse)."
                )
            else:
                self.btn_run_stage1.setToolTip(
                    "Select inputs to enable Stage-1 run."
                )
        else:
            self.btn_run_stage1.setEnabled(True)
            self.btn_run_stage1.setIcon(self._run_icon_filled)
            self.btn_run_stage1.setToolTip(
                "Run Stage-1 preprocessing"
            )

    def _wire(self) -> None:
        # -------------------------------------------------
        # Head actions
        # -------------------------------------------------
        self.btn_prep_refresh.clicked.connect(
            self.request_refresh.emit
        )
        self.btn_prep_browse_root.clicked.connect(
            self.request_browse_results_root.emit
        )
        self.btn_prep_open_city_root.clicked.connect(
            self.request_open_city_root.emit
        )

        # -------------------------------------------------
        # Mode switch
        # -------------------------------------------------
        self.btn_mode_build.clicked.connect(
            lambda: self._set_view_mode("build")
        )
        self.btn_mode_inspect.clicked.connect(
            lambda: self._set_view_mode("inspect")
        )

        # -------------------------------------------------
        # Navigator (A)
        # -------------------------------------------------
        self.nav_inputs.clicked.connect(
            lambda: self._goto_build_card(self.card_inputs)
        )
        self.nav_policy.clicked.connect(
            lambda: self._goto_build_card(self.card_policy)
        )
        self.nav_status.clicked.connect(
            lambda: self._goto_build_card(self.card_status)
        )

        self.nav_readiness.clicked.connect(
            lambda: self._goto_inspect_tab("Readiness")
        )
        self.nav_inspect.clicked.connect(
            lambda: self._goto_inspect_tab("Quicklook")
        )
        self.nav_history.clicked.connect(
            lambda: self._goto_inspect_tab("Run history")
        )
        self.nav_artifacts.clicked.connect(
            lambda: self._goto_inspect_tab("Artifacts")
        )

        # -------------------------------------------------
        # Inputs card
        # -------------------------------------------------
        self.btn_prep_open_dataset.clicked.connect(
            self.request_open_dataset.emit
        )
        self.btn_prep_feature_cfg.clicked.connect(
            self.request_feature_cfg.emit
        )

        # -------------------------------------------------
        # Run
        # -------------------------------------------------
        self.btn_run_stage1.clicked.connect(
            self.request_run_stage1.emit
        )

        # -------------------------------------------------
        # Status card actions
        # -------------------------------------------------
        self.btn_prep_open_manifest.clicked.connect(
            self.request_open_manifest.emit
        )
        self.btn_prep_open_stage1_dir.clicked.connect(
            self.request_open_stage1_dir.emit
        )
        self.btn_prep_use_for_city.clicked.connect(
            self.request_use_for_city.emit
        )

        # Preview card shortcuts (D)
        self.btn_prep_preview_open_dir.clicked.connect(
            self.request_open_stage1_dir.emit
        )
        self.btn_prep_preview_open_mf.clicked.connect(
            self.request_open_manifest.emit
        )
        self.btn_prep_preview_readiness.clicked.connect(
            lambda: self._goto_inspect_tab("Readiness")
        )

        # -------------------------------------------------
        # Options -> store
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Workspace signals -> controller
        # -------------------------------------------------
        self.stage1_ws.request_open_path.connect(
            self.request_open_path.emit
        )
        self.stage1_ws.request_set_active_stage1.connect(
            self.request_set_active_stage1.emit
        )
        self.stage1_ws.request_refresh_history.connect(
            self.request_refresh_history.emit
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

                self._refresh_preview_only()

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
            self._refresh_preview_only()
            return

        if rr_path is None:
            self.set_stage1_status(
                state_text="Stage-1: (no results root selected)",
                manifest_text="Manifest: -",
            )
            self._set_stage1_actions(stage1_dir=None, manifest_path=None)
            self._refresh_preview_only()

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
            self._prep_cache_best = None
            self._prep_cache_manifest = None
            self._prep_cache_audit = None
            self._refresh_preview_only()
  
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

        reason = (
            "Config matches current Stage-1."
            if best.config_match
            else "Config mismatch with current Stage-1."
        )
        
        dec, rsn = self._compute_preview_decision(
            has_best=True,
            is_complete=bool(best.is_complete),
            config_match=bool(best.config_match),
        )

        self._update_run_preview(
            decision=dec,
            reason=rsn,
            stage1_dir=stage1_dir,
            manifest_path=mf_path,
        )

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
            is_complete=bool(best.is_complete),
            config_match=bool(best.config_match),
            decision=dec,
            reason=rsn,
        )
        self._prep_cache_manifest = manifest
        self._prep_cache_audit = audit
        
        self._update_recap()

    def _refresh_runtime_snapshot(
        self,
        *,
        keys: Optional[set[str]],
    ) -> None:
        # Avoid heavy GPU detection on every tiny change.
        if keys is not None and not (keys & self._RUNTIME_KEYS):
            return
    
        txt = runtime_summary_text(self._store)
        self.set_runtime_snapshot(txt)
    
    
    def _runtime_rows_from_text(
        self,
        text: str,
    ) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        cur_k: Optional[str] = None
        cur_v: List[str] = []
    
        def flush() -> None:
            nonlocal cur_k, cur_v
            if cur_k is None:
                return
            v = "\n".join(cur_v).strip() or "-"
            rows.append((cur_k, v))
            cur_k = None
            cur_v = []
    
        for raw in (text or "").splitlines():
            line = raw.rstrip()
            if not line:
                continue
    
            if line.startswith(" - "):
                # continuation for last key
                if cur_k is None:
                    cur_k = "Details"
                cur_v.append(line[3:])
                continue
    
            if ":" in line:
                flush()
                k, v = line.split(":", 1)
                cur_k = k.strip()
                cur_v = [v.strip()]
                continue
    
            # fallback: treat as continuation
            if cur_k is None:
                cur_k = "Details"
            cur_v.append(line)
    
        flush()
        return rows
    
    
    def set_runtime_snapshot(self, text: str) -> None:
        rows = self._runtime_rows_from_text(text)
        try:
            self.comp_panel.set_rows(rows)
        except Exception:
            # last-resort: show raw text if panel API differs
            try:
                self.comp_panel.setText(text)
            except Exception:
                pass
