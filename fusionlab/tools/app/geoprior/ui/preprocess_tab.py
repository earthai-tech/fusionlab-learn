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
)

from ..config.prior_schema import FieldKey
from ..config.store import GeoConfigStore
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

        self.stage1_ws.set_context(
            city=str(city or ""),
            csv_path=ds,
            runs_root=rr,
            stage1_dir=self._ws_stage1_dir,
            model=self._ws_model,
        )

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
        p_layout = QVBoxLayout(self)
        p_layout.setContentsMargins(6, 6, 6, 6)
        p_layout.setSpacing(8)

        # -------------------------------------------------
        # Row 1: Results root (full width)
        # -------------------------------------------------
        root_row = QHBoxLayout()
        root_row.setSpacing(10)

        lbl_root = QLabel("Results root:")
        self.ed_prep_root = QLineEdit()
        self.ed_prep_root.setReadOnly(True)

        self.btn_prep_browse_root = QPushButton("Browse…")
        self.btn_prep_refresh = QPushButton("Refresh")

        root_row.addWidget(lbl_root)
        root_row.addWidget(self.ed_prep_root, 1)
        root_row.addWidget(self.btn_prep_browse_root)
        root_row.addWidget(self.btn_prep_refresh)
        p_layout.addLayout(root_row)

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
        opt_card, opt_box = self._make_card(
            "Stage-1 options"
        )

        self.chk_prep_clean = QCheckBox(
            "Clean Stage-1 run dir before build"
        )
        self.chk_prep_auto_reuse = QCheckBox(
            "Auto-reuse compatible Stage-1 run"
        )
        self.chk_prep_force_rebuild = QCheckBox(
            "Force rebuild if mismatch"
        )
        self.chk_prep_build_future = QCheckBox(
            "Build future NPZ"
        )

        for cb in (
            self.chk_prep_clean,
            self.chk_prep_auto_reuse,
            self.chk_prep_force_rebuild,
            self.chk_prep_build_future,
        ):
            opt_box.addWidget(cb)

        opt_box.addStretch(1)
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

        self.stage1_ws = Stage1Workspace()
        ws_box.addWidget(self.stage1_ws, 1)
        p_layout.addWidget(ws_card, 1)

        # -------------------------------------------------
        # Bottom: Run (right)
        # -------------------------------------------------
        run_row = QHBoxLayout()
        run_row.setSpacing(10)

        self.btn_run_stage1 = self._make_run_button(
            "Run Stage-1 preprocessing"
        )

        run_row.addStretch(1)
        run_row.addWidget(self.btn_run_stage1)
        p_layout.addLayout(run_row)

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
