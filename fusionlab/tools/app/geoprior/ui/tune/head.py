# geoprior/ui/tune/head.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Sequence

from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QCheckBox, 
    QSizePolicy
)

from ...config.prior_schema import FieldKey
from ...config.store import GeoConfigStore

from ..icon_utils import try_icon
from ..common.lifecycle_strip import LifecycleStrip


__all__ = ["TuneHeadBar"]


# UI-only keys (store._extra)
_PRESET_KEY = "tune.preset_name"
_SEARCH_KEY = "tune.head.search"
_DIR_KEY = "tune.direction"  # "min" | "max"

# Optional mirror keys (UI-only)
_TUNE_LIFE_KEY = "tune.lifecycle"
_TUNE_BASE_KEY = "tune.base_model_path"

# Shared training keys (current pipeline truth)
_TRAIN_LIFE_KEY = "train.lifecycle"
_TRAIN_BASE_KEY = "train.base_model_path"

# Schema keys (GeoPriorConfig via FieldKey)
_TRIALS_FK = FieldKey("tuner_max_trials")
_OBJ_FK = FieldKey("tuner_objective")
_EVAL_TUNED_FK = FieldKey("evaluate_tuned")


# User label -> stored objective key
_OBJ_ITEMS = [
    ("Validation loss", "val_loss"),
    ("Validation MAE", "val_mae"),
    ("Validation RMSE", "val_rmse"),
    ("Stability score (PSS)", "val_PSS"),
    ("Time-weighted accuracy (TWA)", "val_TWA"),
]


class TuneHeadBar(QFrame):
    """
    Tune head (pinned command bar).

    Row 1
    -----
    - lifecycle strip
    - preset combo
    - reset / copy plan / config icons

    Row 2
    -----
    - objective + direction
    - trials
    - search box + filter icon
    """

    toast = pyqtSignal(str)

    reset_requested = pyqtSignal()
    copy_plan_requested = pyqtSignal()
    config_clicked = pyqtSignal()
    filter_clicked = pyqtSignal()

    search_changed = pyqtSignal(str)
    preset_changed = pyqtSignal(str)

    objective_changed = pyqtSignal(str)
    direction_changed = pyqtSignal(str)
    trials_changed = pyqtSignal(int)
    
    filter_toggled = pyqtSignal(bool)

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._plan_text = ""

        self.setObjectName("tuneHead")
        self.setFrameShape(QFrame.NoFrame)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._build_ui()
        self._wire()
        self.refresh_from_store()

    # -------------------------
    # Icons
    # -------------------------
    def _std_icon(self, sp: QStyle.StandardPixmap):
        return self.style().standardIcon(sp)

    def _set_icon(
        self,
        btn: QToolButton,
        name: str,
        fallback: QStyle.StandardPixmap,
    ) -> None:
        ic = try_icon(name)
        if ic is None:
            ic = self._std_icon(fallback)
        btn.setIcon(ic)

    def _mk_icon_btn(
        self,
        tip: str,
        icon_name: str,
        fallback: QStyle.StandardPixmap,
    ) -> QToolButton:
        b = QToolButton(self)
        b.setAutoRaise(True)
        b.setToolButtonStyle(Qt.ToolButtonIconOnly)
        b.setToolTip(tip)
        self._set_icon(b, icon_name, fallback)
        return b

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)
    
        # =================================================
        # Row 1: Lifecycle + preset + actions (right)
        # =================================================
        r1w = QWidget(self)
        r1 = QHBoxLayout(r1w)
        r1.setContentsMargins(0, 0, 0, 0)
        r1.setSpacing(10)
    
        self.lifecycle = LifecycleStrip(
            store=self._store,
            life_key=_TRAIN_LIFE_KEY,
            base_key=_TRAIN_BASE_KEY,
        )
        self._patch_lifecycle_labels()
        self.lifecycle.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        r1.addWidget(self.lifecycle, 1)
    
        r1.addWidget(QLabel("Preset:"), 0)
    
        self.cmb_preset = QComboBox(self)
        self.cmb_preset.addItem("Custom")
        self.cmb_preset.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
        self.cmb_preset.setMinimumContentsLength(10)
        self.cmb_preset.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        r1.addWidget(self.cmb_preset, 0)
    
        # push actions to the far right
        r1.addStretch(1)
    
        self.btn_reset = self._mk_icon_btn(
            "Reset to defaults",
            "reset.svg",
            QStyle.SP_BrowserReload,
        )
        self.btn_cfg = self._mk_icon_btn(
            "Open config",
            "settings.svg",
            QStyle.SP_FileDialogDetailedView,
        )
        for b in (self.btn_reset, self.btn_cfg):
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    
        r1.addWidget(self.btn_reset, 0)
        r1.addWidget(self.btn_cfg, 0)
    
        root.addWidget(r1w)
    
        # =================================================
        # Row 2: Objective / trials / eval + search capsule
        # =================================================
        r2w = QWidget(self)
        r2 = QHBoxLayout(r2w)
        r2.setContentsMargins(0, 0, 0, 0)
        r2.setSpacing(10)
    
        r2.addWidget(QLabel("Objective:"), 0)
    
        self.cmb_obj = QComboBox(self)
        self.cmb_obj.addItem("Auto", "")
        for lab, key in _OBJ_ITEMS:
            self.cmb_obj.addItem(lab, key)
        self.cmb_obj.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
        self.cmb_obj.setMinimumContentsLength(14)
        self.cmb_obj.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        r2.addWidget(self.cmb_obj, 0)
    
        r2.addWidget(QLabel("Direction:"), 0)
    
        self.cmb_dir = QComboBox(self)
        self.cmb_dir.addItem("Minimize", "min")
        self.cmb_dir.addItem("Maximize", "max")
        self.cmb_dir.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
        self.cmb_dir.setMinimumContentsLength(10)
        self.cmb_dir.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        r2.addWidget(self.cmb_dir, 0)
    
        r2.addWidget(QLabel("Trials:"), 0)
    
        self.sp_trials = QSpinBox(self)
        self.sp_trials.setRange(1, 5000)
        self.sp_trials.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
        self.sp_trials.setFixedWidth(74)
        r2.addWidget(self.sp_trials, 0)
    
        self.chk_eval_tuned = QCheckBox("Eval tuned model", self)
        self.chk_eval_tuned.setChecked(False)
        self.chk_eval_tuned.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )
        r2.addWidget(self.chk_eval_tuned, 0)
    
        # push search to the far right
        r2.addStretch(1)
    
        # --- Search capsule (DO NOT FIX WIDTH) ---
        self.search_wrap = QFrame(self)
        self.search_wrap.setObjectName("searchWrap")
        self.search_wrap.setFrameShape(QFrame.NoFrame)
        self.search_wrap.setAttribute(Qt.WA_StyledBackground, True)
    
        # Key: don't push main window -> no fixed/min width, only a max
        self.search_wrap.setSizePolicy(
            QSizePolicy.Maximum,
            QSizePolicy.Fixed,
        )
        self.search_wrap.setMinimumWidth(0)
        self.search_wrap.setMaximumWidth(280)
    
        sw = QHBoxLayout(self.search_wrap)
        sw.setContentsMargins(8, 2, 8, 2)
        sw.setSpacing(6)
    
        self.btn_filter = self._mk_icon_btn(
            "Highlight matches",
            "filter2.svg",
            QStyle.SP_FileDialogContentsView,
        )
        self.btn_filter.setCheckable(True)
        self.btn_filter.setObjectName("filterToggle")
        self.btn_filter.setFixedSize(24, 24)
    
        self.ed_search = QLineEdit(self)
        self.ed_search.setObjectName("searchEdit")
        self.ed_search.setPlaceholderText("Search settings…")
        self.ed_search.setFrame(False)
    
        # Allow shrinking inside capsule
        self.ed_search.setMinimumWidth(0)
        self.ed_search.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
    
        sw.addWidget(self.btn_filter, 0)
        sw.addWidget(self.ed_search, 1)
    
        r2.addWidget(self.search_wrap, 0)
    
        root.addWidget(r2w)


    def _patch_lifecycle_labels(self) -> None:
        """
        Make the Tune tab say "New tuning" (UI only).

        We do this safely (no dependency on internals).
        """
        cmb = getattr(self.lifecycle, "cmb_mode", None)
        if cmb is None:
            return

        if cmb.count() <= 0:
            return

        txt0 = str(cmb.itemText(0) or "")
        if "New" in txt0 and "training" in txt0.lower():
            cmb.setItemText(0, "New tuning")

    # -------------------------
    # Wiring
    # -------------------------
    def _wire(self) -> None:
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        self.btn_cfg.clicked.connect(self.config_clicked.emit)
    
        self.ed_search.textChanged.connect(self._on_search)
    
        self.cmb_preset.currentIndexChanged.connect(
            lambda _=0: self._on_preset()
        )
        self.cmb_obj.currentIndexChanged.connect(
            lambda _=0: self._on_obj()
        )
        self.cmb_dir.currentIndexChanged.connect(
            lambda _=0: self._on_dir()
        )
        self.sp_trials.valueChanged.connect(self._on_trials)
    
        sig = getattr(self.lifecycle, "changed", None)
        if sig is not None:
            sig.connect(self._mirror_lifecycle_to_tune)
    
        self.btn_filter.toggled.connect(self._on_filter)
        self.btn_filter.clicked.connect(
            lambda: self.ed_search.setFocus()
        )
    
        self.chk_eval_tuned.toggled.connect(self._on_eval_tuned)

    def _on_eval_tuned(self, on: bool) -> None:
        try:
            self._store.set_value_by_key(_EVAL_TUNED_FK, bool(on))
        except Exception:
            # best-effort fallback (in case store API shifts)
            try:
                self._store.patch_fields(
                    {"evaluate_tuned": bool(on)}
                )
            except Exception:
                pass

    def _on_filter(self, on: bool) -> None:
        if on:
            self.ed_search.setFocus()
        self.filter_clicked.emit()
        self.filter_toggled.emit(bool(on))

    # -------------------------
    # Public API
    # -------------------------
    def set_presets(self, names: Sequence[str]) -> None:
        cur = str(self._store.get(_PRESET_KEY, "Custom") or "")
        cur = cur.strip() or "Custom"

        with QSignalBlocker(self.cmb_preset):
            self.cmb_preset.clear()
            if not names:
                self.cmb_preset.addItem("Custom")
            else:
                for n in names:
                    self.cmb_preset.addItem(str(n))
            j = self.cmb_preset.findText(cur)
            self.cmb_preset.setCurrentIndex(max(j, 0))

    def set_plan_text(self, text: str) -> None:
        self._plan_text = str(text or "")

    def refresh_from_store(self) -> None:
        # preset
        nm = str(self._store.get(_PRESET_KEY, "Custom") or "")
        nm = nm.strip() or "Custom"
        with QSignalBlocker(self.cmb_preset):
            j = self.cmb_preset.findText(nm)
            self.cmb_preset.setCurrentIndex(max(j, 0))

        # search
        s = str(self._store.get(_SEARCH_KEY, "") or "")
        if self.ed_search.text() != s:
            with QSignalBlocker(self.ed_search):
                self.ed_search.setText(s)

        # direction
        d = str(self._store.get(_DIR_KEY, "min") or "").strip()
        d = d if d in {"min", "max"} else "min"
        with QSignalBlocker(self.cmb_dir):
            j = self.cmb_dir.findData(d)
            self.cmb_dir.setCurrentIndex(max(j, 0))

        # objective (FieldKey)
        obj = ""
        try:
            obj = str(self._store.get_value(_OBJ_FK, default=""))
        except Exception:
            obj = ""
        obj = str(obj or "").strip()

        with QSignalBlocker(self.cmb_obj):
            j = self.cmb_obj.findData(obj)
            if j < 0:
                j = 0
            self.cmb_obj.setCurrentIndex(j)

        # trials (FieldKey)
        mt = 20
        try:
            mt = int(self._store.get_value(_TRIALS_FK, default=20))
        except Exception:
            mt = 20
        with QSignalBlocker(self.sp_trials):
            self.sp_trials.setValue(int(mt))

        # eval tuned (FieldKey)
        ev = False
        try:
            ev = bool(
                self._store.get_value(
                    _EVAL_TUNED_FK,
                    default=False,
                )
            )
        except Exception:
            ev = False

        with QSignalBlocker(self.chk_eval_tuned):
            self.chk_eval_tuned.setChecked(bool(ev))


        # If tune.* lifecycle exists, sync to train.*
        self._maybe_sync_tune_life_to_train()

        self.lifecycle.refresh_from_store()
        self._patch_lifecycle_labels()

    # -------------------------
    # Events -> store
    # -------------------------
    def _on_preset(self) -> None:
        nm = str(self.cmb_preset.currentText() or "")
        nm = nm.strip() or "Custom"
        self._store.set(_PRESET_KEY, nm)
        self.preset_changed.emit(nm)

    def _on_obj(self) -> None:
        key = str(self.cmb_obj.currentData() or "").strip()
        try:
            self._store.set_value_by_key(_OBJ_FK, key)
        except Exception:
            pass
        self.objective_changed.emit(key)

    def _on_dir(self) -> None:
        v = str(self.cmb_dir.currentData() or "min")
        v = v if v in {"min", "max"} else "min"
        self._store.set(_DIR_KEY, v)
        self.direction_changed.emit(v)

    def _on_trials(self, v: int) -> None:
        try:
            self._store.set_value_by_key(_TRIALS_FK, int(v))
        except Exception:
            pass
        self.trials_changed.emit(int(v))

    def _on_search(self, text: str) -> None:
        self._store.set(_SEARCH_KEY, str(text or ""))
        self.search_changed.emit(str(text or ""))

    def _on_copy(self) -> None:
        txt = (self._plan_text or "").strip()
        if not txt:
            self.copy_plan_requested.emit()
            return
        QApplication.clipboard().setText(txt)
        self.toast.emit("Tune plan copied.")

    # -------------------------
    # Lifecycle mirroring
    # -------------------------
    def _mirror_lifecycle_to_tune(self) -> None:
        """
        Mirror current shared train.* values into tune.*.

        This keeps future decoupling easy and avoids confusion
        when other tune modules read tune.lifecycle later.
        """
        life = str(self._store.get(_TRAIN_LIFE_KEY, "") or "")
        base = str(self._store.get(_TRAIN_BASE_KEY, "") or "")
        self._store.set(_TUNE_LIFE_KEY, life)
        self._store.set(_TUNE_BASE_KEY, base)

    def _maybe_sync_tune_life_to_train(self) -> None:
        """
        If tune.* was loaded (e.g. from a preset later),
        sync it into train.* so the shared LifecycleStrip
        reflects it.
        """
        tl = str(self._store.get(_TUNE_LIFE_KEY, "") or "").strip()
        tb = str(self._store.get(_TUNE_BASE_KEY, "") or "").strip()

        if tl:
            self._store.set(_TRAIN_LIFE_KEY, tl)
        if tb:
            self._store.set(_TRAIN_BASE_KEY, tb)
