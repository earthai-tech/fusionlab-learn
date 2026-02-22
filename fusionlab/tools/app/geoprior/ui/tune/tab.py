# geoprior/ui/tune/tab.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, Optional, Tuple
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QDialog, 
    QMessageBox
)

from ...config.store import GeoConfigStore
from ...utils.components import RangeListEditor

from .center_panel import TuneCenterPanel
from .details import TuneDetailsCard
from .head import TuneHeadBar
from .navigator import TuneNavigator
from .plan import _get_fk, build_plan_text
from .preview import TunePreviewPanel
from .status import compute_tune_nav

# -------------------------
# Dialogs used by Advanced
# -------------------------
from ...dialogs.hp_arch_dialog import ArchHPDialog
from ...dialogs.hp_phys_dialog import (
    PhysHPDialog,
    PhysSwitchesDetailsDialog,
)
from ...dialogs.hp_export_dialog import ExportDialog
from ...dialogs.export_actions import export_with_saved_prefs
from ...dialogs.scalars_loss_dialog import ScalarsLossDialog
from ...dialogs.model_params_dialog import ModelParamsDialog
from ...dialogs.tune_options import TuneOptionsDialog
from ...dialogs.feature_dialog import FeatureConfigDialog
from ...dialogs.architecture_dialog import ArchitectureConfigDialog
from ...dialogs.prob_dialog import ProbConfigDialog
from ...dialogs.phys_dialogs import PhysicsConfigDialog

MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]
MakeRunBtnFn = Callable[[str], QWidget]

__all__ = ["TuneTab"]


def _ints_to_csv(vals) -> str:
    try:
        return ", ".join(str(int(v)) for v in vals)
    except Exception:
        return ""


def _strs_to_csv(vals) -> str:
    try:
        return ", ".join(str(v) for v in vals)
    except Exception:
        return ""


class TuneTab(QWidget):
    """
    Tune tab v3.x layout (Train-like).

    [A] left: navigator + details
    [B] head: pinned over workspace
    [C] center: cards (inline expand)
    [D] right: preview
    """

    features_clicked = pyqtSignal()
    arch_clicked = pyqtSignal()
    prob_clicked = pyqtSignal()
    physics_clicked = pyqtSignal()

    run_clicked = pyqtSignal()
    reset_requested = pyqtSignal()

    # legacy hook (app.py may connect it)
    advanced_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: MakeCardFn,
        make_run_button: MakeRunBtnFn,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._make_card = make_card
        self._make_run_button = make_run_button

        self._build_ui()
        self._build_compat_widgets()
        self._wire()
        self.refresh_from_store()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        main = QSplitter(Qt.Horizontal, self)
        main.setHandleWidth(6)
        main.setChildrenCollapsible(False)
        outer.addWidget(main, 1)

        # -------------------------
        # Left [A]
        # -------------------------
        left = QWidget(self)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        self.nav = TuneNavigator(parent=left)
        ll.addWidget(self.nav, 0)

        self.details = TuneDetailsCard(
            store=self._store,
            parent=left,
        )
        ll.addWidget(self.details, 1)

        main.addWidget(left)

        # -------------------------
        # Right: [B] + workspace
        # -------------------------
        right = QWidget(self)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        self.head = TuneHeadBar(store=self._store)
        rl.addWidget(self.head, 0)

        work = QSplitter(Qt.Horizontal, right)
        work.setHandleWidth(6)
        work.setChildrenCollapsible(False)
        rl.addWidget(work, 1)

        # -------------------------
        # Center [C]
        # -------------------------
        self._c_scroll = QScrollArea(right)
        self._c_scroll.setWidgetResizable(True)
        self._c_scroll.setFrameShape(QFrame.NoFrame)

        c_page = QWidget(self._c_scroll)
        self._c_scroll.setWidget(c_page)

        c_l = QVBoxLayout(c_page)
        c_l.setContentsMargins(0, 0, 0, 0)
        c_l.setSpacing(10)

        self.center = TuneCenterPanel(
            store=self._store,
            make_card=self._make_card,
            parent=c_page,
        )
        c_l.addWidget(self.center, 1)

        work.addWidget(self._c_scroll)

        # -------------------------
        # Preview [D]
        # -------------------------
        self._p_scroll = QScrollArea(right)
        self._p_scroll.setWidgetResizable(True)
        self._p_scroll.setFrameShape(QFrame.NoFrame)

        p_page = QWidget(self._p_scroll)
        self._p_scroll.setWidget(p_page)

        p_l = QVBoxLayout(p_page)
        p_l.setContentsMargins(0, 0, 0, 0)
        p_l.setSpacing(10)

        card, box = self._make_card("Run preview")
        self.preview = TunePreviewPanel(store=self._store)
        box.addWidget(self.preview, 1)

        p_l.addWidget(card)
        p_l.addStretch(1)

        work.addWidget(self._p_scroll)

        main.addWidget(right)

        # -------------------------
        # Split sizes
        # -------------------------
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setSizes([260, 1200])

        work.setStretchFactor(0, 1)
        work.setStretchFactor(1, 0)
        work.setSizes([860, 360])

        # -------------------------
        # Bottom bar
        # -------------------------
        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(False)
        self.lbl_status.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        self.lbl_status.setMinimumHeight(18)
        self.lbl_status.setMaximumHeight(18)

        self.btn_run = self._make_run_button("Run")
        self.lbl_run = QLabel("Run:")
        self.lbl_run.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )

        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(8)
        run_row.addStretch(1)
        run_row.addWidget(self.lbl_run)
        run_row.addWidget(self.btn_run)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(8)
        bottom.addWidget(self.lbl_status, 1)
        bottom.addLayout(run_row)

        outer.addLayout(bottom)

    # -----------------------------------------------------------------
    # Backward-compat attributes (app.py aliases)
    # -----------------------------------------------------------------
    def _build_compat_widgets(self) -> None:
        self._compat = QWidget(self)
        self._compat.setVisible(False)

        self.hp_embed_dim = QLineEdit(self._compat)
        self.hp_hidden_units = QLineEdit(self._compat)
        self.hp_lstm_units = QLineEdit(self._compat)
        self.hp_attention_units = QLineEdit(self._compat)
        self.hp_num_heads = QLineEdit(self._compat)
        self.hp_vsn_units = QLineEdit(self._compat)

        self.hp_dropout = RangeListEditor(
            self._compat,
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=3,
            show_sampling=False,
        )
        self.hp_hd = RangeListEditor(
            self._compat,
            min_allowed=0.0,
            max_allowed=2.0,
            decimals=3,
            show_sampling=False,
        )

        self.hp_pde_mode = QLineEdit(self._compat)
        self.hp_kappa_mode = QLineEdit(self._compat)
        self.hp_scale_pde_bool = QCheckBox(self._compat)

        self.chk_eval_tuned = QCheckBox(self._compat)

        self.spin_max_trials = QSpinBox(self._compat)
        self.spin_max_trials.setRange(1, 10_000)

        self.btn_model_params = QPushButton(self._compat)
        self.btn_scalars = QPushButton(self._compat)
        self.btn_tune_options = QPushButton(self._compat)

        self.btn_run_tune = self.btn_run

        self.btn_model_params.clicked.connect(
            self.advanced_clicked.emit
        )
        self.btn_scalars.clicked.connect(
            self.advanced_clicked.emit
        )
        self.btn_tune_options.clicked.connect(
            self.advanced_clicked.emit
        )

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire(self) -> None:
        self.nav.section_changed.connect(self._scroll_to)

        # run
        clicked = getattr(self.btn_run, "clicked", None)
        if clicked is not None:
            clicked.connect(self.run_clicked.emit)

        # head reset
        self.head.reset_requested.connect(
            self.reset_requested.emit
        )

        # head filter
        self.head.search_changed.connect(self._apply_filter)
        if hasattr(self.head, "filter_toggled"):
            self.head.filter_toggled.connect(
                lambda _=False: self._apply_filter(
                    self.head.ed_search.text()
                )
            )
            
        self.nav.features_clicked.connect(
            self._on_feature_cfg
        )
        self.nav.arch_clicked.connect(
            self._on_arch_cfg
        )
        self.nav.prob_clicked.connect(
            self._on_prob_cfg
        )
        self.nav.physics_clicked.connect(
            self._on_physics_cfg
        )

        # center edit -> scroll
        self.center.edit_requested.connect(self._scroll_to)

        # search space quick HP buttons
        self.center.arch_hp_clicked.connect(
            self._on_more_arch_hp
        )
        self.center.phys_hp_clicked.connect(
            self._on_more_phys_hp
        )

        # reset space (head + center advanced button)
        self.center.reset_space_requested.connect(
            self.reset_requested.emit
        )

        # advanced hub click (legacy)
        self.center.advanced_clicked.connect(
            self.advanced_clicked.emit
        )
        self.advanced_clicked.connect(self._on_tune_options)

        # advanced hub buttons -> dialogs
        self._wire_advanced_buttons()

        # refresh on card changes
        self._wire_card_changes()

    def _wire_advanced_buttons(self) -> None:
        """
        Hook Advanced card signals (forwarded by center)
        to actual dialog openers.
        """
        adv = getattr(self.center, "card_adv", None)

        if adv is None:
            return

        sig = getattr(adv, "model_params_clicked", None)
        if sig is not None:
            sig.connect(self._on_model_params)

        sig = getattr(adv, "scalars_losses_clicked", None)
        if sig is not None:
            sig.connect(self._on_scalars_losses)

        sig = getattr(adv, "arch_hp_clicked", None)
        if sig is not None:
            sig.connect(self._on_more_arch_hp)

        sig = getattr(adv, "phys_hp_clicked", None)
        if sig is not None:
            sig.connect(self._on_more_phys_hp)

        sig = getattr(adv, "tune_options_clicked", None)
        if sig is not None:
            sig.connect(self._on_tune_options)

        sig = getattr(adv, "export_clicked", None)
        if sig is not None:
            sig.connect(self._on_export)

        sig = getattr(adv, "reset_space_clicked", None)
        if sig is not None:
            sig.connect(self.reset_requested.emit)

        # optional: physics mini-gear dialog
        phys = getattr(self.center, "card_phys", None)
        sig = getattr(phys, "hd_details_clicked", None)
        if sig is not None:
            sig.connect(self._on_phys_switches_details)

    def _wire_card_changes(self) -> None:
        for nm in (
            "card_space",
            "card_phys",
            "card_algo",
            "card_trial",
            "card_compute",
            "card_adv",
        ):
            c = getattr(self.center, nm, None)
            sig = getattr(c, "changed", None)
            if sig is not None:
                try:
                    sig.connect(self.refresh_from_store)
                except Exception:
                    pass

    def _on_feature_cfg(self) -> None:
        csv = self._store.get("dataset_path", None)
        if not csv:
            QMessageBox.information(
                self,
                "Dataset required",
                "Please open/select a dataset first.",
            )
            return
    
        try:
            csv_path = (
                csv if isinstance(csv, Path) else Path(str(csv))
            ).expanduser()
        except Exception:
            QMessageBox.warning(
                self,
                "Invalid dataset path",
                f"Cannot use dataset_path: {csv!r}",
            )
            return
    
        base_cfg = self._store.get("_base_cfg", {}) or {}
        cur = self._store.get("feature_overrides", {}) or {}
    
        dlg = FeatureConfigDialog(
            csv_path=csv_path,
            base_cfg=base_cfg,
            current_overrides=cur,
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
    
        overs = dlg.get_overrides() or {}
        self._store.patch({"feature_overrides": overs})
        self.refresh_from_store()
    
    
    def _on_arch_cfg(self) -> None:
        base_cfg = self._store.get("_base_cfg", {}) or {}
        cur = self._store.get("arch_overrides", {}) or {}
    
        dlg = ArchitectureConfigDialog(
            base_cfg=base_cfg,
            current_overrides=cur,
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
    
        delta = dlg.get_overrides() or {}
    
        # IMPORTANT: don't wipe existing overrides on "no change"
        if not delta:
            return
    
        merged = dict(cur)
        merged.update(delta)
        self._store.patch({"arch_overrides": merged})
        self.refresh_from_store()
    
    def _on_prob_cfg(self) -> None:
        ok = ProbConfigDialog.edit(
            store=self._store,
            parent=self,
        )
        if ok:
            self.refresh_from_store()
    
    
    def _on_physics_cfg(self) -> None:
        patch = PhysicsConfigDialog.edit(
            parent=self,
            store=self._store,
        )
        if patch is not None:
            self.refresh_from_store()

    # -----------------------------------------------------------------
    # Dialog actions (Advanced)
    # -----------------------------------------------------------------
    def _on_more_phys_hp(self) -> None:
        ok = PhysHPDialog.edit(store=self._store, parent=self)
        if ok:
            self.refresh_from_store()

    def _on_more_arch_hp(self) -> None:
        ok = ArchHPDialog.edit(store=self._store, parent=self)
        if ok:
            self.refresh_from_store()

    def _on_export(self) -> None:
        # You can choose one:
        # - open ExportDialog
        # - or perform export with saved prefs
        try:
            ExportDialog.edit(store=self._store, parent=self)
        except Exception:
            export_with_saved_prefs(self._store, parent=self)
        self.refresh_from_store()

    def _on_scalars_losses(self) -> None:
        ok = ScalarsLossDialog.edit(
            store=self._store,
            range_editor_cls=RangeListEditor,
            parent=self,
        )
        if ok:
            self.refresh_from_store()

    def _on_model_params(self) -> None:
        ok = ModelParamsDialog.edit(store=self._store, parent=self)
        if ok:
            self.refresh_from_store()

    def _on_tune_options(self) -> None:
        ok, job = TuneOptionsDialog.edit(
            store=self._store,
            parent=self,
        )
        if ok:
            self.refresh_from_store()

        if job is not None:
            self.run_clicked.emit()

    def _on_phys_switches_details(self) -> None:
        try:
            ok = PhysSwitchesDetailsDialog.edit(
                store=self._store,
                parent=self,
            )
        except Exception:
            ok = False
        if ok:
            self.refresh_from_store()

    # -----------------------------------------------------------------
    # Filter + scroll
    # -----------------------------------------------------------------
    def _apply_filter(self, text: str) -> None:
        q = str(text or "").strip().lower()
        on = False
        try:
            on = bool(self.head.btn_filter.isChecked())
        except Exception:
            on = False

        try:
            self.nav.apply_filter(q, on)
        except Exception:
            pass

        try:
            self.center.apply_filter(q, on)
        except Exception:
            pass

    def _scroll_to(self, key: str) -> None:
        card = None
        try:
            card = self.center.card_for(key)
        except Exception:
            card = None

        if card is None:
            return

        self._c_scroll.ensureWidgetVisible(card, 0, 24)

    # -----------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------
    def refresh_from_store(self) -> None:
        self.details.refresh_from_store()

        try:
            self.center.refresh_from_store()
        except Exception:
            pass

        txt = build_plan_text(self._store)
        self.preview.refresh_from_store()
        self.head.set_plan_text(txt)

        chips = compute_tune_nav(self._store)
        for k, v in chips.items():
            self.nav.set_chip(k, v["status"], v["text"])

        # legacy sync
        space = _get_fk(self._store, "tuner_search_space", {})
        if isinstance(space, dict):
            self.hp_embed_dim.setText(
                _ints_to_csv(space.get("embed_dim", []))
            )
            self.hp_hidden_units.setText(
                _ints_to_csv(space.get("hidden_units", []))
            )
            self.hp_lstm_units.setText(
                _ints_to_csv(space.get("lstm_units", []))
            )
            self.hp_attention_units.setText(
                _ints_to_csv(space.get("attention_units", []))
            )
            self.hp_num_heads.setText(
                _ints_to_csv(space.get("num_heads", []))
            )
            self.hp_vsn_units.setText(
                _ints_to_csv(space.get("vsn_units", []))
            )

            self.hp_pde_mode.setText(
                _strs_to_csv(space.get("pde_mode", []))
            )
            self.hp_kappa_mode.setText(
                _strs_to_csv(space.get("kappa_mode", []))
            )

            self.hp_dropout.from_search_space_value(
                space.get("dropout_rate"),
                space.get("dropout_rate"),
            )
            self.hp_hd.from_search_space_value(
                space.get("hd_factor"),
                space.get("hd_factor"),
            )

            v = space.get("scale_pde_residuals", None)
            is_bool = (
                isinstance(v, dict)
                and str(v.get("type", "")).lower()
                == "bool"
            )
            self.hp_scale_pde_bool.setChecked(is_bool)

        mt = int(_get_fk(self._store, "tuner_max_trials", 20))
        self.spin_max_trials.setValue(mt)

    def set_run_status(self, text: str) -> None:
        fm = QFontMetrics(self.lbl_status.font())
        el = fm.elidedText(
            str(text),
            Qt.ElideRight,
            self.lbl_status.width(),
        )
        self.lbl_status.setText(el)
