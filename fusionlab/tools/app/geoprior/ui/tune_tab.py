# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

r"""
tune_tab

GeoPrior GUI: Tune tab (hyperparameter search).

- UI isolated from app.py (like TrainTab).
- GeoConfigStore is the single source of truth for
  `tuner_search_space` and `tuner_max_trials`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

from PyQt5.QtCore import QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config.prior_schema import FieldKey
from ..config.store import GeoConfigStore
from ..config.geoprior_config import default_tuner_search_space
from ..dialogs.hp_arch_dialog import ArchHPDialog
from ..dialogs.hp_phys_dialog import PhysHPDialog
from ..dialogs.hp_search_dialog import SearchAlgoDialog
from ..dialogs.hp_export_dialog import ExportDialog
from ..dialogs.export_actions import export_with_saved_prefs
from ..dialogs.scalars_loss_dialog import ScalarsLossDialog
from ..dialogs.model_params_dialog import ModelParamsDialog

MakeCardFn = Callable[[str], tuple[QWidget, QVBoxLayout]]
MakeRunBtnFn = Callable[[str], QPushButton]

def _csv_to_ints(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",")]
    out: list[int] = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out


def _csv_to_strs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


def _ints_to_csv(vals: Iterable[int]) -> str:
    return ", ".join(str(v) for v in vals)


def _strs_to_csv(vals: Iterable[str]) -> str:
    return ", ".join(str(v) for v in vals)


class _Skip:
    pass


class _Remove:
    pass


_SKIP = _Skip()
_REMOVE = _Remove()


class TuneTab(QWidget):
    """Store-driven Tune tab (v3.2)."""

    run_clicked = pyqtSignal()
    advanced_clicked = pyqtSignal()
    # model_params_clicked = pyqtSignal()
    # scalars_clicked = pyqtSignal()
    # search_algo_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: MakeCardFn,
        make_run_button: MakeRunBtnFn,
        range_editor_cls: Optional[type] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._make_card = make_card
        self._make_run_button = make_run_button
        self._range_editor_cls = range_editor_cls

        self._writing = False

        self._build_ui()
        self._wire_ui()
        self._wire_store()
        self.refresh_from_store()

    # -----------------------------------------------------------------
    # Public helpers (used by app.py)
    # -----------------------------------------------------------------
    def refresh_from_store(self) -> None:
        """Reload UI from GeoConfigStore."""
        self._writing = True
        try:
            space = self._get_space()

            with QSignalBlocker(self.hp_embed_dim):
                self.hp_embed_dim.setText(
                    _ints_to_csv(space.get("embed_dim", []))
                )
            with QSignalBlocker(self.hp_hidden_units):
                self.hp_hidden_units.setText(
                    _ints_to_csv(space.get("hidden_units", []))
                )
            with QSignalBlocker(self.hp_lstm_units):
                self.hp_lstm_units.setText(
                    _ints_to_csv(space.get("lstm_units", []))
                )
            with QSignalBlocker(self.hp_attention_units):
                self.hp_attention_units.setText(
                    _ints_to_csv(space.get("attention_units", []))
                )
            with QSignalBlocker(self.hp_num_heads):
                self.hp_num_heads.setText(
                    _ints_to_csv(space.get("num_heads", []))
                )
            with QSignalBlocker(self.hp_vsn_units):
                self.hp_vsn_units.setText(
                    _ints_to_csv(space.get("vsn_units", []))
                )

            with QSignalBlocker(self.hp_pde_mode):
                self.hp_pde_mode.setText(
                    _strs_to_csv(space.get("pde_mode", []))
                )
            with QSignalBlocker(self.hp_kappa_mode):
                self.hp_kappa_mode.setText(
                    _strs_to_csv(space.get("kappa_mode", []))
                )

            with QSignalBlocker(self.hp_scale_pde_bool):
                v = space.get("scale_pde_residuals", None)
                self.hp_scale_pde_bool.setChecked(
                    isinstance(v, dict) and str(v.get("type","")).lower() == "bool"
                )
                # self.hp_scale_pde_bool.setChecked(
                #     bool(space.get("tune_scale_pde_bool", False))
                # )

            self._apply_range_defaults(
                editor=self.hp_dropout,
                spec=space.get("dropout_rate"),
                dmin=0.05,
                dmax=0.20,
            )
            self._apply_range_defaults(
                editor=self.hp_hd,
                spec=space.get("hd_factor"),
                dmin=0.50,
                dmax=0.70,
            )

            max_trials = int(
                self._store.get_value(
                    FieldKey("tuner_max_trials"),
                    default=20,
                )
            )
            with QSignalBlocker(self.spin_max_trials):
                self.spin_max_trials.setValue(max_trials)

            self._refresh_overview(space)

        finally:
            self._writing = False

    def build_space_from_ui(self) -> Dict[str, Any]:
        """Build a tuner_search_space fragment from current UI."""
        space: Dict[str, Any] = {}

        def _list_field(ed: QLineEdit, key: str) -> Any:
            raw = ed.text().strip()
            if not raw:
                return _REMOVE
            vals = _csv_to_ints(raw)
            if not vals:
                return _SKIP
            return vals

        def _str_list_field(ed: QLineEdit, key: str) -> Any:
            raw = ed.text().strip()
            if not raw:
                return _REMOVE
            vals = _csv_to_strs(raw)
            if not vals:
                return _SKIP
            return vals

        space["embed_dim"] = _list_field(
            self.hp_embed_dim,
            "embed_dim",
        )
        space["hidden_units"] = _list_field(
            self.hp_hidden_units,
            "hidden_units",
        )
        space["lstm_units"] = _list_field(
            self.hp_lstm_units,
            "lstm_units",
        )
        space["attention_units"] = _list_field(
            self.hp_attention_units,
            "attention_units",
        )
        space["num_heads"] = _list_field(
            self.hp_num_heads,
            "num_heads",
        )
        space["vsn_units"] = _list_field(
            self.hp_vsn_units,
            "vsn_units",
        )

        space["pde_mode"] = _str_list_field(
            self.hp_pde_mode,
            "pde_mode",
        )
        space["kappa_mode"] = _str_list_field(
            self.hp_kappa_mode,
            "kappa_mode",
        )

        dr = self._read_range_spec(self.hp_dropout)
        if dr is not None:
            space["dropout_rate"] = dr

        hd = self._read_range_spec(self.hp_hd)
        if hd is not None:
            space["hd_factor"] = hd

        space["scale_pde_residuals"] = (
            {"type": "bool"} if self.hp_scale_pde_bool.isChecked()
            else _SKIP
        )

        return space

    def commit_ui_to_store(self) -> None:
        """Push current UI -> store."""
        base = self._get_space()
        frag = self.build_space_from_ui()

        new_space = dict(base)
        for k, v in frag.items():
            if v is _SKIP:
                continue
            if v is _REMOVE:
                new_space.pop(k, None)
                continue
            new_space[k] = v

        max_trials = int(self.spin_max_trials.value())

        self._writing = True
        try:
            self._set_space(new_space)
            self._store.set_value_by_key(
                FieldKey("tuner_max_trials"),
                max_trials,
            )
        finally:
            self._writing = False

    # -----------------------------------------------------------------
    # UI build
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        """
        Build Tune tab UI with a 3-row / 2-col grid:
    
        (0,0) Architecture       (0,1) Physics
        (1,0) Tuning overview    (1,1) Tuning controls (6 items)
        (2,0) Tuning overview    (2,1) Run button
        """
        # -------------------------------------------------
        # Root layout
        # -------------------------------------------------
        u_layout = QVBoxLayout(self)
        u_layout.setContentsMargins(6, 6, 6, 6)
        u_layout.setSpacing(8)
    
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        u_layout.addLayout(grid, 1)
    
        # Two columns
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
    
        # Row 0 is "fixed-ish", row 1 grows, row 2 fixed
        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 0)
    
        def _field(w: QWidget) -> None:
            w.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Fixed,
            )
            w.setMinimumHeight(26)
            w.setMinimumWidth(140)
    
        # =================================================
        # (0,0) Architecture card
        # =================================================
        arch_card, arch_box = self._make_card(
            "Architecture search"
        )
        arch_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
    
        self.hp_embed_dim = QLineEdit()
        self.hp_hidden_units = QLineEdit()
        self.hp_lstm_units = QLineEdit()
        self.hp_attention_units = QLineEdit()
        self.hp_num_heads = QLineEdit()
        self.hp_vsn_units = QLineEdit()
    
        for w in (
            self.hp_embed_dim,
            self.hp_hidden_units,
            self.hp_lstm_units,
            self.hp_attention_units,
            self.hp_num_heads,
            self.hp_vsn_units,
        ):
            _field(w)
    
        self.hp_embed_dim.setPlaceholderText("32, 48, 64")
        self.hp_hidden_units.setPlaceholderText("64, 96, 128")
        self.hp_lstm_units.setPlaceholderText("64, 96")
        self.hp_attention_units.setPlaceholderText("32, 48")
        self.hp_num_heads.setPlaceholderText("2, 4")
        self.hp_vsn_units.setPlaceholderText("24, 32, 40")
    
        if self._range_editor_cls is None:
            raise RuntimeError(
                "TuneTab requires range_editor_cls."
            )
    
        self.hp_dropout = self._range_editor_cls(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=3,
            show_sampling=False,
        )
        self.hp_dropout.set_defaults(
            0.05,
            0.20,
            sampling=None,
        )
    
        grid_a = QGridLayout()
        r = 0
    
        grid_a.addWidget(
            QLabel("Embedding dim (comma-sep):"),
            r,
            0,
        )
        grid_a.addWidget(self.hp_embed_dim, r, 1)
        grid_a.addWidget(QLabel("LSTM units:"), r, 2)
        grid_a.addWidget(self.hp_lstm_units, r, 3)
        r += 1
    
        grid_a.addWidget(QLabel("Hidden units:"), r, 0)
        grid_a.addWidget(self.hp_hidden_units, r, 1)
        grid_a.addWidget(QLabel("Attention units:"), r, 2)
        grid_a.addWidget(self.hp_attention_units, r, 3)
        r += 1
    
        grid_a.addWidget(QLabel("Attention heads:"), r, 0)
        grid_a.addWidget(self.hp_num_heads, r, 1)
        grid_a.addWidget(QLabel("VSN units:"), r, 2)
        grid_a.addWidget(self.hp_vsn_units, r, 3)
        r += 1
    
        grid_a.addWidget(QLabel("Dropout:"), r, 0)
        grid_a.addWidget(self.hp_dropout, r, 1, 1, 3)
        r += 1
    
        self.btn_more_arch_hp = QPushButton("More architecture HP...")
        grid_a.addWidget(self.btn_more_arch_hp, r, 0, 1, 4)

    
        arch_box.addLayout(grid_a)
        grid.addWidget(arch_card, 0, 0)
    
        # =================================================
        # (0,1) Physics card
        # =================================================
        phys_card, phys_box = self._make_card(
            "Physics switches"
        )
        phys_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
    
        self.hp_pde_mode = QLineEdit()
        self.hp_kappa_mode = QLineEdit()
        _field(self.hp_pde_mode)
        _field(self.hp_kappa_mode)
    
        self.hp_pde_mode.setPlaceholderText("both, gw, cons")
        self.hp_kappa_mode.setPlaceholderText("bar, kb")
    
        self.hp_scale_pde_bool = QCheckBox(
            "Tune 'scale PDE residuals' as boolean HP"
        )
    
        self.hp_hd = self._range_editor_cls(
            min_allowed=0.0,
            max_allowed=2.0,
            decimals=3,
            show_sampling=False,
        )
        self.hp_hd.set_defaults(
            0.50,
            0.70,
            sampling=None,
        )
    
        grid_p = QGridLayout()
        p = 0
    
        grid_p.addWidget(QLabel("PDE modes:"), p, 0)
        grid_p.addWidget(self.hp_pde_mode, p, 1)
        p += 1
    
        grid_p.addWidget(QLabel("k mode (bar/kb):"), p, 0)
        grid_p.addWidget(self.hp_kappa_mode, p, 1)
        p += 1
    
        grid_p.addWidget(self.hp_scale_pde_bool, p, 0, 1, 2)
        p += 1
    
        grid_p.addWidget(QLabel("HD factor:"), p, 0)
        grid_p.addWidget(self.hp_hd, p, 1)
        p += 1
    
        self.btn_more_phys_hp = QPushButton("More physics HP...")
        grid_p.addWidget(self.btn_more_phys_hp, p, 0, 1, 2)

        phys_box.addLayout(grid_p)
        grid.addWidget(phys_card, 0, 1)
    
        # =================================================
        # (1,0) + (2,0) Tuning overview (spans 2 rows)
        # =================================================
        ov_card, ov_box = self._make_card("Tuning overview")
        ov_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
    
        self.lbl_space_keys = QLabel("Keys: -")
        self.lbl_trial_hint = QLabel("Trial: -")
        self.lbl_device_hint = QLabel("Device: -")
    
        self.btn_reset_space = QPushButton("Reset space")
        self.btn_export_space = QPushButton("Export...")
        # self.btn_export_space.setEnabled(True)  # optional; it’s enabled by default

        self.btn_reset_space.setFixedWidth(100)
        self.btn_export_space.setFixedWidth(110)
    
        top_ov = QHBoxLayout()
        top_ov.setSpacing(14)
    
        top_ov.addWidget(self.lbl_space_keys)
        top_ov.addSpacing(12)
        top_ov.addWidget(self.lbl_trial_hint)
        top_ov.addSpacing(12)
        top_ov.addWidget(self.lbl_device_hint)
    
        top_ov.addStretch(1)
        top_ov.addWidget(self.btn_reset_space)
        top_ov.addWidget(self.btn_export_space)
    
        ov_box.addLayout(top_ov)
    
        self.txt_space_preview = QPlainTextEdit()
        self.txt_space_preview.setReadOnly(True)
        self.txt_space_preview.setMinimumHeight(110)
        self.txt_space_preview.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        ov_box.addWidget(self.txt_space_preview, 1)
    
        # Span overview across rows 1 and 2 on left column
        grid.addWidget(ov_card, 1, 0, 2, 1)
    
        # =================================================
        # (1,1) Tuning controls (6 items) card
        # =================================================
        ctrl_card, ctrl_box = self._make_card("Tuning controls")
        ctrl_card.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
    
        # Buttons / controls (wired in _wire_ui)
        self.btn_tune_options = QPushButton("Advanced options...")
        self.btn_model_params = QPushButton("Model params...")
        self.btn_scalars = QPushButton("Scalars & losses...")
    
        self.btn_search_algo = QPushButton("Search algo...")

        self.chk_eval_tuned = QCheckBox("Evaluate tuned model")
        self.chk_eval_tuned.setChecked(False)
    
        self.spin_max_trials = QSpinBox()
        self.spin_max_trials.setRange(1, 999)
        self.spin_max_trials.setValue(20)
    
        max_trials_w = QWidget()
        max_trials_l = QHBoxLayout(max_trials_w)
        max_trials_l.setContentsMargins(0, 0, 0, 0)
        max_trials_l.setSpacing(6)
        max_trials_l.addWidget(QLabel("Max trials:"))
        max_trials_l.addWidget(self.spin_max_trials)
        max_trials_l.addStretch(1)
    
        ctrl_grid = QGridLayout()
        ctrl_grid.setHorizontalSpacing(10)
        ctrl_grid.setVerticalSpacing(10)
    
        # Row 0 (3 buttons)
        ctrl_grid.addWidget(self.btn_tune_options, 0, 0)
        ctrl_grid.addWidget(self.btn_model_params, 0, 1)
        ctrl_grid.addWidget(self.btn_scalars, 0, 2)
    
        # Row 1 (search + evaluate + max trials)
        ctrl_grid.addWidget(self.btn_search_algo, 1, 0)
        ctrl_grid.addWidget(self.chk_eval_tuned, 1, 1)
        ctrl_grid.addWidget(max_trials_w, 1, 2)
    
        # Make columns behave nicely
        ctrl_grid.setColumnStretch(0, 1)
        ctrl_grid.setColumnStretch(1, 1)
        ctrl_grid.setColumnStretch(2, 1)
    
        ctrl_box.addLayout(ctrl_grid)
        ctrl_box.addStretch(1)
    
        grid.addWidget(ctrl_card, 1, 1)
    
        # =================================================
        # (2,1) Run button (separate from controls)
        # =================================================
        run_w = QWidget()
        run_l = QHBoxLayout(run_w)
        run_l.setContentsMargins(0, 0, 0, 0)
        run_l.addStretch(1)
    
        self.btn_run_tune = self._make_run_button("Run tuning")
        run_l.addWidget(self.btn_run_tune)
    
        grid.addWidget(run_w, 2, 1)

    def _on_more_phys_hp(self) -> None:
        if PhysHPDialog.edit(store=self._store, parent=self):
            self.refresh_from_store()
            
    def _on_more_arch_hp(self) -> None:
        if ArchHPDialog.edit(store=self._store, parent=self):
            self.refresh_from_store()
    def _on_search_algo(self) -> None:
        if SearchAlgoDialog.edit(store=self._store, parent=self):
            self.refresh_from_store()

    def _on_export_clicked(self) -> None:
        # optional: show export prefs dialog first
        ok = ExportDialog.edit(store=self._store, parent=self)
        if not ok:
            return
    
        export_with_saved_prefs(self._store, parent=self)
    
    def _on_scalars_losses(self) -> None:
        ok = ScalarsLossDialog.edit(
            store=self._store,
            range_editor_cls=self._range_editor_cls,
            parent=self,
        )
        if ok:
            self.refresh_from_store()
            
    def _on_model_params(self) -> None:
        ok = ModelParamsDialog.edit(store=self._store, parent=self)
        if ok:
            self.refresh_from_store()

    # -----------------------------------------------------------------
    # Wiring
    # -----------------------------------------------------------------
    def _wire_ui(self) -> None:
        self.btn_run_tune.clicked.connect(self.run_clicked.emit)
        self.btn_tune_options.clicked.connect(
            self.advanced_clicked.emit
        )
        self.btn_model_params.clicked.connect(self._on_model_params)
        self.btn_scalars.clicked.connect(self._on_scalars_losses)
        self.btn_search_algo.clicked.connect(self._on_search_algo)


        self.btn_reset_space.clicked.connect(self._on_reset)

        # Commit edits -> store
        for ed in (
            self.hp_embed_dim,
            self.hp_hidden_units,
            self.hp_lstm_units,
            self.hp_attention_units,
            self.hp_num_heads,
            self.hp_vsn_units,
            self.hp_pde_mode,
            self.hp_kappa_mode,
        ):
            ed.editingFinished.connect(self._on_ui_commit)

        self.hp_scale_pde_bool.toggled.connect(self._on_ui_commit)

        self._connect_range_editor(self.hp_dropout)
        self._connect_range_editor(self.hp_hd)

        self.spin_max_trials.valueChanged.connect(
            self._on_ui_commit
        )
        self.chk_eval_tuned.toggled.connect(self._on_ui_commit)
        self.btn_more_arch_hp.clicked.connect(self._on_more_arch_hp)
        self.btn_more_phys_hp.clicked.connect(self._on_more_phys_hp)
        
        self.btn_export_space.clicked.connect(self._on_export_clicked)



    def _wire_store(self) -> None:
        def _on_store_changed(*_: Any) -> None:
            if self._writing:
                return
            self.refresh_from_store()

        try:
            self._store.config_changed.connect(_on_store_changed)
        except Exception:
            pass

        try:
            self._store.config_replaced.connect(_on_store_changed)
        except Exception:
            pass

    def _on_ui_commit(self) -> None:
        if self._writing:
            return
        self.commit_ui_to_store()

    def _on_reset(self) -> None:
        off = self._store.get_value(
            FieldKey("offset_mode"),
            default="mul",
        )
        try:
            space = default_tuner_search_space(
                offset_mode=str(off),
            )
        except Exception:
            space = {}

        self._writing = True
        try:
            self._set_space(space)
        finally:
            self._writing = False

        self.refresh_from_store()

    # -----------------------------------------------------------------
    # Store helpers
    # -----------------------------------------------------------------
    def _get_space(self) -> Dict[str, Any]:
        val = self._store.get_value(
            FieldKey("tuner_search_space"),
            default={},
        )
        if isinstance(val, dict):
            return dict(val)
        return {}

    def _set_space(self, space: Dict[str, Any]) -> None:
        self._store.merge_dict_field(
            "tuner_search_space",
            dict(space),
            replace=True,
        )

    def _pretty_name(self, key: str) -> str:
        mapping = {
            "embed_dim": "Embedding dim",
            "hidden_units": "Hidden units",
            "lstm_units": "LSTM units",
            "attention_units": "Attention units",
            "num_heads": "Attention heads",
            "vsn_units": "VSN units",
            "dropout_rate": "Dropout",
            "pde_mode": "PDE mode(s)",
            "kappa_mode": "k mode (bar/kb)",
            "hd_factor": "HD factor",
            "learning_rate": "Learning rate",
            "lambda_cons": "Lambda (cons)",
            "lambda_gw": "Lambda (gw)",
            "lambda_prior": "Lambda (prior)",
            "lambda_smooth": "Lambda (smooth)",
            "lambda_bounds": "Lambda (bounds)",
            "lambda_offset": "Lambda offset",
            "lambda_q": "Lambda (q)",
            "memory_size": "Memory size",
            "max_window_size": "Max window size",
            "attention_levels": "Attention levels",
            "scale_pde_residuals": "Scale PDE residuals",
        }
        return mapping.get(key, key.replace("_", " ").title())
    
    
    def _fmt_value(self, v: Any) -> str:
        if isinstance(v, list):
            if v and isinstance(v[0], list):
                parts = []
                for row in v:
                    parts.append(", ".join(str(x) for x in row))
                return " | ".join(parts)
            return ", ".join(str(x) for x in v)
    
        if isinstance(v, dict):
            t = str(v.get("type", "")).lower()
            if t in ("float", "int", "range"):
                vmin = v.get("min_value", v.get("min"))
                vmax = v.get("max_value", v.get("max"))
                step = v.get("step", None)
                samp = v.get("sampling", None)
                s = f"{vmin} -> {vmax}"
                if step is not None:
                    s += f"  (step {step})"
                if samp:
                    s += f"  [{samp}]"
                return s
    
            if t == "choice":
                vals = v.get("values", [])
                return ", ".join(str(x) for x in vals)
    
            if t in ("bool", "boolean"):
                return "True / False"
    
            return ", ".join(f"{k}={v[k]}" for k in sorted(v.keys()))
    
        if isinstance(v, bool):
            return "True" if v else "False"
    
        return str(v)
    
    
    def _format_space_pretty(self, space: Dict[str, Any]) -> str:
        groups = [
            ("Architecture", [
                "embed_dim",
                "hidden_units",
                "lstm_units",
                "attention_units",
                "num_heads",
                "vsn_units",
                "dropout_rate",
                "attention_levels",
            ]),
            ("Physics", [
                "pde_mode",
                "kappa_mode",
                "hd_factor",
                "scale_pde_residuals",
                "kappa",
                "mv",
            ]),
            ("Optimization", [
                "learning_rate",
                "kappa_lr_mult",
                "mv_lr_mult",
            ]),
            ("Loss weights", [
                "lambda_cons",
                "lambda_gw",
                "lambda_prior",
                "lambda_smooth",
                "lambda_bounds",
                "lambda_offset",
                "lambda_q",
                "scale_q_with_offset",
                "scale_mv_with_offset",
            ]),
            ("Data / memory", [
                "max_window_size",
                "memory_size",
                "scales",
            ]),
        ]
    
        used = set()
        lines: list[str] = []
        idx = 1
    
        for title, keys in groups:
            present = [k for k in keys if k in space]
            if not present:
                continue
            lines.append(f"{title}")
            lines.append("-" * len(title))
            for k in present:
                used.add(k)
                name = self._pretty_name(k)
                val = self._fmt_value(space.get(k))
                lines.append(f"[{idx:02d}] {name}: {val}")
                idx += 1
            lines.append("")
    
        leftovers = [k for k in sorted(space.keys()) if k not in used]
        if leftovers:
            lines.append("Other")
            lines.append("-----")
            for k in leftovers:
                name = self._pretty_name(k)
                val = self._fmt_value(space.get(k))
                lines.append(f"[{idx:02d}] {name}: {val}")
                idx += 1
    
        return "\n".join(lines).strip()

    # -----------------------------------------------------------------
    # Overview helpers
    # -----------------------------------------------------------------
    def _refresh_overview(self, space: Dict[str, Any]) -> None:
        try:
            keys_n = len(space)
        except Exception:
            keys_n = 0

        epochs = self._store.get_value(
            FieldKey("epochs"),
            default=None,
        )
        batch = self._store.get_value(
            FieldKey("batch_size"),
            default=None,
        )
        dev = self._store.get_value(
            FieldKey("tf_device_mode"),
            default="auto",
        )

        self.lbl_space_keys.setText(f"Keys: {keys_n}")
        self.lbl_trial_hint.setText(
            f"Trial: epochs={epochs}  batch={batch}"
        )
        self.lbl_device_hint.setText(f"Device: {dev}")

        txt = self._format_space_pretty(space)

        with QSignalBlocker(self.txt_space_preview):
            self.txt_space_preview.setPlainText(txt)

    # -----------------------------------------------------------------
    # RangeListEditor best-effort bridges
    # -----------------------------------------------------------------
    def _connect_range_editor(self, editor: QWidget) -> None:
        for sig_name in (
            "changed",
            "valueChanged",
            "sig_changed",
            "signalChanged",
        ):
            sig = getattr(editor, sig_name, None)
            if sig is None:
                continue
            if hasattr(sig, "connect"):
                try:
                    sig.connect(self._on_ui_commit)
                    return
                except Exception:
                    continue

    def _apply_range_defaults(
        self,
        *,
        editor: Any,
        spec: Any,
        dmin: float,
        dmax: float,
    ) -> None:
        try:
            if isinstance(spec, dict):
                vmin = float(spec.get("min", dmin))
                vmax = float(spec.get("max", dmax))
                sampling = spec.get("sampling", None)
                editor.set_defaults(vmin, vmax, sampling=sampling)
                return
        except Exception:
            pass
        try:
            editor.set_defaults(dmin, dmax, sampling=None)
        except Exception:
            pass

    def _read_range_spec(self, editor: Any) -> Any:
        for meth in (
            "to_spec",
            "get_spec",
            "export_spec",
            "to_dict",
            "value",
        ):
            fn = getattr(editor, meth, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    continue

        vmin = None
        vmax = None

        for nm in ("sp_min", "sb_min", "min_spin"):
            w = getattr(editor, nm, None)
            if w is not None and hasattr(w, "value"):
                try:
                    vmin = float(w.value())
                    break
                except Exception:
                    continue

        for nm in ("sp_max", "sb_max", "max_spin"):
            w = getattr(editor, nm, None)
            if w is not None and hasattr(w, "value"):
                try:
                    vmax = float(w.value())
                    break
                except Exception:
                    continue

        if vmin is None or vmax is None:
            return None

        return {"type": "range", "min": vmin, "max": vmax}
