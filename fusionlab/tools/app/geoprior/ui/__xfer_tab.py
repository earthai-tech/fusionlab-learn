# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.xfer_tab

Transferability tab (cross-city transfer matrix).

v3.2 goals
----------
- UI is centralized in this module (app.py stays clean).
- Store is the single source of truth (GeoConfigStore).
- Advanced knobs are store-backed (xfer.* keys).
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)

from PyQt5.QtCore import (
    Qt,
    QSignalBlocker,
    pyqtSignal,
    QEvent,
)

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QButtonGroup,
    QStackedWidget,
    QFrame,
    QStyle,
    QSizePolicy,
    QBoxLayout,
)

from ..config.store import GeoConfigStore
from .map_view import MapView, CityPoint

def _blocked(w: QWidget):
    return QSignalBlocker(w)


def _parse_float_list(text: str) -> Optional[List[float]]:
    raw = (text or "").strip()
    if not raw:
        return None

    bits = [b.strip() for b in raw.split(",")]
    out: List[float] = []
    for b in bits:
        if not b:
            continue
        try:
            out.append(float(b))
        except Exception:
            return None

    return out or None


class XferTab(QWidget):
    """Transferability tab widget."""

    run_clicked = pyqtSignal()
    view_clicked = pyqtSignal()

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: Callable[
            [str],
            tuple[QWidget, QVBoxLayout],
        ],
        make_run_button: Callable[[str], QToolButton],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._make_card = make_card
        self._make_run_button = make_run_button

        self._build_ui()
        self._connect_signals()
        self._sync_from_store()
        self._apply_view_mode_from_store()
        self._update_map_from_store()


        store.config_changed.connect(self._on_store_changed)

    # -------------------------------------------------
    # Public helpers
    # -------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Collect a controller-friendly snapshot."""
        return {
            "city_a": self.xfer_city_a.text().strip(),
            "city_b": self.xfer_city_b.text().strip(),
            "results_root": (
                self.xfer_results_root.text().strip()
                or None
            ),
            "splits": self._get_checked_splits(),
            "calib_modes": self._get_checked_calib(),
            "batch_size": int(self.sp_xfer_batch.value()),
            "rescale_to_source": bool(
                self.chk_xfer_rescale.isChecked()
            ),
            "quantiles_override": self.get_quantiles(),
            "write_json": bool(
                self.chk_xfer_json.isChecked()
            ),
            "write_csv": bool(
                self.chk_xfer_csv.isChecked()
            ),
            "prefer_tuned": bool(
                self.chk_xfer_prefer_tuned.isChecked()
            ),
            "align_policy": str(
                self.cmb_xfer_align.currentData()
                or "align_by_name_pad"
            ),
            "interval_target": float(
                self.sp_xfer_interval.value()
            ),
            "load_endpoint": str(
                self.cmb_xfer_endpoint.currentData()
                or "serve"
            ),
            "export_physics_payload": bool(
                self.chk_xfer_phys_payload.isChecked()
            ),
            "export_physical_parameters_csv": bool(
                self.chk_xfer_phys_csv.isChecked()
            ),
            "write_eval_future_csv": bool(
                self.chk_xfer_eval_future.isChecked()
            ),
            "view_kind": str(
                self.cmb_xfer_view.currentData()
                or "calib_panel"
            ),
            "view_split": str(
                self.cmb_xfer_view_split.currentData()
                or "val"
            ),
            "strategies": self._get_checked_strategies(),
            "rescale_modes": self._get_checked_rescale_modes(),
            "allow_reorder_dynamic": self._opt_bool_from_combo(
                self.cmb_xfer_allow_re_dyn
            ),
            "allow_reorder_future": self._opt_bool_from_combo(
                self.cmb_xfer_allow_re_fut
            ),
            "warm_split": str(
                self.cmb_xfer_warm_split.currentData()
                or "train"
            ),
            "warm_samples": int(
                self.sp_xfer_warm_samples.value()
            ),
            "warm_frac": float(self.sp_xfer_warm_frac.value()),
            "warm_epochs": int(self.sp_xfer_warm_epochs.value()),
            "warm_lr": float(self.sp_xfer_warm_lr.value()),
            "warm_seed": int(self.sp_xfer_warm_seed.value()),

        }

    def set_last_output(self, out_dir: Optional[str]) -> None:
        txt = out_dir or "No transfer run yet."
        self.lbl_xfer_last_out.setText(txt)
        self.btn_xfer_view.setVisible(bool(out_dir))

    def set_view_enabled(self, enabled: bool) -> None:
        self.btn_xfer_view.setEnabled(bool(enabled))

    def set_run_enabled(self, enabled: bool) -> None:
        self.btn_run_xfer.setEnabled(bool(enabled))

    def get_quantiles(self) -> Optional[Sequence[float]]:
        return _parse_float_list(
            self.ed_xfer_quantiles.text()
        )

    # -------------------------------------------------
    # UI
    # -------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(10)
        row.setAlignment(Qt.AlignTop)

        cities_card, cities_box = self._make_card(
            "Cities & splits"
        )
        self._build_cities_box(cities_box)
        row.addWidget(cities_card, 1, Qt.AlignTop)

        res_card, res_box = self._make_card(
            "Results & view"
        )
        self._build_results_box(res_box)
        row.addWidget(res_card, 1, Qt.AlignTop)

        root.addLayout(row)

        # ---------------------------------------------
        # Mini tabs: [ Map ] [ Options ]
        # ---------------------------------------------
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        
        self.btn_mode_map = QToolButton()
        self.btn_mode_map.setText("Map")
        self.btn_mode_map.setCheckable(True)
        
        self.btn_mode_opts = QToolButton()
        self.btn_mode_opts.setText("Options")
        self.btn_mode_opts.setCheckable(True)
        
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self.btn_mode_map, 0)
        self._mode_group.addButton(self.btn_mode_opts, 1)
        
        mode_row.addWidget(self.btn_mode_map)
        mode_row.addWidget(self.btn_mode_opts)
        mode_row.addStretch(1)
        
        self.btn_run_xfer = self._make_run_button(
            "Run transfer matrix"
        )
        mode_row.addWidget(self.btn_run_xfer)
        
        root.addLayout(mode_row)
        
        # ---------------------------------------------
        # Pages: Map + Options
        # ---------------------------------------------
        self._stack = QStackedWidget()
        
        self.map_view = MapView(parent=self)
        self._stack.addWidget(self.map_view)
        
        self.advanced_box = self._build_advanced_box()
        adv_scroll = QScrollArea()
        adv_scroll.setWidgetResizable(True)
        adv_scroll.setFrameShape(QScrollArea.NoFrame)
        adv_scroll.setWidget(self.advanced_box)
        
        adv_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        adv_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )
        
        adv_scroll.horizontalScrollBar().setSingleStep(24)
        adv_scroll.horizontalScrollBar().setPageStep(240)
        
        self._adv_scroll = adv_scroll
        adv_scroll.viewport().installEventFilter(self)
        
        self._stack.addWidget(adv_scroll)

        root.addWidget(self._stack, 1)

    def _build_cities_box(self, box: QVBoxLayout) -> None:
        self.xfer_city_a = QLineEdit()
        self.xfer_city_a.setPlaceholderText("nansha")

        self.xfer_city_b = QLineEdit()
        self.xfer_city_b.setPlaceholderText("zhongshan")

        self.chk_xfer_split_train = QCheckBox("train")
        self.chk_xfer_split_val = QCheckBox("val")
        self.chk_xfer_split_test = QCheckBox("test")

        self.chk_xfer_cal_none = QCheckBox("none")
        self.chk_xfer_cal_source = QCheckBox("source")
        self.chk_xfer_cal_target = QCheckBox("target")

        self.sp_xfer_batch = QSpinBox()
        self.sp_xfer_batch.setRange(1, 2048)

        self.chk_xfer_rescale = QCheckBox(
            "Rescale target city to source domain"
        )

        grid = QGridLayout()
        r = 0

        grid.addWidget(QLabel("City A (source):"), r, 0)
        grid.addWidget(self.xfer_city_a, r, 1)
        r += 1

        grid.addWidget(QLabel("City B (target):"), r, 0)
        grid.addWidget(self.xfer_city_b, r, 1)
        r += 1

        grid.addWidget(QLabel("Splits:"), r, 0)
        splits = QHBoxLayout()
        splits.addWidget(self.chk_xfer_split_train)
        splits.addWidget(self.chk_xfer_split_val)
        splits.addWidget(self.chk_xfer_split_test)
        splits.addStretch(1)
        grid.addLayout(splits, r, 1)
        r += 1

        grid.addWidget(QLabel("Calibration:"), r, 0)
        cal = QHBoxLayout()
        cal.addWidget(self.chk_xfer_cal_none)
        cal.addWidget(self.chk_xfer_cal_source)
        cal.addWidget(self.chk_xfer_cal_target)
        cal.addStretch(1)
        grid.addLayout(cal, r, 1)
        r += 1

        grid.addWidget(QLabel("Batch size:"), r, 0)
        grid.addWidget(self.sp_xfer_batch, r, 1)
        r += 1

        grid.addWidget(self.chk_xfer_rescale, r, 0, 1, 2)

        box.addLayout(grid)

    def _build_results_box(self, box: QVBoxLayout) -> None:
        self.xfer_results_root = QLineEdit()
        self.xfer_results_root_btn = QPushButton("Browse…")

        self.lbl_xfer_last_out = QLabel(
            "No transfer run yet."
        )
        self.lbl_xfer_last_out.setObjectName(
            "xferLastOutLabel"
        )

        self.cmb_xfer_view = QComboBox()
        self.cmb_xfer_view.addItem(
            "Calibration vs error (scatter panel)",
            "calib_panel",
        )
        self.cmb_xfer_view.addItem(
            "Per-horizon MAE + cov/sharp (summary)",
            "summary_panel",
        )

        self.cmb_xfer_view_split = QComboBox()
        self.cmb_xfer_view_split.addItem(
            "Validation (val)",
            "val",
        )
        self.cmb_xfer_view_split.addItem(
            "Test (test)",
            "test",
        )

        self.btn_xfer_view = QPushButton(
            "Make view figure…"
        )
        self.btn_xfer_view.setVisible(False)

        grid = QGridLayout()
        r = 0

        grid.addWidget(QLabel("Results root:"), r, 0)
        grid.addWidget(self.xfer_results_root, r, 1)
        grid.addWidget(self.xfer_results_root_btn, r, 2)
        r += 1

        grid.addWidget(QLabel("Last output folder:"), r, 0)
        grid.addWidget(self.lbl_xfer_last_out, r, 1, 1, 2)
        r += 1

        grid.addWidget(QLabel("View type:"), r, 0)
        grid.addWidget(self.cmb_xfer_view, r, 1, 1, 2)
        r += 1

        grid.addWidget(QLabel("View split:"), r, 0)
        grid.addWidget(self.cmb_xfer_view_split, r, 1, 1, 2)
        r += 1

        grid.addWidget(self.btn_xfer_view, r, 0, 1, 3)

        box.addLayout(grid)

    def _build_advanced_box(self) -> QWidget:
        outer = QWidget()

        # -------------------------------------------------
        # Fill width (no "center host" + big empty margins)
        # -------------------------------------------------
        root = QHBoxLayout(outer)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
    
        host = QWidget()
        host.setObjectName("xferAdvContent")
        host.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
    
        lay = QVBoxLayout(host)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(10)
    
        root.addWidget(host, 1)
    
        def _section(
            title: str,
            icon: QStyle.StandardPixmap,
        ) -> tuple[QFrame, QVBoxLayout]:
            frame = QFrame()
            frame.setObjectName("xferAdvSection")
    
            v = QVBoxLayout(frame)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(6)
    
            header = QToolButton()
            header.setObjectName("xferAdvToggle")
            header.setCheckable(True)
            header.setChecked(True)
            header.setToolButtonStyle(
                Qt.ToolButtonTextBesideIcon
            )
            header.setIcon(
                self.style().standardIcon(icon)
            )
            header.setText(title)
            header.setArrowType(Qt.DownArrow)
    
            body = QWidget()
            body.setObjectName("xferAdvBody")
    
            body_lay = QVBoxLayout(body)
            body_lay.setContentsMargins(10, 10, 10, 10)
            body_lay.setSpacing(10)
    
            def _toggle(on: bool) -> None:
                body.setVisible(bool(on))
                header.setArrowType(
                    Qt.DownArrow if on else Qt.RightArrow
                )
    
            header.toggled.connect(_toggle)
    
            v.addWidget(header)
            v.addWidget(body)
            return frame, body_lay
    
        def _field(
            title: str,
            body: QWidget,
            tip: str,
            *,
            right: Optional[QWidget] = None,
        ) -> QFrame:
            f = QFrame()
            f.setObjectName("xferField")
    
            v = QVBoxLayout(f)
            v.setContentsMargins(10, 8, 10, 10)
            v.setSpacing(6)
    
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            top.setSpacing(6)
    
            lbl = QLabel(title)
            lbl.setObjectName("xferFieldTitle")
    
            top.addWidget(lbl)
            top.addStretch(1)
    
            if right is not None:
                top.addWidget(right)
    
            top.addWidget(self._make_help_btn(tip))
    
            v.addLayout(top)
            v.addWidget(body)
            return f
    
        # -------------------------------------------------
        # Header row: title + reset
        # -------------------------------------------------
        top = QHBoxLayout()
        top.setSpacing(10)
    
        tcol = QVBoxLayout()
        tcol.setContentsMargins(0, 0, 0, 0)
        tcol.setSpacing(2)
    
        ttl = QLabel("Advanced options")
        ttl.setObjectName("xferAdvTitle")
    
        sub = QLabel(
            "Fine control for audits and transfer semantics."
        )
        sub.setObjectName("setupCardSubtitle")
    
        tcol.addWidget(ttl)
        tcol.addWidget(sub)
        top.addLayout(tcol, 1)
    
        self.btn_xfer_adv_reset = QToolButton()
        self.btn_xfer_adv_reset.setObjectName("miniAction")
        self.btn_xfer_adv_reset.setIcon(
            self.style().standardIcon(
                QStyle.SP_BrowserReload
            )
        )
        self.btn_xfer_adv_reset.setToolTip(
            "Reset Advanced options to defaults."
        )
        self.btn_xfer_adv_reset.clicked.connect(
            self._on_adv_reset
        )
        top.addWidget(self.btn_xfer_adv_reset)
    
        lay.addLayout(top)
    
        # -------------------------------------------------
        # Responsive columns (2-col -> 1-col)
        # -------------------------------------------------
        self._adv_cols = QBoxLayout(QBoxLayout.LeftToRight)
        self._adv_cols.setContentsMargins(0, 0, 0, 0)
        self._adv_cols.setSpacing(12)
        
        self._adv_left = QWidget()
        self._adv_left.setObjectName("xferAdvColLeft")
        left = QVBoxLayout(self._adv_left)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)
        left.setAlignment(Qt.AlignTop)
        
        self._adv_right = QWidget()
        self._adv_right.setObjectName("xferAdvColRight")
        right = QVBoxLayout(self._adv_right)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.setAlignment(Qt.AlignTop)
        
        self._adv_cols.addWidget(self._adv_left, 1)
        self._adv_cols.addWidget(self._adv_right, 1)
        
        lay.addLayout(self._adv_cols)

        # =================================================
        # LEFT: Outputs & alignment
        # =================================================
        s1, s1_lay = _section(
            "Outputs & alignment",
            QStyle.SP_FileDialogDetailedView,
        )
    
        grid1 = QGridLayout()
        grid1.setHorizontalSpacing(10)
        grid1.setVerticalSpacing(10)
        grid1.setContentsMargins(0, 0, 0, 0)
        grid1.setColumnStretch(0, 1)
        grid1.setColumnStretch(1, 1)
        s1_lay.addLayout(grid1)
    
        # Quantiles
        self.ed_xfer_quantiles = QLineEdit()
        self.ed_xfer_quantiles.setPlaceholderText(
            "0.1,0.5,0.9"
        )
    
        self.lbl_xfer_q_chip = QLabel("AUTO")
        self.lbl_xfer_q_chip.setObjectName("xferAdvChip")
    
        self.btn_xfer_q_clear = QToolButton()
        self.btn_xfer_q_clear.setObjectName("miniAction")
        self.btn_xfer_q_clear.setIcon(
            self.style().standardIcon(
                QStyle.SP_DialogCloseButton
            )
        )
        self.btn_xfer_q_clear.setToolTip(
            "Clear quantiles override"
        )
        self.btn_xfer_q_clear.clicked.connect(
            lambda: self.ed_xfer_quantiles.setText("")
        )
        self.btn_xfer_q_clear.clicked.connect(
            self._update_quantiles_chip
        )
        self.btn_xfer_q_clear.clicked.connect(
            self._push_to_store
        )
    
        q_body = QWidget()
        q_row = QHBoxLayout(q_body)
        q_row.setContentsMargins(0, 0, 0, 0)
        q_row.setSpacing(6)
        q_row.addWidget(self.ed_xfer_quantiles, 1)
        q_row.addWidget(self.lbl_xfer_q_chip)
        q_row.addWidget(self.btn_xfer_q_clear)
    
        f_q = _field(
            "Quantiles override",
            q_body,
            "Override output quantiles for eval/export.\n"
            "Leave empty to use model defaults.",
        )
        grid1.addWidget(f_q, 0, 0, 1, 2)
    
        # Formats
        self.chk_xfer_json = QCheckBox("Write JSON")
        self.chk_xfer_csv = QCheckBox("Write CSV")
    
        io_body = QWidget()
        io = QHBoxLayout(io_body)
        io.setContentsMargins(0, 0, 0, 0)
        io.setSpacing(12)
        io.addWidget(self.chk_xfer_json)
        io.addWidget(self.chk_xfer_csv)
        io.addStretch(1)
    
        f_io = _field(
            "Formats",
            io_body,
            "CSV is easier to audit.\n"
            "JSON enables richer summary panels.",
        )
        grid1.addWidget(f_io, 1, 0)
    
        # Prefer tuned
        self.chk_xfer_prefer_tuned = QCheckBox(
            "Prefer tuned calibrator if available"
        )
    
        pt_body = QWidget()
        pt = QVBoxLayout(pt_body)
        pt.setContentsMargins(0, 0, 0, 0)
        pt.setSpacing(0)
        pt.addWidget(self.chk_xfer_prefer_tuned)
    
        f_pt = _field(
            "Calibration",
            pt_body,
            "If tuning artifacts exist, reuse them\n"
            "instead of fitting a fresh calibrator.",
        )
        grid1.addWidget(f_pt, 1, 1)
    
        # Align policy
        self.cmb_xfer_align = QComboBox()
        self.cmb_xfer_align.addItem(
            "Align by name (pad)",
            "align_by_name_pad",
        )
        self.cmb_xfer_align.addItem(
            "Strict (same columns)",
            "strict",
        )
    
        f_align = _field(
            "Align policy",
            self.cmb_xfer_align,
            "How to align feature columns across cities.\n"
            "Strict = fail fast when mismatched.",
        )
        grid1.addWidget(f_align, 2, 0)
    
        # Reorder dynamic
        self.cmb_xfer_allow_re_dyn = QComboBox()
        self._fill_opt_bool_combo(self.cmb_xfer_allow_re_dyn)
    
        f_rdyn = _field(
            "Reorder dynamic",
            self.cmb_xfer_allow_re_dyn,
            "Auto = follow align policy.\n"
            "Allow = reorder if names match.\n"
            "Block = treat reorder as mismatch.",
        )
        grid1.addWidget(f_rdyn, 2, 1)
    
        # Reorder future
        self.cmb_xfer_allow_re_fut = QComboBox()
        self._fill_opt_bool_combo(self.cmb_xfer_allow_re_fut)
    
        f_rfut = _field(
            "Reorder future",
            self.cmb_xfer_allow_re_fut,
            "Same as above, for future-known inputs.",
        )
        grid1.addWidget(f_rfut, 3, 0)
    
        # Interval target
        self.sp_xfer_interval = QDoubleSpinBox()
        self.sp_xfer_interval.setRange(0.10, 0.99)
        self.sp_xfer_interval.setSingleStep(0.01)
        self.sp_xfer_interval.setDecimals(2)
    
        f_int = _field(
            "Interval target",
            self.sp_xfer_interval,
            "Controls calibration coverage.",
        )
        grid1.addWidget(f_int, 3, 1)
    
        # Load endpoint
        self.cmb_xfer_endpoint = QComboBox()
        self.cmb_xfer_endpoint.addItem("serve", "serve")
        self.cmb_xfer_endpoint.addItem("export", "export")
    
        f_end = _field(
            "Load endpoint",
            self.cmb_xfer_endpoint,
            "Choose where to load artifacts from.",
        )
        grid1.addWidget(f_end, 4, 0)
    
        # Exports
        self.chk_xfer_phys_payload = QCheckBox(
            "Export physics payload"
        )
        self.chk_xfer_phys_csv = QCheckBox(
            "Export physical parameters CSV"
        )
        self.chk_xfer_eval_future = QCheckBox(
            "Write future evaluation CSV"
        )
    
        ex_body = QWidget()
        ex = QGridLayout(ex_body)
        ex.setContentsMargins(0, 0, 0, 0)
        ex.setHorizontalSpacing(12)
        ex.setVerticalSpacing(8)
        ex.addWidget(self.chk_xfer_phys_payload, 0, 0)
        ex.addWidget(self.chk_xfer_phys_csv, 0, 1)
        ex.addWidget(
            self.chk_xfer_eval_future,
            1,
            0,
            1,
            2,
        )
    
        f_ex = _field(
            "Exports",
            ex_body,
            "Export physics/closures for audits.\n"
            "Write extra CSVs when available.",
        )
        grid1.addWidget(f_ex, 4, 1)
    
        left.addWidget(s1)
    
        # =================================================
        # RIGHT: Strategies & warm-start
        # =================================================
        s2, s2_lay = _section(
            "Strategies & warm-start",
            QStyle.SP_ArrowRight,
        )
    
        grid2 = QGridLayout()
        grid2.setHorizontalSpacing(10)
        grid2.setVerticalSpacing(10)
        grid2.setContentsMargins(0, 0, 0, 0)
        grid2.setColumnStretch(0, 1)
        grid2.setColumnStretch(1, 1)
        s2_lay.addLayout(grid2)
    
        self.chk_xfer_strat_baseline = QCheckBox(
            "In-domain baselines (A→A, B→B)"
        )
        self.chk_xfer_strat_xfer = QCheckBox(
            "Zero-shot transfer (A→B, B→A)"
        )
        self.chk_xfer_strat_warm = QCheckBox(
            "Warm-start fine-tune (A→(few B)→B)"
        )
    
        st_body = QWidget()
        st = QVBoxLayout(st_body)
        st.setContentsMargins(0, 0, 0, 0)
        st.setSpacing(6)
        st.addWidget(self.chk_xfer_strat_baseline)
        st.addWidget(self.chk_xfer_strat_xfer)
        st.addWidget(self.chk_xfer_strat_warm)
    
        f_st = _field(
            "Strategies",
            st_body,
            "Pick which comparisons to run.\n"
            "Warm-start unlocks the panel below.",
        )
        grid2.addWidget(f_st, 0, 0, 1, 2)
    
        self.chk_xfer_rmode_as_is = QCheckBox("as-is")
        self.chk_xfer_rmode_strict = QCheckBox("strict")
    
        rm_body = QWidget()
        rm = QHBoxLayout(rm_body)
        rm.setContentsMargins(0, 0, 0, 0)
        rm.setSpacing(12)
        rm.addWidget(self.chk_xfer_rmode_as_is)
        rm.addWidget(self.chk_xfer_rmode_strict)
        rm.addStretch(1)
    
        f_rm = _field(
            "Rescale variants",
            rm_body,
            "Run multiple rescale variants for audits.\n"
            "Strict keeps scalers consistent.",
        )
        grid2.addWidget(f_rm, 1, 0)
    
        self.cmb_xfer_warm_split = QComboBox()
        self.cmb_xfer_warm_split.addItem("train", "train")
        self.cmb_xfer_warm_split.addItem("val", "val")
        self.cmb_xfer_warm_split.addItem("test", "test")
    
        self.sp_xfer_warm_samples = QSpinBox()
        self.sp_xfer_warm_samples.setRange(1, 1_000_000)
    
        self.sp_xfer_warm_frac = QDoubleSpinBox()
        self.sp_xfer_warm_frac.setRange(0.0, 1.0)
        self.sp_xfer_warm_frac.setSingleStep(0.05)
        self.sp_xfer_warm_frac.setDecimals(2)
    
        self.sp_xfer_warm_epochs = QSpinBox()
        self.sp_xfer_warm_epochs.setRange(1, 10_000)
    
        self.sp_xfer_warm_lr = QDoubleSpinBox()
        self.sp_xfer_warm_lr.setRange(1e-8, 1.0)
        self.sp_xfer_warm_lr.setDecimals(8)
        self.sp_xfer_warm_lr.setSingleStep(1e-4)
    
        self.sp_xfer_warm_seed = QSpinBox()
        self.sp_xfer_warm_seed.setRange(0, 2_147_483_647)
    
        self._warm_box = QGroupBox("Warm-start settings")
        self._warm_box.setObjectName("xferWarmBox")
    
        warm = QGridLayout(self._warm_box)
        warm.setContentsMargins(8, 8, 8, 8)
        warm.setHorizontalSpacing(10)
        warm.setVerticalSpacing(8)
    
        warm.addWidget(QLabel("Warm split:"), 0, 0)
        warm.addWidget(self.cmb_xfer_warm_split, 0, 1)
        warm.addWidget(QLabel("Warm epochs:"), 0, 2)
        warm.addWidget(self.sp_xfer_warm_epochs, 0, 3)
    
        warm.addWidget(QLabel("Warm samples:"), 1, 0)
        warm.addWidget(self.sp_xfer_warm_samples, 1, 1)
        warm.addWidget(QLabel("Warm lr:"), 1, 2)
        warm.addWidget(self.sp_xfer_warm_lr, 1, 3)
    
        warm.addWidget(QLabel("Warm frac:"), 2, 0)
        warm.addWidget(self.sp_xfer_warm_frac, 2, 1)
        warm.addWidget(QLabel("Warm seed:"), 2, 2)
        warm.addWidget(self.sp_xfer_warm_seed, 2, 3)
    
        warm.setColumnStretch(1, 1)
        warm.setColumnStretch(3, 1)
    
        f_warm = _field(
            "Warm-start",
            self._warm_box,
            "Settings used when warm-start is enabled.",
        )
        grid2.addWidget(f_warm, 1, 1)
    
        right.addWidget(s2)
    
        # Keep top alignment even if one column is shorter
        left.addStretch(1)
        right.addStretch(1)
    
        self._update_quantiles_chip()
        self._update_warm_enabled()
    
        return outer

    def eventFilter(self, obj, ev):
        if not hasattr(self, "_adv_scroll"):
            return super().eventFilter(obj, ev)
    
        vp = self._adv_scroll.viewport()
    
        if obj is vp and ev.type() == QEvent.Resize:
            self._update_adv_layout(vp.width())
    
        return super().eventFilter(obj, ev)
    
    
    def _update_adv_layout(self, w: int) -> None:
        if not hasattr(self, "_adv_cols"):
            return
    
        # Tune this breakpoint to taste.
        # ~980–1100 usually feels right.
        bp = 1020
    
        stacked = w < bp
    
        if stacked:
            self._adv_cols.setDirection(
                QBoxLayout.TopToBottom
            )
            if hasattr(self, "_adv_scroll"):
                self._adv_scroll.setHorizontalScrollBarPolicy(
                    Qt.ScrollBarAlwaysOff
                )
            return
    
        self._adv_cols.setDirection(
            QBoxLayout.LeftToRight
        )
        if hasattr(self, "_adv_scroll"):
            self._adv_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarAsNeeded
            )
            
    # -------------------------------------------------
    # Wiring
    # -------------------------------------------------
    def _connect_signals(self) -> None:
        self.btn_run_xfer.clicked.connect(self.run_clicked)
        self.btn_xfer_view.clicked.connect(self.view_clicked)

        self.xfer_results_root_btn.clicked.connect(
            self._on_browse_root
        )

        for w in (
            self.xfer_city_a,
            self.xfer_city_b,
            self.xfer_results_root,
            self.ed_xfer_quantiles,
        ):
            w.editingFinished.connect(self._push_to_store)

        for cb in (
            self.chk_xfer_split_train,
            self.chk_xfer_split_val,
            self.chk_xfer_split_test,
            self.chk_xfer_cal_none,
            self.chk_xfer_cal_source,
            self.chk_xfer_cal_target,
            self.chk_xfer_rescale,
            self.chk_xfer_json,
            self.chk_xfer_csv,
            self.chk_xfer_prefer_tuned,
            self.chk_xfer_phys_payload,
            self.chk_xfer_phys_csv,
            self.chk_xfer_eval_future,
            self.chk_xfer_strat_baseline,
            self.chk_xfer_strat_xfer,
            self.chk_xfer_strat_warm,
            self.chk_xfer_rmode_as_is,
            self.chk_xfer_rmode_strict,

        ):
            cb.toggled.connect(self._push_to_store)

        self.sp_xfer_batch.valueChanged.connect(
            self._push_to_store
        )
        self.sp_xfer_interval.valueChanged.connect(
            self._push_to_store
        )

        self.cmb_xfer_align.currentIndexChanged.connect(
            self._push_to_store
        )
        self.cmb_xfer_endpoint.currentIndexChanged.connect(
            self._push_to_store
        )
        self.cmb_xfer_view.currentIndexChanged.connect(
            self._push_to_store
        )
        self.cmb_xfer_view_split.currentIndexChanged.connect(
            self._push_to_store
        )
        self._mode_group.buttonClicked.connect(
            self._on_mode_changed
        )
        
        self.cmb_xfer_allow_re_dyn.currentIndexChanged.connect(
            self._push_to_store
        )
        self.cmb_xfer_allow_re_fut.currentIndexChanged.connect(
            self._push_to_store
        )
        
        self.ed_xfer_quantiles.editingFinished.connect(
            self._update_quantiles_chip
        )

        self.cmb_xfer_warm_split.currentIndexChanged.connect(
            self._push_to_store
        )
        self.sp_xfer_warm_samples.valueChanged.connect(
            self._push_to_store
        )
        self.sp_xfer_warm_frac.valueChanged.connect(
            self._push_to_store
        )
        self.sp_xfer_warm_epochs.valueChanged.connect(
            self._push_to_store
        )
        self.sp_xfer_warm_lr.valueChanged.connect(
            self._push_to_store
        )
        self.sp_xfer_warm_seed.valueChanged.connect(
            self._push_to_store
        )

        self.chk_xfer_strat_warm.toggled.connect(
            lambda _v: self._update_warm_enabled()
        )

        self.map_view.request_open_options.connect(
            self._show_options
        )
        self.map_view.request_fit.connect(self._on_fit_requested)

    def _on_fit_requested(self) -> None:
        if not self._coords_complete_from_store():
            self._show_options()
            return
        self._update_map_from_store()
        self.map_view.fit_to_cities()


    def _on_browse_root(self) -> None:
        start = self.xfer_results_root.text().strip()
        start = start or str(
            self._store.get("results_root", "")
        )
        root = QFileDialog.getExistingDirectory(
            self,
            "Select results root (Stage-1/2/xfer)",
            start,
        )
        if not root:
            return

        root_path = Path(root).expanduser()
        self._store.set("results_root", str(root_path))
        self.xfer_results_root.setText(str(root_path))

    def _apply_view_mode_from_store(self) -> None:
        mode = self._store.get("xfer.view_mode", "map")
        mode = (mode or "map").strip().lower()
        self._set_view_mode(mode)
    
    def _set_view_mode(self, mode: str) -> None:
        mode = (mode or "map").strip().lower()
        if mode == "options":
            self.btn_mode_opts.setChecked(True)
            self._stack.setCurrentIndex(1)
            return
    
        self.btn_mode_map.setChecked(True)
        self._stack.setCurrentIndex(0)
        
    def _on_mode_changed(self) -> None:
        idx = self._mode_group.checkedId()
        mode = "options" if idx == 1 else "map"
        self._store.set("xfer.view_mode", mode)
        self._set_view_mode(mode)
    
        if mode == "map":
            self._update_map_from_store()

    def _show_options(self) -> None:
        self._store.set("xfer.view_mode", "options")
        self._set_view_mode("options")

    # -------------------------------------------------
    # Store I/O
    # -------------------------------------------------
    def _on_store_changed(self, keys: object) -> None:
        try:
            changed = set(keys or [])
        except Exception:
            changed = set()
        if not changed:
            return
        self._sync_from_store(keys=changed)
        if changed & {
            "xfer.city_a",
            "xfer.city_b",
            "xfer.city_a_lat",
            "xfer.city_a_lon",
            "xfer.city_b_lat",
            "xfer.city_b_lon",
        }:
            self._update_map_from_store()

    # Your MapView.set_cities(...) signature depends on your
    # implementation. If it expects CityPoint objects, adapt
    # accordingly

    def _update_map_from_store(self) -> None:
        s = self._store
    
        a = (s.get("xfer.city_a", "") or "").strip()
        b = (s.get("xfer.city_b", "") or "").strip()
    
        lat_a = s.get("xfer.city_a_lat", None)
        lon_a = s.get("xfer.city_a_lon", None)
        lat_b = s.get("xfer.city_b_lat", None)
        lon_b = s.get("xfer.city_b_lon", None)
    
        # If coords are missing, just clear (MapView must be safe; see section 2)
        if None in (lat_a, lon_a, lat_b, lon_b):
            self.map_view.clear()
            return
    
        try:
            src = CityPoint(a or "City A", float(lat_a), float(lon_a))
            tgt = CityPoint(b or "City B", float(lat_b), float(lon_b))
            self.map_view.set_cities(src, tgt)
        except Exception:
            self.map_view.clear()

    def _sync_from_store(
        self,
        keys: Optional[set[str]] = None,
    ) -> None:
        s = self._store
        keys = keys or set()

        def wants(k: str) -> bool:
            return (not keys) or (k in keys)

        if wants("results_root"):
            v = s.get("results_root", "")
            if v:
                with _blocked(self.xfer_results_root):
                    self.xfer_results_root.setText(str(v))

        if wants("xfer.city_a"):
            v = s.get("xfer.city_a", "")
            if v:
                with _blocked(self.xfer_city_a):
                    self.xfer_city_a.setText(str(v))

        if wants("xfer.city_b"):
            v = s.get("xfer.city_b", "")
            if v:
                with _blocked(self.xfer_city_b):
                    self.xfer_city_b.setText(str(v))

        if wants("xfer.batch_size"):
            v = s.get("xfer.batch_size", None)
            if v is not None:
                with _blocked(self.sp_xfer_batch):
                    self.sp_xfer_batch.setValue(int(v))

        if wants("xfer.rescale_to_source"):
            v = bool(
                s.get("xfer.rescale_to_source", False)
            )
            with _blocked(self.chk_xfer_rescale):
                self.chk_xfer_rescale.setChecked(v)

        if wants("xfer.splits"):
            self._sync_splits(
                s.get("xfer.splits", ("val", "test"))
            )

        if wants("xfer.calib_modes"):
            self._sync_calib_modes(
                s.get(
                    "xfer.calib_modes",
                    ("none", "source", "target"),
                )
            )

        if wants("xfer.write_json"):
            v = bool(s.get("xfer.write_json", True))
            with _blocked(self.chk_xfer_json):
                self.chk_xfer_json.setChecked(v)

        if wants("xfer.write_csv"):
            v = bool(s.get("xfer.write_csv", True))
            with _blocked(self.chk_xfer_csv):
                self.chk_xfer_csv.setChecked(v)

        if wants("xfer.quantiles_override"):
            q = s.get("xfer.quantiles_override", None)
            if q:
                txt = ",".join(str(x) for x in q)
                with _blocked(self.ed_xfer_quantiles):
                    self.ed_xfer_quantiles.setText(txt)

        if wants("xfer.prefer_tuned"):
            v = bool(s.get("xfer.prefer_tuned", True))
            with _blocked(self.chk_xfer_prefer_tuned):
                self.chk_xfer_prefer_tuned.setChecked(v)

        if wants("xfer.align_policy"):
            v = s.get(
                "xfer.align_policy",
                "align_by_name_pad",
            )
            self._set_combo_data(self.cmb_xfer_align, v)

        if wants("xfer.interval_target"):
            v = float(s.get("xfer.interval_target", 0.80))
            with _blocked(self.sp_xfer_interval):
                self.sp_xfer_interval.setValue(v)

        if wants("xfer.load_endpoint"):
            v = s.get("xfer.load_endpoint", "serve")
            self._set_combo_data(self.cmb_xfer_endpoint, v)

        if wants("xfer.export_physics_payload"):
            v = bool(
                s.get("xfer.export_physics_payload", True)
            )
            with _blocked(self.chk_xfer_phys_payload):
                self.chk_xfer_phys_payload.setChecked(v)

        if wants("xfer.export_physical_parameters_csv"):
            v = bool(
                s.get(
                    "xfer.export_physical_parameters_csv",
                    True,
                )
            )
            with _blocked(self.chk_xfer_phys_csv):
                self.chk_xfer_phys_csv.setChecked(v)

        if wants("xfer.write_eval_future_csv"):
            v = bool(
                s.get("xfer.write_eval_future_csv", True)
            )
            with _blocked(self.chk_xfer_eval_future):
                self.chk_xfer_eval_future.setChecked(v)

        if wants("xfer.view_kind"):
            v = s.get("xfer.view_kind", "calib_panel")
            self._set_combo_data(self.cmb_xfer_view, v)

        if wants("xfer.view_split"):
            v = s.get("xfer.view_split", "val")
            self._set_combo_data(self.cmb_xfer_view_split, v)
            
        if wants("xfer.view_mode"):
            self._apply_view_mode_from_store()


        if wants("xfer.strategies"):
            self._sync_strategies(
                s.get("xfer.strategies", None)
            )

        if wants("xfer.rescale_modes"):
            self._sync_rescale_modes(
                s.get("xfer.rescale_modes", None)
            )

        if wants("xfer.allow_reorder_dynamic"):
            v = s.get("xfer.allow_reorder_dynamic", None)
            self._set_opt_bool_combo(self.cmb_xfer_allow_re_dyn, v)
        
        if wants("xfer.allow_reorder_future"):
            v = s.get("xfer.allow_reorder_future", None)
            self._set_opt_bool_combo(self.cmb_xfer_allow_re_fut, v)

        if wants("xfer.warm_split"):
            v = s.get("xfer.warm_split", "train")
            self._set_combo_data(self.cmb_xfer_warm_split, v)

        if wants("xfer.warm_samples"):
            v = s.get("xfer.warm_samples", None)
            if v is not None:
                with _blocked(self.sp_xfer_warm_samples):
                    self.sp_xfer_warm_samples.setValue(int(v))

        if wants("xfer.warm_frac"):
            v = s.get("xfer.warm_frac", None)
            if v is not None:
                with _blocked(self.sp_xfer_warm_frac):
                    self.sp_xfer_warm_frac.setValue(float(v))

        if wants("xfer.warm_epochs"):
            v = s.get("xfer.warm_epochs", None)
            if v is not None:
                with _blocked(self.sp_xfer_warm_epochs):
                    self.sp_xfer_warm_epochs.setValue(int(v))

        if wants("xfer.warm_lr"):
            v = s.get("xfer.warm_lr", None)
            if v is not None:
                with _blocked(self.sp_xfer_warm_lr):
                    self.sp_xfer_warm_lr.setValue(float(v))

        if wants("xfer.warm_seed"):
            v = s.get("xfer.warm_seed", None)
            if v is not None:
                with _blocked(self.sp_xfer_warm_seed):
                    self.sp_xfer_warm_seed.setValue(int(v))


        self._update_quantiles_chip()
        
        # keep warm controls enabled/disabled consistent
        self._update_warm_enabled()

    def _push_to_store(self) -> None:
        s = self._store

        s.set(
            "results_root",
            self.xfer_results_root.text().strip(),
        )
        s.set(
            "xfer.city_a",
            self.xfer_city_a.text().strip(),
        )
        s.set(
            "xfer.city_b",
            self.xfer_city_b.text().strip(),
        )

        s.set("xfer.splits", self._get_checked_splits())
        s.set("xfer.calib_modes", self._get_checked_calib())

        s.set(
            "xfer.batch_size",
            int(self.sp_xfer_batch.value()),
        )
        s.set(
            "xfer.rescale_to_source",
            bool(self.chk_xfer_rescale.isChecked()),
        )

        s.set(
            "xfer.write_json",
            bool(self.chk_xfer_json.isChecked()),
        )
        s.set(
            "xfer.write_csv",
            bool(self.chk_xfer_csv.isChecked()),
        )

        # s.set(
        #     "xfer.quantiles_override",
        #     self.get_quantiles(),
        # )
        q, ok, _txt = self._quantiles_ui_status()
        raw = (self.ed_xfer_quantiles.text() or "").strip()
        if ok or (not raw):
            s.set("xfer.quantiles_override", q)

        s.set(
            "xfer.prefer_tuned",
            bool(self.chk_xfer_prefer_tuned.isChecked()),
        )

        s.set(
            "xfer.align_policy",
            str(self.cmb_xfer_align.currentData()),
        )
        s.set(
            "xfer.interval_target",
            float(self.sp_xfer_interval.value()),
        )
        s.set(
            "xfer.load_endpoint",
            str(self.cmb_xfer_endpoint.currentData()),
        )

        s.set(
            "xfer.export_physics_payload",
            bool(self.chk_xfer_phys_payload.isChecked()),
        )
        s.set(
            "xfer.export_physical_parameters_csv",
            bool(self.chk_xfer_phys_csv.isChecked()),
        )
        s.set(
            "xfer.write_eval_future_csv",
            bool(self.chk_xfer_eval_future.isChecked()),
        )

        s.set(
            "xfer.view_kind",
            str(self.cmb_xfer_view.currentData()),
        )
        s.set(
            "xfer.view_split",
            str(self.cmb_xfer_view_split.currentData()),
        )
        mode = "map"
        if self._stack.currentIndex() == 1:
            mode = "options"
            
        s.set("xfer.view_mode", mode)

        s.set("xfer.strategies", self._get_checked_strategies())
        s.set("xfer.rescale_modes", self._get_checked_rescale_modes())

        s.set(
            "xfer.allow_reorder_dynamic",
            self._opt_bool_from_combo(self.cmb_xfer_allow_re_dyn),
        )
        s.set(
            "xfer.allow_reorder_future",
            self._opt_bool_from_combo(self.cmb_xfer_allow_re_fut),
        )

        s.set(
            "xfer.warm_split",
            str(self.cmb_xfer_warm_split.currentData()),
        )
        s.set(
            "xfer.warm_samples",
            int(self.sp_xfer_warm_samples.value()),
        )
        s.set(
            "xfer.warm_frac",
            float(self.sp_xfer_warm_frac.value()),
        )
        s.set(
            "xfer.warm_epochs",
            int(self.sp_xfer_warm_epochs.value()),
        )
        s.set(
            "xfer.warm_lr",
            float(self.sp_xfer_warm_lr.value()),
        )
        s.set(
            "xfer.warm_seed",
            int(self.sp_xfer_warm_seed.value()),
        )

    # -------------------------------------------------
    # Small helpers
    # -------------------------------------------------
    def _get_checked_splits(self) -> List[str]:
        out: List[str] = []
        if self.chk_xfer_split_train.isChecked():
            out.append("train")
        if self.chk_xfer_split_val.isChecked():
            out.append("val")
        if self.chk_xfer_split_test.isChecked():
            out.append("test")
        return out

    def _sync_splits(self, splits: Sequence[str]) -> None:
        s = set(splits or [])
        with _blocked(self.chk_xfer_split_train):
            self.chk_xfer_split_train.setChecked("train" in s)
        with _blocked(self.chk_xfer_split_val):
            self.chk_xfer_split_val.setChecked("val" in s)
        with _blocked(self.chk_xfer_split_test):
            self.chk_xfer_split_test.setChecked("test" in s)

    def _get_checked_calib(self) -> List[str]:
        out: List[str] = []
        if self.chk_xfer_cal_none.isChecked():
            out.append("none")
        if self.chk_xfer_cal_source.isChecked():
            out.append("source")
        if self.chk_xfer_cal_target.isChecked():
            out.append("target")
        return out

    def _sync_calib_modes(self, modes: Sequence[str]) -> None:
        m = set(modes or [])
        with _blocked(self.chk_xfer_cal_none):
            self.chk_xfer_cal_none.setChecked("none" in m)
        with _blocked(self.chk_xfer_cal_source):
            self.chk_xfer_cal_source.setChecked("source" in m)
        with _blocked(self.chk_xfer_cal_target):
            self.chk_xfer_cal_target.setChecked("target" in m)

    def _set_combo_data(
        self,
        cmb: QComboBox,
        data: Any,
    ) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == data:
                with _blocked(cmb):
                    cmb.setCurrentIndex(i)
                return

    def set_has_result(self, has: bool) -> None:
        self.btn_xfer_view.setVisible(bool(has))
        self.btn_xfer_view.setEnabled(bool(has))
        
    def _as_float(self, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
        try:
            return float(v)
        except Exception:
            return None

    def _coords_complete_from_store(self) -> bool:
        """
        Return True if both cities have valid lat/lon in the store.
        """
        s = self._store

        lat_a = self._as_float(s.get("xfer.city_a_lat", None))
        lon_a = self._as_float(s.get("xfer.city_a_lon", None))
        lat_b = self._as_float(s.get("xfer.city_b_lat", None))
        lon_b = self._as_float(s.get("xfer.city_b_lon", None))

        if None in (lat_a, lon_a, lat_b, lon_b):
            return False

        # basic geographic bounds
        if not (-90.0 <= lat_a <= 90.0 and -180.0 <= lon_a <= 180.0):
            return False
        if not (-90.0 <= lat_b <= 90.0 and -180.0 <= lon_b <= 180.0):
            return False

        return True

    def _make_help_btn(self, tip: str) -> QToolButton:
        b = QToolButton()
        b.setObjectName("miniAction")
        b.setAutoRaise(True)
        b.setIcon(
            self.style().standardIcon(
                QStyle.SP_MessageBoxInformation
            )
        )
        b.setToolTip(tip)
        b.setCursor(Qt.WhatsThisCursor)
        b.setFixedSize(22, 22)
        return b
    
    
    def _fill_opt_bool_combo(self, cmb: QComboBox) -> None:
        cmb.clear()
        cmb.addItem("Auto", None)
        cmb.addItem("Allow", True)
        cmb.addItem("Block", False)
    
    
    def _opt_bool_from_combo(
        self,
        cmb: QComboBox,
    ) -> Optional[bool]:
        return cmb.currentData()
    
    
    def _set_opt_bool_combo(
        self,
        cmb: QComboBox,
        v: Optional[bool],
    ) -> None:
        for i in range(cmb.count()):
            if cmb.itemData(i) == v:
                with _blocked(cmb):
                    cmb.setCurrentIndex(i)
                return
    
    
    def _quantiles_ui_status(
        self,
    ) -> tuple[Optional[List[float]], bool, str]:
        raw = (self.ed_xfer_quantiles.text() or "").strip()
        if not raw:
            return None, True, "AUTO"
    
        q = _parse_float_list(raw)
        if q is None:
            return None, False, "INVALID"
    
        return list(q), True, "OK"
    
    
    def _update_quantiles_chip(self) -> None:
        _q, ok, txt = self._quantiles_ui_status()
        self.lbl_xfer_q_chip.setText(txt)
        self.lbl_xfer_q_chip.setProperty("ok", bool(ok))
        self.lbl_xfer_q_chip.style().unpolish(self.lbl_xfer_q_chip)
        self.lbl_xfer_q_chip.style().polish(self.lbl_xfer_q_chip)
    
    
    def _on_adv_reset(self) -> None:
        s = self._store
        with s.batch():
            s.set("xfer.quantiles_override", None)
            s.set("xfer.write_json", True)
            s.set("xfer.write_csv", True)
            s.set("xfer.prefer_tuned", True)
            s.set("xfer.align_policy", "align_by_name_pad")
            s.set("xfer.allow_reorder_dynamic", None)
            s.set("xfer.allow_reorder_future", None)
            s.set("xfer.interval_target", 0.80)
            s.set("xfer.load_endpoint", "serve")
            s.set("xfer.export_physics_payload", True)
            s.set("xfer.export_physical_parameters_csv", True)
            s.set("xfer.write_eval_future_csv", True)
            s.set("xfer.rescale_modes", None)
            s.set("xfer.strategies", None)
            s.set("xfer.warm_split", "train")
            s.set("xfer.warm_samples", 20000)
            s.set("xfer.warm_frac", 0.0)
            s.set("xfer.warm_epochs", 3)
            s.set("xfer.warm_lr", 1e-4)
            s.set("xfer.warm_seed", 0)

    # -------------------------------------------------
    # New helpers (v3.2)
    # -------------------------------------------------
    def _opt_bool_from_cb(
        self,
        cb: QCheckBox,
    ) -> Optional[bool]:
        st = cb.checkState()
        if st == Qt.PartiallyChecked:
            return None
        return st == Qt.Checked

    def _set_cb_from_opt_bool(
        self,
        cb: QCheckBox,
        v: Optional[bool],
    ) -> None:
        cb.setTristate(True)
        if v is None:
            cb.setCheckState(Qt.PartiallyChecked)
            return
        cb.setCheckState(
            Qt.Checked if v else Qt.Unchecked
        )

    def _get_checked_strategies(self) -> Optional[List[str]]:
        out: List[str] = []
        if self.chk_xfer_strat_baseline.isChecked():
            out.append("baseline")
        if self.chk_xfer_strat_xfer.isChecked():
            out.append("xfer")
        if self.chk_xfer_strat_warm.isChecked():
            out.append("warm")
        return out or None

    def _sync_strategies(self, v: Any) -> None:
        s = set(v or [])
        with _blocked(self.chk_xfer_strat_baseline):
            self.chk_xfer_strat_baseline.setChecked(
                "baseline" in s
            )
        with _blocked(self.chk_xfer_strat_xfer):
            self.chk_xfer_strat_xfer.setChecked(
                "xfer" in s
            )
        with _blocked(self.chk_xfer_strat_warm):
            self.chk_xfer_strat_warm.setChecked(
                "warm" in s
            )
        self._update_warm_enabled()

    def _get_checked_rescale_modes(self) -> Optional[List[str]]:
        out: List[str] = []
        if self.chk_xfer_rmode_as_is.isChecked():
            out.append("as_is")
        if self.chk_xfer_rmode_strict.isChecked():
            out.append("strict")
        return out or None

    def _sync_rescale_modes(self, v: Any) -> None:
        s = set(v or [])
        with _blocked(self.chk_xfer_rmode_as_is):
            self.chk_xfer_rmode_as_is.setChecked(
                "as_is" in s
            )
        with _blocked(self.chk_xfer_rmode_strict):
            self.chk_xfer_rmode_strict.setChecked(
                "strict" in s
            )

    def _update_warm_enabled(self) -> None:
        on = bool(self.chk_xfer_strat_warm.isChecked())

        if hasattr(self, "_warm_box"):
            self._warm_box.setEnabled(on)

        for w in (
            self.cmb_xfer_warm_split,
            self.sp_xfer_warm_samples,
            self.sp_xfer_warm_frac,
            self.sp_xfer_warm_epochs,
            self.sp_xfer_warm_lr,
            self.sp_xfer_warm_seed,
        ):
            w.setEnabled(on)

