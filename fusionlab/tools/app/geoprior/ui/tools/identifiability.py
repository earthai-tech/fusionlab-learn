# geoprior/ui/tools/identifiability.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QPlainTextEdit,
    QMessageBox,
    QStackedWidget,
    QDialog,
    QDialogButtonBox,
    QSpinBox,
    QFormLayout,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)

from .identifiability_core import (
    run_identifiability_from_npz,
    try_truth_prior_in_units,
)
from .identifiability_viz import (
    make_payload_timescale_figure,
)


@dataclass
class IdentState:
    mode: str = "workflow"
    show_toolbar: bool = False
    source_path: str = ""
    report_units: str = "auto"
    kappa_mode: str = "fixed"
    kappa_b: float = 1.0
    show_prior: bool = False
    out_dir: str = ""
    out_fmt: str = "png"

@dataclass
class SM3SandboxConfig:
    n_realizations: int = 30
    n_years: int = 20
    time_steps: int = 5
    val_tail: int = 5
    seed: int = 123

    epochs: int = 50
    batch: int = 16
    lr: float = 1e-3

    noise_std: float = 0.02
    load_type: str = "step"
    alpha: float = 1000.0

    hd_factor: float = 0.6
    thickness_cap: float = 30.0
    kappa_b: float = 1.0
    gamma_w: float = 9810.0

    tau_min: float = 0.3
    tau_max: float = 10.0
    tau_spread_dex: float = 0.3
    Ss_spread_dex: float = 0.4


class SM3SandboxDialog(QDialog):
    def __init__(
        self,
        *,
        defaults: SM3SandboxConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SM3 Sandbox setup")
        self._cfg = SM3SandboxConfig(**defaults.__dict__)
        self._build_ui()

    @property
    def cfg(self) -> SM3SandboxConfig:
        return self._cfg

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)

        def _spin(v: int, lo: int, hi: int) -> QSpinBox:
            sp = QSpinBox(self)
            sp.setRange(lo, hi)
            sp.setValue(int(v))
            return sp

        def _dspin(v: float, lo: float, hi: float,
                   dec: int = 6) -> QDoubleSpinBox:
            sp = QDoubleSpinBox(self)
            sp.setDecimals(dec)
            sp.setRange(lo, hi)
            sp.setValue(float(v))
            return sp

        self.sp_nreal = _spin(self._cfg.n_realizations, 1, 9999)
        self.sp_nyrs = _spin(self._cfg.n_years, 5, 500)
        self.sp_tsteps = _spin(self._cfg.time_steps, 1, 200)
        self.sp_vtail = _spin(self._cfg.val_tail, 1, 200)
        self.sp_seed = _spin(self._cfg.seed, 0, 10**9)

        self.sp_epochs = _spin(self._cfg.epochs, 1, 5000)
        self.sp_batch = _spin(self._cfg.batch, 1, 2048)
        self.sp_lr = _dspin(self._cfg.lr, 1e-7, 10.0, dec=8)

        self.sp_noise = _dspin(self._cfg.noise_std, 0.0, 10.0, dec=6)

        self.cmb_load = QComboBox(self)
        self.cmb_load.addItems(["step", "ramp"])
        self.cmb_load.setCurrentText(self._cfg.load_type)

        self.sp_alpha = _dspin(self._cfg.alpha, 0.0, 1e9, dec=3)

        self.sp_hd = _dspin(self._cfg.hd_factor, 0.0, 10.0, dec=6)
        self.sp_cap = _dspin(self._cfg.thickness_cap, 0.0, 1e6, dec=3)
        self.sp_kappa = _dspin(self._cfg.kappa_b, 1e-12, 1e12, dec=6)
        self.sp_gamma = _dspin(self._cfg.gamma_w, 0.0, 1e9, dec=3)

        self.sp_tmin = _dspin(self._cfg.tau_min, 1e-6, 1e6, dec=6)
        self.sp_tmax = _dspin(self._cfg.tau_max, 1e-6, 1e6, dec=6)
        self.sp_tspr = _dspin(self._cfg.tau_spread_dex, 0.0, 10.0, dec=6)
        self.sp_sspr = _dspin(self._cfg.Ss_spread_dex, 0.0, 10.0, dec=6)

        form.addRow("Number of realizations", self.sp_nreal)
        form.addRow("Number of years", self.sp_nyrs)
        form.addRow("Time steps (history length)", self.sp_tsteps)
        form.addRow("Validation tail (samples)", self.sp_vtail)
        form.addRow("Random seed", self.sp_seed)

        form.addRow("Epochs", self.sp_epochs)
        form.addRow("Batch size", self.sp_batch)
        form.addRow("Learning rate", self.sp_lr)

        form.addRow("Noise (std)", self.sp_noise)
        form.addRow("Load type", self.cmb_load)
        form.addRow("Alpha (scale factor)", self.sp_alpha)

        form.addRow("Hd factor", self.sp_hd)
        form.addRow("Thickness cap", self.sp_cap)
        form.addRow("Kappa (kappa_b)", self.sp_kappa)
        form.addRow("Gamma_w", self.sp_gamma)

        form.addRow("Tau min", self.sp_tmin)
        form.addRow("Tau max", self.sp_tmax)
        form.addRow("Tau spread (dex)", self.sp_tspr)
        form.addRow("Ss spread (dex)", self.sp_sspr)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def accept(self) -> None:
        self._cfg.n_realizations = int(self.sp_nreal.value())
        self._cfg.n_years = int(self.sp_nyrs.value())
        self._cfg.time_steps = int(self.sp_tsteps.value())
        self._cfg.val_tail = int(self.sp_vtail.value())
        self._cfg.seed = int(self.sp_seed.value())

        self._cfg.epochs = int(self.sp_epochs.value())
        self._cfg.batch = int(self.sp_batch.value())
        self._cfg.lr = float(self.sp_lr.value())

        self._cfg.noise_std = float(self.sp_noise.value())
        self._cfg.load_type = str(self.cmb_load.currentText())
        self._cfg.alpha = float(self.sp_alpha.value())

        self._cfg.hd_factor = float(self.sp_hd.value())
        self._cfg.thickness_cap = float(self.sp_cap.value())
        self._cfg.kappa_b = float(self.sp_kappa.value())
        self._cfg.gamma_w = float(self.sp_gamma.value())

        self._cfg.tau_min = float(self.sp_tmin.value())
        self._cfg.tau_max = float(self.sp_tmax.value())
        self._cfg.tau_spread_dex = float(self.sp_tspr.value())
        self._cfg.Ss_spread_dex = float(self.sp_sspr.value())

        super().accept()

class IdentifiabilityTool(QWidget):
    """
    Tools tab: Identifiability runner.

    v3.2:
    - infer payload units robustly (default SI/sec)
    - convert BEFORE diagnostics/plots
    - store prefs under ident.*
    """

    def __init__(
        self,
        *,
        app_ctx: Optional[object] = None,
        store: Optional[object] = None,
        geo_cfg: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_ctx = app_ctx
        self._store = store
        self._geo_cfg = geo_cfg
        self._state = IdentState()
        self._sandbox_cfg: Optional[SM3SandboxConfig] = None
        
        self._fig = None
        self._canvas = FigureCanvas()
        self._toolbar = NavigationToolbar2QT(
            self._canvas,
            self,
        )
        self._toolbar.setVisible(False)

        self._build_ui()
        self._load_from_store()
        self._sync_ui_from_state()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        hdr = QHBoxLayout()
        title = QLabel("Identifiability (SM3)")
        title.setStyleSheet("font-weight: 600;")

        self._cmb_mode = QComboBox(self)
        self._cmb_mode.addItems(
            [
                "Workflow (payload diagnostics)",
                "Sandbox (SM3 synthetic)",
            ]
        )
        self._cmb_mode.currentIndexChanged.connect(
            self._on_mode_changed
        )

        self._btn_tools = QPushButton("Plot tools", self)
        self._btn_tools.setCheckable(True)
        self._btn_tools.toggled.connect(
            self._on_plot_tools_toggled
        )

        self._status = QLabel("No input")
        self._status.setStyleSheet(
            "padding:2px 8px;"
            "border-radius:10px;"
            "background: palette(midlight);"
        )

        hdr.addWidget(title)
        hdr.addStretch(1)
        hdr.addWidget(self._cmb_mode)
        hdr.addWidget(self._btn_tools)
        hdr.addWidget(self._status)
        root.addLayout(hdr)

        body = QHBoxLayout()
        body.setSpacing(10)
        root.addLayout(body, 1)

        # Left controls
        # Left controls
        left = QVBoxLayout()
        left.setSpacing(8)
        body.addLayout(left, 0)

        self._mode_stack = QStackedWidget(self)

        left.addWidget(self._mode_stack, 1)

        # -------------------------
        # Workflow page (existing)
        # -------------------------
        wf_page = QWidget(self)
        wf_lay = QVBoxLayout(wf_page)
        wf_lay.setContentsMargins(0, 0, 0, 0)
        wf_lay.setSpacing(8)

        box_src = QGroupBox("Payload (NPZ)")
        wf_lay.addWidget(box_src)
        src = QVBoxLayout(box_src)

        row0 = QHBoxLayout()
        self._edit_path = QLineEdit()
        self._edit_path.setPlaceholderText(
            "Select phys_payload_*.npz ..."
        )
        self._btn_browse = QPushButton("Browse…")
        row0.addWidget(self._edit_path, 1)
        row0.addWidget(self._btn_browse)
        src.addLayout(row0)

        box_opt = QGroupBox("Units & closure")
        wf_lay.addWidget(box_opt)
        opt = QVBoxLayout(box_opt)

        row1 = QHBoxLayout()
        self._cmb_units = QComboBox()
        self._cmb_units.addItems(["auto", "year", "sec"])
        row1.addWidget(QLabel("Report units"))
        row1.addWidget(self._cmb_units, 1)
        opt.addLayout(row1)

        row2 = QHBoxLayout()
        self._cmb_kappa = QComboBox()
        self._cmb_kappa.addItems(["fixed"])
        row2.addWidget(QLabel("kappa"))
        row2.addWidget(self._cmb_kappa, 1)
        opt.addLayout(row2)

        row3 = QHBoxLayout()
        self._sp_kappa = QDoubleSpinBox()
        self._sp_kappa.setRange(1e-12, 1e12)
        self._sp_kappa.setDecimals(6)
        self._sp_kappa.setSingleStep(0.1)
        row3.addWidget(QLabel("kappa_b"))
        row3.addWidget(self._sp_kappa, 1)
        opt.addLayout(row3)

        self._chk_prior = QCheckBox("Show prior")
        opt.addWidget(self._chk_prior)

        wf_lay.addStretch(1)

        # -------------------------
        # Sandbox page (new)
        # -------------------------
        sb_page = QWidget(self)
        sb_lay = QVBoxLayout(sb_page)
        sb_lay.setContentsMargins(0, 0, 0, 0)
        sb_lay.setSpacing(8)

        box_sb = QGroupBox("Sandbox (SM3 synthetic)")
        sb_lay.addWidget(box_sb)
        sb = QVBoxLayout(box_sb)

        self._lbl_sb = QLabel(
            "No experiment config.\n"
            "Click Setup… then Run.",
            self,
        )
        self._lbl_sb.setWordWrap(True)
        self._lbl_sb.setStyleSheet("color: palette(mid);")
        sb.addWidget(self._lbl_sb)

        row_sb = QHBoxLayout()
        self._btn_sb_setup = QPushButton("Setup…", self)
        self._chk_sb_csv = QCheckBox("Export CSV", self)
        row_sb.addWidget(self._btn_sb_setup)
        row_sb.addStretch(1)
        row_sb.addWidget(self._chk_sb_csv)
        sb.addLayout(row_sb)

        sb_lay.addStretch(1)

        # Stack pages: 0=workflow, 1=sandbox
        self._mode_stack.addWidget(wf_page)
        self._mode_stack.addWidget(sb_page)

        # -------------------------
        # Output (common)
        # -------------------------
        box_out = QGroupBox("Output")
        left.addWidget(box_out)
        out = QVBoxLayout(box_out)

        row4 = QHBoxLayout()
        self._edit_out = QLineEdit()
        self._btn_out = QPushButton("Dir…")
        row4.addWidget(self._edit_out, 1)
        row4.addWidget(self._btn_out)
        out.addLayout(row4)

        row5 = QHBoxLayout()
        self._cmb_fmt = QComboBox()
        self._cmb_fmt.addItems(["png", "pdf"])
        self._btn_run = QPushButton("Run")
        self._btn_save = QPushButton("Save fig")
        row5.addWidget(self._cmb_fmt, 1)
        row5.addWidget(self._btn_run)
        row5.addWidget(self._btn_save)
        out.addLayout(row5)

        left.addStretch(1)

        # Right preview + log
        right = QVBoxLayout()
        right.setSpacing(8)
        body.addLayout(right, 1)

        prev_box = QGroupBox("Preview")
        right.addWidget(prev_box, 1)
        prev = QVBoxLayout(prev_box)
        prev.addWidget(self._toolbar)
        prev.addWidget(self._canvas, 1)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setPlaceholderText("Logs…")
        right.addWidget(self._log, 0)

        # signals
        self._btn_browse.clicked.connect(self._on_browse)
        self._btn_out.clicked.connect(self._on_pick_out)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_sb_setup.clicked.connect(
            self._on_sandbox_setup
        )

        self._edit_path.textChanged.connect(self._on_any)
        self._cmb_units.currentTextChanged.connect(self._on_any)
        self._cmb_kappa.currentTextChanged.connect(self._on_any)
        self._sp_kappa.valueChanged.connect(self._on_any)
        self._chk_prior.toggled.connect(self._on_any)
        self._edit_out.textChanged.connect(self._on_any)
        self._cmb_fmt.currentTextChanged.connect(self._on_any)

    # -------------------------
    # Store sync
    # -------------------------
    def _load_from_store(self) -> None:
        if self._store is None:
            return

        g = self._store.get
        st = self._state
        
        st.mode = str(g("ident.mode", "workflow"))
        st.show_toolbar = bool(
            g("ident.show_toolbar", False)
        )

        st.source_path = str(g("ident.source_path", ""))
        st.report_units = str(g("ident.report_units", "auto"))
        st.kappa_mode = str(g("ident.kappa_mode", "fixed"))
        st.kappa_b = float(g("ident.kappa_b", 1.0))
        st.show_prior = bool(g("ident.show_prior", False))
        st.out_dir = str(g("ident.out_dir", ""))
        st.out_fmt = str(g("ident.out_fmt", "png"))

        if not st.out_dir:
            rr = str(g("results_root", ""))
            if rr:
                st.out_dir = str(Path(rr) / "figs")
                
        # ensure mode defaults match UI
        if st.mode not in ("workflow", "sandbox"):
            st.mode = "workflow"

    def _persist_common_to_store(self) -> None:
        if self._store is None:
            return

        s = self._store.set
        st = self._state
        s("ident.mode", st.mode)
        s("ident.out_dir", st.out_dir)
        s("ident.out_fmt", st.out_fmt)
        s("ident.show_toolbar", bool(st.show_toolbar))

    def _persist_workflow_to_store(self) -> None:
        if self._store is None:
            return

        s = self._store.set
        st = self._state
        s("ident.source_path", st.source_path)
        s("ident.report_units", st.report_units)
        s("ident.kappa_mode", st.kappa_mode)
        s("ident.kappa_b", float(st.kappa_b))
        s("ident.show_prior", bool(st.show_prior))

    def _sync_ui_from_state(self) -> None:
        st = self._state

        with QSignalBlocker(self._cmb_mode):
            self._cmb_mode.setCurrentIndex(
                1 if st.mode == "sandbox" else 0
            )

        with QSignalBlocker(self._btn_tools):
            self._btn_tools.setChecked(bool(st.show_toolbar))

        self._toolbar.setVisible(bool(st.show_toolbar))
        self._mode_stack.setCurrentIndex(
            1 if st.mode == "sandbox" else 0
        )
        
        self._btn_tools.setChecked(bool(st.show_toolbar))
        self._toolbar.setVisible(bool(st.show_toolbar))
        self._mode_stack.setCurrentIndex(
            1 if st.mode == "sandbox" else 0
        )

        # Workflow widgets
        self._edit_path.setText(st.source_path)
        self._cmb_units.setCurrentText(st.report_units)
        self._cmb_kappa.setCurrentText(st.kappa_mode)
        self._sp_kappa.setValue(float(st.kappa_b))
        self._chk_prior.setChecked(bool(st.show_prior))

        # Output widgets (common)
        self._edit_out.setText(st.out_dir)
        self._cmb_fmt.setCurrentText(st.out_fmt)


    # -------------------------
    # Slots
    # -------------------------

    def _on_mode_changed(self) -> None:
        idx = int(self._cmb_mode.currentIndex())
        self._state.mode = "sandbox" if idx == 1 else "workflow"
        self._mode_stack.setCurrentIndex(idx)
        self._persist_common_to_store()

        if self._state.mode == "sandbox":
            if self._sandbox_cfg is None:
                self._status.setText("Needs setup")
            else:
                self._status.setText("Ready")

    def _on_plot_tools_toggled(self, on: bool) -> None:
        self._state.show_toolbar = bool(on)
        self._toolbar.setVisible(bool(on))
        self._persist_common_to_store()

    def _on_sandbox_setup(self) -> None:
        # Prefill from store (read-only usage), then keep
        # only in memory (never store.set()).
        defaults = SM3SandboxConfig()

        dlg = SM3SandboxDialog(defaults=defaults, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return

        self._sandbox_cfg = dlg.cfg
        self._lbl_sb.setText(
            "Configured:\n"
            f"- n_realizations={self._sandbox_cfg.n_realizations}\n"
            f"- n_years={self._sandbox_cfg.n_years}\n"
            f"- time_steps={self._sandbox_cfg.time_steps}\n"
            f"- epochs={self._sandbox_cfg.epochs}"
        )
        self._status.setText("Ready")
        self._log.appendPlainText("[sandbox] setup updated")

    def _on_any(self) -> None:
        st = self._state

        # Workflow values (only meaningful in workflow)
        st.source_path = self._edit_path.text().strip()
        st.report_units = self._cmb_units.currentText()
        st.kappa_mode = self._cmb_kappa.currentText()
        st.kappa_b = float(self._sp_kappa.value())
        st.show_prior = bool(self._chk_prior.isChecked())

        # Common output
        st.out_dir = self._edit_out.text().strip()
        st.out_fmt = self._cmb_fmt.currentText()

        # Persist:
        self._persist_common_to_store()
        if st.mode == "workflow":
            self._persist_workflow_to_store()

    def _on_browse(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select physics payload",
            "",
            "NPZ (*.npz);;All (*.*)",
        )
        if p:
            self._edit_path.setText(p)

    def _on_pick_out(self) -> None:
        p = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            "",
        )
        if p:
            self._edit_out.setText(p)

    def _on_run(self) -> None:
        if self._state.mode == "sandbox":
            self._run_sandbox_sm3()
            return
        self._run_workflow_payload()

    def _run_workflow_payload(self) -> None:
        path = self._edit_path.text().strip()
        if not path:
            QMessageBox.warning(
                self,
                "Identifiability",
                "Select a payload NPZ first.",
            )
            return
        if not os.path.exists(path):
            QMessageBox.warning(
                self,
                "Identifiability",
                "File not found:\n\n" + path,
            )
            return

        self._status.setText("Running…")
        self._log.appendPlainText(f"[load] {path}")

        bundle, diag = run_identifiability_from_npz(
            path,
            ui_report_units=self._cmb_units.currentText(),
        )

        tp = try_truth_prior_in_units(
            bundle.meta,
            report_units=bundle.report_units,
            sec_per_year=bundle.sec_per_year,
        )
        tau_true = None if tp is None else tp["tau_true"]

        self._fig = make_payload_timescale_figure(
            bundle.payload,
            report_units=bundle.report_units,
            kappa_b=float(self._sp_kappa.value()),
            tau_true=tau_true,
        )
        self._canvas.figure = self._fig
        self._canvas.draw_idle()

        self._log.appendPlainText(
            f"[units] payload={bundle.payload_units} "
            f"report={bundle.report_units} "
            f"({bundle.units_reason})"
        )

        if diag is None:
            self._status.setText("OK (no truth)")
            self._log.appendPlainText(
                "[diag] SM3 truth/prior not found "
                "(self-consistency only)."
            )
            return

        q50 = diag["tau_rel_error"].get("q50", float("nan"))
        q95 = diag["tau_rel_error"].get("q95", float("nan"))
        self._status.setText("OK")
        self._log.appendPlainText(
            f"[diag] tau_rel q50={q50:.4g} "
            f"q95={q95:.4g}"
        )
        
    def _run_sandbox_sm3(self) -> None:
        if self._sandbox_cfg is None:
            QMessageBox.warning(
                self,
                "Sandbox",
                "Click Setup… first to configure the "
                "SM3 sandbox experiment.",
            )
            self._status.setText("Needs setup")
            return

        self._status.setText("Sandbox (TODO)")
        self._log.appendPlainText(
            "[sandbox] runner not wired yet "
            "(we implement in F+)."
        )

    def _on_save(self) -> None:
        if self._fig is None:
            QMessageBox.warning(
                self,
                "Identifiability",
                "Run first to generate a figure.",
            )
            return

        out_dir = self._edit_out.text().strip() or "."
        os.makedirs(out_dir, exist_ok=True)

        fmt = self._cmb_fmt.currentText().strip().lower()
        out = os.path.join(out_dir, f"ident_preview.{fmt}")

        self._fig.savefig(out, bbox_inches="tight")
        self._log.appendPlainText(f"[save] {out}")
        self._status.setText("Saved")
