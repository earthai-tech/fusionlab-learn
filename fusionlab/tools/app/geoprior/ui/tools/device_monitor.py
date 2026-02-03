# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
    QFileDialog,
    QApplication,
    QHeaderView,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure


try:
    from numpy.linalg import LinAlgError
except Exception:  # pragma: no cover
    LinAlgError = Exception  # type: ignore


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return "?"


class _Chip(QLabel):
    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self.setObjectName("chip")
        self.setMinimumHeight(22)
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setStyleSheet(
            "padding:2px 10px;"
            "border-radius:11px;"
            "background: palette(midlight);"
            "color: palette(text);"
        )


@dataclass
class _GpuRow:
    idx: int
    name: str
    util: Optional[float] = None
    mem_used: Optional[float] = None
    mem_total: Optional[float] = None
    temp: Optional[float] = None


class DeviceMonitorTool(QWidget):
    """
    GPU / device monitor.

    Embedded under ToolPageFrame: no big titles here.
    """

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._app_ctx = app_ctx
        self._settings = QSettings("fusionlab", "geoprior")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._hist_t: List[float] = []
        self._hist_gpu_util: List[float] = []
        self._hist_gpu_mem: List[float] = []
        self._hist_cpu: List[float] = []
        self._hist_ram: List[float] = []
        
        self._hist_gpu_temp_c: List[float] = []
        self._hist_gpu_mem_used_mib: List[float] = []
        self._hist_gpu_mem_total_mib: List[float] = []

        self._max_hist = 120
        self._last_status = ""

        self._build_ui()
        self._load_prefs()
        self.refresh()

    # -----------------------------------------------------------------
    # Public (ToolPageFrame will call refresh())
    # -----------------------------------------------------------------
    def refresh(self) -> None:
        self._refresh_probe()
        self._refresh_report()
        self._refresh_controls()
        self._refresh_plot()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ------------------------------
        # Top row: chips + actions
        # ------------------------------
        top = QHBoxLayout()
        top.setSpacing(8)

        self._chip_dev = _Chip("Device: —", self)
        self._chip_tf = _Chip("TF: —", self)
        self._chip_torch = _Chip("Torch: —", self)
        self._chip_sm = _Chip("nvidia-smi: —", self)

        top.addWidget(self._chip_dev)
        top.addWidget(self._chip_tf)
        top.addWidget(self._chip_torch)
        top.addWidget(self._chip_sm)
        top.addStretch(1)

        self._btn_copy = QToolButton(self)
        self._btn_copy.setAutoRaise(True)
        self._btn_copy.setObjectName("miniAction")
        self._btn_copy.setToolTip("Copy report")
        self._btn_copy.setIcon(
            self.style().standardIcon(
                QStyle.SP_DialogSaveButton
            )
        )

        self._btn_save = QToolButton(self)
        self._btn_save.setAutoRaise(True)
        self._btn_save.setObjectName("miniAction")
        self._btn_save.setToolTip("Save report to file")
        self._btn_save.setIcon(
            self.style().standardIcon(
                QStyle.SP_DialogOpenButton
            )
        )

        self._btn_refresh = QToolButton(self)
        self._btn_refresh.setAutoRaise(True)
        self._btn_refresh.setObjectName("miniAction")
        self._btn_refresh.setToolTip("Refresh now")
        self._btn_refresh.setIcon(
            self.style().standardIcon(
                QStyle.SP_BrowserReload
            )
        )

        top.addWidget(self._btn_copy)
        top.addWidget(self._btn_save)
        top.addWidget(self._btn_refresh)

        root.addLayout(top)

        # ------------------------------
        # Options row (runtime knobs)
        # ------------------------------
        opt = QFrame(self)
        opt.setObjectName("deviceMonOptions")
        ol = QVBoxLayout(opt)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(6)

        r1 = QHBoxLayout()
        r1.setSpacing(10)

        self._ck_force_cpu = QCheckBox(
            "Force CPU (restart)",
            self,
        )
        self._ck_mp = QCheckBox(
            "Mixed precision (TF)",
            self,
        )
        self._ck_mem_growth = QCheckBox(
            "GPU memory growth (TF)",
            self,
        )
        self._ck_xla = QCheckBox(
            "XLA (TF, restart)",
            self,
        )

        r1.addWidget(self._ck_force_cpu)
        r1.addWidget(self._ck_mp)
        r1.addWidget(self._ck_mem_growth)
        r1.addWidget(self._ck_xla)
        r1.addStretch(1)

        r2 = QHBoxLayout()
        r2.setSpacing(10)

        self._edit_cuda = QLineEdit(self)
        self._edit_cuda.setPlaceholderText(
            "CUDA_VISIBLE_DEVICES "
            "(e.g. 0 or 0,1 or -1)"
        )
        self._edit_cuda.setMinimumWidth(260)

        self._cmb_gpu = QComboBox(self)
        self._cmb_gpu.setMinimumWidth(220)

        r2.addWidget(QLabel("Visible GPUs:", self))
        r2.addWidget(self._edit_cuda, 1)
        r2.addSpacing(10)
        r2.addWidget(QLabel("Monitor:", self))
        r2.addWidget(self._cmb_gpu, 0)
        r2.addStretch(1)

        r3 = QHBoxLayout()
        r3.setSpacing(10)

        self._ck_auto = QCheckBox(
            "Auto refresh",
            self,
        )
        self._sp_int = QSpinBox(self)
        self._sp_int.setRange(1, 30)
        self._sp_int.setValue(2)
        self._sp_int.setSuffix(" s")

        self._btn_clear = QPushButton(
            "Clear history",
            self,
        )
        self._btn_clear.setObjectName("miniBtn")

        r3.addWidget(self._ck_auto)
        r3.addWidget(QLabel("Interval:", self))
        r3.addWidget(self._sp_int)
        r3.addSpacing(10)
        r3.addWidget(self._btn_clear)
        r3.addStretch(1)

        note = QLabel(
            "Some options are applied on next run/restart.",
            self,
        )
        note.setStyleSheet(
            "color: palette(mid);"
            "font-size: 9pt;"
        )

        ol.addLayout(r1)
        ol.addLayout(r2)
        ol.addLayout(r3)
        ol.addWidget(note)

        root.addWidget(opt, 0)

        # ------------------------------
        # Plot area
        # ------------------------------
        # self._fig = Figure(
        #     figsize=(6, 2.6),
        #     tight_layout=True,
        # )
        self._fig = Figure(figsize=(6, 2.6))

        self._canvas = FigureCanvas(self._fig)
        root.addWidget(self._canvas, 1)

        # ------------------------------
        # Report
        # ------------------------------
        self._tree = QTreeWidget(self)
        self._tree.setObjectName("deviceReportTree")
        self._tree.setHeaderLabels(["Item", "Value"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setIndentation(14)
        self._tree.setMinimumHeight(180)

        hdr = self._tree.header()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(
            0, QHeaderView.Interactive
        )
        hdr.setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self._tree.setColumnWidth(0, 240)

        self._tree.setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self._tree.customContextMenuRequested.connect(
            self._on_report_menu
        )

        root.addWidget(self._tree, 0)

        # signals
        self._btn_refresh.clicked.connect(self.refresh)
        self._btn_copy.clicked.connect(self._copy_report)
        self._btn_save.clicked.connect(self._save_report)

        self._btn_clear.clicked.connect(self._clear_hist)

        self._ck_auto.toggled.connect(self._on_auto)
        self._sp_int.valueChanged.connect(self._on_int)

        self._ck_force_cpu.toggled.connect(
            self._on_prefs_changed
        )
        self._ck_mp.toggled.connect(
            self._on_prefs_changed
        )
        self._ck_mem_growth.toggled.connect(
            self._on_prefs_changed
        )
        self._ck_xla.toggled.connect(
            self._on_prefs_changed
        )
        self._edit_cuda.editingFinished.connect(
            self._on_prefs_changed
        )
        self._cmb_gpu.currentIndexChanged.connect(
            lambda _i: self._refresh_plot()
        )

    # -----------------------------------------------------------------
    # Timer / auto refresh
    # -----------------------------------------------------------------
    def _on_auto(self, on: bool) -> None:
        self._settings.setValue(
            "device.auto_refresh",
            bool(on),
        )
        self._apply_timer()

    def _on_int(self, _v: int) -> None:
        self._settings.setValue(
            "device.refresh_interval_s",
            int(self._sp_int.value()),
        )
        self._apply_timer()

    def _apply_timer(self) -> None:
        on = bool(self._ck_auto.isChecked())
        if not on:
            self._timer.stop()
            return
        ms = int(self._sp_int.value()) * 1000
        self._timer.start(ms)
        
    def _tick(self) -> None:
        if not self.isVisible():
            return
        if self._canvas.width() <= 2:
            return
        if self._canvas.height() <= 2:
            return
    
        self._refresh_probe(only_live=True)
        self._refresh_plot()


    # -----------------------------------------------------------------
    # Prefs (store + settings)
    # -----------------------------------------------------------------
    def _load_prefs(self) -> None:
        self._ck_force_cpu.setChecked(
            bool(self._settings.value(
                "device.force_cpu",
                False,
            ))
        )
        self._ck_mp.setChecked(
            bool(self._settings.value(
                "device.mixed_precision",
                False,
            ))
        )
        self._ck_mem_growth.setChecked(
            bool(self._settings.value(
                "device.tf_memory_growth",
                True,
            ))
        )
        self._ck_xla.setChecked(
            bool(self._settings.value(
                "device.tf_xla",
                False,
            ))
        )

        self._edit_cuda.setText(
            _safe_str(self._settings.value(
                "device.cuda_visible",
                "",
            ))
        )

        self._ck_auto.setChecked(
            bool(self._settings.value(
                "device.auto_refresh",
                False,
            ))
        )
        self._sp_int.setValue(
            int(self._settings.value(
                "device.refresh_interval_s",
                2,
            ))
        )
        self._apply_timer()
        
    def _on_report_menu(self, pos) -> None:
        it = self._tree.itemAt(pos)
        if it is None:
            return

        menu = QMenu(self)
        a_val = menu.addAction("Copy value")
        a_row = menu.addAction("Copy row")
        act = menu.exec_(
            self._tree.viewport().mapToGlobal(pos)
        )
        if act is None:
            return

        if act == a_val:
            txt = it.text(1)
        else:
            txt = f"{it.text(0)}\t{it.text(1)}"

        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(txt)

    def _on_prefs_changed(self) -> None:
        self._settings.setValue(
            "device.force_cpu",
            bool(self._ck_force_cpu.isChecked()),
        )
        self._settings.setValue(
            "device.mixed_precision",
            bool(self._ck_mp.isChecked()),
        )
        self._settings.setValue(
            "device.tf_memory_growth",
            bool(self._ck_mem_growth.isChecked()),
        )
        self._settings.setValue(
            "device.tf_xla",
            bool(self._ck_xla.isChecked()),
        )
        self._settings.setValue(
            "device.cuda_visible",
            _safe_str(self._edit_cuda.text()).strip(),
        )

        self._try_write_store()
        self._refresh_report()

    def _try_write_store(self) -> None:
        ctx = self._app_ctx
        store = getattr(ctx, "config_store", None)
        if store is None:
            store = getattr(ctx, "store", None)
        if store is None:
            return

        kv = {
            "device.force_cpu":
            bool(self._ck_force_cpu.isChecked()),
            "device.mixed_precision":
            bool(self._ck_mp.isChecked()),
            "device.tf_memory_growth":
            bool(self._ck_mem_growth.isChecked()),
            "device.tf_xla":
            bool(self._ck_xla.isChecked()),
            "device.cuda_visible":
            _safe_str(self._edit_cuda.text()).strip(),
        }

        fn = getattr(store, "set", None)
        if not callable(fn):
            return

        for k, v in kv.items():
            try:
                fn(k, v)
            except Exception:
                pass

    # -----------------------------------------------------------------
    # Probe (TF / Torch / nvidia-smi / CPU)
    # -----------------------------------------------------------------
    def _refresh_probe(self, only_live: bool = False) -> None:
        tf_v, tf_gpu = self._tf_status()
        th_v, th_gpu = self._torch_status()

        sm_ok = self._has_nvidia_smi()
        sm_txt = "OK" if sm_ok else "—"

        dev = "GPU" if (tf_gpu or th_gpu) else "CPU"
        self._chip_dev.setText(f"Device: {dev}")
        self._chip_tf.setText(f"TF: {tf_v}")
        self._chip_torch.setText(f"Torch: {th_v}")
        self._chip_sm.setText(f"nvidia-smi: {sm_txt}")

        if only_live:
            self._push_live_sample()
            return

        self._refresh_gpu_combo()
        self._push_live_sample()

    def _refresh_gpu_combo(self) -> None:
        rows = self._query_gpus()
        cur = self._cmb_gpu.currentIndex()

        self._cmb_gpu.blockSignals(True)
        self._cmb_gpu.clear()

        if not rows:
            self._cmb_gpu.addItem("No GPU")
            self._cmb_gpu.setEnabled(False)
        else:
            self._cmb_gpu.setEnabled(True)
            for r in rows:
                label = f"[{r.idx}] {r.name}"
                self._cmb_gpu.addItem(label, r.idx)

            if 0 <= cur < self._cmb_gpu.count():
                self._cmb_gpu.setCurrentIndex(cur)
            else:
                self._cmb_gpu.setCurrentIndex(0)

        self._cmb_gpu.blockSignals(False)

    def _push_live_sample(self) -> None:
        t = time.time()

        cpu, ram = self._cpu_ram()
        util, mem_used, mem_total, temp = self._gpu_snapshot()

        mem_pct = 0.0
        if mem_total > 0:
            mem_pct = 100.0 * (mem_used / mem_total)

        self._hist_t.append(t)
        self._hist_cpu.append(cpu)
        self._hist_ram.append(ram)

        self._hist_gpu_util.append(util)
        self._hist_gpu_mem.append(mem_pct)

        self._hist_gpu_temp_c.append(temp)
        self._hist_gpu_mem_used_mib.append(mem_used)
        self._hist_gpu_mem_total_mib.append(mem_total)

        if len(self._hist_t) > self._max_hist:
            self._hist_t = self._hist_t[-self._max_hist :]
            self._hist_cpu = self._hist_cpu[-self._max_hist :]
            self._hist_ram = self._hist_ram[-self._max_hist :]
            self._hist_gpu_util = self._hist_gpu_util[-self._max_hist :]
            self._hist_gpu_mem = self._hist_gpu_mem[-self._max_hist :]

            self._hist_gpu_temp_c = (
                self._hist_gpu_temp_c[-self._max_hist :]
            )
            self._hist_gpu_mem_used_mib = (
                self._hist_gpu_mem_used_mib[-self._max_hist :]
            )
            self._hist_gpu_mem_total_mib = (
                self._hist_gpu_mem_total_mib[-self._max_hist :]
            )

    def _cpu_ram(self) -> Tuple[float, float]:
        # psutil is optional
        try:
            import psutil  # type: ignore

            cpu = float(psutil.cpu_percent(None))
            ram = float(psutil.virtual_memory().percent)
            return cpu, ram
        except Exception:
            return 0.0, 0.0
        
    def _gpu_snapshot(
        self,
    ) -> Tuple[float, float, float, float]:
        """
        Return a single-GPU snapshot for the currently selected GPU.

        Returns
        -------
        util_pct : float
        mem_used_mib : float
        mem_total_mib : float
        temp_c : float
        """
        rows = self._query_gpus()
        if not rows:
            return 0.0, 0.0, 0.0, 0.0

        idx = self._cmb_gpu.currentData()
        if idx is None:
            idx = rows[0].idx

        pick: Optional[_GpuRow] = None
        for r in rows:
            if r.idx == int(idx):
                pick = r
                break
        if pick is None:
            pick = rows[0]

        util = float(pick.util or 0.0)
        mem_used = float(pick.mem_used or 0.0)
        mem_total = float(pick.mem_total or 0.0)
        temp = float(pick.temp or 0.0)
        return util, mem_used, mem_total, temp

    def _tf_status(self) -> Tuple[str, bool]:
        try:
            import tensorflow as tf  # type: ignore

            v = _safe_str(getattr(tf, "__version__", "?"))
            gpus = []
            try:
                gpus = tf.config.list_physical_devices("GPU")
            except Exception:
                pass
            return v, bool(gpus)
        except Exception:
            return "—", False

    def _torch_status(self) -> Tuple[str, bool]:
        try:
            import torch  # type: ignore

            v = _safe_str(getattr(torch, "__version__", "?"))
            ok = False
            try:
                ok = bool(torch.cuda.is_available())
            except Exception:
                ok = False
            return v, ok
        except Exception:
            return "—", False

    def _has_nvidia_smi(self) -> bool:
        if shutil.which("nvidia-smi"):
            return True
        return False

    def _query_gpus(self) -> List[_GpuRow]:
        if not self._has_nvidia_smi():
            return []

        cmd = [
            "nvidia-smi",
            "--query-gpu="
            "index,name,utilization.gpu,"
            "memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = self._run_cmd(cmd)
        if not out:
            return []

        rows: List[_GpuRow] = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                idx = int(parts[0])
            except Exception:
                continue
            name = parts[1]

            def _f(x: str) -> Optional[float]:
                try:
                    return float(x)
                except Exception:
                    return None

            rows.append(
                _GpuRow(
                    idx=idx,
                    name=name,
                    util=_f(parts[2]),
                    mem_used=_f(parts[3]),
                    mem_total=_f(parts[4]),
                    temp=_f(parts[5]),
                )
            )
        return rows

    def _run_cmd(self, cmd: List[str]) -> str:
        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if p.returncode != 0:
                return ""
            return (p.stdout or "").strip()
        except Exception:
            return ""

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    def _refresh_plot(self) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        if len(self._hist_t) < 2:
            ax.text(
                0.5,
                0.5,
                "No live samples yet.",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            self._canvas.draw()
            return

        t0 = self._hist_t[0]
        xs = [t - t0 for t in self._hist_t]

        # Left axis: percentages
        l1 = ax.plot(xs, self._hist_cpu, label="CPU %")
        l2 = ax.plot(xs, self._hist_ram, label="RAM %")
        l3 = ax.plot(xs, self._hist_gpu_util, label="GPU %")
        l4 = ax.plot(xs, self._hist_gpu_mem, label="GPU mem %")

        ax.set_xlabel("Seconds")
        ax.set_ylabel("Percent")
        ax.set_title("Live utilization")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 100)

        handles = []
        labels = []
        for ln in (l1 + l2 + l3 + l4):
            handles.append(ln)
            labels.append(ln.get_label())

        # Right axis #1: temperature (°C)
        show_temp = (
            len(self._hist_gpu_temp_c) == len(xs)
            and max(self._hist_gpu_temp_c or [0.0]) > 0.0
        )
        ax_t = None
        if show_temp:
            ax_t = ax.twinx()
            tln = ax_t.plot(
                xs,
                self._hist_gpu_temp_c,
                label="GPU temp °C",
            )
            ax_t.set_ylabel("Temp (°C)")

            for ln in tln:
                handles.append(ln)
                labels.append(ln.get_label())

        # Right axis #2 (offset): memory MiB used/total
        show_mem_mib = (
            len(self._hist_gpu_mem_used_mib) == len(xs)
            and max(self._hist_gpu_mem_used_mib or [0.0]) > 0.0
        )
        ax_m = None
        if show_mem_mib:
            ax_m = ax.twinx()
            ax_m.spines["right"].set_position(("axes", 1.10))
            ax_m.spines["right"].set_visible(True)

            mln1 = ax_m.plot(
                xs,
                self._hist_gpu_mem_used_mib,
                label="GPU mem used (MiB)",
            )

            # total may be constant; still useful
            if (
                len(self._hist_gpu_mem_total_mib) == len(xs)
                and max(self._hist_gpu_mem_total_mib or [0.0]) > 0.0
            ):
                mln2 = ax_m.plot(
                    xs,
                    self._hist_gpu_mem_total_mib,
                    label="GPU mem total (MiB)",
                )
            else:
                mln2 = []

            ax_m.set_ylabel("GPU mem (MiB)")

            for ln in (mln1 + mln2):
                handles.append(ln)
                labels.append(ln.get_label())

        # One compact legend (combined)
        ax.legend(handles, labels, loc="upper right")

        # self._fig.tight_layout()
        # self._canvas.draw()
        try:
            w, h = self._canvas.get_width_height()
            if w <= 2 or h <= 2:
                return
            self._fig.tight_layout()
        except (LinAlgError, ValueError):
            # tight_layout can fail on transient sizes
            # or complex twinx configurations.
            pass
        
        self._canvas.draw()

    def _clear_hist(self) -> None:
        self._hist_t.clear()
        self._hist_cpu.clear()
        self._hist_ram.clear()
        self._hist_gpu_util.clear()
        self._hist_gpu_mem.clear()

        self._hist_gpu_temp_c.clear()
        self._hist_gpu_mem_used_mib.clear()
        self._hist_gpu_mem_total_mib.clear()

        self._refresh_plot()

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    def _refresh_report(self) -> None:
        self._refresh_report_tree()

    def _sec(
        self,
        title: str,
    ) -> QTreeWidgetItem:
        it = QTreeWidgetItem([title, ""])
        it.setFlags(Qt.ItemIsEnabled)
        f = it.font(0)
        f.setBold(True)
        it.setFont(0, f)
        it.setFirstColumnSpanned(True)
        self._tree.addTopLevelItem(it)
        return it

    def _kv(
        self,
        parent: QTreeWidgetItem,
        key: str,
        val: str,
    ) -> QTreeWidgetItem:
        it = QTreeWidgetItem([key, val])
        it.setFlags(
            Qt.ItemIsEnabled | Qt.ItemIsSelectable
        )
        parent.addChild(it)
        return it

    def _status_brush(
        self,
        status: str,
    ) -> Optional[QBrush]:
        s = (status or "").lower()
        if "ok" in s:
            return QBrush(QColor(0, 140, 0))
        if "need" in s or "warn" in s:
            return QBrush(QColor(180, 120, 0))
        if "error" in s or "fail" in s:
            return QBrush(QColor(160, 0, 0))
        return None

    def _refresh_report_tree(self) -> None:
        self._tree.clear()

        tf_v, tf_gpu = self._tf_status()
        th_v, th_gpu = self._torch_status()
        sm_ok = self._has_nvidia_smi()

        cpu, ram = self._cpu_ram()
        has_psutil = not (cpu == 0.0 and ram == 0.0)

        warns: List[str] = []
        if not sm_ok:
            warns.append("nvidia-smi not found.")
        if not has_psutil:
            warns.append("psutil not available.")
        if not (tf_gpu or th_gpu):
            warns.append("No GPU detected by frameworks.")

        status = "OK"
        if warns:
            status = "Needs attention"

        # -------------------------
        # Readiness
        # -------------------------
        s_read = self._sec("Readiness")
        it = self._kv(s_read, "Status", status)
        br = self._status_brush(status)
        if br is not None:
            it.setForeground(1, br)
        self._kv(s_read, "Warnings", str(len(warns)))
        if warns:
            s_w = QTreeWidgetItem(["Warnings", ""])
            s_w.setFlags(Qt.ItemIsEnabled)
            f = s_w.font(0)
            f.setBold(True)
            s_w.setFont(0, f)
            s_w.setFirstColumnSpanned(True)
            s_read.addChild(s_w)
            for w in warns:
                self._kv(s_w, "•", w)

        # -------------------------
        # Frameworks
        # -------------------------
        s_fw = self._sec("Frameworks")
        self._kv(
            s_fw,
            "TensorFlow",
            f"{tf_v} (gpu={tf_gpu})",
        )
        self._kv(
            s_fw,
            "PyTorch",
            f"{th_v} (gpu={th_gpu})",
        )
        self._kv(
            s_fw,
            "nvidia-smi",
            "OK" if sm_ok else "—",
        )

        # -------------------------
        # Env (selected)
        # -------------------------
        s_env = self._sec("Env (selected)")
        keys = [
            "CUDA_VISIBLE_DEVICES",
            "TF_CPP_MIN_LOG_LEVEL",
            "TF_FORCE_GPU_ALLOW_GROWTH",
            "TF_GPU_ALLOCATOR",
            "XLA_FLAGS",
            "OMP_NUM_THREADS",
        ]
        any_env = False
        for k in keys:
            v = (os.environ.get(k) or "").strip()
            if v:
                any_env = True
                self._kv(s_env, k, v)
        if not any_env:
            self._kv(s_env, "(none)", "")

        # -------------------------
        # GPUs
        # -------------------------
        s_gpu = self._sec("GPUs (nvidia-smi)")
        rows = self._query_gpus()
        if not rows:
            self._kv(
                s_gpu,
                "Info",
                "No GPU info available.",
            )
        else:
            for r in rows:
                g = QTreeWidgetItem(
                    [f"GPU [{r.idx}] {r.name}", ""]
                )
                g.setFlags(Qt.ItemIsEnabled)
                f = g.font(0)
                f.setBold(True)
                g.setFont(0, f)
                g.setFirstColumnSpanned(True)
                s_gpu.addChild(g)

                util = (
                    f"{float(r.util or 0.0):.0f} %"
                )
                mem = (
                    f"{float(r.mem_used or 0.0):.0f} / "
                    f"{float(r.mem_total or 0.0):.0f} MiB"
                )
                temp = (
                    f"{float(r.temp or 0.0):.0f} °C"
                )
                self._kv(g, "Utilization", util)
                self._kv(g, "Memory", mem)
                self._kv(g, "Temperature", temp)

        # -------------------------
        # Host
        # -------------------------
        s_host = self._sec("Host")
        self._kv(s_host, "Platform", platform.platform())
        self._kv(
            s_host,
            "Python",
            sys.version.split()[0],
        )
        self._kv(s_host, "Executable", sys.executable)
        if has_psutil:
            self._kv(s_host, "CPU", f"{cpu:.1f} %")
            self._kv(s_host, "RAM", f"{ram:.1f} %")
        else:
            self._kv(s_host, "psutil", "—")

        # -------------------------
        # Preferences (GUI)
        # -------------------------
        s_pref = self._sec("Preferences (GUI)")
        self._kv(
            s_pref,
            "force_cpu",
            str(self._ck_force_cpu.isChecked()),
        )
        self._kv(
            s_pref,
            "mixed_precision",
            str(self._ck_mp.isChecked()),
        )
        self._kv(
            s_pref,
            "tf_memory_growth",
            str(self._ck_mem_growth.isChecked()),
        )
        self._kv(
            s_pref,
            "tf_xla",
            str(self._ck_xla.isChecked()),
        )
        self._kv(
            s_pref,
            "cuda_visible",
            _safe_str(self._edit_cuda.text()).strip(),
        )
        self._kv(
            s_pref,
            "auto_refresh",
            str(self._ck_auto.isChecked()),
        )
        self._kv(
            s_pref,
            "interval_s",
            str(int(self._sp_int.value())),
        )

        self._tree.expandToDepth(1)

    def _make_report(self) -> str:
        lines: List[str] = []

        lines.append("=== System ===")
        lines.append(f"Python: {sys.version.split()[0]}")
        lines.append(f"Platform: {platform.platform()}")
        lines.append(f"Executable: {sys.executable}")
        lines.append("")

        lines.append("=== Env (selected) ===")
        keys = [
            "CUDA_VISIBLE_DEVICES",
            "TF_CPP_MIN_LOG_LEVEL",
            "TF_FORCE_GPU_ALLOW_GROWTH",
            "TF_GPU_ALLOCATOR",
            "XLA_FLAGS",
            "OMP_NUM_THREADS",
        ]
        for k in keys:
            v = os.environ.get(k, "")
            if v:
                lines.append(f"{k}={v}")
        if len(lines) == 2:
            lines.append("(no relevant env vars set)")
        lines.append("")

        tf_v, tf_gpu = self._tf_status()
        th_v, th_gpu = self._torch_status()

        lines.append("=== Frameworks ===")
        lines.append(f"TensorFlow: {tf_v}  gpu={tf_gpu}")
        lines.append(f"PyTorch:    {th_v}  gpu={th_gpu}")
        lines.append("")

        rows = self._query_gpus()
        lines.append("=== GPUs (nvidia-smi) ===")
        if not rows:
            lines.append("No GPU info (nvidia-smi missing).")
        else:
            for r in rows:
                lines.append(
                    f"[{r.idx}] {r.name}"
                )
                lines.append(
                    f"  util={r.util}%"
                )
                lines.append(
                    f"  mem={r.mem_used}/"
                    f"{r.mem_total} MiB"
                )
                lines.append(
                    f"  temp={r.temp} C"
                )
        lines.append("")

        cpu, ram = self._cpu_ram()
        lines.append("=== Host (psutil) ===")
        if cpu == 0.0 and ram == 0.0:
            lines.append("psutil not available.")
        else:
            lines.append(f"CPU: {cpu:.1f}%")
            lines.append(f"RAM: {ram:.1f}%")
        lines.append("")

        lines.append("=== Preferences (GUI) ===")
        lines.append(
            f"force_cpu="
            f"{self._ck_force_cpu.isChecked()}"
        )
        lines.append(
            f"mixed_precision="
            f"{self._ck_mp.isChecked()}"
        )
        lines.append(
            f"tf_memory_growth="
            f"{self._ck_mem_growth.isChecked()}"
        )
        lines.append(
            f"tf_xla="
            f"{self._ck_xla.isChecked()}"
        )
        lines.append(
            f"cuda_visible="
            f"{_safe_str(self._edit_cuda.text()).strip()}"
        )
        lines.append(
            f"auto_refresh="
            f"{self._ck_auto.isChecked()}"
        )
        lines.append(
            f"interval_s="
            f"{int(self._sp_int.value())}"
        )

        return "\n".join(lines)

    def _copy_report(self) -> None:
        txt = self._export_report_text()
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(txt)

    def _save_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save device report",
            "device_report.txt",
            "Text files (*.txt)",
        )
        if not path:
            return
        try:
            Path(path).write_text(
                self._export_report_text(),
                encoding="utf-8",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Save failed",
                _safe_str(exc),
            )
            
    def _export_report_text(self) -> str:
        lines: List[str] = []
        for i in range(self._tree.topLevelItemCount()):
            sec = self._tree.topLevelItem(i)
            if sec is None:
                continue
            title = sec.text(0)
            lines.append(f"[{title}]")
            for j in range(sec.childCount()):
                it = sec.child(j)
                if it is None:
                    continue
                k = it.text(0)
                v = it.text(1)
                if it.childCount() > 0 and not v:
                    lines.append(f"  {k}:")
                    for kk in range(it.childCount()):
                        sub = it.child(kk)
                        if sub is None:
                            continue
                        lines.append(
                            f"    {sub.text(0)} {sub.text(1)}"
                        )
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")
        return "\n".join(lines).strip()

    def _refresh_controls(self) -> None:
        # If user typed CUDA_VISIBLE_DEVICES in GUI prefs,
        # reflect it visually (no env mutation here).
        pass
    
    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        self._apply_timer()
    
    def hideEvent(self, ev) -> None:
        self._timer.stop()
        super().hideEvent(ev)
