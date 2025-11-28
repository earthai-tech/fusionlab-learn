# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Device / processor options helpers for GeoPrior GUI.

Provides:

- DeviceOptions dataclass  – config-side representation
  of TF device / threading options.

- DeviceOptionsWidget      – small Qt widget that lets the
  user choose CPU vs GPU, CPU threads and GPU memory policy.

The widget maps directly to NAT-style config keys:

    TF_DEVICE_MODE          : "auto" | "cpu" | "gpu"
    TF_INTRA_THREADS        : int or None
    TF_INTER_THREADS        : int or None
    TF_GPU_ALLOW_GROWTH     : bool
    TF_GPU_MEMORY_LIMIT_MB  : int or None

These keys are consumed by::

    fusionlab.backends.devices.configure_tf_from_cfg(cfg, logger)

so Train / Tune just need to pass cfg_overrides that include
these names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
)


# ----------------------------------------------------------------------
# Dataclass: config-side representation
# ----------------------------------------------------------------------


@dataclass
class DeviceOptions:
    """
    High-level device / processor options for TensorFlow.

    Parameters
    ----------
    device_mode : {"auto", "cpu", "gpu}, default="auto"
        - "auto" : use GPU if available, else CPU.
        - "cpu"  : force CPU only.
        - "gpu"  : force GPU only.

    intra_threads : int or None, default=None
        Number of intra-op threads. None lets TensorFlow decide.
        If you agree on a convention, `0` or any non-positive
        value can be interpreted as "auto".

    inter_threads : int or None, default=None
        Number of inter-op threads. Same convention as for
        ``intra_threads``.

    gpu_allow_growth : bool, default=True
        If True, enable "allow growth" on the visible GPU(s),
        letting TF allocate memory on demand instead of
        grabbing all available memory up front.

    gpu_memory_limit_mb : int or None, default=None
        Optional per-process memory cap (in megabytes). None or
        a non-positive value means "no explicit cap".
    """

    device_mode: str = "auto"
    intra_threads: Optional[int] = None
    inter_threads: Optional[int] = None
    gpu_allow_growth: bool = True
    gpu_memory_limit_mb: Optional[int] = None

    # --------------------------- cfg mapping ---------------------------

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "DeviceOptions":
        """
        Build DeviceOptions from a flat NAT-style config dict.

        Expected keys (all optional)::

            TF_DEVICE_MODE
            TF_INTRA_THREADS
            TF_INTER_THREADS
            TF_GPU_ALLOW_GROWTH
            TF_GPU_MEMORY_LIMIT_MB
        """
        mode = str(cfg.get("TF_DEVICE_MODE", "auto")).lower()

        def _as_int_or_none(name: str) -> Optional[int]:
            val = cfg.get(name, None)
            if val is None:
                return None
            try:
                iv = int(val)
            except Exception:
                return None
            if iv <= 0:
                return None
            return iv

        intra = _as_int_or_none("TF_INTRA_THREADS")
        inter = _as_int_or_none("TF_INTER_THREADS")

        allow_growth_raw = cfg.get("TF_GPU_ALLOW_GROWTH", True)
        allow_growth = bool(allow_growth_raw)

        mem_raw = cfg.get("TF_GPU_MEMORY_LIMIT_MB", None)
        mem_limit: Optional[int]
        try:
            mem_limit = int(mem_raw) if mem_raw is not None else None
        except Exception:
            mem_limit = None
        if mem_limit is not None and mem_limit <= 0:
            mem_limit = None

        return cls(
            device_mode=mode or "auto",
            intra_threads=intra,
            inter_threads=inter,
            gpu_allow_growth=allow_growth,
            gpu_memory_limit_mb=mem_limit,
        )

    def to_cfg_overrides(self) -> Dict[str, Any]:
        """
        Convert to NAT-style cfg_overrides dict.

        Only sets keys explicitly; callers can ``dict.update`` this
        into their existing overrides.
        """
        cfg: Dict[str, Any] = {}

        cfg["TF_DEVICE_MODE"] = self.device_mode or "auto"

        if self.intra_threads is not None and self.intra_threads > 0:
            cfg["TF_INTRA_THREADS"] = int(self.intra_threads)
        else:
            # Explicitly clear to "auto" if desired:
            # cfg["TF_INTRA_THREADS"] = None
            pass

        if self.inter_threads is not None and self.inter_threads > 0:
            cfg["TF_INTER_THREADS"] = int(self.inter_threads)
        else:
            # cfg["TF_INTER_THREADS"] = None
            pass

        cfg["TF_GPU_ALLOW_GROWTH"] = bool(self.gpu_allow_growth)

        if self.gpu_memory_limit_mb is not None and self.gpu_memory_limit_mb > 0:
            cfg["TF_GPU_MEMORY_LIMIT_MB"] = int(self.gpu_memory_limit_mb)
        else:
            # cfg["TF_GPU_MEMORY_LIMIT_MB"] = None
            pass

        return cfg

# ----------------------------------------------------------------------
# Qt widget: minimal Processor / Devices block
# ----------------------------------------------------------------------

class DeviceOptionsWidget(QWidget):
    """
    Compact UI widget to configure DeviceOptions.

    Layout
    ------
    [Backend]  [Auto (GPU if avail) | CPU only | GPU only]
    [CPU threads]  Intra: [spin]  Inter: [spin]   (0 → auto)
    [GPU memory]   [x] Allow growth
                   Limit (MB): [spin]  (0 → no limit)

    Use :meth:`to_options` to get a DeviceOptions instance,
    then :meth:`DeviceOptions.to_cfg_overrides` to update
    cfg_overrides passed to training / tuning.
    """

    def __init__(
        self,
        initial: Optional[DeviceOptions] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._build_ui()

        if initial is None:
            initial = DeviceOptions()
        self.load_from_options(initial)

    # --------------------------- UI building ---------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        group = QGroupBox("Processor / devices")
        group_layout = QGridLayout(group)
        group_layout.setContentsMargins(8, 6, 8, 8)
        group_layout.setSpacing(6)

        row = 0

        # --- Backend: CPU vs GPU vs auto ---
        lbl_backend = QLabel("Backend:")
        self.cmb_backend = QComboBox()
        self.cmb_backend.addItem("Auto (GPU if available)", "auto")
        self.cmb_backend.addItem("CPU only", "cpu")
        self.cmb_backend.addItem("GPU only", "gpu")
        self.cmb_backend.setToolTip(
            "Choose where to run TensorFlow:\n"
            " - Auto: GPU if available, else CPU\n"
            " - CPU only: ignore GPUs\n"
            " - GPU only: force GPU (if present)"
        )

        group_layout.addWidget(lbl_backend, row, 0)
        group_layout.addWidget(self.cmb_backend, row, 1, 1, 2)
        row += 1

        # --- CPU threads (intra / inter) ---
        lbl_threads = QLabel("CPU threads:")
        threads_row = QHBoxLayout()

        self.spin_intra = QSpinBox()
        self.spin_intra.setRange(0, 512)
        self.spin_intra.setValue(0)
        self.spin_intra.setToolTip(
            "Intra-op threads (0 = let TensorFlow decide)."
        )

        self.spin_inter = QSpinBox()
        self.spin_inter.setRange(0, 512)
        self.spin_inter.setValue(0)
        self.spin_inter.setToolTip(
            "Inter-op threads (0 = let TensorFlow decide)."
        )

        threads_row.addWidget(QLabel("Intra:"))
        threads_row.addWidget(self.spin_intra)
        threads_row.addSpacing(8)
        threads_row.addWidget(QLabel("Inter:"))
        threads_row.addWidget(self.spin_inter)
        threads_row.addStretch(1)

        group_layout.addWidget(lbl_threads, row, 0, Qt.AlignTop)
        group_layout.addLayout(threads_row, row, 1, 1, 2)
        row += 1

        # --- GPU memory policy ---
        lbl_gpu = QLabel("GPU memory:")
        mem_col = QVBoxLayout()

        self.chk_allow_growth = QCheckBox("Allow memory growth")
        self.chk_allow_growth.setChecked(True)
        self.chk_allow_growth.setToolTip(
            "When checked, TensorFlow allocates GPU memory on demand,\n"
            "instead of reserving all available GPU memory at startup."
        )

        mem_limit_row = QHBoxLayout()
        self.spin_mem_limit = QSpinBox()
        self.spin_mem_limit.setRange(0, 1_000_000)  # up to 1 TB, just in case
        self.spin_mem_limit.setValue(0)
        self.spin_mem_limit.setSingleStep(256)
        self.spin_mem_limit.setToolTip(
            "Optional per-process GPU memory cap (in MB).\n"
            "0 means 'no explicit cap'."
        )

        mem_limit_row.addWidget(QLabel("Limit (MB):"))
        mem_limit_row.addWidget(self.spin_mem_limit)
        mem_limit_row.addStretch(1)

        mem_col.addWidget(self.chk_allow_growth)
        mem_col.addLayout(mem_limit_row)

        group_layout.addWidget(lbl_gpu, row, 0, Qt.AlignTop)
        group_layout.addLayout(mem_col, row, 1, 1, 2)
        row += 1

        root.addWidget(group)

    # ---------------------- options <-> UI bridge ---------------------

    def load_from_options(self, opts: DeviceOptions) -> None:
        """Populate widgets from an existing DeviceOptions instance."""
        # Backend
        mode = (opts.device_mode or "auto").lower()
        idx = self.cmb_backend.findData(mode)
        if idx < 0:
            idx = self.cmb_backend.findData("auto")
        if idx >= 0:
            self.cmb_backend.setCurrentIndex(idx)

        # Threads
        self.spin_intra.setValue(int(opts.intra_threads or 0))
        self.spin_inter.setValue(int(opts.inter_threads or 0))

        # GPU memory
        self.chk_allow_growth.setChecked(bool(opts.gpu_allow_growth))
        self.spin_mem_limit.setValue(int(opts.gpu_memory_limit_mb or 0))

    def to_options(self) -> DeviceOptions:
        """
        Collect current widget values into a DeviceOptions instance.

        Conventions
        -----------
        - 0 threads → None (let TensorFlow decide).
        - 0 memory limit → None (no explicit cap).
        """
        mode = self.cmb_backend.currentData() or "auto"

        intra = int(self.spin_intra.value())
        if intra <= 0:
            intra_opt: Optional[int] = None
        else:
            intra_opt = intra

        inter = int(self.spin_inter.value())
        if inter <= 0:
            inter_opt: Optional[int] = None
        else:
            inter_opt = inter

        mem = int(self.spin_mem_limit.value())
        if mem <= 0:
            mem_opt: Optional[int] = None
        else:
            mem_opt = mem

        return DeviceOptions(
            device_mode=str(mode),
            intra_threads=intra_opt,
            inter_threads=inter_opt,
            gpu_allow_growth=self.chk_allow_growth.isChecked(),
            gpu_memory_limit_mb=mem_opt,
        )

    def to_cfg_overrides(self) -> Dict[str, Any]:
        """
        Shortcut: return a cfg_overrides dict directly.

        Equivalent to::

            self.to_options().to_cfg_overrides()
        """
        return self.to_options().to_cfg_overrides()

