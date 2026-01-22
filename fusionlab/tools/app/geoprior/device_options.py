# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
device_options

Device / processor options for GeoPrior GUI (v3.2+).

v3.0 (deprecated)
-----------------
The old widget mapped directly to NAT-style flat keys like:
TF_DEVICE_MODE, TF_INTRA_THREADS, ...

v3.2 (current)
--------------
This module binds to the *store* as the single source of truth:
GeoConfigStore + GeoPriorConfig fields, e.g.

- tf_device_mode
- tf_intra_threads
- tf_inter_threads
- tf_gpu_allow_growth
- tf_gpu_memory_limit_mb

Train/Tune should rely on:
    store.cfg.to_cfg_overrides()

which is responsible for producing the NAT-style overrides consumed
by backend configuration (e.g., configure_tf_from_cfg).

Notes
-----
- This widget updates the store live on user changes.
- It also listens to store.config_changed to refresh UI if config
  is loaded / reset elsewhere.
- Uses conservative fallbacks so the widget remains robust if some
  APIs evolve (patch_fields vs set_value, cfg vs config, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ( 
    List, Tuple, 
    Any, Dict, Optional,
    Mapping
)

import os
import platform
import sys

from PyQt5.QtCore import Qt, QSignalBlocker
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
    QPushButton,
    QPlainTextEdit,
)

# GeoPrior v3.2 store
from .config.store import GeoConfigStore


# ----------------------------------------------------------------------
# Store-side dataclass (GeoPriorConfig field names, not NAT keys)
# ----------------------------------------------------------------------

@dataclass
class DeviceOptions:
    """
    Store-side device/runtime options (v3.2+).

    Parameters
    ----------
    tf_device_mode : {"auto", "cpu", "gpu"}, default="auto"
        - "auto": use GPU if available else CPU
        - "cpu" : force CPU only
        - "gpu" : force GPU only (if present)

    tf_intra_threads : int or None, default=None
        Intra-op threads. None lets TF decide.

    tf_inter_threads : int or None, default=None
        Inter-op threads. None lets TF decide.

    tf_gpu_allow_growth : bool, default=True
        If True, allow GPU memory growth (allocate on demand).

    tf_gpu_memory_limit_mb : int or None, default=None
        Optional GPU memory cap (MB). None means no explicit cap.
    """

    tf_device_mode: str = "auto"
    tf_intra_threads: Optional[int] = None
    tf_inter_threads: Optional[int] = None
    tf_gpu_allow_growth: bool = True
    tf_gpu_memory_limit_mb: Optional[int] = None

    # --------------------------- store mapping ---------------------------

    @classmethod
    def from_store(cls, store: GeoConfigStore) -> "DeviceOptions":
        cfg = _store_cfg(store)
        return cls(
            tf_device_mode=str(getattr(cfg, "tf_device_mode", "auto") or "auto"),
            tf_intra_threads=_none_if_nonpos(getattr(cfg, "tf_intra_threads", None)),
            tf_inter_threads=_none_if_nonpos(getattr(cfg, "tf_inter_threads", None)),
            tf_gpu_allow_growth=bool(getattr(cfg, "tf_gpu_allow_growth", True)),
            tf_gpu_memory_limit_mb=_none_if_nonpos(
                getattr(cfg, "tf_gpu_memory_limit_mb", None)
            ),
        )

    def apply_to_store(self, store: GeoConfigStore) -> None:
        _store_patch_fields(
            store,
            {
                "tf_device_mode": (self.tf_device_mode or "auto").strip().lower(),
                "tf_intra_threads": self.tf_intra_threads,
                "tf_inter_threads": self.tf_inter_threads,
                "tf_gpu_allow_growth": bool(self.tf_gpu_allow_growth),
                "tf_gpu_memory_limit_mb": self.tf_gpu_memory_limit_mb,
            },
        )

    # --------------------------- legacy mapping ---------------------------

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "DeviceOptions":
        """Create options from a legacy NAT-style config dict.

        Accepts either v3.2 field names (tf_*) or NAT-style keys (TF_*).
        Missing entries fall back to dataclass defaults.
        """
        cfg = cfg or {}

        def pick(*names: str):
            for n in names:
                if n in cfg:
                    return cfg.get(n)
            return None

        mode = pick("tf_device_mode", "TF_DEVICE_MODE", "device_mode")
        mode = str(mode or "auto").strip().lower()

        intra = _none_if_nonpos(
            pick("tf_intra_threads", "TF_INTRA_THREADS", "intra_threads")
        )
        inter = _none_if_nonpos(
            pick("tf_inter_threads", "TF_INTER_THREADS", "inter_threads")
        )

        ag = pick("tf_gpu_allow_growth", "TF_GPU_ALLOW_GROWTH", "gpu_allow_growth")
        allow_growth = bool(True if ag is None else ag)

        mem = _none_if_nonpos(
            pick(
                "tf_gpu_memory_limit_mb",
                "TF_GPU_MEMORY_LIMIT_MB",
                "gpu_memory_limit_mb",
            )
        )

        return cls(
            tf_device_mode=mode,
            tf_intra_threads=intra,
            tf_inter_threads=inter,
            tf_gpu_allow_growth=allow_growth,
            tf_gpu_memory_limit_mb=mem,
        )

    def to_cfg_overrides(self) -> Dict[str, Any]:
        """Export NAT-style overrides consumed by backend config (TF_* keys)."""
        out: Dict[str, Any] = {
            "TF_DEVICE_MODE": (self.tf_device_mode or "auto").strip().lower(),
            "TF_GPU_ALLOW_GROWTH": bool(self.tf_gpu_allow_growth),
        }
        if self.tf_intra_threads is not None:
            out["TF_INTRA_THREADS"] = int(self.tf_intra_threads)
        if self.tf_inter_threads is not None:
            out["TF_INTER_THREADS"] = int(self.tf_inter_threads)
        if self.tf_gpu_memory_limit_mb is not None:
            out["TF_GPU_MEMORY_LIMIT_MB"] = int(self.tf_gpu_memory_limit_mb)
        return out

# ----------------------------------------------------------------------
# Qt widget
# ----------------------------------------------------------------------

class DeviceOptionsWidget(QWidget):
    """
    Modern device/runtime widget bound to GeoConfigStore (v3.2+).

    UI (compact + robust)
    ---------------------
    Execution:
        Backend: [Auto | CPU only | GPU only]

    CPU threads:
        [x] Override CPU threading
            Intra: [spin]  Inter: [spin]

    GPU memory:
        [x] Enable GPU memory controls
            [x] Allow memory growth
            [x] Cap GPU memory (MB): [spin]

    Diagnostics:
        Small read-only panel showing detected CPU/GPU info.

    Notes
    -----
    - Updates store live.
    - Refreshes from store on store.config_changed.
    - Avoids feedback loops using QSignalBlocker.
    """

    def __init__(
        self,
        store: Optional[GeoConfigStore] = None,
        parent: Optional[QWidget] = None,
        *,
        initial: Optional[DeviceOptions] = None,
    ) -> None:
        super().__init__(parent)
        self._store = store

        # Baseline values used when a control group is disabled
        # (meaning: "do not change").
        if initial is not None:
            self._baseline_opts = initial
        elif self._store is not None:
            self._baseline_opts = DeviceOptions.from_store(self._store)
        else:
            self._baseline_opts = DeviceOptions()

        self._build_ui()
        self._connect()

        if self._store is not None:
            self.refresh_from_store()
        else:
            self.refresh_from_options(self._baseline_opts)

        self._refresh_diag()

    # --------------------------- UI building ---------------------------
    
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(10)
    
        # ===============================================================
        # Group: Processor / devices (left)
        # ===============================================================
        grp = QGroupBox("Processor / devices")
        grid = QGridLayout(grp)
        grid.setContentsMargins(10, 8, 10, 10)
        grid.setSpacing(8)
    
        row = 0
    
        # --- Backend mode
        grid.addWidget(QLabel("Backend:"), row, 0, Qt.AlignTop)
    
        self.cmb_backend = QComboBox()
        self.cmb_backend.addItem("Auto (GPU if available)", "auto")
        self.cmb_backend.addItem("CPU only", "cpu")
        self.cmb_backend.addItem("GPU only", "gpu")
        self.cmb_backend.setToolTip(
            "Execution target for TensorFlow.\n"
            "Auto: use GPU if available else CPU.\n"
            "CPU only: ignore GPUs.\n"
            "GPU only: force GPU (if present)."
        )
        grid.addWidget(self.cmb_backend, row, 1, 1, 2)
        row += 1
    
        # --- CPU threads
        grid.addWidget(QLabel("CPU threads:"), row, 0, Qt.AlignTop)
    
        cpu_col = QVBoxLayout()
        self.chk_threads_override = QCheckBox(
            "Override CPU threading"
        )
        self.chk_threads_override.setToolTip(
            "When unchecked, TensorFlow chooses thread counts.\n"
            "When checked, you can set intra/inter threads."
        )
    
        threads_row = QHBoxLayout()
    
        self.spin_intra = QSpinBox()
        self.spin_intra.setRange(1, 512)
        self.spin_intra.setSingleStep(1)
        self.spin_intra.setToolTip(
            "Intra-op threads (>= 1)."
        )
    
        self.spin_inter = QSpinBox()
        self.spin_inter.setRange(1, 512)
        self.spin_inter.setSingleStep(1)
        self.spin_inter.setToolTip(
            "Inter-op threads (>= 1)."
        )
    
        threads_row.addWidget(QLabel("Intra:"))
        threads_row.addWidget(self.spin_intra)
        threads_row.addSpacing(10)
        threads_row.addWidget(QLabel("Inter:"))
        threads_row.addWidget(self.spin_inter)
        threads_row.addStretch(1)
    
        cpu_col.addWidget(self.chk_threads_override)
        cpu_col.addLayout(threads_row)
    
        grid.addLayout(cpu_col, row, 1, 1, 2)
        row += 1
    
        # --- GPU memory
        grid.addWidget(QLabel("GPU memory:"), row, 0, Qt.AlignTop)
    
        gpu_col = QVBoxLayout()
        self.chk_gpu_controls = QCheckBox(
            "Enable GPU memory controls"
        )
        self.chk_gpu_controls.setToolTip(
            "When unchecked, GPU memory settings remain unchanged.\n"
            "When checked, you can manage growth and memory cap."
        )
    
        self.chk_allow_growth = QCheckBox(
            "Allow memory growth"
        )
        self.chk_allow_growth.setToolTip(
            "If enabled, TF allocates GPU memory on demand,\n"
            "instead of reserving it all at startup."
        )
    
        mem_row = QHBoxLayout()
        self.chk_mem_cap = QCheckBox(
            "Cap GPU memory (MB):"
        )
        self.chk_mem_cap.setToolTip(
            "Enable a per-process GPU memory cap.\n"
            "If disabled, no explicit memory cap is set."
        )
    
        self.spin_mem_limit = QSpinBox()
        self.spin_mem_limit.setRange(256, 1_000_000)
        self.spin_mem_limit.setSingleStep(256)
        self.spin_mem_limit.setToolTip(
            "GPU memory cap in MB."
        )
    
        mem_row.addWidget(self.chk_mem_cap)
        mem_row.addWidget(self.spin_mem_limit)
        mem_row.addStretch(1)
    
        gpu_col.addWidget(self.chk_gpu_controls)
        gpu_col.addWidget(self.chk_allow_growth)
        gpu_col.addLayout(mem_row)
    
        grid.addLayout(gpu_col, row, 1, 1, 2)
        row += 1
    
        # ===============================================================
        # Group: Runtime diagnostics (right)
        # ===============================================================
        diag_grp = QGroupBox("Runtime diagnostics")
        diag_layout = QVBoxLayout(diag_grp)
        diag_layout.setContentsMargins(10, 8, 10, 10)
        diag_layout.setSpacing(6)
    
        diag_top = QHBoxLayout()
        self.btn_refresh_diag = QPushButton("Refresh")
        self.btn_refresh_diag.setToolTip(
            "Refresh runtime info (CPU/GPU detection).\n"
            "Does not change settings."
        )
        diag_top.addStretch(1)
        diag_top.addWidget(self.btn_refresh_diag)
    
        self.txt_diag = QPlainTextEdit()
        self.txt_diag.setReadOnly(True)
        self.txt_diag.setMaximumHeight(160)
    
        diag_layout.addLayout(diag_top)
        diag_layout.addWidget(self.txt_diag, 1)
    
        # ---------------------------------------------------------------
        # Add both groups side-by-side
        # ---------------------------------------------------------------
        root.addWidget(grp, 2)
        root.addWidget(diag_grp, 1)
    
        # initial enable/disable
        self._apply_enable_state()

    # --------------------------- wiring ---------------------------

    def _connect(self) -> None:
        # UI -> store
        self.cmb_backend.currentIndexChanged.connect(self._commit_to_store)
        self.chk_threads_override.toggled.connect(self._on_threads_toggle)
        self.spin_intra.valueChanged.connect(self._commit_to_store)
        self.spin_inter.valueChanged.connect(self._commit_to_store)

        self.chk_gpu_controls.toggled.connect(self._on_gpu_controls_toggle)
        self.chk_allow_growth.toggled.connect(self._commit_to_store)
        self.chk_mem_cap.toggled.connect(self._on_memcap_toggle)
        self.spin_mem_limit.valueChanged.connect(self._commit_to_store)

        self.btn_refresh_diag.clicked.connect(self._refresh_diag)

        # store -> UI
        sig = getattr(self._store, "config_changed", None)
        if sig is not None:
            try:
                sig.connect(self.refresh_from_store)
            except Exception:
                pass

    # --------------------------- store sync ---------------------------

    def refresh_from_store(self, *_: Any) -> None:
        """Refresh UI from current store.cfg values."""
        opts = DeviceOptions.from_store(self._store)

        with QSignalBlocker(self.cmb_backend):
            idx = self.cmb_backend.findData((opts.tf_device_mode or "auto").lower())
            if idx < 0:
                idx = self.cmb_backend.findData("auto")
            if idx >= 0:
                self.cmb_backend.setCurrentIndex(idx)

        # threads
        has_threads = (opts.tf_intra_threads is not None) or (opts.tf_inter_threads is not None)
        with QSignalBlocker(self.chk_threads_override):
            self.chk_threads_override.setChecked(bool(has_threads))

        with QSignalBlocker(self.spin_intra):
            self.spin_intra.setValue(int(opts.tf_intra_threads or max(1, self.spin_intra.value())))
        with QSignalBlocker(self.spin_inter):
            self.spin_inter.setValue(int(opts.tf_inter_threads or max(1, self.spin_inter.value())))

        # gpu controls
        # We consider GPU controls "enabled" if either growth differs from
        # default or a cap is set; but user can toggle it explicitly too.
        cap_set = opts.tf_gpu_memory_limit_mb is not None
        with QSignalBlocker(self.chk_gpu_controls):
            self.chk_gpu_controls.setChecked(True)  # default to enabled UX

        with QSignalBlocker(self.chk_allow_growth):
            self.chk_allow_growth.setChecked(bool(opts.tf_gpu_allow_growth))

        with QSignalBlocker(self.chk_mem_cap):
            self.chk_mem_cap.setChecked(bool(cap_set))

        with QSignalBlocker(self.spin_mem_limit):
            self.spin_mem_limit.setValue(
                int(opts.tf_gpu_memory_limit_mb or max(
                    256, self.spin_mem_limit.value())))

        self._apply_enable_state()

    # --------------------------- enable logic ---------------------------

    def _apply_enable_state(self) -> None:
        threads_on = self.chk_threads_override.isChecked()
        self.spin_intra.setEnabled(threads_on)
        self.spin_inter.setEnabled(threads_on)

        gpu_on = self.chk_gpu_controls.isChecked()
        self.chk_allow_growth.setEnabled(gpu_on)

        self.chk_mem_cap.setEnabled(gpu_on)
        self.spin_mem_limit.setEnabled(gpu_on and self.chk_mem_cap.isChecked())

    def _on_threads_toggle(self, _: bool) -> None:
        self._apply_enable_state()
        self._commit_to_store()

    def _on_gpu_controls_toggle(self, _: bool) -> None:
        self._apply_enable_state()
        self._commit_to_store()

    def _on_memcap_toggle(self, _: bool) -> None:
        self._apply_enable_state()
        self._commit_to_store()

    # --------------------------- diagnostics ---------------------------

    def _refresh_diag(self, *_: Any) -> None:
        self.txt_diag.setPlainText(
            runtime_summary_text(self._store)
        )

    def refresh_from_options(self, opts: DeviceOptions) -> None:
        """Refresh UI from a standalone options object (legacy dialogs)."""
        with QSignalBlocker(self.cmb_backend):
            idx = self.cmb_backend.findData((opts.tf_device_mode or "auto").lower())
            if idx < 0:
                idx = self.cmb_backend.findData("auto")
            if idx >= 0:
                self.cmb_backend.setCurrentIndex(idx)

        has_threads = (opts.tf_intra_threads is not None) or (
            opts.tf_inter_threads is not None
        )
        with QSignalBlocker(self.chk_threads_override):
            self.chk_threads_override.setChecked(bool(has_threads))

        with QSignalBlocker(self.spin_intra):
            self.spin_intra.setValue(
                int(opts.tf_intra_threads or max(1, self.spin_intra.value()))
            )
        with QSignalBlocker(self.spin_inter):
            self.spin_inter.setValue(
                int(opts.tf_inter_threads or max(1, self.spin_inter.value()))
            )

        cap_set = opts.tf_gpu_memory_limit_mb is not None
        with QSignalBlocker(self.chk_gpu_controls):
            self.chk_gpu_controls.setChecked(True)

        with QSignalBlocker(self.chk_allow_growth):
            self.chk_allow_growth.setChecked(bool(opts.tf_gpu_allow_growth))

        with QSignalBlocker(self.chk_mem_cap):
            self.chk_mem_cap.setChecked(bool(cap_set))

        with QSignalBlocker(self.spin_mem_limit):
            self.spin_mem_limit.setValue(
                int(opts.tf_gpu_memory_limit_mb or max(256, self.spin_mem_limit.value()))
            )

        self._apply_enable_state()

    def _collect_options(self) -> DeviceOptions:
        """Collect current UI state into a DeviceOptions object.

        If a whole control group is disabled (unchecked), fall back
        to baseline values, meaning "do not change".
        """
        mode = str(self.cmb_backend.currentData() or "auto").strip().lower()

        # threads
        if self.chk_threads_override.isChecked():
            intra = int(self.spin_intra.value())
            inter = int(self.spin_inter.value())
        else:
            intra = None
            inter = None

        # gpu
        if self.chk_gpu_controls.isChecked():
            allow_growth = bool(self.chk_allow_growth.isChecked())
            mem = int(self.spin_mem_limit.value()) if self.chk_mem_cap.isChecked() else None
        else:
            allow_growth = bool(self._baseline_opts.tf_gpu_allow_growth)
            mem = self._baseline_opts.tf_gpu_memory_limit_mb

        return DeviceOptions(
            tf_device_mode=mode,
            tf_intra_threads=intra,
            tf_inter_threads=inter,
            tf_gpu_allow_growth=allow_growth,
            tf_gpu_memory_limit_mb=mem,
        )

    def to_cfg_overrides(self) -> Dict[str, Any]:
        """Return NAT-style TF_* overrides from current UI state."""
        return self._collect_options().to_cfg_overrides()

        
    def _commit_to_store(self, *_: Any) -> None:
        """Collect UI state and push to store."""
        mode = str(self.cmb_backend.currentData() or "auto").strip().lower()

        # threads
        if self.chk_threads_override.isChecked():
            intra = int(self.spin_intra.value())
            inter = int(self.spin_inter.value())
        else:
            intra = None
            inter = None

        # gpu memory
        if self.chk_gpu_controls.isChecked():
            allow_growth = bool(self.chk_allow_growth.isChecked())
            if self.chk_mem_cap.isChecked():
                mem = int(self.spin_mem_limit.value())
            else:
                mem = None
        else:
            # if user disables GPU controls, keep current store values
            # (do not force reset); we just avoid writing changes.
            allow_growth = None
            mem = None

        patch: Dict[str, Any] = {
            "tf_device_mode": mode,
            "tf_intra_threads": intra,
            "tf_inter_threads": inter,
        }
        if allow_growth is not None:
            patch["tf_gpu_allow_growth"] = allow_growth
        if self.chk_gpu_controls.isChecked():
            patch["tf_gpu_memory_limit_mb"] = mem

        _store_patch_fields(self._store, patch)
        self._apply_enable_state()
# ----------------------------------------------------------------------
# Internal helpers (robust against small API shifts)
# ----------------------------------------------------------------------

def _none_if_nonpos(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        iv = int(v)
    except Exception:
        return None
    return iv if iv > 0 else None


def _store_cfg(store: GeoConfigStore) -> Any:
    """
    Return store config object (GeoPriorConfig).

    Tries common attribute names to stay robust.
    """
    for name in ("cfg", "config"):
        if hasattr(store, name):
            return getattr(store, name)
    raise AttributeError("GeoConfigStore has no cfg/config attribute.")


def _store_patch_fields(store: GeoConfigStore, patch: Mapping[str, Any]) -> None:
    """
    Apply patch to store in the most compatible way.
    """
    if hasattr(store, "patch_fields"):
        store.patch_fields(dict(patch))
        return
    if hasattr(store, "apply_patch"):
        store.apply_patch(dict(patch))
        return

    # last resort: setattr on cfg then emit if possible
    cfg = _store_cfg(store)
    for k, v in patch.items():
        try:
            setattr(cfg, k, v)
        except Exception:
            pass

    sig = getattr(store, "config_changed", None)
    if sig is not None:
        try:
            sig.emit()
        except Exception:
            pass
        
# ----------------------------------------------------------------------
# Public runtime helpers (shared by Train tab + Devices panel)
# ----------------------------------------------------------------------

def runtime_summary_text(
    store: Optional[GeoConfigStore] = None,
) -> str:

    lines: List[str] = []

    os_s = platform.platform()
    py_s = sys.version.split()[0]
    cpu_n = os.cpu_count() or 0

    lines.append(f"OS: {os_s}")
    lines.append(f"Python: {py_s}")
    lines.append(f"CPU cores: {cpu_n}")

    # RAM
    try:
        import psutil

        total = float(psutil.virtual_memory().total)
        gb = total / (1024.0**3)
        lines.append(f"RAM: {gb:.1f} GB")
    except Exception:
        pass

    # Backend/framework choice + versions
    lines.extend(_backend_lines(store))

    # GPU summary (TF first, then Torch)
    lines.extend(_gpu_lines())

    return "\n".join(lines)


def _backend_lines(
    store: Optional[GeoConfigStore],
) -> List[str]:
    chosen = _backend_choice_from_store(store)
    det, vers = _backend_detected()

    if chosen != "Auto":
        v = vers.get(chosen, "")
        if v:
            return [f"Backend: {chosen} ({v})"]
        return [f"Backend: {chosen}"]

    if det == "None":
        return ["Backend: Auto (none detected)"]

    if det == "Both":
        tfv = vers.get("TensorFlow", "")
        thv = vers.get("PyTorch", "")
        msg = "Backend: Auto (TF + Torch available)"
        if tfv or thv:
            msg += f"  TF={tfv}  Torch={thv}"
        return [msg]

    v = vers.get(det, "")
    if v:
        return [f"Backend: Auto ({det}, {v})"]
    return [f"Backend: Auto ({det})"]


def _backend_choice_from_store(
    store: Optional[GeoConfigStore],
) -> str:
    if store is None:
        return "Auto"

    # Try cfg attributes first (robust)
    try:
        cfg = _store_cfg(store)
    except Exception:
        cfg = None

    names = (
        "backend",
        "engine",
        "framework",
        "dl_backend",
        "nn_backend",
        "trainer_backend",
        "compute_backend",
    )

    for n in names:
        v = None
        if cfg is not None and hasattr(cfg, n):
            v = getattr(cfg, n)
        if v is None:
            continue

        s = str(v).strip().lower()
        if not s:
            continue

        if "torch" in s or "pytorch" in s:
            return "PyTorch"
        if "tf" in s or "tensorflow" in s or "keras" in s:
            return "TensorFlow"
        if "jax" in s:
            return "JAX"
        if s in ("auto", "default"):
            return "Auto"

    return "Auto"


def _backend_detected() -> Tuple[str, dict]:
    vers: dict[str, str] = {}

    have_tf = False
    have_th = False

    try:
        import tensorflow as tf

        have_tf = True
        vers["TensorFlow"] = getattr(tf, "__version__", "")
    except Exception:
        pass

    try:
        import torch

        have_th = True
        vers["PyTorch"] = getattr(torch, "__version__", "")
    except Exception:
        pass

    if have_tf and have_th:
        return "Both", vers
    if have_tf:
        return "TensorFlow", vers
    if have_th:
        return "PyTorch", vers
    return "None", vers


def _gpu_lines() -> List[str]:
    out: List[str] = []

    # -----------------------------
    # TensorFlow (preferred)
    # -----------------------------
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            out.append(f"GPU: {len(gpus)} device(s)")
            for i, dev in enumerate(gpus):
                # dev.name is like "/physical_device:GPU:0"
                label = getattr(dev, "name", f"GPU:{i}")

                details: Dict[str, Any] = {}
                try:
                    details = (
                        tf.config.experimental
                        .get_device_details(dev)
                    )
                except Exception:
                    details = {}

                # Human device name (if available)
                pretty = details.get("device_name") or ""
                if pretty:
                    label = pretty

                # Extra details (safe)
                extras: List[str] = []

                cc = details.get("compute_capability")
                if isinstance(cc, (tuple, list)) and len(cc) >= 2:
                    extras.append(f"cc {cc[0]}.{cc[1]}")
                elif cc:
                    extras.append(f"cc {cc}")

                mem = details.get("memory_limit")
                if isinstance(mem, (int, float)) and mem > 0:
                    mb = mem / (1024.0**2)
                    extras.append(f"{mb:.0f} MB")

                if extras:
                    out.append(
                        f" - {label} ({', '.join(extras)})"
                    )
                else:
                    out.append(f" - {label}")

            return out
    except Exception:
        pass

    # -----------------------------
    # PyTorch fallback
    # -----------------------------
    try:
        import torch

        if torch.cuda.is_available():
            n = int(torch.cuda.device_count())
            out.append(f"GPU: {n} CUDA device(s)")

            cuda_v = getattr(torch.version, "cuda", None)
            if cuda_v:
                out.append(f"CUDA: {cuda_v}")

            try:
                cudnn_v = torch.backends.cudnn.version()
                if cudnn_v:
                    out.append(f"cuDNN: {cudnn_v}")
            except Exception:
                pass

            for i in range(n):
                nm = torch.cuda.get_device_name(i)
                out.append(f" - cuda:{i} {nm}")

            return out
    except Exception:
        pass

    out.append("GPU: none detected")
    return out
