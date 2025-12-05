# geoprior/workflows/base.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any

from ..config import GeoPriorConfig


@dataclass
class GUIHooks:
    """
    Thin abstraction over GUI side-effects, so workflows/services
    don't talk to PyQt directly.
    """
    log: Callable[[str], None]
    status: Callable[[str], None]
    update_progress: Callable[[float], None]

    ask_yes_no: Callable[[str, str], bool]
    warn: Callable[[str, str], None]
    error: Callable[[str, str], None]


@dataclass
class RunEnv:
    """
    Shared runtime environment for all workflows.

    - gui_runs_root: base results directory under ~/.fusionlab_runs/...
    - geo_cfg: current GeoPriorConfig (already synced from GUI).
    - device_overrides: GPU/CPU flags etc.
    - dry_mode: whether we are in dry-run mode from the GUI.
    """
    gui_runs_root: Path
    geo_cfg: GeoPriorConfig
    device_overrides: Dict[str, Any] = field(default_factory=dict)
    dry_mode: bool = False


class BaseWorkflowController:
    """
    Common base for Train / Tune / Infer / Xfer controllers.

    Provides a single place to build cfg_overrides.
    """

    def __init__(self, env: RunEnv, hooks: GUIHooks):
        self.env = env
        self.hooks = hooks

    # --- small helpers -------------------------------------------------
    def build_cfg_overrides(self) -> Dict[str, Any]:
        """
        Merge GeoPriorConfig overrides with device-level overrides.

        This mirrors what you currently do in the GUI before launching
        threads.
        """
        overrides = self.env.geo_cfg.to_cfg_overrides()
        if self.env.device_overrides:
            overrides.update(self.env.device_overrides)
        return overrides

    # (optional convenience)
    def log(self, msg: str) -> None:
        self.hooks.log(msg)

    def status(self, msg: str) -> None:
        self.hooks.status(msg)

    def progress(self, frac: float) -> None:
        self.hooks.update_progress(frac)
