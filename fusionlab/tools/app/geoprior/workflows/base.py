# geoprior/workflows/base.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

from ..config import GeoPriorConfig


class StoreLike(Protocol):
    """Minimal store surface required by workflows."""

    @property
    def cfg(self) -> GeoPriorConfig: ...

    def snapshot_overrides(self) -> Dict[str, Any]: ...


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

    v3.2 prefers `store` as the single source of truth.
    v3.0 can still pass `geo_cfg` without a store.

    Rules
    -----
    - If `store` is provided -> it is primary.
    - Else if `geo_cfg` is provided -> use it.
    - Else -> fall back to a default GeoPriorConfig().
    """

    gui_runs_root: Path

    # v3.0 compat
    geo_cfg: Optional[GeoPriorConfig] = None

    # v3.2 primary
    store: Optional[StoreLike] = None

    device_overrides: Dict[str, Any] = field(
        default_factory=dict
    )
    dry_mode: bool = False

    def __post_init__(self) -> None:
        # Migration safety: keep legacy access working.
        if self.geo_cfg is None:
            if self.store is not None:
                self.geo_cfg = self.store.cfg
            else:
                self.geo_cfg = GeoPriorConfig()
                
    def has_store(self) -> bool:
        return self.store is not None

    def resolve_cfg(self) -> GeoPriorConfig:
        if self.store is not None:
            return self.store.cfg
        # geo_cfg is guaranteed non-None after __post_init__
        return self.geo_cfg  # type: ignore[return-value]

    def resolve_cfg_overrides(self) -> Dict[str, Any]:
        """
        Get GeoPrior overrides from the preferred source.

        - Store path uses store.snapshot_overrides().
        - Config path uses GeoPriorConfig.to_cfg_overrides().
        """
    #     if self.store is not None:
    #         try:
    #             return dict(self.store.snapshot_overrides())
    #         except Exception:
    #             # Fallback to cfg if store is present but
    #             # snapshot fails for any reason.
    #             return dict(self.store.cfg.to_cfg_overrides())
    #     cfg = self.resolve_cfg()
    #     return dict(cfg.to_cfg_overrides())
    
    # def resolve_cfg_overrides(self) -> Dict[str, Any]:
        if self.store is not None:
            try:
                return dict(self.store.snapshot_overrides())
            except Exception:
                # Fallback to cfg if store is present but
                # snapshot fails for any reason.
                return dict(self.store.cfg.to_cfg_overrides())

        # geo_cfg is guaranteed non-None after __post_init__
        return dict(self.geo_cfg.to_cfg_overrides())  # type: ignore[union-attr]



class BaseWorkflowController:
    """
    Common base for Train / Tune / Infer / Xfer controllers.

    Provides a single place to build cfg_overrides and apply
    device-level overrides.
    """

    def __init__(self, env: RunEnv, hooks: GUIHooks):
        self.env = env
        self.hooks = hooks

    # --- small helpers -------------------------------------------------
    def build_cfg_overrides(self) -> Dict[str, Any]:
        """
        Merge config/store overrides with device-level overrides.

        Store is preferred when available.
        """
        overrides = self.env.resolve_cfg_overrides()
        dev = self.env.device_overrides or {}
        if dev:
            overrides.update(dev)
            
        return overrides
    
    def resolve_cfg(self) -> GeoPriorConfig:
        """Convenience for jobs needing the live config."""
        return self.env.resolve_cfg()

    # (optional convenience)
    def log(self, msg: str) -> None:
        self.hooks.log(msg)

    def status(self, msg: str) -> None:
        self.hooks.status(msg)

    def progress(self, frac: float) -> None:
        self.hooks.update_progress(frac)
