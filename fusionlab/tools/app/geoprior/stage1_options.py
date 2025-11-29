# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class Stage1Options:
    """
    High-level behaviour flags for Stage-1 management.

    These are GUI-level hints that control how the smart Stage-1
    handshake behaves (auto-reuse / auto-rebuild / cleaning).

    Parameters
    ----------
    clean_dir : bool, default=False
        If True, remove the entire
        ``<city>_GeoPriorSubsNet_stage1/`` directory for the
        next training run before (re)building Stage-1.

    auto_reuse_if_match : bool, default=True
        If True, when a complete Stage-1 run exists whose
        manifest matches the current GUI configuration, reuse
        it automatically (no dialog).

    force_rebuild_if_mismatch : bool, default=True
        If True, when no compatible Stage-1 run exists
        (config mismatch), rebuild Stage-1 automatically
        instead of prompting the user.
    """

    clean_dir: bool = False
    auto_reuse_if_match: bool = True
    force_rebuild_if_mismatch: bool = True

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "Stage1Options":
        """
        Build from a flat config mapping (optional).

        Expected keys (all optional)
        ----------------------------
        STAGE1_CLEAN_DIR : bool
        STAGE1_AUTO_REUSE_IF_MATCH : bool
        STAGE1_FORCE_REBUILD_IF_MISMATCH : bool
        """
        return cls(
            clean_dir=bool(cfg.get("STAGE1_CLEAN_DIR", False)),
            auto_reuse_if_match=bool(
                cfg.get("STAGE1_AUTO_REUSE_IF_MATCH", True)
            ),
            force_rebuild_if_mismatch=bool(
                cfg.get(
                    "STAGE1_FORCE_REBUILD_IF_MISMATCH",
                    True,
                )
            ),
        )

    def to_cfg_overrides(
        self,
        base: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Convert back to a flat overrides dict (NAT style).

        Only keys whose values differ from the base mapping
        (or from the dataclass defaults if base is empty)
        are returned.
        """
        base = dict(base or {})
        overrides: Dict[str, Any] = {}

        def maybe(key: str, value: Any, default: Any) -> None:
            if base.get(key, default) != value:
                overrides[key] = value

        maybe("STAGE1_CLEAN_DIR", self.clean_dir, False)
        maybe("STAGE1_AUTO_REUSE_IF_MATCH", self.auto_reuse_if_match, True)
        maybe(
            "STAGE1_FORCE_REBUILD_IF_MISMATCH",
            self.force_rebuild_if_mismatch,
            True,
        )
        return overrides
