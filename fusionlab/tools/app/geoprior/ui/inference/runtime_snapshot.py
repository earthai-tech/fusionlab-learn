# geoprior/ui/inference/runtime_snapshot.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


__all__ = ["InferRuntimeSnapshot"]


@dataclass(frozen=True)
class InferRuntimeSnapshot:
    """
    Runtime-only inference state resolved from UI + preview.

    This is intentionally separate from GeoConfigStore.
    Navigator and head bar can render it without owning
    inference logic.
    """

    mode: str = "evaluate"  # or "forecast"
    dataset_key: str = "test"
    use_future: bool = False

    model_path: str = ""
    manifest_path: str = ""

    inputs_npz: str = ""
    targets_npz: str = ""

    has_targets: bool = False

    warnings: List[str] = field(default_factory=list)

    last_run_dir: str = ""
    last_eval_csv: str = ""
    last_future_csv: str = ""
    last_json: str = ""

    @property
    def warning_count(self) -> int:
        return len(self.warnings or [])

    @property
    def ready(self) -> bool:
        return self.warning_count == 0 and bool(self.model_path)
