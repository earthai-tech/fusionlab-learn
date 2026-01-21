# geoprior/ui/console/kinds.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KindSpec:
    kind: str
    label: str
    prefix: str


K_PREP = KindSpec("preprocess", "Preprocess", "PREP")
K_S1 = KindSpec("stage1", "Stage-1", "S1")
K_TR = KindSpec("train", "Train", "TR")
K_TU = KindSpec("tune", "Tune", "TU")
K_IN = KindSpec("infer", "Inference", "IN")
K_XF = KindSpec("xfer", "Transfer", "XF")
K_XV = KindSpec("xfer_view", "Xfer View", "XV")

K_MAIN = KindSpec("main", "Main", "MAIN")
K_ALL = KindSpec("all", "All", "ALL")

KINDS = {
    K_PREP.kind: K_PREP,
    K_S1.kind: K_S1,
    K_TR.kind: K_TR,
    K_TU.kind: K_TU,
    K_IN.kind: K_IN,
    K_XF.kind: K_XF,
    K_XV.kind: K_XV,
    K_MAIN.kind: K_MAIN,
    K_ALL.kind: K_ALL,
}
