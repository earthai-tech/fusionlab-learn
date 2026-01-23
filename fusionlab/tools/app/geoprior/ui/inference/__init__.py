# geoprior/ui/inference/__init__.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Inference UI package.

Layout targets (Train-like):
[A] left navigator + [E] extras
[B] head bar (global actions)
[C] center cards (edit disclosures)
[D] preview/review (plan + readiness)
bottom: status + Run

This package is UI-only:
- widgets + store bindings
- emits signals for controller (app.py)
"""

from .tab import InferenceTab

__all__ = ["InferenceTab"]
