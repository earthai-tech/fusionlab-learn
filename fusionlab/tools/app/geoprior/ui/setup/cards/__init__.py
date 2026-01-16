# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""geoprior.ui.setup.cards

Card widgets used by the Setup tab.

Each card is a small, store-driven UI component. New cards should live
here to keep `setup.panel` slim.
"""

from __future__ import annotations

from .base import CardBase
from .summary import SummaryCard

__all__ = [
    "CardBase",
    "SummaryCard",
]
