# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
GeoPrior GUI styles package.

This package centralises Qt stylesheets and palette constants.
It keeps backward compatibility with the former `styles.py`
module by re-exporting its public API explicitly from `._styles`.

Public API:
- Legacy exports (from `._styles`) so old imports keep working
  (e.g. SECONDARY_TBLUE, FLAB_STYLE_SHEET, etc.).
- `PREP_PATCH_LIGHT`, `PREP_PATCH_DARK` from preprocess patches.
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
# Legacy styles (former styles.py -> _styles.py)
# ------------------------------------------------------------------ #

from ._styles import (
    PRIMARY,
    SECONDARY,
    BG_LIGHT,
    FG_DARK,
    PRIMARY_T75,
    SECONDARY_T70,
    SECONDARY_TBLUE,
    INFERENCE_ON,
    INFERENCE_OFF,
    PALETTE,
    MODE_DRY_COLOR,
    MODE_TRAIN_COLOR,
    MODE_TUNE_COLOR,
    MODE_INFER_COLOR,
    MODE_XFER_COLOR,
    MODE_RESULTS_COLOR,
    MODE_DATA_COLOR,
    MODE_SETUP_COLOR,
    MODE_PREPROCESS_COLOR,
    MODE_MAP_COLOR,
    MODE_TOOLS_COLOR,
    RUN_BUTTON_IDLE,
    RUN_BUTTON_HOVER,
    RUN_BUTTON_DISABLED,
    FLAB_STYLE_SHEET,
    DARK_THEME_STYLESHEET,
    TAB_STYLES,
    LOG_STYLES,
    ERROR_STYLES,
    TUNER_STYLES,
    TUNER_DIALOG_STYLES,
    MAIN_TAB_STYLES_LIGHT,
    MAIN_TAB_STYLES_DARK,
    SEARCH_STYLE,
    TRAIN_TAB_PATCH_LIGHT,
    TRAIN_TAB_PATCH_DARK,
    TRAIN_COMP_SCROLL_LIGHT,
    TRAIN_COMP_SCROLL_DARK,
    TRAIN_NAV_LIGHT,
    TRAIN_NAV_DARK,
    TRAIN_NAV_ROW,
    INF_COMP_SCROLL_LIGHT,
    INF_COMP_SCROLL_DARK,
    INFER_CHIP,
    _CONSOLE_STYLES_DARK,
    _CONSOLE_STYLES_LIGHT, 
    _DOCK_CHROME_LIGHT, 
    _DOCK_CHROME_DARK
    
)

# ------------------------------------------------------------------ #
# Preprocess patches (prep_styles.py -> _prep_styles.py)
# ------------------------------------------------------------------ #

from ._prep_styles import PREP_PATCH_DARK, PREP_PATCH_LIGHT
from ._xfer_advsec import XFER_ADVSEC_LIGHT, XFER_ADVSEC_DARK

FLAB_STYLE_SHEET = ( 
    FLAB_STYLE_SHEET 
    + XFER_ADVSEC_LIGHT
    + _DOCK_CHROME_LIGHT
    + _CONSOLE_STYLES_LIGHT 
    + INF_COMP_SCROLL_LIGHT
    + TRAIN_TAB_PATCH_LIGHT
    + TRAIN_NAV_LIGHT 
    + TRAIN_NAV_ROW 
    + SEARCH_STYLE 
    + INFER_CHIP 
    + TRAIN_COMP_SCROLL_LIGHT
    + PREP_PATCH_LIGHT
)


DARK_THEME_STYLESHEET = (
    DARK_THEME_STYLESHEET 
    + XFER_ADVSEC_DARK
    + _DOCK_CHROME_DARK
    + _CONSOLE_STYLES_DARK 
    + INF_COMP_SCROLL_DARK
    + TRAIN_NAV_DARK
    + TRAIN_TAB_PATCH_DARK
    + TRAIN_NAV_ROW 
    + SEARCH_STYLE 
    + INFER_CHIP 
    + TRAIN_COMP_SCROLL_DARK
    + PREP_PATCH_DARK

)


__all__ = [
    # palette + base constants
    "PRIMARY",
    "SECONDARY",
    "BG_LIGHT",
    "FG_DARK",
    "PRIMARY_T75",
    "SECONDARY_T70",
    "SECONDARY_TBLUE",
    "INFERENCE_ON",
    "INFERENCE_OFF",
    "PALETTE",
    # mode colors
    "MODE_DRY_COLOR",
    "MODE_TRAIN_COLOR",
    "MODE_TUNE_COLOR",
    "MODE_INFER_COLOR",
    "MODE_XFER_COLOR",
    "MODE_RESULTS_COLOR",
    "MODE_DATA_COLOR",
    "MODE_SETUP_COLOR",
    "MODE_PREPROCESS_COLOR",
    "MODE_MAP_COLOR",
    "MODE_TOOLS_COLOR",
    # run button colors
    "RUN_BUTTON_IDLE",
    "RUN_BUTTON_HOVER",
    "RUN_BUTTON_DISABLED",
    # main stylesheets
    "FLAB_STYLE_SHEET",
    "DARK_THEME_STYLESHEET",
    "TAB_STYLES",
    "LOG_STYLES",
    "ERROR_STYLES",
    "TUNER_STYLES",
    "TUNER_DIALOG_STYLES",
    "MAIN_TAB_STYLES_LIGHT",
    "MAIN_TAB_STYLES_DARK",
    # misc building blocks (still useful to import directly)
    "SEARCH_STYLE",
    "TRAIN_TAB_PATCH_LIGHT",
    "TRAIN_TAB_PATCH_DARK",
    "TRAIN_COMP_SCROLL_LIGHT",
    "TRAIN_COMP_SCROLL_DARK",
    "TRAIN_NAV_LIGHT",
    "TRAIN_NAV_DARK",
    "TRAIN_NAV_ROW",
    "INF_COMP_SCROLL_LIGHT",
    "INF_COMP_SCROLL_DARK",
    "INFER_CHIP",
    # preprocess patches
    "PREP_PATCH_LIGHT",
    "PREP_PATCH_DARK",
]
