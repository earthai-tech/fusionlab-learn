# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Small helper for the GeoPrior GUI.

`GeoPriorConfig` is a thin adapter around the
central NATCOM `config.py` / `config.json`
handled by :mod:`nat_utils`.

It:

* loads the current config via
  ``load_nat_config_payload``;
* exposes only the subset of fields that the
  Train tab needs;
* can turn the current GUI values into a
  ``cfg_overrides`` dict that is passed to
  ``run_stage1`` / ``run_training`` so that
  changes apply **only for this run** (the
  on-disk config remains the single source of
  truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict#, Mapping, Optional

import os

GUI_CONFIG_DIR = os.path.dirname(__file__) 

# -------------------------------------------------------------------
# Robust import of nat_utils
# -------------------------------------------------------------------

from ....utils.nat_utils import load_nat_config_payload
from ..smart_stage1 import build_stage1_cfg_from_nat
from .stage1_options import Stage1Options
# -----------------------------------------------
# Default tuner search space (GUI + NAT config)
# -----------------------------------------------
def default_tuner_search_space() -> Dict[str, Any]:
    """Return a fresh default TUNER_SEARCH_SPACE dict."""
    return {
        # --- Architecture (model.__init__) ---
        "embed_dim": [32, 48, 64],
        "hidden_units": [64, 96],
        "lstm_units": [64, 96],
        "attention_units": [32, 48],
        "num_heads": [2, 4],

        "dropout_rate": {
            "type": "float",
            "min_value": 0.05,
            "max_value": 0.20,
        },

        "vsn_units": [24, 32, 40],

        # --- Physics switches ---
        "pde_mode": ["both"],
        "scale_pde_residuals": {"type": "bool"},
        "kappa_mode": ["bar", "kb"],

        "hd_factor": {
            "type": "float",
            "min_value": 0.50,
            "max_value": 0.70,
        },

        # --- Learnable scalar initials ---
        "mv": {
            "type": "float",
            "min_value": 5e-8,
            "max_value": 3e-7,
            "sampling": "log",
        },
        "kappa": {
            "type": "float",
            "min_value": 0.8,
            "max_value": 1.2,
        },

        # --- Compile-only (model.compile) ---
        "learning_rate": {
            "type": "float",
            "min_value": 7e-5,
            "max_value": 2e-4,
            "sampling": "log",
        },

        "lambda_gw": {
            "type": "float",
            "min_value": 0.005,
            "max_value": 0.03,
        },
        "lambda_cons": {
            "type": "float",
            "min_value": 0.05,
            "max_value": 0.20,
        },
        "lambda_prior": {
            "type": "float",
            "min_value": 0.05,
            "max_value": 0.20,
        },
        "lambda_smooth": {
            "type": "float",
            "min_value": 0.005,
            "max_value": 0.05,
        },
        "lambda_mv": {
            "type": "float",
            "min_value": 0.005,
            "max_value": 0.05,
        },

        "mv_lr_mult": {
            "type": "float",
            "min_value": 0.5,
            "max_value": 2.0,
        },
        "kappa_lr_mult": {
            "type": "float",
            "min_value": 2.0,
            "max_value": 8.0,
        },
    }


@dataclass
class GeoPriorConfig:
    """
    Hold GUI-visible configuration for GeoPrior.

    Attributes
    ----------
    train_end_year :
        Last year included in the training set.
    forecast_start_year :
        First year in the forecasting window.
    forecast_horizon_years :
        Forecast length (years).
    time_steps :
        Length of the historical look-back window.

    epochs, batch_size, learning_rate :
        Core training hyper-parameters.

    pde_mode :
        Physics mode, mirrors ``PDE_MODE_CONFIG``.
    lambda_cons, lambda_gw, lambda_prior,
    lambda_smooth, lambda_mv :
        Physics loss weights.

    clean_stage1_dir :
        Whether Stage-1 should start from a clean
        run directory.
    evaluate_training :
        Whether to compute metrics on the train /
        validation sets at the end of training.
    """

    # --- temporal window ---
    train_end_year: int = 2022
    forecast_start_year: int = 2023
    forecast_horizon_years: int = 3
    time_steps: int = 5

    # --- training hyper-params ---
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4

    # --- physics configuration ---
    pde_mode: str = "off"
    lambda_cons: float = 0.10
    lambda_gw: float = 0.01
    lambda_prior: float = 0.10
    lambda_smooth: float = 0.01
    lambda_mv: float = 0.01

    # --- flags tied to nat config / Stage-1 ---
    build_future_npz: bool = False
    
    # --- flags (not part of nat config) ---
    evaluate_training: bool = True

    # --- GUI layout / window sizing (GUI-only) -------------------------
    # Design size (what you used in early screenshots)
    ui_base_width: int = 980
    ui_base_height: int = 660

    # Smallest size we allow the main window to shrink to
    ui_min_width: int = 800
    ui_min_height: int = 600

    # How much of the available screen we allow the window to occupy
    # (0.9 = at most 90% of the current monitor)
    ui_max_ratio: float = 0.90

    ui_font_scale: float = 1.0  # 1.0 = default, 1.1 = +10%, etc.

    # --- Stage-1 behaviour (GUI-only flags) -----------------------------
    clean_stage1_dir: bool = False
    stage1_auto_reuse_if_match: bool = True
    stage1_force_rebuild_if_mismatch: bool = True
    
    # The raw config dict coming from nat_utils.
    _base_cfg: Dict[str, Any] = field(
        default_factory=dict,
        repr=False,
    )
    _meta: Dict[str, Any] = field(
        default_factory=dict,
        repr=False,
    )
    
    feature_overrides: Dict[str, Any] = field(
        default_factory=dict,
        repr=False,
    )
    
    arch_overrides: Dict[str, Any] = field(     
        default_factory=dict,
        repr=False,
    )
    prob_overrides: Dict[str, Any] = field(      
        default_factory=dict,
        repr=False,
    )

    # NEW: full tuner search space that the GUI can edit and
    # push back into NAT config as TUNER_SEARCH_SPACE.
    tuner_search_space: Dict[str, Any] = field(
        default_factory=default_tuner_search_space,
        repr=False,
    )
    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_nat_config(
        cls,
    ) -> "GeoPriorConfig":
        """
        Build an instance from :mod:`nat_utils`.

        If :mod:`nat_utils` is unavailable (for
        example in a minimal test environment),
        fall back to the class defaults.
        """
        base: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}

        # Tell nat_utils to use the GUI config root
        payload = load_nat_config_payload(root=GUI_CONFIG_DIR)
        base = payload.get("config", {}) or {}
        meta = payload.get("__meta__", {}) or {}

        # Helper for safe extraction with fallback.
        def iget(key: str, default: Any) -> Any:
            if key not in base:
                return default
            val = base[key]
            # cast conservatively
            if isinstance(default, bool):       
                if isinstance(val, str):
                    v = val.strip().lower()
                    if v in {"1", "true", "yes", "on"}:
                        return True
                    if v in {"0", "false", "no", "off"}:
                        return False
                return bool(val)
            if isinstance(default, int):
                try:
                    return int(val)
                except Exception:
                    return default
            if isinstance(default, float):
                try:
                    return float(val)
                except Exception:
                    return default
            if isinstance(default, str):
                return str(val)
            return val
        # Decide tuner search space: config wins, else defaults
        space_cfg = base.get("TUNER_SEARCH_SPACE", None)
        if isinstance(space_cfg, dict) and space_cfg:
            tuner_space = space_cfg
        else:
            tuner_space = default_tuner_search_space()
            
        obj = cls(
            train_end_year=iget(
                "TRAIN_END_YEAR",
                cls.train_end_year,
            ),
            forecast_start_year=iget(
                "FORECAST_START_YEAR",
                cls.forecast_start_year,
            ),
            forecast_horizon_years=iget(
                "FORECAST_HORIZON_YEARS",
                cls.forecast_horizon_years,
            ),
            time_steps=iget(
                "TIME_STEPS",
                cls.time_steps,
            ),
            epochs=iget(
                "EPOCHS",
                cls.epochs,
            ),
            batch_size=iget(
                "BATCH_SIZE",
                cls.batch_size,
            ),
            learning_rate=iget(
                "LEARNING_RATE",
                cls.learning_rate,
            ),
            pde_mode=iget(
                "PDE_MODE_CONFIG",
                cls.pde_mode,
            ),
            lambda_cons=iget(
                "LAMBDA_CONS",
                cls.lambda_cons,
            ),
            lambda_gw=iget(
                "LAMBDA_GW",
                cls.lambda_gw,
            ),
            lambda_prior=iget(
                "LAMBDA_PRIOR",
                cls.lambda_prior,
            ),
            lambda_smooth=iget(
                "LAMBDA_SMOOTH",
                cls.lambda_smooth,
            ),
            lambda_mv=iget(
                "LAMBDA_MV",
                cls.lambda_mv,
            ),
            build_future_npz=iget(               
                "BUILD_FUTURE_NPZ",
                cls.build_future_npz,
            ),
            _base_cfg=base,
            _meta=meta,
            tuner_search_space=tuner_space,   
        )
        return obj

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def get_stage1_options(self) -> "Stage1Options":
        """
        Return a :class:`Stage1Options` view over Stage-1 flags.
        """
        
        return Stage1Options(
            clean_dir=bool(self.clean_stage1_dir),
            auto_reuse_if_match=bool(
                getattr(self, "stage1_auto_reuse_if_match", True)
            ),
            force_rebuild_if_mismatch=bool(
                getattr(self, "stage1_force_rebuild_if_mismatch", True)
            ),
        )

    def to_cfg_overrides(
        self,
    ) -> Dict[str, Any]:
        """
        Return overrides relative to the base NAT
        config.

        Only keys whose values differ from the
        original config are included.
        """
        overrides: Dict[str, Any] = {}
        base = self._base_cfg or {}

        def maybe(
            key: str,
            value: Any,
        ) -> None:
            if key not in base or base[key] != value:
                overrides[key] = value

        maybe("TRAIN_END_YEAR", self.train_end_year)
        maybe(
            "FORECAST_START_YEAR",
            self.forecast_start_year,
        )
        maybe(
            "FORECAST_HORIZON_YEARS",
            self.forecast_horizon_years,
        )
        maybe("TIME_STEPS", self.time_steps)

        maybe("EPOCHS", self.epochs)
        maybe("BATCH_SIZE", self.batch_size)
        maybe("LEARNING_RATE", self.learning_rate)

        maybe("PDE_MODE_CONFIG", self.pde_mode)
        maybe("LAMBDA_CONS", self.lambda_cons)
        maybe("LAMBDA_GW", self.lambda_gw)
        maybe("LAMBDA_PRIOR", self.lambda_prior)
        maybe("LAMBDA_SMOOTH", self.lambda_smooth)
        maybe("LAMBDA_MV", self.lambda_mv)
        
        # propagate BUILD_FUTURE_NPZ to Stage-1
        maybe("BUILD_FUTURE_NPZ", self.build_future_npz)
        
        # tuner search space
        maybe("TUNER_SEARCH_SPACE", self.tuner_search_space)
        
        # Extra overrides coming from dialogs.
        if self.feature_overrides:
            overrides.update(self.feature_overrides)
        if self.arch_overrides:                  
            overrides.update(self.arch_overrides)
        if self.prob_overrides:                  
            overrides.update(self.prob_overrides)

        return overrides
    
    def to_stage1_config(self) -> Dict[str, Any]:
        """Build a minimal Stage-1 configuration snapshot.

        This mirrors what Stage-1 would see after applying
        :meth:`to_cfg_overrides` on top of the base nat.com config,
        but keeps only the keys that
        :mod:`fusionlab.tools.app.smart_stage1` needs for
        compatibility checks (``TIME_STEPS``,
        ``FORECAST_HORIZON_YEARS``, ``TRAIN_END_YEAR``,
        ``FORECAST_START_YEAR``, ``MODE``, ``censoring``,
        ``features``, ``cols``).
        """
        base_cfg: Dict[str, Any] = self._base_cfg or {}
        overrides = self.to_cfg_overrides()
        
        return build_stage1_cfg_from_nat(
            base_cfg=base_cfg,
            overrides=overrides,
            feature_overrides=self.feature_overrides or None,
        )

    def ensure_valid(self) -> None:
        """
        Validate high-level configuration consistency.

        Raises
        ------
        ValueError
            If any setting is inconsistent.
        """
        # --- Temporal logic ---
        if self.forecast_start_year <= self.train_end_year:
            raise ValueError(
                "FORECAST_START_YEAR must be greater than TRAIN_END_YEAR "
                f"(got TRAIN_END_YEAR={self.train_end_year}, "
                f"FORECAST_START_YEAR={self.forecast_start_year})."
            )

        if self.forecast_horizon_years <= 0:
            raise ValueError(
                "FORECAST_HORIZON_YEARS must be a positive integer "
                f"(got {self.forecast_horizon_years})."
            )

        if self.time_steps <= 0:
            raise ValueError(
                "TIME_STEPS must be a positive integer "
                f"(got {self.time_steps})."
            )

        # --- Training hyper-parameters ---
        if self.epochs <= 0:
            raise ValueError(f"EPOCHS must be > 0 (got {self.epochs}).")

        if self.batch_size <= 0:
            raise ValueError(
                f"BATCH_SIZE must be > 0 (got {self.batch_size})."
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                f"LEARNING_RATE must be > 0 (got {self.learning_rate})."
            )

        # --- Physics configuration ---
        allowed_pde_modes = {
            "both",
            "on",
            "consolidation",
            "gw_flow",
            "none",
            "off",
        }
        if self.pde_mode not in allowed_pde_modes:
            raise ValueError(
                "PDE_MODE_CONFIG must be one of "
                f"{sorted(allowed_pde_modes)}, got {self.pde_mode!r}."
            )

        for name in (
            "lambda_cons",
            "lambda_gw",
            "lambda_prior",
            "lambda_smooth",
            "lambda_mv",
        ):
            val = getattr(self, name)
            if val < 0.0:
                raise ValueError(
                    f"{name.upper()} must be non-negative (got {val})."
                )

        # Basic sanity for tuner space
        if not isinstance(self.tuner_search_space, dict):
            raise ValueError("tuner_search_space must be a dict.")
            
        # --- GUI layout sanity (optional) ------------------------------
        for name in (
            "ui_base_width",
            "ui_base_height",
            "ui_min_width",
            "ui_min_height",
        ):
            val = getattr(self, name, 0)
            if val <= 0:
                raise ValueError(
                    f"{name} must be > 0 (got {val})."
                )

        r = float(getattr(self, "ui_max_ratio", 0.0))
        if not (0.0 < r <= 1.0):
            raise ValueError(
                f"ui_max_ratio must be in (0, 1], got {r}."
            )

        if (
            self.ui_min_width > self.ui_base_width
            or self.ui_min_height > self.ui_base_height
        ):
            raise ValueError(
                "ui_min_* should not exceed ui_base_* "
                f"(base=({self.ui_base_width}, {self.ui_base_height}), "
                f"min=({self.ui_min_width}, {self.ui_min_height}))."
            )

    def as_dict(self) -> Dict[str, Any]:
        """Dump the current values as a plain dict."""
        return {
            "train_end_year": self.train_end_year,
            "forecast_start_year": (
                self.forecast_start_year
            ),
            "forecast_horizon_years": (
                self.forecast_horizon_years
            ),
            "time_steps": self.time_steps,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "pde_mode": self.pde_mode,
            "lambda_cons": self.lambda_cons,
            "lambda_gw": self.lambda_gw,
            "lambda_prior": self.lambda_prior,
            "lambda_smooth": self.lambda_smooth,
            "lambda_mv": self.lambda_mv,
            "build_future_npz": (               
                self.build_future_npz
            ),
            "clean_stage1_dir": (
                self.clean_stage1_dir
            ),
            "evaluate_training": (
                self.evaluate_training
            ),
            # not strictly needed, but handy for debugging
            "tuner_search_space": self.tuner_search_space,
            
            "ui_base_width": self.ui_base_width,
            "ui_base_height": self.ui_base_height,
            "ui_min_width": self.ui_min_width,
            "ui_min_height": self.ui_min_height,
            "ui_max_ratio": self.ui_max_ratio,
            "ui_font_scale": self.ui_font_scale, 

        }
    
