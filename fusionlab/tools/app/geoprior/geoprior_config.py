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
from typing import Any, Dict, Mapping, Optional

import os

GUI_CONFIG_DIR = os.path.dirname(__file__) 

# -------------------------------------------------------------------
# Robust import of nat_utils
# -------------------------------------------------------------------
try:  # normal package layout
    from ...utils.nat_utils import (  # type: ignore
        load_nat_config_payload,
    )
except Exception:  # pragma: no cover - fallback for tests
    try:
        # direct import when running the file standalone
        from nat_utils import (  # type: ignore
            load_nat_config_payload,
        )
    except Exception:  # pragma: no cover
        load_nat_config_payload = None  # type: ignore


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

    # --- flags (not part of nat config) ---
    clean_stage1_dir: bool = False
    evaluate_training: bool = True

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

        if load_nat_config_payload is not None:
            try:
                # Tell nat_utils to use the GUI config root
                payload = load_nat_config_payload(root=GUI_CONFIG_DIR)
                base = payload.get("config", {}) or {}
                meta = payload.get("__meta__", {}) or {}
            except Exception:  # pragma: no cover
                base, meta = {}, {}
        else:
            base, meta = {}, {}
            
        # Helper for safe extraction with fallback.
        def iget(
            key: str,
            default: Any,
        ) -> Any:
            if key not in base:
                return default
            val = base[key]
            # cast conservatively
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
            _base_cfg=base,
            _meta=meta,
        )
        return obj

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
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
        
        # Extra overrides coming from the feature dialog.
        if self.feature_overrides:
            overrides.update(self.feature_overrides)

        return overrides

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
            "clean_stage1_dir": (
                self.clean_stage1_dir
            ),
            "evaluate_training": (
                self.evaluate_training
            ),
        }
