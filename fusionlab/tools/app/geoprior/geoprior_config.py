# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Unified configuration object for the GeoPrior GUI.
#
# This module used to expose two different ``GeoPriorConfig`` classes:
#
#   * one "thin" adapter around the NATCOM ``config.py`` / ``config.json``
#     handled by :mod:`nat_utils`;
#   * another GUI-oriented dataclass carrying layout information and a few
#     high-level training knobs.
#
# Having both quickly became confusing and was the source of subtle bugs
# (for example when changing defaults in one place but not the other).
#
# The goal of this rewrite is to provide **one** canonical
# :class:`GeoPriorConfig` that:
#
#   * knows about the pieces the GUI needs (temporal window, physics
#     toggles, core training hyper-parameters, Stage-1 handshake flags,
#     basic device options, tuner search space, window layout);
#   * can still be initialised from the legacy NATCOM ``config.py`` via
#     :func:`nat_utils.load_nat_config_payload` for back-compat;
#   * can be constructed purely from in-code defaults (no ``config.json``
#     dependency) via :meth:`from_defaults`;
#   * can emit **override dictionaries** for Stage-1 and Stage-2 that are
#     compatible with the existing scripts.
#
# In practice the GUI can keep calling :meth:`GeoPriorConfig.from_nat_config`
# on startup so that Stage-1 still sees the full NATCOM defaults.
# Once Stage-1 has been fully migrated away from :mod:`nat_utils`, we can
# switch the GUI to :meth:`GeoPriorConfig.from_defaults` to remove the
# ``config.json`` dependency entirely.
#

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Directory where this GUI config lives (used as root for nat_utils).
GUI_CONFIG_DIR = os.path.dirname(__file__)

# ----------------------------------------------------------------------
# NAT / Stage-1 helpers
# ----------------------------------------------------------------------
from ....utils.nat_utils import load_nat_config_payload
from ..smart_stage1 import build_stage1_cfg_from_nat
from .stage1_options import Stage1Options

# ----------------------------------------------------------------------
# Default tuner search space
# ----------------------------------------------------------------------
def default_tuner_search_space() -> Dict[str, Any]:
    """Return a fresh default ``TUNER_SEARCH_SPACE`` dict.

    This mirrors the contents of the original NATCOM ``config.py`` but is
    kept here so that the GUI can work even if ``config.json`` is missing.
    """
    return {
        # --- Architecture (model.__init__) -----------------------------
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

        # --- Physics switches -----------------------------------------
        "pde_mode": ["both"],
        "scale_pde_residuals": {"type": "bool"},
        "kappa_mode": ["bar", "kb"],

        # Around GEOPRIOR_HD_FACTOR = 0.6
        "hd_factor": {
            "type": "float",
            "min_value": 0.50,
            "max_value": 0.70,
        },

        # --- Learnable scalar initials (model.__init__) ---------------
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

        # --- Compile-only (model.compile) -----------------------------
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


# ----------------------------------------------------------------------
# Unified GeoPriorConfig
# ----------------------------------------------------------------------
@dataclass
class GeoPriorConfig:
    """
    Hold GUI-visible configuration for GeoPrior.

    This unified dataclass plays three roles:

    * It stores the knobs that the *Train* tab exposes directly
      (temporal window, physics mode, epochs, batch size, etc.).
    * It carries a few GUI-only flags (window sizing, Stage-1 reuse
      preferences, whether to run evaluation at the end of training).
    * It keeps a cached copy of the underlying NATCOM configuration
      (``_base_cfg``) so that we can emit **overrides** relative to those
      defaults when talking to :mod:`nat_utils` / Stage-1 / Stage-2.

    Parameters are initialised either from in-code defaults
    (:meth:`from_defaults`) or from ``config.py`` via
    :meth:`from_nat_config`.
    """

    # ------------------------------------------------------------------
    # GUI / dataset-specific bits (GUI-only)
    # ------------------------------------------------------------------
    # What the GUI calls "City / dataset".  This is *not* taken from
    # NATCOM's ``CITY_NAME``; the GUI owns it.
    city: str = ""

    # Absolute path of the CSV that the user selected.
    dataset_path: Path | None = None

    # Root under which Stage-1 / training / tuning / inference folders
    # will be created.  The default mirrors the CLI behaviour but users
    # can change it from the GUI.
    results_root: Path = field(
        default_factory=lambda: Path.home() / ".fusionlab_runs"
    )

    # ------------------------------------------------------------------
    # Temporal window (mirrors NAT ``config.py`` section 1.3)
    # ------------------------------------------------------------------
    train_end_year: int = 2022
    forecast_start_year: int = 2023
    forecast_horizon_years: int = 3
    time_steps: int = 5  # historical look-back length

    # ------------------------------------------------------------------
    # Stage-1 layout & columns
    # ------------------------------------------------------------------
    mode: str = "tft_like"  # {"tft_like", "pihal_like"}

    # Column names
    time_col: str = "year"
    lon_col: str = "longitude"
    lat_col: str = "latitude"
    subs_col: str = "subsidence"
    gwl_col: str = "GWL_depth_bgs_z"
    h_field_col: str = "soil_thickness"

    # ------------------------------------------------------------------
    # Feature registry & censoring (Stage-1)
    # ------------------------------------------------------------------
    optional_numeric_features: List[List[str]] = field(
        default_factory=lambda: [
            ["rainfall_mm", "rainfall", "rain_mm", "precip_mm"],
            ["urban_load_global", "normalized_density", "urban_load"],
        ]
    )
    # make sure Any is imported at top

    optional_categorical_features: List[Any] = field(
        default_factory=lambda: [
            ["lithology", "geology"],
            "lithology_class",
        ]
    )
    already_normalized_features: List[str] = field(
        default_factory=lambda: ["urban_load_global"]
    )
    future_driver_features: List[str] = field(
        default_factory=lambda: ["rainfall_mm"]
    )

    censoring_specs: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            dict(
                col="soil_thickness",
                direction="right",
                cap=30.0,
                tol=1e-6,
                flag_suffix="_censored",
                eff_suffix="_eff",
                eff_mode="clip",
                eps=0.02,
                impute={"by": ["year"], "func": "median"},
                flag_threshold=0.5,
            ),
        ]
    )
    include_censor_flags_as_dynamic: bool = True
    use_effective_h_field: bool = True

    # ------------------------------------------------------------------
    # Training hyper-parameters (config 4.5)
    # ------------------------------------------------------------------
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4

    # ------------------------------------------------------------------
    # Physics configuration (config 4.3 & 4.4)
    # ------------------------------------------------------------------
    # GUI exposes a friendly ``pde_mode`` string; this is mapped back to
    # ``PDE_MODE_CONFIG`` in the overrides.
    pde_mode: str = "off"

    lambda_cons: float = 0.10
    lambda_gw: float = 0.01
    lambda_prior: float = 0.10
    lambda_smooth: float = 0.01
    lambda_mv: float = 0.01
    mv_lr_mult: float = 1.0
    kappa_lr_mult: float = 5.0

    geoprior_init_mv: float = 1e-7
    geoprior_init_kappa: float = 1.0
    geoprior_gamma_w: float = 9810.0
    geoprior_h_ref: float = 0.0
    geoprior_kappa_mode: str = "kb"   # {"bar", "kb"}
    geoprior_hd_factor: float = 0.6

   # --- probabilistic outputs & weights ---
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    subs_weights: Dict[float, float] = field(
        default_factory=lambda: {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
    )
    gwl_weights: Dict[float, float] = field(
        default_factory=lambda: {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}
    )
    
    # Whether Stage-1 should also pre-build future_* NPZ for Stage-3.
    build_future_npz: bool = False

    # ------------------------------------------------------------------
    # Device configuration (config 4.6)
    # ------------------------------------------------------------------
    tf_device_mode: str = "auto"         # {"auto","cpu","gpu"}
    tf_gpu_allow_growth: bool = True

    # ------------------------------------------------------------------
    # GUI-only flags
    # ------------------------------------------------------------------
    evaluate_training: bool = True

    # Stage-1 behaviour (used only by the smart Stage-1 handshake code).
    clean_stage1_dir: bool = False
    stage1_auto_reuse_if_match: bool = True
    stage1_force_rebuild_if_mismatch: bool = True

    # GUI layout / window sizing
    ui_base_width: int = 980
    ui_base_height: int = 660
    ui_min_width: int = 800
    ui_min_height: int = 600
    ui_max_ratio: float = 0.90    # max fraction of screen
    ui_font_scale: float = 1.0    # 1.0 = default, 1.1 = +10%, etc.

    # ------------------------------------------------------------------
    # Tuner configuration
    # ------------------------------------------------------------------
    tuner_max_trials: int = 20
    tuner_search_space: Dict[str, Any] = field(
        default_factory=default_tuner_search_space,
        repr=False,
    )

    # ------------------------------------------------------------------
    # Low-level NATCOM config cache + override buckets
    # ------------------------------------------------------------------
    # Raw config dict coming from :mod:`nat_utils`.  Used only so that we
    # can compute *differences* in :meth:`to_cfg_overrides`.
    _base_cfg: Dict[str, Any] = field(default_factory=dict, repr=False)
    _meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Extra overrides injected by the "Features", "Architecture" and
    # "Scalars & loss weights" dialogs.  Those are merged into the
    # overrides returned by :meth:`to_cfg_overrides`.
    feature_overrides: Dict[str, Any] = field(default_factory=dict, repr=False)
    arch_overrides: Dict[str, Any] = field(default_factory=dict, repr=False)
    prob_overrides: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_nat_config(cls) -> "GeoPriorConfig":
        """
        Build an instance from :mod:`nat_utils`.

        This reads the legacy ``config.py`` (through
        :func:`load_nat_config_payload`) and maps the relevant keys into
        the dataclass fields.  The dataset-specific pieces (city label,
        CSV path, results root) remain under the GUI's control and are
        **not** taken from the NATCOM config.
        """
        payload = load_nat_config_payload(root=GUI_CONFIG_DIR)
        base = payload.get("config", {}) or {}
        meta = payload.get("__meta__", {}) or {}

        def iget(key: str, default: Any) -> Any:
            """Internal helper: get & coerce to the type of *default*."""
            if key not in base:
                return default
            val = base[key]

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

            # Fallback: do not coerce complex objects (dict, list, ...)
            return val

        # Tuner search space: config wins, else defaults.
        space_cfg = base.get("TUNER_SEARCH_SPACE", None)
        if isinstance(space_cfg, dict) and space_cfg:
            tuner_space = space_cfg
        else:
            tuner_space = default_tuner_search_space()

        tuner_max_trials = iget("TUNER_MAX_TRIALS", cls.tuner_max_trials)

        obj = cls(
            # Temporal window -------------------------------------------------
            train_end_year=iget("TRAIN_END_YEAR", cls.train_end_year),
            forecast_start_year=iget(
                "FORECAST_START_YEAR",
                cls.forecast_start_year,
            ),
            forecast_horizon_years=iget(
                "FORECAST_HORIZON_YEARS",
                cls.forecast_horizon_years,
            ),
            time_steps=iget("TIME_STEPS", cls.time_steps),
            
            # Stage-1 layout & columns
            mode=iget("MODE", cls.mode),
            time_col=iget("TIME_COL", cls.time_col),
            lon_col=iget("LON_COL", cls.lon_col),
            lat_col=iget("LAT_COL", cls.lat_col),
            subs_col=iget("SUBSIDENCE_COL", cls.subs_col),
            gwl_col=iget("GWL_COL", cls.gwl_col),
            h_field_col=iget("H_FIELD_COL_NAME", cls.h_field_col),

            # Training hyper-params ------------------------------------------
            epochs=iget("EPOCHS", cls.epochs),
            batch_size=iget("BATCH_SIZE", cls.batch_size),
            learning_rate=iget("LEARNING_RATE", cls.learning_rate),
            # Physics configuration ------------------------------------------
            pde_mode=iget("PDE_MODE_CONFIG", cls.pde_mode),
            lambda_cons=iget("LAMBDA_CONS", cls.lambda_cons),
            lambda_gw=iget("LAMBDA_GW", cls.lambda_gw),
            lambda_prior=iget("LAMBDA_PRIOR", cls.lambda_prior),
            lambda_smooth=iget("LAMBDA_SMOOTH", cls.lambda_smooth),
            lambda_mv=iget("LAMBDA_MV", cls.lambda_mv),
            build_future_npz=iget("BUILD_FUTURE_NPZ", cls.build_future_npz),
            
            # Scalar physics parameters --------------------------------
            mv_lr_mult=iget("MV_LR_MULT", cls.mv_lr_mult),
            kappa_lr_mult=iget("KAPPA_LR_MULT", cls.kappa_lr_mult),
            geoprior_init_mv=iget(
                "GEOPRIOR_INIT_MV", cls.geoprior_init_mv
            ),
            geoprior_init_kappa=iget(
                "GEOPRIOR_INIT_KAPPA", cls.geoprior_init_kappa
            ),
            geoprior_gamma_w=iget(
                "GEOPRIOR_GAMMA_W", cls.geoprior_gamma_w
            ),
            geoprior_h_ref=iget("GEOPRIOR_H_REF", cls.geoprior_h_ref),
            geoprior_kappa_mode=iget(
                "GEOPRIOR_KAPPA_MODE", cls.geoprior_kappa_mode
            ),
            geoprior_hd_factor=iget(
                "GEOPRIOR_HD_FACTOR", cls.geoprior_hd_factor
            ),

            # Device configuration -------------------------------------------
            tf_device_mode=iget("TF_DEVICE_MODE", cls.tf_device_mode),
            tf_gpu_allow_growth=iget(
                "TF_GPU_ALLOW_GROWTH",
                cls.tf_gpu_allow_growth,
            ),
            # Tuner bits ------------------------------------------------------
            tuner_max_trials=tuner_max_trials,
            _base_cfg=base,
            _meta=meta,
            tuner_search_space=tuner_space,
        )
        
        # Feature registry & censoring: take from NAT if present,
        # otherwise keep dataclass defaults.
        obj.optional_numeric_features = base.get(
            "OPTIONAL_NUMERIC_FEATURES", 
            obj.optional_numeric_features
        )
        obj.optional_categorical_features = base.get(
            "OPTIONAL_CATEGORICAL_FEATURES",
            obj.optional_categorical_features
        )
        obj.already_normalized_features = base.get(
            "ALREADY_NORMALIZED_FEATURES",
            obj.already_normalized_features
        )
        obj.future_driver_features = base.get(
            "FUTURE_DRIVER_FEATURES", 
            obj.future_driver_features
        )
        obj.censoring_specs = base.get(
            "CENSORING_SPECS",
            obj.censoring_specs
        )
        obj.include_censor_flags_as_dynamic = base.get(
            "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC",
            obj.include_censor_flags_as_dynamic,
        )
        obj.use_effective_h_field = base.get(
            "USE_EFFECTIVE_H_FIELD",
            obj.use_effective_h_field,
        )

        obj.quantiles = base.get(
            "QUANTILES",
            obj.quantiles,
        )

        obj.subs_weights = base.get(
            "SUBS_WEIGHTS",
            obj.subs_weights,
        )

        obj.gwl_weights = base.get(
            "GWL_WEIGHTS",
            obj.gwl_weights,
        )
        
        return obj 

    @classmethod
    def from_defaults(cls) -> "GeoPriorConfig":
        """
        Build an instance using only in-code defaults.

        This does **not** touch :mod:`nat_utils` or ``config.json`` and is
        mainly intended for tests or highly customised setups.
        Note that the current smart Stage-1 handshake still expects a
        fully-populated NATCOM config, so the main GeoPrior GUI should
        keep using :meth:`from_nat_config` for now.

        With an empty ``_base_cfg`` the :meth:`to_cfg_overrides` method
        simply emits absolute values instead of "diffs" relative to a
        NAT base config.
        """
        return cls(_base_cfg={}, _meta={})

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------
    def get_stage1_options(self) -> Stage1Options:
        """
        Return a :class:`Stage1Options` view over Stage-1 flags.

        This is used by the smart Stage-1 handshake code when deciding
        whether to reuse / rebuild Stage-1 directories.
        """
        return Stage1Options(
            clean_dir=bool(self.clean_stage1_dir),
            auto_reuse_if_match=bool(self.stage1_auto_reuse_if_match),
            force_rebuild_if_mismatch=bool(
                self.stage1_force_rebuild_if_mismatch
            ),
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def to_cfg_overrides(self) -> Dict[str, Any]:
        """
        Return overrides relative to the base NAT config.

        Only keys whose values differ from ``_base_cfg`` are included.
        If ``_base_cfg`` is empty (for example when created via
        :meth:`from_defaults`), all keys are treated as overrides.

        The result is suitable for passing as ``cfg_overrides`` to
        ``run_stage1`` / :func:`run_training`.
        """
        overrides: Dict[str, Any] = {}
        base = self._base_cfg or {}

        def maybe(key: str, value: Any) -> None:
            if key not in base or base[key] != value:
                overrides[key] = value

        # Temporal window
        maybe("TRAIN_END_YEAR", self.train_end_year)
        maybe("FORECAST_START_YEAR", self.forecast_start_year)
        maybe("FORECAST_HORIZON_YEARS", self.forecast_horizon_years)
        maybe("TIME_STEPS", self.time_steps)

        # Stage-1 layout & columns
        maybe("MODE", self.mode)
        maybe("TIME_COL", self.time_col)
        maybe("LON_COL", self.lon_col)
        maybe("LAT_COL", self.lat_col)
        maybe("SUBSIDENCE_COL", self.subs_col)
        maybe("GWL_COL", self.gwl_col)
        maybe("H_FIELD_COL_NAME", self.h_field_col)
        
        # Training hyper-parameters
        maybe("EPOCHS", self.epochs)
        maybe("BATCH_SIZE", self.batch_size)
        maybe("LEARNING_RATE", self.learning_rate)

        # Physics configuration
        maybe("PDE_MODE_CONFIG", self.pde_mode)
        maybe("LAMBDA_CONS", self.lambda_cons)
        maybe("LAMBDA_GW", self.lambda_gw)
        maybe("LAMBDA_PRIOR", self.lambda_prior)
        maybe("LAMBDA_SMOOTH", self.lambda_smooth)
        maybe("LAMBDA_MV", self.lambda_mv)

        # Scalar physics parameters
        maybe("MV_LR_MULT", self.mv_lr_mult)
        maybe("KAPPA_LR_MULT", self.kappa_lr_mult)
        maybe("GEOPRIOR_INIT_MV", self.geoprior_init_mv)
        maybe("GEOPRIOR_INIT_KAPPA", self.geoprior_init_kappa)
        maybe("GEOPRIOR_GAMMA_W", self.geoprior_gamma_w)
        maybe("GEOPRIOR_H_REF", self.geoprior_h_ref)
        maybe("GEOPRIOR_KAPPA_MODE", self.geoprior_kappa_mode)
        maybe("GEOPRIOR_HD_FACTOR", self.geoprior_hd_factor)
        
        # Stage-1 extras
        maybe("BUILD_FUTURE_NPZ", self.build_future_npz)

        # Feature registry & censoring
        maybe("OPTIONAL_NUMERIC_FEATURES", self.optional_numeric_features)
        maybe("OPTIONAL_CATEGORICAL_FEATURES", self.optional_categorical_features)
        maybe("ALREADY_NORMALIZED_FEATURES", self.already_normalized_features)
        maybe("FUTURE_DRIVER_FEATURES", self.future_driver_features)
        maybe("CENSORING_SPECS", self.censoring_specs)
        maybe("INCLUDE_CENSOR_FLAGS_AS_DYNAMIC",
              self.include_censor_flags_as_dynamic)
        maybe("USE_EFFECTIVE_H_FIELD", self.use_effective_h_field)
        
        # Device configuration
        maybe("TF_DEVICE_MODE", self.tf_device_mode)
        maybe("TF_GPU_ALLOW_GROWTH", self.tf_gpu_allow_growth)

        # Tuner bits
        maybe("TUNER_MAX_TRIALS", self.tuner_max_trials)
        maybe("TUNER_SEARCH_SPACE", self.tuner_search_space)
        
        # Probabilistic
        maybe("QUANTILES", self.quantiles)
        maybe("SUBS_WEIGHTS", self.subs_weights)
        maybe("GWL_WEIGHTS", self.gwl_weights)
        
        # Extra overrides coming from dialogs.
        if self.feature_overrides:
            overrides.update(self.feature_overrides)
        if self.arch_overrides:
            overrides.update(self.arch_overrides)
        if self.prob_overrides:
            overrides.update(self.prob_overrides)

        return overrides

    def to_stage1_config(self) -> Dict[str, Any]:
        """
        Build a minimal Stage-1 configuration snapshot.

        This mirrors what Stage-1 would see after applying
        :meth:`to_cfg_overrides` on top of the base NATCOM config, but
        keeps only the keys that
        :mod:`fusionlab.tools.app.smart_stage1` needs for compatibility
        checks (``TIME_STEPS``, ``FORECAST_HORIZON_YEARS``,
        ``TRAIN_END_YEAR``, ``FORECAST_START_YEAR``, ``MODE``,
        ``censoring``, ``features``, ``cols``).
        """
        base_cfg: Dict[str, Any] = self._base_cfg or {}
        overrides = self.to_cfg_overrides()

        return build_stage1_cfg_from_nat(
            base_cfg=base_cfg,
            overrides=overrides,
            feature_overrides=self.feature_overrides or None,
        )

    # Small alias: some callers may prefer the shorter name.
    def to_stage1_cfg(self, pure: bool = False) -> Dict[str, Any]:
        """
        Convenience alias for building a Stage-1 configuration snapshot.

        Parameters
        ----------
        pure : bool, default=False
            If True, call :meth:`to_stage1_cfg_pure` (no NATCOM
            dependency). If False, call :meth:`to_stage1_config`
            which still uses :func:`build_stage1_cfg_from_nat`.
        """
        if pure:
            return self.to_stage1_cfg_pure()
        
        return self.to_stage1_config()

    def to_stage1_cfg_pure(self) -> Dict[str, Any]:
        """
        Build a Stage-1 configuration dict using only the dataclass
        fields.

        Unlike :meth:`to_stage1_config`, this does **not** call
        :func:`build_stage1_cfg_from_nat` and therefore does not depend
        on the legacy NATCOM ``config.py`` / ``config.json``. It is
        useful for tests or for a fully GUI-driven stack.

        The structure mirrors what the smart Stage-1 handshake expects
        for compatibility checks:

        - top-level temporal keys:
          ``TIME_STEPS``, ``FORECAST_HORIZON_YEARS``,
          ``TRAIN_END_YEAR``, ``FORECAST_START_YEAR``, ``MODE``,
          ``BUILD_FUTURE_NPZ``
        - a ``cols`` block for column names;
        - a ``features`` block for the feature registry;
        - a ``censoring`` block for censoring configuration.

        Returns
        -------
        dict
            Minimal Stage-1 configuration snapshot built purely from
            :class:`GeoPriorConfig` fields.
        """
        return {
            # --- temporal window -----------------------------------------
            "TIME_STEPS": self.time_steps,
            "FORECAST_HORIZON_YEARS": self.forecast_horizon_years,
            "TRAIN_END_YEAR": self.train_end_year,
            "FORECAST_START_YEAR": self.forecast_start_year,
            "MODE": self.mode,
            "BUILD_FUTURE_NPZ": self.build_future_npz,

            # --- column names --------------------------------------------
            "cols": {
                "TIME_COL": self.time_col,
                "LON_COL": self.lon_col,
                "LAT_COL": self.lat_col,
                "SUBSIDENCE_COL": self.subs_col,
                "GWL_COL": self.gwl_col,
                "H_FIELD_COL_NAME": self.h_field_col,
            },

            # --- feature registry ---------------------------------------
            "features": {
                "OPTIONAL_NUMERIC_FEATURES":
                    self.optional_numeric_features,
                "OPTIONAL_CATEGORICAL_FEATURES":
                    self.optional_categorical_features,
                "ALREADY_NORMALIZED_FEATURES":
                    self.already_normalized_features,
                "FUTURE_DRIVER_FEATURES":
                    self.future_driver_features,
            },

            # --- censoring configuration --------------------------------
            "censoring": {
                "CENSORING_SPECS": self.censoring_specs,
                "INCLUDE_CENSOR_FLAGS_AS_DYNAMIC":
                    self.include_censor_flags_as_dynamic,
                "USE_EFFECTIVE_H_FIELD": self.use_effective_h_field,
            },
        }

    # ------------------------------------------------------------------
    # Validation & dumping
    # ------------------------------------------------------------------
    def ensure_valid(self) -> None:
        """
        Validate high-level configuration consistency.

        Raises
        ------
        ValueError
            If any setting is inconsistent.
        """
        # --- Temporal logic -------------------------------------------
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

        # --- Training hyper-parameters --------------------------------
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

        # --- Physics configuration ------------------------------------
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
                "PDE_MODE_CONFIG / pde_mode must be one of "
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

        allowed_modes = {"tft_like", "pihal_like"}
        if self.mode not in allowed_modes:
            raise ValueError(
                f"MODE must be one of {sorted(allowed_modes)}, "
                f"got {self.mode!r}."
            )

        # --- GUI layout sanity ----------------------------------------
        for name in (
            "ui_base_width",
            "ui_base_height",
            "ui_min_width",
            "ui_min_height",
        ):
            val = getattr(self, name, 0)
            if val <= 0:
                raise ValueError(f"{name} must be > 0 (got {val}).")

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

        # Scalar physics checks
        if self.mv_lr_mult < 0.0 or self.kappa_lr_mult < 0.0:
            raise ValueError(
                "MV_LR_MULT and KAPPA_LR_MULT must be non-negative."
            )

        if self.geoprior_init_mv <= 0.0:
            raise ValueError(
                f"GEOPRIOR_INIT_MV must be > 0 (got {self.geoprior_init_mv})."
            )

        if self.geoprior_init_kappa <= 0.0:
            raise ValueError(
                "GEOPRIOR_INIT_KAPPA must be > 0 "
                f"(got {self.geoprior_init_kappa})."
            )

        if self.geoprior_gamma_w <= 0.0:
            raise ValueError(
                f"GEOPRIOR_GAMMA_W must be > 0 (got {self.geoprior_gamma_w})."
            )

        if not (0.0 <= self.geoprior_hd_factor <= 1.0):
            raise ValueError(
                "GEOPRIOR_HD_FACTOR must be in [0, 1], "
                f"got {self.geoprior_hd_factor}."
            )

        if self.geoprior_kappa_mode not in {"bar", "kb"}:
            raise ValueError(
                "GEOPRIOR_KAPPA_MODE must be 'bar' or 'kb', "
                f"got {self.geoprior_kappa_mode!r}."
            )

        # --- Tuner search space sanity --------------------------------
        if not isinstance(self.tuner_search_space, dict):
            raise ValueError("tuner_search_space must be a dict.")

    def as_dict(self) -> Dict[str, Any]:
        """Dump the current values as a plain dictionary."""
        return {
            # GUI / dataset
            "city": self.city,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "results_root": str(self.results_root),
            
            "mode": self.mode,
            "time_col": self.time_col,
            "lon_col": self.lon_col,
            "lat_col": self.lat_col,
            "subs_col": self.subs_col,
            "gwl_col": self.gwl_col,
            "h_field_col": self.h_field_col,
            "optional_numeric_features": self.optional_numeric_features,
            "optional_categorical_features": self.optional_categorical_features,
            "already_normalized_features": self.already_normalized_features,
            "future_driver_features": self.future_driver_features,
            "censoring_specs": self.censoring_specs,
            "include_censor_flags_as_dynamic": (
                self.include_censor_flags_as_dynamic
            ),
            "use_effective_h_field": self.use_effective_h_field,

            # Temporal window
            "train_end_year": self.train_end_year,
            "forecast_start_year": self.forecast_start_year,
            "forecast_horizon_years": self.forecast_horizon_years,
            "time_steps": self.time_steps,
            # Training hyper-params
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            # Physics configuration
            "pde_mode": self.pde_mode,
            "lambda_cons": self.lambda_cons,
            "lambda_gw": self.lambda_gw,
            "lambda_prior": self.lambda_prior,
            "lambda_smooth": self.lambda_smooth,
            "lambda_mv": self.lambda_mv,
            "mv_lr_mult": self.mv_lr_mult,
            "kappa_lr_mult": self.kappa_lr_mult,
            "geoprior_init_mv": self.geoprior_init_mv,
            "geoprior_init_kappa": self.geoprior_init_kappa,
            "geoprior_gamma_w": self.geoprior_gamma_w,
            "geoprior_h_ref": self.geoprior_h_ref,
            "geoprior_kappa_mode": self.geoprior_kappa_mode,
            "geoprior_hd_factor": self.geoprior_hd_factor,
            "build_future_npz": self.build_future_npz,
            # Device configuration
            "tf_device_mode": self.tf_device_mode,
            "tf_gpu_allow_growth": self.tf_gpu_allow_growth,
            # Stage-1 / evaluation flags
            "clean_stage1_dir": self.clean_stage1_dir,
            "stage1_auto_reuse_if_match": self.stage1_auto_reuse_if_match,
            "stage1_force_rebuild_if_mismatch": (
                self.stage1_force_rebuild_if_mismatch
            ),
            "evaluate_training": self.evaluate_training,
            # Tuner bits
            "tuner_max_trials": self.tuner_max_trials,
            "tuner_search_space": self.tuner_search_space,
            # GUI layout
            "ui_base_width": self.ui_base_width,
            "ui_base_height": self.ui_base_height,
            "ui_min_width": self.ui_min_width,
            "ui_min_height": self.ui_min_height,
            "ui_max_ratio": self.ui_max_ratio,
            "ui_font_scale": self.ui_font_scale,
        }
