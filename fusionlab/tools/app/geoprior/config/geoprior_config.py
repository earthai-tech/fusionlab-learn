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
from typing import Any, Dict, List, Optional, Tuple, Mapping

# Directory where this GUI config lives (used as root for nat_utils).
GUI_CONFIG_DIR = os.path.dirname(__file__)

# ----------------------------------------------------------------------
# NAT / Stage-1 helpers
# ----------------------------------------------------------------------
from .....utils.nat_utils import load_nat_config_payload
from .smart_stage1 import build_stage1_cfg_from_nat
from .stage1_options import Stage1Options

# ----------------------------------------------------------------------
# Default tuner search space
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Default tuner search space
# ----------------------------------------------------------------------
def default_tuner_search_space(
    offset_mode: str = "mul",
) -> Dict[str, Any]:
    """Return a fresh default ``TUNER_SEARCH_SPACE`` dict.

    The GUI uses this fallback when a legacy NATCOM payload does not
    provide a ``TUNER_SEARCH_SPACE`` entry.

    Parameters
    ----------
    offset_mode : str, default="mul"
        Controls the sampling range for ``lambda_offset`` in v3.2.

        - ``"mul"``: ``physics_mult = lambda_offset`` (must be > 0).
        - ``"log10"``: ``physics_mult = 10 ** lambda_offset`` (can be < 0).

    Notes
    -----
    The returned dictionary mirrors the v3.2 NATCOM tuning space:
    ``TUNER_SEARCH_SPACE_BASE_V32`` plus an offset-mode-aware
    ``lambda_offset`` so that the effective physics multiplier spans
    roughly ``[0.1, 50]`` in both regimes.

    """
    space: Dict[str, Any] = {
        # ----------------------------
        # Architecture (model.__init__)
        # ----------------------------
        "embed_dim": [32, 48, 64],
        "hidden_units": [64, 96, 128],
        "lstm_units": [64, 96],
        "attention_units": [32, 48],
        "num_heads": [2, 4],
        "dropout_rate": {
            "type": "float",
            "min_value": 0.05,
            "max_value": 0.25,
            "step": 0.05,
        },
        "max_window_size": [8, 10, 12],
        "memory_size": [50, 100],
        # ----------------------------
        # Optimizer / training
        # ----------------------------
        # VSN & BatchNorm are *not* tuned here:
        #   USE_VSN       = True
        #   USE_BATCH_NORM= False
        # We keep them fixed via the main config, because
        # the preprocessing + scaling is designed for that.
        #
        # Still allow some variation of VSN width:
        "vsn_units": [24, 32, 40],

        # --- Physics switches ---
        #
        # Always keep full physics active by default.
        "pde_mode": ["both"],

        "scale_pde_residuals": {"type": "bool"},

        # Config default is "kb", but we can still let tuner
        # choose between bar/kb if useful.
        "kappa_mode": ["bar", "kb"],

        # Around GEOPRIOR_HD_FACTOR = 0.6
        "hd_factor": {
            "type": "float",
            "min_value": 0.50,
            "max_value": 0.70,
        },

        # --- Learnable scalar initials (model.__init__) ---
        #
        # Around GEOPRIOR_INIT_MV = 1e-7
        "mv": {
            "type": "float",
            "min_value": 5e-8,
            "max_value": 3e-7,
            "sampling": "log",
        },

        # Around GEOPRIOR_INIT_KAPPA = 1.0
        "kappa": {
            "type": "float",
            "min_value": 0.8,
            "max_value": 1.2,
        },
        
        "learning_rate": {
            "type": "float",
            "min_value": 3e-4,
            "max_value": 3e-3,
            "sampling": "log",
        },
        
        "attention_levels": [ 
                ['cross', 'memory', 'hierachical']
            ], 
        "scales": [ 
            # [1], 
            [1, 2]
        ], 
        # ----------------------------
        # Physics loss weights (compile-time)
        # ----------------------------
        "lambda_gw": {
            "type": "float",
            "min_value": 0.01,
            "max_value": 0.20,
        },
        "lambda_cons": {
            "type": "float",
            "min_value": 0.05,
            "max_value": 0.30,
        },
        "lambda_prior": {
            "type": "float",
            "min_value": 0.02,
            "max_value": 0.30,
        },
        "lambda_smooth": {
            "type": "float",
            "min_value": 1e-5,
            "max_value": 5e-3,
            "sampling": "log",
        },
        "lambda_bounds": {
            "type": "float",
            "min_value": 1e-4,
            "max_value": 1.0,
            "sampling": "log",
        },
        "lambda_mv": {
            "type": "float",
            "min_value": 0.0,
            "max_value": 0.05,
        },
        "lambda_q": {
            "type": "choice",
            "values": [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
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
        "scale_mv_with_offset": {
            "type": "choice",
            "values": [False, True],
        },
        "scale_q_with_offset": {
            "type": "choice",
            "values": [True, False],
        },
    }

    mode = str(offset_mode or "mul").lower()
    if mode == "mul":
        space["lambda_offset"] = {
            "type": "float",
            "min_value": 0.1,
            "max_value": 50.0,
            "sampling": "log",
        }
    else:
        space["lambda_offset"] = {
            "type": "float",
            "min_value": -1.0,
            "max_value": 1.7,
            "step": 0.05,
        }

    return space
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

    # v3.2: groundwater representation and surface elevation.
    # - GWL_KIND defines whether the input column is a depth (BGS)
    #   or a hydraulic head already.
    # - GWL_SIGN declares the sign convention in the raw dataset.
    gwl_kind: str = "depth_bgs"          # {"depth_bgs","head"}
    gwl_sign: str = "down_positive"      # {"down_positive","up_positive"}
    use_head_proxy: bool = False

    # Optional surface elevation column used to build head:
    #   head = z_surf - z_gwl (depth-bgs, down_positive)
    z_surf_col: Optional[str] = None
    include_z_surf_as_static: bool = True
    head_col: str = "head_m"
    gwl_dyn_index: Optional[int] = None

    # Stage-1 scaling controls (v3.2)
    normalize_coords: bool = True
    keep_coords_raw: bool = False
    shift_raw_coords: bool = True
    scale_h_field: bool = False
    scale_gwl: bool = False
    scale_z_surf: bool = False
    subsidence_kind: str = "cumulative"  # {"cumulative","rate"}


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
    dynamic_driver_features: List[str] = field(
        default_factory=lambda: ['GWL_depth_bgs_z']
    )
    static_driver_features: List[str] = field(
        default_factory=lambda: ['lithology']
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

    # --- near feature registry fields ---
    dynamic_feature_names: Optional[List[str]] = None
    future_feature_names: Optional[List[str]] = None
    
    include_censor_flags_as_future: bool = False

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

    # v3.2: physics warmup / ramp (stage-2)
    physics_warmup_steps: int = 500
    physics_ramp_steps: int = 500
    scale_pde_residuals: bool = True

    lambda_cons: float = 0.10
    lambda_gw: float = 0.01
    lambda_prior: float = 0.10
    lambda_smooth: float = 0.01
    lambda_mv: float = 0.01
    lambda_bounds: float = 1.0
    lambda_q: float = 0.0

    # v3.2: global physics-loss offset
    offset_mode: str = "mul"          # {"mul","log10"}
    lambda_offset: float = 1.0
    use_lambda_offset_scheduler: bool = True
    lambda_offset_unit: str = "epoch"  # {"epoch","step"}
    lambda_offset_when: str = "begin"  # {"begin","end"}
    lambda_offset_warmup: int = 20
    lambda_offset_start: float = 0.1
    lambda_offset_end: float = 1.0
    lambda_offset_schedule: Optional[Dict[int, float]] = None

    mv_lr_mult: float = 1.0
    kappa_lr_mult: float = 5.0

    geoprior_init_mv: float = 1e-7
    geoprior_init_kappa: float = 1.0
    geoprior_gamma_w: float = 9810.0
    geoprior_h_ref: float = 0.0
    geoprior_kappa_mode: str = "kb"   # {"bar", "kb"}
    geoprior_hd_factor: float = 0.6

    # ------------------------------------------------------------------
    # v3.2: bounds, units and source-term controls
    # ------------------------------------------------------------------
    physics_bounds: Dict[str, float] = field(
        default_factory=lambda: {
            "K_min": 1e-9,
            "K_max": 1e-3,
            "Ss_min": 1e-7,
            "Ss_max": 1e-2,
            "tau_min": 1e5,
            "tau_max": 1e10,
            "H_min": 5.0,
            "H_max": 500.0,
        }
    )
    bounds_mode: str = "soft"          # {"soft","hard"}
    time_units: str = "year"           # {"year","day","second"}

    # SI affine mapping (used by scaling helpers)
    subs_unit_to_si: float = 1e-3
    subs_scale_si: float = 1.0
    subs_bias_si: float = 0.0
    head_unit_to_si: float = 1.0
    head_scale_si: Optional[float] = None
    head_bias_si: Optional[float] = None
    thickness_unit_to_si: float = 1.0
    z_surf_unit_to_si: float = 1.0
    h_field_min_si: float = 0.1
    auto_si_affine_from_stage1: bool = True

    # Coordinate interpretation (stage-1 / scaling)
    coord_mode: str = "degrees"
    utm_epsg: int = 32649
    coord_src_epsg: int = 4326

    # Consolidation & groundwater residual scaling
    residual_method: str = "exact"     # {"exact","fd"}
    cons_residual_units: str = "second"
    gw_residual_units: str = "second"
    cons_scale_floor: str = "auto"
    gw_scale_floor: str = "auto"
    allow_subs_residual: bool = True
    dt_min_units: float = 1e-6

    # Q forcing / drainage
    q_wrt_normalized_time: bool = False
    q_in_si: bool = False
    q_in_per_second: bool = False
    q_kind: str = "per_volume"
    q_length_in_si: bool = False
    drainage_mode: str = "double"
    
    # Training strategy (v3.2)
    training_strategy: str = "physics_first"
    
    # --------------------------
    # Physics-first knobs
    # --------------------------
    q_policy_physics_first: str = "warmup_off"
    q_warmup_epochs_physics_first: int = 20
    q_ramp_epochs_physics_first: int = 10
    lambda_q_physics_first: float = 1e-5
    loss_weight_gwl_physics_first: float = 0.5
    subs_resid_policy_physics_first: str = "warmup_off"
    subs_resid_warmup_epochs_physics_first: int = 15
    subs_resid_ramp_epochs_physics_first: int = 10
    
    # --------------------------
    # Data-first knobs
    # --------------------------
    loss_weight_gwl_data_first: float = 1.0
    lambda_q_data_first: float = 1e-3
    q_policy_data_first: str = "always_on"
    q_warmup_epochs_data_first: int = 0
    q_ramp_epochs_data_first: int = 0
    subs_resid_policy_data_first: str = "always_on"
    subs_resid_warmup_epochs_data_first: int = 0
    subs_resid_ramp_epochs_data_first: int = 0
    
    log_q_diagnostics: bool = True
    track_aux_metrics: bool = False

    # --- near physics config fields ---
    physics_baseline_mode: str = "none"
    debug_physics_grads: bool = False
    
    mv_prior_units: str = "strict"
    mv_alpha_disp: float = 0.1
    mv_huber_delta: float = 1.0
    mv_prior_mode: str = "calibrate"
    mv_weight: float = 1e-3
    
    mv_schedule_unit: str = "epoch"
    mv_delay_epochs: int = 1
    mv_warmup_epochs: int = 2
    mv_delay_steps: Optional[int] = None
    mv_warmup_steps: Optional[int] = None
    
    geoprior_use_effective_h: bool = True 
    
    # Scaling / stability safeguards
    scaling_error_policy: str = "raise"
    clip_global_norm: float = 5.0

    # Evaluation JSON units
    eval_json_units_mode: str = "interpretable"
    eval_json_units_scope: str = "all"
    
    # External scaling kwargs (v3.2)
    scaling_kwargs_json_path: Optional[str] = None

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
    
    # --- near device configuration fields ---
    tf_intra_threads: Optional[int] = None
    tf_inter_threads: Optional[int] = None
    tf_gpu_memory_limit_mb: Optional[int] = None
    
    use_tf_savedmodel: bool = False
    use_in_memory_model: bool = True
    debug: bool = False
    
    audit_stages: Any = "*"

    # --- optional but recommended: architecture defaults ---
    model_name: str = "GeoPriorSubsNet"
    
    attention_levels: List[str] = field(
        default_factory=lambda: [
            "cross",
            "hierarchical",
            "memory",
        ]
    )
    embed_dim: int = 32
    hidden_units: int = 64
    lstm_units: int = 64
    attention_units: int = 64
    num_heads: int = 2
    dropout_rate: float = 0.10
    
    memory_size: int = 50
    scales: List[int] = field(
        default_factory=lambda: [1, 2]
    )
    use_residuals: bool = True
    use_batch_norm: bool = False
    use_vsn: bool = True
    vsn_units: int = 32
    
    # ------------------------------------------------------------------
    # Uncertainty controls (v3.2+)
    # ------------------------------------------------------------------
    interval_level: float = 0.8
    crossing_penalty: float = 0.0
    crossing_margin: float = 0.0
    calibration_mode: str = "none"
    calibration_temperature: float = 1.0

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
            tuner_space = default_tuner_search_space(
                offset_mode=base.get(
                    "OFFSET_MODE",
                    cls.offset_mode,
                ),
            )

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

            gwl_kind=iget("GWL_KIND", cls.gwl_kind),
            gwl_sign=iget("GWL_SIGN", cls.gwl_sign),
            use_head_proxy=iget(
                "USE_HEAD_PROXY",
                cls.use_head_proxy,
            ),
            z_surf_col=base.get("Z_SURF_COL", cls.z_surf_col),
            include_z_surf_as_static=iget(
                "INCLUDE_Z_SURF_AS_STATIC",
                cls.include_z_surf_as_static,
            ),
            head_col=iget("HEAD_COL", cls.head_col),
            gwl_dyn_index=base.get(
                "GWL_DYN_INDEX",
                cls.gwl_dyn_index,
            ),
            normalize_coords=iget(
                "NORMALIZE_COORDS",
                cls.normalize_coords,
            ),
            keep_coords_raw=iget(
                "KEEP_COORDS_RAW",
                cls.keep_coords_raw,
            ),
            shift_raw_coords=iget(
                "SHIFT_RAW_COORDS",
                cls.shift_raw_coords,
            ),
            scale_h_field=iget(
                "SCALE_H_FIELD",
                cls.scale_h_field,
            ),
            scale_gwl=iget("SCALE_GWL", cls.scale_gwl),
            scale_z_surf=iget(
                "SCALE_Z_SURF",
                cls.scale_z_surf,
            ),
            subsidence_kind=iget(
                "SUBSIDENCE_KIND",
                cls.subsidence_kind,
            ),
            # --- runtime / format / debugging / audit ---
            tf_intra_threads=base.get(
                "TF_INTRA_THREADS",
                cls.tf_intra_threads,
            ),
            tf_inter_threads=base.get(
                "TF_INTER_THREADS",
                cls.tf_inter_threads,
            ),
            tf_gpu_memory_limit_mb=base.get(
                "TF_GPU_MEMORY_LIMIT_MB",
                cls.tf_gpu_memory_limit_mb,
            ),
            
            use_tf_savedmodel=iget(
                "USE_TF_SAVEDMODEL",
                cls.use_tf_savedmodel,
            ),
            use_in_memory_model=iget(
                "USE_IN_MEMORY_MODEL",
                cls.use_in_memory_model,
            ),
            debug=iget("DEBUG", cls.debug),
            
            audit_stages=base.get(
                "AUDIT_STAGES",
                cls.audit_stages,
            ),
            
            # Training hyper-params ------------------------------------------
            epochs=iget("EPOCHS", cls.epochs),
            batch_size=iget("BATCH_SIZE", cls.batch_size),
            learning_rate=iget("LEARNING_RATE", cls.learning_rate),
            # Physics configuration ------------------------------------------
            pde_mode=base.get("PDE_MODE_CONFIG", cls.pde_mode),
            lambda_cons=iget("LAMBDA_CONS", cls.lambda_cons),
            lambda_gw=iget("LAMBDA_GW", cls.lambda_gw),
            lambda_prior=iget("LAMBDA_PRIOR", cls.lambda_prior),
            lambda_smooth=iget("LAMBDA_SMOOTH", cls.lambda_smooth),
            lambda_mv=iget("LAMBDA_MV", cls.lambda_mv),
            lambda_bounds=iget(
                "LAMBDA_BOUNDS",
                cls.lambda_bounds,
            ),
            lambda_q=iget("LAMBDA_Q", cls.lambda_q),
            offset_mode=iget(
                "OFFSET_MODE",
                cls.offset_mode,
            ),
            lambda_offset=iget(
                "LAMBDA_OFFSET",
                cls.lambda_offset,
            ),
            use_lambda_offset_scheduler=iget(
                "USE_LAMBDA_OFFSET_SCHEDULER",
                cls.use_lambda_offset_scheduler,
            ),
            lambda_offset_unit=iget(
                "LAMBDA_OFFSET_UNIT",
                cls.lambda_offset_unit,
            ),
            lambda_offset_when=iget(
                "LAMBDA_OFFSET_WHEN",
                cls.lambda_offset_when,
            ),
            lambda_offset_warmup=iget(
                "LAMBDA_OFFSET_WARMUP",
                cls.lambda_offset_warmup,
            ),
            lambda_offset_start=iget(
                "LAMBDA_OFFSET_START",
                cls.lambda_offset_start,
            ),
            lambda_offset_end=iget(
                "LAMBDA_OFFSET_END",
                cls.lambda_offset_end,
            ),
            lambda_offset_schedule=base.get(
                "LAMBDA_OFFSET_SCHEDULE",
                cls.lambda_offset_schedule,
            ),
            physics_warmup_steps=iget(
                "PHYSICS_WARMUP_STEPS",
                cls.physics_warmup_steps,
            ),
            physics_ramp_steps=iget(
                "PHYSICS_RAMP_STEPS",
                cls.physics_ramp_steps,
            ),
            scale_pde_residuals=iget(
                "SCALE_PDE_RESIDUALS",
                cls.scale_pde_residuals,
            ),
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

            physics_bounds=base.get(
                "PHYSICS_BOUNDS",
                cls.physics_bounds,
            ),
            bounds_mode=iget(
                "PHYSICS_BOUNDS_MODE",
                cls.bounds_mode,
            ),
            time_units=iget("TIME_UNITS", cls.time_units),
            subs_unit_to_si=iget(
                "SUBS_UNIT_TO_SI",
                cls.subs_unit_to_si,
            ),
            subs_scale_si=iget(
                "SUBS_SCALE_SI",
                cls.subs_scale_si,
            ),
            subs_bias_si=iget(
                "SUBS_BIAS_SI",
                cls.subs_bias_si,
            ),
            head_unit_to_si=iget(
                "HEAD_UNIT_TO_SI",
                cls.head_unit_to_si,
            ),
            head_scale_si=base.get(
                "HEAD_SCALE_SI",
                cls.head_scale_si,
            ),
            head_bias_si=base.get(
                "HEAD_BIAS_SI",
                cls.head_bias_si,
            ),
            thickness_unit_to_si=iget(
                "THICKNESS_UNIT_TO_SI",
                cls.thickness_unit_to_si,
            ),
            z_surf_unit_to_si=iget(
                "Z_SURF_UNIT_TO_SI",
                cls.z_surf_unit_to_si,
            ),
            h_field_min_si=iget(
                "H_FIELD_MIN_SI",
                cls.h_field_min_si,
            ),
            auto_si_affine_from_stage1=iget(
                "AUTO_SI_AFFINE_FROM_STAGE1",
                cls.auto_si_affine_from_stage1,
            ),
            coord_mode=iget("COORD_MODE", cls.coord_mode),
            utm_epsg=iget("UTM_EPSG", cls.utm_epsg),
            coord_src_epsg=iget(
                "COORD_SRC_EPSG",
                cls.coord_src_epsg,
            ),
            residual_method=iget(
                "CONSOLIDATION_STEP_RESIDUAL_METHOD",
                cls.residual_method,
            ),
            cons_residual_units=iget(
                "CONSOLIDATION_RESIDUAL_UNITS",
                cls.cons_residual_units,
            ),
            gw_residual_units=iget(
                "GW_RESIDUAL_UNITS",
                cls.gw_residual_units,
            ),
            cons_scale_floor=iget(
                "CONS_SCALE_FLOOR",
                cls.cons_scale_floor,
            ),
            gw_scale_floor=iget(
                "GW_SCALE_FLOOR",
                cls.gw_scale_floor,
            ),
            allow_subs_residual=iget(
                "ALLOW_SUBS_RESIDUAL",
                cls.allow_subs_residual,
            ),
            dt_min_units=iget(
                "DT_MIN_UNITS",
                cls.dt_min_units,
            ),
            q_wrt_normalized_time=iget(
                "Q_WRT_NORMALIZED_TIME",
                cls.q_wrt_normalized_time,
            ),
            q_in_si=iget("Q_IN_SI", cls.q_in_si),
            q_in_per_second=iget(
                "Q_IN_PER_SECOND",
                cls.q_in_per_second,
            ),
            q_kind=iget("Q_KIND", cls.q_kind),
            q_length_in_si=iget(
                "Q_LENGTH_IN_SI",
                cls.q_length_in_si,
            ),
            drainage_mode=iget(
                "DRAINAGE_MODE",
                cls.drainage_mode,
            ),
            training_strategy=iget(
                "TRAINING_STRATEGY",
                cls.training_strategy,
            ),
            q_policy_physics_first=iget(
                "Q_POLICY_PHYSICS_FIRST",
                cls.q_policy_physics_first,
            ),
            q_warmup_epochs_physics_first=iget(
                "Q_WARMUP_EPOCHS_PHYSICS_FIRST",
                cls.q_warmup_epochs_physics_first,
            ),
            q_ramp_epochs_physics_first=iget(
                "Q_RAMP_EPOCHS_PHYSICS_FIRST",
                cls.q_ramp_epochs_physics_first,
            ),
            lambda_q_physics_first=iget(
                "LAMBDA_Q_PHYSICS_FIRST",
                cls.lambda_q_physics_first,
            ),
            loss_weight_gwl_physics_first=iget(
                "LOSS_WEIGHT_GWL_PHYSICS_FIRST",
                cls.loss_weight_gwl_physics_first,
            ),
            subs_resid_policy_physics_first=iget(
                "SUBS_RESID_POLICY_PHYSICS_FIRST",
                cls.subs_resid_policy_physics_first,
            ),
            subs_resid_warmup_epochs_physics_first=iget(
                "SUBS_RESID_WARMUP_EPOCHS_PHYSICS_FIRST",
                cls.subs_resid_warmup_epochs_physics_first,
            ),
            subs_resid_ramp_epochs_physics_first=iget(
                "SUBS_RESID_RAMP_EPOCHS_PHYSICS_FIRST",
                cls.subs_resid_ramp_epochs_physics_first,
            ),
            loss_weight_gwl_data_first=iget(
                "LOSS_WEIGHT_GWL_DATA_FIRST",
                cls.loss_weight_gwl_data_first,
            ),
            lambda_q_data_first=iget(
                "LAMBDA_Q_DATA_FIRST",
                cls.lambda_q_data_first,
            ),
            q_policy_data_first=iget(
                "Q_POLICY_DATA_FIRST",
                cls.q_policy_data_first,
            ),
            q_warmup_epochs_data_first=iget(
                "Q_WARMUP_EPOCHS_DATA_FIRST",
                cls.q_warmup_epochs_data_first,
            ),
            q_ramp_epochs_data_first=iget(
                "Q_RAMP_EPOCHS_DATA_FIRST",
                cls.q_ramp_epochs_data_first,
            ),
            subs_resid_policy_data_first=iget(
                "SUBS_RESID_POLICY_DATA_FIRST",
                cls.subs_resid_policy_data_first,
            ),
            subs_resid_warmup_epochs_data_first=iget(
                "SUBS_RESID_WARMUP_EPOCHS_DATA_FIRST",
                cls.subs_resid_warmup_epochs_data_first,
            ),
            subs_resid_ramp_epochs_data_first=iget(
                "SUBS_RESID_RAMP_EPOCHS_DATA_FIRST",
                cls.subs_resid_ramp_epochs_data_first,
            ),
            log_q_diagnostics=iget(
                "LOG_Q_DIAGNOSTICS",
                cls.log_q_diagnostics,
            ),
            track_aux_metrics=iget(
                "TRACK_AUX_METRICS",
                cls.track_aux_metrics,
            ),
            scaling_error_policy=iget(
                "SCALING_ERROR_POLICY",
                cls.scaling_error_policy,
            ),
            clip_global_norm=iget(
                "CLIP_GLOBAL_NORM",
                cls.clip_global_norm,
            ),
            eval_json_units_mode=iget(
                "EVAL_JSON_UNITS_MODE",
                cls.eval_json_units_mode,
            ),
            eval_json_units_scope=iget(
                "EVAL_JSON_UNITS_SCOPE",
                cls.eval_json_units_scope,
            ),
            scaling_kwargs_json_path=iget(
                "SCALING_KWARGS_JSON_PATH",
                cls.scaling_kwargs_json_path,
            ),
            # --- physics baseline / debug grads ---
            physics_baseline_mode=iget(
                "PHYSICS_BASELINE_MODE",
                cls.physics_baseline_mode,
            ),
            debug_physics_grads=iget(
                "DEBUG_PHYSICS_GRADS",
                cls.debug_physics_grads,
            ),
            # --- MV prior scheduler parity ---
            mv_prior_units=iget(
                "MV_PRIOR_UNITS",
                cls.mv_prior_units,
            ),
            mv_alpha_disp=iget(
                "MV_ALPHA_DISP",
                cls.mv_alpha_disp,
            ),
            mv_huber_delta=iget(
                "MV_HUBER_DELTA",
                cls.mv_huber_delta,
            ),
            mv_prior_mode=iget(
                "MV_PRIOR_MODE",
                cls.mv_prior_mode,
            ),
            mv_weight=iget("MV_WEIGHT", cls.mv_weight),
            
            mv_schedule_unit=iget(
                "MV_SCHEDULE_UNIT",
                cls.mv_schedule_unit,
            ),
            mv_delay_epochs=iget(
                "MV_DELAY_EPOCHS",
                cls.mv_delay_epochs,
            ),
            mv_warmup_epochs=iget(
                "MV_WARMUP_EPOCHS",
                cls.mv_warmup_epochs,
            ),
            mv_delay_steps=base.get(
                "MV_DELAY_STEPS",
                cls.mv_delay_steps,
            ),
            mv_warmup_steps=base.get(
                "MV_WARMUP_STEPS",
                cls.mv_warmup_steps,
            ),

            # --- optional: architecture defaults ---
            model_name=iget("MODEL_NAME", cls.model_name),
            attention_levels=base.get(
                "ATTENTION_LEVELS",
                cls.attention_levels,
            ),
            embed_dim=iget("EMBED_DIM", cls.embed_dim),
            hidden_units=iget("HIDDEN_UNITS", cls.hidden_units),
            lstm_units=iget("LSTM_UNITS", cls.lstm_units),
            attention_units=iget(
                "ATTENTION_UNITS",
                cls.attention_units,
            ),
            num_heads=iget(
                "NUM_HEADS",
                cls.num_heads,
            ),
            dropout_rate=iget(
                "DROPOUT_RATE",
                cls.dropout_rate,
            ),
            memory_size=iget("MEMORY_SIZE", cls.memory_size),
            scales=base.get("SCALES", cls.scales),
            use_residuals=iget(
                "USE_RESIDUALS",
                cls.use_residuals,
            ),
            use_batch_norm=iget(
                "USE_BATCH_NORM",
                cls.use_batch_norm,
            ),
            use_vsn=iget("USE_VSN", cls.use_vsn),
            vsn_units=iget("VSN_UNITS", cls.vsn_units),

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
        obj.static_driver_features = base.get(
            "STATIC_DRIVER_FEATURES", 
            obj.static_driver_features
        )
        obj.dynamic_driver_features= base.get(
            "DYNAMIC_DRIVER_FEATURES", 
            obj.dynamic_driver_features
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
        
        obj.dynamic_feature_names = base.get(
            "DYNAMIC_FEATURE_NAMES",
            obj.dynamic_feature_names,
        )
        obj.future_feature_names = base.get(
            "FUTURE_FEATURE_NAMES",
            obj.future_feature_names,
        )
        
        obj.include_censor_flags_as_future = base.get(
            "INCLUDE_CENSOR_FLAGS_AS_FUTURE",
            obj.include_censor_flags_as_future,
        )
        obj.interval_level = base.get(
            "INTERVAL_LEVEL", 
            obj.interval_level
        )

        obj.crossing_penalty = base.get(
            "CROSSING_PENALTY", obj.crossing_penalty
            )

        obj.unc_crossing_margin = base.get(
            "CROSSING_MARGIN", obj.crossing_margin
            )
        obj.calibration_mode = base.get(
            "CALIBRATION_MODE",  obj.calibration_mode
        )
        obj.calibration_temperature = base.get(
                "CALIBRATION_TEMPERATURE",
                obj.calibration_temperature,
            )


        return obj 
    
    def refresh_from_nat(self) -> None:
        """Reload NAT config without touching UI-owned fields."""
        keep_city = self.city
        keep_path = self.dataset_path
        keep_root = self.results_root
    
        new = type(self).from_nat_config()
    
        self.__dict__.update(new.__dict__)
    
        self.city = keep_city
        self.dataset_path = keep_path
        self.results_root = keep_root

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
        maybe("GWL_KIND", self.gwl_kind)
        maybe("GWL_SIGN", self.gwl_sign)
        maybe("USE_HEAD_PROXY", self.use_head_proxy)
        maybe("Z_SURF_COL", self.z_surf_col)
        maybe(
            "INCLUDE_Z_SURF_AS_STATIC",
            self.include_z_surf_as_static,
        )
        maybe("HEAD_COL", self.head_col)
        maybe("GWL_DYN_INDEX", self.gwl_dyn_index)
        maybe("NORMALIZE_COORDS", self.normalize_coords)
        maybe("KEEP_COORDS_RAW", self.keep_coords_raw)
        maybe("SHIFT_RAW_COORDS", self.shift_raw_coords)
        maybe("SCALE_H_FIELD", self.scale_h_field)
        maybe("SCALE_GWL", self.scale_gwl)
        maybe("SCALE_Z_SURF", self.scale_z_surf)
        maybe("SUBSIDENCE_KIND", self.subsidence_kind)
        
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
        maybe("LAMBDA_BOUNDS", self.lambda_bounds)
        maybe("LAMBDA_Q", self.lambda_q)
        maybe("OFFSET_MODE", self.offset_mode)
        maybe("LAMBDA_OFFSET", self.lambda_offset)
        maybe(
            "USE_LAMBDA_OFFSET_SCHEDULER",
            self.use_lambda_offset_scheduler,
        )
        maybe("LAMBDA_OFFSET_UNIT", self.lambda_offset_unit)
        maybe("LAMBDA_OFFSET_WHEN", self.lambda_offset_when)
        maybe("LAMBDA_OFFSET_WARMUP", self.lambda_offset_warmup)
        maybe("LAMBDA_OFFSET_START", self.lambda_offset_start)
        maybe("LAMBDA_OFFSET_END", self.lambda_offset_end)
        maybe(
            "LAMBDA_OFFSET_SCHEDULE",
            self.lambda_offset_schedule,
        )
        maybe(
            "PHYSICS_WARMUP_STEPS",
            self.physics_warmup_steps,
        )
        maybe("PHYSICS_RAMP_STEPS", self.physics_ramp_steps)
        maybe(
            "SCALE_PDE_RESIDUALS",
            self.scale_pde_residuals,
        )

        # Scalar physics parameters
        maybe("MV_LR_MULT", self.mv_lr_mult)
        maybe("KAPPA_LR_MULT", self.kappa_lr_mult)
        maybe("GEOPRIOR_INIT_MV", self.geoprior_init_mv)
        maybe("GEOPRIOR_INIT_KAPPA", self.geoprior_init_kappa)
        maybe("GEOPRIOR_GAMMA_W", self.geoprior_gamma_w)
        maybe("GEOPRIOR_H_REF", self.geoprior_h_ref)
        maybe("GEOPRIOR_KAPPA_MODE", self.geoprior_kappa_mode)
        maybe("GEOPRIOR_HD_FACTOR", self.geoprior_hd_factor)
        maybe("PHYSICS_BOUNDS", self.physics_bounds)
        maybe("PHYSICS_BOUNDS_MODE", self.bounds_mode)
        maybe("TIME_UNITS", self.time_units)
        maybe("SUBS_UNIT_TO_SI", self.subs_unit_to_si)
        maybe("SUBS_SCALE_SI", self.subs_scale_si)
        maybe("SUBS_BIAS_SI", self.subs_bias_si)
        maybe("HEAD_UNIT_TO_SI", self.head_unit_to_si)
        maybe("HEAD_SCALE_SI", self.head_scale_si)
        maybe("HEAD_BIAS_SI", self.head_bias_si)
        maybe("THICKNESS_UNIT_TO_SI", self.thickness_unit_to_si)
        maybe("Z_SURF_UNIT_TO_SI", self.z_surf_unit_to_si)
        maybe("H_FIELD_MIN_SI", self.h_field_min_si)
        maybe(
            "AUTO_SI_AFFINE_FROM_STAGE1",
            self.auto_si_affine_from_stage1,
        )
        maybe("COORD_MODE", self.coord_mode)
        maybe("UTM_EPSG", self.utm_epsg)
        maybe("COORD_SRC_EPSG", self.coord_src_epsg)
        maybe(
            "CONSOLIDATION_STEP_RESIDUAL_METHOD",
            self.residual_method,
        )
        maybe("CONSOLIDATION_RESIDUAL_UNITS", self.cons_residual_units)
        maybe("GW_RESIDUAL_UNITS", self.gw_residual_units)
        maybe("CONS_SCALE_FLOOR", self.cons_scale_floor)
        maybe("GW_SCALE_FLOOR", self.gw_scale_floor)
        maybe("ALLOW_SUBS_RESIDUAL", self.allow_subs_residual)
        maybe("DT_MIN_UNITS", self.dt_min_units)
        maybe("Q_WRT_NORMALIZED_TIME", self.q_wrt_normalized_time)
        maybe("Q_IN_SI", self.q_in_si)
        maybe("Q_IN_PER_SECOND", self.q_in_per_second)
        maybe("Q_KIND", self.q_kind)
        maybe("Q_LENGTH_IN_SI", self.q_length_in_si)
        maybe("DRAINAGE_MODE", self.drainage_mode)
        maybe("TRAINING_STRATEGY", self.training_strategy)
        maybe(
            "Q_POLICY_PHYSICS_FIRST",
            self.q_policy_physics_first,
        )
        maybe(
            "Q_WARMUP_EPOCHS_PHYSICS_FIRST",
            self.q_warmup_epochs_physics_first,
        )
        maybe(
            "Q_RAMP_EPOCHS_PHYSICS_FIRST",
            self.q_ramp_epochs_physics_first,
        )
        maybe(
            "LAMBDA_Q_PHYSICS_FIRST",
            self.lambda_q_physics_first,
        )
        maybe(
            "LOSS_WEIGHT_GWL_PHYSICS_FIRST",
            self.loss_weight_gwl_physics_first,
        )
        maybe(
            "SUBS_RESID_POLICY_PHYSICS_FIRST",
            self.subs_resid_policy_physics_first,
        )
        maybe(
            "SUBS_RESID_WARMUP_EPOCHS_PHYSICS_FIRST",
            self.subs_resid_warmup_epochs_physics_first,
        )
        maybe(
            "SUBS_RESID_RAMP_EPOCHS_PHYSICS_FIRST",
            self.subs_resid_ramp_epochs_physics_first,
        )
        maybe(
            "LOSS_WEIGHT_GWL_DATA_FIRST",
            self.loss_weight_gwl_data_first,
        )
        maybe(
            "LAMBDA_Q_DATA_FIRST",
            self.lambda_q_data_first,
        )
        maybe(
            "Q_POLICY_DATA_FIRST",
            self.q_policy_data_first,
        )
        maybe(
            "Q_WARMUP_EPOCHS_DATA_FIRST",
            self.q_warmup_epochs_data_first,
        )
        maybe(
            "Q_RAMP_EPOCHS_DATA_FIRST",
            self.q_ramp_epochs_data_first,
        )
        maybe(
            "SUBS_RESID_POLICY_DATA_FIRST",
            self.subs_resid_policy_data_first,
        )
        maybe(
            "SUBS_RESID_WARMUP_EPOCHS_DATA_FIRST",
            self.subs_resid_warmup_epochs_data_first,
        )
        maybe(
            "SUBS_RESID_RAMP_EPOCHS_DATA_FIRST",
            self.subs_resid_ramp_epochs_data_first,
        )
        maybe("LOG_Q_DIAGNOSTICS", self.log_q_diagnostics)
        maybe("TRACK_AUX_METRICS", self.track_aux_metrics)
        maybe("SCALING_ERROR_POLICY", self.scaling_error_policy)
        maybe(
            "SCALING_KWARGS_JSON_PATH",
            self.scaling_kwargs_json_path,
        )

        maybe("CLIP_GLOBAL_NORM", self.clip_global_norm)
        maybe("EVAL_JSON_UNITS_MODE", self.eval_json_units_mode)
        maybe("EVAL_JSON_UNITS_SCOPE", self.eval_json_units_scope)
        
        # Stage-1 extras
        maybe("BUILD_FUTURE_NPZ", self.build_future_npz)

        # Feature registry & censoring
        maybe("OPTIONAL_NUMERIC_FEATURES", self.optional_numeric_features)
        maybe("OPTIONAL_CATEGORICAL_FEATURES", self.optional_categorical_features)
        maybe("ALREADY_NORMALIZED_FEATURES", self.already_normalized_features)
        maybe("FUTURE_DRIVER_FEATURES", self.future_driver_features)
        maybe("STATIC_DRIVER_FEATURES", self.static_driver_features)
        maybe("DYNAMIC_DRIVER_FEATURES", self.dynamic_driver_features)
        maybe("CENSORING_SPECS", self.censoring_specs)
        maybe("INCLUDE_CENSOR_FLAGS_AS_DYNAMIC",
              self.include_censor_flags_as_dynamic)
        maybe("USE_EFFECTIVE_H_FIELD", self.use_effective_h_field)
        
        # Device configuration
        maybe("TF_DEVICE_MODE", self.tf_device_mode)
        maybe("TF_GPU_ALLOW_GROWTH", self.tf_gpu_allow_growth)

        maybe("TF_INTRA_THREADS", self.tf_intra_threads)
        maybe("TF_INTER_THREADS", self.tf_inter_threads)
        maybe("TF_GPU_MEMORY_LIMIT_MB", self.tf_gpu_memory_limit_mb)
        
        maybe("USE_TF_SAVEDMODEL", self.use_tf_savedmodel)
        maybe("USE_IN_MEMORY_MODEL", self.use_in_memory_model)
        maybe("DEBUG", self.debug)
        
        maybe("AUDIT_STAGES", self.audit_stages)

        # Tuner bits
        maybe("TUNER_MAX_TRIALS", self.tuner_max_trials)
        maybe("TUNER_SEARCH_SPACE", self.tuner_search_space)
        
        # Probabilistic
        maybe("QUANTILES", self.quantiles)
        maybe("SUBS_WEIGHTS", self.subs_weights)
        maybe("GWL_WEIGHTS", self.gwl_weights)
        
        maybe("PHYSICS_BASELINE_MODE", self.physics_baseline_mode)
        maybe("DEBUG_PHYSICS_GRADS", self.debug_physics_grads)

        maybe("MV_PRIOR_UNITS", self.mv_prior_units)
        maybe("MV_ALPHA_DISP", self.mv_alpha_disp)
        maybe("MV_HUBER_DELTA", self.mv_huber_delta)
        maybe("MV_PRIOR_MODE", self.mv_prior_mode)
        maybe("MV_WEIGHT", self.mv_weight)
        
        maybe("MV_SCHEDULE_UNIT", self.mv_schedule_unit)
        maybe("MV_DELAY_EPOCHS", self.mv_delay_epochs)
        maybe("MV_WARMUP_EPOCHS", self.mv_warmup_epochs)
        maybe("MV_DELAY_STEPS", self.mv_delay_steps)
        maybe("MV_WARMUP_STEPS", self.mv_warmup_steps)

        maybe("DYNAMIC_FEATURE_NAMES", self.dynamic_feature_names)
        maybe("FUTURE_FEATURE_NAMES", self.future_feature_names)
        
        maybe(
            "INCLUDE_CENSOR_FLAGS_AS_FUTURE",
            self.include_censor_flags_as_future,
        )

        maybe("MODEL_NAME", self.model_name)
        maybe("ATTENTION_LEVELS", self.attention_levels)
        maybe("EMBED_DIM", self.embed_dim)
        maybe("HIDDEN_UNITS", self.hidden_units)
        maybe("LSTM_UNITS", self.lstm_units)
        maybe("ATTENTION_UNITS", self.attention_units)
        maybe("NUM_HEADS", self.num_heads)
        maybe("DROPOUT_RATE", self.dropout_rate)
        
        maybe("MEMORY_SIZE", self.memory_size)
        maybe("SCALES", self.scales)
        maybe("USE_RESIDUALS", self.use_residuals)
        maybe("USE_BATCH_NORM", self.use_batch_norm)
        maybe("USE_VSN", self.use_vsn)
        maybe("VSN_UNITS", self.vsn_units)

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

        if isinstance(self.pde_mode, (list, tuple)):
            self.pde_mode = (
                self.pde_mode[0]
                if self.pde_mode
                else "off"
            )
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


        # --- v3.2: additional sanity checks -------------------------
        allowed_offset_modes = {"mul", "log10"}
        if self.offset_mode not in allowed_offset_modes:
            raise ValueError(
                "offset_mode must be one of "
                f"{sorted(allowed_offset_modes)} "
                f"(got {self.offset_mode!r})."
            )

        allowed_gwl_kind = {"depth_bgs", "head"}
        if self.gwl_kind not in allowed_gwl_kind:
            raise ValueError(
                "gwl_kind must be one of "
                f"{sorted(allowed_gwl_kind)} "
                f"(got {self.gwl_kind!r})."
            )

        allowed_subs_kind = {"cumulative", "rate"}
        if self.subsidence_kind not in allowed_subs_kind:
            raise ValueError(
                "subsidence_kind must be one of "
                f"{sorted(allowed_subs_kind)} "
                f"(got {self.subsidence_kind!r})."
            )

        allowed_bounds_mode = {"soft", "hard"}
        if self.bounds_mode not in allowed_bounds_mode:
            raise ValueError(
                "bounds_mode must be one of "
                f"{sorted(allowed_bounds_mode)} "
                f"(got {self.bounds_mode!r})."
            )

        allowed_residual_method = {"exact", "fd"}
        if self.residual_method not in allowed_residual_method:
            raise ValueError(
                "residual_method must be one of "
                f"{sorted(allowed_residual_method)} "
                f"(got {self.residual_method!r})."
            )

        if self.clip_global_norm is not None:
            if float(self.clip_global_norm) < 0.0:
                raise ValueError(
                    "clip_global_norm must be >= 0 "
                    f"(got {self.clip_global_norm!r})."
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
            # --------------------------------------------------------------
            # GUI / dataset
            # --------------------------------------------------------------
            "city": self.city,
            "dataset_path": (
                str(self.dataset_path)
                if self.dataset_path
                else None
            ),
            "results_root": str(self.results_root),
    
            # --------------------------------------------------------------
            # Temporal window
            # --------------------------------------------------------------
            "train_end_year": self.train_end_year,
            "forecast_start_year": self.forecast_start_year,
            "forecast_horizon_years": self.forecast_horizon_years,
            "time_steps": self.time_steps,
    
            # --------------------------------------------------------------
            # Stage-1 layout & columns
            # --------------------------------------------------------------
            "mode": self.mode,
            "time_col": self.time_col,
            "lon_col": self.lon_col,
            "lat_col": self.lat_col,
            "subs_col": self.subs_col,
            "gwl_col": self.gwl_col,
            "h_field_col": self.h_field_col,
    
            # v3.2: GWL/head conventions
            "gwl_kind": self.gwl_kind,
            "gwl_sign": self.gwl_sign,
            "use_head_proxy": self.use_head_proxy,
            "z_surf_col": self.z_surf_col,
            "include_z_surf_as_static": (
                self.include_z_surf_as_static
            ),
            "head_col": self.head_col,
            "gwl_dyn_index": self.gwl_dyn_index,
    
            # Stage-1 scaling controls
            "normalize_coords": self.normalize_coords,
            "keep_coords_raw": self.keep_coords_raw,
            "shift_raw_coords": self.shift_raw_coords,
            "scale_h_field": self.scale_h_field,
            "scale_gwl": self.scale_gwl,
            "scale_z_surf": self.scale_z_surf,
            "subsidence_kind": self.subsidence_kind,
    
            # --------------------------------------------------------------
            # Feature registry & censoring (Stage-1)
            # --------------------------------------------------------------
            "optional_numeric_features": (
                self.optional_numeric_features
            ),
            "optional_categorical_features": (
                self.optional_categorical_features
            ),
            "already_normalized_features": (
                self.already_normalized_features
            ),
            "dynamic_driver_features": self.dynamic_driver_features,
            "static_driver_features": self.static_driver_features,
            "future_driver_features": self.future_driver_features,
            "censoring_specs": self.censoring_specs,
            "include_censor_flags_as_dynamic": (
                self.include_censor_flags_as_dynamic
            ),
            "use_effective_h_field": self.use_effective_h_field,
            "dynamic_feature_names": self.dynamic_feature_names,
            "future_feature_names": self.future_feature_names,
            "include_censor_flags_as_future": (
                self.include_censor_flags_as_future
            ),
    
            # --------------------------------------------------------------
            # Training hyper-params
            # --------------------------------------------------------------
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
    
            # --------------------------------------------------------------
            # Physics configuration (Stage-2)
            # --------------------------------------------------------------
            "pde_mode": self.pde_mode,
            "physics_warmup_steps": self.physics_warmup_steps,
            "physics_ramp_steps": self.physics_ramp_steps,
            "scale_pde_residuals": self.scale_pde_residuals,
    
            "lambda_cons": self.lambda_cons,
            "lambda_gw": self.lambda_gw,
            "lambda_prior": self.lambda_prior,
            "lambda_smooth": self.lambda_smooth,
            "lambda_mv": self.lambda_mv,
            "lambda_bounds": self.lambda_bounds,
            "lambda_q": self.lambda_q,
    
            "offset_mode": self.offset_mode,
            "lambda_offset": self.lambda_offset,
            "use_lambda_offset_scheduler": (
                self.use_lambda_offset_scheduler
            ),
            "lambda_offset_unit": self.lambda_offset_unit,
            "lambda_offset_when": self.lambda_offset_when,
            "lambda_offset_warmup": self.lambda_offset_warmup,
            "lambda_offset_start": self.lambda_offset_start,
            "lambda_offset_end": self.lambda_offset_end,
            "lambda_offset_schedule": self.lambda_offset_schedule,
    
            "mv_lr_mult": self.mv_lr_mult,
            "kappa_lr_mult": self.kappa_lr_mult,
    
            "geoprior_init_mv": self.geoprior_init_mv,
            "geoprior_init_kappa": self.geoprior_init_kappa,
            "geoprior_gamma_w": self.geoprior_gamma_w,
            "geoprior_h_ref": self.geoprior_h_ref,
            "geoprior_kappa_mode": self.geoprior_kappa_mode,
            "geoprior_hd_factor": self.geoprior_hd_factor,
    
            # bounds / units / residual scaling / source term controls
            "physics_bounds": self.physics_bounds,
            "bounds_mode": self.bounds_mode,
            "time_units": self.time_units,
    
            "subs_unit_to_si": self.subs_unit_to_si,
            "subs_scale_si": self.subs_scale_si,
            "subs_bias_si": self.subs_bias_si,
            "head_unit_to_si": self.head_unit_to_si,
            "head_scale_si": self.head_scale_si,
            "head_bias_si": self.head_bias_si,
            "thickness_unit_to_si": self.thickness_unit_to_si,
            "z_surf_unit_to_si": self.z_surf_unit_to_si,
            "h_field_min_si": self.h_field_min_si,
            "auto_si_affine_from_stage1": (
                self.auto_si_affine_from_stage1
            ),
    
            "coord_mode": self.coord_mode,
            "utm_epsg": self.utm_epsg,
            "coord_src_epsg": self.coord_src_epsg,
    
            "residual_method": self.residual_method,
            "cons_residual_units": self.cons_residual_units,
            "gw_residual_units": self.gw_residual_units,
            "cons_scale_floor": self.cons_scale_floor,
            "gw_scale_floor": self.gw_scale_floor,
            "allow_subs_residual": self.allow_subs_residual,
            "dt_min_units": self.dt_min_units,
    
            "q_wrt_normalized_time": self.q_wrt_normalized_time,
            "q_in_si": self.q_in_si,
            "q_in_per_second": self.q_in_per_second,
            "q_kind": self.q_kind,
            "q_length_in_si": self.q_length_in_si,
            "drainage_mode": self.drainage_mode,
    
            "training_strategy": self.training_strategy,
            "q_policy_physics_first": self.q_policy_physics_first,
            "q_warmup_epochs_physics_first": (
                self.q_warmup_epochs_physics_first
            ),
            "q_ramp_epochs_physics_first": (
                self.q_ramp_epochs_physics_first
            ),
            "lambda_q_physics_first": self.lambda_q_physics_first,
            "loss_weight_gwl_physics_first": (
                self.loss_weight_gwl_physics_first
            ),
            "subs_resid_policy_physics_first": (
                self.subs_resid_policy_physics_first
            ),
            "subs_resid_warmup_epochs_physics_first": (
                self.subs_resid_warmup_epochs_physics_first
            ),
            "subs_resid_ramp_epochs_physics_first": (
                self.subs_resid_ramp_epochs_physics_first
            ),
            "loss_weight_gwl_data_first": (
                self.loss_weight_gwl_data_first
            ),
            "lambda_q_data_first": self.lambda_q_data_first,
            "q_policy_data_first": self.q_policy_data_first,
            "q_warmup_epochs_data_first": (
                self.q_warmup_epochs_data_first
            ),
            "q_ramp_epochs_data_first": (
                self.q_ramp_epochs_data_first
            ),
            "subs_resid_policy_data_first": (
                self.subs_resid_policy_data_first
            ),
            "subs_resid_warmup_epochs_data_first": (
                self.subs_resid_warmup_epochs_data_first
            ),
            "subs_resid_ramp_epochs_data_first": (
                self.subs_resid_ramp_epochs_data_first
            ),
            "log_q_diagnostics": self.log_q_diagnostics,
            "track_aux_metrics": self.track_aux_metrics,
    
            "physics_baseline_mode": self.physics_baseline_mode,
            "debug_physics_grads": self.debug_physics_grads,
    
            "mv_prior_units": self.mv_prior_units,
            "mv_alpha_disp": self.mv_alpha_disp,
            "mv_huber_delta": self.mv_huber_delta,
            "mv_prior_mode": self.mv_prior_mode,
            "mv_weight": self.mv_weight,
    
            "mv_schedule_unit": self.mv_schedule_unit,
            "mv_delay_epochs": self.mv_delay_epochs,
            "mv_warmup_epochs": self.mv_warmup_epochs,
            "mv_delay_steps": self.mv_delay_steps,
            "mv_warmup_steps": self.mv_warmup_steps,
    
            "geoprior_use_effective_h": self.geoprior_use_effective_h,
    
            "scaling_error_policy": self.scaling_error_policy,
            "clip_global_norm": self.clip_global_norm,
    
            "eval_json_units_mode": self.eval_json_units_mode,
            "eval_json_units_scope": self.eval_json_units_scope,
            "scaling_kwargs_json_path": self.scaling_kwargs_json_path,
    
            # Probabilistic outputs & weights
            "quantiles": self.quantiles,
            "subs_weights": self.subs_weights,
            "gwl_weights": self.gwl_weights,
    
            # Stage-1 helper output
            "build_future_npz": self.build_future_npz,
    
            # --------------------------------------------------------------
            # Device / runtime
            # --------------------------------------------------------------
            "tf_device_mode": self.tf_device_mode,
            "tf_gpu_allow_growth": self.tf_gpu_allow_growth,
            "tf_intra_threads": self.tf_intra_threads,
            "tf_inter_threads": self.tf_inter_threads,
            "tf_gpu_memory_limit_mb": self.tf_gpu_memory_limit_mb,
            "use_tf_savedmodel": self.use_tf_savedmodel,
            "use_in_memory_model": self.use_in_memory_model,
            "debug": self.debug,
            "audit_stages": self.audit_stages,
    
            # --------------------------------------------------------------
            # Architecture defaults
            # --------------------------------------------------------------
            "model_name": self.model_name,
            "attention_levels": self.attention_levels,
            "embed_dim": self.embed_dim,
            "hidden_units": self.hidden_units,
            "lstm_units": self.lstm_units,
            "attention_units": self.attention_units,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "memory_size": self.memory_size,
            "scales": self.scales,
            "use_residuals": self.use_residuals,
            "use_batch_norm": self.use_batch_norm,
            "use_vsn": self.use_vsn,
            "vsn_units": self.vsn_units,
            
            # --------------------------------------------------------------
            # Calibration Prob
            # --------------------------------------------------------------
            "interval_level": self.interval_level, 
            "crossing_penalty": self.crossing_penalty, 
            "crossing_margin":self.crossing_margin, 
            "calibration_mode": self.calibration_mode, 
            "calibration_temperature":self.calibration_temperature, 
    
            # --------------------------------------------------------------
            # GUI-only flags
            # --------------------------------------------------------------
            "evaluate_training": self.evaluate_training,
            "clean_stage1_dir": self.clean_stage1_dir,
            "stage1_auto_reuse_if_match": (
                self.stage1_auto_reuse_if_match
            ),
            "stage1_force_rebuild_if_mismatch": (
                self.stage1_force_rebuild_if_mismatch
            ),
    
            # GUI layout
            "ui_base_width": self.ui_base_width,
            "ui_base_height": self.ui_base_height,
            "ui_min_width": self.ui_min_width,
            "ui_min_height": self.ui_min_height,
            "ui_max_ratio": self.ui_max_ratio,
            "ui_font_scale": self.ui_font_scale,
    
            # --------------------------------------------------------------
            # Tuner
            # --------------------------------------------------------------
            "tuner_max_trials": self.tuner_max_trials,
            "tuner_search_space": self.tuner_search_space,
    
            # --------------------------------------------------------------
            # Dialog override buckets (GUI)
            # --------------------------------------------------------------
            "feature_overrides": self.feature_overrides,
            "arch_overrides": self.arch_overrides,
            "prob_overrides": self.prob_overrides,
        }
