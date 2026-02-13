# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio
#
# scripts/config.py
#
# Single source of truth for:
# - output directories (scripts/out, scripts/figs)
# - file discovery patterns (GeoPrior JSON, eval JSON, CSVs, payloads)
# - paper-wide units + human labels (drivers, targets, physics fields)
# - physics closures / formula strings used in captions and legends

from __future__ import annotations

from pathlib import Path

# -------------------------------------------------------------------
# Filesystem layout (v3.2)
# -------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPTS_DIR / "out"
FIG_DIR = SCRIPTS_DIR / "figs"

# -------------------------------------------------------------------
# Common discovery patterns (src may be a directory or a file)
# -------------------------------------------------------------------
# NOTE:
# - We search recursively under src when src is a directory.
# - For GeoPrior eval JSON we prefer the interpretable one when
#   available (it may carry human units like "mm").
PATTERNS = {
    # GeoPrior evaluation JSON
    "phys_json": (
        "geoprior_eval_phys_*_interpretable.json",
        "geoprior_eval_phys_*.json",
    ),
    # Eval diagnostics fallback JSON
    "eval_diag_json": (
        "eval_diagnostics*.json",
        "diagnostics*.json",
        "*eval*diagnostic*.json",
    ),
    "forecast_test_csv": (
        "*forecast*TestSet*calibrated*.csv",
        "*forecast_TestSet*_calibrated.csv",
        "*_forecast_TestSet_*_calibrated.csv",
        "*TestSet*calibrated*.csv",
    ),
    "forecast_test_future_csv": (
        "*forecast*TestSet*future*.csv",
        "*forecast_TestSet*_future.csv",
        "*_forecast_TestSet_*_future.csv",
        "*TestSet*future*.csv",
    ),
    # Typical calibrated validation forecasts (CSV)
    "forecast_val_csv": (
        "*forecast*Validation*calibrated*.csv",
        "*forecast*Validation*Fallback*calibrated*.csv",
        "*Validation*calibrated*.csv",
        "*_calibrated.csv",
    ),
    # Typical future forecasts (CSV)
    "forecast_future_csv": (
        "*future*.csv",
        "*H*_future*.csv",
        "*forecast*future*.csv",
    ),
    # Physics payloads
    "physics_payload": (
        "*physics_payload*.npz",
        "*phys_payload*.npz",
        "*payload*.npz",
    ),
    # Optional coords file (used by some physics maps)
    "coords_npz": (
        "*coords*.npz",
        "*xy*.npz",
        "*lonlat*.npz",
    ),
    # Ablation records (Supplement S6/S7)
    "ablation_record_jsonl": (
        "ablation_records/ablation_record*.jsonl",
        "ablation_records/ablation_record.updated*.jsonl",
        "ablation_record*.jsonl",
        "ablation_record.updated*.jsonl",
    ),
}

# -------------------------------------------------------------------
# Plot metric metadata (titles / labels / format)
# Used by multi-panel figures to avoid duplicated strings.
# -------------------------------------------------------------------
PLOT_METRIC_META = {
    "r2": {
        "title": r"$R^2$ (↑)",
        "ylabel": r"$R^2$",
        "fmt": "{:.2f}",
    },
    "mae": {
        "title": "MAE (↓, {unit})",
        "ylabel": "MAE ({unit})",
        "fmt": "{:.2f}",
        "unit": "mm",
    },
    "rmse": {
        "title": "RMSE (↓, {unit})",
        "ylabel": "RMSE ({unit})",
        "fmt": "{:.2f}",
        "unit": "mm",
    },
    "mse": {
        "title": "MSE (↓, {unit})",
        "ylabel": "MSE ({unit})",
        "fmt": "{:.2f}",
        "unit": "mm²",
    },
    "coverage80": {
        "title": "Coverage (@80) (↑)",
        "ylabel": "Coverage",
        "fmt": "{:.3f}",
    },
    "sharpness80": {
        "title": "Sharpness (@80) (↓, {unit})",
        "ylabel": "Sharpness ({unit})",
        "fmt": "{:.3f}",
        "unit": "mm",
    },
}

# -------------------------------------------------------------------
# Matplotlib defaults (paper-friendly)
# -------------------------------------------------------------------
PAPER_DPI = 600
PAPER_FONT = 8

# -------------------------------------------------------------------
# City styling (keep consistent across all scripts)
# (Used repeatedly across figures)
# -------------------------------------------------------------------
CITY_CANON = {
    "nansha": "Nansha",
    "zhongshan": "Zhongshan",
    "ns": "Nansha",
    "zh": "Zhongshan",
}

CITY_COLORS = {
    "Nansha": "#1F78B4",     # blue
    "Zhongshan": "#E31A1C",  # red
}

# -------------------------------------------------------------------
# Units: payload + drivers + targets
# -------------------------------------------------------------------
# Keep *units* separate from *labels* so plots can do:
#   label = LABELS[key]
#   unit  = UNITS.get(key)
#   ax.set_xlabel(f"{label} ({unit})" if unit else label)
UNITS = {
    # Drivers / coordinates
    "GWL": "m",
    "rainfall_mm": "mm/yr",
    "soil_thickness": "m",
    "normalized_urban_load_proxy": "1",
    "longitude": "deg",
    "latitude": "deg",
    "year": "year",
    # Targets
    "subsidence": "mm/yr",
    "subsidence_cum": "mm",
    # Forecast CSV canonical columns (no units)
    "forecast_step": "step",
    "sample_idx": "id",
}

LABELS = {
    "GWL": "Groundwater level",
    "rainfall_mm": "Rainfall",
    "soil_thickness": "Soil thickness",
    "normalized_urban_load_proxy": "Urban load",
    "longitude": "Longitude",
    "latitude": "Latitude",
    "year": "Year",
    "subsidence": "Subsidence",
    "subsidence_cum": "Cumulative subsidence",
    "lithology_class": "Lithology class",
    "city": "City",
    "forecast_step": "Forecast step",
    "sample_idx": "Sample id",
}

UNITS.update(
    {
        "GWL_depth_bgs_m": "m",
        "GWL_depth_bgs": "m",
        "GWL_depth_bgs_raw": "m",
        "rainfall_mm": "mm/yr",
        "soil_thickness_eff": "m",
        "soil_thickness": "m",
        "soil_thickness_imputed": "m",
        "urban_load_global": "1",
        "normalized_urban_load_proxy": "1",
        "head_m": "m",
        "z_surf_m": "m",
        "z_surf": "m",
    }
)

LABELS.update(
    {
        "GWL_depth_bgs_m": "Depth to water table",
        "GWL_depth_bgs": "Depth to water table",
        "GWL_depth_bgs_raw": "Depth to water table",
        "rainfall_mm": "Rainfall",
        "soil_thickness_eff": "Effective soil thickness",
        "soil_thickness": "Soil thickness",
        "soil_thickness_imputed": "Imputed soil thickness",
        "urban_load_global": "Urban load",
        "normalized_urban_load_proxy": "Urban load",
        "head_m": "Hydraulic head",
        "z_surf_m": "Surface elevation",
        "z_surf": "Surface elevation",
    }
)


COLUMN_ALIASES = {
    # Drivers (prefer the “best” columns)
    "GWL_depth_bgs_m": (
        "GWL_depth_bgs_m",
        "GWL_depth_bgs",
        "GWL",
        "GWL_depth_bgs_raw",
    ),
    "rainfall_mm": (
        "rainfall_mm",
        "rainfall",
        "rainfall_m",
    ),
    "soil_thickness_eff": (
        "soil_thickness_eff",
        "soil_thickness_imputed",
        "soil_thickness",
    ),
    "urban_load_global": (
        "urban_load_global",
        "normalized_urban_load_proxy",
    ),
    # Responses
    "subsidence_cum": ("subsidence_cum",),
    "subsidence": ("subsidence",),
    "head_m": ("head_m",),
}

DRIVER_RESPONSE_DEFAULT_DRIVERS = (
    "GWL_depth_bgs_m",
    "rainfall_mm",
    "soil_thickness_eff",
    "urban_load_global",
)

DRIVER_RESPONSE_DEFAULT_RESPONSE = "subsidence_cum"

# -------------------------------------------------------------------
# Units: GeoPrior physics fields / diagnostics
# (Used for physics maps and sanity plots)
# -------------------------------------------------------------------
PHYS_UNITS = {
    "K": "m/s",
    "Ss": "1/m",
    "Hd": "m",
    "H": "m",
    "tau": "s",
    "tau_prior": "s",
    "log10_K": "log10(m/s)",
    "log10_Ss": "log10(1/m)",
    "log10_tau": "log10(s)",
    # Diagnostics (dimensionless unless marked raw)
    "epsilon_prior": "1",
    "epsilon_cons": "1",
    "epsilon_gw": "1",
    "epsilon_cons_raw": "varies",
    "epsilon_gw_raw": "varies",
}

PHYS_LABELS = {
    "K": "Hydraulic conductivity",
    "Ss": "Specific storage",
    "Hd": "Drainage thickness",
    "H": "Effective thickness",
    "tau": "Consolidation time scale",
    "tau_prior": "Prior time scale",
    "log10_K": "log10 K",
    "log10_Ss": "log10 Ss",
    "log10_tau": "log10 τ",
    "epsilon_prior": r"$\epsilon_{\mathrm{prior}}$",
    "epsilon_cons": r"$\epsilon_{\mathrm{cons}}$",
    "epsilon_gw": r"$\epsilon_{\mathrm{gw}}$",
    "epsilon_cons_raw": r"$\epsilon_{\mathrm{cons,raw}}$",
    "epsilon_gw_raw": r"$\epsilon_{\mathrm{gw,raw}}$",
}

# -------------------------------------------------------------------
# Time + scaling (used to harmonize JSON variants)
# -------------------------------------------------------------------
SECONDS_PER_YEAR = 31556952.0

# -------------------------------------------------------------------
# Closure strings for captions/legends (math “single source”)
# -------------------------------------------------------------------
CLOSURES = {
    "tau_prior": (
        r"$\tau_{\mathrm{prior}}"
        r"\approx H_d^2\,S_s"
        r"/(\pi^2\,\kappa_b\,K)$"
    ),
    "s_eq": r"$s_{\mathrm{eq}}\approx S_s\,\Delta h\,H$",
    "R_cons": (
        r"$R_{\mathrm{cons}}="
        r"\partial_t \bar{s}"
        r"-(s_{\mathrm{eq}}(\bar{h})-\bar{s})/\tau$"
    ),
    "R_gw": (
        r"$R_{\mathrm{gw}}="
        r"S_s\,\partial_t \bar{h}"
        r"-\nabla\cdot(K\nabla\bar{h})$"
    ),
}


# ---------------------------------------------------------------------
# Column canonicalization
# ---------------------------------------------------------------------
_BASE_ALIASES = {
    "sample_idx": ("sample_idx", "sample_id", "sample"),
    "forecast_step": ("forecast_step", "forecast_s", "h"),
    "coord_t": ("coord_t", "year", "t"),
    "coord_x": ("coord_x", "lon", "longitude", "x"),
    "coord_y": ("coord_y", "lat", "latitude", "y"),
    "subsidence_actual": ("subsidence_actual", "actual"),
    "subsidence_q50": ("subsidence_q50", "q50", "p50"),
    "subsidence_unit": ("subsidence_unit", "unit"),
}


_CALIB_REQUIRED = [
    "sample_idx",
    "forecast_step",
    "coord_t",
    "coord_x",
    "coord_y",
    "subsidence_actual",
    "subsidence_q50",
]


_FUT_REQUIRED = [
    "sample_idx",
    "forecast_step",
    "coord_t",
    "coord_x",
    "coord_y",
    "subsidence_q50",
]

_CAL_ORDER = ("none", "source", "target")

_CAL_LABEL = {
    "none": "None",
    "source": "Source",
    "target": "Target",
}
_CAL_MARKER = {
    "none": "o",
    "source": "^",
    "target": "s",
}

_STRAT_DEFAULT = ("baseline", "xfer", "warm")

_STRAT_LABEL = {
    "baseline": "Baseline (target)",
    "xfer": "Transfer",
    "warm": "Warm-start",
}
_STRAT_HATCH = {
    "baseline": "//",
    "xfer": "",
    "warm": "xx",
}
_STRAT_EDGE = {
    "baseline": "#111111",
    "xfer": "#111111",
    "warm": "#111111",
}

_BASELINE_MAP = {
    "A_to_B": "B_to_B",
    "B_to_A": "A_to_A",
}


_METRIC_DEF = {
    "mae": ("overall_mae", "MAE (mm)", "min"),
    "mse": ("overall_mse", "MSE (mm^2)", "min"),
    "r2": ("overall_r2", r"$R^2$", "max"),
}
