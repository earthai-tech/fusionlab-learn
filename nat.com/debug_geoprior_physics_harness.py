#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_geoprior_physics_harness_v32.py

Physics-debug harness for GeoPriorSubsNet v3.2 (Option-1: physics-driven
subsidence mean via consolidation integrator).

This harness builds synthetic data with a Stage-1-like pack:
- coords: (B,H,3) as (t,x,y) (shifted; optional normalized)
- dynamic_features: (B,T,D)
- future_features:  (B,T+H,F)
- static_features:  (B,S)
- H_field: (B,H,1)
- h_ref_si: (B,1,1)   (optional but recommended)
- s_init_si: (B,1,1)  (optional but recommended)
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from fusionlab.nn.pinn._geoprior_subnet import GeoPriorSubsNet
from fusionlab.nn.pinn.op import seconds_per_time_unit
from fusionlab.utils.subsidence_utils import make_txy_coords

# ---------------------------------------------------------------------
# Debug toggles
# ---------------------------------------------------------------------
WORK_WITH_NORMALIZED_DATA = False        # if True, use *_z cols + affines
WORK_WITH_NORMALIZED_COORDS = False    # if True, normalize coords to ~[0,1]

# If you do not provide z_surf as an explicit input, the model can use a
# proxy head conversion (e.g. h = -depth) driven by h_ref_si.
USE_HEAD_PROXY = True

def infer_work_with_normalized_data(cfg: Dict[str, Any]) -> bool:
    gwl_col = str(cfg.get("GWL_COL", ""))
    subs_col = str(cfg.get("SUBSIDENCE_COL", ""))
    return gwl_col.endswith("_z") or subs_col.endswith("_z")


# ---------------------------------------------------------------------
# Minimal config
# ---------------------------------------------------------------------
DEFAULT_CFG: Dict[str, Any] = dict(
    CITY_NAME="zhongshan",
    MODEL_NAME="GeoPriorSubsNet",

    TRAIN_END_YEAR=2022,
    FORECAST_START_YEAR=2023,
    FORECAST_HORIZON_YEARS=3,
    TIME_STEPS=5,
    MODE="tft_like",

    TIME_COL="year",
    LON_COL="longitude",
    LAT_COL="latitude",

    # columns used in df
    GWL_COL="GWL_depth_bgs",          # updated below if normalized
    SUBSIDENCE_COL="subsidence_cum",  # updated below if normalized
    H_FIELD_COL_NAME="soil_thickness_eff",

    # Architecture
    EMBED_DIM=32,
    HIDDEN_UNITS=64,
    LSTM_UNITS=64,
    ATTENTION_UNITS=64,
    NUMBER_HEADS=2,
    DROPOUT_RATE=0.10,

    # Quantiles
    QUANTILES=[0.1, 0.5, 0.9],

    # Physics switches/weights
    PDE_MODE_CONFIG="both",
    SCALE_PDE_RESIDUALS=True,
    LAMBDA_CONS=0.10,
    LAMBDA_GW=0.01,
    LAMBDA_PRIOR=0.10,
    LAMBDA_SMOOTH=0.01,
    LAMBDA_MV=0.001,
    LAMBDA_BOUNDS=1e-3,
    OFFSET_MODE="mul",
    LAMBDA_OFFSET=1.0,

    PHYSICS_BOUNDS=dict(
        H_min=0.1,    H_max=35.0,
        K_min=1e-10,  K_max=1e-9,
        Ss_min=1e-6,  Ss_max=1e-3,
    ),

    TIME_UNITS="year",

    # SI conversion (harness uses SI meters already)
    SUBS_UNIT_TO_SI=1e-3, 
    HEAD_UNIT_TO_SI=1.0,
    THICKNESS_UNIT_TO_SI=1.0,

    # GeoPrior scalars
    GEOPRIOR_INIT_MV=1e-7,
    GEOPRIOR_INIT_KAPPA=1.0,
    GEOPRIOR_GAMMA_W=9810.0,
    GEOPRIOR_H_REF=0.0,
    GEOPRIOR_KAPPA_MODE="kb",
    GEOPRIOR_HD_FACTOR=0.6,

    # Optim
    LEARNING_RATE=1e-4,

    # coord CRS (optional)
    COORD_SRC_EPSG=4326,
    COORD_TARGET_EPSG=32649,
)


# ---------------------------------------------------------------------
# Synthetic dataframe
# ---------------------------------------------------------------------
def make_fake_df(
    cfg: Dict[str, Any],
    n_sites: int = 16,
    years: Optional[Sequence[int]] = None,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if years is None:
        y0 = 2015
        y1 = int(cfg["FORECAST_START_YEAR"] + cfg["FORECAST_HORIZON_YEARS"] + 1)
        years = list(range(y0, y1))

    lon0, lat0 = 113.3, 22.85
    lons = lon0 + 0.08 * rng.random(n_sites)
    lats = lat0 + 0.08 * rng.random(n_sites)

    litho_classes = [
        "Fine-Grained Soil", "Coarse-Grained Soil",
        "Mixed Clastics", "Carbonate",
    ]
    lithos = [
        "Mudstone–Siltstone", "Conglomerate–Sandstone",
        "Sandstone–Siltstone", "Limestone–Sandstone",
    ]

    rows = []
    for i in range(n_sites):
        city = str(cfg.get("CITY_NAME", "zhongshan")).title()
        lithology = lithos[i % len(lithos)]
        lithology_class = litho_classes[i % len(litho_classes)]

        # static-ish
        Ustar = rng.uniform(0.05, 0.95)         # U* in [0,1]
        H0 = rng.uniform(0.5, 25.0)             # m
        rain0 = rng.uniform(700.0, 1400.0)      # mm

        # surface elevation + baseline depth-to-water
        z_surf = float(rng.uniform(0.0, 30.0))  # m
        depth0 = float(rng.uniform(5.0, 60.0))  # m (down positive)

        subs_annual_mm = []
        depth_series = []
        head_series = []
        H_series = []
        rain_series = []

        head_ref = z_surf - depth0  # baseline head

        for y in years:
            rain = rain0 + 80.0 * math.sin((y - years[0]) * 0.7) + rng.normal(0, 40.0)
            rain = float(np.clip(rain, 200.0, 2500.0))

            depth = depth0 + 0.03 * (900.0 - rain) + rng.normal(0.0, 1.5)
            depth = float(np.clip(depth, 0.5, 120.0))

            head_m = z_surf - depth  # h = z_surf - z_GWL

            H = float(np.clip(H0 + rng.normal(0, 0.3), 0.1, 40.0))

            drawdown = max(0.0, head_ref - head_m)  # positive for head loss

            s_mm = (
                2.0
                + 10.0 * (Ustar ** 1.3)
                + 1.5 * drawdown
                + 0.002 * max(0.0, (rain - 900.0))
                + rng.normal(0.0, 1.0)
            )
            s_mm = float(np.clip(s_mm, 0.0, 40.0))

            subs_annual_mm.append(s_mm)
            depth_series.append(depth)
            head_series.append(head_m)
            H_series.append(H)
            rain_series.append(rain)

        subs_annual_mm = np.asarray(subs_annual_mm, float)
        subs_cum_mm = np.cumsum(subs_annual_mm)  # millimeters

        for j, y in enumerate(years):
            H = float(H_series[j])
            H_eff = float(min(H, 30.0))
            censored = bool(H >= 30.0 - 1e-6)

            rows.append(
                dict(
                    longitude=float(lons[i]),
                    latitude=float(lats[i]),
                    year=int(y),
                    lithology=lithology,
                    lithology_class=lithology_class,
                    city=city,

                    # head/depth series
                    GWL_depth_bgs=float(depth_series[j]),
                    head_m=float(head_series[j]),
                    z_surf_m=float(z_surf),

                    rainfall_mm=float(rain_series[j]),
                    urban_load_global=float(Ustar),

                    soil_thickness=float(H),
                    soil_thickness_eff=float(H_eff),
                    soil_thickness_censored=float(censored),

                    subsidence=float(subs_annual_mm[j]),     # mm/yr (toy)
                    subsidence_cum=float(subs_cum_mm[j]),     #m meters
                )
            )

    df = pd.DataFrame(rows)

    def _zscore(col: str) -> Tuple[pd.Series, float, float]:
        mu = float(df[col].mean())
        sig = float(df[col].std(ddof=0))
        sig = max(sig, 1e-12)
        return (df[col] - mu) / sig, mu, sig

    # z-scored versions (for the normalized-data toggle)
    df["GWL_depth_bgs_z"], depth_mu, depth_std = _zscore("GWL_depth_bgs")
    df["subsidence_cum_z"], subs_mu, subs_std = _zscore("subsidence_cum")

    df.attrs["depth_mu_m"] = depth_mu
    df.attrs["depth_std_m"] = depth_std
    df.attrs["subs_mu_m"] = subs_mu
    df.attrs["subs_std_m"] = subs_std

    return df


# ---------------------------------------------------------------------
# Stage-1-like pack
# ---------------------------------------------------------------------
@dataclass
class Stage1LikePack:
    inputs: Dict[str, np.ndarray]
    scaling_kwargs: Dict[str, Any]
    meta: Dict[str, Any]


def _minmax_fit_transform(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x = np.asarray(x, float)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    rng = max(xmax - xmin, 1e-12)
    return (x - xmin) / rng, xmin, rng


def pack_inputs(
    coords: np.ndarray,
    X_hist: np.ndarray,
    X_fut: np.ndarray,
    static_features: np.ndarray,
    H_future_m: np.ndarray,
    h_ref_si: np.ndarray,
    s_init_si: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "coords": coords.astype("float32"),                 # (B,H,3)
        "dynamic_features": X_hist.astype("float32"),       # (B,T,D)
        "future_features": X_fut.astype("float32"),         # (B,T+H,F)
        "static_features": static_features.astype("float32"),
        "H_field": H_future_m.astype("float32"),            # (B,H,1)
        # optional but strongly recommended for option-1 stability
        "h_ref_si": h_ref_si.astype("float32"),             # (B,1,1)
        "s_init_si": s_init_si.astype("float32"),           # (B,1,1)
    }


def build_one_sample(cfg: Dict[str, Any], df: pd.DataFrame) -> Stage1LikePack:
    T = int(cfg["TIME_STEPS"])
    H = int(cfg["FORECAST_HORIZON_YEARS"])
    y_start = int(cfg["FORECAST_START_YEAR"])
    y_hist_last = y_start - 1

    hist_years = list(range(y_start - T, y_start))
    fut_years = list(range(y_start, y_start + H))
    years_all = hist_years + fut_years  # length T+H

    gcols = [cfg["LON_COL"], cfg["LAT_COL"]]
    sites = list(df.groupby(gcols).groups.keys())[:8]
    B = len(sites)

    # Coordinate transform (lon/lat -> UTM meters), fallback if pyproj missing
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs(
            f"EPSG:{cfg.get('COORD_SRC_EPSG', 4326)}",
            f"EPSG:{cfg.get('COORD_TARGET_EPSG', 32649)}",
            always_xy=True,
        )
        def ll_to_xy(lo, la):
            return tr.transform(float(lo), float(la))
    except Exception:
        # crude fallback: treat degrees as meters-ish (debug only)
        lat_mean = float(np.mean([la for _, la in sites]))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_mean))
        def ll_to_xy(lo, la):
            return float(lo) * m_per_deg_lon, float(la) * m_per_deg_lat

    def get_site_year(df_site: pd.DataFrame, year: int, col: str) -> float:
        s = df_site.loc[df_site[cfg["TIME_COL"]] == year, col]
        if len(s) == 0:
            raise ValueError(f"Missing year={year} for col={col}")
        return float(s.iloc[0])

    gwl_col = str(cfg["GWL_COL"])
    subs_col = str(cfg["SUBSIDENCE_COL"])

    # --- SI affine mapping (model-space -> SI meters) for GWL/depth
    if gwl_col.endswith("_z"):
        head_scale_si = float(df.attrs["depth_std_m"]) * float(cfg["HEAD_UNIT_TO_SI"])
        head_bias_si  = float(df.attrs["depth_mu_m"])  * float(cfg["HEAD_UNIT_TO_SI"])
    else:
        head_scale_si = float(cfg["HEAD_UNIT_TO_SI"])
        head_bias_si  = 0.0

    # --- SI affine mapping (model-space -> SI meters) for subsidence
    if subs_col.endswith("_z"):
        subs_scale_si = float(df.attrs["subs_std_m"]) * float(cfg["SUBS_UNIT_TO_SI"])
        subs_bias_si  = float(df.attrs["subs_mu_m"])  * float(cfg["SUBS_UNIT_TO_SI"])
    else:
        # RAW column: value is in "native units" -> convert to SI via SUBS_UNIT_TO_SI
        subs_scale_si = float(cfg["SUBS_UNIT_TO_SI"])
        subs_bias_si  = 0.0

    # v3.2 feature layout (matches your new description)
    DYN_FEATURE_NAMES = [
        gwl_col,               # 0
        "rainfall_mm",         # 1
        "urban_load_global",   # 2  (U*)
        "soil_thickness_eff",  # 3  (H_eff)
    ]
    FUT_FEATURE_NAMES = [
        "rainfall_mm",         # 0
        "urban_load_global",   # 1
        "soil_thickness_eff",  # 2
    ]
    GWL_DYN_INDEX = int(DYN_FEATURE_NAMES.index(gwl_col))

    # Allocate arrays
    X_hist = np.zeros((B, T, len(DYN_FEATURE_NAMES)), float)
    X_fut  = np.zeros((B, T+ H, len(FUT_FEATURE_NAMES)), float) # auto it does T +H for TFT mode. no need 
                                                             # to explicitly set T +H

    # Horizon coords in meters + years
    lon_m = np.zeros((B, H, 1), float)
    lat_m = np.zeros((B, H, 1), float)
    t_raw = np.zeros((B, H, 1), float)

    # Initial conditions (SI)
    h_ref_si = np.zeros((B, 1, 1), float)
    s_init_si = np.zeros((B, 1, 1), float)

    # Static: [lithology_class_id] (keep simple, numeric)
    classes = {
        c: i for i, c in enumerate(
            sorted(df["lithology_class"].dropna().unique().tolist())
        )
    }
    static_features = np.zeros((B, 1), float)

    for i, (lo, la) in enumerate(sites):
        df_site = df[
            (df[cfg["LON_COL"]] == lo) & (df[cfg["LAT_COL"]] == la)
        ].sort_values(cfg["TIME_COL"])

        static_features[i, 0] = float(
            classes.get(str(df_site["lithology_class"].iloc[0]), 0)
        )

        # Fill future covariates over full window (hist + fut)
        for k, y in enumerate(years_all):
            rain = get_site_year(df_site, y, "rainfall_mm")
            ust  = get_site_year(df_site, y, "urban_load_global")
            H_eff = get_site_year(df_site, y, "soil_thickness_eff")
            X_fut[i, k, :] = [rain, ust, H_eff]

        # Fill dynamic history (history-only)
        for j, y in enumerate(hist_years):
            gwl = get_site_year(df_site, y, gwl_col)
            rain = get_site_year(df_site, y, "rainfall_mm")
            ust  = get_site_year(df_site, y, "urban_load_global")
            H_eff = get_site_year(df_site, y, "soil_thickness_eff")
            X_hist[i, j, :] = [gwl, rain, ust, H_eff]

        # Coords + init conditions from last history year
        # - h_ref_si uses proxy head if USE_HEAD_PROXY else true head needs z_surf
        gwl_ref_model = get_site_year(df_site, y_hist_last, gwl_col)
        depth_ref_m = gwl_ref_model * head_scale_si + head_bias_si  # if gwl is depth
        h_ref_si[i, 0, 0] = -depth_ref_m if USE_HEAD_PROXY else 0.0

        subs_init_model = get_site_year(df_site, y_hist_last, subs_col)
        s_init_si[i, 0, 0] = subs_init_model * subs_scale_si + subs_bias_si

        for j, y in enumerate(fut_years):
            x, y_ = ll_to_xy(lo, la)
            lon_m[i, j, 0] = float(x)
            lat_m[i, j, 0] = float(y_)
            t_raw[i, j, 0] = float(y)

    # coords: shift mins to 0
    coords_pack = make_txy_coords(
        t=t_raw[..., 0],        # (B,H)
        x=lon_m[..., 0],        # (B,H)
        y=lat_m[..., 0],        # (B,H)
        time_shift="min",
        xy_shift="min",
    )
    coords = coords_pack.coords  # (B,H,3)

    # optional normalization
    t_rng = float(coords_pack.coord_ranges["t"])
    x_rng = float(coords_pack.coord_ranges["x"])
    y_rng = float(coords_pack.coord_ranges["y"])
    if WORK_WITH_NORMALIZED_COORDS:
        coords = coords.copy()
        coords[..., 0] = coords[..., 0] / max(t_rng, 1e-12)
        coords[..., 1] = coords[..., 1] / max(x_rng, 1e-12)
        coords[..., 2] = coords[..., 2] / max(y_rng, 1e-12)

    # Optional feature scaling for rainfall + H_eff when normalized-data mode
    if infer_work_with_normalized_data(cfg):
        rain_all = X_fut[..., 0]
        H_all = X_fut[..., 2]
        rain_s, _, _ = _minmax_fit_transform(rain_all)
        H_s, _, _ = _minmax_fit_transform(H_all)

        X_hist_s = X_hist.copy()
        X_fut_s  = X_fut.copy()

        # dynamic history: rainfall + H_eff channels
        X_hist_s[..., 1] = rain_s[:, :T]
        X_hist_s[..., 3] = H_s[:, :T]

        # future: rainfall + H_eff channels
        X_fut_s[..., 0] = rain_s
        X_fut_s[..., 2] = H_s
    else:
        X_hist_s = X_hist
        X_fut_s  = X_fut

    # H_field on horizon only: use H_eff from future covariates window
    H_future_m = X_fut[..., 2:3].astype("float32")[:, T:, :]  # (B,H,1)

    scaling_kwargs = dict(
        coords_normalized=bool(WORK_WITH_NORMALIZED_COORDS),
        coords_in_degrees=False,
        time_units=str(cfg.get("TIME_UNITS", "year")),

        coord_ranges=dict(t=t_rng, x=x_rng, y=y_rng),
        coord_mins=dict(
            t=float(coords_pack.coord_mins["t"]),
            x=float(coords_pack.coord_mins["x"]),
            y=float(coords_pack.coord_mins["y"]),
        ),

        bounds=cfg.get("PHYSICS_BOUNDS", {}),

        subs_unit_to_si=float(cfg["SUBS_UNIT_TO_SI"]),
        head_unit_to_si=float(cfg["HEAD_UNIT_TO_SI"]),
        thickness_unit_to_si=float(cfg["THICKNESS_UNIT_TO_SI"]),

        subs_scale_si=float(subs_scale_si),
        subs_bias_si=float(subs_bias_si),
        head_scale_si=float(head_scale_si),
        head_bias_si=float(head_bias_si),

        dynamic_feature_names=list(DYN_FEATURE_NAMES),
        future_feature_names=list(FUT_FEATURE_NAMES),
        gwl_col=gwl_col,
        gwl_dyn_index=int(GWL_DYN_INDEX),

        # option-1 behavior
        subsidence_kind="cumulative",
        allow_subs_residual=True,

        # depth->head rules
        gwl_kind="depth_bgs",
        gwl_sign="down_positive",
        use_head_proxy=bool(USE_HEAD_PROXY),
        z_surf_col=None,

        mv_units= "kPa^-1",     # or "Pa^-1"
        gamma_w_units= "Pa/m",  # or "kPa/m"

    )

    inputs = pack_inputs(
        coords=coords,
        X_hist=X_hist_s,
        X_fut=X_fut_s,
        static_features=static_features,
        H_future_m=H_future_m,
        h_ref_si=h_ref_si,
        s_init_si=s_init_si,
    )

    meta = dict(
        B=B, T=T, H=H,
        hist_years=hist_years,
        fut_years=fut_years,
        coord_ranges=(t_rng, x_rng, y_rng),
        subs_affine_si=(subs_scale_si, subs_bias_si),
        head_affine_si=(head_scale_si, head_bias_si),
        dyn_names=DYN_FEATURE_NAMES,
        fut_names=FUT_FEATURE_NAMES,
    )
    return Stage1LikePack(inputs=inputs, scaling_kwargs=scaling_kwargs, meta=meta)


# ---------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------
def build_model(cfg: Dict[str, Any], scaling_kwargs: Dict[str, Any],
                sample_inputs: Dict[str, np.ndarray]) -> GeoPriorSubsNet:

    static_dim = int(sample_inputs["static_features"].shape[-1])
    dyn_dim    = int(sample_inputs["dynamic_features"].shape[-1])
    fut_dim    = int(sample_inputs["future_features"].shape[-1])

    init_kwargs = dict(
        static_input_dim=static_dim,
        dynamic_input_dim=dyn_dim,
        future_input_dim=fut_dim,

        embed_dim=cfg["EMBED_DIM"],
        hidden_units=cfg["HIDDEN_UNITS"],
        lstm_units=cfg["LSTM_UNITS"],
        attention_units=cfg["ATTENTION_UNITS"],
        num_heads=cfg["NUMBER_HEADS"],
        dropout_rate=cfg["DROPOUT_RATE"],
        forecast_horizon=cfg["FORECAST_HORIZON_YEARS"],
        quantiles=cfg["QUANTILES"],

        mv=cfg["GEOPRIOR_INIT_MV"],
        kappa=cfg["GEOPRIOR_INIT_KAPPA"],
        gamma_w=cfg["GEOPRIOR_GAMMA_W"],
        h_ref=cfg["GEOPRIOR_H_REF"],
        hd_factor=cfg["GEOPRIOR_HD_FACTOR"],
        kappa_mode=cfg["GEOPRIOR_KAPPA_MODE"],
        offset_mode=cfg["OFFSET_MODE"],
        time_units=cfg.get("TIME_UNITS", "year"),

        pde_mode=cfg["PDE_MODE_CONFIG"],
        scale_pde_residuals=cfg["SCALE_PDE_RESIDUALS"],
        scaling_kwargs=scaling_kwargs,

        mode=cfg.get("MODE", None),
        bounds_mode=cfg.get("BOUNDS_MODE", "soft"),
        verbose=7,
    )

    model = GeoPriorSubsNet(**init_kwargs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["LEARNING_RATE"]),
        lambda_cons=cfg["LAMBDA_CONS"],
        lambda_gw=cfg["LAMBDA_GW"],
        lambda_prior=cfg["LAMBDA_PRIOR"],
        lambda_smooth=cfg["LAMBDA_SMOOTH"],
        lambda_mv=cfg["LAMBDA_MV"],
        lambda_bounds=cfg["LAMBDA_BOUNDS"],
        lambda_offset=cfg["LAMBDA_OFFSET"],
    )
    return model


# ---------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------
def _to_np(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

def summarize(name: str, x) -> None:
    arr = np.asarray(_to_np(x), dtype=float).ravel()
    if arr.size == 0:
        print(f"{name}: <empty>")
        return
    q = np.quantile(arr, [0.0, 0.5, 0.95, 1.0])
    print(
        f"{name:>20s} | shape={_to_np(x).shape} "
        f"min={q[0]:.3e} p50={q[1]:.3e} p95={q[2]:.3e} max={q[3]:.3e}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg = DEFAULT_CFG.copy()

    # Switch columns based on WORK_WITH_NORMALIZED_DATA
    if WORK_WITH_NORMALIZED_DATA:
        cfg["GWL_COL"] = "GWL_depth_bgs_z"
        cfg["SUBSIDENCE_COL"] = "subsidence_cum_z"
    else:
        cfg["GWL_COL"] = "GWL_depth_bgs"
        cfg["SUBSIDENCE_COL"] = "subsidence_cum"

    work_norm = infer_work_with_normalized_data(cfg)
    if work_norm != WORK_WITH_NORMALIZED_DATA:
        raise ValueError(
            f"Mismatch: WORK_WITH_NORMALIZED_DATA={WORK_WITH_NORMALIZED_DATA} "
            f"but columns imply normalized={work_norm} "
            f"(GWL_COL={cfg['GWL_COL']}, SUBSIDENCE_COL={cfg['SUBSIDENCE_COL']})."
        )

    print("[CFG] normalized_data =", WORK_WITH_NORMALIZED_DATA)
    print("[CFG] normalized_coords =", WORK_WITH_NORMALIZED_COORDS)
    print("[CFG] USE_HEAD_PROXY =", USE_HEAD_PROXY)
    print("[CFG] GWL_COL =", cfg["GWL_COL"], "| SUBSIDENCE_COL =", cfg["SUBSIDENCE_COL"])

    df = make_fake_df(cfg, n_sites=16)

    pack = build_one_sample(cfg, df)
    print("[DATA] meta:", pack.meta)
    print("[SANITY] dyn names:", pack.scaling_kwargs["dynamic_feature_names"])
    print("[SANITY] fut names:", pack.scaling_kwargs["future_feature_names"])
    print("[SANITY] gwl_dyn_index:", pack.scaling_kwargs["gwl_dyn_index"])

    model = build_model(cfg, pack.scaling_kwargs, pack.inputs)
    xb = {k: tf.convert_to_tensor(v) for k, v in pack.inputs.items()}

    print("[SHAPES] coords:", xb["coords"].shape)
    print("[SHAPES] H_field:", xb["H_field"].shape)
    print("[SHAPES] dyn:", xb["dynamic_features"].shape)
    print("[SHAPES] fut:", xb["future_features"].shape)
    print("[SHAPES] h_ref_si:", xb["h_ref_si"].shape)
    print("[SHAPES] s_init_si:", xb["s_init_si"].shape)

    # Forward pass sanity
    _ = model(xb, training=False)
    print("[OK] Forward pass succeeded.")

    # Physics diagnostics
    phys = model.evaluate_physics(xb, return_maps=True, max_batches=None, batch_size=None)
    print("[OK] evaluate_physics(...) succeeded.")

    # Print whatever keys are available (robust across refactors)
    for k in [
        "epsilon_prior", "epsilon_cons", "epsilon_gw",
        "physics_loss_raw", "physics_mult", "physics_loss_scaled",
        "consolidation_loss", "gw_flow_loss", "prior_loss",
        "smooth_loss", "mv_prior_loss", "bounds_loss",
        "loss_consolidation", "loss_gw_flow", "loss_prior",
        "loss_smooth", "loss_mv", "loss_bounds",
    ]:
        if k in phys:
            summarize(k, phys[k])

    for k in ["K_field", "Ss_field", "tau_field", "tau_phys", "Hd_eff", "H_field", "Q_si"]:
        if k in phys:
            summarize(k, phys[k])

    sec_per = seconds_per_time_unit(model.time_units)
    if "epsilon_cons" in phys:
        tf.print("epsilon_cons [m/s] =", phys["epsilon_cons"])
        tf.print("epsilon_cons [mm/time_unit] =", phys["epsilon_cons"] * sec_per * 1000.0)

    if "tau_field" in phys:
        tf.print("tau_field [time_unit] =", phys["tau_field"] / sec_per)

    tf.print("pde_modes_active =", model.pde_modes_active)

    print("\nDone.")

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
