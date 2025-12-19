#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_geoprior_physics_harness.py

Minimal physics-debug harness for GeoPriorSubsNet (synthetic data).
"""

from __future__ import annotations

import os
import math
import inspect
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from fusionlab.nn.pinn._geoprior_subnet import GeoPriorSubsNet
from fusionlab.nn.pinn.op import seconds_per_time_unit
# ---------------------------------------------------------------------
# 0) Defaults mirroring your nat.com/config.py (only what we need here)
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
    SUBSIDENCE_COL="subsidence_cum",
    GWL_COL="head_z",
    H_FIELD_COL_NAME="soil_thickness",

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
    LAMBDA_MV=0.01,
    LAMBDA_BOUNDS = 1e-3,  # or 1e-2 if you see drift; 0.0 is okay for pure debug
    OFFSET_MODE="mul",
    LAMBDA_OFFSET=1.0,

    PHYSICS_BOUNDS=dict(
        H_min=5.0, H_max=80.0,
        K_min=1e-8, K_max=1e-3,
        Ss_min=1e-4, Ss_max=1e-3,
    ),

    TIME_UNITS="yr",

    # Units
    SUBS_UNIT_TO_SI=1e-3,   # mm -> m
    HEAD_UNIT_TO_SI=1.0,    # m -> m
    THICKNESS_UNIT_TO_SI=1.0, # m  -> m

    # GeoPrior scalars
    GEOPRIOR_INIT_MV=1e-7,
    GEOPRIOR_INIT_KAPPA=1.0,
    GEOPRIOR_GAMMA_W=9810.0,
    GEOPRIOR_H_REF=0.0,
    GEOPRIOR_KAPPA_MODE="kb",
    GEOPRIOR_HD_FACTOR=0.6,

    # Optim
    LEARNING_RATE=1e-4,
)


# ---------------------------------------------------------------------
# 2) Synthetic dataframe with your columns
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

    litho_classes = ["Fine-Grained Soil", "Coarse-Grained Soil",
                     "Mixed Clastics", "Carbonate"]
    lithos = ["Mudstone–Siltstone", "Conglomerate–Sandstone",
              "Sandstone–Siltstone", "Limestone–Sandstone"]

    rows = []
    for i in range(n_sites):
        city = str(cfg.get("CITY_NAME", "zhongshan")).title()
        lithology = lithos[i % len(lithos)]
        lithology_class = litho_classes[i % len(litho_classes)]

        # static-ish site properties
        urban0 = rng.uniform(0.05, 0.95)          # already [0,1]
        H0 = rng.uniform(0.5, 25.0)               # m
        rain0 = rng.uniform(700.0, 1400.0)        # mm

        # NEW: surface elevation (datum) and baseline depth-to-water
        z_surf = float(rng.uniform(0.0, 30.0))    # m (synthetic DEM)
        depth0 = float(rng.uniform(5.0, 60.0))    # m (positive downward)
        # depth-to-water (BGS), positive downward
        depth = depth0 + 0.03 * (900.0 - rain0) + rng.normal(0.0, 1.5)
        depth = float(np.clip(depth, 0.5, 120.0))

        
        subs_annual = []
        depth_series = []
        head_series = []
        H_series = []
        rain_series = []

        for y in years:
            rain = rain0 + 80.0 * math.sin((y - years[0]) * 0.7) + rng.normal(0, 40.0)
            rain = float(np.clip(rain, 200.0, 2500.0))

            # depth-to-water (BGS), positive downward
            depth = depth0 + 0.03 * (900.0 - rain) + rng.normal(0.0, 1.5)
            depth = float(np.clip(depth, 0.5, 120.0))  # m
            
            # hydraulic head in meters (datum = z_surf)
            head_m = z_surf - depth  # h = z_surf - z_GWL
            
            depth_series.append(depth)
            head_series.append(head_m)

            # hydraulic head in meters
            head_m = z_surf - depth  # h = z_surf - z_GWL

            H = float(np.clip(H0 + rng.normal(0, 0.3), 0.1, 40.0))

            # # drawdown relative to baseline year (positive for head loss)
            head_ref = z_surf - depth0
            drawdown = max(0.0, head_ref - head_m)  # = max(0, depth - depth0)

            # toy annual subsidence in mm/yr (depends on drawdown + urban + rain)
            s_mm = (
                2.0
                + 10.0 * (urban0 ** 1.3)
                + 1.5 * drawdown               # mm per meter drawdown (toy)
                + 0.002 * max(0.0, (rain - 900.0))
                + rng.normal(0.0, 1.0)
            )
            s_mm = float(np.clip(s_mm, 0.0, 40.0))

            subs_annual.append(s_mm)
            depth_series.append(depth)
            head_series.append(head_m)
            H_series.append(H)
            rain_series.append(rain)

        subs_annual = np.asarray(subs_annual, float)
        subs_cum = np.cumsum(subs_annual)  # cumulative mm

        for j, y in enumerate(years):
            H = float(H_series[j])
            censored = bool(H >= 30.0 - 1e-6)
            H_eff = float(min(H, 30.0))

            depth = float(depth_series[j])
            head_m = float(head_series[j])

            rows.append(
                dict(
                    longitude=float(lons[i]),
                    latitude=float(lats[i]),
                    year=int(y),
                    lithology=lithology,

                    # keep your columns (now *filled*)
                    GWL=float(depth),                 # depth-to-water (m)
                    GWL_depth_bgs=float(depth),       # depth-to-water (m)
      
                    head_m=float(head_m),
                    z_surf_m=float(z_surf),

                    rainfall_mm=float(rain_series[j]),
                    soil_thickness=float(H),
                    normalized_urban_load_proxy=float(urban0),
                    subsidence=float(subs_annual[j]),
                    subsidence_cum=float(subs_cum[j]),
                    city=city,
                    lithology_class=lithology_class,

                    # # NEW: surface + head in meters
                    # z_surf_m=float(z_surf),
                    # head_m=float(head_m),

                    # censoring/thickness fields
                    soil_thickness_censored=censored,
                    soil_thickness_imputed=float(H),
                    soil_thickness_eff=float(H_eff),
                    urban_load_global=float(urban0),
                )
            )

    df = pd.DataFrame(rows)

    # Create z-scored versions (like your real dataset)
    def _zscore(col: str) -> Tuple[pd.Series, float, float]:
        mu = float(df[col].mean())
        sig = float(df[col].std(ddof=0))
        sig = max(sig, 1e-12)
        return (df[col] - mu) / sig, mu, sig


    df["GWL_depth_bgs_z"], depth_mu, depth_std = _zscore("GWL_depth_bgs")
    
    # standardize hydraulic head (meters) for the model
    df["head_z"], head_mu, head_std = _zscore("head_m")
    
    df.attrs["head_mu_m"]  = head_mu
    df.attrs["head_std_m"] = head_std
    df.attrs["depth_mu_m"] = depth_mu
    df.attrs["depth_std_m"]= depth_std


    return df


# ---------------------------------------------------------------------
# 3) Build one tft_like sample + scaling kwargs
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
    litho_id: np.ndarray,
    H_future_raw_m: np.ndarray,
) -> Dict[str, np.ndarray]:


    return {
        "coords": coords.astype("float32"),
        "dynamic_features": X_hist.astype("float32"),
        "future_features": X_fut.astype("float32"),
        # simplest static: (B,1) float feature
        "static_features": litho_id.astype("float32"),
        # physics thickness input (train_step accepts either key)
        # "soil_thickness": H_future_raw_m.astype("float32"),
        # OR use "H_field" instead if you prefer:
        "H_field": H_future_raw_m.astype("float32"),
    }


def build_one_sample(cfg: Dict[str, Any], df: pd.DataFrame) -> Stage1LikePack:
    T = int(cfg["TIME_STEPS"])
    H = int(cfg["FORECAST_HORIZON_YEARS"])
    y_start = int(cfg["FORECAST_START_YEAR"])

    hist_years = list(range(y_start - T, y_start))
    fut_years = list(range(y_start, y_start + H))

    gcols = [cfg["LON_COL"], cfg["LAT_COL"]]
    sites = list(df.groupby(gcols).groups.keys())[:8]
    
    lat_mean = float(np.mean([la for _, la in sites]))
    deg_to_m_lat = 111_320.0
    deg_to_m_lon = 111_320.0 * math.cos(math.radians(lat_mean))

    B = len(sites)

    def get_site_year(df_site: pd.DataFrame, year: int, col: str) -> float:
        s = df_site.loc[df_site[cfg["TIME_COL"]] == year, col]
        if len(s) == 0:
            raise ValueError(f"Missing year={year} for col={col}")
        return float(s.iloc[0])

    # TH = T + H
    # years_all = hist_years + fut_years  # length TH

    lon = np.zeros((B, H, 1), float)
    lat = np.zeros((B, H, 1), float)
    t_raw = np.zeros((B, H, 1), float)


    # hist: [gwl_z, rainfall_mm, urban_load_global, H_eff, censor_flag]
    X_hist = np.zeros((B, T, 5), float)
    # fut:  [rainfall_mm, urban_load_global, H_eff, censor_flag]
    TH = T + H
    X_fut = np.zeros((B, TH, 4), float)   # (B, T+H, 4)

    # future (for SI affine inference)
    y_subs_fut = np.zeros((B, H, 1), float)
    y_gwl_fut = np.zeros((B, H, 1), float)

    litho_id = np.zeros((B, 1), int)
    classes = {c: i for i, c in enumerate(sorted(df["lithology_class"].dropna().unique().tolist()))}
    
    # n_classes = len(classes)
    # static_inputs = np.eye(n_classes, dtype="float32")[litho_id[:, 0]]  # (B, C)

    for i, (lo, la) in enumerate(sites):
        df_site = df[(df[cfg["LON_COL"]] == lo) & (df[cfg["LAT_COL"]] == la)].sort_values(cfg["TIME_COL"])
        litho_id[i, 0] = classes.get(str(df_site["lithology_class"].iloc[0]), 0)
    
        # ---- fill FUTURE COVARIATES over FULL window: hist + fut ----
        years_all = hist_years + fut_years  # length TH
    
        for k, y in enumerate(years_all):
            rain = get_site_year(df_site, y, "rainfall_mm")
            urb  = get_site_year(df_site, y, "urban_load_global")
            H_eff = get_site_year(df_site, y, "soil_thickness_eff")
            cens = float(bool(get_site_year(df_site, y, "soil_thickness_censored")))
            X_fut[i, k, :] = [rain, urb, H_eff, cens]
    
        # ---- coords + targets remain on horizon only ----
        for j, y in enumerate(fut_years):
            lon[i, j, 0] = float(lo)
            lat[i, j, 0] = float(la)
            t_raw[i, j, 0] = float(y)
    
            y_subs_fut[i, j, 0] = get_site_year(df_site, y, cfg["SUBSIDENCE_COL"])
            y_gwl_fut[i, j, 0]  = get_site_year(df_site, y, cfg["GWL_COL"])
    
        # ---- dynamic history stays history-only ----
        for j, y in enumerate(hist_years):
            gwl = get_site_year(df_site, y, cfg["GWL_COL"])
            rain = get_site_year(df_site, y, "rainfall_mm")
            urb  = get_site_year(df_site, y, "urban_load_global")
            H_eff = get_site_year(df_site, y, "soil_thickness_eff")
            cens = float(bool(get_site_year(df_site, y, "soil_thickness_censored")))
            X_hist[i, j, :] = [gwl, rain, urb, H_eff, cens]


    # --- coords minmax -> [0,1] over FULL window (T+H)
    t_norm,  t_min, t_rng = _minmax_fit_transform(t_raw)
    lon_norm, x_min, x_rng = _minmax_fit_transform(lon)
    lat_norm, y_min, y_rng = _minmax_fit_transform(lat)
    coords = np.concatenate([t_norm, lon_norm, lat_norm], axis=-1).astype("float32")


    # --- scale rainfall + H_eff; keep urban_load_global as-is
    # rain_all = np.concatenate([X_hist[..., 1], X_fut[..., 0]], axis=1)
    # H_all = np.concatenate([X_hist[..., 3], X_fut[..., 2]], axis=1)
    rain_all = X_fut[..., 0]     # (B, T+H)
    H_all    = X_fut[..., 2]     # (B, T+H)

    # rain_s, rain_min, rain_rng = _minmax_fit_transform(rain_all)
    # H_s, H_min, H_rng = _minmax_fit_transform(H_all)

    # X_hist_s = X_hist.copy()
    # X_fut_s = X_fut.copy()
    # X_hist_s[..., 1] = rain_s[:, :T]
    # X_fut_s[..., 0] = rain_s[:, T:T + H]
    # X_hist_s[..., 3] = H_s[:, :T]
    # X_fut_s[..., 2] = H_s[:, T:T + H]
    rain_s, rain_min, rain_rng = _minmax_fit_transform(rain_all)
    H_s,    H_min,    H_rng    = _minmax_fit_transform(H_all)
    
    X_hist_s = X_hist.copy()
    X_fut_s  = X_fut.copy()
    
    # history part
    X_hist_s[..., 1] = rain_s[:, :T]
    X_hist_s[..., 3] = H_s[:, :T]
    
    # full future-cov window
    X_fut_s[..., 0]  = rain_s
    X_fut_s[..., 2]  = H_s

    # --- SI affine mapping (model-space -> SI)
    # (best-effort for physics debug; adapt if your real Stage-1 differs)
    _, subs_min, subs_rng = _minmax_fit_transform(y_subs_fut)
    _, head_min, head_rng = _minmax_fit_transform(y_gwl_fut)

    subs_scale_si = subs_rng * float(cfg["SUBS_UNIT_TO_SI"])
    subs_bias_si = subs_min * float(cfg["SUBS_UNIT_TO_SI"])
    head_scale_si = float(df.attrs["head_std_m"]) * float(cfg["HEAD_UNIT_TO_SI"])
    head_bias_si  = float(df.attrs["head_mu_m"])  * float(cfg["HEAD_UNIT_TO_SI"])


    scaling_kwargs = dict(
        coords_normalized=True,
        coords_in_degrees=False,
        time_units=cfg.get("TIME_UNITS", "year"),

        # multiple spellings for your _coord_ranges() implementation
        coord_ranges=dict(t=t_rng, x=x_rng, y=y_rng),
        t_range=t_rng, x_range=x_rng, y_range=y_rng,
        coord_range_t=t_rng, coord_range_x=x_rng, coord_range_y=y_rng,
        coord_mins=dict(t=t_min, x=x_min, y=y_min),

        bounds=cfg.get("PHYSICS_BOUNDS", {}),

        subs_unit_to_si=float(cfg["SUBS_UNIT_TO_SI"]),
        head_unit_to_si=float(cfg["HEAD_UNIT_TO_SI"]),
        thickness_unit_to_si=float(cfg.get("THICKNESS_UNIT_TO_SI", 1.0)),

        subs_scale_si=subs_scale_si,
        subs_bias_si=subs_bias_si,
        head_scale_si=head_scale_si,
        head_bias_si=head_bias_si,
    )
    scaling_kwargs.update(dict(
        coords_in_degrees=True,
        deg_to_m_lon=deg_to_m_lon,
        deg_to_m_lat=deg_to_m_lat,
    ))

    # H-field over full window (T+H), in raw meters
    # # IMPORTANT: feed H in RAW meters for physics (reconstruct from scaler)
    # # H_future_raw_m = (X_fut_s[..., 2:3] * H_rng + H_min).astype("float32")
    # H_future_raw_m = H_all_raw_m
    H_fut_raw_m  = X_fut[..., 2:3].astype("float32")[:, T:, :]   # (B, H, 1)
    H_future_raw_m = H_fut_raw_m

    inputs = pack_inputs(
        coords=coords,
        X_hist=X_hist_s,
        X_fut=X_fut_s,
        litho_id=litho_id,
        H_future_raw_m=H_future_raw_m,
    )


    meta = dict(
        B=B, T=T, H=H,
        hist_years=hist_years,
        fut_years=fut_years,
        coord_ranges=(t_rng, x_rng, y_rng),
        subs_affine_si=(subs_scale_si, subs_bias_si),
        head_affine_si=(head_scale_si, head_bias_si),
    )
    return Stage1LikePack(inputs=inputs, scaling_kwargs=scaling_kwargs, meta=meta)


# ---------------------------------------------------------------------
# 4) Build model with signature-filtered kwargs
# ---------------------------------------------------------------------
def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(callable_obj)
    params = sig.parameters

    # If callable accepts **kwargs, DO NOT drop extra keys
    has_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )
    if has_varkw:
        return dict(kwargs)

    return {k: v for k, v in kwargs.items() if k in params}


def build_model(cfg: Dict[str, Any], scaling_kwargs: Dict[str, Any],
                sample_inputs: Dict[str, np.ndarray]):

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

        mode=cfg.get("MODE", None),  # optional but often helpful
    )
    init_kwargs["bounds_mode"] = cfg.get("BOUNDS_MODE", "soft")
    model = GeoPriorSubsNet(**init_kwargs)

    # compile is optional for evaluate_physics, but ok to keep
    compile_kwargs = dict(
        optimizer=tf.keras.optimizers.Adam(cfg["LEARNING_RATE"]),
        lambda_cons=cfg["LAMBDA_CONS"],
        lambda_gw=cfg["LAMBDA_GW"],
        lambda_prior=cfg["LAMBDA_PRIOR"],
        lambda_smooth=cfg["LAMBDA_SMOOTH"],
        lambda_mv=cfg["LAMBDA_MV"],
        lambda_bounds=cfg["LAMBDA_BOUNDS"],
        lambda_offset=cfg["LAMBDA_OFFSET"],
    )
    compile_kwargs["lambda_bounds"] = cfg.get("LAMBDA_BOUNDS", 0.0)
    model.compile(**compile_kwargs)
    return model



# ---------------------------------------------------------------------
# 5) Small printing helpers
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
        f"{name:>18s} | shape={_to_np(x).shape} "
        f"min={q[0]:.3e} p50={q[1]:.3e} p95={q[2]:.3e} max={q[3]:.3e}"
    )


# ---------------------------------------------------------------------
# 6) Main
# ---------------------------------------------------------------------
def main():
    cfg = DEFAULT_CFG.copy() 

    print("[CFG] City:", cfg["CITY_NAME"], "| Model:", cfg["MODEL_NAME"])
    print("[CFG] T:", cfg["TIME_STEPS"], "H:", cfg["FORECAST_HORIZON_YEARS"],
          "| train_end:", cfg["TRAIN_END_YEAR"], "forecast_start:", cfg["FORECAST_START_YEAR"])

    df = make_fake_df(cfg, n_sites=16)
    pack = build_one_sample(cfg, df)
    print("[DATA] meta:", pack.meta)

    model = build_model(cfg, pack.scaling_kwargs, pack.inputs)
    xb = {k: tf.convert_to_tensor(v) for k, v in pack.inputs.items()}
    
    print("coords:", pack.inputs["coords"].shape)
    print("H_field:", pack.inputs["H_field"].shape)
    print("dyn:", pack.inputs["dynamic_features"].shape)
    print("fut:", pack.inputs["future_features"].shape)

    # Forward pass
    try:
        _ = model(xb, training=False)
        print("[OK] Forward pass succeeded.")
    except Exception as e:
        print("\n[ERR] Forward pass failed (input keys/shapes mismatch).")
        print("      Available keys:", list(xb.keys()))
        print("      Fix: edit pack_inputs() to match your model. Error:", repr(e))
        raise

    # Physics diagnostics
    phys = model.evaluate_physics(xb, return_maps=True, max_batches=None, batch_size=None)
    print("[OK] evaluate_physics(...) succeeded.")

    for k in [
        "epsilon_prior", "epsilon_cons",
        "physics_loss_raw", "physics_mult", "physics_loss_scaled",
        "consolidation_loss", "gw_flow_loss", "prior_loss",
        "smooth_loss", "mv_prior_loss", "bounds_loss",
    ]:
        if k in phys:
            summarize(k, phys[k])

    if ("tau" in phys) and ("tau_prior" in phys):
        summarize("tau", phys["tau"])
        summarize("tau_phys", phys["tau_prior"])
        ratio = _to_np(phys["tau"]) / np.maximum(_to_np(phys["tau_prior"]), 1e-30)
        summarize("tau/tau_phys", ratio)

    for k in ["K", "Ss", "Hd", "H_field", "H_in"]:
        if k in phys:
            summarize(k, phys[k])

    if hasattr(model, "current_mv"):
        summarize("mv", model.current_mv())
    if hasattr(model, "current_kappa"):
        summarize("kappa", model.current_kappa())
        
    sec_per = seconds_per_time_unit(cfg["TIME_UNITS"])
    eps_cons_mm_per_unit = phys["epsilon_cons"] * sec_per * 1000.0
    tf.print("epsilon_cons [mm/time_unit] =", eps_cons_mm_per_unit)
    tf.print("tau [time_unit] =", phys["tau"] / sec_per)
    tf.print("pde_modes_active =", model.pde_modes_active)


    print("\nDone. If tau and tau_phys differ by many orders at init,")
    print("you’ll typically see a big consolidation/prior term at epoch-1.")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()


