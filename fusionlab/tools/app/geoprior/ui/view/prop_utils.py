# geoprior/ui/map/prop_utils.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.prop_utils

Utilities for Subsidence Propagation & Simulation.
Handles temporal extrapolation and spatial 
gradient (flow) analysis.
"""

from __future__ import annotations

from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import griddata 

from ..map.coord_utils import ensure_lonlat

def extrapolate_scenarios(
    df: pd.DataFrame,
    *,
    years_to_add: int = 5,
    time_col: str = "coord_t",
    id_cols: List[str] = ["coord_x", "coord_y"],
    value_col: str = "subsidence_q50",
) -> pd.DataFrame:
    """
    Simulate future subsidence based on existing trend per location.
    
    Returns a DataFrame containing original data + N future years.
    """
    if df.empty or years_to_add <= 0:
        return df.copy()

    # --- 1. Robust Column Detection (Fix for KeyError: '') ---
    use_t = time_col
    if not use_t or use_t not in df.columns:
        # Fallbacks: standard 't' (from build_points), or common names
        for c in ["t", "coord_t", "year", "date"]:
            if c in df.columns:
                use_t = c
                break
        else:
            # No time column found -> cannot simulate
            return df.copy()

    use_v = value_col
    if not use_v or use_v not in df.columns:
        # Fallbacks: standard 'v' (from build_points), or common names
        for c in ["v", "subsidence", "subsidence_q50", "subsidence_pred"]:
            if c in df.columns:
                use_v = c
                break
        else:
            return df.copy()

    # --- 2. Data Preparation & Type Safety ---
    d = df.copy()
    
    # Safely convert time to integer years (Fix for TypeError: Timestamp vs int)
    if pd.api.types.is_datetime64_any_dtype(d[use_t]):
        d["_t"] = d[use_t].dt.year
    else:
        # Try numeric first (float/int)
        d["_t"] = pd.to_numeric(d[use_t], errors="coerce")
        
        # If numeric failed (e.g. object column with Dates), try datetime parse
        if d["_t"].isna().any():
            try:
                dt_series = pd.to_datetime(d[use_t], errors='coerce')
                mask = d["_t"].isna() & dt_series.notna()
                d.loc[mask, "_t"] = dt_series.loc[mask].dt.year
            except Exception:
                pass

    # Drop invalid rows (missing time or value)
    d = d.dropna(subset=["_t", use_v])
    
    if d.empty:
        return df.copy()

    # Force internal time to int for consistent math/sorting
    d["_t"] = d["_t"].astype(int)
    
    # Overwrite valid time column with clean integers for result consistency
    # This prevents 'int' vs 'Timestamp' comparison errors later
    d[use_t] = d["_t"]

    # --- 3. Identification ---
    # Create composite key for location grouping
    # If id_cols missing, try to use coordinates
    valid_ids = [c for c in id_cols if c in d.columns]
    if not valid_ids:
        # Fallback to lon/lat if standard ID cols are missing
        if "lon" in d.columns and "lat" in d.columns:
            valid_ids = ["lon", "lat"]
        else:
            # Last resort: treat entire dataset as one group (unlikely useful but safe)
            d["_loc_id"] = "all"
    
    if "_loc_id" not in d.columns:
        d["_loc_id"] = d[valid_ids].astype(str).agg('_'.join, axis=1)
    
    # --- 4. Extrapolation Setup ---
    max_year = int(d["_t"].max())
    future_years = np.arange(
        max_year + 1, 
        max_year + 1 + years_to_add
    )
    
    # ---------------------------------------------------------
    # Vectorized Extrapolation (Linear Trend)
    # y = mx + c
    # ---------------------------------------------------------
    def _forecast_group(g):
        if len(g) < 2:
            # Not enough data for trend, repeat last value
            last_val = g[use_v].iloc[-1]
            return pd.DataFrame({
                use_t: future_years,
                use_v: last_val,
                "_is_simulated": True
            })
            
        # Linear regression
        x = g["_t"].values
        y = g[use_v].values
        
        # Slope (m) and intercept (c) via Least Squares
        # A = [x, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Predict
        y_pred = m * future_years + c
        
        return pd.DataFrame({
            use_t: future_years,
            use_v: y_pred,
            "_is_simulated": True
        })

    # Apply forecast
    simulated = d.groupby("_loc_id").apply(_forecast_group).reset_index()
    
    # Merge metadata (coordinates/IDs) back to simulated rows
    meta = d.drop_duplicates(subset="_loc_id")[["_loc_id"] + valid_ids]
    
    # Merge
    final_sim = pd.merge(simulated, meta, on="_loc_id", how="left")
    
    # Cleanup
    final_sim = final_sim.drop(
        columns=["_loc_id", "level_1"], 
        errors="ignore"
    )
    
    # Combine original + simulated
    d["_is_simulated"] = False
    combined = pd.concat([d, final_sim], ignore_index=True)
    
    # Sort by the validated time column
    return combined.sort_values([use_t])

def compute_propagation_vectors(
    df_slice: pd.DataFrame,
    value_col: str = "subsidence_q50",
    grid_res: int = 50
) -> List[Dict[str, float]]:
    """
    Calculate spatial gradients (direction of subsidence expansion).
    Returns a list of vectors {lat, lon, angle, mag} for Quiver plots.
    """
    if df_slice.empty:
        return []

    # Handle missing/empty value_col for vectors too
    use_v = value_col
    if not use_v or use_v not in df_slice.columns:
        for c in ["v", "subsidence", "subsidence_q50"]:
            if c in df_slice.columns:
                use_v = c
                break
        else:
            return []

    # 1. Normalize coordinates to lon/lat
    # Pass actual EPSG if known (omitted here for simplicity, relies on auto)
    pts, ok, _ = ensure_lonlat(df_slice, mode="auto", src_epsg=None) 
    if not ok: 
        return []

    x = pts["lon"].values
    y = pts["lat"].values
    z = pts[use_v].values

    # 2. Grid interpolation (to calculate smooth gradients)
    # Create a regular grid
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)
    
    try:
        ZI = griddata((x, y), z, (XI, YI), method='linear')
    except Exception:
        return []

    # 3. Compute Gradients (dy, dx)
    # Gradient points "uphill". Subsidence flows "downhill" (usually).
    # We want arrows pointing towards greater values (or gradients).
    
    dy, dx = np.gradient(ZI)
    
    # 4. Downsample for visibility
    skip = 2
    vectors = []
    
    rows, cols = ZI.shape
    
    for r in range(0, rows, skip):
        for c in range(0, cols, skip):
            mag = np.sqrt(dx[r,c]**2 + dy[r,c]**2)
            if np.isnan(mag) or mag < 0.1: # Threshold for flat areas
                continue
                
            # Angle in degrees
            angle = np.degrees(np.arctan2(dy[r,c], dx[r,c]))
            
            vectors.append({
                "lon": float(XI[r,c]),
                "lat": float(YI[r,c]),
                "angle": float(angle),
                "mag": float(mag)
            })
            
    return vectors