# add_zsurf_from_coords.py
# -*- coding: utf-8 -*-
"""
Add surface elevation (z_surf) to the main harmonized dataset by
merging on (longitude, latitude). Works for Nansha / Zhongshan by
switching `CITY`.

It also optionally computes hydraulic head (head_m) if a depth-bgs
column is present (depth positive downward): head = z_surf - depth.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
CITY = "zhongshan"  # <-- switch to "zhongshan"

DATA_ROOT = r"F:\repositories\fusionlab-learn\data"
COORDS_ROOT = (
    r"F:\repositories\fusionlab-learn\data"
    r"\prd_coords_with_elevation\prd_coords_with_elevation"
)

# merge tolerance control (float coordinates)
ROUND_DECIMALS = 6

# output naming
OUT_SUFFIX = ".with_zsurf.csv"


def _round_coords(df: pd.DataFrame, lon="longitude", lat="latitude",
                  decimals: int = 6) -> pd.DataFrame:
    df = df.copy()
    df[lon] = pd.to_numeric(df[lon], errors="coerce").round(decimals)
    df[lat] = pd.to_numeric(df[lat], errors="coerce").round(decimals)
    return df


def add_zsurf_to_city_dataset(
    city: str,
    *,
    data_root: str,
    coords_root: str,
    round_decimals: int = 6,
    out_suffix: str = ".with_zsurf.csv",
) -> str:
    # ---- paths -------------------------------------------------------
    main_path = os.path.join(data_root, f"{city}_final_main_std.harmonized.csv")
    elev_path = os.path.join(coords_root, f"{city}_coords_with_elevation.csv")

    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Main dataset not found: {main_path}")
    if not os.path.exists(elev_path):
        raise FileNotFoundError(f"Elevation coords file not found: {elev_path}")

    # ---- load --------------------------------------------------------
    main = pd.read_csv(main_path)
    elev = pd.read_csv(elev_path)

    # ---- normalize column names a bit --------------------------------
    # (keeps your existing column spellings, but ensures the merge cols exist)
    for df, label in [(main, "main"), (elev, "elev")]:
        missing = {"longitude", "latitude"} - set(df.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {missing}")

    if "elevation" not in elev.columns:
        raise ValueError("Elevation file must contain an 'elevation' column.")

    # ---- round coords to avoid float merge misses --------------------
    main_r = _round_coords(main, decimals=round_decimals)
    elev_r = _round_coords(elev, decimals=round_decimals)

    # ---- de-duplicate elevation per (lon,lat) if needed --------------
    # (mean is safe; you can also use median)
    elev_r = (
        elev_r.groupby(["longitude", "latitude"], as_index=False)["elevation"]
        .mean()
    )

    # ---- merge -------------------------------------------------------
    merged = main_r.merge(
        elev_r,
        on=["longitude", "latitude"],
        how="left",
        validate="m:1",  # many rows per coord in main (years), 1 in elev
    )

    # rename to the name we want in physics code
    # (keep raw 'elevation' too if you prefer; here we use z_surf)
    merged = merged.rename(columns={"elevation": "z_surf"})

    # ---- quick diagnostics ------------------------------------------
    miss = merged["z_surf"].isna().mean()
    n_miss = int(merged["z_surf"].isna().sum())
    n_tot = len(merged)
    print(f"[{city}] merged rows: {n_tot}")
    print(f"[{city}] missing z_surf: {n_miss} ({miss:.2%})")

    # ---- OPTIONAL: compute hydraulic head ----------------------------
    # If you have depth below ground surface (positive down):
    # head (elevation datum) = z_surf - depth
    depth_col = None
    for c in ("GWL_depth_bgs", "GWL_depth_bgs_m", "gwl_depth_bgs"):
        if c in merged.columns:
            depth_col = c
            break

    if depth_col is not None:
        merged["head_m"] = merged["z_surf"] - pd.to_numeric(
            merged[depth_col], errors="coerce"
        )
        print(f"[{city}] computed head_m from z_surf and {depth_col}")
    else:
        print(f"[{city}] depth-bgs column not found -> head_m not computed")

    # ---- save --------------------------------------------------------
    out_path = os.path.join(
        data_root,
        f"{city}_final_main_std.harmonized{out_suffix}",
    )
    merged.to_csv(out_path, index=False)
    print(f"[{city}] saved: {out_path}")

    return out_path


if __name__ == "__main__":
    add_zsurf_to_city_dataset(
        CITY,
        data_root=DATA_ROOT,
        coords_root=COORDS_ROOT,
        round_decimals=ROUND_DECIMALS,
        out_suffix=OUT_SUFFIX,
    )
