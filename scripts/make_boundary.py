# scripts/make_boundary.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as cfg
from . import utils
try:
    import geopandas as gpd
    from shapely.geometry import mapping
except Exception as e:
    raise SystemExit(
        "Need geopandas + shapely installed for boundary export. "
        f"Error: {e}"
    )
from shapely import concave_hull

_CITY_A = cfg.CITY_CANON.get("ns", "Nansha")
_CITY_B = cfg.CITY_CANON.get("zh", "Zhongshan")


def _pick_paths(
    art: utils.Artifacts,
    split: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    if split == "val":
        return art.forecast_val_csv, art.forecast_future_csv
    if split == "test":
        return art.forecast_test_csv, art.forecast_test_future_csv
    if (
        art.forecast_test_csv is not None
        and art.forecast_test_future_csv is not None
    ):
        return art.forecast_test_csv, art.forecast_test_future_csv
    return art.forecast_val_csv, art.forecast_future_csv


def _load_xy(path: str) -> np.ndarray:
    df = pd.read_csv(utils.as_path(path))
    utils.ensure_columns(df, aliases=cfg._BASE_ALIASES)
    for c in ("coord_x", "coord_y"):
        if c not in df.columns:
            raise KeyError(f"{path}: missing {c}")
    x = pd.to_numeric(df["coord_x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["coord_y"], errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    return np.column_stack([x[m], y[m]])


def _resolve_city(
    *,
    city: str,
    src: Optional[str],
    eval_csv: Optional[str],
    future_csv: Optional[str],
    split: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": city}
    if eval_csv and future_csv:
        out["eval_csv"] = str(utils.as_path(eval_csv))
        out["future_csv"] = str(utils.as_path(future_csv))
        return out

    if not src:
        raise ValueError(f"{city}: provide --*-src or --*-eval/--*-future")

    art = utils.detect_artifacts(src)
    ev, fu = _pick_paths(art, split)
    if ev is None or fu is None:
        raise FileNotFoundError(f"{city}: missing eval/future under {src}")

    out["eval_csv"] = str(ev)
    out["future_csv"] = str(fu)
    return out


def _poly_from_points(
    xy: np.ndarray,
    *,
    method: str,
    alpha: float,
):
    from shapely.geometry import MultiPoint

    pts = MultiPoint([tuple(p) for p in xy])
    if method == "convex":
        return pts.convex_hull

    # concave hull / alpha shape (shapely>=2.0)
    try:
        return concave_hull(pts, ratio=float(alpha))
    except Exception:
        # fallback: convex hull
        return pts.convex_hull


def make_boundary_main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="make-boundary",
        description="Create a boundary polygon from forecast points.",
    )
    utils.add_city_flags(ap, default_both=True)

    ap.add_argument("--ns-src", type=str, default=None)
    ap.add_argument("--zh-src", type=str, default=None)
    ap.add_argument("--ns-eval", type=str, default=None)
    ap.add_argument("--zh-eval", type=str, default=None)
    ap.add_argument("--ns-future", type=str, default=None)
    ap.add_argument("--zh-future", type=str, default=None)

    ap.add_argument(
        "--split",
        choices=["auto", "val", "test"],
        default="auto",
    )

    ap.add_argument(
        "--method",
        choices=["convex", "concave"],
        default="convex",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Concave hull ratio (0..1).",
    )

    ap.add_argument(
        "--format",
        choices=["geojson", "shp", "both"],
        default="geojson",
    )

    ap.add_argument(
        "--out",
        type=str,
        default="boundary",
        help="Output stem (scripts/out if relative).",
    )
    args = ap.parse_args(argv)

    utils.ensure_script_dirs()

    cities0 = utils.resolve_cities(args) or [_CITY_A, _CITY_B]

    jobs: List[Dict[str, Any]] = []
    if _CITY_A in cities0:
        jobs.append(
            _resolve_city(
                city=_CITY_A,
                src=args.ns_src,
                eval_csv=args.ns_eval,
                future_csv=args.ns_future,
                split=args.split,
            )
        )
    if _CITY_B in cities0:
        jobs.append(
            _resolve_city(
                city=_CITY_B,
                src=args.zh_src,
                eval_csv=args.zh_eval,
                future_csv=args.zh_future,
                split=args.split,
            )
        )



    for j in jobs:
        city = str(j["name"])
        xy1 = _load_xy(j["eval_csv"])
        xy2 = _load_xy(j["future_csv"])
        xy = np.vstack([xy1, xy2])

        poly = _poly_from_points(
            xy,
            method=str(args.method),
            alpha=float(args.alpha),
        )

        gdf = gpd.GeoDataFrame(
            {"city": [city]},
            geometry=[poly],
            crs=None,  # keep None unless you know EPSG
        )

        stem = utils.resolve_out_out(
            f"{args.out}_{city}".lower().replace(" ", "_")
        )
        stem = stem.with_suffix("")

        if args.format in ("geojson", "both"):
            p = stem.with_suffix(".geojson")
            gdf.to_file(p, driver="GeoJSON")
            print(f"[OK] wrote {p}")

        if args.format in ("shp", "both"):
            p = stem.with_suffix(".shp")
            gdf.to_file(p)
            print(f"[OK] wrote {p}")


def main(argv: Optional[List[str]] = None) -> None:
    make_boundary_main(argv)


if __name__ == "__main__":
    main()