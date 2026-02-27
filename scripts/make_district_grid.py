# scripts/make_district_grid.py
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


_CITY_A = cfg.CITY_CANON.get("ns", "Nansha")
_CITY_B = cfg.CITY_CANON.get("zh", "Zhongshan")


def _slug(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


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
        out["eval_csv"] = utils.as_path(eval_csv)
        out["future_csv"] = utils.as_path(future_csv)
        out["note"] = "manual"
        return out

    if not src:
        raise ValueError(
            f"{city}: provide --*-src "
            "or both --*-eval and --*-future."
        )

    art = utils.detect_artifacts(src)
    ev, fu = _pick_paths(art, split)

    if ev is None:
        raise FileNotFoundError(
            f"{city}: eval CSV not found under {src}"
        )
    if fu is None:
        raise FileNotFoundError(
            f"{city}: future CSV not found under {src}"
        )

    out["eval_csv"] = ev
    out["future_csv"] = fu
    out["note"] = f"{ev.name}+{fu.name}"
    return out


def _load_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    utils.ensure_columns(df, aliases=cfg._BASE_ALIASES)

    need = ["sample_idx", "coord_x", "coord_y"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{path}: missing {miss}")

    df["sample_idx"] = pd.to_numeric(
        df["sample_idx"], errors="coerce"
    )
    df["coord_x"] = pd.to_numeric(
        df["coord_x"], errors="coerce"
    )
    df["coord_y"] = pd.to_numeric(
        df["coord_y"], errors="coerce"
    )

    df = df.dropna(subset=need).copy()
    df["sample_idx"] = df["sample_idx"].astype(int)

    # keep one coordinate per sample
    df = df.drop_duplicates("sample_idx", keep="first")
    return df[need].copy()


def _extent_xy(
    df: pd.DataFrame,
    *,
    pad_frac: float,
) -> Tuple[float, float, float, float]:
    x = df["coord_x"].to_numpy(float)
    y = df["coord_y"].to_numpy(float)

    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))

    dx = xmax - xmin
    dy = ymax - ymin
    if not np.isfinite(dx) or dx <= 0:
        dx = 1.0
    if not np.isfinite(dy) or dy <= 0:
        dy = 1.0

    pad = float(pad_frac)
    xmin = xmin - pad * dx
    xmax = xmax + pad * dx
    ymin = ymin - pad * dy
    ymax = ymax + pad * dy

    return xmin, xmax, ymin, ymax


def _grid_edges(
    extent: Tuple[float, float, float, float],
    *,
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = extent
    x_edges = np.linspace(xmin, xmax, int(nx) + 1)
    y_edges = np.linspace(ymin, ymax, int(ny) + 1)
    return x_edges, y_edges


def _zone_id(
    row_north: int,
    col: int,
    *,
    ny: int,
    nx: int,
) -> str:
    # row_north: 0 at north (top), increases southward
    # col: 0 at west (left), increases eastward
    # Ex: Z0305 = row 3, col 5 (1-based)
    rr = int(row_north) + 1
    cc = int(col) + 1
    w = max(2, len(str(ny)))
    z = max(2, len(str(nx)))
    return f"Z{rr:0{w}d}{cc:0{z}d}"


def _try_load_boundary(path: Optional[str]):
    if not path:
        return None
    try:
        import geopandas as gpd
    except Exception:
        print("[WARN] geopandas missing; skip boundary.")
        return None
    try:
        gdf = gpd.read_file(utils.as_path(path))
        if gdf.empty:
            return None
        # dissolve to one geometry (union)
        gdf2 = gdf[["geometry"]].copy()
        geom = gdf2.unary_union
        return geom
    except Exception as e:
        print(f"[WARN] boundary load failed: {e}")
        return None


def _build_grid_gdf(
    *,
    city: str,
    extent: Tuple[float, float, float, float],
    nx: int,
    ny: int,
    boundary_geom,
    clip_boundary: bool,
    min_area_frac: float,
):
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except Exception as e:
        raise SystemExit(
            "Need geopandas + shapely for district grid. "
            f"Error: {e}"
        )

    xmin, xmax, ymin, ymax = extent
    x_edges, y_edges = _grid_edges(extent, nx=int(nx), ny=int(ny))

    rows: List[Dict[str, Any]] = []
    for iy in range(int(ny)):
        for ix in range(int(nx)):
            x0 = float(x_edges[ix])
            x1 = float(x_edges[ix + 1])
            y0 = float(y_edges[iy])
            y1 = float(y_edges[iy + 1])

            # row_north: 0 at top
            row_north = int(ny) - 1 - int(iy)

            zid = _zone_id(
                row_north,
                int(ix),
                ny=int(ny),
                nx=int(nx),
            )

            geom = box(x0, y0, x1, y1)

            rows.append(
                {
                    "city": str(city),
                    "zone_id": zid,
                    "zone_label": f"Zone {zid}",
                    "row": int(row_north),
                    "col": int(ix),
                    "xmin": x0,
                    "xmax": x1,
                    "ymin": y0,
                    "ymax": y1,
                    "geometry": geom,
                }
            )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=None)

    if clip_boundary and boundary_geom is not None:
        # clip each cell to boundary; drop empty / tiny intersections
        cell_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        cell_area = float(cell_area) if cell_area > 0 else 1.0

        inter = gdf.geometry.intersection(boundary_geom)
        keep = ~inter.is_empty

        gdf = gdf.loc[keep].copy()
        gdf["geometry"] = inter.loc[keep].values

        # drop tiny slivers if requested
        mf = float(min_area_frac)
        if mf > 0:
            a = gdf.geometry.area.to_numpy(float)
            good = a >= (mf * cell_area)
            gdf = gdf.loc[good].copy()

    return gdf


def _assign_points_to_grid(
    pts: pd.DataFrame,
    *,
    city: str,
    extent: Tuple[float, float, float, float],
    nx: int,
    ny: int,
    zone_ids: Optional[set],
) -> pd.DataFrame:
    xmin, xmax, ymin, ymax = extent
    x_edges, y_edges = _grid_edges(extent, nx=int(nx), ny=int(ny))

    x = pts["coord_x"].to_numpy(float)
    y = pts["coord_y"].to_numpy(float)

    ix = np.searchsorted(x_edges, x, side="right") - 1
    iy = np.searchsorted(y_edges, y, side="right") - 1

    ix = np.clip(ix, 0, int(nx) - 1)
    iy = np.clip(iy, 0, int(ny) - 1)

    row_north = (int(ny) - 1) - iy
    zid = [
        _zone_id(int(r), int(c), ny=int(ny), nx=int(nx))
        for r, c in zip(row_north, ix)
    ]

    out = pts.copy()
    out["city"] = str(city)
    out["zone_row"] = row_north.astype(int)
    out["zone_col"] = ix.astype(int)
    out["zone_id"] = zid

    if zone_ids is not None:
        out["zone_present"] = out["zone_id"].isin(zone_ids)

    return out[["city", "sample_idx", "coord_x", "coord_y",
                "zone_id", "zone_row", "zone_col"] +
               (["zone_present"] if zone_ids is not None else [])]


def make_district_grid_main(
    argv: Optional[List[str]] = None,
) -> None:
    ap = argparse.ArgumentParser(
        prog="make-district-grid",
        description=(
            "Create a grid-based district layer "
            "(Zone IDs) from spatial points."
        ),
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
        "--nx",
        type=int,
        default=12,
        help="Grid columns (east-west).",
    )
    ap.add_argument(
        "--ny",
        type=int,
        default=12,
        help="Grid rows (north-south).",
    )
    ap.add_argument(
        "--pad",
        type=float,
        default=0.02,
        help="Extent padding fraction.",
    )

    ap.add_argument(
        "--boundary",
        type=str,
        default=None,
        help="Optional boundary GeoJSON/SHP.",
    )
    ap.add_argument(
        "--clip-boundary",
        action="store_true",
        help="Clip grid cells to boundary polygon.",
    )
    ap.add_argument(
        "--min-area-frac",
        type=float,
        default=0.01,
        help="Drop clipped cells below this area fraction.",
    )

    ap.add_argument(
        "--format",
        choices=["geojson", "shp", "both"],
        default="geojson",
    )

    ap.add_argument(
        "--assign-samples",
        action="store_true",
        help="Also write sample_idx -> zone_id CSV.",
    )

    ap.add_argument(
        "--out",
        type=str,
        default="district_grid",
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

    boundary_geom = _try_load_boundary(args.boundary)

    for j in jobs:
        city = str(j["name"])

        d1 = _load_points(Path(j["eval_csv"]))
        d2 = _load_points(Path(j["future_csv"]))
        pts = pd.concat([d1, d2], ignore_index=True)
        pts = pts.drop_duplicates("sample_idx", keep="first")

        ext = _extent_xy(pts, pad_frac=float(args.pad))

        gdf = _build_grid_gdf(
            city=city,
            extent=ext,
            nx=int(args.nx),
            ny=int(args.ny),
            boundary_geom=boundary_geom,
            clip_boundary=bool(args.clip_boundary),
            min_area_frac=float(args.min_area_frac),
        )

        stem = utils.resolve_out_out(
            f"{args.out}_{_slug(city)}"
        ).with_suffix("")

        if args.format in ("geojson", "both"):
            p = stem.with_suffix(".geojson")
            gdf.to_file(p, driver="GeoJSON")
            print(f"[OK] wrote {p}")

        if args.format in ("shp", "both"):
            p = stem.with_suffix(".shp")
            gdf.to_file(p)
            print(f"[OK] wrote {p}")

        if args.assign_samples:
            zone_ids = set(gdf["zone_id"].astype(str).to_list())
            a = _assign_points_to_grid(
                pts,
                city=city,
                extent=ext,
                nx=int(args.nx),
                ny=int(args.ny),
                zone_ids=zone_ids,
            )
            p2 = utils.resolve_out_out(
                f"district_assignments_{_slug(city)}.csv"
            )
            a.to_csv(p2, index=False)
            print(f"[OK] wrote {p2}")


def main(argv: Optional[List[str]] = None) -> None:
    make_district_grid_main(argv)


if __name__ == "__main__":
    main()