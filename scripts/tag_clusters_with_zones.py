# scripts/tag_clusters_with_zones.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import utils


@dataclass(frozen=True)
class _Grid:
    zone_ids: List[str]
    zone_labels: List[str]
    geoms: List[Any]
    attrs: List[Dict[str, Any]]


def _as_path(x: str) -> Path:
    return utils.as_path(x)


def _load_clusters(path: str) -> pd.DataFrame:
    df = pd.read_csv(_as_path(path))

    need = ["centroid_x", "centroid_y"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"clusters: missing {miss}")

    for c in [
        "centroid_x",
        "centroid_y",
        "cluster_id",
        "year",
        "priority_rank",
        "metric_mean",
        "risk_mean",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "cluster_id" in df.columns:
        df["cluster_id"] = (
            pd.to_numeric(df["cluster_id"], errors="coerce")
            .astype("Int64")
        )
    if "year" in df.columns:
        df["year"] = (
            pd.to_numeric(df["year"], errors="coerce")
            .astype("Int64")
        )
    if "priority_rank" in df.columns:
        df["priority_rank"] = (
            pd.to_numeric(df["priority_rank"], errors="coerce")
            .astype("Int64")
        )

    return df


def _try_read_grid_geopandas(
    path: Path,
    *,
    zone_id_field: str,
    zone_label_field: str,
) -> Optional[_Grid]:
    try:
        import geopandas as gpd
    except Exception:
        return None

    gdf = gpd.read_file(path)
    if gdf.empty:
        return None

    if zone_id_field not in gdf.columns:
        raise KeyError(
            f"grid: missing {zone_id_field!r}"
        )

    if zone_label_field not in gdf.columns:
        gdf[zone_label_field] = (
            "Zone " + gdf[zone_id_field].astype(str)
        )

    zone_ids = gdf[zone_id_field].astype(str).tolist()
    zone_labs = gdf[zone_label_field].astype(str).tolist()
    geoms = list(gdf.geometry.values)

    attrs: List[Dict[str, Any]] = []
    for _, r in gdf.drop(columns="geometry").iterrows():
        attrs.append(dict(r))

    return _Grid(zone_ids, zone_labs, geoms, attrs)


def _read_grid_geojson_fallback(
    path: Path,
    *,
    zone_id_field: str,
    zone_label_field: str,
) -> _Grid:
    from shapely.geometry import shape

    obj = json.loads(path.read_text(encoding="utf-8"))
    feats = obj.get("features", [])
    if not feats:
        raise ValueError("grid: empty GeoJSON")

    zone_ids: List[str] = []
    zone_labs: List[str] = []
    geoms: List[Any] = []
    attrs: List[Dict[str, Any]] = []

    for f in feats:
        props = dict(f.get("properties", {}) or {})
        if zone_id_field not in props:
            raise KeyError(
                f"grid: missing {zone_id_field!r}"
            )

        zid = str(props.get(zone_id_field))
        zlab = props.get(zone_label_field)
        if zlab is None:
            zlab = "Zone " + zid

        geom = shape(f.get("geometry"))
        zone_ids.append(zid)
        zone_labs.append(str(zlab))
        geoms.append(geom)
        attrs.append(props)

    return _Grid(zone_ids, zone_labs, geoms, attrs)


def _load_grid(
    path: str,
    *,
    zone_id_field: str,
    zone_label_field: str,
) -> _Grid:
    p = _as_path(path)

    g = _try_read_grid_geopandas(
        p,
        zone_id_field=zone_id_field,
        zone_label_field=zone_label_field,
    )
    if g is not None:
        return g

    # fallback: GeoJSON only
    if p.suffix.lower() not in (".geojson", ".json"):
        raise SystemExit(
            "grid: geopandas not available. "
            "Use GeoJSON, or install geopandas."
        )

    return _read_grid_geojson_fallback(
        p,
        zone_id_field=zone_id_field,
        zone_label_field=zone_label_field,
    )


def _match_points_to_polys(
    xs: np.ndarray,
    ys: np.ndarray,
    grid: _Grid,
    *,
    nearest: bool,
) -> Tuple[List[Optional[str]], List[Optional[str]]]:
    from shapely.geometry import Point

    try:
        from shapely.strtree import STRtree
    except Exception:
        STRtree = None

    pts = [Point(float(x), float(y)) for x, y in zip(xs, ys)]

    # Build spatial index if available
    tree = None
    geom_to_idx: Dict[int, int] = {}
    if STRtree is not None:
        tree = STRtree(grid.geoms)
        for i, g in enumerate(grid.geoms):
            geom_to_idx[id(g)] = i

    out_id: List[Optional[str]] = []
    out_lab: List[Optional[str]] = []

    for p in pts:
        idx = None

        if tree is not None:
            cand = tree.query(p)
            for g in cand:
                if g.contains(p):
                    idx = geom_to_idx.get(id(g))
                    break
        else:
            for i, g in enumerate(grid.geoms):
                if g.contains(p):
                    idx = i
                    break

        if idx is None and nearest:
            best_i = None
            best_d = None
            for i, g in enumerate(grid.geoms):
                d = float(g.distance(p))
                if best_d is None or d < best_d:
                    best_d = d
                    best_i = i
            idx = best_i

        if idx is None:
            out_id.append(None)
            out_lab.append(None)
        else:
            out_id.append(grid.zone_ids[int(idx)])
            out_lab.append(grid.zone_labels[int(idx)])

    return out_id, out_lab


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tag-clusters-with-zones",
        description=(
            "Assign hotspot cluster centroids to "
            "grid Zone IDs (GeoJSON/SHP)."
        ),
    )

    ap.add_argument(
        "--clusters",
        required=True,
        type=str,
        help="CSV with cluster centroids.",
    )
    ap.add_argument(
        "--grid",
        required=True,
        type=str,
        help="District grid GeoJSON/SHP.",
    )

    ap.add_argument(
        "--zone-id-field",
        type=str,
        default="zone_id",
        help="Field in grid for Zone ID.",
    )
    ap.add_argument(
        "--zone-label-field",
        type=str,
        default="zone_label",
        help="Field in grid for Zone label.",
    )

    ap.add_argument(
        "--city",
        type=str,
        default=None,
        help="Optional filter for city.",
    )
    ap.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional filter for year.",
    )

    ap.add_argument(
        "--nearest",
        action="store_true",
        help="If no polygon contains centroid, "
        "assign nearest zone.",
    )

    ap.add_argument(
        "--out",
        "-o",
        type=str,
        default="cluster_zones",
        help="Output stem/path (scripts/out if rel).",
    )

    return ap


def tag_clusters_with_zones_main(
    argv: Optional[List[str]] = None,
) -> None:
    args = build_parser().parse_args(argv)

    utils.ensure_script_dirs()

    cl = _load_clusters(str(args.clusters))

    if args.city and "city" in cl.columns:
        cl = cl.loc[
            cl["city"].astype(str).eq(str(args.city))
        ].copy()

    if args.year is not None and "year" in cl.columns:
        cl = cl.loc[
            cl["year"].astype("Int64").eq(int(args.year))
        ].copy()

    if cl.empty:
        raise SystemExit("No clusters after filtering.")

    grid = _load_grid(
        str(args.grid),
        zone_id_field=str(args.zone_id_field),
        zone_label_field=str(args.zone_label_field),
    )

    xs = pd.to_numeric(
        cl["centroid_x"], errors="coerce"
    ).to_numpy(float)
    ys = pd.to_numeric(
        cl["centroid_y"], errors="coerce"
    ).to_numpy(float)

    ok = np.isfinite(xs) & np.isfinite(ys)
    cl = cl.loc[ok].copy()
    xs = xs[ok]
    ys = ys[ok]

    zid, zlab = _match_points_to_polys(
        xs,
        ys,
        grid,
        nearest=bool(args.nearest),
    )

    out = cl.copy()
    out["zone_id"] = zid
    out["zone_label"] = zlab

    # Keep a clean, table-friendly set
    keep = []
    for c in [
        "city",
        "year",
        "cluster_id",
        "priority_rank",
        "zone_id",
        "zone_label",
        "centroid_x",
        "centroid_y",
        "n_cells",
        "metric_mean",
        "metric_max",
        "risk_mean",
        "risk_max",
    ]:
        if c in out.columns:
            keep.append(c)

    out2 = out[keep].copy() if keep else out

    p = utils.resolve_out_out(str(args.out))
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")
    p.parent.mkdir(parents=True, exist_ok=True)

    out2.to_csv(p, index=False)
    print(f"[OK] wrote {p}")


def main(argv: Optional[List[str]] = None) -> None:
    tag_clusters_with_zones_main(argv)


if __name__ == "__main__":
    main()