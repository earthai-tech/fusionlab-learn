# scripts/make_exposure.py
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
    from sklearn.neighbors import NearestNeighbors
except Exception as e:
    raise SystemExit(
        "density exposure requires scikit-learn. "
        f"Error: {e}"
    )

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


def _load_points(path: str) -> pd.DataFrame:
    df = pd.read_csv(utils.as_path(path))
    utils.ensure_columns(df, aliases=cfg._BASE_ALIASES)
    need = ["sample_idx", "coord_x", "coord_y"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"{path}: missing {c}")
    df["sample_idx"] = pd.to_numeric(df["sample_idx"], errors="coerce")
    df["coord_x"] = pd.to_numeric(df["coord_x"], errors="coerce")
    df["coord_y"] = pd.to_numeric(df["coord_y"], errors="coerce")
    df = df.dropna(subset=need).copy()
    df["sample_idx"] = df["sample_idx"].astype(int)
    return df[need].copy()


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


def _density_exposure(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    # simple kNN density proxy using squared distance
 
    pts = np.column_stack([x, y])
    nn = NearestNeighbors(n_neighbors=int(k) + 1)
    nn.fit(pts)
    d, _ = nn.kneighbors(pts)
    # skip self (0)
    d = d[:, 1:]
    # density proxy: inverse mean distance
    m = np.mean(d, axis=1)
    m = np.where(m <= 0, np.nan, m)
    z = 1.0 / m
    # normalize to mean 1
    z = z / np.nanmean(z)
    z = np.where(np.isfinite(z), z, 1.0)
    return z


def make_exposure_main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="make-exposure",
        description="Build exposure.csv from spatial points (proxy).",
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
        "--mode",
        choices=["uniform", "density"],
        default="density",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=30,
        help="kNN size for density exposure.",
    )

    ap.add_argument(
        "--out",
        type=str,
        default="exposure",
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

    all_rows: List[pd.DataFrame] = []

    for j in jobs:
        city = str(j["name"])
        d1 = _load_points(j["eval_csv"])
        d2 = _load_points(j["future_csv"])

        d = pd.concat([d1, d2], ignore_index=True)
        d = d.drop_duplicates("sample_idx").copy()

        if args.mode == "uniform":
            d["exposure"] = 1.0
        else:
            x = d["coord_x"].to_numpy(float)
            y = d["coord_y"].to_numpy(float)
            d["exposure"] = _density_exposure(x, y, k=int(args.k))

        d["city"] = city
        all_rows.append(d[["city", "sample_idx", "exposure"]])

    out = pd.concat(all_rows, ignore_index=True)
    p = utils.resolve_out_out(str(args.out)).with_suffix(".csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print(f"[OK] wrote {p}")


def main(argv: Optional[List[str]] = None) -> None:
    make_exposure_main(argv)


if __name__ == "__main__":
    main()