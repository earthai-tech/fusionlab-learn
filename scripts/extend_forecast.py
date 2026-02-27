# scripts/extend_forecast.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg
from . import extend_utils as ex
from . import utils


def _city_ns() -> str:
    return cfg.CITY_CANON.get("ns", "Nansha")


def _city_zh() -> str:
    return cfg.CITY_CANON.get("zh", "Zhongshan")


def _slug_city(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _resolve_cities(args) -> List[str]:
    picked: List[str] = []

    if getattr(args, "use_ns", False):
        picked.append(_city_ns())
    if getattr(args, "use_zh", False):
        picked.append(_city_zh())

    if picked:
        return picked

    raw = str(getattr(args, "cities", "") or "")
    parts = [p.strip().lower() for p in raw.split(",")]

    out: List[str] = []
    for p in parts:
        if not p:
            continue
        out.append(cfg.CITY_CANON.get(p, p.title()))
    return out


def _pick_paths(
    art: utils.Artifacts,
    split: str,
) -> Tuple[Optional[Path], Optional[Path], str]:
    if split == "val":
        return (
            art.forecast_val_csv,
            art.forecast_future_csv,
            "val",
        )
    if split == "test":
        return (
            art.forecast_test_csv,
            art.forecast_test_future_csv,
            "test",
        )

    if (
        art.forecast_test_csv is not None
        and art.forecast_test_future_csv is not None
    ):
        return (
            art.forecast_test_csv,
            art.forecast_test_future_csv,
            "test",
        )

    return (
        art.forecast_val_csv,
        art.forecast_future_csv,
        "val",
    )


def _resolve_one_city(
    *,
    city: str,
    src: Optional[str],
    eval_csv: Optional[str],
    future_csv: Optional[str],
    split: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"city": city}

    if eval_csv and future_csv:
        out["eval_csv"] = utils.as_path(eval_csv)
        out["future_csv"] = utils.as_path(future_csv)
        out["split"] = split
        out["src_note"] = "manual"
        return out

    if not src:
        raise ValueError(
            f"{city}: provide --*-src or both "
            "--*-eval and --*-future."
        )

    art = utils.detect_artifacts(src)
    ev, fu, lab = _pick_paths(art, split)

    if ev is None:
        raise FileNotFoundError(
            f"{city}: no eval CSV under {src}"
        )
    if fu is None:
        raise FileNotFoundError(
            f"{city}: no future CSV under {src}"
        )

    out["eval_csv"] = ev
    out["future_csv"] = fu
    out["split"] = lab
    out["src_note"] = f"{ev.name}+{fu.name} ({lab})"
    return out


def _out_path(
    out_arg: str,
    *,
    city: str,
    multi: bool,
) -> Path:
    p = Path(out_arg).expanduser()

    if p.suffix.lower() != ".csv":
        stem = str(p)
        if multi:
            stem = f"{stem}_{_slug_city(city)}"
        p = Path(stem + ".csv")
    else:
        if multi:
            stem = p.with_suffix("")
            p = Path(f"{stem}_{_slug_city(city)}.csv")

    return utils.resolve_out_out(str(p))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="extend-forecast",
        description=(
            "Extend future forecast CSV by "
            "1-2+ years (extrapolation)."
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
        type=str,
        choices=["auto", "val", "test"],
        default="auto",
        help="When using --*-src, pick val/test.",
    )

    ap.add_argument(
        "--subsidence-kind",
        type=str,
        default="cumulative",
        choices=["cumulative", "rate", "increment"],
        help="Input meaning (default cumulative).",
    )
    ap.add_argument(
        "--out-kind",
        type=str,
        default="same",
        choices=["same", "cumulative", "rate"],
        help="Output kind (default same).",
    )

    ap.add_argument(
        "--method",
        type=str,
        default="linear_fit",
        choices=["linear_fit", "linear_last"],
        help="Trend method for extrapolation.",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=3,
        help="Fit window (years) for trend.",
    )

    ap.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Explicit years to add (e.g. 2026 2027).",
    )
    ap.add_argument(
        "--add-years",
        type=int,
        default=1,
        help="If --years omitted, add N years.",
    )

    ap.add_argument(
        "--unc-growth",
        type=str,
        default="sqrt",
        choices=["hold", "sqrt", "linear"],
        help="Uncertainty widening across added years.",
    )
    ap.add_argument(
        "--unc-scale",
        type=float,
        default=1.0,
        help="Multiplier for uncertainty widening.",
    )

    ap.add_argument(
        "--out",
        "-o",
        type=str,
        default="future_extended",
        help="Output stem/path (scripts/out if rel).",
    )

    return ap


def extend_forecast_main(
    argv: Optional[List[str]] = None,
) -> None:
    args = build_parser().parse_args(argv)

    utils.ensure_script_dirs()

    cities = _resolve_cities(args)
    if not cities:
        cities = [_city_ns(), _city_zh()]

    want_ns = _city_ns() in cities
    want_zh = _city_zh() in cities

    if not want_ns and not want_zh:
        want_ns = True
        want_zh = True

    jobs: List[Dict[str, Any]] = []

    if want_ns:
        jobs.append(
            _resolve_one_city(
                city=_city_ns(),
                src=args.ns_src,
                eval_csv=args.ns_eval,
                future_csv=args.ns_future,
                split=args.split,
            )
        )

    if want_zh:
        jobs.append(
            _resolve_one_city(
                city=_city_zh(),
                src=args.zh_src,
                eval_csv=args.zh_eval,
                future_csv=args.zh_future,
                split=args.split,
            )
        )

    multi = len(jobs) > 1

    cc = ex.ExtendCfg(
        kind=str(args.subsidence_kind),
        out_kind=str(args.out_kind),
        method=str(args.method),
        window=int(args.window),
        unc_growth=str(args.unc_growth),
        unc_scale=float(args.unc_scale),
    )

    for j in jobs:
        city = str(j["city"])

        ev = ex.load_eval_csv(j["eval_csv"], cc=cc)
        fu = ex.load_future_csv(j["future_csv"], cc=cc)

        out_df = ex.extend_future_df(
            fu,
            eval_df=ev,
            add_years=int(args.add_years),
            years=list(args.years) if args.years else None,
            cc=cc,
        )

        out_p = _out_path(str(args.out), city=city, multi=multi)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        out_df.to_csv(out_p, index=False)

        print(
            f"[OK] {city}: wrote {out_p} "
            f"({j.get('src_note','')})"
        )


def main(argv: Optional[List[str]] = None) -> None:
    extend_forecast_main(argv)


if __name__ == "__main__":
    main()