# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import OrderedDict
import os
import re
import json
import datetime as dt
from typing import Any, Dict, Iterable, Optional, Set

import numpy as np

from .generic_utils import print_config_table



# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _minmaxmeanstd(a: np.ndarray):
    a = np.asarray(a)
    if a.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    a = a.astype(np.float64, copy=False)
    return {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
    }

def _finite_ratio(a: np.ndarray):
    a = np.asarray(a)
    if a.size == 0:
        return 1.0
    ok = np.isfinite(a)
    return float(ok.sum() / ok.size)

def _fmt_range(d: Any) -> str:
    if not isinstance(d, dict) or not d:
        return "None"
    parts = []
    for k in ("t", "x", "y"):
        if k in d:
            vv = _safe_float(d.get(k))
            if vv is None:
                parts.append(f"{k}=None")
            else:
                parts.append(f"{k}={vv:.6g}")
    return "{ " + ", ".join(parts) + " }" if parts else str(d)


def _guess_xy_units(x, y) -> str:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return "unknown"
    # crude but effective: lon ~ [-180,180], lat ~ [-90,90]
    if (np.nanmax(np.abs(x)) <= 180.0) and (np.nanmax(np.abs(y)) <= 90.0):
        return "degrees-like"
    return "meters-like"


def _utm_plausibility(x, y, epsg_used) -> str:
    """
    Very lightweight plausibility check.
    For UTM, easting is typically ~[166k, 834k] and northing ~[0, 10M].
    """
    ex = _safe_float(np.nanmin(x)), _safe_float(np.nanmax(x))
    ny = _safe_float(np.nanmin(y)), _safe_float(np.nanmax(y))
    if epsg_used is None:
        return "epsg=None (skip)"

    try:
        epsg_i = int(epsg_used)
    except Exception:
        return f"epsg={epsg_used!r} (skip)"

    is_utm = (32600 <= epsg_i <= 32660) or (32700 <= epsg_i <= 32760)
    if not is_utm:
        return f"epsg={epsg_i} (non-UTM, skip)"

    # quick checks
    e_ok = (ex[0] is not None and ex[1] is not None and (0 <= ex[0] <= 1_000_000) and (0 <= ex[1] <= 1_000_000))
    n_ok = (ny[0] is not None and ny[1] is not None and (0 <= ny[0] <= 10_000_000) and (0 <= ny[1] <= 10_000_000))
    if e_ok and n_ok:
        return "UTM-like (OK)"
    return "UTM-like (CHECK): ranges look unusual"


# ---------------------------------------------------------------------
# AUDIT STAGES resolver
# ---------------------------------------------------------------------
_CANON = {
    "stage1": {"stage1", "stage-1", "stage_1", "s1", "1"},
    "stage2": {"stage2", "stage-2", "stage_2", "s2", "2"},
    "stage3": {"stage3", "stage-3", "stage_3", "s3", "3"},
}


def resolve_audit_stages(
    audit_stages: Any,
    *,
    known: Iterable[str] = ("stage1", "stage2", "stage3"),
    default: Any = None,
) -> Set[str]:
    """
    Resolve cfg["AUDIT_STAGES"] into a canonical set like {"stage1","stage2"}.

    Accepted forms
    --------------
    - None, "", False, "off", "none", 0 -> set()
    - True, "*", "all" -> all known stages
    - "stage1" / "stage-1" / "1" / "s1" -> {"stage1"}
    - "stage1,stage2" / "stage1/2" / "1 2" / "both" -> {"stage1","stage2"}
    - list/tuple/set of tokens -> union
    - dict like {"stage1": True, "stage2": False} -> enabled keys
    """
    known_set = set(known)

    if audit_stages is None:
        audit_stages = default

    # hard disable
    if audit_stages in (None, "", False, 0):
        return set()

    # dict form
    if isinstance(audit_stages, dict):
        out = set()
        for k, v in audit_stages.items():
            if not v:
                continue
            kk = str(k).strip().lower()
            for canon, aliases in _CANON.items():
                if kk in aliases or kk == canon:
                    if canon in known_set:
                        out.add(canon)
        return out

    # boolean enable all
    if audit_stages is True:
        return set(known_set)

    # list-like
    if isinstance(audit_stages, (list, tuple, set)):
        out = set()
        for item in audit_stages:
            out |= resolve_audit_stages(item, known=known, default=None)
        return out

    # string parsing
    s = str(audit_stages).strip().lower()
    if s in ("*", "all", "true", "yes", "y", "on", "any"):
        return set(known_set)
    if s in ("none", "off", "false", "no", "n", "0", "disable", "disabled"):
        return set()

    # common shorthand meaning stage1+stage2
    if s in ("both", "stage1/2", "stage1-2", "1-2", "1/2", "s1+s2", "s1,s2", "12"):
        out = set()
        for canon in ("stage1", "stage2"):
            if canon in known_set:
                out.add(canon)
        return out

    tokens = [t for t in re.split(r"[,\s;/+|]+", s) if t]
    out = set()
    for t in tokens:
        # direct match
        if t in known_set:
            out.add(t)
            continue

        # map aliases
        for canon, aliases in _CANON.items():
            if t in aliases and canon in known_set:
                out.add(canon)
                break

        # allow "stage4" style (ignored if not in known_set)
        if t.startswith("stage") and t[5:].isdigit():
            canon = "stage" + t[5:]
            if canon in known_set:
                out.add(canon)

    return out


def should_audit(audit_stages: Any, *, stage: str, default: Any = None) -> bool:
    """
    Convenience: should we audit this stage?
    """
    st = str(stage).strip().lower()
    # normalize stage tokens like "stage-1"
    for canon, aliases in _CANON.items():
        if st == canon or st in aliases:
            st = canon
            break
    return st in resolve_audit_stages(audit_stages, default=default)


# ---------------------------------------------------------------------
# Stage-1 Audit (coords + feature scaling sanity)
# ---------------------------------------------------------------------
def audit_stage1_scaling(
    *,
    df_train,
    inputs_train: Dict[str, Any],
    targets_train: Dict[str, Any],
    coord_scaler: Any = None,
    coord_ranges: Optional[Dict[str, float]] = None,
    # provenance
    coord_mode: str = "auto",
    coords_in_degrees: bool = False,
    coord_epsg_used: Any = None,
    coord_x_col_used: str = "x",
    coord_y_col_used: str = "y",
    x_col_used: str = "x",
    y_col_used: str = "y",
    time_col_used: str = "t",
    normalize_coords: bool = True,
    keep_coords_raw: bool = False,
    shift_raw_coords: bool = False,
    # physics / column names
    subs_model_col: Optional[str] = None,
    gwl_dyn_col: Optional[str] = None,
    gwl_target_col: Optional[str] = None,
    h_field_col: Optional[str] = None,
    # feature lists
    dynamic_features: Optional[Iterable[str]] = None,
    static_features: Optional[Iterable[str]] = None,
    future_features: Optional[Iterable[str]] = None,
    scaled_ml_numeric_cols: Optional[Iterable[str]] = None,
    main_scaler_path: Optional[str] = None,
    scaler_info: Optional[dict] = None,
    # UI + saving
    save_dir: Optional[str] = None,
    table_width: int = 110,
    title_prefix: str = "COORDINATE + FEATURE SCALING AUDIT (Stage-1)",
    city: str = "Unknown",
    model_name: str = "Model",
    sample_rows: int = 5,
    log_fn = None, 
) -> Optional[str]:
    """
    Stage-1 audit:
    - raw df_train coord stats (t/x/y) + heuristic units
    - model-fed coords stats from inputs_train["coords"] (flattened)
    - coord scaler min/max + coord_ranges
    - SI channel sanity for physics cols (if present)
    - target arrays sanity
    - split of features: scaled ML vs __si vs other
    Saves a machine-readable JSON if save_dir is provided.
    """
    audit: Dict[str, Any] = {}

    # ---- 0) Basic extraction
    df_cols = set(getattr(df_train, "columns", []))
    if time_col_used not in df_cols:
        raise RuntimeError(f"[Stage-1 audit] time_col_used={time_col_used!r} not found in df_train.")
    if x_col_used not in df_cols:
        raise RuntimeError(f"[Stage-1 audit] x_col_used={x_col_used!r} not found in df_train.")
    if y_col_used not in df_cols:
        raise RuntimeError(f"[Stage-1 audit] y_col_used={y_col_used!r} not found in df_train.")

    t_raw = df_train[time_col_used].to_numpy(dtype=float)
    x_raw = df_train[x_col_used].to_numpy(dtype=float)
    y_raw = df_train[y_col_used].to_numpy(dtype=float)

    units_guess = _guess_xy_units(x_raw, y_raw)
    utm_check = _utm_plausibility(x_raw, y_raw, coord_epsg_used)

    # ---- 1) Model-fed coords
    coords_arr = np.asarray(inputs_train["coords"], dtype=float)  # (N, H, 3)
    coords_flat = coords_arr.reshape(-1, 3)

    # ---- 2) Targets sanity
    y_sub = np.asarray(targets_train.get("subsidence"), dtype=float)
    y_gwl = np.asarray(targets_train.get("gwl"), dtype=float)

    # ---- 3) Physics SI columns sanity (df_train)
    phys_cols = []
    for c in (subs_model_col, gwl_dyn_col, gwl_target_col, h_field_col):
        if c:
            phys_cols.append(c)

    phys_stats = OrderedDict()
    for c in phys_cols:
        if c in df_cols:
            phys_stats[c] = _minmaxmeanstd(df_train[c].to_numpy(dtype=float))
        else:
            phys_stats[c] = {"missing": True}

    # ---- 4) Feature split summary
    scaled_set = set(scaled_ml_numeric_cols or [])
    dyn = list(dynamic_features or [])
    sta = list(static_features or [])
    fut = list(future_features or [])

    def _split_feats(feats: list) -> Dict[str, Any]:
        scaled = [c for c in feats if c in scaled_set]
        si = [c for c in feats if str(c).endswith("__si")]
        other = [c for c in feats if (c not in scaled) and (c not in si)]
        return {
            "count": int(len(feats)),
            "scaled_by_main_scaler": scaled,
            "si_unscaled": si,
            "other_unscaled_mixed": other,
        }

    feat_split = OrderedDict()
    feat_split["dynamic_features"] = _split_feats(dyn)
    feat_split["static_features"] = _split_feats(sta)
    feat_split["future_features"] = _split_feats(fut)

    # ---- 5) Build table sections
    provenance = OrderedDict(
        COORD_MODE=str(coord_mode),
        coords_in_degrees=bool(coords_in_degrees),
        coord_epsg_used=coord_epsg_used,
        COORD_X_COL_USED=str(coord_x_col_used),
        COORD_Y_COL_USED=str(coord_y_col_used),
        X_COL_USED=str(x_col_used),
        Y_COL_USED=str(y_col_used),
        TIME_COL_USED=str(time_col_used),
        NORMALIZE_COORDS=bool(normalize_coords),
        KEEP_COORDS_RAW=bool(keep_coords_raw),
        SHIFT_RAW_COORDS=bool(shift_raw_coords),
    )

    raw_stats = OrderedDict()
    raw_stats["xy_units(heuristic)"] = units_guess
    raw_stats["utm_plausibility"] = utm_check
    raw_stats["t_raw(df_train)"] = _minmaxmeanstd(t_raw)
    raw_stats["x_raw(df_train)"] = _minmaxmeanstd(x_raw)
    raw_stats["y_raw(df_train)"] = _minmaxmeanstd(y_raw)

    # sample rows
    samp_lines = []
    n_samp = int(min(sample_rows, len(df_train)))
    for i in range(n_samp):
        try:
            samp_lines.append(
                f"- {df_train.iloc[i][time_col_used]:.6g}, "
                f"{df_train.iloc[i][x_col_used]:.6g}, "
                f"{df_train.iloc[i][y_col_used]:.6g}"
            )
        except Exception:
            break
    raw_samples = OrderedDict()
    raw_samples["sample_df_train_coords(t,x,y)"] = "\n".join(samp_lines) if samp_lines else "(unavailable)"

    model_coords = OrderedDict()
    model_coords["coords.shape"] = str(tuple(coords_arr.shape))
    model_coords["t_in_model"] = _minmaxmeanstd(coords_flat[:, 0])
    model_coords["x_in_model"] = _minmaxmeanstd(coords_flat[:, 1])
    model_coords["y_in_model"] = _minmaxmeanstd(coords_flat[:, 2])
    # show unique t values (helps interpret horizons)
    try:
        model_coords["t_unique_norm"] = np.unique(coords_flat[:, 0]).tolist()
    except Exception:
        pass

    scaler_sec = OrderedDict()
    if coord_scaler is None:
        scaler_sec["coord_scaler"] = "None (coords not normalized inside sequence builder)"
    else:
        scaler_sec["coord_scaler"] = "MinMaxScaler fitted (coords normalized inside sequence builder)"
        if hasattr(coord_scaler, "data_min_"):
            scaler_sec["data_min_(t,x,y)"] = np.asarray(coord_scaler.data_min_, float).tolist()
        if hasattr(coord_scaler, "data_max_"):
            scaler_sec["data_max_(t,x,y)"] = np.asarray(coord_scaler.data_max_, float).tolist()
        scaler_sec["coord_ranges(chain_rule)"] = coord_ranges if coord_ranges is not None else None

    targets_sec = OrderedDict()
    targets_sec["targets.subsidence"] = _minmaxmeanstd(y_sub)
    targets_sec["targets.gwl(head)"] = _minmaxmeanstd(y_gwl)

    scaled_ml_sec = OrderedDict()
    scaled_ml_sec["scaled_ml_numeric_cols"] = list(scaled_set)

    feat_sec = OrderedDict()
    for k, v in feat_split.items():
        feat_sec[f"{k}.count"] = v["count"]
        feat_sec[f"{k}.scaled_by_main_scaler"] = v["scaled_by_main_scaler"]
        feat_sec[f"{k}.si_unscaled"] = v["si_unscaled"]
        # keep table readable
        other = v["other_unscaled_mixed"]
        feat_sec[f"{k}.other_unscaled_mixed"] = other if len(other) <= 25 else (other[:25] + ["..."])

    phys_sec = OrderedDict()
    for k, v in phys_stats.items():
        phys_sec[k] = v

    sections = [
        ("Stage-1 coord provenance", provenance),
        ("Raw coord stats in df_train (pre-normalization)", {k: str(v) for k, v in raw_stats.items()}),
        ("Sample df_train coords (t,x,y)", raw_samples),
        ("Model-fed coords stats (inputs_train['coords'])", {k: str(v) for k, v in model_coords.items()}),
        ("Coord scaler", {k: str(v) for k, v in scaler_sec.items()}),
        ("Physics SI channels sanity (df_train)", {k: str(v) for k, v in phys_sec.items()}),
        ("Targets arrays sanity (targets_train)", {k: str(v) for k, v in targets_sec.items()}),
        ("Stage-1 main scaler summary (ML drivers only)", {k: str(v) for k, v in scaled_ml_sec.items()}),
        ("Feature role split", {k: str(v) for k, v in feat_sec.items()}),
    ]

    print_config_table(
        sections,
        table_width=table_width,
        title=f"{title_prefix} — {city.upper()} {model_name} ({dt.datetime.now():%Y-%m-%d %H:%M:%S})",
        log_fn = log_fn, 
    )

    # ---- 6) Save JSON
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        audit["provenance"] = dict(provenance)
        audit["raw_stats"] = dict(raw_stats)
        audit["raw_samples"] = dict(raw_samples)
        audit["model_coords"] = dict(model_coords)
        audit["coord_scaler"] = dict(scaler_sec)
        audit["physics_df_stats"] = dict(phys_stats)
        audit["targets_stats"] = dict(targets_sec)
        audit["scaled_ml_numeric_cols"] = list(scaled_set)
        audit["feature_split"] = dict(feat_split)
        audit["scalers"] = {
            "main_scaler_path": main_scaler_path,
            "scaled_ml_numeric_cols": list(scaled_ml_numeric_cols or []),
            "scaler_info": scaler_info or {},
        }

        path = os.path.join(save_dir, "stage1_scaling_audit.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        print(f"\n[Audit] Saved Stage-1 scaling audit -> {path}")
        return path

    return None


# =============================================================================
# Stage-2 Handshake Audit (coords + NPZ tensors + scaling_kwargs)
# =============================================================================

def _summarize_scaling_kwargs(sk: dict) -> dict:
    # Keep table-friendly (no huge lists / nested blobs)
    out = OrderedDict()

    # --- coords & chain-rule ---
    out["coords_normalized"] = bool(sk.get("coords_normalized", False))
    out["coord_order"] = str(sk.get("coord_order", ["t", "x", "y"]))
    out["coord_ranges"] = _fmt_range(sk.get("coord_ranges", None))
    out["coords_in_degrees"] = bool(sk.get("coords_in_degrees", False))
    out["deg_to_m_lon"] = sk.get("deg_to_m_lon", None)
    out["deg_to_m_lat"] = sk.get("deg_to_m_lat", None)
    out["lat_ref_deg"] = sk.get("lat_ref_deg", None)

    out["coord_epsg_used"] = sk.get("coord_epsg_used", None)
    out["coord_target_epsg"] = sk.get("coord_target_epsg", None)

    # --- SI affine maps ---
    out["subs_scale_si"] = sk.get("subs_scale_si", None)
    out["subs_bias_si"] = sk.get("subs_bias_si", None)
    out["head_scale_si"] = sk.get("head_scale_si", None)
    out["head_bias_si"] = sk.get("head_bias_si", None)
    out["H_scale_si"] = sk.get("H_scale_si", None)
    out["H_bias_si"] = sk.get("H_bias_si", None)

    # --- channel indices (critical for physics) ---
    out["gwl_dyn_name"] = sk.get("gwl_dyn_name", None)
    out["gwl_dyn_index"] = sk.get("gwl_dyn_index", None)
    out["z_surf_static_index"] = sk.get("z_surf_static_index", None)
    out["subs_dyn_index"] = sk.get("subs_dyn_index", None)

    # --- GWL semantics ---
    out["gwl_kind"] = sk.get("gwl_kind", None)
    out["gwl_sign"] = sk.get("gwl_sign", None)
    out["use_head_proxy"] = sk.get("use_head_proxy", None)
    out["z_surf_col"] = sk.get("z_surf_col", None)

    # --- residual scaling knobs / safety ---
    out["time_units"] = sk.get("time_units", None)
    out["cons_residual_units"] = sk.get("cons_residual_units", None)
    out["gw_residual_units"] = sk.get("gw_residual_units", None)
    out["cons_scale_floor"] = sk.get("cons_scale_floor", None)
    out["gw_scale_floor"] = sk.get("gw_scale_floor", None)
    out["dt_min_units"] = sk.get("dt_min_units", None)

    out["clip_global_norm"] = sk.get("clip_global_norm", None)
    out["scaling_error_policy"] = sk.get("scaling_error_policy", None)
    out["debug_physics_grads"] = sk.get("debug_physics_grads", None)

    # --- Q forcing (if enabled) ---
    out["Q_kind"] = sk.get("Q_kind", None)
    out["Q_in_si"] = sk.get("Q_in_si", None)
    out["Q_in_per_second"] = sk.get("Q_in_per_second", None)
    out["Q_wrt_normalized_time"] = sk.get("Q_wrt_normalized_time", None)
    out["Q_length_in_si"] = sk.get("Q_length_in_si", None)
    out["drainage_mode"] = sk.get("drainage_mode", None)

    return out

def audit_stage2_handshake(
    *,
    X_train: dict,
    X_val: dict,
    y_train: dict,
    y_val: dict,
    time_steps: int,
    forecast_horizon: int,
    mode: str,
    dyn_names: list,
    fut_names: list,
    sta_names: list,
    coord_scaler=None,
    sk_final: dict,
    save_dir: str,
    table_width: int = 100,
    title_prefix: str = "STAGE-2 HANDSHAKE AUDIT",
    city='Unkown', 
    model_name ="Model", 
    log_fn =None, 
):
    audit = {}

    # ----------------------------------------------------------
    # 1) Shape expectations
    # ----------------------------------------------------------
    Ntr = int(X_train["dynamic_features"].shape[0])
    Nva = int(X_val["dynamic_features"].shape[0])

    exp = OrderedDict()
    exp["TIME_STEPS"] = int(time_steps)
    exp["FORECAST_HORIZON"] = int(forecast_horizon)
    exp["MODE"] = str(mode)

    # coords always horizon length in Stage-1/2 for GeoPrior PINN
    exp["coords.shape"] = f"(N,{forecast_horizon},3)"

    # dynamic is always (N,T,D)
    exp["dynamic_features.shape"] = f"(N,{time_steps},{len(dyn_names)})"

    # future depends on mode
    fut_T = time_steps + forecast_horizon if mode == "tft_like" else forecast_horizon
    exp["future_features.shape"] = f"(N,{fut_T},{len(fut_names)})"
    exp["static_features.shape"] = f"(N,{len(sta_names)})"
    exp["H_field.shape"] = f"(N,{forecast_horizon},1)"
    exp["targets.shape"] = f"(N,{forecast_horizon},1)"

    got = OrderedDict()
    got["N_train"] = Ntr
    got["N_val"] = Nva
    got["coords"] = tuple(X_train.get("coords").shape)
    got["dynamic_features"] = tuple(X_train.get("dynamic_features").shape)
    got["future_features"] = tuple(X_train.get("future_features").shape)
    got["static_features"] = tuple(X_train.get("static_features").shape)
    got["H_field"] = tuple(X_train.get("H_field").shape)
    got["y_subs_pred"] = tuple(y_train.get("subs_pred").shape)
    got["y_gwl_pred"] = tuple(y_train.get("gwl_pred").shape)

    # hard checks (fail fast)
    if got["coords"][1] != forecast_horizon or got["coords"][-1] != 3:
        raise RuntimeError(
            f"[Audit] coords shape mismatch: got {got['coords']}"
            F" expected (N,{forecast_horizon},3)")

    if got["dynamic_features"][1] != time_steps:
        raise RuntimeError(
            f"[Audit] dynamic_features time dim mismatch:"
            f" got {got['dynamic_features']} expected (N,{time_steps},D)"
        )

    if got["dynamic_features"][-1] != len(dyn_names):
        raise RuntimeError(
            f"[Audit] dynamic_features dim mismatch: "
            f"got D={got['dynamic_features'][-1]} vs len(DYN_NAMES)={len(dyn_names)}"
        )

    if got["future_features"][1] != fut_T:
        raise RuntimeError(
            f"[Audit] future_features time dim mismatch: "
            f"got {got['future_features']} expected (N,{fut_T},F)"
        )

    if got["future_features"][-1] != len(fut_names):
        raise RuntimeError(
            f"[Audit] future_features dim mismatch: got"
            f" F={got['future_features'][-1]} vs len(FUT_NAMES)={len(fut_names)}"
        )

    if got["H_field"][1] != forecast_horizon:
        raise RuntimeError(
            f"[Audit] H_field horizon mismatch: got"
            f" {got['H_field']} expected (N,{forecast_horizon},1)"
        )

    # ----------------------------------------------------------
    # 2) Finite checks (cheap but catches NaNs early)
    # ----------------------------------------------------------
    finite = OrderedDict()
    for k in ("coords", "dynamic_features", "future_features", "static_features", "H_field"):
        finite[f"X_train.{k}.finite_ratio"] = _finite_ratio(X_train[k])
        finite[f"X_val.{k}.finite_ratio"] = _finite_ratio(X_val[k])
    for k in ("subs_pred", "gwl_pred"):
        finite[f"y_train.{k}.finite_ratio"] = _finite_ratio(y_train[k])
        finite[f"y_val.{k}.finite_ratio"] = _finite_ratio(y_val[k])

    # ----------------------------------------------------------
    # 3) Coordinate audit (normalized + inverse if scaler exists)
    # ----------------------------------------------------------
    coords = np.asarray(X_train["coords"], dtype=np.float64)
    coords2d = coords.reshape(-1, coords.shape[-1])

    coord_stats_norm = OrderedDict()
    for j, nm in enumerate(["t", "x", "y"]):
        coord_stats_norm[f"{nm}.norm"] = _minmaxmeanstd(coords2d[:, j])

    coords_normalized = bool(sk_final.get("coords_normalized", False))
    eps = 1e-6
    coord_checks = OrderedDict()
    if coords_normalized:
        coord_checks["expected_in_[0,1]"] = True
        coord_checks["t_outside_01?"] = bool((coords2d[:, 0] < -eps).any() or (coords2d[:, 0] > 1 + eps).any())
        coord_checks["x_outside_01?"] = bool((coords2d[:, 1] < -eps).any() or (coords2d[:, 1] > 1 + eps).any())
        coord_checks["y_outside_01?"] = bool((coords2d[:, 2] < -eps).any() or (coords2d[:, 2] > 1 + eps).any())
    else:
        coord_checks["expected_in_[0,1]"] = False


    coord_stats_raw = None
    range_check = OrderedDict()
    
    if coords_normalized and (coord_scaler is not None) and hasattr(coord_scaler, "inverse_transform"):
        n = coords2d.shape[0]
        m = min(5000, n)
        # deterministic random sample -> avoids horizon-aliasing
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=m, replace=False)
    
        samp = coords2d[idx].astype(np.float64, copy=False)
        raw = coord_scaler.inverse_transform(samp)
    
        coord_stats_raw = OrderedDict()
        for j, nm in enumerate(["t", "x", "y"]):
            coord_stats_raw[f"{nm}.raw"] = _minmaxmeanstd(raw[:, j])
    
        # Compare recorded coord_ranges vs scaler span (unchanged)
        if hasattr(coord_scaler, "data_min_") and hasattr(coord_scaler, "data_max_"):
            span = (coord_scaler.data_max_ - coord_scaler.data_min_).astype(float)
            span_dict = {"t": float(span[0]), "x": float(span[1]), "y": float(span[2])}
            range_check["scaler_span"] = span_dict
    
            cr = sk_final.get("coord_ranges") or {}
            for k in ("t", "x", "y"):
                if k in cr and cr[k] not in (None, 0):
                    rel = abs(float(cr[k]) - span_dict[k]) / max(abs(span_dict[k]), 1e-12)
                    range_check[f"coord_ranges_rel_err_{k}"] = float(rel)

    # ----------------------------------------------------------
    # 4) scaling_kwargs audit summary table (table-friendly)
    # ----------------------------------------------------------
    sk_summary = _summarize_scaling_kwargs(sk_final)

    # Extra consistency checks on the final scaling_kwargs
    sk_checks = OrderedDict()
    sk_checks["has_coord_ranges"] = bool(sk_final.get("coord_ranges"))
    sk_checks["coords_in_degrees"] = bool(sk_final.get("coords_in_degrees", False))
    if sk_checks["coords_in_degrees"]:
        sk_checks["has_deg_to_m_lon"] = sk_final.get("deg_to_m_lon") is not None
        sk_checks["has_deg_to_m_lat"] = sk_final.get("deg_to_m_lat") is not None
    else:
        sk_checks["deg_to_m_lon_needed"] = False
        sk_checks["deg_to_m_lat_needed"] = False

    # indices sanity
    gwl_idx = sk_final.get("gwl_dyn_index", None)
    sk_checks["gwl_dyn_index_in_bounds"] = (
        gwl_idx is not None and 0 <= int(gwl_idx) < len(dyn_names)
    )

    # ----------------------------------------------------------
    # Print config tables
    # ----------------------------------------------------------
    sections = [
        ("Expected shapes", exp),
        ("Loaded NPZ shapes", got),
        ("Finite ratios", finite),
        ("Coords (normalized) stats", {k: str(v) for k, v in coord_stats_norm.items()}),
        ("Coords checks", coord_checks),
        ("Scaling kwargs (summary)", sk_summary),
        ("Scaling kwargs checks", sk_checks),
    ]
    if coord_stats_raw is not None:
        sections.insert(5, ("Coords (inverse to raw via coord_scaler)", {k: str(v) for k, v in coord_stats_raw.items()}))
    if range_check:
        sections.insert(6, ("Coord range cross-check", {k: str(v) for k, v in range_check.items()}))

    print_config_table(
        sections,
        table_width=table_width,
        title=f"{title_prefix} — {city.upper()} {model_name} ({dt.datetime.now():%Y-%m-%d %H:%M:%S})",
        log_fn = log_fn, 
    )

    # ----------------------------------------------------------
    # Save machine-readable audit JSON
    # ----------------------------------------------------------
    audit["expected"] = dict(exp)
    audit["got"] = {k: list(v) if isinstance(v, tuple) else v for k, v in got.items()}
    audit["finite"] = dict(finite)
    audit["coord_stats_norm"] = {k: v for k, v in coord_stats_norm.items()}
    audit["coord_checks"] = dict(coord_checks)
    audit["coord_stats_raw"] = {k: v for k, v in (coord_stats_raw or {}).items()}
    audit["coord_range_check"] = dict(range_check)
    audit["sk_summary"] = dict(sk_summary)
    audit["sk_checks"] = dict(sk_checks)

    audit_path = os.path.join(save_dir, "stage2_handshake_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"\n[Audit] Saved Stage-2 handshake audit -> {audit_path}")
    return audit_path

def audit_stage1_stage2_coord_consistency(
    *,
    X_train: dict,
    coord_scaler,
    sk_final: dict,
    time_steps: int,
    forecast_horizon: int,
    time_units: str = "year",
    save_dir: str | None = None,
    table_width: int = 110,
    title_prefix: str = "STAGE-1 ↔ STAGE-2 COORD CONSISTENCY",
    city: str = "Unknown",
    model_name: str = "Model",
    log_fn=None, 
):
    """
    Cross-check coordinate semantics between Stage-1 scaler and Stage-2 NPZ coords.

    Key facts for GeoPrior Stage-2:
      - coords are (N, H, 3) and correspond to *target horizon times*
        not the full dynamic history. So t has exactly H unique values.
      - x/y typically cover full normalized [0,1] range if you have
        spatial coverage (often min=0 and max=1).

    This audit:
      - computes normalized min/max for t/x/y in X_train["coords"]
      - derives implied raw min/max using MinMaxScaler data_min_/data_max_
      - checks raw ranges are within Stage-1 scaler bounds
      - checks t_unique count == H and t_raw_unique spacing (≈1 year)
      - provides UTM plausibility hint if epsg is UTM-like
    """
    coords = np.asarray(X_train.get("coords"), dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise RuntimeError(
            f"[CoordConsistency] Expected coords (N,H,3), got {coords.shape}"
        )

    coords2d = coords.reshape(-1, 3)

    coords_normalized = bool(sk_final.get("coords_normalized", False))
    coord_order = sk_final.get("coord_order", ["t", "x", "y"])
    if not isinstance(coord_order, (list, tuple)) or len(coord_order) != 3:
        coord_order = ["t", "x", "y"]

    audit = {}
    sec = []

    # --------------------------------------------------------------
    # 0) Quick summary of coordinate input
    # --------------------------------------------------------------
    basic = OrderedDict()
    basic["coords.shape"] = str(tuple(coords.shape))
    basic["coords_normalized"] = coords_normalized
    basic["coord_order"] = str(list(coord_order))
    basic["forecast_horizon(H)"] = int(forecast_horizon)
    basic["time_steps(T)"] = int(time_steps)
    basic["time_units"] = str(time_units)
    sec.append(("Coords input summary", basic))

    # --------------------------------------------------------------
    # 1) Normalized stats
    # --------------------------------------------------------------
    norm_stats = OrderedDict()
    for j, nm in enumerate(coord_order):
        norm_stats[f"{nm}.norm"] = _minmaxmeanstd(coords2d[:, j])
    sec.append(("Coords (normalized) stats", {k: str(v) for k, v in norm_stats.items()}))

    # If not normalized, we can still sanity check ranges but cannot compare
    if not coords_normalized:
        note = OrderedDict()
        note["note"] = "coords_normalized=False, skipping scaler-based inverse checks."
        sec.append(("Consistency note", note))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            audit_path = os.path.join(save_dir, "stage1_stage2_coord_consistency.json")
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump({"basic": dict(basic), "norm_stats": dict(norm_stats)}, f, indent=2)
            print(f"\n[Audit] Saved coord consistency audit -> {audit_path}")
            return audit_path
        return None

    # --------------------------------------------------------------
    # 2) Require a MinMaxScaler-like scaler
    # --------------------------------------------------------------
    if coord_scaler is None or not hasattr(
            coord_scaler, "data_min_") or not hasattr(coord_scaler, "data_max_"):
        raise RuntimeError(
            "[CoordConsistency] coords_normalized=True but coord_scaler "
            "missing data_min_/data_max_."
        )

    data_min = np.asarray(coord_scaler.data_min_, dtype=np.float64)
    data_max = np.asarray(coord_scaler.data_max_, dtype=np.float64)
    span = (data_max - data_min).astype(np.float64)

    scaler_sec = OrderedDict()
    scaler_sec["data_min_(t,x,y)"] = data_min.tolist()
    scaler_sec["data_max_(t,x,y)"] = data_max.tolist()
    scaler_sec["span_(t,x,y)"] = span.tolist()
    scaler_sec["coord_ranges(manifest)"] = _fmt_range(sk_final.get("coord_ranges"))
    scaler_sec["coord_epsg_used"] = sk_final.get("coord_epsg_used", None)
    sec.append(("Stage-1 coord_scaler reference", {k: str(v) for k, v in scaler_sec.items()}))

    # --------------------------------------------------------------
    # 3) Derive implied raw min/max from normalized min/max
    #    raw = data_min + norm * span
    # --------------------------------------------------------------
    derived = OrderedDict()
    checks = OrderedDict()

    eps = 1e-6
    for j, nm in enumerate(coord_order):
        nmin = float(np.nanmin(coords2d[:, j]))
        nmax = float(np.nanmax(coords2d[:, j]))

        rmin = float(data_min[j] + nmin * span[j])
        rmax = float(data_min[j] + nmax * span[j])

        derived[f"{nm}.norm_min"] = nmin
        derived[f"{nm}.norm_max"] = nmax
        derived[f"{nm}.raw_min(derived)"] = rmin
        derived[f"{nm}.raw_max(derived)"] = rmax

        # must be within scaler bounds (allow tiny epsilon)
        lo = float(data_min[j] - 1e-9 * max(abs(span[j]), 1.0))
        hi = float(data_max[j] + 1e-9 * max(abs(span[j]), 1.0))
        checks[f"{nm}.raw_within_scaler_bounds?"] = bool((rmin >= lo) and (rmax <= hi))

        # does this dim cover full [0,1]?
        checks[f"{nm}.covers_full_[0,1]?"] = bool((abs(nmin - 0.0) <= eps) and (abs(nmax - 1.0) <= 5e-5))

        # if it covers full [0,1], raw should ~equal scaler min/max
        if checks[f"{nm}.covers_full_[0,1]?"] and abs(span[j]) > 0:
            rel_min = abs(rmin - data_min[j]) / max(abs(span[j]), 1e-12)
            rel_max = abs(rmax - data_max[j]) / max(abs(span[j]), 1e-12)
            checks[f"{nm}.rel_err_raw_min_vs_scaler_min"] = float(rel_min)
            checks[f"{nm}.rel_err_raw_max_vs_scaler_max"] = float(rel_max)

    sec.append(("Derived raw ranges from normalized coords", {k: str(v) for k, v in derived.items()}))

    # --------------------------------------------------------------
    # 4) Time semantics: t has only H unique values (horizon-only coords)
    # --------------------------------------------------------------
    t_index = 0
    # if coord_order is weird, find 't'
    if "t" in coord_order:
        t_index = int(list(coord_order).index("t"))

    t_unique_norm = np.unique(coords[:, :, t_index]).astype(np.float64)
    # invert those EXACT unique values
    # raw = data_min + t_norm * span for time dim
    t_unique_raw = (data_min[t_index] + t_unique_norm * span[t_index]).astype(np.float64)

    time_sem = OrderedDict()
    time_sem["t.unique_norm.count"] = int(t_unique_norm.size)
    time_sem["t.unique_norm.values"] = t_unique_norm.tolist()
    time_sem["t.unique_raw.values(derived)"] = t_unique_raw.tolist()
    time_sem["expected_unique_count==H?"] = bool(t_unique_norm.size == int(forecast_horizon))

    # spacing check (year-like)
    if t_unique_raw.size >= 2:
        diffs = np.diff(np.sort(t_unique_raw))
        time_sem["t.raw.diffs"] = diffs.tolist()
        # if using years, diff ~ 1
        if str(time_units).lower().startswith("year"):
            time_sem["t.raw.diffs_approx_1yr?"] = bool(
                np.all(np.isfinite(diffs)) and np.all(np.abs(diffs - 1.0) < 1e-2))

    # explain the “3 unique t” situation explicitly
    time_sem["note"] = (
        "GeoPrior Stage-2 coords are horizon-only (N,H,3). "
        "So t has exactly H unique values (e.g., 2020,2021,2022). "
        "This is expected and correct."
    )
    sec.append(("Time-axis semantics (horizon-only coords)", {k: str(v) for k, v in time_sem.items()}))

    # --------------------------------------------------------------
    # 5) UTM plausibility hint for x/y if EPSG suggests UTM
    # --------------------------------------------------------------
    epsg_used = sk_final.get("coord_epsg_used", None)
    utm_hint = OrderedDict()
    if epsg_used is not None:
        try:
            epsg_i = int(epsg_used)
            is_utm = (32600 <= epsg_i <= 32660) or (32700 <= epsg_i <= 32760)
        except Exception:
            is_utm = False
        if is_utm:
            # use derived raw bounds for x/y
            # try locate x/y indices in coord_order
            xi = int(coord_order.index("x")) if "x" in coord_order else 1
            yi = int(coord_order.index("y")) if "y" in coord_order else 2
            x_rmin = float(data_min[xi] + float(np.nanmin(coords2d[:, xi])) * span[xi])
            x_rmax = float(data_min[xi] + float(np.nanmax(coords2d[:, xi])) * span[xi])
            y_rmin = float(data_min[yi] + float(np.nanmin(coords2d[:, yi])) * span[yi])
            y_rmax = float(data_min[yi] + float(np.nanmax(coords2d[:, yi])) * span[yi])

            utm_hint["epsg_used"] = epsg_i
            utm_hint["x_raw_minmax(derived)"] = [x_rmin, x_rmax]
            utm_hint["y_raw_minmax(derived)"] = [y_rmin, y_rmax]
            utm_hint["heuristic"] = _utm_plausibility(np.array([x_rmin, x_rmax]),
                                                      np.array([y_rmin, y_rmax]),
                                                      epsg_i)
        else:
            utm_hint["epsg_used"] = epsg_used
            utm_hint["heuristic"] = "epsg not UTM-like (skip)"
    else:
        utm_hint["heuristic"] = "epsg missing (skip)"

    sec.append(("Spatial plausibility hint", {k: str(v) for k, v in utm_hint.items()}))

    # --------------------------------------------------------------
    # 6) Print table
    # --------------------------------------------------------------
    sec.append(("Consistency checks", checks))

    print_config_table(
        sec,
        table_width=table_width,
        title=f"{title_prefix} — {city.upper()} {model_name} ({dt.datetime.now():%Y-%m-%d %H:%M:%S})",
        log_fn =log_fn, 
    )

    # --------------------------------------------------------------
    # 7) Save JSON
    # --------------------------------------------------------------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        audit = {
            "basic": dict(basic),
            "norm_stats": {k: v for k, v in norm_stats.items()},
            "scaler_ref": dict(scaler_sec),
            "derived_raw": dict(derived),
            "time_semantics": dict(time_sem),
            "spatial_hint": dict(utm_hint),
            "checks": dict(checks),
        }
        audit_path = os.path.join(save_dir, "stage1_stage2_coord_consistency.json")
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)

        print(f"\n[Audit] Saved coord consistency audit -> {audit_path}")
        return audit_path

    return None

# ---------------------------------------------------------------------
# Stage-3 Audit (tuning + saving + eval artifacts)
# ---------------------------------------------------------------------


def _json_safe(x: Any) -> Any:
    if isinstance(x, (np.generic,)):
        try:
            return x.item()
        except Exception:
            return str(x)

    if isinstance(x, np.ndarray):
        try:
            return x.tolist()
        except Exception:
            return str(x)

    return x


def _subset_dict(
    d: Optional[Dict[str, Any]],
    keys: Iterable[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = OrderedDict()
    if not d:
        return out
    for k in keys:
        if k in d:
            out[k] = d[k]
    return out


def audit_stage3_run(
    *,
    manifest_path: Optional[str],
    manifest: Dict[str, Any],
    cfg: Dict[str, Any],
    fixed_params: Dict[str, Any],
    best_hps: Optional[Dict[str, Any]],
    run_dir: str,
    best_model_path: Optional[str],
    best_weights_path: Optional[str],
    use_tf_savedmodel: bool,
    quantiles: Any,
    forecast_horizon: int,
    mode: str,
    pred_shapes: Optional[Dict[str, Any]] = None,
    eval_results: Optional[Dict[str, Any]] = None,
    phys_diag: Optional[Dict[str, Any]] = None,
    calibrator_factors: Any = None,
    forecast_csv_eval: Optional[str] = None,
    forecast_csv_future: Optional[str] = None,
    metrics_json_path: Optional[str] = None,
    physics_payload_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    table_width: int = 100,
    title_prefix: str = "STAGE-3 AUDIT",
    city: str = "Unknown",
    model_name: str = "Model",
    log_fn=None, 
) -> Optional[str]:
    """Stage-3 audit: tuned artifacts + eval sanity."""
    os.makedirs(run_dir, exist_ok=True)

    audit: Dict[str, Any] = {}

    prov = OrderedDict()
    prov["city"] = str(city)
    prov["model"] = str(model_name)
    prov["run_dir"] = str(run_dir)
    prov["manifest_path"] = str(manifest_path)
    prov["use_tf_savedmodel"] = bool(use_tf_savedmodel)
    prov["forecast_horizon"] = int(forecast_horizon)
    prov["mode"] = str(mode)
    prov["quantiles"] = (
        list(quantiles)
        if isinstance(quantiles, (list, tuple))
        else quantiles
    )

    art = OrderedDict()

    def _exists(p: Optional[str]) -> Optional[bool]:
        if not p:
            return None
        return bool(os.path.exists(p))

    art["best_model_path"] = str(best_model_path)
    art["best_model_exists"] = _exists(best_model_path)
    art["best_weights_path"] = str(best_weights_path)
    art["best_weights_exists"] = _exists(best_weights_path)

    art["forecast_csv_eval"] = str(forecast_csv_eval)
    art["forecast_csv_eval_exists"] = _exists(forecast_csv_eval)

    art["forecast_csv_future"] = str(forecast_csv_future)
    art["forecast_csv_future_exists"] = _exists(
        forecast_csv_future
    )

    art["metrics_json_path"] = str(metrics_json_path)
    art["metrics_json_exists"] = _exists(metrics_json_path)

    art["physics_payload_path"] = str(physics_payload_path)
    art["physics_payload_exists"] = _exists(
        physics_payload_path
    )

    fixed = OrderedDict()
    fixed_keys = (
        "static_input_dim",
        "dynamic_input_dim",
        "future_input_dim",
        "output_subsidence_dim",
        "output_gwl_dim",
        "pde_mode",
        "offset_mode",
        "bounds_mode",
        "residual_method",
        "time_units",
        "scale_pde_residuals",
        "use_effective_h",
    )
    fixed.update(_subset_dict(fixed_params, fixed_keys))

    sk = fixed_params.get("scaling_kwargs", {}) or {}
    sk_sum = OrderedDict()
    if isinstance(sk, dict):
        sk_sum = _summarize_scaling_kwargs(sk)

    hp = OrderedDict()
    hp["n_best_hps"] = int(len(best_hps or {}))
    hp_keys = (
        "learning_rate",
        "lambda_gw",
        "lambda_cons",
        "lambda_prior",
        "lambda_smooth",
        "lambda_bounds",
        "lambda_q",
        "lambda_offset",
        "lambda_mv",
        "mv_lr_mult",
        "kappa_lr_mult",
        "scale_mv_with_offset",
        "scale_q_with_offset",
    )
    hp.update(_subset_dict(best_hps, hp_keys))

    preds = OrderedDict()
    if pred_shapes:
        for k, v in pred_shapes.items():
            preds[f"{k}.shape"] = str(v)

    cal = OrderedDict()
    f = calibrator_factors
    if f is None:
        cal["cal_factors"] = "None"
    else:
        try:
            f_arr = np.asarray(f, dtype=float)
            cal["factors.shape"] = str(tuple(f_arr.shape))
            cal["factors.stats"] = _minmaxmeanstd(f_arr)
            cal["factors.finite_ratio"] = _finite_ratio(f_arr)
        except Exception:
            cal["cal_factors"] = str(f)

    ev = OrderedDict()
    if eval_results:
        ev["n_keys"] = int(len(eval_results))
        keep = (
            "loss",
            "total_loss",
            "data_loss",
            "physics_loss",
            "physics_loss_scaled",
            "consolidation_loss",
            "gw_flow_loss",
            "prior_loss",
            "smooth_loss",
            "bounds_loss",
            "mv_prior_loss",
            "epsilon_prior",
            "epsilon_cons",
        )
        ev.update(_subset_dict(eval_results, keep))

    ph = OrderedDict()
    if phys_diag:
        ph.update(phys_diag)

    sections = [
        ("Stage-3 provenance", prov),
        ("Artifacts", art),
        ("Fixed params (key subset)", fixed),
        ("Scaling kwargs (summary)", sk_sum),
        ("Best HPs (summary)", hp),
        ("Pred shapes", preds),
        ("Calibrator", {k: str(v) for k, v in cal.items()}),
        ("Evaluate() summary", {k: str(v) for k, v in ev.items()}),
        ("Physics diag", {k: str(v) for k, v in ph.items()}),
    ]

    print_config_table(
        sections,
        table_width=table_width,
        title=(
            f"{title_prefix} — {str(city).upper()} "
            f"{model_name} ({dt.datetime.now():%Y-%m-%d %H:%M:%S})"
        ),
        log_fn = log_fn, 
    )

    out_dir = save_dir or run_dir
    if not out_dir:
        return None

    os.makedirs(out_dir, exist_ok=True)

    audit["provenance"] = dict(prov)
    audit["artifacts"] = dict(art)
    audit["fixed_params"] = dict(fixed)
    audit["scaling_kwargs_summary"] = dict(sk_sum)
    audit["best_hps"] = {k: _json_safe(v) for k, v in (best_hps or {}).items()}
    audit["pred_shapes"] = dict(preds)
    audit["calibrator"] = dict(cal)
    audit["eval_results"] = {
        k: _json_safe(v)
        for k, v in (eval_results or {}).items()
    }
    audit["physics_diag"] = {
        k: _json_safe(v)
        for k, v in (phys_diag or {}).items()
    }

    audit_path = os.path.join(out_dir, "stage3_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"\n[Audit] Saved Stage-3 audit -> {audit_path}")
    return audit_path
