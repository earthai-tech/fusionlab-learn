# -*- coding: utf-8 -*-
# License : BSD-3-Clause
# Author  : LKouadio

"""Debug helpers for GeoPriorSubsNet.

Keep *all* verbosity + shape/unit printing here so
`_geoprior_subnet.py` stays clean.

All functions are safe to call inside `tf.function`:
they use `tf.print` and TensorFlow assertions.
- Compare in-memory vs loaded inference model on the *same* batch.
- Report prediction diffs + weight name/shape diffs + key attribute digests.

"""

from __future__ import annotations

from typing import ( 
    Any, Dict, Optional, Sequence, 
    Callable, List, Tuple, 
    Iterable
)

import hashlib
import json

import numpy as np

from ... import KERAS_DEPS 

from .maths import (
    tf_print_nonfinite,
    _assert_grads_finite,
    resolve_cons_units,
    to_rms, _stats, 
    seconds_per_time_unit,
    dt_to_seconds,

)

Tensor = KERAS_DEPS.Tensor

tf_abs = KERAS_DEPS.abs
tf_cast = KERAS_DEPS.cast
tf_cond = KERAS_DEPS.cond
tf_constant = KERAS_DEPS.constant
tf_debugging = KERAS_DEPS.debugging
tf_equal = KERAS_DEPS.equal
tf_float32 = KERAS_DEPS.float32
tf_greater = KERAS_DEPS.greater
tf_int32 = KERAS_DEPS.int32
tf_print = KERAS_DEPS.print
tf_rank = KERAS_DEPS.rank
tf_reduce_max = KERAS_DEPS.reduce_max
tf_reduce_mean = KERAS_DEPS.reduce_mean
tf_reduce_min = KERAS_DEPS.reduce_min
tf_shape = KERAS_DEPS.shape
tf_square = KERAS_DEPS.square
tf_where = KERAS_DEPS.where
tf_zeros_like = KERAS_DEPS.zeros_like
tf_math =KERAS_DEPS.math 
tf_sqrt = KERAS_DEPS.sqrt 
tf_maximum = KERAS_DEPS.maximum
tf_string = KERAS_DEPS.string



def _to_numpy(x: Any) -> np.ndarray:
    # Works for tf.Tensor, np.ndarray, python numbers
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _get_one_batch(dataset: Iterable) -> Tuple[Any, Any]:
    # Works for tf.data.Dataset or any iterator yielding (xb, yb)
    return next(iter(dataset))


def _extract_output(model: Any, out: Any, key: str):
    """
    Support dict outputs (preferred) and
    list/tuple outputs (Keras predict()).
    """
    if isinstance(out, dict):
        if key not in out:
            raise KeyError(
                f"Output key {key!r} not found."
                f" Got keys={list(out.keys())}")
        return out[key]

    if isinstance(out, (list, tuple)):
        names = list(getattr(model, "output_names", []) or [])
        if names and key in names:
            return out[names.index(key)]
        # fallback common ordering
        if key == "subs_pred" and len(out) >= 1:
            return out[0]
        if key == "gwl_pred" and len(out) >= 2:
            return out[1]
        raise KeyError(
            f"Cannot map output key {key!r} from"
            f" list outputs; output_names={names}")

    raise TypeError(f"Unexpected model output type: {type(out)}")


def _weight_key(
    w: Any,
    occ: int,
) -> str:
    """
    Build a stable, mostly-unique key for a weight.

    Priority:
      1) Keras weight `path` (unique, best)
      2) name + occurrence index (fallback)
    """
    p = getattr(w, "path", None)
    if p:
        return f"path:{p}"
    n = getattr(w, "name", "<noname>")
    return f"name:{n}#{occ}"


def _index_weights(
    model: Any,
) -> Tuple[
    Dict[str, Any],
    Dict[str, str],
    List[str],
]:
    """
    Index model weights.

    Returns:
      wmap: key -> weight
      nmap: key -> original name
      keys: keys in traversal order
    """
    weights = list(getattr(model, "weights", []) or [])
    counts: Dict[str, int] = {}
    wmap: Dict[str, Any] = {}
    nmap: Dict[str, str] = {}
    keys: List[str] = []

    for w in weights:
        name = getattr(w, "name", "<noname>")
        occ = counts.get(name, 0)
        counts[name] = occ + 1

        key = _weight_key(w, occ)
        wmap[key] = w
        nmap[key] = name
        keys.append(key)

    return wmap, nmap, keys

def weight_diff_report(
    m1: Any,
    m2: Any,
    *,
    top: int = 30,
    include_ok: bool = False,
    include_extra: bool = False,
) -> List[Tuple[float, str, str, Any]]:
    """
    Compare weights between two models using stable keys.

    Each row is:
      (max_abs_diff or inf, tag, weight_id, shape_info)

    Where tag in:
      {"OK", "MISSING", "SHAPE", "EXTRA"}

    Notes
    -----
    - Uses w.path when available (best).
    - Otherwise uses name + occurrence index.
    - Set top<=0 to return all rows.
    """
    w1, n1, k1 = _index_weights(m1)
    w2, n2, k2 = _index_weights(m2)

    rows: List[Tuple[float, str, str, Any]] = []

    # --- compare m1 -> m2
    for key in k1:
        a_w = w1[key]
        a_name = n1[key]
        wid = f"{a_name} | {key}"

        if key not in w2:
            shp = tuple(getattr(a_w, "shape", ()))
            rows.append((np.inf, "MISSING", wid, shp))
            continue

        b_w = w2[key]

        a = _to_numpy(a_w)
        b = _to_numpy(b_w)

        if a.shape != b.shape:
            rows.append((np.inf, "SHAPE", wid, (a.shape, b.shape)))
            continue

        if a.size:
            d = float(np.nanmax(np.abs(a - b)))
            if np.isnan(d):
                d = float(np.inf)
        else:
            d = 0.0

        if include_ok or d > 0.0:
            rows.append((d, "OK", wid, a.shape))

    # --- optionally report m2 extras
    if include_extra:
        k1_set = set(k1)
        for key in k2:
            if key in k1_set:
                continue
            b_w = w2[key]
            b_name = n2[key]
            wid = f"{b_name} | {key}"
            shp = tuple(getattr(b_w, "shape", ()))
            rows.append((np.inf, "EXTRA", wid, shp))

    # Sort with inf first, then by diff desc
    rows.sort(key=lambda x: (np.isfinite(x[0]), x[0]), reverse=True)

    if top is None or int(top) <= 0:
        return rows
    return rows[: int(top)]

def _weight_diff_report(
    m1: Any,
    m2: Any,
    *,
    top: int = 30,
    include_ok: bool = False,
) -> List[Tuple[float, str, str, Any]]:
    """
    Compare weights by name; returns rows sorted by worst diff first.

    Each row is:
      (max_abs_diff or inf, tag, weight_name, shape_info)
    where tag in {"OK","MISSING","SHAPE"}.
    """
    w2 = {w.name: w for w in getattr(m2, "weights", [])}
    rows: List[Tuple[float, str, str, Any]] = []

    for w in getattr(m1, "weights", []):
        name = w.name
        if name not in w2:
            rows.append(
                (np.inf, "MISSING", name, tuple(
                    getattr(w, "shape", ()))))
            continue

        a = _to_numpy(w)
        b = _to_numpy(w2[name])

        if a.shape != b.shape:
            rows.append((np.inf, "SHAPE", name, (a.shape, b.shape)))
            continue

        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        if include_ok or d > 0.0:
            rows.append((d, "OK", name, a.shape))

    rows.sort(key=lambda x: x[0], reverse=True)

    # If include_ok=False, the tail may be empty; keep only top anyway
    return rows[: int(top)]


def model_scaling_digest(model: Any) -> str:
    """
    Stable digest of `model.scaling_kwargs` to verify the same config
    survived reload.
    """
    sk = getattr(model, "scaling_kwargs", {}) or {}
    s = json.dumps(sk, sort_keys=True, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def debug_model_reload(
    mem_model: Any,
    load_model: Any,
    dataset: Iterable,
    *,
    pred_key: str = "subs_pred",
    also_check: Optional[List[str]] = None,
    top_weights: int = 30,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Run a compact reload debug on one batch and return a dict report.

    - Compares predictions (max/mean abs diff) for `pred_key` (+ optional keys).
    - Compares weights by name (MISSING/SHAPE/OK).
    - Compares scaling_kwargs digest + time_units attribute.
    """
    log = log_fn or print
    also = list(also_check or [])
    if pred_key not in also:
        also = [pred_key] + also

    xb, _ = _get_one_batch(dataset)

    out_mem = mem_model(xb, training=False)
    out_ld  = load_model(xb, training=False)

    pred_report: Dict[str, Dict[str, float]] = {}

    for k in also:
        a = _to_numpy(_extract_output(mem_model, out_mem, k))
        b = _to_numpy(_extract_output(load_model, out_ld, k))

        diff = np.abs(a - b)
        max_abs = float(np.max(diff)) if diff.size else 0.0
        mean_abs = float(np.mean(diff)) if diff.size else 0.0

        # relative check (avoid division by zero)
        denom = np.maximum(np.abs(a), np.abs(b))
        rel = diff / np.maximum(denom, 1e-12)
        max_rel = float(np.max(rel)) if rel.size else 0.0

        pred_report[k] = {
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "max_rel_diff": max_rel,
        }

    mv_present = any("mv" in w.name.lower() for w in getattr(mem_model, "weights", []))
    kappa_present = any("kappa" in w.name.lower() for w in getattr(mem_model, "weights", []))

    # w_rows = weight_diff_report(
    #     mem_model, load_model, top=top_weights, 
    #     include_ok=False, 
    #     include_extra= True, 
    # )
    # n_missing = sum(1 for _, tag, _, _ in w_rows if tag == "MISSING")
    # n_shape   = sum(1 for _, tag, _, _ in w_rows if tag == "SHAPE")
    
    w_all = weight_diff_report(
        mem_model,
        load_model,
        top=0,              # all
        include_ok=False,
        include_extra=True,
    )
    w_top = w_all[:top_weights]
    n_missing = sum(1 for _, t, _, _ in w_all if t == "MISSING")
    n_shape = sum(1 for _, t, _, _ in w_all if t == "SHAPE")

    sk_mem = model_scaling_digest(mem_model)
    sk_ld  = model_scaling_digest(load_model)

    tu_mem = getattr(mem_model, "time_units", None)
    tu_ld  = getattr(load_model, "time_units", None)

    # Pretty logs
    log("=" * 72)
    log("[GeoPrior][Reload Debug] One-batch prediction diffs")
    for k, d in pred_report.items():
        log(
            f"  - {k}: max_abs={d['max_abs_diff']:.6g} "
            f"mean_abs={d['mean_abs_diff']:.6g} "
            f"max_rel={d['max_rel_diff']:.6g}"
        )

    log("[GeoPrior][Reload Debug] Weight presence (mem model)")
    log(f"  - mv in weights?    {mv_present}")
    log(f"  - kappa in weights? {kappa_present}")

    log("[GeoPrior][Reload Debug] scaling_kwargs digest / time_units")
    log(f"  - sk_digest mem : {sk_mem}")
    log(f"  - sk_digest load: {sk_ld}")
    log(f"  - time_units mem : {tu_mem}")
    log(f"  - time_units load: {tu_ld}")

    if w_top:
        log("[GeoPrior][Reload Debug] Top weight issues (by name)")
        for d, tag, name, shp in w_top:
            # d is inf for MISSING/SHAPE
            log(f"  {tag:7s} max_abs={d} | {name} | {shp}")
    else:
        log("[GeoPrior][Reload Debug] No nonzero/shape/missing issues in top scan.")

    # Simple pass/fail (you can tweak thresholds)
    # - fail if shape/missing weights exist
    # - fail if prediction diffs exceed (atol + rtol * scale)
    ok_preds = True
    for k, d in pred_report.items():
        if d["max_abs_diff"] > (atol + rtol * 1.0):
            ok_preds = False
            break

    ok = (n_missing == 0) and (n_shape == 0) and ok_preds

    report = {
        "ok": bool(ok),
        "pred": pred_report,
        "weights_top": w_top,
        "n_missing_or_shape_in_top": {"missing": int(n_missing), "shape": int(n_shape)},
        "mv_in_weights": bool(mv_present),
        "kappa_in_weights": bool(kappa_present),
        "sk_digest_mem": sk_mem,
        "sk_digest_load": sk_ld,
        "time_units_mem": tu_mem,
        "time_units_load": tu_ld,
    }

    log(f"[GeoPrior][Reload Debug] ok={report['ok']}")
    log("=" * 72)

    return report

# ==============================================================
# VERBOSE SHAPE DIAGNOSTICS (paste-only snippets)
# - Uses tf_shape() (works in graph) + a small helper.
# - Triggered when verbose > 3.
# ==============================================================

def _vshape(tag, x):
    """Print runtime shape + rank (graph-safe)."""
    if x is None:
        tf_print("[shape]", tag, "= None")
        return
    xr = tf_rank(x)
    xs = tf_shape(x)
    tf_print("[shape]", tag, "rank=", xr, "shape=", xs)

def _vshapes(title, items):
    """Convenience: print a block of shapes."""
    tf_print("\n==========", title, "==========")
    for tag, x in items:
        _vshape(tag, x)
    tf_print("================================\n")

# ---------------------------------------------------------------------
# Small gates
# ---------------------------------------------------------------------
def dbg_on(verbose: int, level: int) -> bool:
    """Return True if verbose is strictly above level."""
    return int(verbose) >= int(level)


def dbg_run_first_iter(
    *,
    verbose: int,
    level: int,
    iterations: Tensor,
    fn,
) -> None:
    """
    Run fn() only at optimizer.iterations == 0.

    Graph-safe: uses tf.cond and returns a dummy scalar.
    """
    if not dbg_on(verbose, level):
        return

    it = tf_cast(iterations, tf_int32)
    z = tf_constant(0, tf_int32)

    def _do():
        fn()
        return z

    tf_cond(tf_equal(it, 0), _do, lambda: z)

# ---------------------------------------------------------------------
# Basic stats helpers
# ---------------------------------------------------------------------
def dbg_stats(tag: str, x: Tensor) -> None:
    """Print min/max/mean (graph-safe)."""
    tf_print(
        tag,
        "min/max/mean=",
        tf_reduce_min(x),
        tf_reduce_max(x),
        tf_reduce_mean(x),
    )


def dbg_pde_divergence_maxabs(
    *,
    verbose: int,
    raw_dKdhx_dcoords: Tensor,
    raw_d_K_dh_dx_dx: Tensor,
    raw_d_K_dh_dy_dy: Tensor,
    d_K_dh_dx_dx: Optional[Tensor] = None,
    d_K_dh_dy_dy: Optional[Tensor] = None,
    level: int = 7,
    prefix: str = "pde/div",
) -> None:
    """
    Print max-abs diagnostics for the divergence terms, before and
    optionally after normalization/chain-rule correction.

    Usage
    -----
    # before normalization only:
    dbg_pde_divergence_maxabs(
        verbose=verbose,
        raw_dKdhx_dcoords=dKdhx_dcoords,
        raw_d_K_dh_dx_dx=d_K_dh_dx_dx_raw,
        raw_d_K_dh_dy_dy=d_K_dh_dy_dy_raw,
    )

    # after normalization too:
    dbg_pde_divergence_maxabs(
        verbose=verbose,
        raw_dKdhx_dcoords=dKdhx_dcoords,
        raw_d_K_dh_dx_dx=d_K_dh_dx_dx_raw,
        raw_d_K_dh_dy_dy=d_K_dh_dy_dy_raw,
        d_K_dh_dx_dx=d_K_dh_dx_dx,
        d_K_dh_dy_dy=d_K_dh_dy_dy,
    )
    """
    if not dbg_on(verbose, level):
        return

    tf_print(f"[{prefix}] before normalization:")
    tf_print(
        "max|dKdhx_dcoords|=",
        tf_reduce_max(tf_abs(raw_dKdhx_dcoords)),
    )
    tf_print(
        "max|d_K_dh_dx_dx_raw|=",
        tf_reduce_max(tf_abs(raw_d_K_dh_dx_dx)),
    )
    tf_print(
        "max|d_K_dh_dy_dy_raw|=",
        tf_reduce_max(tf_abs(raw_d_K_dh_dy_dy)),
    )

    if (d_K_dh_dx_dx is None) or (d_K_dh_dy_dy is None):
        return

    tf_print(f"[{prefix}] after normalization:")
    tf_print(
        "max|d_K_dh_dx_dx|=",
        tf_reduce_max(tf_abs(d_K_dh_dx_dx)),
    )
    tf_print(
        "max|d_K_dh_dy_dy|=",
        tf_reduce_max(tf_abs(d_K_dh_dy_dy)),
    )


def dbg_gw_units_and_sec_scale(
    *,
    verbose: int,
    gw_units: Any,
    gw_res_before: Tensor,
    gw_res_after: Tensor,
    level: int = 7,
    prefix: str = "gw/units",
) -> None:
    """
    Print GW residual diagnostics before/after applying sec_u scaling.

    Call this right after you do:
        gw_res_before = gw_res
        gw_res = gw_res * sec_u
        gw_res_after = gw_res

    Parameters
    ----------
    gw_units:
        Usually resolve_gw_units(sk). Keep it as Any so callers can
        pass python strings without TF ops.

    Notes
    -----
    We print RMS to catch accidental unit explosions.
    """
    if not dbg_on(verbose, level):
        return

    tf_print(
        f"[{prefix}] resolve_gw_units(sk)=",
        gw_units,
    )
    tf_print(
        f"[{prefix}] gw_res BEFORE sec_u: RMS=",
        to_rms(gw_res_before),
        "| max|.|=",
        tf_reduce_max(tf_abs(gw_res_before)),
    )
    tf_print(
        f"[{prefix}] gw_res AFTER  sec_u: RMS=",
        to_rms(gw_res_after),
        "| max|.|=",
        tf_reduce_max(tf_abs(gw_res_after)),
    )

    # Optional: if you want to catch NaNs early (cheap)
    tf_print_nonfinite(f"{prefix}/gw_res_before", gw_res_before)
    tf_print_nonfinite(f"{prefix}/gw_res_after", gw_res_after)


def dbg_mae(tag: str, y: Tensor, yhat: Tensor) -> None:
    """Print batch MAE for y vs yhat."""
    tf_print(tag, "mae(batch)=", tf_reduce_mean(tf_abs(y - yhat)))


def dbg_chk_finite(tag: str, x: Tensor) -> Tensor:
    """Assert finite, return x (small helper)."""
    tf_debugging.assert_all_finite(x, f"{tag} has NaN/Inf")
    return x


def _median_index(quantiles: Optional[Sequence[float]]) -> Optional[int]:
    if quantiles is None:
        return None
    qs = list(quantiles)
    try:
        return int(qs.index(0.5))
    except ValueError:
        return int(len(qs) // 2)


def _pick_q50(
    y_pred: Tensor,
    quantiles: Optional[Sequence[float]],
) -> Tensor:
    """Pick the median quantile if `y_pred` is quantile-aware.

    Uses static rank whenever possible (safe in `tf.function`).
    """
    q_idx = _median_index(quantiles)
    if q_idx is None:
        return y_pred

    r = y_pred.shape.rank
    # (B,H,Q,1) -> (B,H,1)
    if r == 4:
        return y_pred[:, :, q_idx, :]
    # (B,H,Q) -> (B,H,1)
    if r == 3:
        return y_pred[:, :, q_idx : q_idx + 1]

    # Unknown rank: keep original tensor (debug-only).
    return y_pred

    r = tf_rank(y_pred)
    # (B,H,Q,1) -> (B,H,1)
    if int(r) == 4:
        return y_pred[:, :, q_idx, :]
    # (B,H,Q) -> (B,H,1)
    if int(r) == 3:
        return y_pred[:, :, q_idx : q_idx + 1]
    return y_pred

# ---------------------------------------------------------------------
# Step 0/1/2: input packaging
# ---------------------------------------------------------------------
def dbg_step0_inputs_targets(
    *,
    verbose: int,
    inputs: Dict[str, Any],
    targets: Any,
    level: int = 12
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("\n[train_step] verbose=", verbose)
    tf_print("[train_step] --- step 0/1 inputs+targets ---")
    tf_print("[train_step] inputs keys:", list(inputs.keys()))
    if isinstance(targets, dict):
        tf_print("[train_step] targets keys:", list(targets.keys()))

    _vshapes(
        "STEP 0/1 (raw tensors)",
        [
            ("inputs['H_field']", inputs.get("H_field", None)),
            ("inputs['soil_thickness']",
             inputs.get("soil_thickness", None)),
            ("inputs['coords']", inputs.get("coords", None)),
            ("inputs['static_features']",
             inputs.get("static_features", None)),
            ("inputs['dynamic_features']",
             inputs.get("dynamic_features", None)),
            ("inputs['future_features']",
             inputs.get("future_features", None)),
        ],
    )


def dbg_step1_thickness(
    *,
    verbose: int,
    H_field: Tensor,
    H_si: Tensor,
    level: int = 12
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step 1 thickness ---")
    _vshapes(
        "STEP 1 (thickness)",
        [
            ("H_field", H_field),
            ("H_si", H_si),
        ],
    )


def dbg_step2_coords_checks(
    *,
    verbose: int,
    coords: Tensor,
    inputs: Dict[str, Any],
    level: int= 12, 
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_debugging.assert_equal(
        tf_rank(coords),
        3,
        "coords must be rank-3 (B,H,3)",
    )
    tf_debugging.assert_equal(
        tf_shape(coords)[-1],
        3,
        "coords last dim must be 3",
    )

    tf_print_nonfinite("train_step/coords", coords)

    for k in (
        "static_features",
        "dynamic_features",
        "future_features",
        "H_field",
    ):
        v = inputs.get(k, None)
        if v is None:
            continue
        tf_print_nonfinite(
            f"train_step/{k}",
            tf_cast(v, tf_float32),
        )


# ---------------------------------------------------------------------
# Units / layout checks
# ---------------------------------------------------------------------
def dbg_units_once(
    *,
    verbose: int,
    iterations: Tensor,
    targets: Dict[str, Tensor],
    gwl_pred_final: Tensor,
    s_pred_final: Tensor,
    quantiles: Optional[Sequence[float]],
    level: int = 7, 
) -> None:
    if not dbg_on(verbose, level):
        return

    it = tf_cast(iterations, tf_int32)

    def _dbg() -> Tensor:
        yt_g = tf_cast(targets["gwl_pred"], tf_float32)
        yt_s = tf_cast(targets["subs_pred"], tf_float32)

        yp_g = _pick_q50(tf_cast(gwl_pred_final, tf_float32),
                         quantiles)
        yp_s = _pick_q50(tf_cast(s_pred_final, tf_float32),
                         quantiles)

        tf_print("\n[train_step][units] iter=", it)

        tf_print("[shapes] yt_g=", tf_shape(yt_g),
                 "yp_g=", tf_shape(yp_g))
        tf_print("[shapes] yt_s=", tf_shape(yt_s),
                 "yp_s=", tf_shape(yp_s))

        tf_print("[gwl] y_true min/max/mean =",
                 tf_reduce_min(yt_g),
                 tf_reduce_max(yt_g),
                 tf_reduce_mean(yt_g))
        tf_print("[gwl] y_pred(p50) min/max/mean =",
                 tf_reduce_min(yp_g),
                 tf_reduce_max(yp_g),
                 tf_reduce_mean(yp_g))

        tf_print("[subs] y_true min/max/mean =",
                 tf_reduce_min(yt_s),
                 tf_reduce_max(yt_s),
                 tf_reduce_mean(yt_s))
        tf_print("[subs] y_pred(p50) min/max/mean =",
                 tf_reduce_min(yp_s),
                 tf_reduce_max(yp_s),
                 tf_reduce_mean(yp_s))

        tf_print("[gwl] mean(|y_true|) =",
                 tf_reduce_mean(tf_abs(yt_g)))
        tf_print("[gwl] mae(batch,p50) =",
                 tf_reduce_mean(tf_abs(yt_g - yp_g)))
        return tf_constant(0, tf_int32)

    _ = tf_cond(
        tf_equal(it, 0),
        _dbg,
        lambda: tf_constant(0, tf_int32),
    )


def dbg_assert_data_layout(
    *,
    verbose: int,
    data_final: Tensor,
    data_mean_raw: Optional[Tensor],
    quantiles: Optional[Sequence[float]],
    level: int =12, 
) -> None:
    if not dbg_on(verbose, level):
        return

    if quantiles is None:
        tf_debugging.assert_equal(
            tf_rank(data_final),
            3,
            "data_final must be (B,H,O)",
        )
    else:
        tf_debugging.assert_equal(
            tf_rank(data_final),
            4,
            "data_final must be (B,H,Q,O)",
        )
        tf_debugging.assert_equal(
            tf_shape(data_final)[2],
            len(list(quantiles)),
            "Q axis mismatch",
        )

    if data_mean_raw is not None:
        tf_debugging.assert_equal(
            tf_rank(data_mean_raw),
            3,
            "data_mean_raw must be (B,H,O)",
        )


# ---------------------------------------------------------------------
# Step 3: mean-head prep
# ---------------------------------------------------------------------
def dbg_step3_mean_head(
    *,
    verbose: int,
    gwl_mean_raw: Tensor,
    gwl_si: Tensor,
    h_si: Tensor,
    level: int = 12, 
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step: mean-head prep ---")
    _vshapes(
        "STEP 3.2 (mean head)",
        [
            ("gwl_mean_raw", gwl_mean_raw),
            ("gwl_si", gwl_si),
            ("h_si", h_si),
        ],
    )


def dbg_step31_forward_outputs(
    *,
    verbose: int,
    data_final: Tensor,
    s_pred_final: Tensor,
    gwl_pred_final: Tensor,
    data_mean_raw: Optional[Tensor],
    phys_mean_raw: Tensor,
    level: int =12
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step: forward outputs ---")
    _vshape("data_final", data_final)
    _vshape("s_pred_final", s_pred_final)
    _vshape("gwl_pred_final", gwl_pred_final)
    if data_mean_raw is not None:
        _vshape("data_mean_raw", data_mean_raw)
    _vshape("phys_mean_raw", phys_mean_raw)


# ---------------------------------------------------------------------
# Step 3.3: physics fields
# ---------------------------------------------------------------------
def dbg_step33_physics_logits(
    *,
    verbose: int,
    K_logits: Tensor,
    Ss_logits: Tensor,
    dlogtau_logits: Tensor,
    Q_logits: Optional[Tensor],
    K_base: Tensor,
    Ss_base: Tensor,
    dlogtau_base: Tensor,
    level: int =12, 
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step: physics logits/base ---")
    _vshapes(
        "STEP 3.3 (physics logits)",
        [
            ("K_logits", K_logits),
            ("Ss_logits", Ss_logits),
            ("dlogtau_logits", dlogtau_logits),
            ("Q_logits", Q_logits),
            ("K_base", K_base),
            ("Ss_base", Ss_base),
            ("dlogtau_base", dlogtau_base),
        ],
    )


def dbg_step33_physics_fields(
    *,
    verbose: int,
    K_field: Tensor,
    Ss_field: Tensor,
    tau_field: Tensor,
    tau_phys: Tensor,
    Hd_eff: Tensor,
    delta_log_tau: Tensor,
    logK: Tensor,
    logSs: Tensor,
    log_tau: Tensor,
    log_tau_phys: Tensor,
    level: int =12, 
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step: physics fields (SI) ---")
    _vshapes(
        "STEP 3.3 (physics fields)",
        [
            ("K_field", K_field),
            ("Ss_field", Ss_field),
            ("tau_field", tau_field),
            ("tau_phys", tau_phys),
            ("Hd_eff", Hd_eff),
            ("delta_log_tau", delta_log_tau),
            ("logK", logK),
            ("logSs", logSs),
            ("log_tau", log_tau),
            ("log_tau_phys", log_tau_phys),
        ],
    )


# ---------------------------------------------------------------------
# Step 4: AD derivatives
# ---------------------------------------------------------------------
def dbg_step4_ad_raw(
    *,
    verbose: int,
    dh_dcoords: Tensor,
    dh_dt_raw: Tensor,
    dh_dx_raw: Tensor,
    dh_dy_raw: Tensor,
    K_dh_dx: Tensor,
    K_dh_dy: Tensor,
    dKdhx_dcoords: Tensor,
    dKdhy_dcoords: Tensor,
    d_K_dh_dx_dx_raw: Tensor,
    d_K_dh_dy_dy_raw: Tensor,
    dK_dcoords: Tensor,
    dSs_dcoords: Tensor,
    dK_dx_raw: Tensor,
    dK_dy_raw: Tensor,
    dSs_dx_raw: Tensor,
    dSs_dy_raw: Tensor,
    level: int= 12, 
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step: AD derivatives (raw) ---")
    _vshapes(
        "STEP 4 (AD raw grads)",
        [
            ("dh_dcoords", dh_dcoords),
            ("dh_dt_raw", dh_dt_raw),
            ("dh_dx_raw", dh_dx_raw),
            ("dh_dy_raw", dh_dy_raw),
            ("K_dh_dx", K_dh_dx),
            ("K_dh_dy", K_dh_dy),
            ("dKdhx_dcoords", dKdhx_dcoords),
            ("dKdhy_dcoords", dKdhy_dcoords),
            ("d_K_dh_dx_dx_raw", d_K_dh_dx_dx_raw),
            ("d_K_dh_dy_dy_raw", d_K_dh_dy_dy_raw),
            ("dK_dcoords", dK_dcoords),
            ("dSs_dcoords", dSs_dcoords),
            ("dK_dx_raw", dK_dx_raw),
            ("dK_dy_raw", dK_dy_raw),
            ("dSs_dx_raw", dSs_dx_raw),
            ("dSs_dy_raw", dSs_dy_raw),
        ],
    )


def dbg_step41_si_grads(
    *,
    verbose: int,
    dh_dt: Tensor,
    d_K_dh_dx_dx: Tensor,
    d_K_dh_dy_dy: Tensor,
    dK_dx: Tensor,
    dK_dy: Tensor,
    dSs_dx: Tensor,
    dSs_dy: Tensor,
    level: int =12, 
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step: chain-rule corrected ---")
    _vshapes(
        "STEP 4.1 (SI grads)",
        [
            ("dh_dt", dh_dt),
            ("d_K_dh_dx_dx", d_K_dh_dx_dx),
            ("d_K_dh_dy_dy", d_K_dh_dy_dy),
            ("dK_dx", dK_dx),
            ("dK_dy", dK_dy),
            ("dSs_dx", dSs_dx),
            ("dSs_dy", dSs_dy),
        ],
    )


# ---------------------------------------------------------------------
# Step 5: Q forcing
# ---------------------------------------------------------------------
def dbg_step5_q_source(
    *,
    verbose: int,
    Q_si: Tensor,
    dh_dt: Tensor,
    level: int= 12, 
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step: Q source term ---")
    _vshapes(
        "STEP 5 (Q)",
        [
            ("Q_si", Q_si),
            ("dh_dt", dh_dt),
        ],
    )


# ---------------------------------------------------------------------
# Step 6: consolidation
# ---------------------------------------------------------------------
def dbg_cons_units_rms(
    *,
    verbose: int,
    sk: Dict[str, Any],
    cons_res: Tensor,
    level: int = 7, 
) -> None:
    if not dbg_on(verbose, level):
        return

    mode = resolve_cons_units(sk)
    tf_print("cons_units=", mode, "rms=", to_rms(cons_res))


def dbg_step6_consolidation(
    *,
    verbose: int,
    allow_resid: bool,
    cons_active: bool,
    s_mean_raw: Tensor,
    s_pred_si: Tensor,
    dt_units: Tensor,
    s0_cum_11: Tensor,
    s_inc_pred: Tensor,
    s_state: Tensor,
    h_ref_si_11: Tensor,
    h_state: Tensor,
    cons_step_m: Tensor,
    cons_res: Tensor,
) -> None:
    if not dbg_on(verbose, 10):
        return

    tf_print("[train_step] --- step: consolidation branch ---")
    tf_print("[train_step] allow_resid=", allow_resid,
             "| cons_active=", cons_active)

    _vshapes(
        "STEP 6 (consolidation state build)",
        [
            ("s_mean_raw", s_mean_raw),
            ("s_pred_si", s_pred_si),
            ("dt_units", dt_units),
            ("s0_cum_11", s0_cum_11),
            ("s_inc_pred", s_inc_pred),
            ("s_state", s_state),
            ("h_ref_si_11", h_ref_si_11),
            ("h_state", h_state),
            ("cons_step_m", cons_step_m),
            ("cons_res", cons_res),
        ],
    )


# ---------------------------------------------------------------------
# Step 7: residuals + priors
# ---------------------------------------------------------------------
def dbg_step7_residuals(
    *,
    verbose: int,
    gw_res: Tensor,
    prior_res: Tensor,
    smooth_res: Tensor,
    loss_mv: Tensor,
    bounds_res: Tensor,
    loss_bounds: Tensor,
    level: int =12
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step: residuals + priors ---")
    _vshapes(
        "STEP 7 (residual tensors)",
        [
            ("gw_res", gw_res),
            ("prior_res", prior_res),
            ("smooth_res", smooth_res),
            ("loss_mv", loss_mv),
            ("bounds_res", bounds_res),
            ("loss_bounds", loss_bounds),
        ],
    )


# ---------------------------------------------------------------------
# Step 8: scaling / finiteness checks
# ---------------------------------------------------------------------
def dbg_step8_scaling(
    *,
    verbose: int,
    cons_res_raw: Tensor,
    gw_res_raw: Tensor,
    cons_res: Tensor,
    gw_res: Tensor,
    level: int =7, 
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step 8 scaling ---")
    _vshapes(
        "STEP 8 (before/after scaling)",
        [
            ("cons_res_raw", cons_res_raw),
            ("gw_res_raw", gw_res_raw),
            ("cons_res", cons_res),
            ("gw_res", gw_res),
        ],
    )


def dbg_chk_scales(
    *,
    verbose: int,
    scales: Dict[str, Tensor],
    level: int = 2, 
) -> None:
    if not dbg_on(verbose, level):
        return

    for k in ("cons_scale", "gw_scale"):
        v = scales.get(k, None)
        if v is None:
            continue
        tf_debugging.assert_all_finite(
            v,
            f"{k} has NaN/Inf",
        )


def dbg_chk_core_finite(
    *,
    verbose: int,

    cons_res: Tensor,
    gw_res: Tensor,
    tau_field: Tensor,
    K_field: Tensor,
    Ss_field: Tensor,
    level: int=2, 
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_debugging.assert_all_finite(
        cons_res,
        "cons_res has NaN/Inf",
    )
    tf_debugging.assert_all_finite(
        gw_res,
        "gw_res has NaN/Inf",
    )
    tf_debugging.assert_all_finite(
        tau_field,
        "tau_field has NaN/Inf",
    )
    tf_debugging.assert_all_finite(
        K_field,
        "K_field has NaN/Inf",
    )
    tf_debugging.assert_all_finite(
        Ss_field,
        "Ss_field has NaN/Inf",
    )


# ---------------------------------------------------------------------
# Step 9/10: loss + grads
# ---------------------------------------------------------------------
def dbg_step9_losses(
    *,
    verbose: int,
    data_loss: "Tensor | None" = None,
    loss_cons: "Tensor | None" = None,
    loss_gw: "Tensor | None" = None,
    loss_prior: "Tensor | None" = None,
    loss_smooth: "Tensor | None" = None,
    physics_loss_raw: "Tensor | None" = None,
    physics_loss_scaled: "Tensor | None" = None,
    total_loss: "Tensor | None" = None,
    level: int = 7,
) -> None:
    """Debug-print loss scalars (only those provided)."""
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step 9 losses ---")

    pairs = []
    if data_loss is not None:
        pairs.append(("data_loss", data_loss))
    if loss_cons is not None:
        pairs.append(("loss_cons", loss_cons))
    if loss_gw is not None:
        pairs.append(("loss_gw", loss_gw))
    if loss_prior is not None:
        pairs.append(("loss_prior", loss_prior))
    if loss_smooth is not None:
        pairs.append(("loss_smooth", loss_smooth))
    if physics_loss_raw is not None:
        pairs.append(("physics_loss_raw", physics_loss_raw))
    if physics_loss_scaled is not None:
        pairs.append(("physics_loss_scaled", physics_loss_scaled))
    if total_loss is not None:
        pairs.append(("total_loss", total_loss))

    if not pairs:
        tf_print("STEP 9 (loss scalars): <no losses provided>")
        return

    _vshapes("STEP 9 (loss scalars)", pairs)


def dbg_step10_grads(
    *,
    verbose: int,
    trainable_vars: Sequence[Tensor],
    grads: Sequence[Optional[Tensor]],
    level: int = 9,
) -> None:
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- step 10 gradients ---")
    tf_print("[train_step] n_trainable_vars =",
             tf_constant(len(trainable_vars), tf_int32))

    n_show = min(12, len(trainable_vars))
    for v, g in list(zip(trainable_vars, grads))[:n_show]:
        if g is None:
            tf_print("[grad] NONE | var:", v.name,
                     "| var_shape:", tf_shape(v))
        else:
            tf_print("[grad]", v.name,
                     "| var_shape:", tf_shape(v),
                     "| grad_shape:", tf_shape(g))


def dbg_term_grads_finite(
    *,
    verbose: int,
    debug_grads: bool,
    trainable_vars: Sequence[Tensor],
    data_loss: Tensor,
    terms_scaled: Dict[str, Tensor],
    tape: Any,
    level: int =1
) -> None:
    if not debug_grads:
        return
    if not dbg_on(verbose, level):
        # debug_grads is an explicit toggle, but keep a
        # tiny verbosity gate to avoid surprises.
        pass

    # Check each term independently.
    for name, lt in {"data": data_loss, **terms_scaled}.items():
        g = tape.gradient(lt, trainable_vars)
        _assert_grads_finite(g, trainable_vars)


def dbg_done_apply_gradients(
    *,
    debug_grads: bool =False, 
    verbose: int=1
) -> None:
    if not debug_grads:
        return
    
    if not dbg_on(verbose, 1):
        pass
    tf_print("[train_step] --- apply_gradients done ---\n")

def dbg_select_q(
    y: Tensor,
    quantiles: Optional[Sequence[float]],
    *,
    q: float = 0.5,
) -> Tensor:
    """
    Select a quantile slice if y is (B,H,Q,1)/(B,H,Q).

    If quantiles is None, returns y as-is.
    """
    if quantiles is None:
        return y

    qs = np.asarray(list(quantiles), dtype=float)
    idx = int(np.argmin(np.abs(qs - float(q))))

    r = int(y.shape.rank or 0)
    if r >= 3:
        # (B,H,Q,1) or (B,H,Q)
        yq = y[..., idx : idx + 1, ...]
        # If last dim is 1, squeeze Q only.
        # Keep (B,H,1) convention when possible.
        if yq.shape.rank == 4:
            return yq[:, :, 0, :]
        return yq
    return y


def dbg_step5_q(
    *,
    verbose: int,
    Q_si: Tensor,
    dh_dt: Tensor,
    level: int = 12, 
) -> None:
    """Print Q source term block."""
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- Q source term ---")
    _vshapes(
        "STEP 5 (Q)",
        [
            ("Q_si", Q_si),
            ("dh_dt", dh_dt),
        ],
    )



def _dbg_stats_line(tag: str, x: Tensor) -> None:
    """Print min/mean/max + rms for a tensor."""
    x = tf_cast(x, tf_float32)
    tf_print(
        "[stats]",
        tag,
        "min/mean/max=",
        tf_reduce_min(x),
        tf_reduce_mean(x),
        tf_reduce_max(x),
        "| rms=",
        tf_sqrt(tf_reduce_mean(tf_square(x))),
    )


def dbg_step8_residual_scale_stats(
    *,
    verbose: int,
    level: int = 3,
    cons_res_raw: Tensor,
    cons_scale: Tensor,
    gw_res_raw: Tensor,
    gw_scale: Tensor,
) -> None:
    """
    Debug block for residuals + scaling stats.

    Replaces:
        _stats("cons_res_raw", cons_res)
        _stats("cons_scale", scales["cons_scale"])
        _stats("gw_res_raw", gw_res)
        _stats("gw_scale", scales["gw_scale"])
    """
    if not dbg_on(verbose, level):
        return

    tf_print("[train_step] --- residual/scale stats ---")
    _dbg_stats_line("cons_res_raw", cons_res_raw)
    _dbg_stats_line("cons_scale", cons_scale)
    _dbg_stats_line("gw_res_raw", gw_res_raw)
    _dbg_stats_line("gw_scale", gw_scale)


def dbg_dt_debug(
    *,
    verbose: int,
    level: int = 3,
    time_units: str,
    dt_units: Tensor,
    t: Tensor,
) -> None:
    """
    Debug dt conversion and t-grid sanity.

    Replaces the "dt debug" block.
    """
    if not dbg_on(verbose, level):
        return

    sec_u = seconds_per_time_unit(
        time_units,
        dtype=tf_float32,
    )
    dt_sec = dt_to_seconds(
        dt_units,
        time_units=time_units,
    )

    tf_print("[dt debug] time_units =", time_units)
    tf_print("[dt debug] sec_u =", sec_u)

    tf_print(
        "[dt debug] dt_units min/mean/max =",
        tf_reduce_min(dt_units),
        tf_reduce_mean(dt_units),
        tf_reduce_max(dt_units),
    )

    tf_print(
        "[dt debug] dt_sec   min/mean/max =",
        tf_reduce_min(dt_sec),
        tf_reduce_mean(dt_sec),
        tf_reduce_max(dt_sec),
    )

    tf_print(
        "[dt debug] dt_sec / dt_units (mean) =",
        tf_reduce_mean(
            dt_sec / tf_maximum(dt_units, tf_constant(1e-12, tf_float32))
        ),
    )

    # show the first sample time grid (B=0)
    tf_print("[t debug] t[0,:,0] =", t[0, :, 0])


def dbg_call_nonfinite(
    *,
    verbose: int,
    level: int = 9,
    coords_for_decoder: Tensor,
    H_si: Tensor,
    K_base: Tensor,
    Ss_base: Tensor,
    dlogtau_base: Tensor,
    tau_field: Tensor,
) -> None:
    """
    Debug non-finite checks for call() internal tensors.

    Replaces:
        tf_print_nonfinite("call/coords_for_decoder", coords_for_decoder)
        ...
    """
    if not dbg_on(verbose, level):
        return

    tf_print_nonfinite("call/coords_for_decoder", coords_for_decoder)
    tf_print_nonfinite("call/H_si", H_si)
    tf_print_nonfinite("call/K_base", K_base)
    tf_print_nonfinite("call/Ss_base", Ss_base)
    tf_print_nonfinite("call/tau_base", dlogtau_base)
    tf_print_nonfinite(
        "call/tau_field(pre-integrator)",
        tau_field,
    )

def dbg_step3_residual_scales(
    *,
    verbose: int,
    cons_res: Tensor,
    gw_res: Tensor,
    scales: dict,
    level: int = 3,
) -> None:
    """Print raw residual stats + scaling factors."""
    if not dbg_on(verbose, level):
        return

    cons_scale = scales.get("cons_scale", None)
    gw_scale = scales.get("gw_scale", None)

    _stats("cons_res_raw", cons_res)
    if cons_scale is not None:
        _stats("cons_scale", cons_scale)

    _stats("gw_res_raw", gw_res)
    if gw_scale is not None:
        _stats("gw_scale", gw_scale)


def dbg_dt_diag(
    *,
    verbose: int,
    time_units: str,
    dt_units: Tensor,
    t: Tensor,
    level: int = 3,
) -> None:
    """Print dt consistency checks in time_units and seconds."""
    if not dbg_on(verbose, level):
        return

    sec_u = seconds_per_time_unit(
        time_units,
        dtype=tf_float32,
    )
    dt_sec = dt_to_seconds(
        dt_units,
        time_units=time_units,
    )

    tf_print("[dt debug] time_units =", time_units)
    tf_print("[dt debug] sec_u =", sec_u)

    tf_print(
        "[dt debug] dt_units min/mean/max =",
        tf_reduce_min(dt_units),
        tf_reduce_mean(dt_units),
        tf_reduce_max(dt_units),
    )

    tf_print(
        "[dt debug] dt_sec   min/mean/max =",
        tf_reduce_min(dt_sec),
        tf_reduce_mean(dt_sec),
        tf_reduce_max(dt_sec),
    )

    tf_print(
        "[dt debug] dt_sec / dt_units (mean) =",
        tf_reduce_mean(
            dt_sec / tf_maximum(dt_units, 1e-12),
        ),
    )

    tf_print("[t debug] t[0,:,0] =", t[0, :, 0])


def dbg_call_nonfinite_diag(
    *,
    verbose: int,
    coords_for_decoder: Tensor,
    H_si: Tensor,
    K_base: Tensor,
    Ss_base: Tensor,
    dlogtau_base: Tensor,
    tau_field: Tensor,
    level: int = 9,
) -> None:
    """Print non-finite diagnostics inside call()."""
    if not dbg_on(verbose, level):
        return

    tf_print_nonfinite(
        "call/coords_for_decoder",
        coords_for_decoder,
    )
    tf_print_nonfinite("call/H_si", H_si)
    tf_print_nonfinite("call/K_base", K_base)
    tf_print_nonfinite("call/Ss_base", Ss_base)
    tf_print_nonfinite(
        "call/tau_base",
        dlogtau_base,
    )
    tf_print_nonfinite(
        "call/tau_field(pre-integrator)",
        tau_field,
    )

def dbg_gw_grad_flux_rms(
    *,
    verbose: int,
    dh_dx_raw: Tensor,
    dh_dy_raw: Tensor,
    K_field: Tensor,
    level: int = 3,
    prefix: str = "gw/gradflux",
) -> None:
    """
    Print RMS diagnostics for spatial head gradients and Darcy-like
    flux terms K*∂h/∂x, K*∂h/∂y (raw coord units).

    Replaces:
        tf_print("to_rms(dh_dx)=", to_rms(dh_dx_raw))
        tf_print("to_rms(dh_dy)=", to_rms(dh_dy_raw))
        tf_print("to_rms(K_field * dh_dx)=", to_rms(K_field * dh_dx_raw))
        tf_print("to_rms(K_field * dh_dy)=", to_rms(K_field * dh_dy_raw))
    """
    if not dbg_on(verbose, level):
        return

    tf_print(f"[{prefix}] rms(dh_dx_raw)=", to_rms(dh_dx_raw))
    tf_print(f"[{prefix}] rms(dh_dy_raw)=", to_rms(dh_dy_raw))

    fx = K_field * dh_dx_raw
    fy = K_field * dh_dy_raw

    tf_print(f"[{prefix}] rms(K*dh_dx_raw)=", to_rms(fx))
    tf_print(f"[{prefix}] rms(K*dh_dy_raw)=", to_rms(fy))
