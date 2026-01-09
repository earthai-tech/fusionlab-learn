## -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author : LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import numpy as np 

from . import KERAS_DEPS, dependency_message

# ---------------------------------------------------------------------
# Dev note:
# This module provides **shape-safe** helpers used by metrics/losses.
# Goal: avoid silent broadcasting bugs (esp. Keras 3 multi-output)
# when targets/preds carry quantile axes or accidental tiling.
# ---------------------------------------------------------------------

Tensor = KERAS_DEPS.Tensor

tf_float32 = KERAS_DEPS.float32
tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor
tf_expand_dims = KERAS_DEPS.expand_dims
tf_squeeze = KERAS_DEPS.squeeze
tf_shape = KERAS_DEPS.shape
tf_cast = KERAS_DEPS.cast
tf_reduce_sum = KERAS_DEPS.reduce_sum
tf_size = KERAS_DEPS.size
tf_reshape = KERAS_DEPS.reshape
tf_broadcast_to = KERAS_DEPS.broadcast_to
tf_gather = KERAS_DEPS.gather
tf_where = KERAS_DEPS.where 
tf_equal = KERAS_DEPS.equal 
tf_executing_eagerly =KERAS_DEPS.executing_eagerly
tf_rank = KERAS_DEPS.rank 
tf_cond = KERAS_DEPS.cond 
tf_sort = KERAS_DEPS.sort 
tf_reduce_mean = KERAS_DEPS.reduce_mean 
tf_transpose =KERAS_DEPS.transpose 
tf_abs = KERAS_DEPS.abs

DEP_MSG = dependency_message("nn._shapes")


def _infer_quantile_axis(t: Tensor, n_q: int = 3):
    """Infer quantile axis (static, conservative)."""
    shape = getattr(t, "shape", None)
    rank = getattr(shape, "rank", None)

    if rank is None:
        return None

    # Canonical: (B,H,Q,O) => axis=2
    if rank == 4:
        d2 = shape[2]
        d3 = shape[3]

        if d2 == n_q:
            return 2

        # Alternate: (B,H,O,Q) => axis=3
        if d3 == n_q:
            return 3

        return None

    # Rank-3: only accept (B,H,Q) on last axis
    if rank == 3:
        return 2 if shape[2] == n_q else None

    return None


def _as_BHO(y_true: Tensor, y_pred: Tensor | None = None):
    """Normalize y_true to (B,H,O) float32."""
    y = tf_convert_to_tensor(y_true)

    # (B,H) -> (B,H,1)
    if y.shape.rank == 2:
        y = tf_expand_dims(y, axis=-1)

    # (B,H,O,1) -> (B,H,O)
    if (y.shape.rank == 4) and (y.shape[-1] == 1):
        y = tf_squeeze(y, axis=-1)

    # If targets got tiled to (B,H,Q,O), drop Q.
    if (y.shape.rank == 4) and (y_pred is not None):
        yp_shape = getattr(y_pred, "shape", None)
        yp_rank = getattr(yp_shape, "rank", None)

        # Dev note: in this codebase rank-4 preds mean (B,H,Q,O)
        # so any rank-4 y_true is suspicious and should be untiled.
        if yp_rank == 4:
            y = y[:, :, 0, :]

    return tf_cast(y, tf_float32)


def _as_BHO_like_pred(x: Tensor, dtype=tf_float32):
    """Normalize (B,H,*) to (B,H,1)/(B,H,O)."""
    t = tf_convert_to_tensor(x)

    # (B,H) -> (B,H,1)
    if t.shape.rank == 2:
        t = tf_expand_dims(t, axis=-1)

    return tf_cast(t, dtype)


def _weighted_sum_and_count(
    values: Tensor,
    sample_weight: Tensor | None = None,
):
    """Return (sum, count) as float32."""
    v = tf_cast(values, tf_float32)

    # Unweighted: count is number of elements.
    if sample_weight is None:
        s = tf_reduce_sum(v)
        c = tf_cast(tf_size(v), tf_float32)
        return s, c

    w = tf_cast(sample_weight, tf_float32)

    # Make weights broadcastable to v.
    # Common: (B,), (B,H), (B,H,1), (B,H,O)
    if (
        (w.shape.rank == 1)
        and (v.shape.rank is not None)
        and (v.shape.rank >= 2)
    ):
        w = tf_reshape(w, [-1, 1, 1])

    elif (
        (w.shape.rank == 2)
        and (v.shape.rank is not None)
        and (v.shape.rank >= 3)
    ):
        w = tf_expand_dims(w, axis=-1)

    w = tf_broadcast_to(w, tf_shape(v))

    s = tf_reduce_sum(v * w)
    c = tf_reduce_sum(w)

    return s, c


def _q50_as_BHO(
    y_pred: Tensor,
    q_axis: int = 2,
    q_index: int = 1,
):
    """Extract q50 forecast as (B,H,O)/(B,H,1)."""
    yp = tf_convert_to_tensor(y_pred)
    r = yp.shape.rank

    # Rank-4: (B,H,Q,O)
    if r == 4:
        # Quantiles: pick q_index along q_axis.
        if yp.shape[q_axis] == 3:
            out = tf_gather(yp, q_index, axis=q_axis)
            return tf_cast(out, tf_float32)

        # Packed interval: (...,2) -> midpoint.
        if yp.shape[-1] == 2:
            mid = 0.5 * (yp[..., 0] + yp[..., 1])
            return tf_cast(mid, tf_float32)

        # Trailing singleton: (B,H,O,1) -> (B,H,O)
        if yp.shape[-1] == 1:
            out = tf_squeeze(yp, axis=-1)
            return tf_cast(out, tf_float32)

        # Last-resort: take first "O" slice.
        return tf_cast(yp[:, :, :, 0], tf_float32)

    # Rank-3: (B,H,Q) or (B,H,O)
    if r == 3:
        # (B,H,Q) -> pick q_index -> (B,H,1)
        if yp.shape[-1] == 3:
            out = tf_gather(yp, q_index, axis=-1)
            out = tf_expand_dims(out, axis=-1)
            return tf_cast(out, tf_float32)

        # Already (B,H,O)
        return tf_cast(yp, tf_float32)

    # Rank-2: (B,H) -> (B,H,1)
    if r == 2:
        out = tf_expand_dims(yp, axis=-1)
        return tf_cast(out, tf_float32)

    return tf_cast(yp, tf_float32)


def _interval80_as_BHO(y_pred: Tensor, q_axis: int = 2):
    """Return (lo, hi) broadcastable to (B,H,O)."""
    yp = tf_convert_to_tensor(y_pred)
    qax = _infer_quantile_axis(yp, n_q=3)

    # Quantiles: pick q10 and q90.
    if qax is not None:
        lo = tf_gather(yp, 0, axis=qax)
        hi = tf_gather(yp, -1, axis=qax)
        return (
            tf_cast(lo, tf_float32),
            tf_cast(hi, tf_float32),
        )

    # Packed interval: (...,2) == (lo,hi)
    if (yp.shape.rank is not None) and (yp.shape[-1] == 2):
        lo = yp[..., 0]
        hi = yp[..., 1]
        return (
            tf_cast(lo, tf_float32),
            tf_cast(hi, tf_float32),
        )

    # Fallback: treat point forecast as (lo==hi).
    p = _q50_as_BHO(yp, q_axis=q_axis)
    return tf_cast(p, tf_float32), tf_cast(p, tf_float32)



def infer_quantile_axis(t, n_q=3):
    """
    Infer which axis holds quantiles of length n_q (typically 3: q10,q50,q90).

    Returns
    -------
    int | None
        Axis index of the quantile dimension, or None if not found / packed.
    """
    shape = getattr(t, "shape", None)
    if shape is None:
        return None

    rank = shape.rank
    if rank is None:
        # Can't safely infer in graph if rank unknown; treat as "no quantile axis".
        return None

    # ---- 1) Packed interval guard: (...,2) means (lo,hi) not quantiles ----
    # Covers (B,H,2), (B,H,O,2), etc.
    if shape[-1] == 2:
        return None

    # ---- 2) Collect candidate axes with dim == n_q (static only) ----
    dims = list(shape)
    cand = [i for i, d in enumerate(dims) if d == n_q]

    # If exactly one axis matches, it's almost certainly the quantile axis.
    if len(cand) == 1:
        return cand[0]

    # ---- 3) Strong conventions / common layouts ----
    # (B,H,Q,O)
    if rank == 4 and dims[2] == n_q and dims[-1] not in (n_q, 2):
        return 2

    # (B,H,O,Q)
    if rank == 4 and dims[-1] == n_q:
        return rank - 1

    # (B,H,Q)
    if rank == 3 and dims[-1] == n_q:
        return rank - 1

    # ---- 4) Ambiguous: multiple axes have size n_q ----
    if len(cand) >= 2:
        # Prefer "quantiles before output dim" when last dim looks like O.
        # Typical: (B,H,Q,O) => rank-2
        pref = rank - 2
        if pref in cand and dims[-1] not in (n_q, 2):
            return pref

        # Otherwise, prefer last axis if it's a candidate (B,H,O,Q style)
        if (rank - 1) in cand:
            return rank - 1

        # Otherwise pick the last non-batch candidate
        non_batch = [c for c in cand if c != 0]
        return non_batch[-1] if non_batch else cand[-1]

    return None


def _mean_crossing_score(
    arr: np.ndarray,
    q_axis: int,
    q_idx: Sequence[int] = (0, 1, 2),
) -> float:
    # Compute quantile crossing score:
    #   mean(q10>q50) + mean(q50>q90) + mean(q10>q90)
    q10 = np.take(arr, int(q_idx[0]), axis=q_axis)
    q50 = np.take(arr, int(q_idx[1]), axis=q_axis)
    q90 = np.take(arr, int(q_idx[2]), axis=q_axis)

    # squeeze last dim if O=1
    if q10.ndim >= 1 and q10.shape[-1] == 1:
        q10 = q10[..., 0]
        q50 = q50[..., 0]
        q90 = q90[..., 0]

    c1 = float(np.mean(q10 > q50))
    c2 = float(np.mean(q50 > q90))
    c3 = float(np.mean(q10 > q90))
    return c1 + c2 + c3


def _safe_transpose(
    y: np.ndarray,
    axes: Tuple[int, int, int, int],
) -> np.ndarray:
    return np.transpose(y, axes)


def canonicalize_BHQO_quantiles_np(
    y: Any,
    n_q: int = 3,
    *,
    verbose: int = 0,
    log_fn: Callable[[str], None] = print,
) -> Any:
    """
    Return y in canonical (B,H,Q,O).

    Accepts common layouts:
      - (B,H,Q,O) -> unchanged
      - (B,Q,H,O) -> transpose(0,2,1,3)
      - (B,H,O,Q) -> transpose(0,1,3,2)

    If ambiguous (multiple axes match n_q), choose the transform
    with minimal quantile crossing score.
    """
    y_np = np.asarray(y)

    # Not quantile tensor (we only canonicalize rank-4).
    if y_np.ndim != 4:
        if verbose:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                f"rank={y_np.ndim}, skipping."
            )
        return y

    n_q = int(n_q)
    if n_q <= 0:
        raise ValueError("n_q must be a positive integer.")

    # candidates: interpret which axis is Q among {1,2,3}
    cand = [ax for ax in (1, 2, 3) if y_np.shape[ax] == n_q]

    # No axis matches n_q => not quantile mode, return unchanged.
    if not cand:
        if verbose:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                f"no axis matches n_q={n_q}, "
                "return unchanged."
            )
        return y_np

    options: list[tuple[str, np.ndarray]] = []

    # already (B,H,Q,O): q_axis=2 and O axis=3
    if y_np.shape[2] == n_q:
        # Keep as-is. This is the canonical layout.
        options.append(("BHQO", y_np))

    # (B,Q,H,O) -> (B,H,Q,O)
    if y_np.shape[1] == n_q:
        # Swap axes 1 and 2.
        options.append(
            (
                "BQHO->BHQO",
                _safe_transpose(y_np, (0, 2, 1, 3)),
            )
        )

    # (B,H,O,Q) -> (B,H,Q,O)
    if y_np.shape[3] == n_q:
        # Swap axes 2 and 3.
        options.append(
            (
                "BHOQ->BHQO",
                _safe_transpose(y_np, (0, 1, 3, 2)),
            )
        )

    # If nothing matched the supported transforms, keep unchanged.
    if not options:
        if verbose:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                "no supported transform matched; "
                "return unchanged."
            )
        return y_np

    # If only one option, pick it directly.
    if len(options) == 1:
        name, arr = options[0]
        if verbose:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                f"chose={name} (only option)."
            )
        return arr

    # Multiple candidates (e.g., H==Q==3):
    # pick best by minimal quantile crossing score.
    best_name: Optional[str] = None
    best_arr: Optional[np.ndarray] = None
    best_score = float("inf")

    for name, arr in options:
        # score in canonical BHQO along axis=2
        sc = _mean_crossing_score(arr, q_axis=2)
        if verbose >= 2:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                f"candidate={name} score={sc:.6f}"
            )

        if sc < best_score:
            best_score = sc
            best_name = name
            best_arr = arr

    if best_arr is None:
        # Defensive fallback; should not happen.
        if verbose:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                "no best candidate found; "
                "return unchanged."
            )
        return y_np

    if verbose:
        log_fn(
            "canonicalize_BHQO_quantiles_np: "
            f"chose={best_name} score={best_score:.6f}"
        )

    return best_arr


# def canonicalize_BHQO_quantiles_np(y, n_q=3):
#     """
#     Return y in canonical (B,H,Q,O).
#     Accepts common layouts:
#       - (B,H,Q,O) -> unchanged
#       - (B,Q,H,O) -> transpose(0,2,1,3)
#       - (B,H,O,Q) -> transpose(0,1,3,2)
#     If ambiguous (H==Q), choose the axis that minimizes quantile crossing.
#     """
#     y = np.asarray(y)
#     if y.ndim != 4:
#         return y

#     B, d1, d2, d3 = y.shape

#     # candidates: interpret which axis is Q among {1,2,3}
#     cand = [ax for ax in (1,2,3) if y.shape[ax] == n_q]
#     if not cand:
#         return y  # not quantile mode

#     def crossing_score(arr, q_axis):
#         q10 = np.take(arr, 0, axis=q_axis)
#         q50 = np.take(arr, 1, axis=q_axis)
#         q90 = np.take(arr, 2, axis=q_axis)
#         # squeeze last dim if O=1
#         if q10.ndim >= 3 and q10.shape[-1] == 1:
#             q10 = q10[..., 0]; q50 = q50[..., 0]; q90 = q90[..., 0]
#         return (np.mean(q10 > q50) + np.mean(q50 > q90) + np.mean(q10 > q90))

#     # build possible canonical transforms and pick best by crossing score
#     options = []

#     # already (B,H,Q,O): q_axis=2 and O axis=3
#     if y.shape[2] == n_q:
#         options.append(("BHQO", y))

#     # (B,Q,H,O) -> (B,H,Q,O)
#     if y.shape[1] == n_q:
#         options.append(("BQHO->BHQO", np.transpose(y, (0,2,1,3))))

#     # (B,H,O,Q) -> (B,H,Q,O)
#     if y.shape[3] == n_q:
#         options.append(("BHOQ->BHQO", np.transpose(y, (0,1,3,2))))

#     if not options:
#         return y

#     # score in canonical (B,H,Q,O) along axis=2
#     best_name, best_arr = None, None
#     best_score = 1e9
#     for name, arr in options:
#         sc = crossing_score(arr, q_axis=2)
#         if sc < best_score:
#             best_score = sc
#             best_name, best_arr = name, arr

#     # optional: print which transform chosen
#     print("[canonicalize] chose:", best_name, "score:", best_score)

#     return best_arr


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor/array-like to numpy array."""
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def _static_shape(x: Any) -> Optional[Tuple[int, ...]]:
    """Return static shape tuple if available."""
    shp = getattr(x, "shape", None)
    if shp is None:
        return None

    # tf.TensorShape
    if hasattr(shp, "as_list"):
        lst = shp.as_list()
        if lst is None:
            return None
        out = []
        for d in lst:
            out.append(-1 if d is None else int(d))
        return tuple(out)

    # numpy / python
    try:
        return tuple(int(d) for d in shp)
    except Exception:
        return None


def infer_q_axis(
    yq: Any,
    n_q: int = 3,
    default: int = 2,
    prefer: Sequence[int] = (2, 1, 3, 0),
    verbose: int = 0,
    log_fn: Callable[[str], None] = print,
) -> int:
    """
    Infer quantile axis for common BHQO layouts.

    Supports (B,H,Q,1) and (B,Q,H,1) (and similar ranks).
    """
    if n_q is None or int(n_q) <= 0:
        raise ValueError("n_q must be a positive integer.")

    n_q = int(n_q)
    default = int(default)

    shp = _static_shape(yq)
    if shp is not None and len(shp) > 0:
        axes = [i for i, d in enumerate(shp) if d == n_q]
        if len(axes) == 1:
            return int(axes[0])

        if len(axes) > 1:
            for ax in prefer:
                if int(ax) in axes:
                    return int(ax)
            return int(axes[0])

    if not tf_executing_eagerly():
        if verbose:
            msg = (
                "infer_q_axis: non-eager mode; "
                f"returning default={default}"
            )
            log_fn(msg)
        return default

    try:
        dyn = tf_shape(yq)
        rank = int(tf_rank(yq).numpy())
    except :
        if verbose:
            msg = (
                "infer_q_axis: dynamic shape failed; "
                f"returning default={default}"
            )
            log_fn(msg)
        return default

    # We only resolve common ranks safely.
    if rank < 2:
        return default

    axes_try = list(prefer)
    axes_try = [ax for ax in axes_try if ax < rank]

    for ax in axes_try:
        try:
            if int(dyn[ax].numpy()) == n_q:
                return int(ax)
        except Exception:
            continue

    if verbose:
        msg = (
            "infer_q_axis: could not locate n_q axis; "
            f"rank={rank}, returning default={default}"
        )
        log_fn(msg)

    return default


def _crossing_rates(
    q_lo: np.ndarray,
    q_mid: np.ndarray,
    q_hi: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute basic crossing rates."""
    c1 = float(np.mean(q_lo > q_mid))
    c2 = float(np.mean(q_mid > q_hi))
    c3 = float(np.mean(q_lo > q_hi))
    return c1, c2, c3


def debug_val_interval(
    model: Any,
    ds: Any,
    n_q: int = 3,
    max_batches: int = 2,
    y_key: str = "subs_pred",
    out_key: str = "subs_pred",
    default_q_axis: int = 2,
    verbose: int = 1,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """
    Debug quantile crossing on a tf.data.Dataset.

    Returns a small summary dict.
    """

    if max_batches is None:
        max_batches = 2
    max_batches = int(max_batches)

    summary: Dict[str, Any] = {
        "batches": 0,
        "q_axis": None,
        "crossings": [],
    }

    if verbose:
        log_fn("debug_val_interval: start")

    for i, (xb, yb) in enumerate(ds.take(max_batches)):
        out = model(xb, training=False)
        s = out[out_key]
        y = yb[y_key]

        s_np = _to_numpy(s)
        y_np = _to_numpy(y)

        q_axis = infer_q_axis(
            s,
            n_q=n_q,
            default=default_q_axis,
            verbose=max(0, verbose - 1),
            log_fn=log_fn,
        )

        summary["batches"] += 1
        summary["q_axis"] = int(q_axis)

        if verbose:
            msg = (
                f"\n[BATCH {i}] "
                f"y_true={y_np.shape} "
                f"pred={s_np.shape} "
                f"q_axis={q_axis}"
            )
            log_fn(msg)

        # Expect a 4D tensor for quantiles.
        if s_np.ndim != 4:
            if verbose:
                msg = (
                    "  expected rank-4 preds; "
                    f"got rank={s_np.ndim}"
                )
                log_fn(msg)
            continue

        # Slice (assumes last dim is output=1).
        try:
            if q_axis == 2:
                q10 = s_np[:, :, 0, 0]
                q50 = s_np[:, :, 1, 0]
                q90 = s_np[:, :, 2, 0]
            elif q_axis == 1:
                q10 = s_np[:, 0, :, 0]
                q50 = s_np[:, 1, :, 0]
                q90 = s_np[:, 2, :, 0]
            elif q_axis == 3:
                # Rare layout: (B,H,1,Q)
                q10 = s_np[:, :, 0, 0]
                q50 = s_np[:, :, 0, 1]
                q90 = s_np[:, :, 0, 2]
            else:
                if verbose:
                    msg = (
                        "  unsupported q_axis="
                        f"{q_axis}; skipping"
                    )
                    log_fn(msg)
                continue
        except Exception as e:
            if verbose:
                msg = (
                    "  slicing failed: "
                    f"{type(e).__name__}: {e}"
                )
                log_fn(msg)
            continue

        cr = _crossing_rates(q10, q50, q90)
        summary["crossings"].append(cr)

        if verbose:
            msg = (
                "  crossing (q10>q50, q50>q90, q10>q90): "
                f"{cr}"
            )
            log_fn(msg)

    if verbose:
        log_fn("\ndebug_val_interval: done")

    return summary


def debug_tensor_interval(
    y_true: Any,
    s_q: Any,
    n_q: int = 3,
    name: str = "RAW",
    default_q_axis: int = 2,
    verbose: int = 1,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """
    Debug quantile crossing for pre-collected tensors/arrays.
    """
    y_np = _to_numpy(y_true)
    s_np = _to_numpy(s_q)

    out: Dict[str, Any] = {
        "name": str(name),
        "y_shape": tuple(y_np.shape),
        "s_shape": tuple(s_np.shape),
        "q_axis": None,
        "crossing": None,
    }

    if verbose:
        msg = (
            f"\n[{name}] "
            f"y_true shape={y_np.shape} "
            f"s_q shape={s_np.shape}"
        )
        log_fn(msg)

    if s_np.ndim != 4:
        if verbose:
            msg = (
                f"[{name}] expected rank-4 s_q; "
                f"got rank={s_np.ndim}"
            )
            log_fn(msg)
        return out

    # Infer q axis using static shape only.
    q_axis = None
    if s_np.shape[2] == n_q:
        q_axis = 2
    elif s_np.shape[1] == n_q:
        q_axis = 1
    elif s_np.shape[3] == n_q:
        q_axis = 3
    else:
        q_axis = int(default_q_axis)

    out["q_axis"] = int(q_axis)

    if verbose:
        log_fn(f"[{name}] inferred q_axis={q_axis}")

    if int(n_q) < 3:
        if verbose:
            msg = (
                f"[{name}] n_q={n_q} < 3; "
                "crossing checks need >= 3."
            )
            log_fn(msg)
        return out

    try:
        if q_axis == 2:
            q10 = s_np[:, :, 0, 0]
            q50 = s_np[:, :, 1, 0]
            q90 = s_np[:, :, 2, 0]
        elif q_axis == 1:
            q10 = s_np[:, 0, :, 0]
            q50 = s_np[:, 1, :, 0]
            q90 = s_np[:, 2, :, 0]
        elif q_axis == 3:
            q10 = s_np[:, :, 0, 0]
            q50 = s_np[:, :, 0, 1]
            q90 = s_np[:, :, 0, 2]
        else:
            if verbose:
                msg = (
                    f"[{name}] unsupported q_axis={q_axis}; "
                    "skipping."
                )
                log_fn(msg)
            return out
    except Exception as e:
        if verbose:
            msg = (
                f"[{name}] slicing failed: "
                f"{type(e).__name__}: {e}"
            )
            log_fn(msg)
        return out

    cr = _crossing_rates(q10, q50, q90)
    out["crossing"] = cr

    if verbose:
        msg = (
            f"[{name}] cross q10>q50: {cr[0]:.6f}\n"
            f"[{name}] cross q50>q90: {cr[1]:.6f}\n"
            f"[{name}] cross q10>q90: {cr[2]:.6f}"
        )
        log_fn(msg)

    return out

def _logs_to_py(logs, *, keep_none=True):
    out = {}
    for k, v in (logs or {}).items():
        if v is None:
            if keep_none:
                out[k] = None
            continue

        # tf.Tensor / tf.Variable
        if hasattr(v, "numpy"):
            v = v.numpy()

        # numpy scalar / array
        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
            else:
                v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()

        out[k] = v
    return out

def canonicalize_to_BHQO_using_ytrue(
        s_pred, y_true, q_values=(0.1,0.5,0.9)
    ):
    """
    s_pred: (B,?, ?,1) rank-4
    y_true: (B,H,1)
    Returns: s_pred_BHQO (B,H,Q,1)
    """

    q_values = list(q_values)
    med = int(min(range(len(
        q_values)), key=lambda i: abs(q_values[i]-0.5)))

    # Candidate A: assume BHQO already
    A = s_pred
    A_q50 = A[:, :, med, :]          # (B,H,1)

    # Candidate B: assume BQHO -> transpose to BHQO
    B = tf_transpose(s_pred, [0, 2, 1, 3])
    B_q50 = B[:, :, med, :]          # (B,H,1)

    # Score by MAE against y_true in model space
    maeA = tf_reduce_mean(tf_abs(A_q50 - y_true))
    maeB = tf_reduce_mean(tf_abs(B_q50 - y_true))

    out = tf_cond(maeB < maeA, lambda: B, lambda: A)

    # Enforce monotonic quantiles after canonicalization (prevents inverted bounds)
    out = tf_sort(out, axis=2)
    return out

def debug_quantile_crossing_np(
    s_pred: Any,
    n_q: int = 3,
    *,
    name: str = "subs_pred",
    verbose: int = 1,
    log_fn: Callable[[str], None] = print,
    try_repair: bool = False,
    repair_mode: str = "sort",
) -> Dict[str, Any]:
    """
    Debug quantile crossing rates for two common layouts.

    Checks both:
      A) (B,H,Q,1) => q_axis=2
      B) (B,Q,H,1) => q_axis=1

    Optionally suggests / applies a repair:
      - repair_mode="flip": reverse quantile order along Q axis
      - repair_mode="sort": enforce monotonic quantiles via sort
    """
    s_np = _to_numpy(s_pred)

    out: Dict[str, Any] = {
        "name": str(name),
        "shape": tuple(s_np.shape),
        "n_q": int(n_q),
        "layout_A": None,
        "layout_B": None,
        "best_layout": None,
        "best_score": None,
        "repaired": False,
        "repair_mode": None,
        "notes": [],
    }

    if verbose:
        log_fn(
            f"\n[debug_quantile_crossing_np] "
            f"name={name} shape={s_np.shape}"
        )

    if s_np.ndim != 4:
        out["notes"].append("expected rank-4 tensor")
        if verbose:
            log_fn(
                "[debug_quantile_crossing_np] "
                f"expected rank-4, got {s_np.ndim}"
            )
        return out

    if int(n_q) < 3:
        out["notes"].append("n_q < 3; need >=3")
        if verbose:
            log_fn(
                "[debug_quantile_crossing_np] "
                f"n_q={n_q} < 3; skipping"
            )
        return out

    # We assume O=1 (last dim) for this debug utility.
    if s_np.shape[-1] != 1:
        out["notes"].append("last dim != 1 (O != 1)")
        if verbose:
            log_fn(
                "[debug_quantile_crossing_np] "
                "expected last dim O=1"
            )

    def _rates_and_score(
        q10: np.ndarray,
        q50: np.ndarray,
        q90: np.ndarray,
    ) -> Tuple[Tuple[float, float, float], float]:
        rates = _crossing_rates(q10, q50, q90)
        score = float(rates[0] + rates[1] + rates[2])
        return rates, score

    # ---- Layout A: (B,H,Q,1) ----
    if s_np.shape[2] == n_q:
        try:
            q10 = s_np[:, :, 0, 0]
            q50 = s_np[:, :, 1, 0]
            q90 = s_np[:, :, 2, 0]
            rates, score = _rates_and_score(q10, q50, q90)
            out["layout_A"] = {
                "layout": "(B,H,Q,1)",
                "q_axis": 2,
                "rates": rates,
                "score": score,
            }
            if verbose:
                log_fn(
                    "layout (B,H,Q,1) crossing: "
                    f"{rates} score={score:.6f}"
                )
        except Exception as e:
            out["notes"].append(f"layout_A slicing failed: {e}")

    # ---- Layout B: (B,Q,H,1) ----
    if s_np.shape[1] == n_q:
        try:
            q10 = s_np[:, 0, :, 0]
            q50 = s_np[:, 1, :, 0]
            q90 = s_np[:, 2, :, 0]
            rates, score = _rates_and_score(q10, q50, q90)
            out["layout_B"] = {
                "layout": "(B,Q,H,1)",
                "q_axis": 1,
                "rates": rates,
                "score": score,
            }
            if verbose:
                log_fn(
                    "layout (B,Q,H,1) crossing: "
                    f"{rates} score={score:.6f}"
                )
        except Exception as e:
            out["notes"].append(f"layout_B slicing failed: {e}")

    # Pick best available layout by minimal score.
    cand = []
    if out["layout_A"] is not None:
        cand.append(("A", out["layout_A"]["score"]))
    if out["layout_B"] is not None:
        cand.append(("B", out["layout_B"]["score"]))

    if not cand:
        out["notes"].append("no matching Q axis found")
        if verbose:
            log_fn(
                "[debug_quantile_crossing_np] "
                "no axis matched n_q"
            )
        return out

    best_layout, best_score = min(cand, key=lambda x: x[1])
    out["best_layout"] = best_layout
    out["best_score"] = float(best_score)

    if verbose:
        log_fn(
            "[debug_quantile_crossing_np] "
            f"best_layout={best_layout} "
            f"best_score={best_score:.6f}"
        )

    # Optional repair suggestion / apply.
    if not try_repair:
        return out

    # Only repair when we can interpret canonical BHQO first.
    repaired = None
    repair_mode = str(repair_mode).lower().strip()

    if best_layout == "B":
        # Convert to BHQO first.
        repaired = np.transpose(s_np, (0, 2, 1, 3))
    else:
        repaired = s_np

    # Now repaired is BHQO (best guess).
    if repaired.shape[2] != n_q:
        out["notes"].append("repair: not BHQO after step")
        return out

    if repair_mode == "flip":
        repaired = repaired[:, :, ::-1, :]
        out["repaired"] = True
        out["repair_mode"] = "flip"
        if verbose:
            log_fn(
                "[debug_quantile_crossing_np] "
                "repair applied: flip Q axis"
            )

    elif repair_mode == "sort":
        # Enforce q10<=q50<=q90 pointwise.
        repaired = np.sort(repaired, axis=2)
        out["repaired"] = True
        out["repair_mode"] = "sort"
        if verbose:
            log_fn(
                "[debug_quantile_crossing_np] "
                "repair applied: sort on Q axis"
            )
    else:
        out["notes"].append(
            "repair_mode must be 'flip' or 'sort'"
        )
        return out

    # Re-score repaired tensor.
    try:
        q10 = repaired[:, :, 0, 0]
        q50 = repaired[:, :, 1, 0]
        q90 = repaired[:, :, 2, 0]
        rates, score = _rates_and_score(q10, q50, q90)
        out["after_repair"] = {
            "layout": "(B,H,Q,1)",
            "q_axis": 2,
            "rates": rates,
            "score": score,
        }
        if verbose:
            log_fn(
                "after repair crossing: "
                f"{rates} score={score:.6f}"
            )
    except Exception as e:
        out["notes"].append(f"after_repair failed: {e}")

    return out
