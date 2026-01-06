## -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author : LKouadio <etanoyau@gmail.com>

from __future__ import annotations

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


# def _as_BHO(y_true, y_pred=None):
#     y = tf_convert_to_tensor(y_true)

#     # (B,H) -> (B,H,1)
#     if y.shape.rank == 2:
#         y = tf_expand_dims(y, axis=-1)

#     # If Keras expanded a trailing singleton: (B,H,O,1) -> (B,H,O)
#     if y.shape.rank == 4 and y.shape[-1] == 1:
#         y = tf_squeeze(y, axis=-1)

#     # If something upstream accidentally tiled targets: (B,H,Q,O) -> (B,H,O)
#     if (y.shape.rank == 4) and (y_pred is not None) and (getattr(y_pred.shape, "rank", None) == 4):
#         # If axis=2 matches Q, drop it
#         y = y[:, :, 0, :]

#     return tf_cast(y, tf_float32)