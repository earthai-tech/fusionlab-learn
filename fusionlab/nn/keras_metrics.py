# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations

import numpy as np 

from . import KERAS_DEPS, dependency_message

Layer = KERAS_DEPS.Layer
Loss = KERAS_DEPS.Loss
Model = KERAS_DEPS.Model
Sequential = KERAS_DEPS.Sequential
Metric = KERAS_DEPS.Metric

tf_divide_no_nan = KERAS_DEPS.divide_no_nan
tf_reduce_sum = KERAS_DEPS.reduce_sum
tf_cast = KERAS_DEPS.cast
tf_size = KERAS_DEPS.size
tf_float32 = KERAS_DEPS.float32
tf_abs = KERAS_DEPS.abs
tf_gather = KERAS_DEPS.gather
tf_reduce_mean = KERAS_DEPS.reduce_mean
tf_shape = KERAS_DEPS.shape
tf_expand_dims = KERAS_DEPS.expand_dims
tf_broadcast_to = KERAS_DEPS.broadcast_to
tf_shape = KERAS_DEPS.shape
register_keras_serializable=KERAS_DEPS.register_keras_serializable

DEP_MSG = dependency_message("nn.keras_metrics")


def _normalize_index(idx, dim):
    # idx can be python int; dim can be python int or a scalar tensor
    if isinstance(idx, int) and idx >= 0:
        return idx
    if isinstance(dim, int):
        return dim + idx if isinstance(idx, int) and idx < 0 else idx
    # dim is dynamic tensor → return a tensor index
    if isinstance(idx, int) and idx < 0:
        return dim + idx
    return idx

def _split_interval(y_pred_interval, *, q_axis=None, lo_index=0, hi_index=-1):
    r"""
    Return (lo, hi) for coverage/sharpness.

    - If y_pred_interval is (lo, hi) or has last-dim==2 → use that.
    - If q_axis is not None → treat that axis as quantile axis and select
      lo_index/hi_index (defaults 0 and -1 for [0.1,0.5,0.9]).
    """
    if isinstance(y_pred_interval, (list, tuple)):
        lo, hi = y_pred_interval[0], y_pred_interval[1]
        return lo, hi

    t = y_pred_interval
    if q_axis is not None:
        qdim_static = t.shape[q_axis]
        qdim = qdim_static if qdim_static is not None else tf_shape(t)[q_axis]
        lo_idx = _normalize_index(lo_index, qdim)
        hi_idx = _normalize_index(hi_index, qdim)
        lo = tf_gather(t, lo_idx, axis=q_axis)
        hi = tf_gather(t, hi_idx, axis=q_axis)
        return lo, hi

    # fallback: assume last dim packs (lo, hi)
    lo, hi = t[..., 0], t[..., 1]
    return lo, hi


class CentralCoverage(Metric):
    r"""
    Empirical coverage of a central prediction interval.

    This metric computes the fraction of targets :math:`y` that fall
    inside a predicted central interval :math:`[q_{\alpha/2},\,
    q_{1-\alpha/2}]`. It supports optional sample weights.

    The estimator is

    .. math::

        \widehat{\mathrm{Cov}}_{1-\alpha}
        \;=\;
        \frac{\sum_i w_i \,\mathbb{1}
        \{ q_{\alpha/2,i} \le y_i \le q_{1-\alpha/2,i} \}}
        {\sum_i w_i},

    with :math:`w_i \equiv 1` in the unweighted case.

    Parameters
    ----------
    name : str, default='coverage'
        Name of the metric in Keras history.
    dtype : Any, default=tf.float32
        Internal dtype for accumulators.

    Notes
    -----
    * `y_pred_interval` can be passed as a pair `(q_lo, q_hi)` or as a
      single tensor with last dimension 2: `[..., 2] = (lo, hi)`.
    * `sample_weight`, when provided, is broadcast against the hit mask.

    Examples
    --------
    >>> cov = CentralCoverage()
    >>> # Unweighted coverage over a batch
    >>> cov.update_state(y_true, (q_lo, q_hi))
    >>> float(cov.result().numpy())  # coverage in [0, 1]
    """

    def __init__(self, name: str = "coverage", dtype=tf_float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(
            "total", initializer="zeros", dtype=self.dtype
        )
        self.hits = self.add_weight(
            "hits", initializer="zeros", dtype=self.dtype
        )

    def update_state(self, y_true, y_pred_interval, sample_weight=None):
        r"""
        Update internal counts for the current batch.

        Parameters
        ----------
        y_true : Tensor
            Ground-truth targets, arbitrary shape.
        y_pred_interval : Tuple[Tensor, Tensor] | Tensor
            Either `(q_lo, q_hi)` or a tensor with `[..., 2]`.
        sample_weight : Tensor | None
            Optional non-negative weights broadcastable to `y_true`.
        """
        y = tf_cast(y_true, self.dtype)
        lo, hi = _split_interval(y_pred_interval)
        
        # Make ranks compatible: if y has one extra trailing dim (e.g., y.shape=(N,1))
        if ( 
                y.shape.rank is not None and 
                lo.shape.rank is not None 
                and y.shape.rank == lo.shape.rank + 1
            ):
            lo = tf_expand_dims(lo, axis=-1)
            hi = tf_expand_dims(hi, axis=-1)
        
        # Broadcast lo/hi to y's shape to avoid (N,1) vs (N,) → (N,N) blowup
        lo = tf_broadcast_to(tf_cast(lo, self.dtype), tf_shape(y))
        hi = tf_broadcast_to(tf_cast(hi, self.dtype), tf_shape(y))
        
        hit = tf_cast((y >= lo) & (y <= hi), self.dtype)

        if sample_weight is not None:
            w = tf_cast(sample_weight, self.dtype)
            self.hits.assign_add(tf_reduce_sum(hit * w))
            self.total.assign_add(tf_reduce_sum(w))
        else:
            self.hits.assign_add(tf_reduce_sum(hit))
            self.total.assign_add(tf_cast(tf_size(y), self.dtype))

    def result(self):
        """Return the coverage fraction in [0, 1]."""
        return tf_divide_no_nan(self.hits, self.total)

    def reset_state(self):
        """Reset accumulators to zero."""
        self.hits.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        return cfg


class IntervalSharpness(Metric):
    r"""
    Mean (or weighted-mean) width of a central prediction interval.

    This metric summarizes the sharpness (narrowness) of a central
    interval by averaging its width :math:`q_{1-\alpha/2}-q_{\alpha/2}`.

    The estimator is

    .. math::

        \mathrm{Sharpness}
        \;=\;
        \frac{\sum_i w_i \,(q_{1-\alpha/2,i}-q_{\alpha/2,i})}
        {\sum_i w_i},

    with :math:`w_i \equiv 1` in the unweighted case.

    Parameters
    ----------
    name : str, default='sharpness'
        Name of the metric in Keras history.
    dtype : Any, default=tf.float32
        Internal dtype for accumulators.

    Notes
    -----
    * `y_true` is accepted for Keras API compatibility but not used.
    * `y_pred_interval` can be a pair `(q_lo, q_hi)` or a single
      tensor with last dimension 2.

    Examples
    --------
    >>> shp = IntervalSharpness()
    >>> shp.update_state(y_true, (q_lo, q_hi))
    >>> float(shp.result().numpy())  # average width (units of y)
    """

    def __init__(self, name: str = "sharpness", dtype=tf_float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.sumw = self.add_weight(
            "sumw", initializer="zeros", dtype=self.dtype
        )
        self.count = self.add_weight(
            "count", initializer="zeros", dtype=self.dtype
        )

    def update_state(self, y_true, y_pred_interval, sample_weight=None):
        r"""
        Update internal sums for the current batch.

        Parameters
        ----------
        y_true : Tensor
            Unused (kept for Keras signature compatibility).
        y_pred_interval : Tuple[Tensor, Tensor] | Tensor
            Either `(q_lo, q_hi)` or a tensor with `[..., 2]`.
        sample_weight : Tensor | None
            Optional non-negative weights broadcastable to the width.
        """
        lo, hi = _split_interval(y_pred_interval)
        
        # Align shapes as above
        y = y_true  # unused, but we align against y's shape if helpful
        if y is not None:
            if ( 
                    y.shape.rank is not None and 
                    lo.shape.rank is not None and 
                    y.shape.rank == lo.shape.rank + 1
                ):
                lo = tf_expand_dims(lo, axis=-1)
                hi = tf_expand_dims(hi, axis=-1)
            # For width aggregation, broadcasting to common shape is also fine:
            # but if you want 1:1 with targets, do:
            lo = tf_broadcast_to(tf_cast(lo, self.dtype), tf_shape(y))
            hi = tf_broadcast_to(tf_cast(hi, self.dtype), tf_shape(y))
        
        width = tf_abs(hi - lo)

        if sample_weight is not None:
            w = tf_cast(sample_weight, self.dtype)
            self.sumw.assign_add(tf_reduce_sum(width * w))
            self.count.assign_add(tf_reduce_sum(w))
        else:
            self.sumw.assign_add(tf_reduce_sum(width))
            self.count.assign_add(tf_cast(tf_size(width), self.dtype))

    def result(self):
        """Return the (weighted) average interval width."""
        return tf_divide_no_nan(self.sumw, self.count)

    def reset_state(self):
        """Reset accumulators to zero."""
        self.sumw.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        return cfg

def _to_py(x):
    """Best-effort cast to JSON-safe Python scalars."""
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, (np.generic,)):
        return x.item()
    try:
        return float(x)
    except Exception:
        try:
            return x.tolist()
        except Exception:
            return str(x)
    

def _infer_quantile_axis(t, n_q=3):
    """Pick an axis that (statically) looks like the quantile axis."""
    stat = t.shape
    cand = [i for i, d in enumerate(stat) if d == n_q]
    if cand:
        return cand[-1]  # prefer the last matching axis
    # Fallback: assume last dim packs (lo, hi) or (… , n_q) at runtime
    # If we follow the convention (B,H,Q,O), force Q-axis=2 for rank-4
    if getattr(t, "shape", None) is not None and t.shape.rank == 4:
        return 2
    return None

@register_keras_serializable(
    "fusionlab.nn.keras_metrics",name="coverage80_fn"
)
def coverage80_fn(y_true, y_pred):
    """
    Empirical coverage for the central [0.1, 0.9] predictive interval.

    This metric returns the fraction of targets that fall inside the
    model's 80% interval. It supports two prediction layouts:

    1) Packed interval: the last dimension encodes (lo, hi), i.e.
       ``y_pred.shape = (..., 2)``.
    2) Quantile axis: any axis of length 3 encodes (q10, q50, q90),
       e.g. ``(..., 3, ...)`` or ``(..., H, 3, O)``.

    The function automatically detects the quantile axis (if present)
    and extracts (lo, hi) as (q10, q90). Packed (lo, hi) inputs are
    also supported. Shapes are broadcast to ``y_true`` to avoid NxN
    cartesian products.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth targets. Can be any shape; commonly
        ``(batch, horizon, output_dim)`` or ``(batch, output_dim)``.
    y_pred : tf.Tensor
        Predictive intervals or quantiles. Either (..., 2) with
        (lo, hi) or has a 3-length quantile axis containing
        (q10, q50, q90) at any position.

    Returns
    -------
    tf.Tensor
        A scalar tensor (float32) with the empirical coverage in
        ``[0, 1]``.

    Notes
    -----
    * Coverage is computed element-wise over the full tensor and
      averaged globally.
    * If your model outputs additional dimensions (e.g., features or
      outputs), they are included in the element-wise average.
    * To compute per-horizon or per-output coverage, slice ``y_true``
      and ``y_pred`` before passing them to this metric.

    Examples
    --------
    >>> # Keras compile with quantile outputs (q10, q50, q90):
    >>> model.compile(
    ...     optimizer="adam",
    ...     loss=pinball_loss_for_quantiles,
    ...     metrics=[coverage80_fn]
    ... )

    See Also
    --------
    sharpness80_fn : Mean width of the 80% interval.
    make_coverage80 : Coverage metric with a fixed quantile axis.
    fit_interval_calibrator_on_val : Post-hoc calibration utility.
    """
    qax = _infer_quantile_axis(y_pred, n_q=3)
    if qax is None:
        # assume last-dim packed (lo, hi)
        lo, hi = _split_interval(y_pred_interval=y_pred, q_axis=None)
    else:
        lo, hi = _split_interval(y_pred_interval=y_pred, q_axis=qax,
                                 lo_index=0, hi_index=-1)

    # Align shapes to y_true
    y = tf_cast(y_true, tf_float32)
    if (y.shape.rank is not None and lo.shape.rank is not None
            and y.shape.rank == lo.shape.rank + 1):
        # common (N,1) vs (N,) mismatch
        lo = tf_expand_dims(lo, axis=-1)
        hi = tf_expand_dims(hi, axis=-1)
    lo = tf_broadcast_to(tf_cast(lo, tf_float32), tf_shape(y))
    hi = tf_broadcast_to(tf_cast(hi, tf_float32), tf_shape(y))

    hit = tf_cast((y >= lo) & (y <= hi), tf_float32)
    num = tf_reduce_sum(hit)
    den = tf_cast(tf_size(hit), tf_float32)
    return tf_divide_no_nan(num, den)

coverage80_fn.__name__ = "coverage80"

@register_keras_serializable(
    "fusionlab.nn.keras_metrics",name="sharpness80_fn"
)
def sharpness80_fn(y_true, y_pred):
    """
    Mean width of the central [0.1, 0.9] predictive interval.

    This metric reports the average interval width, i.e.
    ``E[hi - lo]``, for the model's 80% interval. It supports both
    packed (lo, hi) and a 3-length quantile axis (q10, q50, q90).
    Shapes are broadcast to align with ``y_true`` so that averaging
    is 1:1 with the targets.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth targets. Used only for shape alignment and
        broadcasting. Values are not used in the width computation.
    y_pred : tf.Tensor
        Predictive intervals or quantiles. Either (..., 2) with
        (lo, hi) or has a 3-length quantile axis containing
        (q10, q50, q90) at any position.

    Returns
    -------
    tf.Tensor
        A scalar tensor (float32) with the mean interval width.
        Units match the scale of ``y_pred``; if predictions are in
        model-scaled space, widths will be in that space unless
        inverse-transformed upstream.

    Notes
    -----
    * Lower values indicate sharper (narrower) predictive intervals,
      but sharpness should be interpreted together with coverage.
    * For calibrated intervals, apply calibration to the quantiles
      upstream and then compute sharpness.

    Examples
    --------
    >>> model.compile(
    ...     optimizer="adam",
    ...     loss=pinball_loss_for_quantiles,
    ...     metrics=[coverage80_fn, sharpness80_fn]
    ... )

    See Also
    --------
    coverage80_fn : Empirical coverage of the 80% interval.
    fit_interval_calibrator_on_val : Post-hoc calibration utility.
    """
    qax = _infer_quantile_axis(y_pred, n_q=3)
    if qax is None:
        lo, hi = _split_interval(y_pred_interval=y_pred, q_axis=None)
    else:
        lo, hi = _split_interval(y_pred_interval=y_pred, q_axis=qax,
                                 lo_index=0, hi_index=-1)

    y = y_true  # only for aligning shapes; not used in width math
    if y is not None:
        if (y.shape.rank is not None and lo.shape.rank is not None
                and y.shape.rank == lo.shape.rank + 1):
            lo = tf_expand_dims(lo, axis=-1)
            hi = tf_expand_dims(hi, axis=-1)
        lo = tf_broadcast_to(tf_cast(lo, tf_float32), tf_shape(y))
        hi = tf_broadcast_to(tf_cast(hi, tf_float32), tf_shape(y))

    width = tf_abs(hi - lo)
    return tf_reduce_mean(width)

sharpness80_fn.__name__ = "sharpness80"


def make_coverage80(q_axis):
    """
    Build a coverage-80% metric bound to a fixed quantile axis.

    Use this factory when your model's quantiles are always arranged
    along a known axis (length 3, encoding q10/q50/q90). Fixing the
    axis avoids runtime axis inference and yields a metric with a
    stable name (useful in CI and logs).

    Parameters
    ----------
    q_axis : int
        The axis index that contains the 3 quantiles
        (q10, q50, q90). Negative indexing is supported
        (e.g., ``-2``).

    Returns
    -------
    Callable
        A Keras-compatible metric function
        ``fn(y_true, y_pred) -> tf.Tensor`` that returns the scalar
        coverage in ``[0, 1]``. The function name is set to
        ``coverage80_qax{q_axis}``.

    Notes
    -----
    * This function expects a 3-length quantile axis. If your model
      emits packed (lo, hi), use :func:`coverage80_fn` instead.
    * The metric broadcasts predictions to the shape of ``y_true`` to
      avoid NxN blowups.

    Examples
    --------
    >>> cov_qm2 = make_coverage80(q_axis=-2)
    >>> model.compile(
    ...     optimizer="adam",
    ...     loss=pinball_loss_for_quantiles,
    ...     metrics=[cov_qm2]
    ... )

    See Also
    --------
    coverage80_fn : Axis-agnostic coverage metric.
    sharpness80_fn : Mean width for the 80% interval.
    """
    def fn(y_true, y_pred):
        lo, hi = _split_interval(y_pred_interval=y_pred, q_axis=q_axis,
                                 lo_index=0, hi_index=-1)
        y = tf_cast(y_true, tf_float32)
        if (y.shape.rank is not None and lo.shape.rank is not None
                and y.shape.rank == lo.shape.rank + 1):
            lo = tf_expand_dims(lo, axis=-1); hi = tf_expand_dims(hi, axis=-1)
        lo = tf_broadcast_to(tf_cast(lo, tf_float32), tf_shape(y))
        hi = tf_broadcast_to(tf_cast(hi, tf_float32), tf_shape(y))
        hit = tf_cast((y >= lo) & (y <= hi), tf_float32)
        return tf_divide_no_nan(tf_reduce_sum(hit),
                                tf_cast(tf_size(hit), tf_float32))
    fn.__name__ = f"coverage80_qax{q_axis}"
    return fn

@register_keras_serializable(
    "fusionlab.nn.keras_metrics",
    name="mae_q50_fn",
)
def mae_q50_fn(y_true, y_pred):
    r"""
    Mean absolute error using the q50 point forecast.

    This metric is designed for models that output either:

    * Quantiles in a dedicated axis of length 3:
      ``(q10, q50, q90)`` in this exact order.
    * A packed central interval in the last dimension:
      ``(..., 2)`` interpreted as ``(lo, hi)``.

    If a quantile axis is detected, the function extracts
    the median (q50) and computes the mean absolute error
    (MAE) against ``y_true``. If a packed interval is
    detected, the midpoint ``0.5 * (lo + hi)`` is used as
    a fallback point forecast. Otherwise, ``y_pred`` is
    used as-is.

    The MAE is computed as:

    .. math::

        \mathrm{MAE} \;=\;
        \frac{1}{N}\sum_{i=1}^{N}
        \lvert y_i - \hat{y}_i\rvert

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth targets.
    y_pred : tf.Tensor
        Model predictions, either a point forecast, a
        packed interval ``(..., 2)``, or quantiles with a
        3-length axis containing ``(q10, q50, q90)``.

    Returns
    -------
    tf.Tensor
        Scalar MAE aggregated over all elements.

    Notes
    -----
    * Quantile axis detection is delegated to
      :func:`~fusionlab.nn.keras_metrics`
      ``._infer_quantile_axis``.
    * If you want the progress-bar name to be exactly
      "mae", set ``mae_q50_fn.__name__ = "mae"`` before
      calling ``model.compile(...)``. Keras snapshots
      names at compile time.

    Examples
    --------
    >>> import tensorflow as tf
    >>> y = tf.zeros((2, 3, 1), dtype=tf.float32)
    >>> q10 = tf.zeros((2, 3, 1), dtype=tf.float32)
    >>> q50 = tf.ones((2, 3, 1), dtype=tf.float32)
    >>> q90 = 2.0 * tf.ones((2, 3, 1), dtype=tf.float32)
    >>> yp = tf.stack([q10, q50, q90], axis=2)
    >>> float(mae_q50_fn(y, yp).numpy()) > 0.0
    True

    See Also
    --------
    mse_q50_fn : Mean squared error for q50 point
        forecast.
    coverage80_fn : Empirical 80 percent interval
        coverage.
    sharpness80_fn : Mean width of 80 percent
        interval.

    References
    ----------
    .. [1] Koenker, R., and Bassett, G. (1978).
       "Regression Quantiles". Econometrica, 46(1),
       33-50.
    """
    # Detect a quantile axis of length 3, if any.
    qax = _infer_quantile_axis(y_pred, n_q=3)

    # Extract the q50 point forecast when available.
    yp = y_pred
    if qax is not None:
        yp = tf_gather(yp, 1, axis=qax)
    else:
        # Packed (lo, hi) interval -> midpoint fallback.
        if yp.shape.rank is not None and yp.shape[-1] == 2:
            yp = 0.5 * (yp[..., 0] + yp[..., 1])

    # Cast to a stable dtype for metric computations.
    y = tf_cast(y_true, tf_float32)
    yp = tf_cast(yp, tf_float32)

    # If targets have a trailing singleton dim, align yp.
    if (
        y.shape.rank is not None
        and yp.shape.rank is not None
        and y.shape.rank == yp.shape.rank + 1
    ):
        yp = tf_expand_dims(yp, axis=-1)

    # Broadcast prediction to target shape.
    yp = tf_broadcast_to(yp, tf_shape(y))

    return tf_reduce_mean(tf_abs(y - yp))


mae_q50_fn.__name__ = "mae"


@register_keras_serializable(
    "fusionlab.nn.keras_metrics",
    name="mse_q50_fn",
)
def mse_q50_fn(y_true, y_pred):
    r"""
    Mean squared error using the q50 point forecast.

    This metric mirrors :func:`mae_q50_fn`, but uses the
    mean squared error (MSE):

    .. math::

        \mathrm{MSE} \;=\;
        \frac{1}{N}\sum_{i=1}^{N}
        (y_i - \hat{y}_i)^2

    The point forecast :math:`\hat{y}` is derived from
    ``y_pred`` as follows:

    * If a 3-length quantile axis is detected, extract
      q50.
    * Else if ``y_pred.shape[-1] == 2``, use midpoint of
      ``(lo, hi)``.
    * Otherwise, treat ``y_pred`` as a point forecast.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth targets.
    y_pred : tf.Tensor
        Model predictions, either a point forecast, a
        packed interval ``(..., 2)``, or quantiles with a
        3-length axis containing ``(q10, q50, q90)``.

    Returns
    -------
    tf.Tensor
        Scalar MSE aggregated over all elements.

    Notes
    -----
    For naming in Keras logs, see the note in
    :func:`mae_q50_fn`.

    Examples
    --------
    >>> import tensorflow as tf
    >>> y = tf.zeros((1, 2, 1), dtype=tf.float32)
    >>> lo = tf.zeros((1, 2, 1), dtype=tf.float32)
    >>> hi = tf.ones((1, 2, 1), dtype=tf.float32)
    >>> yp = tf.concat([lo, hi], axis=-1)
    >>> float(mse_q50_fn(y, yp).numpy()) >= 0.0
    True

    See Also
    --------
    mae_q50_fn : Mean absolute error for q50 point
        forecast.
    coverage80_fn : Empirical 80 percent interval
        coverage.
    sharpness80_fn : Mean width of 80 percent
        interval.

    References
    ----------
    .. [1] Koenker, R., and Bassett, G. (1978).
       "Regression Quantiles". Econometrica, 46(1),
       33-50.
    """
    # Detect a quantile axis of length 3, if any.
    qax = _infer_quantile_axis(y_pred, n_q=3)

    # Extract the q50 point forecast when available.
    yp = y_pred
    if qax is not None:
        yp = tf_gather(yp, 1, axis=qax)
    else:
        # Packed (lo, hi) interval -> midpoint fallback.
        if yp.shape.rank is not None and yp.shape[-1] == 2:
            yp = 0.5 * (yp[..., 0] + yp[..., 1])

    # Cast to a stable dtype for metric computations.
    y = tf_cast(y_true, tf_float32)
    yp = tf_cast(yp, tf_float32)

    # If targets have a trailing singleton dim, align yp.
    if (
        y.shape.rank is not None
        and yp.shape.rank is not None
        and y.shape.rank == yp.shape.rank + 1
    ):
        yp = tf_expand_dims(yp, axis=-1)

    # Broadcast prediction to target shape.
    yp = tf_broadcast_to(yp, tf_shape(y))

    d = y - yp
    return tf_reduce_mean(d * d)


mse_q50_fn.__name__ = "mse"

