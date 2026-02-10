
import numpy as np

from .._optdeps import with_progress 
from ..utils.validator import check_is_fitted 
from ._shapes import (
    canonicalize_to_BHQO_using_ytrue,
)
from . import KERAS_DEPS

tf_concat=KERAS_DEPS.concat 
tf_stack = KERAS_DEPS.stack 
tf_expand_dims = KERAS_DEPS.expand_dims 
tf_convert = KERAS_DEPS.convert_to_tensor

class IntervalCalibrator:
    r"""
    Horizon-wise symmetric interval scaling to reach a target
    empirical coverage.

    This calibrator rescales the lower and upper quantiles around the
    median so that the central interval (by default [0.1, 0.9])
    achieves a desired coverage on validation data. The scaling is
    performed independently for each forecast horizon.

    The scale factor :math:`f \ge 1` is found via bisection so that
    for each horizon :math:`h`,

    .. math::

        \tilde q^{(h)}_{lo} = q^{(h)}_{50}
            - f\\,(q^{(h)}_{50} - q^{(h)}_{10}) \\
        \tilde q^{(h)}_{hi} = q^{(h)}_{50}
            + f\\,(q^{(h)}_{90} - q^{(h)}_{50})

    where the median :math:`q^{(h)}_{50}` is preserved.

    Parameters
    ----------
    target : float, default=0.80
        Desired empirical coverage for the central interval (e.g.,
        q10–q90).
    max_iter : int, default=30
        Maximum bisection iterations per horizon.
    tol : float, default=1e-3
        Absolute tolerance on coverage for early stopping.

    Attributes
    ----------
    factors_ : ndarray of shape (H,)
        Learned scale factor per horizon. Populated after
        :meth:`fit`.

    Notes
    -----
    * Scaling is symmetric about the median; dispersion increases but
      the central tendency remains unchanged.
    * Factors are learned on the validation split and then applied to
      test or deployment predictions of the same model.
    * Inputs are expected to be shaped either ``(N, H, 1)`` or
      ``(N, H)``; the final singleton dimension is optional.

    Examples
    --------
    >>> cal = IntervalCalibrator(target=0.80)
    >>> cal.fit(y_val, q10_val, q50_val, q90_val)
    >>> q10_t, q90_t = cal.transform(q10_test, q50_test, q90_test)
    """

    def __init__(self, target=0.80, max_iter=30, tol=1e-3):
        self.target = float(target)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        

    @staticmethod
    def _coverage(y, lo, hi):
        return np.mean((y >= lo) & (y <= hi))

    def _fit_one(self, y, qlo, qmed, qhi):
        # Flatten batch dims; keep scalar factor per horizon
        y  = y.reshape(-1)
        lo = qlo.reshape(-1)
        md = qmed.reshape(-1)
        hi = qhi.reshape(-1)

        # if already >= target, return f=1
        cov = self._coverage(y, lo, hi)
        if cov >= self.target - self.tol:
            return 1.0

        # bisection on f in [1, fmax]
        f_lo, f_hi = 1.0, 5.0
        for _ in range(self.max_iter):
            f_mid = 0.5 * (f_lo + f_hi)
            lo_mid = md - f_mid * (md - lo)
            hi_mid = md + f_mid * (hi - md)
            cov_mid = self._coverage(y, lo_mid, hi_mid)
            if cov_mid < self.target:
                f_lo = f_mid
            else:
                f_hi = f_mid
            if abs(cov_mid - self.target) < self.tol:
                break
        return f_hi

    def fit(self, y_true, q_lo, q_med, q_hi):
        """
        Fit per-horizon scale factors on validation data.

        Parameters
        ----------
        y_true : array-like of shape (N, H, 1) or (N, H)
            Ground-truth targets for subsidence.
        q_lo : array-like of shape (N, H, 1) or (N, H)
            Lower quantile (e.g., q10).
        q_med : array-like of shape (N, H, 1) or (N, H)
            Median quantile (q50).
        q_hi : array-like of shape (N, H, 1) or (N, H)
            Upper quantile (e.g., q90).

        Returns
        -------
        self : IntervalCalibrator
            The fitted instance with ``factors_`` set.

        Notes
        -----
        The method flattens batch dimensions and solves a scalar
        bisection per horizon to hit the requested coverage within
        ``tol`` or until ``max_iter`` is reached.
        """
        # self.factors_ = None  # shape: (H,)
        y_true = np.array(y_true)
        q_lo   = np.array(q_lo)
        q_med  = np.array(q_med)
        q_hi   = np.array(q_hi)

        if y_true.ndim == 3 and y_true.shape[-1] == 1:
            y_true = y_true[..., 0]
            q_lo   = q_lo[..., 0]
            q_med  = q_med[..., 0]
            q_hi   = q_hi[..., 0]

        H = q_med.shape[1]

        # Optionally wrap the horizon loop with tqdm for a progress bar
        indices = range(H)
        indices = with_progress(
            indices,
            desc="Fitting interval factors per horizon",
            ascii=True,
            leave=False,
        )
        fs = []
        for h in indices:
            fs.append(
                self._fit_one(
                    y_true[:, h],
                    q_lo[:, h],
                    q_med[:, h],
                    q_hi[:, h],
                )
            )

        self.factors_ = np.array(fs, dtype=np.float32)
        return self

    def transform(self, q_lo, q_med, q_hi):
        """
        Apply learned factors to new predictions.

        Parameters
        ----------
        q_lo : array-like of shape (N, H, 1) or (N, H)
            Lower quantile to be widened.
        q_med : array-like of shape (N, H, 1) or (N, H)
            Median (remains unchanged).
        q_hi : array-like of shape (N, H, 1) or (N, H)
            Upper quantile to be widened.

        Returns
        -------
        lo_c : ndarray of shape like ``q_lo``
            Calibrated lower bound.
        hi_c : ndarray of shape like ``q_hi``
            Calibrated upper bound.

        Raises
        ------
        AssertionError
            If ``fit`` has not been called and ``factors_`` is
            ``None``.

        Notes
        -----
        * The median is preserved exactly.
        * The output keeps the input dimensionality (``(N, H)`` or
          ``(N, H, 1)``).
        """
        check_is_fitted (self, attributes =['factors_'])
 
        q_lo = np.array(q_lo); q_med = np.array(q_med); q_hi = np.array(q_hi)
        squeeze = False
        if q_med.ndim == 3 and q_med.shape[-1] == 1:
            squeeze = True
            q_lo  = q_lo[..., 0]; q_med = q_med[..., 0]; q_hi = q_hi[..., 0]

        fs = self.factors_[None, :]  # (1,H)
        lo_c = q_med - fs * (q_med - q_lo)
        hi_c = q_med + fs * (q_hi  - q_med)

        if squeeze:
            lo_c = lo_c[..., None]; hi_c = hi_c[..., None]
        return lo_c, hi_c

def _extract_subs_pred(model, out):
    r"""
    Extract subsidence predictions from a model output.

    Supports both the new GeoPrior output dict and the legacy
    ``data_final`` path used by older checkpoints.

    Parameters
    ----------
    model : object
        Model instance. If the output is legacy, the model must
        implement ``split_data_predictions(data_final)``.

    out : dict
        Output dict returned by ``model(x, training=False)``.

    Returns
    -------
    subs_pred : Tensor
        Subsidence prediction tensor in model space.

        Expected shapes:

        - Point mode: ``(B, H, 1)``
        - Quantile mode: ``(B, H, Q, 1)``

    Raises
    ------
    TypeError
        If ``out`` is not a dict.
    KeyError
        If required keys are missing.

    Notes
    -----
    New path:
    ``out["subs_pred"]`` is used when available.

    Legacy path:
    ``out["data_final"]`` is split via
    ``model.split_data_predictions``.

    Examples
    --------
    >>> out = model(x, training=False)
    >>> s_pred = _extract_subs_pred(model, out)
    """
    if not isinstance(out, dict):
        raise TypeError(
            "Expected `out` to be a dict. "
            f"Got type={type(out)!r}."
        )

    if ("subs_pred" in out) and ("gwl_pred" in out):
        return out["subs_pred"]

    if "data_final" in out:
        s_pred, _ = model.split_data_predictions(
            out["data_final"],
        )
        return s_pred

    raise KeyError(
        "Unsupported model output keys. Expected "
        "{'subs_pred','gwl_pred'} or {'data_final'}. "
        f"Got keys={list(out.keys())!r}."
    )

def _stack_subs_quantiles(s_pred_q):
    r"""
    Extract (lo, med, hi) from subsidence quantiles.

    Parameters
    ----------
    s_pred_q : Tensor or np.ndarray
        Subsidence quantiles shaped ``(B, H, Q)`` or
        ``(B, H, Q, 1)``. The function assumes the quantile
        axis order is ``[q10, q50, q90]``.

    Returns
    -------
    lo : Tensor
        Lower quantile shaped ``(B, H, 1)``.
    med : Tensor
        Median quantile shaped ``(B, H, 1)``.
    hi : Tensor
        Upper quantile shaped ``(B, H, 1)``.

    Raises
    ------
    ValueError
        If ``s_pred_q`` does not look like a quantile tensor.

    Notes
    -----
    This helper is used by interval calibration and expects
    three quantiles (Q=3).

    Examples
    --------
    >>> lo, med, hi = _stack_subs_quantiles(s_pred_q)
    """
    if len(s_pred_q.shape) == 3:
        s_pred_q = tf_expand_dims(  # -> (B,H,Q,1)
            s_pred_q,
            axis=-1,
        )

    rank = getattr(getattr(s_pred_q, "shape", None), "rank", None)
    if rank is not None and rank != 4:
        raise ValueError(
            "Expected rank-4 quantile tensor. "
            f"Got rank={rank!r}."
        )

    q_dim = getattr(s_pred_q.shape, "__getitem__", None)
    if q_dim is not None:
        qn = s_pred_q.shape[2]
        if (qn is not None) and (int(qn) < 3):
            raise ValueError(
                "Need at least 3 quantiles on axis=2. "
                f"Got Q={int(qn)!r}."
            )

    lo = s_pred_q[:, :, 0, :]
    med = s_pred_q[:, :, 1, :]
    hi = s_pred_q[:, :, 2, :]
    return lo, med, hi

def fit_interval_calibrator_on_val(
    model,
    ds_val,
    target=0.80,
    log_fn=None,
    q_values =None, 
    **tqdm_kws,
):
    r"""
    Fit a horizon-wise symmetric interval calibrator on a
    validation dataset.

    This routine runs the model on ``ds_val``, extracts the
    subsidence quantiles (q10, q50, q90), and learns per-horizon
    scale factors so the central interval reaches ``target``
    empirical coverage.

    The function supports both output formats:

    - New: ``{"subs_pred": ..., "gwl_pred": ...}``
    - Legacy: ``{"data_final": ...}`` plus
      ``model.split_data_predictions``

    Parameters
    ----------
    model : tf.keras.Model
        Trained model used for inference on ``ds_val``.
    ds_val : tf.data.Dataset
        Validation dataset yielding ``(x, y)`` where
        ``y["subs_pred"]`` is shaped ``(B, H, 1)``.
    target : float, default=0.80
        Desired coverage for the central interval.
    log_fn : callable or None, optional
        Optional logger sink for progress output.
    **tqdm_kws : dict
        Extra keyword arguments forwarded to ``with_progress``.

    Returns
    -------
    cal : IntervalCalibrator
        Fitted calibrator.

    Raises
    ------
    ValueError
        If the model does not output quantiles.

    Notes
    -----
    Only the subsidence head is calibrated.

    Examples
    --------
    >>> cal = fit_interval_calibrator_on_val(
    ...     model,
    ...     ds_val,
    ...     target=0.8,
    ... )
    """
    cal = IntervalCalibrator(target=target)

    y_true_list = []
    lo_list, med_list, hi_list = [], [], []

    iterator = with_progress(
        ds_val,
        desc="Calibrating intervals on val",
        ascii=True,
        leave=False,
        log_fn=log_fn,
        **tqdm_kws,
    )

    for x, y in iterator:
        out = model(x, training=False)

        s_pred = _extract_subs_pred(model, out)

        # Disambiguate BHQO vs BQHO when H == Q (e.g. 3 and 3).
        if q_values is not None:
            s_pred = canonicalize_to_BHQO_using_ytrue(
                s_pred,
                y["subs_pred"],
                q_values=(0.1, 0.5, 0.9),
            )
 
        rank = getattr(getattr(s_pred, "shape", None), "rank", None)
        if rank is not None and rank != 4:
            raise ValueError(
                "Interval calibration requires quantiles. "
                "Expected subs_pred shape (B,H,Q,1). "
                f"Got rank={rank!r}."
            )

        lo, med, hi = _stack_subs_quantiles(s_pred)

        y_true_list.append(y["subs_pred"])
        lo_list.append(lo)
        med_list.append(med)
        hi_list.append(hi)

    y_true = tf_concat(y_true_list, axis=0).numpy()
    q_lo = tf_concat(lo_list, axis=0).numpy()
    q_med = tf_concat(med_list, axis=0).numpy()
    q_hi = tf_concat(hi_list, axis=0).numpy()

    cal.fit(y_true, q_lo, q_med, q_hi)
    return cal

def _fit_interval_calibrator_on_val(
        model, ds_val, target=0.80, 
        log_fn =None, **tqdm_kws
    ):
    """
    Fit a horizon-wise symmetric interval calibrator on a validation
    dataset.

    The function runs the provided Keras model on the validation
    dataset, extracts subsidence quantiles (q10, q50, q90) using the
    model's ``split_data_predictions`` helper, and learns scale
    factors that make the central interval reach ``target`` coverage.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model that, when called as ``model(x, training=False)``,
        returns a dict with key ``"data_final"``. The model must also
        implement ``split_data_predictions(tensor) -> (s_q, h_q)``,
        where ``s_q`` has shape ``(B, H, Q, 1)`` with ``Q = 3``.
    ds_val : tf.data.Dataset
        Validation dataset yielding ``(x, y)`` pairs. The target dict
        ``y`` must contain key ``"subs_pred"`` of shape ``(B, H, 1)``.
    target : float, default=0.80
        Desired coverage for the central interval.
    log_fn : callable or None, optional
        If provided, tqdm's progress bar output is redirected into
        this logger instead of the terminal. The logger must accept
        a single string argument.
    Returns
    -------
    cal : IntervalCalibrator
        Calibrator fitted on the validation split.

    Notes
    -----
    * Only the **subsidence** head is calibrated. GWL or other heads
      are not modified by this routine.
    * The function concatenates all validation batches along the
      batch axis before fitting.

    Examples
    --------
    >>> cal = fit_interval_calibrator_on_val(model, ds_val, target=0.8)
    >>> # Later, apply to test subsidence quantiles:
    >>> # s_q_test = model.split_data_predictions(...)[0]
    >>> # s_q_test_cal = apply_calibrator_to_subs(cal, s_q_test)
    """

    cal = IntervalCalibrator(target=target)

    y_true_list, lo_list, med_list, hi_list = [], [], [], []

    # Wrap with progress bar if available  otherwise use plain iterator
    iterator = with_progress(
        ds_val,
        desc="Calibrating intervals on val",
        ascii=True,
        leave=False,
        log_fn= log_fn, 
        **tqdm_kws
    )

    for x, y in iterator:  # y needs 'subs_pred' with shape (B,H,1)
        out = model(x, training=False)
        s_pred_q, _ = model.split_data_predictions(out["data_final"])  # (B,H,Q,1), (B,H,Q,1)
        lo, med, hi = _stack_subs_quantiles(s_pred_q)

        y_true_list.append(y["subs_pred"])
        lo_list.append(lo)
        med_list.append(med)
        hi_list.append(hi)

    y_true = tf_concat(y_true_list, axis=0).numpy()
    q_lo   = tf_concat(lo_list,   axis=0).numpy()
    q_med  = tf_concat(med_list,  axis=0).numpy()
    q_hi   = tf_concat(hi_list,   axis=0).numpy()

    cal.fit(y_true, q_lo, q_med, q_hi)  # learns factors_ per horizon
    return cal

def apply_calibrator_to_subs(cal, s_pred_q):
    """
    Apply a fitted interval calibrator to **subsidence** quantiles.

    Parameters
    ----------
    cal : IntervalCalibrator
        Fitted calibrator with ``factors_`` of shape ``(H,)``.
    s_pred_q : tf.Tensor or np.ndarray
        Subsidence quantiles with shape ``(B, H, Q, 1)`` where
        ``Q = 3`` corresponds to ``[q10, q50, q90]``.

    Returns
    -------
    s_cal : tf.Tensor
        Calibrated subsidence quantiles with the **same** shape as
        ``s_pred_q`` (``(B, H, 3, 1)``). The median (q50) is
        preserved exactly; q10/q90 are symmetrically scaled about the
        median using ``cal.factors_``.

    Notes
    -----
    * Only the subsidence head is transformed. Apply analogous logic
      if you need to calibrate other heads.
    * This function is typically used **before** formatting outputs
      to a pandas DataFrame, so that all downstream evaluation and
      visualization reflect calibrated intervals.

    Examples
    --------
    >>> s_q_cal = apply_calibrator_to_subs(cal, s_q_test)
    """

    lo, med, hi = _stack_subs_quantiles(s_pred_q)      # (B,H,1) each
    lo_c, hi_c  = cal.transform(lo, med, hi)           # (B,H,1) each
    s_cal = tf_stack([lo_c[...,0], med[...,0], hi_c[...,0]], axis=2)[..., None]
    return s_cal
