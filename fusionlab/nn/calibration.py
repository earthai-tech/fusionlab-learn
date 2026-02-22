import numpy as np

from .._optdeps import with_progress
from ..utils.validator import check_is_fitted
from . import KERAS_DEPS

tf_concat = KERAS_DEPS.concat
tf_stack = KERAS_DEPS.stack
tf_expand_dims = KERAS_DEPS.expand_dims
tf_convert = KERAS_DEPS.convert_to_tensor
tf_minimum = KERAS_DEPS.minimum
tf_maximum = KERAS_DEPS.maximum


def _as_rank4_bhqo(s):
    if len(s.shape) == 3:
        s = tf_expand_dims(s, axis=-1)
    rank = getattr(getattr(s, "shape", None), "rank", None)
    if rank is not None and rank != 4:
        raise ValueError(
            "Expected subs_pred rank=4 BHQO. "
            f"Got rank={rank!r}."
        )
    return s


def _interval_indices(q_values):
    if q_values is None:
        return 0, 1, 2

    q = np.asarray(q_values, dtype=float).ravel()
    if q.size < 3:
        raise ValueError(
            "Need >=3 quantiles for interval calibration."
        )

    lo_i = int(np.argmin(q))
    hi_i = int(np.argmax(q))

    med_i = int(np.argmin(np.abs(q - 0.5)))
    if abs(float(q[med_i]) - 0.5) > 1e-6:
        raise ValueError(
            "Interval calibration requires q=0.5 in "
            "q_values."
        )

    return lo_i, med_i, hi_i


def _split_interval_bhqo(s, *, q_values=None):
    s = _as_rank4_bhqo(s)

    lo_i, med_i, hi_i = _interval_indices(q_values)

    qn = getattr(s.shape, "__getitem__", None)
    if qn is not None:
        qn = s.shape[2]
        if qn is not None:
            qn = int(qn)
            if qn <= max(lo_i, med_i, hi_i):
                raise ValueError(
                    "Quantile axis too small for requested "
                    "indices. "
                    f"Q={qn} idx={(lo_i, med_i, hi_i)}"
                )

    lo = s[:, :, lo_i, :]
    med = s[:, :, med_i, :]
    hi = s[:, :, hi_i, :]
    return lo, med, hi


def resolve_crossing_keep_median(lo, med, hi):
    lo = tf_minimum(lo, med)
    hi = tf_maximum(hi, med)

    lo2 = tf_minimum(lo, hi)
    hi2 = tf_maximum(hi, lo2)

    return lo2, med, hi2


def interval_from_bhqo(
    s_pred_q,
    *,
    q_values=None,
    enforce_monotone=True,
):
    lo, med, hi = _split_interval_bhqo(
        s_pred_q,
        q_values=q_values,
    )
    if enforce_monotone:
        lo, med, hi = resolve_crossing_keep_median(
            lo,
            med,
            hi,
        )
    return lo, med, hi


def interval_coverage_np(y, lo, hi):
    y = np.asarray(y).reshape(-1)
    lo = np.asarray(lo).reshape(-1)
    hi = np.asarray(hi).reshape(-1)
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_sharpness_np(lo, hi):
    lo = np.asarray(lo).reshape(-1)
    hi = np.asarray(hi).reshape(-1)
    return float(np.mean(hi - lo))


def interval_crossing_np(lo, med, hi):
    lo = np.asarray(lo)
    med = np.asarray(med)
    hi = np.asarray(hi)

    c1 = float(np.mean(lo > med))
    c2 = float(np.mean(med > hi))
    c3 = float(np.mean(lo > hi))
    return c1, c2, c3


class IntervalCalibrator:
    def __init__(self, target=0.80, max_iter=30, tol=1e-3):
        self.target = float(target)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def _coverage(self, y, lo, hi):
        return interval_coverage_np(y, lo, hi)

    def _fit_one(self, y, qlo, qmed, qhi):
        y = y.reshape(-1)
        lo = qlo.reshape(-1)
        md = qmed.reshape(-1)
        hi = qhi.reshape(-1)

        cov = self._coverage(y, lo, hi)
        if cov >= self.target - self.tol:
            return 1.0

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
        y_true = np.asarray(y_true)
        q_lo = np.asarray(q_lo)
        q_med = np.asarray(q_med)
        q_hi = np.asarray(q_hi)

        if y_true.ndim == 3 and y_true.shape[-1] == 1:
            y_true = y_true[..., 0]
            q_lo = q_lo[..., 0]
            q_med = q_med[..., 0]
            q_hi = q_hi[..., 0]

        h = int(q_med.shape[1])

        indices = with_progress(
            range(h),
            desc="Fitting interval factors per horizon",
            ascii=True,
            leave=False,
        )

        fs = []
        for i in indices:
            fs.append(
                self._fit_one(
                    y_true[:, i],
                    q_lo[:, i],
                    q_med[:, i],
                    q_hi[:, i],
                )
            )

        self.factors_ = np.asarray(fs, dtype=np.float32)
        return self

    def transform(self, q_lo, q_med, q_hi):
        check_is_fitted(self, attributes=["factors_"])

        q_lo = np.asarray(q_lo)
        q_med = np.asarray(q_med)
        q_hi = np.asarray(q_hi)

        squeeze = False
        if q_med.ndim == 3 and q_med.shape[-1] == 1:
            squeeze = True
            q_lo = q_lo[..., 0]
            q_med = q_med[..., 0]
            q_hi = q_hi[..., 0]

        fs = self.factors_[None, :]
        lo_c = q_med - fs * (q_med - q_lo)
        hi_c = q_med + fs * (q_hi - q_med)

        if squeeze:
            lo_c = lo_c[..., None]
            hi_c = hi_c[..., None]

        return lo_c, hi_c


def _extract_subs_pred(model, out):
    if not isinstance(out, dict):
        raise TypeError(
            "Expected `out` to be a dict. "
            f"Got type={type(out)!r}."
        )

    if ("subs_pred" in out) and ("gwl_pred" in out):
        return out["subs_pred"]

    if "data_final" in out: 
        s_pred, _ = model.split_data_predictions(out["data_final"])
        return s_pred

    raise KeyError(
        "Unsupported model output keys. Expected "
        "{'subs_pred','gwl_pred'} or {'data_final'}. "
        f"Got keys={list(out.keys())!r}."
    )


def fit_interval_calibrator_on_val(
    model,
    ds_val,
    target=0.80,
    log_fn=None,
    q_values=None,
    **tqdm_kws,
):
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

        s_pred = _as_rank4_bhqo(s_pred)

        lo, med, hi = interval_from_bhqo(
            s_pred,
            q_values=q_values,
            enforce_monotone=True,
        )

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


def apply_calibrator_to_subs(
    cal,
    s_pred_q,
    *,
    q_values=None,
):
    s_pred_q = _as_rank4_bhqo(s_pred_q)

    lo, med, hi = interval_from_bhqo(
        s_pred_q,
        q_values=q_values,
        enforce_monotone=True,
    )

    lo_c, hi_c = cal.transform(lo, med, hi)

    s_cal = tf_stack(
        [lo_c[..., 0], med[..., 0], hi_c[..., 0]],
        axis=2,
    )[..., None]

    return s_cal
