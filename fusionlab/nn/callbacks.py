# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Utility callbacks for training and tuning.

"""

from __future__ import annotations

from typing import ( 
    Iterable, Optional, Set, Any, 
    Callable, Mapping, Sequence, 
    Union, Dict, Tuple 
)
import numpy as np

from . import KERAS_DEPS
from ._shapes import canonicalize_to_BHQO_using_ytrue
from .keras_metrics import (
    MAEQ50,
    MSEQ50,
)


Callback = KERAS_DEPS.Callback
Tensor = KERAS_DEPS.Tensor 

tf_nest = KERAS_DEPS.nest 
tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor
tf_reshape = KERAS_DEPS.reshape 
tf_shape = KERAS_DEPS.shape 
tf_transpose = KERAS_DEPS.transpose 
tf_reduce_mean = KERAS_DEPS.reduce_mean 
tf_cast = KERAS_DEPS.cast 
tf_float32 = KERAS_DEPS.float32 
tf_minimum = KERAS_DEPS.minimum 
tf_maximum = KERAS_DEPS.maximum 
tf_abs = KERAS_DEPS.abs 
tf_square = KERAS_DEPS.square 
tf_identity = KERAS_DEPS.identity 
tf_expand_dims = KERAS_DEPS.expand_dims 
tf_broadcast_to = KERAS_DEPS.broadcast_to
tf_logical_and = KERAS_DEPS.logical_and

ScheduleType = Union[
    Callable[[Optional[int], int, float], float],
    Mapping[int, float],
    Sequence[float],
    None,
]


__all__ = [
    # "ResetOptimizerStats",
    # "make_lr_scheduler",
    # "summarize_lr_schedule",
    "NaNGuard",
    "FrozenValQuantileMonitor",
    "FrozenValQuantilePrinter",
    "FrozenValQuantileLogger",
    "LambdaOffsetScheduler", 
    "LambdaOffsetStepScheduler"
    
]


def _linear_warmup_value(
    idx: int,
    start: float,
    end: float,
    warmup: int,
) -> float:
    """Linear ramp from start to end over warmup steps/epochs."""
    if warmup <= 0:
        return float(end)
    if idx <= 0:
        return float(start)
    if idx >= warmup:
        return float(end)
    frac = float(idx) / float(warmup)
    return float(start + (end - start) * frac)


class LambdaOffsetScheduler(Callback):
    r"""
    Schedule GeoPrior's global physics-loss offset ``_lambda_offset``.

    This callback updates the non-trainable TF variable
    ``model._lambda_offset`` via ``assign()``, which is safe under
    ``tf.function`` tracing (the new value is visible to the graph).

    It supports both:
    - epoch-based schedules (default) via ``unit="epoch"``
    - step-based schedules via ``unit="step"``

    Parameters
    ----------
    schedule : callable or mapping or sequence or None, optional
        How to set the offset.

        * callable:
            ``schedule(epoch, step, current) -> new_value``

        * mapping:
            ``{index: value}`` where index is epoch or step depending on
            ``unit``. Missing keys keep the current value.

        * sequence:
            ``values[index]`` where index is epoch or step.

        * None (default):
            Use an internal warmup schedule controlled by ``warmup``,
            ``start`` and ``end`` (and adapted to ``model.offset_mode`` if
            start/end are not provided).

    unit : {"epoch", "step"}, default="epoch"
        Schedule index type.

    when : {"begin", "end"}, default="begin"
        When to apply the update.

    warmup : int, default=10
        Warmup length when ``schedule is None``. Meaning depends on
        ``unit`` (epochs or steps).

    start : float or None, optional
        Start value for the warmup when ``schedule is None``.
        If None, a mode-aware default is chosen:
        - offset_mode="mul"   -> start=0.1
        - offset_mode="log10" -> start=-1.0  (multiplier 0.1)

    end : float or None, optional
        End value for the warmup when ``schedule is None``.
        If None, a mode-aware default is chosen:
        - offset_mode="mul"   -> end=1.0
        - offset_mode="log10" -> end=0.0   (multiplier 1.0)

    clamp_positive : bool, default=True
        Enforce ``_lambda_offset > 0`` when ``offset_mode="mul"``.

    verbose : int, default=1
        Print updates.

    Notes
    -----
    * This callback expects the model to expose:
      - ``model._lambda_offset`` (tf.Variable; non-trainable)
      - ``model.offset_mode`` in {"mul", "log10"}
    """

    def __init__(
        self,
        schedule: ScheduleType = None,
        unit: str = "epoch",
        when: str = "begin",
        warmup: int = 10,
        start: Optional[float] = None,
        end: Optional[float] = None,
        clamp_positive: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__()
        self.schedule = schedule
        self.unit = str(unit)
        self.when = str(when)

        self.warmup = int(warmup)
        self.start = start
        self.end = end

        self.clamp_positive = bool(clamp_positive)
        self.verbose = int(verbose)

        self.step_: int = 0
        self.last_value_: Optional[float] = None

        if self.unit not in ("epoch", "step"):
            raise ValueError("unit must be 'epoch' or 'step'.")
        if self.when not in ("begin", "end"):
            raise ValueError("when must be 'begin' or 'end'.")

    # -----------------------
    # Lifecycle
    # -----------------------
    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        self.step_ = 0
        self.last_value_ = None

        if not hasattr(self.model, "_lambda_offset"):
            raise AttributeError(
                "LambdaOffsetScheduler requires `model._lambda_offset` "
                "(created with add_weight(trainable=False))."
            )
        if not hasattr(self.model, "offset_mode"):
            raise AttributeError(
                "LambdaOffsetScheduler requires `model.offset_mode`."
            )

        # Apply initial update (epoch 0 / step 0) if configured at begin.
        if self.when == "begin":
            epoch0 = 0 if self.unit == "epoch" else None
            self._maybe_update(epoch=epoch0, step=self.step_)

    # -----------------------
    # Epoch hooks
    # -----------------------
    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        if self.unit == "epoch" and self.when == "begin":
            self._maybe_update(epoch=epoch, step=self.step_)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if self.unit == "epoch" and self.when == "end":
            self._maybe_update(epoch=epoch, step=self.step_)

    # -----------------------
    # Step hooks
    # -----------------------
    def on_train_batch_begin(self, batch: int, logs: Optional[dict] = None) -> None:
        if self.unit == "step" and self.when == "begin":
            self._maybe_update(epoch=None, step=self.step_)

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        if self.unit == "step" and self.when == "end":
            self._maybe_update(epoch=None, step=self.step_)
        self.step_ += 1

    # -----------------------
    # Core helpers
    # -----------------------
    def _current_value(self) -> float:
        try:
            return float(self.model._lambda_offset.numpy())
        except Exception:
            return float(self.model._lambda_offset)

    def _mode_defaults(self) -> tuple[float, float]:
        mode = str(getattr(self.model, "offset_mode", "mul"))
        if mode == "log10":
            # multiplier goes 10**(-1)=0.1 -> 10**0=1.0
            return -1.0, 0.0
        # mode == "mul"
        return 0.1, 1.0

    def _default_schedule_value(self, epoch: Optional[int], step: int) -> float:
        idx = int(epoch) if self.unit == "epoch" else int(step)
        d_start, d_end = self._mode_defaults()
        start = float(self.start) if self.start is not None else d_start
        end = float(self.end) if self.end is not None else d_end
        return _linear_warmup_value(idx, start=start, end=end, warmup=self.warmup)

    def _get_scheduled_value(
        self,
        epoch: Optional[int],
        step: int,
        current: float,
    ) -> Optional[float]:
        idx = int(epoch) if self.unit == "epoch" else int(step)

        if self.schedule is None:
            return self._default_schedule_value(epoch=epoch, step=step)

        if callable(self.schedule):
            return float(self.schedule(epoch, step, current))

        if isinstance(self.schedule, Mapping):
            v = self.schedule.get(idx, None)
            return None if v is None else float(v)

        if isinstance(self.schedule, Sequence):
            if 0 <= idx < len(self.schedule):
                return float(self.schedule[idx])
            return None

        raise TypeError(
            "schedule must be callable, mapping, sequence, or None."
        )

    def _validate_value(self, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("lambda_offset must be finite.")

        mode = str(getattr(self.model, "offset_mode", "mul"))
        if self.clamp_positive and mode == "mul" and value <= 0.0:
            raise ValueError(
                "lambda_offset must be > 0 when offset_mode='mul'."
            )

    def _maybe_update(self, epoch: Optional[int], step: int) -> None:
        cur = self._current_value()
        new = self._get_scheduled_value(epoch=epoch, step=step, current=cur)

        if new is None:
            return

        self._validate_value(new)
        self.model._lambda_offset.assign(float(new))
        self.last_value_ = float(new)

        if self.verbose:
            unit = "epoch" if self.unit == "epoch" else "step"
            idx = epoch if self.unit == "epoch" else step
            print(
                f"[LambdaOffsetScheduler] {unit}={idx}: "
                f"lambda_offset={float(new):g}"
            )

    def __repr__(self) -> str:
        return (
            "LambdaOffsetScheduler("
            f"unit={self.unit!r}, when={self.when!r}, "
            f"warmup={self.warmup}, start={self.start}, end={self.end}, "
            f"clamp_positive={self.clamp_positive}, verbose={self.verbose})"
        )

class LambdaOffsetStepScheduler(LambdaOffsetScheduler):
    def __init__(self, *args, **kwargs):
        kwargs["unit"] = "step"
        super().__init__(*args, **kwargs)

class NaNGuard(Callback):
    r"""
    Early-stop a training run if any watched metric is non-finite.

    This callback inspects the Keras ``logs`` dict after each *train batch*,
    *validation batch*, and/or *epoch end* (configurable). If any selected
    metric is ``NaN`` or ``Inf``, it sets ``model.stop_training = True`` so
    the current run/trial ends immediately.

    Parameters
    ----------
    limit_to : Iterable[str] or None, optional
        If provided, only these metric keys are checked (e.g.,
        ``{"loss", "val_loss", "total_loss", "physics_loss"}``).
        If ``None`` (default), all numeric entries in ``logs`` are checked.
    check_train : bool, default True
        Inspect metrics after each training batch
        (``on_train_batch_end``).
    check_val : bool, default True
        Inspect metrics after each validation batch
        (``on_test_batch_end`` as used by Keras during fit()).
    check_epoch_end : bool, default True
        Inspect metrics at the end of each epoch (common place to see
        ``val_*`` keys).
    raise_on_nan : bool, default False
        If True, raise ``RuntimeError`` when a non-finite value is found
        (useful to make outer orchestration detect a "failed trial"
        immediately). If False, only stops training.
    verbose : int, default 1
        0 = silent, 1 = brief one-line notices.

    Attributes
    ----------
    tripped_ : bool
        Whether the guard has been triggered for this run.
    last_bad_key_ : str or None
        The metric key that triggered the stop (e.g., ``"val_loss"``).
    last_bad_value_ : Any
        The offending value as captured from ``logs``.
    last_bad_phase_ : {"train-batch", "val-batch", "epoch-end"} or None
        Where the issue was detected.

    Notes
    -----

    During hyperparameter search (e.g., with KerasTuner), exploding losses can
    cascade into repeated trial failures. ``NaNGuard`` stops the current trial
    as soon as a non-finite metric is observed, helping you fail fast, save
    time, and surface bad configurations cleanly.
    
    * The callback resets its state at ``on_train_begin`` so it can be reused
      across multiple tuner trials.
    * Values in ``logs`` are often Python floats, but may also be NumPy arrays
      or eager tensors. This class normalizes them to NumPy and tests
      ``np.all(np.isfinite(...))`` safely.
    * Messages use ASCII only to avoid Windows cp1252 console issues.

    Examples
    --------
    Basic usage:

    >>> from fusionlab.nn.callbacks import NaNGuard
    >>> nan_guard = NaNGuard(
    ...     limit_to={"loss", "val_loss", "total_loss",
    ...               "data_loss", "physics_loss",
    ...               "consolidation_loss", "gw_flow_loss"},
    ...     raise_on_nan=False,
    ...     verbose=1
    ... )
    >>> model.fit(
    ...     train_ds,
    ...     validation_data=val_ds,
    ...     epochs=50,
    ...     callbacks=[nan_guard],
    ... )

    With KerasTuner (recommended together with EarlyStopping):

    >>> from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
    >>> early = EarlyStopping(monitor="val_loss", patience=10,
    ...                       restore_best_weights=True, verbose=1)
    >>> ton = TerminateOnNaN()
    >>> tuner.search(
    ...     train_ds,
    ...     validation_data=val_ds,
    ...     epochs=50,
    ...     callbacks=[early, ton, nan_guard],
    ... )
    """

    def __init__(
        self,
        limit_to: Optional[Iterable[str]] = None,
        check_train: bool = True,
        check_val: bool = True,
        check_epoch_end: bool = True,
        raise_on_nan: bool = False,
        verbose: int = 1,
    ) -> None:
        super().__init__()
        self.limit_to: Optional[Set[str]] = set(limit_to) if limit_to else None
        self.check_train = bool(check_train)
        self.check_val = bool(check_val)
        self.check_epoch_end = bool(check_epoch_end)
        self.raise_on_nan = bool(raise_on_nan)
        self.verbose = int(verbose)

        # Runtime state
        self.tripped_: bool = False
        self.last_bad_key_: Optional[str] = None
        self.last_bad_value_: Any = None
        self.last_bad_phase_: Optional[str] = None

    # -----------------------
    # Lifecycle helpers
    # -----------------------
    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        # Reset state for a fresh run/trial
        self.tripped_ = False
        self.last_bad_key_ = None
        self.last_bad_value_ = None
        self.last_bad_phase_ = None

    # -----------------------
    # Hooks
    # -----------------------
    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        if not self.check_train or self.tripped_:
            return
        self._scan_logs_and_maybe_trip(logs or {}, where="train-batch")

    def on_test_batch_end(self, batch: int, logs: Optional[dict] = None) -> None:
        # Called for validation batches during fit()
        if not self.check_val or self.tripped_:
            return
        self._scan_logs_and_maybe_trip(logs or {}, where="val-batch")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if not self.check_epoch_end or self.tripped_:
            return
        self._scan_logs_and_maybe_trip(logs or {}, where="epoch-end")

    # -----------------------
    # Core logic
    # -----------------------
    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Best-effort conversion of a logs value to a NumPy array."""
        try:
            if hasattr(value, "numpy"):
                return np.asarray(value.numpy())
            return np.asarray(value)
        except Exception:
            # If conversion fails, return an empty array that is "finite"
            return np.asarray([], dtype=float)

    @staticmethod
    def _is_nonfinite(arr: np.ndarray) -> bool:
        """True if any element is NaN or Inf (also handles scalars)."""
        if arr.size == 0:
            return False
        try:
            return not np.all(np.isfinite(arr))
        except Exception:
            # Be conservative: if we can't decide, don't trip on it
            return False

    def _scan_logs_and_maybe_trip(self, logs: dict, where: str) -> None:
        if not logs:
            return

        keys = (self.limit_to & logs.keys()) if self.limit_to else logs.keys()

        for k in keys:
            v = logs.get(k, None)
            if v is None:
                continue
            arr = self._to_numpy(v)
            if self._is_nonfinite(arr):
                # Trip
                self.tripped_ = True
                self.last_bad_key_ = k
                self.last_bad_value_ = v
                self.last_bad_phase_ = where
                if self.verbose:
                    print(f"[NaNGuard] Non-finite metric '{k}' detected in {where}; stopping.")
                # Stop current fit() cleanly
                self.model.stop_training = True
                if self.raise_on_nan:
                    # Raise after signaling stop so outer orchestrators can catch
                    raise RuntimeError(f"NaNGuard tripped on '{k}' during {where}.")
                break

    # -----------------------
    # Representations
    # -----------------------
    def __repr__(self) -> str:
        lim = sorted(self.limit_to) if self.limit_to else None
        return (
            "NaNGuard("
            f"limit_to={lim}, "
            f"check_train={self.check_train}, "
            f"check_val={self.check_val}, "
            f"check_epoch_end={self.check_epoch_end}, "
            f"raise_on_nan={self.raise_on_nan}, "
            f"verbose={self.verbose})"
        )

class FrozenValQuantileMonitor(Callback):
    """
   Print the same diagnostics every epoch on a frozen val batch.

   This callback freezes the *first* batch of the provided
   validation data (dataset / sequence / tuple) at train start,
   then re-evaluates that exact batch at every epoch end.

   It is designed to catch:
   - quantile crossings (e.g., q10 > q90)
   - when crossings begin and whether they recover
   - fixed-batch coverage/sharpness/MAE/MSE for q50

   Parameters
   ----------
   val_data : Any
        Validation source. One of:
        - tf.data.Dataset yielding (x, y) or (x, y, sw)
        - Keras Sequence / iterable yielding batches
        - Tuple (x, y) or (x, y, sw)

   outputs : Sequence[str] | None, optional
        Output names to monitor. If None, monitors all keys
        when y is a dict; otherwise monitors a single output.

   quantiles : Sequence[float] | None, optional
        Quantiles expected in predictions (e.g. (0.1,0.5,0.9)).
        If None, the callback treats predictions as point
        forecasts and skips band diagnostics.

   alpha : float, optional
        Central interval mass for coverage/sharpness.
        Default is 0.8 (i.e., 80% interval).

   every : int, optional
        Print every `every` epochs. Default is 1.

   prefix : str, optional
        Prefix for printed blocks.

   print_fn : Callable[[str], None] | None, optional
        Custom printer. Default uses builtin print().
    """

    def __init__(
        self,
        val_data: Any,
        *,
        outputs: Sequence[str] | None = None,
        quantiles: Sequence[float] | None = (0.1, 0.5, 0.9),
        alpha: float = 0.8,
        every: int = 1,
        prefix: str = "[frozen-val]",
        print_fn: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._val_data = val_data
        self._outputs = None if outputs is None else list(outputs)
        self._qs = None if quantiles is None else list(quantiles)
        self._alpha = float(alpha)
        self._every = max(1, int(every))
        self._prefix = str(prefix)
        self._print = print if print_fn is None else print_fn

        self._xb = None
        self._yb = None
        self._sw = None

        self._q50 = None
        self._qlo = None
        self._qhi = None

    # ------------------------------
    # Helpers
    # ------------------------------
    def _take_one_batch(self) -> tuple[Any, Any, Any]:
        vd = self._val_data

        if isinstance(vd, (tuple, list)):
            if len(vd) == 2:
                return vd[0], vd[1], None
            if len(vd) == 3:
                return vd[0], vd[1], vd[2]
            raise ValueError(
                "val_data tuple must be (x,y) or (x,y,sw)."
            )

        # tf.data.Dataset or general iterable / Sequence
        it = iter(vd)
        batch = next(it)

        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                return batch[0], batch[1], None
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]

        raise ValueError(
            "val_data must yield (x,y) or (x,y,sw)."
        )


    def _to_tensor_tree(self, obj: Any) -> Any:
        return tf_nest.map_structure(tf_convert_to_tensor, obj)

    def _as_dict(
        self,
        y: Any,
        *,
        names: Sequence[str] | None,
    ) -> dict[str, Any]:
        if isinstance(y, Mapping):
            return dict(y)

        if isinstance(y, (tuple, list)):
            out: dict[str, Any] = {}
            if names and len(names) == len(y):
                for k, v in zip(names, y):
                    out[str(k)] = v
                return out
            for i, v in enumerate(y):
                out[f"y{i}"] = v
            return out

        if names and len(names) == 1:
            return {str(names[0]): y}

        return {"y": y}

    def _nearest_idx(self, qs: Sequence[float], q: float) -> int:
        q = float(q)
        best = 0
        best_d = float("inf")
        for i, v in enumerate(qs):
            d = abs(float(v) - q)
            if d < best_d:
                best = i
                best_d = d
        return int(best)

    def _to_BHO(self, y_true: Tensor) -> Tensor:
        y = tf_convert_to_tensor(y_true)
        r = y.shape.rank

        if r == 3:
            return y
        if r == 2:
            return y[..., None]
        if r == 1:
            return y[:, None, None]

        # fallback: keep last dim as O
        y = tf_reshape(y, [tf_shape(y)[0], -1, 1])
        return y

    def _to_BHQO(
        self,
        y_pred: Tensor,
        *,
        n_q: int,
        q_axis: int | None = None,
    ) -> tuple[Tensor, bool]:
        """
        Return (B,H,Q,O), and whether Q>1 was detected.
        """
        y = tf_convert_to_tensor(y_pred)
        r = y.shape.rank

        if r == 4:
            # assume (B,H,Q,O) unless q_axis says otherwise
            if q_axis is None or q_axis == 2:
                qn = y.shape[2]
                has_q = (qn is not None and qn > 1)
                return y, bool(has_q)

            # move q_axis -> 2
            axes = list(range(4))
            src = int(q_axis)
            axes.pop(src)
            axes.insert(2, src)
            y = tf_transpose(y, perm=axes)
            qn = y.shape[2]
            has_q = (qn is not None and qn > 1)
            return y, bool(has_q)

        if r == 3:
            # common cases:
            # (B,H,1) point -> (B,H,1,1)
            # (B,H,Q) quant -> (B,H,Q,1)
            # (B,H,O) point -> (B,H,1,O)
            last = y.shape[-1]
            if last == 1:
                y = y[..., None]
                return y, False
            if last == n_q:
                y = y[..., None]
                return y, True
            y = y[:, :, None, :]
            return y, False

        if r == 2:
            # (B,H) -> (B,H,1,1)
            y = y[:, :, None, None]
            return y, False

        if r == 1:
            y = y[:, None, None, None]
            return y, False

        y = tf_reshape(y, [tf_shape(y)[0], -1, 1, 1])
        return y, False

    def _f(self, x: Any) -> float:
        try:
            return float(x.numpy())
        except Exception:
            try:
                return float(x)
            except Exception:
                return float("nan")

    def _fmt(self, x: float, n: int = 4) -> str:
        if not (x == x):  # NaN
            return "nan"
        return f"{x:.{n}f}"

    # ------------------------------
    # Keras hooks
    # ------------------------------
    def on_train_begin(self, logs: dict | None = None) -> None:
        xb, yb, sw = self._take_one_batch()

        self._xb = self._to_tensor_tree(xb)
        self._yb = self._to_tensor_tree(yb)
        self._sw = self._to_tensor_tree(sw) if sw is not None else None

        qs = self._qs
        if qs is None or len(qs) == 0:
            self._q50 = None
            self._qlo = None
            self._qhi = None
            return

        self._q50 = self._nearest_idx(qs, 0.5)

        lo_q = 0.5 * (1.0 - float(self._alpha))
        hi_q = 1.0 - lo_q
        self._qlo = self._nearest_idx(qs, lo_q)
        self._qhi = self._nearest_idx(qs, hi_q)

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,
    ) -> None:
        ep = int(epoch) + 1
        if (ep % self._every) != 0:
            return

        if self._xb is None or self._yb is None:
            return

        logs = {} if logs is None else dict(logs)

        yp = self.model(self._xb, training=False)

        out_names = list(getattr(self.model, "output_names", []))

        y_true_d = self._as_dict(self._yb, names=out_names)
        y_pred_d = self._as_dict(yp, names=out_names)

        keys = self._outputs
        if keys is None:
            keys = list(y_true_d.keys())

        # header
        loss = self._fmt(self._f(logs.get("loss", float("nan"))))
        vloss = self._fmt(self._f(logs.get("val_loss", float("nan"))))

        self._print(
            f"{self._prefix} epoch={ep} "
            f"loss={loss} val_loss={vloss}"
        )

        for k in keys:
            if k not in y_true_d or k not in y_pred_d:
                continue

            yt_raw = y_true_d[k]
            yp_raw = y_pred_d[k]

            # metrics (reuse your exact metric logic)
            m_mae = MAEQ50()
            m_mse = MSEQ50()
            
            m_mae.update_state(yt_raw, yp_raw)
            m_mse.update_state(yt_raw, yp_raw)
            
            mae = self._fmt(self._f(m_mae.result()))
            mse = self._fmt(self._f(m_mse.result()))
            
            do_band = self._qs is not None and len(self._qs) >= 2
            if do_band:
                cov, shp = self._cov_shp_from_band(yt_raw, yp_raw)
            else:
                cov, shp = "na", "na"

            # crossings (fixed-batch, epoch-by-epoch)
            cross_1090 = "na"
            cross_1050 = "na"
            cross_5090 = "na"

            if do_band and self._qlo is not None:
                qs = self._qs or []
                n_q = max(1, len(qs))

                q, has_q = self._to_BHQO(
                    yp_raw,
                    n_q=n_q,
                    q_axis=None,
                )

                if has_q and q.shape.rank == 4:
                    lo = q[:, :, int(self._qlo), :]
                    md = q[:, :, int(self._q50), :]
                    hi = q[:, :, int(self._qhi), :]

                    c1090 = tf_reduce_mean(
                        tf_cast(lo > hi, tf_float32)
                    )
                    c1050 = tf_reduce_mean(
                        tf_cast(lo > md, tf_float32)
                    )
                    c5090 = tf_reduce_mean(
                        tf_cast(md > hi, tf_float32)
                    )

                    cross_1090 = self._fmt(self._f(c1090))
                    cross_1050 = self._fmt(self._f(c1050))
                    cross_5090 = self._fmt(self._f(c5090))

            self._print(
                f"  {k}: mae50={mae} mse50={mse} "
                f"cov{int(self._alpha*100):d}={cov} "
                f"shp{int(self._alpha*100):d}={shp}"
            )
            self._print(
                f"    cross(q10>q90)={cross_1090} "
                f"cross(q10>q50)={cross_1050} "
                f"cross(q50>q90)={cross_5090}"
            )

    def _cov_shp_from_band(
        self,
        yt_raw: Any,
        yp_raw: Any,
    ) -> tuple[str, str]:
        if self._qs is None or self._qlo is None:
            return "na", "na"

        qs = self._qs
        n_q = max(1, len(qs))

        y = self._to_BHO(yt_raw)

        q, has_q = self._to_BHQO(
            yp_raw,
            n_q=n_q,
            q_axis=None,
        )
        if not has_q or q.shape.rank != 4:
            return "na", "na"

        lo = q[:, :, int(self._qlo), :]
        hi = q[:, :, int(self._qhi), :]

        # crossing-safe band
        lo2 = tf_minimum(lo, hi)
        hi2 = tf_maximum(lo, hi)

        lo2 = tf_broadcast_to(lo2, tf_shape(y))
        hi2 = tf_broadcast_to(hi2, tf_shape(y))

        inside = tf_cast(
            tf_logical_and(y >= lo2, y <= hi2),
            tf_float32,
        )

        cov = tf_reduce_mean(inside)
        shp = tf_reduce_mean(tf_abs(hi2 - lo2))

        return self._fmt(self._f(cov)), self._fmt(self._f(shp))


class FrozenValQuantilePrinter(KERAS_DEPS.Callback):
    """
    Print quantile diagnostics on a frozen val batch.

    This helps catch:
      - quantile crossings that appear mid-training
      - broken coverage due to layout/shape mistakes

    Parameters
    ----------
    val_data : tf.data.Dataset | tuple
        Either a dataset yielding (x,y) or a pre-fetched (xb,yb).
        The first batch is frozen at init time.
    y_key : str
        Key in y dict (e.g. "subs_pred").
    pred_key : str
        Key in model outputs (e.g. "subs_pred").
    q_values : tuple[float,...]
        Quantile levels (default (0.1,0.5,0.9)).
    every : int
        Print every N epochs (default 1).
    prefix : str
        Print prefix.
    """

    def __init__(
        self,
        *,
        val_data: Any,
        y_key: str = "subs_pred",
        pred_key: str = "subs_pred",
        q_values: Tuple[float, ...] = (0.1, 0.5, 0.9),
        every: int = 1,
        prefix: str = "[val-qdiag]",
    ) -> None:
        super().__init__()


        self._y_key = str(y_key)
        self._p_key = str(pred_key)
        self._qv = tuple(float(q) for q in q_values)
        self._nq = int(len(self._qv))
        self._every = max(int(every), 1)
        self._prefix = str(prefix)

        xb, yb = self._take_one(val_data)
        self._xb = self._freeze(xb)
        self._yb = self._freeze(yb)

    def _take_one(self, val_data: Any) -> Tuple[Any, Any]:
        if hasattr(val_data, "take"):
            it = iter(val_data.take(1))
            xb, yb = next(it)
            return xb, yb

        if isinstance(val_data, (tuple, list)) and len(val_data) == 2:
            return val_data[0], val_data[1]

        raise TypeError(
            "val_data must be a Dataset or (xb,yb) tuple."
        )

    def _freeze(self, x: Any) -> Any:

        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                out[k] = tf_identity(tf_convert_to_tensor(v))
            return out

        if isinstance(x, (tuple, list)):
            typ = type(x)
            return typ(self._freeze(v) for v in x)

        return tf_identity(tf_convert_to_tensor(x))

    def _to_BHQO(self, y_pred: Any, y_true: Any) -> Any:
        """
        Canonicalize quantiles to (B,H,Q,O).

        Uses your existing helper:
          canonicalize_to_BHQO_using_ytrue
        """

        yp = tf_convert_to_tensor(y_pred)
        r = yp.shape.rank

        # Rank-4: try common layouts, else use MAE chooser.
        if r == 4:
            s = yp

            if s.shape[2] == self._nq:
                return s

            if s.shape[1] == self._nq:
                return tf_transpose(s, [0, 2, 1, 3])

            if s.shape[3] == self._nq:
                return tf_transpose(s, [0, 1, 3, 2])

            yt = tf_convert_to_tensor(y_true)
            return canonicalize_to_BHQO_using_ytrue(
                s,
                yt,
                q_values=self._qv,
            )

        # Rank-3: (B,H,Q) -> (B,H,Q,1)
        if r == 3 and yp.shape[-1] == self._nq:
            return tf_expand_dims(yp, axis=-1)

        # Otherwise treat as point forecast -> Q=1
        if r == 3:
            return tf_expand_dims(yp, axis=2)

        if r == 2:
            s = tf_expand_dims(yp, axis=-1)
            return tf_expand_dims(s, axis=2)

        return yp

    def _mean(self, x: Any) -> float:

        return float(
            tf_reduce_mean(tf_cast(x, tf_float32)).numpy()
        )

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if (epoch % self._every) != 0:
            return

        xb = self._xb
        yb = self._yb

        y_true = yb[self._y_key]
        yp_all = self.model(xb, training=False)

        # model may return dict or list/tuple
        if isinstance(yp_all, dict):
            y_pred = yp_all[self._p_key]
        else:
            y_pred = yp_all

        q = self._to_BHQO(y_pred, y_true)

        # y_true is (B,H,1) in SM3 -> (B,H)
        yt = tf_convert_to_tensor(y_true)[..., 0]

        q10 = q[:, :, 0, 0]
        q50 = q[:, :, 1, 0] if self._nq >= 2 else q[:, :, 0, 0]
        q90 = q[:, :, -1, 0]

        c10_50 = self._mean(q10 > q50)
        c50_90 = self._mean(q50 > q90)
        c10_90 = self._mean(q10 > q90)

        lo = tf_minimum(q10, q90)
        hi = tf_maximum(q10, q90)

        cov80 = self._mean((yt >= lo) & (yt <= hi))
        shp80 = self._mean(hi - lo)
        mae50 = self._mean(tf_abs(yt - q50))
        mse50 = self._mean(tf_square(yt - q50))

        msg = (
            f"{self._prefix} epoch={epoch + 1} "
            f"qshape={tuple(q.shape)} "
            f"yshape={tuple(y_true.shape)}\n"
            f"{self._prefix} cross "
            f"q10>q50={c10_50:.6f} "
            f"q50>q90={c50_90:.6f} "
            f"q10>q90={c10_90:.6f}\n"
            f"{self._prefix} cov80={cov80:.6f} "
            f"shp80={shp80:.6f} "
            f"mae50={mae50:.6f} "
            f"mse50={mse50:.6f}"
        )
        print(msg)

class FrozenValQuantileLogger(Callback):
    """
    Frozen val quantile diagnostics that write into `logs`.

    Parameters
    ----------
    val_data : tf.data.Dataset | tuple
        Dataset or (xb,yb). First batch is frozen.
    y_key, pred_key : str
        Dict keys for targets and preds.
    q_values : tuple[float,...]
        Quantile levels.
    every : int
        Log every N epochs.
    log_prefix : str
        Prefix for log keys (e.g. "diag/").
    also_print : bool
        If True, print a short one-liner each epoch.
    """

    def __init__(
        self,
        *,
        val_data: Any,
        y_key: str = "subs_pred",
        pred_key: str = "subs_pred",
        q_values: Tuple[float, ...] = (0.1, 0.5, 0.9),
        every: int = 1,
        log_prefix: str = "diag/",
        also_print: bool = False,
    ) -> None:
        super().__init__()

        self._y_key = str(y_key)
        self._p_key = str(pred_key)
        self._qv = tuple(float(q) for q in q_values)
        self._nq = int(len(self._qv))
        self._every = max(int(every), 1)
        self._lp = str(log_prefix)
        self._also_print = bool(also_print)

        xb, yb = self._take_one(val_data)
        self._xb = self._freeze(xb)
        self._yb = self._freeze(yb)

    # same helpers as printer
    def _take_one(self, val_data: Any) -> Tuple[Any, Any]:
        if hasattr(val_data, "take"):
            it = iter(val_data.take(1))
            xb, yb = next(it)
            return xb, yb

        if isinstance(val_data, (tuple, list)) and len(val_data) == 2:
            return val_data[0], val_data[1]

        raise TypeError(
            "val_data must be a Dataset or (xb,yb) tuple."
        )

    def _freeze(self, x: Any) -> Any:
 
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                out[k] = tf_identity(tf_convert_to_tensor(v))
            return out

        if isinstance(x, (tuple, list)):
            typ = type(x)
            return typ(self._freeze(v) for v in x)

        return tf_identity(tf_convert_to_tensor(x))

    def _to_BHQO(self, y_pred: Any, y_true: Any) -> Any:

        yp = tf_convert_to_tensor(y_pred)
        r = yp.shape.rank

        if r == 4:
            s = yp

            if s.shape[2] == self._nq:
                return s

            if s.shape[1] == self._nq:
                return tf_transpose(s, [0, 2, 1, 3])

            if s.shape[3] == self._nq:
                return tf_transpose(s, [0, 1, 3, 2])

            yt = tf_convert_to_tensor(y_true)
            return canonicalize_to_BHQO_using_ytrue(
                s,
                yt,
                q_values=self._qv,
            )

        if r == 3 and yp.shape[-1] == self._nq:
            return tf_expand_dims(yp, axis=-1)

        if r == 3:
            return tf_expand_dims(yp, axis=2)

        if r == 2:
            s = tf_expand_dims(yp, axis=-1)
            return tf_expand_dims(s, axis=2)

        return yp

    def _mean(self, x: Any) -> float:
        return float(
            tf_reduce_mean(tf_cast(x, tf_float32)).numpy()
        )

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if (epoch % self._every) != 0:
            return


        logs = logs if logs is not None else {}

        xb = self._xb
        yb = self._yb

        y_true = yb[self._y_key]
        yp_all = self.model(xb, training=False)

        if isinstance(yp_all, dict):
            y_pred = yp_all[self._p_key]
        else:
            y_pred = yp_all

        q = self._to_BHQO(y_pred, y_true)
        yt = tf_convert_to_tensor(y_true)[..., 0]

        q10 = q[:, :, 0, 0]
        q50 = q[:, :, 1, 0] if self._nq >= 2 else q[:, :, 0, 0]
        q90 = q[:, :, -1, 0]

        c10_50 = self._mean(q10 > q50)
        c50_90 = self._mean(q50 > q90)
        c10_90 = self._mean(q10 > q90)

        lo = tf_minimum(q10, q90)
        hi = tf_maximum(q10, q90)

        cov80 = self._mean((yt >= lo) & (yt <= hi))
        shp80 = self._mean(hi - lo)
        mae50 = self._mean(tf_abs(yt - q50))
        mse50 = self._mean(tf_square(yt - q50))

        logs[self._lp + "cov80"] = cov80
        logs[self._lp + "shp80"] = shp80
        logs[self._lp + "mae50"] = mae50
        logs[self._lp + "mse50"] = mse50
        logs[self._lp + "cross10_50"] = c10_50
        logs[self._lp + "cross50_90"] = c50_90
        logs[self._lp + "cross10_90"] = c10_90

        if self._also_print:
            msg = (
                f"[{self._lp}epoch={epoch + 1}] "
                f"cov80={cov80:.4f} "
                f"shp80={shp80:.4f} "
                f"c10_90={c10_90:.4f}"
            )
            print(msg)
            
    def _cov_shp_from_band(
        self,
        yt_raw: Any,
        yp_raw: Any,
    ) -> tuple[str, str]:
        if self._qs is None or self._qlo is None:
            return "na", "na"

        qs = self._qs
        n_q = max(1, len(qs))

        y = self._to_BHO(yt_raw)

        q, has_q = self._to_BHQO(
            yp_raw,
            n_q=n_q,
            q_axis=None,
        )
        if not has_q or q.shape.rank != 4:
            return "na", "na"

        lo = q[:, :, int(self._qlo), :]
        hi = q[:, :, int(self._qhi), :]

        # crossing-safe band
        lo2 = tf_minimum(lo, hi)
        hi2 = tf_maximum(lo, hi)

        lo2 = tf_broadcast_to(lo2, tf_shape(y))
        hi2 = tf_broadcast_to(hi2, tf_shape(y))

        inside = tf_cast(
            tf_logical_and(y >= lo2, y <= hi2),
            tf_float32,
        )

        cov = tf_reduce_mean(inside)
        shp = tf_reduce_mean(tf_abs(hi2 - lo2))

        return self._fmt(self._f(cov)), self._fmt(self._f(shp))
