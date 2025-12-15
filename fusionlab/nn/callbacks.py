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
    Union
)
import numpy as np

from . import KERAS_DEPS

Callback = KERAS_DEPS.Callback



ScheduleType = Union[
    Callable[[Optional[int], int, float], float],
    Mapping[int, float],
    Sequence[float],
    None,
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


