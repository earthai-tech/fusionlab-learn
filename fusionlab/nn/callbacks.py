# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Utility callbacks for training and tuning.

"""

from __future__ import annotations

from typing import Iterable, Optional, Set, Any
import numpy as np

from . import KERAS_DEPS

Callback = KERAS_DEPS.Callback


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
