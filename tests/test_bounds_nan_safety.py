# tests/test_bounds_nan_safety.py
# -*- coding: utf-8 -*-
"""
Tests for bounds helpers: ensure they never
silently return NaN/Inf tensors.

If a configuration is invalid (NaN bounds,
non-positive bounds, etc.), we prefer:
- raise ValueError, OR
- return (None, None, None, None)
  when bounds are genuinely missing.

These tests are intentionally strict. If
they fail, it means the code needs a
robustness patch (like we did for
compute_mv_prior_loss).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


from fusionlab.nn.pinn._geoprior_maths import (
    bounded_exp,
    finite_floor,
    get_log_bounds,
    get_log_tau_bounds,
)


def _is_finite_tensor(x: Any) -> bool:
    if x is None:
        return True
    x = tf.convert_to_tensor(x)
    return bool(tf.reduce_all(tf.math.is_finite(x)).numpy())


def _assert_all_finite(*xs: Any) -> None:
    for x in xs:
        assert _is_finite_tensor(x)


@dataclass
class _DummyMVConfig:
    initial_value: float = 1e-7


@dataclass
class _DummyModel:
    scaling_kwargs: Dict[str, Any]
    gamma_w: Optional[Any] = None
    mv_config: Optional[Any] = None
    time_units: Optional[str] = None


def _make_model(
    *,
    bounds: Optional[Dict[str, Any]] = None,
    gamma_w: Any = 9810.0,
    mv0: float = 1e-7,
    time_units: str = "yr",
) -> _DummyModel:
    sk: Dict[str, Any] = {}
    if bounds is not None:
        sk["bounds"] = bounds

    return _DummyModel(
        scaling_kwargs=sk,
        gamma_w=gamma_w,
        mv_config=_DummyMVConfig(initial_value=mv0),
        time_units=time_units,
    )


# ---------------------------------------------------------------------
# get_log_bounds
# ---------------------------------------------------------------------
def test_get_log_bounds_linear_returns_finite():
    model = _make_model(
        bounds={
            "K_min": 1e-12,
            "K_max": 1e-7,
            "Ss_min": 1e-6,
            "Ss_max": 1e-3,
        },
    )

    out = get_log_bounds(model, as_tensor=True)
    assert isinstance(out, tuple)
    assert len(out) == 4

    _assert_all_finite(*out)


def test_get_log_bounds_log_returns_finite():
    model = _make_model(
        bounds={
            "logK_min": np.log(1e-12),
            "logK_max": np.log(1e-7),
            "logSs_min": np.log(1e-6),
            "logSs_max": np.log(1e-3),
        },
    )

    out = get_log_bounds(model, as_tensor=True)
    _assert_all_finite(*out)


def test_get_log_bounds_missing_returns_none_tuple():
    model = _make_model(bounds={})
    out = get_log_bounds(model, as_tensor=True)
    assert out == (None, None, None, None)


def test_get_log_bounds_rejects_non_positive():
    model = _make_model(
        bounds={
            "K_min": 0.0,
            "K_max": 1e-7,
            "Ss_min": 1e-6,
            "Ss_max": 1e-3,
        },
    )

    with pytest.raises(ValueError):
        get_log_bounds(model, as_tensor=False)


def test_get_log_bounds_rejects_nan_bounds():
    model = _make_model(
        bounds={
            "K_min": np.nan,
            "K_max": 1e-7,
            "Ss_min": 1e-6,
            "Ss_max": 1e-3,
        },
    )

    # We want a hard failure, not NaN logs.
    with pytest.raises(ValueError):
        get_log_bounds(model, as_tensor=False)


# ---------------------------------------------------------------------
# get_log_tau_bounds
# ---------------------------------------------------------------------
def test_get_log_tau_bounds_defaults_are_finite():
    model = _make_model(bounds={})
    log_min, log_max = get_log_tau_bounds(
        model,
        as_tensor=True,
    )
    _assert_all_finite(log_min, log_max)

    # sanity: min < max in linear space
    assert float(tf.exp(log_min).numpy()) < float(
        tf.exp(log_max).numpy(),
    )


def test_get_log_tau_bounds_swaps_if_reversed():
    model = _make_model(
        bounds={
            "tau_min": 100.0,
            "tau_max": 10.0,
        },
    )
    log_min, log_max = get_log_tau_bounds(
        model,
        as_tensor=False,
    )
    assert np.isfinite(log_min)
    assert np.isfinite(log_max)
    assert log_min <= log_max


def test_get_log_tau_bounds_rejects_nan():
    model = _make_model(
        bounds={
            "tau_min": np.nan,
            "tau_max": 10.0,
        },
    )

    with pytest.raises(ValueError):
        get_log_tau_bounds(model, as_tensor=False)


# ---------------------------------------------------------------------
# bounded_exp
# ---------------------------------------------------------------------
def test_bounded_exp_finite_and_in_range():
    raw = tf.constant([-100.0, 0.0, 100.0], tf.float32)
    log_min = tf.constant(np.log(1e-12), tf.float32)
    log_max = tf.constant(np.log(1e-7), tf.float32)

    out = bounded_exp(
        raw,
        log_min,
        log_max,
        eps=1e-12,
        return_log=False,
    )

    _assert_all_finite(out)

    lo = float(tf.exp(log_min).numpy()) + 1e-12
    hi = float(tf.exp(log_max).numpy()) + 1e-12

    out_np = out.numpy()
    assert np.all(out_np >= lo)
    assert np.all(out_np <= hi)


def test_bounded_exp_handles_nan_raw_no_nan():
    raw = tf.constant([np.nan, 0.0, 1.0], tf.float32)
    log_min = tf.constant(np.log(1e-12), tf.float32)
    log_max = tf.constant(np.log(1e-7), tf.float32)

    out = bounded_exp(
        raw,
        log_min,
        log_max,
        eps=1e-12,
        return_log=True,
    )
    val, logv = out

    # This is the robustness contract:
    # bounded_exp should not emit NaN even
    # if raw contains NaN.
    _assert_all_finite(val, logv)


# ---------------------------------------------------------------------
# finite_floor
# ---------------------------------------------------------------------
def test_finite_floor_clamps_nan_inf_and_small():
    x = tf.constant(
        [np.nan, np.inf, -np.inf, -1.0, 0.0, 1e-15, 2.0],
        tf.float32,
    )
    y = finite_floor(x, eps=1e-12)

    _assert_all_finite(y)
    y_np = y.numpy()
    assert np.all(y_np >= 1e-12)
