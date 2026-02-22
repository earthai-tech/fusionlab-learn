# -*- coding: utf-8 -*-
# tests/test_scales_nan_safety.py
"""
NaN/Inf safety tests for residual scaling utilities.

Contract (robustness):
- _scale_residual() must not emit NaN/Inf when residual is finite,
  even if scale contains NaN/Inf/<=0.
- _compute_scales() should not emit NaN/Inf in returned scales,
  even if inputs contain NaN/Inf (this test suite is meant to
  *detect* remaining NaN leaks and lock the behavior once fixed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import tensorflow as tf

from fusionlab.nn.pinn._geoprior_maths import (
    compute_scales,
    scale_residual,
)

def _py(fn):
    return getattr(fn, "python_function", getattr(fn, "__wrapped__", fn))


_compute_scales = _py(compute_scales)
_scale_residual = _py(scale_residual)

# -----------------------------
# Helpers
# -----------------------------
def _is_finite_tensor(x: Any) -> bool:
    x = tf.convert_to_tensor(x)
    ok = tf.reduce_all(tf.math.is_finite(x))
    return bool(ok.numpy())


def _assert_all_finite(*xs: Any) -> None:
    for x in xs:
        assert _is_finite_tensor(x)


def _mk_bad(shape: Tuple[int, ...], kind: str) -> tf.Tensor:
    """Create tensors with NaN/Inf patterns."""
    a = np.ones(shape, dtype=np.float32)

    if kind == "nan":
        a[...] = np.nan
    elif kind == "inf":
        a[...] = np.inf
    elif kind == "neg":
        a[...] = -1.0
    elif kind == "zero":
        a[...] = 0.0
    elif kind == "mix":
        a = np.linspace(-2.0, 2.0, int(np.prod(shape))).astype(np.float32)
        a = a.reshape(shape)
        flat = a.reshape(-1)
        if flat.size >= 1:
            flat[0] = np.nan
        if flat.size >= 2:
            flat[1] = np.inf
        if flat.size >= 3:
            flat[2] = -1.0
        if flat.size >= 4:
            flat[3] = 0.0
        a = flat.reshape(shape)
    else:
        raise ValueError(f"Unknown kind={kind!r}")

    return tf.constant(a, dtype=tf.float32)


def _mk_time(B: int, H: int) -> tf.Tensor:
    """(B,H,1) time grid."""
    t = np.linspace(0.0, 1.0, H, dtype=np.float32)[None, :, None]
    t = np.repeat(t, B, axis=0)
    return tf.constant(t, tf.float32)


def _mk_series(B: int, H: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Simple finite (B,H,1) s_mean/h_mean."""
    s = np.linspace(0.0, 0.2, H, dtype=np.float32)[None, :, None]
    h = np.linspace(10.0, 9.5, H, dtype=np.float32)[None, :, None]
    s = np.repeat(s, B, axis=0)
    h = np.repeat(h, B, axis=0)
    return tf.constant(s, tf.float32), tf.constant(h, tf.float32)


@dataclass
class _DummyModel:
    scaling_kwargs: Dict[str, Any]
    time_units: str = "yr"
    h_ref: float = 0.0


def _default_model(
    *,
    cons_units: Optional[str] = None,
    gw_units: Optional[str] = None,
    cons_floor: Optional[float] = None,
    gw_floor: Optional[float] = None,
) -> _DummyModel:
    sk: Dict[str, Any] = {}
    # These keys may or may not be used by your resolvers;
    # tests still exercise code paths.
    if cons_units is not None:
        sk["cons_units"] = cons_units
    if gw_units is not None:
        sk["gw_units"] = gw_units
    if cons_floor is not None:
        sk["cons_scale_floor"] = cons_floor
    if gw_floor is not None:
        sk["gw_scale_floor"] = gw_floor
    return _DummyModel(scaling_kwargs=sk)


# -----------------------------
# _scale_residual tests
# -----------------------------
@pytest.mark.parametrize("bad_scale", ["nan", "inf", "neg", "zero", "mix"])
def test_scale_residual_no_nan_for_bad_scale(bad_scale: str) -> None:
    residual = tf.constant([1.0, -2.0, 3.0], tf.float32)
    scale = _mk_bad((3,), bad_scale)

    out = _scale_residual(
        residual,
        scale,
        floor=1e-6,
    )
    _assert_all_finite(out)

    # Output should remain finite and roughly scale with floor.
    denom = 1e-6  # + internal epsilon
    expect = residual.numpy() / denom
    got = out.numpy()
    assert np.all(np.isfinite(got))
    assert np.all(np.abs(got) > 0.0)
    assert np.allclose(got, expect, rtol=1e-2, atol=1e-2)


# -----------------------------
# _compute_scales tests
# -----------------------------
def test_compute_scales_clean_is_finite() -> None:
    B, H = 4, 6
    model = _default_model(cons_units="si", gw_units="si")

    t = _mk_time(B, H)
    s, h = _mk_series(B, H)

    K = tf.ones((B, H, 1), tf.float32)
    Ss = tf.ones((B, H, 1), tf.float32) * 1e-5

    out = _compute_scales(
        model,
        t=t,
        s_mean=s,
        h_mean=h,
        K_field=K,
        Ss_field=Ss,
        time_units="yr",
        verbose=0,
    )

    assert "cons_scale" in out
    assert "gw_scale" in out
    _assert_all_finite(out["cons_scale"], out["gw_scale"])


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_compute_scales_handles_bad_dt_no_nan(bad_kind: str) -> None:
    """
    Desired contract:
    dt containing NaN/Inf should not make scales NaN.

    If this fails, sanitize dt_step / dt_sec before reductions
    (e.g., finite_floor or tf.where(is_finite,...)).
    """
    B, H = 4, 6
    model = _default_model(cons_units="si", gw_units="si")

    t = _mk_time(B, H)
    s, h = _mk_series(B, H)

    dt = _mk_bad((B, H, 1), bad_kind)

    K = tf.ones((B, H, 1), tf.float32)
    Ss = tf.ones((B, H, 1), tf.float32) * 1e-5

    out = _compute_scales(
        model,
        t=t,
        s_mean=s,
        h_mean=h,
        K_field=K,
        Ss_field=Ss,
        dt=dt,
        time_units="yr",
        verbose=0,
    )

    _assert_all_finite(out["cons_scale"], out["gw_scale"])


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_compute_scales_handles_bad_s_mean_no_nan(bad_kind: str) -> None:
    """
    Desired contract:
    NaN/Inf in s_mean should not propagate to cons_scale.

    If this fails, sanitize s (and ds) before reduce_mean/max.
    """
    B, H = 4, 6
    model = _default_model(cons_units="si", gw_units="si")

    t = _mk_time(B, H)
    s, h = _mk_series(B, H)

    s_bad = _mk_bad((B, H, 1), bad_kind)

    K = tf.ones((B, H, 1), tf.float32)
    Ss = tf.ones((B, H, 1), tf.float32) * 1e-5

    out = _compute_scales(
        model,
        t=t,
        s_mean=s_bad,
        h_mean=h,
        K_field=K,
        Ss_field=Ss,
        time_units="yr",
        verbose=0,
    )

    _assert_all_finite(out["cons_scale"], out["gw_scale"])


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_compute_scales_handles_bad_Ss_field_no_nan(bad_kind: str) -> None:
    """
    Desired contract:
    NaN/Inf in Ss_field should not propagate to gw_scale.

    If this fails, sanitize Ss_field before Ss_ref computations.
    """
    B, H = 4, 6
    model = _default_model(cons_units="si", gw_units="si")

    t = _mk_time(B, H)
    s, h = _mk_series(B, H)

    K = tf.ones((B, H, 1), tf.float32)
    Ss_bad = _mk_bad((B, H, 1), bad_kind)

    out = _compute_scales(
        model,
        t=t,
        s_mean=s,
        h_mean=h,
        K_field=K,
        Ss_field=Ss_bad,
        time_units="yr",
        verbose=0,
    )

    _assert_all_finite(out["cons_scale"], out["gw_scale"])


@pytest.mark.parametrize("bad_tau,bad_H", [("nan", "mix"), ("mix", "nan")])
def test_compute_scales_handles_bad_tau_H_no_nan(
    bad_tau: str,
    bad_H: str,
) -> None:
    """
    Desired contract:
    If tau_field or H_field contains NaN/Inf, cons_scale should remain finite.

    If this fails, sanitize tau/H and intermediate relax terms.
    """
    B, H = 4, 6
    model = _default_model(cons_units="si", gw_units="si")

    t = _mk_time(B, H)
    s, h = _mk_series(B, H)

    K = tf.ones((B, H, 1), tf.float32)
    Ss = tf.ones((B, H, 1), tf.float32) * 1e-5

    tau = _mk_bad((B, H, 1), bad_tau)
    Hf = _mk_bad((B, H, 1), bad_H)

    out = _compute_scales(
        model,
        t=t,
        s_mean=s,
        h_mean=h,
        K_field=K,
        Ss_field=Ss,
        tau_field=tau,
        H_field=Hf,
        time_units="yr",
        verbose=0,
    )

    _assert_all_finite(out["cons_scale"], out["gw_scale"])


def test_compute_scales_rank2_inputs_no_nan() -> None:
    """s_mean/h_mean rank-2 inputs should work and be finite."""
    B, H = 3, 5
    model = _default_model(cons_units="si", gw_units="si")

    t = _mk_time(B, H)[:, :, 0]  # (B,H)
    s, h = _mk_series(B, H)
    s2 = s[:, :, 0]  # (B,H)
    h2 = h[:, :, 0]  # (B,H)

    K = tf.ones((B, H, 1), tf.float32)
    Ss = tf.ones((B, H, 1), tf.float32) * 1e-5

    out = _compute_scales(
        model,
        t=t,
        s_mean=s2,
        h_mean=h2,
        K_field=K,
        Ss_field=Ss,
        time_units="yr",
        verbose=0,
    )

    _assert_all_finite(out["cons_scale"], out["gw_scale"])
