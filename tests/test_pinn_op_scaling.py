# -*- coding: utf-8 -*-
# License: BSD-3-Clause

import numpy as np
import pytest

from fusionlab.nn.pinn.op import (
    positive,
    default_scales,
    scale_residual,
    _SMALL,
)

import tensorflow as tf


def test_positive_softplus_eps():
    x = tf.constant([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=tf.float32)
    y = positive(x)  # softplus(x) + eps
    y_np = y.numpy()
    # strictly positive
    assert np.all(y_np > 0.0)
    # for x=0, output is softplus(0)=~0.693... + eps
    assert np.isclose(y_np[2], tf.math.softplus(0.0).numpy() + _SMALL, rtol=1e-6, atol=1e-7)


def test_default_scales_shapes_and_values():
    # simple 3D (B,T,1) tensors
    B, T = 2, 5
    h = tf.ones([B, T, 1], dtype=tf.float32) * 2.0     # |h| mean = 2
    s = tf.ones([B, T, 1], dtype=tf.float32) * 5.0     # |s| mean = 5
    dt = tf.ones([B, T, 1], dtype=tf.float32) * 0.25   # |dt| mean = 0.25
    Ss = tf.ones([], dtype=tf.float32) * 1e-4          # |Ss| mean = 1e-4

    scales = default_scales(h=h, s=s, dt=dt, Ss=Ss)
    for k in ["h_ref", "s_ref", "dt_ref", "gw_scale", "cons_scale"]:
        assert k in scales

    # expected references (with +_SMALL inside function)
    h_ref = (tf.reduce_mean(tf.abs(h)) + _SMALL).numpy()
    s_ref = (tf.reduce_mean(tf.abs(s)) + _SMALL).numpy()
    dt_ref = (tf.reduce_mean(tf.abs(dt)) + _SMALL).numpy()
    Ss_ref = (tf.abs(Ss) + _SMALL).numpy()

    # gw_scale ≈ Ss_ref * h_ref / dt_ref
    gw_scale = scales["gw_scale"].numpy()
    assert np.isclose(gw_scale, Ss_ref * h_ref / dt_ref, rtol=1e-6, atol=1e-12)

    # cons_scale ≈ s_ref / dt_ref
    cons_scale = scales["cons_scale"].numpy()
    assert np.isclose(cons_scale, s_ref / dt_ref, rtol=1e-6, atol=1e-12)


def test_scale_residual_basic_and_invariance():
    residual = tf.constant([1.0, 2.0, 4.0], dtype=tf.float32)
    scale = tf.constant(2.0, dtype=tf.float32)

    dimless = scale_residual(residual, scale).numpy()
    expected = (residual.numpy()) / (scale.numpy() + _SMALL)
    assert np.allclose(dimless, expected, rtol=1e-6, atol=1e-12)

    # invariance: multiply both residual and scale by the same factor → ratio unchanged (up to eps effects)
    alpha = 7.5
    dimless2 = scale_residual(alpha * residual, alpha * scale).numpy()
    assert np.allclose(dimless, dimless2, rtol=1e-6, atol=1e-6)
