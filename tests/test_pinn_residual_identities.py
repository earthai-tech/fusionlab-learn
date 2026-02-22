# License: BSD-3-Clause

import numpy as np
import pytest
import tensorflow as tf


def _make_grids(B=1, T=8):
    """Create (t,x,y) grids of shape [B,T,1] suitable for tf.GradientTape."""
    t = tf.linspace(0.0, 1.0, T)[tf.newaxis, :, tf.newaxis]     # [1,T,1]
    x = tf.zeros_like(t)                                        # no x dependence
    y = tf.zeros_like(t)                                        # no y dependence
    return t, x, y


def test_consolidation_rate_identity_zero_residual():
    """
    Rate-form consolidation residual:
        R_cons = ds/dt + C * dh/dt

    For s(t) = -C * h(t) + c, R_cons == 0 identically.
    """
    B, T = 1, 16
    C = tf.constant(0.25, dtype=tf.float32)

    t, x, y = _make_grids(B=B, T=T)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([t, x, y])

        a = tf.constant(3.0, dtype=tf.float32)     # intercept for h
        b = tf.constant(-2.0, dtype=tf.float32)    # slope (dh/dt = b)
        h = a + b * t                              # linear in time
        c0 = tf.constant(1.0, dtype=tf.float32)    # intercept for s
        s = -C * h + c0

        ds_dt = tape.gradient(s, t)
        dh_dt = tape.gradient(h, t)

    del tape

    R_cons = ds_dt + C * dh_dt
    # residual should be (near) zero everywhere
    assert np.allclose(R_cons.numpy(), 0.0, rtol=1e-6, atol=1e-6)


def test_groundwater_zero_residual_laplacian_free_case():
    """
    Transient GW residual (canonical sign):
        R_gw = K * (d2h/dx2 + d2h/dy2) + Q - Ss * dh/dt

    Choose h(t) = a + b t (no x,y dependence) → Laplacian = 0.
    Set Q = Ss * b so R_gw = 0.
    """
    B, T = 1, 12
    K = tf.constant(1.0, dtype=tf.float32)   # arbitrary (won't matter since laplacian=0)
    Ss = tf.constant(5e-5, dtype=tf.float32)

    t, x, y = _make_grids(B=B, T=T)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([t, x, y])

        a = tf.constant(2.0, dtype=tf.float32)
        b = tf.constant(0.01, dtype=tf.float32)  # dh/dt = b
        h = a + b * t

        dh_dt = tape.gradient(h, t)
        dh_dx = tape.gradient(h, x)
        dh_dy = tape.gradient(h, y)

    d2h_dx2 = tf.gradients_function(lambda x_: tf.identity(dh_dx))(x) if hasattr(tf, "gradients_function") else tf.zeros_like(h)
    # Simpler: since h doesn't depend on x or y, first derivatives are zero → second derivatives are zero too.
    d2h_dx2 = tf.zeros_like(h)
    d2h_dy2 = tf.zeros_like(h)

    # Q chosen to cancel Ss * dh/dt
    # dh_dt is constant 'b'; we broadcast to the same shape as h
    Q = Ss * b

    R_gw = K * (d2h_dx2 + d2h_dy2) + Q - Ss * dh_dt
    assert np.allclose(R_gw.numpy(), 0.0, rtol=1e-6, atol=1e-6)
