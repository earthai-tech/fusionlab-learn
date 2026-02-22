# -*- coding: utf-8 -*-
import numpy as np
import pytest
import tensorflow as tf

from fusionlab.nn.pinn._geoprior_maths import compute_gw_flow_residual


class DummyModel:
    def __init__(self, active=True):
        self.pde_modes_active = {"gw_flow"} if active else set()


def _finite0(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 (matches the common _finite_or_zero contract)."""
    return np.where(np.isfinite(x), x, 0.0)


def _make_base_tensors(B=2, H=3, dtype=tf.float32):
    dh_dt = tf.constant(np.full((B, H, 1), 4.0, np.float32), dtype=dtype)
    dKdx  = tf.constant(np.full((B, H, 1), 1.0, np.float32), dtype=dtype)
    dKdy  = tf.constant(np.full((B, H, 1), 2.0, np.float32), dtype=dtype)
    Ss    = tf.constant(np.full((B, H, 1), 3.0, np.float32), dtype=dtype)
    return dh_dt, dKdx, dKdy, Ss


def test_gw_flow_inactive_returns_zeros_even_if_inputs_nonfinite():
    model = DummyModel(active=False)

    dh_dt = tf.constant([[[np.nan], [np.inf], [-np.inf]]], dtype=tf.float32)
    dKdx  = tf.ones_like(dh_dt)
    dKdy  = tf.ones_like(dh_dt)
    Ss    = tf.ones_like(dh_dt)

    out = compute_gw_flow_residual(model, dh_dt, dKdx, dKdy, Ss, Q=np.nan)

    np.testing.assert_allclose(out.numpy(), np.zeros_like(dh_dt.numpy()))
    assert np.isfinite(out.numpy()).all()


def test_gw_flow_active_no_Q_matches_expected():
    model = DummyModel(active=True)
    dh_dt, dKdx, dKdy, Ss = _make_base_tensors()

    out = compute_gw_flow_residual(model, dh_dt, dKdx, dKdy, Ss, Q=None)

    # expected: Ss*dh_dt - (dKdx + dKdy) - 0
    expected = 3.0 * 4.0 - (1.0 + 2.0)  # = 9
    np.testing.assert_allclose(out.numpy(), expected * np.ones((2, 3, 1)), rtol=1e-6, atol=1e-6)


def test_gw_flow_active_scalar_Q_broadcasts():
    model = DummyModel(active=True)
    dh_dt, dKdx, dKdy, Ss = _make_base_tensors()

    out = compute_gw_flow_residual(model, dh_dt, dKdx, dKdy, Ss, Q=5.0)

    # expected: 12 - 3 - 5 = 4
    expected = 4.0
    np.testing.assert_allclose(out.numpy(), expected * np.ones((2, 3, 1)), rtol=1e-6, atol=1e-6)


def test_gw_flow_active_rank1_Q_broadcasts_over_batch_and_last_dim():
    model = DummyModel(active=True)
    B, H = 2, 3
    dh_dt, dKdx, dKdy, Ss = _make_base_tensors(B=B, H=H)

    Q = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)  # (H,)

    out = compute_gw_flow_residual(model, dh_dt, dKdx, dKdy, Ss, Q=Q)

    base = 3.0 * 4.0 - (1.0 + 2.0)  # 9
    expected = np.zeros((B, H, 1), np.float32)
    expected[:, :, 0] = base - np.array([1.0, 2.0, 3.0], np.float32)

    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_gw_flow_active_rank2_Q_broadcasts_to_rank3():
    model = DummyModel(active=True)
    B, H = 2, 3
    dh_dt, dKdx, dKdy, Ss = _make_base_tensors(B=B, H=H)

    Q = tf.constant([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=tf.float32)  # (B,H)

    out = compute_gw_flow_residual(model, dh_dt, dKdx, dKdy, Ss, Q=Q)

    base = 3.0 * 4.0 - (1.0 + 2.0)  # 9
    expected = np.zeros((B, H, 1), np.float32)
    expected[:, :, 0] = base - np.array([[1.0, 2.0, 3.0],
                                         [4.0, 5.0, 6.0]], np.float32)

    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_gw_flow_nan_inf_safe_contract_output_is_finite_and_matches_finite0_rule():
    """
    Contract test:
    - any NaN/Inf in inputs is treated as 0
    - output must be finite
    """
    model = DummyModel(active=True)
    B, H = 2, 3

    dh_dt = np.full((B, H, 1), 4.0, np.float32)
    dKdx  = np.full((B, H, 1), 1.0, np.float32)
    dKdy  = np.full((B, H, 1), 2.0, np.float32)
    Ss    = np.full((B, H, 1), 3.0, np.float32)
    Q     = np.full((B, H, 1), 0.5, np.float32)

    # inject non-finite
    dh_dt[0, 1, 0] = np.nan
    dKdx[1, 0, 0]  = np.inf
    dKdy[0, 2, 0]  = -np.inf
    Ss[1, 2, 0]    = np.nan
    Q[0, 0, 0]     = np.inf

    out = compute_gw_flow_residual(
        model,
        tf.constant(dh_dt),
        tf.constant(dKdx),
        tf.constant(dKdy),
        tf.constant(Ss),
        Q=tf.constant(Q),
    ).numpy()

    # expected under finite->0 rule
    dh_dt0 = _finite0(dh_dt)
    dKdx0  = _finite0(dKdx)
    dKdy0  = _finite0(dKdy)
    Ss0    = _finite0(Ss)
    Q0     = _finite0(Q)

    expected = Ss0 * dh_dt0 - (dKdx0 + dKdy0) - Q0

    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
