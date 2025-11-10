# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from fusionlab.nn.pinn._geoprior_subnet import GeoPriorSubsNet
from fusionlab.params import LearnableMV, LearnableKappa


# ------------ helpers ---------------------------------------------------------
def _make_batch(B=4, T=6, H=3, sd=2, dd=3, fd=2):
    """
    Build one toy batch with the exact dict structure GeoPriorSubsNet expects.
    mode='tft_like' => future_features has shape (B, T+H, fd).
    """
    static = tf.zeros([B, sd], dtype=tf.float32)
    dyn = tf.random.normal([B, T, dd], dtype=tf.float32)
    fut = tf.random.normal([B, T + H, fd], dtype=tf.float32)  # tft_like routing
    coords = tf.random.normal([B, H, 3], dtype=tf.float32)    # (t, x, y)
    H_field = tf.ones([B, H, 1], dtype=tf.float32) * 10.0

    inputs = {
        "static_features": static,
        "dynamic_features": dyn,
        "future_features": fut,
        "coords": coords,
        "H_field": H_field,
    }
    # data heads: (subs, gwl) each (B,H,1)
    targets = {
        "subs_pred": tf.zeros([B, H, 1], dtype=tf.float32),
        "gwl_pred":  tf.zeros([B, H, 1], dtype=tf.float32),
    }
    return inputs, targets


def _make_model(T=6, H=3, sd=2, dd=3, fd=2, pde_mode="none"):
    """Small GeoPriorSubsNet suitable for quick unit tests."""
    return GeoPriorSubsNet(
        static_input_dim=sd,
        dynamic_input_dim=dd,
        future_input_dim=fd,
        output_subsidence_dim=1,
        output_gwl_dim=1,
        forecast_horizon=H,
        max_window_size=T,
        mode="tft_like",               # <-- important: future_features is T+H
        quantiles=None,                # means only (simpler shapes)
        pde_mode=pde_mode,
        mv=LearnableMV(1e-7),
        kappa=LearnableKappa(1.0),
        architecture_config={"encoder_type": "hybrid"},
        name="GeoPriorSubsNet_Test",
    )


def _train_n_steps(model, inputs, targets, n=5):
    for _ in range(int(n)):
        model.train_on_batch(inputs, targets)

def _get_mv(model) -> float:
    return float(model.current_mv().numpy())

def _get_kappa(model) -> float:
    return float(model.current_kappa().numpy())
# ------------ tests -----------------------------------------------------------
def test_mv_moves_when_lambda_mv_positive():
    """
    With the storage-identity penalty active (lambda_mv > 0),
    m_v should get a nonzero gradient and change from its init.
    This does NOT require PDE residuals, so pde_mode='none' is fine.
    """
    tf.random.set_seed(0)
    np.random.seed(0)

    model = _make_model(pde_mode="none")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss={"subs_pred": "mse", "gwl_pred": "mse"},
        lambda_cons=0.0,
        lambda_gw=0.0,
        lambda_prior=0.0,
        lambda_smooth=0.0,
        lambda_mv=1.0,          # <-- turn on storage-identity penalty
        mv_lr_mult=50.0,        # give it a bit more bite
        kappa_lr_mult=1.0,
    )

    inputs, targets = _make_batch()
    mv0 = _get_mv(model)

    _train_n_steps(model, inputs, targets, n=8)
    mv1 = _get_mv(model)

    assert not np.isclose(mv1, mv0, rtol=0.0, atol=1e-12), f"m_v did not change: {mv0} -> {mv1}"


def test_kappa_moves_when_prior_active():
    """
    With the τ-consistency prior active (lambda_prior > 0),
    κ should move even without PDE terms (it depends on K, Ss, τ, H).
    """
    tf.random.set_seed(1)
    np.random.seed(1)

    model = _make_model(pde_mode="none")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss={"subs_pred": "mse", "gwl_pred": "mse"},
        lambda_cons=0.0,
        lambda_gw=0.0,
        lambda_prior=1.0,      
        lambda_smooth=0.0,
        lambda_mv=0.0,
        mv_lr_mult=1.0,
        kappa_lr_mult=20.0,     
    )

    inputs, targets = _make_batch()
    k0 = _get_kappa(model)

    _train_n_steps(model, inputs, targets, n=8)
    k1 = _get_kappa(model)

    assert not np.isclose(k1, k0, rtol=0.0, atol=1e-12), f"kappa did not change: {k0} -> {k1}"


def test_scalars_frozen_when_no_paths():
    """
    If pde_mode='none' and all physics penalties are off,
    scalars should not move (no gradient path).
    """
    tf.random.set_seed(2)
    np.random.seed(2)

    model = _make_model(pde_mode="none")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss={"subs_pred": "mse", "gwl_pred": "mse"},
        lambda_cons=0.0,
        lambda_gw=0.0,
        lambda_prior=0.0,
        lambda_smooth=0.0,
        lambda_mv=0.0,
        mv_lr_mult=100.0,     # should not matter
        kappa_lr_mult=100.0,  # should not matter
    )

    inputs, targets = _make_batch()
    mv0, k0 = _get_mv(model), _get_kappa(model)

    _train_n_steps(model, inputs, targets, n=8)
    mv1, k1 = _get_mv(model), _get_kappa(model)

    assert np.isclose(mv1, mv0, rtol=0.0, atol=1e-12), f"m_v changed unexpectedly: {mv0} -> {mv1}"
    assert np.isclose(k1, k0, rtol=0.0, atol=1e-12), f"kappa changed unexpectedly: {k0} -> {k1}"
