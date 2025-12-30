# test_mv_prior_loss.py
# Pytest coverage for compute_mv_prior_loss() NaN safety.
#
# Run:
#   pytest -q
#

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import tensorflow as tf 

from fusionlab.nn.pinn._geoprior_maths import compute_mv_prior_loss 


@dataclass
class DummyModel:
    """Minimal model stub for mv prior tests."""
    scaling_kwargs: dict[str, Any]

    # Optional parameters used by the mv prior.
    log_mv: Any | None = None
    _mv_fixed: Any | None = None
    gamma_w: Any | None = None


def _make_model(
    *,
    mv_units: str = "strict",
    mv_kind: str = "learnable",
    mv_value: float = 1e-7,
    gamma_w: float = 9810.0,
    mv_nan: bool = False,
    gw_missing: bool = False,
    mv_missing: bool = False,
) -> DummyModel:
    sk = {"mv_prior_units": mv_units}

    model = DummyModel(scaling_kwargs=sk)

    if not gw_missing:
        model.gamma_w = tf.constant(gamma_w, tf.float32)

    if mv_missing:
        return model

    if mv_kind == "learnable":
        v = np.nan if mv_nan else np.log(mv_value)
        model.log_mv = tf.Variable(v, dtype=tf.float32)
        model._mv_fixed = None
        return model

    # Fixed
    model.log_mv = None
    model._mv_fixed = tf.constant(mv_value, tf.float32)
    return model


def _assert_finite_scalar(x):
    """Assert tensor is scalar-like and finite."""
    x = tf.convert_to_tensor(x)
    # Force eager materialization for clearer errors.
    xv = x.numpy()
    assert np.ndim(xv) == 0, "loss must be a scalar"
    assert np.isfinite(xv), f"loss is not finite: {xv!r}"


def _bad_ss_tensor(kind: str, shape=(32, 16)):
    """Generate Ss_field with tricky values."""
    if kind == "positive":
        x = np.exp(np.random.randn(*shape)).astype("float32")
        return tf.constant(x)

    if kind == "zeros":
        return tf.zeros(shape, tf.float32)

    if kind == "negatives":
        x = -np.abs(np.random.randn(*shape)).astype("float32")
        return tf.constant(x)

    if kind == "has_nan":
        x = np.exp(np.random.randn(*shape)).astype("float32")
        x.ravel()[0] = np.nan
        return tf.constant(x)

    if kind == "has_inf":
        x = np.exp(np.random.randn(*shape)).astype("float32")
        x.ravel()[0] = np.inf
        x.ravel()[1] = -np.inf
        return tf.constant(x)

    if kind == "huge":
        x = (1e30 * np.ones(shape)).astype("float32")
        return tf.constant(x)

    if kind == "tiny":
        x = (1e-30 * np.ones(shape)).astype("float32")
        return tf.constant(x)

    raise ValueError(f"Unknown kind={kind!r}.")


@pytest.mark.parametrize(
    "ss_kind",
    [
        "positive",
        "zeros",
        "negatives",
        "has_nan",
        "has_inf",
        "huge",
        "tiny",
    ],
)
def test_mv_prior_loss_strict_is_finite(ss_kind):
    model = _make_model(
        mv_units="strict",
        mv_kind="learnable",
        mv_value=1e-7,
        gamma_w=9810.0,
    )
    Ss = _bad_ss_tensor(ss_kind)

    loss = compute_mv_prior_loss(
        model,
        Ss,
        alpha_disp=0.1,
        delta=1.0,
        eps=1e-12,
        verbose=0,
    )
    _assert_finite_scalar(loss)


@pytest.mark.parametrize(
    "ss_kind",
    [
        "positive",
        "zeros",
        "negatives",
        "has_nan",
        "has_inf",
        "huge",
        "tiny",
    ],
)
def test_mv_prior_loss_auto_is_finite(ss_kind):
    model = _make_model(
        mv_units="auto",
        mv_kind="learnable",
        mv_value=1e-7,
        gamma_w=9810.0,
    )
    Ss = _bad_ss_tensor(ss_kind)

    loss = compute_mv_prior_loss(
        model,
        Ss,
        alpha_disp=0.1,
        delta=1.0,
        eps=1e-12,
        verbose=0,
    )
    _assert_finite_scalar(loss)


def test_mv_prior_loss_handles_missing_gamma_w():
    model = _make_model(
        mv_units="auto",
        mv_kind="learnable",
        gw_missing=True,
    )
    Ss = _bad_ss_tensor("has_inf")

    loss = compute_mv_prior_loss(
        model,
        Ss,
        alpha_disp=0.25,
        delta=1.0,
        eps=1e-12,
        verbose=0,
    )
    _assert_finite_scalar(loss)


def test_mv_prior_loss_handles_missing_mv():
    model = _make_model(
        mv_units="strict",
        mv_missing=True,
        gw_missing=False,
    )
    Ss = _bad_ss_tensor("has_nan")

    loss = compute_mv_prior_loss(
        model,
        Ss,
        alpha_disp=0.25,
        delta=1.0,
        eps=1e-12,
        verbose=0,
    )
    _assert_finite_scalar(loss)


def test_mv_prior_loss_handles_nan_log_mv():
    model = _make_model(
        mv_units="auto",
        mv_kind="learnable",
        mv_nan=True,
        gamma_w=9810.0,
    )
    Ss = _bad_ss_tensor("positive")

    loss = compute_mv_prior_loss(
        model,
        Ss,
        alpha_disp=0.1,
        delta=1.0,
        eps=1e-12,
        verbose=0,
    )
    _assert_finite_scalar(loss)


def test_mv_prior_loss_grad_is_finite_wrt_log_mv():
    model = _make_model(
        mv_units="strict",
        mv_kind="learnable",
        mv_value=1e-7,
        gamma_w=9810.0,
    )
    Ss = _bad_ss_tensor("positive", shape=(64, 32))

    with tf.GradientTape() as tape:
        loss = compute_mv_prior_loss(
            model,
            Ss,
            alpha_disp=0.1,
            delta=1.0,
            eps=1e-12,
            verbose=0,
        )
    grad = tape.gradient(loss, [model.log_mv])[0]

    _assert_finite_scalar(loss)
    assert grad is not None, "gradient must not be None"
    gv = tf.convert_to_tensor(grad).numpy()
    assert np.ndim(gv) == 0, "grad(log_mv) must be scalar"
    assert np.isfinite(gv), f"grad is not finite: {gv!r}"


def test_mv_prior_loss_random_stress_no_nan():
    rng = np.random.default_rng(0)

    def _random_ss():
        x = rng.normal(size=(32, 32)).astype("float32")
        x = np.exp(x)
        # Inject occasional pathologies.
        u = rng.uniform()
        if u < 0.2:
            x.ravel()[0] = np.nan
        elif u < 0.4:
            x.ravel()[1] = np.inf
        elif u < 0.6:
            x.ravel()[2] = -np.inf
        elif u < 0.8:
            x = -np.abs(x)
        else:
            x.ravel()[3] = 0.0
        return tf.constant(x)

    for mv_units in ("strict", "auto"):
        model = _make_model(
            mv_units=mv_units,
            mv_kind="learnable",
            mv_value=1e-7,
            gamma_w=9810.0,
        )
        for _ in range(50):
            Ss = _random_ss()
            loss = compute_mv_prior_loss(
                model,
                Ss,
                alpha_disp=0.5,
                delta=0.0,  # edge: delta=0 should stay finite
                eps=1e-12,
                verbose=0,
            )
            _assert_finite_scalar(loss)
