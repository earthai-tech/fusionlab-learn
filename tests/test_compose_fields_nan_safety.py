# tests/test_compose_fields_nan_safety.py
# -*- coding: utf-8 -*-
"""
NaN-safety tests for physics field composition.

Goal:
- Hard-bounds mode should *not* emit NaN/Inf even
  if raw inputs (coords / bases / MLP outputs)
  contain NaN/Inf.
- Soft-bounds mode should *fail fast* (raise)
  when NaN/Inf would otherwise propagate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


from fusionlab.nn.pinn._geoprior_maths import (  # noqa: E501
    compose_physics_fields,
    tau_phys_from_fields,
)


def _is_finite_tensor(x: Any) -> bool:
    x = tf.convert_to_tensor(x)
    ok = tf.reduce_all(tf.math.is_finite(x))
    return bool(ok.numpy())


def _assert_all_finite(*xs: Any) -> None:
    for x in xs:
        assert _is_finite_tensor(x)


def _mk_bad(shape: Tuple[int, ...], kind: str) -> tf.Tensor:
    if kind == "nan":
        a = np.full(shape, np.nan, dtype=np.float32)
        return tf.constant(a)

    if kind == "inf":
        a = np.full(shape, np.inf, dtype=np.float32)
        return tf.constant(a)

    if kind == "mix":
        a = np.ones(shape, dtype=np.float32)
        a = a.reshape(-1)
        if a.size >= 1:
            a[0] = np.nan
        if a.size >= 2:
            a[1] = np.inf
        if a.size >= 3:
            a[2] = -np.inf
        return tf.constant(a.reshape(shape))

    raise ValueError(f"Unknown kind={kind!r}")


class _ZeroMLP:
    """Return zeros with shape (..., 1)."""

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        _ = training
        x = tf.convert_to_tensor(x)
        return tf.zeros_like(x[..., :1])


class _NanMLP:
    """Return NaNs with shape (..., 1)."""

    def __call__(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        _ = training
        x = tf.convert_to_tensor(x)
        z = tf.zeros_like(x[..., :1])
        return z / z


@dataclass
class _MVConfig:
    initial_value: float = 1e-7


class _DummyModel:
    def __init__(
        self,
        *,
        bounds_mode: str,
        bounds: Optional[Dict[str, Any]] = None,
        mlp_kind: str = "zero",
    ) -> None:
        self.bounds_mode = bounds_mode
        self.scaling_kwargs = {"bounds": bounds or {}}

        if mlp_kind == "zero":
            self.K_coord_mlp = _ZeroMLP()
            self.Ss_coord_mlp = _ZeroMLP()
            self.tau_coord_mlp = _ZeroMLP()
        elif mlp_kind == "nan":
            self.K_coord_mlp = _NanMLP()
            self.Ss_coord_mlp = _NanMLP()
            self.tau_coord_mlp = _NanMLP()
        else:
            raise ValueError(f"mlp_kind={mlp_kind!r}")

        # Used by get_log_bounds heuristic (optional).
        self.gamma_w = tf.constant(9810.0, tf.float32)
        self.mv_config = _MVConfig()

        # Used by tau_phys_from_fields.
        self.use_effective_thickness = False
        self.Hd_factor = 1.0
        self.kappa_mode = "bar"

    def _kappa_value(self) -> tf.Tensor:
        return tf.constant(1.0, tf.float32)


def _default_bounds() -> Dict[str, Any]:
    # Linear bounds are fine; get_log_bounds will log them.
    sec_year = 31556952.0
    return {
        "K_min": 1e-12,
        "K_max": 1e-6,
        "Ss_min": 1e-10,
        "Ss_max": 1e-2,
        "tau_min": 7.0 * 86400.0,
        "tau_max": 300.0 * sec_year,
    }


@pytest.mark.parametrize(
    "badK,badSs,badH",
    [
        ("nan", "mix", "mix"),
        ("mix", "nan", "mix"),
        ("mix", "mix", "nan"),
        ("inf", "mix", "mix"),
        ("mix", "inf", "mix"),
        ("mix", "mix", "inf"),
    ],
)
def test_tau_phys_from_fields_no_nan_for_bad_inputs(
    badK: str,
    badSs: str,
    badH: str,
):
    """
    Desired contract:
    tau_phys_from_fields should not emit NaN/Inf.

    If this test fails, sanitize K/Ss/H inputs inside
    tau_phys_from_fields (e.g., finite_floor + eps).
    """
    model = _DummyModel(
        bounds_mode="hard",
        bounds=_default_bounds(),
    )

    K = _mk_bad((5, 1), badK)
    Ss = _mk_bad((5, 1), badSs)
    H = _mk_bad((5, 1), badH)

    tau_phys, Hd = tau_phys_from_fields(
        model,
        K,
        Ss,
        H,
        verbose=0,
    )

    _assert_all_finite(tau_phys, Hd)

    # Should also be strictly positive after floors.
    assert float(tf.reduce_min(tau_phys).numpy()) > 0.0
    assert float(tf.reduce_min(Hd).numpy()) > 0.0


def test_compose_fields_hard_bounds_nan_inputs_no_nan():
    """
    Hard bounds path must be robust:
    - rawK/rawSs may contain NaN/Inf
    - coords may contain NaN/Inf
    - MLP outputs may contain NaN/Inf
    bounded_exp + log bounds should prevent NaNs.
    """
    model = _DummyModel(
        bounds_mode="hard",
        bounds=_default_bounds(),
        mlp_kind="nan",
    )

    coords = _mk_bad((8, 3), "mix")

    # Bases are log-space tensors in your design.
    K_base = _mk_bad((8, 1), "mix")
    Ss_base = _mk_bad((8, 1), "mix")
    tau_base = tf.zeros((8, 1), tf.float32)

    H_si = tf.ones((8, 1), tf.float32)

    out = compose_physics_fields(
        model,
        coords_flat=coords,
        H_si=H_si,
        K_base=K_base,
        Ss_base=Ss_base,
        tau_base=tau_base,
        training=False,
        eps_KSs=1e-12,
        eps_tau=1e-6,
        verbose=0,
    )

    (
        K_field,
        Ss_field,
        tau_field,
        tau_phys,
        Hd_eff,
        delta_log_tau,
        logK,
        logSs,
        log_tau,
        log_tau_phys,
    ) = out

    _assert_all_finite(
        K_field,
        Ss_field,
        tau_field,
        tau_phys,
        Hd_eff,
        delta_log_tau,
        logK,
        logSs,
        log_tau,
        log_tau_phys,
    )


def test_compose_fields_soft_bounds_raises_on_nan_raw():
    """
    Soft mode currently asserts rawK/rawSs finite.
    This test enforces "fail fast" instead of
    silently producing NaNs.
    """
    model = _DummyModel(
        bounds_mode="soft",
        bounds=_default_bounds(),
        mlp_kind="zero",
    )

    coords = tf.zeros((4, 3), tf.float32)
    H_si = tf.ones((4, 1), tf.float32)

    K_base = _mk_bad((4, 1), "nan")
    Ss_base = tf.zeros((4, 1), tf.float32)
    tau_base = tf.zeros((4, 1), tf.float32)

    with pytest.raises(Exception):
        _ = compose_physics_fields(
            model,
            coords_flat=coords,
            H_si=H_si,
            K_base=K_base,
            Ss_base=Ss_base,
            tau_base=tau_base,
            training=False,
            eps_KSs=1e-12,
            eps_tau=1e-6,
            verbose=0,
        )


def test_compose_fields_soft_bounds_finite_inputs_no_nan():
    """Soft mode should work for normal finite inputs."""
    model = _DummyModel(
        bounds_mode="soft",
        bounds=_default_bounds(),
        mlp_kind="zero",
    )

    coords = tf.zeros((6, 3), tf.float32)
    H_si = tf.ones((6, 1), tf.float32)

    K_base = tf.fill((6, 1), np.log(1e-8)).numpy()
    K_base = tf.constant(K_base, tf.float32)

    Ss_base = tf.fill((6, 1), np.log(1e-5)).numpy()
    Ss_base = tf.constant(Ss_base, tf.float32)

    tau_base = tf.zeros((6, 1), tf.float32)

    out = compose_physics_fields(
        model,
        coords_flat=coords,
        H_si=H_si,
        K_base=K_base,
        Ss_base=Ss_base,
        tau_base=tau_base,
        training=False,
        eps_KSs=1e-12,
        eps_tau=1e-6,
        verbose=0,
    )

    K_field, Ss_field, tau_field, tau_phys, Hd_eff = out[:5]
    _assert_all_finite(K_field, Ss_field, tau_field, tau_phys, Hd_eff)


def test_compose_fields_tau_guard_prevents_overflow():
    """
    Even if delta_log_tau is huge, soft mode should
    clip log_tau_safe and avoid exp overflow.
    """
    model = _DummyModel(
        bounds_mode="soft",
        bounds=_default_bounds(),
        mlp_kind="zero",
    )

    # Override only tau MLP to return huge values.
    class _HugeTauMLP:
        def __call__(
            self,
            x: tf.Tensor,
            training: bool = False,
        ) -> tf.Tensor:
            _ = training
            x = tf.convert_to_tensor(x)
            return tf.fill(x[..., :1].shape, 1e9)

    model.tau_coord_mlp = _HugeTauMLP()

    coords = tf.zeros((3, 3), tf.float32)
    H_si = tf.ones((3, 1), tf.float32)

    K_base = tf.zeros((3, 1), tf.float32)
    Ss_base = tf.zeros((3, 1), tf.float32)
    tau_base = tf.zeros((3, 1), tf.float32)

    out = compose_physics_fields(
        model,
        coords_flat=coords,
        H_si=H_si,
        K_base=K_base,
        Ss_base=Ss_base,
        tau_base=tau_base,
        training=False,
        eps_KSs=1e-12,
        eps_tau=1e-6,
        verbose=0,
    )

    tau_field = out[2]
    log_tau = out[8]
    _assert_all_finite(tau_field, log_tau)
