# tests/test_consolidation_nan_safety.py
# -*- coding: utf-8 -*-
"""
NaN/Inf safety tests for consolidation helpers.

Contract:
- Bad (NaN/Inf) inputs must not make outputs NaN/Inf.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pytest
import tensorflow as tf


from fusionlab.nn.pinn._geoprior_maths import (  # type: ignore
    equilibrium_compaction_si,
    integrate_consolidation_mean,
    compute_consolidation_step_residual,
)

def _all_finite(x: Any) -> bool:
    x = tf.convert_to_tensor(x)
    ok = tf.reduce_all(tf.math.is_finite(x))
    return bool(ok.numpy())


def _assert_finite(*xs: Any) -> None:
    for x in xs:
        assert _all_finite(x)


def _mk_good(shape: Tuple[int, ...], v: float) -> tf.Tensor:
    return tf.ones(shape, tf.float32) * tf.constant(v, tf.float32)


def _mk_bad(shape: Tuple[int, ...], kind: str) -> tf.Tensor:
    arr = np.ones(shape, dtype=np.float32)

    if kind == "nan":
        arr.ravel()[0] = np.nan
    elif kind == "inf":
        arr.ravel()[0] = np.inf
    elif kind == "mix":
        n = arr.size
        k = max(1, n // 3)
        arr.ravel()[:k] = np.nan
        arr.ravel()[k : 2 * k] = np.inf
    else:
        raise ValueError("bad kind")

    return tf.constant(arr, tf.float32)


def _mk_base(B: int, H: int) -> dict[str, tf.Tensor]:
    out: dict[str, tf.Tensor] = {}
    out["h_mean"] = _mk_good((B, H, 1), 10.0)
    out["h_ref"] = _mk_good((B, H, 1), 12.0)
    out["Ss"] = _mk_good((B, H, 1), 1e-5)
    out["Hf"] = _mk_good((B, H, 1), 20.0)
    out["tau"] = _mk_good((B, H, 1), 3.15e7)
    out["tau_step"] = _mk_good((B, H - 1, 1), 3.15e7)
    out["s0"] = _mk_good((B, 1, 1), 0.0)
    out["s_state"] = _mk_good((B, H, 1), 0.02)
    out["dt"] = _mk_good((B, H, 1), 1.0)
    out["dt_step"] = _mk_good((B, H - 1, 1), 1.0)
    return out


# ------------------------------------------------------------
# equilibrium_compaction_si
# ------------------------------------------------------------
@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_eq_compaction_bad_h_mean_no_nan(bad_kind: str) -> None:
    B, H = 4, 6
    d = _mk_base(B, H)
    h_bad = _mk_bad((B, H, 1), bad_kind)

    s_eq = equilibrium_compaction_si(
        h_mean_si=h_bad,
        h_ref_si=d["h_ref"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        verbose=0,
    )
    _assert_finite(s_eq)


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_eq_compaction_bad_Ss_no_nan(bad_kind: str) -> None:
    B, H = 4, 6
    d = _mk_base(B, H)
    Ss_bad = _mk_bad((B, H, 1), bad_kind)

    s_eq = equilibrium_compaction_si(
        h_mean_si=d["h_mean"],
        h_ref_si=d["h_ref"],
        Ss_field=Ss_bad,
        H_field_si=d["Hf"],
        verbose=0,
    )
    _assert_finite(s_eq)


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_eq_compaction_bad_H_no_nan(bad_kind: str) -> None:
    B, H = 4, 6
    d = _mk_base(B, H)
    H_bad = _mk_bad((B, H, 1), bad_kind)

    s_eq = equilibrium_compaction_si(
        h_mean_si=d["h_mean"],
        h_ref_si=d["h_ref"],
        Ss_field=d["Ss"],
        H_field_si=H_bad,
        verbose=0,
    )
    _assert_finite(s_eq)


# ------------------------------------------------------------
# integrate_consolidation_mean
# ------------------------------------------------------------
@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_integrate_bad_dt_no_nan(bad_kind: str) -> None:
    B, H = 4, 6
    d = _mk_base(B, H)
    dt_bad = _mk_bad((B, H, 1), bad_kind)

    s_bar = integrate_consolidation_mean(
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=d["tau"],
        h_ref_si=d["h_ref"],
        s_init_si=d["s0"],
        dt=dt_bad,
        time_units="yr",
        method="exact",
        verbose=0,
    )
    _assert_finite(s_bar)


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_integrate_bad_tau_no_nan(bad_kind: str) -> None:
    B, H = 4, 6
    d = _mk_base(B, H)
    tau_bad = _mk_bad((B, H, 1), bad_kind)

    s_bar = integrate_consolidation_mean(
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=tau_bad,
        h_ref_si=d["h_ref"],
        s_init_si=d["s0"],
        dt=d["dt"],
        time_units="yr",
        method="exact",
        verbose=0,
    )
    _assert_finite(s_bar)


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_integrate_bad_s0_no_nan(bad_kind: str) -> None:
    B, H = 4, 6
    d = _mk_base(B, H)
    s0_bad = _mk_bad((B, 1, 1), bad_kind)

    s_bar = integrate_consolidation_mean(
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=d["tau"],
        h_ref_si=d["h_ref"],
        s_init_si=s0_bad,
        dt=d["dt"],
        time_units="yr",
        method="exact",
        verbose=0,
    )
    _assert_finite(s_bar)


# ------------------------------------------------------------
# compute_consolidation_step_residual
# ------------------------------------------------------------
@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_step_resid_bad_dt_no_nan(bad_kind: str) -> None:
    B, H = 4, 6
    d = _mk_base(B, H)
    dt_bad = _mk_bad((B, H - 1, 1), bad_kind)

    res = compute_consolidation_step_residual(
        s_state_si=d["s_state"],
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=d["tau_step"],
        h_ref_si=d["h_ref"],
        dt=dt_bad,
        time_units="yr",
        method="exact",
        verbose=0,
    )
    _assert_finite(res)


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_step_resid_bad_tau_no_nan(bad_kind: str) -> None:
    B, H = 4, 6
    d = _mk_base(B, H)
    tau_bad = _mk_bad((B, H, 1), bad_kind)

    res = compute_consolidation_step_residual(
        s_state_si=d["s_state"],
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=tau_bad,
        h_ref_si=d["h_ref"],
        dt=d["dt_step"],
        time_units="yr",
        method="exact",
        verbose=0,
    )
    _assert_finite(res)


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "mix"])
def test_step_resid_bad_s_state_no_nan(bad_kind: str) -> None:
    B, H = 4, 6
    d = _mk_base(B, H)
    s_bad = _mk_bad((B, H, 1), bad_kind)

    res = compute_consolidation_step_residual(
        s_state_si=s_bad,
        h_mean_si=d["h_mean"],
        Ss_field=d["Ss"],
        H_field_si=d["Hf"],
        tau_field=d["tau_step"],
        h_ref_si=d["h_ref"],
        dt=d["dt_step"],
        time_units="yr",
        method="exact",
        verbose=0,
    )
    _assert_finite(res)
