# -*- coding: utf-8 -*-
"""
Derivative helpers for GeoPrior PINN blocks.

Goal: keep train_step() and _evaluate_physics_on_batch() consistent and DRY
for coordinate chain-rule conversions.

Conventions
-----------
- Raw autodiff derivatives are w.r.t. the coordinates tensor fed to call().
- This module converts those derivatives to **SI-consistent** forms:
  - time derivatives -> per-second
  - spatial derivatives -> per-meter (and per-meter^2 for second derivatives)

The helper is "conversion-aware":
- If coords are normalized and `scaling_kwargs` provides `coord_ranges_si`,
  those SI spans are used directly (t in seconds, x/y in meters).
- Otherwise, it falls back to `coord_ranges()` plus optional `deg_to_m()`
  and finally `rate_to_per_second()` for time.

It also returns `t_range_units_tf` (the *original* time span in `time_units`)
for Q conversion (because Q scaling typically expects the span in the same
time units used by the dataset, not seconds).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from collections.abc import Mapping

from ... import KERAS_DEPS
from .utils import get_sk, coord_ranges, deg_to_m
from .maths import rate_to_per_second

Tensor = KERAS_DEPS.Tensor
tf_constant = KERAS_DEPS.constant
tf_float32 = KERAS_DEPS.float32

def compute_head_pde_derivatives_raw(
    tape,
    coords,
    h_si,
    K_field,
    Ss_field,
):
    """AD derivatives in raw coord units."""
    dh_dcoords = tape.gradient(h_si, coords)
    if dh_dcoords is None:
        raise ValueError(
            "dh_dcoords is None: graph not connected "
            "to coords."
        )

    dh_dt_raw = dh_dcoords[..., 0:1]
    dh_dx_raw = dh_dcoords[..., 1:2]
    dh_dy_raw = dh_dcoords[..., 2:3]

    K_dh_dx = K_field * dh_dx_raw
    K_dh_dy = K_field * dh_dy_raw

    dKdhx_dcoords = tape.gradient(K_dh_dx, coords)
    dKdhy_dcoords = tape.gradient(K_dh_dy, coords)
    if (dKdhx_dcoords is None) or (dKdhy_dcoords is None):
        raise ValueError(
            "Second-order PDE gradients are None."
        )

    d_K_dh_dx_dx_raw = dKdhx_dcoords[..., 1:2]
    d_K_dh_dy_dy_raw = dKdhy_dcoords[..., 2:3]

    dK_dcoords = tape.gradient(K_field, coords)
    dSs_dcoords = tape.gradient(Ss_field, coords)
    if (dK_dcoords is None) or (dSs_dcoords is None):
        raise ValueError("K/Ss spatial grads are None.")

    dK_dx_raw = dK_dcoords[..., 1:2]
    dK_dy_raw = dK_dcoords[..., 2:3]
    dSs_dx_raw = dSs_dcoords[..., 1:2]
    dSs_dy_raw = dSs_dcoords[..., 2:3]

    return {
        "dh_dt_raw": dh_dt_raw,
        "d_K_dh_dx_dx_raw": d_K_dh_dx_dx_raw,
        "d_K_dh_dy_dy_raw": d_K_dh_dy_dy_raw,
        "dK_dx_raw": dK_dx_raw,
        "dK_dy_raw": dK_dy_raw,
        "dSs_dx_raw": dSs_dx_raw,
        "dSs_dy_raw": dSs_dy_raw,
    }

def ensure_si_derivative_frame(
    *,
    dh_dt_raw: Tensor,
    d_K_dh_dx_dx_raw: Tensor,
    d_K_dh_dy_dy_raw: Tensor,
    dK_dx_raw: Tensor,
    dK_dy_raw: Tensor,
    dSs_dx_raw: Tensor,
    dSs_dy_raw: Tensor,
    scaling_kwargs: Optional[Dict[str, Any]],
    time_units: Optional[str],
    coords_normalized: Optional[bool] = None,
    coords_in_degrees: Optional[bool] = None,
    eps: float = 1e-12,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    """Convert autodiff derivative tensors into SI-consistent derivatives.

    Parameters
    ----------
    dh_dt_raw, d_K_dh_dx_dx_raw, d_K_dh_dy_dy_raw, dK_dx_raw, dK_dy_raw,
    dSs_dx_raw, dSs_dy_raw : Tensor
        Raw autodiff derivatives w.r.t. the input coords tensor.
        Expected shape is (B,H,1) for all tensors.
    scaling_kwargs : dict or None
        Scaling parameters that describe coordinate normalization and units.
    time_units : str or None
        The dataset time unit for `t` (e.g., "year", "day", "second").
    coords_normalized, coords_in_degrees : bool, optional
        If provided, overrides the values in `scaling_kwargs`.

    Returns
    -------
    deriv : dict[str, Tensor]
        SI derivatives:
        - dh_dt   : m/s
        - d_K_dh_dx_dx, d_K_dh_dy_dy : (m/s)/m  (i.e., 1/s)
        - dK_dx, dK_dy : (m/s)/m
        - dSs_dx, dSs_dy : (1/m)/m
    meta : dict[str, Any]
        Metadata:
        - used_coord_ranges_si : bool
        - time_already_si : bool (True only if coord_ranges_si provided)
        - deg_already_applied : bool (True if ranges already converted to meters)
        - t_range_units_tf : Tensor or None (span in original time units)
    """
    sk = scaling_kwargs or {}

    coords_norm = (
        bool(get_sk(sk, "coords_normalized", default=False))
        if coords_normalized is None
        else bool(coords_normalized)
    )
    coords_deg = (
        bool(get_sk(sk, "coords_in_degrees", default=False))
        if coords_in_degrees is None
        else bool(coords_in_degrees)
    )

    dtype = dh_dt_raw.dtype
    eps_tf = tf_constant(float(eps), dtype)

    # Start from raw tensors.
    dh_dt = dh_dt_raw
    d_K_dh_dx_dx = d_K_dh_dx_dx_raw
    d_K_dh_dy_dy = d_K_dh_dy_dy_raw
    dK_dx, dK_dy = dK_dx_raw, dK_dy_raw
    dSs_dx, dSs_dy = dSs_dx_raw, dSs_dy_raw

    used_coord_ranges_si = False
    time_already_si = False
    deg_already_applied = False

    # For Q scaling, we still want t_range in the *original* time units.
    t_range_units_tf = None

    if coords_norm:
        # Always compute the original unit spans (for Q scaling).
        tR_u, xR_u, yR_u = coord_ranges(sk)
        if tR_u is None or xR_u is None or yR_u is None:
            raise ValueError("coords_normalized=True but coord_ranges missing.")
        t_range_units_tf = tf_constant(float(tR_u), dtype)

        # Prefer precomputed SI spans (t seconds; x/y meters).
        cr_si = get_sk(sk, "coord_ranges_si", default=None)
        if isinstance(cr_si, Mapping) and all(k in cr_si for k in ("t", "x", "y")):
            used_coord_ranges_si = True
            time_already_si = True
            deg_already_applied = True  # x/y already meters if coord_ranges_si exists

            tR = tf_constant(float(cr_si["t"]), dtype)
            xR = tf_constant(float(cr_si["x"]), dtype)
            yR = tf_constant(float(cr_si["y"]), dtype)

            # First derivative: /range
            dh_dt = dh_dt / (tR + eps_tf)

            # Second derivative: /range^2
            d_K_dh_dx_dx = d_K_dh_dx_dx / (xR * xR + eps_tf)
            d_K_dh_dy_dy = d_K_dh_dy_dy / (yR * yR + eps_tf)

            # Smoothness: /range
            dK_dx = dK_dx / (xR + eps_tf)
            dK_dy = dK_dy / (yR + eps_tf)
            dSs_dx = dSs_dx / (xR + eps_tf)
            dSs_dy = dSs_dy / (yR + eps_tf)

        else:
            # Fallback: use raw spans (in original coordinate units).
            tR = tf_constant(float(tR_u), dtype)
            xR = tf_constant(float(xR_u), dtype)
            yR = tf_constant(float(yR_u), dtype)

            dh_dt = dh_dt / (tR + eps_tf)
            d_K_dh_dx_dx = d_K_dh_dx_dx / (xR * xR + eps_tf)
            d_K_dh_dy_dy = d_K_dh_dy_dy / (yR * yR + eps_tf)

            dK_dx = dK_dx / (xR + eps_tf)
            dK_dy = dK_dy / (yR + eps_tf)
            dSs_dx = dSs_dx / (xR + eps_tf)
            dSs_dy = dSs_dy / (yR + eps_tf)

    # Degrees -> meters conversion (only if ranges were NOT already meters)
    if coords_deg and (not deg_already_applied):
        deg2m_x = deg_to_m("x", sk)  # m/deg
        deg2m_y = deg_to_m("y", sk)

        d_K_dh_dx_dx = d_K_dh_dx_dx / (deg2m_x * deg2m_x + eps_tf)
        d_K_dh_dy_dy = d_K_dh_dy_dy / (deg2m_y * deg2m_y + eps_tf)

        dK_dx = dK_dx / (deg2m_x + eps_tf)
        dK_dy = dK_dy / (deg2m_y + eps_tf)
        dSs_dx = dSs_dx / (deg2m_x + eps_tf)
        dSs_dy = dSs_dy / (deg2m_y + eps_tf)

    # Time derivative must be per-second for SI PDE.
    if not time_already_si:
        dh_dt = rate_to_per_second(dh_dt, time_units=time_units)

    deriv = {
        "dh_dt": dh_dt,
        "d_K_dh_dx_dx": d_K_dh_dx_dx,
        "d_K_dh_dy_dy": d_K_dh_dy_dy,
        "dK_dx": dK_dx,
        "dK_dy": dK_dy,
        "dSs_dx": dSs_dx,
        "dSs_dy": dSs_dy,
    }
    meta = {
        "used_coord_ranges_si": used_coord_ranges_si,
        "time_already_si": time_already_si,
        "deg_already_applied": deg_already_applied,
        "t_range_units_tf": t_range_units_tf,
    }
    return deriv, meta

