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
    r"""
    Compute raw autodiff derivatives for the groundwater-flow PDE.
    
    This helper computes first- and second-order derivatives needed by
    the GeoPrior groundwater-flow residual using automatic
    differentiation (AD). All derivatives returned by this function are
    in the "raw" coordinate units of the ``coords`` tensor supplied to
    the model, without chain-rule rescaling to SI units.
    
    The returned tensors are intended to be passed to
    :func:`ensure_si_derivative_frame` to obtain SI-consistent forms
    (per-second time derivatives and per-meter spatial derivatives).
    
    Parameters
    ----------
    tape : tf.GradientTape
        Gradient tape that recorded operations connecting ``h_si``,
        ``K_field``, and ``Ss_field`` to ``coords``.
    
        Notes
        -----
        The tape must watch ``coords``. A common pattern is:
    
        .. code-block:: python
    
           with tf.GradientTape(persistent=True) as tape:
               tape.watch(coords)
               # compute h_si, K_field, Ss_field from model forward pass
    
    coords : Tensor
        Coordinate tensor used as the differentiation variable.
    
        Expected shape is ``(B, H, 3)`` where the last axis stores
        coordinates ordered as ``['t', 'x', 'y']``. The order must be
        consistent with how ``dh_dcoords[..., i]`` is interpreted.
    
        Notes
        -----
        * ``coords`` may be normalized or unnormalized.
        * Units may be dataset units or degrees/meters. This function
          does not apply any unit conversion.
    h_si : Tensor
        Hydraulic head in SI-consistent units (or the internal head unit
        chosen by the pipeline).
    
        Expected shape is ``(B, H, 1)``. The tensor must be connected to
        ``coords`` through the computation graph, otherwise AD gradients
        will be None.
    K_field : Tensor
        Hydraulic conductivity field :math:`K` evaluated on the same
        batch and horizon grid.
    
        Expected shape is ``(B, H, 1)``. The tensor must be connected to
        ``coords`` for spatial gradients to be defined.
    Ss_field : Tensor
        Specific storage field :math:`S_s` evaluated on the same batch
        and horizon grid.
    
        Expected shape is ``(B, H, 1)``. The tensor must be connected to
        ``coords`` for spatial gradients to be defined.
    
    Returns
    -------
    grads : dict of str to Tensor
        Dictionary containing raw derivatives in the coordinate units
        of ``coords``. Keys include:
    
        ``'dh_dt_raw'``
            Raw time derivative :math:`\partial h / \partial t_{raw}`.
    
        ``'d_K_dh_dx_dx_raw'``
            Raw x-direction divergence term:
            :math:`\partial_x (K \partial_x h)` in raw coord units.
    
        ``'d_K_dh_dy_dy_raw'``
            Raw y-direction divergence term:
            :math:`\partial_y (K \partial_y h)` in raw coord units.
    
        ``'dK_dx_raw'``, ``'dK_dy_raw'``
            Raw spatial gradients of :math:`K` w.r.t. x and y.
    
        ``'dSs_dx_raw'``, ``'dSs_dy_raw'``
            Raw spatial gradients of :math:`S_s` w.r.t. x and y.
    
        Notes
        -----
        All tensors are expected to have shape ``(B, H, 1)``. No scaling
        by coordinate ranges is applied here.
    
    Raises
    ------
    ValueError
        If any required gradient is None, indicating the computation
        graph is not connected to ``coords`` or the tape did not watch
        ``coords``.
    ValueError
        If any second-order gradients required for the divergence form
        are None.
    
    Notes
    -----
    Groundwater-flow residual context
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This function provides building blocks for the divergence form used
    in the groundwater-flow residual:
    
    .. math::
    
       R_{gw} = S_s \\, \partial_t h
                - \nabla \cdot (K \\, \nabla h) - Q
    
    The divergence term in 2D can be expressed as:
    
    .. math::
    
       \nabla \cdot (K \nabla h)
       = \partial_x (K \partial_x h)
         + \partial_y (K \partial_y h)
    
    This helper returns the two directional components separately so
    that downstream code can apply unit conversions and scaling
    consistently.
    
    Implementation details
    ~~~~~~~~~~~~~~~~~~~~~~
    * First-order gradients are computed as:
    
      .. math::
    
         \nabla_{coords} h = \frac{\partial h}{\partial coords}
    
      and then split by coordinate axis index.
    * Second-order divergence terms are computed by differentiating the
      products ``K_field * dh_dx_raw`` and ``K_field * dh_dy_raw`` with
      respect to ``coords`` and extracting the x and y components.
    
    Examples
    --------
    Compute raw derivatives and then convert to SI:
    
    >>> from fusionlab.nn.pinn.geoprior.derivatives import ( 
    ...     compute_head_pde_derivatives_raw
    ...     )
    >>> with tf.GradientTape(persistent=True) as tape:
    ...     tape.watch(coords)
    ...     # forward pass returns h_si, K_field, Ss_field
    ...     raw = compute_head_pde_derivatives_raw(
    ...         tape=tape,
    ...         coords=coords,
    ...         h_si=h_si,
    ...         K_field=K_field,
    ...         Ss_field=Ss_field,
    ...     )
    >>> deriv, meta = ensure_si_derivative_frame(
    ...     dh_dt_raw=raw["dh_dt_raw"],
    ...     d_K_dh_dx_dx_raw=raw["d_K_dh_dx_dx_raw"],
    ...     d_K_dh_dy_dy_raw=raw["d_K_dh_dy_dy_raw"],
    ...     dK_dx_raw=raw["dK_dx_raw"],
    ...     dK_dy_raw=raw["dK_dy_raw"],
    ...     dSs_dx_raw=raw["dSs_dx_raw"],
    ...     dSs_dy_raw=raw["dSs_dy_raw"],
    ...     scaling_kwargs=scaling_kwargs,
    ...     time_units=time_units,
    ... )
    
    See Also
    --------
    ensure_si_derivative_frame
        Convert raw derivatives to SI-consistent derivatives.
    
    fusionlab.nn.pinn.geoprior.losses
        Physics losses that consume SI-consistent PDE derivatives.
    
    References
    ----------
    .. [1] Bear, J. Dynamics of Fluids in Porous Media. Dover
       Publications, 1988.
    
    .. [2] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
       Physics-informed neural networks: A deep learning framework
       for solving forward and inverse problems involving nonlinear
       partial differential equations. Journal of Computational
       Physics, 2019.
    """

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
    r"""
    Convert autodiff derivative tensors into SI-consistent derivatives.
    
    This helper is the canonical "chain-rule bridge" between raw
    autodiff gradients taken with respect to the model input
    ``coords`` tensor and the SI-consistent derivatives required by
    GeoPrior physics losses.
    
    It is designed to keep ``train_step()`` and
    ``_evaluate_physics_on_batch()`` consistent and DRY:
    
    * Raw derivatives are w.r.t. the coords tensor passed to ``call()``.
    * If coords are normalized, derivatives are rescaled by coordinate
      spans (and spans squared for second derivatives).
    * If spatial coords are degrees, spatial derivatives are converted
      to per-meter forms using a degrees-to-meters factor.
    * Time derivatives are converted to per-second using ``time_units``
      unless SI time spans are already supplied.
    
    Parameters
    ----------
    dh_dt_raw : Tensor
        Raw autodiff time derivative of head w.r.t. the first coord
        axis, i.e. :math:`\partial h / \partial t_{raw}`.
    
        Expected shape is ``(B, H, 1)``. The tensor is assumed to be
        computed w.r.t. the coords tensor fed to ``call()``.
    d_K_dh_dx_dx_raw : Tensor
        Raw second-order x-direction PDE term computed as the x
        component of :math:`\nabla \cdot (K \nabla h)` in raw coord
        units. Conceptually:
    
        .. math::
    
           \partial_x (K \partial_x h)
    
        Expected shape is ``(B, H, 1)``.
    d_K_dh_dy_dy_raw : Tensor
        Raw second-order y-direction PDE term in raw coord units:
    
        .. math::
    
           \partial_y (K \partial_y h)
    
        Expected shape is ``(B, H, 1)``.
    dK_dx_raw : Tensor
        Raw spatial gradient of :math:`K` in the x direction in raw
        coord units.
    
        Expected shape is ``(B, H, 1)``.
    dK_dy_raw : Tensor
        Raw spatial gradient of :math:`K` in the y direction in raw
        coord units.
    
        Expected shape is ``(B, H, 1)``.
    dSs_dx_raw : Tensor
        Raw spatial gradient of :math:`S_s` in the x direction in raw
        coord units.
    
        Expected shape is ``(B, H, 1)``.
    dSs_dy_raw : Tensor
        Raw spatial gradient of :math:`S_s` in the y direction in raw
        coord units.
    
        Expected shape is ``(B, H, 1)``.
    scaling_kwargs : dict or None
        Scaling and convention payload (resolved config) that
        describes coordinate normalization and units.
    
        This function primarily consults the following keys:
    
        * ``coords_normalized`` : bool
            If True, apply span-based chain-rule scaling.
        * ``coord_ranges`` : dict with keys {'t','x','y'}
            Original coordinate spans in dataset units.
            Required when ``coords_normalized=True``.
        * ``coord_ranges_si`` : dict with keys {'t','x','y'}
            Coordinate spans in SI units (t in seconds, x/y in meters).
            If present, this is preferred over ``coord_ranges``.
        * ``coords_in_degrees`` : bool
            If True, spatial axes are in degrees and must be converted
            to meters if SI spans were not already provided.
    
        Notes
        -----
        The payload is treated as an audit-friendly source-of-truth.
        When residual magnitudes look inconsistent across sites, inspect
        the resolved coordinate spans first.
    time_units : str or None
        Dataset time unit name for the t axis, used to convert the time
        derivative to per-second when SI time spans are not already
        provided.
    
        Typical values include ``'year'``, ``'day'``, or ``'second'``.
    coords_normalized : bool, optional
        Optional override for ``scaling_kwargs['coords_normalized']``.
        If provided, this value takes precedence over the payload.
    coords_in_degrees : bool, optional
        Optional override for ``scaling_kwargs['coords_in_degrees']``.
        If provided, this value takes precedence over the payload.
    eps : float, default 1e-12
        Numerical stabilizer added to denominators to avoid division by
        zero when spans are extremely small or missing.
    
    Returns
    -------
    deriv : dict of str to Tensor
        Dictionary of SI-consistent derivative tensors. Keys include:
    
        ``'dh_dt'``
            Time derivative converted to per-second:
            :math:`\partial h / \partial t` in SI time.
    
        ``'d_K_dh_dx_dx'`` and ``'d_K_dh_dy_dy'``
            Spatial second-derivative PDE terms converted to per-meter
            squared scaling (via span squared), consistent with the
            divergence form.
    
        ``'dK_dx'``, ``'dK_dy'``, ``'dSs_dx'``, ``'dSs_dy'``
            Spatial gradients converted to per-meter scaling.
    
        Notes
        -----
        The exact physical units of the returned tensors depend on the
        units of ``h_si`` and the representation of ``K`` and ``S_s``.
        The purpose of this function is to enforce correct coordinate
        scaling (per-second, per-meter, per-meter squared).
    meta : dict of str to Any
        Metadata describing which conversion path was used:
    
        ``'used_coord_ranges_si'`` : bool
            True if SI spans were taken from ``coord_ranges_si``.
        ``'time_already_si'`` : bool
            True if SI time span (seconds) was provided.
        ``'deg_already_applied'`` : bool
            True if x/y spans were already in meters and no degree-to-
            meter correction was applied.
        ``'t_range_units_tf'`` : Tensor or None
            The original time span in dataset time units (not seconds).
            This is returned for downstream Q scaling logic, where some
            conversions are defined in dataset time units rather than
            SI seconds.
    
    Notes
    -----
    Chain-rule scaling for normalized coordinates
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If normalized coordinates are defined as:
    
    .. math::
    
       u' = (u - u_0) / \Delta u
    
    then derivatives transform as:
    
    .. math::
    
       \frac{\partial}{\partial u}
       = \frac{1}{\Delta u} \frac{\partial}{\partial u'}
    
    and second derivatives as:
    
    .. math::
    
       \frac{\partial^2}{\partial u^2}
       = \frac{1}{(\Delta u)^2} \frac{\partial^2}{\partial (u')^2}
    
    This function applies these rules using either ``coord_ranges_si``
    (preferred) or ``coord_ranges`` plus unit conversion.
    
    Degrees to meters conversion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If spatial coords are degrees (longitude/latitude), the function
    converts spatial derivative scaling using a degrees-to-meters factor
    derived from the scaling payload. This is only applied when SI spans
    were not already provided.
    
    Examples
    --------
    Convert derivatives for normalized coords with SI spans:
    
    >>> from fusionlab.nn.pinn.geoprior.derivatives import ( 
    ...    ensure_si_derivative_frame 
    ...    )
    >>> deriv, meta = ensure_si_derivative_frame(
    ...     dh_dt_raw=dh_dt_raw,
    ...     d_K_dh_dx_dx_raw=dKdhx_dx_raw,
    ...     d_K_dh_dy_dy_raw=dKdhy_dy_raw,
    ...     dK_dx_raw=dK_dx_raw,
    ...     dK_dy_raw=dK_dy_raw,
    ...     dSs_dx_raw=dSs_dx_raw,
    ...     dSs_dy_raw=dSs_dy_raw,
    ...     scaling_kwargs={
    ...         "coords_normalized": True,
    ...         "coord_ranges": {"t": 7.0, "x": 4.4e4, "y": 3.9e4},
    ...         "coord_ranges_si": {
    ...             "t": 2.2e8, "x": 4.4e4, "y": 3.9e4
    ...         },
    ...     },
    ...     time_units="year",
    ... )
    >>> bool(meta["used_coord_ranges_si"])
    True
    
    Fallback when SI spans are absent (time converted using time_units):
    
    >>> deriv, meta = ensure_si_derivative_frame(
    ...     dh_dt_raw=dh_dt_raw,
    ...     d_K_dh_dx_dx_raw=dKdhx_dx_raw,
    ...     d_K_dh_dy_dy_raw=dKdhy_dy_raw,
    ...     dK_dx_raw=dK_dx_raw,
    ...     dK_dy_raw=dK_dy_raw,
    ...     dSs_dx_raw=dSs_dx_raw,
    ...     dSs_dy_raw=dSs_dy_raw,
    ...     scaling_kwargs={
    ...         "coords_normalized": True,
    ...         "coord_ranges": {"t": 7.0, "x": 4.4e4, "y": 3.9e4},
    ...     },
    ...     time_units="year",
    ... )
    >>> bool(meta["time_already_si"])
    False
    
    See Also
    --------
    compute_head_pde_derivatives_raw
        Compute raw autodiff derivatives w.r.t. input coords.
    
    fusionlab.nn.pinn.geoprior.maths.rate_to_per_second
        Convert a time rate from dataset units to per-second.
    
    fusionlab.nn.pinn.geoprior.utils.coord_ranges
        Extract coordinate spans from a scaling payload.
    
    fusionlab.nn.pinn.geoprior.utils.deg_to_m
        Convert degrees to meters scaling for spatial axes.
    
    References
    ----------
    .. [1] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
       Physics-informed neural networks: A deep learning framework
       for solving forward and inverse problems involving nonlinear
       partial differential equations. Journal of Computational
       Physics, 2019.
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

