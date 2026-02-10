# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

r"""
fusionlab.nn.pinn.geoprior.doc

Shared documentation fragments for GeoPrior PINN models.

This module stores:
* Parameter documentation components (re-usable).
* Long-form docstring templates (format-ready).
"""

from __future__ import annotations

from ....api.docs import DocstringComponents, _halnet_core_params


# ---------------------------------------------------------------------
# GeoPrior-specific parameter docs (only what base docs do not cover).
# ---------------------------------------------------------------------
_geoprior_core_params = dict(
    output_subsidence_dim=r"""
output_subsidence_dim : int, default 1
    Number of subsidence series emitted at each forecast step.

    This is the dimensionality of the *subsidence* output head
    per horizon step. If ``forecast_horizon = H``, the raw model
    output for subsidence has shape ``(B, H, output_subsidence_dim)``
    (or ``(B, H, Q, output_subsidence_dim)`` when quantiles are
    enabled).

    Typical use-cases
    -----------------
    * ``1``: a single subsidence target per spatial sample.
    * ``n_benchmarks``: multiple benchmarks (e.g., leveling points)
      predicted jointly.
    * Multi-task: different subsidence variants (e.g., cumulative and
      incremental) if your dataset explicitly defines them as distinct
      targets.

    Notes
    -----
    The physics terms are usually computed on the *mean* subsidence
    path implied by the head forecast and the consolidation closure.
    Additional subsidence channels are still supported, but developers
    should ensure the closure and target formatting are consistent with
    the chosen multi-output convention.
""",
    output_gwl_dim=r"""
output_gwl_dim : int, default 1
    Number of groundwater-level (GWL) series emitted at each
    forecast step.

    This sets the dimensionality of the *GWL/head* output head per
    horizon step. For ``forecast_horizon = H``, the raw output for
    GWL has shape ``(B, H, output_gwl_dim)`` (or with quantiles
    ``(B, H, Q, output_gwl_dim)``).

    Typical use-cases
    -----------------
    * ``1``: single well / single aquifer representation.
    * ``n_aquifers``: multi-aquifer heads predicted jointly.
    * ``n_wells``: multiple wells predicted jointly for the same
      spatial sample, when the dataset is organized accordingly.

    Notes
    -----
    Internally, GeoPriorSubsNet converts the predicted GWL to head
    using scaling conventions and an optional reference datum.
    Developers must ensure that the indexing used to fetch GWL from
    dynamic features (e.g., ``gwl_dyn_index`` in ``scaling_kwargs``)
    is consistent with the configured output dimensionality.
""",
    pde_mode=r"""
pde_mode : {{'consolidation', 'gw_flow', 'both', 'none', \
'on', 'off'}} or list of str, default 'both'
    Select which physics residuals participate in the loss.

    GeoPriorSubsNet can enforce two main constraints:

    (1) Groundwater-flow residual (diffusivity equation)
    .. math::

       R_gw = S_s \\, \partial_t h
              - \nabla \cdot (K \\, \nabla h) - Q

    (2) Consolidation residual (relaxation closure)
    .. math::

       R_cons = \partial_t s - \frac{s_eq(h) - s}{tau}

    where :math:`h` is hydraulic head, :math:`s` is subsidence,
    :math:`K` is hydraulic conductivity, :math:`S_s` is specific
    storage, :math:`Q` is a volumetric forcing term, and :math:`tau`
    is a relaxation time scale.

    Accepted values
    --------------
    'consolidation'
        Only :math:`R_cons` is active in the physics loss.
    'gw_flow'
        Only :math:`R_gw` is active in the physics loss.
    'both' or 'on'
        Both residuals are active (recommended).
    'none' or 'off'
        All physics residuals are disabled. The model behaves as a
        purely data-driven encoder-decoder.

    A list of strings enables multiple modes explicitly, e.g.
    ``['consolidation', 'gw_flow']``. This form is useful if you plan
    to introduce additional residual families in the future (e.g.,
    bounds-only, priors-only), while keeping a stable API.

    Notes
    -----
    Even when PDE residuals are disabled, the model may still produce
    and regularize physics fields (e.g., via priors or smoothness),
    depending on how the training step is configured. For a fully
    data-driven run, ensure physics loss weights are also set to zero.
""",
    mv=r"""
mv : LearnableMV or float, default LearnableMV(1e-7)
    Compressibility-like coefficient for the consolidation closure.

    In GeoPriorSubsNet, ``mv`` represents a one-dimensional
    compressibility factor that links drawdown to equilibrium
    settlement through an effective closure.

    A common conceptual form used by the model is:
    .. math::

       s_eq(h) \approx S_s \\, \Delta h \\, H

    where the drawdown :math:`\Delta h` depends on a reference head
    datum, and :math:`H` is an effective thickness. ``mv`` can act as
    an additional closure/scale factor depending on the chosen
    formulation (e.g., calibration of settlement amplitude).

    Accepted forms
    --------------
    * LearnableMV:
        Uses the provided configuration (trainable or fixed).
    * float:
        Converted to ``LearnableMV(initial_value=float(mv),
        trainable=False)`` for reproducibility and serialization.

    Notes
    -----
    * Trainable MV parameters are commonly optimized in log space to
      enforce positivity and improve numerical stability.
    * If your formulation already fully specifies settlement amplitude
      via :math:`S_s` and :math:`H`, keep ``mv`` fixed and let the
      priors/smoothness regularize the learned fields instead.
""",
    kappa=r"""
kappa : LearnableKappa or float, default LearnableKappa(1.0)
    Closure / unit-conversion factor used by the time-scale prior.

    ``kappa`` enters the prior used to regularize the relaxation
    time scale :math:`tau`. A typical conceptual relationship is:

    .. math::

       tau_{prior} \approx
       \frac{H_d^2 \\, S_s}{\pi^2 \\, \kappa \\, K}

    where:
    * :math:`H_d` is an effective drainage thickness
      (see ``use_effective_h`` and ``hd_factor``),
    * :math:`S_s` is specific storage,
    * :math:`K` is hydraulic conductivity,
    * :math:`kappa` absorbs convention differences and/or an
      effective vertical conductivity closure.

    Accepted forms
    --------------
    * LearnableKappa:
        Uses the provided configuration (trainable by default).
    * float:
        Converted to ``LearnableKappa(initial_value=float(kappa))``.

    Notes
    -----
    * ``kappa`` is intended to make the :math:`tau` prior robust when
      different groundwater formulations or scaling conventions are
      used across datasets.
    * If you have strong external knowledge of the closure, you may
      fix ``kappa`` to 1.0 (or another calibrated value) and rely on
      learned :math:`K` and :math:`S_s` fields for adaptation.
""",
    gamma_w=r"""
gamma_w : FixedGammaW or float, default FixedGammaW(9810.0)
    Unit weight of water used by the consolidation closure.

    ``gamma_w`` is typically used to translate between pressure and
    head or to express effective stress changes under a head change,
    depending on the chosen closure.

    Accepted forms
    --------------
    * FixedGammaW:
        Uses the provided fixed value.
    * float:
        Converted to ``FixedGammaW(value=float(gamma_w))``.

    Notes
    -----
    The default value ``9810.0`` corresponds to approximately
    :math:`rho_w g` in SI units (N/m^3). Keep this fixed unless your
    unit system or dataset conventions differ.
""",
    h_ref=r"""
h_ref : FixedHRef or float or {{'auto', 'fixed'}} or None, \
default FixedHRef(0.0, mode='auto')
    Reference head datum configuration used to define drawdown.

    Drawdown is commonly computed as:
    .. math::

       \Delta h = h_{ref} - h

    where :math:`h` is head and :math:`h_{ref}` is a chosen reference
    datum. This datum impacts equilibrium settlement and therefore the
    consolidation residual.

    Accepted forms
    --------------
    * FixedHRef:
        Fully specified configuration (value + mode).
    * float:
        Treated as a fixed datum and converted to
        ``FixedHRef(value=float(h_ref), mode='fixed')``.
    * 'auto' (or aliases such as 'history', 'last', 'last_obs'):
        Converted to ``FixedHRef(value=0.0, mode='auto')``. The actual
        datum is inferred from the available history during training
        or evaluation.
    * None:
        Same as 'auto'.

    Notes
    -----
    * A stable and well-defined :math:`h_{ref}` improves training
      stability when using consolidation constraints.
    * When using cumulative subsidence targets, ensure the chosen
      reference convention matches how cumulative drawdown is encoded
      in the data pipeline.
""",
    use_effective_h=r"""
use_effective_h : bool, default False
    Whether to use an effective drainage thickness in the
    time-scale prior for :math:`tau`.

    When enabled, an effective thickness :math:`H_d` is used:
    .. math::

       H_d = f \\, H

    where ``f`` is ``hd_factor`` and :math:`H` is the thickness field
    used by the closure. This affects the prior:
    .. math::

       tau_{prior} \propto H_d^2

    Notes
    -----
    Enable this if you want the :math:`tau` prior to reflect
    single-drainage vs double-drainage conditions or other
    drainage assumptions without changing the dataset thickness.
""",
    hd_factor=r"""
hd_factor : float, default 1.0
    Factor used to form effective drainage thickness:
    ``Hd = hd_factor * H``.

    Constraints
    -----------
    Must satisfy ``0 < hd_factor <= 1``.

    Common choices
    --------------
    * ``1.0``: single-drainage thickness (no adjustment).
    * ``0.5``: double-drainage effective thickness (common
      consolidation assumption).

    Notes
    -----
    This parameter only affects the :math:`tau` prior when
    ``use_effective_h=True`` (or when drainage_mode in the scaling
    configuration activates effective thickness automatically).
""",
    kappa_mode=r"""
kappa_mode : {{'bar', 'kb'}}, default 'kb'
    Convention selector for how ``kappa`` relates to conductivity
    in the time-scale prior.

    ``kappa_mode`` is used to match the internal :math:`tau` prior
    closure to your preferred groundwater convention. Two common
    interpretations are:

    'kb'
        Interpret ``kappa`` as a factor applied to a base
        conductivity-like scale (e.g., vertical hydraulic
        conductivity closure).
    'bar'
        Interpret ``kappa`` as a factor applied to an effective or
        averaged conductivity scale.

    Notes
    -----
    This option is primarily for developers maintaining consistent
    mapping between learned fields and priors across formulations.
    If you do not have a specific convention, keep the default.
""",
    offset_mode=r"""
offset_mode : {{'mul', 'log10'}}, default 'mul'
    Global scaling mode for physics loss contribution via
    ``lambda_offset``.

    GeoPriorSubsNet exposes a scalar weight ``lambda_offset`` that
    can be used to control the overall influence of physics terms
    during training (for example, warm-up schedules).

    * 'mul'
        Use ``physics_scale = lambda_offset``.
    * 'log10'
        Use ``physics_scale = 10**lambda_offset``.

    Notes
    -----
    * 'log10' is convenient when you want to tune physics strength
      across multiple orders of magnitude.
    * ``lambda_offset`` is non-trainable by design; it is intended
      to be updated by callbacks or external schedulers.
""",
    bounds_mode=r"""
bounds_mode : {{'soft', 'hard'}} or None, default 'soft'
    How parameter bounds are enforced when bounds are provided.

    Bounds can be declared in ``scaling_kwargs`` (for example on
    log-conductivity or specific storage). This option controls
    enforcement:

    'soft'
        Add a differentiable penalty when predicted values exceed
        the bounds. This preserves gradient flow and often improves
        training stability.
    'hard'
        Clip / project parameters to the valid range. This enforces
        physical plausibility strictly but may introduce gradient
        discontinuities.

    Notes
    -----
    If ``bounds_mode`` is None and the resolved scaling configuration
    specifies a bounds policy, the scaling configuration wins.
    Otherwise, this argument provides the default behavior.
""",
    residual_method=r"""
residual_method : {{'exact', 'euler'}}, default 'exact'
    Time-integration method used inside the consolidation residual.

    GeoPriorSubsNet supports two update styles for the relaxation
    closure. Conceptually, the closure is:

    .. math::

       \partial_t s = \frac{s_eq(h) - s}{tau}

    'exact'
        Use a closed-form relaxation update over the time step
        (recommended). This is usually more stable for stiff
        dynamics (small :math:`tau`).
    'euler'
        Use an explicit Euler discretization. This is simpler but
        can be unstable unless time steps are small.

    Notes
    -----
    Developers should prefer 'exact' for modern training loops,
    especially when learning :math:`tau(x,y)` jointly with the
    forecasting network.
""",
    time_units=r"""
time_units : str or None, default None
    Name of the time unit used by the coordinates supplied to
    the model, e.g. ``'year'``, ``'month'``, ``'day'``, or
    ``'second'``.

    This value is injected into the resolved scaling configuration
    if the scaling configuration does not already define it. It can
    influence:
    * conversion from dataset time steps to SI seconds,
    * the magnitude of time derivatives in PDE residuals,
    * the interpretation of :math:`tau` priors.

    Notes
    -----
    For reproducibility, prefer setting ``time_units`` explicitly
    (either here or in ``scaling_kwargs``) rather than relying on
    implicit defaults.
""",
    scale_pde_residuals=r"""
scale_pde_residuals : bool, default True
    Whether to rescale PDE derivatives by coordinate ranges when
    coordinates are normalized.

    When coordinates are normalized to roughly ``[0, 1]`` ranges,
    raw derivatives computed in normalized space do not match the
    intended physical scaling. This option applies a chain-rule
    correction using the coordinate ranges:

    If :math:`t' = (t - t_0) / \Delta t`, then:
    .. math::

       \partial_t = \frac{1}{\Delta t} \\, \partial_{t'}

    Similarly for spatial coordinates:
    .. math::

       \nabla = \left[
       \frac{1}{\Delta x}\partial_{x'},
       \frac{1}{\Delta y}\partial_{y'}\right]

    Notes
    -----
    Enable this whenever ``scaling_kwargs['coords_normalized']=True``
    and accurate ``coord_ranges`` are available. Disable only if you
    intentionally want a purely normalized PDE residual (rare).
""",
    scaling_kwargs=r"""
scaling_kwargs : mapping or str or GeoPriorScalingConfig \
or None, default None
    Scaling, conventions, and physics-configuration payload.

    This is the single most important configuration for GeoPrior.
    It defines how raw dataset variables (subsidence, groundwater
    level, coordinates, thickness) are interpreted, scaled, and
    converted into the internal quantities used by physics losses.

    The resolved payload is treated as source-of-truth and is used
    by:
    * target formatting (subsidence and head),
    * coordinate handling (order, normalization, SI conversion),
    * PDE derivative scaling (chain-rule corrections),
    * physics closures (drawdown rules, head-from-depth proxy),
    * priors and bounds (K, Ss, tau, H ranges),
    * training policies (warmups, ramps, diagnostics).

    Accepted forms
    --------------
    mapping
        A configuration dictionary. The dictionary is copied,
        validated, and resolved into a canonical payload.
    str
        A filesystem path to a JSON or YAML file containing the
        payload.
    GeoPriorScalingConfig
        A validated configuration object. Its resolved payload is
        used as source-of-truth.
    None
        Use internal defaults (recommended only for development
        tests, not for scientific runs).

    Core keys
    ---------
    The following keys are commonly used. Unrecognized keys may be
    preserved for provenance and debugging, but only documented keys
    are guaranteed stable.

    1) Target scaling and conventions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    subsidence_kind : {'cumulative', 'incremental'}, optional
        Declares the semantics of the subsidence target.

        * 'cumulative': the target is cumulative settlement
          relative to an origin (e.g., first year).
        * 'incremental': the target is stepwise increment between
          consecutive time steps.

        Notes
        -----
        The consolidation residual typically expects a consistent
        notion of :math:`s(t)` and its time derivative. Mixing a
        cumulative target with an incremental residual definition
        will produce unstable training.

    subs_scale_si, subs_bias_si : float, optional
        Linear transform to convert stored subsidence values to the
        model's internal SI-consistent units.

        The transform is applied as:
        .. math::

           s_{si} = s_{raw} \\, subs\\_scale\\_si + subs\\_bias\\_si

        Examples
        --------
        * If subsidence is stored in millimeters and internal units
          are meters, use ``subs_scale_si = 1e-3``.
        * If the dataset is already in SI, keep the defaults.

    head_scale_si, head_bias_si : float, optional
        Linear transform applied to head values in the same form as
        subsidence:
        .. math::

           h_{si} = h_{raw} \\, head\\_scale\\_si + head\\_bias\\_si

        Notes
        -----
        These fields are required when the dataset mixes unit
        systems or when preprocessing stores standardized variables.

    H_scale_si, H_bias_si : float, optional
        Linear transform applied to the thickness field used by
        consolidation or priors:
        .. math::

           H_{si} = H_{raw} \\, H\\_scale\\_si + H\\_bias\\_si

        Notes
        -----
        Thickness enters the equilibrium settlement approximation
        and the :math:`tau` prior. Incorrect scaling strongly
        destabilizes physics losses.

    allow_subs_residual : bool, optional
        If True, the model may output an additive residual around
        the physics-implied mean subsidence. This is used when the
        physics closure captures the mean trend but data contain
        unmodeled components.

    2) Coordinates, normalization, and SI conversion
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    coord_order : list of str, optional
        Order of the coordinate columns in the ``coords`` tensor.
        The default expectation is ``['t', 'x', 'y']``.

        Notes
        -----
        GeoPrior derivatives assume a known axis order. If your
        pipeline uses ``['x','y','t']`` you must declare it here.

    coords_normalized : bool, optional
        Whether the coordinates provided to the model are already
        normalized (typically to [0, 1] or standardized).

        If True, PDE derivatives computed w.r.t. normalized coords
        must be rescaled to physical units by chain rule.

    coord_ranges : dict, optional
        Ranges of each coordinate axis in the *current coordinate
        system* used by ``coords``. Example:
        ``{'t': 7.0, 'x': 44494.875, 'y': 39275.0}``.

        When coordinates are normalized using:
        .. math::

           u' = (u - u_0) / \\Delta u

        the chain-rule correction is:
        .. math::

           \\partial_u = (1/\\Delta u) \\, \\partial_{u'}

        Therefore, for the groundwater residual:
        .. math::

           \\partial_t h = (1/\\Delta t) \\, \\partial_{t'} h

        and
        .. math::

           \\nabla h =
           [ (1/\\Delta x) \\partial_{x'} h,
             (1/\\Delta y) \\partial_{y'} h ]

        Notes
        -----
        Provide accurate ``coord_ranges`` whenever
        ``coords_normalized=True``. Incorrect ranges will inflate or
        deflate residual magnitudes and can dominate training.

    coord_mode : {'degrees', 'meters'}, optional
        Declares how raw coordinates were represented at ingestion.
        This is used for provenance checks and optional conversion.

    coord_src_epsg, coord_target_epsg, coord_epsg_used : int, optional
        EPSG metadata used when converting geographic coordinates
        (degrees) to projected coordinates (meters). These keys are
        primarily for audit and reproducibility.

    coords_in_degrees : bool, optional
        Whether current coordinates are in degrees. If False, the
        system treats x/y as meter-like.

    time_units : str, optional
        Name of the dataset time unit for the coordinate time axis,
        e.g. 'year', 'day', or 'second'.

    seconds_per_time_unit : float, optional
        Explicit conversion factor from one dataset time unit to
        seconds. When provided, the system can derive SI ranges:
        .. math::

           \\Delta t_{si} = \\Delta t \\cdot seconds\\_per\\_time\\_unit

    coord_ranges_si : dict, optional
        Coordinate ranges converted to SI for derivative scaling.
        If present, it can override implicit conversions.

    coord_inv_ranges_si : dict, optional
        Precomputed inverse SI ranges:
        ``{'t': 1/Delta_t_si, 'x': 1/Delta_x, 'y': 1/Delta_y}``.
        This is a convenience for fast derivative rescaling.

    dt_min_units : float, optional
        Lower bound for time-step size in dataset units to avoid
        division-by-zero and extreme gradients in finite-difference
        approximations.

    3) Groundwater level (GWL) to head mapping
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Many datasets store groundwater as depth below ground surface
    (positive downward). Physics residuals usually require head
    (positive upward). GeoPrior provides a structured set of keys
    to define this mapping robustly.

    gwl_kind : {'depth_bgs', 'head', ...}, optional
        Declares what the stored GWL represents. Common:
        * 'depth_bgs': depth below ground surface.
        * 'head': hydraulic head.

    gwl_sign : {'down_positive', 'up_positive'}, optional
        Sign convention of the stored GWL quantity.

    gwl_target_kind : {'head', 'depth', ...}, optional
        The target quantity used internally by physics. Common is
        'head'.

    gwl_target_sign : {'up_positive', 'down_positive'}, optional
        Sign convention for the internal target quantity.

    use_head_proxy : bool, optional
        If True, compute head from depth and surface elevation using
        a declared rule. A typical rule is:
        .. math::

           h = z_{surf} - depth

    z_surf_col : str, optional
        Column name for surface elevation (static) used to convert
        depth to head.

    gwl_col : str, optional
        Column name for the groundwater driver series.

    gwl_z_meta : dict, optional
        Structured metadata describing the full conversion pipeline
        from raw columns to model columns. This is recommended for
        robust audits and for ensuring the same conventions are
        used across Stage-1 and Stage-2.

        Typical entries include:
        * ``raw_kind`` / ``raw_sign`` (as stored),
        * ``driver_kind`` / ``driver_sign`` (as used as input),
        * ``target_kind`` / ``target_sign`` (as used in physics),
        * ``z_surf_col`` and conversion rule,
        * ``cols`` mapping from raw to model column names.

    Notes
    -----
    If the GWL-to-head mapping is inconsistent, the groundwater
    residual becomes physically meaningless. Always verify the
    resolved ``gwl_z_meta`` in the Stage-1 audit.

    4) Dynamic and static feature indexing
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gwl_dyn_index : int, optional
        Index of the GWL driver channel inside ``dynamic_features``
        (last axis). This must match the feature order exported by
        Stage-1.

    subs_dyn_index : int, optional
        Index of the subsidence channel inside ``dynamic_features``
        when subsidence is included as an input signal.

    z_surf_static_index : int, optional
        Index of surface elevation inside ``static_features`` when
        it is included as a static channel.

    dynamic_feature_names, future_feature_names, static_feature_names \
: list of str, optional
        Feature name lists for audit, debugging, and alignment checks.
        These lists are strongly recommended for reproducible runs.

    5) Physics residual scaling and floors
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cons_residual_units : {'second', 'time_unit'}, optional
        Declares the unit convention for the consolidation residual
        scaling. If set to 'second', time derivatives are interpreted
        in SI seconds after conversion.

    gw_residual_units : {'second', 'time_unit'}, optional
        Same idea as ``cons_residual_units`` but for groundwater flow
        residual scaling.

    cons_scale_floor, gw_scale_floor : {'auto'} or float, optional
        Lower bounds for residual scaling factors. Floors prevent
        extremely small scales that would blow up residuals after
        normalization. 'auto' selects a safe heuristic.

    Q_wrt_normalized_time : bool, optional
        If True, interpret any time dependence of Q with respect to
        normalized time. If False, interpret it with respect to the
        physical time axis after conversion.

    Q_kind : {'per_volume', 'recharge_rate', 'head_rate'}, optional
        Declares the physical interpretation of the forcing Q.

        'per_volume'
            Q is in 1/s in the groundwater PDE:
            .. math::

               R_gw = S_s \\, \\partial_t h
                      - \\nabla \\cdot (K \\nabla h) - Q

        Other kinds require additional conversion rules and should be
        used only if your implementation explicitly supports them.

    Q_in_si : bool, optional
        Whether Q provided by the model/pipeline is already in SI.

    Q_in_per_second : bool, optional
        Whether Q is already expressed as 1/s.

    Q_length_in_si : bool, optional
        Whether any length scale used in Q conversions is in SI.

    6) Priors, bounds, and constraint metadata
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bounds : dict, optional
        Bounds for learned fields and/or their log transforms.
        Common entries include:
        * ``K_min`` / ``K_max``,
        * ``Ss_min`` / ``Ss_max``,
        * ``tau_min`` / ``tau_max``,
        * ``H_min`` / ``H_max``,
        * ``logK_min`` / ``logK_max``, etc.

        Notes
        -----
        Bounds can be enforced softly (penalty) or hard (projection),
        depending on the model's bounds policy.

    drainage_mode : {'single', 'double'}, optional
        Declares drainage assumption used for the tau prior. When
        'double', a typical effective thickness rule is used
        internally (e.g., Hd_factor=0.5).

    scaling_error_policy : {'raise', 'warn', 'ignore'}, optional
        Policy used when resolving scaling inconsistencies (missing
        keys, invalid ranges, unknown conventions). 'raise' is
        recommended for scientific workflows.

    7) Training policies and diagnostics (optional)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    training_strategy : str, optional
        High-level policy hint, e.g. 'physics_first'. This may be
        used by callbacks or training orchestration code.

    physics_warmup_steps, physics_ramp_steps : int, optional
        Warm-up and ramp schedule for physics loss activation.

    q_policy : str, optional
        Policy for enabling the forcing Q term (e.g. warmup_off).

    q_warmup_epochs, q_ramp_epochs : int, optional
        Epoch-based schedule for Q activation.

    q_warmup_steps, q_ramp_steps : int, optional
        Step-based schedule for Q activation.

    subs_resid_policy : str, optional
        Policy for enabling subsidence residual head (if allowed).

    subs_resid_warmup_epochs, subs_resid_ramp_epochs : int, optional
        Epoch-based schedule for subsidence residual activation.

    subs_resid_warmup_steps, subs_resid_ramp_steps : int, optional
        Step-based schedule for subsidence residual activation.

    clip_global_norm : float, optional
        Global norm threshold for gradient clipping used by the
        training loop (when enabled).

    debug_physics_grads : bool, optional
        If True, enable additional gradient diagnostics for physics
        terms. This can be expensive and should be disabled for
        production training.

    log_q_diagnostics : bool, optional
        If True, emit additional logs related to Q behavior.

    track_aux_metrics : bool, optional
        Enable auxiliary epsilon-style metrics and trackers.

    Notes
    -----
    The resolved configuration should be saved alongside model
    artifacts (Stage-1/Stage-2 audits). If results differ across
    sites, compare resolved scaling payloads first.
""",
)

# ---------------------------------------------------------------------
# Nested doc components:
#   {params.base.<...>}     come from _halnet_core_params
#   {params.geoprior.<...>} come from _geoprior_core_params
# ---------------------------------------------------------------------
_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_halnet_core_params),
    geoprior=DocstringComponents(_geoprior_core_params),
)


# ---------------------------------------------------------------------
# GeoPriorSubsNet main docstring template.
# ---------------------------------------------------------------------
GEOPRIOR_SUBSNET_DOC = r"""
GeoPriorSubsNet
================

Prior-regularized physics-informed network for multi-step
subsidence forecasting with groundwater coupling.

GeoPriorSubsNet combines a BaseAttentive encoder-decoder with
a set of physics losses that constrain the forecast to respect
a simplified groundwater-flow equation and a consolidation
closure. In addition, it learns spatially varying physics
fields and regularizes them against geologically motivated
priors.

Key ideas
---------
* Predict groundwater level (GWL) over a forecast horizon.
* Convert GWL to hydraulic head using a scaling configuration.
* Compute mean subsidence from head via a relaxation-style
  consolidation closure.
* Learn effective fields :math:`K(x,y)`, :math:`S_s(x,y)`,
  :math:`tau(x,y)`, and a forcing term :math:`Q(t,x,y)`.
* Penalize PDE residuals and prior inconsistency during
  training.

Physics residuals (conceptual)
------------------------------
Let :math:`h(t,x,y)` be head and :math:`s(t,x,y)` subsidence.

Groundwater flow residual:

.. math::

   R_gw = S_s \\, \partial_t h - \nabla \cdot (K \\, \nabla h) - Q

Consolidation residual (relaxation form):

.. math::

   R_cons = \partial_t s - \frac{{s_eq(h) - s}}{{tau}}

A common equilibrium approximation is:

.. math::

   s_eq(h) \approx S_s \\, \Delta h \\, H

where :math:`H` is an effective thickness and
:math:`\Delta h` is drawdown w.r.t. a reference datum.

Time-scale prior (conceptual):

.. math::

   tau_{{prior}} \approx \frac{{H_d^2 \\, S_s}}
   {{\pi^2 \\, \kappa \\, K}}

and the prior residual is expressed in log space.

Inputs
------
GeoPriorSubsNet follows the FusionLab "dict input" API.
A typical batch is a mapping containing:

* ``static_features``  : ``(B, S)``
* ``dynamic_features`` : ``(B, T_in, D)``
* ``future_features``  : ``(B, T_out, F)``
* ``coords``           : ``(B, T_out, 3)``

The coordinate axis order is expected to be ``['t','x','y']``
unless overridden by ``scaling_kwargs['coord_order']``.

If your physics closure requires a thickness field, provide a
compatible entry (for example ``H_field``) as supported by the
GeoPrior data pipeline.

Parameters
----------
{params.base.static_input_dim}
{params.base.dynamic_input_dim}
{params.base.future_input_dim}

{params.geoprior.output_subsidence_dim}
{params.geoprior.output_gwl_dim}

forecast_horizon : int, default 1
    Forecast horizon length :math:`H`. The model emits
    :math:`H` steps and evaluates physics terms on these
    steps (when enabled by ``pde_mode``).

quantiles : list of float or None, default None
    Optional quantile levels for probabilistic forecasting.
    When provided, outputs include a quantile axis.

{params.base.embed_dim}
{params.base.hidden_units}
{params.base.lstm_units}
{params.base.attention_units}
{params.base.num_heads}
{params.base.dropout_rate}
{params.base.max_window_size}
{params.base.memory_size}
{params.base.scales}
{params.base.multi_scale_agg}
{params.base.final_agg}
{params.base.activation}
{params.base.use_residuals}
{params.base.use_batch_norm}
{params.base.use_vsn}
{params.base.vsn_units}

{params.geoprior.pde_mode}
{params.geoprior.mv}
{params.geoprior.kappa}
{params.geoprior.gamma_w}
{params.geoprior.h_ref}
{params.geoprior.use_effective_h}
{params.geoprior.hd_factor}
{params.geoprior.kappa_mode}
{params.geoprior.offset_mode}
{params.geoprior.bounds_mode}
{params.geoprior.residual_method}
{params.geoprior.time_units}
{params.geoprior.scale_pde_residuals}
{params.geoprior.scaling_kwargs}

mode : str or None, default None
    Routing mode for known-future covariates, forwarded to
    BaseAttentive.

objective : str or None, default None
    Backbone selection, forwarded to BaseAttentive.

attention_levels : str or list of str or None, default None
    Which attention tensors are returned in inference mode,
    forwarded to BaseAttentive.

architecture_config : dict or None, default None
    Optional architecture override, forwarded to BaseAttentive.

name : str, default "GeoPriorSubsNet"
    Keras model name / scope.

verbose : int, default 0
    Verbosity level used by internal logging hooks.

**kwargs
    Forwarded to ``keras.Model``.

Notes
-----
* Physics losses are added inside the custom training loop.
  Use compile-time weights (for example ``lambda_cons``,
  ``lambda_gw``, ``lambda_prior``, ``lambda_smooth``,
  ``lambda_bounds``) to tune the balance between data fit and
  physics regularization.

* ``scaling_kwargs`` is central: it defines coordinate
  conventions, unit conversions, and (optionally) parameter
  bounds. If coordinates are normalized, provide accurate
  ``coord_ranges`` to keep PDE residual magnitudes consistent.

* The model supports auxiliary diagnostics (for example
  epsilon-style residual summaries) when enabled in the
  scaling configuration.

See Also
--------
fusionlab.nn._base_attentive.BaseAttentive
    Core encoder-decoder backbone used by GeoPriorSubsNet.

fusionlab.nn.pinn.TransFlowSubsNet
    Related PINN that couples groundwater flow and settlement
    with a simpler coefficientization.

Examples
--------
>>> import tensorflow as tf
>>> from fusionlab.nn.pinn.geoprior import GeoPriorSubsNet
>>> model = GeoPriorSubsNet(
...     static_input_dim=3,
...     dynamic_input_dim=8,
...     future_input_dim=4,
...     output_subsidence_dim=1,
...     output_gwl_dim=1,
...     forecast_horizon=3,
...     quantiles=[0.1, 0.5, 0.9],
...     pde_mode='both',
...     scaling_kwargs={{'time_units': 'year'}},
... )
>>> batch = {{
...     "static_features": tf.zeros([8, 3]),
...     "dynamic_features": tf.zeros([8, 12, 8]),
...     "future_features": tf.zeros([8, 3, 4]),
...     "coords": tf.zeros([8, 3, 3]),
... }}
>>> y = model(batch, training=False)
>>> sorted(y.keys())
['gwl_pred', 'subs_pred']

References
----------
.. [1] Lim, B., Arik, S. O., Loeff, N., and Pfister, T.
   Temporal Fusion Transformers for Interpretable Multi-horizon
   Time Series Forecasting. International Journal of
   Forecasting, 2021.

.. [2] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
   Physics-informed neural networks: A deep learning framework
   for solving forward and inverse problems involving
   nonlinear partial differential equations. Journal of
   Computational Physics, 2019.

.. [3] Biot, M. A. General theory of three-dimensional
   consolidation. Journal of Applied Physics, 1941.
""".format(params=_param_docs)

# ---------------------------------------------------------------------
# PoroElasticSubsNet documentation strategy (doc.py)
# ---------------------------------------------------------------------
#
# Goal
# ----
# PoroElasticSubsNet is a "preset" variant of GeoPriorSubsNet.
# We should not duplicate the full parameter list. Instead:
#
# 1) Reuse the exact same nested param components:
#       base  -> _halnet_core_params
#       geoprior -> _geoprior_core_params
#
# 2) Add ONLY a tiny "delta" component describing what differs:
#       * changed defaults
#       * extra bound injection behavior
#       * stronger compile() defaults
#
# 3) Make the PoroElastic docstring explicitly reference
#    GeoPriorSubsNet for the full parameter definitions, while
#    still being copy-pasteable and useful in API docs.
#
# This avoids repetition while keeping the docstring self-
# sufficient and developer-friendly.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Delta params: only describe what is different from GeoPriorSubsNet.
# ---------------------------------------------------------------------
_poroelastic_delta_params = dict(
    poroelastic_overview=r"""
Notes
-----
PoroElasticSubsNet is a preset configuration of GeoPriorSubsNet.
It reuses the same architecture, I/O contract, and parameter
definitions. Only a few defaults and policies differ (listed
below). For the complete parameter reference, see
:class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet`.
""",
    poro_pde_mode_default=r"""
pde_mode : str, default 'consolidation'
    Same meaning as in GeoPriorSubsNet, but the default is set to
    ``'consolidation'`` to disable the groundwater-flow residual
    by default.

    The intent is to provide a poroelastic surrogate baseline:
    consolidation-driven settlement with a strong time-scale prior.

    See Also
    --------
    GeoPriorSubsNet.pde_mode
        Full definition and accepted values.
""",
    poro_effective_thickness_defaults=r"""
use_effective_h : bool, default True
hd_factor : float, default 0.6
    Same meaning as in GeoPriorSubsNet, but defaults are chosen to
    activate an effective drainage thickness.

    The effective thickness is:
    .. math::

       H_d = f \\, H

    with ``f = hd_factor``. Since the time-scale prior scales as
    :math:`tau_{prior} \\propto H_d^2`, lowering ``hd_factor`` makes
    the prior more sensitive to thickness and can yield more
    physically constrained training in poroelastic baselines.

    See Also
    --------
    GeoPriorSubsNet.use_effective_h, GeoPriorSubsNet.hd_factor
""",
    poro_kappa_mode_default=r"""
kappa_mode : {'bar', 'kb'}, default 'bar'
    Same meaning as in GeoPriorSubsNet, but default is set to 'bar'
    to match the poroelastic surrogate convention used by this
    variant.

    See Also
    --------
    GeoPriorSubsNet.kappa_mode
""",
    poro_bounds_injection=r"""
scaling_kwargs : mapping or str or GeoPriorScalingConfig or None
    Same meaning as in GeoPriorSubsNet.

    PoroElasticSubsNet additionally applies a "fill-missing-bounds"
    policy during initialization:

    * The user-provided ``scaling_kwargs['bounds']`` dictionary is
      copied.
    * A default bound set is applied with ``setdefault`` semantics,
      so user keys are never overwritten.

    This affects only missing keys in ``bounds`` and is meant to
    provide a safe poroelastic prior envelope for (H, logK, logSs).

    Notes
    -----
    This variant does not change the definition of bounds enforcement
    (soft vs hard). It only ensures typical geomechanical bounds are
    available when the user did not provide them.

    See Also
    --------
    GeoPriorSubsNet.scaling_kwargs
        Full key documentation and recommended payload structure.
""",
    poro_compile_defaults=r"""
compile(...) defaults
    PoroElasticSubsNet overrides compile defaults to enforce stronger
    geomechanical consistency:

    * ``lambda_gw = 0.0``  (groundwater-flow residual disabled)
    * ``lambda_prior`` increased to tighten :math:`tau` toward the
      time-scale prior.
    * ``lambda_bounds`` increased to keep fields within plausible
      envelopes.
    * smaller LR multipliers for ``mv`` and ``kappa`` so these scalars
      adapt conservatively.

    These changes are defaults only; users can still override all
    compile weights explicitly.
""",
)


# ---------------------------------------------------------------------
# Parameter docs: reuse base + geoprior, then add poroelastic delta.
# ---------------------------------------------------------------------
_poro_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_halnet_core_params),
    geoprior=DocstringComponents(_geoprior_core_params),
    poro=DocstringComponents(_poroelastic_delta_params),
)

# ---------------------------------------------------------------------
# PoroElastic docstring template.
# ---------------------------------------------------------------------
POROELASTIC_SUBSNET_DOC = r"""
PoroElasticSubsNet
==================

Poroelastic surrogate variant of GeoPriorSubsNet.

This model is architecturally identical to GeoPriorSubsNet and
follows the same dict-input API, outputs, and parameter semantics.
It is provided as a physics-driven baseline for ablation and
comparison runs.

Differences from GeoPriorSubsNet
--------------------------------
* Default ``pde_mode='consolidation'`` (no groundwater-flow residual).
* Effective drainage thickness enabled by default:
  ``use_effective_h=True`` and ``hd_factor < 1``.
* A "fill-missing-bounds" policy is applied to
  ``scaling_kwargs['bounds']`` (user keys never overwritten).
* compile() uses stronger prior and bounds defaults, and disables
  groundwater-flow by default.

{params.poro.poroelastic_overview}

Parameters
----------
{params.base.static_input_dim}
{params.base.dynamic_input_dim}
{params.base.future_input_dim}

pde_mode : str, default 'consolidation'
    {params.poro.poro_pde_mode_default}

use_effective_h : bool, default True
hd_factor : float, default 0.6
    {params.poro.poro_effective_thickness_defaults}

kappa_mode : {{'bar', 'kb'}}, default 'bar'
    {params.poro.poro_kappa_mode_default}

scale_pde_residuals : bool, default True
    Same meaning as in GeoPriorSubsNet. Kept enabled by default to
    ensure derivative rescaling when coordinates are normalized.

scaling_kwargs : mapping or str or GeoPriorScalingConfig or None
    {params.poro.poro_bounds_injection}

name : str, default "PoroElasticSubsNet"
    Keras model name / scope.

**kwargs
    Forwarded to GeoPriorSubsNet and BaseAttentive.

Notes
-----
{params.poro.poro_compile_defaults}

See Also
--------
fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet
    Full parameter reference and the groundwater-coupled variant.

Examples
--------
>>> import tensorflow as tf
>>> from fusionlab.nn.pinn.geoprior import PoroElasticSubsNet
>>> model = PoroElasticSubsNet(
...     static_input_dim=3,
...     dynamic_input_dim=8,
...     future_input_dim=4,
...     forecast_horizon=3,
...     scaling_kwargs={{'time_units': 'year'}},
... )
>>> batch = {{
...     "static_features": tf.zeros([8, 3]),
...     "dynamic_features": tf.zeros([8, 12, 8]),
...     "future_features": tf.zeros([8, 3, 4]),
...     "coords": tf.zeros([8, 3, 3]),
... }}
>>> y = model(batch, training=False)
>>> sorted(y.keys())
['gwl_pred', 'subs_pred']
""".format(params=_poro_param_docs)
