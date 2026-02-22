.. _geoprior_data_and_units:

==============================
Data, Units, and Conventions
==============================

GeoPrior is extremely sensitive to **units** and **coordinate conventions**
because physics residuals rely on derivatives w.r.t. ``(t, x, y)``.
A “reasonable” model can look completely broken if the scaling metadata
is inconsistent (classic symptom: exploding ``epsilon_*`` even when the
data loss looks fine).

This page defines the **input batch contract**, the meaning of the
**scaling payload** (``scaling_kwargs``), and the conventions used in the
Stage-1/Stage-2 pipeline audits.

See also
--------
* :doc:`overview` (high-level picture)
* :doc:`pipeline` (Stage-1 artifacts, Stage-2 handshake audit)
* :doc:`core` (where derivatives are taken, and how they are rescaled)
* :doc:`losses` (how epsilon metrics and physics losses are computed)


1. Input batch contract (dict inputs)
======================================

GeoPrior expects **dict inputs**. At minimum, physics needs:

* ``coords`` : spatio-temporal coordinates for PINN derivatives
* a thickness field (commonly ``H_field``) for consolidation / closures

In practice (most common):

.. code-block:: text

   inputs = {
       "static_features":  (B, S),
       "dynamic_features": (B, T, D),
       "future_features":  (B, H, F),
       "coords":           (B, H, 3),  # (t, x, y) in coord_order
       "H_field":          (B, H, 1),  # thickness [m] after scaling
   }

Shapes and axes
----------------

* ``B`` : batch size
* ``T`` : historical/past window length (encoder length)
* ``H`` : forecast horizon (decoder length)
* ``S, D, F`` : feature dims for static, dynamic past, known future

Coordinates (``coords``)
-------------------------

``coords`` **must** align with the forecast horizon:

* shape: ``(B, H, 3)``
* axes are ordered by ``scaling_kwargs["coord_order"]`` (default: ``["t","x","y"]``)

.. note::

   GeoPrior’s physics derivatives are taken w.r.t. the same ``coords`` tensor
   passed into the forward call. If you accidentally feed different coords to
   the forward pass than the ones used in the GradientTape branch, your physics
   residuals become meaningless. This is enforced in the core workflow
   (see :doc:`core`).


2. The scaling payload (``scaling_kwargs``)
============================================

GeoPrior uses a single audit-friendly dictionary called ``scaling_kwargs``.
It defines how to interpret model-space values as physical quantities.

This payload is validated at runtime and is also serialized with the model
(via a small Keras-serializable container).

:API Reference:
   :class:`~fusionlab.nn.pinn.geoprior.scaling.GeoPriorScalingConfig`

Typical location in the pipeline:
----------------------------------

* ``scaling_kwargs.json``: exported by Stage-1
* ``stage1_scaling_audit.json``: plausibility checks and summaries
* ``stage2_handshake_audit.json``: confirms Stage-2 inputs match the contract

.. note::

   When physics diagnostics look inconsistent across cities/sites,
   the first thing to check is **resolved coordinate spans** (``coord_ranges``
   or ``coord_ranges_si``) and time unit conversion.


3. Coordinate conventions (x/y/t)
==================================

GeoPrior supports two common coordinate modes:

A) Meters-like coordinates (recommended)
------------------------------------------

* ``coords_in_degrees = False``
* x/y are already in meters (e.g., UTM)
* ``coord_epsg_used`` often corresponds to a projected CRS (e.g., EPSG:326xx)

In this case, spatial derivatives can be interpreted directly in SI after
normalization scaling is applied (see below).

B) Degree coordinates (lon/lat)
---------------------------------

If your coords are lon/lat degrees:

* ``coords_in_degrees = True``
* you **must** supply conversion factors:

  * ``deg_to_m_lon`` : meters per degree longitude (site latitude-dependent)
  * ``deg_to_m_lat`` : meters per degree latitude

GeoPrior validates these factors when ``coords_in_degrees=True``.

.. warning::

   Do **not** run spatial PDE residuals on raw degrees without meters/degree
   conversion. The PDE residual scale will be off by orders of magnitude.


4. Normalized coords and chain-rule scaling
============================================

Many pipelines normalize ``coords`` (and optionally also features).
If ``coords_normalized=True``, then derivatives computed by autodiff are
with respect to normalized coordinates and **must be rescaled** back to
dataset/SI spans via the chain rule.

Key fields
-----------

``coords_normalized`` : bool
   If True, coords have been scaled to a normalized range.

``coord_ranges`` : dict
   Coordinate spans in *dataset units* (must include ``"t"``, ``"x"``, ``"y"``).

Example:

.. code-block:: json

   {
     "coords_normalized": true,
     "coord_order": ["t", "x", "y"],
     "coord_ranges": {"t": 7.0, "x": 44494.875, "y": 39275.0},
     "time_units": "year"
   }

Optional but preferred
------------------------

``coord_ranges_si`` : dict
   Coordinate spans already in SI (t in seconds, x/y in meters).
   If present, GeoPrior prefers this over ``coord_ranges`` + unit conversion.

.. note::

   Second derivatives scale with the **square** of spans (e.g., ``1 / x_span^2``).
   This is a common reason ``gw_flow`` residuals explode if spans are wrong.

The actual derivative rescaling is performed by internal helpers that take the
raw derivatives (from normalized coords) and return an SI-consistent “derivative
frame” used by the physics residual builders (see :doc:`core`).


5. Time units (critical)
=========================

GeoPrior requires a declared time unit:

``time_units`` : str
   Dataset time unit for the t axis.

Common values:
* ``"year"``
* ``"day"``
* ``"second"``

If your time coordinate is “year index” (e.g., 2015, 2016, …) the model can still
work, but **physics derivatives must be converted** to per-second internally.

Practical guidance
-------------------

* If your t coordinate increments by 1 each year: set ``time_units="year"``.
* If your t coordinate is in days: set ``time_units="day"``.
* If you already express t in seconds: set ``time_units="second"``.

.. warning::

   A wrong ``time_units`` silently corrupts :math:`\\partial_t` terms
   and can make consolidation / groundwater residuals meaningless even if
   the supervised fit looks good.


6. Target variables and sign conventions
=========================================

Subsidence (``subs_pred``)
---------------------------

GeoPrior typically models subsidence as **cumulative settlement**,
but supports different conventions through metadata.

Recommended scaling keys:

``subs_scale_si`` : float
   Multiply your stored subsidence values by this factor to convert to **meters**.

   * If your targets are in **mm**, use ``subs_scale_si = 1e-3``.
   * If your targets are already in **m**, use ``subs_scale_si = 1.0``.

``subs_bias_si`` : float
   Additive bias (rare; usually 0).

``subsidence_kind`` : {"cumulative", "incremental"}
   Declares whether your subsidence target is cumulative over time
   or per-step increments.

.. note::

   Your Stage-2 CSV exports may carry a display unit (e.g. ``mm``),
   but physics should ideally run in SI (meters).

Groundwater variable (``gwl_pred`` / head conventions)
--------------------------------------------------------

Different datasets store “groundwater level” in different ways:

* **Hydraulic head** (recommended): already an elevation-like head.
* **Depth-to-water** (common): depth below surface (positive downward).

GeoPrior supports both, but **depth-to-water requires surface elevation**
to convert to head:

.. math::

   h = z_{surf} - z_{GWL}

Common metadata keys used by the pipeline:

``gwl_kind`` : str (dataset-dependent)
   Indicates whether the groundwater variable is head-like or depth-like.

``z_surf_col`` / ``z_surf_static_index``
   Where to find surface elevation (or DEM) if needed for conversions.

The pipeline typically records these conventions in a nested metadata object
(e.g., ``gwl_z_meta``) and validates that the required ``z_surf`` information
is present when the groundwater variable is depth-like.

Head scaling keys:

``head_scale_si`` : float
   Multiply stored head values by this factor to convert to meters.

``head_bias_si`` : float
   Additive bias (rare; usually 0).

Thickness field (``H_field``)
------------------------------

GeoPrior uses an effective thickness field in several places
(e.g., equilibrium settlement, time-scale closures).

Keys:

``H_scale_si`` : float
   Multiply stored thickness by this factor to convert to meters.

``H_bias_si`` : float
   Additive bias (rare; usually 0).

``H_floor`` : float (model-side safety)
   Small positive floor (in meters) to avoid division-by-zero or invalid logs.


7. Feature indexing metadata (dynamic channels)
=================================================

GeoPrior often needs to know **which dynamic channels correspond to what**
for diagnostics and physics assembly.

Common keys:

``dynamic_feature_names`` : list[str]
   Optional but recommended; used for audits and sanity checks.

``gwl_dyn_index`` : int
   Index of the groundwater variable inside ``dynamic_features`` (if applicable).

``subs_dyn_index`` : int
   Index of the subsidence variable inside ``dynamic_features`` (if applicable).

.. note::

   These indices are especially useful when you compute auxiliary signals
   (e.g., drawdown) or want consistent logging across datasets.


8. Forcing term Q (sources/sinks) conventions
===============================================

GeoPrior may include an optional forcing :math:`Q` in the groundwater PDE:

.. math::

   R_{gw} = S_s \\, \\partial_t h - \\nabla \\cdot (K \\, \\nabla h) - Q

The *meaning* and units of :math:`Q` must be declared.

Common flags in the scaling payload:

``Q_in_per_second`` : bool
   If True, interpret Q as already in **1/s**.

``Q_in_si`` : bool
   Legacy/ambiguous flag (use with care). Prefer explicit settings.

Recommended interpretation:

*Default:* treat Q as a **rate in 1/time_units**, then convert to **1/s**
using ``time_units``.

If you later support multiple Q “kinds” (per-volume vs recharge-rate, etc.),
document them in :doc:`maths` and ensure the PDE residual matches the declared
meaning.


9. Validation & audit files (how to use them)
===============================================

Stage-1 scaling audit (``stage1_scaling_audit.json``)
--------------------------------------------------------

This file is designed to answer:

* Are x/y plausibly meters (UTM-like) or degrees?
* What are the min/max/mean/std of raw t/x/y in train split?
* Were coords shifted/normalized?
* Which columns were used as x/y/t?

Use it to detect “silent pipeline drift” between sites.

Stage-2 handshake audit (``stage2_handshake_audit.json``)
-----------------------------------------------------------

This file is designed to answer:

* Do input dict keys exist and match expectations?
* Are shapes consistent with ``forecast_horizon`` and window sizes?
* Does the resolved scaling payload contain required keys?

If Stage-2 throws shape errors or physics anomalies, this audit is the
fastest way to determine whether the issue is data-side or model-side.


10. Checklist: make physics stable on day one
================================================

**Coords**
* [ ] ``coord_order == ["t","x","y"]`` (or your declared order matches reality)
* [ ] If ``coords_in_degrees=True``: you provided ``deg_to_m_lon/lat``
* [ ] If ``coords_normalized=True``: you provided ``coord_ranges`` (or SI spans)

**Time**
* [ ] ``time_units`` matches your dataset t-axis meaning
* [ ] Your t coordinate increments are consistent (e.g., yearly steps)

**Targets**
* [ ] ``subs_scale_si`` converts your stored unit to meters (mm → 1e-3)
* [ ] If groundwater is depth-like: you provided ``z_surf`` metadata/values

**Thickness**
* [ ] ``H_field`` is positive and in meters after scaling (or set a safe floor)

If you complete this checklist, GeoPrior’s physics losses and epsilon metrics
are usually well-behaved, and remaining issues are typically architectural or
optimization-related (see :doc:`training` and :doc:`diagnostics`).
