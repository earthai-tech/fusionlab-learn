.. _geoprior_core_guide:

===========================================================
GeoPrior Core: Physics Core and Residual Assembly
===========================================================

:API Reference:
    - :func:`~fusionlab.nn.pinn.geoprior.step_core.physics_core`
    - :func:`~fusionlab.nn.pinn.geoprior.derivatives.compute_head_pde_derivatives_raw`
    - :func:`~fusionlab.nn.pinn.geoprior.derivatives.ensure_si_derivative_frame`

GeoPrior is trained with a **hybrid objective**: a supervised data loss
(subsidence/head targets) plus a **physics regularizer** that penalizes
violations of the governing flow and compaction dynamics.

The central design goal of the GeoPrior implementation is to keep all
physics computations (units, derivatives, residuals, scaling, warmup
gates) in a **single source of truth**: the function
:func:`~fusionlab.nn.pinn.geoprior.step_core.physics_core`.

This page documents the core workflow at a *conceptual* and *mathematical*
level. It is intentionally implementation-close: the section numbering
mirrors the internal assembly steps and the returned diagnostic bundle.

.. note::

   This file is written in two parts.

   **Part 1** explains what ``physics_core`` computes and how
   the residuals are formed.

   **Part 2** (to be written next) will dive into the derivative engine,
   chain-rule conversion, and stability safeguards (finite checks, NaN-safe
   mapping, scaling floors, and debug modes).

Why a dedicated physics core?
-----------------------------

A GeoPrior training/evaluation step needs to compute, consistently:

1) SI-consistent **physical fields** (:math:`K, S_s, \tau, H`) from raw logits
   and coordinate-correction MLPs.

2) **Derivatives** of predicted head with respect to the same coordinates
   tensor fed into the forward pass.

3) Physics **residuals** for:
   - groundwater flow (PDE),
   - consolidation/compaction (ODE/PDE-like relaxation),
   - priors/regularizers (timescale prior, smoothness, bounds, optional MV prior),
   - and optional source/sink forcing :math:`Q(t,x,y)`.

4) Optional **nondimensionalization** and **warmup gating** so physics terms
   do not destabilize early training.

The function :func:`~fusionlab.nn.pinn.geoprior.step_core.physics_core`
returns a “physics bundle” with all residuals, scaled residuals, loss
components, and epsilon diagnostics used in logs/plots.

.. _geoprior_physics_core:

-------------------------------------------------------------------------------
Physics core
-------------------------------------------------------------------------------

At a high level, ``physics_core`` does:

.. code-block:: text

   (predictions + aux logits)
            |
            v
   compose_physics_fields  ->  K, Ss, tau (+ logs, prior residual)
            |
            v
   derivatives (raw)  ->  chain-rule -> derivatives in SI
            |
            v
   Q conversion + gate
            |
            v
   consolidation residual     groundwater residual
            |                         |
            v                         v
      priors & bounds  + optional nondimensionalization
            |
            v
   physics losses + warmup gate  ->  physics bundle (metrics + terms)

Inputs and contracts
--------------------

``physics_core`` is called from the model’s custom training/evaluation logic,
and assumes the standard GeoPrior batch layout:

- ``inputs_fwd``: dictionary inputs forwarded to the model, including
  ``coords`` for physics derivatives.

- ``y_pred``: model outputs (mean paths and/or quantiles) for head and subsidence.

- ``aux``: auxiliary forward payload (raw logits for physical fields, plus
  optional internal tensors needed for diagnostics).

- ``scaling_kwargs``: the canonical scaling/units dictionary produced in Stage-1
  and validated/resolved in the model constructor.

The scaling payload includes important flags such as:

- ``time_units`` (e.g. ``"year"``),
- ``coords_normalized`` and ``coord_ranges`` (for chain-rule),
- ``gwl_kind`` / ``gwl_sign`` / optional head proxy behavior,
- residual unit display choices (e.g. seconds),
- consolidation drawdown options,
- and physics scaling floors and warmup schedules.

.. tip::

   Treat ``scaling_kwargs`` as part of the model definition. If you change how
   Stage-1 generates it, Stage-2 physics will change.

Governing equations enforced by GeoPrior
----------------------------------------

GeoPrior targets two coupled dynamical components:

**(1) Groundwater flow (2D transient PDE)**

.. math::

   \mathcal{R}_{gw}
   \;=\;
   S_s\,\frac{\partial h}{\partial t}
   \;-\;
   \nabla\cdot\Bigl(K\,\nabla h\Bigr)
   \;-\;
   Q
   \;=\; 0

Expanded in 2D:

.. math::

   \nabla\cdot(K\nabla h)
   \;=\;
   \frac{\partial}{\partial x}\Bigl(K\,\frac{\partial h}{\partial x}\Bigr)
   \;+\;
   \frac{\partial}{\partial y}\Bigl(K\,\frac{\partial h}{\partial y}\Bigr)

**(2) Consolidation / compaction (relaxation ODE)**

GeoPrior uses a relaxation form where incremental compaction evolves toward
an equilibrium compaction :math:`s_{eq}` with timescale :math:`\tau`:

.. math::

   \frac{\partial s}{\partial t}
   \;=\;
   \frac{s_{eq}(h) - s}{\tau}

A typical equilibrium proxy used in hydrogeology is proportional to drawdown:

.. math::

   s_{eq}(h)\;\approx\; S_s\,\Delta h \, H

where :math:`H` is an effective compressible thickness and
:math:`\Delta h` is a drawdown defined relative to a reference head.

GeoPrior computes a **step residual** (with an “exact” step integrator)
and then converts it to a residual form used as a loss.

Step-by-step: what ``physics_core`` computes
--------------------------------------------

This section follows the internal assembly order.

1) Compose physical fields from logits + coordinate corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GeoPrior predicts (or receives via ``aux``) raw log-parameters for physical
fields, and then maps them into safe, bounded SI fields:

- hydraulic conductivity :math:`K` (m/s),
- specific storage :math:`S_s`,
- consolidation timescale :math:`\tau` (s),
- thickness proxies :math:`H` (m) and derived drainage thickness :math:`H_d`.

The mapping is **NaN-safe** and **overflow-safe**: raw log-parameters are
never directly exponentiated without guards/bounds (hard or soft bound mode).
The function also returns log-space values (``logK``, ``logSs``, ``log_tau``)
for priors/diagnostics.

.. note::

   The bounds mode affects whether the logs are clipped (“hard”) or only
   guarded at exp-time (“soft”). Bounds penalties are handled later.

2) Compute head derivatives in the same coordinate system as the forward pass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GeoPrior computes derivatives using automatic differentiation, *then*
converts them to SI via a chain-rule transform when coordinates are normalized
(and optionally when coordinates originate from degrees).

This yields SI derivatives such as:

- :math:`\partial h/\partial t`,
- divergence terms :math:`\partial_x(K\partial_x h)`, :math:`\partial_y(K\partial_y h)`.

(Full details will be documented in :ref:`Derivatives and stability controls <geoprior_core_derivatives>`.)


3) Convert source/sink forcing ``Q`` to SI and apply the Q gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the model provides a learnable forcing logits tensor (``Q_logits``),
it is converted into an SI-consistent source term ``Q_si`` using the same
metadata that controls time normalization and units.

A train-time **gate** is applied to avoid early instability:

- if the model defines ``model._q_gate()``, it is used,
- otherwise the gate defaults to 1.

Additionally, GeoPrior tracks a small regularization on the forcing magnitude:

.. math::

   \mathcal{L}_{Q} \;=\; \mathbb{E}[Q_{si}^2]

This keeps forcing from becoming the “easy shortcut” that explains everything.

4) Consolidation residual (exact-step) using incremental settlement state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consolidation physics is only active if:

- the model’s PDE mode includes consolidation, and
- the scaling config allows a subsidence residual pathway (``allow_subs_residual``).

GeoPrior then:

- converts the predicted subsidence mean to SI,
- builds an *incremental* settlement state suitable for step integration,
- concatenates initial state + predicted increments,
- forms a head state using a reference head :math:`h_{ref}` and predicted head,
- applies drawdown options (mode/rule/clipping/stop-grad policy),
- computes an exact-step residual,
- converts the step residual to the requested residual units.

Mathematically, the “exact-step” integrator corresponds to:

.. math::

   s_{k+1}
   \;=\;
   s_k + \bigl(s_{eq}(h_k) - s_k\bigr)
   \Bigl(1 - \exp(-\Delta t/\tau)\Bigr)

and the residual measures mismatch between the predicted increment and the
physics-driven increment.

5) Groundwater flow residual + priors (timescale, smoothness, MV, bounds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once SI derivatives and ``Q_si`` are available, GeoPrior forms:

**Groundwater residual** (PDE):

.. math::

   \mathcal{R}_{gw}
   \;=\;
   S_s\,\frac{\partial h}{\partial t}
   \;-\;
   \Bigl[\partial_x(K\partial_x h) + \partial_y(K\partial_y h)\Bigr]
   \;-\;
   Q_{si}

**Timescale prior residual** (log-space):

GeoPrior exposes a log-timescale mismatch residual (typically
:math:`\log \tau - \log\tau_{prior}`), returned as ``prior_res`` and used
as a penalty term.

**Smoothness prior**:

Spatial gradients of the fields are penalized to encourage physically plausible
smooth parameter fields.

**MV prior (optional)**:

A learnable MV prior can be computed from :math:`S_s` and its log, with a
schedule controlled by scaling config.

**Bounds residual**:

Bounds are enforced both by mapping and by an explicit residual/penalty on
out-of-range values for thickness and log-fields.

6) Display units for diagnostics (raw only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For logging and debugging, GeoPrior can convert residuals into a
“display unit” (e.g., multiply by seconds-per-time-unit when the configured
unit is ``time_unit``). This does not change the SI residual used for
physics scaling; it is a **diagnostic view**.

7) Optional nondimensionalization (auto-scaling of residuals)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``model.scale_pde_residuals`` is enabled, GeoPrior computes characteristic
scales for consolidation and groundwater residuals and rescales them:

.. math::

   \tilde{\mathcal{R}} \;=\; \frac{\mathcal{R}}{\max(\mathrm{scale}, \mathrm{floor})}

Scaling is made robust by:

- automatic floors (``"auto"``) or explicit floors from config,
- guarding scales against non-finite values,
- optionally stopping gradients through scale estimates.

8) Physics losses, epsilons, and the warmup gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GeoPrior forms scalar losses as mean-square residuals:

.. math::

   \mathcal{L}_{cons}=\mathbb{E}[\tilde{\mathcal{R}}_{cons}^2],
   \quad
   \mathcal{L}_{gw}=\mathbb{E}[\tilde{\mathcal{R}}_{gw}^2]

and similarly for priors/smoothness/bounds/MV/Q regularization.

It also computes RMS “epsilons” for both raw and scaled residuals, e.g.:

- ``eps_cons_raw`` vs ``eps_cons``,
- ``eps_gw_raw`` vs ``eps_gw``,
- ``eps_prior``.

Finally, during training, GeoPrior applies a **warmup+linear ramp gate**
to the *scaled physics* term, controlled by:

- ``physics_warmup_steps``,
- ``physics_ramp_steps``.

This ensures early epochs learn a stable data-driven solution before physics
regularization becomes dominant.

What gets returned (physics bundle)
-----------------------------------

``physics_core`` returns a dictionary-like bundle containing, at minimum:

- residual tensors (raw + scaled): consolidation and groundwater,
- auxiliary residuals: timescale prior, smoothness, bounds,
- scalar losses per component and assembled physics loss terms,
- epsilon diagnostics (RMS),
- optional computed scales and scaling metadata,
- Q diagnostics (``q_rms``) and Q regularization loss.

This bundle is attached to training logs and is the basis of the
GeoPrior diagnostics utilities (history plots, epsilon plots, residual plots).

.. _geoprior_core_derivatives:

-------------------------------------------------------------------------------
Derivatives, chain rule, and stability controls
-------------------------------------------------------------------------------

This section documents the derivative engine and the stability machinery that
makes GeoPrior’s PINN losses trainable on real data (normalized coordinates,
heterogeneous scales, and imperfect signals).

The key idea is:

1) **Differentiate in model space** (whatever coords you feed to the forward
   pass).
2) **Convert derivatives to SI** (meters + seconds) using a chain-rule frame.
3) Build residuals and (optionally) **nondimensionalize** them for stable
   weighting.
4) Apply **warmup/ramp gates** so physics influence increases gradually.

The derivative engine
---------------------

GeoPrior computes PDE derivatives with automatic differentiation using the
same ``coords`` tensor that is injected into the forward pass. Conceptually:

.. math::

   h(t,x,y) = \text{model}( \text{features}, \text{coords} )

and we need:

.. math::

   \partial_t h,\quad \partial_x h,\quad \partial_y h,\quad
   \partial_{xx} h,\quad \partial_{yy} h

Implementation pattern (conceptual)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A robust PDE derivative computation has three non-negotiable rules:

1) **Differentiate w.r.t. the exact coords used in the forward call**.
   Any mismatch makes residuals meaningless.

2) Use a tape configuration that supports second derivatives.
   Practically, this means either:
   - nested tapes, or
   - a persistent tape (careful with memory).

3) Differentiate only **the differentiable mean head path**.
   If quantiles are enabled, derivatives are computed from the mean (or q50)
   to avoid non-smooth behavior in quantile assembly.

A minimal conceptual skeleton:

.. code-block:: python
   :linenos:

   with tf.GradientTape(persistent=True) as tape2:
       tape2.watch(coords)
       with tf.GradientTape() as tape1:
           tape1.watch(coords)
           y_pred = model(inputs, training=True)   # coords included
           h = head_mean(y_pred)                   # (B,H,1)
       dh_dcoords = tape1.gradient(h, coords)      # (B,H,3)
   d2h_dx2 = tape2.gradient(dh_dcoords[..., x_i], coords)[..., x_i]
   d2h_dy2 = tape2.gradient(dh_dcoords[..., y_i], coords)[..., y_i]

Raw derivative frame (what it represents)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The derivative helper returns a *raw derivative frame*:

- derivatives are taken w.r.t. **the coords as provided to the model**
  (possibly normalized, possibly in degrees).
- units are therefore “model-space units”, not necessarily SI.

Typical keys (conceptual):

.. code-block:: text

   dh_dt_raw, dh_dx_raw, dh_dy_raw
   d2h_dxx_raw, d2h_dyy_raw
   (optionally) divKgrad_raw or components needed to build it

The chain-rule conversion to SI
-------------------------------

GeoPrior supports normalized coordinates and (optionally) lon/lat degrees.
The conversion is applied after autodiff to obtain **SI-consistent** derivatives.

Case A: coords are normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let the model use normalized coordinates:

.. math::

   \hat{t},\hat{x},\hat{y} \in [0, 1]

and let the physical coordinates be:

.. math::

   t = t_{min} + \hat{t}\,\Delta t,\qquad
   x = x_{min} + \hat{x}\,\Delta x,\qquad
   y = y_{min} + \hat{y}\,\Delta y

Then chain rule gives:

.. math::

   \frac{\partial h}{\partial t}
   = \frac{1}{\Delta t}\,\frac{\partial h}{\partial \hat{t}},\qquad
   \frac{\partial h}{\partial x}
   = \frac{1}{\Delta x}\,\frac{\partial h}{\partial \hat{x}},\qquad
   \frac{\partial h}{\partial y}
   = \frac{1}{\Delta y}\,\frac{\partial h}{\partial \hat{y}}

Second derivatives scale quadratically:

.. math::

   \frac{\partial^2 h}{\partial x^2}
   = \frac{1}{\Delta x^2}\,\frac{\partial^2 h}{\partial \hat{x}^2},\qquad
   \frac{\partial^2 h}{\partial y^2}
   = \frac{1}{\Delta y^2}\,\frac{\partial^2 h}{\partial \hat{y}^2}

**This is the most common source of physics explosions**:
a wrong ``coord_ranges["x"]`` or ``coord_ranges["y"]`` will distort
second derivatives by orders of magnitude.

GeoPrior gets :math:`\Delta t,\Delta x,\Delta y` from:

- ``scaling_kwargs["coord_ranges"]`` (dataset units), and then converts time to
  seconds if needed, OR
- ``scaling_kwargs["coord_ranges_si"]`` if provided (preferred).

Case B: coords are in degrees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If spatial coords are lon/lat degrees, GeoPrior requires meters-per-degree
conversion:

- ``deg_to_m_lon``
- ``deg_to_m_lat``

Then:

.. math::

   \Delta x_{m} = \Delta x_{deg} \cdot \mathrm{deg\_to\_m\_lon},\qquad
   \Delta y_{m} = \Delta y_{deg} \cdot \mathrm{deg\_to\_m\_lat}

and the same chain-rule formulas apply using :math:`\Delta x_m, \Delta y_m`.

Time-unit conversion
^^^^^^^^^^^^^^^^^^^^

Physics residuals are enforced in SI. If ``time_units != "second"``, GeoPrior
converts the time span:

.. math::

   \Delta t_{sec} = \Delta t_{unit}\cdot \mathrm{sec\_per\_unit}

and uses :math:`\Delta t_{sec}` in the chain rule. This ensures
:math:`\partial_t h` has units of meters per second.

The SI derivative frame
-----------------------

After conversion, GeoPrior constructs an *SI derivative frame* that is used
downstream by residual builders. Conceptually, it contains:

- SI derivatives: :math:`\partial_t h`, :math:`\partial_x h`, :math:`\partial_y h`,
  :math:`\partial_{xx}h`, :math:`\partial_{yy}h`
- scaling metadata used for diagnostics (spans, sec/unit, degree conversion)
- safe flags for finiteness and fallback behaviors

This separation is deliberate:
derivative computation stays model-agnostic, while SI conversion is fully
controlled by ``scaling_kwargs``.

Building divergence terms robustly
----------------------------------

The groundwater PDE uses a divergence form:

.. math::

   \nabla\cdot(K\nabla h)
   =
   \partial_x(K\,\partial_x h) + \partial_y(K\,\partial_y h)

GeoPrior computes this in a numerically stable way. In heterogeneous fields,
it is usually better to compute:

- :math:`K` and :math:`\partial_x h` in SI,
- then form :math:`K\,\partial_x h`,
- then take derivatives consistently w.r.t. the same coordinates.

If you instead expand to :math:`K\,\partial_{xx}h + (\partial_x K)(\partial_x h)`
you must also compute :math:`\partial_x K`, which is more sensitive to noise.
GeoPrior uses the divergence-form workflow.

Residual scaling (nondimensionalization)
----------------------------------------

Raw residual magnitudes can differ wildly across datasets and training phases.
GeoPrior therefore supports optional nondimensionalization:

.. math::

   \tilde{R} = \frac{R}{\max(s, s_{floor})}

where:

- :math:`R` is the raw residual (consolidation or groundwater),
- :math:`s` is a characteristic scale estimated from batch statistics,
- :math:`s_{floor}` is a robust floor to prevent division by tiny numbers.

This has two benefits:

1) The physics loss becomes **comparable across runs**.
2) The physics weight hyperparameters become **less brittle**.

In logs you typically see both:

- ``epsilon_*_raw``: RMS of raw residual
- ``epsilon_*``: RMS of scaled residual

Warmup and ramp gates
---------------------

Physics regularization is intentionally introduced gradually.

GeoPrior uses a gate:

- warmup steps: physics multiplier stays near 0 (or a small offset),
- ramp steps: multiplier increases linearly to 1,
- thereafter: multiplier stays at 1.

Conceptually:

.. math::

   g(\text{step}) =
   \begin{cases}
     0, & \text{step} < W\\
     \frac{\text{step}-W}{R}, & W \le \text{step} < W+R\\
     1, & \text{step} \ge W+R
   \end{cases}

and the final physics contribution is:

.. math::

   \mathcal{L}_{phys,final} =
   \lambda_{offset}\,g(\text{step})\cdot \mathcal{L}_{phys}

This is what prevents early training from being dominated by noisy
second-derivative terms.

Numerical stability policies
----------------------------

GeoPrior includes multiple stability guards. The most important ones:

Finite guards (NaN/Inf)
^^^^^^^^^^^^^^^^^^^^^^^

- All critical tensors are checked for finiteness before they can dominate loss.
- Non-finite values are replaced with safe defaults for *diagnostics* and the
  step can optionally skip/clip the offending term.

Floors for thickness and scales
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Thickness-like signals (``H_field``) use a positive floor (e.g. ``H_floor``).
- Residual scaling uses robust floors to prevent divide-by-zero.

NaN-safe log-parameter mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Physical fields are produced from logits in log-space.
To prevent NaN propagation:

- logits are sanitized before clipping,
- exponentiation is done after bounding.

This matters because ``clip_by_value(NaN, a, b)`` remains NaN.

Drawdown construction policies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If head is constructed from groundwater depth (or a proxy), GeoPrior supports:

- stop-gradient on reference head,
- clipping drawdown ranges,
- optional “always positive drawdown” modes.

These policies live in scaling config to keep data conventions explicit.

Debugging derivative/residual failures
--------------------------------------

If physics looks wrong, isolate the issue in this order:

1) **Confirm coordinate spans**
   - Are x/y in meters?
   - If coords are normalized, are ``coord_ranges`` correct?

2) **Confirm time units**
   - Is ``time_units`` correct (year vs day vs second)?
   - Is ``Δt`` consistent with your dataset?

3) **Confirm head semantics**
   - If groundwater is depth-like, is ``z_surf`` available?
   - If not, is the head proxy documented and acceptable?

4) **Compare raw vs scaled epsilons**
   - huge ``epsilon_gw_raw`` with modest ``epsilon_gw`` often indicates
     scaling is working but raw units are large (still check spans).
   - huge values in both raw and scaled typically indicate wrong spans/units
     or derivative mismatch.

5) **Quantile layout sanity**
   - If quantiles are enabled, ensure you compute derivatives on mean/q50,
     not on a non-smooth quantile-assembled tensor.

Common symptom → likely cause
-----------------------------

**``epsilon_gw_raw`` explodes immediately**
  - Wrong ``coord_ranges["x"]`` / ``coord_ranges["y"]``,
    or missing degree→meter conversion.

**Consolidation residual explodes but GW residual is fine**
  - Wrong ``time_units`` (Δt in years but treated as seconds),
    or thickness field not in meters / near zero.

**Residuals are ~0 but predictions are poor**
  - ``pde_mode`` is off, warmup gate never ramps up, or physics weights are 0.

**Training becomes NaN at a random step**
  - Non-finite logits, thickness underflow, or unguarded exp/log;
    enable finiteness checks and inspect aux tensors from
    ``forward_with_aux``.

What’s next
-----------

The next documentation pages connect these mechanics to user-facing knobs:

- :doc:`losses` (exact loss decomposition and epsilon definitions)
- :doc:`training` (practical schedules for warmup/ramp and weights)
- :doc:`diagnostics` (plots that catch unit/derivative issues early)
