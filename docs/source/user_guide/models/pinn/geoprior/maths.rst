.. _geoprior_maths_guide:

=========================================================
GeoPrior Maths: Definitions, Closures, and Conventions
=========================================================

:API Reference:
    - :mod:`~fusionlab.nn.pinn.geoprior.maths`
    - :func:`~fusionlab.nn.pinn.geoprior.maths.compose_physics_fields`
    - :func:`~fusionlab.nn.pinn.geoprior.maths.tau_phys_from_fields`
    - :func:`~fusionlab.nn.pinn.geoprior.maths.equilibrium_compaction_si`
    - :func:`~fusionlab.nn.pinn.geoprior.maths.compute_mv_prior`
    - :func:`~fusionlab.nn.pinn.geoprior.maths.q_to_gw_source_term_si`
    - :func:`~fusionlab.nn.pinn.geoprior.maths.compute_bounds_residual`

This page defines the mathematical objects used by GeoPrior’s PINN engine:
effective physical fields, closures (especially :math:`\tau`), equilibrium
settlement, soft/hard bounds, the optional :math:`m_v` prior, and the
groundwater forcing term :math:`Q`.

It is intentionally written in **two layers**:

1) **Definitions & conventions** (this file section): what each symbol means,
   its units, and how GeoPrior interprets it.

2) **Behind-the-scenes derivations & implementation details** (added later):
   how each residual/closure is assembled, stabilized, and differentiated.

.. seealso::

   For how these definitions appear in the training loop, see:
   :doc:`core` (physics core + derivative conversion) and :doc:`losses`
   (term-by-term loss assembly and log names).

-------------------------------------------------------------------------------
1. Notation and units
-------------------------------------------------------------------------------

GeoPrior couples two primary state variables:

- :math:`h(t,x,y)` : hydraulic head in meters (or a head-like proxy that is
  treated consistently as meters within the physics core).
- :math:`s(t,x,y)` : subsidence (settlement) in meters.

Coordinates are:

- :math:`t` : time (converted internally to seconds for physics).
- :math:`x,y` : planar coordinates in meters (UTM recommended).

The physics core enforces equations in **SI**:

- :math:`K > 0` : hydraulic conductivity (m/s).
- :math:`S_s > 0` : specific storage (1/m).
- :math:`H > 0` : drained (compressible) thickness (m).
- :math:`\tau > 0` : consolidation timescale (s).
- :math:`Q_{term}` : groundwater source/sink term entering the PDE residual
  (units 1/s in GeoPrior’s residual form).

.. note::

   This project typically uses **normalized coordinates** in the model
   forward pass. Derivatives are converted back to SI using a chain-rule
   frame (see :doc:`core`). The definitions here always describe the
   **SI-consistent** quantities used in residual evaluation.

-------------------------------------------------------------------------------
2. Learned effective fields: K, Ss, tau
-------------------------------------------------------------------------------

GeoPrior learns **effective** (grid-scale) physical fields rather than
laboratory parameters. These are produced by the physics head and then mapped
into positive SI quantities by
:func:`~fusionlab.nn.pinn.geoprior.maths.compose_physics_fields`.

2.1 Raw log-parameters and coordinate corrections
-------------------------------------------------

Let the model produce base logits (typically shaped like ``(B, H, 1)``):

- :math:`\ell_K^{base}` for :math:`\log K`
- :math:`\ell_{S_s}^{base}` for :math:`\log S_s`
- :math:`\ell_{\tau}^{base}` for a residual in :math:`\log\tau`

GeoPrior adds **time-invariant** spatial corrections from coordinate MLPs
evaluated on zeroed-time coordinates:

.. math::

   \tilde{\mathbf{c}} = (0, x, y)

.. math::

   \ell_K = \ell_K^{base}(t,x,y) + \Delta \ell_K(\tilde{\mathbf{c}})

.. math::

   \ell_{S_s} = \ell_{S_s}^{base}(t,x,y) + \Delta \ell_{S_s}(\tilde{\mathbf{c}})

The physical fields are exponentials of these logs:

.. math::

   K = \exp(\ell_K), \qquad S_s = \exp(\ell_{S_s})

subject to the configured bounds policy (Section 5).

2.2 Timescale as “closure + residual”
-------------------------------------

GeoPrior does not learn :math:`\tau` purely freely. Instead it uses:

- a **physics closure** :math:`\tau_{phys}` derived from :math:`K,S_s,H`, and
- a **learned residual** in log-space.

Let :math:`\log\tau_{phys} = f_\tau(K,S_s,H;\text{options})`. The model predicts
a residual :math:`\Delta \log\tau` and forms:

.. math::

   \log\tau = \log\tau_{phys} + \Delta \log\tau

.. math::

   \tau = \exp(\log\tau) + \varepsilon_\tau

where :math:`\varepsilon_\tau > 0` is a small floor (``eps_tau``) for stability.

See :func:`~fusionlab.nn.pinn.geoprior.maths.tau_phys_from_fields` for the exact
closure modes.

-------------------------------------------------------------------------------
3. Tau closure: :math:`\tau_{phys}` and effective drainage thickness
-------------------------------------------------------------------------------

The closure returned by :func:`~fusionlab.nn.pinn.geoprior.maths.tau_phys_from_fields`
also defines an effective drainage thickness :math:`H_d`.

3.1 Effective drainage thickness
--------------------------------

GeoPrior optionally uses:

.. math::

   H_d =
   \begin{cases}
     H \cdot f_{Hd}, & \text{if use\_effective\_thickness is True} \\
     H,              & \text{otherwise}
   \end{cases}

where :math:`f_{Hd}` is ``model.Hd_factor``. This is a pragmatic knob to account
for unresolved drainage geometry at grid scale.

3.2 Two closure modes (kappa_mode)
----------------------------------

Let :math:`\kappa>0` be a model-provided scalar (``model._kappa_value()``) and
:math:`\pi^2` the constant.

GeoPrior supports two branches:

**(A) ``kappa_mode="bar"``**

.. math::

   \tau_{phys}
   = \frac{\kappa \, H^2 \, S_s}{\pi^2 \, K}

**(B) other modes (“non-bar” branch)**

.. math::

   \tau_{phys}
   = \frac{H_d^2 \, S_s}{\pi^2 \, \kappa \, K}

To improve numerical stability, the implementation computes
:math:`\log(\tau_{phys})` first and exponentiates at the end.

.. note::

   These are **effective closures** designed for stable learning, not a claim
   that the medium strictly follows a single analytic derivation everywhere.
   The learned residual :math:`\Delta\log\tau` compensates for closure mismatch.

-------------------------------------------------------------------------------
4. Equilibrium settlement: :math:`s_{eq}`
-------------------------------------------------------------------------------

The consolidation component compares the current settlement state to an
equilibrium settlement implied by drawdown. GeoPrior defines equilibrium
compaction in SI meters via
:func:`~fusionlab.nn.pinn.geoprior.maths.equilibrium_compaction_si`.

4.1 Drawdown definition
-----------------------

Let :math:`h_{mean}` be the mean head and :math:`h_{ref}` a reference head.
A raw drawdown is formed by a rule:

.. math::

   \Delta h_{raw} =
   \begin{cases}
     h_{ref} - h_{mean}, & \text{rule = ref\_minus\_mean} \\
     h_{mean} - h_{ref}, & \text{rule = mean\_minus\_ref}
   \end{cases}

GeoPrior then applies a gating operator to enforce non-negative drawdown:

.. math::

   \Delta h = [\Delta h_{raw}]_+

The exact gating operator is selected by ``drawdown_mode`` (e.g. hard ReLU,
softplus, smooth-ReLU with ``relu_beta``). Optional clipping can be applied to
prevent extreme values early in training.

4.2 Equilibrium settlement
--------------------------

With :math:`S_s` (1/m) and thickness :math:`H` (m):

.. math::

   s_{eq} = S_s \, \Delta h \, H

Units are consistent: :math:`(1/m)\cdot(m)\cdot(m)=m`.

.. note::

   GeoPrior can stop gradients through the reference head
   (:math:`h_{ref} := stop\_gradient(h_{ref})`) to prevent the model from
   “cheating” by shifting the reference rather than correcting the predicted
   head field.

-------------------------------------------------------------------------------
5. Bounds, safe transforms, and the bounds residual
-------------------------------------------------------------------------------

GeoPrior uses bounds in two complementary ways:

1) **parameter mapping bounds** (during field construction), and
2) **a soft penalty** based on bound violation residuals.

The penalty is produced by
:func:`~fusionlab.nn.pinn.geoprior.maths.compute_bounds_residual`.

5.1 Bounds mode: hard vs soft
-----------------------------

GeoPrior supports two conceptual policies:

- ``bounds_mode="hard"``:
  log-parameters are clipped into configured intervals before exponentiation.

- ``bounds_mode="soft"``:
  raw logs are kept (so violations can be penalized), while exponentiation is
  guarded to prevent float32 overflow. This is often preferable for training
  stability while still providing a meaningful penalty signal.

5.2 Bound violation residual (definition)
-----------------------------------------

For a scalar parameter :math:`z` with bounds :math:`z_{min}\le z \le z_{max}`,
define:

.. math::

   v(z) = \max(z_{min} - z, 0) + \max(z - z_{max}, 0)

GeoPrior returns a range-normalized residual:

.. math::

   R(z) = \frac{v(z)}{\max(z_{max} - z_{min}, \varepsilon)}

For log-bounded parameters, this definition is applied in log space. For
instance, if :math:`\ell_K=\log K` has bounds
:math:`\ell_{K,min} \le \ell_K \le \ell_{K,max}`, then:

.. math::

   R_K = R(\ell_K;\ell_{K,min},\ell_{K,max})

The bounds residual term used by the physics loss is typically:

.. math::

   \mathcal{L}_{bounds} = \mathrm{mean}(R_H^2 + R_K^2 + R_{S_s}^2 + R_\tau^2)

where each residual map is computed by
:func:`~fusionlab.nn.pinn.geoprior.maths.compute_bounds_residual`.

-------------------------------------------------------------------------------
6. The m_v prior: tying Ss to compressibility
-------------------------------------------------------------------------------

GeoPrior optionally adds a prior linking storage to compressibility through
water unit weight:

.. math::

   S_s \approx m_v\,\gamma_w

This is implemented by :func:`~fusionlab.nn.pinn.geoprior.maths.compute_mv_prior`.

6.1 Log-space residual
----------------------

The constraint is applied in log space:

.. math::

   r = \log(S_s) - \log(m_v\,\gamma_w)

6.2 Two-part objective (global + dispersion)
--------------------------------------------

When used as a loss, GeoPrior penalizes:

- a global mismatch based on the mean residual :math:`\bar{r}`,
- a dispersion term discouraging scatter around :math:`\bar{r}`.

Using a Huber penalty :math:`\mathrm{Huber}(\cdot;\delta)`:

.. math::

   L_g = \mathrm{Huber}(\bar{r};\delta)

.. math::

   L_d = \mathrm{mean}\bigl(\mathrm{Huber}(r-\bar{r};\delta)\bigr)

.. math::

   L = L_g + \alpha_{disp}\,L_d

The **mode** controls gradient flow:

- ``calibrate`` (default): stop gradients through :math:`S_s` before logging it,
  so :math:`m_v` is calibrated without reshaping the learned :math:`S_s` field.
- ``field``: allow gradients through :math:`S_s` (can amplify gradients as
  :math:`\partial\log S_s/\partial S_s = 1/S_s`).
- ``logss``: allow gradients through :math:`\log S_s` directly (often more
  stable than ``field``).

-------------------------------------------------------------------------------
7. Groundwater forcing Q: conversion and meaning
-------------------------------------------------------------------------------

GeoPrior supports an optional learnable forcing term used in the groundwater
residual:

.. math::

   R_{gw} = S_s\,\frac{\partial h}{\partial t} - \nabla\cdot(K\nabla h) - Q_{term}

The key design choice is that the residual expects **Q in 1/s**, regardless of
how the network produced it. The conversion is implemented by
:func:`~fusionlab.nn.pinn.geoprior.maths.q_to_gw_source_term_si`.

7.1 Supported meanings (Q_kind)
-------------------------------

GeoPrior resolves a ``Q_kind`` string from scaling configuration:

- ``per_volume`` (default):
  the network output is an inverse-time forcing already (1/time_unit or 1/s).

- ``recharge_rate``:
  the network output is a length-rate recharge :math:`R` (m/time_unit or m/s)
  converted to a volumetric rate by dividing by thickness:

  .. math::

     Q_{term} = \frac{R}{H_{safe}}

- ``head_rate``:
  the network output is a head-rate forcing :math:`q_h` (m/time_unit or m/s)
  entering the storage term. The equivalent forcing in the residual is:

  .. math::

     Q_{term} = S_s\,q_h

7.2 Normalized-time correction (coords_normalized)
--------------------------------------------------

If coords are normalized, the model may produce :math:`Q` with respect to the
normalized time coordinate. GeoPrior applies a chain-rule correction to map it
back to the intended time units, and then converts to SI seconds using the
configured ``time_units`` mapping.

.. note::

   This time-normalization rule is the same conceptual correction used for
   :math:`\partial h/\partial t` conversion in the derivative frame
   (see :doc:`core`).

-------------------------------------------------------------------------------
Part 2: Residual assembly, exact-step integrator, scaling, and stability
-------------------------------------------------------------------------------

This section expands the “definitions” into the concrete mathematics used
inside GeoPrior to build residual maps and stable loss terms.

-------------------------------------------------------------------------------
A) Residual assembly from derivatives
-------------------------------------------------------------------------------

A.1 Groundwater residual :math:`R_{gw}`
=======================================

GeoPrior enforces the transient groundwater flow equation in divergence form:

.. math::

   R_{gw}
   \;=\;
   S_s\,\frac{\partial h}{\partial t}
   \;-\;
   \nabla\cdot(K\nabla h)
   \;-\;
   Q_{term}.

In 2D, the divergence term is:

.. math::

   \nabla\cdot(K\nabla h)
   \;=\;
   \frac{\partial}{\partial x}\Bigl(K\,\frac{\partial h}{\partial x}\Bigr)
   \;+\;
   \frac{\partial}{\partial y}\Bigl(K\,\frac{\partial h}{\partial y}\Bigr).

GeoPrior computes derivatives by automatic differentiation w.r.t. the same
coordinates used in the forward pass, then converts derivatives to SI
(using chain rule when coords are normalized or in degrees; see :doc:`core`).

Let the SI derivative frame provide:

.. math::

   h_t = \frac{\partial h}{\partial t},\quad
   h_x = \frac{\partial h}{\partial x},\quad
   h_y = \frac{\partial h}{\partial y}.

Define flux-like terms:

.. math::

   u = K\,h_x,\qquad v = K\,h_y.

Then:

.. math::

   \nabla\cdot(K\nabla h) \;=\; u_x + v_y.

and the residual is:

.. math::

   R_{gw} \;=\; S_s\,h_t - (u_x + v_y) - Q_{term}.

.. note::

   The divergence form is preferred because it remains meaningful when
   :math:`K(x,y)` is heterogeneous. Expanding it into
   :math:`K h_{xx} + (\partial_x K)h_x` requires reliable :math:`\partial_x K`,
   which is often noisier and more sensitive to coordinate scaling.

A.2 Consolidation residual :math:`R_{cons}`
===========================================

GeoPrior uses a relaxation form for compaction/settlement:

.. math::

   \frac{\partial s}{\partial t}
   \;=\;
   \frac{s_{eq}(h) - s}{\tau}.

In GeoPrior, :math:`s_{eq}` is computed from drawdown and thickness:

.. math::

   s_{eq}(h)
   \;=\;
   S_s\,\Delta h\,H,

with a drawdown gate that enforces non-negative drawdown:

.. math::

   \Delta h = [h_{ref} - h]_{+}.

The residual is defined as:

.. math::

   R_{cons} \;=\; \frac{\partial s}{\partial t} - \frac{s_{eq}(h) - s}{\tau}.

In practice, GeoPrior evaluates this in a **step-consistent** manner using an
exact-step integrator (next section) rather than a noisy finite difference of
:math:`\partial_t s`.

-------------------------------------------------------------------------------
B) Exact-step consolidation integrator and residual definition
-------------------------------------------------------------------------------

GeoPrior’s consolidation term is designed to be stable at annual time steps.
Rather than discretizing :math:`\partial_t s` directly, GeoPrior compares the
predicted *increment* to the physics-implied increment under the relaxation law.

Let :math:`s_k` be the settlement at time step :math:`k` and
:math:`\Delta t` the (SI) time step in seconds.

The relaxation ODE:

.. math::

   \frac{ds}{dt} = \frac{s_{eq} - s}{\tau}

has the exact solution over one step (with :math:`s_{eq},\tau` treated as
constant within the step):

.. math::

   s_{k+1}
   =
   s_k
   + (s_{eq,k} - s_k)\,\Bigl(1 - e^{-\Delta t/\tau_k}\Bigr).

Therefore, the physics-implied increment is:

.. math::

   \Delta s_{k}^{phys}
   =
   (s_{eq,k} - s_k)\,\Bigl(1 - e^{-\Delta t/\tau_k}\Bigr).

Let the model produce a predicted increment:

.. math::

   \Delta s_{k}^{pred} = s_{k+1}^{pred} - s_{k}^{pred}.

GeoPrior defines a step residual:

.. math::

   r_k
   =
   \Delta s_{k}^{pred} - \Delta s_{k}^{phys}.

The consolidation residual map returned by the core is built from
:math:`r_k` (with appropriate broadcasting across batch/horizon).

This exact-step residual has two advantages:

1) it remains stable for large :math:`\Delta t` (e.g., yearly),
2) it matches the intended physics even when :math:`\tau` varies spatially.

If ``residual_method="euler"``, GeoPrior uses the explicit Euler increment:

.. math::

   \Delta s_{k}^{phys,euler} = \frac{\Delta t}{\tau_k}\,(s_{eq,k} - s_k),

and forms the same step residual with this approximation.

-------------------------------------------------------------------------------
C) Nondimensionalization and epsilon diagnostics
-------------------------------------------------------------------------------

Physics residual magnitudes depend strongly on dataset scale, coordinate spans,
and time units. GeoPrior optionally rescales residuals to yield more stable and
transferable physics weights.

C.1 Residual scaling
====================

For any residual map :math:`R` (consolidation or groundwater), define a
characteristic scale :math:`c` and floor :math:`c_{min}>0`. GeoPrior forms:

.. math::

   R^{*} = \frac{R}{\max(c, c_{min})}.

Physics losses are computed from :math:`R^{*}`:

.. math::

   \mathcal{L} = \mathbb{E}\left[(R^{*})^2\right].

This scaling is enabled when ``scale_pde_residuals=True``.

C.2 Epsilon metrics (RMS)
=========================

GeoPrior logs RMS diagnostics (“epsilons”) in two families:

- **raw epsilons** (unit-bearing; diagnostic view):
  .. math::
     \epsilon_{raw} = \sqrt{\mathbb{E}[R^2]}

- **scaled epsilons** (unitless; optimization view):
  .. math::
     \epsilon = \sqrt{\mathbb{E}[(R^{*})^2]}

Typical log keys:

- ``epsilon_cons_raw`` and ``epsilon_cons``
- ``epsilon_gw_raw`` and ``epsilon_gw``
- ``epsilon_prior`` (RMS of log-timescale mismatch residual)

.. note::

   If raw epsilons are huge but scaled epsilons are moderate, scaling is doing
   its job; you should still check coordinate spans and time-unit consistency.

-------------------------------------------------------------------------------
D) Stability guards and NaN-safe math
-------------------------------------------------------------------------------

GeoPrior includes multiple “hardening” policies to prevent numerical failures.

D.1 Floors and safe thickness
=============================

Thickness fields enter divisions (e.g. recharge-rate Q conversion) and the
equilibrium compaction closure. GeoPrior uses a positive floor:

.. math::

   H_{safe} = \max(H, H_{floor})

with ``H_floor`` configurable (in meters).

Similarly, :math:`\tau` is floored:

.. math::

   \tau_{safe} = \max(\tau, \tau_{floor}).

D.2 Guarded exponentials for log-parameters
===========================================

Physical fields are produced by exponentiating logits. GeoPrior avoids:

- overflow in float32 exp,
- propagation of NaNs through clipping,
- brittle gradients from unbounded exp/log.

Conceptually, with a log parameter :math:`\ell`:

1) sanitize:
   .. math::
      \ell \leftarrow \ell_{min}\quad \text{if } \ell \text{ is non-finite}

2) bound (hard mode) or guard (soft mode)

3) exponentiate:
   .. math::
      z = \exp(\ell).

D.3 Finite checks and fallback behavior
=======================================

Before residuals contribute to scalar losses, GeoPrior checks for non-finite
values. If non-finite values are detected in critical tensors (derivatives,
fields, residuals), the core can:

- replace them with safe sentinels for diagnostics,
- apply clipping,
- or isolate the offending term (depending on debug settings).

D.4 Drawdown gates
==================

Drawdown is gated to avoid unphysical negative values:

.. math::

   \Delta h = [h_{ref}-h]_{+}

with a selectable smooth gate (ReLU vs softplus vs smooth-ReLU). Optional
clipping is also supported:

.. math::

   \Delta h \leftarrow \mathrm{clip}(\Delta h, 0, \Delta h_{max}).

This is critical early in training when head predictions may be noisy.

-------------------------------------------------------------------------------
Where to go next
-------------------------------------------------------------------------------

The math in this section connects directly to:

- :doc:`core` (derivative conversion and the physics core pipeline),
- :doc:`losses` (weights, warmup gates, and log names),
- :doc:`diagnostics` (how to plot and interpret residual and epsilon curves).

