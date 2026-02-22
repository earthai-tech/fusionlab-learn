.. _geoprior_losses_guide:

=====================================================
GeoPrior Losses: Data + Physics (Term-by-Term)
=====================================================

:API Reference:
    - :class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet`
    - :func:`~fusionlab.nn.pinn.geoprior.step_core.physics_core`
    - :func:`~fusionlab.nn.pinn.geoprior.losses.assemble_physics_loss`
    - :func:`~fusionlab.nn.pinn.geoprior.losses.pack_step_results`

GeoPrior is optimized with a **hybrid objective**:

1) a **supervised data loss** defined via ``model.compile(loss=...)`` (standard
   Keras multi-output loss), and

2) a **physics loss** assembled from PDE/ODE residuals, priors, and bounds
   penalties computed by :func:`~fusionlab.nn.pinn.geoprior.step_core.physics_core`.

This page documents each loss component mathematically and maps every term
to its **training log name**.

-------------------------------------------------------------------------------
1. Total objective optimized by ``train_step``
-------------------------------------------------------------------------------

Let :math:`\mathcal{L}_{data}` be the compiled supervised objective (including
any Keras regularization losses), and let
:math:`\mathcal{L}_{phys,scaled}` be the **scaled** physics objective produced
by the physics core.

The objective optimized at each step is:

.. math::

   \mathcal{L}_{total}
   \;=\;
   \mathcal{L}_{data}
   \;+\;
   \mathcal{L}_{phys,scaled}.

**Warmup/ramp gating (training only).**
During training, GeoPrior may apply a warmup+ramp gate :math:`g(step)\in[0,1]`
to introduce physics gradually:

.. math::

   \mathcal{L}_{phys,scaled}^{train}
   \;=\;
   g(step)\,\mathcal{L}_{phys,scaled}^{pre}.

In evaluation/inference, the gate is not applied.

.. note::

   In the logs, ``physics_loss_scaled`` is the *training-effective* physics
   objective (after warmup/ramp if enabled). The value ``physics_loss`` remains
   the ungated “raw physics” diagnostic.

-------------------------------------------------------------------------------
2. Supervised data losses (compiled Keras loss)
-------------------------------------------------------------------------------

GeoPrior is typically trained with two supervised heads:

- subsidence output: ``"subs_pred"``
- groundwater/head output: ``"gwl_pred"``

Let these targets be :math:`s` and :math:`h` and their predictions be
:math:`\hat{s}` and :math:`\hat{h}` (mean path or quantile tensor depending
on configuration).

Keras defines (user-configurable) per-output losses:

.. math::

   \mathcal{L}_{subs} = \ell_{subs}(s, \hat{s}),\qquad
   \mathcal{L}_{gwl}  = \ell_{gwl}(h, \hat{h})

and the compiled data loss is:

.. math::

   \mathcal{L}_{data}
   \;=\;
   \mathcal{L}_{subs}
   \;+\;
   \mathcal{L}_{gwl}
   \;+\;
   \mathcal{L}_{reg}

where :math:`\mathcal{L}_{reg}` is the sum of regularization losses from
layers (passed as ``regularization_losses=self.losses``).

**Quantile outputs.**
If quantiles are enabled, the prediction tensor often has a quantile axis
(e.g. :math:`\hat{s}\in\mathbb{R}^{B\times H\times Q\times 1}`).
Targets are *not tiled across quantiles*; broadcast is used instead.

Logged names (data)
-------------------

Keras will log per-output compiled losses when available, typically:

- ``subs_pred_loss``  : :math:`\mathcal{L}_{subs}`
- ``gwl_pred_loss``   : :math:`\mathcal{L}_{gwl}`
- ``data_loss``       : :math:`\mathcal{L}_{data}` (explicit in GeoPrior)
- ``loss`` / ``total_loss`` : :math:`\mathcal{L}_{total}`

-------------------------------------------------------------------------------
3. Physics residual losses (base definitions)
-------------------------------------------------------------------------------

GeoPrior builds residual maps (per batch, per horizon, optionally per grid)
and converts them into scalar penalties via mean-square.

The main residual families are:

A) Consolidation / compaction residual
--------------------------------------

GeoPrior enforces a relaxation form:

.. math::

   \frac{\partial s}{\partial t}
   \;=\;
   \frac{s_{eq}(h) - s}{\tau}

with an equilibrium proxy often proportional to drawdown:

.. math::

   s_{eq}(h)\;\approx\; S_s\,\Delta h\,H.

The physics core computes a **step-consistency** residual map
:math:`R_{cons}` (via an exact-step integrator) and forms:

.. math::

   \mathcal{L}_{cons}
   \;=\;
   \mathbb{E}\left[\bigl(R_{cons}^{*}\bigr)^2\right]

where :math:`R_{cons}^{*}` is the (optionally) nondimensionalized version
of the residual (see Section 5).

B) Groundwater flow residual
----------------------------

GeoPrior enforces the transient 2D groundwater equation:

.. math::

   R_{gw}
   \;=\;
   S_s\,\frac{\partial h}{\partial t}
   \;-\;
   \nabla\cdot(K\nabla h)
   \;-\;
   Q
   \;=\; 0.

The scalar penalty is:

.. math::

   \mathcal{L}_{gw}
   \;=\;
   \mathbb{E}\left[\bigl(R_{gw}^{*}\bigr)^2\right]

where :math:`R_{gw}^{*}` is optionally nondimensionalized.

C) Timescale prior loss (log-space)
-----------------------------------

GeoPrior maintains an explicit log-timescale mismatch residual, e.g.:

.. math::

   R_{prior} \;=\; \log\tau \;-\; \log\tau_{prior}

and penalizes:

.. math::

   \mathcal{L}_{prior}
   \;=\;
   \mathbb{E}\left[R_{prior}^2\right].

D) Smoothness prior loss (spatial regularity)
---------------------------------------------

To encourage physically plausible spatial fields, GeoPrior penalizes
spatial variation in learned parameters (e.g. :math:`K` and :math:`S_s`).
Let :math:`R_{smooth}` be the smoothness residual map (constructed from
spatial gradients of :math:`K` and :math:`S_s`); then:

.. math::

   \mathcal{L}_{smooth}
   \;=\;
   \mathbb{E}\left[R_{smooth}^2\right].

E) Bounds loss (soft/hard physical ranges)
------------------------------------------

Bounds are enforced both by safe parameter mapping and by an explicit penalty
on excursions outside configured ranges. If :math:`R_{bounds}` is the stacked
bounds residual:

.. math::

   \mathcal{L}_{bounds}
   \;=\;
   \mathbb{E}\left[R_{bounds}^2\right].

F) MV prior loss (optional)
---------------------------

GeoPrior can include an optional calibration-style penalty on the
consolidation coefficient :math:`m_v` (or an MV proxy derived from :math:`S_s`).
Let :math:`\mathcal{L}_{mv}` denote the MV penalty (often Huber-like).

G) Q regularization (optional)
------------------------------

If a learnable forcing term :math:`Q` is enabled, GeoPrior adds an L2 penalty:

.. math::

   \mathcal{L}_{q}
   \;=\;
   \mathbb{E}\left[Q_{si}^2\right].

This discourages the forcing from becoming an unconstrained “shortcut”.

-------------------------------------------------------------------------------
4. Physics term weights and how lambdas combine
-------------------------------------------------------------------------------

GeoPrior exposes scalar weights (configured in ``GeoPriorSubsNet.compile``):

- ``lambda_cons``   : weight for :math:`\mathcal{L}_{cons}`
- ``lambda_gw``     : weight for :math:`\mathcal{L}_{gw}`
- ``lambda_prior``  : weight for :math:`\mathcal{L}_{prior}`
- ``lambda_smooth`` : weight for :math:`\mathcal{L}_{smooth}`
- ``lambda_bounds`` : weight for :math:`\mathcal{L}_{bounds}`
- ``lambda_mv``     : weight for :math:`\mathcal{L}_{mv}`
- ``lambda_q``      : weight for :math:`\mathcal{L}_{q}`
- ``lambda_offset`` : global physics multiplier (see below)

Define weighted terms:

.. math::

   T_{cons}   = \lambda_{cons}\,\mathcal{L}_{cons},\quad
   T_{gw}     = \lambda_{gw}\,\mathcal{L}_{gw},\quad
   T_{prior}  = \lambda_{prior}\,\mathcal{L}_{prior}

.. math::

   T_{smooth} = \lambda_{smooth}\,\mathcal{L}_{smooth},\quad
   T_{bounds} = \lambda_{bounds}\,\mathcal{L}_{bounds}

.. math::

   T_{mv}     = \lambda_{mv}\,\mathcal{L}_{mv},\quad
   T_{q}      = \lambda_{q}\,\mathcal{L}_{q}.

Core vs. auxiliary physics (raw)
--------------------------------

GeoPrior separates PDE-style terms from calibration-style terms:

.. math::

   \mathcal{L}_{core} \;=\; T_{cons}+T_{gw}+T_{prior}+T_{smooth}+T_{bounds}

.. math::

   \mathcal{L}_{phys,raw} \;=\; \mathcal{L}_{core} + T_{mv} + T_{q}.

This is logged as:

- ``physics_loss`` : :math:`\mathcal{L}_{phys,raw}` (ungated diagnostic)

Global physics multiplier: ``lambda_offset`` and ``offset_mode``
---------------------------------------------------------------

GeoPrior defines a global multiplier:

.. math::

   m \;=\; \texttt{physics\_mult}
   \;=\;
   \begin{cases}
     \lambda_{offset}, & \texttt{offset\_mode}=\text{"mul"}\\
     10^{\lambda_{offset}}, & \texttt{offset\_mode}=\text{"log10"}
   \end{cases}

This multiplier is logged as:

- ``lambda_offset`` : the configured offset parameter
- ``physics_mult``  : :math:`m`

Scaled physics objective (pre-gate)
-----------------------------------

The scaled physics loss is assembled as:

.. math::

   \mathcal{L}_{phys,scaled}^{pre}
   \;=\;
   m\,\mathcal{L}_{core}
   \;+\;
   s_{mv}\,T_{mv}
   \;+\;
   s_{q}\,T_{q}

with scaling switches:

- :math:`s_{mv}=m` if ``scale_mv_with_offset=True``, else :math:`s_{mv}=1`
- :math:`s_{q}=m`  if ``scale_q_with_offset=True``,  else :math:`s_{q}=1`

Default behavior is typically:

- MV is **not** scaled by the offset (kept as calibration signal)
- Q regularization **is** scaled with the offset (kept consistent with physics)

Warmup/ramp gate (training)
---------------------------

During training only, GeoPrior applies:

.. math::

   \mathcal{L}_{phys,scaled}
   \;=\;
   g(step)\,\mathcal{L}_{phys,scaled}^{pre}

and uses this in the total objective:

.. math::

   \mathcal{L}_{total}=\mathcal{L}_{data}+\mathcal{L}_{phys,scaled}.

This is logged as:

- ``physics_loss_scaled`` : :math:`\mathcal{L}_{phys,scaled}` (effective term)

Per-term scaled contributions (for interpretability)
----------------------------------------------------

GeoPrior also tracks per-term contributions consistent with
:math:`\mathcal{L}_{phys,scaled}`:

- ``terms_scaled["cons"]``   : :math:`g(step)\,m\,T_{cons}`
- ``terms_scaled["gw"]``     : :math:`g(step)\,m\,T_{gw}`
- ``terms_scaled["prior"]``  : :math:`g(step)\,m\,T_{prior}`
- ``terms_scaled["smooth"]`` : :math:`g(step)\,m\,T_{smooth}`
- ``terms_scaled["bounds"]`` : :math:`g(step)\,m\,T_{bounds}`
- ``terms_scaled["mv"]``     : :math:`g(step)\,s_{mv}\,T_{mv}`
- ``terms_scaled["q_reg"]``  : :math:`g(step)\,s_{q}\,T_{q}`

-------------------------------------------------------------------------------
5. Residual nondimensionalization (optional)
-------------------------------------------------------------------------------

To stabilize training, GeoPrior can scale the primary PDE residuals.

Let :math:`R` be a raw residual map (consolidation or groundwater).
GeoPrior computes a characteristic scale :math:`s` and applies:

.. math::

   R^{*} \;=\; \frac{R}{\max(s, s_{floor})}.

Then the loss uses :math:`R^{*}`:

.. math::

   \mathcal{L}=\mathbb{E}\left[(R^{*})^2\right].

This produces two useful diagnostics:

- “raw” RMS: large numbers reflect SI magnitude and units
- “scaled” RMS: unitless/normalized magnitude used in optimization

-------------------------------------------------------------------------------
6. Logging names (what you see during training)
-------------------------------------------------------------------------------

The following names appear in ``model.fit`` logs.

Total / data
------------

- ``loss``:
  The optimized objective :math:`\mathcal{L}_{total}`.

- ``total_loss``:
  Alias of ``loss`` (same scalar).

- ``data_loss``:
  The compiled supervised objective :math:`\mathcal{L}_{data}`.

Per-output compiled losses (typical)
------------------------------------

- ``subs_pred_loss``: :math:`\mathcal{L}_{subs}`
- ``gwl_pred_loss``: :math:`\mathcal{L}_{gwl}`

Physics assembly (top-level)
----------------------------

- ``physics_loss``:
  Raw physics objective :math:`\mathcal{L}_{phys,raw}` (ungated diagnostic).

- ``physics_loss_scaled``:
  Effective physics contribution :math:`\mathcal{L}_{phys,scaled}`
  (includes warmup/ramp gate if enabled).

- ``physics_mult``:
  :math:`m` computed from ``lambda_offset`` and ``offset_mode``

- ``lambda_offset``:
  The configured offset parameter (used to build :math:`m`)

Physics components (scalar losses)
----------------------------------

These are the **unweighted** base components (i.e., they correspond to
:math:`\mathcal{L}_{*}` above, before multiplying by :math:`\lambda_*` and
before the global multiplier):

- ``consolidation_loss``: :math:`\mathcal{L}_{cons}`
- ``gw_flow_loss``: :math:`\mathcal{L}_{gw}`
- ``prior_loss``: :math:`\mathcal{L}_{prior}`
- ``smooth_loss``: :math:`\mathcal{L}_{smooth}`
- ``bounds_loss``: :math:`\mathcal{L}_{bounds}`
- ``mv_prior_loss``: :math:`\mathcal{L}_{mv}`

Optional Q diagnostics (only if enabled in scaling config)
---------------------------------------------------------

When ``log_q_diagnostics=True`` in scaling config:

- ``q_reg_loss``: :math:`\mathcal{L}_{q}`
- ``q_rms``: RMS magnitude of :math:`Q_{si}`
- ``q_gate``: gating value in :math:`[0,1]` used for Q injection
- ``subs_resid_gate``: gating value in :math:`[0,1]` used for subs residual head

Epsilon metrics (RMS residual diagnostics)
------------------------------------------

- ``epsilon_prior``:
  RMS of prior residual map (:math:`R_{prior}` or its configured diagnostic view)

- ``epsilon_cons``:
  RMS of scaled consolidation residual :math:`R_{cons}^{*}`

- ``epsilon_gw``:
  RMS of scaled groundwater residual :math:`R_{gw}^{*}`

- ``epsilon_cons_raw``:
  RMS of raw consolidation residual (pre scaling)

- ``epsilon_gw_raw``:
  RMS of raw groundwater residual (pre scaling / diagnostic units)

-------------------------------------------------------------------------------
7. Practical interpretation of log lines
-------------------------------------------------------------------------------

A typical training line:

.. code-block:: text

   ... - subs_pred_loss: 0.04 - gwl_pred_loss: 1.70 - loss: 0.71
       - data_loss: 0.90 - physics_loss: 4.7e-04 - physics_mult: 2.0
       - physics_loss_scaled: 9.3e-04 - consolidation_loss: 4.7e-04
       - gw_flow_loss: 2.4e-09 - prior_loss: 5.6e-07 ...

can be read as:

- the model is mostly driven by the supervised objective (``data_loss``),
- physics is active but small in magnitude (``physics_loss_scaled``),
- groundwater PDE is currently negligible (``gw_flow_loss`` tiny),
- consolidation residual dominates the physics bundle (``consolidation_loss``),
- ``physics_mult`` tells you how strongly physics is globally scaled.

.. tip::

   If you enable warmup/ramp, expect ``physics_loss_scaled`` to start near zero
   and grow over steps, while ``physics_loss`` (raw) may be large early on.

-------------------------------------------------------------------------------
8. Configuring loss weights (what to tune)
-------------------------------------------------------------------------------

Tuning guidelines (very practical)
----------------------------------

- Start with moderate PDE weights (e.g. ``lambda_cons=lambda_gw=1``),
  and adjust ``lambda_offset`` to control overall physics strength.

- Use warmup/ramp (scaling config) to prevent early training instability.

- If MV prior is meant as a calibration signal, keep
  ``scale_mv_with_offset=False`` (default).

- If Q is enabled, keep its regularization aligned with physics strength
  (default ``scale_q_with_offset=True``), and enable Q warmup/ramp if needed.

Example ``compile`` snippet
---------------------------

.. code-block:: python
   :linenos:

   model.compile(
       optimizer="adam",
       loss={"subs_pred": "mse", "gwl_pred": "mse"},
       lambda_cons=1.0,
       lambda_gw=1.0,
       lambda_prior=0.1,
       lambda_smooth=0.01,
       lambda_bounds=0.0,
       lambda_mv=0.0,
       lambda_q=0.0,
       lambda_offset=0.1,
       offset_mode="mul",
   )
