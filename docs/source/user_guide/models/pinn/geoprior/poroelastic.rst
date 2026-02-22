.. _geoprior_poroelastic:

===========================================================
PoroElasticSubsNet: consolidation-first GeoPrior preset
===========================================================

:API Reference: :class:`~fusionlab.nn.pinn.geoprior.models.PoroElasticSubsNet`

``PoroElasticSubsNet`` is a **specialized preset** built on top of
:class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet`.
It targets workflows where the **poroelastic / consolidation dynamics**
are the dominant and best-constrained physics, while the full
groundwater-flow PDE is either:

- not reliably constrained (missing pumping/recharge proxies),
- too sensitive to derivative noise early in training, or
- intentionally disabled for ablation and regime studies.

In other words:

**GeoPriorSubsNet** = coupled flow + consolidation (general-purpose)

**PoroElasticSubsNet** = consolidation-first (poroelastic regime) with
**stronger physical regularization by default**.

When should you use it?
-----------------------

Use ``PoroElasticSubsNet`` when one or more of the following is true:

- You want **physically plausible subsidence trajectories** even when
  head data are noisy or sparse.
- You trust the consolidation closure more than the groundwater PDE on
  your dataset (e.g., uncertain boundaries, unknown pumping, unknown
  recharge distribution).
- You are doing **inverse learning / parameter discovery** (e.g.
  :math:`\tau`, MV-related parameters) and want a more stable prior-driven
  setup.
- You want a strong baseline for “subsidence from drawdown” behavior
  before turning on the full coupled PDE.

If you have reliable forcing proxies (pumping/recharge), good spatial
coverage, and stable coordinate scaling, then
:class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet` with
``pde_mode="both"`` is usually the better default.

What is different from GeoPriorSubsNet?
---------------------------------------

``PoroElasticSubsNet`` inherits the entire architecture and I/O contract
of :class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet`:

- same dict inputs (``static_features``, ``dynamic_features``,
  ``future_features``, ``coords`` + optional physics keys),
- same supervised outputs (``subs_pred``, ``gwl_pred``),
- same quantile layout (when enabled, quantile axis is ``q_axis=2``),
- same ``forward_with_aux`` diagnostics interface.

The difference is **configuration defaults** that bias the model toward
a poroelastic / consolidation-first regime:

- **PDE mode defaults to consolidation** (groundwater residual is off by
  default).
- **Regularization weights are stronger by default** (timescale prior and
  bounds are emphasized).
- **More conservative learning-rate multipliers** are used for some
  physical parameters so they do not drift aggressively at the start.

These defaults make the model a stable choice for:
- regime testing (consolidation-only vs coupled),
- identifiability experiments,
- and early-stage debugging of units and derivatives.

Default physics behavior (conceptual)
-------------------------------------

The consolidation pathway enforces a relaxation dynamics:

.. math::

   \frac{\partial s}{\partial t}
   \;=\;
   \frac{s_{eq}(h) - s}{\tau}

where an equilibrium settlement proxy is:

.. math::

   s_{eq}(h)\;\approx\; S_s\,\Delta h \, H

This yields a **physics-driven mean subsidence path**, with an optional
additive residual (see :doc:`model` and :doc:`core`).

Because the groundwater PDE term is off by default, the model focuses on:

- stable learning of :math:`\tau`-controlled relaxation,
- stable drawdown-to-compaction mapping,
- and robust bounds/priors for physical fields.

Minimal usage example
---------------------

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.models import PoroElasticSubsNet

   model = PoroElasticSubsNet(
       static_input_dim=3,
       dynamic_input_dim=8,
       future_input_dim=4,
       forecast_horizon=3,
       max_window_size=12,
       # optional:
       quantiles=[0.1, 0.5, 0.9],
       scaling_kwargs="path/to/scaling_kwargs.json",
   )

   model.compile(
       optimizer="adam",
       loss={"subs_pred": "mse", "gwl_pred": "mse"},
       # The preset typically emphasizes priors/bounds, but you can override:
       # lambda_prior=..., lambda_bounds=..., lambda_cons=...
   )

Turn the groundwater PDE back on (optional)
-------------------------------------------

If your dataset supports it (reliable coordinates, stable scaling, and
meaningful forcing configuration), you can enable the coupled physics:

.. code-block:: python
   :linenos:

   model = PoroElasticSubsNet(
       static_input_dim=3,
       dynamic_input_dim=8,
       future_input_dim=4,
       forecast_horizon=3,
       max_window_size=12,
       pde_mode="both",   # or "gw_flow"
       scaling_kwargs="path/to/scaling_kwargs.json",
   )

   model.compile(
       optimizer="adam",
       loss={"subs_pred": "mse", "gwl_pred": "mse"},
       lambda_gw=1.0,      # enable groundwater loss weight
       lambda_cons=1.0,    # consolidation weight
   )

This is a useful workflow pattern:

1) train a stable consolidation-first model,
2) then turn on GW flow residual and fine-tune (lower LR recommended),
3) compare epsilons and coverage/sharpness diagnostics.

Practical notes and gotchas
---------------------------

- **This is not a different I/O model**: it is a preset.
  Treat it as a “recommended configuration” for a specific regime.

- If you enable GW flow residual, coordinate scaling becomes critical.
  Verify ``coords_normalized``, ``coord_ranges``, and ``time_units`` in
  :doc:`data_and_units` and :doc:`pipeline`.

- If your GWL target is depth-like rather than head-like, make sure the
  head proxy configuration is correct (e.g., availability of ``z_surf_m``
  and the sign convention). See :doc:`data_and_units`.

See also
--------

- :doc:`model` for the full GeoPriorSubsNet constructor and output shapes.
- :doc:`core` for the physics-core assembly and residual definitions.
- :doc:`losses` for term-by-term loss composition and lambda wiring.
- :doc:`diagnostics` for epsilon/residual plots and common failure modes.
