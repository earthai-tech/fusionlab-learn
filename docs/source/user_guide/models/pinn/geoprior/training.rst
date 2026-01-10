.. _geoprior_training:

==============================
Training GeoPrior (Stage-2)
==============================

:API Reference:
    - :class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet`
    - :func:`~fusionlab.nn.pinn.geoprior.step_core.physics_core`

This page explains how to train GeoPrior reliably in practice.
It focuses on the knobs you actually tune: batch construction, compile-time
loss weights, physics warmup/ramp, residual scaling, quantile training, and
common stability patterns.

For the end-to-end pipeline context, see :doc:`pipeline`.
For the exact loss definitions and log names, see :doc:`losses`.
For unit/scaling contracts, see :doc:`data_and_units`.

-------------------------------------------------------------------------------
1) Training entry points
-------------------------------------------------------------------------------

There are two supported ways to train GeoPrior:

1. **Stage-2 script** (recommended):
   you run ``stage2.py`` which loads Stage-1 exports, writes
   ``scaling_kwargs.json``, runs handshake audits, trains, and exports results.

2. **Direct API training**:
   you instantiate :class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet`
   and call ``model.fit`` yourself.

Both approaches follow the same core contract: dict inputs, dict targets,
and a validated ``scaling_kwargs`` payload.

-------------------------------------------------------------------------------
2) Datasets: dict inputs + dict targets
-------------------------------------------------------------------------------

2.1 Minimal required keys
-------------------------

A GeoPrior batch is a dictionary. At minimum:

- ``static_features``: ``(B, S)``
- ``dynamic_features``: ``(B, T, D)``
- ``future_features``: ``(B, H, F)``
- ``coords``: ``(B, H, 3)`` ordered as ``(t, x, y)``
- ``H_field``: broadcastable to ``(B, H, 1)`` (thickness proxy)

Targets are also a dictionary:

- ``subs_pred``: ``(B, H, 1)`` (or quantile-aware shape)
- ``gwl_pred``: ``(B, H, 1)`` (or quantile-aware shape)

.. note::

   Even if you do not use future-known covariates, keep ``future_features`` as
   a shape-consistent tensor (possibly width 0) so the pipeline stays uniform.

2.2 tf.data recommendation
--------------------------

Always use ``tf.data`` for stable and performant training:

.. code-block:: python
   :linenos:

   import tensorflow as tf

   ds_train = (
       tf.data.Dataset.from_tensor_slices((x_train, y_train))
       .shuffle(50_000)
       .batch(BATCH_SIZE)
       .prefetch(tf.data.AUTOTUNE)
   )

-------------------------------------------------------------------------------
3) Compile: the knobs that matter
-------------------------------------------------------------------------------

GeoPrior uses standard Keras compilation for supervised heads plus additional
physics weights. A practical compile template:

.. code-block:: python
   :linenos:

   model.compile(
       optimizer="adam",
       # Supervised losses
       loss={"subs_pred": "mse", "gwl_pred": "mse"},
       # Physics weights (start simple)
       lambda_cons=1.0,
       lambda_gw=1.0,
       lambda_prior=0.1,
       lambda_smooth=0.01,
       lambda_bounds=0.0,
       lambda_mv=0.0,
       lambda_q=0.0,
       # Global physics multiplier
       lambda_offset=0.1,
       offset_mode="mul",
   )

Interpretation (quick)
----------------------

- ``lambda_cons`` / ``lambda_gw``:
  how strongly the model is penalized for violating consolidation / flow.
- ``lambda_prior``:
  how strongly timescale closure is enforced.
- ``lambda_smooth``:
  smoothness regularization on learned fields (prevents noisy K/Ss maps).
- ``lambda_bounds``:
  penalizes out-of-range parameters (useful when ``bounds_mode="soft"``).
- ``lambda_offset`` + ``offset_mode``:
  overall physics multiplier (see :doc:`losses`).

-------------------------------------------------------------------------------
4) Physics warmup and ramp (stability-critical)
-------------------------------------------------------------------------------

Physics terms contain second derivatives and can dominate early training.
GeoPrior supports a warmup+ramp schedule controlled by scaling config.

Conceptually:

- warmup: physics contribution ~0
- ramp: linear increase to full strength
- steady: full physics contribution

Recommended defaults (bring-up)
-------------------------------

- warmup: 500–2000 steps
- ramp: 2000–10000 steps

Use longer schedules when:
- coords are normalized,
- horizon is long,
- subsidence is cumulative with large magnitudes,
- you observe early NaNs or exploding epsilons.

.. tip::

   If you see ``physics_loss_scaled`` near zero for many steps, confirm
   warmup/ramp settings are not too large for your training length.

-------------------------------------------------------------------------------
5) Residual scaling (nondimensionalization)
-------------------------------------------------------------------------------

When enabled, residual scaling makes physics weights much less brittle.

Enable with:

- ``scale_pde_residuals=True`` (model option or scaling config)
- optional floors (``*_scale_floor``) to avoid divide-by-zero.

How to interpret epsilons
-------------------------

- ``epsilon_gw_raw`` / ``epsilon_cons_raw``:
  unit-bearing RMS diagnostics (can be huge if units are large).
- ``epsilon_gw`` / ``epsilon_cons``:
  scaled RMS used by optimization (unitless-ish and comparable across runs).

If both raw and scaled epsilons explode:
- check coord ranges, time units, and head semantics first.

-------------------------------------------------------------------------------
6) Quantile training (uncertainty)
-------------------------------------------------------------------------------

6.1 Enabling quantiles
----------------------

Pass quantiles at construction time, e.g.:

.. code-block:: python

   model = GeoPriorSubsNet(..., quantiles=[0.1, 0.5, 0.9])

The supervised outputs will include a quantile axis (see :doc:`model`).

6.2 Choosing the supervised loss
--------------------------------

For probabilistic forecasts, use a quantile/pinball loss (recommended),
or a composite loss where q50 uses MAE/MSE and tails use pinball.

Example (conceptual):

.. code-block:: python
   :linenos:

   model.compile(
       optimizer="adam",
       loss={
           "subs_pred": "pinball",   # replace with your project loss function
           "gwl_pred": "pinball",
       },
       lambda_cons=1.0,
       lambda_gw=1.0,
       lambda_offset=0.1,
   )

.. note::

   The physics derivatives are computed on the mean/q50 path, not on the full
   quantile tensor. This avoids non-smoothness and keeps physics stable.

6.3 Interval calibration
------------------------

After training, Stage-2 can calibrate predictive intervals horizon-wise
(central interval scaling) to hit a desired empirical coverage on validation
data. See :doc:`inference_and_export`.

-------------------------------------------------------------------------------
7) Practical schedules and recipes
-------------------------------------------------------------------------------

Recipe A: bring-up (debugging shapes/units)
-------------------------------------------

Use this when you are not sure about the data contract yet.

- Set ``pde_mode="none"`` or set ``lambda_offset=0``.
- Train 1–3 epochs.
- Confirm losses decrease, outputs are finite, and forecasting export works.
- Then enable physics.

Recipe B: stable physics start (recommended)
--------------------------------------------

- Enable physics: ``pde_mode="both"``
- Enable warmup+ramp
- Start with moderate weights:

  - ``lambda_cons=1``, ``lambda_gw=1``
  - ``lambda_prior=0.1``, ``lambda_smooth=0.01``
  - ``lambda_offset=0.05`` to ``0.2``

- If physics is too weak (epsilons remain large and unchanged), increase
  ``lambda_offset``.
- If physics dominates (data loss stops improving), decrease
  ``lambda_offset`` or extend warmup/ramp.

Recipe C: inverse field discovery
---------------------------------

Use this when you expect :math:`K` and :math:`S_s` to be learned from data.

- Use ``bounds_mode="soft"`` + ``lambda_bounds>0``.
- Keep ``lambda_smooth`` non-zero.
- Increase ``lambda_prior`` moderately so :math:`\tau` stays realistic.
- Consider freezing the backbone for a short warmup, then unfreezing.

-------------------------------------------------------------------------------
8) Reading training logs
-------------------------------------------------------------------------------

Here is how to read the most important fields:

- ``data_loss``: supervised objective (sum of per-output losses)
- ``physics_loss``: raw physics objective (ungated)
- ``physics_mult``: global multiplier derived from ``lambda_offset``
- ``physics_loss_scaled``: effective physics contribution added to total loss
- ``consolidation_loss`` / ``gw_flow_loss``: unweighted residual MSEs
- ``epsilon_*``: RMS residual diagnostics (raw vs scaled)

If a run goes unstable:
- inspect ``epsilon_gw_raw`` and ``epsilon_cons_raw``,
- inspect whether Q is active (q_gate, q_rms),
- verify coordinate spans and time units in ``scaling_kwargs.json``.

-------------------------------------------------------------------------------
9) Common failure modes and fixes
-------------------------------------------------------------------------------

**Exploding ``epsilon_gw_raw``**
  - x/y spans wrong (coord_ranges mismatch)
  - normalized coords without correct chain-rule scaling
  - degrees treated as meters

**Exploding consolidation residual**
  - time units mismatch (year vs second)
  - thickness too small or not in meters
  - drawdown rule/sign mismatch

**Physics dominates (data loss stuck)**
  - reduce ``lambda_offset`` or extend warmup/ramp
  - reduce ``lambda_cons``/``lambda_gw`` temporarily

**Physics does nothing**
  - check pde_mode and lambdas are non-zero
  - check warmup/ramp gate isn't stuck at 0
  - ensure derivatives are computed on the correct coords tensor

-------------------------------------------------------------------------------
10) Next steps
-------------------------------------------------------------------------------

- :doc:`diagnostics` for plotting epsilons/residuals and debugging
- :doc:`inference_and_export` for saving models and forecast CSVs
- :doc:`faq` for quick answers to common “why is this exploding?” questions
