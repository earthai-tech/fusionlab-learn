.. _geoprior_faq:

==========================
GeoPrior FAQ (practical)
==========================

This FAQ focuses on the issues that actually block users during Stage-2:
unit mismatches, exploding epsilons, quantile layout confusion, Keras loading
differences, and “why is physics doing nothing?”.

If you do not find your issue here, check:

- :doc:`diagnostics` (plots + payloads)
- :doc:`losses` (log names and weights)
- :doc:`data_and_units` (unit contract)
- :doc:`pipeline` (Stage-1/Stage-2 responsibilities)

-------------------------------------------------------------------------------
Q1) My training explodes: ``epsilon_gw_raw`` is huge (or becomes NaN). Why?
-------------------------------------------------------------------------------

**Most common cause:** coordinate scaling mismatch.

GeoPrior computes PDE derivatives w.r.t. the coords used in the forward pass
and then converts them to SI using chain rule. If your coordinate spans
(``coord_ranges``) do not match the actual coordinate values you feed, second
derivatives explode.

**Checklist:**

1) Confirm your x/y are **meters** (UTM-like) not degrees.
2) If coords are normalized, confirm:

   - ``coords_normalized=True``
   - ``coord_ranges["x"]`` and ``coord_ranges["y"]`` are correct (meters)
   - time span also matches your dataset horizon

3) If coords are in degrees, confirm degree→meter conversion is enabled and
   consistent.

**What to do:**

- Inspect ``scaling_kwargs.json`` and Stage-1 audit files.
- Run the physics harness: :doc:`identifiability` (Harness section).
- Plot epsilons: :doc:`diagnostics`.

-------------------------------------------------------------------------------
Q2) ``epsilon_cons_raw`` explodes but GW residual seems fine. Why?
-------------------------------------------------------------------------------

This is usually a **time unit mismatch** or **thickness underflow**.

**Time units:**
If your data uses “year” steps but physics treats :math:`\Delta t` as seconds,
then :math:`\partial s/\partial t` is wrong by ~:math:`3.15\times 10^7`.

**Thickness:**
Equilibrium compaction uses :math:`s_{eq}=S_s\Delta h H`. If ``H_field`` is
near zero (or in wrong units), you’ll see instability.

**Fix:**

- Verify ``time_units`` in scaling config (year/day/second).
- Verify ``H_field`` is in meters and has a positive floor.
- Confirm drawdown construction (sign conventions).

-------------------------------------------------------------------------------
Q3) Physics seems to do nothing: ``physics_loss_scaled`` stays ~0.
-------------------------------------------------------------------------------

Most likely:

- physics is disabled (``pde_mode="none"`` or weights are zero), or
- warmup/ramp gating is keeping physics near 0 for too long.

**Checklist:**

- ``pde_mode`` is ``"both"`` or the desired physics mode.
- ``lambda_offset`` > 0 and at least one of ``lambda_cons`` or ``lambda_gw`` > 0.
- warmup/ramp schedule is not longer than the actual training run.

**Debug:**

- In logs, check ``physics_mult`` and (if logged) gate values.
- Plot physics terms using :doc:`diagnostics`.

-------------------------------------------------------------------------------
Q4) Physics dominates: data loss stops improving.
-------------------------------------------------------------------------------

This means the effective physics contribution is too large:

- ``lambda_offset`` too high, or
- ``lambda_cons`` / ``lambda_gw`` too high, or
- warmup/ramp is too short (physics hits full strength too early).

**Fix options:**

1) Reduce ``lambda_offset``.
2) Extend warmup/ramp.
3) Temporarily reduce PDE weights (``lambda_cons``, ``lambda_gw``) and
   increase later.

-------------------------------------------------------------------------------
Q5) My quantiles look wrong (crossing, huge spread, negative intervals).
-------------------------------------------------------------------------------

Two separate issues are common:

**(A) Layout confusion**
GeoPrior uses q-axis = 2 by default, so:

- ``(B, H, Q, 1)`` and q index order matches ``quantiles=[q10,q50,q90]``

Extract:

.. code-block:: python

   s = y_pred["subs_pred"]   # (B,H,Q,1)
   q10, q50, q90 = s[:, :, 0, 0], s[:, :, 1, 0], s[:, :, 2, 0]

If your outputs are shaped differently (e.g. ``(B,Q,H,1)``), your extraction is
wrong.

**(B) Quantile crossing**
Crossing can happen with unconstrained quantile heads.

**Fix options:**

- Apply post-hoc interval calibration (Stage-2 calibrator).
- Add a quantile crossing penalty (if enabled in your project).
- Consider a monotone reparameterization for quantiles (advanced).

Use the quantile debug utility (if available in your workflow) and see
:doc:`diagnostics`.

-------------------------------------------------------------------------------
Q6) I can’t load a model in Keras 3 from a TensorFlow SavedModel folder.
-------------------------------------------------------------------------------

This is expected: Keras 3 does not load TF SavedModel via ``load_model`` the
same way as tf.keras historically did. The recommended approach is:

- load a ``.keras`` artifact if possible, or
- rebuild from manifest + load weights, or
- wrap a TF SavedModel export using ``TFSMLayer`` for inference-only use.

See :doc:`inference_and_export` for the recommended portable bundle workflow.

-------------------------------------------------------------------------------
Q7) My physics fields (K/Ss/tau) are NaN or Inf.
-------------------------------------------------------------------------------

This almost always comes from log-parameter mapping:

- logits become non-finite,
- then exponentiation or later math propagates NaNs.

GeoPrior includes NaN-safe clipping constraints for log-params (and floors),
but you should still confirm:

- bounds are reasonable (log ranges not too wide),
- learning rate is not too high,
- warmup/ramp is enabled (avoid early second-derivative shocks).

**Debug:**

- run ``forward_with_aux`` to inspect logits (see :doc:`inference_and_export`).
- export a physics payload (see :doc:`diagnostics`).

----------------------------------------------------------------------------------
Q8) The closure prior says tau should be X, but the learned tau is very different.
----------------------------------------------------------------------------------

This can be correct or a sign of mismatch.

Common causes:

- closure settings mismatch (kappa mode, Hd_factor, thickness proxy),
- weak prior weight (``lambda_prior`` too small),
- the closure is not appropriate for the data regime (effective medium differs).

**What to do:**

- Plot ``log10_tau`` vs ``log10_tau_prior`` using the payload tools.
- Check ``epsilon_prior`` and closure residual statistics.
- If you trust the closure, increase ``lambda_prior`` moderately.
- If you do not, treat tau as a learned effective timescale and keep
  the prior weak.

See :doc:`maths` and :doc:`identifiability`.

-------------------------------------------------------------------------------
Q9) Do I need perfect head in meters? My dataset has depth-to-water.
-------------------------------------------------------------------------------

GeoPrior can work with a *head proxy*, but you must be consistent:

- If you treat GWL as depth (positive downward), then head should be derived
  using surface elevation:

  .. math::

     h = z_{surf} - z_{GWL}

This is the recommended physical convention.

If you cannot obtain ``z_surf``, you can use a proxy, but interpret recovered
fields (K/Ss/tau) as *effective parameters relative to the proxy*, not
absolute physical parameters.

See :doc:`data_and_units`.

-------------------------------------------------------------------------------
Q10) Which “minimum config” should I start with for a new city/dataset?
-------------------------------------------------------------------------------

Start with the most stable setup:

- ``pde_mode="both"``
- warmup+ramp enabled
- residual scaling enabled
- moderate weights:

  - ``lambda_cons=1``, ``lambda_gw=1``
  - ``lambda_prior=0.1``, ``lambda_smooth=0.01``
  - ``lambda_offset=0.05`` to ``0.2``

Then:

1) run 1–3 epochs,
2) plot epsilons and physics terms,
3) adjust ``lambda_offset`` first.

See :doc:`training` for recipes.

-------------------------------------------------------------------------------
Q11) How do I quickly confirm my physics pipeline works before full training?
-------------------------------------------------------------------------------

Use the debug harness and the SM3 synthetic identifiability script:

- :doc:`identifiability`

They isolate unit/derivative/closure issues from the full dataset complexity.

-------------------------------------------------------------------------------
Q12) Where are the exact names of the terms I see in training logs?
-------------------------------------------------------------------------------

See :doc:`losses`, which maps each log key to a mathematical definition and
explains how lambdas and the global multiplier combine.
