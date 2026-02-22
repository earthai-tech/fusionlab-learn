.. _geoprior_scaling:

=============================================================
Scaling configuration (``scaling_kwargs``): units & control
=============================================================

:API Reference:
   - :class:`~fusionlab.nn.pinn.geoprior.scaling.GeoPriorScalingConfig`

GeoPrior’s physics is **only meaningful if units and coordinate spans are
consistent**. All of that information is centralized in a single payload:
``scaling_kwargs``. It is produced during Stage-1 and consumed during Stage-2,
and it governs:

- **Unit conversions** (subsidence/head/thickness to SI),
- **Coordinate normalization and chain-rule** (critical for derivatives),
- **Head proxy semantics** (depth-below-ground → head),
- **Residual display/scaling** (raw vs nondimensionalized),
- **Warmup/ramp gates** (physics and forcing),
- **Bounds and priors** (K, Ss, τ, H),
- **MV prior schedule** and related diagnostics.

If you edit this file manually, you are effectively changing the *physics
problem definition*.

Where ``scaling_kwargs`` comes from
------------------------------------

Stage-1 writes ``scaling_kwargs.json`` (plus audit files) as a compact,
fully-resolved record of:

1) dataset conventions (time units, coordinate mode),
2) feature naming and indices,
3) SI conversion constants,
4) physics-control knobs.

Stage-2 loads it and attaches it to the model so that training, evaluation,
and exported inference reproduce the same physics interpretation.

The recommended practice is:

- Treat ``scaling_kwargs.json`` as a **versioned artifact** of Stage-1,
- Keep it next to Stage-2 outputs (manifest, weights, audits),
- Never “guess” spans/units inside the physics code; always read them from
  this payload.

How GeoPrior loads and freezes scaling
----------------------------------------

GeoPrior wraps the payload using a Keras-serializable container
:class:`~fusionlab.nn.pinn.geoprior.scaling.GeoPriorScalingConfig`.
This matters because Keras model saving/reloading must not silently change
physics behavior.

Construction pattern:

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.scaling import GeoPriorScalingConfig

   cfg = GeoPriorScalingConfig.from_any("path/to/scaling_kwargs.json")
   scaling_kwargs = cfg.resolve()  # canonical + validated dict

The resolution pipeline is:

1) load (dict or file path),
2) canonicalize keys + fill defaults,
3) enforce alias consistency,
4) validate schema and ranges.

This is the same logic used during training, so reloaded models behave
identically. 

Why scaling is *physics-critical*
----------------------------------

GeoPrior computes derivatives with autodiff in **model coordinate space**,
then converts derivatives to SI using a chain rule driven by
``coord_ranges`` / ``coord_ranges_si``.

If coordinates are normalized:

.. math::

   t = t_{min} + \hat{t}\,\Delta t,\quad
   x = x_{min} + \hat{x}\,\Delta x,\quad
   y = y_{min} + \hat{y}\,\Delta y

then:

.. math::

   \frac{\partial h}{\partial t}
   = \frac{1}{\Delta t}\,\frac{\partial h}{\partial \hat{t}},\qquad
   \frac{\partial^2 h}{\partial x^2}
   = \frac{1}{\Delta x^2}\,\frac{\partial^2 h}{\partial \hat{x}^2}

Second derivatives scale with :math:`1/\Delta x^2` and :math:`1/\Delta y^2`,
so **a small mistake in spans can explode PDE residuals by orders of magnitude**.

In your example payload:

- ``coords_normalized = true``
- ``time_units = "year"``
- ``seconds_per_time_unit = 31556952.0``
- ``coord_ranges_si["t"] = 220898664.0`` (7 years in seconds)

and these values directly set the derivative conversion. 


Scaling control table (most important knobs)
--------------------------------------------

This table focuses on the knobs that most strongly change physics meaning,
numerical stability, and logged “epsilon” diagnostics.

.. list-table::
   :header-rows: 1
   :widths: 18 28 16 38

   * - Key
     - Meaning
     - Typical values
     - Why it matters
   * - ``time_units``
     - Dataset time unit used in coords (Stage-1 → Stage-2 contract).
     - ``"year"``, ``"day"``, ``"second"``
     - Controls :math:`\Delta t` conversion and thus :math:`\partial_t h`
       and consolidation steps.
   * - ``seconds_per_time_unit``
     - Seconds per one ``time_units`` (SI conversion constant).
     - float (e.g., ~3.1557e7 for year)
     - Wrong value ⇒ wrong derivative scaling and unstable residual magnitudes.
   * - ``coords_normalized``
     - Whether coords fed to model are normalized (usually to [0,1]).
     - ``true`` / ``false``
     - If ``true``, chain-rule must use correct spans.
   * - ``coord_order``
     - Order of coords axes in ``coords[..., :]``.
     - ``["t","x","y"]``
     - If order mismatches, derivatives target wrong axes.
   * - ``coord_ranges`` / ``coord_ranges_si``
     - Coordinate spans in dataset units / SI units.
     - dict with ``t,x,y``
     - The **most common source** of PDE “epsilon explosions”.
   * - ``coord_mode`` + ``coords_in_degrees``
     - Coordinate origin convention (degrees vs meters).
     - ``"degrees"`` + ``false`` (after projection)
     - If degrees are treated as meters, PDE becomes meaningless.
   * - ``subs_scale_si`` / ``head_scale_si`` / ``H_scale_si``
     - Multipliers to convert model targets/fields to SI.
     - floats
     - Guarantees residuals computed in SI even if dataset uses other units.
   * - ``subsidence_kind``
     - Meaning of the subsidence target series.
     - ``"cumulative"`` / ``"incremental"``
     - Determines how settlement state is built for the integrator/residual.
   * - ``allow_subs_residual``
     - Enables the learnable residual around physics mean subsidence.
     - ``true`` / ``false``
     - When off, subsidence mean is fully physics-driven (strong prior).
   * - ``gwl_kind`` / ``gwl_sign``
     - Semantics of the groundwater variable (e.g. depth below ground).
     - ``"depth_bgs"`` + ``"down_positive"``
     - Prevents sign mistakes when converting to head / drawdown.
   * - ``use_head_proxy`` + ``z_surf_col``
     - Whether to convert depth to head using surface elevation.
     - ``true`` + ``"z_surf_m__si"``
     - Critical for correct head definition if the target is depth-like.
   * - ``cons_drawdown_mode``
     - How drawdown :math:`\Delta h` is constrained/activated.
     - ``"softplus"`` / ``"linear"`` / ``"relu"``
     - Prevents negative/unstable drawdown driving consolidation.
   * - ``cons_drawdown_rule``
     - Definition of drawdown relative to reference head.
     - e.g. ``"ref_minus_mean"``
     - Changes the physics meaning of “compaction driving”.
   * - ``cons_stop_grad_ref``
     - Stop-grad on reference head used for drawdown.
     - ``true`` / ``false``
     - Stabilizes training by preventing reference drift.
   * - ``cons_residual_units`` / ``gw_residual_units``
     - Diagnostic display units for residuals.
     - ``"second"`` (common)
     - Does **not** change SI physics; it changes “raw” diagnostic view.
   * - ``cons_scale_floor`` / ``gw_scale_floor``
     - Floors for nondimensionalization scales.
     - ``"auto"`` or float
     - Prevents divide-by-tiny-scale explosions during residual scaling.
   * - ``physics_warmup_steps`` / ``physics_ramp_steps``
     - Warmup and linear ramp for physics multiplier.
     - ints
     - Keeps early training stable; physics ramps in gradually.
   * - ``Q_kind``
     - Meaning of forcing ``Q`` used in GW residual.
     - ``"per_volume"`` (default)
     - Determines how Q is interpreted and converted to SI.
   * - ``q_policy`` + ``q_warmup_*`` / ``q_ramp_*``
     - Schedules when Q is allowed to influence training.
     - e.g. ``"warmup_off"``
     - Prevents Q from becoming an early “shortcut”.
   * - ``lambda_q``
     - Regularization weight on Q magnitude.
     - small float (e.g. 1e-5)
     - Controls forcing size so Q doesn’t absorb model error.
   * - ``bounds``
     - Physical bounds for H, K, Ss, τ (and their log versions).
     - dict (min/max)
     - Prevents non-physical parameters and stabilizes exponentiation.
   * - ``scaling_error_policy``
     - What to do if scaling validation fails.
     - ``"raise"`` / ``"warn"``
     - “Raise” is safer: it prevents silent physics drift.
   * - ``debug_physics_grads``
     - Enables extra diagnostics for gradients/derivatives.
     - ``false`` / ``true``
     - Useful for isolating derivative explosions.

(Values and keys shown above correspond to an example ``scaling_kwargs.json``.
A file contains additional convenience fields such as feature name lists,
index hints, and derived inverse spans for faster chain-rule conversion.) 


Recommended workflow for editing scaling
-----------------------------------------

1) Prefer regenerating ``scaling_kwargs.json`` via Stage-1 rather than manual edits.
2) If you must edit manually:
   - keep ``time_units`` consistent with your dataset timeline,
   - verify ``coord_ranges_si`` matches the true spatial/time span,
   - re-run Stage-2 handshake/audit and check epsilons early.
3) If epsilons explode at epoch 1:
   - check ``coord_order``,
   - check time conversion (year vs second),
   - check degree vs meter handling,
   - check head proxy semantics.

See also
--------

- :doc:`data_and_units` for dataset-side units and column conventions.
- :doc:`pipeline` for Stage-1/Stage-2 artifacts and audits.
- :doc:`core` for how scaling drives derivatives, residual scaling, and gates.
- :doc:`losses` for how residuals/lambdas combine into the final objective.
