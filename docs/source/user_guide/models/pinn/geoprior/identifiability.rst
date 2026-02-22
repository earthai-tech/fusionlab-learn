.. _geoprior_identifiability:

==========================================
Identifiability and Physics Debug Harness
==========================================

GeoPrior can learn *effective physical fields* (e.g., :math:`K`, :math:`S_s`,
:math:`\tau`, :math:`H_d`) from data while being regularized by PDE/ODE physics.
This page answers two practical questions:

1) **Identifiability**: what can GeoPrior truly infer from observations, and
   which parameters are only recoverable up to a trade-off?

2) **Harnessing**: how to run quick, controlled experiments that isolate
   physics issues (units, derivatives, gating, NaN/Inf) from the full pipeline.

This page is built around two utilities shipped with the project:

- ``sm3_synthetic_identifiability.py``: a controlled 1-pixel synthetic
  identifiability experiment (SM3). :contentReference[oaicite:0]{index=0}
- ``debug_geoprior_physics_harness.py``: a Stage-1-like physics debug harness
  for GeoPriorSubsNet v3.2 (Option-1). :contentReference[oaicite:1]{index=1}


1) What “identifiability” means in GeoPrior
---------------------------------------------

In GeoPrior, the physics module introduces a *timescale closure* linking
hydraulics and consolidation dynamics. In its common form:

.. math::

   \tau_{\mathrm{prior}}
   \;\approx\;
   \frac{H_d^2\,S_s}{\pi^2\,\kappa_b\,K}

where:

- :math:`H_d` is an effective drainage thickness (often :math:`H_d = \alpha H`
  via ``hd_factor``),
- :math:`S_s` is specific storage,
- :math:`K` is hydraulic conductivity,
- :math:`\kappa_b` is a stiffness/compressibility-like factor.

**Key implication (trade-off):**
even with perfect data, the closure mostly identifies the *combination*

.. math::

   \frac{H_d^2\,S_s}{K}

not necessarily :math:`H_d`, :math:`S_s`, and :math:`K` independently.
This is why GeoPrior uses priors, bounds, and regularizers:

- **Log-space closure prior** (``prior_loss``) nudges :math:`\tau` toward
  :math:`\tau_{\mathrm{prior}}`.
- **Bounds** keep parameters in plausible ranges.
- **Smoothness** prevents fitting noise with unrealistic spatial oscillations.


2) Identifiability targets and diagnostics
----------------------------------------------

GeoPrior’s identifiability is usually discussed at two levels:

A) **Timescale identifiability**
   - Can the model recover :math:`\tau` (or :math:`\log\tau`) accurately?

B) **Field identifiability (conditional)**
   - Given the closure, bounds, and priors, does the model recover consistent
     :math:`K`, :math:`S_s`, :math:`H_d` patterns?

A practical approach is to measure:

- Relative error in :math:`\tau` (median and high quantiles)
- Closure residual in log space:

  .. math::

     r_{\tau} = \log(\tau) - \log(\tau_{\mathrm{prior}})

- Offset diagnostics vs truth and vs prior (when truth/prior are known)

These are exactly what the SM3 experiment reports. 


3) SM3 synthetic identifiability experiment (recommended)
-----------------------------------------------------------

SM3 is designed to test “can GeoPrior learn tau?” under controlled conditions:

- single pixel (no spatial PDE),
- consolidation-only physics (``pde_mode="consolidation"``),
- horizon :math:`H=1` (one-step-ahead),
- synthetic drawdown forcing + exponential-response settlement,
- exports a **physics payload** and computes identifiability diagnostics.

3.1 What SM3 generates
*************************

SM3 samples priors and truth for lithology-dependent values:

- :math:`K_{\mathrm{prior}}, S_{s,\mathrm{prior}}, H_{\min}, H_{\max}`
- truth offsets around priors (log-spread in :math:`S_s` and :math:`\tau`)
- drawdown series :math:`\Delta h(t)` (step or ramp)
- settlement :math:`s(t)` from a simple exponential memory with timescale
  :math:`\tau_{\mathrm{true}}`.

It then trains GeoPrior on 1-step windows with:

- ``dynamic_features`` = past depth-to-water :math:`z_{\mathrm{GWL}}`
- ``coords`` = time coordinate only (x=y=0)
- ``H_field`` = a constant thickness per sample

See the script for the full generator and the window layout. 

3.2 Run SM3
*************

Example invocation (from the script header):

.. code-block:: bash

   python nat.com/sm3_synthetic_identifiability.py \
     --outdir results/sm3_synth \
     --n-realizations 30 \
     --n-years 20 \
     --time-steps 5 \
     --epochs 50 \
     --noise-std 0.02 \
     --load-type step \
     --seed 123

The script writes:

- ``sm3_synth_runs.csv`` (one row per realization)
- ``sm3_synth_summary.csv`` (summary statistics)
- for each realization:
  - ``phys_payload_val.npz``
  - ``sm3_identifiability_diag.json``
  - ``best.keras`` checkpoint

3.3 What to look at in SM3 outputs
************************************

The flattened diagnostics include (by design):

- ``tau_rel_q50``, ``tau_rel_q90``, ``tau_rel_q95``:
  relative error in tau estimate (median / tail)

- ``closure_log_resid_mean`` and ``closure_log_resid_q95``:
  distribution of :math:`r_{\tau}`

- offsets (quantiles), both **vs truth** and **vs prior**:
  ``vs_true_delta_K_*``, ``vs_prior_delta_K_*``, etc.

Interpretation guidelines
******************************

- If ``tau_rel_q50`` is small but ``tau_rel_q95`` is large:
  tau is “mostly” identifiable but can fail in hard cases (noise, short series,
  weak forcing). Consider increasing years, reducing noise, or strengthening
  consolidation loss.

- If closure residuals are large but tau errors are small:
  your model may learn tau via data fit while violating the closure prior,
  which usually means ``lambda_prior`` is too small (or closure settings mismatch).

- If :math:`K`, :math:`S_s`, :math:`H_d` offsets show large spreads but tau is OK:
  this is expected: tau is easier to identify than individual fields.

3.4 Time-unit hygiene in payloads (important)
***********************************************

SM3 explicitly carries both “payload time units” and “report time units” and
converts for consistent diagnostics. :contentReference[oaicite:4]{index=4}

Rule of thumb:

- Always compute errors in the **same unit system** for
  (:math:`\tau`, :math:`K`) and for truth/prior.
- When reporting tau in “years”, remember that K becomes “m/year” if you
  keep the closure consistent.


4) Physics Debug Harness (fastest way to isolate physics bugs)
---------------------------------------------------------------

When something explodes in real training (epsilons, derivatives, NaNs),
you want a minimal script that:

- builds Stage-1-like inputs,
- runs a forward pass,
- calls ``evaluate_physics(...)``,
- prints distribution summaries for residuals and fields.

That is exactly what the GeoPrior physics harness does. 

4.1 What the harness constructs
***********************************

The harness builds a synthetic dataframe and packs it into the GeoPrior input
dict with the same conventions as Stage-1 exports:

- ``coords``: ``(B,H,3)`` ordered as ``(t,x,y)``
- ``dynamic_features``: ``(B,T,D)``
- ``future_features``: ``(B,T+H,F)`` (TFT-like compatibility)
- ``static_features``: ``(B,S)``
- ``H_field``: ``(B,H,1)``
- optional stability inputs:
  - ``h_ref_si``: ``(B,1,1)``
  - ``s_init_si``: ``(B,1,1)``

It also assembles a local ``scaling_kwargs`` dict to exercise:

- coordinate shifting + optional normalization,
- affine SI mappings (scale+bias),
- gwl indexing (``gwl_dyn_index``),
- bounds payload.

See ``pack_inputs(...)`` and ``build_one_sample(...)``. 

4.2 Run the harness
*************************

.. code-block:: bash

   python nat.com/debug_geoprior_physics_harness.py

At startup, it prints toggles like:

- ``WORK_WITH_NORMALIZED_DATA``
- ``WORK_WITH_NORMALIZED_COORDS``
- ``USE_HEAD_PROXY``

and then executes:

- forward pass: ``model(xb, training=False)``
- physics evaluation: ``model.evaluate_physics(xb, return_maps=True)``

Finally it prints summary stats for keys such as:

- ``epsilon_prior``, ``epsilon_cons``, ``epsilon_gw``
- loss components (``prior_loss``, ``consolidation_loss``, etc.)
- learned fields (``K_field``, ``Ss_field``, ``tau_field``)
- optional source term ``Q_si``

See the bottom of the script for the printed key list.

4.3 How to use the harness for “why is epsilon exploding?”
************************************************************

Common investigations:

**A) Coordinate normalization chain-rule**
  Toggle ``WORK_WITH_NORMALIZED_COORDS``.
  If epsilons jump by orders of magnitude, your chain-rule scaling or
  coord_ranges are inconsistent.

**B) Time units**
  Set ``TIME_UNITS="year"`` (default) and print conversions:
  the harness shows how to convert :math:`\epsilon_{cons}` from m/s to
  mm/time_unit using ``seconds_per_time_unit``. :contentReference[oaicite:8]{index=8}

**C) Head semantics**
  Toggle ``USE_HEAD_PROXY``.
  If head is computed as a proxy (e.g., :math:`h \approx -z_{\mathrm{GWL}}`)
  you can isolate whether your :math:`z_\mathrm{surf}` / depth-to-head conversion
  is the root cause.


5) Recommended “identifiability workflow” for real projects
-------------------------------------------------------------

1) **Run the harness** to validate:
   - dict input shapes,
   - scaling_kwargs consistency,
   - finite physics outputs and stable epsilons.

2) **Run SM3** (or a variant) to validate:
   - tau recovery across realizations,
   - sensitivity to noise, series length, forcing type.

3) **Only then** run full Stage-2 training.

If SM3 fails systematically, do *not* trust recovered :math:`K,S_s,\tau` in
real experiments yet—fix closure settings, scaling, or loss weights first.


6) What’s next
-----------------

- :doc:`diagnostics` explains how to plot epsilons and physics payload maps.
- :doc:`losses` and :doc:`maths` provide the exact residual definitions.
- After this page, we can add:
  - a “field identifiability” section (spatial K/Ss/Hd maps),
  - recommended experiment matrix (ablation table),
  - guidance on interpreting closure residual maps.
