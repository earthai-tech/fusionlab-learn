.. _geoprior_diagnostics:

=========================================
Diagnostics: Physics sanity checks & plots
=========================================

GeoPrior exposes two complementary diagnostic layers:

1) **Training-time diagnostics** from the Keras ``History`` object
   (loss terms, epsilons, gates, etc.).

2) **Post-training physics payloads** exported from a trained model and a
   dataset, used for sanity plots (maps, histograms, scatter plots) and for
   identifiability summaries.

This page shows the recommended workflow and the exact helper functions you
can call.

.. seealso::

   - For definitions of epsilon metrics and residual scaling, see :doc:`maths`.
   - For log names and how lambdas combine, see :doc:`losses`.
   - For end-to-end training flow, see :doc:`training` and :doc:`pipeline`.


1) Quick checklist: what to look at first
-------------------------------------------

When a run is unstable, or you suspect unit/scale issues, check in this order:

1. **History curves**
   - ``data_loss`` vs ``physics_loss_scaled``
   - ``epsilon_gw_raw`` / ``epsilon_cons_raw`` (explosion = unit mismatch)
   - ``epsilon_gw`` / ``epsilon_cons`` (scaled view)

2. **Gates / warmup**
   - ``physics_loss_scaled`` stuck ~0 => warmup/ramp too long, or physics disabled
   - if enabled: ``q_gate``, ``subs_resid_gate``

3. **Physics payload**
   - sanity of distributions: ``log10_tau``, ``log10_tau_prior``
   - scatter: closure consistency (``tau_prior`` vs ``tau``)
   - maps: spatial patterns of ``K``, ``Ss``, ``Hd``


2) Training-history plotting (the “must-have” plots)
------------------------------------------------------

The GeoPrior plotting helper lives in:

- :mod:`~fusionlab.nn.pinn.geoprior.plot` :contentReference[oaicite:0]{index=0}

It provides robust, crash-resistant plotting and two GeoPrior-specific
autoplots.

2.1 Plot only epsilons
***********************

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.plot import plot_epsilons_in

   plot_epsilons_in(
       history,                         # keras History or dict-like
       title="GeoPrior | epsilons",
       savefig="figs/geoprior_epsilons.png",
   )

This plots all keys starting with ``epsilon_`` (including ``*_raw``) using a
safe log scale that falls back to *symlog* if values are non-positive.

2.2 Plot physics loss terms (log/symlog)
******************************************

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.plot import plot_physics_losses_in

   plot_physics_losses_in(
       history,
       title="GeoPrior | physics terms",
       savefig="figs/geoprior_physics_terms.png",
   )

This plots key physics-related scalars such as:

- ``physics_loss``, ``physics_loss_scaled``
- ``consolidation_loss``, ``gw_flow_loss``
- ``prior_loss``, ``smooth_loss``, ``bounds_loss``, ``mv_prior_loss``
- optional: ``q_reg_loss``, ``q_rms``, ``q_gate``, ``subs_resid_gate``

2.3 One-call autoplot for GeoPrior
*************************************

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.plot import autoplot_geoprior_history

   autoplot_geoprior_history(
       history,
       outdir="figs",
       prefix="nansha_geoprior",
   )

This produces:

- ``<prefix>_epsilons.png``
- ``<prefix>_physics_terms.png``


3) Physics payloads: export, save, load
-----------------------------------------

For post-training diagnostics (maps/scatter/hist), GeoPrior can export a
**flat payload** that contains one-dimensional arrays for fields and residuals.

The payload helper lives in:

- :mod:`~fusionlab.nn.pinn.geoprior.diagnostics.payloads` 

3.1 Gather a payload from a dataset
******************************************

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.diagnostics.payloads import (
       gather_physics_payload, default_meta_from_model
   )

   payload = gather_physics_payload(
       model,
       ds_val,           # tf.data.Dataset yielding inputs or (inputs, targets)
       max_batches=50,   # optional: keep it small for quick plots
   )

   meta = default_meta_from_model(model)

**What you get (typical keys)**


The payload is designed for plotting. It usually includes:

- ``tau`` (s)
- ``tau_prior`` / ``tau_closure`` (s)
- ``K`` (m/s)
- ``Ss`` (1/m)
- ``Hd`` (m)  (or falls back to H if Hd absent)
- ``cons_res_vals`` (m/s, consolidation residual diagnostic)
- derived:
  - ``log10_tau``
  - ``log10_tau_prior`` / ``log10_tau_closure``

It also contains ``payload["metrics"]`` with quick sanity stats, e.g.:

- ``eps_prior_rms`` = RMS of :math:`\log(\tau)-\log(\tau_{prior})`
- ``r2_logtau`` = R² between log-timescales
- optional: ``closure_consistency_rms`` if scalar kappa can be extracted

.. note::

   The payload layer **does not convert units** itself. It assumes that
   ``model.evaluate_physics(..., return_maps=True)`` returns SI-consistent
   values (as intended by GeoPrior).

3.2 Save payload + sidecar metadata
****************************************

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.diagnostics.payloads import save_physics_payload

   out_path = save_physics_payload(
       payload,
       meta,
       path="exports/physics_payload.npz",
       format="npz",        # "npz" (recommended), "csv", or "parquet"
       overwrite=True,
   )

This writes:

- ``exports/physics_payload.npz``
- ``exports/physics_payload.npz.meta.json``

3.3 Load payload later
****************************

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.diagnostics.payloads import load_physics_payload

   payload, meta = load_physics_payload("exports/physics_payload.npz")


4) Plotting payload values (maps + histograms)
------------------------------------------------

The plotting helper can visualize any numeric arrays in the payload as:

- **maps** (colored scatter in x–y)
- **histograms**
- or **both**

It also provides coordinate collection from a dataset.

4.1 Gather flat coords from a dataset
*******************************************

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.plot import gather_coords_flat

   coords = gather_coords_flat(ds_val)   # returns {"t","x","y"} as 1D arrays

4.2 Map plot: K, Ss, tau, residuals
***************************************

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.plot import plot_physics_values_in

   plot_physics_values_in(
       payload,
       coords=coords,
       mode="map",
       keys=["log10_tau", "log10_tau_prior", "K", "Ss", "Hd", "cons_res_vals"],
       title="GeoPrior | physics payload maps",
       savefig="figs/payload_maps.png",
   )

4.3 Histograms (distribution sanity checks)
**************************************************

.. code-block:: python
   :linenos:

   plot_physics_values_in(
       payload,
       mode="hist",
       keys=["log10_tau", "log10_tau_prior", "K", "Ss", "Hd"],
       title="GeoPrior | parameter distributions",
       savefig="figs/payload_hists.png",
       bins=80,
   )

4.4 Both map + hist for each key
**************************************

.. code-block:: python
   :linenos:

   plot_physics_values_in(
       payload,
       dataset=ds_val,        # if coords not provided, it will derive coords
       mode="both",
       keys=["cons_res_vals", "K", "Ss", "Hd"],
       transform="signed_log10",   # useful for signed residuals
       title="GeoPrior | maps + hists",
       savefig="figs/payload_both.png",
   )

.. tip::

   Use ``transform="signed_log10"`` for residuals that can be negative,
   and ``transform="log10"`` for strictly-positive quantities.


5) Interpreting the most important diagnostic plots
-------------------------------------------------------

5.1 epsilons (training-time)
******************************

- ``epsilon_cons_raw`` / ``epsilon_gw_raw``:
  SI-magnitude diagnostics. If these **explode**, it is usually a unit mismatch
  (time units, coordinate spans, or head/subsidence scale).

- ``epsilon_cons`` / ``epsilon_gw``:
  scaled (nondimensionalized) residual RMS. If these explode, the issue is
  deeper than a simple scale mismatch (often derivative conversion or NaNs).

5.2 tau vs tau_prior (post-training payload)
************************************************

Plotting the distributions:

- ``log10_tau``:
  learned effective relaxation time
- ``log10_tau_prior``:
  closure-implied timescale from :math:`(H_d,S_s,K,\kappa)`

Rules of thumb:

- If the two distributions are completely disjoint, the closure is not
  compatible with the learned fields or the closure settings (kappa mode,
  Hd_factor, etc.) are inconsistent.

- If ``eps_prior_rms`` is small and ``r2_logtau`` is high, your closure is
  internally consistent and the residual :math:`\Delta\log\tau` is not doing
  extreme compensation.

5.3 Spatial maps
*********************

- ``K`` and ``Ss`` should be smooth-ish (unless you intentionally allow sharp
  changes). If you see salt-and-pepper noise, increase ``lambda_smooth`` or
  constrain the correction network.

- ``cons_res_vals`` maps should not show extreme coherent artifacts aligned
  with coordinate normalization/scaling issues (e.g., stripes aligned with x or y).


6) Common gotchas (and how the helpers handle them)
----------------------------------------------------

**(A) Log-scale plotting crashes**
  The plotting code requests ``log`` but automatically falls back to ``symlog``
  if values include zeros/negatives.

**(B) Payload values and coords length mismatch**
  Payload arrays may be per-point while coords may be per-sequence (or vice
  versa). The plotting helper tries simple downsampling/upsampling alignment
  and falls back to truncation if needed.

**(C) Huge file size**
  Use ``max_batches`` in ``gather_physics_payload`` or subsample payload rows
  before saving.


7) Minimal “diagnostics workflow” snippet (recommended)
--------------------------------------------------------

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.plot import (
       autoplot_geoprior_history, plot_physics_values_in, gather_coords_flat
   )
   from fusionlab.nn.pinn.geoprior.diagnostics.payloads import (
       gather_physics_payload, default_meta_from_model, save_physics_payload
   )

   # 1) History plots (fast)
   autoplot_geoprior_history(history, outdir="figs", prefix="run01")

   # 2) Physics payload (slower, but most informative)
   payload = gather_physics_payload(model, ds_val, max_batches=50)
   meta = default_meta_from_model(model)
   save_physics_payload(payload, meta, path="exports/run01_payload.npz", overwrite=True)

   # 3) Payload plots
   coords = gather_coords_flat(ds_val, max_batches=50)
   plot_physics_values_in(
       payload,
       coords=coords,
       mode="both",
       keys=["log10_tau", "log10_tau_prior", "K", "Ss", "Hd", "cons_res_vals"],
       transform=None,
       savefig="figs/run01_payload_both.png",
   )
