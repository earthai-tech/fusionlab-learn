.. _geoprior_inference_export:

=================================
Inference, export, and deployment
=================================

GeoPrior Stage-2 produces two kinds of outputs:

1) **Model artifacts** (portable model bundle + weights + manifest)
2) **Forecast artifacts** (evaluation CSVs, future forecasts, and optional
   calibrated quantile intervals)

This page documents the recommended inference workflow, the export formats,
and the practical notes for using GeoPrior predictions downstream.

.. seealso::

   - :doc:`pipeline` for Stage-1/Stage-2 responsibilities and files.
   - :doc:`model` for output tensor layouts (quantile axis).
   - :doc:`diagnostics` for sanity plots and payload exports.


1) Inference modes
---------------------

GeoPrior supports two inference modes:

A) **In-memory model inference** (same Python process)
   Use this for validation runs and for quick iterations inside Stage-2.

B) **Portable inference from saved artifacts**
   Use this for deployment or for running inference later without retraining.

Both modes use the same input contract: dict inputs with the required keys
(``static_features``, ``dynamic_features``, ``future_features``, ``coords``,
and any physics auxiliaries such as ``H_field``).


2) Predicting: what to call and what you get back
-------------------------------------------------

2.1 Keras ``model.predict`` (supervised outputs only)
******************************************************

The simplest inference call:

.. code-block:: python
   :linenos:

   y_pred = model.predict(ds_val)

For GeoPrior, predictions are typically returned in the Keras multi-output
format. In most pipelines, you want the dict:

.. code-block:: python
   :linenos:

   y_pred = model(batch, training=False)
   s_pred = y_pred["subs_pred"]
   h_pred = y_pred["gwl_pred"]

**Output shapes**


- Without quantiles: ``(B, H, 1)``
- With quantiles: ``(B, H, Q, 1)`` (q-axis = 2)

See :doc:`model` for details and safe extraction patterns.

2.2 Aux inference for debugging physics
******************************************

When you need physics logits and intermediate tensors:

.. code-block:: python
   :linenos:

   y_pred, aux = model.forward_with_aux(batch, training=False)

This is recommended for debugging unit issues (non-finite logits, thickness
floors, etc.) and for generating physics payloads (see :doc:`diagnostics`).


3) Exported forecast CSVs (Stage-2)
-------------------------------------

Stage-2 writes forecast CSVs in two common forms:

A) **Evaluation CSV** (has actuals)
   Contains one row per sample and per horizon step.

B) **Future forecast CSV** (no actuals)
   Same schema but without the target columns.

**Typical evaluation columns**


A common schema looks like:

- ``sample_idx``: integer id of the sample within the evaluated dataset
- ``forecast_step``: 1..H
- ``coord_t``: physical time label (e.g. year)
- ``coord_x``, ``coord_y``: coordinates in dataset units (often meters)
- ``subsidence_actual``: observed subsidence
- ``subsidence_q10``, ``subsidence_q50``, ``subsidence_q90``: predicted quantiles
- optional: ``gwl_actual`` and GWL quantiles (if exported)

.. note::

   If you do not enable quantiles, Stage-2 exports point predictions as
   ``*_pred`` or maps q50 to a single column. The exact naming is configured
   in the Stage-2 formatter.


4) Quantile interval calibration (optional)
--------------------------------------------

GeoPrior supports post-hoc **symmetric interval scaling** per horizon to match
a target empirical coverage on validation data.

Concept
********

Given predicted quantiles (e.g. q10, q50, q90), the calibrator rescales the
interval around the median:

.. math::

   \tilde{q}_{lo}
   =
   q_{50} - f\,(q_{50}-q_{lo}),
   \qquad
   \tilde{q}_{hi}
   =
   q_{50} + f\,(q_{hi}-q_{50})

where :math:`f \ge 1` is chosen by bisection to hit a target coverage level
(e.g. 0.8 for the central 80% interval).

How Stage-2 uses it
********************

- Fit calibrator on validation outputs (per horizon)
- Apply to evaluation and/or future forecasts
- Save calibrated CSV (often suffixed with ``_calibrated.csv``)

.. tip::

   Calibrating intervals is almost always worth it when you care about
   coverage metrics. It does **not** change the median prediction; it only
   scales the spread.


5) Saving the model: portable artifacts
----------------------------------------

GeoPrior export can produce:

- a Keras-native file (``.keras``) and/or weights file (``.weights.h5``)
- a manifest JSON (builder config, scalers, feature lists, stage metadata)
- optionally a TensorFlow SavedModel directory (when configured)

**Recommended bundle layout**


.. code-block:: text

   run_dir/
     model.keras
     weights.weights.h5
     manifest.json
     scaling_kwargs.json
     stage2_handshake_audit.json

The manifest is designed to be sufficient to rebuild the model and restore
weights even when full deserialization is unavailable (e.g. strict Keras 3
environments).


6) Loading for inference (portable)
-------------------------------------

The recommended load strategy is:

1) Try native load (``keras.models.load_model`` on ``.keras`` / ``.h5``)
2) If not possible, rebuild from manifest + load weights
3) If TensorFlow SavedModel is used in Keras 3, wrap as ``TFSMLayer`` for inference

**Example (conceptual)**


.. code-block:: python
   :linenos:

   from fusionlab.compat.keras import load_model_bundle

   model, manifest = load_model_bundle(
       run_dir="path/to/run_dir",
       endpoint="serve",     # when using TF SavedModel export
   )

   y = model.predict(ds_future)

.. note::

   If your deployment environment differs from training (Keras 2 vs Keras 3),
   prefer the “rebuild + load weights” path for maximum portability.


7) Exporting predictions from raw arrays
------------------------------------------

If you are not using Stage-2’s CSV writer, you can construct your own
forecast DataFrame by aligning:

- sample index
- horizon step
- coordinate labels (time + x + y)
- actuals (optional)
- predictions (point or quantiles)

**Quantile extraction reminder**


For q-axis = 2 and output dim = 1:

.. code-block:: python
   :linenos:

   s = y_pred["subs_pred"]          # (B,H,Q,1)
   q10 = s[:, :, 0, 0]
   q50 = s[:, :, 1, 0]
   q90 = s[:, :, 2, 0]


8) Deployment notes and best practices
---------------------------------------

8.1 Always ship scaling_kwargs + feature metadata
**************************************************

GeoPrior predictions are only meaningful when:

- coordinates are normalized the same way,
- time units are consistent,
- thickness and head conventions match training.

Therefore, always ship:

- ``scaling_kwargs.json``
- the feature lists (static/dynamic/future)
- any scalers/encoders used in Stage-1

8.2 Determinism and reproducibility
**************************************

For stable inference:

- freeze dropout (use ``training=False``)
- set random seeds if you rebuild models
- keep exact versions of TF/Keras for strict bit-level reproducibility

8.3 Batch size at inference
*****************************

Large batches speed inference but can increase memory due to:

- auxiliary physics tensors (if you use aux forward),
- large horizon size,
- large spatial sample count.

Start with modest batch sizes for export tasks.


9) Next steps
---------------

- :doc:`diagnostics` to validate exported models and forecasts
- :doc:`faq` for common load/export issues (Keras 2 vs Keras 3, SavedModel, etc.)
- :doc:`references` for citations and the theory background
