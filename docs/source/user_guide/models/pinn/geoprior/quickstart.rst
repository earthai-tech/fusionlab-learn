.. _geoprior_quickstart:

=====================
GeoPrior: Quickstart
=====================

:API Reference: :class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet`

This quickstart shows the **minimum, correct workflow** to:

1) build GeoPrior dict inputs (including ``coords`` and ``H_field``),
2) compile the model with supervised losses + physics weights,
3) run a short training loop,
4) run inference and interpret the output structure.

If you are integrating GeoPrior into the Stage-1/Stage-2 pipeline, read
:doc:`pipeline` and :doc:`data_and_units` alongside this page.

.. important::

   GeoPrior is physics-informed: if ``coords`` units, normalization spans
   (``coord_ranges``), or ``time_units`` are wrong, you can get exploding
   physics epsilons even when the supervised loss decreases.
   Always validate your ``scaling_kwargs`` first (see :doc:`data_and_units`).


Prerequisites
=============

GeoPrior works with either standalone Keras or ``tf.keras``.
Use whichever is installed in your environment:

.. code-block:: python
   :linenos:

   import tensorflow as tf

   try:
       import keras
   except Exception:  # pragma: no cover
       from tensorflow import keras


1) Smoke-test with synthetic tensors
====================================

This section is a **shape-and-plumbing** test. It verifies:

* dict inputs are accepted,
* model forward pass returns ``{"subs_pred", "gwl_pred"}``,
* training runs end-to-end.

You can start with physics disabled (``pde_mode="none"``), then enable it.

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.pinn.geoprior.models import GeoPriorSubsNet

   # ----------------------------
   # Dimensions (example)
   # ----------------------------
   B = 16          # batch size
   T = 12          # past window length
   H = 3           # forecast horizon
   S = 4           # static dims
   D = 10          # dynamic dims
   F = 6           # future-known dims

   # ----------------------------
   # Build a synthetic batch
   # ----------------------------
   x = {
       "static_features": tf.random.normal([B, S]),
       "dynamic_features": tf.random.normal([B, T, D]),
       "future_features": tf.random.normal([B, H, F]),
       "coords": tf.random.normal([B, H, 3]),          # (t, x, y)
       "H_field": tf.ones([B, H, 1]) * 50.0,           # thickness [m]
   }

   y = {
       "subs_pred": tf.random.normal([B, H, 1]),
       "gwl_pred": tf.random.normal([B, H, 1]),
   }

   # ----------------------------
   # Minimal scaling payload
   # (for real runs, load from Stage-1 exports)
   # ----------------------------
   scaling_kwargs = {
       "time_units": "year",
       "coords_normalized": False,
       "coords_in_degrees": False,
       "coord_order": ["t", "x", "y"],
       "subs_scale_si": 1.0,
       "head_scale_si": 1.0,
       "H_scale_si": 1.0,
       "subsidence_kind": "cumulative",
   }

   # ----------------------------
   # 1) Bring-up mode: physics off
   # ----------------------------
   model = GeoPriorSubsNet(
       static_input_dim=S,
       dynamic_input_dim=D,
       future_input_dim=F,
       forecast_horizon=H,
       max_window_size=T,
       pde_mode="none",                 # <-- start here
       scaling_kwargs=scaling_kwargs,
   )

   model.compile(
       optimizer=keras.optimizers.Adam(1e-3),
       loss={"subs_pred": "mse", "gwl_pred": "mse"},
   )

   # Single-step smoke training
   history = model.fit(x, y, epochs=2, verbose=1)

   # Forward pass
   yhat = model(x, training=False)
   print("outputs:", list(yhat.keys()))
   print("subs_pred:", yhat["subs_pred"].shape)
   print("gwl_pred:", yhat["gwl_pred"].shape)


2) Enable physics (PINN mode)
=============================

Once the dict shapes are correct, enable physics. GeoPrior uses a composite
objective:

* supervised data loss from ``compile(..., loss=...)``
* + physics loss assembled internally and weighted by lambdas (set in ``compile``)

.. code-block:: python
   :linenos:

   model_phys = GeoPriorSubsNet(
       static_input_dim=S,
       dynamic_input_dim=D,
       future_input_dim=F,
       forecast_horizon=H,
       max_window_size=T,
       pde_mode="both",                 # <-- enable physics
       residual_method="exact",
       scale_pde_residuals=True,
       scaling_kwargs=scaling_kwargs,
   )

   model_phys.compile(
       optimizer=keras.optimizers.Adam(1e-3),
       loss={"subs_pred": "mse", "gwl_pred": "mse"},
       # Physics component weights (typical bring-up values)
       lambda_cons=1.0,
       lambda_gw=1.0,
       lambda_prior=1.0,
       lambda_smooth=0.1,
       lambda_bounds=0.0,
       lambda_q=0.0,
       # Global physics multiplier (depends on offset_mode)
       lambda_offset=0.1,
   )

   history = model_phys.fit(x, y, epochs=2, verbose=1)


3) Plug in Stage-1 exports (recommended workflow)
=================================================

In a real run, you should load:

* arrays/sequences exported by Stage-1 (NPZ or similar)
* ``scaling_kwargs.json``
* optionally the Stage-1/Stage-2 audits for sanity checks

The exact filenames depend on your pipeline, but the pattern is always:

1) load arrays,
2) build ``x_dict`` and ``y_dict`` with required keys,
3) pass the Stage-1 scaling payload into the model.

.. code-block:: python
   :linenos:

   import json
   import numpy as np
   import tensorflow as tf

   from fusionlab.nn.pinn.geoprior.models import GeoPriorSubsNet

   # ----------------------------
   # Load Stage-1 artifacts
   # ----------------------------
   # Example paths (adjust to your outputs)
   npz_path = "path/to/stage1_arrays_train.npz"
   scaling_path = "path/to/scaling_kwargs.json"

   arrays = np.load(npz_path, allow_pickle=True)
   with open(scaling_path, "r", encoding="utf-8") as f:
       scaling_kwargs = json.load(f)

   # ----------------------------
   # Map arrays -> dict inputs
   # (use your Stage-1 naming convention)
   # ----------------------------
   x_train = {
       "static_features": arrays["X_static"],     # (N, S)
       "dynamic_features": arrays["X_dynamic"],   # (N, T, D)
       "future_features": arrays["X_future"],     # (N, H, F)
       "coords": arrays["coords"],                # (N, H, 3)
       "H_field": arrays["H_field"],              # (N, H, 1)
   }

   y_train = {
       "subs_pred": arrays["y_subs"],             # (N, H, 1)
       "gwl_pred": arrays["y_gwl"],               # (N, H, 1)
   }

   # ----------------------------
   # Build tf.data (recommended)
   # ----------------------------
   BATCH_SIZE = 256
   ds_train = (
       tf.data.Dataset.from_tensor_slices((x_train, y_train))
       .shuffle(10_000)
       .batch(BATCH_SIZE)
       .prefetch(tf.data.AUTOTUNE)
   )

   # ----------------------------
   # Instantiate + compile
   # ----------------------------
   S = x_train["static_features"].shape[-1]
   D = x_train["dynamic_features"].shape[-1]
   F = x_train["future_features"].shape[-1]
   H = x_train["coords"].shape[1]
   T = x_train["dynamic_features"].shape[1]

   model = GeoPriorSubsNet(
       static_input_dim=S,
       dynamic_input_dim=D,
       future_input_dim=F,
       forecast_horizon=H,
       max_window_size=T,
       pde_mode="both",
       scaling_kwargs=scaling_kwargs,
   )

   model.compile(
       optimizer=keras.optimizers.Adam(1e-3),
       loss={"subs_pred": "mse", "gwl_pred": "mse"},
       lambda_cons=1.0,
       lambda_gw=1.0,
       lambda_prior=1.0,
       lambda_smooth=0.1,
       lambda_offset=0.1,
   )

   model.fit(ds_train, epochs=20, verbose=1)


4) Inference: interpreting outputs
==================================

GeoPrior returns a dict with at least:

* ``subs_pred``
* ``gwl_pred``

For point forecasting (no quantiles), typical shapes are:

* ``subs_pred``: ``(B, H, 1)``
* ``gwl_pred``:  ``(B, H, 1)``

.. code-block:: python
   :linenos:

   yhat = model.predict(ds_train.take(1))
   print(yhat.keys())
   print(yhat["subs_pred"].shape, yhat["gwl_pred"].shape)


Optional: probabilistic forecasts (quantiles)
=============================================

To enable quantiles, pass a list such as:

.. code-block:: python

   quantiles=[0.1, 0.5, 0.9]

GeoPrior will then produce quantile-aware tensors (with an additional quantile
axis). The exact layout and recommended losses/metrics are described in
:doc:`model` and :doc:`losses`.

.. note::

   For real probabilistic training, use a **quantile/pinball loss** (and optional
   calibration). The quickstart uses MSE for simplicity only.

Next steps
==========

* Understand the scaling contract and audits: :doc:`data_and_units`
* Learn the model outputs, quantile layout, and physics toggles: :doc:`model`
* Dive into the physics core (derivatives, residual assembly): :doc:`core`
* Plot and debug epsilon/residual curves: :doc:`diagnostics`
* Export forecasts and artifacts: :doc:`inference_and_export`
