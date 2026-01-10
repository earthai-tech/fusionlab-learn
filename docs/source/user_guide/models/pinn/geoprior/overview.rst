.. _geoprior_overview:

====================
GeoPrior: Overview
====================

:API Reference: :class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet`

GeoPrior is a physics-informed forecasting module for **land subsidence**
and **groundwater head/level** dynamics. It combines:

1) a **data-driven backbone** (attention/LSTM/Transformer style, via
   :class:`~fusionlab.nn.models.BaseAttentive` inheritance),
2) **learned effective physical fields** (e.g., :math:`K`, :math:`S_s`,
   :math:`\\tau`, and optional forcing :math:`Q`),
3) a **PINN-style physics pathway** that penalizes violations of governing
   residuals during training.

The core idea is simple:

*Use ML to learn what data explains well, but require the outputs to remain
physically plausible by enforcing soft constraints derived from hydrogeology
and poroelastic consolidation.*

What GeoPrior predicts
-----------------------

GeoPrior produces two supervised outputs (returned as a dict):

* ``subs_pred``: subsidence forecast (typically cumulative subsidence)
* ``gwl_pred``: groundwater variable forecast (either head, or a depth-like
  proxy depending on your data conventions)

If you enable probabilistic forecasting via ``quantiles``, outputs may include
a quantile axis (the exact layout is described in :doc:`model`).

When to use GeoPrior
---------------------

Use GeoPrior when one (or more) of the following apply:

* You want **physics consistency** in forecasts (not only accuracy).
* You have limited/noisy observations and need a regularizer grounded in
  governing processes.
* You want to recover **effective** spatial closures (e.g., :math:`K(x,y)`,
  :math:`S_s(x,y)`, :math:`\\tau(x,y)`), not only predictions.
* You want stable training with a **physics warmup/ramp** and residual scaling
  strategies (see :doc:`training` and :doc:`core`).

High-level architecture
------------------------

GeoPrior can be read as a pipeline with two coupled branches:

**A) Supervised branch (data-driven)**
  Learns to map rich covariates (static, dynamic past, known future) into
  forecasts of subsidence and groundwater variable.

**B) Physics branch (PINN)**
  Uses the same forward tensors and the same spatio-temporal coordinates
  to compute residual maps and physics losses.

A useful mental diagram (conceptual):

.. code-block:: text

   inputs(dict)
     ├─ static_features      (B, S)
     ├─ dynamic_features     (B, T, D)
     ├─ future_features      (B, H, F)
     ├─ coords               (B, H, 3)   -> (t, x, y)
     └─ H_field / thickness  (B, H, 1)   -> aquifer / drainage thickness
           |
           v
   shared encoder/decoder (BaseAttentive)
           |
           +--> data head  -> {subs_pred, gwl_pred}
           |
           +--> physics head logits -> (K_logits, Ss_logits, dlogtau_logits, Q_logits)
                 |
                 v
           bounded physical fields -> (K, Ss, tau, tau_prior, Q)
                 |
                 v
           autodiff derivatives wrt coords
                 |
                 v
           residual maps (gw_flow, consolidation, priors, bounds)
                 |
                 v
           physics loss terms + epsilon diagnostics

Batch contract (dict inputs)
----------------------------

GeoPrior expects **dict inputs** (not a single tensor). This is non-negotiable
because the physics pathway needs ``coords`` and typically a thickness field.

Required keys (most common)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``coords``:
  Tensor of shape ``(B, H, 3)`` ordered as ``(t, x, y)``.
* ``H_field`` (or alias such as ``soil_thickness`` depending on your pipeline):
  thickness field broadcastable to ``(B, H, 1)``.

Common optional keys
^^^^^^^^^^^^^^^^^^^^

* ``static_features``: ``(B, S)``
* ``dynamic_features``: ``(B, T, D)``
* ``future_features``: ``(B, H, F)``

.. note::

   The exact required keys depend on your Stage-1 export and scaling conventions.
   The authoritative description of scaling and required signals lives in
   :doc:`data_and_units` and :doc:`pipeline`.

Core physics residual families
------------------------------

GeoPrior supports multiple residuals, enabled by ``pde_mode``:

* Groundwater flow residual (``gw_flow``):

  .. math::

     R_{gw} = S_s \\, \\partial_t h - \\nabla \\cdot (K \\, \\nabla h) - Q

* Consolidation relaxation residual (``consolidation``):

  .. math::

     R_{cons} = \\partial_t s - \\frac{s_{eq}(h) - s}{\\tau}

* Time-scale closure / prior residual (always available when physics is on),
  tying learned :math:`\\tau` to a closure :math:`\\tau_{phys}` built from
  learned fields and thickness.

* Optional regularizers: smoothness priors and bounds penalties.

For the full definitions and how they are assembled into a total physics loss,
see :doc:`losses` and :doc:`maths`.

Units, scaling, and why they matter
-----------------------------------

GeoPrior is strict about **units** and **coordinate conventions** because
physics derivatives are sensitive to scaling:

* If coords are normalized (common), derivatives must be rescaled by coordinate
  spans (and squared for second derivatives).
* If coords are in degrees, they must be mapped to meters before spatial PDE
  evaluation.
* If time is expressed as “year index”, derivatives must be converted to
  per-second (unless already SI).

All of this is governed by ``scaling_kwargs`` (or a
:class:`~fusionlab.nn.pinn.geoprior.scaling.GeoPriorScalingConfig` wrapper).

This is why the recommended reading order is:

:doc:`data_and_units` → :doc:`core` → :doc:`losses` → :doc:`maths`

Typical workflow (Stage-1 → Stage-2)
-------------------------------------

A practical GeoPrior run usually follows the project pipeline:

1) **Stage-1**: prepare data, build sequences, export arrays and metadata.
2) **Stage-2**: instantiate model, compile, train, evaluate, export artifacts.
3) **Diagnostics**: inspect epsilon metrics and residual plots; export physics
   payloads for paper-quality figures.

See :doc:`pipeline`, :doc:`training`, :doc:`diagnostics`,
and :doc:`inference_and_export`.

Minimal example (shape-only)
-------------------------------

.. code-block:: python
   :linenos:

   import tensorflow as tf
   from fusionlab.nn.pinn.geoprior.models import GeoPriorSubsNet

   B, T, H = 8, 12, 3
   S, D, F = 4, 10, 6

   batch = {
       "static_features": tf.random.normal([B, S]),
       "dynamic_features": tf.random.normal([B, T, D]),
       "future_features": tf.random.normal([B, H, F]),
       "coords": tf.random.normal([B, H, 3]),      # (t, x, y)
       "H_field": tf.ones([B, H, 1]) * 50.0,       # thickness in meters
   }

   model = GeoPriorSubsNet(
       static_input_dim=S,
       dynamic_input_dim=D,
       future_input_dim=F,
       forecast_horizon=H,
       pde_mode="both",
       scaling_kwargs={  # minimal sketch; see data_and_units
           "time_units": "year",
           "coords_normalized": False,
           "coords_in_degrees": False,
       },
   )

   y = model(batch, training=False)
   print(y.keys())  # subs_pred, gwl_pred

Where to go next
------------------

* If you want an end-to-end runnable recipe: :doc:`quickstart`
* If you need to understand units and coordinate conventions: :doc:`data_and_units`
* If you want the full model API and outputs: :doc:`model`
* If you want the physics pathway and derivative handling: :doc:`core`
* If you want the full loss decomposition and epsilon terms: :doc:`losses`
* If you want the exact math and closure definitions: :doc:`maths`
