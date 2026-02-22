.. _geoprior_model:

===========================================
GeoPriorSubsNet: model API and I/O contract
===========================================

:API Reference: :class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet`

``GeoPriorSubsNet`` is the flagship **GeoPrior (PINN)** model used in the
Stage-2 training workflow. It couples a feature-driven
encoder/decoder backbone (inherited from :class:`~fusionlab.nn.models.BaseAttentive`)
with a physics pathway that regularizes the forecast via:

* a **groundwater flow** residual (Darcy-style PDE),
* a **consolidation / relaxation** residual (ODE closure),
* learned effective fields (:math:`K`, :math:`S_s`, :math:`\tau`) with priors.

The most important design choice is that the **mean subsidence path** is
physics-driven: it is computed from the predicted head through the
consolidation integrator, with an optional learnable residual around that
mean.

What the model returns
----------------------

Keras supervised outputs
^^^^^^^^^^^^^^^^^^^^^^^^
The public :meth:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet.call`
returns **only** the supervised outputs used by Keras losses/metrics:

* ``"subs_pred"``: subsidence forecast
* ``"gwl_pred"``: groundwater level (or head-like target) forecast

Both are returned as a **dict**, and the key order is stable
(``model.output_names`` / ``GeoPriorSubsNet.OUTPUT_KEYS``).

Auxiliary tensors for diagnostics (opt-in)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For research/diagnostics, use
:meth:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet.forward_with_aux`
to obtain **both** the supervised outputs and the auxiliary tensors used by
the physics pathway.

This dual interface keeps the Keras contract stable while still exposing
everything needed for debugging.

Constructor overview
--------------------

The constructor mixes four families of arguments:

1) **Backbone / sequence modeling** (inherited patterns)
   ``embed_dim``, ``hidden_units``, ``lstm_units``, ``attention_units``,
   ``num_heads``, ``dropout_rate``, ``max_window_size``, ``memory_size``,
   ``scales``, ``multi_scale_agg``, ``final_agg``, ``activation``,
   ``use_vsn``, ``vsn_units``, ``architecture_config``

2) **Supervised output contract**
   ``output_subsidence_dim``, ``output_gwl_dim``, ``forecast_horizon``

3) **Uncertainty / quantiles**
   ``quantiles`` (when provided, outputs include a quantile axis)

4) **Physics configuration and units**
   ``pde_mode``, ``residual_method``, ``mv``, ``kappa``, ``gamma_w``,
   ``h_ref``, ``use_effective_h``, ``hd_factor``, ``kappa_mode``,
   ``offset_mode``, ``bounds_mode``, ``time_units``, ``scale_pde_residuals``,
   and ``scaling_kwargs`` (unit/coordinate scaling + bounds/prior settings)


Important GeoPrior-specific switches
------------------------------------

``pde_mode``
^^^^^^^^^^^^
Select which physics constraints are active during training:

* ``"both"`` (default): enforce groundwater flow + consolidation
* ``"gw_flow"``: enforce groundwater flow only
* ``"consolidation"``: enforce consolidation only
* ``"none"`` / ``"off"``: disable all physics losses (ablation / debugging)

``residual_method``
^^^^^^^^^^^^^^^^^^^
Controls the consolidation integrator used to compute the physics mean
subsidence path from head:

* ``"exact"`` (default): exact-step / closed-form update (stable for larger
  time steps)
* ``"euler"``: explicit Euler update (useful for debugging / comparisons)

``use_residuals``
^^^^^^^^^^^^^^^^^
If ``True`` (default), the network can learn an additive residual around the
physics-driven subsidence mean. The residual is optionally gated by
``scaling_kwargs["allow_subs_residual"]`` during forward passes.

``scaling_kwargs``
^^^^^^^^^^^^^^^^^^
Unit handling, coordinate normalization, bounds, priors, and several
pipeline toggles are centralized in ``scaling_kwargs``. You can pass a
Python ``dict`` or a path to a JSON file. The model resolves it to a
canonical, validated dict at construction time.


A minimal instantiation looks like:

.. code-block:: python
   :linenos:

   from fusionlab.nn.pinn.geoprior.models import GeoPriorSubsNet

   model = GeoPriorSubsNet(
       static_input_dim=3,
       dynamic_input_dim=8,
       future_input_dim=4,
       forecast_horizon=3,
       max_window_size=12,
       pde_mode="both",
       quantiles=[0.1, 0.5, 0.9],   # optional
       scaling_kwargs="path/to/scaling_kwargs.json",  # or a dict
   )

Input dictionary contract
--------------------------

GeoPrior models are trained and inferred with **dict inputs**. The minimal
set used by the backbone is:

* ``static_features``: shape ``(B, S)``
* ``dynamic_features``: shape ``(B, T, D)`` (historical window)
* ``future_features``: shape ``(B, H, F)`` (known future features)
* ``coords``: shape ``(B, H, 3)`` representing ``(t, x, y)`` for PINN grads

The GeoPrior pipeline also injects additional entries used for unit
conversions and the physics closure. Depending on your configuration, you
may see keys such as:

* ``H_field`` (or ``soil_thickness``): thickness field (broadcastable to ``(B, H, 1)``)
* ``Ss_field``: storage/closure support field when needed (broadcastable to ``(B, H, 1)``)
* ``z_surf_m`` (or equivalent): surface elevation used to convert GWL to head
* initial conditions (e.g. ``subsidence_t0`` / ``head_ref``) when enabled

.. note::

   The authoritative list of optional keys is defined by the Stage-1/Stage-2
   pipeline and the resolved ``scaling_kwargs``. See :doc:`data_and_units`
   and :doc:`pipeline` for the dataset-side contract.

Outputs and tensor shapes
-------------------------

Point forecasts (``quantiles=None``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When ``quantiles`` is ``None``, each supervised output is:

* ``subs_pred``: ``(B, H, output_subsidence_dim)``
* ``gwl_pred``: ``(B, H, output_gwl_dim)``

Probabilistic forecasts (``quantiles=[...]``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When ``quantiles`` is a list of floats, outputs include a quantile axis:

* ``subs_pred``: ``(B, H, Q, output_subsidence_dim)``
* ``gwl_pred``: ``(B, H, Q, output_gwl_dim)``

where ``Q = len(quantiles)`` and the **quantile axis is axis 2**
(``q_axis=2`` in GeoPrior diagnostic trackers).

Quantile layout and extraction
------------------------------

With the default 3-quantile setup ``[0.1, 0.5, 0.9]``, a common extraction
pattern is:

.. code-block:: python
   :linenos:

   y = model(batch, training=False)      # supervised-only
   s = y["subs_pred"]                    # (B, H, Q, 1)

   q10 = s[:, :, 0, 0]
   q50 = s[:, :, 1, 0]
   q90 = s[:, :, 2, 0]

If you use a different quantile list, indices follow your provided order.

Aux forward (predictions + diagnostics)
---------------------------------------

Use :meth:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet.forward_with_aux`
when you need physics heads, intermediate tensors, or debugging signals:

.. code-block:: python
   :linenos:

   y_pred, aux = model.forward_with_aux(batch, training=False)

   # supervised outputs (same as model(batch))
   s_pred = y_pred["subs_pred"]
   h_pred = y_pred["gwl_pred"]

   # auxiliary tensors
   data_final = aux["data_final"]
   data_mean_raw = aux["data_mean_raw"]
   phys_logits = aux["phys_mean_raw"]
   phys_feat_raw_3d = aux["phys_features_raw_3d"]

Aux keys and meaning
^^^^^^^^^^^^^^^^^^^^
The auxiliary dict currently exposes:

* ``data_final``:
  final supervised tensor used to form ``subs_pred`` / ``gwl_pred``
  (includes quantiles if enabled).

* ``data_mean_raw``:
  mean-path tensor before quantile modeling
  (shape ``(B, H, output_subsidence_dim + output_gwl_dim)``).

* ``phys_mean_raw``:
  concatenated **physics logits** (usually ``[K, S_s, log(τ), Q]``),
  shape ``(B, H, P)`` with ``P = 4`` by default.

* ``phys_features_raw_3d``:
  the stacked physics-feature tensor used by the physics core; useful for
  finiteness checks and debugging preprocessing/units.

.. tip::

   The physics core consumes these auxiliary tensors to compute residual
   maps and loss terms. For details, see :doc:`core` and :doc:`losses`.

