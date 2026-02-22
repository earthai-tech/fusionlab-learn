.. _geoprior_tuner:

=========================================================
GeoPriorTuner: hyperparameter tuning for GeoPriorSubsNet
=========================================================

:API Reference: :class:`~fusionlab.nn.forecast_tuner.GeoPriorTuner`

``GeoPriorTuner`` is the dedicated hyperparameter tuner for
:class:`~fusionlab.nn.pinn.geoprior.models.GeoPriorSubsNet`. It is designed to
search *architecture*, *training*, and *physics-weighting* knobs while keeping
GeoPrior’s **dict I/O contract** and **physics configuration** consistent.

Under the hood, it builds a ``GeoPriorSubsNet`` from a merged parameter set
(``fixed_params`` + sampled hyperparameters), compiles it with appropriate
losses/metrics (MSE/MAE or weighted pinball when quantiles are enabled), and
runs a ``keras-tuner`` search. 


Why a specialized tuner?
--------------------------

GeoPrior is not a plain “forecast head” network: training uses a hybrid
objective that combines supervised losses and physics regularization (PDE/ODE),
and the forward pass relies on a consistent set of unit/scaling flags
(``scaling_kwargs``). The tuner therefore needs to:

- validate *dict inputs* (including mandatory physics keys),
- keep the model constructor parameters and compile-time parameters separate,
- remain compatible with GeoPrior’s quantile output layout,
- preserve reproducibility and artifact paths across runs. 


Quickstart
------------

The tuner is usually created from in-memory arrays exported by Stage-1
(``*.npz`` or ``*.npy``) and then run with a search space specification.

.. code-block:: python
   :linenos:

   from fusionlab.nn.forecast_tuner import GeoPriorTuner

   # inputs_data: dict of numpy arrays (or tensors) keyed by GeoPrior input names
   # targets_data: dict of numpy arrays keyed by "subs_pred" and "gwl_pred"
   tuner = GeoPriorTuner.create(
       inputs_data=inputs_data,
       targets_data=targets_data,
       search_space=search_space,
       fixed_params=fixed_params,  # optional
       directory="geoprior_tuner_results",
       project_name="nansha_geoprior_h3",
       overwrite=True,
   )

   best_model, best_hps, tuner_obj = tuner.run(
       batch_size=256,
       epochs=30,
       patience=5,
       verbose=1,
   )

``run(...)`` returns the best Keras model, the best hyperparameter set, and the
tuner object itself. 


Input and target contracts
----------------------------

Inputs are **dicts** (not tuples)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GeoPrior models are trained and tuned with **dict inputs**. At minimum the
tuner expects the same core keys required by the model backbone *plus* the
physics coordinate tensor:

- ``"dynamic_features"`` (historical window)
- ``"coords"`` (PINN coordinate tensor, typically ``(t, x, y)``)
- thickness-like ``"H_field"`` (required for the consolidation closure)

The tuner includes a helper that enforces the presence of ``H_field`` (or
promotes a dataset-specific thickness key into that canonical name). 


Targets are a dict with canonical keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The supervised targets must be provided as a dict with keys matching GeoPrior’s
supervised outputs:

- ``"subs_pred"``
- ``"gwl_pred"``

For convenience, the factory may accept common aliases (e.g. ``"subsidence"``,
``"gwl"``) and normalizes them into the canonical names before search. 


Fixed parameters vs search space
----------------------------------

Two parameter families
^^^^^^^^^^^^^^^^^^^^^^^^

``GeoPriorTuner`` splits configuration into:

- ``fixed_params``: anything you want to hold constant (model dimensions, physics
  mode, scaling config, etc.).
- ``search_space``: hyperparameters to explore (learning rate, hidden sizes,
  physics multipliers, etc.).

The tuner also infers a few required dimensions from the provided arrays (when
not explicitly included) to avoid “dim mismatch” boilerplate. 


Compile-only hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some hyperparameters **do not** belong to the model constructor: they control
``model.compile(...)`` and the training objective. ``GeoPriorTuner`` treats
these as “compile-only” and does not pass them into the constructor, e.g.:

- ``learning_rate``
- physics weights: ``lambda_gw``, ``lambda_cons``, ``lambda_prior``,
  ``lambda_smooth``, ``lambda_bounds``, ``lambda_mv_prior``
- optional forcing regularization: ``lambda_q``

This prevents accidental constructor errors and makes tuning reproducible. 


Losses and metrics chosen by the tuner
--------------------------------------

Point forecasts (``quantiles=None``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If quantiles are disabled, the tuner compiles the model with MSE losses for both
outputs and MAE metrics. (Your existing Stage-2 loss/metric plumbing still
applies; the tuner only selects the supervised loss/metrics family.)


Probabilistic forecasts (``quantiles=[...]``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If quantiles are enabled, the tuner switches to a **weighted pinball loss**
factory to match the quantile layout and avoid shape pitfalls. In this mode,
metrics are typically minimized (or disabled) to keep the compile graph clean
during search, and diagnostics are handled by GeoPrior’s internal trackers.


Search-space specification
----------------------------

``search_space`` is passed through the common tuner utilities
(``PINNTunerBase``). The recommended style is a list of parameter specs, e.g.:

.. code-block:: python
   :linenos:

   search_space = [
       # architecture
       {"name": "embed_dim", "type": "choice", "values": [32, 64, 96]},
       {"name": "hidden_units", "type": "choice", "values": [64, 128, 256]},
       {"name": "dropout_rate", "type": "float", "min": 0.0, "max": 0.3, "step": 0.05},

       # optimizer / compile-only
       {"name": "learning_rate", "type": "float", "min": 1e-4, "max": 5e-3, "sampling": "log"},

       # physics weighting (compile-only)
       {"name": "lambda_gw", "type": "float", "min": 0.1, "max": 5.0, "sampling": "log"},
       {"name": "lambda_cons", "type": "float", "min": 0.1, "max": 5.0, "sampling": "log"},
   ]

The *exact* supported schema depends on your shared tuner helpers; keep the
search space small at first, then expand to physics knobs only after the model
stably trains. 


Recommended tuning strategy
-----------------------------

A practical order that reduces failure rate:

1) Tune backbone capacity (``embed_dim``, ``hidden_units``, attention/LSTM sizes)
   with physics enabled but conservative weights.
2) Tune optimization (``learning_rate``, batch size, dropout).
3) Only then tune physics weights (``lambda_gw``, ``lambda_cons``, priors/bounds).

This mirrors how PINNs tend to train: get the supervised path stable first, then
let physics regularize. 


How Stage-3 uses the tuner (script-level integration)
-------------------------------------------------------

The pipeline tuning script (``stage3.py``) loads Stage-1 exports (arrays +
``scaling_kwargs``), instantiates the tuner via ``GeoPriorTuner.create(...)``,
runs the search, then saves:

- the best hyperparameters,
- the best trained model,
- and a manifest entry for reproducible inference. 


