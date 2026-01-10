.. _geoprior_pipeline:

==============================
Pipeline (Stage-1 → Stage-2)
==============================

This project is organized as a **two-stage** pipeline:

- **Stage-1 (`stage1.py`)**: data cleaning, feature selection, scaling/encoding,
  sequence building, and export of ready-to-train tensors (NPZ) + metadata.
- **Stage-2 (`stage2.py`)**: load Stage-1 exports, finalize scaling + unit
  conventions, run handshake audits, train GeoPriorSubsNet, optionally calibrate
  prediction intervals, and export forecasts.

The two stages communicate via:
- exported tensors (NPZ),
- exported scalers/encoders (joblib),
- and a compact unit/scaling contract: ``scaling_kwargs.json``.


Pipeline at a glance
----------------------

1. **Stage-1** builds and exports:
   - input tensors: ``coords``, ``dynamic_features``, ``static_features``,
     ``future_features``, ``H_field``.
   - target tensors: ``subs_pred`` (subsidence), ``gwl_pred`` (groundwater/head).
   - coordinate scaling metadata (``coord_ranges``) required for chain-rule
     rescaling of PDE derivatives in Stage-2.
   - Stage-1 scaling audit (sanity checks + provenance).

2. **Stage-2** loads Stage-1 exports and:
   - finalizes SI affine maps and unit settings (writes ``scaling_kwargs.json``),
   - runs Stage-2 “handshake” audit (shape + scaling invariants),
   - trains a model and saves a bundle (model + weights + manifest),
   - formats and exports forecast CSVs (calibrated + future grids).


Directory layout
------------------

Both scripts write into a city/model-specific run directory, typically under
``results/`` (configurable via ``BASE_OUTPUT_DIR``).

A typical run structure looks like:

.. code-block:: text

   results/
     <CITY>_<MODEL>_stage1/
       train_<timestamp>/
         artifacts/
         stage1_scaling_audit.json
         ... (npz, joblib, manifests)
     <CITY>_<MODEL>_stage2/
       train_<timestamp>/
         scaling_kwargs.json
         stage2_handshake_audit.json
         ... (model bundle, logs)
         <CITY>_<MODEL>_forecast_<DATASET>_H<H>_calibrated.csv
         <CITY>_<MODEL>_forecast_<DATASET>_H<H>_future.csv


Stage-1: preprocessing & sequence export
------------------------------------------

What Stage-1 does
~~~~~~~~~~~~~~~~~~

Stage-1 executes (conceptually):

1. Load dataset and apply basic cleanup.
2. Select/assemble dynamic, static, and future-known feature blocks.
3. Encode and scale ML features.
4. Build PINN/TFT-like sequences (depending on ``MODE``).
5. Export NumPy tensors for training/validation (NPZ).
6. Save scalers and write audits (optional but recommended).

Exported tensors (NPZ contract)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage-1 exports the following **input keys**:

- ``coords``:
  a (N, T, 3) tensor where the last dimension is ordered as
  ``(t, x, y)``. Coordinates may be normalized to [0, 1] if
  ``NORMALIZE_COORDS=True``.
- ``dynamic_features``:
  (N, T, D_dyn) tensor of time-varying drivers.
- ``static_features``:
  (N, D_static) tensor (may be empty if no static covariates).
- ``future_features``:
  (N, T_future, D_future) tensor (may be empty if no future-known features).
- ``H_field``:
  (N, T, ...) tensor (effective thickness field used by the physics loss).

Stage-1 exports the following **target keys**:

- ``subs_pred``: subsidence target tensor
- ``gwl_pred``: groundwater/head target tensor

Stage-1 also fills missing optional blocks with **empty but shape-consistent**
arrays (e.g., static with width 0, future with width 0) so downstream code does
not need special-casing.

Coordinate handling and chain-rule metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Stage-1 normalizes coords, the model sees normalized coordinates (typically
in [0, 1]). However, PDE residuals must still be correctly scaled back to
physical units.

For this reason, Stage-1 computes and records:

- ``coords_normalized`` (bool)
- ``coord_order`` = ``["t", "x", "y"]``
- ``coord_ranges`` = span of raw coords for (t, x, y)

These spans are later used in Stage-2 to rescale derivatives via the chain rule.

Stage-1 scaling audit
~~~~~~~~~~~~~~~~~~~~~~

When enabled, Stage-1 writes a scaling audit JSON containing:

- provenance (coord mode, EPSG used, whether normalization/shift was enabled),
- raw coord stats (heuristics like “UTM-like”),
- model coord stats (min/max/mean/std in model space),
- coord scaler min/max and recorded chain-rule ranges,
- physics column stats for sanity checking.


Stage-1 unit conventions (high level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage-1 is responsible for producing **SI-consistent physics columns** that
the model will train against (or recording the affine maps if non-SI is used).

In the current configuration:

- subsidence affine map is identity (``subs_scale_si=1, subs_bias_si=0``),
- head affine map is identity (``head_scale_si=1, head_bias_si=0``),
- thickness affine map is identity (``H_scale_si=1, H_bias_si=0``).

Stage-1 also encodes the groundwater semantics split:

- **driver** is groundwater level as **depth below ground surface** (down-positive)
- **target** is groundwater **head** (up-positive)

(See Stage-2 section for how these semantics are recorded and validated.)


Stage-2: training, calibration, forecasting
-------------------------------------------

What Stage-2 does
~~~~~~~~~~~~~~~~~

Stage-2 performs:

1. Load Stage-1 exports (NPZ + scalers + manifests).
2. Build ``subsmodel_params`` and finalize ``scaling_kwargs``.
3. Write the final unit/scaling contract to ``scaling_kwargs.json``.
4. Run Stage-2 handshake audit (recommended).
5. Build, compile, and train the model.
6. Save model artifacts (format depends on Keras/TensorFlow settings).
7. Optionally calibrate prediction intervals (quantile mode).
8. Export forecast CSVs (calibrated + future grid).

Finalizing scaling + units (``scaling_kwargs.json``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage-2 writes a single, canonical “contract” file:

- ``scaling_kwargs.json``

This file includes:

- SI affine maps (subsidence/head/thickness),
- coordinate normalization metadata (coord order + coord ranges),
- PDE residual units and floors,
- Q-forcing conventions (e.g., ``Q_kind``),
- indices that map physics-relevant channels inside tensors.

A typical example includes:

- ``time_units = "year"``
- ``cons_residual_units = "second"``
- ``gw_residual_units = "second"``
- ``Q_kind = "per_volume"``
- ``gwl_dyn_index``, ``subs_dyn_index``, ``z_surf_static_index`` (see below)

The “indices, not names” rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage-2 (and the model) should not rely on a feature name to locate physics
channels. Instead, Stage-1/2 provide indices:

- ``gwl_dyn_index``: where groundwater *driver* lives in ``dynamic_features``
- ``subs_dyn_index``: where subsidence driver/feedback lives in ``dynamic_features``
- ``z_surf_static_index``: where surface elevation lives in ``static_features``

These indices prevent silent mistakes when feature sets are reordered.

Groundwater semantics + head-from-depth rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage-2 stores a groundwater metadata block in ``scaling_kwargs`` (``gwl_z_meta``)
so the model knows:

- what the raw config *meant* (kind/sign),
- what Stage-1 *produced* (driver vs target semantics),
- whether it is using a head proxy,
- and whether ``z_surf`` is available to compute head from depth.

In words:

- If a surface elevation column is available, head can be computed as:

  ``head = z_surf - depth``

- If not, Stage-2 may fall back to a proxy:

  ``head ≈ -depth``

Stage-2 handshake audit (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When enabled, Stage-2 writes ``stage2_handshake_audit.json`` which checks:

- tensor shapes are consistent with ``TIME_STEPS`` and ``FORECAST_HORIZON``,
- no NaNs/Infs in training/validation tensors,
- ``coord_ranges`` matches the coord scaler span (relative error checks),
- scaling contract is internally consistent (e.g., coords normalized implies
  coord ranges exist).

This audit is designed to catch the most common silent failure modes early.

Common failure mode: missing coord_ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Stage-1 used normalized coords but did not record ranges, Stage-2 attempts to
infer them from the coord scaler. If it still cannot infer, it raises a hard
error:

``coords_normalized=True but coord_ranges missing and cannot infer ...``

Forecast exports (CSV)
~~~~~~~~~~~~~~~~~~~~~~

Stage-2 writes two main forecast outputs for a chosen dataset split:

- ``<CITY>_<MODEL>_forecast_<DATASET>_H<H>_calibrated.csv``
- ``<CITY>_<MODEL>_forecast_<DATASET>_H<H>_future.csv``

The calibrated file contains evaluation rows (with actuals where available) and
quantile predictions when quantiles are enabled. The future file contains the
future time grid in physical time units (e.g., years) and the predicted forecast
trajectories.

Minimal run recipe
------------------

From your environment (with config set appropriately):

.. code-block:: bash

   python stage1.py
   python stage2.py

Stage-2 expects the Stage-1 run artifacts to exist and match the active config
(city/model/mode/time steps/horizon).

See also
--------

- :ref:`data_and_units` (units and sign conventions)
- :ref:`quickstart` (end-to-end run examples)
