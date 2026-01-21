.. _geoprior_v3_experiment_setup_tab:

======================
Experiment Setup tab
======================

The **Experiment Setup** tab is the configuration center of GeoPrior v3.
It collects all run-critical settings (paths, time window, data
semantics, model/training knobs, physics constraints, probabilistic
outputs, tuning, device/runtime) into a single, store-backed workspace.

Unlike the **Data** tab (which focuses on dataset selection and preview),
Experiment Setup focuses on **declaring intent**: *what experiment you
want to run*, *with what semantics*, and *with which defaults or
overrides*.

.. figure:: /_static/apps/geoprior_v3/02_experiment_setup_tab.png
   :alt: Experiment Setup tab (GeoPrior v3.2).
   :width: 96%

   Experiment Setup tab. A sticky header (actions + search + lock),
   a left navigation with sections, and a right scroll area with
   section cards (Summary, Paths, Time window, Data semantics, etc.).

What this tab controls
----------------------

The Experiment Setup tab is the single place where you define:

- **Project context**: city name, model name, dataset path, results root.
- **Time window & forecast**: training end year, forecast start, horizon,
  time steps, and whether to build a future NPZ.
- **Data columns & semantics**: mapping dataset columns to model roles
  (time, lon, lat, subsidence, GWL, H-field, head proxy settings, etc.).
- **Coordinates & CRS**: coordinate mode (degrees/meters), EPSG settings,
  normalization rules.
- **Architecture / training / physics / probabilistic outputs**:
  the knobs that define the Stage-2 model run.
- **Tuning and runtime**: hyperparameter tuning settings and device/
  runtime preferences.

The key idea is that these choices are saved as a reproducible config
snapshot and are reused by downstream tabs (**Preprocess**, **Train**,
**Tune**, **Inference**, **Transfer**, **Results**, **Map**).

Layout overview
---------------

The tab is composed of three parts:

1) Sticky header (actions + search + lock)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the top you will see a compact header with:

- **Load**: import a configuration snapshot (JSON) and patch it into the
  current configuration.
- **Save**: save the current config snapshot to the last used JSON path.
- **Apply**: broadcast the current config to the rest of the GUI so other
  tabs refresh immediately (useful after bulk edits).
- **Diff**: view the current override diff (what changed relative to the
  baseline/default config).
- **Reset**: restore defaults (use with care).
- **Overrides counter**: the pill showing how many keys differ from the
  baseline (example: ``162 overrides``).
- **Lock**: toggle *read-only* mode for the entire setup panel.
- **Search** + **More**: filter sections/cards by text and access extra
  actions (export/copy helpers, depending on build).

This header is designed to make configuration management explicit:
you can quickly see whether the setup is “clean”, how many overrides are
active, and whether editing is currently locked.

.. note::

   **Load/Save** here refer to configuration snapshots (JSON). They do
   not load datasets. Datasets are loaded in the **Data** tab.

2) Left navigation (sections)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The left panel lists all setup sections (for example: Summary, Project &
paths, Time window & forecast, Data columns & semantics, Coordinates &
CRS, Training basics, Physics & constraints, Probabilistic outputs,
Tuning, Device & runtime, UI preferences).

Each item includes a short description, and selecting it scrolls the
right-hand side to the corresponding card. This is intentionally
“manual” and predictable: you always know which card you are editing.

3) Right scroll area (cards)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main workspace is a scrollable stack of **cards**, one per section.
Cards are presented in the same order as the navigation list.

Each card typically contains:

- a title (matching the section),
- an optional status pill (e.g., **Missing paths**, **Unlocked**),
- grouped widgets with labels and tooltips,
- and immediate feedback (validation state, detected dataset columns,
  etc., depending on the card).

Config store and overrides
--------------------------

GeoPrior v3 uses a single configuration store as the source of truth.
Every widget change patches one or more keys in that store.

Two concepts are visible in the UI:

- **Snapshot**: the full configuration as a JSON-friendly dict.
- **Overrides (diff)**: only the keys that differ from the baseline
  defaults.

That’s why you see an “overrides count” pill in the header. It is a
lightweight audit indicator: a run with many overrides is not “bad”, but
it reminds you the configuration is far from defaults and should be
saved as a snapshot for reproducibility.

.. tip::

   Use **Diff** before long training or tuning runs. It is the fastest
   way to verify you did not accidentally change something critical
   (years, horizon, semantics, physics weights, etc.).

Locking the setup (prevent accidental edits)
--------------------------------------------

The **Lock** button turns the setup panel into a read-only view. This is
useful when you are:

- inspecting an existing run setup,
- browsing multiple sections without wanting to change values,
- presenting a configuration (screenshots/demos),
- or using the tab as an “audit dashboard”.

When locked, widgets are disabled and the header shows the lock state
(e.g., **Unlocked** vs locked). Unlock when you are ready to edit again.

Recommended workflow
--------------------

A minimal but reliable workflow is:

1. In **Data**, select or import your dataset.
2. In **Experiment Setup**, verify the three essentials:

   - **Project & paths**: city, dataset path, results root.
   - **Time window & forecast**: train end, forecast start, horizon, time
     steps (and future NPZ toggle).
   - **Data columns & semantics**: confirm the correct time/lon/lat/target
     columns and GWL semantics.

3. Click **Apply** to refresh dependent tabs immediately.
4. Continue to **Preprocess** (Stage-1), then **Train/Tune/Inference**.

If you change any of the essentials after running Stage-1, re-run Stage-1
to keep artifacts consistent.

.. warning::

   Changing time window, horizon, or column semantics after Stage-1 can
   invalidate previously generated tensors/manifests. If you are unsure,
   rerun Stage-1 in the **Preprocess** tab.

Key cards you will use most
---------------------------

Summary
^^^^^^^

The **Summary** card provides a compact, read-only preview of your
current experiment setup: city, model, mode, PDE mode, dataset path,
results root, and the most important time/training settings. It is meant
to answer, at a glance: “What run am I about to launch?”

Project & paths
^^^^^^^^^^^^^^^

This card defines where inputs come from and where outputs go.

Typical controls include:

- **City**: used to label runs and organize outputs.
- **Dataset path**: canonical dataset path used by the GUI.
- **Results root**: root folder for all run artifacts and exports.
- **Stage-1 reuse / rebuild policies** (advanced): options that control
  whether Stage-1 preprocessing can be reused automatically when settings
  match, and whether mismatches force a rebuild.
- **Audit stages** (advanced): select which stages are audited.

If the dataset path is missing or invalid, the card can show a status
pill such as **Missing paths** until the paths are configured.

Time window & forecast
^^^^^^^^^^^^^^^^^^^^^^

This card defines the temporal scope of the run:

- **Train end year**: last observed year included in training history.
- **Forecast start year**: first year predicted by inference.
- **Horizon (years)**: number of years to predict beyond forecast start.
- **Time steps**: sequence length / lookback used by the model.
- **Build future NPZ**: whether Stage-1 should create a future-known NPZ
  for forecasting mode.

Data columns & semantics
^^^^^^^^^^^^^^^^^^^^^^^^

This card maps dataset columns to GeoPrior’s expected semantics. The
drop-downs are populated from the active dataset columns (provided by
the Data tab). Most fields are also **editable**, so you can type a
column name manually when needed.

Typical fields include:

- time column, lon/lat columns,
- subsidence target column,
- GWL (groundwater level) column + its *kind* and *sign conventions*,
- optional H-field and z_surf columns,
- head proxy options when deriving head from depth-to-water + elevation.

Coordinates & CRS
^^^^^^^^^^^^^^^^^

This card controls how coordinates are interpreted and normalized:

- coordinate mode (degrees vs meters),
- source EPSG and UTM EPSG settings (when projection is required),
- optional normalization/shift rules.

Other sections
^^^^^^^^^^^^^^

Other cards (architecture, training, physics, probabilistic outputs,
tuning, device/runtime, UI preferences) are designed to be edited when
you move beyond a minimal run. In early experiments, you can keep these
at defaults and focus on paths + time window + semantics.

Search and section filtering
----------------------------

Use the search box in the header to filter sections. Filtering hides
non-matching cards and updates the navigation to keep only visible
sections. This is the fastest way to jump to a control when you already
know its category (e.g., type “epsg”, “horizon”, “gwl”, “lambda”, “heads”,
“quantiles”).

Saving your setup (recommended)
-------------------------------

Once your run configuration is stable, save a snapshot:

- Click **Save as** (or **Save** if a path is already set),
- store the JSON alongside your results archive or in a separate
  experiment registry folder.

Saved snapshots make it easy to reproduce experiments, compare runs, and
audit exactly what changed.

See also
--------

- :doc:`preprocess_tab` for Stage-1 preprocessing
- :doc:`train_tab` and :doc:`inference_tab` for Stage-2 runs
- :doc:`../reference/file_layout_outputs` for output directory layout
- :doc:`../reference/configuration_keys` for a key-level reference
