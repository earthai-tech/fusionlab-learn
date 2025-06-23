.. _user_guide_tools:

==============================
Command-Line Interface (CLI)
==============================

``fusionlab-learn`` includes a powerful and extensible Command-Line
Interface (CLI) for executing common workflows directly from your
terminal. This allows you to run forecasting pipelines, data
processing tasks, and launch applications without writing any
Python code.

The main entry point for all tools is the ``fusionlab-learn``
command.

Getting Help
------------
You can get a list of all available commands and their descriptions by
running:

.. code-block:: bash

   fusionlab-learn --help

To get help for a specific command group (e.g., `forecast`) or a
sub-command, you can use:

.. code-block:: bash

   # Help for a command group
   fusionlab-learn forecast --help

   # Help for a specific sub-command
   fusionlab-learn forecast xtft-proba --help

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Command Groups
==============

The CLI is organized into several logical groups to make finding the
right tool intuitive and easy.

.. contents::
   :local:
   :depth: 1

Forecasting Tools (`forecast`)
------------------------------
This command group contains high-level tools for running end-to-end
forecasting workflows. Each command encapsulates a full pipeline, from
data loading and preprocessing to model training, prediction, and
visualization.

**`xtft-proba`**
****************
This command trains and runs an
:class:`~fusionlab.nn.models.XTFT` model to generate **probabilistic
(quantile)** forecasts. It provides a comprehensive set of options to
control the entire workflow.

**Usage:**

.. code-block:: bash

   fusionlab-learn forecast xtft-proba --data-path <PATH> --target <NAME> [OPTIONS]

**Key Options:**

* ``--data-path`` (Required): The path to the directory containing
  your input CSV files.
* ``--target`` (Required): The name of the target variable you wish
  to predict.
* ``--cat-features``, ``--num-features``: Comma-separated lists of
  your categorical and numerical feature column names.
* ``--epochs``, ``--batch-size``, ``--time-steps``, ``--horizon``:
  Standard parameters to control the training process and sequence
  generation.
* ``--quantiles``: A comma-separated list of quantiles for the
  probabilistic forecast (e.g., `"0.1,0.5,0.9"`).

**`xtft-point`**
****************
This command is similar to `xtft-proba` but is specifically for
generating **deterministic (point)** forecasts. It trains an `XTFT`
model using a standard regression loss like Mean Squared Error.

**Usage:**

.. code-block:: bash

   fusionlab-learn forecast xtft-point --data-path <PATH> [OPTIONS]

**Key Options:**

* ``--data-path`` (Required): The path to the data directory.
* ``--epochs``, ``--batch-size``, ``--time-steps``, ``--horizon``:
  Parameters to control the training and forecasting dimensions.
    
.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">
   
Data Processing Tools (`process`)
---------------------------------
This group provides powerful utilities for inspecting and transforming
your forecast data files. These tools are designed to help you
automate common post-processing tasks, such as reshaping data for
analysis or scripting workflows based on the data format.

**`pivot-forecast`**
********************
Converts a **long-format** forecast DataFrame, where each row is a
single time step for a specific entity, into a **wide-format**
DataFrame, where each entity has a single row and the time steps are
spread across the columns. This is essential for creating summary
tables or preparing data for specific analysis tools that require a
wide data structure.

**Usage:**

.. code-block:: bash

   fusionlab-learn process pivot-forecast -i <INPUT> -o <OUTPUT> --id-vars <COLS> --time-col <COL> --prefixes <PREFIXES>

**Key Options:**

* ``-i, --input-file`` (Required): The path to the input CSV file that
  is in a long format.
* ``-o, --output-file`` (Required): The path where the new, wide-format
  CSV file will be saved.
* ``--id-vars`` (Required): A comma-separated list of columns that
  uniquely identify each time series (e.g., `"sample_idx,longitude"`).
  These columns will be preserved as the index of the new wide table.
* ``--time-col`` (Required): The name of the column that contains the
  temporal information (e.g., `"year"` or `"forecast_step"`).
* ``--prefixes`` (Required): A comma-separated list of the base names
  of the value columns to be pivoted (e.g., `"subsidence,GWL"`).
* ``--static-cols``: An optional comma-separated list of columns that
  contain static "ground truth" values. These columns will be merged
  back into the final wide DataFrame without being pivoted.

**`format-forecast`**
*********************
This is a smart wrapper around the `pivot-forecast` command. It
automatically detects if the input file is already in a wide format.
If it is, the command does nothing. If it detects a long format, it
will automatically run the pivot operation. This is useful for
ensuring a dataset conforms to the wide format without causing errors
if it's already been processed.

**Usage:**

.. code-block:: bash

   fusionlab-learn process format-forecast -i <INPUT_CSV> -o <OUTPUT_CSV> ...

**`detect-forecast-type`**
**************************
A useful utility for scripting that inspects the columns of a CSV file
and reports whether it contains a ``'quantile'`` or a ``'deterministic'``
forecast. This allows automated pipelines to make decisions based on
the type of data they are processing.

**Usage:**

.. code-block:: bash

   fusionlab-learn process detect-forecast-type -i <INPUT_CSV>

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Geospatial Utilities (`geotools`)
---------------------------------
This group contains tools specifically designed for preparing and
augmenting geospatial time series data before it is used for training.

**`generate-dummy`**
********************
Creates a synthetic dataset with a structure suitable for the library's
PINN models. This is invaluable for quick tests, creating reproducible
examples, or prototyping a workflow without needing a real dataset.

**Usage:**

.. code-block:: bash

   fusionlab-learn geotools generate-dummy -o dummy_data.csv --n-samples 5000

**`sample`**
************
Performs **stratified spatial sampling** on a large dataset. This is
more intelligent than a simple random sample because it ensures that the
smaller, output dataset is a representative microcosm of the original,
preserving the spatial and categorical distribution of the data.

**Usage:**

.. code-block:: bash

   fusionlab-learn geotools sample -i <INPUT> -o <OUTPUT> --sample-size 0.1 --stratify-by "year,category"

**`augment`**
*************
Applies advanced data augmentation techniques to a spatiotemporal
dataset to increase the size and diversity of the training data, which
can lead to more robust models. It can operate in two modes:
``interpolate`` (fills in missing time steps) and ``augment_features``
(adds random noise to feature columns), or ``both``.

**Usage:**

.. code-block:: bash

   fusionlab-learn geotools augment -i data.csv -o aug.csv --mode both --group-by "lon,lat" --time-col "date"

**Key Options:**

* ``--interp-kwargs`` & ``--augment-kwargs``: These  options
  allow you to pass a JSON string to control the behavior of the
  underlying functions. For example:
  `--interp-kwargs '{"freq": "D", "method": "linear"}'`

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Hyperparameter Tuning (`tune`)
------------------------------
This command group contains tools for running automated hyperparameter
tuning sessions for the various models in the library.

**`legacy-pihalnet`**
*********************
Launches the hyperparameter tuning workflow for the original, legacy
``PIHALNet`` model. This script is designed as a self-contained
workflow with its own internal data loading and processing logic,
making it a quick way to run a predefined tuning experiment.

**Usage:**

.. code-block:: bash

   fusionlab-learn tune legacy-pihalnet

.. raw:: html

   <hr style="margin-top: 1.5em; margin-bottom: 1.5em;">

Mini-GUI Applications (`app`)
---------------------------------
This group contains commands for launching graphical user interfaces,
providing a user-friendly way to interact with the library's
workflows without writing code.

**`launch-forecaster`**
***********************
Launches the **Subsidence PINN Mini GUI**. This desktop application
provides a complete, interactive interface for loading data,
configuring all model and training parameters, running an end-to-end
forecasting pipeline, and visualizing the results.

**Usage:**

.. code-block:: bash

   fusionlab-learn app launch-mini-forecaster

.. seealso::
   For a complete walkthrough of all the features and panels in the
   application, please refer to the detailed :doc:`/user_guide/pinn_gui_guide`.