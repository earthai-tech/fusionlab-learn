.. _example_cli_usage:

==========================
Using Command-Line Tools
==========================

``fusionlab`` provides convenient command-line interface (CLI) tools,
typically located in the ``fusionlab/tools/`` directory of the
installation. These tools allow you to run standard training and
prediction workflows directly from your terminal without writing custom
Python scripts, making common tasks faster and more reproducible.

This page provides examples based on the tools described in the
:doc:`../tools` section. Remember to replace placeholder paths like
`path/to/fusionlab/tools/` with the actual location in your
environment.

Using the General TFT CLI (`tft_cli.py`)
----------------------------------------

This is the recommended tool for general-purpose training and prediction
with the :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
due to its configurability.

**Example: Training a Quantile TFT Model**

This command trains a TFT model for quantile forecasting, specifying
data paths, features, sequence parameters, training settings, and
output locations.

.. code-block:: bash
   :linenos:

   python path/to/fusionlab/tools/tft_cli.py \
      --mode train \
      --data path/to/your_training_data.csv \
      --target Sales \
      --dt_col Date \
      --dynamic_features Price Promotion Temperature \
      --static_features StoreID Region \
      --future_features PlannedHoliday \
      --time_steps 24 \
      --forecast_horizon 12 \
      --quantiles 0.1 0.5 0.9 \
      --epochs 75 \
      --batch_size 64 \
      --learning_rate 0.002 \
      --hidden_units 64 \
      --num_heads 4 \
      --patience 15 \
      --output_dir ./tft_training_run1 \
      --model_name my_sales_tft \
      --verbose 1

**Example: Generating Predictions with the Trained TFT Model**

This command uses the model and scalers saved during the training run
to generate forecasts. It loads the necessary artifacts and historical
data to prepare prediction inputs.

.. code-block:: bash
   :linenos:

   python path/to/fusionlab/tools/tft_cli.py \
      --mode predict \
      --data path/to/your_LATEST_historical_data.csv \
      --target Sales \
      --dt_col Date \
      --dynamic_features Price Promotion Temperature \
      --static_features StoreID Region \
      --future_features PlannedHoliday \
      --time_steps 24 \
      --forecast_horizon 12 \
      --load_model_path ./tft_training_run1/my_sales_tft_best.keras \
      --load_scalers_path ./tft_training_run1/scalers.joblib \
      --quantiles 0.1 0.5 0.9 \
      --predictions_output_file ./tft_training_run1/forecast_output.csv \
      --verbose 1


Running Specific Application Scripts (Example)
----------------------------------------------

These scripts might be less configurable but demonstrate specific,
potentially pre-configured, workflows.

**Example: Running the Deterministic XTFT Application**

This script (as documented previously) has most parameters hardcoded
and mainly accepts a verbosity level.

.. code-block:: bash
   :linenos:

   # Run with detailed DEBUG logging
   python path/to/fusionlab/tools/xtft_determinic_p.py --verbose 2

   # Run with standard INFO logging
   python path/to/fusionlab/tools/xtft_determinic_p.py --verbose 1


.. topic:: Why Use the Command-Line Tools?

   * **Convenience:** Quickly run standard training or prediction tasks
     without writing Python code for common configurations.
   * **Reproducibility:** Command-line calls with specific arguments
     can be easily logged and rerun, ensuring reproducible results.
   * **Scripting:** Integrate ``fusionlab`` workflows into shell scripts
     or automated batch processing jobs.
   * **Accessibility:** Allows users less familiar with the intricacies
     of the Python API to leverage the core models for standard tasks.

   However, for highly customized data pipelines, non-standard model
   architectures, complex evaluation procedures, or fine-grained control
   over the process, using the ``fusionlab`` library components directly
   within your own Python scripts offers greater flexibility. The CLI tools
   are designed for the most common use cases.