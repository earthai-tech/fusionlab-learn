.. _user_guide_tools:

====================
Command-Line Tools
====================

Beyond the core library components, ``fusionlab`` also provides
command-line tools and applications within the ``fusionlab.tools``
subpackage. These tools aim to streamline common workflows, such as
training a standard model configuration or generating forecasts
directly from the terminal.

.. _tft_cli:

General TFT Command-Line Interface (CLI)
------------------------------------------
*(Script: ``fusionlab/tools/tft_cli.py`` or similar)*

**Purpose:**
This script provides a configurable command-line interface for
training :class:`~fusionlab.nn.transformers.TemporalFusionTransformer`
models and generating forecasts. Unlike the more specialized scripts
described previously, this tool allows users to specify datasets,
features, hyperparameters, and training options via command-line
arguments, making it adaptable to various time series forecasting tasks.

**Key Features:**

* **Argument-Driven:** Configure most aspects of the workflow via
    command-line arguments (data paths, feature lists, model settings,
    training parameters, etc.).
* **Dual Mode:** Supports both training a new model (`--mode train`) and
    generating predictions using a pre-trained model (`--mode predict`).
* **Data Handling:** Includes steps for loading data, handling missing
    values, scaling numerical features, and potentially encoding
    categorical features (depending on implementation details of helper
    functions).
* **Sequence Preparation:** Uses utilities like
    :func:`~fusionlab.utils.ts_utils.reshape_xtft_data` to transform
    input data into the sequence format required by TFT (static,
    dynamic, future, target arrays).
* **Flexible Model:** Configures the `TemporalFusionTransformer` based
    on provided hyperparameters (hidden units, attention heads, dropout,
    LSTM settings, etc.).
* **Quantile & Point Forecasts:** Supports both probabilistic (quantile)
    and deterministic (point) forecasting based on the `--quantiles`
    argument and corresponding loss function selection.
* **Standard Training:** Implements a standard Keras training loop with
    `EarlyStopping` and `ModelCheckpoint` callbacks.
* **Prediction Workflow:** Includes loading saved models and scalers,
    preparing inputs for future prediction (using logic similar to
    :func:`~fusionlab.utils.ts_utils.prepare_spatial_future_data`),
    running predictions, inverse scaling results, and saving outputs.

**Workflow Overview:**

* **Training Mode (`--mode train`):**
    1. Load training data (`--data`).
    2. Preprocess data (handle NA, scale numericals based on `--scaler`,
        potentially encode categoricals). Save fitted scalers/encoders
        to `--output_dir`.
    3. Reshape the processed data into static, dynamic, future, and
        target sequences using feature lists (`--dynamic_features`, etc.),
        `--time_steps`, and `--forecast_horizon`.
    4. Split sequences into training and validation sets
        (`--validation_split`).
    5. Build the `TemporalFusionTransformer` model using specified
        hyperparameters.
    6. Compile the model with specified optimizer, learning rate, and
        loss (MSE or quantile loss).
    7. Train the model using `.fit()`, saving the best model to
        `--output_dir`/`--model_name`.keras.

* **Prediction Mode (`--mode predict`):**
    1. Load the pre-trained model specified by `--load_model_path`.
    2. Load the pre-fitted scalers/encoders specified by
        `--load_scalers_path`.
    3. Load historical data (`--data`) required to establish context for
        prediction inputs.
    4. Preprocess the historical data using the *loaded* scalers/encoders.
    5. Prepare the specific input arrays (`X_static_pred`,
        `X_dynamic_pred`, `X_future_pred`) needed for forecasting the
        next `--forecast_horizon` steps, typically based on the last
        sequence of the processed historical data (using logic similar
        to `prepare_spatial_future_data`).
    6. Generate scaled predictions using `model.predict()`.
    7. Apply inverse scaling transformations to the predictions using the
        loaded target scaler.
    8. Format the results into a Pandas DataFrame, including identifiers
        (spatial columns, time steps/dates).
    9. Save the formatted predictions to `--predictions_output_file`.

**Usage:**
Run the script from your terminal, specifying the mode and relevant options:

.. code-block:: bash

   # Training Example
   python path/to/tft_cli.py --mode train \
      --data path/to/my_data.csv \
      --target Sales \
      --dt_col Date \
      --dynamic_features Price Promotion Temperature \
      --static_features StoreID Region \
      --future_features PlannedPromotion \
      --time_steps 12 \
      --forecast_horizon 3 \
      --epochs 50 \
      --batch_size 64 \
      --quantiles 0.1 0.5 0.9 \
      --output_dir ./my_tft_output \
      --model_name sales_tft \
      --verbose 1

   # Prediction Example
   python path/to/tft_cli.py --mode predict \
      --data path/to/my_data.csv \
      --target Sales \
      --dt_col Date \
      --dynamic_features Price Promotion Temperature \
      --static_features StoreID Region \
      --future_features PlannedPromotion \
      --time_steps 12 \
      --forecast_horizon 3 \
      --load_model_path ./my_tft_output/sales_tft_best.keras \
      --load_scalers_path ./my_tft_output/scalers.joblib \
      --quantiles 0.1 0.5 0.9 \
      --predictions_output_file ./my_tft_output/predictions.csv \
      --verbose 1

**Key Command-Line Arguments:**

*(Note: Default values are shown in parentheses)*

*General:*

* `--mode` (Required): 'train' or 'predict'.
* `--data` (Required): Path to the main dataset CSV.
* `--target` (Required): Name of the target column.
* `--dt_col` (Required): Name of the datetime column.
* `--output_dir`: Directory for saving outputs ('./tft_output').
* `--verbose`: Logging level (0-2, default: 1).
* `--seed`: Random seed (42).

*Features:*

* `--dynamic_features` (Required): List of dynamic column names.
* `--static_features`: List of static column names (None).
* `--future_features`: List of known future column names (None).
* `--categorical_features`: List of categorical feature names among all
    features (None).
* `--spatial_cols`: List of spatial identifier columns (None).

*Preprocessing:*

* `--scaler`: Scaler for numerical features ('z-norm', 'minmax', 'none',
    default: 'z-norm').
* `--handle_na`: Strategy for missing values ('drop', 'ffill', default:
    'ffill').

*Sequence Parameters:*

* `--time_steps`: Input sequence length (10).
* `--forecast_horizon`: Output prediction length (1).

*Model Hyperparameters:*

* `--hidden_units`: Hidden units for GRNs/Dense layers (32).
* `--num_heads`: Number of attention heads (4).
* `--dropout_rate`: Dropout rate (0.1).
* `--quantiles`: List of quantiles for prediction (None = point forecast).
* `--activation`: Activation function ('elu').
* `--use_batch_norm`: Flag to use Batch Normalization (False).
* `--num_lstm_layers`: Number of LSTM layers (1).
* `--lstm_units`: Units per LSTM layer (None = use hidden_units).

*Training Specific:*

* `--epochs`: Number of training epochs (50).
* `--batch_size`: Training batch size (32).
* `--learning_rate`: Optimizer learning rate (0.001).
* `--optimizer`: Optimizer name ('adam').
* `--validation_split`: Fraction for validation set (0.2).
* `--patience`: Early stopping patience (10).
* `--model_name`: Base name for saved model ('tft_model').

*Prediction Specific:*

* `--load_model_path`: Path to load a trained model (None). Required
    for predict mode.
* `--load_scalers_path`: Path to load saved scalers (None). Required
    for predict mode.
* `--predictions_output_file`: Path to save predictions (defaults to
    `<output_dir>/<model_name>_predictions.csv`). Required for predict mode.

**Dependencies:**

Ensure the following libraries are installed:
`pandas`, `numpy`, `scikit-learn`, `tensorflow` (which includes Keras),
`matplotlib` (likely used by internal helpers), `joblib`, 
(likely used by internal helpers), and `fusionlab` itself.
.. _xtft_proba_app:

XTFT Probabilistic Prediction Application
-------------------------------------------

*(Script: ``fusionlab/tools/xtft_proba_app.py``)*

**Purpose:**
This application provides a command-line interface for executing a
complete workflow to train an :class:`~fusionlab.nn.XTFT` model and
generate probabilistic (quantile) forecasts for time series data.
It handles data loading, preprocessing, sequence creation, training,
prediction, and basic visualization.

**Key Features:**

* **Data Handling:** Loads datasets (CSV recommended), performs basic
    preprocessing like handling missing values, scales numerical
    features using StandardScaler, and encodes categorical features
    using OneHotEncoder. Scalers are saved for inverse transformation.
* **Sequence Generation:** Uses internal utilities (likely related to
    :func:`~fusionlab.nn.utils.create_sequences` or
    :func:`~fusionlab.nn.utils.reshape_xtft_data`) to create input
    sequences based on specified `time_steps` and `forecast_horizon`.
* **Model Training:** Defines and compiles an XTFT model based on
    inferred input dimensions and specified `quantiles`. Trains the
    model using Adam optimizer and appropriate loss (quantile or MSE),
    incorporating `EarlyStopping` and `ModelCheckpoint` callbacks.
* **Probabilistic Prediction:** Loads the best trained model and
    generates predictions for the specified `quantiles` on validation
    or future data.
* **Inverse Scaling:** Reverses the scaling applied during
    preprocessing to present predictions in the original data scale.
* **Output & Visualization:** Saves the formatted predictions to a
    CSV file and generates basic spatial visualizations of the
    forecasts for specified future years using `matplotlib`.

**Workflow Overview:**
The script typically executes the following steps:

1.  Parses command-line arguments (`argparse`).
2.  Loads the main dataset specified by `--data`.
3.  Preprocesses the data: scales numerical features, one-hot encodes
    categorical features. Saves scalers.
4.  Defines static, dynamic, and future feature sets based on user
    inputs.
5.  Creates input sequences (`X_static`, `X_dynamic`, `X_future`) and
    target sequences (`y`) for training.
6.  Splits the sequences into training and validation sets.
7.  Builds the XTFT model architecture based on data dimensions and
    specified `forecast_horizon` and `quantiles`.
8.  Compiles the model with Adam optimizer and quantile loss (or MSE).
9.  Trains the model using `.fit()`, saving the best model based on
    validation loss (`ModelCheckpoint`).
10. Loads the best saved model weights.
11. Generates predictions on the validation set using `.predict()`.
12. Applies inverse scaling to the predictions and relevant input
    features (like coordinates) using the saved scalers.
13. Formats the predictions into a Pandas DataFrame, adding spatial
    coordinates and future time identifiers (years).
14. Saves the prediction DataFrame to a CSV file.
15. Creates and displays visualizations of the predictions for the
    specified `visualize_years`.

**Usage:**
Run the script from your terminal, providing necessary arguments:

.. code-block:: bash

   python path/to/fusionlab/tools/xtft_proba_app.py --data <DATA_PATH> --target <TARGET_COL> [OPTIONS]

**Example Command:**

.. code-block:: bash

   python path/to/fusionlab/tools/xtft_proba_app.py \
      --data /path/to/final_data.csv \
      --target subsidence \
      --categorical_features geological_category bc_category \
      --numerical_features longitude latitude year GWL soil_thickness \
      --epochs 100 \
      --batch_size 32 \
      --time_steps 4 \
      --forecast_horizon 4 \
      --quantiles 0.1 0.5 0.9 \
      --visualize_years 2024 2025 \
      --output_file my_predictions.csv

**Key Command-Line Arguments:**

* `--data` (Required): Path to the input dataset (CSV format recommended).
* `--target`: Name of the target variable column (default: 'subsidence').
* `--categorical_features` (Required): List of categorical feature column names.
* `--numerical_features` (Required): List of numerical feature column names
    (should include coordinates if used, time column, etc.).
* `--epochs`: Number of training epochs (default: 100).
* `--batch_size`: Batch size for training (default: 32).
* `--time_steps`: Lookback window size for input sequences (default: 4).
* `--forecast_horizon`: Number of future steps to predict (default: 4).
* `--quantiles`: List of quantiles for probabilistic forecast
    (default: [0.1, 0.5, 0.9]). Use this for quantile mode. If omitted
    or set carefully, might imply point forecast mode (check script logic).
* `--visualize_years`: List of future years for which to generate
    prediction plots (default: [2024, 2025, 2026]).
* `--output_file`: Name for the output CSV file containing predictions
    (default: 'xtft_quantile_predictions.csv').
* `--helpdoc`: Display the script's full docstring and exit.

**Dependencies:**
Ensure the following libraries are installed:
`pandas`, `numpy`, `scikit-learn`, `tensorflow` (which includes Keras),
`matplotlib`, `joblib`, `gofast` (appears to be an external or related
project dependency used for reading data), and `fusionlab` itself.


.. _xtft_deterministic_app:

XTFT Deterministic Prediction Application
-----------------------------------------

*(Script: ``fusionlab/tools/xtft_determinic_p.py`` or similar)*

**Purpose:**
This application provides a command-line script to train an
:class:`~fusionlab.nn.XTFT` model specifically for *deterministic*
(point) prediction. The script appears configured for a particular
use case, likely subsidence prediction from 2023-2026, using a
defined set of features and internal helper functions for the
workflow.

**Key Features/Enhancements (as noted in script docstring):**

* **Deterministic Focus:** Trains the XTFT model using Mean Squared
    Error (MSE) loss for single-value predictions per time step.
* **Configurable Verbosity:** Allows setting logging levels (DEBUG,
    INFO, WARNING) via the `--verbose` command-line flag for detailed
    monitoring or debugging.
* **Structured Workflow:** Organizes the process into logical steps
    within functions (though many helper functions like `load_data`,
    `preprocess_data`, etc., are assumed to be defined elsewhere).
* **Hardcoded Configuration:** Utilizes predefined paths, feature names
    (e.g., 'subsidence', 'GWL', 'geological_category'), sequence
    lengths, and forecast horizons within the script, making it specific
    to a particular dataset structure and forecasting goal.

**Workflow Overview:**
The script's `main` function executes the following pipeline:

1.  Sets up logging based on the `--verbose` argument.
2.  Defines a hardcoded data path.
3.  Loads data (e.g., `final_data.csv`, `final_data.bc_cat.csv`) using
    `load_data`.
4.  Performs specific preprocessing: renames columns (e.g., 'x' to
    'longitude'), handles missing values.
5.  Defines hardcoded lists of `categorical_features` and
    `numerical_features`, and sets the `target` variable ('subsidence').
6.  Applies One-Hot Encoding to categorical features and saves the encoder.
7.  Applies StandardScaler to numerical features and the target variable,
    separately handling coordinates ('longitude', 'latitude'). Saves scalers.
8.  Combines processed features into `final_processed_data`.
9.  Sorts data by year.
10. Creates sequences using `create_sequences` (or similar helper) with
    hardcoded `sequence_length=4` and `forecast_horizon=4`.
11. Splits sequences into training and validation sets using
    `train_test_split`.
12. Further splits the sequence data into static and dynamic arrays using
    `split_static_dynamic` based on derived feature indices.
13. Builds an XTFT model instance (`build_model`) with dimensions inferred
    from the data and importantly, `quantiles=None` for deterministic
    output. Other hyperparameters (embedding dim, heads, units, etc.)
    appear hardcoded.
14. Compiles the model with Adam optimizer and 'mse' loss.
15. Trains the model (`train_model`) using `.fit()` with EarlyStopping and
    ModelCheckpoint callbacks, saving the best model.
16. Loads the best saved model weights.
17. Generates point predictions on the validation set using `.predict()`.
18. Reverses scaling on predictions and coordinates using saved scalers.
19. Formats predictions into a DataFrame, adding coordinates and hardcoded
    future years (2023-2026).
20. Saves the prediction DataFrame (`save_predictions`).
21. Visualizes predictions using `visualize_predictions`.

**Usage:**
Run the script from your terminal. The primary control offered via
command line is the verbosity level.

.. code-block:: bash

   python path/to/fusionlab/tools/xtft_determinic_p.py [--verbose LEVEL]

**Example Command:**

.. code-block:: bash

   # Run with detailed DEBUG logging
   python path/to/fusionlab/tools/xtft_determinic_p.py --verbose 2

   # Run with standard INFO logging
   python path/to/fusionlab/tools/xtft_determinic_p.py --verbose 1

   # Run with minimal WARNING logging (default if flag omitted)
   python path/to/fusionlab/tools/xtft_determinic_p.py --verbose 0
   # or simply: python path/to/fusionlab/tools/xtft_determinic_p.py

**Key Command-Line Arguments:**

* `--verbose`: Sets the logging level. 0 for WARNING, 1 for INFO,
    2 for DEBUG (default: 1 based on `main` signature, but the
    `argparse` default is 1).

*(Note: Unlike the probabilistic application described earlier, this
script has most parameters like data paths, feature names, target name,
sequence lengths, horizons, and model hyperparameters defined internally.
It is less configurable via the command line and more tailored to its
specific subsidence prediction task.)*

**Dependencies:**
Ensure the following libraries are installed:
`pandas`, `numpy`, `scikit-learn`, `tensorflow` (which includes Keras),
`matplotlib`, `joblib`, and `fusionlab` itself.

