# TransFlowSubsNet: A Coupled Physics-Data Model for Land Subsidence & GWL Forecasting

This Code Ocean capsule provides the code and a demonstration workflow for
the research paper:
“Forecasting Urban Land Subsidence in the Era of Rapid Urbanization and
Climate Stress” (Revised and Resubmitted to Nature Communications)

The primary purpose of this capsule is to ensure the reproducibility of the
**Zhongshan** land subsidence and Groundwater Level (GWL) forecasting
results presented in the paper, utilizing the **Transient Flow and
Subsidence Network (TransFlowSubsNet)** model implemented in the
`fusionlab-learn` Python library.

---

## Table of Contents

1.  [The TransFlowSubsNet Model](#1-the-transflowsubsnet-model)
2.  [The fusionlab-learn Library](#2-the-fusionlab-learn-library)
3.  [Code Ocean Capsule Contents](#3-code-ocean-capsule-contents)
4.  [Setup and Execution](#4-setup-and-execution)
    * Environment
    * Data (Zhongshan Dataset)
    * Running `main_script.py`
    * Expected Outputs
5.  [Understanding the Main Workflow](#5-understanding-the-main-workflow)
6.  [Hyperparameter Tuning (Optional)](#6-hyperparameter-tuning-optional)
7.  [Short Summary of the Workflow](#7-short-summary-of-the-workflow)
8.  [Citation](#8-citation)
9.  [Support & Contact](#9-support--contact)

---

## 1. The TransFlowSubsNet Model

As detailed in our revised paper, the **Transient Flow and Subsidence
Network (TransFlowSubsNet)** is a deep learning framework designed to
forecast land subsidence and groundwater levels by integrating data-driven
insights with the fundamental, coupled physical laws governing the system.

**Abstract Summary:**

> TransFlowSubsNet builds upon a hybrid attention-based,
> data-driven architecture and significantly enhances it by incorporating
> a coupled, physics-informed neural network (PINN) framework. It
> simultaneously predicts subsidence and GWL, constrained by two
> representations of physical law: the **transient groundwater flow
> equation** and **Terzaghi's consolidation theory**. This approach
> produces physically plausible forecasts and allows for the data-driven
> estimation of a suite of effective hydrogeological parameters,
> including hydraulic conductivity ($K$), specific storage ($S_s$),
> and the consolidation coefficient ($C$). When evaluated on the
> Zhongshan land subsidence and GWL prediction task, TransFlowSubsNet
> demonstrated robust performance, physically consistent results,
> achieving a Mean Absolute Error (MAE) of 0.026, a Mean Squared Error
> (MSE) of 0.0029, and an R² score of 0.71. The probabilistic forecasts
> were well-calibrated, with a prediction interval coverage of >81.7%.

TransFlowSubsNet effectively captures complex temporal patterns using its
data-driven core, while the coupled-physics module ensures predictions
align with domain knowledge, offering a path towards more interpretable
and reliable multi-horizon forecasting in complex geophysical systems.

---

## 2. The fusionlab-learn Library

This work utilizes **`fusionlab-learn`**, a modular Python library for
building and experimenting with advanced time-series models, including
PINN-augmented architectures.

* **PyPI Package:**

    ```bash
    pip install fusionlab-learn
    ```

* **Documentation:**
    [https://fusion-lab.readthedocs.io/](https://fusion-lab.readthedocs.io/)
    

---

## 3. Code Ocean Capsule Contents

This capsule is structured for clarity and reproducibility:

* **`main_script.py`:**
    The primary Python script that executes the Zhongshan land subsidence and
    GWL forecasting workflow using `TransFlowSubsNet`. It covers data loading,
    preprocessing, PINN-specific data preparation, model training with a
    composite loss, prediction, and visualization.

* **`data/` (User Provided or via `fetch_zhongshan_data`):**
    The `main_script.py` will prioritize loading `zhongshan_500k.csv`
    from a `data/` directory at the root of the capsule. If it fails, it will
    attempt to download a 2,000-sample subset using the library's built-in
    fetcher.
    * **For Reviewers:** To reproduce the results from the paper, please
        ensure the full `zhongshan_500k.csv` dataset is uploaded to the
        `data/` folder in this capsule.

* **`results/`:**
    This directory will be created by `main_script.py` to store
    all output artifacts:
    * Preprocessed data stages (as `.csv` files).
    * Fitted data scalers (as `.joblib` files).
    * Generated sequence data for the model input (as `.joblib` files).
    * Trained `TransFlowSubsNet` model checkpoints (as `.keras` files).
    * Forecast prediction DataFrames (as `.csv` files).
    * Output plots and figures (as `.png` and `.pdf` files).

* **`README.md`:**
    This file.

* **`fusionlab/` directory (if including library source):**
    Contains the `fusionlab-learn` library modules required to run the
    script, including the `TransFlowSubsNet` model definition.

---

## 4. Setup and Execution

### Environment

* **Python:** The library requires Python 3.9 or higher.
* **Dependencies:** The primary dependencies are `tensorflow`, `pandas`,
    `scikit-learn`, `matplotlib`, and `keras-tuner`. You can install
    them directly or by installing the `fusionlab-learn` package.

    ```bash
    pip install fusionlab-learn matplotlib scikit-learn joblib tensorflow "keras-tuner>=1.4.0" pandas
    ```

### Data (Zhongshan Dataset)

The `main_script.py` is configured to find the Zhongshan data:

1.  **Primary Method:** Looks for `zhongshan_500k.csv` in the `data/`
    directory.
2.  **Fallback Method:** If the primary file isn't found, it uses
    `fusionlab.datasets.fetch_zhongshan_data()` to download a smaller,
    representative sample.

> **For Reviewers:** To use the full dataset from the paper, please
> upload it as `zhongshan_500k.csv` to the root `data/` folder.

### Running the Script

Once the environment is set up and the data is in place, execute the main
script from your terminal:

```bash
python main_script.py
```
The script will log its progress through each major stage of the workflow.


### Expected Outputs

All generated files will be saved into the
`./results/zhongshan_TransFlowSubsNet_run/` directory. Key outputs will
include:

* `zhongshan_01_raw_data.csv`
* `zhongshan_02_cleaned_data.csv`
* `zhongshan_03_processed_scaled_data.csv`
* `zhongshan_main_scaler.joblib` (and other scalers if used)
* `zhongshan_train_pinn_sequences_T{TIME_STEPS}_H{HORIZON}.joblib`
* `zhongshan_TransFlowSubsNet_H{HORIZON}.keras` (the trained model)
* `zhongshan_TransFlowSubsNet_forecast_TestSet_{YEAR_RANGE}.csv`
* Various plot files (e.g., `zhongshan_TransFlowSubsNet_plot_subsidence_spatial.png`, 
` training_history_plot.png`)

### Graphical User Interface (Subsidence PINN Mini GUI)

To maximize accessibility and ensure the workflow is easily reproducible, 
this capsule includes a standalone desktop **Subsidence PINN Mini GUI**. 
The application offers a complete, code-free environment for running the 
entire forecasting pipeline

The GUI allows users to: 

* visually load a dataset
* configure all major model and training parameters
* run the end-to-end workflow
* inspect real-time logs and generated plots

![Screenshot of the GUI showing final results and plot viewer](../data/nature_com_gui_explan.jpg)

*After a successful run, the status bar indicates completion and an interactive
plot-viewer window pops up automatically.*

**How to Launch the GUI:**

> **Prerequisite:** activate the same Conda/virtual-env used to install **fusionlab-learn**.

From your terminal, ensure your environment is activated, then run:

```bash
fusionlab-learn app launch-mini-forecaster
```

—or, if you installed the convenience entry point—

```bash
pinn-mini-forecaster
```

This will open the main application window. The GUI provides a more
user-friendly way to interact with the underlying `main_script.py` logic
and is the recommended starting point for users who prefer a visual
interface.

**Viewing Results:**

As shown in the image above, after a run is complete:
1.  The main window updates to show a **"Forecast finished"** status and
    displays the final calculated coverage score in the footer.
2.  An **interactive plot viewer** opens automatically, displaying all
    visualizations generated during the run. This viewer includes controls
    to zoom, pan, save the plot as an image, or copy it to the clipboard.

> For a complete, step-by-step walkthrough of every panel and feature
> in the application, please refer to the detailed
> **[Subsidence PINN Mini GUI Guide](https://fusion-lab.readthedocs.io/en/latest/user_guide/pinn_gui_guide.html)**
> in our full documentation.

---

## 5. Understanding the Main Workflow

The `main_script.py` executes a comprehensive, multi-stage pipeline
designed to train, evaluate, and visualize the `TransFlowSubsNet`
model. Each stage is designed to be a logical and reproducible step,
from raw data to final forecast plots.

The key stages are as follows:

1.  **Configuration**
    This initial block acts as a centralized control panel for the
    entire experiment. It sets top-level parameters like the `MODEL_NAME`,
    the forecast horizon and lookback window (`TIME_STEPS`), and the
    crucial physics-informed settings, including the `pde_mode` and the
    weights for the physics loss terms (`lambda_cons`, `lambda_gw`).

2.  **Data Loading**
    The script robustly loads the Zhongshan dataset. It is designed to
    first search for the full, large-scale dataset required for
    reproducing the paper's results. If this file is not found, it has a
    fallback mechanism to fetch a smaller, representative sample,
    ensuring the script can run in different environments.

3.  **Preprocessing**
    This stage prepares the raw data for the model through several key
    actions:
    * **Feature Selection:** It selects only the columns relevant to the
      forecasting task.
    * **Coordinate Preparation:** It creates a continuous numerical time
      coordinate from the `year` column. This is essential for the
      physics module, which requires derivatives with respect to time
      (:math:`t`).
    * **Data Cleaning:** It handles any missing values in the dataset to
      prevent errors during training.
    * **Encoding & Scaling:** It converts categorical features (like
      `geology`) into a numerical format via one-hot encoding.
      Crucially, it then scales all numerical features to a common range
      (typically 0-1) to ensure numerical stability and improve model
      convergence during training.

4.  **Feature Set Definition**
    This step explicitly maps the column names from the processed
    `DataFrame` to their roles within the `TransFlowSubsNet`
    architecture. It defines distinct lists for `static_features`,
    `dynamic_features`, and `future_features`, which are fed into the
    data-driven core of the model.

5.  **Master Data Splitting**
    A simple but critical temporal split is performed. The preprocessed
    data is divided into a training set (`df_train_master`) and a test
    set (`df_test_master`) based on the `TRAIN_END_YEAR`, ensuring the
    model is evaluated on unseen future data.

6.  **PINN Sequence Generation**
    This is where the 2D tabular data is transformed into 3D sequences
    that the model can process. The specialized
    `prepare_pinn_data_sequences` utility is used to create:
    * An `inputs` dictionary containing four key tensors:
      `'static_features'`, `'dynamic_features'`, `'future_features'`,
      and the crucial `'coords'` tensor for the physics module.
    * A `targets` dictionary containing the corresponding output sequences
      for `'subsidence'` and `'gwl'`.

7.  **`tf.data.Dataset` Creation**
    The NumPy arrays generated in the previous step are converted into
    `tf.data.Dataset` objects. This is a performance optimization that
    enables efficient batching, shuffling, and prefetching, which helps
    maximize GPU utilization during training. The training sequences are
    also split here to create a validation set.

8.  **`TransFlowSubsNet` Model Training**
    This is the core modeling stage:
    * `TransFlowSubsNet` is instantiated with the correct data dimensions
      inferred from the prepared sequences.
    * The model is compiled with its unique **composite loss function**. This
      involves defining standard data-fidelity losses (like MSE) for the
      subsidence and GWL outputs, as well as the physics-based loss
      weights, `lambda_cons` and `lambda_gw`, which control the strength
      of the physical regularization.
    * The model is trained using `.fit()`. Callbacks like `EarlyStopping`
      and `ModelCheckpoint` are used to prevent overfitting and save the
      best-performing model based on the total validation loss
      (`val_loss`).

9.  **Forecasting on Test Data**
    The best saved model from the training phase is loaded and used to
    generate predictions on the test set. The `format_pihalnet_predictions`
    utility then converts the model's raw dictionary output into a
    structured and human-readable `pandas` DataFrame, which includes the
    predictions, true values, and coordinates for easy analysis.

10. **Visualization & Artifact Saving**
    The final stage involves saving all experiment artifacts. The forecast
    DataFrame is saved to a `.csv` file. The script then uses the
    library's plotting utilities to generate and save spatial maps and
    other relevant figures visualizing the forecast results.
    
---
 
## 6. Hyperparameter Tuning (Optional)

The `main_script.py` uses a well-chosen, but fixed, set of
hyperparameters. For optimal performance on a specific dataset or for
research purposes, these parameters should be tuned. The
`fusionlab-learn` library provides the **`HydroTuner`** utility for this
purpose, which automates the search for the best model configuration.

Below is a conceptual snippet demonstrating how to set up and run a
tuning job for the `TransFlowSubsNet` model.

```python

import tensorflow as tf
from fusionlab.nn.forecast_tuner import HydroTuner
from fusionlab.nn.pinn.models import TransFlowSubsNet
# Assume data preparation steps have been run to get NumPy arrays
# from your main_script.py (e.g., train_inputs, train_targets, etc.)

# 1. Define the Hyperparameter Search Space for TransFlowSubsNet
# This dictionary tells the tuner what to optimize.
search_space = {
    # Architectural HPs from BaseAttentive
    "embed_dim": [32, 64],
    "num_heads": [2, 4],
    "dropout_rate": {"type": "float", "min_value": 0.0, "max_value": 0.3},
    
    # Physics HPs for TransFlowSubsNet
    "K": ["learnable", 1e-5, 1e-4], # Tune between learnable or fixed values
    "Ss": {"type": "float", "min_value": 1e-6, "max_value": 1e-4, "sampling": "log"},
    "pinn_coefficient_C": ["learnable"], # Test only the learnable case for C
    
    # Compile-time HPs
    "learning_rate": {"type": "choice", "values": [1e-3, 5e-4, 1e-4]},
    "lambda_gw": {"type": "float", "min_value": 0.5, "max_value": 2.0},
    "lambda_cons": {"type": "float", "min_value": 0.1, "max_value": 1.0}
}

# 2. Instantiate the Tuner using the .create() factory method
# This is the recommended approach as it infers data dimensions automatically.
tuner = HydroTuner.create(
    model_name_or_cls=TransFlowSubsNet, # Specify the model to tune
    inputs_data=train_inputs,          # Your dict of training input arrays
    targets_data=train_targets,        # Your dict of training target arrays
    search_space=search_space,
    # --- Keras Tuner Settings ---
    objective='val_loss', # Monitor total validation loss
    max_trials=50,        # Number of HP combinations to test
    project_name="Zhongshan_TransFlowSubsNet_Tuning",
    directory=os.path.join(RUN_OUTPUT_PATH, "tuning_results"),
    executions_per_trial=1,
    tuner_type='bayesian', # BayesianOptimization is often efficient
)

# 3. Run the tuning process using the high-level .run() method
best_model, best_hps, tuner_instance = tuner.run(
    inputs=train_inputs,
    y=train_targets,
    validation_data=(val_inputs, val_targets),
    epochs=100, # Max epochs per trial
    batch_size=256,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
    ],
    verbose=1
)

# 4. Analyze the results
if best_model:
    print("\n--- Best Hyperparameters Found ---")
    for hp, value in best_hps.values.items():
        print(f"  - {hp}: {value}")
    
    # The best_model is already retrained and can be saved or used for prediction
    best_model.save("best_transflow_model.keras")
    
```

---

## 7. Short Summary of the Workflow

The `main_script.py` executes a complete, end-to-end pipeline for
hybrid physics-data forecasting:

1.  **Setup & Data:** Configures parameters and loads the Zhongshan dataset.
2.  **Preprocessing:** Cleans the data, creates a numerical time coordinate,
    encodes categorical features, and scales numerical features.
3.  **PINN Data Sequencing:** Uses `prepare_pinn_data_sequences` to create
    dictionaries of inputs (including `coords` for physics) and targets
    (`subsidence`, `gwl`) required by `TransFlowSubsNet`.
4.  **Dataset Creation:** Converts the NumPy sequence dictionaries into
    `tf.data.Dataset` objects for efficient training and validation.
5.  **Model Training:** Instantiates `TransFlowSubsNet`, then compiles it
    with a composite loss function. This includes data-driven losses for
    subsidence and GWL, plus two weighted physics terms for both
    **consolidation** and **groundwater flow**, controlled by `lambda_cons`
    and `lambda_gw`. The model is then trained using its custom `train_step`.
6.  **Forecasting & Output:** Generates predictions on the test data and
    formats the results into a structured `.csv` file for analysis using
    `format_pihalnet_predictions`.
7.  **Visualization:** Creates and saves spatial and temporal plots of the
    forecast results.

This workflow demonstrates a reproducible, physics-informed approach to
forecasting coupled hydrogeological processes.

---

## 8. Citation

If you use `TransFlowSubsNet` or this coupled physics-informed approach in
your research, please cite our paper:

> Liu Rong & Laurent Kouadio (2025). Forecasting Urban Land Subsidence in the Era
> of Rapid Urbanization and Climate Stress. *Submitted to Nature
> Communications*.

**BibTeX entry (to be updated upon publication/preprint):**

```bibtex
@article{liurong2025transflowsubsnet,
  title={Forecasting Urban Land Subsidence in the Era of Rapid Urbanization and Climate Stress},
  author={ Rong Liu and Kouadio, Laurent},
  journal={Submitted to Nature Communications},
  year={2025}
}
```