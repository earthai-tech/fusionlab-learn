# PIHALNet: Physics-Informed Land Subsidence & GWL Forecasting for Nature Communications Reproducibility

This Code Ocean capsule provides the code and a demonstration workflow for
the research paper:
“Forecasting Urban Land Subsidence in the Era of Rapid Urbanization and
Climate Stress” (Revised and Resubmitted to Nature Communications)

The primary purpose of this capsule is to ensure the reproducibility of the
**Zhongshan** land subsidence and Groundwater Level (GWL) forecasting
results presented in the paper, utilizing the **Physics-Informed Hybrid
Attentive LSTM Network (PIHALNet)** model implemented in the
`fusionlab-learn` Python library.

---

## Table of Contents

1. [The PIHALNet Model](#1-the-pihalnet-model)  
2. [The fusionlab-learn Library](#2-the-fusionlab-learn-library)  
3. [Code Ocean Capsule Contents](#3-code-ocean-capsule-contents)  
4. [Setup and Execution](#4-setup-and-execution)  
   * Environment  
   * Data (Zhongshan Dataset)  
   * Running `main_zhongshan_pihalnet.py`  
   * Expected Outputs  
5. [Understanding the `main_zhongshan_pihalnet.py` Workflow](#5-understanding-the-main_zhongshan_pihalnetpy-workflow)  
6. [Hyperparameter Tuning (Optional)](#6-hyperparameter-tuning-optional)  
7. [Short Summary of the Workflow](#7-short-summary-of-the-workflow)  
8. [Citation](#8-citation)  
9. [Support & Contact](#9-support--contact)

---

## 1. The PIHALNet Model

As detailed in our revised paper, the **Physics-Informed Hybrid Attentive
LSTM Network (PIHALNet)** is a deep learning framework designed to forecast
land subsidence and groundwater levels by integrating data-driven insights
with fundamental physical principles.

**Abstract Summary:**

> PIHALNet builds upon a hybrid LSTM-Attention architecture (previously
> explored as HALNet) and significantly enhances it by incorporating
> physics-informed neural network (PINN) principles. It simultaneously
> predicts subsidence and GWL, constrained by a  representation
> of Terzaghi’s consolidation theory, where the rate of subsidence is linked
> to GWL changes via a learnable physical coefficient. This approach aims
> to produce more physically plausible forecasts and allows for the
> data-driven estimation of effective system parameters. When evaluated on
> the Zhongshan land subsidence and GWL prediction task, PIHALNet
> demonstrated [mention key improvements, e.g., robust performance,
> physically consistent results, and novel insights into parameter C, with
> MAE/RMSE values for both subsidence and GWL, and R² scores].

PIHALNet effectively captures complex temporal patterns using its
multi-scale LSTM and attention core, while the PINN component ensures
predictions align with domain knowledge, offering a path towards more
interpretable and reliable multi-horizon forecasting in complex
geophysical systems.

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
  (Update if URL changes)

---

## 3. Code Ocean Capsule Contents

This capsule is structured for clarity and reproducibility:

* **`main_zhongshan_pihalnet.py`:**
  The primary Python script that executes the Zhongshan land subsidence and
  GWL forecasting workflow using `PIHALNet`. It covers data loading,
  preprocessing, PINN-specific data preparation, model training with a
  composite loss, prediction, and visualization.

* **`data/` (User Provided or via `fetch_zhongshan_data`):**

  * The `main_zhongshan_pihalnet.py` script will prioritize loading
    `zhongshan_500000.csv` (if this is the name of your target 500k sample
    dataset) from a `data/` directory at the root of the capsule.
  * If that fails, it attempts to download a 2,000-sample subset using
    `fusionlab.datasets.fetch_zhongshan_data()`.
  * As a further fallback, it may look for `zhongshan_2000.csv` in the
    `data/` folder.
  * **For Reviewers:** Please ensure `zhongshan_500000.csv` (or the
    specific large dataset file you are using) is uploaded to the `data/`
    folder in this Code Ocean capsule.

* **`results_pinn/` (Note: changed from `results/` to distinguish from
  previous runs):**
  This directory will be created by `main_zhongshan_pihalnet.py` to store
  all output artifacts:

  * Preprocessed data (CSV files)
  * Fitted data scalers (Joblib files)
  * Generated sequence data for `PIHALNet` input (Joblib files, output by
    `prepare_pinn_data_sequences`)
  * Trained `PIHALNet` model checkpoints (Keras format)
  * Forecast predictions for subsidence and GWL (CSV files)
  * Output plots (PNG, PDF files)

* **`README.md`:**
  This file.

* **`fusionlab/` directory (if including library source):**
  Contains the `fusionlab-learn` library modules, including:

  * `nn/pinn/models.py` (contains `PIHALNet`)
  * `nn/pinn/op.py` (contains `compute_consolidation_residual`, etc.)
  * `nn/pinn/utils.py` (contains `prepare_pinn_data_sequences`,
    `format_pihalnet_predictions`)
  * `nn/components.py` (standard NN components)
  * `plot/forecast.py` (contains `plot_forecasts`)

---

## 4. Setup and Execution

### Environment

* **Python:** `fusionlab-learn` typically requires Python 3.9 or higher.
* **Dependencies:**
  The `main_zhongshan_pihalnet.py` script uses `pandas`, `numpy`,
  `scikit-learn`, `tensorflow`, `matplotlib`, `joblib`, and `fusionlab-learn`.
  Recommended installation:

  ```bash
  pip install fusionlab-learn matplotlib scikit-learn joblib tensorflow
  pandas keras-tuner
  ```

  (Add `keras-tuner` if you include the tuning section and it’s not a core
  dependency of `fusionlab-learn[full]`.)

### Data (Zhongshan Dataset)

The `main_zhongshan_pihalnet.py` script is configured to handle Zhongshan
data:

1. **Primary Method (Large Dataset):**
   Looks for `zhongshan_500000.csv` (or your specified large dataset name)
   in `data/` (or `../data/`).
2. **Fallback 1 (Package Fetch):**
   Uses `fusionlab.datasets.fetch_zhongshan_data(as_frame=True,
   download_if_missing=True)` to download a 2,000-sample subset if the larger
   file isn’t found.
3. **Fallback 2 (Smaller Local CSV):**
   May look for a `zhongshan_2000.csv` locally if the fetch fails.

> **For Reviewers:** To use the intended 500,000 sample dataset, please
> upload it as `zhongshan_500000.csv` (or the correct filename) to the root
> `data/` folder of this capsule.

### Running `main_zhongshan_pihalnet.py`

Once the environment is set up and data is accessible, run:

```bash
python main_zhongshan_pihalnet.py
```

The script will output progress for each major step.

### Expected Outputs

All generated files will be saved into the
`./results_pinn/zhongshan_PIHALNet_run/` directory. Key outputs will
include:

* `zhongshan_01_raw_data.csv`
* `zhongshan_02_cleaned_data.csv`
* `zhongshan_03_processed_scaled_data.csv`
* `zhongshan_main_scaler.joblib` (and other scalers if used)
* `zhongshan_train_pinn_sequences_T{TIME_STEPS}_H{HORIZON}.joblib`
* `zhongshan_PIHALNet_H{HORIZON}.keras` (the trained model)
* `zhongshan_PIHALNet_forecast_TestSet_{YEAR_RANGE}.csv`
* Various plot files (e.g., `zhongshan_PIHALNet_plot_subsidence_spatial.png`)

---

## 5. Understanding the `main_zhongshan_pihalnet.py` Workflow

The script follows these key stages for `PIHALNet` and the Zhongshan dataset:

1. **Configuration:**
   Sets parameters like `CITY_NAME = "zhongshan"`, `MODEL_NAME = "PIHALNet"`,
   forecast horizon, time steps, PINN settings (`pde_mode`,
   `pinn_coefficient_C`, `lambda_pde`), learning rate, paths, etc.

2. **Data Loading:**
   Loads the Zhongshan dataset using the prioritized strategy (500k sample
   CSV preferred).

3. **Preprocessing:**

   * Selects relevant features from the Zhongshan dataset.
   * Converts the `year` column to a proper datetime object and then to a
     numerical `TIME_COL_NUMERIC_PINN` for use as the “t” coordinate in
     PINN.
   * Handles missing values.
   * Encodes categorical features (e.g., `geology`).
   * Scales numerical features (including targets `subsidence` and `GWL`,
     and coordinates `longitude`, `latitude`, `TIME_COL_NUMERIC_PINN`).

4. **Feature Set Definition for PINN:**
   Defines column names for coordinates (`time_col_numeric_pinn`,
   `lon_col`, `lat_col`), targets (`subsidence_col`, `gwl_col`), and
   static, dynamic, and future features for
   `prepare_pinn_data_sequences`.

5. **Master Data Splitting:**
   Splits the processed DataFrame by `year` into `df_train_master` and
   `df_test_master`.

6. **PINN Sequence Generation:**
   Uses `prepare_pinn_data_sequences` on `df_train_master` (and later
   `df_test_master`) to create input dictionaries (`inputs_dict`) and
   target dictionaries (`targets_dict`):

   * `inputs_dict` contains `'coords'`, `'static_features'`,
     `'dynamic_features'`, `'future_features'`.
   * `targets_dict` contains `'subsidence'` and `'gwl'`.

7. **`tf.data.Dataset` Creation & Train/Validation Split:**
   Converts the sequence dictionaries into `tf.data.Dataset` objects and
   splits the training sequences into training and validation datasets.

8. **`PIHALNet` Model Training:**

   * Instantiates `PIHALNet` with dimensions derived from the prepared data
     and configured hyperparameters (including `pde_mode`,
     `pinn_coefficient_C`).
   * Compiles the model with a composite loss: data fidelity losses for
     ‘subs\_pred’ and ‘gwl\_pred’ (e.g., MSE or `combined_quantile_loss`)
     and the physics-based loss weighted by `lambda_pde`. Appropriate
     metrics are also specified for each output.
   * Uses `EarlyStopping` and `ModelCheckpoint` (monitoring `val_total_loss`).
   * Trains the model using its custom `train_step`.
   * Loads the best model saved by `ModelCheckpoint`.

9. **Forecasting:**

   * Prepares test sequences from `df_test_master` using
     `prepare_pinn_data_sequences`.
   * Generates predictions using `pihal_model_loaded.predict(inputs_test_dict)`.
     The output is a dictionary.
   * Formats predictions using `format_pihalnet_predictions` into a
     comprehensive DataFrame including actuals, coordinates, and optionally
     inverse-scaled values.
   * Saves the forecast DataFrame as CSV.

10. **Visualization:**
    If forecasts were generated, uses `plot_forecasts` (called separately
    for subsidence and GWL) to create spatial or temporal plots.

11. **Save All Figures:**
    Saves all open Matplotlib figures.

---

## 6. Hyperparameter Tuning (Optional)

The `main_zhongshan_pihalnet.py` script uses a predefined set of
hyperparameters for `PIHALNet`. For optimal performance, these can be tuned
using the `PIHALTuner` utility from `fusionlab.nn.pinn.tuning`. Below is a
conceptual snippet:

```python
from fusionlab.nn.pinn.tuning import PIHALTuner
from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences
# Assuming tf, pd, np, and your column names (S_DIM, D_DIM, etc.) are
# defined

# 1. Prepare your full training data (e.g., df_train_master from main script)
# inputs_np_dict, targets_np_dict = prepare_pinn_data_sequences(
#     df=df_train_master,
#     time_col=TIME_COL_NUMERIC_PINN,
#     lon_col=LON_COL, lat_col=LAT_COL,
#     subsidence_col=SUBSIDENCE_COL, gwl_col=GWL_COL,
#     dynamic_cols=dynamic_features_list, static_cols=static_features_list,
#     future_cols=future_features_list, group_id_cols=GROUP_ID_COLS_SEQ,
#     time_steps=TIME_STEPS, forecast_horizon=FORECAST_HORIZON_YEARS,
#     verbose=1
# )
# val_inputs_np_dict, val_targets_np_dict = ... (for validation data)

# 2. Instantiate PIHALTuner using the .create() factory method
# This infers dimensions from your data.
# tuner = PIHALTuner.create(
#     inputs_data=inputs_np_dict,
#     targets_data=targets_np_dict,
#     forecast_horizon=FORECAST_HORIZON_YEARS, # Must match data
#     quantiles=QUANTILES, # Must match data if used
#     objective='val_total_loss', # Monitor total PINN loss
#     max_trials=30, # Example
#     project_name="Zhongshan_PIHALNet_Tuning",
#     directory=os.path.join(RUN_OUTPUT_PATH, "tuning_results"),
#     executions_per_trial=1,
#     tuner_type='bayesianoptimization', # or 'randomsearch'
#     param_space_config={ # Optional: Override default HP search ranges
#         'embed_dim': [32, 64],
#         'lambda_pde': [0.1, 1.0, 10.0]
#     }
# )

# 3. Run the tuning process using the tuner's fit method
# best_model, best_hps, tuner_obj = tuner.fit(
#     inputs=inputs_np_dict, # Pass NumPy dicts directly
#     y=targets_np_dict,
#     validation_data=(val_inputs_np_dict, val_targets_np_dict),
#     epochs=50, # Max epochs per trial
#     batch_size=128, # Batch size for creating tf.data.Dataset internally
#     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_total_loss',
#                                                patience=7)],
#     verbose=1
# )

# if best_model:
#     print("Best HPs:", best_hps.values)
#     best_model.summary()
```

## 7. Short Summary of the Workflow

The `main_zhongshan_pihalnet.py` script executes a PINN-based forecasting
pipeline:

1. **Setup & Data:** Configures parameters and loads the Zhongshan dataset
   (prioritizing a 500k sample file).
2. **Preprocessing:** Cleans data, creates a numerical time coordinate,
   encodes categoricals, and scales features.
3. **PINN Data Sequencing:** Uses `prepare_pinn_data_sequences` to create
   dictionaries of inputs (including `coords` for physics) and targets
   (`subsidence`, `gwl`) for `PIHALNet`.
4. **Dataset Creation:** Converts NumPy sequence dictionaries into
   `tf.data.Dataset` objects for training/validation.
5. **Model Training:** Instantiates `PIHALNet` with appropriate dimensions
   and PINN settings. Compiles with a composite loss (data terms for
   subsidence & GWL + weighted physics term for consolidation). Trains
   using its custom `train_step`.
6. **Forecasting & Output:** Generates predictions for subsidence and GWL
   on test sequences. Formats these multi-output predictions into a
   structured CSV using `format_pihalnet_predictions`.
7. **Visualization:** Creates and saves plots for both subsidence and GWL
   forecasts.

This workflow demonstrates a reproducible, physics-informed approach to
forecasting land subsidence and GWL.

---

## 8. Citation

If you use `PIHALNet` or this physics-informed approach in your research,
please cite our paper:

> Laurent Kouadio & Liu Rong. (2025). Forecasting Urban Land Subsidence in the Era
> of Rapid Urbanization and Climate Stress. *Submitted to Nature
> Communications*.

**BibTeX entry (to be updated upon publication/preprint):**

```bibtex
@article{yourlastname2025pihalnet,
  title={Forecasting Urban Land Subsidence in the Era of Rapid Urbanization
         and Climate Stress},
  author={[Your Name] and [Co-author Names]},
  journal={Submitted to Nature Communications},
  year={2025}
}
```

---

## 9. Support & Contact

*For questions or issues, please contact the corresponding authors or open
an [issue](https://github.com/earthai-tech/fusionlab-learn/issues) on the
`fusionlab-learn` GitHub repository (if applicable).*

```

