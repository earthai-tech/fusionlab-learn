# -*- coding: utf-8 -*-

""" 
xtft_determinic_p.py
====================

**Description:**
This script trains an XTFT (Extended Temporal Fusion Transformer) model to 
predict subsidence from the years 2023 to 2026 using historical data 
and future covariates.

**Key Enhancements:**
1. **Configurable Verbosity for Detailed Debugging:**
   - Users can set the verbosity level externally (0: WARNING, 1: INFO, 2: DEBUG) 
   to control the amount of logging output.
2. **Proper Scope and Definition of Variables:**
   - Ensures variables like `encoded_cat_columns` are properly defined and accessible
   throughout the script.
3. **Modularized Structure for Better Readability and Maintenance:**
   - Encapsulates functionality into functions, promoting code reusability 
   and clarity.

**Usage:**
Run the script from the command line and specify the verbosity level using
the `--verbose` flag.

**Example:**
```bash
python xtft_determinic_p.py --verbose 2


**Usage:**
Run the script from the command line and specify the verbosity level using
 the `--verbose` flag.

**Example:**
```bash
python xtft_determinic_p.py --verbose 2

**Dependencies:**

pandas
numpy
scikit-learn
joblib
matplotlib
tensorflow
scikeras
gofast
Author: Daniel Date: 2024-12-17 

"""
import os
import logging
from typing import List, Union, Tuple

import pandas as pd
import numpy as np
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Suppress warnings for clarity
import warnings
import sklearn
import gofast as gf
from gofast.utils.data_utils import pop_labels_in 
from gofast.nn.tft import XTFT
# Versioning of packages for reproducibility
pkgs_versions = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": sklearn.__version__,
    "joblib": joblib.__version__,
    "tensorflow": tf.__version__,
    "gofast": gf.__version__,
    "matplotlib": mpl.__version__,
    "scikeras": "Not Used",  # Not used in this script
}

# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(verbose: int = 0):
    """
    Configure the logging settings based on verbosity level.

    Parameters
    ----------
    verbose : int
        Verbosity level (0: WARNING, 1: INFO, 2: DEBUG).
    """
    level = logging.WARNING  # Default level
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

# =============================================================================
# Data Loading and Preprocessing Functions
# =============================================================================

def load_data(data_path: str, filename: str = 'final_data.csv') -> pd.DataFrame:
    """
    Load the main dataset into a pandas DataFrame.

    Parameters
    ----------
    data_path : str
        Path to the dataset directory.
    filename : str, optional
        Name of the CSV file to load, by default 'final_data.csv'.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    data_file = os.path.join(data_path, filename)
    logging.info(f"Loading data from {data_file}")
    data = pd.read_csv(data_file)
    logging.debug(f"Data shape: {data.shape}")
    return data.copy()

def preprocess_data(
    batch_data: pd.DataFrame,
    categorical_features: List[str],
    numerical_features: List[str],
    target: str,
    encoder: OneHotEncoder,
    encoded_cat_columns: List[str],
    data_path: str,
    batch_number: int
) -> pd.DataFrame:
    """
    Preprocess a single batch of data: encoding, scaling, and cleaning.

    Parameters
    ----------
    batch_data : pd.DataFrame
        DataFrame of the batch.
    categorical_features : List[str]
        List of categorical feature names.
    numerical_features : List[str]
        List of numerical feature names.
    target : str
        Target column name.
    encoder : OneHotEncoder
        Fitted OneHotEncoder.
    encoded_cat_columns : List[str]
        List of encoded categorical column names.
    data_path : str
        Path to save the scaler and encoder.
    batch_number : int
        Batch number for file naming.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    logging.info(f"Preprocessing batch {batch_number}")
    
    # # Uncomment this, if you want to pop the year 2023 in the data --------
    # # Remove rows where 'year' is 2023 for prediction purposes
    # batch_data = pop_labels_in(batch_data, categories='year', labels=2023)
    # logging.debug(f"Batch {batch_number} shape after removing year 2023: {batch_data.shape}")
    # ----------------------------------------------------------------------------

    # Select relevant features
    selected_cols = categorical_features + numerical_features + [target]
    batch_data = batch_data[selected_cols]
    logging.debug(f"Batch {batch_number} selected columns: {selected_cols}")
    
    # Drop missing values
    initial_shape = batch_data.shape
    batch_data.dropna(inplace=True)
    logging.debug(f"Batch {batch_number} shape after dropping NA: {batch_data.shape} (Dropped {initial_shape[0] - batch_data.shape[0]} rows)")
    
    # One-hot encode categorical features
    encoded_cats = encoder.transform(batch_data[categorical_features])
    encoded_cat_df = pd.DataFrame(
        encoded_cats, columns=encoded_cat_columns, index=batch_data.index
    )
    logging.debug(f"Batch {batch_number} encoded categorical features shape: {encoded_cat_df.shape}")
    
    # Concatenate encoded features with numerical features
    processed_data = pd.concat(
        [batch_data[numerical_features], encoded_cat_df, batch_data[target]],
        axis=1
    )
    logging.debug(f"Batch {batch_number} processed data shape: {processed_data.shape}")
    
    # Initialize scalers
    numerical_scaler = StandardScaler()
    target_scaler = StandardScaler()
    long_lat_scaler = StandardScaler()
    
    # Scale longitude and latitude separately
    scaled_long_lat = long_lat_scaler.fit_transform(processed_data[['longitude', 'latitude']])
    scaled_long_lat_df = pd.DataFrame(
        scaled_long_lat, columns=['longitude', 'latitude'], index=batch_data.index
    )
    logging.debug(f"Batch {batch_number} scaled longitude and latitude shape: {scaled_long_lat_df.shape}")
    
    # Scale other numerical features
    scaled_numerical = numerical_scaler.fit_transform(processed_data[numerical_features])
    scaled_numerical_df = pd.DataFrame(
        scaled_numerical, columns=numerical_features, index=batch_data.index
    )
    logging.debug(f"Batch {batch_number} scaled numerical features shape: {scaled_numerical_df.shape}")
    
    # Scale the target variable
    scaled_target = target_scaler.fit_transform(processed_data[[target]])
    scaled_target_df = pd.DataFrame(
        scaled_target, columns=[target], index=batch_data.index
    )
    logging.debug(f"Batch {batch_number} scaled target shape: {scaled_target_df.shape}")
    
    # Combine all processed features
    final_processed_data = pd.concat(
        [scaled_numerical_df, encoded_cat_df, scaled_target_df],
        axis=1
    ).sort_values('year')
    logging.debug(f"Batch {batch_number} final processed data shape: {final_processed_data.shape}")
    
    # Save scalers
    numerical_scaler_path = os.path.join(
        data_path, f'xtft.numerical_scaler_batch{batch_number}.joblib'
    )
    lonlat_path = os.path.join(
        data_path, f'xtft.lonlat_batch{batch_number}.joblib'
    )
    target_scaler_path = os.path.join(
        data_path, f'xtft.target_scaler_batch{batch_number}.joblib'
    )
    
    dict_scalers = {
        "target_scaler": target_scaler,
        "numerical_scaler": numerical_scaler,
        "categorical_scaler": encoder,
        "__version__": pkgs_versions
    }
    
    joblib.dump(dict_scalers, target_scaler_path)
    joblib.dump(numerical_scaler, numerical_scaler_path)
    joblib.dump(long_lat_scaler, lonlat_path)
    logging.info(f"Batch {batch_number} scalers saved.")
    
    return final_processed_data

def prepare_sequences(
    data: pd.DataFrame,
    sequence_length: int,
    forecast_horizon: int,
    target_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and corresponding targets for forecasting.

    Parameters
    ----------
    data : pd.DataFrame
        The processed DataFrame containing features and target.
    sequence_length : int
        The number of past time steps to include in each input sequence.
    forecast_horizon : int
        The number of future time steps to predict.
    target_col : str
        The name of the target column.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of input sequences and corresponding multi-step targets.
    """
    logging.info("Creating sequences for training and validation")
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        seq = data.iloc[i:i+sequence_length]
        target = data.iloc[i+sequence_length:i+sequence_length + forecast_horizon][target_col]
        sequences.append(seq.values)
        targets.append(target.values)
    
    sequences_array = np.array(sequences)
    targets_array = np.array(targets)
    logging.debug(f"Sequences shape: {sequences_array.shape}")
    logging.debug(f"Targets shape: {targets_array.shape}")
    return sequences_array, targets_array

def split_static_dynamic(
    sequences: np.ndarray,
    static_indices: List[int],
    dynamic_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split sequences into static and dynamic inputs.

    Parameters
    ----------
    sequences : np.ndarray
        Array of input sequences.
    static_indices : List[int]
        Indices of static features.
    dynamic_indices : List[int]
        Indices of dynamic features.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of static and dynamic inputs.
    """
    logging.debug("Splitting sequences into static and dynamic inputs")
    # Extract static inputs from the first time step
    static_inputs = sequences[:, 0, static_indices]
    static_inputs = static_inputs.reshape(-1, len(static_indices), 1)
    logging.debug(f"Static inputs shape: {static_inputs.shape}")
    
    # Extract dynamic inputs from all time steps
    dynamic_inputs = sequences[:, :, dynamic_indices]
    dynamic_inputs = dynamic_inputs.reshape(-1, sequences.shape[1], len(dynamic_indices), 1)
    logging.debug(f"Dynamic inputs shape: {dynamic_inputs.shape}")
    
    return static_inputs, dynamic_inputs

# =============================================================================
# Prediction Preparation Functions
# =============================================================================

def prepare_future_data(
    final_processed_data: pd.DataFrame,
    feature_columns: List[str],
    dynamic_feature_indices: List[int],
    static_feature_indices: List[int],
    sequence_length: int,
    forecast_horizon: int,
    future_years: List[int],
    encoded_cat_columns: List[str],
    logging_enabled: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int], List[float], List[float]]:
    """
    Prepare future static and dynamic inputs for making predictions.

    Parameters
    ----------
    final_processed_data : pd.DataFrame
        The processed DataFrame containing all features and target.
    feature_columns : List[str]
        List of feature column names.
    dynamic_feature_indices : List[int]
        Indices of dynamic features in feature_columns.
    static_feature_indices : List[int]
        Indices of static features in feature_columns.
    sequence_length : int
        The number of past time steps to include in each input sequence.
    forecast_horizon : int
        The number of future time steps to predict.
    future_years : List[int]
        List of future years to predict.
    encoded_cat_columns : List[str]
        List of encoded categorical column names.
    logging_enabled : bool, optional
        Whether to enable logging within this function, by default True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[int], List[int], List[float], List[float]]
        Future static inputs, future dynamic inputs, future years list, location IDs list,
        longitudes, latitudes.
    """
    if logging_enabled:
        logging.info("Preparing future data for predictions")
    
    future_static_inputs_list = []
    future_dynamic_inputs_list = []
    future_years_list = []
    location_ids_list = []
    longitudes = []
    latitudes = []

    # Mean and std for scaling 'year'
    year_mean = final_processed_data['year'].mean()
    year_std = final_processed_data['year'].std()

    # Group by 'location_id'
    grouped = final_processed_data.groupby('location_id')

    for name, group in grouped:
        group = group.sort_values('year').reset_index(drop=True)
        if len(group) >= sequence_length:
            last_sequence = group.iloc[-sequence_length:]
            last_sequence_features = last_sequence[feature_columns]

            # Static features: longitude, latitude, encoded categories
            static_features = last_sequence_features.iloc[0][['longitude', 'latitude'] + encoded_cat_columns].values
            static_inputs = static_features.reshape(1, len(static_features))
            static_inputs = static_inputs.astype(np.float32)

            # Dynamic features
            dynamic_features = last_sequence_features.iloc[:, dynamic_feature_indices].values
            dynamic_inputs = dynamic_features.reshape(sequence_length, len(dynamic_feature_indices), 1)

            # Future inputs for each year
            for year in future_years:
                future_dynamic_inputs = dynamic_inputs.copy()

                # Update 'year' feature if present
                if 'year' in feature_columns:
                    year_idx = feature_columns.index('year')
                    if year_idx in dynamic_feature_indices:
                        dyn_year_idx = dynamic_feature_indices.index(year_idx)
                        future_dynamic_inputs[-1, dyn_year_idx, 0] = (year - year_mean) / year_std

                future_static_inputs_list.append(static_inputs)
                future_dynamic_inputs_list.append(future_dynamic_inputs)
                future_years_list.append(year)
                location_ids_list.append(name)
                longitudes.append(static_features[0])
                latitudes.append(static_features[1])

    # Convert lists to arrays
    future_static_inputs = np.array(future_static_inputs_list)
    future_dynamic_inputs = np.array(future_dynamic_inputs_list)

    # Reshape static inputs
    future_static_inputs = future_static_inputs.reshape(
        future_static_inputs.shape[0],
        future_static_inputs.shape[1],
        1
    )

    if logging_enabled:
        logging.debug(f"Future static inputs shape: {future_static_inputs.shape}")
        logging.debug(f"Future dynamic inputs shape: {future_dynamic_inputs.shape}")

    return (
        future_static_inputs,
        future_dynamic_inputs,
        future_years_list,
        location_ids_list,
        longitudes,
        latitudes
    )

# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_predictions(
    future_data: pd.DataFrame,
    data_path: str,
    visualize_years: List[int] = [2024, 2025, 2026],
    cmap: str = 'jet_r',
    output_filename: str = 'xtft_prediction_2024_2026.png',
    logging_enabled: bool = True
) -> None:
    """
    Visualize the predictions for selected years.

    Parameters
    ----------
    future_data : pd.DataFrame
        DataFrame containing future predictions.
    data_path : str
        Path to save the visualization.
    visualize_years : List[int], optional
        List of years to visualize, by default [2024, 2025, 2026].
    cmap : str, optional
        Colormap for the scatter plot, by default 'jet_r'.
    output_filename : str, optional
        Filename for the saved plot, by default 'xtft_prediction_2024_2026.png'.
    logging_enabled : bool, optional
        Whether to enable logging within this function, by default True.
    """
    if logging_enabled:
        logging.info("Visualizing predictions")
    
    fig, axes = plt.subplots(1, len(visualize_years), figsize=(18, 6), constrained_layout=True)

    for i, year in enumerate(visualize_years):
        ax = axes[i]
        year_data = future_data[['longitude', 'latitude', f'predicted_subsidence_{year}']]
        sc = ax.scatter(
            year_data['longitude'],
            year_data['latitude'],
            c=year_data[f'predicted_subsidence_{year}'],
            cmap=cmap,
            s=10,
            alpha=0.8
        )
        ax.set_title(f'Subsidence Prediction for {year}')
        ax.axis('off')  # Remove axes for cleaner plots

    # Add a single colorbar
    cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.1)
    cbar.set_label('Subsidence (mm)')

    # Save the figure
    output_path = os.path.join(data_path, output_filename)
    plt.savefig(output_path, dpi=300)
    logging.info(f"Prediction visualization saved to {output_path}")
    plt.show()

# =============================================================================
# Main Function
# =============================================================================

def main(verbose: int = 1):
    """
    Main function to execute the XTFT batch prediction workflow.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level (0: WARNING, 1: INFO, 2: DEBUG), by default 1.
    """
    # Setup logging
    setup_logging(verbose)
    
    logging.info("Starting XTFT prediction script")
    
    # Define paths
    data_path = r'J:\first_data\Nansha\new\xtft\new'
    
    # =============================================================================
    # 1. Load the dataset
    # =============================================================================
    final_data = load_data(data_path, 'final_data.csv')
    logging.info("Loaded final data")
    
    # Create a copy of the data for processing
    data0 = final_data.copy()
    logging.debug(f"Data0 shape: {data0.shape}")
    
    # Display the first few rows of the data for inspection
    logging.info("Initial data preview:")
    logging.info(f"\n{data0.head()}")
    
    # =============================================================================
    # 2. Optional: Batch Spatial Sampling
    # =============================================================================
    # Uncomment the following section if batching is required
    # batches = batch_spatial_sampling(data0, sample_size=len(data0), n_batches=5)
    # for idx, batch_df in enumerate(batches):
    #     batch_df.to_csv(os.path.join(data_path, f'nsh_tft_data_batch{idx+1}.csv'), index=False)
    
    # =============================================================================
    # 3. Data Preprocessing
    # =============================================================================
    
    # Categorizing building concentration into A, B, C categories
    batch_data = pd.read_csv(os.path.join(data_path, 'final_data.bc_cat.csv'))
    
    # Rename columns for clarity (longitude, latitude, etc.)
    batch_data.rename(columns={
        'x': 'longitude',
        'y': 'latitude',
        'groundwater_levels': 'GWL',
        'lithology': 'geological_category'
    }, inplace=True)
    logging.debug(f"Batch data columns after renaming: {batch_data.columns.tolist()}")
    
    # =============================================================================
    # 4. Define Features and Target
    # =============================================================================
    
    # List of categorical and numerical features for model training
    categorical_features = ['geological_category', 'bc_category']
    numerical_features = [
        'longitude', 'latitude', 'year', 'GWL',
        'soil_thickness', 'soil_quality'
    ]
    
    target = 'subsidence'  # The target variable we aim to predict
    
    # Select relevant features from the dataset
    batch_data = batch_data[categorical_features + numerical_features + [target]]
    logging.debug(f"Batch data selected columns: {batch_data.columns.tolist()}")
    
    # Check for any missing values in the data
    missing_values = batch_data.isnull().sum()
    logging.info(f"Missing values in data:\n{missing_values}")
    
    # Drop rows with missing values to ensure clean data
    initial_shape = batch_data.shape
    batch_data.dropna(inplace=True)
    logging.debug(f"Data shape after dropping NA: {batch_data.shape} (Dropped {initial_shape[0] - batch_data.shape[0]} rows)")
    
    # =============================================================================
    # 5. One-Hot Encoding for Categorical Variables
    # =============================================================================
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(batch_data[categorical_features])
    encoded_cat_columns = encoder.get_feature_names_out(categorical_features).tolist()
    logging.info(f"Encoded categorical columns: {encoded_cat_columns}")
    
    # One-hot encode categorical features
    encoded_cats = encoder.transform(batch_data[categorical_features])
    
    # Convert the encoded categorical data into a DataFrame
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoded_cat_columns, index=batch_data.index)
    logging.debug(f"Encoded categorical DataFrame shape: {encoded_cat_df.shape}")
    
    # Combine the encoded categorical features with numerical features and the target
    processed_data = pd.concat([batch_data[numerical_features], encoded_cat_df, batch_data[target]], axis=1)
    logging.debug(f"Processed data shape: {processed_data.shape}")
    
    # Save the encoder for future use
    encoder_path = os.path.join(data_path, 'xtft_onehot_encoder.joblib')
    joblib.dump(encoder, encoder_path)
    logging.info(f"OneHotEncoder saved to: {encoder_path}")
    
    # =============================================================================
    # 6. Scaling Numerical Data
    # =============================================================================
    
    # Initialize the scalers for different feature groups
    numerical_scaler = StandardScaler()
    target_scaler = StandardScaler()
    long_lat_scaler = StandardScaler()
    
    # Scale the longitude and latitude features separately
    scaled_long_lat = long_lat_scaler.fit_transform(processed_data[['longitude', 'latitude']])
    scaled_long_lat_df = pd.DataFrame(
        scaled_long_lat, columns=['longitude', 'latitude'], index=batch_data.index
    )
    logging.debug(f"Scaled longitude and latitude shape: {scaled_long_lat_df.shape}")
    
    # Scale the numerical features (excluding longitude and latitude)
    scaled_numerical = numerical_scaler.fit_transform(processed_data[numerical_features])
    scaled_numerical_df = pd.DataFrame(
        scaled_numerical, columns=numerical_features, index=batch_data.index
    )
    logging.debug(f"Scaled numerical features shape: {scaled_numerical_df.shape}")
    
    # Scale the target variable (subsidence)
    scaled_target = target_scaler.fit_transform(processed_data[[target]])
    scaled_target_df = pd.DataFrame(
        scaled_target, columns=[target], index=batch_data.index
    )
    logging.debug(f"Scaled target shape: {scaled_target_df.shape}")
    
    # =============================================================================
    # 7. Combine Processed Features
    # =============================================================================
    
    # Combine all processed data: numerical features, encoded categories, and target
    final_processed_data = pd.concat([scaled_numerical_df, encoded_cat_df, scaled_target_df], axis=1)
    logging.debug(f"Final processed data shape: {final_processed_data.shape}")
    
    # Save the scalers for future use
    numerical_scaler_path = os.path.join(data_path, 'xtft.numerical_scaler.joblib')
    lonlat_path = os.path.join(data_path, 'xtft.lonlat.joblib')
    
    joblib.dump(numerical_scaler, numerical_scaler_path)
    joblib.dump(long_lat_scaler, lonlat_path)
    logging.info(f"Numerical scaler saved to: {numerical_scaler_path}")
    logging.info(f"Longitude and latitude scaler saved to: {lonlat_path}")
    
    # =============================================================================
    # 8. Save Combined Scalers Dictionary
    # =============================================================================
    
    # Save the combined dictionary of scalers (numerical, categorical, target)
    target_scaler_path = os.path.join(data_path, 'xtft.target_scaler.joblib')
    dict_scalers = {
        "target_scaler": target_scaler,
        "numerical_scaler": numerical_scaler,
        "categorical_scaler": encoder,
        "__version__": pkgs_versions
    }
    
    joblib.dump(dict_scalers, target_scaler_path)
    logging.info(f"Combined scalers saved to: {target_scaler_path}")
    
    # =============================================================================
    # 9. Sort Data by Year
    # =============================================================================
    
    # Ensure the data is sorted by 'year'
    final_processed_data.sort_values('year', inplace=True)
    logging.debug("Final processed data sorted by year")
    
    # =============================================================================
    # 10. Prepare Sequences for Model Training
    # =============================================================================
    
    # Define sequence length and forecast horizon
    sequence_length = 4  # Corresponds to 2015 to 2018
    forecast_horizon = 4  # Predicting from 2023 to 2026
    
    # Define feature columns
    feature_columns = numerical_features + encoded_cat_columns
    target_col = target
    
    # Create sequences
    sequences, targets = prepare_sequences(
        final_processed_data[feature_columns + [target_col]],
        sequence_length,
        forecast_horizon,
        target_col
    )
    
    # =============================================================================
    # 11. Split Data into Train and Validation Sets
    # =============================================================================
    
    logging.info("Splitting data into training and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, targets, test_size=0.3, random_state=42, shuffle=False
    )
    logging.debug(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.debug(f"Validation set shape: X_val={X_val.shape}, y_val={y_val.shape}")
    
    # =============================================================================
    # 12. Split Sequences into Static and Dynamic Inputs
    # =============================================================================
    
    # Identify static and dynamic feature indices
    static_feature_names = ['longitude', 'latitude'] + encoded_cat_columns
    static_feature_indices = [feature_columns.index(col) for col in static_feature_names]
    dynamic_feature_indices = [i for i in range(len(feature_columns)) if i not in static_feature_indices]
    logging.debug(f"Static feature indices: {static_feature_indices}")
    logging.debug(f"Dynamic feature indices: {dynamic_feature_indices}")
    
    # Split sequences into static and dynamic inputs
    static_train, dynamic_train = split_static_dynamic(
        X_train, static_feature_indices, dynamic_feature_indices
    )
    static_val, dynamic_val = split_static_dynamic(
        X_val, static_feature_indices, dynamic_feature_indices
    )
    
    # Reshape targets to match model expectations
    y_train = y_train.reshape(-1, forecast_horizon, 1)
    y_val = y_val.reshape(-1, forecast_horizon, 1)
    logging.debug(f"Reshaped y_train shape: {y_train.shape}")
    logging.debug(f"Reshaped y_val shape: {y_val.shape}")
    
    # =============================================================================
    # 13. Define and Compile the XTFT Model
    # =============================================================================
    
    # XTFT Model Parameters
    static_input_dim = 1  # Each static feature has a single value
    dynamic_input_dim = 1  # Each dynamic feature has a single value
    future_covariate_dim = len(encoded_cat_columns)  # Repeat encoded categories as future covariates
    output_dim = 1  # Continuous target
    
    # Initialize the XTFT model
    model = XTFT(
        static_input_dim=static_input_dim,
        dynamic_input_dim=dynamic_input_dim,
        future_covariate_dim=future_covariate_dim,
        embed_dim=32,
        forecast_horizons=forecast_horizon,
        quantiles=None,  # Deterministic prediction
        max_window_size=10,
        memory_size=100,
        num_heads=4,
        dropout_rate=0.1,
        output_dim=output_dim,
        attention_units=32,
        hidden_units=64,
        lstm_units=64,
        activation='relu',
        use_residuals=True,
        use_batch_norm=False,
        final_agg='last'
    )
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    logging.info("XTFT model compiled")
    
    # =============================================================================
    # 14. Define Callbacks for Training
    # =============================================================================
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True, 
        verbose=1 if verbose else 0
    )
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(data_path, 'best_xtft_model'),
        monitor='val_loss',
        save_best_only=True, 
        save_format='tf',
        verbose=1 if verbose else 0
    )
    
    # =============================================================================
    # 15. Train the XTFT Model
    # =============================================================================
    
    logging.info("Starting model training")
    history = model.fit(
        [static_train, dynamic_train, encoded_cats[:static_train.shape[0]]], y_train,
        validation_data=([static_val, dynamic_val, encoded_cats[-static_val.shape[0]:]], y_val),
        epochs=100,  # Change this as needed for deep training
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1 if verbose else 0
    )
    
    # =============================================================================
    # 16. Make Predictions on Validation Set
    # =============================================================================
    
    logging.info("Making predictions on the validation set")
    predictions_scaled = model.predict([static_val, dynamic_val, encoded_cats[-static_val.shape[0]:]])
    logging.debug(f"Predictions scaled shape: {predictions_scaled.shape}")
    
    # =============================================================================
    # 17. Reverse Scaling of Predictions
    # =============================================================================
    
    # Reshape predictions to 2D for inverse scaling (flattening the last two dimensions)
    predictions_reshaped = predictions_scaled.reshape(-1, predictions_scaled.shape[-1])
    
    # Reverse scaling for the predicted subsidence values using the target scaler
    target_scaler_loaded = joblib.load(os.path.join(data_path, 'xtft.target_scaler.joblib'))["target_scaler"]
    predictions_inverse = target_scaler_loaded.inverse_transform(predictions_reshaped)
    logging.debug(f"Predictions inverse scaled shape: {predictions_inverse.shape}")
    
    # Reshape back to the original shape (the forecast horizon)
    predictions = predictions_inverse.reshape(-1, forecast_horizon)
    logging.debug(f"Predictions reshaped shape: {predictions.shape}")
    
    # =============================================================================
    # 18. Reverse Scaling for Longitude and Latitude
    # =============================================================================
    
    # Reverse scaling for longitude and latitude using the longitude-latitude scaler
    longitude_latitude_scaled = static_val[:, :2]  # Assuming first two columns are longitude and latitude
    longitude_latitude_original = joblib.load(os.path.join(data_path, 'xtft.lonlat.joblib')).inverse_transform(longitude_latitude_scaled)
    
    # =============================================================================
    # 19. Create DataFrame for the Predicted Data (For 2023–2026)
    # =============================================================================
    
    future_data = pd.DataFrame({
        'longitude': longitude_latitude_original[:, 0],
        'latitude': longitude_latitude_original[:, 1],
        'predicted_subsidence_2023': predictions[:, 0],
        'predicted_subsidence_2024': predictions[:, 1],
        'predicted_subsidence_2025': predictions[:, 2],
        'predicted_subsidence_2026': predictions[:, 3],
    })
    logging.info(f"Future data shape: {future_data.shape}")
    
    # =============================================================================
    # 20. Save Future Predictions to CSV
    # =============================================================================
    
    future_predictions_path = os.path.join(data_path, 'xtft_prediction_val_2023_2026.csv')
    future_data.to_csv(future_predictions_path, index=False)
    logging.info(f"Future predictions saved to {future_predictions_path}")
    
    # =============================================================================
    # 21. Visualize Predictions
    # =============================================================================
    
    visualize_predictions(
        future_data, 
        data_path,
        visualize_years=[2024, 2025, 2026],
        cmap='jet_r',
        output_filename='xtft_prediction_2024_2026.png',
        logging_enabled=True
    )
    
    logging.info("XTFT prediction script completed successfully")

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="XTFT Deterministic Prediction Script")
    parser.add_argument(
        '--verbose', 
        type=int, 
        default=1, 
        choices=[0, 1, 2],
        help='Verbosity level: 0=WARNING, 1=INFO, 2=DEBUG'
    )
    
    args = parser.parse_args()
    
    main(verbose=args.verbose)
