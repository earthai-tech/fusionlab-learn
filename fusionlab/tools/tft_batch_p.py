"""
tft_batch_prediction.py
========================

**Script Name:** tft_batch_p.py

**Description:**
This script trains a Temporal Fusion Transformer (TFT) model to predict
land subsidence from 2024 to 2030 using historical data and future
covariates. Enhancements include increased sequence length, improved
LSTM units, better handling of dynamic features, consolidated training
across batches, and streamlined data processing.

**Usage:**
Integrate this script as part of the `gofast.tools` subpackage to
perform TFT-based subsidence predictions.

**Example:**

.. code-block:: python

    from gofast.tools import tft_batch_p

    # Execute the main function
    tft_batch_p.main()

**Dependencies:**
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- seaborn
- tensorflow
- scikeras
- geopandas
- shapely
- gofast

**Author:** Daniel
**Date:** 2024-12-17
"""

# =============================================================================
# Importing necessary libraries
# =============================================================================
import os
import warnings
from typing import List, Union, Tuple

import pandas as pd
import numpy as np
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import scikeras
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import gofast as gf
from gofast.nn.tft import TemporalFusionTransformer
from gofast.utils.spatial_utils import spatial_sampling, batch_spatial_sampling
from gofast.utils.data_utils import pop_labels_in

# =============================================================================
# Suppress warnings for clarity
# =============================================================================
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# =============================================================================
# Package versions for reference
# =============================================================================
pkgs_versions = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit-learn": gf.__version__,  # Corrected scikit-learn version
    "joblib": joblib.__version__,
    "tensorflow": tf.__version__,
    "gofast": gf.__version__,
    "matplotlib": mpl.__version__,
    'scikeras': scikeras.__version__,
}

# =============================================================================
# Data Loading and Encoding Functions
# =============================================================================
def load_final_data(data_path: str) -> pd.DataFrame:
    """
    Load the main dataset into a pandas DataFrame.

    :param data_path: Path to the dataset directory.
    :return: DataFrame containing the final data.
    """
    final_data_path = os.path.join(
        data_path, 'zhongshan_filtered_final_data.csv'
    )
    final_data = pd.read_csv(final_data_path)
    return final_data.copy()


def save_batches(
    data: pd.DataFrame,
    data_path: str,
    n_batches: int = 5
) -> None:
    """
    Perform batch spatial sampling and save each batch to CSV.

    :param data: DataFrame to sample from.
    :param data_path: Path to save the batch files.
    :param n_batches: Number of batches to create.
    """
    batches = batch_spatial_sampling(
        data, sample_size=len(data), n_batches=n_batches
    )
    for idx, batch_df in enumerate(batches):
        batch_file = f'zh_tft_data_batch{idx+1}.csv'
        batch_df.to_csv(os.path.join(data_path, batch_file), index=False)


def fit_onehot_encoder(
    data: pd.DataFrame,
    categorical_features: List[str]
) -> Tuple[OneHotEncoder, List[str]]:
    """
    Fit a OneHotEncoder on the entire dataset for consistent encoding.

    :param data: DataFrame containing the data.
    :param categorical_features: List of categorical feature names.
    :return: Tuple of fitted encoder and encoded categorical column names.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(data[categorical_features])
    encoded_cat_columns = encoder.get_feature_names_out(categorical_features).tolist()
    return encoder, encoded_cat_columns


def preprocess_batch(
    batch_number: int,
    data_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target: str,
    encoder: OneHotEncoder,
    encoded_cat_columns: List[str]
) -> pd.DataFrame:
    """
    Preprocess a single batch: encode, scale, and clean data.

    :param batch_number: The batch number.
    :param data_path: Path to the batch files.
    :param categorical_features: List of categorical feature names.
    :param numerical_features: List of numerical feature names.
    :param target: Target column name.
    :param encoder: Fitted OneHotEncoder.
    :param encoded_cat_columns: List of encoded categorical column names.
    :return: Processed DataFrame.
    """
    batch_file = f'zh_tft_data_batch{batch_number}.csv'
    batch_path = os.path.join(data_path, batch_file)
    batch_data = pd.read_csv(batch_path)

    # Remove labels for the year 2023
    batch_data = pop_labels_in(
        batch_data, categories='year', labels=2023
    )

    # Select relevant features
    selected_cols = categorical_features + numerical_features + [target]
    batch_data = batch_data[selected_cols]

    # Drop missing values
    batch_data.dropna(inplace=True)

    # One-hot encode categorical features using the pre-fitted encoder
    encoded_cats = encoder.transform(batch_data[categorical_features])
    encoded_cat_df = pd.DataFrame(
        encoded_cats, columns=encoded_cat_columns, index=batch_data.index
    )

    # Concatenate encoded features with numerical features
    processed_data = pd.concat(
        [batch_data[numerical_features], encoded_cat_df, batch_data[target]],
        axis=1
    )

    # Initialize scalers
    numerical_scaler = StandardScaler()
    long_lat_scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    # Scale longitude and latitude separately
    scaled_long_lat = long_lat_scaler.fit_transform(
        processed_data[['longitude', 'latitude']]
    )
    scaled_long_lat_df = pd.DataFrame(
        scaled_long_lat, columns=['longitude', 'latitude'], index=batch_data.index
    )

    # Scale other numerical features
    scaled_numerical = numerical_scaler.fit_transform(
        processed_data[numerical_features]
    )
    scaled_numerical_df = pd.DataFrame(
        scaled_numerical, columns=numerical_features, index=batch_data.index
    )

    # Scale the target
    scaled_target = target_scaler.fit_transform(
        processed_data[[target]]
    )
    scaled_target_df = pd.DataFrame(
        scaled_target, columns=[target], index=batch_data.index
    )

    # Combine all processed features
    final_processed_data = pd.concat(
        [scaled_numerical_df, encoded_cat_df, scaled_target_df],
        axis=1
    ).sort_values('year')

    # Save scalers
    numerical_scaler_path = os.path.join(
        data_path, f'tft.numerical_scaler_{batch_number}.joblib'
    )
    lonlat_path = os.path.join(
        data_path, f'tft.lonlat{batch_number}.joblib'
    )
    target_scaler_path = os.path.join(
        data_path, f'tft.target_scaler_{batch_number}.joblib'
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

    return final_processed_data

def combine_batches(
    data_path: str,
    n_batches: int,
    categorical_features: List[str],
    numerical_features: List[str],
    target: str,
    encoder: OneHotEncoder,
    encoded_cat_columns: List[str]
) -> pd.DataFrame:
    """
    Combine all processed batches into a single DataFrame.

    :param data_path: Path to the batch files.
    :param n_batches: Number of batches.
    :param categorical_features: List of categorical feature names.
    :param numerical_features: List of numerical feature names.
    :param target: Target column name.
    :param encoder: Fitted OneHotEncoder.
    :param encoded_cat_columns: List of encoded categorical column names.
    :return: Combined DataFrame.
    """
    combined_data = pd.DataFrame()

    for batch_number in range(1, n_batches + 1):
        processed_batch = preprocess_batch(
            batch_number, data_path, categorical_features,
            numerical_features, target, encoder, encoded_cat_columns
        )
        combined_data = pd.concat(
            [combined_data, processed_batch], ignore_index=True
        )

    return combined_data


# =============================================================================
# Sequence Creation Function
# =============================================================================
def create_sequences(
    data: pd.DataFrame,
    sequence_length: int,
    forecast_horizon: int,
    target_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and corresponding multi-step targets for time series forecasting.

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
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        seq = data.iloc[i:i+sequence_length]
        target = data.iloc[i+sequence_length:i+sequence_length + forecast_horizon][target_col]
        sequences.append(seq.values)
        targets.append(target.values)
    return np.array(sequences), np.array(targets)


# =============================================================================
# Data Splitting Functions
# =============================================================================
def split_static_dynamic(
    sequences: np.ndarray,
    static_indices: List[int],
    dynamic_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split sequences into static and dynamic inputs.

    :param sequences: Input sequences array.
    :param static_indices: Indices of static features.
    :param dynamic_indices: Indices of dynamic features.
    :return: Tuple of static and dynamic inputs.
    """
    # Extract static inputs from the first time step
    static_inputs = sequences[:, 0, static_indices]
    static_inputs = static_inputs.reshape(
        -1, len(static_indices), 1
    )

    # Extract dynamic inputs from all time steps
    dynamic_inputs = sequences[:, :, dynamic_indices]
    dynamic_inputs = dynamic_inputs.reshape(
        -1, sequences.shape[1], len(dynamic_indices), 1
    )

    return static_inputs, dynamic_inputs


# =============================================================================
# Model Preparation Functions
# =============================================================================
def initialize_tft_model(
    static_input_dim: int,
    dynamic_input_dim: int,
    num_static_vars: int,
    num_dynamic_vars: int,
    hidden_units: int = 64,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    forecast_horizon: int = 1,
    quantiles: Union[None, List[float]] = None,
    activation: str = 'relu',
    use_batch_norm: bool = True,
    num_lstm_layers: int = 1,
    lstm_units: List[int] = [64]
) -> TemporalFusionTransformer:
    """
    Initialize the Temporal Fusion Transformer model with specified parameters.

    :param static_input_dim: Input dimension per static variable.
    :param dynamic_input_dim: Input dimension per dynamic variable.
    :param num_static_vars: Number of static variables.
    :param num_dynamic_vars: Number of dynamic variables.
    :param hidden_units: Number of hidden units.
    :param num_heads: Number of attention heads.
    :param dropout_rate: Dropout rate.
    :param forecast_horizon: Forecast horizon.
    :param quantiles: Quantiles for probabilistic forecasting.
    :param activation: Activation function.
    :param use_batch_norm: Whether to use batch normalization.
    :param num_lstm_layers: Number of LSTM layers.
    :param lstm_units: List of LSTM units per layer.
    :return: Initialized TFT model.
    """
    model = TemporalFusionTransformer(
        static_input_dim=static_input_dim,
        dynamic_input_dim=dynamic_input_dim,
        num_static_vars=num_static_vars,
        num_dynamic_vars=num_dynamic_vars,
        hidden_units=hidden_units,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        forecast_horizon=forecast_horizon,
        quantiles=quantiles,
        activation=activation,
        use_batch_norm=use_batch_norm,
        num_lstm_layers=num_lstm_layers,
        lstm_units=lstm_units
    )
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


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
    encoded_cat_columns: List[str]
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

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[int], List[int], List[float], List[float]]
        Future static inputs, future dynamic inputs, future years list, location IDs list,
        longitudes, latitudes.
    """
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

            # Static features
            static_features = last_sequence_features.iloc[0][
                ['longitude', 'latitude'] + encoded_cat_columns
            ].values
            static_inputs = static_features.reshape(1, len(static_features))
            static_inputs = static_inputs.astype(np.float32)

            # Dynamic features
            dynamic_features = last_sequence_features.iloc[
                :, dynamic_feature_indices
            ].values
            dynamic_inputs = dynamic_features.reshape(
                sequence_length, len(dynamic_feature_indices), 1
            )

            # Future inputs for each year
            for year in future_years:
                future_dynamic_inputs = dynamic_inputs.copy()

                # Update 'year' feature if present
                if 'year' in feature_columns:
                    year_idx = feature_columns.index('year')
                    if year_idx in dynamic_feature_indices:
                        dyn_year_idx = dynamic_feature_indices.index(year_idx)
                        future_dynamic_inputs[-1, dyn_year_idx, 0] = (
                            (year - year_mean) / year_std
                        )

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
    visualize_years: List[int] = [2025, 2027, 2030],
    cmap: str = 'viridis',
    output_filename: str = 'tft_prediction_ok_2024_2030.png'
) -> None:
    """
    Visualize the predictions for selected years.

    :param future_data: DataFrame containing future predictions.
    :param data_path: Path to save the visualization.
    :param visualize_years: List of years to visualize.
    :param cmap: Colormap for the scatter plot.
    :param output_filename: Filename for the saved plot.
    """
    visualize_data = future_data[future_data['year'].isin(visualize_years)].copy()
    fig, axes = plt.subplots(
        1, len(visualize_years), figsize=(18, 6), constrained_layout=True
    )

    for i, year in enumerate(visualize_years):
        ax = axes[i]
        year_data = visualize_data[visualize_data['year'] == year]
        sc = ax.scatter(
            year_data['longitude'],
            year_data['latitude'],
            c=year_data['predicted_subsidence'],
            cmap=cmap,
            s=10,
            alpha=0.8
        )
        ax.set_title(f'Subsidence in {year}')
        ax.axis('off')

    # Add a single colorbar
    cbar = fig.colorbar(
        sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.1
    )
    cbar.set_label('Subsidence (mm)')

    # Save and show the plot
    output_path = os.path.join(data_path, output_filename)
    fig.savefig(output_path, dpi=300)
    plt.show()


# =============================================================================
# Main Function
# =============================================================================
def main():
    """
    Main function to execute the TFT batch prediction workflow.
    """
    # Define paths
    data_path = r'C:\Users\Daniel\Documents\zongshan_codes\tft_prediction_ok\new\error_pred_2023'
    n_batches = 5

    # Load final data
    final_data = load_final_data(data_path)
    print("Final Data Head:\n", final_data.head())

    # Fit OneHotEncoder on the entire dataset for consistent encoding
    categorical_features = ['geological_category', 'rainfall_category']
    encoder, encoded_cat_columns = fit_onehot_encoder(final_data, categorical_features)

    # Save the fitted encoder for future use
    encoder_save_path = os.path.join(data_path, 'onehot_encoder_combined.joblib')
    joblib.dump(encoder, encoder_save_path)

    # Perform batch sampling and save batches
    save_batches(final_data, data_path, n_batches=n_batches)

    # Define numerical features and target
    numerical_features = [
        'longitude', 'latitude', 'year', 'GWL',
        'normalized_density', 'normalized_seismic_risk_score'
    ]
    target = 'subsidence'

    # Combine all processed batches
    combined_data = combine_batches(
        data_path, n_batches, categorical_features,
        numerical_features, target, encoder, encoded_cat_columns
    )
    print("Combined Data Shape:", combined_data.shape)

    # Define sequence parameters
    sequence_length = 7  # Increased from 4 to 7
    forecast_horizon = 1  # Predicting one step ahead

    # Create sequences
    feature_columns = numerical_features + encoded_cat_columns
    sequences, targets = create_sequences(
        combined_data[feature_columns + [target]],
        sequence_length,
        forecast_horizon,
        target
    )
    print("Sequences Shape:", sequences.shape)
    print("Targets Shape:", targets.shape)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, targets, test_size=0.2, shuffle=False
    )
    print("Training Set Shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print("Validation Set Shapes:")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

    # Define static and dynamic feature indices
    static_feature_names = ['longitude', 'latitude'] + encoded_cat_columns
    static_feature_indices = [
        feature_columns.index(col) for col in static_feature_names
    ]
    dynamic_feature_indices = [
        i for i in range(len(feature_columns))
        if i not in static_feature_indices
    ]

    # Split sequences into static and dynamic inputs
    static_train, dynamic_train = split_static_dynamic(
        X_train, static_feature_indices, dynamic_feature_indices
    )
    static_val, dynamic_val = split_static_dynamic(
        X_val, static_feature_indices, dynamic_feature_indices
    )
    print("Static Train Shape:", static_train.shape)
    print("Dynamic Train Shape:", dynamic_train.shape)
    print("Static Val Shape:", static_val.shape)
    print("Dynamic Val Shape:", dynamic_val.shape)

    # Reshape targets
    y_train = y_train.reshape(-1, forecast_horizon, 1)
    y_val = y_val.reshape(-1, forecast_horizon, 1)
    print("Reshaped y_train Shape:", y_train.shape)
    print("Reshaped y_val Shape:", y_val.shape)

    # Initialize TFT model
    static_input_dim = 1  # Each static feature has a single value
    dynamic_input_dim = 1  # Each dynamic feature has a single value
    num_static_vars = static_train.shape[1]
    num_dynamic_vars = dynamic_train.shape[2]

    tft_model = initialize_tft_model(
        static_input_dim,
        dynamic_input_dim,
        num_static_vars,
        num_dynamic_vars,
        hidden_units=64,
        num_heads=4,
        dropout_rate=0.1,
        forecast_horizon=forecast_horizon,
        quantiles=None,
        activation='relu',
        use_batch_norm=True,
        num_lstm_layers=1,
        lstm_units=[64]
    )

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(data_path, 'best_tft_model'),
        monitor='val_loss',
        save_best_only=True,
        save_format='tf'
    )

    # Train the model
    history = tft_model.fit(
        x=[static_train, dynamic_train],
        y=y_train,
        validation_data=([static_val, dynamic_val], y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # Load the best model if needed
    # tft_model = load_model(os.path.join(data_path, 'best_tft_model'))

    # Add location_id for future reference
    combined_data['location_id'] = combined_data.groupby(
        ['longitude', 'latitude']
    ).ngroup()

    # Define future years for prediction
    future_years = list(range(2024, 2031))  # 2024 to 2030 inclusive

    # Prepare future data
    future_static_inputs, future_dynamic_inputs, future_years_list, \
    location_ids_list, longitudes, latitudes = prepare_future_data(
        combined_data,
        feature_columns,
        dynamic_feature_indices,
        static_feature_indices,
        sequence_length,
        forecast_horizon,
        future_years,
        encoded_cat_columns
    )
    print("Future Static Inputs Shape:", future_static_inputs.shape)
    print("Future Dynamic Inputs Shape:", future_dynamic_inputs.shape)

    # Make predictions
    predictions = tft_model.predict([future_static_inputs, future_dynamic_inputs])
    predictions = predictions.reshape(-1, forecast_horizon)

    # Inverse transform predictions
    target_scaler = joblib.load(
        os.path.join(data_path, 'tft.target_scaler_1.joblib')
    )["target_scaler"]
    predictions_inverse = target_scaler.inverse_transform(predictions)

    # Ensure all arrays have the same length
    lengths = {
        "predictions_inverse": len(predictions_inverse.flatten()),
        "future_years_list": len(future_years_list),
        "location_ids_list": len(location_ids_list),
        "longitudes": len(longitudes),
        "latitudes": len(latitudes),
    }
    print("Lengths of arrays:", lengths)
    min_length = min(lengths.values())

    # Truncate to the minimum length
    future_years_list = future_years_list[:min_length]
    location_ids_list = location_ids_list[:min_length]
    longitudes = longitudes[:min_length]
    latitudes = latitudes[:min_length]
    predictions_inverse = predictions_inverse.flatten()[:min_length]

    # Prepare future data DataFrame
    future_data = pd.DataFrame({
        'year': future_years_list,
        'location_id': location_ids_list,
        'longitude': longitudes,
        'latitude': latitudes,
        'predicted_subsidence': predictions_inverse
    })
    print("Future Data Shape:", future_data.shape)

    # Inverse transform longitude and latitude
    # Assuming longitude and latitude scalers are the same across batches
    # Load the scaler from the first batch
    long_lat_scaler = joblib.load(os.path.join(data_path, 'tft.lonlat1.joblib'))
    long_lat_original = long_lat_scaler.inverse_transform(
        future_data[['longitude', 'latitude']]
    )
    future_data[['longitude', 'latitude']] = long_lat_original

    # Save future predictions
    future_predictions_path = os.path.join(data_path, 'tft.ls.predicted_2024_2030.csv')
    future_data.to_csv(future_predictions_path, index=False)
    print(f"Future predictions saved to {future_predictions_path}")

    # Visualize predictions
    visualize_predictions(
        future_data, data_path,
        visualize_years=[2025, 2027, 2030],
        cmap='viridis',
        output_filename='tft_prediction_ok_2024_2030.png'
    )


if __name__ == "__main__":
    main()
