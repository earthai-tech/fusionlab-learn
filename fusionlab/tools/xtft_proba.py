"""
XTFT Probabilistic Prediction Application
=========================================

This application provides a flexible and robust tool for performing probabilistic
predictions using the Extended Temporal Fusion Transformer (XTFT) with quantiles.
Designed to handle various datasets, the tool allows users to specify key parameters
such as feature types, target variables, and model hyperparameters, ensuring adaptability
to different forecasting tasks.

Features:
---------
- **Data Loading and Preprocessing**: Loads datasets, handles missing values, encodes
  categorical variables, and scales numerical features based on user specifications.
- **Sequence Generation**: Creates sequences of data suitable for time-series forecasting,
  with customizable time steps and forecast horizons.
- **Model Training**: Defines, compiles, and trains the XTFT model with support for early
  stopping and model checkpointing. Hyperparameters are adjustable via command-line arguments.
- **Prediction and Inversion**: Generates probabilistic predictions using specified quantiles
  and reverses scaling to obtain interpretable results.
- **Visualization**: Visualizes the prediction results geographically, supporting multiple
  quantiles and forecast years.
- **Flexibility**: Exposes key variables and parameters, allowing the application to work
  with any compatible dataset.

Usage:
------
This application is intended to be used as part of the `gofast.tools` subpackage. Users can
manipulate external parameters such as data paths, feature lists, model hyperparameters, and other
configurations to tailor the tool to their specific forecasting tasks.

Example:
--------
```bash
python xtft_proba_app.py 
    --data /path/to/final_data.bc_cat.csv 
    --target subsidence 
    --categorical_features geological_category bc_category 
    --numerical_features longitude latitude year GWL soil_thickness soil_quality 
    --epochs 100 
    --batch_size 32 
    --time_steps 4 
    --forecast_horizon 4 
    --quantiles 0.1 0.5 0.9
```

Dependencies:
-------------
- Python 3.7+
- pandas
- numpy
- scikeras
- scikit-learn
- tensorflow
- matplotlib
- joblib
- gofast

Ensure all dependencies are installed before running the application.

"""

import os
import sys
import argparse
import warnings
import logging

import pandas as pd
import numpy as np
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import gofast as gf
from gofast.core.io import read_data, print_script_info, show_usage 
from gofast.nn.tft import XTFT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# A module-level flag to ensure docstring is printed only once
_DOCSTRING_PRINTED = False
 
def load_data(main_data_file):
    """
    Load the main dataset from the specified path.
    :param main_data_file: Filename of the main dataset CSV.
    :type main_data_file: str
    :return: Loaded DataFrame.
    :rtype: pandas.DataFrame
    """
    try:
        file_path = os.path.dirname(main_data_file)
        final_data = read_data(main_data_file)
        logger.info("Data loaded successfully from %s", file_path)
        return final_data.copy()
    except Exception as e:
        logger.error("Failed to load data from %s: %s", file_path, e)
        raise

def preprocess_data(
        final_data, 
        data_path, 
        # cat_data_file,
        categorical_features,
        numerical_features, target):
    """
    Preprocess the data by encoding categorical variables and scaling numerical features.

    :param final_data: Original DataFrame.
    :type final_data: pandas.DataFrame
    :param data_path: Path to save encoders and scalers.
    :type data_path: str
    :param cat_data_file: Filename of the categorical data CSV.
    :type cat_data_file: str
    :param categorical_features: List of categorical feature names.
    :type categorical_features: list
    :param numerical_features: List of numerical feature names.
    :type numerical_features: list
    :param target: Name of the target variable.
    :type target: str
    :return: Processed DataFrame and dictionary of fitted scalers/encoders.
    :rtype: tuple[pandas.DataFrame, dict]
    """
    logger.info("Starting data preprocessing...")

    try:
        # Load categorized data
        cat_data_path = os.path.join(final_data)
        batch_data = pd.read_csv(cat_data_path)
        logger.info("Categorical data loaded from %s", cat_data_path)

        # Rename columns for clarity if necessary
        rename_dict = {}
        if 'x' in batch_data.columns and 'longitude' not in batch_data.columns:
            rename_dict['x'] = 'longitude'
        if 'y' in batch_data.columns and 'latitude' not in batch_data.columns:
            rename_dict['y'] = 'latitude'
        if 'groundwater_levels' in batch_data.columns and 'GWL' not in batch_data.columns:
            rename_dict['groundwater_levels'] = 'GWL'
        if 'lithology' in batch_data.columns and 'geological_category' not in batch_data.columns:
            rename_dict['lithology'] = 'geological_category'

        if rename_dict:
            batch_data.rename(columns=rename_dict, inplace=True)
            logger.info("Renamed columns: %s", rename_dict)

        # Select relevant columns
        required_columns = categorical_features + numerical_features + [target]
        missing_cols = set(required_columns) - set(batch_data.columns)
        if missing_cols:
            logger.error("Missing columns in categorical data: %s", missing_cols)
            raise ValueError(f"Missing columns: {missing_cols}")
        batch_data = batch_data[required_columns]
        logger.info("Selected relevant features: %s", required_columns)

        # Handle missing values
        missing_values = batch_data.isnull().sum()
        logger.info("Missing values in data before dropping:\n%s", missing_values)
        batch_data.dropna(inplace=True)
        logger.info("Dropped rows with missing values.")

        # One-Hot Encoding for categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(batch_data[categorical_features])
        encoded_cat_columns = encoder.get_feature_names_out(categorical_features)
        encoded_cat_df = pd.DataFrame(
            encoded_cats, columns=encoded_cat_columns, index=batch_data.index
        )
        logger.info("Categorical variables encoded.")

        # Combine numerical and encoded categorical features
        processed_data = pd.concat(
            [batch_data[numerical_features], encoded_cat_df, batch_data[target]],
            axis=1
        )

        # Scaling numerical features
        numerical_scaler = StandardScaler()
        scaled_numerical = numerical_scaler.fit_transform(processed_data[numerical_features])
        scaled_numerical_df = pd.DataFrame(
            scaled_numerical, columns=numerical_features, index=processed_data.index
        )
        logger.info("Numerical features scaled.")

        # Scaling the target variable
        target_scaler = StandardScaler()
        scaled_target = target_scaler.fit_transform(processed_data[[target]])
        scaled_target_df = pd.DataFrame(
            scaled_target, columns=[target], index=processed_data.index
        )
        logger.info("Target variable scaled.")

        # Scaling longitude and latitude separately
        long_lat_scaler = StandardScaler()
        scaled_long_lat = long_lat_scaler.fit_transform(processed_data[['longitude', 'latitude']])
        scaled_long_lat_df = pd.DataFrame(
            scaled_long_lat, columns=['longitude', 'latitude'], index=processed_data.index
        )
        logger.info("Longitude and latitude scaled separately.")

        # Replace scaled longitude and latitude in the processed data
        processed_data[['longitude', 'latitude']] = scaled_long_lat_df

        # Combine all processed data
        final_processed_data = pd.concat(
            [scaled_numerical_df, encoded_cat_df, scaled_target_df],
            axis=1
        )
        logger.info("All features combined into final processed data.")

        # Save the encoder
        encoder_path = os.path.join(data_path, 'xtft_onehot_encoder.joblib')
        joblib.dump(encoder, encoder_path)
        logger.info("OneHotEncoder saved to: %s", encoder_path)

        # Save scalers
        numerical_scaler_path = os.path.join(data_path, 'xtft_numerical_scaler.joblib')
        lonlat_scaler_path = os.path.join(data_path, 'xtft_lonlat_scaler.joblib')
        joblib.dump(numerical_scaler, numerical_scaler_path)
        joblib.dump(long_lat_scaler, lonlat_scaler_path)
        logger.info("Numerical and longitude-latitude scalers saved to: %s and %s",
                    numerical_scaler_path, lonlat_scaler_path)

        # Save combined scalers and versions
        pkg_versions = {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "tensorflow": tf.__version__,
            "gofast": gf.__version__,
            "matplotlib": mpl.__version__,
            # "scikeras": scikeras.__version__,  # Uncomment if scikeras is used
        }
        target_scaler_path = os.path.join(data_path, 'xtft_target_scaler.joblib')
        dict_scalers = {
            "target_scaler": target_scaler,
            "numerical_scaler": numerical_scaler,
            "categorical_scaler": encoder,
            "lonlat_scaler": long_lat_scaler,
            "__version__": pkg_versions
        }
        joblib.dump(dict_scalers, target_scaler_path)
        logger.info("Combined scalers and package versions saved to: %s", target_scaler_path)

        return final_processed_data, dict_scalers
    except Exception as e:
        logger.error("Failed to save preprocess_data: %s", e)
        raise

def create_sequences(final_processed_data, time_steps, forecast_horizon,
                     static_features, dynamic_features, target):
    """
    Create sequences of data for the XTFT model.

    :param final_processed_data: Processed DataFrame.
    :type final_processed_data: pandas.DataFrame
    :param time_steps: Number of time steps in each sequence.
    :type time_steps: int
    :param forecast_horizon: Number of future steps to predict.
    :type forecast_horizon: int
    :param static_features: List of static feature names.
    :type static_features: list
    :param dynamic_features: List of dynamic feature names.
    :type dynamic_features: list
    :param target: Name of the target variable.
    :type target: str
    :return: Static inputs, dynamic inputs, future covariates, and target outputs.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    logger.info("Creating sequences for the XTFT model...")
    try:
        static_input = []
        dynamic_input = []
        future_covariate_input = []
        target_output = []

        total_sequences = len(final_processed_data) - time_steps - forecast_horizon + 1
        logger.info("Total sequences to create: %d", total_sequences)

        for i in range(total_sequences):
            sequence_data = final_processed_data.iloc[i:i + time_steps]

            static_data = sequence_data[static_features].iloc[0].values
            dynamic_data = sequence_data[dynamic_features].values
            future_covariate_data = np.repeat(
                sequence_data[static_features].iloc[0].values.reshape(1, -1),
                time_steps,
                axis=0
            )
            target_data = final_processed_data.iloc[
                i + time_steps:i + time_steps + forecast_horizon
            ][target].values

            static_input.append(static_data)
            dynamic_input.append(dynamic_data)
            future_covariate_input.append(future_covariate_data)
            target_output.append(target_data)

            if (i + 1) % 100000 == 0:
                logger.info("Processed %d/%d sequences...", i + 1, total_sequences)

        static_input = np.array(static_input)
        dynamic_input = np.array(dynamic_input)
        future_covariate_input = np.array(future_covariate_input)
        target_output = np.array(target_output)

        logger.info("Sequences created successfully.")
        logger.debug("Static input shape: %s", static_input.shape)
        logger.debug("Dynamic input shape: %s", dynamic_input.shape)
        logger.debug("Future covariate input shape: %s", future_covariate_input.shape)
        logger.debug("Target output shape: %s", target_output.shape)

        return static_input, dynamic_input, future_covariate_input, target_output

    except Exception as e:
        logger.error("Failed to create sequences: %s", e)
        raise

def split_data(static_input, dynamic_input, future_covariate_input, target_output,
               test_size=0.3, random_state=42):
    """
    Split the data into training and validation sets.

    :param static_input: Static features.
    :type static_input: np.ndarray
    :param dynamic_input: Dynamic features.
    :type dynamic_input: np.ndarray
    :param future_covariate_input: Future covariates.
    :type future_covariate_input: np.ndarray
    :param target_output: Target variables.
    :type target_output: np.ndarray
    :param test_size: Proportion of the dataset to include in the validation split.
    :type test_size: float
    :param random_state: Seed used by the random number generator.
    :type random_state: int
    :return: Split datasets.
    :rtype: tuple
    """
    logger.info("Splitting data into training and validation sets...")
    try:
        X_train_static, X_val_static, X_train_dynamic, X_val_dynamic, \
        X_train_future_covariate, X_val_future_covariate, y_train, y_val = train_test_split(
            static_input, dynamic_input, future_covariate_input, target_output,
            test_size=test_size, random_state=random_state
        )
        logger.info("Data split completed.")
        logger.debug("Training static shape: %s", X_train_static.shape)
        logger.debug("Validation static shape: %s", X_val_static.shape)
        return (
            X_train_static, X_val_static,
            X_train_dynamic, X_val_dynamic,
            X_train_future_covariate, X_val_future_covariate,
            y_train, y_val
        )
    except Exception as e:
        logger.error("Data splitting failed: %s", e)
        raise

def build_model(
    static_input_dim, dynamic_input_dim, future_covariate_dim,
    forecast_horizon, output_dim, quantiles
):
    """
    Define and compile the XTFT model.

    :param static_input_dim: Dimension of static inputs.
    :type static_input_dim: int
    :param dynamic_input_dim: Dimension of dynamic inputs.
    :type dynamic_input_dim: int
    :param future_covariate_dim: Dimension of future covariates.
    :type future_covariate_dim: int
    :param forecast_horizon: Number of steps to forecast.
    :type forecast_horizon: int
    :param output_dim: Dimension of the output.
    :type output_dim: int
    :param quantiles: List of quantiles for probabilistic prediction.
    :type quantiles: list
    :return: Compiled XTFT model.
    :rtype: XTFT
    """
    logger.info("Building the XTFT model...")
    try:
        model = XTFT(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_covariate_dim=future_covariate_dim,
            embed_dim=32,
            forecast_horizons=forecast_horizon,
            quantiles=quantiles,
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
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        logger.info("Model built and compiled successfully.")
        return model
    except Exception as e:
        logger.error("Failed to build the model: %s", e)
        raise

def train_xtft_model(model, X_train, y_train, X_val, y_val,
                    data_path, epochs=100, batch_size=32):
    """
    Train the XTFT model with early stopping and model checkpointing.

    :param model: Compiled XTFT model.
    :type model: XTFT
    :param X_train: Training static, dynamic, and future covariate inputs.
    :type X_train: list or tuple
    :param y_train: Training targets.
    :type y_train: np.ndarray
    :param X_val: Validation static, dynamic, and future covariate inputs.
    :type X_val: list or tuple
    :param y_val: Validation targets.
    :type y_val: np.ndarray
    :param data_path: Path to save the best model.
    :type data_path: str
    :param epochs: Number of training epochs.
    :type epochs: int
    :param batch_size: Training batch size.
    :type batch_size: int
    :return: Training history.
    :rtype: History
    """
    logger.info("Starting XTFT model training...")
    try:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        checkpoint_path = os.path.join(data_path, 'best_xtft_q_model')
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_format='tf'
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        logger.info("XTFT model training completed.")
        return history
    except Exception as e:
        logger.error("XTFT model training failed: %s", e)
        raise

def predict_and_inverse_scale(
    model, X_val, dict_scalers, forecast_horizon, quantiles
):
    """
    Generate predictions using the trained model and reverse the scaling.

    :param model: Trained XTFT model.
    :type model: XTFT
    :param X_val: Validation static, dynamic, and future covariate inputs.
    :type X_val: list or tuple
    :param dict_scalers: Dictionary of fitted scalers and encoders.
    :type dict_scalers: dict
    :param forecast_horizon: Number of steps to forecast.
    :type forecast_horizon: int
    :param quantiles: List of quantiles used for prediction.
    :type quantiles: list
    :return: Dictionary of quantile predictions.
    :rtype: dict
    """
    logger.info("Generating predictions...")
    try:
        predictions_scaled = model.predict(X_val)
        logger.info("Predictions generated with shape: %s", predictions_scaled.shape)

        predictions_scaled = predictions_scaled.squeeze(-1)
        logger.info("Predictions reshaped to: %s", predictions_scaled.shape)

        target_scaler = dict_scalers['target_scaler']

        quantile_predictions = {}
        for i, q in enumerate(quantiles):
            predictions_reshaped = predictions_scaled[:, :, i].reshape(-1, 1)
            predictions_inverse = target_scaler.inverse_transform(predictions_reshaped)
            quantile_predictions[q] = predictions_inverse.reshape(-1, forecast_horizon)
            logger.debug("Quantile %s predictions shape: %s", q, quantile_predictions[q].shape)

        logger.info("Inverse scaling completed for all quantiles.")
        return quantile_predictions
    except Exception as e:
        logger.error("Prediction and inverse scaling failed: %s", e)
        raise

def visualize_predictions(future_data, quantiles, visualize_years, data_path):
    """
    Visualize the prediction results for each quantile.

    :param future_data: DataFrame containing predictions.
    :type future_data: pandas.DataFrame
    :param quantiles: List of quantiles.
    :type quantiles: list
    :param visualize_years: List of years to visualize.
    :type visualize_years: list
    :param data_path: Path to save the visualization.
    :type data_path: str
    """
    logger.info("Visualizing predictions...")
    try:
        fig, axes = plt.subplots(
            len(quantiles), len(visualize_years), figsize=(24, 6 * len(quantiles)),
            constrained_layout=True
        )
        if len(quantiles) == 1:
            axes = [axes]  # Make it iterable

        for i, q in enumerate(quantiles):
            for j, year in enumerate(visualize_years):
                ax = axes[i, j] if len(quantiles) > 1 else axes[j]
                year_col = f'predicted_subsidence_{year}_q{q}'
                if year_col not in future_data.columns:
                    logger.warning("Column %s not found in future_data. Skipping...", year_col)
                    continue
                year_data = future_data[['longitude', 'latitude', year_col]]
                sc = ax.scatter(
                    year_data['longitude'],
                    year_data['latitude'],
                    c=year_data[year_col],
                    cmap='jet_r',
                    s=10,
                    alpha=0.8
                )
                ax.set_title(f'Subsidence Prediction for {year} (q={q})')
                ax.axis('off')

                # Add a colorbar to the last subplot in each row
                if j == len(visualize_years) - 1:
                    cbar = fig.colorbar(
                        sc, ax=ax, orientation='vertical', fraction=0.02, pad=0.04
                    )
                    cbar.set_label('Subsidence (mm)')

        output_filename = os.path.join(
            data_path, 'xtft_quantile_predictions_visualization.png'
        )
        fig.savefig(output_filename, dpi=300)
        plt.close(fig)
        logger.info("Visualization saved to %s", output_filename)
    except Exception as e:
        logger.error("Visualization failed: %s", e)
        raise

def save_predictions(future_data, data_path, output_file='xtft_quantile_predictions.csv'):
    """
    Save the prediction results to a CSV file.

    :param future_data: DataFrame containing predictions.
    :type future_data: pandas.DataFrame
    :param data_path: Path to save the predictions.
    :type data_path: str
    :param output_file: Filename for the output CSV.
    :type output_file: str
    """
    try:
        output_csv = os.path.join(data_path, output_file)
        future_data.to_csv(output_csv, index=False)
        logger.info("Predictions saved to %s", output_csv)
    except Exception as e:
        logger.error("Failed to save predictions: %s", e)
        raise

def main(args):
    """
    Main function to execute the XTFT probabilistic prediction workflow.

    :param args: Command-line arguments.
    :type args: argparse.Namespace
    """
    global _DOCSTRING_PRINTED
    if not _DOCSTRING_PRINTED:
        print_script_info(__doc__)
        _DOCSTRING_PRINTED = True
    
    main_data_file = args.data
    # cat_data_file = args.cat_data_file
    target = args.target
    categorical_features = args.categorical_features
    numerical_features = args.numerical_features
    epochs = args.epochs
    batch_size = args.batch_size
    time_steps = args.time_steps
    forecast_horizon = args.forecast_horizon
    quantiles = args.quantiles
    visualize_years = args.visualize_years
 
    # Suppress warnings for clarity
    warnings.filterwarnings('ignore')
    tf.get_logger().setLevel('ERROR')

    # Load data
    final_data = load_data(main_data_file)
    data_path = os.path.dirname (final_data)
    # Preprocess data
    final_processed_data, dict_scalers = preprocess_data(
        final_data, data_path, # cat_data_file,
        categorical_features, numerical_features, target
    )

    # Define features
    encoded_cat_columns = [
        col for col in final_processed_data.columns
        if any(cf in col for cf in categorical_features)
    ]
    static_features = ['longitude', 'latitude'] + encoded_cat_columns
    dynamic_features = [col for col in numerical_features if col not in ['longitude', 'latitude']] + [target]
    
    logger.info("Static features: %s", static_features)
    logger.info("Dynamic features: %s", dynamic_features)

    # Create sequences
    static_input, dynamic_input, future_covariate_input, target_output = create_sequences(
        final_processed_data, time_steps, forecast_horizon, static_features, dynamic_features, target
    )

    # Split data
    (
        X_train_static, X_val_static,
        X_train_dynamic, X_val_dynamic,
        X_train_future_covariate, X_val_future_covariate,
        y_train, y_val
    ) = split_data(static_input, dynamic_input, future_covariate_input, target_output)

    # Reshape targets
    y_train = y_train.reshape(-1, forecast_horizon, 1)
    y_val = y_val.reshape(-1, forecast_horizon, 1)
    logger.info("Target variables reshaped for training.")

    # Build model
    model = build_model(
        static_input_dim=X_train_static.shape[1],
        dynamic_input_dim=X_train_dynamic.shape[2],
        future_covariate_dim=X_train_future_covariate.shape[2],
        forecast_horizon=forecast_horizon,
        output_dim=1,
        quantiles=quantiles
    )

    # Train model
    history = train_xtft_model(
        model,
        [X_train_static, X_train_dynamic, X_train_future_covariate],
        y_train,
        [X_val_static, X_val_dynamic, X_val_future_covariate],
        y_val,
        data_path,
        epochs=epochs,
        batch_size=batch_size
    )
    logger.info("History model loaded from %s", history.history)
    # Load the best model
    best_model_path = os.path.join(data_path, 'best_xtft_q_model')
    try:
        model = tf.keras.models.load_model(best_model_path, custom_objects={'XTFT': XTFT})
        logger.info("Best model loaded from %s", best_model_path)
    except Exception as e:
        logger.error("Failed to load the best model from %s: %s", best_model_path, e)
        raise
    
    # Make predictions
    quantile_predictions = predict_and_inverse_scale(
        model,
        [X_val_static, X_val_dynamic, X_val_future_covariate],
        dict_scalers,
        forecast_horizon,
        quantiles
    )

    # Reverse scaling for longitude and latitude
    try:
        long_lat_scaler = dict_scalers['lonlat_scaler']
        longitude_latitude_scaled = X_val_static[:, :2]
        longitude_latitude_original = long_lat_scaler.inverse_transform(longitude_latitude_scaled)
        X_val_static[:, :2] = longitude_latitude_original
        logger.info("Reversed scaling for longitude and latitude.")
    except Exception as e:
        logger.error("Failed to reverse scaling for longitude and latitude: %s", e)
        raise

    # Create DataFrame for future predictions
    future_data = pd.DataFrame({
        'longitude': X_val_static[:, 0],
        'latitude': X_val_static[:, 1],
    })

    # Add predictions to DataFrame
    max_year = final_processed_data['year'].max()
    future_years = range(max_year + 1, max_year + 1 + forecast_horizon)
    for i, year in enumerate(future_years):
        for q in quantiles:
            col_name = f'predicted_subsidence_{year}_q{q}'
            future_data[col_name] = quantile_predictions[q][:, i]
            logger.debug("Added column %s to future_data", col_name)

    # Save predictions
    save_predictions(future_data, data_path, output_file=args.output_file)

    # Visualize predictions
    visualize_predictions(future_data, quantiles, visualize_years, data_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="XTFT Probabilistic Prediction Application using Quantiles."
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the the main dataset preferably in CSV format.'
    )

    parser.add_argument(
        '--target',
        type=str,
        default='subsidence',
        help='Name of the target variable.'
    )
    parser.add_argument(
        '--categorical_features',
        type=str,
        nargs='+',
        required=True,
        help='List of categorical feature names.'
    )
    parser.add_argument(
        '--numerical_features',
        type=str,
        nargs='+',
        required=True,
        help='List of numerical feature names.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size.'
    )
    parser.add_argument(
        '--time_steps',
        type=int,
        default=4,
        help='Number of time steps in each sequence.'
    )
    parser.add_argument(
        '--forecast_horizon',
        type=int,
        default=4,
        help='Number of future steps to predict.'
    )
    parser.add_argument(
        '--quantiles',
        type=float,
        nargs='+',
        default=[0.1, 0.5, 0.9],
        help='List of quantiles for probabilistic prediction.'
    )
    parser.add_argument(
        '--visualize_years',
        type=int,
        nargs='+',
        default=[2024, 2025, 2026],
        help='List of future years to visualize predictions.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='xtft_quantile_predictions.csv',
        help='Filename for saving prediction results.'
    )
    parser.add_argument(
        "--helpdoc", 
        action="store_true", 
        help='Script documentation'
        )
    # 1) If no arguments or only the script name, show usage and exit.
    # if len(sys.argv) == 1:
    #     show_usage()
    #     sys.exit(0)
    
    if len(sys.argv) == 1:
        show_usage(parser, script_name="app_xtft_proba")
        sys.exit(0)
        
    # 2) Parse arguments
    args = parser.parse_args()
    
    # 3) If user specifically asked for helpdoc, show usage and exit.
    if args.helpdoc:
        print_script_info(__doc__)
        sys.exit(0)

    # 3) Otherwise, run main logic
    main(args)
    