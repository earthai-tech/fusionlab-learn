# tft_cli.py
import argparse
import logging
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

try:
    from fusionlab.core.io import read_data 
    from fusionlab.utils.ts_utils import (
        ts_validator, reshape_xtft_data, prepare_spatial_future_data
    )
    from fusionlab.nn.transformers import TemporalFusionTransformer
    from fusionlab.nn.losses import combined_quantile_loss
    from fusionlab.utils.generic_utils import setup_logging, set_random_seed
except ImportError as e:
    print(f"Error importing fusionlab components: {e}")
    print("Please ensure fusionlab is installed correctly and accessible.")
    sys.exit(1)
# --- End FusionLab Imports ---

def load_data(path):
    """Loads data, preferably CSV."""
    logging.info(f"Loading data from: {path}")
    # Example implementation:
    if not os.path.exists(path):
        logging.error(f"Data file not found: {path}")
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        # Use fusionlab's reader or pandas
        df = read_data(path) # Adjust if read_data is elsewhere
        # df = pd.read_csv(path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        raise

def preprocess_data(df, numerical_features, categorical_features, target_col,
                    scaler_type, handle_na, dt_col, output_dir=None, scalers=None):
    """Handles NA, scales numerical, encodes categorical (optional)."""
    logging.info("Starting data preprocessing...")
    df_processed = df.copy()

    # 1. Handle NA
    if handle_na == 'drop':
        df_processed.dropna(inplace=True)
    elif handle_na == 'ffill':
        df_processed.ffill(inplace=True)
    logging.info(f"Handled NA values using '{handle_na}'. Shape: {df_processed.shape}")

    # 2. Validate datetime
    df_processed, dt_col = ts_validator(
        df_processed, dt_col=dt_col, to_datetime='auto', as_index=False,
        error='raise', return_dt_col=True
        )
    logging.info(f"Validated datetime column: {dt_col}")

    # 3. Scaling Numerical Features
    num_features_to_scale = [f for f in numerical_features if f != dt_col] # Exclude dt_col
    target_included = False
    if target_col in num_features_to_scale:
         num_features_to_scale.remove(target_col)
         target_included = True

    fitted_scalers = {}
    if scaler_type:
        if scalers: # Use pre-fitted scalers (prediction mode)
            logging.info("Using pre-fitted scalers.")
            numerical_scaler = scalers.get('numerical_scaler')
            target_scaler = scalers.get('target_scaler')
            if not numerical_scaler or (target_included and not target_scaler) :
                 raise ValueError("Required scalers not found in loaded dictionary.")
        else: # Fit new scalers (training mode)
            logging.info(f"Fitting '{scaler_type}' scaler...")
            scaler_class = StandardScaler if scaler_type == 'z-norm' else MinMaxScaler
            numerical_scaler = scaler_class()
            if num_features_to_scale:
                df_processed[num_features_to_scale] = numerical_scaler.fit_transform(
                    df_processed[num_features_to_scale])
            if target_included:
                 target_scaler= scaler_class()
                 df_processed[[target_col]]= target_scaler.fit_transform(
                      df_processed[[target_col]])
            else:
                 target_scaler= None

            fitted_scalers['numerical_scaler'] = numerical_scaler
            fitted_scalers['target_scaler'] = target_scaler

            # Save scalers if in training mode
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                scaler_path = os.path.join(output_dir, "scalers.joblib")
                joblib.dump(fitted_scalers, scaler_path)
                logging.info(f"Scalers saved to {scaler_path}")
        # Apply scaling using fitted/loaded scalers
        if num_features_to_scale:
             df_processed[num_features_to_scale] = numerical_scaler.transform(
                  df_processed[num_features_to_scale])
        if target_included and target_scaler:
             df_processed[[target_col]] = target_scaler.transform(
                  df_processed[[target_col]])
        logging.info("Numerical features scaled.")
    else:
        logging.info("Skipping scaling.")
        fitted_scalers = scalers if scalers else {} # Pass through if no scaling

    # 4. Encoding Categorical Features (Placeholder - TFT might handle internally)
    # If explicit encoding is needed:
    # encoder = OneHotEncoder(...)
    # df_processed[encoded_cols] = encoder.fit_transform(...) / encoder.transform(...)
    # save/load encoder
    # fitted_scalers['encoder'] = encoder # Add encoder if used
    logging.warning("Categorical encoding placeholder - assuming model handles it or"
                    " data is pre-encoded if needed.")
    encoder = None # Placeholder

    logging.info("Preprocessing completed.")
    return df_processed, fitted_scalers, encoder

def build_tft_model(args, input_dims):
    """Builds and compiles the TemporalFusionTransformer model."""
    logging.info("Building TemporalFusionTransformer model...")
    # Example: Extract dims needed for constructor
    static_dim, dynamic_dim, future_dim = input_dims

    # Handle potential list format for lstm_units from argparse
    lstm_units_arg = args.lstm_units
    if lstm_units_arg and isinstance(lstm_units_arg, list) and len(lstm_units_arg)==1:
        lstm_units_processed = lstm_units_arg[0] # Use single int if list has one item
    elif lstm_units_arg and isinstance(lstm_units_arg, int):
         lstm_units_processed = lstm_units_arg
    else:
        # Default or handle multi-layer list if supported by your TFT version
        lstm_units_processed = 64 # Default single value if not specified or complex list
        if lstm_units_arg:
            logging.warning(f"Complex lstm_units list {lstm_units_arg} not fully"
                            f" handled by CLI default builder, using {lstm_units_processed}.")
                            
    model = TemporalFusionTransformer(
        dynamic_input_dim=dynamic_dim,
        static_input_dim=static_dim if static_dim > 0 else None,
        future_input_dim=future_dim if future_dim > 0 else None,
        hidden_units=args.hidden_units,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        forecast_horizon=args.forecast_horizon,
        quantiles=args.quantiles,
        activation=args.activation,
        use_batch_norm=args.use_batch_norm,
        num_lstm_layers=args.num_lstm_layers,
        lstm_units=lstm_units_processed # Pass the processed value
    )
    # Determine loss
    if args.quantiles:
        loss = combined_quantile_loss(args.quantiles)
    else:
        loss = 'mse' # Default for point forecasts

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss
    )
    logging.info("Model built and compiled successfully.")
    return model

def train_model(model, train_inputs, y_train, val_inputs, y_val, args):
    """Trains the model with callbacks."""
    logging.info("Starting model training...")
    output_dir = args.output_dir or "."
    model_path = os.path.join(output_dir, f"{args.model_name}_best.keras")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, save_weights_only=False)
    ]

    history = model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1 if args.verbose > 0 else 0
    )
    logging.info("Model training completed.")
    return history

def prepare_prediction_inputs(processed_history_df, args, scalers):
    """Prepares inputs for prediction using logic like prepare_spatial_future_data."""
    logging.info("Preparing inputs for future prediction...")
    # This function needs detailed implementation based on how future years/dates
    # are handled and how features are projected.
    # It should leverage prepare_spatial_future_data logic.

    # Example conceptual steps:
    # 1. Identify future time steps/years based on horizon and last date in history
    future_years = list(range(int(processed_history_df[args.dt_col].max()) + 1,
                         int(processed_history_df[args.dt_col].max()) + 1 + args.forecast_horizon))

    # 2. Call prepare_spatial_future_data (or replicate its logic)
    # Need to carefully map args to prepare_spatial_future_data params
    # Need feature columns IDENTICAL to training (including encoded)
    all_feature_cols = args.dynamic_features + (args.static_features or []) + (args.future_features or [])
    # Remove duplicates and target
    all_feature_cols = sorted(list(set(all_feature_cols) - {args.target, args.dt_col}))
    
    # Derive dynamic_feature_indices relative to `all_feature_cols`
    dynamic_indices = [i for i, col in enumerate(all_feature_cols) if col in args.dynamic_features]

    # Derive static_feature_names (ensure consistency with training)
    static_names= args.static_features # Assuming these exist in df

    static_pred, dynamic_pred, _, _, _, _ = prepare_spatial_future_data(
         final_processed_data=processed_history_df,
         feature_columns=all_feature_cols, # All features fed to sequence model
         dynamic_feature_indices=dynamic_indices, # Indices WITHIN feature_columns
         sequence_length=args.time_steps,
         dt_col=args.dt_col,
         static_feature_names=static_names, # Names of static cols
         forecast_horizon=args.forecast_horizon,
         future_years=future_years, # List of years/dates to predict
         encoded_cat_columns=None, # Assuming handled elsewhere or not applicable here
         scaling_params=scalers, # Pass loaded scalers for time scaling
         spatial_cols=args.spatial_cols,
         squeeze_last=False, # Keep last dim for model
         verbosity=args.verbose
     )
    
    # 3. Prepare Future Covariates (X_future_pred) - Crucial step!
    #    This part is tricky and highly dependent on the specific model and data.
    #    The logic from reshape_xtft_data (repeating first step) or a more
    #    sophisticated approach might be needed. For CLI simplicity, we might
    #    need to assume future covariates are either constant or loaded externally.
    #    Here, let's create a dummy one based on the last dynamic step's shape
    #    and assume it needs to match dynamic_pred's time steps.
    num_future_features = len(args.future_features or [])
    if num_future_features > 0:
         # This is a placeholder - real future features need proper handling!
         # Replicating the logic seen in reshape_xtft_data's internal calls:
         # Take the FIRST time step's future features from the last DYNAMIC sequence
         # and tile it.
         # This requires having the original non-sequenced data accessible or rethinking.
         # A simpler approach for CLI might be to REQUIRE a separate future data file.
         # Let's use zeros as a placeholder for now.
         logging.warning("Future feature preparation logic is simplified (using zeros)."
                         " Provide actual future features if needed.")
         future_pred = np.zeros((static_pred.shape[0], args.time_steps, num_future_features, 1 ))
    else:
         future_pred = None # Or handle as needed by model if dim is None

    logging.info("Prediction inputs prepared.")
    # Model expects list: [static, dynamic, future]
    # Handle cases where static/future might be None or zero-dim if model allows
    pred_inputs_list = []
    pred_inputs_list.append(static_pred.astype(np.float32))
    pred_inputs_list.append(dynamic_pred.astype(np.float32))
    if future_pred is not None:
         pred_inputs_list.append(future_pred.astype(np.float32))
    elif args.future_features: # Need future features but couldn't create them
         raise ValueError("Future features specified but could not be prepared for prediction.")
    # Else: No future features specified, don't append

    # Final check for models requiring 3 inputs
    if len(pred_inputs_list)<3 and (args.static_features or args.future_features):
         # Pad with dummy if needed, depending on model strictness
         logging.warning("Model might expect 3 inputs, padding missing static/future with zeros.")
         # This padding is complex and model-dependent. Omitted for brevity.
         # Ensure the loaded model can handle potentially fewer inputs if appropriate.
         pass

    return pred_inputs_list


def format_predictions(predictions_scaled, pred_inputs, scalers, args):
    """Inverse transforms predictions and formats into a DataFrame."""
    logging.info("Formatting predictions...")
    # 1. Inverse transform predictions
    target_scaler = scalers.get('target_scaler')
    if not target_scaler:
        logging.warning("Target scaler not found. Predictions will remain scaled.")
        predictions_inv = predictions_scaled.squeeze() # Remove trailing dim if present
    else:
        # Reshape for scaler: (samples * horizon, num_outputs_per_step)
        num_samples = predictions_scaled.shape[0]
        num_outputs = predictions_scaled.shape[-1] # Usually num_quantiles or 1
        pred_reshaped = predictions_scaled.reshape(-1, num_outputs)
        pred_inv_flat = target_scaler.inverse_transform(pred_reshaped)
        # Reshape back: (samples, horizon, num_outputs_per_step)
        predictions_inv = pred_inv_flat.reshape(num_samples, args.forecast_horizon, num_outputs)

    # 2. Create DataFrame
    # Need identifier columns (e.g., spatial cols, time)
    # This part requires access to how `prepare_prediction_inputs` stored identifiers
    # Assuming pred_inputs[0] (static) holds spatial cols if used
    pred_df = pd.DataFrame()
    num_samples = predictions_inv.shape[0]

    if args.spatial_cols:
        static_input_original_shape = pred_inputs[0] # Need original unscaled static if possible
        # Placeholder: Assuming first columns are spatial coords
        # This ideally needs inverse transform if coords were scaled
        pred_df[args.spatial_cols[0]] = static_input_original_shape[:, 0].squeeze()
        if len(args.spatial_cols) > 1:
            pred_df[args.spatial_cols[1]] = static_input_original_shape[:, 1].squeeze()

    # Add columns for each horizon step and quantile/point
    # Requires knowing the future dates/times
    # Placeholder for time - needs proper generation
    future_times = [f"t+{i+1}" for i in range(args.forecast_horizon)]

    rows = []
    for sample_idx in range(num_samples):
        base_row = {}
        if args.spatial_cols:
            base_row[args.spatial_cols[0]] = pred_df.loc[sample_idx, args.spatial_cols[0]]
            if len(args.spatial_cols) > 1:
                 base_row[args.spatial_cols[1]] = pred_df.loc[sample_idx, args.spatial_cols[1]]

        for step_idx in range(args.forecast_horizon):
            row = base_row.copy()
            row[args.dt_col or 'time_step'] = future_times[step_idx] # Use placeholder time
            if args.quantiles:
                 for q_idx, q in enumerate(args.quantiles):
                      col_name = f"{args.target}_q{int(q*100)}"
                      row[col_name] = predictions_inv[sample_idx, step_idx, q_idx]
            else: # Point forecast
                 col_name = f"{args.target}_pred"
                 row[col_name] = predictions_inv[sample_idx, step_idx, 0] # Assuming point is first/only output
            rows.append(row)

    final_pred_df = pd.DataFrame(rows)
    logging.info("Predictions formatted into DataFrame.")
    return final_pred_df

# --- End Helper Functions ---

def main_cli(args):
    """Main logic dispatcher based on mode."""
    setup_logging(args.verbose)
    set_random_seed(args.seed)
    warnings.filterwarnings('ignore')
    # Suppress excessive TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ERROR
    tf.get_logger().setLevel('ERROR')
    logging.getLogger('tensorflow').setLevel(logging.ERROR)


    if args.mode == 'train':
        # --- Training Workflow ---
        logging.info("Starting Training Workflow...")
        # 1. Load Data
        df = load_data(args.data)

        # 2. Preprocess
        numerical_features = list(args.numerical_features) if args.numerical_features else []
        categorical_features = list(args.categorical_features) if args.categorical_features else []
        processed_df, scalers, encoder = preprocess_data(
            df, numerical_features, categorical_features, args.target,
            args.scaler, args.handle_na, args.dt_col, args.output_dir
        )

        # 3. Define features for reshaping (ensure consistency)
        static_cols = list(args.static_features) if args.static_features else None
        dynamic_cols = list(args.dynamic_features) if args.dynamic_features else None
        future_cols = list(args.future_features) if args.future_features else None
        spatial_cols = list(args.spatial_cols) if args.spatial_cols else None
        
        if dynamic_cols is None:
             raise ValueError("`--dynamic_features` are required for training.")

        # 4. Reshape into sequences
        logging.info("Reshaping data into sequences...")
        static_data, dynamic_data, future_data, target_data = reshape_xtft_data(
            df=processed_df, # Use processed data
            dt_col=args.dt_col,
            target_col=args.target,
            dynamic_cols=dynamic_cols,
            static_cols=static_cols,
            future_cols=future_cols,
            spatial_cols=spatial_cols, # Pass spatial cols if provided
            time_steps=args.time_steps,
            forecast_horizons=args.forecast_horizon,
            verbose=args.verbose
        )

        # Basic validation of reshaped data
        if dynamic_data is None or target_data is None:
             raise ValueError("Failed to create dynamic sequences or targets.")
        logging.info(f"Sequence shapes: Dynamic={dynamic_data.shape},"
                      f" Target={target_data.shape}")
        if static_data is not None:
             logging.info(f"Static shape: {static_data.shape}")
        if future_data is not None:
             logging.info(f"Future shape: {future_data.shape}")
             
        # 5. Split Train/Val
        logging.info(f"Splitting data with validation split: {args.validation_split}")
        # Need to handle optional static/future data in split
        # This requires careful indexing or a dedicated split function
        indices = np.arange(dynamic_data.shape[0])
        train_indices, val_indices = train_test_split(
             indices, test_size=args.validation_split, shuffle=False # Time series split
        )

        X_train_dynamic = dynamic_data[train_indices]
        X_val_dynamic = dynamic_data[val_indices]
        y_train = target_data[train_indices]
        y_val = target_data[val_indices]

        X_train_static = static_data[train_indices] if static_data is not None else None
        X_val_static = static_data[val_indices] if static_data is not None else None
        X_train_future = future_data[train_indices] if future_data is not None else None
        X_val_future = future_data[val_indices] if future_data is not None else None

        # Package inputs for model training
        train_inputs = [d for d in [X_train_static, X_train_dynamic, X_train_future] if d is not None]
        val_inputs = [d for d in [X_val_static, X_val_dynamic, X_val_future] if d is not None]

        # Determine input dimensions for model building
        # Handle case where static/future might be None
        static_dim = X_train_static.shape[-1] if X_train_static is not None else 0 # Use feature dim
        dynamic_dim = X_train_dynamic.shape[-1] # Feature dim
        future_dim = X_train_future.shape[-1] if X_train_future is not None else 0 # Feature dim

        # Ensure dimensions match model expectations (e.g., TFT might need >=1 dim)
        # The TFT code might handle None dims, check its __init__
        if static_dim==0 and args.static_features:
             logging.warning("Static features specified but data has 0 dim after reshaping.")
             # Adjust or raise error based on model requirements
        if future_dim==0 and args.future_features:
             logging.warning("Future features specified but data has 0 dim after reshaping.")

        # 6. Build Model
        model = build_tft_model(args, (static_dim, dynamic_dim, future_dim))

        # 7. Train Model
        train_model(model, train_inputs, y_train, val_inputs, y_val, args)

        logging.info("Training Workflow Completed.")

    elif args.mode == 'predict':
        # --- Prediction Workflow ---
        logging.info("Starting Prediction Workflow...")
        # 1. Load Model
        if not args.load_model_path or not os.path.exists(args.load_model_path):
            raise ValueError("`--load_model_path` must be provided and exist for prediction.")
        logging.info(f"Loading model from: {args.load_model_path}")
        # Ensure custom objects are passed if needed (e.g., custom loss)
        custom_objects = {'TemporalFusionTransformer': TemporalFusionTransformer}
        if args.quantiles:
             # Need to register the specific quantile loss function used during training
             loss_fn_name= f"combined_quantile_loss_{'_'.join(map(str, args.quantiles))}" # Example name
             # Or pass the function directly if possible
             custom_objects['combined_quantile_loss'] = combined_quantile_loss(args.quantiles) # Pass the actual function

        model = tf.keras.models.load_model(args.load_model_path, custom_objects=custom_objects)

        # 2. Load Scalers
        if not args.load_scalers_path or not os.path.exists(args.load_scalers_path):
             raise ValueError("`--load_scalers_path` must be provided and exist for prediction.")
        logging.info(f"Loading scalers from: {args.load_scalers_path}")
        scalers = joblib.load(args.load_scalers_path)
        encoder = scalers.get('encoder') # Load encoder if saved

        # 3. Load Historical Data for context/preparation
        if not args.data:
             raise ValueError("Historical data path (`--data`) required for preparing prediction inputs.")
        history_df = load_data(args.data)

        # 4. Preprocess Historical Data using LOADED scalers
        logging.info("Preprocessing historical data using loaded scalers...")
        numerical_features = list(args.numerical_features) if args.numerical_features else []
        categorical_features = list(args.categorical_features) if args.categorical_features else []

        processed_history_df, _, _ = preprocess_data(
             history_df, numerical_features, categorical_features, args.target,
             args.scaler, args.handle_na, args.dt_col, scalers=scalers # Pass loaded scalers
        )

        # 5. Prepare Prediction Inputs using logic from prepare_spatial_future_data
        # This requires careful mapping of arguments and potentially loading encoded cols
        logging.info("Preparing prediction inputs based on historical data...")
        pred_inputs = prepare_prediction_inputs(processed_history_df, args, scalers)

        # 6. Predict
        logging.info("Generating predictions...")
        predictions_scaled = model.predict(pred_inputs)

        # 7. Inverse Transform & Format
        logging.info("Inverse transforming predictions and formatting output...")
        predictions_df = format_predictions(predictions_scaled, pred_inputs, scalers, args)

        # 8. Save Predictions
        if not args.predictions_output_file:
             raise ValueError("`--predictions_output_file` must be specified for saving predictions.")
        predictions_df.to_csv(args.predictions_output_file, index=False)
        logging.info(f"Predictions saved to {args.predictions_output_file}")
        logging.info("Prediction Workflow Completed.")
    else:
        logging.error(f"Invalid mode specified: {args.mode}. Choose 'train' or 'predict'.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command-Line Interface for Temporal Fusion Transformer (TFT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Mode ---
    parser.add_argument(
        '--mode', type=str, required=True, choices=['train', 'predict'],
        help="Operation mode: 'train' a new model or 'predict' using a saved one."
    )

    # --- Common Arguments ---
    parser.add_argument(
        '--data', type=str, required=True,
        help="Path to the main dataset CSV file (used for training history or prediction context)."
    )
    parser.add_argument(
        '--target', type=str, required=True,
        help="Name of the target variable column to predict."
    )
    parser.add_argument(
        '--dt_col', type=str, required=True,
        help="Name of the datetime column."
    )
    parser.add_argument(
        '--dynamic_features', type=str, nargs='+', required=True,
        help="List of dynamic (time-varying) feature column names."
    )
    parser.add_argument(
        '--static_features', type=str, nargs='+', default=None,
        help="List of static (time-invariant) feature column names (optional)."
    )
    parser.add_argument(
        '--future_features', type=str, nargs='+', default=None,
        help="List of known future feature column names (optional)."
    )
    parser.add_argument(
        '--categorical_features', type=str, nargs='+', default=None,
        help="List of categorical features (subset of dynamic/static/future) "
             "that might require special handling or embedding."
    )
    parser.add_argument(
        '--spatial_cols', type=str, nargs='+', default=None,
        help="List of spatial identifier columns (e.g., longitude latitude) "
             "for grouping data (used in reshaping/prediction prep)."
    )
    parser.add_argument(
        '--time_steps', type=int, default=10,
        help="Lookback sequence length (number of past time steps for input)."
    )
    parser.add_argument(
        '--forecast_horizon', type=int, default=1,
        help="Prediction horizon length (number of future steps to predict)."
    )
    parser.add_argument(
        '--output_dir', type=str, default="./tft_output",
        help="Directory to save models, scalers, logs during training."
             " Also used as base for default prediction output file."
    )
    parser.add_argument(
        '--verbose', type=int, default=1, choices=[0, 1, 2],
        help="Verbosity level (0: WARNING, 1: INFO, 2: DEBUG)."
    )
    parser.add_argument(
        '--seed', type=int, default=42, help="Random seed for reproducibility."
    )

    # --- Preprocessing Arguments ---
    parser.add_argument(
        '--scaler', type=str, default='z-norm', choices=['z-norm', 'minmax', 'none'],
        help="Scaler for numerical features ('z-norm': StandardScaler, "
             "'minmax': MinMaxScaler, 'none': no scaling)."
    )
    parser.add_argument(
        '--handle_na', type=str, default='ffill', choices=['drop', 'ffill'],
        help="Strategy for handling missing values ('drop' rows, 'ffill')."
    )

    # --- Model Hyperparameters ---
    parser.add_argument(
        '--hidden_units', type=int, default=32, help="Hidden units for GRNs/Dense layers."
    )
    parser.add_argument(
        '--num_heads', type=int, default=4, help="Number of attention heads."
    )
    parser.add_argument(
        '--dropout_rate', type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        '--quantiles', type=float, nargs='+', default=None,
        help="List of quantiles for probabilistic forecast (e.g., 0.1 0.5 0.9)."
             " If None, performs point forecast using MSE loss."
    )
    parser.add_argument(
        '--activation', type=str, default='elu', help="Activation function (e.g., 'relu', 'elu')."
    )
    parser.add_argument(
        '--use_batch_norm', action='store_true', help="Use Batch Normalization in GRNs."
    )
    parser.add_argument(
        '--num_lstm_layers', type=int, default=1, help="Number of LSTM layers."
    )
    parser.add_argument(
        '--lstm_units', type=int, nargs='+', default=None, # Allow list for future? Use single for now.
        help="Units per LSTM layer. Provide one value for consistent units."
             " If None, defaults are used (e.g., hidden_units)."
    )

    # --- Training Arguments ---
    parser.add_argument(
        '--epochs', type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help="Training batch size."
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001, help="Optimizer learning rate."
    )
    parser.add_argument(
        '--optimizer', type=str, default='adam', help="Optimizer name (e.g., 'adam')."
    )
    parser.add_argument(
        '--validation_split', type=float, default=0.2,
        help="Fraction of training data to use for validation (chronological split)."
    )
    # parser.add_argument('--val_data', type=str, default=None, help='Path to separate validation data CSV.') # Alternative split method
    parser.add_argument(
        '--patience', type=int, default=10, help="Early stopping patience."
    )
    parser.add_argument(
        '--model_name', type=str, default='tft_model', help="Base name for saved model file."
    )
    # parser.add_argument('--save_scalers', action='store_true', help="Save fitted scalers.") # Handled internally
    # parser.add_argument('--save_model', action='store_true', help="Save the best trained model.") # Handled by ModelCheckpoint

    # --- Prediction Arguments ---
    parser.add_argument(
        '--load_model_path', type=str, default=None,
        help="Path to load a pre-trained Keras model file for prediction."
    )
    parser.add_argument(
        '--load_scalers_path', type=str, default=None,
        help="Path to load pre-fitted scalers (.joblib file) for prediction."
    )
    # parser.add_argument('--predict_input_data', type=str, default=None, help='Path to data file for prediction.') # Use --data instead for context
    parser.add_argument(
        '--predictions_output_file', type=str, default=None,
        help="Path to save the generated predictions CSV file."
             " Defaults to '<output_dir>/<model_name>_predictions.csv'"
    )

    args = parser.parse_args()

    # --- Post-processing/Defaults for Prediction Output ---
    if args.mode == 'predict' and args.predictions_output_file is None:
        output_dir = args.output_dir or "."
        model_name = os.path.splitext(os.path.basename(args.load_model_path or 'model'))[0] if args.load_model_path else args.model_name
        args.predictions_output_file = os.path.join(output_dir, f"{model_name}_predictions.csv")
        os.makedirs(os.path.dirname(args.predictions_output_file), exist_ok=True)

    # --- Run Main Logic ---
    main_cli(args)