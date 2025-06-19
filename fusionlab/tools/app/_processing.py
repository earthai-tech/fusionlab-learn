# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides a configuration class to manage all parameters for the
subsidence forecasting GUI and its underlying processing script.
"""
import os
import numpy as np
import pandas as pd
import math 
from typing import List, Optional, Dict, Any, Callable
from typing import Tuple
import joblib
import shutil # noqa

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from fusionlab.utils.generic_utils import ensure_directory_exists, save_all_figures
from fusionlab.utils.generic_utils import normalize_time_column 
from fusionlab.datasets import fetch_zhongshan_data
from fusionlab.utils.data_utils import nan_ops
from fusionlab.utils.io_utils import save_job
from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences, format_pinn_predictions
from fusionlab.nn.utils import extract_batches_from_dataset # noqa
from fusionlab.nn import KERAS_DEPS
from fusionlab.params import LearnableK, LearnableSs, LearnableQ
from fusionlab.nn.losses import combined_quantile_loss
from fusionlab.nn.models import PIHALNet, TransFlowSubsNet
from fusionlab.plot.forecast import plot_forecasts, forecast_view

Callback =KERAS_DEPS.Callback 
Dataset = KERAS_DEPS.Dataset
AUTOTUNE = KERAS_DEPS.AUTOTUNE
EarlyStopping = KERAS_DEPS.EarlyStopping
ModelCheckpoint = KERAS_DEPS.ModelCheckpoint
load_model = KERAS_DEPS.load_model
custom_object_scope = KERAS_DEPS.custom_object_scope
Adam = KERAS_DEPS.Adam
MeanSquaredError = KERAS_DEPS.MeanSquaredError
experimental =KERAS_DEPS.experimental 

#%
class SubsConfig:
    """
    A configuration class to manage parameters for the subsidence
    forecasting GUI and the underlying processing script.

    This class centralizes all configurable options, from file paths
    and model names to feature definitions and training parameters,
    making it easy to manage and pass settings between the GUI and
    the backend processing logic.
    """
    def __init__(
            self, log_callback: Optional[callable] = None, 
            progress_callback: Optional[Callable[[int], None]] = None,
        **kwargs):
        """
        Initializes the configuration with default values, which can be
        overridden by keyword arguments.
        """
        # --- File and Path Configuration ---
        self.city_name: str = 'zhongshan'
        self.model_name: str = 'TransFlowSubsNet'
        self.data_dir: str = os.getenv("JUPYTER_PROJECT_ROOT", "../..")
        self.data_filename: str = "zhongshan_500k.csv"
        self.output_dir: str = os.path.join(os.getcwd(), "results_pinn")

        # --- Time and Horizon Configuration ---
        self.train_end_year: int = 2022
        self.forecast_start_year: int = 2023
        self.forecast_horizon_years: int = 3
        self.time_steps: int = 5 # Lookback window
        
        
        # --- PINN and Model Configuration ---
        self.pde_mode: str = 'both'
        # self.pinn_coeff_c: str = 'learnable'
        # self.pde_mode: str = 'consolidation'
        self.pinn_coeff_c: str = 'learnable'
        self.lambda_pde: float = 1.0
        self.lambda_cons: float = 1.0
        self.lambda_gw: float = 1.0
        self.gwflow_init_k: float = 1e-4
        self.gwflow_init_ss: float = 1e-5
        self.gwflow_init_q: float = 0.0
        
        self.save_format : str = 'keras'
        
        self.quantiles: Optional[List[float]] = [0.1, 0.5, 0.9]
        self.mode: str = 'pihal'
        self.attention_levels: List[str] = ['1', '2', '3']
        
        # --- Training Hyperparameters ---
        self.epochs: int = 50
        self.learning_rate: float = 0.001
        self.batch_size: int = 256
        self.patience : int =15
        self.fit_verbose: int = 1 
        
        # --- Data Splitting ---
        self.validation_size: float = 0.2
        self.seed: int = 42

        self.weight_subs_pred = 1.0 
        self.weight_gwl_pred =.5 
        
        # --- Evaluation ---
        self.evaluate_coverage: bool = True

        # --- Column Definitions ---
        # User can set these to 'auto' for detection or provide specific lists.
        self.time_col: str = 'auto' # e.g., 'year'
        self.lon_col: str = 'longitude'
        self.lat_col: str = 'latitude'
        self.subsidence_col: str = 'subsidence'
        self.gwl_col: str = 'GWL'
        self.categorical_cols: Optional[List[str]] = 'auto' # or user can provide like ['geology']
        self.numerical_cols: Optional[List[str]] = 'auto'
        self.static_features: Optional[List[str]] = 'auto'
        self.dynamic_features: Optional[List[str]] = 'auto'
        self.future_features: Optional[List[str]] = ['rainfall_mm'] # Example

        # --- Internal State ---
        self.run_output_path: Optional[str] = None
        self.save_intermediate: bool = True
        self.include_gwl_in_df: bool = False # Control GWL inclusion in plots
        # --- Progress & logging hooks -----------------------------------
        self.log: Callable[[str], None]       = log_callback or print
        self.progress_callback: Callable[[int], None] = (
            progress_callback or (lambda *_: None)   # ← safe no-op default
        )
        self.verbose = 1 
        
        # Override defaults with any user-provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self._build_paths()

    def _build_paths(self):
        """Constructs the full run output path based on current config."""
        self.run_output_path = os.path.join(
            self.output_dir, f"{self.city_name}_{self.model_name}_run"
        )

    def update_from_gui(self, gui_config: Dict[str, Any]):
        """Updates configuration from a dictionary, e.g., from the GUI state."""
        for key, value in gui_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._build_paths() # Re-build paths in case model name changed
        
    def auto_detect_columns(self, df: pd.DataFrame ):
        """
        Automatically detects and sets column lists (categorical, numerical,
        static, dynamic) if they are set to 'auto'. This method respects
        user-provided lists and ensures special columns are handled correctly.
        """
        self.log("Auto-detecting column types and roles...")
        
        # --- Step 1: Detect Time Column (if set to 'auto') ---
        if self.time_col == 'auto':
            # Use a predefined list of common time-related column names
            for col in ['year', 'date', 'time', 'timestamp']:
                if col in df.columns:
                    self.time_col = col
                    self.log(f"  Detected time column: '{self.time_col}'")
                    break
            if self.time_col == 'auto':
                raise ValueError("Could not auto-detect time column. Please "
                                 "specify it in the configuration.")

        # --- Step 2: Detect Categorical Columns (if set to 'auto') ---
        if self.categorical_cols == 'auto':
            categoricals = df.select_dtypes(exclude=np.number).columns.tolist()
            # Ensure the main time column is not treated as a category
            if self.time_col in categoricals:
                categoricals.remove(self.time_col)
            self.categorical_cols = categoricals
            self.log(f"  Auto-detected categorical columns: {self.categorical_cols}")

        # --- Step 3: Detect Numerical Columns (if set to 'auto') ---
        if self.numerical_cols == 'auto':
            numerics = df.select_dtypes(include=np.number).columns.tolist()
            # Ensure the main time column is not treated as a numerical feature
            if self.time_col in numerics:
                numerics.remove(self.time_col)
            self.numerical_cols = numerics
            self.log(f"  Auto-detected numerical columns: {self.numerical_cols}")

        # --- Step 4: Determine Feature Roles (if set to 'auto') ---
        if self.static_features == 'auto':
            # Heuristic: Assume that the identified categorical columns
            # (which will be one-hot encoded) are static features.
            self.static_features = self.categorical_cols or []
            self.log(f"  Auto-set static features: {self.static_features}")

        if self.dynamic_features == 'auto':
            # Heuristic: Assume all other numerical columns are dynamic,
            # after excluding special ID, target, and known future columns.
            exclude_cols = {
                self.time_col, self.lon_col, self.lat_col,
                self.subsidence_col, self.gwl_col
            }
            # Also exclude columns already assigned to be static or future
            exclude_cols.update(self.static_features or [])
            exclude_cols.update(self.future_features or [])
            
            # Use the detected numerical_cols list as the base
            self.dynamic_features = [
                col for col in self.numerical_cols if col not in exclude_cols
            ]
            self.log(f"  Auto-set dynamic features: {self.dynamic_features}")


    def __repr__(self) -> str:
        """Provides a string representation of the configuration."""
        params = "\n".join(f"  {key}: {value}" for key, value in self.__dict__.items())
        return f"SubsConfig(\n{params}\n)"


class DataProcessor:
    """
    Handles the data loading and preprocessing workflow (Steps 1-5).
    """
    def __init__(
            self, config: SubsConfig, 
            log_callback: Optional[callable] = None):
        """
        Initializes the processor with a configuration object.

        Args:
            config (SubsConfig): The configuration object with all parameters.
            log_callback (callable, optional): A function to receive log messages.
        """
        self.config = config
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.log = log_callback or print
        
        # To store fitted objects
        self.encoder = None
        self.scaler = None

    def load_data(self) -> pd.DataFrame:
        """
        Handles Step 1: Loading the dataset from a file or by fetching it.
        """
        self.log("Step 1: Loading Dataset...")
        data_path = os.path.join(
            self.config.data_dir, self.config.data_filename)
        
        if os.path.exists(data_path):
            try:
                self.raw_df = pd.read_csv(data_path)
                self.log(f"  Successfully loaded '{self.config.data_filename}'."
                         f" Shape: {self.raw_df.shape}")
            except Exception as e:
                raise IOError(f"Error loading data from '{data_path}': {e}")
        else:
            self.log("  Local file not found. Attempting to fetch sample data...")
            try:
                data_bunch = fetch_zhongshan_data()
                self.raw_df = data_bunch.frame
                self.log(f"  Successfully fetched sample data. Shape: {self.raw_df.shape}")
            except Exception as e:
                raise FileNotFoundError(
                    "Data could not be loaded from local paths or fetched."
                    f" Error: {e}"
                )
        
        if self.config.save_intermediate:
            ensure_directory_exists(self.config.run_output_path)
            save_path = os.path.join(self.config.run_output_path, "01_raw_data.csv")
            self.raw_df.to_csv(save_path, index=False)
            self.log(f"  Saved raw data artifact to: {save_path}")
            
        return self.raw_df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Handles Steps 2-4: Feature selection, cleaning, encoding, and scaling.
        """
        if self.raw_df is None:
            raise RuntimeError("Raw data must be loaded before preprocessing.")
        
        self.log("Step 2 & 3: Preprocessing Data...")
        
        # --- Auto-detect columns if needed ---
        if self.config.time_col == 'auto' or self.config.categorical_cols == 'auto':
            self.config.auto_detect_columns(self.raw_df)
        
        # --- Feature Selection ---
        base_cols = [
            self.config.lon_col, self.config.lat_col, self.config.time_col,
            self.config.subsidence_col, self.config.gwl_col
        ]
        all_cols = base_cols + (self.config.categorical_cols or []) + \
                   (self.config.numerical_cols or []) + (self.config.future_features or [])
        
        df_selected = self.raw_df[[
            col for col in set(all_cols) if col in self.raw_df.columns
        ]].copy()
        
        # --- Cleaning ---
        self.log("  Cleaning NaN values...")
        df_cleaned = nan_ops(df_selected, ops='sanitize', action='fill')
        
        # --- Encoding ---
        df_processed = df_cleaned
        if self.config.categorical_cols:
            self.log(f"  One-hot encoding: {self.config.categorical_cols}")
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
            encoded_data = self.encoder.fit_transform(df_cleaned[self.config.categorical_cols])
            # encoded_cols = self.encoder.get_feature_names_out(self.config.categorical_cols)
            
            self.static_features_encoded = self.encoder.get_feature_names_out(
                self.config.categorical_cols).tolist()
            encoded_df = pd.DataFrame(
                encoded_data, columns=self.static_features_encoded, index=df_cleaned.index)
            
            df_processed = pd.concat([
                df_cleaned.drop(columns=self.config.categorical_cols),
                encoded_df
                # pd.DataFrame(
                #     encoded_data, columns=encoded_cols, index=df_cleaned.index)
            ], axis=1)

        # --- Time Coordinate and Normalization ---
        self.log("  Creating and normalizing time coordinate...")
        # Ensure time column is datetime for processing
        temp_dt_col = 'datetime_temp'
        try: 
            
            df_processed[temp_dt_col] = pd.to_datetime(
                df_processed[self.config.time_col], errors='coerce')
            df_processed['time_numeric'] = (
                df_processed[temp_dt_col].dt.year + 
                (df_processed[temp_dt_col].dt.dayofyear - 1) / 366.0
            )
            df_processed.drop(columns=[temp_dt_col], inplace=True)
        except: 
            # Convert year to datetime for consistent processing, then to numeric for PINN
            # print(df_processed.columns)
            df_processed = normalize_time_column(
                df_processed, time_col= self.config.time_col, 
                datetime_col= temp_dt_col, 
                year_col= self.config.time_col,
                drop_orig= True 
            )
 
        # --- Scaling ---
        self.log("  Scaling numerical features...")
        cols_to_scale = [self.config.subsidence_col, self.config.gwl_col] + \
                        (self.config.future_features or [])
        self.scaler = MinMaxScaler()
        # Filter to only scale columns that actually exist
        existing_cols_to_scale = [
            col for col in cols_to_scale if col in df_processed.columns]
        if existing_cols_to_scale:
            df_processed[existing_cols_to_scale] = self.scaler.fit_transform(
                df_processed[existing_cols_to_scale]
            )
        
        self.processed_df = df_processed
        self.log("  Data preprocessing complete.")
        
        if self.config.save_intermediate:
            # Save artifacts
            save_path = os.path.join(
                self.config.run_output_path, "02_processed_data.csv"
                )
            self.processed_df.to_csv(save_path, index=False)
            self.log(f"  Saved processed data to: {save_path}")
            if self.encoder:
                save_job(self.encoder, os.path.join(
                    self.config.run_output_path, "ohe_encoder.joblib"))
            if self.scaler:
                save_job(self.scaler, os.path.join(
                    self.config.run_output_path, "main_scaler.joblib"))
                
        return self.processed_df

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handles Step 5: Splitting the data into training and test sets."""
        if self.processed_df is None:
            raise RuntimeError("Data must be preprocessed before splitting.")

        self.log("Step 4 & 5: Defining Feature Sets and Splitting Data...")
        
        # Use the config to perform the split using the configured time column
        time_col = self.config.time_col
        self.train_df = self.processed_df[
            self.processed_df[time_col] <= self.config.train_end_year
        ].copy()
        
        self.test_df = self.processed_df[
            self.processed_df[time_col] >= self.config.train_end_year
        ].copy()
        
        self.log(f"  Data split complete. Train shape: {self.train_df.shape},"
                 f" Test shape: {self.test_df.shape}")
                 
        return self.train_df, self.test_df

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the full data processing pipeline from loading to splitting.
        """
        self.load_data()
        return self.preprocess_data()
        # self.split_data()
        
        # return self.train_df, self.test_df

class SequenceGenerator:
    """
    Handles sequence generation and dataset creation (Steps 5-6).
    """
    def __init__(
            self, config: SubsConfig, log_callback: Optional[callable] = None):
        """
        Initializes the generator with a configuration object.
        """
        self.config = config
        self.log = log_callback or print
        self.inputs_train: Optional[Dict[str, np.ndarray]] = None
        self.targets_train: Optional[Dict[str, np.ndarray]] = None
        self.coord_scaler = None
        self.train_df = None
        self.test_df = None

    def run(
            self, processed_df: pd.DataFrame, 
            static_features_encoded: List[str]) -> Tuple[Any, Any]:
        """
        Executes the full sequence and dataset creation pipeline.
        
        Returns:
            A tuple of (train_dataset, validation_dataset).
        """
        self._split_data(processed_df)
        self._generate_sequences(self.train_df, static_features_encoded)
        return self._create_tf_datasets()


    def _split_data(
            self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handles Step 5: Splitting the data into training and test sets
        robustly.
        """
        if df is None:
            raise RuntimeError("Data must be preprocessed before splitting.")

        self.log("Step 4 & 5: Splitting master data...")
        
        time_col = self.config.time_col
        if time_col not in df.columns:
            raise ValueError(
                f"Time column '{time_col}' not found in DataFrame.")

        # --- Robust time conversion for comparison ---
        # Convert the column to datetime objects, coercing any errors to NaT
        datetime_series = pd.to_datetime(df[time_col], errors='coerce')
        
        # Extract the year as a numerical series for reliable comparison
        year_series = datetime_series.dt.year
        
        # Check if conversion resulted in NaNs, which indicates a format issue
        if year_series.isnull().any():
            self.log(f"  [Warning] Could not parse all values in time column"
                     f" '{time_col}' as dates. Rows with invalid date formats"
                     " will be dropped.")
            # Create a mask to keep only valid rows
            valid_mask = year_series.notna()
            df = df[valid_mask].copy()
            year_series = year_series[valid_mask]

        # Use the numerical year series for the split
        train_mask = year_series <= self.config.train_end_year
        test_mask = year_series >= self.config.forecast_start_year
        
        self.train_df = df[train_mask].copy()
        # For the test set, we need to include a lookback period
        # for sequence generation. The `prepare_pinn_data_sequences`
        # handles the windowing, so we just need to provide the data
        # from the start of the forecast period. The sequence generator
        # will look back from there.
        self.test_df = df[test_mask].copy()
        
        self.log(f"  Data split complete. Train shape: {self.train_df.shape},"
                 f" Test shape: {self.test_df.shape}")
                 
        if self.train_df.empty:
            raise ValueError(
                f"Training data is empty after splitting on year <="
                f" {self.config.train_end_year}."
            )
            
        return self.train_df, self.test_df
    
    def _generate_sequences(
            self, train_master_df: pd.DataFrame, 
            static_features: List[str]
        ):
        """Generates PINN-compatible sequences from the training data."""
        self.log("  Generating PINN training sequences...")
        
        # Define feature roles for the sequence generator
        dynamic_features = [
            self.config.gwl_col, 'rainfall_mm', 
            # 'normalized_density'
        ] or self.dynamic_features 
        dynamic_features = [
            c for c in dynamic_features if c in train_master_df.columns]
        
        inputs, targets, scaler = prepare_pinn_data_sequences(
            df=train_master_df,
            time_col='time_numeric',
            lon_col=self.config.lon_col,
            lat_col=self.config.lat_col,
            subsidence_col=self.config.subsidence_col,
            gwl_col=self.config.gwl_col,
            dynamic_cols=dynamic_features,
            static_cols=static_features or self.static_features,
            future_cols=self.config.future_features,
            group_id_cols=[self.config.lon_col, self.config.lat_col],
            time_steps=self.config.time_steps,
            forecast_horizon=self.config.forecast_horizon_years,
            output_subsidence_dim=1,
            output_gwl_dim=1,
            normalize_coords=True,
            return_coord_scaler=True,
            mode=self.config.mode,
            verbose=self.config.verbose,  # Can be linked to a config verbose level, 
            _logger = self.log 
        )

        if targets['subsidence'].shape[0] == 0:
            raise ValueError(
                "Sequence generation produced no training samples.")
            
        self.inputs_train = inputs
        self.targets_train = targets
        self.coord_scaler = scaler
        self.log(
            "  Training sequences generated successfully."
            )

    def _create_tf_datasets(self) -> Tuple[Any, Any]:
        """Creates and splits tf.data.Dataset objects."""
        if self.inputs_train is None:
            raise RuntimeError("Sequences must be generated before creating datasets.")
            
        self.log("Step 6: Creating TensorFlow Datasets...")
        
        num_samples = self.inputs_train['dynamic_features'].shape[0]
        
        # --- handle optional features safely -----------------------------------
        static_feats = self.inputs_train.get('static_features')
        if static_feats is None:
            static_feats = np.zeros((num_samples, 0), dtype=np.float32)
    
        future_feats = self.inputs_train.get('future_features')
        if future_feats is None:
            future_feats = np.zeros(
                (num_samples, self.config.forecast_horizon_years, 0),
                dtype=np.float32,
            )
            
        dataset_inputs = {
        'coords':            self.inputs_train['coords'],
        'dynamic_features':  self.inputs_train['dynamic_features'],
        'static_features':   static_feats,
        'future_features':   future_feats,
        }
        dataset_targets = {
            'subs_pred': self.targets_train['subsidence'],
            'gwl_pred':  self.targets_train['gwl'],
        }
        
        full_dataset = Dataset.from_tensor_slices((dataset_inputs, dataset_targets))
        
        total_size = num_samples
        val_size = int(self.config.validation_size * total_size)
        train_size = total_size - val_size
        
        full_dataset = full_dataset.shuffle(
            buffer_size=total_size, seed=self.config.seed )
        train_dataset = full_dataset.take(train_size).batch(
            self.config.batch_size).prefetch(AUTOTUNE)
        val_dataset = full_dataset.skip(train_size).batch(
            self.config.batch_size).prefetch(AUTOTUNE)
        
        self.log(f"  Training dataset created with {train_size} samples.")
        self.log(f"  Validation dataset created with {val_size} samples.")
        
        return train_dataset, val_dataset


class ModelTrainer:
    """
    Handles model definition, compilation, training, and loading.
    """
    def __init__(
            self, config: SubsConfig,
            log_callback: Optional[callable] = None):
        """
        Initializes the trainer with a configuration object.
        """
        self.config = config
        self.log = log_callback or print
        self.model: Optional[Any] = None
        self.history: Optional[Any] = None

    def run(
        self,
        train_dataset: Any,
        val_dataset: Any,
        input_shapes: Dict[str, Tuple]
    ) -> Any:
        """
        Executes the full model training pipeline.
        """
        self.log("Step 7: Defining, Compiling, and Training the Model...")
        self._define_and_compile_model(input_shapes)
        self._train_model(train_dataset, val_dataset)
        return self._load_best_model()

    def _define_and_compile_model(self, input_shapes: Dict[str, Tuple]):
        """Defines the model architecture and compiles it."""
        self.log("  Defining model architecture...")
        
        # Determine which model class to use
        ModelClass = ( 
            TransFlowSubsNet if self.config.model_name == 'TransFlowSubsNet'
            else PIHALNet
        )
        # Prepare parameters for the model constructor
        model_params = {
            'static_input_dim': input_shapes['static_features'][-1],
            'dynamic_input_dim': input_shapes['dynamic_features'][-1],
            'future_input_dim': input_shapes['future_features'][-1],
            'output_subsidence_dim': 1,
            'output_gwl_dim': 1,
            'forecast_horizon': self.config.forecast_horizon_years,
            'quantiles': self.config.quantiles,
            'pde_mode': self.config.pde_mode,
            'pinn_coefficient_C': self.config.pinn_coeff_c,
            'mode': self.config.mode,
            'max_window_size': self.config.time_steps,
        }
        
        physics_loss_weights = {}
        if ModelClass is TransFlowSubsNet:
            model_params.update({
                "K": LearnableK(initial_value=self.config.gwflow_init_k),
                "Ss": LearnableSs(initial_value=self.config.gwflow_init_ss),
                "Q": LearnableQ(initial_value=self.config.gwflow_init_q),
            })
            physics_loss_weights = {
                "lambda_cons": self.config.lambda_cons,
                "lambda_gw": self.config.lambda_gw
            }
        else: # PIHALNet
            physics_loss_weights = {"lambda_pde": self.config.lambda_pde}

        self.model = ModelClass(**model_params)
        
        # Compile the model
        self.log("  Compiling model...")
        loss_dict = {
            'subs_pred': ( 
                'mse' if self.config.quantiles is None 
                else combined_quantile_loss(self.config.quantiles)
                ),
            'gwl_pred': ( 
                'mse' if self.config.quantiles is None 
                else combined_quantile_loss(self.config.quantiles)
                )
        }
        metrics_dict = {'subs_pred': ['mae', 'mse'], 'gwl_pred': ['mae', 'mse']}
        loss_weights_dict = {
            'subs_pred':self.config.weight_subs_pred, 
            'gwl_pred': self.config.weight_gwl_pred
        }
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=loss_dict,
            metrics=metrics_dict,
            loss_weights=loss_weights_dict,
            **physics_loss_weights
        )
        self.log("  Model compiled successfully.")

    
    def _train_model(self, train_dataset, val_dataset):
        self.log(f"  Starting model training for {self.config.epochs} epochs...")
    
        # ── 1.  Decide checkpoint format / path....
        fmt = (getattr(self.config, "save_format", "keras") or "keras").lower()
        if fmt not in {"keras", "tf"}:
            self.log(f"[WARNING] Unknown save_format '{fmt}', falling back to 'keras'")
            fmt = "keras"
        
        if fmt == "keras":
            ckpt_name   = f"{self.config.model_name}.keras"
            ckpt_kwargs = {}                      # native .keras saving
        else:  # fmt == "tf"
            ckpt_name   = f"{self.config.model_name}_ckpt"  # directory name
            ckpt_kwargs = {"save_format": "tf"}   # SavedModel
        
        
        # checkpoint_path = os.path.join(
        #     self.config.run_output_path, f"{self.config.model_name}.keras"
        # )
        checkpoint_path = os.path.join(self.config.run_output_path, ckpt_name)
        self.log(
            "  Model checkpoints will be saved to:"
            f" {checkpoint_path}  (format = {fmt})")
    
        # -Keras housekeeping callbacks --
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config.patience,
            restore_best_weights=True,
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            **ckpt_kwargs,              # ← inject format if 'tf'
        )
        self.log(f"  Model checkpoints will be saved to: {checkpoint_path}")
        
        # if none the skip the gui_cb and use verbose = self.config.fit_verbose  
        # else use the Gui
        # GUI progress callback -
        # 1. decide the UI update function
        # ── 2. Optional GUI progress callback ───────────────────────────────
        
        callbacks = [early_stopping, model_checkpoint]
     
        if callable(getattr(self.config, "progress_callback", None)):
            #   a) create the progress-aware Keras callback
            batches_per_epoch = experimental.cardinality(
                train_dataset).numpy()
            gui_cb = GuiProgress(
                total_epochs=self.config.epochs,
                batches_per_epoch=batches_per_epoch,
                update_fn=self.config.progress_callback,   # Qt signal or any callable
                epoch_level=False,                         # smoother, per-batch updates
            )
            callbacks.append(gui_cb)
            fit_verbose = 0       # suppress default ASCII bar
        else:
                #   b) no GUI → honour user's requested verbosity
            fit_verbose = self.config.fit_verbose

        #  launch training --
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=fit_verbose,   # if none          # silence default ASCII bar
        )
        self.log("  Model training complete.")

  
    def _load_best_model(self) -> Any:
        """Loads the best performing model saved by ModelCheckpoint."""
        self.log("  Loading best model from checkpoint...")
        checkpoint_path = os.path.join(
            self.config.run_output_path, f"{self.config.model_name}.keras")
        
        try:
            custom_objects = {}
            if self.config.quantiles:
                custom_objects = {'combined_quantile_loss': combined_quantile_loss(
                    self.config.quantiles)}
            
            with custom_object_scope(custom_objects):
                best_model = load_model(checkpoint_path)
            self.log("  Best model loaded successfully.")
            return best_model
        except Exception as e:
            self.log(f"  Warning: Could not load best model from checkpoint: {e}. "
                     "Returning the model from the end of training.")
            return self.model
        
class Forecaster:
    """
    Handles generating predictions from a trained model and formatting
    the results into a DataFrame.
    """
    def __init__(self, config: SubsConfig, log_callback: Optional[callable] = None):
        """
        Initializes the forecaster with a configuration object.
        """
        self.config = config
        self.log = log_callback or print

    def run(
        self,
        model: Any,
        test_df: pd.DataFrame,
        val_dataset: Any,
        static_features_encoded: List[str],
        coord_scaler: Any
    ) -> Optional[pd.DataFrame]:
        """
        Executes the full forecasting and results formatting pipeline.

        Args:
            model: The trained Keras model.
            test_df: The preprocessed DataFrame for test set predictions.
            val_dataset: The validation tf.data.Dataset, for fallback use.
            static_features_encoded: List of one-hot encoded static columns.
            coord_scaler: The fitted scaler for coordinates.

        Returns:
            A pandas DataFrame with the formatted predictions, or None if
            forecasting fails.
        """
        self.log("Step 8: Forecasting on Test Data...")
        
        inputs_test, targets_test = self._prepare_test_sequences(
            test_df, val_dataset, static_features_encoded, coord_scaler
        )

        if inputs_test is None:
            self.log("  Skipping forecasting as no valid test or validation "
                     "input could be prepared.")
            return None

        return self._predict_and_format(
            model, inputs_test, targets_test, coord_scaler)

    def _prepare_test_sequences(
        self, test_df, val_dataset, static_features, coord_scaler
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Prepares test sequences, with a fallback to the validation set.
        """
        try:
            if test_df.empty:
                raise ValueError("Test DataFrame is empty.")
            
            self.log("  Attempting to generate PINN sequences from test data...")
            inputs, targets = prepare_pinn_data_sequences(
                df=test_df,
                time_col='time_numeric',
                lon_col=self.config.lon_col,
                lat_col=self.config.lat_col,
                subsidence_col=self.config.subsidence_col,
                gwl_col=self.config.gwl_col,
                dynamic_cols=[c for c in [
                    self.config.gwl_col, 'rainfall_mm'] if c in test_df.columns],
                static_cols=static_features,
                future_cols=self.config.future_features,
                group_id_cols=[self.config.lon_col, self.config.lat_col],
                time_steps=self.config.time_steps,
                forecast_horizon=self.config.forecast_horizon_years,
                normalize_coords=True,
                # return_coord_scaler= True, # we will use train scaler
                mode=self.config.mode, 
                _logger = self.log 
            )
            if targets['subsidence'].shape[0] == 0:
                raise ValueError("No test sequences were generated.")
            
            self.log("  Test sequences generated successfully.")
            return inputs, targets

        except Exception as e:
            self.log(f"\n  [WARNING] Could not generate test sequences: {e}")
            self.log("  Falling back to use the validation dataset for forecasting.")
            
            try:
                # Fallback to extracting the first batch from the validation dataset
                for x_val, y_val in val_dataset.take(1):
                    self.log("  Successfully extracted one batch from validation set.")
                    return x_val, y_val
                else: # Loop completed without break
                    self.log("  [ERROR] Fallback failed: Validation dataset is empty.")
                    return None, None
            except Exception as fallback_e:
                self.log(f"  [ERROR] Critical fallback failed: {fallback_e}")
                return None, None

    def _predict_and_format(
        self, model, inputs_test, targets_test, coord_scaler
    ) -> Optional[pd.DataFrame]:
        """
        Runs model prediction and formats the output into a DataFrame.
        """
        self.log("  Generating predictions with the trained model...")
        predictions = model.predict(inputs_test, verbose=0)

        # Standardize target keys for formatting
        y_true_for_format = {
            'subsidence': targets_test['subs_pred'],
            'gwl': targets_test['gwl_pred']
        }
        
        self.log("  Formatting predictions into a structured DataFrame...")
        forecast_df = format_pinn_predictions(
            predictions=predictions,
            y_true_dict=y_true_for_format,
            target_mapping={
                'subs_pred': self.config.subsidence_col,
                'gwl_pred': self.config.gwl_col
            },
            quantiles=self.config.quantiles,
            forecast_horizon=self.config.forecast_horizon_years,
            evaluate_coverage=( 
                True if ( self.config.evaluate_coverage and self.config.quantiles) else False
                ), 
            model_inputs=inputs_test,
            coord_scaler= coord_scaler, 
        )

        if forecast_df is not None and not forecast_df.empty:
            if self.config.save_intermediate:
                save_path = os.path.join(
                    self.config.run_output_path, "03_forecast_results.csv")
                forecast_df.to_csv(save_path, index=False)
                self.log(f"  Saved forecast results to: {save_path}")
            return forecast_df
        
        self.log("  Warning: No final forecast DataFrame was generated.")
        return None


class ResultsVisualizer:
    """
    Handles the visualization and final saving of forecast results.
    """
    def __init__(self, config: SubsConfig, log_callback: Optional[callable] = None):
        """
        Initializes the visualizer with a configuration object.
        """
        self.config = config
        self.log = log_callback or print

    def run(self, forecast_df: Optional[pd.DataFrame]):
        """
        Executes the full visualization and saving pipeline.

        Args:
            forecast_df (pd.DataFrame, optional): The DataFrame containing
                formatted forecast results. If None or empty, the
                visualization steps are skipped.
        """
        if forecast_df is None or forecast_df.empty:
            self.log("Step 9 & 10: Skipping visualization and saving as no "
                     "forecast data was provided.")
            return

        self.log("Step 9: Visualizing Forecasts...")
        self._plot_main_forecasts(forecast_df)

        self.log("Step 10: Finalizing and Saving All Figures...")
        self._run_forecast_view(forecast_df)
        self._save_all_figures()
        
        self.log("\n--- SCRIPT COMPLETED ---")
        self.log(f"All outputs are in: {self.config.run_output_path}")

    def _plot_main_forecasts(self, df: pd.DataFrame):
        """Generates the primary spatial plots for subsidence and GWL."""
        
        coord_cols = ['coord_x', 'coord_y']
        if not all(c in df.columns for c in coord_cols):
            self.log(f"  [Warning] Coordinate columns {coord_cols} not found. "
                     "Spatial plots may fail.")
            spatial_cols_arg = None
        else:
            spatial_cols_arg = coord_cols

        horizon_steps = [1, self.config.forecast_horizon_years] \
            if self.config.forecast_horizon_years > 1 else [1]
            
        forecast_years = [
            self.config.forecast_start_year + i for i in range(
                self.config.forecast_horizon_years)
        ]
        view_years = [forecast_years[step - 1] for step in horizon_steps]

        # Plot for Subsidence
        self.log("  --- Plotting Subsidence Forecasts ---")
        plot_forecasts(
            forecast_df=df,
            target_name=self.config.subsidence_col,
            quantiles=self.config.quantiles,
            output_dim=1,
            kind="spatial",
            horizon_steps=horizon_steps,
            spatial_cols=spatial_cols_arg,
            verbose=self.config.verbose,
            cbar="uniform",
            step_names={
                f"step {step}": f'Subsidence: Year {year}'
                for step, year in zip(horizon_steps, view_years)
            }
        )

        # Plot for GWL if configured
        gwl_pred_col = f"{self.config.gwl_col}_q50" if self.config.quantiles \
            else f"{self.config.gwl_col}_pred"
            
        if self.config.include_gwl_in_df and gwl_pred_col in df.columns:
            self.log("  --- Plotting GWL Forecasts ---")
            plot_forecasts(
                forecast_df=df,
                target_name=self.config.gwl_col,
                quantiles=self.config.quantiles,
                output_dim=1,
                kind="spatial",
                horizon_steps=horizon_steps,
                spatial_cols=spatial_cols_arg,
                verbose=self.config.verbose,
                cbar="uniform", # Can be set differently if desired
                titles=[f'GWL: Year {y}' for y in view_years], 
            )

    def _run_forecast_view(self, df: pd.DataFrame):
        """Runs the yearly comparison plot."""
        try:
            self.log("  Generating yearly forecast comparison view...")
            save_path = os.path.join(
                self.config.run_output_path,
                f"{self.config.city_name}_forecast_comparison_plot_"
            )
            forecast_view(
                df,
                spatial_cols=('coord_x', 'coord_y'),
                time_col='coord_t',
                value_prefixes=[self.config.subsidence_col],
                view_quantiles=[0.5] if self.config.quantiles else None,
                savefig=save_path,
                save_fmts=['.png', '.pdf'],
                verbose=self.config.verbose
            )
            self.log(f"  Forecast view figures saved to: {self.config.run_output_path}")
        except Exception as e:
            self.log(f"  [Warning] Could not generate forecast view plot: {e}")

    def _save_all_figures(self):
        """Saves all open matplotlib figures."""
        try:
            self.log("  Saving all generated figures...")
            save_all_figures(
                output_dir=self.config.run_output_path,
                prefix=f"{self.config.city_name}_{self.config.model_name}_plot_",
                fmts=['.png', '.pdf']
            )
            self.log("  All figures saved successfully.")
        except Exception as e:
            self.log(f"  [Warning] Could not save all figures: {e}")

class PredictionPipeline:
    """
    Handles the end-to-end workflow for making predictions on a new
    dataset using a pre-trained model and its associated artifacts.
    """
    def __init__(
        self,
        config: SubsConfig,
        model_path: str,
        encoder_path: str,
        scaler_path: str,
        coord_scaler_path: str,
        log_callback: Optional[callable] = None
    ):
        """
        Initializes the prediction pipeline.

        Args:
            config (SubsConfig): The configuration object.
            model_path (str): Path to the trained .keras model file.
            encoder_path (str): Path to the fitted OneHotEncoder .joblib file.
            scaler_path (str): Path to the fitted main scaler .joblib file.
            coord_scaler_path (str): Path to the fitted coordinate scaler .joblib.
            log_callback (callable, optional): Function to receive log messages.
        """
        self.config = config
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path
        self.coord_scaler_path = coord_scaler_path
        self.log = log_callback or print
        
        self.model = None
        self.encoder = None
        self.scaler = None
        self.coord_scaler = None

    def run(self, validation_data_path: str):
        """
        Executes the full prediction workflow on a new validation dataset.
        """
        self.log("--- Starting Prediction Pipeline ---")
        
        # 1. Load all necessary artifacts
        self._load_artifacts()
        
        # 2. Process the new validation data
        processed_val_df, static_features_encoded = self._process_validation_data(
            validation_data_path)
        
        # 3. Generate sequences for the validation data
        val_inputs, val_targets = self._generate_validation_sequences(
            processed_val_df, static_features_encoded
        )
        
        # 4. Run forecasting
        forecaster = Forecaster(self.config, self.log)
        forecast_df = forecaster._predict_and_format(
            self.model, val_inputs, val_targets, self.coord_scaler
        )
        
        # 5. Visualize results
        visualizer = ResultsVisualizer(self.config, self.log)
        visualizer.run(forecast_df)

        self.log("--- Prediction Pipeline Finished Successfully ---")

    def _load_artifacts(self):
        """Loads the trained model and preprocessing objects."""
        self.log("Loading trained model and preprocessing artifacts...")
        try:
            self.model = load_model(self.model_path, custom_objects={
                'combined_quantile_loss': combined_quantile_loss(self.config.quantiles)
            } if self.config.quantiles else {})
            self.encoder = joblib.load(self.encoder_path)
            self.scaler = joblib.load(self.scaler_path)
            self.coord_scaler = joblib.load(self.coord_scaler_path)
            self.log("  All artifacts loaded successfully.")
        except Exception as e:
            raise IOError(f"Failed to load a required artifact. Error: {e}")

    def _process_validation_data(
        self, validation_data_path: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Loads and processes the new validation data using saved scalers."""
        self.log(f"Processing new validation data from: {validation_data_path}")
        val_df = pd.read_csv(validation_data_path)
        
        # Use a temporary DataProcessor to apply transformations
        # This assumes DataProcessor can be initialized and used this way.
        temp_processor = DataProcessor(self.config)
        temp_processor.encoder = self.encoder
        temp_processor.scaler = self.scaler
        
        # A simplified transform-only logic
        df_cleaned = nan_ops(val_df, ops='sanitize', action='fill')
        
        encoded_data = self.encoder.transform(
            df_cleaned[self.config.categorical_cols])
        static_features = self.encoder.get_feature_names_out(
            self.config.categorical_cols).tolist()
        encoded_df = pd.DataFrame(
            encoded_data, columns=static_features, index=df_cleaned.index)
        df_processed = pd.concat([
            df_cleaned.drop(
                columns=self.config.categorical_cols), encoded_df], axis=1)
            
        df_processed['time_numeric'] = df_processed[
            self.config.time_col] - df_processed[self.config.time_col].min()
        
        cols_to_scale = [
            self.config.subsidence_col, self.config.gwl_col] + (
                self.config.future_features or [])
        existing_cols = [col for col in cols_to_scale if col in df_processed.columns]
        if existing_cols:
            df_processed[existing_cols] = self.scaler.transform(
                df_processed[existing_cols])

        self.log("  Validation data processed successfully.")
        return df_processed, static_features

    def _generate_validation_sequences(
        self, df: pd.DataFrame, static_features: List[str]
    ) -> Tuple[Dict, Dict]:
        """Generates sequences from the processed validation data."""
        self.log("  Generating validation sequences...")
        
        dynamic_features = [
            c for c in [self.config.gwl_col, 'rainfall_mm'] 
            if c in df.columns]
        
        inputs, targets = prepare_pinn_data_sequences(
            df=df,
            time_col='time_numeric',
            lon_col=self.config.lon_col,
            lat_col=self.config.lat_col,
            subsidence_col=self.config.subsidence_col,
            gwl_col=self.config.gwl_col,
            dynamic_cols=dynamic_features,
            static_cols=static_features,
            future_cols=self.config.future_features,
            group_id_cols=[self.config.lon_col, self.config.lat_col],
            time_steps=self.config.time_steps,
            forecast_horizon=self.config.forecast_horizon_years,
            normalize_coords=True,
            coord_scaler=self.coord_scaler, # Use the loaded scaler
            return_coord_scaler=False,
            mode=self.config.mode,
            _logger = self.log 
        )
        if targets['subsidence'].shape[0] == 0:
            raise ValueError("Sequence generation produced no validation samples.")
            
        self.log("  Validation sequences generated successfully.")
        return inputs, targets
    

class GuiProgress(Callback):
    """
    Emit percentage updates while `model.fit` runs.

    Parameters
    ----------
    total_epochs : int
        Epochs you pass to `model.fit`.
    batches_per_epoch : int
        Length of the training dataset (`len(ds)`).
        Needed only for *batch-level* granularity.
    update_fn : Callable[[int], None]
        Function that receives an **int 0-100**.
        Examples: `my_qprogressbar.setValue`, `signal.emit`.
    epoch_level : bool, default=True
        If True, update once per epoch; otherwise per batch.
    """
    def __init__(
        self,
        total_epochs: int,
        batches_per_epoch: int,
        update_fn: Callable[[int], None],
        *,
        epoch_level: bool = True,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.update_fn = update_fn
        self.epoch_level = epoch_level
        self._seen_batches = 0

    # -------- epoch-level --------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_level:
            pct = int((epoch + 1) / self.total_epochs * 100)
            self.update_fn(pct)

    # -------- batch-level --------------------------------------------------
    def on_train_batch_end(self, batch, logs=None):
        if not self.epoch_level:
            self._seen_batches += 1
            total_batches = self.total_epochs * self.batches_per_epoch
            pct = math.floor(self._seen_batches / total_batches * 100)
            self.update_fn(pct)
    
if __name__ == '__main__':
    print("--- Starting Subsidence Forecasting Workflow ---")
    
    # 1. Configure the workflow
    # For this test, point to the sample data that comes with the library
    config = SubsConfig(
        data_dir='_pinn_works/test_data', #'../../fusionlab/datasets/data',
        data_filename='zhongshan_500k.csv',
        epochs=3, # Use a small number of epochs for a quick test run
        save_intermediate=True,
        verbose=1
    )
    print(f"Configuration loaded for model '{config.model_name}'")

    # 2. Process Data
    processor = DataProcessor(config=config)
    processed_df = processor.run()
    
    # 3. Generate Sequences and Datasets
    sequence_gen = SequenceGenerator(config=config)
    train_dataset, val_dataset = sequence_gen.run(
        processed_df, processor.static_features_encoded
    )

    # 4. Train the Model
    # Get input shapes from a sample batch for model instantiation
    sample_inputs, _ = next(iter(train_dataset))
    input_shapes = {name: tensor.shape for name, tensor in sample_inputs.items()}
    
    trainer = ModelTrainer(config=config)
    best_model = trainer.run(train_dataset, val_dataset, input_shapes)

    # 5. Make Forecasts
    #%
    forecaster = Forecaster(config=config)
    forecast_df = forecaster.run(
        model=best_model,
        test_df=sequence_gen.test_df,
        val_dataset=val_dataset,
        static_features_encoded=processor.static_features_encoded,
        coord_scaler=sequence_gen.coord_scaler
    )

    # 6. Visualize Results
    visualizer = ResultsVisualizer(config=config)
    visualizer.run(forecast_df)

    print("\n--- Workflow Finished Successfully ---")

# if __name__ == '__main__':
#     print("--- Starting Full Training & Prediction Workflow ---")
    
#     # ==================================================================
#     # Part 1: Full Training Workflow
#     # ==================================================================
    
#     # 1. Configure the workflow
#     # For this test, we create a temporary directory for all outputs.
#     # Note: Using a smaller dataset for faster testing is recommended.
#     output_directory = "./app_workflow_test_run"
#     if os.path.exists(output_directory):
#         shutil.rmtree(output_directory) # Clean up previous runs

#     config = SubsConfig(
#         # For a quick test, we can use the smaller built-in dataset
#         # by pointing to a non-existent file and letting it fallback.
#         data_dir='./dummy_data_for_test', 
#         data_filename='non_existent_file.csv',
#         epochs=3, # Use a small number of epochs for a quick test
#         output_dir=output_directory,
#         save_intermediate=True, # Ensure artifacts are saved
#         verbose=1
#     )
#     print(f"Configuration loaded for model '{config.model_name}'")
#     print(f"All artifacts will be saved in: {config.run_output_path}")

#     # 2. Process Data
#     # This will load data, preprocess it, and save the encoder and scaler.
#     processor = DataProcessor(config=config)
#     processed_df = processor.run()
    
#     # 3. Generate Sequences and Datasets
#     # This will create sequences and save the coordinate scaler.
#     sequence_gen = SequenceGenerator(config=config)
#     train_dataset, val_dataset = sequence_gen.run(
#         processed_df, processor.static_features_encoded
#     )

#     # 4. Train the Model
#     # This will train the model and save the best version.
#     sample_inputs, _ = next(iter(train_dataset))
#     input_shapes = {name: tensor.shape for name, tensor in sample_inputs.items()}
    
#     trainer = ModelTrainer(config=config)
#     best_model = trainer.run(train_dataset, val_dataset, input_shapes)

#     print("\n--- Training Workflow Finished Successfully ---")

#     # ==================================================================
#     # Part 2: Inference Workflow using PredictionPipeline
#     # ==================================================================
#     print("\n" + "="*50)
#     print("--- Starting Inference Workflow on Validation Data ---")
#     print("="*50 + "\n")
    
#     # Define the paths to the artifacts we just saved in the training run
#     model_path = os.path.join(
#         config.run_output_path, f"{config.model_name}.keras")
#     encoder_path = os.path.join(
#         config.run_output_path, "ohe_encoder.joblib")
#     scaler_path = os.path.join(
#         config.run_output_path, "main_scaler.joblib")
#     # Note: The SequenceGenerator saves the coord scaler in its own file
#     coord_scaler_path = os.path.join(
#         config.run_output_path, "coord_scaler.joblib")
    
#     # We will use the original data file as the "new" data to predict on.
#     # The loader inside the pipeline needs a valid source.
#     validation_data_path = processor.raw_df_path 

#     # Check if all required artifacts exist before proceeding
#     required_artifacts = [model_path, encoder_path, scaler_path, coord_scaler_path]
#     if not all(os.path.exists(p) for p in required_artifacts):
#         print("[ERROR] Not all required artifacts from the training run were found.")
#         print("Skipping prediction pipeline.")
#     else:
#         # Instantiate the prediction pipeline with the paths to the artifacts
#         prediction_pipeline = PredictionPipeline(
#             config=config,
#             model_path=model_path,
#             encoder_path=encoder_path,
#             scaler_path=scaler_path,
#             coord_scaler_path=coord_scaler_path
#         )
        
#         # Run the entire prediction and visualization workflow
#         prediction_pipeline.run(validation_data_path=validation_data_path)