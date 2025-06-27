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
from typing import List, Optional, Dict, Any
from typing import Tuple, Callable  

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from fusionlab.utils.generic_utils import ( 
    split_train_test_by_time, 
    ensure_directory_exists, 
    normalize_time_column, 
    ensure_cols_exist
) 
from fusionlab.datasets import fetch_zhongshan_data
from fusionlab.utils.data_utils import nan_ops
from fusionlab.utils.io_utils import save_job, _update_manifest 

from fusionlab.nn import KERAS_DEPS
from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences

from fusionlab.tools.app.config import SubsConfig 

Dataset = KERAS_DEPS.Dataset
AUTOTUNE = KERAS_DEPS.AUTOTUNE

class DataProcessor:
    """
    Handles the data loading and preprocessing workflow (Steps 1-5).
    """
    
    def __init__(
        self, config: SubsConfig, 
        log_callback: Optional[callable] = None, 
        raw_df: Optional[pd.DataFrame] = None, 
        kind: Optional[str] =None, 
        ):
        """
        Initializes the processor with a configuration object.

        Args:
            config (SubsConfig): The configuration object with all parameters.
            log_callback (callable, optional): A function to receive log messages.
            kind, str, the kind of data that us used , can be for training or inference
            this is usefull for append in the processing file that is generated 
            if inference, then append .infer to the processing otherwise is None. 
            
        """
        self.config = config
        self.raw_df: Optional[pd.DataFrame] = raw_df
        self.processed_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.log = log_callback or print
        self.kind = kind or '' 
        
        # To store fitted objects
        self.encoder = None
        self.scaler = None

    # PRIVATE helper: emit progress locally (0-100 %)
    def _tick(self, percent: int) -> None:
        """
        Emit <percent> through the SubsConfig.progress_callback
        if that callback exists and is callable.
        """
        cb = getattr(self.config, "progress_callback", None)
        if callable(cb):
            cb(percent)
            
    def load_data(self) -> pd.DataFrame:
        """
        Handles Step 1: Loading the dataset from a file or by fetching it.
        """
        self._tick(0)  
        if self.raw_df is not None:
            self.log(f"Step 1: Using in-memory DataFrame – shape {self.raw_df.shape}")
        else: 
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
            ensure_directory_exists(
                self.config.run_output_path)
            save_path = os.path.join(
                self.config.run_output_path, "01_raw_data.csv")
            self.raw_df.to_csv(save_path, index=False)
            self.log(f"  Saved raw data artifact to: {save_path}")
            
            _update_manifest(
                self.config.registry_path,               # NEW
                "artifacts",
                {"raw_csv": os.path.basename(save_path)})   # NEW
        
        self._tick(20) 
        
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
        def _as_list(x):
            if x in (None, 'auto'):      # keep sentinel
                return []
            return list(x) if not isinstance(x, list) else x
        
        # self.log("categorical_cols {}".format(self.config.categorical_cols ))
        # self.log("numerical_cols {}".format(self.config.numerical_cols ))
        # self.log("future_features {}".format(self.config.future_features ))
        
        cat_cols = _as_list(self.config.categorical_cols)
        num_cols = _as_list(self.config.numerical_cols)
        fut_cols = _as_list(self.config.future_features)
        
        all_cols = base_cols + cat_cols + num_cols + fut_cols
        
        ensure_cols_exist(self.raw_df, *all_cols, error ='raise')

        # all_cols = base_cols + (self.config.categorical_cols or []) + \
        #             (self.config.numerical_cols or []) + (self.config.future_features or [])
        
        df_selected = self.raw_df[[
            col for col in set(all_cols) if col in self.raw_df.columns
        ]].copy()
        
        # --- Cleaning ---
        self.log("  Cleaning NaN values...")
        df_cleaned = nan_ops(df_selected, ops='sanitize', action='fill')
        
        self._tick(40)
        
        # --- Encoding ---
        df_processed = df_cleaned
        if self.config.categorical_cols:
            self.log(f"  One-hot encoding: {self.config.categorical_cols}")
            self.encoder = OneHotEncoder(
                sparse_output=False,handle_unknown='ignore', 
                dtype=np.float32
            )
            encoded_data = self.encoder.fit_transform(
                df_cleaned[self.config.categorical_cols]
            )
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
        
        self._tick(60)
        
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
                      
        # persist the resolved column choices to the manifest
        _update_manifest(
            run_dir=self.config.registry_path,
            section="configuration",
            item={
                "time_col":          self.config.time_col,
                "categorical_cols":  self.config.categorical_cols,
                "numerical_cols":    self.config.numerical_cols,
                "static_features":   self.config.static_features,
                "dynamic_features":  self.config.dynamic_features,
                "future_features":   self.config.future_features,
            },
        )
        
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
        self._tick(80)
        
        if self.config.save_intermediate:
            # Save artifacts
            save_path = os.path.join(
                self.config.run_output_path, f"02_processed_data{self.kind}.csv"
                )
            self.processed_df.to_csv(save_path, index=False)
            self.log(f"  Saved processed data to: {save_path}")
            
            artefacts = {"processed_csv": os.path.basename(save_path)}
            
            if self.encoder:
                enc_name = f"{self.config.model_name}.ohe_encoder.joblib"
                save_job(self.encoder, os.path.join(
                    self.config.run_output_path, enc_name
                    ), 
                    append_date= False, append_versions= False, # leave manifest track it 
                    )
                artefacts["encoder"] = enc_name 
            if self.scaler:
                sc_name = f"{self.config.model_name}.main_scaler.joblib"
                save_job(self.scaler, os.path.join(
                    self.config.run_output_path, sc_name), 
                    append_date= False, append_versions= False, 
                    )
                artefacts["main_scaler"] = sc_name
            
            _update_manifest(
                self.config.registry_path, "artifacts", artefacts)    
            
        
        self._tick(100)
        return self.processed_df

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handles Step 5: Splitting the data into training and test sets."""
        if self.processed_df is None:
            raise RuntimeError("Data must be preprocessed before splitting.")

        self.log("Step 4 & 5: Defining Feature Sets and Splitting Data...")
        
        # Use the config to perform the split using the configured time column
        time_col = self.config.time_col
        
        self.train_df , self.test_df = split_train_test_by_time( 
            self.processed_df , time_col = time_col, 
            cutoff=self.config.train_end_year 
        )
        # self.train_df = self.processed_df[
        #     self.processed_df[time_col] <= self.config.train_end_year
        # ].copy()
        
        # self.test_df = self.processed_df[
        #     self.processed_df[time_col] >= self.config.train_end_year
        # ].copy()
        
        self.log(f"  Data split complete. Train shape: {self.train_df.shape},"
                 f" Test shape: {self.test_df.shape}")
                 
        return self.train_df, self.test_df

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the full data processing pipeline from loading to splitting.
        """
        self.load_data()
        return self.preprocess_data()
        # Henceforth manage by SequenceGenerator 
        # self.split_data()
        # return self.train_df, self.test_df

class SequenceGenerator:
    """
    Handles sequence generation and dataset creation (Steps 5-6).
    """
    ZOOM = staticmethod(lambda frac, lo, hi: int(lo + (hi - lo) * frac))
    def __init__(
        self, config: SubsConfig, 
        log_callback: Optional[callable] = None, 
        ):
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

    def _tick(self, percent: int) -> None:
        """
        Emit <percent> through the SubsConfig.progress_callback
        if that callback exists and is callable.
        """
        cb = getattr(self.config, "progress_callback", None)
        if callable(cb):
            cb(percent)
            
    def run(
        self, processed_df: pd.DataFrame, 
        static_features_encoded: List[str], 
        stop_check: Callable[[], bool] = None,
        ) -> Tuple[Any, Any]:
        """
        Executes the full sequence and dataset creation pipeline.
        
        Returns:
            A tuple of (train_dataset, validation_dataset).
        """
        self._split_data(processed_df)
        self._generate_sequences(
            self.train_df, static_features_encoded, 
            stop_check= stop_check 
        )
        return self._create_tf_datasets()

    def _split_data(
            self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handles Step 5: Splitting the data into training and test sets
        robustly.
        """
        self._tick(0)
        if df is None:
            raise RuntimeError("Data must be preprocessed before splitting.")

        self.log("Step 4 & 5: Splitting master data...")
        
        time_col = self.config.time_col
        if time_col not in df.columns:
            raise ValueError(
                f"Time column '{time_col}' not found in DataFrame.")

        # --- Robust time conversion for comparison ---
        try: 
            self.train_df, _ = split_train_test_by_time(
                df, time_col = time_col , 
                cutoff=self.config.train_end_year )
            _, self.test_df = split_train_test_by_time(
                df, time_col= time_col , 
                cutoff= self.config.forecast_start_year
            )
        except: 
            # do it manually 
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
        self._tick(10)
        
        return self.train_df, self.test_df
    
    def _generate_sequences(
            self, train_master_df: pd.DataFrame, 
            static_features: List[str], 
            stop_check =None, 
        ):
        """Generates PINN-compatible sequences from the training data."""
        self.log("  Generating PINN training sequences...")
        
        # Define feature roles for the sequence generator
        dynamic_features = self.config.dynamic_features 
        # [
        #     self.config.gwl_col, 'rainfall_mm', 
        #     # 'normalized_density'
        # ] or self.config.dynamic_features 
        dynamic_features = [
            c for c in dynamic_features if c in train_master_df.columns]
        
        self._tick(30)
        
        lo, hi = 30, 80                      # global slice for seq-gen
        hook = lambda f: self._tick(self.ZOOM(f, lo, hi))
    
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
            progress_hook=hook, 
            stop_check= stop_check, 
            verbose=self.config.verbose,  
            _logger = self.log 
        )
        self._tick(80)
        if targets['subsidence'].shape[0] == 0:
            raise ValueError(
                "Sequence generation produced no training samples.")
            
        self.inputs_train = inputs
        self.targets_train = targets
        self.coord_scaler = scaler
        self.log(
            "  Training sequences generated successfully."
            )
        if self.config.save_intermediate and self.coord_scaler is not None:
            coord_sc_name = f"{self.config.model_name}.coord_scaler.joblib"
            save_job(self.coord_scaler,
                     os.path.join(self.config.run_output_path, coord_sc_name), 
                     append_date=False, append_versions=False )

            _update_manifest(
                self.config.registry_path, "artifacts", 
                {"coord_scaler": coord_sc_name}
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
        self._tick(100)
        return train_dataset, val_dataset


