# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides the PredictionPipeline class for running a full inference
workflow using artifacts from a previous training run.
"""
from __future__ import annotations 
import os 
from pathlib import Path 
import json 
from typing import Optional, List, Tuple, Dict, Callable, Union   
import joblib 
import pandas as pd 

from ...nn import KERAS_DEPS 
from ...nn.pinn.utils import prepare_pinn_data_sequences 
from ...nn.losses import combined_quantile_loss 
from ...nn.models import TransFlowSubsNet, PIHALNet  # noqa 
from ...params import LearnableK, LearnableSs, LearnableQ # Noqa: E401
from ...registry import  ManifestRegistry, _update_manifest
from ...utils.data_utils import nan_ops
from ...utils.generic_utils import normalize_time_column, rename_dict_keys 
from ...utils.ts_utils import ts_validator 

from .config import SubsConfig 
from .modeling import Forecaster 
from .processing import DataProcessor 
from .utils import ( 
    safe_model_loader, _rebuild_from_arch_cfg, 
    _CUSTOM_OBJECTS
)
from .view import ResultsVisualizer 
load_model = KERAS_DEPS.load_model
Model =KERAS_DEPS.Model 

class PredictionPipeline:
    """
    Orchestrates an end-to-end inference workflow.

    This class is designed to be instantiated after a model has been
    trained. It loads a trained model, its configuration, and all
    necessary preprocessing artifacts (scalers, encoders) from a
    run manifest file. It then uses these artifacts to process a new
    dataset, generate predictions, and visualize the results.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        The path to the 'run_manifest.json' file generated during a
        training run. This file contains all the paths and
        configurations needed for inference.
    log_callback : callable, optional
        A function to receive and handle log messages, such as
        updating a GUI log panel. Defaults to `print`.
    kind: str, 
       Name of the pipeline. If provided, name should be used for append 
       the plot that generated.
    """
    _range = staticmethod(lambda frac, lo, hi: int(lo + (hi - lo)*frac))

    def __init__(
        self,
        manifest_path: Optional[str | os.PathLike] = None,
        *,
        log_callback: Optional[Callable[[str], None]] = None,
        kind : Optional[str]=None, 
        **kws, 
    ):
        """
        Initializes the pipeline by loading a manifest file.

        The manifest can be provided directly or auto-detected as the
        most recent run from the central `ManifestRegistry`.

        Parameters
        ----------
        manifest_path : str or pathlib.Path, optional
            A direct path to a specific `run_manifest.json` file. If
            None, the pipeline will automatically find the most recent
            run. Default is None.
        log_callback : callable, optional
            A function to receive and handle log messages. Defaults to `print`.
        """
        self.log = log_callback or print
        self.kind = kind 
        
        registry = ManifestRegistry()
        if manifest_path is None:
            self.log("No manifest path provided. Locating the latest run...")
            
            self.manifest_path =registry.latest_manifest()
            if not self.manifest_path:
                raise FileNotFoundError(
                    "No training runs found in the manifest registry. "
                    "Please run a training session first."
                )
        else:
            # here user provide explicity manifest file. so make a copy 
            # and kekedp into registry 
            manifest_path = Path(manifest_path)
            self.manifest_path = registry.import_manifest(manifest_path)
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found at: {self.manifest_path}"
            )
        
        for key in list(kws.keys()): 
            setattr (self, key, kws[key]) 
            
        self._load_from_manifest()

    def _load_from_manifest(self):
        """Loads configuration and resolves artifact paths from the manifest."""
        self.log(f"Initializing prediction pipeline from: {self.manifest_path}")
        self._manifest = json.loads(self.manifest_path.read_text("utf-8"))
        
        # The manifest is the single source of truth for configuration.
        self.config = SubsConfig(**self._manifest.get("configuration", {}))
        
        # The key is to use the run_output_path stored *inside* the manifest,
        # as this is where the training artifacts were actually saved.
        self.artifacts_dir = Path(self.config.run_output_path)
        self._resolve_artifact_paths()
        
        self.model = self.encoder = self.scaler = self.coord_scaler = None

    def _resolve_artifact_paths(self):
        """Resolves full paths to all artifacts from the manifest."""
        artifacts = self._manifest.get("artifacts", {})
        training_info = self._manifest.get("training", {})
        
        # Helper to construct full path from relative filename in manifest
        def _get_path(key, default_name):
            filename = artifacts.get(key) or training_info.get(key) or default_name
            return self.artifacts_dir / filename

        self.model_path = _get_path("checkpoint", "model.keras")
        self.encoder_path = _get_path("encoder", "ohe_encoder.joblib")
        self.scaler_path = _get_path("main_scaler", "main_scaler.joblib")
        self.coord_scaler_path = _get_path("coord_scaler", "coord_scaler.joblib")

    def _load_artifacts(self) -> None:
        """
        Loads the trained model and all preprocessing objects from disk.
        
        This method uses the `safe_model_loader` utility to handle
        different Keras save formats, including weights-only files
        which require rebuilding the model architecture from the manifest.
        """
        self.log("Loading trained model and preprocessing artifacts...")
        
        # --- 1. Prepare for Model Loading ---
        custom_objects = _CUSTOM_OBJECTS.copy()
        if self.config.quantiles:
            custom_objects["combined_quantile_loss"] = combined_quantile_loss(
                self.config.quantiles
            )
            
        build_fn = None
        model_cfg = self._manifest.get("training", {})
        # Check if we need to rebuild the model for weights-only loading
        if str(self.model_path).endswith((".weights.h5", ".weights.keras")):
            self.log("Weights-only file detected. Rebuilding model from manifest...")
            
            arch_cfg = model_cfg.get("config")
            if not arch_cfg:
                raise ValueError("Manifest is missing 'training.config' section"
                                 " required to rebuild model for weights-only file.")
            # The build function will use the architecture config
            build_fn = lambda: _rebuild_from_arch_cfg(arch_cfg)
        
        # --- 2. Load the Model ---
        self.model = safe_model_loader(
            model_path=self.model_path,
            build_fn=build_fn,
            custom_objects=custom_objects,
            log=self.log,
            model_cfg=model_cfg  # Pass model_cfg for model.build() hint
        )

        # --- 3. Load Preprocessing Artifacts ---
        try:
            self.encoder = joblib.load(self.encoder_path)
            self.scaler = joblib.load(self.scaler_path)
            self.coord_scaler = joblib.load(self.coord_scaler_path)
            self.log("  All artifacts loaded successfully.")
        except Exception as e:
            raise IOError(f"Failed to load a required preprocessing artifact. Error: {e}")

    def _tick(self, percent: int) -> None:
        cb = getattr(self.config, "progress_callback", None)
        # do NOT recurse if callback is _tick itself
        if callable(cb) and cb is not self._tick:
            cb(percent)
    
    def _check_if_trained(self, validation_df: pd.DataFrame) -> None:
        """
        Safeguard:  ensure the model & artefacts exist **and** the
        validation file has (at least) the columns that were present
        during training.
        """
        missing: List[str] = []

        if self.model is None:
            missing.append("model")
        if self.encoder is None:
            missing.append("encoder")
        if self.scaler is None:
            missing.append("scaler")
        if self.coord_scaler is None:
            missing.append("coord_scaler")

        if missing:
            raise RuntimeError(
                "[PredictionPipeline] The following artefacts have *not* "
                f"been loaded: {', '.join(missing)}"
            )

        trained_cols = (
            list(self.encoder.feature_names_in_) +
            list(self.scaler.feature_names_in_) +
            self.config.dynamic_features +
            (self.config.future_features or [])
        )
        not_found = [c for c in trained_cols if c not in validation_df.columns]
        if not_found:
            raise ValueError(
                "[PredictionPipeline] Validation file is missing "
                f"{len(not_found)} column(s) that were present in training:\n"
                f"{', '.join(not_found[:10])}{' …' if len(not_found) > 10 else ''}"
            )

    def run(self, validation_data: Union [str, pd.DataFrame], 
            stop_check =None 
        ):
        """
        Executes the full prediction workflow on a new validation dataset.
        """
        self._tick(0)
        self.log("--- Starting Prediction Pipeline ---")
        
        # 1. Load all necessary artifacts
        self._load_artifacts()
        self._tick(10)
        
        # 2. Process the new validation data
        processed_val_df, static_features_encoded = self._process_validation_data(
            validation_data, stop_check = stop_check, 
            )
        self._tick(30)
        
        # 3. Generate sequences for the validation data
        self.log("Generating validation sequences …")
    
        val_inputs, val_targets = self._generate_validation_sequences(
            processed_val_df, static_features_encoded,
            stop_check = stop_check,  
        )
        self._tick(80)
        
        # 4. Run forecasting
        forecaster = Forecaster(
            self.config, self.log, kind = self.kind)
        # rename target to fit the exact subsudence value. 
        val_targets = rename_dict_keys(
            val_targets.copy(),
                param_to_rename={
                    "subsidence": "subs_pred", 
                    "gwl": "gwl_pred"
                }
             )
        forecast_df = forecaster._predict_and_format(
            self.model, val_inputs, val_targets, self.coord_scaler, 
            stop_check = stop_check , 
        )
        self._tick(95)
        
        # 5. Visualize results
        visualizer = ResultsVisualizer(
            self.config, self.log, 
            kind = self.kind
        )
        visualizer.run(forecast_df, stop_check = stop_check )
        if self.model_path: 
            inference_ref = 'unset'
            if isinstance(validation_data, str): 
                inference_ref = validation_data # assume a file path 
                
            _update_manifest(
                run_dir = os.path.dirname (str(self.manifest_path)),      
                section = "inference",
                item = {"validation_file": str(inference_ref)},
            )

        self.log("--- Prediction Pipeline Finished Successfully ---")
        self._tick(100)
        
    def _process_validation_data(
        self, validation_data: str,
        stop_check = None, 
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Loads and processes the new validation data using saved scalers."""
        msg ="Processing new validation "
        val_df = validation_data
        if isinstance ( val_df, str): 
            self.log(f"{msg} from {validation_data}")
            # assume a file path 
            val_df = pd.read_csv(val_df)
        elif isinstance (val_df, pd.DataFrame): 
            self.log(msg)

        else: 
            raise ValueError(
                "Inference/Validation data *must* be either a Pathlike object"
                f"or a dataframe. Got {type(validation_data).__name__!r}"
            )
        # check when dataframe contain date time. 
        val_df= ts_validator(
            val_df.copy(),
            dt_col=self.config.time_col,
            to_datetime='auto',
            as_index=False,
            error="raise",
            return_dt_col=False,
            verbose=0
        )
        
        # Use a temporary DataProcessor to apply transformations
        # This assumes DataProcessor can be initialized and used this way.
        temp_processor = DataProcessor(self.config, kind =self.kind)
        temp_processor.encoder = self.encoder
        temp_processor.scaler = self.scaler
        
        if stop_check and stop_check():
            raise InterruptedError("Preprocessing aborted.")
            
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
            # Convert year to datetime for consistent processing,
            # then to numeric for PINN
            # print(df_processed.columns)
            df_processed = normalize_time_column(
                df_processed, time_col= self.config.time_col, 
                datetime_col= temp_dt_col, 
                year_col= self.config.time_col,
                drop_orig= True 
            )
        if stop_check and stop_check():
            raise InterruptedError("Processing aborted.")
        # df_processed['time_numeric'] = df_processed[
        #     self.config.time_col] - df_processed[self.config.time_col].min()
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
        self, df: pd.DataFrame, static_features: List[str], 
        stop_check=None 
    ) -> Tuple[Dict, Dict]:
        """Generates sequences from the processed validation data."""
        self.log("  Generating validation sequences...")
        
        # dynamic_features = [
        #     c for c in [self.config.gwl_col, 'rainfall_mm'] 
        #     if c in df.columns]
        progress_hook = lambda f: self._tick(self._range(f, 30, 80))
        
        # 2. do *not* point the config callback to self._tick
        #    (or add the guard shown above)
        inputs, targets = prepare_pinn_data_sequences(
            df=df,
            time_col='time_numeric',
            lon_col=self.config.lon_col,
            lat_col=self.config.lat_col,
            subsidence_col=self.config.subsidence_col,
            gwl_col=self.config.gwl_col,
            dynamic_cols=self.config.dynamic_features,
            static_cols=static_features,
            future_cols=self.config.future_features,
            group_id_cols=[self.config.lon_col, self.config.lat_col],
            time_steps=self.config.time_steps,
            forecast_horizon=self.config.forecast_horizon_years,
            normalize_coords=True,
            coord_scaler=self.coord_scaler, # Use the loaded scaler
            return_coord_scaler=False,
            mode=self.config.mode,
            stop_check =stop_check, 
             progress_hook =progress_hook,
            _logger = self.log 
        )
        if targets['subsidence'].shape[0] == 0:
            raise ValueError("Sequence generation produced no validation samples.")
            
        self.log("  Validation sequences generated successfully.")
        return inputs, targets
    

