from __future__ import annotations 

import os 
from pathlib import Path 
import json 
from typing import Optional, List, Tuple, Dict  
import joblib 
import pandas as pd 

from fusionlab.nn import KERAS_DEPS 
from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences 
from fusionlab.nn.losses import combined_quantile_loss 
from fusionlab.utils.generic_utils import ( 
    normalize_time_column, rename_dict_keys 
)
from fusionlab.utils.io_utils import _update_manifest
from fusionlab.nn.models import TransFlowSubsNet, PIHALNet  # noqa 
from fusionlab.params import LearnableK, LearnableSs, LearnableQ # Noqa 
from fusionlab.utils.data_utils import nan_ops 
from fusionlab.tools.app.config import SubsConfig 
from fusionlab.tools.app.processing import DataProcessor 
from fusionlab.tools.app.modeling import Forecaster 
from fusionlab.tools.app.view import ResultsVisualizer 
from fusionlab.tools.app.utils import ( 
    safe_model_loader, _rebuild_from_arch_cfg, 
    _CUSTOM_OBJECTS
)

load_model = KERAS_DEPS.load_model
Model =KERAS_DEPS.Model 

class PredictionPipeline:
    _range = staticmethod(lambda frac, lo, hi: int(lo + (hi - lo)*frac))

    def __init__(
        self,
        manifest_path: str | os.PathLike | None = None,   # <- NEW
        *,
        # the “manual” arguments keep working as before
        config: SubsConfig | None = None,
        model_path: str | None = None,
        encoder_path: str | None = None,
        scaler_path: str | None = None,
        coord_scaler_path: str | None = None,
        log_callback: Optional[callable] = None,
    ):
        # --
        self.log = log_callback or print                 # we need it early

        # try to load the JSON manifest
        manifest = {}
        if manifest_path and Path(manifest_path).exists():
            self.log(f"[Prediction] Reading manifest: {manifest_path}")
            manifest = json.loads(Path(manifest_path).read_text("utf-8"))

        #  Config --------------------------------------------------------
        if config is None:
            cfg_dict = manifest.get("configuration", {})
            if not cfg_dict:
                raise ValueError("No SubsConfig provided and manifest "
                                 "contains no 'config' section.")
            config = SubsConfig(**cfg_dict)
        self.config = config

        # decide the *expected* checkpoint name from the chosen save_format
        fmt  = self.config.save_format or "weights"
        ckpt = (f"{self.config.model_name}.weights.h5"
                if fmt == "weights"
                else f"{self.config.model_name}.keras")          # default / .keras
        
        arte   = manifest.get("artefacts", {})
        train  = manifest.get("training", {})
        
        self.model_path        = (model_path
                                  or self._dflt(train.get("model"))
                                  or self._dflt(ckpt))
        
        self.encoder_path      = (encoder_path
                                  or self._dflt(arte.get("encoder"))
                                  or self._dflt(
                                         f"{self.config.model_name}.ohe_encoder.joblib"))
        
        self.scaler_path       = (scaler_path
                                  or self._dflt(arte.get("scaler"))
                                  or self._dflt(
                                         f"{self.config.model_name}.main_scaler.joblib"))
        
        self.coord_scaler_path = (coord_scaler_path
                                  or self._dflt(arte.get("coord_scaler"))
                                  or self._dflt(
                                         f"{self.config.model_name}.coord_scaler.joblib"))

        # since the check point has the model name 
        # ④  runtime-holders
        self.model = self.encoder = self.scaler = self.coord_scaler = None
        # make a cony of Manifest config 
        self._cfg = manifest.copy() # from manifest 
        

    def _dflt(self, candidate: str | None) -> str | None:
        """
        Smart-resolve *candidate* into a **full path**.

        Rules
        -----
        1. ``None``  →  returns ``None`` (caller will try the next option).
        2. *Absolute* path  →  returned unchanged.
        3. *Relative* path **with a directory part**  
           (e.g. ``"checkpoints/ckpt.h5"``) → resolved against *cwd*.
        4. *Bare filename* (no “/”)  →  prefixed with
           ``self.config.run_output_path``.
        5. *Directory only* (ends with “/” or has no filename) → **error** –
           the caller must supply a filename.
        """
        if candidate is None:
            return None

        p = Path(candidate)

        # 1) absolute → done
        if p.is_absolute():
            return str(p)

        # 2) ‘dir / file’
        if p.parent != Path("."):
            if p.name == "":                 # “dir/” without file
                raise ValueError(
                    f"[Prediction] Expected a filename in '{candidate}'.")
            return str(p.resolve())

        # 3) bare filename → prepend run_output_path
        return str(Path(self.config.run_output_path) / p)

    def _tick(self, percent: int) -> None:
        cb = getattr(self.config, "progress_callback", None)
        # do NOT recurse if callback is _tick itself
        if callable(cb) and cb is not self._tick:
            cb(percent)
    
        
    # NEW – checks the artefacts and column compatibility -------------
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

    def run(self, validation_data_path: str):
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
            validation_data_path)
        self._tick(30)
        
        # 3. Generate sequences for the validation data
        self.log("Generating validation sequences …")
        # seq_cb = lambda p: self._tick(
        #         self._range(p/100, 30, 80)        # smooth 30→80 %
        #     )
        # self.config.progress_callback = seq_cb
        
        val_inputs, val_targets = self._generate_validation_sequences(
            processed_val_df, static_features_encoded 
            
        )
        self._tick(80)
        
        # 4. Run forecasting
        forecaster = Forecaster(self.config, self.log)
        # rename target to fit the exact subsudence value. 
        val_targets = rename_dict_keys(
            val_targets.copy(),
                param_to_rename={
                    "subsidence": "subs_pred", 
                    "gwl": "gwl_pred"
                }
             )
        forecast_df = forecaster._predict_and_format(
            self.model, val_inputs, val_targets, self.coord_scaler
        )
        self._tick(95)
        
        # 5. Visualize results
        visualizer = ResultsVisualizer(self.config, self.log)
        visualizer.run(forecast_df)
        if self.model_path: 
            _update_manifest(
                run_dir = Path(self.model_path).parent,      
                section = "inference",
                item = {"validation_file": validation_data_path},
            )

          
        self.log("--- Prediction Pipeline Finished Successfully ---")
        self._tick(100)
        
    def _load_artifacts(self) -> None:
        """Loads the trained model and preprocessing objects."""
        self.log("Loading trained model ...")
        # 
        custom = {}
        if self.config.quantiles:
            custom["combined_quantile_loss"] = combined_quantile_loss(
                self.config.quantiles)
    
        # if weights-only: rebuild from manifest → Keras can revive from config
        build_fn = None
        if str(self.model_path).endswith(".weights.h5"):
            # self._config  = json.loads(Path(self.manifest_path).read_text("utf-8"))
            arch_cfg = self._cfg ["training"]["config"]         # saved earlier
            try: 
                build_fn = lambda: _rebuild_from_arch_cfg(arch_cfg)
            except:
                # This will never reach but let keep it anyway. 
                class_name = self._cfg ["configuration"]["model_name"]
                cls = ( 
                    TransFlowSubsNet if class_name == "TransFlowSubsNet" 
                    else PIHALNet
                )
                build_fn = lambda: cls.from_config(arch_cfg)
             
        self.model = safe_model_loader(
            self.model_path,
            build_fn=build_fn,
            custom_objects={ **custom,** _CUSTOM_OBJECTS} ,
            log=self.log,
        )
        self.log("Loading preprocessing artifacts...")
        
        try:
            self.encoder = joblib.load(self.encoder_path)
            self.scaler = joblib.load(self.scaler_path)
            self.coord_scaler = joblib.load(self.coord_scaler_path)
            self.log("  All artifacts loaded successfully.")
        except Exception as e:
            raise IOError(
                f"Failed to load a required artifact. Error: {e}")
            
    def __load_artifacts(self):
        """Loads the trained model and preprocessing objects."""
        
        self.log("Loading trained model and preprocessing artifacts...")
        try:
            self.model = load_model(self.model_path, custom_objects={
                'combined_quantile_loss': combined_quantile_loss(
                    self.config.quantiles)
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
    

