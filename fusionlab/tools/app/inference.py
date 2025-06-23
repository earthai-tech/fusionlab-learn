
from typing import Optional, List, Tuple, Dict  
import joblib 
import pandas as pd 
from fusionlab.utils.data_utils import nan_ops 
from fusionlab.tools.app.config import SubsConfig 
from fusionlab.tools.app.processing import DataProcessor 
from fusionlab.tools.app.modeling import Forecaster 
from fusionlab.tools.app.view import ResultsVisualizer 
from fusionlab.nn import KERAS_DEPS 
from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences 
from fusionlab.nn.losses import combined_quantile_loss 

load_model = KERAS_DEPS.load_model 

class PredictionPipeline:
    """
    Handles the end-to-end workflow for making predictions on a new
    dataset using a pre-trained model and its associated artifacts.
    """
    _range = staticmethod(lambda frac, lo, hi: int(lo + (hi - lo)*frac))
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

    def _tick(self, percent: int) -> None:
        """
        Emit <percent> through the SubsConfig.progress_callback
        if that callback exists and is callable.
        """
        cb = getattr(self.config, "progress_callback", None)
        if callable(cb):
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
        seq_cb = lambda p: self._tick(
                self._range(p/100, 30, 80)        # smooth 30→80 %
            )
        self.config.progress_callback = seq_cb
        
        val_inputs, val_targets = self._generate_validation_sequences(
            processed_val_df, static_features_encoded
        )
        self._tick(80)
        
        # 4. Run forecasting
        # 4. forecast     (80-95 %)
        fc_cb = lambda p: self._tick(
            self._range(p/100, 80, 95)
        )
        self.config.progress_callback = fc_cb
        
        forecaster = Forecaster(self.config, self.log)
        forecast_df = forecaster._predict_and_format(
            self.model, val_inputs, val_targets, self.coord_scaler
        )
        self._tick(95)
        
        # 5. Visualize results
        visualizer = ResultsVisualizer(self.config, self.log)
        visualizer.run(forecast_df)
          
        self.log("--- Prediction Pipeline Finished Successfully ---")
        self._tick(100)

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
        
        # dynamic_features = [
        #     c for c in [self.config.gwl_col, 'rainfall_mm'] 
        #     if c in df.columns]
        
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
            _logger = self.log 
        )
        if targets['subsidence'].shape[0] == 0:
            raise ValueError("Sequence generation produced no validation samples.")
            
        self.log("  Validation sequences generated successfully.")
        return inputs, targets
    

    