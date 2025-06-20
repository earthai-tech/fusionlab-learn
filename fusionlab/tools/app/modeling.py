
import os
import pandas as pd
from typing import List, Optional, Dict, Any
from typing import Tuple
import shutil # noqa

from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences, format_pinn_predictions
from fusionlab.nn.utils import make_dict_to_tuple_fn
from fusionlab.nn import KERAS_DEPS
from fusionlab.params import LearnableK, LearnableSs, LearnableQ
from fusionlab.nn.losses import combined_quantile_loss
from fusionlab.nn.models import PIHALNet, TransFlowSubsNet

from fusionlab.tools.app.config import SubsConfig 
from fusionlab.tools.app.utils import GuiProgress 

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
        if self.config.bypass_loading: 
            return self.model  # dont load the best model 
        else: 
           return self._load_best_model()

    def _define_and_compile_model(self, input_shapes: Dict[str, Tuple]):
        """Defines the model architecture and compiles it."""
        self.log("  Defining model architecture...")
        
        def _check_physic_params (param_value, learnable_state): 
            if isinstance(param_value, str):# and v.lower() in ['learnable', 'fixed']:
                # either fixed or 'learnable',
                # model will handle it internally 
                return param_value.lower()
            # return learnable_state
            return learnable_state(initial_value= param_value) 
                
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
                "K": _check_physic_params(
                    self.config.gwflow_init_k, LearnableK),
                "Ss": _check_physic_params(
                    self.config.gwflow_init_ss, LearnableSs),
                "Q": _check_physic_params (
                    self.config.gwflow_init_q, LearnableQ),
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
        if fmt not in {"keras", "tf", "weights"}:
            self.log(f"[WARNING] Unknown save_format '{fmt}', falling back to 'keras'")
            fmt = "keras"
        
        if fmt == "tf":                       # SavedModel directory
            ckpt_name   = f"{self.config.model_name}_ckpt"
            ckpt_kwargs = {"save_format": "tf"}          # <─ explicit
        elif fmt == "weights":                # H5 weights only – always works
            ckpt_name   = f"{self.config.model_name}.weights.h5"
            ckpt_kwargs = {"save_weights_only": True}
        else:                                 # default .keras (functional only)
            ckpt_name   = f"{self.config.model_name}.keras"
            ckpt_kwargs = {}                  # beware: subclassed models fail
            
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
        # ── 2. Optional GUI progress callback ───────
        
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

        train_dataset , val_dataset = self._build_tensor_inputs(
            train_dataset, val_dataset)
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
        
    def _build_tensor_inputs (self, train_dataset, val_dataset ): 
        # 1.  Build mapper once – right after you know the dict keys
        
        feature_keys = [
            "coords",            # (B, T, 3)
            "static_features",   # (B, D_stat)               – may be None
            "dynamic_features",  # (B, T, D_dyn)
            "future_features",   # (B, H, D_future)          – may be None
        ]
        # target_keys = ["subsidence", "gwl"]  
        
        dict_to_tuple = make_dict_to_tuple_fn(
            feature_keys,
            target_keys=None,      # or None to passthrough
            allow_missing_optional=True,  # missing keys → None placeholders
        )
        # 2.  Map **all** datasets *before* calling model.fit
        train_dataset = train_dataset.map(
            dict_to_tuple,   num_parallel_calls=AUTOTUNE
        )
        val_dataset   = val_dataset.map(
            dict_to_tuple,   num_parallel_calls=AUTOTUNE
        )
        
        # 3.  (Optional) force one trace so tf.function captures the tuple shape
        # ------------------------------------------------------------------
        if not self.model.built:
            sample_inputs, _ = next(iter(train_dataset))
            self.model(sample_inputs, training=False)   # forward pass
        
        return train_dataset, val_dataset 
    
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
                dynamic_cols=[c for c in self.config.dynamic_features if c in test_df.columns],
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
            _logger = self.config.log, 
            savepath = self.config.run_output_path, 
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