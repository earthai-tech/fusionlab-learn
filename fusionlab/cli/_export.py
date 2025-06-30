# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Core workflow functions for command-line tools related to data
exporting, generation, and transformation. This module has been
refactored to apply the DRY principle by centralizing common I/O
and error handling operations.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from functools import wraps
from typing import List, Optional, Dict, Any
import pandas as pd

from fusionlab.tools.app.config import SubsConfig
from fusionlab.tools.app.processing import DataProcessor, SequenceGenerator 
from fusionlab.tools.app.modeling import ModelTrainer, Forecaster 
from fusionlab.tools.app.view import ResultsVisualizer 
from fusionlab.tools.app.inference import PredictionPipeline
from fusionlab.utils.forecast_utils import pivot_forecast_dataframe
from fusionlab.utils.forecast_utils import format_forecast_dataframe
from fusionlab.utils.generic_utils import ExistenceChecker
from fusionlab.utils.geo_utils import augment_spatiotemporal_data
from fusionlab.utils.geo_utils import generate_dummy_pinn_data

def handle_cli_workflow_errors(func):
    """Decorator to provide a standard try/except block for CLI workflows."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An unexpected error occurred during the workflow: {e}", 
                  file=sys.stderr)
            sys.exit(1)
    return wrapper

@handle_cli_workflow_errors
def run_format_forecast(
    input_file: str,
    output_file: str,
    id_vars: Optional[List[str]] = None,
    value_prefixes: Optional[List[str]] = None,
    time_col: str = 'coord_t',
):
    """
    Main workflow function that transforms a long-format forecast
    DataFrame to a wide format.
    """
    df_long = _read_csv_safely(input_file)
    
    print("Pivoting data to wide format using format_forecast_dataframe...")
    df_wide = format_forecast_dataframe(
        df_long,
        to_wide=True,
        time_col=time_col,
        id_vars=id_vars,
        value_prefixes=value_prefixes,
    )
    
    _save_csv_safely(df_wide, output_file)

@handle_cli_workflow_errors
def run_pivot_forecast(
    input_file: str,
    output_file: str,
    id_vars: List[str],
    time_col: str,
    value_prefixes: List[str],
    static_actuals_cols: Optional[List[str]] = None,
    round_time_col: bool = False,
    verbose: int = 0
):
    """
    Main workflow function that reads a long-format CSV, pivots it to a
    wide format using the core pivot utility, and saves the result.
    """
    df_long = _read_csv_safely(input_file)

    print("Pivoting data to wide format using pivot_forecast_dataframe...")
    df_wide = pivot_forecast_dataframe(
        data=df_long,
        id_vars=id_vars,
        time_col=time_col,
        value_prefixes=value_prefixes,
        static_actuals_cols=static_actuals_cols,
        round_time_col=round_time_col,
        verbose=verbose,
    )
    
    _save_csv_safely(df_wide, output_file)

@handle_cli_workflow_errors
def run_augmentation_workflow(
    input_file: str,
    output_file: str,
    mode: str,
    group_by_cols: Optional[List[str]],
    time_col: Optional[str],
    value_cols_interpolate: Optional[List[str]],
    feature_cols_augment: Optional[List[str]],
    interpolation_kwargs: Optional[Dict[str, Any]],
    augmentation_kwargs: Optional[Dict[str, Any]],
    savefile: Optional [str], 
    verbose: bool
):
    """
    Orchestrates the data augmentation workflow.

    Reads a CSV file, applies spatiotemporal augmentation (interpolation
    and/or feature noise), and saves the resulting DataFrame.
    """
    df = _read_csv_safely(input_file)
    
    print(f"Applying spatiotemporal augmentation (mode: {mode})...")
    augmented_df = augment_spatiotemporal_data(
        df=df,
        mode=mode,
        group_by_cols=group_by_cols,
        time_col=time_col,
        value_cols_interpolate=value_cols_interpolate,
        feature_cols_augment=feature_cols_augment,
        interpolation_kwargs=interpolation_kwargs,
        augmentation_kwargs=augmentation_kwargs,
        verbose=verbose
    )
    
    _save_csv_safely(augmented_df, output_file)


@handle_cli_workflow_errors
def run_dummy_data_generation(
    output_file: str,
    n_samples: int,
    **range_kwargs
):
    """
    Generates a dummy PINN dataset and saves it to a CSV file.
    """
    print(f"Generating {n_samples} dummy samples...")
    dummy_data_dict = generate_dummy_pinn_data(
        n_samples=n_samples,
        **range_kwargs
    )
    
    dummy_df = pd.DataFrame(dummy_data_dict)
    
    _save_csv_safely(dummy_df, output_file)



@handle_cli_workflow_errors
def run_training_workflow(
    data_file: str,
    model_name: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    verbose: int,
    **kwargs
):
    """
    Orchestrates the end-to-end PINN model training and evaluation workflow.

    This function initializes the configuration, runs the data processing
    pipeline, trains the model, generates forecasts, and visualizes the
    results. It is designed to be called by a CLI command.
    """
    print("🚀 Initializing workflow configuration...")
    
    config = SubsConfig(
        data_dir=os.path.dirname(data_file),
        data_filename=os.path.basename(data_file),
        model_name=model_name,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        verbose=verbose,
        save_intermediate=True,
        **kwargs # Pass any other CLI options through
    )
    print(f"Configuration loaded for model '{config.model_name}'.")

    # The rest of the logic is identical to your test script
    processor = DataProcessor(config=config)
    processed_df = processor.run()
    
    sequence_gen = SequenceGenerator(config=config)
    train_dataset, val_dataset = sequence_gen.run(
        processed_df, processor.static_features_encoded
    )
    
    sample_inputs, _ = next(iter(train_dataset))
    input_shapes = {name: tensor.shape for name, tensor in sample_inputs.items()}
    
    trainer = ModelTrainer(config=config)
    best_model = trainer.run(train_dataset, val_dataset, input_shapes)
    
    forecaster = Forecaster(config=config)
    forecast_df = forecaster.run(
        model=best_model,
        test_df=sequence_gen.test_df,
        val_dataset=val_dataset,
        static_features_encoded=processor.static_features_encoded,
        coord_scaler=sequence_gen.coord_scaler
    )
    
    visualizer = ResultsVisualizer(config=config)
    visualizer.run(forecast_df)
    print("\n--- Workflow Finished Successfully ---")
    
@handle_cli_workflow_errors
def run_inference_workflow(
    model_path: str,
    encoder_path: str,
    scaler_path: str,
    coord_scaler_path: str,
    data_file: str,
    output_dir: str,
    model_name: str,
    city_name: str,
    **kwargs
):
    """
    Orchestrates the end-to-end PINN inference workflow.

    This function loads a pre-trained model and its associated
    preprocessing artifacts, applies them to a new dataset, and
    generates forecasts and visualizations.
    """
    print("Initializing inference workflow configuration...")
    
    # Use a base config, as most settings are derived from the artifacts
    # or are not needed for pure prediction.
    config = SubsConfig(
        data_dir=os.path.dirname(data_file),
        data_filename=os.path.basename(data_file),
        model_name=model_name,
        city_name=city_name,
        output_dir=output_dir,
        save_intermediate=True,
        **kwargs # Pass through any other relevant options
    )
    print(f"Configuration loaded for model '{config.model_name}'.")
    
    # Check if all required artifacts exist before proceeding
    required_artifacts = [
        model_path, encoder_path, scaler_path, coord_scaler_path
    ]
    if not all(os.path.exists(p) for p in required_artifacts):
        raise FileNotFoundError(
            "One or more required artifacts (model, encoder, scalers) "
            "were not found. Please provide valid paths."
        )
        
    # Instantiate the prediction pipeline with the paths to the artifacts
    prediction_pipeline = PredictionPipeline(
        config=config,
        model_path=model_path,
        encoder_path=encoder_path,
        scaler_path=scaler_path,
        coord_scaler_path=coord_scaler_path
    )
    
    # Run the entire prediction and visualization workflow
    prediction_pipeline.run(validation_data_path=data_file)
    print("\n--- Workflow Finished Successfully ---")


# Utilities 
def _read_csv_safely(file_path: str) -> pd.DataFrame:
    """Reads a CSV file with standardized error handling."""
    print(f"Reading data from: {file_path}")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

def _save_csv_safely(df: pd.DataFrame, file_path: str):
    """Ensures output directory exists and saves a DataFrame to CSV."""
    output_path = Path(file_path).resolve()
    ExistenceChecker.ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved data to: {output_path}")

