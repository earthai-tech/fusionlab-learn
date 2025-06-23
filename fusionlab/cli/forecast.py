# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Defines the 'forecast' command group for the fusionlab-learn CLI.

This module provides high-level commands for running complete end-to-end
forecasting workflows using the library's advanced models like XTFT.
"""
import os
import click
from typing import List

# --- Reusable Decorator for Common CLI Options ---

def common_forecast_options(f):
    """
    A decorator that applies a standard set of command-line options
    for forecasting workflows to a click command.
    """
    options = [
        click.option(
            '--data-path', required=True,
            type=click.Path(exists=True, file_okay=False, resolve_path=True),
            help='Path to the data directory containing the required CSV files.'
        ),
        click.option(
            '--epochs', default=100, show_default=True, type=int,
            help='Number of training epochs.'
        ),
        click.option(
            '--batch-size', default=32, show_default=True, type=int,
            help='Training batch size.'
        ),
        click.option(
            '--time-steps', default=4, show_default=True, type=int,
            help='Number of look-back time steps.'
        ),
        click.option(
            '--horizon', default=4, show_default=True, type=int,
            help='Number of future steps to predict.'
        ),
        click.option(
            '--verbose', default=1, type=click.IntRange(0, 2), show_default=True,
            help="Verbosity level (0: quiet, 1: info, 2: debug)."
        )
    ]
    for option in reversed(options):
        f = option(f)
    return f

def _parse_quantiles(ctx, param, value: str) -> List[float]:
    """Helper to parse comma-separated quantiles into a list of floats."""
    try:
        return [float(q.strip()) for q in value.split(',')]
    except (ValueError, AttributeError):
        raise click.BadParameter(
            f"Invalid format for quantiles: '{value}'. Please provide a "
            "comma-separated list of numbers (e.g., '0.1,0.5,0.9')."
        )

def _add_pinn_workflow_options(f):
    """
    A decorator that applies a standard set of command-line options
    for the PINN forecasting workflows to a click command.

    This promotes consistency and reduces code duplication between
    related commands like training and inference.
    """
    # Define all options in a list for easy management
    options = [
        # --- File and Path Configuration ---
        click.option(
            '--data-file', '-i', 'data_file', required=True,
            type=click.Path(exists=True, dir_okay=False, resolve_path=True),
            help='Path to the input data CSV file.'
        ),
        click.option(
            '--output-dir', default='./results_pinn', show_default=True,
            type=click.Path(),
            help='Root directory to save all outputs.'
        ),
        click.option(
            '--city-name', default='zhongshan', show_default=True,
            help='Name of the city/dataset for naming output files.'
        ),

        # --- Model and Training Configuration ---
        click.option(
            '--model-name', default='TransFlowSubsNet', show_default=True,
            type=click.Choice(['TransFlowSubsNet', 'PIHALNet']),
            help='The name of the model architecture to use.'
        ),
        click.option(
            '--epochs', default=50, show_default=True, type=int,
            help='Number of training epochs.'
        ),
        click.option(
            '--batch-size', default=256, show_default=True, type=int,
            help='Training batch size.'
        ),
        click.option(
            '--lr', '--learning-rate', 'learning_rate', default=0.001,
            show_default=True, type=float, help='Optimizer learning rate.'
        ),
        click.option(
            '--patience', default=15, show_default=True, type=int,
            help='Patience for early stopping callback.'
        ),

        # --- Time and Horizon Configuration ---
        click.option(
            '--train-end-year', default=2022, show_default=True, type=int,
            help="Last year of data to include in the training set."
        ),
        click.option(
            '--forecast-start-year', default=2023, show_default=True, type=int,
            help="First year for which to generate forecasts."
        ),
        click.option(
            '--horizon', 'forecast_horizon_years', default=3, show_default=True,
            type=int, help="Number of future years to predict."
        ),
        click.option(
            '--time-steps', default=5, show_default=True, type=int,
            help="Number of look-back time steps for the model's encoder."
        ),

        # --- PINN Physics Configuration ---
        click.option(
            '--pde-mode', default='both', show_default=True,
            type=click.Choice(['both', 'consolidation', 'gw_flow', 'none']),
            help='Which physics loss components to activate.'
        ),
        click.option(
            '--lambda-cons', default=1.0, show_default=True, type=float,
            help='Weight for the consolidation physics loss.'
        ),
        click.option(
            '--lambda-gw', default=1.0, show_default=True, type=float,
            help='Weight for the groundwater flow physics loss.'
        ),
        
        # --- General Options ---
        click.option(
            '--verbose', default=1, type=click.IntRange(0, 2), show_default=True,
            help="Verbosity level for logging (0: quiet, 1: info, 2: debug)."
        )
    ]
    # Apply the decorators to the function `f` in reverse order
    for option in reversed(options):
        f = option(f)
    return f

def _add_pinn_inference_options(f):
    """Applies a standard set of PINN inference options to a command."""
    options = [
        click.option(
            '--data-file', '-i', 'data_file', required=True,
            type=click.Path(exists=True, dir_okay=False, resolve_path=True),
            help='Path to the new CSV data file for prediction.'
        ),
        click.option(
            '--model-path', required=True, type=click.Path(exists=True),
            help='Path to the pre-trained .keras model file.'
        ),
        click.option(
            '--artifacts-dir', required=True, type=click.Path(exists=True),
            help='Path to the directory containing training artifacts (scalers, encoders).'
        ),
        click.option(
            '--output-dir', default='./results_pinn_inference', show_default=True,
            type=click.Path(), help='Directory to save inference outputs.'
        ),
        click.option(
            '--model-name', default='TransFlowSubsNet', show_default=True,
            help='Name of the trained model architecture (for naming output files).'
        ),
        click.option(
            '--city-name', default='zhongshan', show_default=True,
            help='Name of the city/dataset (for naming output files).'
        ),
    ]
    for option in reversed(options):
        f = option(f)
    return f

# --- Main 'forecast' Command Group ---
@click.group(name="forecast")
def forecast_group():
    """Commands for running end-to-end forecasting pipelines."""
    pass

try:
    from ..tools.xtft_proba_p import run_workflow as forecast_xtft_proba_main

    @forecast_group.command(name='xtft-proba')
    @common_forecast_options
    @click.option(
        '--quantiles', default='0.1,0.5,0.9', show_default=True,
        callback=_parse_quantiles,
        help='Comma-separated list of quantiles for prediction.'
    )
    def forecast_xtft_proba_command(
        data_path, epochs, batch_size, time_steps, horizon, quantiles, verbose
    ):
        """Runs the probabilistic XTFT forecasting tool."""
        click.echo("🚀 Launching Probabilistic XTFT Forecasting workflow...")
        try:
            forecast_xtft_proba_main(
                data_path=data_path,
                epochs=epochs,
                batch_size=batch_size,
                time_steps=time_steps,
                forecast_horizon=horizon,
                quantiles=quantiles,
                verbose=verbose,
            )
            click.secho("✅ Workflow finished successfully.", fg='green')
        except Exception as e:
            click.secho(
                f"❌ An error occurred during the workflow: {e}",
                fg='red', err=True
            )

except ImportError:
    @forecast_group.command(name='xtft-proba')
    def forecast_xtft_proba_command_dummy(**kwargs):
        click.secho(
            "Error: The 'xtft_proba_p' tool is missing or could not be imported.",
            fg='red', err=True
        )

try:
    from .tools.xtft_point_p import run_point_workflow

    @forecast_group.command(name='xtft-point')
    @common_forecast_options
    def forecast_xtft_point_command(
        data_path, epochs, batch_size, time_steps, horizon, verbose
    ):
        """Runs the deterministic (point) XTFT forecasting tool."""
        click.echo("🚀 Launching Point XTFT Forecasting workflow...")
        try:
            run_point_workflow(
                data_path=data_path,
                epochs=epochs,
                batch_size=batch_size,
                time_steps=time_steps,
                forecast_horizon=horizon,
                verbose=verbose
            )
            click.secho("✅ Workflow finished successfully.", fg='green')
        except Exception as e:
            click.secho(
                f"❌ An error occurred during the workflow: {e}",
                fg='red', err=True
            )

except ImportError:
    @forecast_group.command(name='xtft-point')
    def forecast_xtft_point_command_dummy(**kwargs):
        click.secho(
            "Error: The 'xtft_point_p' tool is missing or could not be imported.",
            fg='red', err=True
        )


try:
    # Import the new workflow function
    from ._export import run_training_workflow

    @forecast_group.command(name='train-pinn')
    @_add_pinn_workflow_options 
    @click.option(
        '--train-end-year', type=int, default=2022, show_default=True,
        help="Last year of data to include in the training set."
    )
    def train_pinn_command(**kwargs):
        """
        Runs the full end-to-end training and forecasting pipeline for a
        PINN model.
        """
        click.echo("🚀 Launching PINN Training and Forecasting Workflow...")
        
        # The kwargs dict will contain all options from the decorator
        # and the command itself. We can pass it directly.
        run_training_workflow(**kwargs)
        
        click.secho(
            "✅ PINN workflow finished successfully.", fg='green'
        )

except ImportError:
    @forecast_group.command(name='train-pinn')
    def train_pinn_command_dummy(**kwargs):
        click.secho(
            "Error: The `_export` module or one of its dependencies"
            " is missing.", fg='red', err=True
        )

try:
    
    from ._export import run_inference_workflow 
    
    @forecast_group.command(name='infer-pinn')
    @_add_pinn_inference_options
    def infer_pinn_command(
        data_file, model_path, artifacts_dir, output_dir, model_name, 
        city_name, **kwargs
    ):
        """Runs inference using a pre-trained PINN model and its artifacts."""
        click.echo("🚀 Launching PINN Inference Workflow...")
        
        # Construct full paths to artifacts from the artifacts_dir
        encoder_path = os.path.join(artifacts_dir, "ohe_encoder.joblib")
        scaler_path = os.path.join(artifacts_dir, "main_scaler.joblib")
        coord_scaler_path = os.path.join(artifacts_dir, "coord_scaler.joblib")
        
        run_inference_workflow(
            data_file=data_file,
            model_path=model_path,
            encoder_path=encoder_path,
            scaler_path=scaler_path,
            coord_scaler_path=coord_scaler_path,
            output_dir=output_dir,
            model_name=model_name,
            city_name=city_name,
            **kwargs
        )
        click.secho("✅ Inference workflow finished successfully.", fg='green')

except ImportError:
    @forecast_group.command(name='infer-pinn')
    def infer_pinn_dummy(**kwargs):
        click.secho("Error: The 'infer-pinn' tool is unavailable.", fg='red')
