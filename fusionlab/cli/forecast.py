
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Main Command-Line Interface (CLI) for the fusionlab-learn package.

This module uses the 'Click' library to create a powerful, nested
command structure, allowing users to run various tools and applications
directly from their terminal.
"""
import click

@click.group()
def forecast_group():
    """Commands for running forecasting pipelines."""
    pass

try:
    from .tools.xtft_proba_p import run_workflow as forecast_xtft_proba_main

    @forecast_group.command(name='xtft-proba')
    @click.option(
        '--data-path', required=True, type=click.Path(exists=True, file_okay=False),
        help='Path to the data directory containing the required CSV files.'
    )
    @click.option('--epochs', default=100, show_default=True, 
                  type=int, help='Number of training epochs.')
    @click.option('--batch-size', default=32, show_default=True, 
                  type=int, help='Training batch size.')
    @click.option('--time-steps', default=4, show_default=True, 
                  type=int, help='Number of look-back time steps.')
    @click.option('--horizon', default=4, show_default=True,
                  type=int, help='Number of future steps to predict.')
    @click.option(
        '--quantiles', default='0.1,0.5,0.9', show_default=True,
        help='Comma-separated list of quantiles for prediction (e.g., "0.1,0.5,0.9").'
    )
    @click.option(
        '--verbose', default=1, type=click.IntRange(0, 2), show_default=True,
        help="Verbosity level (0: quiet, 1: info, 2: debug)."
    )
    def forecast_xtft_proba_command(
        data_path, epochs, batch_size, time_steps, horizon, quantiles, verbose
    ):
        """Runs the probabilistic XTFT forecasting tool."""
        # Parse the comma-separated quantiles into a list of floats
        try:
            quantile_list = [float(q.strip()) for q in quantiles.split(',')]
        except (ValueError, AttributeError):
            click.secho(f"Error: Invalid format for quantiles: '{quantiles}'."
                        " Please provide a comma-separated list of numbers.", fg='red')
            return
        
        click.echo("🚀 Launching Probabilistic XTFT Forecasting workflow...")
        try:
            forecast_xtft_proba_main(
                data_path=data_path,
                epochs=epochs,
                batch_size=batch_size,
                time_steps=time_steps,
                forecast_horizon=horizon,
                quantiles=quantile_list,
                verbose=verbose, 
            )
            click.secho("✅ Workflow finished successfully.", fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred during the workflow: {e}", 
                        fg='red', err=True)

except ImportError:
    @forecast_group.command(name='xtft-proba')
    def forecast_xtft_proba_command(**kwargs):
        click.secho("Error: The 'xtft_proba_p' tool is missing or could not be imported.",
                    fg='red', err=True)
    
try:
    from .tools.xtft_point_p import run_point_workflow

    @forecast_group.command(name='xtft-point')
    @click.option(
        '--data-path', required=True, type=click.Path(exists=True, file_okay=False),
        help='Path to the data directory containing final_data.csv.'
    )
    @click.option('--epochs', default=100, show_default=True,
                  type=int, help='Number of training epochs.')
    @click.option('--batch-size', default=32, show_default=True,
                  type=int, help='Training batch size.')
    @click.option('--time-steps', default=4, show_default=True, 
                  type=int, help='Number of look-back time steps.')
    @click.option('--horizon', default=4, show_default=True, 
                  type=int, help='Number of future steps to predict.')
    @click.option(
        '--verbose', default=1, type=click.IntRange(0, 2), show_default=True,
        help="Verbosity level (0: quiet, 1: info, 2: debug)."
    )
    def forecast_xtft_point_command(
            data_path, epochs, batch_size, time_steps, horizon, verbose):
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
            click.secho(f"❌ An error occurred during the workflow: {e}", 
                        fg='red', err=True)

except ImportError:
    @forecast_group.command(name='xtft-point')
    def forecast_xtft_point_command_dummy(**kwargs):
        click.secho("Error: The 'xtft_point_p' tool is missing or could not be imported.",
                    fg='red', err=True)
        
 