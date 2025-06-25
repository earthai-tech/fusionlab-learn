# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Defines the command-line interface (CLI) for data processing tasks.

This module uses the `click` library to create a group of commands
that serve as entry points for the various data transformation and
utility workflows defined in the `_export.py` module. It handles
parsing of command-line arguments and passes them to the appropriate
backend functions, providing a user-friendly interface for running
data processing tasks from the terminal.
"""

import click
import pandas as pd
import json
from typing import List, Optional

def _parse_str_to_list(value: Optional[str]) -> Optional[List[str]]:
    """Parses a comma-separated string into a list of stripped strings."""
    if value is None:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]

@click.group(name="process")
def process_group():
    """A group of commands for data processing and transformation tasks."""
    pass

try:
    from ._export import run_pivot_forecast

    @process_group.command(name='pivot-forecast')
    @click.option(
        '--input-file', '-i', required=True,
        type=click.Path(exists=True, dir_okay=False, readable=True),
        help='Path to the long-format input CSV file.'
    )
    @click.option(
        '--output-file', '-o', required=True,
        type=click.Path(),
        help='Path to save the new wide-format output CSV file.'
    )
    @click.option(
        '--id-vars', required=True,
        help='Comma-separated ID variables to keep as index columns'
             ' (e.g., "sample_idx,coord_x").'
    )
    @click.option(
        '--time-col', required=True,
        help="Name of the column representing the time step"
             " (e.g., 'coord_t')."
    )
    @click.option(
        '--prefixes', required=True,
        help='Comma-separated value prefixes to pivot'
             ' (e.g., "subsidence,GWL").'
    )
    @click.option(
        '--static-cols',
        help='Optional comma-separated list of static actual columns'
             ' to merge back.'
    )
    def pivot_forecast_command(
        input_file, output_file, id_vars, time_col, prefixes, static_cols
    ):
        """Pivots a long-format forecast DataFrame to a wide format."""
        click.echo("🚀 Launching DataFrame pivoting tool...")
        
        # Use the helper to parse CLI arguments into lists
        id_vars_list = _parse_str_to_list(id_vars)
        prefixes_list = _parse_str_to_list(prefixes)
        static_cols_list = _parse_str_to_list(static_cols)
        
        # The underlying run_pivot_forecast is now wrapped with the
        # error handling decorator, so no try/except block is needed here.
        try:
            run_pivot_forecast(
                input_file=input_file,
                output_file=output_file,
                id_vars=id_vars_list,
                time_col=time_col,
                value_prefixes=prefixes_list,
                static_actuals_cols=static_cols_list
            )
            click.secho("✅ Successfully pivoted data.", fg='green') 
        except Exception as e: 
            click.secho(f"❌ An error occurred: {e}", fg='red', err=True)

except ImportError:
    # Fallback command if the _export module cannot be imported
    @process_group.command(name='pivot-forecast')
    def pivot_forecast_command_dummy(**kwargs):
        click.secho("Error: The `_export` tool module is missing or has "
                    "failed to import.", fg='red')

try:
    from ..utils.forecast_utils import detect_forecast_type

    @process_group.command(name='detect-forecast-type')
    @click.option(
        '--input-file', '-i', required=True,
        type=click.Path(exists=True, dir_okay=False),
        help='Path to the CSV file to inspect.'
    )
    @click.option(
        '--prefixes', help='Optional comma-separated value prefixes to check.'
    )
    def detect_forecast_type_command(input_file, prefixes):
        """Auto-detects if a forecast DataFrame is deterministic or quantile."""
        click.echo(f"Inspecting file: {input_file}")
        try:
            df = pd.read_csv(input_file)
            prefixes_list = _parse_str_to_list(prefixes)
            forecast_type = detect_forecast_type(df, value_prefixes=prefixes_list)
            click.echo("Detection complete.")
            click.secho(f"--> Detected Forecast Type: {forecast_type.upper()}", 
                        fg='cyan', bold=True)
        except Exception as e:
            click.secho(f"❌ An error occurred during detection: {e}",
                        fg='red', err=True)

except ImportError:
    @process_group.command(name='detect-forecast-type')
    def detect_forecast_type_command_dummy(**kwargs):
        click.secho("Error: The `forecast_utils` module is missing.", fg='red')


try:
    from ._export import run_format_forecast

    @process_group.command(name='format-forecast')
    @click.option(
        '--input-file', '-i', required=True,
        type=click.Path(exists=True, dir_okay=False),
        help='Path to the long-format input CSV file.'
    )
    @click.option(
        '--output-file', '-o', required=True,
        type=click.Path(),
        help='Path to save the new wide-format output CSV file.'
    )
    @click.option(
        '--id-vars', help='Comma-separated list of ID variables to keep.'
    )
    @click.option(
        '--prefixes', help='Comma-separated value prefixes to pivot.'
    )
    @click.option(
        '--time-col', default='coord_t', show_default=True,
        help="Name of the column representing the time step."
    )
    def format_forecast_command(
            input_file, output_file, id_vars, prefixes, time_col):
        """Auto-detects format and pivots a long-format DataFrame to wide."""
        click.echo("🚀 Launching DataFrame formatting tool...")
        run_format_forecast(
            input_file=input_file,
            output_file=output_file,
            id_vars=_parse_str_to_list(id_vars),
            value_prefixes=_parse_str_to_list(prefixes),
            time_col=time_col,
        )
        click.secho("✅ Successfully formatted data.", fg='green')

except ImportError:
    @process_group.command(name='format-forecast')
    def format_forecast_command_dummy(**kwargs):
        click.secho("Error: The `_export` tool module is missing.", fg='red')


try:
    from ._export import run_augmentation_workflow

    @process_group.command(name='augment-data')
    @click.option(
        '--input-file', '-i', required=True,
        type=click.Path(exists=True, dir_okay=False),
        help='Path to the input CSV data file.'
    )
    @click.option(
        '--output-file', '-o', required=True,
        type=click.Path(),
        help='Path to save the augmented output CSV file.'
    )
    @click.option(
        '--mode', required=True,
        type=click.Choice(['interpolate', 'augment_features', 'both'], 
                          case_sensitive=False),
        help='The augmentation mode to apply.'
    )
    @click.option(
        '--group-by-cols',
        help='Comma-separated columns to group by (for interpolation).',
    )
    @click.option(
        '--time-col',
        help='Name of the time column (for interpolation).'
    )
    @click.option(
        '--interpolate-cols',
        help='Comma-separated columns to interpolate.'
    )
    @click.option(
        '--augment-cols', 
        help='Comma-separated feature columns for noise augmentation.'
    )
    @click.option(
        '--interp-kwargs',
        help=(
            'JSON string for interpolation'
            ' arguments (e.g., \'{"freq": "D"}\').'
            )
    )
    @click.option(
        '--augment-kwargs',
        help=( 
            'JSON string for augmentation'
            ' arguments (e.g., \'{"noise_level": 0.05}\').'
        )
    )
    @click.option(
        '--verbose', is_flag=True, default=False,
        help='Enable verbose progress messages.'
    )
    def augment_data_command(
        input_file, output_file, mode, group_by_cols, time_col,
        interpolate_cols, augment_cols, interp_kwargs, 
        augment_kwargs, verbose
    ):
        """Applies temporal interpolation and/or feature augmentation."""
        click.echo(f"🔄 Starting data augmentation (mode: {mode})...")
        
        # Parse JSON strings into dictionaries
        interp_dict = json.loads(interp_kwargs) if interp_kwargs else None
        augment_dict = json.loads(augment_kwargs) if augment_kwargs else None

        # Call the core workflow function with parsed arguments
        try:
            run_augmentation_workflow(
                input_file=input_file,
                output_file=output_file,
                mode=mode,
                group_by_cols=_parse_str_to_list(group_by_cols),
                time_col=time_col,
                value_cols_interpolate=_parse_str_to_list(interpolate_cols),
                feature_cols_augment=_parse_str_to_list(augment_cols),
                interpolation_kwargs=interp_dict,
                augmentation_kwargs=augment_dict,
                verbose=verbose
            )
            click.secho(
                "✅ Augmented data saved successfully"
                f" to: {output_file}", fg='green')
        except Exception as e:
            click.secho(
                "❌ An error occurred during"
                f" augmentation: {e}", fg='red', err=True)
except ImportError:
    @process_group.command(name='augment-data')
    def augment_data_command_dummy(**kwargs):
        click.secho("Error: The `geo_utils` module is missing.", fg='red')

