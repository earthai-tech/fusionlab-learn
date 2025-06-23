# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

import click
import pandas as pd
import json

# Define the command group for this module
@click.group()
def process_group():
    """Commands for data processing and transformation tasks."""
    pass

# --- Add the `pivot-forecast` command ---
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
        help='Comma-separated list of ID variables to keep (e.g., "sample_idx,coord_x").'
    )
    @click.option(
        '--time-col', required=True,
        help="Name of the column representing the time step (e.g., 'coord_t')."
    )
    @click.option(
        '--prefixes', required=True,
        help='Comma-separated value prefixes to pivot (e.g., "subsidence,GWL").'
    )
    @click.option(
        '--static-cols',
        help='Optional comma-separated list of static actual columns to merge back.'
    )
    def pivot_forecast_command(
            input_file, output_file, id_vars, time_col, prefixes, static_cols
            ):
        """Pivots a long-format forecast DataFrame to a wide format."""
        click.echo("🚀 Launching DataFrame pivoting tool...")
        try:
            id_vars_list = [v.strip() for v in id_vars.split(',')]
            prefixes_list = [p.strip() for p in prefixes.split(',')]
            static_cols_list = [s.strip() for s in static_cols.split(',')] if static_cols else None
            
            run_pivot_forecast(
                input_file=input_file, output_file=output_file, id_vars=id_vars_list,
                time_col=time_col, value_prefixes=prefixes_list,
                static_actuals_cols=static_cols_list
            )
            click.secho("✅ Successfully pivoted data.", fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred: {e}", fg='red', err=True)

except ImportError:
    @process_group.command(name='pivot-forecast')
    def pivot_forecast_command_dummy(**kwargs):
        click.secho("Error: The `_export` tool is missing.", fg='red')

# --- Add the `detect-forecast-type` command ---
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
            prefixes_list = [p.strip() for p in prefixes.split(',')] if prefixes else None
            forecast_type = detect_forecast_type(df, value_prefixes=prefixes_list)
            click.echo("Detection complete.")
            click.secho(f"--> Detected Forecast Type: {forecast_type.upper()}", fg='cyan', bold=True)
        except Exception as e:
            click.secho(f"❌ An error occurred: {e}", fg='red', err=True)

except ImportError:
    @process_group.command(name='detect-forecast-type')
    def detect_forecast_type_command_dummy(**kwargs):
        click.secho("Error: The `forecast_utils` module is missing.", fg='red')

# --- 3. Add the `format-forecast` command (now much cleaner) ---
try:
    # Import the refactored workflow function
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
    def format_forecast_command(input_file, output_file, id_vars, prefixes, time_col):
        """
        Pivots a long-format forecast DataFrame to a wide format.
        """
        click.echo("🚀 Launching DataFrame formatting tool...")
        try:
            # Parse the comma-separated strings into lists
            id_vars_list = [v.strip() for v in id_vars.split(',')] if id_vars else None
            prefixes_list = [p.strip() for p in prefixes.split(',')] if prefixes else None

            # Call the workflow function directly with parsed arguments
            run_format_forecast(
                input_file=input_file,
                output_file=output_file,
                id_vars=id_vars_list,
                value_prefixes=prefixes_list,
                time_col=time_col,
            )
            click.secho("✅ Successfully formatted data.", fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred: {e}", fg='red', err=True)

except ImportError:
    @process_group.command(name='format-forecast')
    def format_forecast_command_dummy(**kwargs):
        click.secho(
            "Error: The `_export` tool is missing or could not be imported.",
            fg='red'
        )

# --- Command for `augment_spatiotemporal_data` ---
try:
    from ..utils.geo_utils import augment_spatiotemporal_data

    @process_group.command(name='augment-data')
    @click.option(
        '--input-file', '-i', required=True,
        type=click.Path(exists=True, dir_okay=False, readable=True),
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
        '--group-by-cols', help='Comma-separated columns to group by (for interpolation).',
    )
    @click.option(
        '--time-col', help='Name of the time column (for interpolation).'
    )
    @click.option(
        '--interpolate-cols', help='Comma-separated columns to interpolate.'
    )
    @click.option(
        '--augment-cols', help='Comma-separated feature columns for noise augmentation.'
    )
    @click.option(
        '--interp-kwargs',
        help='JSON string for interpolation arguments (e.g., \'{"freq": "D"}\').'
    )
    @click.option(
        '--augment-kwargs',
        help='JSON string for augmentation arguments (e.g., \'{"noise_level": 0.05}\').'
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
        """
        Applies temporal interpolation and/or feature augmentation to a
        spatiotemporal dataset.
        """
        click.echo(f"🔄 Starting data augmentation (mode: {mode})...")
        try:
            df = pd.read_csv(input_file)

            # Parse comma-separated strings into lists
            group_by_list = [c.strip() for c in group_by_cols.split(',')
                             ] if group_by_cols else None
            interpolate_list = [c.strip() for c in interpolate_cols.split(',')
                                ] if interpolate_cols else None
            augment_list = [c.strip() for c in augment_cols.split(',')
                            ] if augment_cols else None

            # Parse JSON strings into dictionaries
            interp_dict = json.loads(interp_kwargs) if interp_kwargs else None
            augment_dict = json.loads(augment_kwargs) if augment_kwargs else None

            # Call the core utility function
            augmented_df = augment_spatiotemporal_data(
                df=df,
                mode=mode,
                group_by_cols=group_by_list,
                time_col=time_col,
                value_cols_interpolate=interpolate_list,
                feature_cols_augment=augment_list,
                interpolation_kwargs=interp_dict,
                augmentation_kwargs=augment_dict,
                verbose=verbose
            )

            # Save the result
            augmented_df.to_csv(output_file, index=False)
            click.secho(f"✅ Augmented data saved successfully to: {output_file}", fg='green')

        except Exception as e:
            click.secho(f"❌ An error occurred during augmentation: {e}", fg='red', err=True)

except ImportError:
    @process_group.command(name='augment-data')
    def augment_data_command_dummy(**kwargs):
        click.secho(
            "Error: The `geo_utils` module is missing or could not be imported.",
            fg='red'
        )

