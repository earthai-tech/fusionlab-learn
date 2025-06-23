# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Defines the 'geotools' command group for the fusionlab-learn CLI,
providing tools for geospatial data generation and augmentation.
"""
import click
import json
import pandas as pd 

# --- Helper function to parse comma-separated values ---
def _parse_list(ctx, param, value):
    if value is None:
        return None
    try:
        # Handles both floats and integers
        return [item.strip() for item in value.split(',')]
    except ValueError:
        raise click.BadParameter(
            f"Could not parse '{value}'. Please provide a comma-separated list."
        )
        
# --- Define the 'geotools' command group ---
@click.group()
def geotools_group():
    """Commands for geospatial data generation and augmentation."""
    pass

try:
    from ._export import run_augmentation_workflow

    @geotools_group.command(name='augment')
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
    @click.option('--group-by',
                  help='Comma-separated columns to group by (for interpolation).')
    @click.option('--time-col', 
                  help='Name of the time column (for interpolation).')
    @click.option('--interp-cols',
                  help='Comma-separated columns to interpolate.')
    @click.option('--augment-cols',
                  help='Comma-separated columns for noise augmentation.')
    @click.option('--interp-kwargs',
                  help='JSON string for interpolation args (e.g., \'{"freq":"D"}\').')
    @click.option('--augment-kwargs',
                  help='JSON string for augmentation args (e.g., \'{"noise_level":0.05}\').')
    def augment_command(
        input_file, output_file, mode, group_by, time_col,
        interp_cols, augment_cols, interp_kwargs, augment_kwargs
    ):
        """Applies temporal interpolation and/or feature augmentation."""
        click.echo(f"🔄 Starting data augmentation (mode: {mode})...")
        try:
            group_list = [c.strip() for c in group_by.split(',')] if group_by else None
            interp_list = [c.strip() for c in interp_cols.split(',')] if interp_cols else None
            augment_list = [c.strip() for c in augment_cols.split(',')] if augment_cols else None
            interp_dict = json.loads(interp_kwargs) if interp_kwargs else None
            augment_dict = json.loads(augment_kwargs) if augment_kwargs else None
            
            run_augmentation_workflow(
                input_file=input_file, output_file=output_file, mode=mode,
                group_by_cols=group_list, time_col=time_col,
                value_cols_interpolate=interp_list, feature_cols_augment=augment_list,
                interpolation_kwargs=interp_dict, augmentation_kwargs=augment_dict,
                verbose=True
            )
            click.secho("✅ Augmented data saved successfully.", fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred: {e}", fg='red', err=True)

except ImportError:
    @geotools_group.command(name='augment')
    def augment_command_dummy(**kwargs):
        click.secho("Error: The `_export` tool is missing.", fg='red')


# --- Command for `generate_dummy_pinn_data` ---
try:
    from ._export import run_dummy_data_generation

    @geotools_group.command(name='generate-dummy')
    @click.option('--output-file', '-o', required=True, type=click.Path(), 
                  help='Path to save the dummy CSV file.')
    @click.option('--n-samples', default=1000, show_default=True,
                  help='Number of samples to generate.')
    def generate_dummy_command(output_file, n_samples):
        """Generates a synthetic dataset for PINN models."""
        click.echo("🧬 Generating synthetic PINN data...")
        try:
            run_dummy_data_generation(output_file=output_file, n_samples=n_samples)
            click.secho("✅ Dummy data saved successfully.", fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred: {e}", fg='red', err=True)
            
except ImportError:
    @geotools_group.command(name='generate-dummy')
    def generate_dummy_command_dummy(**kwargs):
        click.secho("Error: The `_export` tool is missing.", fg='red')


# --- Command for `spatial_sampling` ---
try:
    from ..utils.spatial_utils import spatial_sampling

    @geotools_group.command(name='sample')
    @click.option(
        '--input-file', '-i', required=True,
        type=click.Path(exists=True, dir_okay=False, readable=True),
        help='Path to the input CSV data file to sample from.'
    )
    @click.option(
        '--output-file', '-o', required=True, type=click.Path(),
        help='Path to save the sampled output CSV file.'
    )
    @click.option(
        '--sample-size', default=0.01, show_default=True, type=float,
        help='Fraction (e.g., 0.1) or absolute number of samples to draw.'
    )
    @click.option(
        '--stratify-by', callback=_parse_list,
        help='Comma-separated columns to stratify on (e.g., "year,category").'
    )
    @click.option(
        '--spatial-bins', default=10, show_default=True, type=int,
        help='Number of bins for spatial stratification.'
    )
    @click.option(
        '--method', type=click.Choice(['abs', 'relative'], case_sensitive=False),
        default='abs', show_default=True, help='Sampling method.'
    )
    def sample_command(
        input_file, output_file, sample_size, stratify_by, spatial_bins, method
    ):
        """Performs stratified spatial sampling on a dataset."""
        click.echo("🔬 Performing stratified spatial sampling...")
        try:
            df = pd.read_csv(input_file)
            # The @SaveFile decorator on the function handles the saving
            spatial_sampling(
                data=df,
                sample_size=sample_size,
                stratify_by=stratify_by,
                spatial_bins=spatial_bins,
                method=method,
                savefile=output_file,
                verbose=1
            )
            click.secho(f"✅ Sampled data saved successfully to: {output_file}",
                        fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred during sampling: {e}", 
                        fg='red', err=True)

except ImportError:
    @geotools_group.command(name='sample')
    def sample_command_dummy(**kwargs):
        click.secho("Error: The `spatial_utils` module is missing or has an error.", fg='red')
