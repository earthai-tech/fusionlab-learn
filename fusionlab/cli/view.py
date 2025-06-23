# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>


"""
Defines the 'view' command group for the fusionlab-learn CLI, providing
tools for visualizing forecast results and data.
"""
from __future__ import annotations 

import click
import pandas as pd
from typing import Dict, Any

# --- Helper function to parse comma-separated values ---
def _parse_list(ctx, param, value):
    if value is None:
        return None
    try:
        # Handles both floats and integers
        items = [item.strip() for item in value.split(',')]
        return [float(i) if '.' in i else int(i) for i in items]
    except ValueError:
        raise click.BadParameter(
            f"Could not parse '{value}'. Please provide a comma-separated list of numbers."
        )
        
def _parse_dict(ctx, param, value: str) -> Dict[int, Any] | None:
    """Helper to parse key:value comma-separated strings."""
    if value is None:
        return None
    try:
        items = {}
        for pair in value.split(','):
            if ':' not in pair:
                raise ValueError("Pairs must be in 'key:value' format.")
            key, val = pair.split(':', 1)
            items[int(key.strip())] = val.strip()
        return items
    except Exception as e:
        raise click.BadParameter(f"Could not parse step names: {e}")

# --- Define the 'view' command group ---
@click.group()
def view_group():
    """Commands for visualizing forecast results."""
    pass
# --- Command for `plot_forecasts` ---

try:
    from ..plot.forecast import plot_forecasts

    @view_group.command(name='plot-forecasts')
    @click.option('--input-file', '-i', required=True,
                  type=click.Path(exists=True, dir_okay=False, readable=True),
                  help='Path to the long-format forecast CSV file.')
    @click.option('--target-name', required=True,
                  help='Base name of the target variable to plot (e.g., "subsidence").')
    @click.option('--kind', type=click.Choice(['temporal', 'spatial'], case_sensitive=False),
                  default='temporal', show_default=True, help='Type of plot to generate.')
    @click.option('--quantiles', callback=_parse_list,
                  help='Comma-separated list of quantiles (e.g., "0.1,0.5,0.9").')
    @click.option('--horizon-steps', callback=_parse_list,
                  help='Comma-separated list of horizon steps to plot.')
    @click.option('--spatial-cols', help='Comma-separated lon,lat columns for spatial plots.')
    @click.option('--savefig', type=click.Path(), help='Path to save the output plot file.')
    def plot_forecasts_command(
        input_file, target_name, kind, quantiles, horizon_steps,
        spatial_cols, savefig
    ):
        """
        Generates and displays forecast plots using the plot_forecasts utility.
        """
        click.echo(f"📊 Generating '{kind}' forecast plot for target '{target_name}'...")
        try:
            df = pd.read_csv(input_file)
            
            spatial_cols_list = [c.strip() for c in spatial_cols.split(',')
                                 ] if spatial_cols else None

            plot_forecasts(
                forecast_df=df,
                target_name=target_name,
                kind=kind,
                quantiles=quantiles,
                horizon_steps=horizon_steps,
                spatial_cols=spatial_cols_list,
                savefig=savefig,
                show=True # Always show when run from CLI
            )
            click.secho("✅ Plot generated successfully.", fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred during plotting: {e}", fg='red', err=True)

except ImportError:
    @view_group.command(name='plot-forecasts')
    def plot_forecasts_command_dummy(**kwargs):
        click.secho("Error: The `plot_forecasts` utility is missing or has an error.", fg='red')

# --- Command for `forecast_view` ---
try:
    from ..plot.forecast import forecast_view

    @view_group.command(name='forecast-view')
    @click.option('--input-file', '-i', required=True,
                  type=click.Path(exists=True, dir_okay=False, readable=True),
                  help='Path to the forecast CSV file.')
    @click.option('--prefixes', help='Comma-separated list of value prefixes to view.')
    @click.option('--quantiles', callback=_parse_list, help='Comma-separated quantiles to view.')
    @click.option('--savefig', type=click.Path(), help='Path to save the output plot file.')
    def forecast_view_command(input_file, prefixes, quantiles, savefig):
        """
        Creates a yearly comparison plot of forecast results.
        """
        click.echo("🖼️ Generating  forecast comparison view...")
        try:
            df = pd.read_csv(input_file)
            prefixes_list = [p.strip() for p in prefixes.split(',')] if prefixes else None

            forecast_view(
                forecast_df=df,
                value_prefixes=prefixes_list,
                view_quantiles=quantiles,
                savefig=savefig,
                show=True
            )
            click.secho("✅ Forecast view generated successfully.", fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred: {e}", fg='red', err=True)

except ImportError:
    @view_group.command(name='forecast-view')
    def forecast_view_command_dummy(**kwargs):
        click.secho("Error: The `forecast_view` utility is missing or has an error.", fg='red')

# --- Command for `plot_forecast_by_step` ---
try:
    from ..plot.forecast import plot_forecast_by_step

    @view_group.command(name='plot-by-step')
    @click.option('--input-file', '-i', required=True,
                  type=click.Path(exists=True, dir_okay=False, readable=True),
                  help='Path to the forecast CSV file.')
    @click.option('--prefixes', help='Comma-separated list of value prefixes to plot.')
    @click.option('--steps', callback=_parse_list, help='Comma-separated integer steps to plot.')
    @click.option('--step-names', callback=_parse_dict,
                  help="Mapping of step to name, e.g., '1:2023,3:2025'")
    @click.option('--kind', type=click.Choice(['dual', 'spatial', 'temporal']), default='dual',
                  show_default=True)
    @click.option('--savefig', type=click.Path(), help='Path to save the output plot file.')
    def plot_by_step_command(input_file, prefixes, steps, step_names, kind, savefig):
        """Generates plots organized by forecast step."""
        click.echo(f"📊 Generating '{kind}' plots, organized by forecast step...")
        try:
            df = pd.read_csv(input_file)
            prefixes_list = [
                p.strip() for p in prefixes.split(',')] if prefixes else None
            plot_forecast_by_step(
                df=df,
                value_prefixes=prefixes_list,
                kind=kind,
                steps=steps,
                step_names=step_names,
                savefig=savefig
            )
            click.secho("✅ Step plots generated successfully.", fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred: {e}", fg='red', err=True)
except ImportError:
    @view_group.command(name='plot-by-step')
    def plot_by_step_command_dummy(**kwargs):
        click.secho("Error: The `plot_forecast_by_step` utility is missing.", fg='red')

# --- Command for `visualize_forecasts` ---
try:
    from ..plot.forecast import visualize_forecasts

    @view_group.command(name='visualize')
    @click.option('--input-file', '-i', required=True, type=click.Path(exists=True),
                  help='Path to the forecast CSV file.')
    @click.option('--target-name', required=True, help='Name of the target variable (prefix).')
    @click.option('--dt-col', required=True, help='Name of the datetime/year column.')
    @click.option('--savefig', type=click.Path(), help='Path to save the output plot file.')
    def visualize_command(input_file, target_name, dt_col, savefig):
        """A specialized tool for creating spatial forecast visualizations."""
        click.echo("Generating specialized forecast visualization...")
        try:
            df = pd.read_csv(input_file)
            visualize_forecasts(
                forecast_df=df,
                dt_col=dt_col,
                tname=target_name,
                savefig=savefig,
                show=True
            )
            click.secho("✅ Visualization generated successfully.", fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred: {e}", fg='red', err=True)
except ImportError:
    @view_group.command(name='visualize')
    def visualize_command_dummy(**kwargs):
        click.secho("Error: The `visualize_forecasts` utility is missing.", fg='red')
