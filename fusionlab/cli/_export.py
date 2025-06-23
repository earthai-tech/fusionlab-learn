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

try:
    from fusionlab.utils.geo_utils import (
        augment_spatiotemporal_data,
        generate_dummy_pinn_data
    )
    from fusionlab.utils.generic_utils import ExistenceChecker
    from fusionlab.utils.forecast_utils import pivot_forecast_dataframe
    from fusionlab.utils.forecast_utils import format_forecast_dataframe
except ImportError:
    print("Error: Could not import fusionlab utilities. "
          "Please ensure fusionlab-learn is installed correctly.", file=sys.stderr)
    sys.exit(1)

# --- Private Helper Functions for I/O and Error Handling ---

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

def handle_cli_workflow_errors(func):
    """Decorator to provide a standard try/except block for CLI workflows."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An unexpected error occurred during the workflow: {e}", file=sys.stderr)
            sys.exit(1)
    return wrapper

# --- Public Workflow Functions ---

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
    verbose: bool
):
    """
    Reads a CSV, runs the augmentation pipeline, and saves the result.
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

