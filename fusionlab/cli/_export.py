
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Core workflow functions for command-line tools related to data
exporting, generation, and transformation.
"""
from __future__ import annotations 

import sys
import os

from pathlib import Path
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

    Args:
        input_file (str): Path to the long-format input CSV file.
        output_file (str): Path to save the new wide-format output CSV file.
        id_vars (List[str], optional): List of ID variables to keep.
        value_prefixes (List[str], optional): List of value prefixes to pivot.
        time_col (str): The name of the column representing the time step.
    """
    try:
        print(f"Reading long-format data from: {input_file}")
        df_long = pd.read_csv(input_file)
        
        # Prepare kwargs for the formatting function
        pivot_kwargs = {
            'id_vars': id_vars,
            'value_prefixes': value_prefixes
        }
        
        print("Pivoting data to wide format...")
        df_wide = format_forecast_dataframe(
            df_long,
            to_wide=True,
            time_col=time_col,
            **pivot_kwargs
        )

        # Save the result
        output_path = Path(output_file)
        ExistenceChecker.ensure_directory(output_path.parent)
        df_wide.to_csv(output_path, index=False)
        print(f"Successfully saved wide-format data to: {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'", 
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during processing: {e}", 
              file=sys.stderr)
        sys.exit(1)

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
    wide format, and saves the result.

    Args:
        input_file (str): Path to the long-format input CSV file.
        output_file (str): Path to save the new wide-format output CSV.
        id_vars (List[str]): List of ID variables to keep as index columns.
        time_col (str): The name of the column representing the time step.
        value_prefixes (List[str]): List of value prefixes to pivot.
        static_actuals_cols (List[str], optional): Static columns to merge back.
        round_time_col (bool): Whether to round the time column to integers.
        verbose (int): Verbosity level for logging.
    """
    try:
        print(f"Reading long-format data from: {input_file}")
        df_long = pd.read_csv(input_file)

        print("Pivoting data to wide format...")
        df_wide = pivot_forecast_dataframe(
            data=df_long,
            id_vars=id_vars,
            time_col=time_col,
            value_prefixes=value_prefixes,
            static_actuals_cols=static_actuals_cols,
            round_time_col=round_time_col,
            verbose=verbose,
        )

        # Ensure output directory exists and save the result
        output_path = os.path.abspath(output_file)
        ExistenceChecker.ensure_directory(os.path.dirname(output_path))
        df_wide.to_csv(output_path, index=False)
        print(f"Successfully saved wide-format data to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)

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
    print(f"Reading input data from: {input_file}")
    df = pd.read_csv(input_file)
    
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
    
    # Ensure output directory exists and save the result
    output_path = os.path.abspath(output_file)
    ExistenceChecker.ensure_directory(os.path.dirname(output_path))
    augmented_df.to_csv(output_path, index=False)
    print(f"Augmented data saved successfully to: {output_path}")

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
    
    # The function now returns a DataFrame directly
    dummy_df = pd.DataFrame(dummy_data_dict)

    # Ensure output directory exists and save the result
    output_path = os.path.abspath(output_file)
    ExistenceChecker.ensure_directory(os.path.dirname(output_path))
    dummy_df.to_csv(output_path, index=False)
    print(f"Dummy data saved successfully to: {output_path}")

