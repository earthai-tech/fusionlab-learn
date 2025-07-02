# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Configuration and Environment Setup for the GUI Application.

This module serves as the primary configuration and setup utility
for the `fusionlab` desktop application. It contains the central
`SubsConfig` class, which holds all parameters for the end-to-end
forecasting workflow.

Crucially, this module also performs an initial environment check.
It imports and runs the `_setup_env` function to ensure that all
heavy dependencies, particularly TensorFlow, are installed in the
user's environment before any other application modules are imported.
This prevents `ImportError` exceptions downstream and ensures the
application can start gracefully or fail with a clear, informative
message.
"""

import os 
import json 
from pathlib import Path 
from typing import ( 
    List, Callable,
    Dict, Any, 
    cast, 
    Optional, 
    Union 
)

import pandas as pd 
import numpy as np   

from ...registry import ManifestRegistry
from ...utils.deps_utils import get_versions
from ...utils.generic_utils import ensure_directory_exists

from ._config import setup_environment as _setup_env   

if not globals().get("_FUSIONLAB_ENV_READY", False):
    _setup_env()                           
    globals()["_FUSIONLAB_ENV_READY"] = True

class SubsConfig:
    """
    A configuration class to manage parameters for the subsidence
    forecasting GUI and the underlying processing script.

    This class centralizes all configurable options, from file paths
    and model names to feature definitions and training parameters,
    making it easy to manage and pass settings between the GUI and
    the backend processing logic.
    """
    def __init__(
        self, log_callback: Optional[callable] = None, 
        progress_callback: Optional[Callable[[int], None]] = None,
        **kwargs
    ):
        """
        Initializes the configuration with default values, which can be
        overridden by keyword arguments.
        """
        # --- File and Path Configuration ---
        self.city_name: str = 'zhongshan'
        self.model_name: str = 'TransFlowSubsNet'
        self.data_dir: str = os.getenv("JUPYTER_PROJECT_ROOT", "../..")
        self.data_filename: str = "zhongshan_500k.csv"
        self.output_dir: str = os.path.join(os.getcwd(), "results_pinn")

        # --- Time and Horizon Configuration ---
        self.train_end_year: int = 2022
        self.forecast_start_year: int = 2023
        self.forecast_horizon_years: int = 3
        self.time_steps: int = 5 # Lookback window
        
        
        # --- PINN and Model Configuration ---
        self.pde_mode: str = 'both'
        # self.pinn_coeff_c: str = 'learnable'
        # self.pde_mode: str = 'consolidation'
        self.pinn_coeff_c: str = 'learnable'
        self.lambda_pde: float = 1.0
        self.lambda_cons: float = 1.0
        self.lambda_gw: float = 1.0
        self.gwflow_init_k: float = 1e-4
        self.gwflow_init_ss: float = 1e-5
        self.gwflow_init_q: float = 0.0
        
        self.save_format : str = 'keras'
        self.bypass_loading: bool =False 
        
        self.quantiles: Optional[List[float]] = [0.1, 0.5, 0.9]
        self.mode: str = 'pihal'
        self.attention_levels: List[str] = ['1', '2', '3']
        
        # --- Training Hyperparameters ---
        self.epochs: int = 50
        self.learning_rate: float = 0.001
        self.batch_size: int = 256
        self.patience : int =15
        self.fit_verbose: int = 1 
        
        # --- Data Splitting ---
        self.validation_size: float = 0.2
        self.seed: int = 42

        self.weight_subs_pred = 1.0 
        self.weight_gwl_pred =.5 
        
        # --- Evaluation ---
        self.evaluate_coverage: bool = True

        # --- Column Definitions ---
        # User can set these to 'auto' for detection or provide specific lists.
        self.time_col: str = 'auto' # e.g., 'year'
        self.lon_col: str = 'longitude'
        self.lat_col: str = 'latitude'
        self.subsidence_col: str = 'subsidence'
        self.gwl_col: str = 'GWL'
        self.categorical_cols: Optional[List[str]] = 'auto' # or user can provide like ['geology']
        self.numerical_cols: Optional[List[str]] = 'auto'
        self.static_features: Optional[List[str]] = 'auto'
        self.dynamic_features: Optional[List[str]] = 'auto'
        self.future_features: Optional[List[str]] =  ['rainfall_mm'] # Example
        self.merge_future_into_dynamic: bool = False   # default: keep them separate

        # --- Internal State ---
        self.run_output_path: Optional[str] = None
        self.save_intermediate: bool = True
        self.include_gwl_in_df: bool = False # Control GWL inclusion in plots
        # --- Progress & logging hooks -----------------------------------
        self.log: Callable[[str], None]       = log_callback or print
        self.progress_callback: Callable[[int], None] = (
            progress_callback
            or cast(Callable[[int], None], lambda *_: None)
        )
        self.verbose = 1 
        
        # Override defaults with any user-provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self._build_paths()
        
        # save registry 
        self._registry_path =None 

    def _build_paths(self):
        """Constructs the full run output path based on current config."""
        self.run_output_path = os.path.join(
            self.output_dir, f"{self.city_name}_{self.model_name}_run"
        )
        ensure_directory_exists(self.run_output_path)

    def update_from_gui(self, gui_config: Dict[str, Any]):
        """Updates configuration from a dictionary, e.g., from the GUI state."""
        for key, value in gui_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._build_paths() # Re-build paths in case model name changed
        
    def auto_detect_columns(self, df: pd.DataFrame):
        """
        Automatically detects and sets column lists (categorical, numerical,
        static, dynamic) if they are set to 'auto'. This method respects
        user-provided lists and ensures special columns are handled correctly.
        """
        self.log("Auto-detecting column types and roles...")
        
        # --- Step 1: Detect Time Column (if set to 'auto') ---
        if self.time_col == 'auto':
            # Use a predefined list of common time-related column names
            for col in ['year', 'date', 'time', 'timestamp']:
                if col in df.columns:
                    self.time_col = col
                    self.log(f"  Detected time column: '{self.time_col}'")
                    break
            if self.time_col == 'auto':
                raise ValueError("Could not auto-detect time column. Please "
                                 "specify it in the configuration.")

        # --- Step 2: Detect Categorical Columns (if set to 'auto') ---
        if self.categorical_cols == 'auto':
            categoricals = df.select_dtypes(exclude=np.number).columns.tolist()
            # Ensure the main time column is not treated as a category
            if self.time_col in categoricals:
                categoricals.remove(self.time_col)
            self.categorical_cols = categoricals
            self.log(f"  Auto-detected categorical columns: {self.categorical_cols}")

        # --- Step 3: Detect Numerical Columns (if set to 'auto') ---
        if self.numerical_cols == 'auto':
            numerics = df.select_dtypes(include=np.number).columns.tolist()
            # Ensure the main time column is not treated as a numerical feature
            if self.time_col in numerics:
                numerics.remove(self.time_col)
            self.numerical_cols = numerics
            self.log(f"  Auto-detected numerical columns: {self.numerical_cols}")

        # --- Step 4: Determine Feature Roles (if set to 'auto') ---
        if self.static_features == 'auto':
            # Heuristic: Assume that the identified categorical columns
            # (which will be one-hot encoded) are static features.
            self.static_features = self.categorical_cols or []
            self.log(f"  Auto-set static features: {self.static_features}")

        if self.dynamic_features == 'auto':
            # Heuristic: Assume all other numerical columns are dynamic,
            # after excluding special ID, target, and known future columns.
            exclude_cols = {
                self.time_col, self.lon_col, self.lat_col,
                self.subsidence_col, self.gwl_col
            }
            # Also exclude columns already assigned to be static or future
            exclude_cols.update(self.static_features or [])
            
            if not self.merge_future_into_dynamic:
                exclude_cols.update(self.future_features or [])
            
            # Use the detected numerical_cols list as the base
            self.dynamic_features = [
                col for col in self.numerical_cols if col not in exclude_cols
            ]
            self.log(f"  Auto-set dynamic features: {self.dynamic_features}")
  
    def to_json(
        self,
        path: Optional[Union[str, os.PathLike]] = None,
        extra: Optional[dict] = None, 
        manifest_kind : Optional[str]=None, 
    ) -> Path:
        """Serializes the configuration to a JSON manifest file.

        This method intelligently handles the creation and saving of
        the run manifest.

        - If `path` is None, it creates a new, timestamped run
          directory within the central manifest registry and saves a new
          `run_manifest.json` file inside it.
        - If `path` is provided, it imports the existing manifest from
          that path into the registry, creating a new run directory
          to store a copy of it.

        This ensures that every training run, whether new or imported,
        is self-contained and managed by the registry.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            An explicit path to an existing JSON manifest file to be
            imported into the registry. If None, a new run is created.
            Default is None.
        extra : dict, optional
            Extra top-level keys to merge into the JSON, such as Git
            metadata or package versions. These keys will override any
            duplicates from the main configuration.

        Returns
        -------
        pathlib.Path
            The absolute path to the saved manifest file *inside* the
            central registry.
        """
        registry = ManifestRegistry(
            manifest_kind= manifest_kind or 'training') 
        
        manifest_path: Path

        if path is None:
            # Case 1: No path provided. Create a new run directory.
            run_dir = registry.new_run_dir(
                city=self.city_name, model=self.model_name
            )
            
        else:
            # Case 2: User provided an external manifest. Import it.
            # The import_manifest method copies the file into a new
            # run directory within the registry and returns the new path.
            self.log(f"Importing external manifest from '{path}' into registry.")
            run_dir = registry.import_manifest(path)
            # No need to Update config's run_output_path to point to the new location
            # self.run_output_path = str(manifest_path.parent)
            
        # Keep the registry path as property 
        self._registry_path = run_dir 
        
        # save manifest file absolute path 
        manifest_path = run_dir / registry._manifest_filename
        
        # 1. Create a flat copy of all public attributes for serialization.
        cfg_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
            # skip private internals
            # and isinstance(v, (int, float, str, bool, list, dict))
        }
        # 2. Allow the caller to add extra metadata.
        content = {"configuration": cfg_dict, 'git': get_versions() }
        if extra:
            content.update(extra)
    
        # 3. Write the content to the JSON file using an atomic write.
        tmp_path = manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(
            content, indent=2), encoding="utf-8")
        os.replace(tmp_path, manifest_path)

        # 4. Log the successful export.
        if callable(getattr(self, "log", None)):
            self.log(
                f"  JSON manifest saved successfully: {manifest_path}")
  
        return manifest_path
    
    @property
    def registry_path(self) -> Path:
        return self._registry_path
    
    def __repr__(self) -> str:
        """Provides a string representation of the configuration."""
        params = "\n".join(
            f"  {key}: {value}" for key, value in self.__dict__.items())
        return f"SubsConfig(\n{params}\n)"
    
