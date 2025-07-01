# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Handles visualization and output for the forecasting application.

This module acts as the presentation layer for the backend workflow.
It contains the `ResultsVisualizer` class, which uses plotting
functions from `fusionlab.plot` to generate figures from the final
forecast data.

A key feature of this module is the `VIS_SIGNALS` object, a global
`QObject` with a `figure_saved` signal. This provides a robust,
decoupled mechanism for the visualization logic to notify the main
GUI thread whenever a plot has been saved to disk. This enables
features like the pop-up image previewer without creating tight
dependencies between components.

The module also intelligently configures the `matplotlib` backend.
By default, it uses the non-interactive "Agg" backend, which is
ideal for saving figures without attempting to open a display window.
This behavior can be overridden for interactive debugging by setting
the environment variable ``FUSIONLAB_HEADLESS=0``.

Attributes
----------
VIS_SIGNALS : _VisualizerSignals
    A global instance of a QObject subclass that emits signals related
    to the visualization process, such as `figure_saved`.

Classes
-------
ResultsVisualizer
    An orchestrator class that takes a configuration and a forecast
    DataFrame and generates all the required plots and visualizations.
"""

import os 
from typing import Optional

import pandas as pd 
from PyQt5.QtCore import pyqtSignal, QObject
import matplotlib
import matplotlib.pyplot as plt # noqa 

from ...registry import  ManifestRegistry, _update_manifest 
from ...utils.generic_utils import save_all_figures, apply_affix 
from ...plot.forecast import plot_forecasts, forecast_view 
from .config import SubsConfig 

if os.environ.get("FUSIONLAB_HEADLESS", "1") == "1":
    matplotlib.use("Agg")       
else:
    matplotlib.use("Qt5Agg")     


class _VisualizerSignals(QObject):                   
    figure_saved = pyqtSignal(str)                   
VIS_SIGNALS = _VisualizerSignals()                  

class ResultsVisualizer:
    """
    Handles the visualization and final saving of forecast results.
    """
    def __init__(self, config: SubsConfig,
                 log_callback: Optional[callable] = None, 
                 kind: Optional [str]=None, ):
        """
        Initializes the visualizer with a config uration object.
        """
        self.config = config
        self.log = log_callback or print
        self.kind = kind
        
    def _note(self, fname: str) -> None:
        """
        Append <fname> to the “figures” list in run_manifest.json
        (creates the list on first call).
        """
        try: 
            if self.config is None: 
                # fallback to manifest registery loc
                raise 
            _update_manifest(
                self.config.registry_path, "figures", 
                fname, # value
                as_list=True                # <- append, don’t overwrite
            )
        except: 
            _update_manifest(
                ManifestRegistry().latest_manifest(), "figures",
                fname, # value
                as_list=True  # <- append, don’t overwrite
            )
            
    def run(self, forecast_df: Optional[pd.DataFrame], stop_check =None ):
        """
        Executes the full visualization and saving pipeline.

        Args:
            forecast_df (pd.DataFrame, optional): The DataFrame containing
                formatted forecast results. If None or empty, the
                visualization steps are skipped.
        """
        if forecast_df is None or forecast_df.empty:
            self.log("Step 9 & 10: Skipping visualization and saving as no "
                     "forecast data was provided.")
            return

        self.log("Step 9: Visualizing Forecasts...")
        self._plot_main_forecasts(forecast_df, stop_check=stop_check )

        self.log("Step 10: Finalizing and Saving All Figures...")
        self._run_forecast_view(forecast_df, stop_check =stop_check  )
        # self._save_all_figures() # get the exact name
        
        self.log("\n--- WORKFLOW COMPLETED ---")
        self.log(f"All outputs are in: {self.config.run_output_path}")

    def _plot_main_forecasts(self, df: pd.DataFrame, stop_check =None ):
        """Generates the primary spatial plots for subsidence and GWL."""
        
        coord_cols = ['coord_x', 'coord_y']
        if not all(c in df.columns for c in coord_cols):
            self.log(f"  [Warning] Coordinate columns {coord_cols} not found. "
                     "Spatial plots may fail.")
            spatial_cols_arg = None
        else:
            spatial_cols_arg = coord_cols

        horizon_steps = [1, self.config.forecast_horizon_years] \
            if self.config.forecast_horizon_years > 1 else [1]
            
        forecast_years = [
            self.config.forecast_start_year + i for i in range(
                self.config.forecast_horizon_years)
        ]
        view_years = [forecast_years[step - 1] for step in horizon_steps]

        # Plot for Subsidence
        self.log("  --- Plotting Subsidence Forecasts ---")
    
        png_base_subs = os.path.join(
            self.config.run_output_path,
            f"{self.config.city_name}_{self.config.model_name}_plot_subs"
        )
        png_base_subs = apply_affix(
            png_base_subs, label=self.kind, affix_prefix='.')
        
        if stop_check and stop_check():
            raise InterruptedError("Forecasting plot configuration aborted.")
            
        plot_forecasts(
            forecast_df=df,
            target_name=self.config.subsidence_col,
            quantiles=self.config.quantiles,
            output_dim=1,
            kind="spatial",
            horizon_steps=horizon_steps,
            spatial_cols=spatial_cols_arg,
            verbose=self.config.verbose,
            cbar="uniform",
            step_names={
                f"step {step}": f'Subsidence: Year {year}'
                for step, year in zip(horizon_steps, view_years)
            },
            _logger= self.log , 
            savefig=png_base_subs, 
            save_fmts=['.png', '.pdf'], 
            stop_check = stop_check, 
        )
        png_png = f"{png_base_subs}.png" 
        self.log (f"PNG: png_base_subs= {png_png}")
        self._note(os.path.basename(png_png))  
        VIS_SIGNALS.figure_saved.emit(png_png)
        
        
        # Plot for GWL if configured
        gwl_pred_col = f"{self.config.gwl_col}_q50" if self.config.quantiles \
            else f"{self.config.gwl_col}_pred"
            
        if self.config.include_gwl_in_df and gwl_pred_col in df.columns:
            self.log("  --- Plotting GWL Forecasts ---")
            
            png_base_gwl = os.path.join(
                self.config.run_output_path,
                f"{self.config.city_name}_{self.config.model_name}_plot_gwl"
            )
            png_base_gwl =apply_affix(
                png_base_gwl, label=self.kind, affix_prefix='.')
            
            plot_forecasts(
                forecast_df=df,
                target_name=self.config.gwl_col,
                quantiles=self.config.quantiles,
                output_dim=1,
                kind="spatial",
                horizon_steps=horizon_steps,
                spatial_cols=spatial_cols_arg,
                verbose=self.config.verbose,
                cbar="uniform", # Can be set differently if desired
                titles=[f'GWL: Year {y}' for y in view_years], 
                _logger = self.log , 
                savefig= png_base_gwl,  
                save_fmts= ['.png', '.pdf'], 
                stop_check = stop_check 
            )
            png_gwl_png  = f"{png_base_gwl}.png"
            self._note(os.path.basename(png_gwl_png))  
            VIS_SIGNALS.figure_saved.emit(png_gwl_png)  
        
    def _run_forecast_view(self, df: pd.DataFrame, stop_check =None ):
        """Runs the yearly comparison plot."""
        try:
            self.log("  Generating yearly forecast comparison view...")
            
            save_base = os.path.join(
                self.config.run_output_path,
                f"{self.config.city_name}_forecast_comparison_plot"
            )
            save_base =apply_affix(
                save_base, label=self.kind, affix_prefix='.'
            )
            forecast_view(
                df,
                spatial_cols=('coord_x', 'coord_y'),
                time_col='coord_t',
                value_prefixes=[self.config.subsidence_col],
                view_quantiles=[0.5] if self.config.quantiles else None,
                savefig=save_base, 
                save_fmts=['.png', '.pdf'], 
                verbose=self.config.verbose, 
                _logger = self.log, 
                stop_check= stop_check 
            )
            png_forecast_save= f"{save_base}.png"
            self._note(os.path.basename(png_forecast_save))  
            VIS_SIGNALS.figure_saved.emit(png_forecast_save)
            
            self.log(f"  Forecast view figures saved to: {self.config.run_output_path}")
        except Exception as e:
            self.log(f"  [Warning] Could not generate forecast view plot: {e}")
            
    def _save_all_figures(self):
        """Saves all open matplotlib figures."""
        try:
            self.log("  Saving all generated figures...")
            save_all_figures(
                output_dir=self.config.run_output_path,
                prefix=f"{self.config.city_name}_{self.config.model_name}_plot_",
                fmts=['.png', '.pdf']
            )
            self.log("  All figures saved successfully.")
        except Exception as e:
            self.log(f"  [Warning] Could not save all figures: {e}")


# if __name__=="__main__": # pragma: no-cover 
#     # FUSIONLAB_HEADLESS=0 python my_script.py
#     if os.environ.get("FUSIONLAB_HEADLESS", "1") == "1":
#         matplotlib.use("Agg")        # production / in-GUI mode
#     else:
#         matplotlib.use("Qt5Agg")     # debugging outside the GUI

