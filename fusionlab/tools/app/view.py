# -*- coding: utf-8 -*-
import os 
import pandas as pd 

from typing import Optional
from PyQt5.QtCore import pyqtSignal, QObject

from fusionlab.utils.generic_utils import save_all_figures 
from fusionlab.plot.forecast import plot_forecasts, forecast_view 
from fusionlab.tools.app.config import SubsConfig 

import matplotlib

if os.environ.get("FUSIONLAB_HEADLESS", "1") == "1":
    matplotlib.use("Agg")        # production / in-GUI mode
else:
    matplotlib.use("Qt5Agg")     # debugging outside the GUI

import matplotlib.pyplot as plt # noqa 
# FUSIONLAB_HEADLESS=0 python my_script.py

class _VisualizerSignals(QObject):                   
    figure_saved = pyqtSignal(str)                   
VIS_SIGNALS = _VisualizerSignals()                  


class ResultsVisualizer:
    """
    Handles the visualization and final saving of forecast results.
    """
    def __init__(self, config: SubsConfig, log_callback: Optional[callable] = None):
        """
        Initializes the visualizer with a config uration object.
        """
        self.config = config
        self.log = log_callback or print

    def run(self, forecast_df: Optional[pd.DataFrame]):
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
        self._plot_main_forecasts(forecast_df)

        self.log("Step 10: Finalizing and Saving All Figures...")
        self._run_forecast_view(forecast_df)
        # self._save_all_figures() # get the exact name
        
        self.log("\n--- WORKFLOW COMPLETED ---")
        self.log(f"All outputs are in: {self.config.run_output_path}")

    def _plot_main_forecasts(self, df: pd.DataFrame):
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
            _logger= self.config.log , 
            savefig=png_base_subs, 
            save_fmts=['.png', '.pdf']
        )
        png_png = f"{png_base_subs}.png"          
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
                _logger = self.config.log , 
                savefig= png_base_gwl,  
                save_fmts= ['.png', '.pdf'], 
            )
            png_gwl_png  = f"{png_base_gwl}.png"
            VIS_SIGNALS.figure_saved.emit(png_gwl_png)  
           
    def _run_forecast_view(self, df: pd.DataFrame):
        """Runs the yearly comparison plot."""
        try:
            self.log("  Generating yearly forecast comparison view...")
            
            save_base = os.path.join(
                self.config.run_output_path,
                f"{self.config.city_name}_forecast_comparison_plot"
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
                _logger = self.config.log
            )
            png_forecast_save= f"{save_base}.png"
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
