# fusionlab/cli/__init__.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Main entry point for the fusionlab-learn Command-Line Interface (CLI).

This script initializes the main `click` group and dynamically registers
all the command sub-groups from the other modules in this package. It also
includes a safeguard to ensure the `click` library is installed.
"""

# --- Safeguard for Core CLI Dependency ---
try:
    import click
except ImportError:
    raise ImportError(
        "The 'click' library is required to use the fusionlab-learn CLI.\n"
        "It should be installed automatically as a core dependency.\n\n"
        "Please try reinstalling the package or install it manually:\n"
        "    pip install click"
    )

from .app import app_group
from .forecast import forecast_group
from .process import process_group
from .tune import tune_group
from .view import view_group
from .geotools import geotools_group
from .inference import inference_group 

@click.group()
def cli():
    """
    FusionLab-learn: A command-line toolkit for advanced time series
    forecasting and analysis.
    """
    pass

# --- Register all Command Groups ---
# This attaches each sub-command group to the main `cli` object.
cli.add_command(app_group, name='app')
cli.add_command(forecast_group, name='forecast')
cli.add_command(process_group, name='process')
cli.add_command(tune_group, name='tune')
cli.add_command(view_group, name='view')
cli.add_command(geotools_group, name='geotools')
cli.add_command(inference_group,   name='inference')   