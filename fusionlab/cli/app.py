# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Defines the command-line interface (CLI) for launching GUI applications.

This module uses the `click` library to create a dedicated command
group, `app`, which serves as the entry point for starting the various
graphical user interfaces provided by the `fusionlab-learn` library,
such as the "Subsidence PINN Mini GUI".
"""

import click
import sys

# --- Main 'app' Command Group ---

@click.group(name="app")
def app_group():
    """Commands for launching GUI applications."""
    pass

# --- Command to Launch the Mini Forecaster GUI ---
try:
    # This assumes the main GUI script has a launch function.
    from ..tools.app.mini_forecaster_gui import launch_cli as launch_gui_main

    @app_group.command(name='launch-mini-forecaster')
    @click.option(
        '--theme',
        default='fusionlab',
        type=click.Choice(['dark', 'light', 'fusionlab'], case_sensitive=False),
        help='The visual theme to apply to the GUI.'
    )
    def launch_forecaster_command(theme):
        """Launches the Subsidence PINN Mini GUI application."""
        click.echo(
            f"🚀 Launching Subsidence PINN Mini GUI with '{theme}' theme..."
        )
        try:
            # Pass the theme argument from the CLI to the launch function.
            # The launch_cli function in mini_forecaster_gui.py must be
            # modified to accept this 'theme' keyword argument.
            launch_gui_main(theme=theme)
        except Exception as e:
            click.secho(f"❌ Failed to launch GUI: {e}", fg='red', err=True)
            sys.exit(1)

except ImportError:
    # Fallback command if the GUI application or its dependencies
    # (like PyQt5) are not installed or cannot be imported.
    @app_group.command(name='launch-mini-forecaster')
    def launch_forecaster_command_dummy(**kwargs):
        """Shows an error message if the GUI cannot be launched."""
        click.secho(
            "Error: The 'mini_forecaster_gui' application or its "
            "dependencies (e.g., PyQt5) are missing or could not be "
            "imported.",
            fg='red',
            err=True
        )
