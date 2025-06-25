
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

import click

@click.group()
def tune_group():
    """Commands for running hyperparameter tuning sessions."""
    pass

# --- Add a Specific Command under the "tune" Group ---
try:
    from ..tools.pihalnet_tuner import main as tune_legacy_pihalnet_main
    
    @tune_group.command(name='legacy-pihalnet')
    def tune_legacy_pihalnet_command():
        """
        Runs the hyperparameter tuning script for the legacy PIHALNet model.
        """
        click.echo("🚀 Launching Legacy PIHALNet Tuner workflow...")
        try:
            tune_legacy_pihalnet_main()
            click.secho("✅ Tuning script finished successfully.", fg='green')
        except Exception as e:
            click.secho(f"❌ An error occurred: {e}", fg='red', err=True)

except ImportError:
    @tune_group.command(name='legacy-pihalnet')
    def tune_legacy_pihalnet_command():
        click.secho("Error: The 'pihalnet_tuner' tool is missing or "
                    "could not be imported.", fg='red', err=True)

