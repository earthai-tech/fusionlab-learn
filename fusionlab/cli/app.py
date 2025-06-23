     
import click   
        
# --- 4. Create an "app" Sub-Command Group for GUIs ---
@click.group()
def app_group():
    """Commands for launching GUI applications."""
    pass

# --- 5. Add the Command to Launch the Mini Forecaster GUI ---
try:
    from .tools.app.mini_forecaster_gui import launch_cli as launch_gui_main

    @app_group.command(name='launch-mini-forecaster')
    # Example of adding a command-line option
    @click.option(
        '--theme',
        default='dark',
        type=click.Choice(['dark', 'light'], case_sensitive=False),
        help='The theme to apply to the GUI.'
    )
    def launch_forecaster_command(theme):
        """
        Launches the Subsidence PINN Mini GUI application.
        """
        click.echo(f"🚀 Launching Subsidence PINN Mini GUI with '{theme}' theme...")
        try:
            # Note: The GUI script itself would need to handle the 'theme' argument
            launch_gui_main()
        except Exception as e:
            click.secho(f"❌ Failed to launch GUI: {e}", fg='red', err=True)

except ImportError:
    @app_group.command(name='launch-mini-forecaster')
    def launch_forecaster_command(theme):
        click.secho("Error: The 'mini_forecaster_gui' application is missing "
                    "or could not be imported.", fg='red', err=True)

