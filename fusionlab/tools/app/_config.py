# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides environment setup and dependency validation for the GUI application.
"""
import subprocess
import sys
import importlib.util
import logging

def _install_package(package_name: str, version: str = "2.15"):
    """Installs a specific version of a package using pip."""
    full_package = f"{package_name}=={version}"
    logging.info(f"Attempting to install {full_package}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", full_package]
        )
        logging.info(f"Successfully installed {full_package}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install {full_package}. Please install it manually.")
        raise e

def setup_environment(required_tf_version: str = "2.15"):
    """
    Checks for TensorFlow and attempts to install it if missing.

    This function acts as a guard for the application. It ensures that the
    necessary backend is available before the main application logic, which
    relies on TensorFlow, is executed.

    Args:
        required_tf_version (str): The specific version of TensorFlow to
            install if it is not found.

    Raises:
        ImportError: If TensorFlow is not found and cannot be installed.
    """
    tf_spec = importlib.util.find_spec("tensorflow")
    if tf_spec is None:
        print(
            "--> TensorFlow is not installed, which is required for this application."
        )
        try:
            # Ask user for confirmation before installing
            response = input("    Would you like to try installing it now? (y/n): ").lower()
            if response == 'y':
                _install_package("tensorflow", version=required_tf_version)
                # Re-check after installation
                if importlib.util.find_spec("tensorflow") is None:
                    raise ImportError()
            else:
                raise ImportError("Installation declined by user.")
        except Exception as e:
            error_message = (
                "The application cannot proceed without TensorFlow. "
                f"Please install it manually. Original error: {e}"
            )
            logging.error(error_message)
            raise ImportError(error_message) from e
    else:
        logging.info(f"TensorFlow found (version: {tf_spec.loader.load_module().__version__})."
                     " Environment is correctly configured.")

