# -*- coding: utf-8 -*-

import subprocess
import sys
import importlib.util
from tqdm import tqdm

class Config:
    INSTALL_DEPS = False
    DEPS = None
    WARN_STATUS = 'warn'
    USE_CONDA=False

def install_package(package_name):
    """
    Install the given package using pip with a progress bar.

    Parameters
    ----------
    package_name : str
        The name of the package to install.
    """
    def progress_bar():
        pbar = tqdm(total=100, desc=f"Installing {package_name}", ascii=True)
        while True:
            pbar.update(1)
            if pbar.n >= 100:
                break

    process = subprocess.Popen([sys.executable, "-m", "pip", "install", package_name],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    progress_thread = progress_bar()

    stdout, stderr = process.communicate()
    progress_thread.join()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, process.args, output=stdout, stderr=stderr)

def is_package_installed(package_name):
    """
    Check if a package is installed.

    Parameters
    ----------
    package_name : str
        The name of the package to check.

    Returns
    -------
    bool
        True if the package is installed, False otherwise.
    """
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None

def configure_dependencies(install_dependencies=True):
    """
    Configure the environment by checking and optionally installing required packages.

    Parameters
    ----------
    install_dependencies : bool, optional
        If True, installs TensorFlow or Keras if they are not already installed.
        Default is True.
    """
    required_packages = ['tensorflow', 'keras']
    if Config.DEPS:
        required_packages = [Config.DEPS]
    else : 
        return 
    
    installed_packages = {pkg: is_package_installed(pkg) for pkg in required_packages}

    if not any(installed_packages.values()):
        if install_dependencies:
            try:
                for pkg in required_packages:
                    print(f"Installing {pkg} as it is required for this package...")
                    install_package(pkg)
            except Exception as e:
                if Config.WARN_STATUS == 'warn':
                    print(f"Warning: {e}")
                elif Config.WARN_STATUS == 'ignore':
                    pass
                else:
                    raise e
        else:
            raise ImportError(
                "Required dependencies are not installed. "
                "Please install one of these packages to use the `nn` sub-package."
            )

