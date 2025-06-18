# -*- coding: utf-8 -*-

import subprocess
import sys

from tqdm import tqdm
import warnings 

_WARNING_MSG =(
     "Standalone Keras is installed but TensorFlow is not. While "
     "some functionalities may work, the primary backend for "
     "`fusionlab-learn` is `tensorflow.keras`. For full "
     "functionality and future compatibility, please install "
     "TensorFlow. The standalone Keras fallback may be deprecated."
    )
     
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
    import importlib.util  
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None

def configure_dependencies(install_dependencies=True):
    """
    Configure the environment by checking and optionally installing
    required packages.

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

def import_keras_dependencies(
    extra_msg=None,
    error='warn'
):
    """
    Dynamically loads Keras/TensorFlow dependencies or a dummy object.

    This function checks for the presence of 'tensorflow' first. If found,
    it returns a real dependency loader. If not, it checks for a
    standalone 'keras' installation as a fallback and issues a warning.
    If neither is found, it returns a dummy object that raises errors
    at runtime.
    """
    import importlib.util

    # Prioritize TensorFlow as the primary backend
    if importlib.util.find_spec("tensorflow"):
        from ..compat.tf import KerasDependencies
        return KerasDependencies(extra_msg, error)

    # If TensorFlow is not found, check for standalone Keras as a fallback
    elif importlib.util.find_spec("keras"):
        warnings.warn(
            _WARNING_MSG,
            UserWarning,
            stacklevel=2
        )
        # Still return the real dependency loader, which should handle this
        from ..compat.tf import KerasDependencies
        return KerasDependencies(
             extra_msg=extra_msg, error = error 
         )

    # If neither backend is found,
    # return the dummy object generator
    else:
        if extra_msg:
            warnings.warn(
                str(extra_msg), ImportWarning, stacklevel=2)
            
        from .._dummies import DummyKerasDeps
        return DummyKerasDeps()
    

def check_keras_backend(
    error='warn'
):
    """
    Checks for the presence of 'tensorflow' or 'keras' and returns the
    name of the available backend.

    It prioritizes TensorFlow and falls back to checking for a standalone
    Keras installation, issuing a warning in that case.

    Parameters
    ----------
    error : {'raise', 'warn', 'ignore'}, default='warn'
        Policy for handling the case where neither backend is found.
        - 'raise': Raises an ImportError.
        - 'warn': Issues an ImportWarning.
        - 'ignore': Does nothing.

    Returns
    -------
    str or None
        Returns 'tensorflow' or 'keras' if found, otherwise returns None.
    """
    
    import importlib.util

    try:
        importlib.util.find_spec("tensorflow")
        return 'tensorflow'
    except ImportError:
        try:
            importlib.util.find_spec("keras")
            ## If TensorFlow is not found, check for standalone Keras
            # Issue a clear warning that TensorFlow is preferred
            warnings.warn(
                _WARNING_MSG,
                UserWarning,
                stacklevel=2
            )
            return 'keras'
        
        except ImportError as e:
            message = (
                "Neither TensorFlow nor Keras is installed."
            )
            if error == 'warn':
                warnings.warn(message)
            elif error == 'raise':
                raise ImportError(message) from e
            return None