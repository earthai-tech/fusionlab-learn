# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

# Source: Adapted from earthai-tech/gofast/utils/deps_utils (https://github.com/earthai-tech/gofast)
# This module is included in the FusionLab package, with modifications
# to fit FusionLab’s API conventions.

"""
Dependency utilities providing functions to handle package installation,
checking, and ensuring that optional dependencies are available.
"""

import importlib
import warnings
import sys
import functools
import subprocess

from .._fusionlog import fusionlog
from ..api.types import _T, Any, Callable, List, Optional, Union
from ..core.handlers import delegate_on_error
from ._dependency import import_optional_dependency
from ..decorators import Deprecated

_logger = fusionlog.get_fusionlab_logger(__name__)


__all__ = [
    "ensure_pkg", 
    "ensure_pkgs", 
    "install_package",
    "is_installing",
    "get_installation_name", 
    "is_module_installed", 
    "import_optional_dependency",
    "ensure_module_installed", 
    "get_versions"
]

def get_versions(
    extras=None,
    distribution_mapping=None
):
    """
    Retrieve a dictionary containing version information for
    common libraries, as well as any user-specified packages
    and distribution name mappings.

    Parameters
    ----------
    extras : list of str, optional
        Additional packages for which to attempt version
        retrieval. By default, None, which means no extra
        packages beyond the defaults.

    distribution_mapping : dict, optional
        Mapping from import-like names to actual distribution
        names. For example, the import name ``'sklearn'``
        corresponds to the distribution name
        ``'scikit-learn'``. Default is None, which uses
        a built-in mapping for scikit-learn and any
        user-provided dictionary overrides or additions.

    Returns
    -------
    dict
        Dictionary of the form:

        .. code-block:: python

           {
               "__version__": {
                   "numpy": "1.24.2",
                   "pandas": "1.5.0",
                   "sklearn": "1.3.2",
                   ...
               }
           }

    Notes
    -----
    - By default, this function attempts to retrieve versions
      for the following packages:
      ``['numpy', 'pandas', 'sklearn', 'joblib', 'tensorflow',
      'keras', 'torch']``.
    - If a package is not installed, it is skipped (no error
      is raised).
    - If `<distribution_mapping>` is provided, it merges with
      the built-in mapping (for ``"sklearn"`` → ``"scikit-learn"``),
      allowing users to specify additional name differences.
    - Python 3.8+ is recommended to ensure
      ``importlib.metadata`` is available.

    Examples
    --------
    >>> get_versions()
    {
      "__version__": {
        "numpy": "1.24.2",
        "pandas": "1.5.0",
        ...
      }
    }

    >>> # Add custom package and distribution mapping:
    >>> get_versions(
    ...   extras=["spacy"],
    ...   distribution_mapping={"spacy": "spacy-legacy"}
    ... )
    {
      "__version__": {
        "numpy": "1.24.2",
        "pandas": "1.5.0",
        "spacy": "3.5.1"
      }
    }
    """
    if extras is None:
        extras = []

    # Default packages to check
    default_pkgs = [
        "numpy",
        "pandas",
        "sklearn",  # we expect distribution 'scikit-learn'
        "joblib",
        "tensorflow",
        "keras",
        "torch"
    ]

    # Base distribution mapping for known discrepancies
    base_mapping = {
        "sklearn": "scikit-learn"
    }
    # Merge user-provided distribution mapping, if any
    if distribution_mapping is not None:
        base_mapping.update(distribution_mapping)

    all_pkgs = default_pkgs + list(extras)
    version_dict = {}

    for pkg in all_pkgs:
        # Determine the actual distribution name for version lookup
        dist_name = base_mapping.get(pkg, pkg)

        try:
            # Check if the package is findable
            spec = importlib.util.find_spec(pkg)
            if spec is None:
                # Not installed or can't be found
                continue

            # Attempt to retrieve version from distribution name
            metadata = importlib.metadata
            version = metadata.version(dist_name)

            # Store the version under the original pkg key
            version_dict[pkg] = version

        except (importlib.metadata.PackageNotFoundError,
                ModuleNotFoundError):
            # Not installed or cannot detect version
            continue
        except Exception as e:
            # Catch other unexpected issues, warn and skip
            warnings.warn(
                f"Could not retrieve version for '{pkg}': {e}"
            )
            continue
        
    # After collecting versions in version_dict, 
    # fix distribution names if needed.
    for import_name, dist_name in base_mapping.items():
        if import_name in version_dict:
            # Move the version from import_name => dist_name
            version_dict[dist_name] = version_dict.pop(import_name)
    
    return {"__version__": version_dict}

def ensure_module_installed(
    module_name: str,
    auto_install: bool = False,
    version: Optional[str] = None,
    package_manager: str = "pip",
    dist_name: Optional[str] = None,
    extra_install_args: Optional[List[str]] = None
) -> bool:
    """
    Ensure that the required module is installed, optionally installing it 
    if missing.

    Parameters
    ----------
    module_name : str
        The name of the module to check and install if necessary.
    auto_install : bool, optional
        If ``True``, automatically install the module using the specified 
        package manager if it is not already installed (default is ``False``).
    version : Optional[str], optional
        Specify a version or version range for the module. For example, 
        ">=1.0.0" or "==2.0.1". If ``None``, no version constraints are 
        applied (default is ``None``).
    package_manager : str, optional
        The package manager to use for installation. Currently, only 
        ``"pip"`` is supported. Future versions may support other package 
        managers like ``"conda"`` (default is ``"pip"``).
    dist_name : Optional[str], optional
        Sometimes the module name used for importing is different from the 
        distribution package name. This parameter allows specifying the 
        distribution package name (default is ``None``).
    extra_install_args : Optional[List[str]], optional
        A list of additional arguments to pass to the package manager during 
        installation. For example, ``["--upgrade"]`` to upgrade the package. 
        If ``None``, no extra arguments are passed (default is ``None``).

    Returns
    -------
    bool
        Returns ``True`` if the module is installed or successfully 
        installed, ``False`` otherwise.

    Raises
    ------
    ImportError
        If the module is not installed and ``auto_install`` is ``False``, 
        or if the installation fails.
    ValueError
        If an unsupported package manager is specified.

    .. math::
        P(\text{installed}) = 
        \begin{cases} 
            1 & \text{if module is installed} \\ 
            0 & \text{otherwise} 
        \end{cases}

    Examples
    --------
    >>> from gofast.utils.deps_utils import ensure_module_installed

    >>> # Ensure that 'numpy' is installed
    >>> ensure_module_installed("numpy")

    >>> # Ensure that 'pandas' is installed, automatically installing if missing
    >>> ensure_module_installed("pandas", auto_install=True)

    >>> # Ensure that 'scipy' version >=1.5.0 is installed
    >>> ensure_module_installed("scipy", version=">=1.5.0", auto_install=True)

    >>> # Install with additional arguments
    >>> ensure_module_installed(
    ...     "requests", 
    ...     auto_install=True, 
    ...     extra_install_args=["--upgrade"]
    ... )

    Notes
    -----
    - This function currently supports only ``"pip"`` as the package manager.
    - When specifying a version, ensure that the version string is compatible 
      with the package manager's version specification syntax.
    - For packages that require system-level dependencies, manual installation 
      might be necessary.

    See Also
    --------
    subprocess : For spawning new processes.
    sys : System-specific parameters and functions.

    References
    ----------
    .. [1] Python Packaging User Guide. *Installing Packages*. 
       https://packaging.python.org/tutorials/installing-packages/
    .. [2] pip documentation. *User Guide*. 
       https://pip.pypa.io/en/stable/user_guide/
    """
    try:
        # Attempt to import the module using the module_name
        if dist_name:
            __import__(dist_name)
        else:
            __import__(module_name)
        return True
    except ImportError:
        if not auto_install:
            raise ImportError(
                f"``{module_name}`` is required but not installed."
            )

        if package_manager.lower() != "pip":
            raise ValueError(
                f"Unsupported package manager ``'{package_manager}'``. "
                f"Only ``'pip'`` is supported."
            )

        # If auto-install is true, create the install command
        install_cmd = [sys.executable, "-m", "pip", "install"]

        # Append the module_name and version if provided
        install_cmd.append(module_name)
        if version:
            install_cmd.append(version)
        
        # Include any additional installation arguments
        if extra_install_args:
            install_cmd.extend(extra_install_args)
        
        # Attempt to install the module
        try:
            subprocess.check_call(install_cmd)
            if dist_name:
                __import__(dist_name)
            else:
                __import__(module_name)
            return True
        except subprocess.CalledProcessError as e:
            raise ImportError(
                f"Failed to install ``{module_name}``"
                f" using ``{package_manager}``: {e}"
            )
        except ImportError:
            raise ImportError(
                f"Module ``{module_name}`` was installed but could not be imported."
            )

def install_package(
    name: str,
    dist_name: Optional[str] = None,
    infer_dist_name: bool = False,
    version: Optional[str] = None,
    extra: str = '',
    use_conda: bool = False,
    verbose: bool = True
) -> None:
    r"""
    Install a Python package at runtime, optionally specifying a version
    constraint or other parameters, using either conda or pip. If conda is
    unavailable or disabled, pip is used by default. The function includes
    a check for pre-existing installations, allowing users to skip redundant
    installs.

    Parameters
    ----------
    name : str
        Base name of the package to install (e.g., ``'requests'``).
    dist_name : str, optional
        Distribution name, if different from the import name. For example,
        scikit-learn's import name <sklearn> differs from its distribution
        name ``'scikit-learn'``.
    infer_dist_name : bool, optional
        If True, calls :meth:`get_installation_name` to infer the
        distribution name automatically. Defaults to False.
    version : str, optional
        Version string or comparator. Examples:
          - ``'1.2.0'``  => interpreted as '>=1.2.0'
          - ``'==1.2.0'``
          - ``'<2.0'``
          - ``'>=1.5.3'``
        If None, no version constraint is applied.
        
    extra : str, optional
        Additional install specifiers or command-line flags passed
        to the installation command. For instance, ``' --no-cache-dir'`` 
        or ``'[extra]'``. Default is ``''``.
    use_conda : bool, optional
        If True, attempts installation via conda first. If conda is
        unavailable or fails, falls back to pip. Defaults to False.
    verbose : bool, optional
        If True, prints detailed logs throughout the installation.
        Defaults to True.

    Returns
    -------
    None
        On success, the specified package is installed (or is
        already present). If the installer fails, raises a
        RuntimeError.

    Raises
    ------
    RuntimeError
        If the installation cannot be completed using either
        conda or pip, or if conda is requested but unavailable.

    Notes
    -----
    - If the package is already installed (as determined by
      :meth:`is_module_installed`), no further action is taken.
    - When using pip, a progress bar is displayed (if the `tqdm`
      library is installed). For conda, no progress bar is shown
      due to console I/O capture limitations.


    Mathematically, this function assembles an install spec of the form:

    .. math::
       \text{install\_str} = \langle \text{name} \rangle 
       + \langle \text{version\_spec} \rangle 
       + \langle \text{extra} \rangle

    where :math:`\langle \text{name} \rangle` is the package name,
    :math:`\langle \text{version\_spec} \rangle` is a version comparator
    (e.g., ``>=1.2.0``), and :math:`\langle \text{extra} \rangle` is any
    additional flags or arguments.

    Examples
    --------
    >>> from gofast.utils.deps_utils import install_package
    >>> # Install requests with no version constraint, default pip:
    >>> install_package('requests', verbose=True)

    >>> # Install a specific version via conda (fallback to pip if conda fails):
    >>> install_package(
    ...     'pandas',
    ...     version='==1.2.0',
    ...     use_conda=True,
    ...     verbose=True
    ... )

    See Also
    --------
    is_module_installed:
        Checks whether a Python module or corresponding distribution 
        is already installed.
    get_installation_name:
        Infers a distribution name for the given module name,
        if necessary.

    References
    ----------
    .. [1] Gonsalves, T. et al. (2023). *Dynamic Environment
           Management in Python Projects*. PyCon Proceedings,
           45(7), 31-42.
    """
    # Check if tqdm is available
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        
    # --- Helper Functions ---
    def _format_version_spec(ver_str: str) -> str:
        """
        If `ver_str` starts with a comparator (>, <, ==, !=, >=, <=, ~=, ^),
        return it unchanged. Otherwise, interpret it as '>=ver_str'.
        """
        # List of recognized version operators
        operators = ('>=', '<=', '==', '!=', '>', '<', '~=', '^')
        if any(ver_str.strip().startswith(op) for op in operators):
            return ver_str
        # Default to >= if user just gave a plain version
        return f">={ver_str}"

    def execute_command(command: list, show_progress: bool = False) -> None:
        """
        Execute a system command with optional progress bar for output lines.
        Raises RuntimeError if the command fails.
        """
        if TQDM_AVAILABLE and show_progress:
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            ) as process, tqdm(desc="Installing", unit="line") as pbar:
                for line in process.stdout:
                    if verbose:
                        print(line, end='')
                    pbar.update(1)
                if process.wait() != 0:
                    raise RuntimeError(
                        f"Installation failed for package '{command[-1]}'."
                    )
        else:
            # No tqdm available or progress not requested
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            ) as process:
                for line in process.stdout:
                    if verbose:
                        print(line, end='')
                if process.wait() != 0:
                    raise RuntimeError(
                        f"Installation failed for package '{command[-1]}'."
                    )
 
    # --- Main Logic ---
    # Check if already installed
    if is_module_installed(name, dist_name):
        if verbose:
            print(f"[INFO] '{name}' is already installed.")
        return

    # Potentially infer the distribution name for pip/conda usage
    if infer_dist_name:
        name = get_installation_name(name, dist_name)

    # Format version string if provided
    version_spec = ''
    if version is not None:
        version_spec = _format_version_spec(version)

    # Combine the name + version spec + extra into a single string for pip/conda
    # e.g. "pandas>=1.2.0" or "numpy==1.23.5 --no-cache-dir"
    install_str = name
    if version_spec:
        install_str += version_spec
    if extra.strip():
        # If user wrote extra like "==1.2.0", we typically expect them
        # to put version in "version" param. But let's not override. We'll just
        # append it with a space if it doesn't start with '='.
        # For safety, always add a space. The user might specify advanced pip flags:
        # e.g. extra=" --no-cache-dir"
        install_str += f"{extra}"

    conda_available = _check_conda_installed()
    if use_conda and conda_available:
        if verbose:
            print(
                f"[INFO] Attempting to install '{install_str}' using conda..."
            )
        try:
            # conda install <package>[version spec, etc.] -y
            execute_command(['conda', 'install', install_str, '-y'],
                            show_progress=False)
            if verbose:
                print(f"[INFO] Package '{install_str}' installed via conda.")
            return
        except Exception as e:
            if verbose:
                print(f"[WARN] Conda installation failed: {str(e)}.")
                print("[INFO] Falling back to pip...")
    elif use_conda and not conda_available:
        if verbose:
            print("[WARN] Conda is not available. Falling back to pip...")

    # Fallback to pip
    if verbose:
        print(f"[INFO] Attempting to install '{install_str}' using pip...")
    try:
        execute_command(
            [sys.executable, "-m", "pip", "install", install_str],
            show_progress=True
        )
        if verbose:
            print(f"[INFO] Package '{install_str}' was successfully installed.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to install '{install_str}' via pip: {e}"
        ) from e

@Deprecated (
    "'Install_pkg' is deprecated, fallback to `install_package'"
    " which implement the more robust approach."
 )
@delegate_on_error(
         transfer=install_package,
         delegate_params_mapping={
             'name': 'name', 
             'dist_name': 'dist_name', 
             'infer_dist_name': 'infer_dist_name', 
             'version': 'version', 
             'use_conda':'use_conda', 
             'verbose': 'verbose', 
         }, 
     )
def install_pkg(
    name: str, 
    dist_name: Optional[str]=None,
    infer_dist_name: bool=False, 
    version: str = '', 
    use_conda: bool = False, 
    verbose: bool = True
    ) -> None:
    """
    Install a Python package using either conda or pip, with an option to 
    display installation progress and fallback mechanism.

    This function dynamically chooses between conda and pip for installing 
    Python packages, based on user preference and system configuration. It 
    supports a verbose mode for detailed operation logging and utilizes a 
    progress bar for pip installations.

    Parameters
    ----------
    name : str
        Name of the package to install. Version specification can be included.
    dist_name : str, optional
        The distribution name of the package. Useful for packages where
        the import name differs from the distribution name.
    infer_dist_name : bool, optional
        If True, attempt to infer the distribution name for pip installation,
        defaults to False.
    version : str, optional
        Additional options or version specifier for the package, by default ''.
    use_conda : bool, optional
        Prefer conda over pip for installation, by default False.
    verbose : bool, optional
        Enable detailed output during the installation process, by default True.

    Raises
    ------
    RuntimeError
        If installation fails via both conda and pip, or if the specified installer
        is not available.

    Examples
    --------
    Install a package using pip without version specification:
        >>> from gofast.utils.deps_utils import install_package
        >>> install_package('requests', verbose=True)

    Install a specific version of a package using conda:

        >>> install_package('pandas', extra='==1.2.0', use_conda=True, verbose=True)
    
    Notes
    -----
    Conda installations do not display a progress bar due to limitations in capturing
    conda command line output. Pip installations will show a progress bar indicating
    the number of processed output lines from the installation command.
    """
  
    def execute_command(command: list, progress_bar: bool = False) -> None:
        """
        Execute a system command with optional progress bar for output lines.

        Parameters
        ----------
        command : list
            Command and arguments to execute as a list.
        progress_bar : bool, optional
            Enable a progress bar that tracks the command's output lines, 
            by default False.
        """
        from tqdm import tqdm
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1) as process, \
             tqdm(desc="Installing", unit="line", disable=not progress_bar) as pbar:
            for line in process.stdout:
                if verbose:
                    print(line, end='')
                pbar.update(1)
            if process.wait() != 0:  # Non-zero exit code indicates failure
                raise RuntimeError(f"Installation failed for package '{name}{version}'.")
    
    
    # If the module is installed don't install again.
    if is_module_installed(name, distribution_name= dist_name ): 
        if verbose:
           print(f"{name} is already installed.")
           
        return True
    # If the distribution to pkg name if the pkg name 
    # is different to distribution name .
    if infer_dist_name: 
        name = get_installation_name(name, dist_name)  
        
    conda_available = _check_conda_installed()
    try:
        if use_conda and conda_available:
            if verbose:
                print(f"Attempting to install '{name}{version}' using conda...")
            execute_command(['conda', 'install', f"{name}{version}", '-y'], 
                            progress_bar=False)
        elif use_conda and not conda_available:
            if verbose:
                print("Conda is not available. Falling back to pip...")
            execute_command([sys.executable, "-m", "pip", "install", f"{name}{version}"],
                            progress_bar=True)
        else:
            if verbose:
                print(f"Attempting to install '{name}{version}' using pip...")
            execute_command([sys.executable, "-m", "pip", "install", f"{name}{version}"],
                            progress_bar=True)
        if verbose:
            print(f"Package '{name}{version}' was successfully installed.")
    except Exception as e:
        raise RuntimeError(f"Failed to install '{name}{version}': {e}") from e

def _check_conda_installed() -> bool:
    """
    Check if conda is installed and available in the system's PATH.

    Returns
    -------
    bool
        True if conda is found, False otherwise.
    """
    try:
        subprocess.check_call(
            ['conda', '--version'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    
def ensure_pkg(
    name: str, 
    extra: str = "",
    error: str = "raise",
    min_version: Optional[str] = None,
    exception: Exception = None, 
    dist_name: Optional[str]=None, 
    infer_dist_name: bool=False, 
    auto_install: bool = False,
    use_conda: bool = False, 
    partial_check: bool = False,
    condition: Any = None, 
    verbose: bool = False
) -> Callable[[_T], _T]:
    """
    Decorator to ensure a Python package is installed before function execution.

    If the specified package is not installed, or if its installed version does
    not meet the minimum version requirement, this decorator can optionally 
    install or upgrade the package automatically using either pip or conda.

    Parameters
    ----------
    name : str
        The name of the package.
    extra : str, optional
        Additional specification for the package, such as version or extras.
    error : str, optional
        Error handling strategy if the package is missing: 'raise', 'ignore',
        or 'warn'.
    min_version : str or None, optional
        The minimum required version of the package. If not met, triggers 
        installation.
    exception : Exception, optional
        A custom exception to raise if the package is missing and `errors`
        is 'raise'.
    dist_name : str, optional
        The distribution name of the package as known by package managers (e.g., pip).
        If provided and the module import fails, an additional check based on the
        distribution name is performed. This parameter is useful for packages where
        the distribution name differs from the importable module name.
    infer_dist_name : bool, optional
        If True, attempt to infer the distribution name for pip installation,
        defaults to False.
    auto_install : bool, optional
        Whether to automatically install the package if missing. 
        Defaults to False.
    use_conda : bool, optional
        Prefer conda over pip for automatic installation. Defaults to False.
    partial_check : bool, optional
        If True, checks the existence of the package only if the `condition` 
        is met. This allows for conditional package checking based on the 
        function's arguments or other criteria. If `False`, the check is always
        performed. Defaults to False.
    condition : Any, optional
        A condition that determines whether to check for the package's existence. 
        This can be a callable that takes the same arguments as the decorated function 
        and returns a boolean, a specific argument name to check for truthiness, or 
        any other value that will be evaluated as a boolean. If `None`, the package 
        check is performed unconditionally unless `partial_check` is False.
    verbose : bool, optional
        Enable verbose output during the installation process. Defaults to False.

    Returns
    -------
    Callable
        A decorator that wraps functions to ensure the specified package 
        is installed.

    Examples
    --------
    >>> from gofast.utils.deps_utils import ensure_pkg
    >>> @ensure_pkg("numpy", auto_install=True)
    ... def use_numpy():
    ...     import numpy as np
    ...     return np.array([1, 2, 3])

    >>> @ensure_pkg("pandas", min_version="1.1.0", errors="warn", use_conda=True)
    ... def use_pandas():
    ...     import pandas as pd
    ...     return pd.DataFrame([[1, 2], [3, 4]])

    >>> @ensure_pkg("matplotlib", partial_check=True, condition=lambda x, y: x > 0)
    ... def plot_data(x, y):
    ...     import matplotlib.pyplot as plt
    ...     plt.plot(x, y)
    ...     plt.show()
    
    >>> @ensure_pkg("skimage", partial_check=True, condition=(
    ...     lambda *args, **kwargs: 'method' in kwargs and kwargs['method'] == 'hog')
    ...     )
    >>> def check_package_installed(data, method='hog', **kwargs):
    ...     extractor_function = None
    ...     if method == 'hog':
    ...         from skimage.feature import hog
    ...         extractor_function = lambda image: hog(image, **kwargs)
    ...     return extractor_function
    """
    def decorator(func: _T) -> _T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine if this is a method or a function based on the first argument
            bound_method = hasattr(args[0], func.__name__) if args else False # 
            
            # If partial_check is True, check condition before performing actions
            if not partial_check or  _should_check_condition(
                    condition, *args, **kwargs):
                try:
                    # Attempt to import the package, handling installation 
                    # if necessary and permitted
                    import_optional_dependency(
                        name, extra=extra, errors=error, 
                        min_version=min_version, 
                        exception=exception
                    )
                except (ModuleNotFoundError, ImportError):
                    if auto_install:
                        # Install the package if auto-install is enabled
                        install_package(
                            name, dist_name=dist_name, 
                            infer_dist_name=infer_dist_name, 
                            extra=extra, 
                            version=min_version, 
                            use_conda=use_conda, 
                            verbose=verbose
                        )
                    elif exception is not None:
                        raise exception
                    else:
                        raise
                    
            # If the function is a bound method, call it with 'self' or 'cls'
            if bound_method:
                return func(args[0], *args[1:], **kwargs)
            else:
                return func(*args, **kwargs) # 
        
        return wrapper
    
    return decorator

def _should_check_condition(condition: Any, *args, **kwargs) -> bool:
    """
    Determines whether the condition(s) for checking a package's existence are met, 
    based on the provided arguments and keyword arguments of a decorated function.

    This function offers enhanced flexibility by allowing conditions to be specified 
    as callable functions, tuples for positional argument checks, strings for keyword 
    argument checks, or a list combining any of these types for multiple conditions.

    Parameters
    ----------
    condition : Any
        The condition(s) that determine whether to perform the package check. Can be:
        - A callable that takes `*args` and `**kwargs` and returns a boolean.
        - A string specifying a keyword argument name that should be truthy.
        - A tuple `(index, value)` for checking a specific value of a positional argument.
        - A list of any combination of the above to specify multiple conditions.
    *args : tuple
        Positional arguments passed to the decorated function.
    **kwargs : dict
        Keyword arguments passed to the decorated function.

    Returns
    -------
    bool
        `True` if the package check should be performed based on the evaluation of 
        `condition`, `False` otherwise.

    Examples
    --------
    Checking with a single callable condition for partial_check is ``True``:

    >>> _should_check_condition(lambda x, y: x > y, 5, 3)
    True

    Checking with a string condition (keyword argument name):

    >>> _should_check_condition('method', method='hog')
    True

    Checking with a tuple for positional argument value:

    >>> _should_check_condition((0, 'data'), 'data', method='hog')
    True

    Checking with multiple conditions:

    >>> conditions = [(1, 'hog'), lambda *args, **kwargs: kwargs.get('filter', False)]
    >>> _should_check_condition( conditions, 'data', 'hog', filter=True)
    True

    In the last example, the package check is performed because both conditions are met:
    the second positional argument equals 'hog', and the 'filter' keyword argument is `True`.
    """

    def eval_condition(cond):
        # Callable condition with direct application
        if callable(cond):
            return cond(*args, **kwargs)
        # String condition indicating a key in kwargs
        elif isinstance(cond, str) and cond in kwargs:
            return bool(kwargs[cond])
        # Tuple condition indicating positional argument check
        elif isinstance(cond, tuple) and len(cond) == 2:
            index, value = cond
            return index < len(args) and args[index] == value
        return False
    
    # Support for list of conditions: all must be True
    if isinstance(condition, list):
        return all(eval_condition(cond) for cond in condition)
    else:
        return eval_condition(condition)

def ensure_pkgs(
    names: Union[str, List[str]], 
    extra: str = "",
    error: str = "raise",
    min_versions: Optional[Union[str, List[Optional[str]]]] = None,
    exception: Exception = None, 
    dist_names: Optional[Union[str, List[Optional[str]]]] = None, 
    infer_dist_name: bool = False, 
    auto_install: bool = False,
    use_conda: bool = False, 
    partial_check: bool = False,
    condition: Any = None, 
    verbose: bool = False
) -> Callable[[_T], _T]:
    """
    Decorator to ensure Python packages are installed before function execution.

    If the specified packages are not installed, or if their installed versions
    do not meet the minimum version requirements, this decorator can optionally
    install or upgrade the packages automatically using either pip or conda.

    Parameters
    ----------
    names : str or list of str
        The name(s) of the package(s). Can be a single string with package names
        separated by commas, or a list of package names.
    extra : str, optional
        Additional specification for the package(s), such as version or extras.
    error : {'raise', 'ignore', 'warn'}, optional
        Error handling strategy if a package is missing: 'raise', 'ignore',
        or 'warn'. Defaults to 'raise'.
    min_version : str or list of str, optional
        The minimum required version(s) of the package(s). If not met, triggers
        installation. Can be a single version string applied to all packages
        or a list matching the `names` list.
    exception : Exception, optional
        A custom exception to raise if a package is missing and `errors` is 'raise'.
    dist_name : str or list of str, optional
        The distribution name(s) of the package(s) as known by package managers (e.g., pip).
        Useful when the distribution name differs from the importable module name.
        Can be a single string or a list matching the `names` list.
    infer_dist_name : bool, optional
        If True, attempt to infer the distribution name for pip installation.
        Defaults to False.
    auto_install : bool, optional
        Whether to automatically install missing packages. Defaults to False.
    use_conda : bool, optional
        Prefer conda over pip for automatic installation. Defaults to False.
    partial_check : bool, optional
        If True, checks the existence of the packages only if the `condition` is met.
        Allows for conditional package checking based on the function's arguments or
        other criteria. If False, the check is always performed. Defaults to False.
    condition : Any, optional
        A condition that determines whether to check for the packages' existence.
        Can be a callable that takes the same arguments as the decorated function
        and returns a boolean, a specific argument name to check for truthiness, or
        any other value that will be evaluated as a boolean. If None, the package
        check is performed unconditionally unless `partial_check` is False.
    verbose : bool, optional
        Enable verbose output during the installation process. Defaults to False.

    Returns
    -------
    Callable
        A decorator that wraps functions to ensure the specified packages
        are installed.

    Examples
    --------
    >>> from gofast.utils.deps_utils import ensure_pkgs
    >>> @ensure_pkgs("numpy, pandas", auto_install=True)
    ... def use_numpy_pandas():
    ...     import numpy as np
    ...     import pandas as pd
    ...     return np.array([1, 2, 3]), pd.DataFrame([[1, 2], [3, 4]])

    >>> @ensure_pkgs(["matplotlib", "seaborn"], min_version=["3.0.0", "0.11.0"])
    ... def plot_data(x, y):
    ...     import matplotlib.pyplot as plt
    ...     import seaborn as sns
    ...     sns.scatterplot(x=x, y=y)
    ...     plt.show()

    >>> @ensure_pkgs("skimage", partial_check=True, condition=(
    ...     lambda *args, **kwargs: 'method' in kwargs and kwargs['method'] == 'hog')
    ... )
    ... def process_image(data, method='hog', **kwargs):
    ...     if method == 'hog':
    ...         from skimage.feature import hog
    ...         return hog(data, **kwargs)
    ...     else:
    ...         # Other processing
    ...         pass
    """
    def decorator(func: _T) -> _T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine if this is a method or a function based on the first argument
            bound_method = hasattr(args[0], func.__name__) if args else False

            # If partial_check is True, check condition before performing actions
            if not partial_check or _should_check_condition(condition, *args, **kwargs):
                # Parse names into a list
                if isinstance(names, str):
                    pkg_list = [pkg.strip() for pkg in names.split(',')]
                else:
                    pkg_list = names

                # Ensure min_version and dist_name are lists matching pkg_list
                if isinstance(min_versions, (str, type(None))):
                    min_version_list = [min_versions] * len(pkg_list)
                else:
                    min_version_list = min_versions

                if isinstance(dist_names, (str, type(None))):
                    dist_name_list = [dist_names] * len(pkg_list)
                else:
                    dist_name_list = dist_names

                # Iterate over the packages
                for idx, pkg_name in enumerate(pkg_list):
                    pkg_min_version = min_version_list[idx] if min_version_list else None
                    pkg_dist_name = dist_name_list[idx] if dist_name_list else None

                    try:
                        # Attempt to import the package
                        import_optional_dependency(
                            pkg_name,
                            extra=extra,
                            errors=error,
                            min_version=pkg_min_version,
                            exception=exception
                        )
                    except (ModuleNotFoundError, ImportError):
                        if auto_install:
                            # Install the package if auto-install is enabled
                            install_package(
                                pkg_name,
                                dist_name=pkg_dist_name,
                                infer_dist_name=infer_dist_name,
                                version=pkg_min_version,
                                use_conda=use_conda,
                                verbose=verbose
                            )
                        elif exception is not None:
                            raise exception
                        else:
                            raise

            # Call the original function
            if bound_method:
                return func(args[0], *args[1:], **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator

def is_module_installed(module_name: str, distribution_name: str = None) -> bool:
    """
    Check if a Python module is installed by attempting to import it.
    Optionally, a distribution name can be provided if it differs from the module name.

    Parameters
    ----------
    module_name : str
        The import name of the module to check.
    distribution_name : str, optional
        The distribution name of the package as known by package managers (e.g., pip).
        If provided and the module import fails, an additional check based on the
        distribution name is performed. This parameter is useful for packages where
        the distribution name differs from the importable module name.

    Returns
    -------
    bool
        True if the module can be imported or the distribution package is installed,
        False otherwise.

    Examples
    --------
    >>> is_module_installed("sklearn")
    True
    >>> is_module_installed("scikit-learn", "scikit-learn")
    True
    >>> is_module_installed("some_nonexistent_module")
    False
    """
    if _try_import_module(module_name):
        return True
    if distribution_name and _check_distribution_installed(distribution_name):
        return True
    return False

def _try_import_module(module_name: str) -> bool:
    """
    Attempt to import a module by its name.

    Parameters
    ----------
    module_name : str
        The import name of the module.

    Returns
    -------
    bool
        True if the module can be imported, False otherwise.
    """
    # import importlib.util
    # module_spec = importlib.util.find_spec(module_name)
    # return module_spec is not None
    import importlib
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False 
    
def _check_distribution_installed(distribution_name: str) -> bool:
    """
    Check if a distribution package is installed by its name.

    Parameters
    ----------
    distribution_name : str
        The distribution name of the package.

    Returns
    -------
    bool
        True if the distribution package is installed, False otherwise.
    """
    try:
        # Prefer importlib.metadata for Python 3.8 and newer
        from importlib.metadata import distribution
        distribution(distribution_name)
        return True
    except ImportError:
        # Fallback to pkg_resources for older Python versions
        try:
            from pkg_resources import get_distribution, DistributionNotFound
            get_distribution(distribution_name)
            return True
        except DistributionNotFound:
            return False
    except Exception:
        return False
    
def get_installation_name(
        module_name: str, distribution_name: Optional[str] = None, 
        return_bool: bool = False) -> Union[str, bool]:
    """
    Determines the appropriate name for installing a package, considering potential
    discrepancies between the distribution name and the module import name. Optionally,
    returns a boolean indicating if the distribution name matches the import name.

    Parameters
    ----------
    module_name : str
        The import name of the module.
    distribution_name : str, optional
        The distribution name of the package. If None, the function attempts to infer
        the distribution name from the module name.
    return_bool : bool, optional
        If True, returns a boolean indicating whether the distribution name matches
        the module import name. Otherwise, returns the name recommended for installation.

    Returns
    -------
    Union[str, bool]
        Depending on `return_bool`, returns either a boolean indicating if the distribution
        name matches the module name, or the name (distribution or module) recommended for
        installation.
    """
    inferred_name = _infer_distribution_name(module_name)

    # If a distribution name is provided, check if it matches the inferred name
    if distribution_name:
        if return_bool:
            return distribution_name.lower() == inferred_name.lower()
        return distribution_name

    # If no distribution name is provided, return the inferred name or module name
    if return_bool:
        return inferred_name.lower() == module_name.lower()

    return inferred_name or module_name

def _infer_distribution_name(module_name: str) -> str:
    """
    Attempts to infer the distribution name of a package from its module name
    by querying the metadata of installed packages.

    Parameters
    ----------
    module_name : str
        The import name of the module.

    Returns
    -------
    str
        The inferred distribution name. If no specific inference is made, returns
        the module name.
    """
    try:
        # Use importlib.metadata for Python 3.8+; use importlib_metadata for older versions
        from importlib.metadata import distributions
    except ImportError:
        from importlib_metadata import distributions
    #  Loop through all installed distributions
    for distribution in distributions():
        # Check if the module name matches the distribution name directly
        if module_name == distribution.metadata.get('Name').replace('-', '_'):
            return distribution.metadata['Name']

        # Safely attempt to read and split 'top_level.txt'
        top_level_txt = distribution.read_text('top_level.txt')
        if top_level_txt:
            top_level_packages = top_level_txt.split()
            if any(module_name == pkg.split('.')[0] for pkg in top_level_packages):
                return distribution.metadata['Name']

    return module_name

def is_installing (
    module: str , 
    upgrade: bool=True , 
    action: bool=True, 
    DEVNULL: bool=False,
    verbose: int=0,
    **subpkws
    )-> bool: 
    """ Install or uninstall a module/package using the subprocess 
    under the hood.
    
    Parameters 
    ------------
    module: str,
        the module or library name to install using Python Index Package `PIP`
    
    upgrade: bool,
        install the lastest version of the package. *default* is ``True``.   
        
    DEVNULL:bool, 
        decline the stdoutput the message in the console 
    
    action: str,bool 
        Action to perform. 'install' or 'uninstall' a package. *default* is 
        ``True`` which means 'intall'. 
        
    verbose: int, Optional
        Control the verbosity i.e output a message. High level 
        means more messages. *default* is ``0``.
         
    subpkws: dict, 
        additional subprocess keywords arguments 
    Returns 
    ---------
    success: bool 
        whether the package is sucessfully installed or not. 
        
    Example
    --------
    >>> from gofast import is_installing
    >>> is_installing(
        'tqdm', action ='install', DEVNULL=True, verbose =1)
    >>> is_installing(
        'tqdm', action ='uninstall', verbose =1)
    """
    #implement pip as subprocess 
    # refer to https://pythongeeks.org/subprocess-in-python/
    if not action: 
        if verbose > 0 :
            print("---> No action `install`or `uninstall`"
                  f" of the module {module!r} performed.")
        return action  # DO NOTHING 
    
    success=False 

    action_msg ='uninstallation' if action =='uninstall' else 'installation' 

    if action in ('install', 'uninstall', True) and verbose > 0:
        print(f'---> Module {module!r} {action_msg} will take a while,'
              ' please be patient...')
        
    cmdg =f'<pip install {module}> | <python -m pip install {module}>'\
        if action in (True, 'install') else ''.join([
            f'<pip uninstall {module} -y> or <pip3 uninstall {module} -y ',
            f'or <python -m pip uninstall {module} -y>.'])
        
    upgrade ='--upgrade' if upgrade else '' 
    
    if action == 'uninstall':
        upgrade= '-y' # Don't ask for confirmation of uninstall deletions.
    elif action in ('install', True):
        action = 'install'

    cmd = ['-m', 'pip', f'{action}', f'{module}', f'{upgrade}']

    try: 
        STDOUT = subprocess.DEVNULL if DEVNULL else None 
        STDERR= subprocess.STDOUT if DEVNULL else None 
    
        subprocess.check_call(
            [sys.executable] + cmd, stdout= STDOUT, stderr=STDERR,
                              **subpkws)
        if action in (True, 'install'):
            # freeze the dependancies
            reqs = subprocess.check_output(
                [sys.executable,'-m', 'pip','freeze'])
            [r.decode().split('==')[0] for r in reqs.split()]

        success=True
        
    except: 

        if verbose > 0 : 
            print(f'---> Module {module!r} {action_msg} failed. Please use'
                f' the following command: {cmdg} to manually do it.')
    else : 
        if verbose > 0: 
            print(f"{action_msg.capitalize()} of `{module}` "
                      "and dependancies was successfully done!") 
        
    return success 

        