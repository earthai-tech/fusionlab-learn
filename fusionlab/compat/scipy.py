# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides compatibility utilities for different versions of SciPy.
This module includes functions and feature flags to ensure smooth
operation across various SciPy versions, handling breaking changes
and deprecated features. It also sets up logging for the gofast 
package.

Key functionalities include:
- Integration, optimization, and special functions from SciPy
- Linear algebra utilities for matrix operations
- Sparse matrix utilities
- Compatibility checks for different SciPy versions

The module ensures compatibility with SciPy versions less than
1.7, 1.6, 1.5, and 0.15.

Attributes
----------
scipy_version : packaging.version.Version
    The installed SciPy version.
SP_LT_1_7 : bool
    True if the installed SciPy version is less than 1.7.
SP_LT_1_6 : bool
    True if the installed SciPy version is less than 1.6.
SP_LT_1_5 : bool
    True if the installed SciPy version is less than 1.5.
SP_LT_0_15 : bool
    True if the installed SciPy version is less than 0.15.

Functions
---------
integrate_quad
    Perform numerical integration.
optimize_minimize
    Perform optimization using different methods.
special_jn
    Calculate Bessel function of the first kind.
linalg_inv
    Compute the inverse of a matrix.
linalg_solve
    Solve a linear matrix equation.
linalg_det
    Compute the determinant of a matrix.
sparse_csr_matrix
    Create a compressed sparse row matrix.
ensure_scipy_compatibility
    Check and ensure compatibility with the current SciPy version.
calculate_statistics
    Compute various statistical measures.
is_sparse_matrix
    Check if a matrix is sparse.
solve_linear_system
    Solve a system of linear equations.
check_scipy_interpolate
    Check and import interpolation functions.
get_scipy_function
    Retrieve a specific SciPy function.
"""

from packaging.version import Version, parse
import warnings
# import logging
import numpy as np

import scipy
from scipy import special
from scipy import stats
from .._fusionlog import fusionlog 
# Setup logging
_logger = fusionlog().get_fusionlab_logger(__name__)

__all__ = [
    "integrate_quad",
    "optimize_minimize",
    "special_jn",
    "linalg_inv",
    "linalg_solve",
    "linalg_det",
    "sparse_csr_matrix",
    "ensure_scipy_compatibility", 
    "calculate_statistics", 
    "is_sparse_matrix", 
    "solve_linear_system",
    "check_scipy_interpolate",
    "get_scipy_function", 
    "SP_LT_1_6",
    "SP_LT_1_5",
    "SP_LT_1_7", 
    "SP_LT_0_15", 
    "probplot", 
]
# Version checks
scipy_version = parse(scipy.__version__)
SP_LT_1_7 = scipy_version < Version("1.6.99")
SP_LT_1_6 = scipy_version < Version("1.5.99")
SP_LT_1_5 = scipy_version < Version("1.4.99")
SP_LT_0_15 = scipy_version < Version("0.14.0")


def probplot(
    x, 
    dist='norm', 
    sparams=(), 
    fit=True, 
    plot=None, 
    xlabel=None, 
    ylabel=None, 
    line='s'
    ):
    """
    A compatible probplot function that attempts to use scipy's probplot.
    If an error occurs (e.g., due to scipy version changes), it falls back
    to using statsmodels' qqplot.

    Parameters
    ----------
    x : array-like
        Ordered sample data.
    dist : str or distribution, optional
        The theoretical distribution to compare to. 
        Currently, only 'norm' is supported.
    sparams : tuple, optional
        Shape parameters for the theoretical distribution. 
        Not utilized in this implementation.
    fit : bool, default=True
        If True, fit the distribution to the data. 
        Not utilized in this implementation.
    plot : matplotlib axes or figure, optional
        If provided, plot the Q-Q plot on the given axes.
    xlabel : str, optional
        Label for the x-axis. 
        If None, defaults to 'Theoretical Quantiles'.
    ylabel : str, optional
        Label for the y-axis. If None, defaults to 'Ordered Residuals'.
    line : {'s', '45'}, default='s'
        Reference line style. 's' for standardized line,
        '45' for 45-degree line.

    Returns
    -------
    None
        The function directly plots the Q-Q plot.
    """
    import matplotlib.pyplot as plt
    
    try:
        # Attempt to use scipy's probplot
        if plot is not None:
            ax = plot
        else:
            ax = plt.gca()
        
        stats.probplot(x, dist=dist, sparams=sparams, fit=fit, plot=ax)
        
    except ValueError as ve:
        # Check if the error is due to too many values to unpack
        if 'too many values to unpack' in str(ve):
            print("scipy.stats.probplot encountered an"
                  " error due to version incompatibility.")
            print("Attempting to use statsmodels'"
                  "qqplot as a fallback.")
            try:
                from statsmodels.graphics.gofplots import qqplot as sm_qqplot
                
                if plot is not None:
                    ax = plot
                else:
                    ax = plt.gca()
                
                sm_qqplot(x, line=line, ax=ax, fit=fit)
                
                # Set labels if provided
                if xlabel is not None:
                    ax.set_xlabel(xlabel)
                else:
                    ax.set_xlabel('Theoretical Quantiles')
                
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel('Ordered Residuals')
                
            except ImportError:
                raise ImportError(
                    "statsmodels is required for generating Q-Q"
                    "plots when scipy's probplot fails. "
                    "Please install statsmodels using 'pip install statsmodels'"
                    " or 'conda install statsmodels'."
                )
        else:
            # Re-raise the exception if it's a different ValueError
            raise ve
    else:
        # Set labels if provided
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel('Theoretical Quantiles')
        
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('Ordered Residuals')
            

# Utilize function-based approach for direct imports and version-based configurations
def get_scipy_function(module_path, function_name, version_requirement=None):
    """
    Dynamically imports a function from scipy based on the version requirement.
    
    Parameters
    ----------
    module_path : str
        The module path within scipy, e.g., 'integrate'.
    function_name : str
        The name of the function to import.
    version_requirement : Version, optional
        The minimum version of scipy required for this function.
        
    Returns
    -------
    function or None
        The scipy function if available and meets version requirements, else None.
    """
    if version_requirement is None: 
        version_requirement = Version("0.14.0")
    if version_requirement and scipy_version < version_requirement:
        _logger.warning(f"{function_name} requires scipy version {version_requirement} or higher.")
        return None
    
    try:
        module = __import__(f"scipy.{module_path}", fromlist=[function_name])
        return getattr(module, function_name)
    except ImportError as e:
        _logger.error(f"Failed to import {function_name} from scipy.{module_path}: {e}")
        return None

# Define functionalities using the utility function with appropriate
# version checks where necessary
integrate_quad = get_scipy_function("integrate", "quad")
# Optimization
optimize_minimize = get_scipy_function("optimize", "minimize")
# Linear algebra
special_jn = get_scipy_function("special", "jn")
linalg_inv = get_scipy_function("linalg", "inv")
linalg_solve = get_scipy_function("linalg", "solve")
linalg_det = get_scipy_function("linalg", "det")
linalg_eigh = get_scipy_function("linalg", "eigh")
# Sparse matrices
sparse_csr_matrix = get_scipy_function("sparse", "csr_matrix")
sparse_csc_matrix = get_scipy_function("sparse", "csc_matrix")

# Stats
if SP_LT_1_7:
    # Use older stats functions or define a fallback
    def stats_norm_pdf(x):
        return "norm.pdf replacement for older scipy versions"
else:
    stats_norm_pdf = stats.norm.pdf

# Define a function that uses version-dependent functionality
def calculate_statistics(data):
    """
    Calculate statistics on data using scipy's stats module,
    with handling for different versions of scipy.
    """
    if SP_LT_1_7:
        # Use an alternative approach or older function if needed
        mean = np.mean(data)
        median = np.median(data)
        pdf = "PDF calculation not available in this scipy version"
    else:
        mean = stats.tmean(data)
        median = stats.median(data)
        pdf = stats_norm_pdf(data)
    
    return mean, median, pdf

def is_sparse_matrix(matrix) -> bool:
    """
    Check if a matrix is a scipy sparse matrix.

    Parameters
    ----------
    matrix : Any
        Matrix to check.

    Returns
    -------
    bool
        True if the matrix is a scipy sparse matrix.
    """
    return scipy.sparse.issparse(matrix)


def solve_linear_system(A, b):
    """
    Solve a linear system Ax = b using scipy's linalg.solve function.

    Parameters
    ----------
    A : ndarray or sparse matrix
        Coefficient matrix.
    b : ndarray
        Ordinate or dependent variable values.

    Returns
    -------
    x : ndarray
        Solution to the system Ax = b.
    """
    if is_sparse_matrix(A):
        A = A.toarray()  # Convert to dense array if A is sparse for compatibility
    return linalg_solve(A, b)


# Special functions
if hasattr(special, 'erf'):
    special_erf = special.erf
else:
    # Define a fallback for erf if it's not present in the current scipy version
    def special_erf(x):
        # Approximation or use an alternative approach
        return "erf function not available in this scipy version"

# Messages
_msg = ''.join([
    'Note: need scipy version 0.14.0 or higher for interpolation,',
    ' might not work.']
)
_msg0 = ''.join([
    'Could not find scipy.interpolate, cannot use method interpolate. ',
    'Check your installation. You can get scipy from scipy.org.']
)

def ensure_scipy_compatibility():
    """
    Ensures that the scipy version is compatible and required modules are available.
    Logs warnings if conditions are not met.
    """
    global interp_import
    try:
        scipy_version = [int(ss) for ss in scipy.__version__.split('.')]
        if scipy_version[0] == 0 and scipy_version[1] < 14:
            warnings.warn(_msg, ImportWarning)
            _logger.warning(_msg)

        # Attempt to import required modules
        import scipy.interpolate as spi # noqa 
        from scipy.spatial import distance # noqa 

        interp_import = True
       # _logger.info("scipy.interpolate and scipy.spatial.distance imported successfully.")

    except ImportError as e: # noqa
        warnings.warn(_msg0, ImportWarning)
       # _logger.warning(_msg0)
        interp_import = False
        #_logger.error(f"ImportError: {e}")

    return interp_import


def check_scipy_interpolate():
    """
    Checks for scipy.interpolate compatibility and imports required modules.
    
    Returns
    -------
    module or None
        The scipy.interpolate module if available, otherwise None.
    """
    try:
        import scipy

        # Checking for the minimum version requirement
        if scipy_version < Version("0.14.0"):
            _logger.warning('Scipy version 0.14.0 or higher is required for'
                            ' interpolation. Might not work.')
            return None
        
        from scipy import interpolate # noqa 
        return scipy.interpolate

    except ImportError:
        _logger.error('Could not find scipy.interpolate, cannot use method'
                      ' interpolate. Check your scipy installation.') 
        return None


