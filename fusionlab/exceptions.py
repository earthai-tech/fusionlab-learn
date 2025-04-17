# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""List of `fusionlab` exceptions for warning users."""

class ArgumentError(Exception):
    """
    Exception raised for errors in the passed argument values.

    This exception is deprecated and should be replaced by `TypeError` or
    `ValueError` in the next release.
    """
    pass

class SiteError(Exception):
    """
    Exception raised for errors related to site or location information.

    For example, this exception is raised when coordinates values provided are
    inappropriate.
    """
    pass

class DatasetError(Exception):
    """
    Exception raised for inconsistencies in the dataset provided.

    This exception is raised when multiple datasets passed as arguments have
    mismatching shapes, columns, or sizes, or when items in the data are not
    valid according to predefined criteria.
    """
    pass

class HeaderError(Exception):
    """
    Exception raised when headers or required columns are missing in the data.
    """
    pass

class ConfigError(Exception):
    """
    Exception raised for errors in configuration setup or execution.
    """
    pass

class FileHandlingError(Exception):
    """
    Exception raised for errors encountered during file manipulation.

    This exception occurs if there are file permission issues or other problems
    encountered when opening, reading, or writing files.
    """
    pass

class TipError(Exception):
    """
    Exception raised for inappropriate tips proposed for plot visualization
    shortcuts.
    """
    pass

class PlotError(Exception):
    """
    Exception raised when a plot cannot be generated successfully.
    """
    pass

class ParameterNumberError(Exception):
    """
    Exception raised when the number of parameters provided is incorrect.

    This exception is raised when the parameters given do not match the expected
    count for proper computation.
    """
    pass

class ProcessingError(Exception):
    """
    Exception raised for failures in the data processing pipeline.
    """
    pass

class ProfileError(Exception):
    """
    Exception raised for mismatches in arguments passed to a profile object.
    """
    pass

class FeatureError(Exception):
    """
    Exception raised for errors in feature processing or handling.
    """
    pass

class EstimatorError(Exception):
    """
    Exception raised when an incorrect estimator or assessor is provided.
    """
    pass

class GeoPropertyError(Exception):
    """
    Exception raised when there is an attempt to externally modify geological
    property objects.
    """
    pass

class GeoArgumentError(Exception):
    """
    Exception raised for inappropriate arguments passed to geology modules.
    """
    pass

class HintError(Exception):
    """
    Exception raised for inappropriate hints proposed for processing shortcuts.
    """
    pass

class SQLError(Exception):
    """
    Exception raised for errors in SQL queries or database interactions.
    """
    pass

class StrataError(Exception):
    """
    Exception raised for incorrect stratum values or missing 'sname' in
    hydro-logging datasets, where 'sname' is the column name for strata.
    """
    pass

class SQLManagerError(Exception):
    """
    Exception raised for failures in SQL request transfers or executions.
    """
    pass

class GeoDatabaseError(Exception):
    """
    Exception raised when a geospatial database fails to respond or process
    requests.
    """
    pass

class ModelError(Exception):
    """
    Exception raised for errors in geospatial or other model constructions.
    """
    pass

class ERPError(Exception):
    """
    Exception raised for invalid electrical resistivity profiling data.

    'station' and 'resistivity' columns must be present in the ERP dataset.
    """
    pass

class ExtractionError(Exception):
    """
    Exception raised for failures in data extraction from path-like objects or
    file formats like JSON or YAML.
    """
    pass

class CoordinateError(Exception):
    """
    Exception raised for issues with coordinate values or computations.
    """
    pass

class TopModuleError(Exception):
    """
    Exception raised when a key dependency package fails to install or load.

    'scikit-learn' is an example of a key dependency for the `gofast` package.
    """
    pass

class NotFittedError(Exception):
    """
    Exception raised when a 'fit' method is called on an unfitted object.

    This is a common exception for classes in `gofast` that implement a 'fit'
    method for parameter initialization.
    """
    pass

class DependencyError(Exception):
    """
    Exception raised for errors related to missing or incompatible 
    dependencies in the `gofast` package.

    This error is used to handle situations where essential dependencies 
    for a particular functionality are either not installed or incompatible 
    with the current environment. The message provided can include the 
    specific package or module causing the issue and potential solutions.

    Examples
    --------
    >>> try:
    >>>     raise DependencyError("PyTorch is not installed. Please install "
    >>>                            "PyTorch by running 'pip install torch'.")
    >>> except DependencyError as e:
    >>>     print(e)
    PyTorch is not installed. Please install PyTorch by running 
    'pip install torch'.

    Notes
    -----
    - This exception can be raised when the `gofast` package detects that 
      a required dependency, such as PyTorch or TensorFlow, is missing or 
      incompatible with the current system.
    - The error message should provide clear instructions to resolve the issue, 
      such as installation or upgrade commands.

    See also
    --------
    - `fusionlab.exceptions`: Other custom exceptions in the `gofast` package.
    - `fusionlab.utils.install_dependencies`: Function that installs missing 
      dependencies for the package.

    References
    ----------
    .. [1] Python Documentation: Custom Exceptions
           https://docs.python.org/3/tutorial/errors.html
    """
    pass 


class NotRunnedError(Exception):
    """
    Exception raised when an operation requiring a 'runned' state 
    is called on an object that has not completed the 'run' method.

    This exception is particularly relevant in `gofast` classes that 
    implement a 'run' method for initialization or execution before 
    further operations are performed.

    Parameters
    ----------
    message : str, optional
        A custom error message to be displayed. If not provided, a default
        message is used, informing the user that the required 'run' method 
        has not been executed.

    Attributes
    ----------
    message : str
        The error message provided during instantiation, or the default 
        message if none was provided.

    Notes
    -----
    This exception can be raised by utility functions like `check_is_runned`, 
    which validates whether a class object has been properly "runned" before 
    executing dependent operations. If an object lacks the `_is_runned` 
    attribute or the `__gofast_is_runned__` method, `NotRunnedError` is 
    triggered to ensure correct usage patterns.

    Examples
    --------
    >>> from fusionlab.utils.validator import check_is_runned, NotRunnedError
    >>> class ExampleClass:
    ...     def __init__(self):
    ...         self._is_runned = False
    ...
    ...     def run(self):
    ...         self._is_runned = True
    ...         print("Run completed.")
    ...
    ...     def process_data(self):
    ...         if not self._is_runned:
    ...             raise NotRunnedError("Object has not been runned.")
    ...         print("Processing data...")
    >>> model = ExampleClass()
    >>> try:
    ...     model.process_data()
    ... except NotRunnedError as e:
    ...     print(e)
    Object has not been runned.

    See Also
    --------
    NotFittedError : Exception raised when a 'fit' method is called on an 
                     unfitted object.
    check_is_runned : Function to validate if an object has been "runned".

    References
    ----------
    .. [1] Scikit-learn's `NotFittedError` for comparison:
           https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NotFittedError.html
    """

    pass 

class GISError(Exception):
    """
    Exception raised for failures in GIS parameter calculations.
    """
    pass


class LearningError(Exception): 
    """
    Raises an Exception for issues during the learning inspection phase 
    of training.
    """
    pass 

class kError (Exception):
    """
    Raises an exception if the permeability coefficient array is missing 
    or if 'kname' is not specified for permeability in Hydro-log data.
    """
    pass

class DepthError (Exception):
    """
    Raises an exception for depth line issues in a multidimensional array. 
    Depth should be one-dimensional and labeled 'z' in pandas dataframes 
    or series.
    """
    pass 

class AquiferGroupError (Exception):
    """
    Raises an exception for issues with aquifer data, which should be 
    one-dimensional and categorical, representing layer/rock names.
    """
    pass 
