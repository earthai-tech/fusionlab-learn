# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

# Source: Adapted from earthai-tech/gofast (https://github.com/earthai-tech/gofast)
# This module is included in the FusionLab package, with modifications
# to fit FusionLab’s utility conventions.

"""
Utility functions for data manipulation, validation, and preprocessing
within FusionLab. Includes routines for handling missing data,
string normalization, type inspection, and other common utilities.

Adapted for FusionLab from the original gofast.core.utils implementation.
"""

from __future__ import print_function, annotations
import os
import re
import time
import inspect
import warnings
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..api.types import (
    Any, Callable, Dict, Union, Tuple, Optional, Iterable,
    _Sub, _F, List, DataFrame
)
from ..compat.scipy import check_scipy_interpolate
from .checks import _assert_all_types, is_iterable, validate_name_in

__all__ = [
    'contains_delimiter',
    'convert_value_in',
    'fill_nan_in',
    'get_confidence_ratio',
    'ismissing',
    'listing_items_format',
    'make_introspection',
    'make_obj_consistent_if',
    'normalize_string',
    'sanitize_frame_cols',
    'type_of_target',
    'unpack_list_of_dicts',
    'run_return',
    'format_to_datetime',
    'error_policy'
]


def run_return(
    self, 
    attribute_name: Optional[str] = None, 
    error: str = 'warn',
    default_value: Optional[Any] = None,
    check_callable: bool = False,
    return_type: str = 'attribute',
    on_callable_error: str = 'warn',
    allow_private: bool = False,
    msg: Optional[str] = None, 
    config_return_type: Optional[Union[str, bool]] = None
) -> Any:
    """
    Return `self`, a specified attribute of `self`, or both, with error handling
    policies. Optionally integrates with global configuration to customize behavior.

    Parameters
    ----------
    attribute_name : str, optional
        The name of the attribute to return. If `None`, returns `self`.
    error : str, optional
        Policy for handling non-existent attributes. Options:
        - `warn` : Warn the user and return `self` or a default value.
        - `ignore` : Silently return `self` or the default value.
        - `raise` : Raise an `AttributeError` if the attribute does not exist.
    default_value : Any, optional
        The default value to return if the attribute does not exist. If `None`,
        and the attribute does not exist, returns `self` based on the error policy.
    check_callable : bool, optional
        If `True`, checks if the attribute is callable and executes it if so.
    return_type : str, optional
        Specifies the return type. Options:
        - `self` : Always return `self`.
        - `attribute` : Return the attribute if it exists.
        - `both` : Return a tuple of (`self`, attribute).
    on_callable_error : str, optional
        How to handle errors when calling a callable attribute. Options:
        - `warn` : Warn the user and return `self`.
        - `ignore` : Silently return `self`.
        - `raise` : Raise the original error.
    allow_private : bool, optional
        If `True`, allows access to private attributes (those starting with '_').
    msg : str, optional
        Custom message for warnings or errors. If `None`, a default message will be used.
    config_return_type : str or bool, optional
        Global configuration to override return behavior. If set to 'self', always
        return `self`. If 'attribute', always return the attribute. If `None`, use
        developer-defined behavior.

    Returns
    -------
    Any
        Returns `self`, the attribute value, or a tuple of both, depending on
        the specified options and the availability of the attribute.

    Raises
    ------
    AttributeError
        If the attribute does not exist and `error_policy` is set to 'raise', or if the
        callable check fails and `on_callable_error` is set to 'raise'.

    Notes
    -----
    The `run_return` function is designed to offer flexibility in determining
    what is returned from a method, allowing developers to either return `self` for
    chaining, return an attribute of the class, or both. By using `global_config`,
    package-wide behavior can be customized.

    Examples
    --------
    >>> from gofast.core.utils import run_return
    >>> class MyModel:
    ...     def __init__(self, name):
    ...         self.name = name
    ...
    >>> model = MyModel(name="example")
    >>> run_return(model, "name")
    'example'

    See Also
    --------
    logging : Python's logging module.
    warnings.warn : Function to issue warning messages.

    References
    ----------
    .. [1] "Python Logging Module," Python Software Foundation.
           https://docs.python.org/3/library/logging.html
    .. [2] "Python Warnings," Python Documentation.
           https://docs.python.org/3/library/warnings.html
    """

    # If global config specifies return behavior, override the return type
    if config_return_type == 'self':
        return self
    elif config_return_type == 'attribute':
        return getattr(self, attribute_name, default_value
                       ) if attribute_name else self

    # If config is None or not available, use developer-defined logic
    if attribute_name:
        # Check for private attributes if allowed
        if not allow_private and attribute_name.startswith('_'):
            custom_msg = msg or ( 
                f"Access to private attribute '{attribute_name}' is not allowed.")
            raise AttributeError(custom_msg)

        # Check if the attribute exists
        if hasattr(self, attribute_name):
            attr_value = getattr(self, attribute_name)

            # If check_callable is True, try executing the attribute if it's callable
            if check_callable and isinstance(attr_value, Callable):
                try:
                    attr_value = attr_value()
                except Exception as e:
                    custom_msg = msg or ( 
                        f"Callable attribute '{attribute_name}'"
                        f" raised an error: {e}."
                        )
                    if on_callable_error == 'raise':
                        raise e
                    elif on_callable_error == 'warn':
                        warnings.warn(custom_msg)
                        return self
                    elif on_callable_error == 'ignore':
                        return self

            # Return based on the return_type provided
            if return_type == 'self':
                return self
            elif return_type == 'both':
                return self, attr_value
            else:
                return attr_value
        else:
            # Handle the case where the attribute does not exist based on the error
            custom_msg = msg or ( 
                f"'{self.__class__.__name__}' object has"
                f"  no attribute '{attribute_name}'."
                )
            if error == 'raise':
                raise AttributeError(custom_msg)
            elif error== 'warn':
                warnings.warn(f"{custom_msg} Returning default value or self.")
            # Return the default value if provided, otherwise return self
            return default_value if default_value is not None else self
    else:
        # If no attribute is provided, return self
        return self

def gen_X_y_batches(
    X, y, *,
    batch_size="auto",
    n_samples=None,
    min_batch_size=0,
    shuffle=True,
    random_state=None,
    return_batches=False,
    default_size=200,
):
    """
    Generate batches of data (`X`, `y`) for machine learning tasks such as 
    training or evaluation. This function slices the dataset into smaller 
    batches, optionally shuffles the data, and returns them as a list of 
    tuples or just the data batches.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data matrix, where each row is a sample and each column 
        represents a feature.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The target variable(s) corresponding to `X`. Can be a vector or 
        matrix depending on the problem (single or multi-output).

    batch_size : int, "auto", default="auto"
        The number of samples per batch. If set to `"auto"`, it uses the 
        minimum between `default_size` and the number of samples, `n_samples`.

    n_samples : int, optional, default=None
        The total number of samples to consider. If `None`, the function 
        defaults to using the number of samples in `X`.

    min_batch_size : int, default=0
        The minimum size for each batch. This parameter ensures that the 
        final batch contains at least `min_batch_size` samples. If the 
        last batch is smaller than `min_batch_size`, it will be excluded 
        from the result.

    shuffle : bool, default=True
        If `True`, the data is shuffled before batching. This helps avoid 
        bias when splitting data for training and validation.

    random_state : int, RandomState instance, or None, default=None
        The seed used by the random number generator for reproducibility. 
        If `None`, the random number generator uses the system time or 
        entropy source.

    return_batches : bool, default=False
        If `True`, the function returns both the data batches and the slice 
        objects used to index into `X` and `y`. If `False`, only the 
        data batches are returned.

    default_size : int, default=200
        The default batch size used when `batch_size="auto"` is selected.

    Returns
    -------
    Xy_batches : list of tuples
        A list of tuples where each tuple contains a batch of `X` and its 
        corresponding batch of `y`.

    batch_slices : list of slice objects, optional
        If `return_batches=True`, this list of `slice` objects is returned, 
        each representing the slice of `X` and `y` used for a specific batch.

    Notes
    -----
    - This function ensures that no empty batches are returned. If a batch 
      contains zero samples (either from improper slicing or due to 
      `min_batch_size`), it will be excluded.
    - The function performs shuffling using scikit-learn's `shuffle` function, 
      which is more stable and reduces memory usage by shuffling indices 
      rather than the whole dataset.
    - The function utilizes the `gen_batches` utility to divide the data into 
      batches.

    Examples
    --------
    >>> from gofast.core.utils import gen_X_y_batches
    >>> X = np.random.rand(2000, 5)
    >>> y = np.random.randint(0, 2, size=(2000,))
    >>> batches = gen_X_y_batches(X, y, batch_size=500, shuffle=True)
    >>> len(batches)
    4

    >>> X = np.random.rand(2000, 5)
    >>> y = np.random.randint(0, 2, size=(2000,))
    >>> batches, slices = gen_X_y_batches(
    >>>     X, y, batch_size=500, shuffle=True, return_batches=True
    >>> )
    >>> len(batches)
    4
    >>> len(slices)
    4

    Notes
    ------
    Given a dataset of size `n_samples` and target `y`, we want to partition 
    the dataset into batches. The `batch_size` parameter defines the maximum 
    number of samples in each batch, and `min_batch_size` ensures that 
    the last batch has a minimum size if possible.

    For each batch, we perform the following steps:
    
    1. **Determine the batch size**:
       - If `batch_size` is "auto", we set:
       
       .. math::
           \text{batch\_size} = \min(\text{default\_size}, n_{\text{samples}})
       
    2. **Validate batch size**:
       - Ensure the batch size does not exceed the total number of samples. 
       If it does, we clip it:
       
       .. math::
           \text{batch\_size} = \min(\text{batch\_size}, n_{\text{samples}})
    
    3. **Generate batches**:
       - Use the `gen_batches` utility to create slice indices that partition 
       the dataset into batches:
       
       .. math::
           \text{batch\_slices} = \text{gen\_batches}(n_{\text{samples}}, 
           \text{batch\_size})
       
    4. **Shuffling** (if enabled):
       - If `shuffle=True`, shuffle the dataset's indices before splitting:
       
       .. math::
           \text{indices} = \text{shuffle}(0, 1, \dots, n_{\text{samples}} - 1)
    
    5. **Return Batches**:
       - After creating the batches, return them as tuples of `(X_batch, y_batch)`.

    See Also
    --------
    gen_batches : A utility function that generates slices of data.
    shuffle : A utility to shuffle data while keeping the data and labels in sync.

    References
    ----------
    [1] Scikit-learn. "sklearn.utils.shuffle". Available at 
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
    """
    from sklearn.utils import shuffle as sk_shuffle, _safe_indexing
    from sklearn.utils import gen_batches
    from .validator import check_X_y, validate_batch_size 
    
    X, y = check_X_y(X, y, to_frame=True)

    # List to store the resulting batches
    Xy_batches = []
    batch_slices = []

    # Default to the number of samples in X if not provided
    if n_samples is None:
        n_samples = X.shape[0]

    # Determine and validate batch size
    if batch_size == "auto":
        batch_size = min(default_size, n_samples)
    else:
        
        if batch_size > n_samples:
            warnings.warn(
                "Got `batch_size` less than 1 or larger than "
                "sample size. It is going to be clipped."
            )
            batch_size = np.clip(batch_size, 1, n_samples)
    # Validate batch size
    batch_size = validate_batch_size( 
        batch_size, n_samples, min_batch_size=min_batch_size
    )
    
    # Generate batch slices
    batches = list(
        gen_batches(n_samples, batch_size, min_batch_size=min_batch_size)
    )

    # Generate an array of indices for shuffling
    indices = np.arange(X.shape[0])

    if shuffle:
        # Shuffle indices for stable randomization
        sample_idx = sk_shuffle(indices, random_state=random_state)

    for batch_idx, batch_slice in enumerate(batches):
        # Slice the training data to obtain the current batch
        if shuffle:
            X_batch = _safe_indexing(X, sample_idx[batch_slice])
            y_batch = y[sample_idx[batch_slice]]
        else:
            X_batch = X[batch_slice]
            y_batch = y[batch_slice]

        if y_batch.size == 0 or X_batch.size == 0:
            if shuffle: 
                X_batch, y_batch = ensure_non_empty_batch(
                    X, y, batch_slice, 
                    random_state=random_state, 
                    error = "warn", 
                ) 
            else:
                continue

        # Append valid batches to the results
        Xy_batches.append((X_batch, y_batch))
        batch_slices.append(batch_slice)

    if len(Xy_batches)==0: 
        # No batch found 
        Xy_batches.append ((X, y)) 
        
    return (Xy_batches, batch_slices) if return_batches else Xy_batches


def ensure_non_empty_batch(
    X, y, *, batch_slice, max_attempts=10, random_state=None,
    error ="raise", 
):
    """
    Shuffle the dataset (`X`, `y`) until the specified `batch_slice` yields 
    a non-empty batch. This function ensures that the batch extracted using 
    `batch_slice` contains at least one sample by repeatedly shuffling the 
    data and reapplying the slice.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data matrix, where each row corresponds to a sample and 
        each column corresponds to a feature.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The target variable(s) corresponding to `X`. It can be a one-dimensional 
        array for single-output tasks or a two-dimensional array for multi-output 
        tasks.

    batch_slice : slice
        A slice object representing the indices for the batch. For example, 
        `slice(0, 512)` would extract the first 512 samples from `X` and `y`.

    max_attempts : int, optional, default=10
        The maximum number of attempts to shuffle the data to obtain a non-empty 
        batch. If the batch remains empty after the specified number of attempts, 
        a `ValueError` is raised.

    random_state : int, RandomState instance, or None, default=None
        Controls the randomness of the shuffling. Pass an integer for reproducible 
        results across multiple function calls. If `None`, the random number 
        generator is the RandomState instance used by `np.random`.

    error: str, default ='raise' 
        Handle error status when empty batch is still present after 
        `max_attempts`. Expect ``{"raise", "warn" "ignore"} , if ``warn``, 
        error is converted in warning message. Any other value ignore the 
        error message. 
        
    Returns
    -------
    X_batch : ndarray of shape (batch_size, n_features)
        The batch of input data extracted using `batch_slice`. Ensures that 
        `X_batch` is not empty.

    y_batch : ndarray of shape (batch_size,) or (batch_size, n_targets)
        The batch of target data corresponding to `X_batch`, extracted using 
        `batch_slice`. Ensures that `y_batch` is not empty.

    Raises
    ------
    ValueError
        If a non-empty batch cannot be obtained after `max_attempts` shuffles.

    Examples
    --------
    >>> from gofast.core.utils import ensure_non_empty_batch
    >>> import numpy as np
    >>> X = np.random.rand(2000, 5)
    >>> y = np.random.randint(0, 2, size=(2000,))
    >>> batch_slice = slice(0, 512)
    >>> X_batch, y_batch = ensure_non_empty_batch(X, y, batch_slice=batch_slice)
    >>> X_batch.shape
    (512, 5)
    >>> y_batch.shape
    (512,)

    >>> # Example where the batch might initially be empty
    >>> X_empty = np.empty((0, 5))
    >>> y_empty = np.empty((0,))
    >>> try:
    ...     ensure_non_empty_batch(X_empty, y_empty, batch_slice=slice(0, 512))
    ... except ValueError as e:
    ...     print(e)
    ...
    Unable to obtain a non-empty batch after 10 attempts.

    Notes
    -----
    Given a dataset with `n_samples` samples, the goal is to find a subset of 
    samples defined by the `batch_slice` such that:

    .. math::
        \text{batch\_size} = \text{len}(X[\text{batch\_slice}])

    The function ensures that:

    .. math::
        \text{batch\_size} > 0

    This is achieved by iteratively shuffling the dataset and reapplying the 
    `batch_slice` until the condition is satisfied or the maximum number of 
    attempts is reached.

    See Also
    --------
    gen_batches : Generate slice objects to divide data into batches.
    shuffle : Shuffle arrays or sparse matrices in a consistent way.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, 
       B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
       learning in Python. *Journal of Machine Learning Research*, 12, 
       2825-2830.
    .. [2] NumPy Developers. (2023). NumPy Documentation. 
       https://numpy.org/doc/
    """
    from sklearn.utils import shuffle as sk_shuffle 
    
    attempts = 0

    while attempts < max_attempts:
        # Extract the batch using the provided slice
        X_batch = X[batch_slice]
        y_batch = y[batch_slice]

        # Check if both X_batch and y_batch are non-empty
        if X_batch.size > 0 and y_batch.size > 0:
            return X_batch, y_batch

        # Shuffle the dataset
        X, y = sk_shuffle(
            X, y, random_state=random_state
        )

        attempts += 1

    msg =  f"Unable to obtain a non-empty batch after {max_attempts} attempts."
    if error=="raise": 
        # If a non-empty batch is not found after max_attempts, raise an error
        raise ValueError(msg)
    elif error =='warn':
        warnings.warn( msg ) 
        
    return X, y 
    

# def get_batch_size(
#     *arrays, 
#     default_size=None, 

#     ):
#     """
#     Determine an optimal batch size based on available memory.

#     This function computes an optimal batch size for
#     processing large arrays in batches, aiming to prevent
#     memory overload by considering the available system
#     memory. If `psutil` is installed, it uses the available
#     memory to calculate the batch size. Otherwise, it warns
#     the user and defaults to a batch size of 64.

#     Parameters
#     ----------
#     *arrays : array-like
#         One or more arrays (e.g., NumPy arrays) for which
#         to compute the batch size. All arrays must have the
#         same number of samples (first dimension).
#     default_size: int, default=64 
#         Value selected when 'psutil' package is not available. However, if 
#         the value is ``given`` psutil is not used instead and fallback to 
#         the default value instead. 

#     Returns
#     -------
#     int
#         The computed batch size, which is at least 1 and
#         at most the number of samples in the arrays.

#     Notes
#     -----
#     The batch size is computed using the formula:

#     .. math::

#         \text{batch\_size} = \min\left(
#             \max\left(
#                 1, \left\lfloor \frac{M \times 0.1}{S}
#                 \right\rfloor
#             \right), N
#         \right)

#     where:

#     - :math:`M` is the available system memory in bytes,
#       obtained via `psutil`.
#     - :math:`S` is the total size in bytes of one sample
#       across all arrays.
#     - :math:`N` is the total number of samples in the
#       arrays.

#     If `psutil` is not installed, a default `batch_size`
#     of 64 is used, or less if there are fewer samples.

#     Examples
#     --------
#     >>> import numpy as np
#     >>> from gofast.core.utils import get_batch_size
#     >>> X = np.random.rand(1000, 20)
#     >>> y = np.random.rand(1000)
#     >>> batch_size = _get_batch_size(X, y)
#     >>> print(batch_size)
#     64

#     See Also
#     --------
#     batch_generator : Generator function to create batches.

#     References
#     ----------
#     .. [1] Giampaolo Rodola, "psutil - process and system
#        utilities", https://psutil.readthedocs.io/

#     """
# def get_batch_size(
#     *arrays, 
#     default_size=None, 

#     ):
#     # revise
#     try: 
#         import psutil 
#     except: 
        
#         warnings.warn(
#             "'psutil' is missing for computing the optimal batch size" 
#             " based on available memory. Use default "
#             f"``batch_size={default_size}`` instead`")

#         default_size = default_size or 64  
        
#     if default_size is not None: 
        
#         # check whether bach_size is not  based on the array length 
#         # for instance of bacth_size > then the array length , it does 
#         # not make sense. 
#         # now if the batch_size is greather than array length and if psutil is installed 
#         # then fall back to psutil determination by warning users 
        
#         # if batch size  is greater than array length and psutil is not install 
#         # then use 1 instead, it means the bacth will use the all array . 
        
#         return default_size 
    
#     arrays = [np.asarray(arr) for arr in arrays ]
#     available_memory = psutil.virtual_memory().available
#     sample_size = sum(arr.strides[0] for arr in arrays)
#     n_samples = arrays[0].shape[0]
#     max_memory_usage = available_memory * 0.1 # 0.1 can be an other parameter 
#     batch_size = int(max_memory_usage // sample_size)
#     batch_size = max(1, min(batch_size, n_samples))
#     return batch_size

@contextmanager
def training_progress_bar(
    epochs, 
    steps_per_epoch, 
    metrics=None, 
    bar_length=30, 
    delay=0.01, 
    obj_name=''
):
    """
    Context manager to display a Keras-like training progress bar for each epoch,
    with dynamic metric display and optional object name.

    This function simulates the progress of training epochs, displaying a progress 
    bar and updating metrics in real-time. After completion, it shows the best 
    metrics achieved. The context manager is suitable for visual feedback in 
    iterative tasks such as training models in machine learning.

    Parameters
    ----------
    epochs : int
        Total number of epochs to simulate training.
        
    steps_per_epoch : int
        Total number of steps (batches) per epoch.
        
    metrics : dict, optional
        Dictionary containing metric names as keys (e.g., `loss`, `accuracy`, 
        `val_loss`, `val_accuracy`) and their initial values as the dictionary 
        values. These metrics are updated and displayed at each step.
        
    bar_length : int, default=30
        Length of the progress bar in characters. Modify this to adjust 
        the visual length of the progress bar.
        
    delay : float, default=0.01
        Time delay (in seconds) between steps, useful for simulating processing 
        time per batch or step.
        
    obj_name : str, optional
        The name of the object being trained, if any. This can be set to 
        ``obj.__class__.__name__`` if an object instance is passed to provide
        customized display.

    Notes
    -----
    - Metrics are simulated in this function and updated using an arbitrary decay 
      for loss and growth for accuracy metrics. Replace these with actual values 
      in a real training loop.
    - `metrics` will show both current and best values across all epochs at the end.
    
    Examples
    --------
    >>> from gofast.utils.sysutils import training_progress_bar
    >>> metrics = {'loss': 1.0, 'accuracy': 0.5, 'val_loss': 1.0, 'val_accuracy': 0.5}
    >>> with training_progress_bar(epochs=5, steps_per_epoch=20, 
                                   metrics=metrics, obj_name="Model") as progress:
    ...     pass  # This context will display progress updates as configured.

    See Also
    --------
    Other tracking utilities, such as TensorBoard, which provides an interactive 
    visual interface for tracking and logging training metrics.

    References
    ----------
    .. [1] Keras Documentation - https://keras.io/api/callbacks/progress_bar/

    """
    import sys
    
    # Initialize metrics if not provided
    if metrics is None:
        metrics = {'loss': 1.0, 'accuracy': 0.5, 'val_loss': 1.0, 'val_accuracy': 0.5}
    best_metrics = {k: v for k, v in metrics.items()}  # Track best metrics

    # Start training simulation
    try:
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            for step in range(1, steps_per_epoch + 1):
                time.sleep(delay)  # Simulate time delay for each step

                # Update each metric for display
                for metric in metrics:
                    # Simulate decreasing loss and increasing accuracy
                    if "loss" in metric:
                        metrics[metric] = max(
                            0, metrics[metric] - 0.001 * step)
                    else:
                        metrics[metric] = min(
                            1.0, metrics[metric] + 0.001 * step)

                    # Update best metric values
                    if "loss" in metric:
                        best_metrics[metric] = min(
                            best_metrics[metric], metrics[metric])
                    else:
                        best_metrics[metric] = max(
                            best_metrics[metric], metrics[metric])

                # Calculate and display progress
                progress = step / steps_per_epoch
                completed = int(progress * bar_length)
                
                # Construct progress bar
                progress_bar = '=' * completed
                if completed < bar_length:
                    progress_bar += '>'
                progress_bar = progress_bar.ljust(bar_length)

                # Display current metrics
                metric_display = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                
                # Output progress bar with metrics
                sys.stdout.write(
                    f"\r{step}/{steps_per_epoch} "
                    f"[{progress_bar}] - {metric_display}"
                )
                sys.stdout.flush()
            print("\n")  # Newline after each epoch
        yield
    finally:
        # Display the best metrics upon completion
        best_metric_display = " - ".join([f"{k}: {v:.4f}" for k, v in best_metrics.items()])
        print("Training complete!")
        if obj_name:
            obj_name += ' - '  # Add hyphen separator if obj_name is specified
        print(f"{obj_name}Best Metrics: {best_metric_display}")
        

def format_to_datetime(data, date_col, verbose=0, **dt_kws):
    """
    Reformats a specified column in a DataFrame to Pandas datetime format.

    This function attempts to convert the values in the specified column of a 
    DataFrame to Pandas datetime objects. If the conversion is successful, 
    the DataFrame with the updated column is returned. If the conversion fails, 
    a message describing the error is printed, and the original 
    DataFrame is returned.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the column to be reformatted.
    date_col : str
        The name of the column to be converted to datetime format.
    verbose : int, optional
        Verbosity mode; 0 or 1. If 1, prints messages about the conversion 
        process.Default is 0 (silent mode).
    **dt_kws : dict, optional
        Additional keyword arguments to pass to `pd.to_datetime` function.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the specified column in datetime format. If conversion
        fails, the original DataFrame is returned.

    Raises
    ------
    ValueError
        If the specified column is not found in the DataFrame.

    Examples
    --------
    >>> from gofast.core.utils import format_to_datetime
    >>> df = pd.DataFrame({
    ...     'Date': ['2021-01-01', '01/02/2021', '03-Jan-2021', '2021.04.01',
                     '05 May 2021'],
    ...     'Value': [1, 2, 3, 4, 5]
    ... })
    >>> df = format_to_datetime(df, 'Date')
    >>> print(df.dtypes)
    Date     datetime64[ns]
    Value             int64
    dtype: object
    """
    if date_col not in data.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")
    
    try:
        data[date_col] = pd.to_datetime(data[date_col], **dt_kws)
        if verbose: 
            print(f"Column '{date_col}' successfully converted to datetime format.")
    except Exception as e:
        print(f"Error converting '{date_col}' to datetime format: {e}")
        return data

    return data


def unpack_list_of_dicts(list_of_dicts):
    """
    Unpacks a list of dictionaries into a single dictionary,
    merging all keys and values.

    Parameters:
    ----------
    list_of_dicts : list of dicts
        A list where each element is a dictionary with a single key-value pair, 
        the value being a list.

    Returns:
    -------
    dict
        A single dictionary with all keys from the original list of dictionaries, 
        each associated with its combined list of values from all occurrences 
        of the key.

    Example:
    --------
    >>> from gofast.core.utils import unpack_list_of_dicts
    >>> list_of_dicts = [
            {'key1': ['value10', 'value11']},
            {'key2': ['value20', 'value21']},
            {'key1': ['value12']},
            {'key2': ['value22']}
        ]
    >>> unpacked_dict = unpack_list_of_dicts(list_of_dicts)
    >>> print(unpacked_dict)
    {'key1': ['value10', 'value11', 'value12'], 'key2': ['value20', 'value21', 'value22']}
    """
    unpacked_dict = defaultdict(list)
    for single_dict in list_of_dicts:
        for key, values in single_dict.items():
            unpacked_dict[key].extend(values)
    return dict(unpacked_dict)  # Convert defaultdict back to dict if required

#XXX TODO 
# # write a decorator 
# def param_deprecated_message (
#         params, 
#         condition=None, # can be a lambda function, boolean condition  etc...
#         message=None, # message to raise when condition is meet or satisfied or any others. 
#         defaut_values=None, # default value to let the parameter take etc ... 
#         warning_categories='FutureWarning' #(default),  
#         extra=''# message to append to default message when message is None 
#         ... # other parameters for flexibility and robustness for customizing 
        
#         )
# # FIRST CASE:
#     # if condition is set, the param should meet this condition before raising warning user 
#     # if parameter meet this conditioon , then raise message and warning category 
#     # if the default value is set then erase the value of parameter to take the default value 
#     # for the function or method or class to proceed. 
    
#     # for instance the TemporalTransformerFusion currently support only quantiles to be None, 
#     # if the other values are set , then user must be warn that quantiles currently support None 
#     # and erase the value to None and let the class, function or method to proceed. 
    
# class TemporalTransformerFusion: 
#      def __init__(
#          self,
#          static_input_dim,
#          dynamic_input_dim,
#          num_static_vars,
#          num_dynamic_vars,
#          hidden_units,
#          num_heads=4,  
#          dropout_rate=0.1,
#          forecast_horizon=1,
#          quantiles=None,# should be None because, current version not support the list [float]
#          activation='elu',
#          use_batch_norm=False,
#          num_lstm_layers=1,
#          lstm_units=None, 
#          **kwargs
#      ):
#     # if quantiles is set and not None, i.e. the condition is met 
#     # then a message is trigger to the user and reset the value of quantile to None 
#     # and continue proceeding 
    
#     # the decorator can operate on class, function or method 
    
#     # also the decorator should handle this case : 
# # SECOND CASE:
# class HWClassifier : 
#     def __init__(
#             self, 
#             nonlinear_input_estimator =None, 
#             nonlinear_output_estimator=None, 
#             ...
#             # new parameter added 
#             input_estimator =None, 
#             output_estimator=None 
#             ...
#         )
#     # Here in this class for instance 
#     # the previous version use nonlinear_input_estimator and nonlinear_output_estimator as parameters names.
#     # now in the future version, we expect to rename these two parameters (nonlinear_input_estimator and nonlinear_output_estimator) 
#     # to be replaced by  input_estimator and output_estimator  respecitively 
#     # so if the user passes the value to nonlinear_output_estimator for instance, a warnings message is 
#     # triggered and delegate the value to output_estimator, similar for the nonlinear_input_estimator
#     # parameter which delegate value to input_estimator and proceed. The user should now that these 
#     # parameter will not longer available in future version ( version can be specify as parameter), 
    
#     # however if input_estimator and output_estimator are set, no message raise since 
#     # they are the new parameters that will be permanent to do the tasks of 
#     # nonlinear_input_estimator and nonlinear_output_estimator respectivey and so on. 
#     # 
# # write this sophysticated decorator that can be used in method, class or function 
# # make the decorator robust and you can check the parameters listed in the decorator function 
# # and rename them or reorganized them if you think they are not intuitive enough etc 
# Use parameters vertical alignment which is more professionals 
# come back to a new line if the code is long using the best vertical alignement as well as the comments.
# beatifully write the function and break the code using vertical alignment and 
# parenthesis to come back to new line make it more beautiful 

# # skip the documentation for brievity, focus on the code writing and comments in the code. 
# # skip any docstring 


def fancy_printer(result, report_name='Data Quality Check Report'):
    """ 
    This _fancy_print function within the check_data_quality function 
    iterates over the results dictionary and prints each category 
    (like missing data, outliers, etc.) in a formatted manner. It only 
    displays categories with findings, making the output more concise and 
    focused on the areas that need attention. The use of .title() 
    and .replace('_', ' ') methods enhances the readability of the 
    category names.

    Parameters 
    -----------
    result: dict,
       the result to print. Must contain a dictionnary. 
    report_name: str, 
       A report to fancy printer. 
       
    """
    if not isinstance ( result, dict): 
        raise TypeError("fancy_printer accepts only a dictionnary type."
                        f" Got {type(result).__name__!r}")
        
    print(f"\n{report_name}:\n")

    for key, value in result.items():
        if value:  # Only display categories with findings
            print(f"--- {key.replace('_', ' ').title()} ---")
            print("Column            | Details")
            print("-" * 40)  # Table header separator

            try : 
                
                for sub_key, sub_value in value.items():
                    # Ensuring column name and details fit into the table format
                    formatted_key = (sub_key[:15] + '..') if len(
                        sub_key) > 17 else sub_key
                    formatted_value = str(sub_value)[:20] + (
                        '..' if len(str(sub_value)) > 22 else '')
                    print(f"{formatted_key:<17} | {formatted_value}")
            except : 
                formatted_key = (key[:15] + '..') if len(key) > 17 else key
                formatted_value = f"{value:.2f}"
                print(f"{formatted_key:<17} | {formatted_value}")

            print("\n")
        else:
            print(f"--- No {key.replace('_', ' ').title()} Found ---\n")

def listing_items_format ( 
        lst,  begintext ='', endtext='' , bullet='-', 
        enum =True , lstyle=None , space =3 , inline =False, verbose=True
        ): 
    """ Format list by enumerate them successively with carriage return
    
    :param lst: list,
        object for listening 
    :param begintext: str, 
        Text to display at the beginning of listing the items in `lst`. 
    :param endtext: str, 
        Text to display at the end of the listing items in `lst`. 
    :param enum:bool, default=True, 
        Count the number of items in `lst` and display it 
    :param lstyle: str, default =None 
        listing marker. 
    :param bullet:str, default='-'
        symbol that is used to introduce item if `enum` is set to False. 
    :param space: int, 
        number of space to keep before each outputted item in `lst`
    :param inline: bool, default=False, 
        Display all element inline rather than carriage return every times. 
    :param verbose: bool, 
        Always True for print. If set to False, return list of string 
        litteral text. 
    :returns: None or str 
        None or string litteral if verbose is set to ``False``.
    Examples
    ---------
    >>> from gofast.core.utils import listing_items_format 
    >>> litems = ['hole_number', 'depth_top', 'depth_bottom', 'strata_name', 
                'rock_name','thickness', 'resistivity', 'gamma_gamma', 
                'natural_gamma', 'sp','short_distance_gamma', 'well_diameter']
    >>> listing_items_format (litems , 'Features' , 
                               'have been successfully drop.' , 
                              lstyle ='.', space=3) 
    """
    out =''
    if not is_iterable(lst): 
        lst=[lst]
   
    if hasattr (lst, '__array__'): 
        if lst.ndim !=1: 
            raise ValueError (" Can not print multidimensional array."
                              " Expect one dimensional array.")
    lst = list(lst)
    begintext = str(begintext); endtext=str(endtext)
    lstyle=  lstyle or bullet  
    lstyle = str(lstyle)
    b= f"{begintext +':' } "   
    if verbose :
        print(b, end=' ') if inline else (
            print(b)  if  begintext!='' else None)
    out += b +  ('\n' if not inline else ' ') 
    for k, item in enumerate (lst): 
        sp = ' ' * space 
        if ( not enum and inline ): lstyle =''
        o = f"{sp}{str(k+1) if enum else bullet+ ' ' }{lstyle} {item}"
        if verbose:
            print (o , end=' ') if inline else print(o)
        out += o + ('\n' if not inline else ' ') 
       
    en= ' ' + endtext if inline else endtext
    if verbose: 
        print(en) if endtext !='' else None 
    out +=en 
    
    return None if verbose else out 

def shrunkformat(
    text: Union[str, Iterable[Any]], 
    chunksize: int = 7,
    insert_at: Optional[str] = None, 
    sep: Optional[str] = None, 
) -> None:
    """ Format class and add ellipsis when classes are greater than maxview 
    
    :param text: str - a text to shrunk and format. Can also be an iterable
        object. 
    :param chunksize: int, the size limit to keep in the formatage text. *default* 
        is ``7``.
    :param insert_at: str, the place to insert the ellipsis. If ``None``,  
        shrunk the text and put the ellipsis, between the text beginning and 
        the text endpoint. Can be ``beginning``, or ``end``. 
    :param sep: str if the text is delimited by a kind of character, the `sep` 
        parameters could be usefull so it would become a starting point for 
        word counting. *default*  is `None` which means word is counting from 
        the space. 
        
    :example: 
        
    >>> import numpy as np 
    >>> from gofast.core.utils import shrunkformat
    >>> text=" I'm a long text and I will be shrunked and replaced by ellipsis."
    >>> shrunkformat (text)
    ... 'Im a long ... and replaced by ellipsis.'
    >>> shrunkformat (text, insert_at ='end')
    ...'Im a long ... '
    >>> arr = np.arange(30)
    >>> shrunkformat (arr, chunksize=10 )
    ... '0 1 2 3 4  ...  25 26 27 28 29'
    >>> shrunkformat (arr, insert_at ='begin')
    ... ' ...  26 27 28 29'
    
    """
    is_str = False 
    chunksize = int (_assert_all_types(chunksize, float, int))
                   
    regex = re.compile (r"(begin|start|beg)|(end|close|last)")
    insert_at = str(insert_at).lower().strip() 
    gp = regex.search (insert_at) 
    if gp is not None: 
        if gp.group (1) is not None:  
            insert_at ='begin'
        elif gp.group(2) is not None: 
            insert_at ='end'
        if insert_at is None: 
            warnings.warn(f"Expect ['begining'|'end'], got {insert_at!r}"
                          " Default value is used instead.")
    if isinstance(text , str): 
        textsplt = text.strip().split(sep) # put text on list 
        is_str =True 
        
    elif hasattr (text , '__iter__'): 
        textsplt = list(text )
        
    if len(textsplt) < chunksize : 
        return  text 
    
    if is_str : 
        rl = textsplt [:len(textsplt)//2][: chunksize//2]
        ll= textsplt [len(textsplt)//2:][-chunksize//2:]
        
        if sep is None: sep =' '
        spllst = [f'{sep}'.join ( rl), f'{sep}'.join ( ll)]
        
    else : spllst = [
        textsplt[: chunksize//2 ] ,textsplt[-chunksize//2:]
        ]
    if insert_at =='begin': 
        spllst.insert(0, ' ... ') ; spllst.pop(1)
    elif insert_at =='end': 
        spllst.pop(-1) ; spllst.extend ([' ... '])
        
    else : 
        spllst.insert (1, ' ... ')
    
    spllst = spllst if is_str else str(spllst)
    
    return re.sub(r"[\[,'\]]", '', ''.join(spllst), 
                  flags=re.IGNORECASE 
                  ) 

def accept_types(
    *objtypes: list,
    format: bool = False
    ) -> Union[List[str], str]:
    """
    List the type formats that can be accepted by a function.

    This function accepts a list of object types and returns either 
    a list of their names or a formatted string with the names of 
    types, depending on the value of the `format` parameter.

    Parameters
    ----------
    objtypes : list
        A variable-length list of object types (e.g., `int`, `str`, 
        `pd.DataFrame`). The function will extract the names of these 
        types and format them accordingly.
    format : bool, optional, default=False
        If `True`, returns a formatted string with the type names 
        separated by commas and an "and" before the last type. If `False`, 
        returns a list of the type names as strings.

    Returns
    -------
    Union[List[str], str]
        If `format` is `False`, returns a list of type names as strings. 
        If `format` is `True`, returns a formatted string with the 
        type names.

    Examples
    --------
    >>> import numpy as np; import pandas as pd 
    >>> from gofast.core.utils import accept_types
    >>> accept_types(pd.Series, pd.DataFrame, tuple, list, str)
    "'Series','DataFrame','tuple','list' and 'str'"

    >>> accept_types(pd.Series, pd.DataFrame, np.ndarray, format=True)
    "'Series','DataFrame' and 'ndarray'"
    """
    return smart_format(
        [f'{o.__name__}' for o in objtypes]
    ) if format else [f'{o.__name__}' for o in objtypes]


def smart_format(iter_obj, choice='and'):
    """
    Smartly format an iterable object into a human-readable string with 
    a conjunction ('and' or 'or') for the last item.

    This function is useful for formatting lists of strings in a natural 
    language style, e.g., 'item1, item2, and item3'.

    Parameters
    ----------
    iter_obj : iterable
        Iterable object to be formatted. Should be an iterable of strings 
        or objects that can be converted to strings.
    choice : str, optional, default='and'
        Conjunction to use between the last two items. Can be 'and' or 'or'.

    Returns
    -------
    str
        A formatted string with the items from the iterable.

    Raises
    ------
    ValueError
        If `choice` is not 'and' or 'or'.
    TypeError
        If `iter_obj` is not an iterable.

    Examples
    --------
    >>> from gofast.core.utils import smart_format
    >>> smart_format(['model', 'iter', 'mesh', 'data'])
    'model', 'iter', 'mesh' and 'data'

    >>> smart_format(['apple', 'orange'], choice='or')
    'apple or orange'

    >>> smart_format(['apple'])
    'apple'

    >>> smart_format([])
    ''
    """

    # Validate 'choice' parameter
    if choice not in ['and', 'or']:
        raise ValueError(f"Invalid choice '{choice}'. Must be 'and' or 'or'.")
    
    # Ensure the input is iterable
    try:
        iter(iter_obj)
    except TypeError:
        raise TypeError("The provided input is not an iterable.")
    
    # Convert all elements to strings
    iter_obj = [str(obj) for obj in iter_obj]

    # Handle edge cases
    if not iter_obj:
        return ''
    elif len(iter_obj) == 1:
        return iter_obj[0]
    else:
        # Join all but the last element with commas
        formatted_str = ', '.join(f"'{item}'" for item in iter_obj[:-1])
        # Add the conjunction before the last item
        formatted_str += f" {choice} '{iter_obj[-1]}'"
    
    return formatted_str

def make_introspection(Obj: object, subObj: _Sub[object]) -> None:
    """
    Make introspection by using the attributes of an instance to populate 
    the new classes created.

    This function copies attributes from `subObj` to `Obj`. If `Obj` has 
    an attribute with the same name as `subObj`, `Obj`'s attribute takes 
    precedence.

    Parameters
    ----------
    Obj : callable
        New object to inherit attributes from `subObj`.
    subObj : callable
        Instance whose attributes will be copied to `Obj`.

    Examples
    --------
    >>> class Base:
    >>>     def __init__(self):
    >>>         self.base_attr = 'Base attribute'
    >>> class Derived:
    >>>     pass
    >>> base_instance = Base()
    >>> derived_instance = Derived()
    >>> make_introspection(Derived, base_instance)
    >>> derived_instance.base_attr
    'Base attribute'
    """
    for key, value in subObj.__dict__.items():
        if not hasattr(Obj, key) and key != ''.join(['__', str(key), '__']):
            setattr(Obj, key, value)

def format_notes(text: str, cover_str: str = '~', inline: int = 70, **kws):
    """
    Format a note by wrapping the text in a given `cover_str` and limiting 
    the number of characters per line.

    This function is useful for formatting notes or messages where the 
    text should be wrapped to a specified length with custom margin and 
    header formatting.

    Parameters
    ----------
    text : str
        The text to be formatted.
    cover_str : str, optional, default='~'
        The string used to surround the text (for header and footer).
    inline : int, optional, default=70
        The number of characters per line before wrapping to the next line.
    margin_space : float, optional, default=0.2
        The margin ratio, expressed as a percentage of the `inline` value, 
        indicating the space between the edge and the start of the text.
        Must be less than 1.

    Examples
    --------
    >>> text = 'Automatic Option is set to ``True``. Composite estimator'
    >>> 'building is triggered.'
    >>> format_notes(text=text, inline=70, margin_space=0.05)

    Notes
    -----
    The `headernotes` keyword can be passed to change the header label 
    (default is 'notes').
    """
    headnotes = kws.pop('headernotes', 'notes')
    margin_ratio = kws.pop('margin_space', 0.2)
    margin = int(margin_ratio * inline)
    init_ = 0
    new_textList = []

    if len(text) <= (inline - margin):
        new_textList = text
    else:
        for kk, char in enumerate(text):
            if kk % (inline - margin) == 0 and kk != 0:
                new_textList.append(text[init_:kk])
                init_ = kk
            if kk == len(text) - 1:
                new_textList.append(text[init_:])

    print('!', headnotes.upper(), ':')
    print('{}'.format(cover_str * inline))
    for k in new_textList:
        fmtin_str = '{' + '0:>{}'.format(margin) + '}'
        print('{0}{1:>2}{2:<51}'.format(fmtin_str.format(cover_str), '', k))

    print('{0}{1:>51}'.format(' ' * (margin - 1), cover_str * (inline - margin + 1)))

def interpol_scipy(
    x_value,
    y_value,
    x_new,
    kind="linear",
    plot=False,
    fill_value="extrapolate"
):
    """
    Interpolate data using scipy's `interp1d` method, if available.

    This function performs interpolation on the given data using 
    scipy's `interp1d` function. It can plot the interpolation result 
    if required.

    Parameters
    ----------
    x_value : np.ndarray
        Array of original abscissa (x) values. These are the independent 
        values corresponding to the `y_value` array.
    
    y_value : np.ndarray
        Array of original ordinate (y) values. These are the dependent 
        values corresponding to the `x_value` array.
    
    x_new : np.ndarray
        New abscissa (x) values for which interpolation is performed. The 
        function will return interpolated `y` values at these new `x` values.
    
    kind : str, optional, default="linear"
        The type of interpolation to use. Possible values include:
        - "linear" (default)
        - "nearest"
        - "zero"
        - "slinear"
        - "quadratic"
        - "cubic"
        - "previous"
        - "next"
        The choice of interpolation affects the smoothness and accuracy 
        of the interpolated values.

    fill_value : str, optional, default="extrapolate"
        The value to use for extrapolation if the new x-values lie outside 
        the range of the original data. Common choices:
        - "extrapolate" (default): use linear extrapolation for values outside 
          the range.
        - A specific numeric value can also be used instead of extrapolating.

    plot : bool, optional, default=False
        If True, a plot will be generated showing the original data points 
        along with the interpolated values. The plot will display both the 
        original data and the interpolation curve.

    Returns
    -------
    np.ndarray
        Interpolated ordinate (y) values corresponding to the provided 
        `x_new` values.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.core.utils import interpol_scipy
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> y = np.array([0, 1, 4, 9, 16])
    >>> x_new = np.array([1.5, 2.5, 3.5])
    >>> interpol_scipy(x, y, x_new, kind="quadratic", plot=True)
    array([2.25, 6.25, 12.25])
    
    Notes
    -----
    This function requires `scipy` to be installed. If `scipy` is not 
    available, the function will return `None`.
    """
    spi = check_scipy_interpolate()  # Check for scipy compatibility
    if spi is None:
        return None
    
    try:
        func_ = spi.interp1d(
            x_value,
            y_value,
            kind=kind,
            fill_value=fill_value
        )
        y_new = func_(x_new)

        if plot:
            plt.plot(x_value, y_value, "o", x_new, y_new, "--")
            plt.legend(["Data", kind.capitalize()], loc="best")
            plt.title(f"Interpolation: {kind.capitalize()}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.show()

        return y_new

    except Exception as e:
        warnings.warn(f"An unexpected error occurred during interpolation: {e}")
        return None


def _remove_str_word (ch, word_to_remove, deep_remove=False):
    """
    Small funnction to remove a word present on  astring character 
    whatever the number of times it will repeated.
    
    Parameters
    ----------
        * ch : str
                may the the str phrases or sentences . main items.
        * word_to_remove : str
                specific word to remove.
        * deep_remove : bool, optional
                use the lower case to remove the word even the word is uppercased 
                of capitalized. The default is False.

    Returns
    -------
        str ; char , new_char without the removed word .
        
    Examples
    ---------
    >>> from gofast.tools import funcutils as func
    >>> ch ='AMTAVG 7.76: "K1.fld", Dated 99-01-01,AMTAVG, 
    ...    Processed 11 Jul 17 AMTAVG'
    >>> ss=func._remove_str_word(char=ch, word_to_remove='AMTAVG', 
    ...                             deep_remove=False)
    >>> print(ss)
    
    """
    if type(ch) is not str : char =str(ch)
    if type(word_to_remove) is not str : word_to_remove=str(word_to_remove)
    
    if deep_remove == True :
        word_to_remove, char =word_to_remove.lower(),char.lower()

    if word_to_remove not in char :
        return char

    while word_to_remove in char : 
        if word_to_remove not in char : 
            break 
        index_wr = char.find(word_to_remove)
        remain_len=index_wr+len(word_to_remove)
        char=char[:index_wr]+char[remain_len:]

    return char

def stn_check_split_type(data_lines): 
    """
    Read data_line and check for data line the presence of 
    split_type < ',' or ' ', or any other marks.>
    Threshold is assume to be third of total data length.
    
    :params data_lines: list of data to parse . 
    :type data_lines: list 
 
    :returns: The split _type
    :rtype: str
    
    :Example: 
        >>> from gofast.tools  import funcutils as func
        >>> path =  data/ K6.stn
        >>> with open (path, 'r', encoding='utf8') as f : 
        ...                     data= f.readlines()
        >>>  print(func.stn_check_split_type(data_lines=data))
        
    """

    split_type =[',', ':',' ',';' ]
    data_to_read =[]
    # change the data if data is not dtype string elements.
    if isinstance(data_lines, np.ndarray): 
        if data_lines.dtype in ['float', 'int', 'complex']: 
            data_lines=data_lines.astype('<U12')
        data_lines= data_lines.tolist()
        
    if isinstance(data_lines, list):
        for ii, item in enumerate(data_lines[:int(len(data_lines)/3)]):
             data_to_read.append(item)
             # be sure the list is str item . 
             data_to_read=[''.join([str(item) for item in data_to_read])] 

    elif isinstance(data_lines, str): data_to_read=[str(data_lines)]
    
    for jj, sep  in enumerate(split_type) :
        if data_to_read[0].find(sep) > 0 :
            if data_to_read[0].count(sep) >= 2 * len(data_lines)/3:
                if sep == ' ': return  None  # use None more conventional 
                else : return sep 


def fr_en_parser (f, delimiter =':'): 
    """ Parse the translated data file. 
    
    :param f: translation file to parse.
    
    :param delimiter: str, delimiter.
    
    :return: generator obj, composed of a list of 
        french  and english Input translation. 
    
    :Example:
        >>> file_to_parse = 'pme.parserf.md'
        >>> path_pme_data = r'C:/Users\Administrator\Desktop\__elodata
        >>> data =list(BS.fr_en_parser(
            os.path.join(path_pme_data, file_to_parse)))
    """
    
    is_file = os.path.isfile (f)
    if not is_file: 
        raise IOError(f'Input {f} is not a file. Please check your file.')
    
    with open(f, 'r', encoding ='utf8') as ft: 
        data = ft.readlines()
        for row in data :
            if row in ( '\n', ' '):
                continue 
            fr, en = row.strip().split(delimiter)
            yield([fr, en])

def drawn_boundaries(erp_data, appRes, index):
    """
    Function to drawn anomaly boundary 
    and return the anomaly with its boundaries
    
    :param erp_data: erp profile 
    :type erp_data: array_like or list 
    
    :param appRes: resistivity value of minimum pk anomaly 
    :type appRes: float 
    
    :param index: index of minimum pk anomaly 
    :type index: int 
    
    :return: anomaly boundary 
    :rtype: list of array_like 

    """
    f = 0 # flag to mention which part must be calculated 
    if index ==0 : 
        f = 1 # compute only right part 
    elif appRes ==erp_data[-1]: 
        f=2 # compute left part 
    
    def loop_sideBound(term):
        """
        loop side bar from anomaly and find the term side 
        
        :param term: is array of left or right side of anomaly.
        :type term: array 
        
        :return: side bar 
        :type: array_like 
        """
        tem_drawn =[]
        maxT=0 

        for ii, tem_rho in enumerate(term) : 

            diffRes_betw_2pts= tem_rho - appRes 
            if diffRes_betw_2pts > maxT : 
                maxT = diffRes_betw_2pts
                tem_drawn.append(tem_rho)
            elif diffRes_betw_2pts < maxT : 
                # rho_limit = tem_rho 
                break 
        return np.array(tem_drawn)
    # first broke erp profile from the anomalies 
    if f ==0 or f==2 : 
        left_term = erp_data[:index][::-1] # flip left term  for looping
        # flip again to keep the order 
        left_limit = loop_sideBound(term=left_term)[::-1] 

    if f==0 or f ==1 : 
        right_term= erp_data[index :]
        right_limit=loop_sideBound(right_term)
    # concat right and left to get the complete anomaly 
    if f==2: 
        anomalyBounds = np.append(left_limit,appRes)
                                   
    elif f ==1 : 
        anomalyBounds = np.array([appRes]+ right_limit.tolist())
    else: 
        left_limit = np.append(left_limit, appRes)
        anomalyBounds = np.concatenate((left_limit, right_limit))
    
    return appRes, index, anomalyBounds 

def fmt_text(
        anFeatures=None, 
        title = None,
        **kwargs) :
    """
    Function format text from anomaly features 
    
    :param anFeatures: Anomaly features 
    :type anFeatures: list or dict
    
    :param title: head lines 
    :type title: list
    
    :Example: 
        
        >>> from gofast.core.utils import fmt_text
        >>> fmt_text(anFeatures =[1,130, 93,(146,145, 125)])
    
    """
    if title is None: 
        title = ['Ranking', 'rho(Ω.m)', 'position pk(m)', 'rho range(Ω.m)']
    inline =kwargs.pop('inline', '-')
    mlabel =kwargs.pop('mlabels', 100)
    line = inline * int(mlabel)
    
    #--------------------header ----------------------------------------
    print(line)
    tem_head ='|'.join(['{:^15}'.format(i) for i in title[:-1]])
    tem_head +='|{:^45}'.format(title[-1])
    print(tem_head)
    print(line)
    #-----------------------end header----------------------------------
    newF =[]
    if isinstance(anFeatures, dict):
        for keys, items in anFeatures.items(): 
            rrpos=keys.replace('_pk', '')
            rank=rrpos[0]
            pos =rrpos[1:]
            newF.append([rank, min(items), pos, items])
            
    elif isinstance(anFeatures, list): 
        newF =[anFeatures]
    
    
    for anFeatures in newF: 
        strfeatures ='|'.join(['{:^15}'.format(str(i)) \
                               for i in anFeatures[:-1]])
        try : 
            iter(anFeatures[-1])
        except : 
            strfeatures +='|{:^45}'.format(str(anFeatures[-1]))
        else : 
            strfeatures += '|{:^45}'.format(
                ''.join(['{} '.format(str(i)) for i in anFeatures[-1]]))
            
        print(strfeatures)
        print(line)
    

def wrap_infos (
        phrase ,
        value ='',
        underline ='-',
        unit ='',
        site_number= '',
        **kws) : 
    """Display info from anomaly details."""
    
    repeat =kws.pop('repeat', 77)
    intermediate =kws.pop('inter+', '')
    begin_phrase_mark= kws.pop('begin_phrase', '--|>')
    on = kws.pop('on', False)
    if not on: return ''
    else : 
        print(underline * repeat)
        print('{0} {1:<50}'.format(begin_phrase_mark, phrase), 
              '{0:<10} {1}'.format(value, unit), 
              '{0}'.format(intermediate), "{}".format(site_number))
        print(underline * repeat )
    

def ismissing(refarr, arr, fill_value = np.nan, return_index =False): 
    """ Get the missing values in array-like and fill it  to match the length
    of the reference array. 
    
    The function makes sense especially for frequency interpollation in the 
    'attenuation band' when using the audio-frequency magnetotelluric methods. 
    
    :param arr: array-like- Array to be extended with fill value. It should be  
        shorter than the `refarr`. Otherwise it returns the same array `arr` 
    :param refarr: array-like- the reference array. It should have a greater 
        length than the array 
    :param fill_value: float - Value to fill the `arr` to match the length of 
        the `refarr`. 
    :param return_index: bool or str - array-like, index of the elements element 
        in `arr`. Default is ``False``. Any other value should returns the 
        mask of existing element in reference array
        
    :returns: array and values missings or indexes in reference array. 
    
    :Example: 
        
    >>> import numpy as np 
    >>> from gofast.core.utils import ismissing
    >>> refreq = np.linspace(7e7, 1e0, 20) # 20 frequencies as reference
    >>> # remove the value between index 7 to 12 and stack again
    >>> freq = np.hstack ((refreq.copy()[:7], refreq.copy()[12:] ))  
    >>> f, m  = ismissing (refreq, freq)
    >>> f, m  
    ...array([7.00000000e+07, 6.63157895e+07, 6.26315791e+07, 5.89473686e+07,
           5.52631581e+07, 5.15789476e+07, 4.78947372e+07,            nan,
                      nan,            nan,            nan,            nan,
           2.57894743e+07, 2.21052638e+07, 1.84210534e+07, 1.47368429e+07,
           1.10526324e+07, 7.36842195e+06, 3.68421147e+06, 1.00000000e+00])
    >>> m # missing values 
    ... array([44210526.68421052, 40526316.21052632, 36842105.73684211,
           33157895.2631579 , 29473684.78947368])
    >>>  _, m_ix  = ismissing (refreq, freq, return_index =True)
    >>> m_ix 
    ... array([ 7,  8,  9, 10, 11], dtype=int64)
    >>> # assert the missing values from reference values 
    >>> refreq[m_ix ] # is equal to m 
    ... array([44210526.68421052, 40526316.21052632, 36842105.73684211,
           33157895.2631579 , 29473684.78947368]) 
        
    """
    return_index = str(return_index).lower() 
    fill_value = _assert_all_types(fill_value, float, int)
    if return_index in ('false', 'value', 'val') :
        return_index ='values' 
    elif return_index  in ('true', 'index', 'ix') :
        return_index = 'index' 
    else : 
        return_index = 'mask'
    
    ref = refarr.copy() ; mask = np.isin(ref, arr)
    miss_values = ref [~np.isin(ref, arr)] 
    miss_val_or_ix  = (ref [:, None] == miss_values).argmax(axis=0
                         ) if return_index =='index' else ref [~np.isin(ref, arr)] 
    
    miss_val_or_ix = mask if return_index =='mask' else miss_val_or_ix 
    # if return_missing_values: 
    ref [~np.isin(ref, arr)] = fill_value 
    #arr= np.hstack ((arr , np.repeat(fill_value, 0 if m <=0 else m  ))) 
    #refarr[refarr ==arr] if return_index else arr 
    return  ref , miss_val_or_ix   

def strip_item(item_to_clean, item=None, multi_space=12):
    """
    Function to strip item around string values.  if the item to clean is None or 
    item-to clean is "''", function will return None value

    Parameters
    ----------
        * item_to_clean : list or np.ndarray of string 
                 List to strip item.
        * cleaner : str , optional
                item to clean , it may change according the use. The default is ''.
        * multi_space : int, optional
                degree of repetition may find around the item. The default is 12.
    Returns
    -------
        list or ndarray
            item_to_clean , cleaned item 
            
    :Example: 
        
     >>> import numpy as np
     >>> new_data=_strip_item (item_to_clean=np.array(['      ss_data','    pati   ']))
     >>>  print(np.array(['      ss_data','    pati   ']))
     ... print(new_data)

    """
    if item==None :
        item = ' '
    
    cleaner =[(''+ ii*'{0}'.format(item)) for ii in range(multi_space)]
    
    if isinstance (item_to_clean, str) : 
        item_to_clean=[item_to_clean] 
        
    # if type(item_to_clean ) != list :#or type(item_to_clean ) !=np.ndarray:
    #     if type(item_to_clean ) !=np.ndarray:
    #         item_to_clean=[item_to_clean]
    if item_to_clean in cleaner or item_to_clean ==['']:
        #warnings.warn ('No data found for sanitization; returns None.')
        return None 
    try : 
        multi_space=int(multi_space)
    except : 
        raise TypeError('argument <multplier> must be an integer'
                        'not {0}'.format(type(multi_space)))
    
    for jj, ss in enumerate(item_to_clean) : 
        for space in cleaner:
            if space in ss :
                new_ss=ss.strip(space)
                item_to_clean[jj]=new_ss
    
    return item_to_clean  
 

def pretty_printer(
        clfs: List[_F],  
        clf_score:List[float]=None, 
        scoring: Optional[str] =None,
        **kws
 )->None: 
    """ Format and pretty print messages after gridSearch using multiples
    estimators.
    
    Display for each estimator, its name, it best params with higher score 
    and the mean scores. 
    
    Parameters
    ----------
    clfs:Callables 
        classifiers or estimators 
    
    clf_scores: array-like
        for single classifier, usefull to provided the 
        cross validation score.
    
    scoring: str 
        Scoring used for grid search.
    """
    empty =kws.pop('empty', ' ')
    e_pad =kws.pop('e_pad', 2)
    p=list()

    if not isinstance(clfs, (list,tuple)): 
        clfs =(clfs, clf_score)

    for ii, (clf, clf_be, clf_bp, clf_sc) in enumerate(clfs): 
        s_=[e_pad* empty + '{:<20}:'.format(
            clf.__class__.__name__) + '{:<20}:'.format(
                'Best-estimator <{}>'.format(ii+1)) +'{}'.format(clf_be),
         e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'Best paramaters') + '{}'.format(clf_bp),
         e_pad* empty  +'{:<20}:'.format(' ') + '{:<20}:'.format(
            'scores<`{}`>'.format(scoring)) +'{}'.format(clf_sc)]
        try : 
            s0= [e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'scores mean')+ '{}'.format(clf_sc.mean())]
        except AttributeError:
            s0= [e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'scores mean')+ 'None']
            s_ +=s0
        else :
            s_ +=s0

        p.extend(s_)
    
    for i in p: 
        print(i)
     
def sanitize_frame_cols(
    d,  func:_F = None , 
    regex=None, 
    pattern:str = None, 
    fill_pattern:str =None, 
    inplace:bool =False 
    ):
    """ Remove an indesirable characters to the dataframe and returns 
    new columns. 
    
    Use regular expression for columns sanitizing 
    
    Parameters 
    -----------
    
    d: list, columns, 
        columns to sanitize. It might contain a list of items to 
        to polish. If dataframe or series are given, the dataframe columns  
        and the name respectively will be polished and returns the same 
        dataframe.
        
    func: _F, callable 
       Universal function used to clean the columns 
       
    regex: `re` object,
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&.)(*@!_,;\s-]\s*', flags=re.IGNORECASE) 
    pattern: str, default = '[_#&.)(*@!_,;\s-]\s*'
        The base pattern to sanitize the text in each column names. 
        
    fill_pattern: str, default='' 
        pattern to replace the non-alphabetic character in each item of 
        columns. 
    inplace: bool, default=False, 
        transform the dataframe of series in place. 

    Returns
    -------
    columns | pd.Series | dataframe. 
        return Serie or dataframe if one is given, otherwise it returns a 
        sanitized columns. 
        
    Examples 
    ---------
    >>> from gofast.core.utils import sanitize_frame_cols 
    >>> from gofast.core.utils import read_data 
    >>> h502= read_data ('data/boreholes/H502.xlsx') 
    >>> h502 = sanitize_frame_cols (h502, fill_pattern ='_' ) 
    >>> h502.columns[:3]
    ... Index(['depth_top', 'depth_bottom', 'strata_name'], dtype='object') 
    >>> f = lambda r : r.replace ('_', "'s ") 
    >>> h502_f= sanitize_frame_cols( h502, func =f )
    >>> h502_f.columns [:3]
    ... Index(["depth's top", "depth's bottom", "strata's name"], dtype='object')
               
    """
    isf , iss= False , False 
    pattern = pattern or r'[_#&.)(*@!_,;\s-]\s*'
    fill_pattern = fill_pattern or '' 
    fill_pattern = str(fill_pattern)
    
    regex = regex or re.compile (pattern, flags=re.IGNORECASE)
    
    if isinstance(d, pd.Series): 
        c = [d.name]  
        iss =True 
    elif isinstance (d, pd.DataFrame ) :
        c = list(d.columns) 
        isf = True
        
    else : 
        c = is_iterable(d, exclude_string= True, transform =True )
 
    if inspect.isfunction(func): 
        c = list( map (func , c ) ) 
    
    else : c =list(map ( 
        lambda r : regex.sub(fill_pattern, r.strip() ), c ))
        
    if isf : 
        if inplace : 
            d.columns = c
        else : 
            d =pd.DataFrame(d.values, columns =c )
        
    elif iss:
        if inplace: 
            d.name = c[0]
        else :
            d= pd.Series (data =d.values, name =c[0] )
        
    else :
        d = c 

    return d 

 
def convert_value_in (v, unit ='m'): 
    """Convert value based on the reference unit.
    
    Parameters 
    ------------
    v: str, float, int, 
      value to convert 
    unit: str, default='m'
      Reference unit to convert value in. Default is 'meters'. Could be 
      'kg' or else. 
      
    Returns
    -------
    v: float, 
       Value converted. 
       
    Examples 
    ---------
    >>> from gofast.core.utils import convert_value_in 
    >>> convert_value_in (20) 
    20.0
    >>> convert_value_in ('20mm') 
    0.02
    >>> convert_value_in ('20kg', unit='g') 
    20000.0
    >>> convert_value_in ('20') 
    20.0
    >>> convert_value_in ('20m', unit='g')
    ValueError: Unknwon unit 'm'...
    """
    c= { 'k':1e3 , 
        'h':1e2 , 
        'dc':1e1 , 
        '':1e0 , 
        'd':1e-1, 
        'c':1e-2 , 
        'm':1e-3  
        }
    c = {k +str(unit).lower(): v for k, v in c.items() }

    v = str(v).lower()  

    regex = re.findall(r'[a-zA-Z]', v) 
    
    if len(regex) !=0: 
        unit = ''.join( regex ) 
        v = v.replace (unit, '')

    if unit not in c.keys(): 
        raise ValueError (
            f"Unknwon unit {unit!r}. Expect {smart_format(c.keys(), 'or' )}."
            f" Or rename the `unit` parameter maybe to {unit[-1]!r}.")
    
    return float ( v) * (c.get(unit) or 1e0) 

def get_confidence_ratio (
    ar, 
    axis = 0, 
    invalid = 'NaN',
    mean=False, 
    ):
    
    """ Get ratio of confidence in array by counting the number of 
    invalid values. 
    
    Parameters 
    ------------
    ar: arraylike 1D or 2D  
      array for checking the ratio of confidence 
      
    axis: int, default=0, 
       Compute the ratio of confidence alongside the rows by defaults. 
       
    invalid: int, foat, default='NaN'
      The value to consider as invalid in the data might be listed if 
      applicable. The default is ``NaN``. 
      
    mean: bool, default=False, 
      Get the mean ratio. Average the percentage of each axis. 
      
      .. versionadded:: 0.2.8 
         Average the ratio of confidence of each axis. 
      
    Returns 
    ---------
    ratio: arraylike 1D 
      The ratio of confidence array alongside the ``axis``. 

    Examples 
    ----------
    >>> import numpy as np 
    >>> np.random.seed (0) 
    >>> test = np.random.randint (1, 20 , 10 ).reshape (5, 2 ) 
    >>> test
    array([[13, 16],
           [ 1,  4],
           [ 4,  8],
           [10, 19],
           [ 5,  7]])
    >>> from gofast.core.utils import get_confidence_ratio 
    >>> get_confidence_ratio (test)
    >>> array([1., 1.])
    >>> get_confidence_ratio (test, invalid= ( 13, 19) )
    array([0.8, 0.8])
    >>> get_confidence_ratio (test, invalid= ( 13, 19, 4) )
    array([0.6, 0.6])
    >>> get_confidence_ratio (test, invalid= ( 13, 19, 4), axis =1 )
    array([0.5, 0.5, 0.5, 0.5, 1. ])
    
    """
    def gfc ( ar, inv):
        """ Get ratio in each column or row in the array. """
        inv = is_iterable(inv, exclude_string=True , transform =True, 
                              )
        # if inv!='NaN': 
        for iv in inv: 
            if iv in ('NAN', np.nan, 'NaN', 'nan', None): 
                iv=np.nan  
            ar [ar ==iv] = np.nan 
                
        return len( ar [ ~np.isnan (ar)])  / len(ar )
    
    # validate input axis name 
    axis = validate_name_in (axis , ('1', 'rows', 'sites', 'stations') ,
                              expect_name=1 )
    if not axis:
        axis =0 
    
    ar = np.array(ar).astype ( np.float64) # for consistency
    ratio = np.zeros(( (ar.shape[0] if axis ==1 else ar.shape [1] )
                      if ar.ndim ==2 else 1, ), dtype= np.float64) 
    
    for i in range (len(ratio)): 
        ratio[i] = gfc ( (ar [:, i] if axis ==0 else ar [i, :])
                        if ar.ndim !=1 else ar , inv= invalid 
                        )
    if mean: 
        ratio = np.array (ratio).mean() 
    return ratio 
    
def make_obj_consistent_if ( 
        item= ... , default = ..., size =None, from_index: bool =True ): 
    """Combine default values to item to create default consistent iterable 
    objects. 
    
    This is valid if  the size of item does not fit the number of 
    expected iterable objects.     
    
    Parameters 
    ------------
    item : Any 
       Object to construct it default values 
       
    default: Any 
       Value to hold in the case the items does not match the size of given items 
       
    size: int, Optional 
      Number of items to return. 
      
    from_index: bool, default=True 
       make an item size to match the exact size of given items 
       
    Returns 
    -------
       item: Iterable object that contain default values. 
       
    Examples 
    ----------
    >>> from gofast.core.utils import make_obj_consistent_if
    >>> from gofast.exlib import SVC, LogisticRegression, XGBClassifier 
    >>> classifiers = ["SVC", "LogisticRegression", "XGBClassifier"] 
    >>> classifier_names = ['SVC', 'LR'] 
    >>> make_obj_consistent_if (classifiers, default = classifier_names ) 
    ['SVC', 'LogisticRegression', 'XGBClassifier']
    >>> make_obj_consistent_if (classifier_names, from_index =False  )
    ['SVC', 'LR']
    >>> >>> make_obj_consistent_if ( classifier_names, 
                                     default= classifiers, size =3 , 
                                     from_index =False  )
    ['SVC', 'LR', 'SVC']
    
    """
    if default==... or None : default =[]
    # for consistency 
    default = list( is_iterable (default, exclude_string =True,
                                 transform =True ) ) 
    
    if item not in ( ...,  None) : 
         item = list( is_iterable( item , exclude_string =True ,
                                  transform = True ) ) 
    else: item = [] 
    
    item += default[len(item):] if from_index else default 
    
    if size is not None: 
        size = int (_assert_all_types(size, int, float,
                                      objname = "Item 'size'") )
        item = item [:size]
        
    return item

def ellipsis2false(
        *parameters, default_value: Any = True
   ):
    """
    Turn all parameter arguments to the default value if ellipsis (`...`)
    is provided.

    This function processes a tuple of parameters, replacing any instance 
    of the ellipsis (`...`) with the specified `default_value`. The output 
    maintains the same order as the input parameters.

    Parameters
    ----------
    parameters : tuple
        A tuple of parameters. Each element in the tuple is checked, 
        and if it is an ellipsis (`...`), it will be replaced by the 
        `default_value`.
    
    default_value : Any, optional, default=False
        The value to replace any ellipsis (`...`) in the `parameters`. 
        By default, this is set to `False`.

    Returns
    -------
    tuple
        A tuple of the same length as `parameters`, with ellipses replaced 
        by `default_value`.

    Examples
    --------
    >>> from gofast.core.utils import ellipsis2false
    >>> var, = ellipsis2false(...)
    >>> var
    False

    >>> data, sep, verbose = ellipsis2false([2, 3, 4], ',', ...)
    >>> verbose
    False

    Notes
    -----
    - This function ensures that the output tuple matches the order of the 
      input parameters, with any ellipses replaced by the `default_value`.
    - If a single parameter is provided, the trailing comma is used to 
      collect it into a tuple.
    """
    return tuple(
        (default_value if param is ... else param for param in parameters)
    )

def type_of_target(y):
    """
    Determine the type of data indicated by the target variable.

    Parameters
    ----------
    y : array-like
        Target values. 

    Returns
    -------
    target_type : string
        Type of target data, such as 'binary', 'multiclass', 'continuous', etc.

    Examples
    --------
    >>> type_of_target([0, 1, 1, 0])
    'binary'
    >>> type_of_target([0.5, 1.5, 2.5])
    'continuous'
    >>> type_of_target([[1, 0], [0, 1]])
    'multilabel-indicator'
    """
    # Check if y is an array-like
    if not isinstance(y, (np.ndarray, list, pd.Series, Sequence, pd.DataFrame)):
        raise ValueError("Expected array-like (array or list), got %s" % type(y))
    
    # check whether it is a series of pandas dataframe with single column.
    if isinstance (y, pd.Series) or ( 
            isinstance (y, pd.DataFrame) and len(y.columns) ==1): 
        y= np.array(y).ravel() # ravel it for 
        
    y= np.asarray(y)
    # Check for valid number type
    if not all(isinstance(i, (int, float, np.integer, np.floating)) 
               for i in np.array(y).flatten()):
        raise ValueError("Input must be a numeric array-like")

    # Continuous data
    if any(isinstance(i, float) for i in np.array(y).flatten()):
        return 'continuous'

    # Binary or multiclass
    unique_values = np.unique(y)
    if len(unique_values) == 2:
        return 'binary'
    elif len(unique_values) > 2 and np.ndim(y) == 1:
        return 'multiclass'

    # Multilabel indicator
    if isinstance(y[0], (np.ndarray, list, Sequence)) and len(y[0]) > 1:
        return 'multilabel-indicator'

    return 'unknown'


def fancier_repr_formatter(obj, max_attrs=7):
    """
    Generates a formatted string representation for any class object.

    Parameters:
    ----------
    obj : object
        The object for which the string representation is generated.

    max_attrs : int, optional
        Maximum number of attributes to display in the representation.

    Returns:
    -------
    str
        A string representation of the object.

    Examples:
    --------
    >>> from gofast.core.utils import fancier_repr_formatter
    >>> class MyClass:
    >>>     def __init__(self, a, b, c):
    >>>         self.a = a
    >>>         self.b = b
    >>>         self.c = c
    >>> obj = MyClass(1, [1, 2, 3], 'hello')
    >>> print(fancier_repr_formatter(obj))
    MyClass(a=1, c='hello', ...)
    """
    attrs = [(name, getattr(obj, name)) for name in dir(obj)
             if not name.startswith('_') and
             (isinstance(getattr(obj, name), str) or
              not hasattr(getattr(obj, name), '__iter__'))]

    displayed_attrs = attrs[:min(len(attrs), max_attrs)]
    attr_str = ', '.join([f'{name}={value!r}' for name, value in displayed_attrs])

    # Add ellipsis if there are more attributes than max_attrs
    if len(attrs) > max_attrs:
        attr_str += ', ...'

    return f'{obj.__class__.__name__}({attr_str})'


def normalize_string(
    input_str: str, 
    target_strs: Optional[List[str]] = None, 
    num_chars_check: Optional[int] = None, 
    deep: bool = False, 
    return_target_str: bool = False,
    return_target_only: bool=False, 
    raise_exception: bool = False,
    ignore_case: bool = True,
    match_method: str = 'exact',
    error_msg: str=None, 
) -> Union[str, Tuple[str, Optional[str]]]:
    """
    Normalizes a string by applying various transformations and optionally checks 
    against a list of target strings based on different matching methods.

    Function normalizes a string by stripping leading/trailing whitespace, 
    converting to lowercase,and optionally checks against a list of target  
    strings. If specified, returns the target string that matches the 
    conditions. Raise an exception if the string is not found.
    
    Parameters
    ----------
    input_str : str
        The string to be normalized.
    target_strs : List[str], optional
        A list of target strings for comparison.
    num_chars_check : int, optional
        The number of characters at the start of the string to check 
        against each target string.
    deep : bool, optional
        If True, performs a deep substring check within each target string.
    return_target_str : bool, optional
        If True and a target string matches, returns the matched target string 
        along with the normalized string.
    return_target_only: bool, optional 
       If True and a target string  matches, returns only the matched string
       target. 
    raise_exception : bool, optional
        If True and the input string is not found in the target strings, 
        raises an exception.
    ignore_case : bool, optional
        If True, ignores case in string comparisons. Default is True.
    match_method : str, optional
        The string matching method: 'exact', 'contains', or 'startswith'.
        Default is 'exact'.
    error_msg: str, optional, 
       Message to raise if `raise_exception` is ``True``. 
       
    Returns
    -------
    Union[str, Tuple[str, Optional[str]]]
        The normalized string. If return_target_str is True and a target 
        string matches, returns a tuple of the normalized string and the 
        matched target string.

    Raises
    ------
    ValueError
        If raise_exception is True and the input string is not found in 
        the target strings.

    Examples
    --------
    >>> from gofast.core.utils import normalize_string
    >>> normalize_string("Hello World", target_strs=["hello", "world"], ignore_case=True)
    'hello world'
    >>> normalize_string("Goodbye World", target_strs=["hello", "goodbye"], 
                         num_chars_check=7, return_target_str=True)
    ('goodbye world', 'goodbye')
    >>> normalize_string("Hello Universe", target_strs=["hello", "world"],
                         raise_exception=True)
    ValueError: Input string not found in target strings.
    """
    normalized_str = str(input_str).lower() if ignore_case else input_str

    if not target_strs:
        return normalized_str
    target_strs = is_iterable(target_strs, exclude_string=True, transform =True)
    normalized_targets = [str(t).lower() for t in target_strs] if ignore_case else target_strs
    matched_target = None

    for target in normalized_targets:
        if num_chars_check is not None:
            condition = (normalized_str[:num_chars_check] == target[:num_chars_check])
        elif deep:
            condition = (normalized_str in target)
        elif match_method == 'contains':
            condition = (target in normalized_str)
        elif match_method == 'startswith':
            condition = normalized_str.startswith(target)
        else:  # Exact match
            condition = (normalized_str == target)

        if condition:
            matched_target = target
            break

    if matched_target is not None:
        if return_target_only: 
            return matched_target 
        return (normalized_str, matched_target) if return_target_str else normalized_str

    if raise_exception:
        error_msg = error_msg or ( 
            f"Invalid input. Expect {smart_format(target_strs, 'or')}."
            f" Got {input_str!r}."
            )
        raise ValueError(error_msg)
    
    if return_target_only: 
        return matched_target 
    
    return ('', None) if return_target_str else ''

def format_and_print_dict(data_dict, front_space=4):
    """
    Formats and prints the contents of a dictionary in a structured way.

    Each key-value pair in the dictionary is printed with the key followed by 
    its associated values. 
    The values are expected to be dictionaries themselves, allowing for a nested 
    representation.
    The inner dictionary's keys are sorted in descending order before printing.

    Parameters
    ----------
    data_dict : dict
        A dictionary where each key contains a dictionary of items to be printed. 
        The key represents a category
        or label, and the value is another dictionary where each key-value pair 
        represents an option or description.
        
    front_space : int, optional
        The number of spaces used for indentation in front of each line (default is 4).


    Returns
    -------
    None
        This function does not return any value. It prints the formatted contents 
        of the provided dictionary.

    Examples
    --------
    >>> from gofast.core.utils import format_and_print_dict
    >>> sample_dict = {
            'gender': {1: 'Male', 0: 'Female'},
            'age': {1: '35-60', 0: '16-35', 2: '>60'}
        }
    >>> format_and_print_dict(sample_dict)
    gender:
        1: Male
        0: Female
    age:
        2: >60
        1: 35-60
        0: 16-35
    """
    if not isinstance(data_dict, dict):
        raise TypeError("The input data must be a dictionary.")

    indent = ' ' * front_space
    for label, options in data_dict.items():
        print(f"{label}:")
        options= is_iterable(options, exclude_string=True, transform=True )
  
        if isinstance(options, (tuple, list)):
            for option in options:
                print(f"{indent}{option}")
        elif isinstance(options, dict):
            for key in sorted(options.keys(), reverse=True):
                print(f"{indent}{key}: {options[key]}")
        print()  # Adds an empty line for better readability between categories


def fill_nan_in(
        data: DataFrame,  method: str = 'constant', 
        value: Optional[Union[int, float, str]] = 0) -> DataFrame:
    """
    Fills NaN values in a Pandas DataFrame using various methods.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be checked and modified.
    method : str, optional
        The method to use for filling NaN values. Options include 'constant',
        'ffill', 'bfill', 'mean', 'median', 'mode'. Default is 'constant'.
    value : int, float, string, optional
        The value used when method is 'constant'. Ignored for other methods.
        Default is 0.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame with NaN values filled.

    Example
    -------
    >>> import pandas as pd
    >>> from gofast.core.utils import fill_nan_in
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 2, 3]})
    >>> df = fill_nan_in(df, method='median')
    >>> print(df)
       A    B
    0  1.0  2.5
    1  2.0  2.0
    2  1.5  3.0
    """
    # Check for NaN values in the DataFrame and apply the specified fill method
    if not data.isna().any().any(): 
        return data 

    fill_methods = {
        'constant': lambda: data.fillna(value, inplace=True),
        'ffill': lambda: data.fillna(method='ffill', inplace=True),
        'bfill': lambda: data.fillna(method='bfill', inplace=True),
        'mean': lambda: data.fillna(data.mean(), inplace=True),
        'median': lambda: data.fillna(data.median(), inplace=True),
        'mode': lambda: data.apply(lambda col: col.fillna(col.mode()[0], inplace=True))
    }
    
    fill_action = fill_methods.get(method)
    if fill_action:
        fill_action()
    else:
        raise ValueError(f"Method '{method}' not recognized for filling NaN values.")
        
    return data 

        
def contains_delimiter(s: str, delimiters: Union[str, list, set]) -> bool:
    """
    Checks if the given string contains any of the specified delimiters.

    Parameters
    ----------
    s : str
        The string to check.
    delimiters : str, list, or set
        Delimiters to check for in the string. Can be specified as a single
        string (for a single delimiter), a list of strings, or a set of strings.

    Returns
    -------
    bool
        True if the string contains any of the delimiters, False otherwise.

    Examples
    --------
    >>> from gofast.core.utils import contains_delimiter
    >>> contains_delimiter("example__string", "__")
    True

    >>> contains_delimiter("example--string", ["__", "--", "&", "@", "!"])
    True

    >>> contains_delimiter("example&string", {"__", "--", "&", "@", "!"})
    True

    >>> contains_delimiter("example@string", "__--&@!")
    True

    >>> contains_delimiter("example_string", {"__", "--", "&", "@", "!"})
    False

    >>> contains_delimiter("example#string", "#$%")
    True

    >>> contains_delimiter("example$string", ["#", "$", "%"])
    True

    >>> contains_delimiter("example%string", "#$%")
    True

    >>> contains_delimiter("example^string", ["#", "$", "%"])
    False
    """
    # for consistency
    s = str(s) 
    # Convert delimiters to a set if it's not already a set
    if not isinstance(delimiters, set):
        if isinstance(delimiters, str):
            delimiters = set(delimiters)
        else:  # Assuming it's a list or similar iterable
            delimiters = set(delimiters)
    
    return any(delimiter in s for delimiter in delimiters)    

def lowertify(
    *values,
    strip: bool = True, 
    return_origin: bool = False, 
    unpack: bool = False
    ) -> Union[Tuple[str, ...], Tuple[Tuple[str, Any], ...], Tuple[Any, ...]]:
    """
    Convert all input values to lowercase strings, optionally stripping 
    whitespace, and optionally return the original values alongside the 
    lowercased versions.
    
    Can also unpack the tuples of lowercased and original values into a single
    flat tuple.

    Parameters
    ----------
    *values : Any
        Arbitrary number of values to be converted to lowercase. Non-string 
        values will be converted to strings before processing.
    strip : bool, optional
        If True (default), leading and trailing whitespace will be removed 
        from the strings.
    return_origin : bool, optional
        If True, each lowercased string is returned as a tuple with its 
        original value; otherwise, only the lowercased strings are returned.
    unpack : bool, optional
        If True, and `return_origin` is also True, the function returns a 
        single flat tuple containing all lowercased and original values 
        alternatively. This parameter is ignored if `return_origin` is False.

    Returns
    -------
    Union[Tuple[str, ...], Tuple[Tuple[str, Any], ...], Tuple[Any, ...]]
        Depending on `return_origin` and `unpack` flags, returns either:
        - A tuple of lowercased (and optionally stripped) strings.
        - A tuple of tuples, each containing the lowercased string and its 
          original value.
        - A single flat tuple containing all lowercased and original values 
          alternatively (if `unpack` is True).

    Examples
    --------
    >>> from gofast.core.utils import lowertify
    >>> lowertify('KIND')
    ('kind',)
    
    >>> lowertify("KIND", return_origin=True)
    (('kind', 'KIND'),)
    
    >>> lowertify("args1", 120, 'ArG3')
    ('args1', '120', 'arg3')
    
    >>> lowertify("args1", 120, 'ArG3', return_origin=True)
    (('args1', 'args1'), ('120', 120), ('arg3', 'ArG3'))
    
    >>> lowertify("KIND", "task ", return_origin=True, unpack=True)
    ('kind', 'KIND', 'task', 'task ')
    """
    processed_values = [(str(val).strip().lower() if strip 
                         else str(val).lower(), val) for val in values]
    if return_origin:
        if unpack:
            # Flatten the list of tuples into a single tuple for unpacking
            return tuple(item for pair in processed_values for item in pair)
        else:
            return tuple(processed_values)
    else:
        return tuple(lowered for lowered, _ in processed_values)
    
def fetch_value_in(
    data: Dict[str, Any],
    key: str,
    kind: str = "metric",
    mode: str = "soft",
    suffixes: Tuple[str, ...] = ("_score", "_loss", "_error"),
    strict_suffix: bool = False,
    error: str = 'raise',
) -> Any:
    """
    Fetch the value from a dictionary based on flexible criteria using regex.

    Parameters
    ----------
    data : dict
        The dictionary to search within.

    key : str
        The key or partial key to search for in the dictionary.

    kind : str, optional
        Specifies the type of data (e.g., 'metric'). Default is 'metric'.

    mode : str, {'soft', 'strict'}, optional
        - If 'soft', attempts a best match with suffixes.
        - If 'strict', the key must match exactly.
        Default is 'soft'.

    suffixes : tuple, optional
        A tuple of suffixes to consider for appending to the search key.
        Default is ("_score", "_loss", "_error").

    strict_suffix : bool, optional
        If True, only keys that match with the suffix are considered in
        'soft' mode. Default is False.

    error : str, {'raise', 'warn', 'ignore'}, optional
        Specifies the behavior when an ambiguous or no match is found.
        - 'raise': Raise a ValueError.
        - 'warn': Issue a warning and use the first match.
        - 'ignore': Silently use the first match if available.

    Returns
    -------
    value : Any
        The value corresponding to the matching key in the dictionary.

    Raises
    ------
    ValueError
        If the key is ambiguous or cannot be found in the dictionary,
        depending on the 'error' parameter.

    Examples
    --------
    >>> from gofast.core.utils import fetch_value_in
    >>> metrics = {
    ...     'accuracy_score': lambda x: x,
    ...     'balanced_accuracy_score': lambda x: x,
    ...     'top_k_accuracy_score': lambda x: x,
    ...     'balanced_accuracy': lambda x: x,
    ... }
    >>> fetch_value_in(metrics, 'accuracy')
    <function <lambda> at 0x...>
    """
    if isinstance(key, str): 
        key = get_full_metric_name(key)
        
    if kind != "metric":
        raise ValueError(
            f"Unsupported kind '{kind}'. Currently only 'metric' is supported.")

    matches = []

    # Compile regex pattern based on mode
    if mode == "soft":
        # Create patterns that start with the key and optionally followed by suffixes
        patterns = [f"^{re.escape(key)}{re.escape(suffix)}$" for suffix in suffixes]
        patterns.insert(0, f"^{re.escape(key)}$")  # Exact key match has highest priority
    elif mode == "strict":
        patterns = [f"^{re.escape(key)}$"]  # Only exact match
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'soft' or 'strict'.")

    # If strict_suffix is True in soft mode, ensure suffixes are present
    if mode == "soft" and strict_suffix:
        patterns = [f"^{re.escape(key)}{re.escape(suffix)}$" for suffix in suffixes]

    # Compile all patterns
    regex_patterns = [re.compile(pattern) for pattern in patterns]

    # Search for matches
    for k in data.keys():
        if any(pattern.match(k) for pattern in regex_patterns):
            matches.append(k)

    if not matches:
        if error == 'raise':
            raise ValueError(
                f"Key '{key}' could not be resolved in the dictionary"
                f" under mode='{mode}' and kind='{kind}'."
            )
        elif error == 'warn':
            warnings.warn(
                f"Key '{key}' could not be resolved. Returning None.",
                UserWarning
            )
            return None
        elif error == 'ignore':
            return None
        else:
            raise ValueError(f"Unsupported error handling option '{error}'."
                             " Use 'raise', 'warn', or 'ignore'.")

    if len(matches) > 1:
        if mode == "soft":
            if error == 'raise':
                raise ValueError(
                    f"Ambiguous key '{key}' matched multiple entries: {matches}."
                )
            elif error == 'warn':
                warnings.warn(
                    f"Ambiguous key '{key}' matched multiple entries: {matches}."
                    " Using the first match '{matches[0]}'.",
                    UserWarning
                )
                return data[matches[0]]
            elif error == 'ignore':
                return data[matches[0]]
            else:
                raise ValueError(f"Unsupported error handling option '{error}'."
                                 " Use 'raise', 'warn', or 'ignore'.")
        else:
            # In strict mode, there should be at most one match
            raise ValueError(
                f"Ambiguous key '{key}' matched multiple entries"
                f" in strict mode: {matches}."
            )

    # Single match found
    return data[matches[0]]

def get_full_metric_name(metric: str) -> str:
    """
    Resolve the complete metric name based on a mapping. If the metric is not found
    in the mapping or the special 'gofast' metrics, return the metric in lowercase.

    Parameters
    ----------
    metric : str
        The input metric name provided by the user.

    Returns
    -------
    str
        The resolved complete metric name.

    Examples
    --------
    >>> get_complete_metric_name('mae')
    'mean_absolute_error'
    >>> get_complete_metric_name('MAPE')
    'mean_absolute_percentage_error'
    >>> get_complete_metric_name('msle')
    'mean_squared_log_error'
    >>> get_complete_metric_name('mean_squared_log_error')
    'mean_squared_log_error'
    >>> get_complete_metric_name('unknown_metric')
    'unknown_metric'
    """
    # Define the mapping from shorthand or alternative
    #  names to full metric names
    _METRIC_MAPPING= {
        # Standard Metrics
        'mae': 'mean_absolute_error',
        'mape': 'mean_absolute_percentage_error',
        'mse': 'mean_squared_error',
        'rmse': 'root_mean_squared_error',  # GOFAST metric
        'r2': 'r2_score',
        'accuracy': 'accuracy_score',
        'f1': 'f1_score',
        'precision': 'precision_score',
        'recall': 'recall_score',
        'auc': 'roc_auc_score',
        'msle': 'mean_squared_log_error',   # a GOFAST metric
        'bacc': 'balanced_accuracy',        # GOFAST metric
        'bacc_score': 'balanced_accuracy_score', # GOFAST metric
        'iv': 'information_value',          # GOFAST metric
        'evs': 'explained_variance_score',  # GOFAST metric
        'mae_dev': 'mean_absolute_deviation',  # GOFAST metric
        'dsc': 'dice_similarity_score',     # GOFAST metric
        'gini': 'gini_score',               # GOFAST metric
        'hloss': 'hamming_loss',            # GOFAST metric
        'fms': 'fowlkes_mallows_score',     # GOFAST metric
        'rmsle': 'root_mean_squared_log_error',  # GOFAST metric
        'mpe': 'mean_percentage_error',     # GOFAST metric
        'pbias': 'percentage_bias',         # GOFAST metric
        'sprank': 'spearmans_rank_score',   # GOFAST metric
        'precision_at_k': 'precision_at_k', # GOFAST metric
        'ndcg_at_k': 'ndcg_at_k',           # GOFAST metric
        'mrs': 'mean_reciprocal_score',     # GOFAST metric
        'adj_r2': 'adjusted_r2_score',      # GOFAST metric
        'likelihood': 'likelihood_score',   # GOFAST metric
        'twa': 'twa_score',                 # GOFAST metric
    }
    if callable (metric): 
        metric = metric.__name__ 
        
    metric_lower = str(metric).lower()

    # Check if the metric is in the predefined mapping
    if metric_lower in _METRIC_MAPPING:
        return _METRIC_MAPPING[metric_lower]

    # If not found, return the metric in lowercase
    return metric_lower

def error_policy(
    error: str | None,
    *,
    policy: str = 'auto',
    base: str = 'ignore',
    exception: type[Exception] = None,
    msg: str | None = None, 
    valid_policies: set =None 
) -> str:
    """
    Manage error-handling policies like 'warn', 'raise', or 'ignore'.

    The `error_policy` function determines how to handle potential
    errors by mapping the user-provided ``error`` argument to a valid
    policy. It helps standardize responses such as warnings, raised
    exceptions, or silent ignores. The function can adapt to different
    modes, allowing for strict or flexible behavior depending on the
    ``policy`` and ``base`` settings.

    Parameters
    ----------
    error : str or None
        The user-provided error setting. Can be `'warn'`, `'raise'`,
        `'ignore'`, or `None`. If `None`, the behavior is resolved
        based on ``policy`` and ``base``.
    
    policy : str, default='auto'
        Determines how to interpret a `None` error setting. Valid
        options:
        
        - `'auto'`: Resolve `None` to the default `base` policy.
        - `'strict'`: Disallows `None` for `error`; raises an error
          if encountered.
        - `None`: Defers strictly to `base`.

    base : str, default='ignore'
        The fallback error policy when `None` is encountered and
        `policy='auto'` or `policy=None`. Must be one of `'warn'`,
        `'raise'`, or `'ignore'`.

    exception : type of Exception, default=ValueError
        The exception class to be raised if an invalid policy or
        error is encountered.

    msg : str, optional
        A custom message for the raised exception if an invalid
        `error` or `policy` is detected. If omitted, a default is
        used.

    Returns
    -------
    str
        A valid error policy: one of `'warn'`, `'raise'`, or
        `'ignore'`.

    Raises
    ------
    ValueError
        If `policy` is invalid or if `None` is not permitted by
        `policy='strict'` but is used. Also raised if `error` cannot
        be resolved to a valid policy or if `base` is invalid when
        `policy='auto'`.

    Notes
    -----
    - If `error` is already a valid policy (`'warn'`, `'raise'`,
      `'ignore'`), it is returned immediately.
    - When `error=None`, the behavior depends on the `policy` and
      `base` parameters. Setting `policy='strict'` disallows `None`
      for `error`.
      
      
    .. math::
       \\text{error\\_policy}:
       \\begin{cases}
         \\text{'warn'}, & \\text{issue a warning} \\\\
         \\text{'raise'}, & \\text{raise an exception} \\\\
         \\text{'ignore'}, & \\text{do nothing}
       \\end{cases}


    Examples
    --------
    >>> from gofast.core.utils import error_policy
    >>> # Basic usage:
    >>> resolved_error = error_policy('warn')
    >>> print(resolved_error)
    'warn'
    
    >>> # Using 'auto' policy with a default base of 'ignore'
    >>> resolved_error = error_policy(None, policy='auto',
    ...                                base='warn')
    >>> print(resolved_error)
    'warn'
    
    >>> # Strict policy disallows None
    >>> error_policy(None, policy='strict')
    ValueError: In strict policy, `None` is not acceptable as error.

    See Also
    --------
    gofast.utils.validator.validate_nan_policy : A function that
        validate NaN policies.
    """  # noqa: E501

    # Predefined valid policies.
    valid_policies = valid_policies or {'warn', 'raise', 'ignore'}

    # Default message if none is provided.
    default_msg = (
        "Invalid error policy: '{error}'. Valid options are "
        f"{valid_policies}."
    )
    if exception is None:
        exception = ValueError

    # Use custom message or default if not supplied.
    msg = msg or default_msg

    # Validate the `policy` argument.
    if policy not in {'auto', 'strict', None}:
        raise ValueError(
            f"Invalid policy: '{policy}'. Valid options are "
            "'auto', 'strict', or None."
        )

    # Resolve None values for `error` according to `policy`.
    if error is None:
        if policy == 'auto':
            # If policy='auto', fallback to `base` if no override is set.
            error = base or 'ignore'
        elif policy == 'strict':
            # If policy='strict', disallow None for `error`.
            raise ValueError(
                "In strict policy, `None` is not acceptable as an "
                "error. Please set `error` explicitly or switch "
                "policy to 'auto'."
            )
        else:
            # policy=None means strictly use `base` for resolution.
            if base not in valid_policies:
                raise ValueError(
                    f"Invalid base policy: '{base}'. Must be one of "
                    f"{valid_policies} when `error` is None and "
                    "policy is None."
                )
            error = base

    # Final check to ensure `error` is valid.
    if error not in valid_policies:
        # Raise the specified exception if the policy is invalid.
        raise exception(msg.format(error=error))

    # Return the resolved error policy.
    return error