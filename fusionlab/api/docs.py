# -*- coding: utf-8 -*-
from __future__ import annotations
from textwrap import dedent 
from typing import Callable 

_shared_docs: dict[str, str] = {}

__all__ =["doc", ]

_shared_docs[
    "data"
] = """data : array-like, pandas.DataFrame, str, dict, or Path-like
    The input `data`, which can be either an array-like object (e.g., 
    list, tuple, or numpy array), a pandas DataFrame, a file path 
    (string or Path-like), or a dictionary. The function will 
    automatically convert it into a pandas DataFrame for further 
    processing based on its type. Here's how each type is handled:

    1. **Array-like (list, tuple, numpy array)**:
       If `data` is array-like (e.g., a list, tuple, or numpy array), 
       it will be converted to a pandas DataFrame. Each element in 
       the array will correspond to a row in the resulting DataFrame. 
       The columns can be specified manually if desired, or pandas 
       will auto-generate column names.

       Example:
       >>> data = [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ]
       >>> df = pd.DataFrame(data, columns=['A', 'B', 'C'])
       >>> print(df)
          A  B  C
       0  1  2  3
       1  4  5  6
       2  7  8  9

    2. **pandas.DataFrame**:
       If `data` is already a pandas DataFrame, it will be returned as 
       is without any modification. This allows flexibility for cases 
       where the input data is already in DataFrame format.

       Example:
       >>> data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
       >>> print(data)
          A  B
       0  1  3
       1  2  4

    3. **File path object (str or Path-like)**:
       If `data` is a file path (either a string or Path-like object), 
       it will be read and converted into a pandas DataFrame. Supported 
       file formats include CSV, Excel, and other file formats that can 
       be read by pandas' `read_*` methods. This enables seamless 
       reading of data directly from files.

       Example:
       >>> data = "data.csv"
       >>> df = pd.read_csv(data)
       >>> print(df)
          A  B  C
       0  1  2  3
       1  4  5  6

    4. **Dictionary**:
       If `data` is a dictionary, it will be converted into a pandas 
       DataFrame. The dictionary's keys become the column names, and 
       the values become the corresponding rows. This is useful when 
       the data is already structured as key-value pairs.

       Example:
       >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
       >>> df = pd.DataFrame(data)
       >>> print(df)
          A  B
       0  1  4
       1  2  5
       2  3  6

    The `data` parameter can accept a variety of input types and will 
    be converted into a pandas DataFrame accordingly. In case of invalid 
    types or unsupported formats, a `ValueError` will be raised to notify 
    the user of the issue.

    Notes
    ------
    If `data` is an unsupported type or cannot be converted into a 
    pandas DataFrame, a `ValueError` will be raised with a clear 
    error message describing the issue.

    The `data` parameter will be returned as a pandas DataFrame, 
    regardless of its initial format.
    
"""

_shared_docs[
    "y_true"
] = """y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels or binary label indicators for the regression or 
        classification problem.
    
        The `y_true` parameter represents the ground truth values. It can be either:
        - A 1D array for binary classification or single-label classification, 
          where each element represents the true class label for a sample.
        - A 2D array for multilabel classification, where each row corresponds to 
          the true labels for a sample in a multi-output problem.

        Example:
        1. ** Regression problem 
        
        >>> y_true = [1.20, 0.62, 0.78, 0.02]
        >>> print(y_true)
        [1.20, 0.62, 0.78, 0.02]
        
        2. **Binary classification (1D array)**:
    
        >>> y_true = [0, 1, 0, 1]
        >>> print(y_true)
        [0, 1, 0, 1]

        3. **Multilabel classification (2D array)**:
    
        >>> y_true = [[0, 1], [1, 0], [0, 1], [1, 0]]
        >>> print(y_true)
        [[0, 1], [1, 0], [0, 1], [1, 0]]
"""

_shared_docs[
    "y_pred"
] = """y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted labels or probabilities, as returned by a classifier.
        
        The `y_pred` parameter contains the predictions made by a classifier. 
        It can be:
        - Predicted class labels (in the case of classification).
        - Probabilities representing the likelihood that each sample belongs 
        to each class. If probabilities are provided, a threshold can be used 
        to convert these into binary labels (e.g., class 1 if the probability 
        exceeds the threshold, otherwise class 0).
        
        Example:
        1. **Predicted regression labels 
        
        >>> y_pred = [1.21, 0.60, 0.76, 0.50]
        >>> print(y_pred)
        [1.21, 0.60, 0.76, 0.50]
        
        1. **Predicted class labels for binary classification (1D array)**:
       
        >>> y_pred = [0, 1, 0, 1]
        >>> print(y_pred)
        [0, 1, 0, 1]
        
        2. **Predicted probabilities for binary classification (1D array)**:
       
        >>> y_pred = [0.1, 0.9, 0.2, 0.7]
        >>> print(y_pred)
        [0.1, 0.9, 0.2, 0.7]
    
        3. **Predicted class labels for multilabel classification (2D array)**:
        
        >>> y_pred = [[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]]
        >>> print(y_pred)
        [[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]]
"""

_shared_docs[
    "alpha"
] = """alpha : float, default={value}
    Decay factor for time weighting, controlling the emphasis on 
    more recent predictions.

    The `alpha` parameter determines how much weight should be assigned to recent 
    predictions when computing metrics. It is used to apply a time-based decay, 
    where a higher value gives more weight to the most recent predictions. 
    `alpha` must be a value in the range (0, 1).

    A higher `alpha` (close to 1) means recent predictions are more heavily 
    weighted, whereas a lower `alpha` (closer to 0) means older predictions 
    are treated more equally.

    Example:
    >>> alpha = 0.95  # Recent predictions are given higher weight.
    >>> alpha = 0.5   # All predictions are treated more equally.
"""

_shared_docs[
    "sample_weight"
] = """sample_weight : array-like of shape (n_samples,), default=None
    Sample weights for computing a weighted accuracy.

    The `sample_weight` parameter allows the user to assign individual weights 
    to each sample in the dataset, which will be taken into account when 
    computing the accuracy or other metrics. This is particularly useful when 
    some samples should have more importance than others. 

    If provided, `sample_weight` is combined with time weights (if any) 
    to compute a weighted accuracy or other metrics. The values in `sample_weight` 
    should correspond to the samples in `y_true` and `y_pred`.

    Example:
    >>> sample_weight = [1, 1.5, 1, 1.2]  # Sample weights for each sample.
    >>> sample_weight = [0.8, 1.0, 1.2]   # Different weight for each sample.
"""


_shared_docs[
    "threshold"
] = """threshold : float, default=%s
    Threshold value for converting probabilities to binary labels 
    in binary or multilabel classification tasks.

    In binary classification or multilabel classification, classifiers 
    often output a probability score for each class. The `threshold` 
    parameter determines the cutoff point for converting these probabilities 
    into binary labels (0 or 1). If the predicted probability for a class 
    exceeds the given `threshold`, the label is assigned to that class (i.e., 
    it is classified as 1). Otherwise, the label is assigned to the alternative 
    class (i.e., 0).

    For example, in a binary classification task where the model outputs 
    probabilities, if the `threshold` is set to `{value}`, any prediction with 
    a probability greater than or equal to `0.5` is classified as class 1, 
    while predictions below `{value}` are classified as class 0.

    If `y_pred` contains probabilities for multiple classes (in multilabel 
    classification), the same logic applies for each class independently.

    Example:
    >>> threshold = 0.7  # Convert probabilities greater than 0.7 to class 1.
    >>> y_pred = [0.4, 0.8, 0.6, 0.2]  # Example predicted probabilities.
    >>> labels = [1 if p > threshold else 0 for p in y_pred]
    >>> print(labels)
    [0, 1, 1, 0]  # Labels are assigned based on threshold of 0.7.
"""

_shared_docs[
    "strategy"
] = """strategy : str, optional, default='%s'
    Computation strategy used for multiclass classification. Can be one of 
    two strategies: ``'ovr'`` (one-vs-rest) or ``'ovo'`` (one-vs-one).

    The `strategy` parameter defines how the classifier handles multiclass or 
    multilabel classification tasks. 

    - **'ovr'** (One-vs-Rest): In this strategy, the classifier compares 
      each class individually against all other classes collectively. For each 
      class, the classifier trains a binary classifier that distinguishes that 
      class from the rest. This is the default strategy and is commonly used 
      in multiclass classification problems.

    - **'ovo'** (One-vs-One): In this strategy, a binary classifier is trained 
      for every pair of classes. This approach can be computationally expensive 
      when there are many classes but might offer better performance in some 
      situations, as it evaluates all possible class pairings.

    Example:
    >>> strategy = 'ovo'  # One-vs-one strategy for multiclass classification.
    >>> strategy = 'ovr'  # One-vs-rest strategy for multiclass classification.
    >>> # 'ovo' will train a separate binary classifier for each pair of classes.
"""


_shared_docs[
    "epsilon"
] = """epsilon : float, optional, default=1e-8
    A small constant added to the denominator to prevent division by 
    zero. This parameter helps maintain numerical stability, especially 
    when dealing with very small numbers in computations that might lead 
    to division by zero errors. 

    In machine learning tasks, especially when calculating metrics like 
    log-likelihood or probabilities, small values are often involved in 
    the computation. The `epsilon` value ensures that these operations 
    do not result in infinite values or errors caused by dividing by zero. 
    The default value is typically `1e-8`, but users can specify their 
    own value.

    Additionally, if the `epsilon` value is set to ``'auto'``, the system 
    will automatically select a suitable epsilon based on the input data 
    or computation method. This ensures that numerical stability is 
    preserved without the need for manual tuning.

    Example:
    >>> epsilon = 1e-6  # A small value to improve numerical stability.
    >>> epsilon = 'auto'  # Automatically selected epsilon based on the input.
"""

_shared_docs[
    "multioutput"
] = """multioutput : str, optional, default='uniform_average'
    Determines how to return the output: ``'uniform_average'`` or 
    ``'raw_values'``. 

    - **'uniform_average'**: This option computes the average of the 
      metrics across all classes, treating each class equally. This is useful 
      when you want an overall average performance score, ignoring individual 
      class imbalances.

    - **'raw_values'**: This option returns the metric for each individual 
      class. This is helpful when you want to analyze the performance of each 
      class separately, especially in multiclass or multilabel classification.

    By using this parameter, you can control whether you get a summary of 
    the metrics across all classes or whether you want detailed metrics for 
    each class separately.

    Example:
    >>> multioutput = 'uniform_average'  # Average metrics across all classes.
    >>> multioutput = 'raw_values'  # Get separate metrics for each class.
"""


_shared_docs[
    "detailed_output"
] = """detailed_output : bool, optional, default=False
    If ``True``, returns a detailed output including individual sensitivity 
    and specificity values for each class or class pair. This is particularly 
    useful for detailed statistical analysis and diagnostics, allowing you 
    to assess the performance of the classifier at a granular level.

    When ``detailed_output`` is enabled, you can inspect the performance 
    for each class separately, including metrics like True Positive Rate, 
    False Positive Rate, and other class-specific statistics. This can help 
    identify if the model performs unevenly across different classes, which 
    is crucial for multiclass or multilabel classification tasks.

    Example:
    >>> detailed_output = True  # Return individual class metrics for analysis.
"""

_shared_docs[
    "zero_division"
] = """zero_division : str, optional, default='warn'
    Defines how to handle division by zero errors during metric calculations: 
    - ``'warn'``: Issues a warning when division by zero occurs, but allows 
      the computation to proceed.
    - ``'ignore'``: Suppresses division by zero warnings and proceeds 
      with the computation. In cases where division by zero occurs, 
      it may return infinity or a default value (depending on the operation).
    - ``'raise'``: Throws an error if division by zero is encountered, 
      halting the computation.

    This parameter gives you control over how to deal with potential issues 
    during metric calculations, especially in cases where numerical instability 
    could arise, like when a sample has no positive labels.

    Example:
    >>> zero_division = 'ignore'  # Ignore division by zero warnings.
    >>> zero_division = 'warn'  # Warn on division by zero, but continue.
    >>> zero_division = 'raise'  # Raise an error on division by zero.
"""

_shared_docs[
    "nan_policy"
] = """nan_policy : str, {'omit', 'propagate', 'raise'}, optional, default='%s'
    Defines how to handle NaN (Not a Number) values in the input arrays
    (`y_true` or `y_pred`):
    
    - ``'omit'``: Ignores any NaN values in the input arrays (`y_true` or
      `y_pred`). This option is useful when you want to exclude samples 
      with missing or invalid data from the metric calculation, effectively 
      removing them from the analysis. If this option is chosen, NaN values 
      are treated as non-existent, and the metric is computed using only the 
      valid samples. It is a common choice in cases where the data set has 
      sparse missing values and you do not want these missing values to affect 
      the result.
      
    - ``'propagate'``: Leaves NaN values in the input data unchanged. This 
      option allows NaNs to propagate through the metric calculation. When 
      this option is selected, any NaN values encountered during the computation 
      process will result in the entire metric (or output) being set to NaN. 
      This is useful when you want to track the occurrence of NaNs or understand 
      how their presence affects the metric. It can be helpful when debugging 
      models or when NaN values themselves are of interest in the analysis.
      
    - ``'raise'``: Raises an error if any NaN values are found in the input 
      arrays. This option is ideal for scenarios where you want to ensure that 
      NaN values do not go unnoticed and potentially disrupt the calculation. 
      Selecting this option enforces data integrity, ensuring that the analysis 
      will only proceed if all input values are valid and non-missing. If a NaN 
      value is encountered, it raises an exception (typically a `ValueError`), 
      allowing you to catch and handle such cases immediately.

    This parameter is especially useful in situations where missing or 
    invalid data is a concern. Depending on how you want to handle incomplete 
    data, you can choose one of the options that best suits your needs.

    Example:
    >>> nan_policy = 'omit'  # Ignore NaNs in `y_true` or `y_pred` and 
    >>> nan_policy = 'propagate'  # Let NaNs propagate; if any NaN is 
    >>> nan_policy = 'raise'  # Raise an error if NaNs are found in the 
"""

_shared_docs [ 
    "activation_transform"
]="""
Apply the activation function element-wise to the input data.

The {fmt} applies the specified activation function to each element 
of the input data array, transforming the values accordingly.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    The input data to transform. Should be an array or matrix with 
    shape (n_samples, n_features), where `n_samples` is the number of 
    samples, and `n_features` is the number of features for each sample.

Returns
-------
X_transformed : array-like, shape (n_samples, n_features)
    The transformed data, with the same shape as the input `X`. The 
    activation function is applied element-wise to each feature of 
    the input data.

Notes
-----
The input data `X` must be numeric (e.g., `int`, `float`). If `X` 
contains non-numeric data types, the transformation may fail or 
produce incorrect results. Additionally, the behavior of the function 
depends on the specific activation function being applied.

Examples
--------
>>> import numpy as np 
>>> from gofast.transformers.activations import {fmt}
>>> X = np.array([[1, 2], [3, 4]])
>>> transformer = {fmt}(
...    scale=1.0, 
...    shift=0.0, 
...    precision=1e-6, 
...    verbose=1
...    ) 
>>> transformer.fit(X)
>>> transformer.transform(X)
array([[0. ..., 0. ...],
       [0. ..., 0. ...]])
"""

_shared_docs [ 
    "gen_neg_sample_strategy"
]="""
strategy : str, optional
    Defines the sampling approach for generating
    negative examples. Available strategies include:

    - ``'landslide'`` or ``'event``:
        Negative samples are generated spatially around
        each known landslide event within a circular
        buffer defined by `buffer_km`. This assumes that
        nearby regions may share similar environmental
        conditions but did not experience a landslide.

        ✅ Pros:
            - Spatially consistent with real events
            - Easy to implement
        ⚠️ Cons:
            - Risk of overlapping with actual landslide
              zones
            - May include false negatives
        📌 Recommended:
            - When landslide events are georeferenced but
              no gauge data is available

    - ``'gauge'``:
        Negative samples are generated around
        meteorological rain gauges. Useful when
        precipitation data is the key driver and negative
        points should reflect areas under monitoring, but
        with no landslide occurrence.

        ✅ Pros:
            - Leverages real measurement locations
            - Works well with rainfall‑based models
        ⚠️ Cons:
            - Not all gauges may represent terrain
              diversity
        📌 Recommended:
            - When rainfall is the main driver and gauge
              locations are reliable

    - ``'random_global'``:
        Negative samples are distributed randomly across
        the entire study region (not tied to events or
        gauges). This provides broad coverage but may
        include unrealistic or low‑risk zones, depending
        on the study area.

        ✅ Pros:
            - Covers full spatial variability
            - Simple and fast
        ⚠️ Cons:
            - May include unrealistic zones (e.g., flat
              deserts, urban areas)
            - Poor environmental control
        📌 Recommended:
            - For exploratory models or when spatial
              context is less critical

    - ``'temporal_shift'``:
        Uses the same location as a positive landslide
        sample, but at a different (non‑disaster) time.
        Ideal when working with temporal data (e.g.,
        rainfall), as it controls for geography while
        testing different temporal conditions.

        ✅ Pros:
            - Controls for geography
            - Focuses on temporal variation (e.g., rainfall)
        ⚠️ Cons:
            - Requires time‑series data
            - May still be close to actual disaster dates
        📌 Recommended:
            - For temporal models or rainfall‑driven
              landslide prediction

    - ``'clustered_negatives'``:
        Applies clustering to the positive samples
        (e.g., KMeans) and generates negatives around the
        resulting cluster centroids. Balances realism and
        variability by shifting away from exact event
        locations.

        ✅ Pros:
            - Reduces overfitting to exact event locations
            - Balances spatial diversity
        ⚠️ Cons:
            - Needs tuning for clustering
            - May miss sparse or isolated regions
        📌 Recommended:
            - For datasets with highly clustered landslides

    - ``'environmental_similarity'``:
        Samples are drawn from regions with environmental
        attributes (e.g., slope, soil type, geology)
        similar to those of the positive class but where
        no landslides have been recorded. Provides
        difficult negative examples and improves model
        generalization.

        ✅ Pros:
            - Generates “hard negatives”
            - Excellent for feature‑sensitive models
        ⚠️ Cons:
            - Computationally expensive
            - Requires complete feature data for all regions
        📌 Recommended:
            - When building highly accurate or
              generalizable ML models

    - ``'elevation_based'``:
        Generates negative samples in areas with low
        elevation or slope, where landslides are
        geologically less likely. This strategy ensures a
        strong negative class with minimal false
        negatives.

        ✅ Pros:
            - Produces clearly negative samples
            - Low chance of false negatives
        ⚠️ Cons:
            - Poor variability
            - Could skew the classifier
        📌 Recommended:
            - When needing a clear separation between
              classes (e.g., binary baseline)

    - ``'hybrid'``:
        Combines two or more of the above strategies
        (e.g., 50 % `landslide` + 50 % `gauge`) to build a
        more diverse and robust negative sample set. Helps
        reduce sampling bias and improve classifier
        performance.

        ✅ Pros:
            - Rich, diverse sample set
            - Reduces sampling bias
        ⚠️ Cons:
            - More complex logic
            - Possible duplicate locations
        📌 Recommended:
            - When you want a robust and general dataset          
"""

def filter_docs(keys, input_dict=None):
    """
    Filters a dictionary to include only the key-value pairs where 
    the key is present in the specified list of keys. By default, 
    filters from the global `_shared_docs` dictionary.

    Parameters
    ----------
    keys : list of str
        A list of keys to keep in the resulting filtered dictionary.

    input_dict : dict, optional, default=_shared_docs
        The dictionary to be filtered. If not provided, uses the global 
        `_shared_docs` dictionary.

    Returns
    -------
    dict
        A new dictionary containing only the key-value pairs where the 
        key is present in the specified `keys` list.

    Examples
    --------
    >>> _shared_docs = {
    >>>     'y_true': [1, 2, 3],
    >>>     'y_pred': [1, 2, 3],
    >>>     'y_t': [1, 2, 3]
    >>> }
    >>> filtered_dict = filter_dict_by_keys(['y_true', 'y_pred'])
    >>> print(filtered_dict)
    {'y_true': [1, 2, 3], 'y_pred': [1, 2, 3]}

    Notes
    -----
    This function returns a new dictionary with only the specified keys
    and their corresponding values. If a key is not found in the original 
    dictionary, it is ignored.
    """
    input_dict = input_dict or _shared_docs  # Default to _shared_docs if None
    return dict(filter(lambda item: item[0] in keys, input_dict.items()))

# doc are derived from pandas._decorators module.  
# module https://pandas.org/users/license.html

def doc(*docstrings: str | Callable, **params) -> Callable[[callable], callable]:
    """
    A decorator take docstring templates, concatenate them and perform string
    substitution on it.

    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : str or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: callable) -> callable:
        # collecting docstring and docstring templates
        docstring_components: list[str | Callable] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if hasattr(docstring, "_docstring_components"):
                # error: Item "str" of "Union[str, Callable[..., Any]]" has no attribute
                # "_docstring_components"
                # error: Item "function" of "Union[str, Callable[..., Any]]" has no
                # attribute "_docstring_components"
                docstring_components.extend(
                    docstring._docstring_components  # type: ignore[union-attr]
                )
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        # formatting templates and concatenating docstring
        decorated.__doc__ = "".join(
            [
                component.format(**params)
                if isinstance(component, str)
                else dedent(component.__doc__ or "")
                for component in docstring_components
            ]
        )

        # error: "F" has no attribute "_docstring_components"
        decorated._docstring_components = (  # type: ignore[attr-defined]
            docstring_components
        )
        return decorated

    return decorator
