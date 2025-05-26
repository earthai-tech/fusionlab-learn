# -*- coding: utf-8 -*-

#   License: BSD-3-Clause
#   Author: LKouadio Laurent <etanoyau@gmail.com>
#
#   Source: Adapted from earthai-tech/gofast (https://github.com/earthai-tech/gofast)
#   This module is included in the FusionLab package, with modifications
#   to fit FusionLab’s API and documentation conventions.

"""
Provides core components and utilities for generating standardized docstrings
across the gofast API, enhancing consistency and readability in documentation.

Adapted for FusionLab from the original gofast implementation.
"""

from __future__ import annotations

import re
from textwrap import dedent 
from typing import Callable 


__all__ = [
    '_core_params',
    'refglossary',
    '_core_docs',
    '_shared_nn_params',
    '_shared_docs', 
    'DocstringComponents',
    'filter_docs', 
    'doc'
]

class DocstringComponents:
    """
    A class for managing and cleaning docstring components for classes, methods,
    or functions. It provides structured access to raw docstrings by parsing 
    them from a dictionary, optionally stripping outer whitespace, and allowing
    dot access to the components.

    This class is typically used to standardize, clean, and manage the 
    docstrings for different components of a codebase (such as methods or classes),
    particularly when docstrings contain multiple components that need to be 
    extracted, cleaned, and accessed easily.

    Parameters
    ----------
    comp_dict : dict
        A dictionary where the keys are component names and the values are 
        the raw docstring contents. The dictionary may contain entries such as 
        "description", "parameters", "returns", etc.

    strip_whitespace : bool, optional, default=True
        If True, it will remove leading and trailing whitespace from each
        entry in the `comp_dict`. If False, the whitespace will be retained.

    Attributes
    ----------
    entries : dict
        A dictionary containing the cleaned or raw docstring components after 
        parsing, depending on the `strip_whitespace` flag. These components 
        are accessible via dot notation.

    Methods
    -------
    __getattr__(attr)
        Provides dot access to the components in `self.entries`. If the requested
        attribute exists in `self.entries`, it is returned. Otherwise, it attempts
        to look for the attribute normally or raise an error if not found.

    from_nested_components(cls, **kwargs)
        A class method that allows combining multiple sub-sets of docstring
        components into a single `DocstringComponents` instance.

    Examples
    --------
    # Example 1: Creating a DocstringComponents object with basic docstrings
    doc_dict = {
        "description": "This function adds two numbers.",
        "parameters": "a : int\n    First number.\nb : int\n    Second number.",
        "returns": "int\n    The sum of a and b."
    }

    doc_comp = DocstringComponents(doc_dict)
    print(doc_comp.description)
    # Output: This function adds two numbers.

    # Example 2: Using `from_nested_components` to add multiple sub-sets
    sub_dict_1 = {
        "description": "This function multiplies two numbers.",
        "parameters": "a : int\n    First number.\nb : int\n    Second number.",
        "returns": "int\n    The product of a and b."
    }
    sub_dict_2 = {
        "example": "example_func(2, 3) # Returns 6"
    }

    doc_comp = DocstringComponents.from_nested_components(sub_dict_1, sub_dict_2)
    print(doc_comp.example)
    # Output: example_func(2, 3) # Returns 6
    """

    regexp = re.compile(r"\n((\n|.)+)\n\s*", re.MULTILINE)

    def __init__(self, comp_dict, strip_whitespace=True):
        """Read entries from a dict, optionally stripping outer whitespace."""
        if strip_whitespace:
            entries = {}
            for key, val in comp_dict.items():
                m = re.match(self.regexp, val)
                if m is None:
                    entries[key] = val
                else:
                    entries[key] = m.group(1)
        else:
            entries = comp_dict.copy()

        self.entries = entries

    def __getattr__(self, attr):
        """Provide dot access to entries for clean raw docstrings."""
        if attr in self.entries:
            return self.entries[attr]
        else:
            try:
                return self.__getattribute__(attr)
            except AttributeError as err:
                # If Python is run with -OO, it will strip docstrings and our lookup
                # from self.entries will fail. We check for __debug__, which is actually
                # set to False by -O (it is True for normal execution).
                # But we only want to see an error when building the docs;
                # not something users should see, so this slight inconsistency is fine.
                if __debug__:
                    raise err
                else:
                    pass

    @classmethod
    def from_nested_components(cls, **kwargs):
        """Add multiple sub-sets of components."""
        return cls(kwargs, strip_whitespace=False)

def doc(
    *docstrings: str | Callable, 
    **params
    ) -> Callable[[callable], callable]:
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

# ------------------------core params ------------------------------------------

_core_params= dict ( 
    data ="""
data: str, filepath_or_buffer, or :class:`pandas.core.DataFrame`
    Data source, which can be a path-like object, a DataFrame, or a file-like object.
    - For path-like objects, data is read, asserted, and validated. Accepts 
    any valid string path, including URLs. Supported URL schemes: http, ftp, 
    s3, gs, and file. For file URLs, a host is expected (e.g., 'file://localhost/path/to/table.csv'). 
    - os.PathLike objects are also accepted.
    - File-like objects should have a `read()` method (
        e.g., opened via the `open` function or `StringIO`).
    When a path-like object is provided, the data is loaded and validated. 
    This flexibility allows for various data sources, including local files or 
    files hosted on remote servers.

    """, 
    X = """
X: ndarray of shape (M, N), where M = m-samples and N = n-features
    Training data; represents observed data at both training and prediction 
    times, used as independent variables in learning. The uppercase notation 
    signifies that it typically represents a matrix. In a matrix form, each 
    sample is represented by a feature vector. Alternatively, X may not be a 
    matrix and could require a feature extractor or a pairwise metric for 
    transformation. It's critical to ensure data consistency and compatibility 
    with the chosen learning model.
    """,
    y = """
y: array-like of shape (m,), where M = m-samples
    Training target; signifies the dependent variable in learning, observed 
    during training but unavailable at prediction time. The target is often 
    the main focus of prediction in supervised learning models. Ensuring the 
    correct alignment and representation of target data is crucial for effective 
    model training.
    """,
    Xt = """
Xt: ndarray, shape (M, N), where M = m-samples and N = n-features
    Test set; denotes data observed during testing and prediction, used as 
    independent variables in learning. Like X, Xt is typically a matrix where 
    each sample corresponds to a feature vector. The consistency between the 
    training set (X) and the test set (Xt) in terms of feature representation 
    and preprocessing is essential for accurate model evaluation.
    """,
    yt = """
yt: array-like, shape (M,), where M = m-samples
    Test target; represents the dependent variable in learning, akin to 'y' 
    but for the testing phase. While yt is observed during training, it is used
    to evaluate the performance of predictive models. The test target helps 
    in assessing the generalization capabilities of the model to unseen data.
    """,
    target_name = """
target_name: str
    Target name or label used in supervised learning. It serves as the reference name 
    for the target variable (`y`) or label. Accurate identification of `target_name` is 
    crucial for model training and interpretation, especially in datasets with multiple 
    potential targets.
""",

   z = """
z: array-like 1D or pandas.Series
    Represents depth values in a 1D array or pandas series. Multi-dimensional arrays 
    are not accepted. If `z` is provided as a DataFrame and `zname` is unspecified, 
    an error is raised. In such cases, `zname` is necessary for extracting the depth 
    column from the DataFrame.
""",
    zname = """
zname: str or int
    Specifies the column name or index for depth values within a DataFrame. If an 
    integer is provided, it is interpreted as the column index for depth values. 
    The integer value should be within the DataFrame's column range. `zname` is 
    essential when the depth information is part of a larger DataFrame.
""",
    kname = """
kname: str or int
    Identifies the column name or index for permeability coefficient ('K') within a 
    DataFrame. An integer value represents the column index for 'K'. It must be within 
    the DataFrame's column range. `kname` is required when permeability data is 
    integrated into a DataFrame, ensuring correct retrieval and processing of 'K' values.
""",
   k = """
k: array-like 1D or pandas.Series
    Array or series containing permeability coefficient ('K') values. Multi-dimensional 
    arrays are not supported. If `K` is provided as a DataFrame without specifying 
    `kname`, an error is raised. `kname` is used to extract 'K' values from the DataFrame 
    and overwrite the original `K` input.
""",
    target = """
target: Array-like or pandas.Series
    The dependent variable in supervised (and semi-supervised) learning, usually 
    denoted as `y` in an estimator's fit method. Also known as the dependent variable, 
    outcome variable, response variable, ground truth, or label. Scikit-learn handles 
    targets with minimal structure: a class from a finite set, a finite real-valued 
    number, multiple classes, or multiple numbers. In this library, `target` is 
    conceptualized as a pandas Series with `target_name` as its name, combining the 
    identifier and the variable `y`.
    Refer to Scikit-learn's documentation on target types for more details:
    [Scikit-learn Target Types](https://scikit-learn.org/stable/glossary.html#glossary-target-types).
""",
    model="""
model: callable, always as a function,    
    A model estimator. An object which manages the estimation and decoding 
    of a model. The model is estimated as a deterministic function of:
        * parameters provided in object construction or with set_params;
        * the global numpy.random random state if the estimator’s random_state 
            parameter is set to None; and
        * any data or sample properties passed to the most recent call to fit, 
            fit_transform or fit_predict, or data similarly passed in a sequence 
            of calls to partial_fit.
    The estimated model is stored in public and private attributes on the 
    estimator instance, facilitating decoding through prediction and 
    transformation methods.
    Estimators must provide a fit method, and should provide `set_params` and 
    `get_params`, although these are usually provided by inheritance from 
    `base.BaseEstimator`.
    The core functionality of some estimators may also be available as a ``function``.
    """,
    clf="""
clf :callable, always as a function, classifier estimator
    A supervised (or semi-supervised) predictor with a finite set of discrete 
    possible output values. A classifier supports modeling some of binary, 
    multiclass, multilabel, or multiclass multioutput targets. Within scikit-learn, 
    all classifiers support multi-class classification, defaulting to using a 
    one-vs-rest strategy over the binary classification problem.
    Classifiers must store a classes_ attribute after fitting, and usually 
    inherit from base.ClassifierMixin, which sets their _estimator_type attribute.
    A classifier can be distinguished from other estimators with is_classifier.
    It must implement::
        * fit
        * predict
        * score
    It may also be appropriate to implement decision_function, predict_proba 
    and predict_log_proba.    
    """,
    reg="""
reg: callable, always as a function
    A regression estimator; Estimators must provide a fit method, and should 
    provide `set_params` and 
    `get_params`, although these are usually provided by inheritance from 
    `base.BaseEstimator`. The estimated model is stored in public and private 
    attributes on the estimator instance, facilitating decoding through prediction 
    and transformation methods.
    The core functionality of some estimators may also be available as a
    ``function``.
    """,
    cv="""
cv: float,    
    A cross validation splitting strategy. It used in cross-validation based 
    routines. cv is also available in estimators such as multioutput. 
    ClassifierChain or calibration.CalibratedClassifierCV which use the 
    predictions of one estimator as training data for another, to not overfit 
    the training supervision.
    Possible inputs for cv are usually::
        * An integer, specifying the number of folds in K-fold cross validation. 
            K-fold will be stratified over classes if the estimator is a classifier
            (determined by base.is_classifier) and the targets may represent a 
            binary or multiclass (but not multioutput) classification problem 
            (determined by utils.multiclass.type_of_target).
        * A cross-validation splitter instance. Refer to the User Guide for 
            splitters available within `Scikit-learn`_
        * An iterable yielding train/test splits.
    With some exceptions (especially where not using cross validation at all 
                          is an option), the default is ``4-fold``.
    .. _Scikit-learn: https://scikit-learn.org/stable/glossary.html#glossary
    """,
    scoring="""
scoring: str, callable
    Specifies the score function to be maximized (usually by :ref:`cross
    validation <cross_validation>`), or -- in some cases -- multiple score
    functions to be reported. The score function can be a string accepted
    by :func:`sklearn.metrics.get_scorer` or a callable :term:`scorer`, not to 
    be confused with an :term:`evaluation metric`, as the latter have a more
    diverse API.  ``scoring`` may also be set to None, in which case the
    estimator's :term:`score` method is used.  See `slearn.scoring_parameter`
    in the `Scikit-learn`_ User Guide.
    """, 
    random_state="""
random_state : int, RandomState instance or None, default=None
    Controls the shuffling applied to the data before applying the split.
    Pass an int for reproducible output across multiple function calls..    
    """,
    test_size="""
test_size : float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the proportion
    of the dataset to include in the test split. If int, represents the
    absolute number of test samples. If None, the value is set to the
    complement of the train size. If ``train_size`` is also None, it will
    be set to 0.25.    
    """, 
    n_jobs="""
n_jobs: int, 
    is used to specify how many concurrent processes or threads should be 
    used for routines that are parallelized with joblib. It specifies the maximum 
    number of concurrently running workers. If 1 is given, no joblib parallelism 
    is used at all, which is useful for debugging. If set to -1, all CPUs are 
    used. For instance::
        * `n_jobs` below -1, (n_cpus + 1 + n_jobs) are used. 
        
        * `n_jobs`=-2, all CPUs but one are used. 
        * `n_jobs` is None by default, which means unset; it will generally be 
            interpreted as n_jobs=1 unless the current joblib.Parallel backend 
            context specifies otherwise.

    Note that even if n_jobs=1, low-level parallelism (via Numpy and OpenMP) 
    might be used in some configuration.  
    """,
    verbose="""
verbose: int, `default` is ``0``    
    Control the level of verbosity. Higher value lead to more messages. 
    """  
) 

_core_docs = dict(
    params=DocstringComponents(_core_params),
)


refglossary =type ('refglossary', (), dict (
    __doc__="""\

.. _GeekforGeeks: https://www.geeksforgeeks.org/style-plots-using-matplotlib/#:~:text=Matplotlib%20is%20the%20most%20popular,without%20using%20any%20other%20GUIs

.. _IUPAC nommenclature: https://en.wikipedia.org/wiki/IUPAC_nomenclature_of_inorganic_chemistry

.. _Matplotlib scatter: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.scatter.html
.. _Matplotlib plot: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html
.. _Matplotlib pyplot: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html
.. _Matplotlib figure: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.figure.html
.. _Matplotlib figsuptitle: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.suptitle.html

.. _Properties of water: https://en.wikipedia.org/wiki/Properties_of_water#Electrical_conductivity 
.. _pandas DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
.. _pandas Series: https://pandas.pydata.org/docs/reference/api/pandas.Series.html

.. _scipy.optimize.curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

"""
    ) 
)


# -------------------------- Share params docs  ------------- -----------------

_shared_nn_params = dict(
    input_dim = """
input_dim: int
    The dimensionality of each input variable. This defines the number of
    features (or the length of the feature vector) for each individual input.
    For scalar features, this value is typically ``1``. However, for more
    complex data types such as embeddings, images, or time series, the input
    dimension can be greater than 1, reflecting the number of dimensions in
    the input vectors or feature matrices. This parameter is important for
    ensuring the correct shape and consistency of input data when training
    the model.

    Example:
    - For a single scalar feature per sample, ``input_dim = 1``.
    - For a word embedding with a 300-dimensional vector for each word, 
      ``input_dim = 300``.
    - For time-series data with 10 features at each time step, 
      ``input_dim = 10``.
    """, 
    
    units = """
units: int
    The number of units in the attention layer. This parameter defines
    the dimensionality of the output space for the attention mechanism.
    It determines the size of the internal representation for each input
    and plays a significant role in model capacity and performance.
    Larger values provide more capacity to capture complex patterns,
    but may also lead to higher computational costs. The number of units
    influences how well the model can learn complex representations from
    the input data. A larger number of units can improve performance on 
    more challenging tasks, but it can also increase memory and 
    computational requirements, so tuning this parameter is important.
    """,

    num_heads = """
num_heads: int
    The number of attention heads in the multi-head attention mechanism.
    Multiple attention heads allow the model to focus on different aspects
    of the input data, capturing more complex relationships within the
    data. More heads provide better representation power but increase
    computational costs. This parameter is crucial in self-attention
    mechanisms where each head can attend to different parts of the input
    data in parallel, improving the model's ability to capture diverse
    features. For example, in natural language processing, multiple heads
    allow the model to attend to different semantic aspects of the text.
    Using more heads can increase the model's capacity to learn complex
    features, but it also requires more memory and computational power.
    """,

    dropout_rate = """
dropout_rate: float, optional
    The dropout rate applied during training to prevent overfitting.
    Dropout is a regularization technique where a fraction of input units
    is randomly set to zero at each training step to prevent the model from
    relying too heavily on any one feature. This helps improve generalization
    and can make the model more robust. Dropout is particularly effective
    in deep learning models where overfitting is a common issue. The value
    should be between 0.0 and 1.0, where a value of ``0.0`` means no dropout
    is applied and a value of ``1.0`` means that all units are dropped. 
    A typical value for ``dropout_rate`` ranges from 0.1 to 0.5.
    """,

    activation = """
activation: str, optional
    The activation function to use in the Gated Recurrent Networks (GRNs).
    The activation function defines how the model's internal representations
    are transformed before being passed to the next layer. Supported values
    include:
    
    - ``'elu'``: Exponential Linear Unit (ELU), a variant of ReLU that
      improves training performance by preventing dying neurons. ELU provides
      a smooth output for negative values, which can help mitigate the issue 
      of vanishing gradients. The mathematical formulation for ELU is:
      
      .. math:: 
          f(x) = 
          \begin{cases}
          x & \text{if } x > 0 \\
          \alpha (\exp(x) - 1) & \text{if } x \leq 0
          \end{cases}
      
      where \(\alpha\) is a constant (usually 1.0).

    - ``'relu'``: Rectified Linear Unit (ReLU), a widely used activation
      function that outputs zero for negative input and the input itself for
      positive values. It is computationally efficient and reduces the risk
      of vanishing gradients. The mathematical formulation for ReLU is:
      
      .. math:: 
          f(x) = \max(0, x)
      
      where \(x\) is the input value.

    - ``'tanh'``: Hyperbolic Tangent, which squashes the outputs into a range 
      between -1 and 1. It is useful when preserving the sign of the input
      is important, but can suffer from vanishing gradients for large inputs.
      The mathematical formulation for tanh is:
      
      .. math::
          f(x) = \frac{2}{1 + \exp(-2x)} - 1

    - ``'sigmoid'``: Sigmoid function, commonly used for binary classification
      tasks, maps outputs between 0 and 1, making it suitable for probabilistic
      outputs. The mathematical formulation for sigmoid is:
      
      .. math:: 
          f(x) = \frac{1}{1 + \exp(-x)}

    - ``'linear'``: No activation (identity function), often used in regression
      tasks where no non-linearity is needed. The output is simply the input value:
      
      .. math:: 
          f(x) = x

    The default activation function is ``'elu'``.
    """,

    use_batch_norm = """
use_batch_norm: bool, optional
    Whether to use batch normalization in the Gated Recurrent Networks (GRNs).
    Batch normalization normalizes the input to each layer, stabilizing and
    accelerating the training process. When set to ``True``, it normalizes the
    activations by scaling and shifting them to maintain a stable distribution
    during training. This technique can help mitigate issues like vanishing and
    exploding gradients, making it easier to train deep networks. Batch normalization
    also acts as a form of regularization, reducing the need for other techniques
    like dropout. By default, batch normalization is turned off (``False``).
    
    """, 
    
    hidden_units = """
hidden_units: int
    The number of hidden units in the model's layers. This parameter 
    defines the size of the hidden layers throughout the model, including 
    Gated Recurrent Networks (GRNs), Long Short-Term Memory (LSTM) layers, 
    and fully connected layers. Increasing the value of ``hidden_units`` 
    enhances the model's capacity to capture more complex relationships and 
    patterns from the data. However, it also increases computational costs 
    due to a higher number of parameters. The choice of hidden units should 
    balance model capacity and computational feasibility, depending on the 
    complexity of the problem and available resources.
    """,

quantiles = """
quantiles: list of float or None, optional
    A list of quantiles to predict for each time step. For example, 
    specifying ``[0.1, 0.5, 0.9]`` would result in the model predicting 
    the 10th, 50th, and 90th percentiles of the target variable at each 
    time step. This is useful for estimating prediction intervals and 
    capturing uncertainty in forecasting tasks. If set to ``None``, the model 
    performs point forecasting and predicts a single value (e.g., the mean 
    or most likely value) for each time step. Quantile forecasting is commonly 
    used for applications where it is important to predict not just the 
    most likely outcome, but also the range of possible outcomes.
    """
)
    
# ---------------------------------------------------------------------
# Shared docstring snippets used across FusionLab metric‑plotting
# utilities.
# ---------------------------------------------------------------------
_shared_metric_plot_params = dict(

    y_true="""
y_true : ndarray of shape (n_samples, …)
    Ground‑truth target values.  Depending on the metric a 1‑D
    array (global aggregation), a 2‑D array *(n_samples, n_outputs)*,
    or a 3‑D array *(n_samples, n_horizons, n_outputs)* may be
    expected.""",

    y_pred="""
y_pred : ndarray
    Point‑forecast predictions with the same shape semantics as
    `y_true`.  Used by deterministic metrics such as MAE or RMSE as
    well as for plotting point predictions alongside intervals.""",

    y_median="""
y_median : ndarray
    Median (50‑th quantile) of a probabilistic forecast.  The array
    must align with `y_true` along every sampled dimension.""",

    y_lower="""
y_lower : ndarray
    Lower‑bound quantile (e.g. 0.05 or 0.10) for an uncertainty
    interval.  Shape must mirror `y_true`.  Required by coverage,
    interval‑width, and WIS plots.""",

    y_upper="""
y_upper : ndarray
    Upper‑bound quantile (e.g. 0.95 or 0.90) paired with `y_lower`.
    Must share the same shape and broadcast semantics as `y_true`. """,

    y_pred_quantiles="""
y_pred_quantiles : ndarray
    Stack of predictive quantiles.  Typical shape is
    *(n_samples, n_horizons, n_quantiles)* or
    *(n_samples, n_quantiles)* for horizon‑aggregated diagnostics.""",

    quantiles="""
quantiles : 1‑D ndarray
    Numeric array of the quantile levels represented in
    `y_pred_quantiles`, sorted in ascending order
    (e.g. ``np.array([0.1, 0.5, 0.9])``).""",

    alphas="""
alphas : 1‑D ndarray
    Alpha levels *α = 2 × min(q, 1−q)* that define the nominal
    coverage *(1 − α)* of each prediction interval used in Weighted
    Interval Score (WIS) computations.""",

    metric_values="""
metric_values : float or ndarray, default=None
    Pre‑computed metric value(s).  Supply this to skip internal
    calculation and plot the given numbers directly.""",

    metric_kws="""
metric_kws : dict, default=None
    Extra keyword arguments forwarded verbatim to the underlying
    metric function (e.g. `coverage_score`).  Use this to tweak
    nan‑handling, sample‑weights, or multi‑output behaviour.""",

    kind="""
kind : {'summary_bar', 'intervals', 'reliability_diagram', ...}
    High‑level style of plot to produce.  The accepted values depend
    on the specific helper, and unsupported kinds raise
    ``ValueError``.""",

    output_idx="""
output_idx : int, optional
    Index of the target variable to visualise when the model
    predicts multiple outputs.  If *None*, the first output or an
    aggregated view is plotted, depending on the function.""",

    sample_idx="""
sample_idx : int, default=0
    Index of the time series (row) to highlight in sample‑wise
    plots (e.g. CRPS ECDF per sample).""",

    figsize="""
figsize : tuple of float, optional
    Size of the figure in inches *(width, height)*.  If omitted the
    helper chooses sensible defaults such as ``(8, 6)``.""",

    title="""
title : str, optional
    Main title for the figure.  If *None*, a context‑aware default
    is generated from the metric name and input parameters.""",

    xlabel="""
xlabel : str, optional
    Label for the x‑axis.  If *None*, a function‑specific default is
    applied.""",

    ylabel="""
ylabel : str, optional
    Label for the y‑axis.  If *None*, a context‑sensitive default is
    used (e.g. 'Coverage', 'Score').""",

    bar_color="""
bar_color : str or list of str, optional
    Bar face‑colour(s).  Accepts any Matplotlib‑recognised colour
    spec or a list for multi‑bar plots.""",

    bar_width="""
bar_width : float, default=0.8
    Relative width of bars in bar‑type plots (0 < bar_width ≤ 1).""",

    score_annotation_format="""
score_annotation_format : str, default='{:.4f}'
    Python format string used for numeric annotations.  Examples:
    ``'{:.4f}'`` → 0.1234, ``'{:.2%}'`` → 12.34 %. """,

    show_score_on_title="""
show_score_on_title : bool, default=True
    If *True*, appends the aggregated metric value to the plot
    title.""",

    show_score="""
show_score : bool, default=True
    Whether to display individual metric values (e.g. bar labels or
    legend entries) on the plot.""",

    show_grid="""
show_grid : bool, default=True
    Toggle the background grid on the plot axes.""",

    grid_props="""
grid_props : dict, optional
    Keyword arguments forwarded to ``Axes.grid`` for fine‑grained
    grid style control (linestyle, linewidth, alpha, etc.).""",

    ax="""
ax : matplotlib.axes.Axes, optional
    Existing Matplotlib axes to draw on.  If *None*, a new figure
    and axes are created internally.""",

    verbose="""
verbose : int, default=0
    Verbosity level.  0 ⇒ silent, 1 ⇒ basic info, 2+ ⇒ debug
    details printed to stdout.""",

    kwargs="""
**kwargs
    Additional keyword arguments passed directly to the underlying
    Matplotlib primitives (``plot``, ``scatter``, ``bar``,
    ``fill_between`` …) for low‑level aesthetic control."""
)
    
# --------------------------------------------------------------------------- #
# Centralised parameter‑descriptions for evaluation / radar‑style plots.
# Each entry is a reStructuredText‑ready snippet that can be injected into
# docstrings via ``.format`` – exactly the pattern used for
# `_shared_metric_plot_params`.
# --------------------------------------------------------------------------- #

_evaluation_plot_params = dict(

    forecast_df="""
forecast_df : pandas.DataFrame
    Long‑format table of predictions.  Must contain
    ``'sample_idx'`` and ``'forecast_step'`` plus the prediction,
    {segment_col}, and actual columns (for instance
    ``'{target_name}_actual'``).""",

    segment_col="""
segment_col : str
    Column whose unique values form the radar spokes
    (e.g. ``'ItemID'``, ``'Month'`` or ``'DayOfWeek'``).""",

    metric="""
metric : str or Callable, default ``'mae'``
    Metric to compute per segment.
    Accepted names: ``'mae'``, ``'mse'``, ``'rmse'``,
    ``'mape'``, ``'smape'``.
    For a custom metric pass a function ``f(y_true, y_pred) -> float``.
    When *quantiles* are supplied the median prediction is forwarded
    to that callable.""",

    target_name="""
target_name : str, default ``"target"``
    Base name used to assemble prediction / actual column names.""",

    quantiles="""
quantiles : list[float], optional
    Quantiles included in *forecast_df* (e.g. ``[0.1, 0.5, 0.9]``).
    If present and a generic metric is chosen the median
    (``0.5`` or nearest) prediction is employed as ``y_pred``.
    Omit for point forecasts.""",

    output_dim="""
output_dim : int, default ``1``
    Number of target dimensions.  A separate radar is generated
    for each dimension when ``output_dim > 1``.""",

    actual_col_pattern="""
actual_col_pattern : str, default ``"{target_name}_actual"``
    Format string for locating actual columns.
    Place‑holders: ``{target_name}``, ``{o_idx}``.""",

    pred_col_pattern_point="""
pred_col_pattern_point : str, default ``"{target_name}_pred"``
    Format string for point‑forecast columns.""",

    pred_col_pattern_quantile="""
pred_col_pattern_quantile : str, default
    ``"{target_name}_q{quantile_int}"``
    Format string for quantile columns.
    Place‑holders: ``{target_name}``, ``{o_idx}``, ``{quantile_int}``.""",

    aggregate_across_horizon="""
aggregate_across_horizon : bool, default ``True``
    If *True* the metric is computed on all time‑steps per segment.
    If *False* the caller must provide pre‑aggregated values or expect
    one metric per step (rare for radar plots).""",

    scaler="""
scaler : Any, optional
    Fitted scikit‑learn‑style transformer used to inverse‑scale data
    before metric evaluation.""",

    scaler_feature_names="""
scaler_feature_names : list[str], optional
    Full feature order that *scaler* was trained on.
    Mandatory when *scaler* is given.""",

    target_idx_in_scaler="""
target_idx_in_scaler : int, optional
    Position of *target_name* inside *scaler_feature_names*.
    Mandatory when *scaler* is given.""",

    figsize="""
figsize : tuple[float, float], default ``(8, 8)``
    Width and height of each radar chart in inches.""",

    max_segments_to_plot="""
max_segments_to_plot : int, optional
    Hard cap on the number of segments shown on one radar.
    Defaults to ``12`` – exceeding this might overcrowd the figure.""",

    verbose="""
verbose : int, default ``0``
    Controls diagnostic output.  ``0`` = silent.""",

    plot_kwargs="""
**plot_kwargs : Any
    Extra arguments forwarded to the underlying Matplotlib
    ``ax.plot`` / ``ax.fill`` calls (e.g. ``color``, ``linewidth``,
    ``alpha``).""",
)

# Common parameter docs reused by XTFTTuner / TFTTuner
_tuner_common_params = dict(

    model_name="""
model_name : str, optional
    Identifier of the model variant to tune.  Must match one of
    the names accepted by the respective tuner class.  Case is
    ignored.  Defaults to ``"xtft"`` for :class:`XTFTTuner` and
    ``"tft"`` for :class:`TFTTuner`.  Validation occurs before
    the base class initialiser is called.
""",

    param_space="""
param_space : dict, optional
    Dictionary mapping hyper‑parameter names to search options
    understood by Keras Tuner (e.g. lists, ranges, Int/Float
    distributions).  When *None* a built‑in default space is
    employed.
""",

    max_trials="""
max_trials : int, default ``10``
    Upper bound on the number of trial configurations that the
    tuner explores.  Must be a positive integer.
""",

    objective="""
objective : str, default ``'val_loss'``
    Metric name that the tuner seeks to minimise (or maximise if
    prefixed with ``'max'``).  Any Keras history key is valid.
""",

    epochs="""
epochs : int, default ``10``
    Training epochs for the *refit* phase carried out on the best
    hyper‑parameters of each batch‑size loop.
""",

    batch_sizes="""
batch_sizes : list[int], default ``[32]``
    Ensemble of batch sizes to iterate over.  A separate tuning
    run is executed for every value.
""",

    validation_split="""
validation_split : float, default ``0.2``
    Fraction of the training data reserved for validation inside
    both the search and refit stages.  Must fall in ``(0, 1)``.
""",

    tuner_dir="""
tuner_dir : str, optional
    Root directory where Keras Tuner artefacts are written
    (trial summaries, checkpoints, logs).  A path within the
    current working directory is autogenerated if omitted.
""",

    project_name="""
project_name : str, optional
    Folder name under *tuner_dir* used to isolate results of one
    tuning job.  Defaults to a slug derived from the model type
    and run description.
""",

    tuner_type="""
tuner_type : {'random', 'bayesian'}, default ``'random'``
    Search strategy.  *'random'* draws configurations uniformly;
    *'bayesian'* performs probabilistic optimisation of the
    objective.
""",

    callbacks="""
callbacks : list[keras.callbacks.Callback], optional
    Extra Keras callbacks active during both the search and refit
    phases.  When *None* a sensible :class:`EarlyStopping` is
    injected automatically.
""",

    model_builder="""
model_builder : Callable[[kt.HyperParameters], Model], optional
    Custom factory returning a compiled Keras model from a
    hyper‑parameter set.  If missing an internal builder
    covering the canonical search space is substituted.
""",

    verbose="""
verbose : int, default ``1``
    Controls console logging produced by the tuner wrapper:
    ``0`` = silent · ``1`` = high‑level · ``2`` = per‑step
    details · ``≥3`` = debug.
""",
)


#---------------------------------Share docs ----------------------------------

_shared_docs: dict[str, str] = {}

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


_shared_docs[
    'tft_params_doc'
 ]="""

Parameters
----------
dynamic_input_dim : int
    The dimensionality of each dynamic input feature. These are the 
    time-varying features (e.g., stock prices, temperature, etc.) that 
    change over time. This should be the number of features in your 
    temporal input data at each time step.
    
static_input_dim : int
    The dimensionality of each static input feature. This is the number 
    of features that do not change over time (e.g., static data such as 
    geographical coordinates, user demographics, etc.). For example, 
    if there are 2 static features (e.g., country, region), set this to 2.
    
hidden_units : int, optional
    The number of hidden units in the layers of the model. This determines 
    the size of the hidden layers in the model architecture. A larger 
    number of hidden units allows the model to capture more complex 
    relationships in the data but may also increase computational cost.
    The default is ``32``.

num_heads: int, optional
    The number of attention heads used in the multi-head attention 
    mechanism. This controls how many separate "attention" operations 
    are run in parallel. More heads typically allow the model to capture 
    more complex interactions within the input data. The default is ``4``.

dropout_rate : float, optional
    The dropout rate used during training to prevent overfitting. This 
    value controls the fraction of input units to drop during training 
    (i.e., setting it to 0.2 means 20% of input units are randomly 
    set to zero in each forward pass). The value should be between 0 and 1.
    The default is ``0.1`` i.e 10%. 

forecast_horizon: int
    The number of time steps ahead to predict. This defines how far into 
    the future the model will generate predictions. For example, if set 
    to 7, the model will predict 7 future time steps from the current 
    data point. The value must be a positive integer (e.g., 1, 7, etc.).

quantiles: list, optional
    A list of quantiles for prediction. These quantiles define the 
    uncertainty in the predictions. For example, if set to `[0.1, 0.5, 0.9]`, 
    the model will output predictions for the 10th, 50th, and 90th percentiles 
    of the forecasted distribution. If set to `None`, the model will output 
    only the mean prediction.

activation : {'elu', 'relu', 'tanh', 'sigmoid', 'linear', 'gelu'}
    The activation function used in the model. Common choices include:
    - `'relu'`: Rectified Linear Unit (recommended for deep models)
    - `'elu'`: Exponential Linear Unit
    - `'tanh'`: Hyperbolic Tangent
    - `'sigmoid'`: Sigmoid function (common for binary classification)
    - `'linear'`: Linear activation (used in regression problems)
    - `'gelu'`: Gaussian Error Linear Unit (often used in transformers)

use_batch_norm : bool, default True
    Whether to use batch normalization in the model. Batch normalization 
    helps improve training by normalizing the output of previous layers 
    and speeding up convergence. Set this to `True` to enable batch 
    normalization, or `False` to disable it.

num_lstm_layers : int
    The number of LSTM layers in the model. LSTMs are used to capture 
    long-term dependencies in the data. More LSTM layers allow the model 
    to capture more complex temporal patterns but may increase the 
    computational cost.

lstm_units : list of int, optional
    The number of units in each LSTM layer. This can be a list of integers 
    where each element corresponds to the number of units in a specific 
    LSTM layer. For example, `[64, 32]` means the model has two LSTM 
    layers with 64 and 32 units, respectively. If set to `None`, the 
    number of units will be inferred from the `hidden_units` parameter.
"""

_shared_docs[ 
   'tft_math_doc'
]="""
    Notes
    -----
    The Temporal Fusion Transformer (TFT) model combines the strengths of
    sequence-to-sequence models and attention mechanisms to handle complex
    temporal dynamics. It provides interpretability by allowing examination
    of variable importance and temporal attention weights.
    
    **Variable Selection Networks (VSNs):**
    
    VSNs select relevant variables by applying Gated Residual Networks (GRNs)
    to each variable and computing variable importance weights via a softmax
    function. This allows the model to focus on the most informative features.
    
    **Gated Residual Networks (GRNs):**
    
    GRNs allow the model to capture complex nonlinear relationships while
    controlling information flow via gating mechanisms. They consist of a
    nonlinear layer followed by gating and residual connections.
    
    **Static Enrichment Layer:**
    
    Enriches temporal features with static context, enabling the model to
    adjust temporal dynamics based on static information. This layer combines
    static embeddings with temporal representations.
    
    **Temporal Attention Layer:**
    
    Applies multi-head attention over the temporal dimension to focus on
    important time steps. This mechanism allows the model to weigh different
    time steps differently when making predictions.
    
    **Mathematical Formulation:**
    
    Let:
    
    - :math:`\mathbf{x}_{\text{static}} \in \mathbb{R}^{n_s \times d_s}` be the
      static inputs,
    - :math:`\mathbf{x}_{\text{dynamic}} \in \mathbb{R}^{T \times n_d \times d_d}`
      be the dynamic inputs,
    - :math:`n_s` and :math:`n_d` are the numbers of static and dynamic variables,
    - :math:`d_s` and :math:`d_d` are their respective input dimensions,
    - :math:`T` is the number of time steps.
    
    **Variable Selection Networks (VSNs):**
    
    For static variables:
    
    .. math::
    
        \mathbf{e}_{\text{static}} = \sum_{i=1}^{n_s} \alpha_i \cdot
        \text{GRN}(\mathbf{x}_{\text{static}, i})
    
    For dynamic variables:
    
    .. math::
    
        \mathbf{E}_{\text{dynamic}} = \sum_{j=1}^{n_d} \beta_j \cdot
        \text{GRN}(\mathbf{x}_{\text{dynamic}, :, j})
    
    where :math:`\alpha_i` and :math:`\beta_j` are variable importance weights
    computed via softmax.
    
    **LSTM Encoder:**
    
    Processes dynamic embeddings to capture sequential dependencies:
    
    .. math::
    
        \mathbf{H} = \text{LSTM}(\mathbf{E}_{\text{dynamic}})
    
    **Static Enrichment Layer:**
    
    Combines static context with temporal features:
    
    .. math::
    
        \mathbf{H}_{\text{enriched}} = \text{StaticEnrichment}(
        \mathbf{e}_{\text{static}}, \mathbf{H})
    
    **Temporal Attention Layer:**
    
    Applies attention over time steps:
    
    .. math::
    
        \mathbf{Z} = \text{TemporalAttention}(\mathbf{H}_{\text{enriched}})
    
    **Position-wise Feedforward Layer:**
    
    Refines the output:
    
    .. math::
    
        \mathbf{F} = \text{GRN}(\mathbf{Z})
    
    **Final Output:**
    
    For point forecasting:
    
    .. math::
    
        \hat{y} = \text{OutputLayer}(\mathbf{F}_{T})
    
    For quantile forecasting (if quantiles are specified):
    
    .. math::
    
        \hat{y}_q = \text{OutputLayer}_q(\mathbf{F}_{T}), \quad q \in \text{quantiles}
    
    where :math:`\mathbf{F}_{T}` is the feature vector at the last time step.
    
    Examples
    --------
    >>> from fusionlab.nn.transformers import TemporalFusionTransformer
    >>> # Define model parameters
    >>> model = TemporalFusionTransformer(
    ...     static_input_dim=1,
    ...     dynamic_input_dim=1,
    ...     hidden_units=64,
    ...     num_heads=4,
    ...     dropout_rate=0.1,
    ...     forecast_horizon=1,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     activation='relu',
    ...     use_batch_norm=True,
    ...     num_lstm_layers=2,
    ...     lstm_units=[64, 32]
    ... )
    >>> model.compile(optimizer='adam', loss='mse')
    >>> # Assume `static_inputs`, `dynamic_inputs`, and `labels` are prepared
    >>> model.fit(
    ...     [static_inputs, dynamic_inputs],
    ...     labels,
    ...     epochs=10,
    ...     batch_size=32
    ... )
    
    Notes
    -----
    When using quantile regression by specifying the ``quantiles`` parameter,
    ensure that your loss function is compatible with quantile prediction,
    such as the quantile loss function. Additionally, the model output will
    have multiple predictions per time step, corresponding to each quantile.
    
    See Also
    --------
    VariableSelectionNetwork : Selects relevant variables.
    GatedResidualNetwork : Processes inputs with gating mechanisms.
    StaticEnrichmentLayer : Enriches temporal features with static context.
    TemporalAttentionLayer : Applies attention over time steps.
    
    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). "Time-series forecasting with deep
           learning: a survey." *Philosophical Transactions of the Royal
           Society A*, 379(2194), 20200209.
    """

_shared_docs[
    'tft_notes_doc'
 ]="""
    Notes
    -----
    - The model's performance can be highly dependent on the choice of 
      hyperparameters such as `hidden_units`, `num_heads`, and `dropout_rate`. 
      Experimentation is encouraged to find the optimal configuration for your 
      specific problem.
    - If `n_features` is set to a value greater than the actual number of 
      features in the data, the model will fail to train properly.
    - A larger `forecast_horizon` results in more complex predictions and 
      higher computational cost. Make sure to set it according to the 
      forecasting needs.
    
    See Also
    --------
    - :class:`fusionlab.nn.transformers.TemporalFusionTransformer`: 
        The main class that implements the Temporal Fusion Transformers supporting
        the keras API.
    
    References
    ----------
    - Borovykh, A., et al. (2017). "Conditional Variational Autoencoder for 
      Time Series". 
    - Lim, B., & Zohdy, M. (2020). "Temporal Fusion Transformers for Time 
      Series". 
"""

_shared_docs[ 
    'xtft_params_doc'
    ]="""\
Parameters
----------
static_input_dim : int
    Dimensionality of static input features (no time dimension).  
    These features remain constant over time steps and provide
    global context or attributes related to the time series. For
    example, a store ID or geographic location. Increasing this
    dimension allows the model to utilize more contextual signals
    that do not vary with time. A larger `static_input_dim` can
    help the model specialize predictions for different entities
    or conditions and improve personalized forecasts.

dynamic_input_dim : int
    Dimensionality of dynamic input features. These features vary
    over time steps and typically include historical observations
    of the target variable, and any time-dependent covariates such
    as past sales, weather variables, or sensor readings. A higher
    `dynamic_input_dim` enables the model to incorporate more
    complex patterns from a richer set of temporal signals. These
    features help the model understand seasonality, trends, and
    evolving conditions over time.

future_input_dim : int
    Dimensionality of future known covariates. These are features
    known ahead of time for future predictions (e.g., holidays,
    promotions, scheduled events, or future weather forecasts).
    Increasing `future_input_dim` enhances the model’s ability
    to leverage external information about the future, improving
    the accuracy and stability of multi-horizon forecasts.

embed_dim : int, optional
    Dimension of feature embeddings. Default is ``32``. After
    variable transformations, inputs are projected into embeddings
    of size `embed_dim`. Larger embeddings can capture more nuanced
    relationships but may increase model complexity. A balanced
    choice prevents overfitting while ensuring the representation
    capacity is sufficient for complex patterns.

forecast_horizon : int, optional
    Number of future time steps to predict. Default is ``1``. This
    parameter specifies how many steps ahead the model provides
    forecasts. For instance, `forecast_horizon=3` means the model
    predicts values for three future periods simultaneously.
    Increasing this allows multi-step forecasting, but may
    complicate learning if too large.

quantiles : list of float or str, optional
    Quantiles to predict for probabilistic forecasting. For example,
    ``[0.1, 0.5, 0.9]`` indicates lower, median, and upper bounds.
    If set to ``'auto'``, defaults to ``[0.1, 0.5, 0.9]``. If
    `None`, the model makes deterministic predictions. Providing
    quantiles helps the model estimate prediction intervals and
    uncertainty, offering more informative and robust forecasts.

max_window_size : int, optional
    Maximum dynamic time window size. Default is ``10``. Defines
    the length of the dynamic windowing mechanism that selects
    relevant recent time steps for modeling. A larger `max_window_size`
    enables the model to consider more historical data at once,
    potentially capturing longer-term patterns, but may also
    increase computational cost.

memory_size : int, optional
    Size of the memory for memory-augmented attention. Default is
    ``100``. Introduces a fixed-size memory that the model can
    attend to, providing a global context or reference to distant
    past information. Larger `memory_size` can help the model
    recall patterns from further back in time, improving long-term
    forecasting stability.

num_heads : int, optional
    Number of attention heads. Default is ``4``. Multi-head
    attention allows the model to attend to different representation
    subspaces of the input sequence. Increasing `num_heads` can
    improve model performance by capturing various aspects of the
    data, but also raises the computational complexity and the
    number of parameters.

dropout_rate : float, optional
    Dropout rate for regularization. Default is ``0.1``. Controls
    the fraction of units dropped out randomly during training.
    Higher values can prevent overfitting but may slow convergence.
    A small to moderate `dropout_rate` (e.g. 0.1 to 0.3) is often
    a good starting point.

output_dim : int, optional
    Dimensionality of the output. Default is ``1``. Determines how
    many target variables are predicted at each forecast horizon.
    For univariate forecasting, `output_dim=1` is typical. For
    multi-variate forecasting, set a larger value to predict
    multiple targets simultaneously.

anomaly_config : dict, optional
        Configuration dictionary for anomaly detection. It may contain 
        the following keys:

        - ``'anomaly_scores'`` : array-like, optional
            Precomputed anomaly scores tensor of shape `(batch_size, forecast_horizon)`. 
            If not provided, anomaly loss will not be applied.

        - ``'anomaly_loss_weight'`` : float, optional
            Weight for the anomaly loss in the total loss computation. 
            Balances the contribution of anomaly detection against the 
            primary forecasting task. A higher value emphasizes identifying 
            and penalizing anomalies, potentially improving robustness to
            irregularities in the data, while a lower value prioritizes
            general forecasting performance.
            If not provided, anomaly loss will not be applied.

        **Behavior:**
        If `anomaly_config` is `None`, both `'anomaly_scores'` and 
        `'anomaly_loss_weight'` default to `None`, and anomaly loss is 
        disabled. This means the model will perform forecasting without 
        considering  any anomaly detection mechanisms.

        **Examples:**
        
        - **Without Anomaly Detection:**
            ```python
            model = XTFT(
                static_input_dim=10,
                dynamic_input_dim=45,
                future_covariate_dim=5,
                anomaly_config=None,
                ...
            )
            ```
        
        - **With Anomaly Detection:**
            ```python
            import tensorflow as tf

            # Define precomputed anomaly scores
            precomputed_anomaly_scores = tf.random.normal((batch_size, forecast_horizon))

            # Create anomaly_config dictionary
            anomaly_config = {{
                'anomaly_scores': precomputed_anomaly_scores,
                'anomaly_loss_weight': 1.0
            }}

            # Initialize the model with anomaly_config
            model = XTFT(
                static_input_dim=10,
                dynamic_input_dim=45,
                future_input_dim=5,
                anomaly_config=anomaly_config,
                ...
            )
            ```

anomaly_loss_weight : float, optional
    Weight of the anomaly loss term. Default is ``1.0``. 

attention_units : int, optional
    Number of units in attention layers. Default is ``32``.
    Controls the dimensionality of internal representations in
    attention mechanisms. More `attention_units` can allow the
    model to represent more complex dependencies, but may also
    increase risk of overfitting and computation.

hidden_units : int, optional
    Number of units in hidden layers. Default is ``64``. Influences
    the capacity of various dense layers within the model, such as
    those processing static features or for residual connections.
    More units allow modeling more intricate functions, but can
    lead to overfitting if not regularized.

lstm_units : int or None, optional
    Number of units in LSTM layers. Default is ``64``. If `None`,
    LSTM layers may be disabled or replaced with another mechanism.
    Increasing `lstm_units` improves the model’s ability to capture
    temporal dependencies, but also raises computational cost and
    potential overfitting.

scales : list of int, str or None, optional
    Scales for multi-scale LSTM. If ``'auto'``, defaults are chosen
    internally. This parameter configures multiple LSTMs to operate
    at different temporal resolutions. For example, `[1, 7, 30]`
    might represent daily, weekly, and monthly scales. Multi-scale
    modeling can enhance the model’s understanding of hierarchical
    time structures and seasonalities.

multi_scale_agg : str or None, optional
    Aggregation method for multi-scale outputs. Options:
    ``'last'``, ``'average'``, ``'flatten'``, ``'auto'``. If `None`,
    no special aggregation is applied. This parameter determines
    how the multiple scales’ outputs are combined. For instance,
    `average` can produce a more stable representation by averaging
    across scales, while `flatten` preserves all scale information
    in a concatenated form.

activation : str or callable, optional
    Activation function. Default is ``'relu'``. Common choices
    include ``'tanh'``, ``'elu'``, or a custom callable. The choice
    of activation affects the model’s nonlinearity and can influence
    convergence speed and final accuracy. For complex datasets,
    experimenting with different activations may yield better
    results.

use_residuals : bool, optional
    Whether to use residual connections. Default is ``True``.
    Residuals help in stabilizing and speeding up training by
    allowing gradients to flow more easily through the model and
    mitigating vanishing gradients. They also enable deeper model
    architectures without significant performance degradation.

use_batch_norm : bool, optional
    Whether to use batch normalization. Default is ``False``.
    Batch normalization can accelerate training by normalizing
    layer inputs, reducing internal covariate shift. It often makes
    model training more stable and can improve convergence,
    especially in deeper architectures. However, it adds complexity
    and may not always be beneficial.

final_agg : str, optional
    Final aggregation of the time window. Options:
    ``'last'``, ``'average'``, ``'flatten'``. Default is ``'last'``.
    Determines how the time-windowed representations are reduced
    into a final vector before decoding into forecasts. For example,
    `last` takes the most recent time step's feature vector, while
    `average` merges information across the entire window. Choosing
    a suitable aggregation can influence forecast stability and
    sensitivity to recent or aggregate patterns.    
    
"""

_shared_docs [
    'xtft_params_doc_minimal'
]= """
    static_input_dim : int
        The dimensionality of the static input features.

    dynamic_input_dim : int
        The dimensionality of the dynamic input features.

    future_input_dim : int
        The dimensionality of the future covariate features.

    embed_dim : int, optional, default=32
        The dimensionality of embeddings used for features.

    forecast_horizon : int, optional, default=1
        The number of steps ahead for which predictions are generated.

    quantiles : Union[str, List[float], None], optional
        Quantiles for probabilistic forecasting. If None, the model 
        may output deterministic forecasts.

    max_window_size : int, optional, default=10
        The maximum window size for attention computations.

    memory_size : int, optional, default=100
        The size of the memory component used in the model.

    num_heads : int, optional, default=4
        The number of attention heads to use in the multi-head attention layer.

    dropout_rate : float, optional, default=0.1
        The dropout rate applied to reduce overfitting.

    output_dim : int, optional, default=1
        The dimensionality of the model output (e.g., univariate or multivariate).

    anomaly_config : Optional[Dict[str, Any]], optional
        Configuration for anomaly detection/loss. If not required, can be None.

    attention_units : int, optional, default=32
        The number of units in the attention layer.

    hidden_units : int, optional, default=64
        The number of units in hidden layers.

    lstm_units : int, optional, default=64
        The number of units in the LSTM layers.

    scales : Union[str, List[int], None], optional
        Scaling strategy or scale values. If None, no scaling is applied.

    multi_scale_agg : Optional[str], optional
        Multi-scale aggregation strategy for time series processing.

    activation : str, optional, default='relu'
        The activation function used in the model.

    use_residuals : bool, optional, default=True
        Whether to use residual connections in the model.

    use_batch_norm : bool, optional, default=False
        Whether to use batch normalization in the model.

    final_agg : str, optional, default='last'
        The final aggregation method used for producing outputs.
    """

_shared_docs [ 
    'xtft_key_improvements'
]=r"""

**Key Enhancements:**

- **Enhanced Variable Embeddings**: 
  Employs learned normalization and multi-modal embeddings to
  flexibly integrate static, dynamic, and future covariates. 
  This allows the model to effectively handle heterogeneous 
  inputs and exploit relevant signals from different data 
  modalities.
  
  The model applies learned normalization and multi-modal embeddings
  to unify static, dynamic, and future covariates into a common
  representation space. Let :math:`\mathbf{x}_{static}`, 
  :math:`\mathbf{X}_{dynamic}`, and :math:`\mathbf{X}_{future}` 
  denote the static, dynamic, and future input tensors:
  .. math::
     \mathbf{x}_{norm} = \frac{\mathbf{x}_{static} - \mu}
     {\sigma + \epsilon}
     
  After normalization, static and dynamic features are embedded:
  .. math::
     \mathbf{E}_{dyn} = \text{MultiModalEmbedding}
     ([\mathbf{X}_{dynamic}, \mathbf{X}_{future}])
     
  and similarly, static embeddings 
  :math:`\mathbf{E}_{static}` are obtained. This enables flexible 
  integration of heterogeneous signals.

- **Multi-Scale LSTM Mechanisms**: 
  Adopts multiple LSTMs operating at various temporal resolutions
  as controlled by `scales`. By modeling patterns at multiple
  time scales (e.g., daily, weekly, monthly), the model can 
  capture long-term trends, seasonalities, and short-term 
  fluctuations simultaneously.
  
  Multiple LSTMs process the input at different scales defined by 
  `scales`. For a set of scales 
  :math:`S = \{s_1, s_2, \ldots, s_k\}`, each scale selects 
  time steps at intervals of :math:`s_i`:
  .. math::
     \mathbf{H}_{lstm} = \text{Concat}(
     [\text{LSTM}_{s_i}(\mathbf{E}_{dyn}^{(s_i)})]_{i=1}^{k})
     
  where :math:`\mathbf{E}_{dyn}^{(s_i)}` represents 
  :math:`\mathbf{E}_{dyn}` sampled at stride :math:`s_i`. This 
  approach captures patterns at multiple temporal resolutions 
  (e.g., daily, weekly).


- **Enhanced Attention Mechanisms**: 
  Integrates hierarchical, cross, and memory-augmented attention. 
  Hierarchical attention highlights critical temporal regions,
  cross attention fuses information from diverse feature spaces,
  and memory-augmented attention references a learned memory to
  incorporate long-range dependencies beyond the immediate 
  input window.
  
  XTFT integrates hierarchical, cross, and memory-augmented attention
  layers to enrich temporal and contextual relationships.  
  Hierarchical attention:
  .. math::
     \mathbf{H}_{hier} = \text{HierarchicalAttention}
     ([\mathbf{X}_{dynamic}, \mathbf{X}_{future}])
  
  Cross attention:
  .. math::
     \mathbf{H}_{cross} = \text{CrossAttention}
     ([\mathbf{X}_{dynamic}, \mathbf{E}_{dyn}])
  
  Memory-augmented attention with memory :math:`\mathbf{M}`:
  .. math::
     \mathbf{H}_{mem} = \text{MemoryAugmentedAttention}(
     \mathbf{H}_{hier}, \mathbf{M})
     
  Together, these attentions enable the model to focus on 
  short-term critical points, fuse different feature spaces,
  and reference long-range contexts.
  

- **Dynamic Quantile Loss**: 
  Implements adaptive quantile loss to produce probabilistic
  forecasts. This enables the model to return predictive intervals
  and quantify uncertainty, offering more robust and informed 
  decision-making capabilities.
  
  For quantiles :math:`q \in \{q_1,\ldots,q_Q\}`, and errors 
  :math:`e = y_{true} - y_{pred}`, quantile loss is defined as:
  .. math::
     \mathcal{L}_{quantile}(q) = \frac{1}{N}\sum_{n=1}^{N} 
     \max(q \cdot e_n, (q-1) \cdot e_n)
     
  This yields predictive intervals rather than single-point
  estimates, facilitating uncertainty-aware decision-making.
  
- **Multi-Horizon Output Strategies**:
  Facilitates forecasting over multiple future steps at once, 
  enabling practitioners to assess future scenarios and plan 
  accordingly. This functionality supports both deterministic 
  and probabilistic forecasts.
  
  XTFT predicts multiple horizons simultaneously. If 
  `forecast_horizon = H`, the decoder produces:
  .. math::
     \mathbf{Y}_{decoder} = \text{MultiDecoder}(\mathbf{H}_{combined})
     
  resulting in a forecast:
  .. math::
     \hat{\mathbf{Y}} \in \mathbb{R}^{B \times H \times D_{out}}
  
  This allows practitioners to assess future scenarios over 
  multiple steps rather than a single forecast instant.

- **Optimization for Complex Time Series**:
  Utilizes multi-resolution attention fusion, dynamic time 
  windowing, and residual connections to handle complex and 
  noisy data distributions. Such mechanisms improve training 
  stability and convergence rates, even in challenging 
  environments.
  
  Multi-resolution attention fusion and dynamic time windowing 
  improve the model's capability to handle complex, noisy data:
  .. math::
     \mathbf{H}_{fused} = \text{MultiResolutionAttentionFusion}(
     \mathbf{H}_{combined})
  
  Along with residual connections:
  .. math::
     \mathbf{H}_{res} = \mathbf{H}_{fused} + \mathbf{H}_{combined}
  
  These mechanisms stabilize training, enhance convergence, and 
  improve performance on challenging datasets.

- **Advanced Output Mechanisms**:
  Employs quantile distribution modeling to generate richer
  uncertainty estimations, thereby enabling the model to
  provide more detailed and informative predictions than 
  single-point estimates.
  
  Quantile distribution modeling converts decoder outputs into a
  set of quantiles:
  .. math::
     \mathbf{Y}_{quantiles} = \text{QuantileDistributionModeling}(
     \mathbf{Y}_{decoder})
  
  enabling richer uncertainty estimation and more informative 
  predictions, such as lower and upper bounds for future values.

When `quantiles` are specified, XTFT delivers probabilistic 
forecasts that include lower and upper bounds, enabling better 
risk management and planning. Moreover, anomaly detection 
capabilities, governed by `anomaly_loss_weight`, allow the 
model to identify and adapt to irregularities or abrupt changes
in the data.

"""

_shared_docs [
    'xtft_key_functions'
]=r"""

Key Functions
--------------
Consider a batch of time series data. Let:

- :math:`\mathbf{x}_{static} \in \mathbb{R}^{B \times D_{static}}`
  represent the static (time-invariant) features, where
  :math:`B` is the batch size and :math:`D_{static}` is the
  dimensionality of static inputs.
  
- :math:`\mathbf{X}_{dynamic} \in \mathbb{R}^{B \times T \times D_{dynamic}}`
  represent the dynamic (time-varying) features over :math:`T` time steps.
  Here, :math:`D_{dynamic}` corresponds to the dimensionality of
  dynamic inputs (e.g., historical observations).

- :math:`\mathbf{X}_{future} \in \mathbb{R}^{B \times T \times D_{future}}`
  represent the future known covariates, also shaped by
  :math:`T` steps and :math:`D_{future}` features. These may
  include planned events or predicted conditions known ahead of time.

The model first embeds dynamic and future features via multi-modal
embeddings, producing a unified representation:
.. math::
   \mathbf{E}_{dyn} = \text{MultiModalEmbedding}\left(
   [\mathbf{X}_{dynamic}, \mathbf{X}_{future}]\right)

To capture temporal dependencies at various resolutions, multi-scale
LSTMs are applied. These can process data at different temporal scales:
.. math::
   \mathbf{H}_{lstm} = \text{MultiScaleLSTM}(\mathbf{E}_{dyn})

Multiple attention mechanisms enhance the model’s representational
capacity:

1. Hierarchical attention focuses on both short-term and long-term
   interactions between dynamic and future features:
   .. math::
      \mathbf{H}_{hier} = \text{HierarchicalAttention}\left(
      [\mathbf{X}_{dynamic}, \mathbf{X}_{future}]\right)

2. Cross attention integrates information from different modalities
   or embedding spaces, here linking original dynamic inputs and
   their embeddings:
   .. math::
      \mathbf{H}_{cross} = \text{CrossAttention}\left(
      [\mathbf{X}_{dynamic}, \mathbf{E}_{dyn}]\right)

3. Memory-augmented attention incorporates an external memory for
   referencing distant past patterns not directly present in the
   current window:
   .. math::
      \mathbf{H}_{mem} = \text{MemoryAugmentedAttention}(\mathbf{H}_{hier})

Next, static embeddings :math:`\mathbf{E}_{static}` (obtained from
processing static inputs) are combined with the outputs from LSTMs
and attention mechanisms:
.. math::
   \mathbf{H}_{combined} = \text{Concatenate}\left(
   [\mathbf{E}_{static}, \mathbf{H}_{lstm}, \mathbf{H}_{mem},
   \mathbf{H}_{cross}]\right)

The combined representation is decoded into multi-horizon forecasts:
.. math::
   \mathbf{Y}_{decoder} = \text{MultiDecoder}(\mathbf{H}_{combined})

For probabilistic forecasting, quantile distribution modeling

transforms the decoder outputs into quantile predictions:
.. math::
   \mathbf{Y}_{quantiles} = \text{QuantileDistributionModeling}\left(
   \mathbf{Y}_{decoder}\right)

The final predictions are thus:
.. math::
   \hat{\mathbf{Y}} = \mathbf{Y}_{quantiles}

The loss function incorporates both quantile loss for probabilistic
forecasting and anomaly loss for robust handling of irregularities:
.. math::
   \mathcal{L} = \mathcal{L}_{quantile} + \lambda \mathcal{L}_{anomaly}

By adjusting :math:`\lambda`, the model can balance predictive
accuracy against robustness to anomalies.

Furthermore: 
    
- Multi-modal embeddings and multi-scale LSTMs enable the model to
  represent complex temporal patterns at various resolutions.
- Attention mechanisms (hierarchical, cross, memory-augmented)
  enrich the context and allow the model to focus on relevant
  aspects of the data.
- Quantile modeling provides probabilistic forecasts, supplying
  uncertainty intervals rather than single-point predictions.
- Techniques like residual connections, normalization, and
  anomaly loss weighting improve training stability and
  model robustness.
"""

_shared_docs[ 
    'xtft_methods'
]="""
   
Methods
-------
call(inputs, training=False)
    Perform the forward pass through the model. Given a tuple
    ``(static_input, dynamic_input, future_covariate_input)``,
    it processes all features through embeddings, LSTMs, and
    attention mechanisms before producing final forecasts.
    
    - ``static_input``: 
      A tensor of shape :math:`(B, D_{static})` representing 
      the static features. These do not vary with time.
    - ``dynamic_input``: 
      A tensor of shape :math:`(B, T, D_{dynamic})` representing
      dynamic features across :math:`T` time steps. These include
      historical values and time-dependent covariates.
    - ``future_covariate_input``: 
      A tensor of shape :math:`(B, T, D_{future})` representing
      future-known features, aiding multi-horizon forecasting.

    Depending on the presence of quantiles:
    - If ``quantiles`` is not `None`: 
      The output shape is :math:`(B, H, Q, D_{out})`, where 
      :math:`H` is `forecast_horizon`, :math:`Q` is the number of
      quantiles, and :math:`D_{out}` is `output_dim`.
    - If ``quantiles`` is `None`: 
      The output shape is :math:`(B, H, D_{out})`, providing a 
      deterministic forecast for each horizon.

    Parameters
    ----------
    inputs : tuple of tf.Tensor
        Input tensors `(static_input, dynamic_input, 
        future_covariate_input)`.
    training : bool, optional
        Whether the model is in training mode (default False).
        In training mode, layers like dropout and batch norm
        behave differently.

    Returns
    -------
    tf.Tensor
        The prediction tensor. Its shape and dimensionality depend
        on the `quantiles` setting. In probabilistic scenarios,
        multiple quantiles are returned. In deterministic mode, 
        a single prediction per horizon is provided.

compute_objective_loss(y_true, y_pred, anomaly_scores)
    Compute the total loss, combining both quantile loss (if 
    `quantiles` is not `None`) and anomaly loss. Quantile loss
    measures forecasting accuracy at specified quantiles, while
    anomaly loss penalizes unusual deviations or anomalies.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth targets. Shape: :math:`(B, H, D_{out})`.
    y_pred : tf.Tensor
        Model predictions. If quantiles are present:
        :math:`(B, H, Q, D_{out})`. Otherwise:
        :math:`(B, H, D_{out})`.
    anomaly_scores : tf.Tensor
        Tensor indicating anomaly severity. Its shape typically
        matches `(B, H, D_{dynamic})` or a related dimension.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the combined loss. Lower 
        values indicate better performance, balancing accuracy
        and anomaly handling.

anomaly_loss(anomaly_scores)
    Compute the anomaly loss component. This term encourages the
    model to be robust against abnormal patterns in the data.
    Higher anomaly scores lead to higher loss, prompting the model
    to adjust predictions or representations to reduce anomalies.

    Parameters
    ----------
    anomaly_scores : tf.Tensor
        A tensor reflecting the presence and intensity of anomalies.
        Its shape often corresponds to time steps and dynamic 
        features, e.g., `(B, H, D_{dynamic})`.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the anomaly loss. Minimizing
        this term encourages the model to learn patterns that 
        mitigate anomalies and produce more stable forecasts.
"""