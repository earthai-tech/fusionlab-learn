# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Utility wrappers for integrating neural networks models with 
scikit-learn estimators.

This module provides wrapper functions to facilitate the creation and 
compatibility of Temporal Fusion Transformer (TFT) models with scikit-learn 
estimators, such as `KerasRegressor`. These wrappers enable seamless integration 
of TFT models into scikit-learn workflows, including hyperparameter tuning 
and cross-validation.
"""

from textwrap import dedent 
from numbers import Real, Integral 
from typing import Optional, List, Union
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np 

from ..api.docs import doc
from ..core.diagnose_q import validate_quantiles_in 
from ..compat.sklearn import validate_params, Interval, StrOptions
from ..utils.deps_utils import ensure_pkg 
from ..utils.validator import check_is_fitted 
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from ._nn_docs import _shared_docs

if KERAS_BACKEND:
    Optimizer=KERAS_DEPS.Optimizer
    Metric=KERAS_DEPS.Metric
    Loss=KERAS_DEPS.Loss
    Model=KERAS_DEPS.Model 
    LSTM = KERAS_DEPS.LSTM
    Dense = KERAS_DEPS.Dense
    Adam= KERAS_DEPS.Adam
    Dropout = KERAS_DEPS.Dropout 
    LayerNormalization = KERAS_DEPS.LayerNormalization 
    MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
    Input = KERAS_DEPS.Input
    Concatenate=KERAS_DEPS.Concatenate 
    
    from ._tft import TemporalFusionTransformer 
    from ._xtft import XTFT
    from ._wrappers import _scigofast_set_X_compat
    from .utils import extract_callbacks_from 
    
DEP_MSG = dependency_message('wrappers') 

__all__ = [
    'TFTWrapper', 
    'TFTRegressor', 
    'XTFTWrapper', 
    "create_tft_model",
    "create_xtft_model"
]


@doc(
tft_params=dedent(_shared_docs['tft_params_doc']),      
math_f= dedent ( 
"""
.. math::
    Y = \text{TFT}(X_{\text{static}}, X_{\text{dynamic}})

Where:
- :math:`X_{\text{static}}` represents the static input features.
- :math:`X_{\text{dynamic}}` represents the dynamic input features.
- :math:`Y` is the predicted output.
"""
),

notes=dedent( 
"""
Notes
-----
- **Data Preparation**:
    - The ``TFTRegressor`` expects the input data ``X`` to be a single 2D 
      array where the static and dynamic features are concatenated. The 
      expected shape of ``X`` is:
      
      .. math::
          X = 
          \begin{bmatrix}
          X_{\text{static}} & X_{\text{dynamic}}
          \end{bmatrix}
      
      Where:
      - :math:`X_{\text{static}}` has shape :math:`(n_{\text{samples}}, 
        \text{num_static_vars} \times \text{static_input_dim})`
      - :math:`X_{\text{dynamic}}` has shape :math:`(n_{\text{samples}}, 
        \text{forecast_horizon} \times \text{num_dynamic_vars} \times 
        \text{dynamic_input_dim})`
    
    - Ensure that the concatenation order matches the expected split in the
      ``_split_X`` method: static features first, followed by dynamic features.

- **Model Training**:
    - The ``fit`` method trains the underlying Keras model using default parameters.
      For customized training (e.g., different number of epochs, batch size),
      consider modifying the ``build_model`` method or extending the wrapper.

- **Hyperparameter Tuning**:
    - Hyperparameter tuning with ``RandomizedSearchCV`` may be computationally 
      intensive.
      Adjust ``n_iter`` and ``cv`` parameters based on available computational
      resources and the size of the parameter grid.

- **Model Limitations**:
    - The current implementation of ``TFTRegressor`` does not support multi-output
      predictions. For multi-output forecasting, additional modifications to the
      wrapper and underlying model are required.

- **Verbose Levels**:
    - The ``verbose`` parameter controls the level of debug information:
      
      - `0`: No output.
      - `1`: Minimal output (e.g., starting messages).
      - `2`: Intermediate output (e.g., model building steps).
      - `3`: Detailed output (e.g., layer-wise information).
      - `4`: Extensive debugging information.
      - `5`: Highly detailed logs for model construction.
      - `6`: Verbose output for input shapes and transformations.
      - `7`: Maximum verbosity with comprehensive debug information.
    
    - Adjust the ``verbose`` level based on the need for debugging or monitoring
      the training and prediction processes.

See Also
--------
sklearn.model_selection.RandomizedSearchCV : Randomized hyperparameter search.
sklearn.model_selection.GridSearchCV : Exhaustive grid hyperparameter search.
tensorflow.keras.Model : TensorFlow Keras Model API.
sklearn.base.BaseEstimator : Base class for all estimators in Scikit-Learn.
sklearn.base.RegressorMixin : Mixin class for all regressors in Scikit-Learn.
gofast.nn.transformers.TemporalFusionTransformer: 
    Gofast Temporal Fusion Transformer Model API.

References
----------
.. [1] Lim, B., & Zohdy, M. A. (2019). Temporal Fusion Transformers for interpretable
    multi-horizon time series forecasting. *International Journal of Forecasting*.
.. [2] McKinney, W. (2017). *Python for Data Analysis: Data Wrangling with Pandas,
    NumPy, and IPython*. O'Reilly Media.
.. [3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is All You Need.
    *Advances in Neural Information Processing Systems*, 30.

""" ), 
examples = dedent ( 
"""
Examples
--------
>>> from gofast.nn.wrappers import TFTRegressor
>>> import numpy as np
>>> from sklearn.model_selection import RandomizedSearchCV
>>> 
>>> # Sample data
>>> X_static = np.random.rand(100, 5, 10)  # 100 samples, 5 static vars, 10 dims each
>>> # 100 samples, 10 time steps, 3 dynamic vars, 15 dims each
>>> X_dynamic = np.random.rand(100, 10, 3, 15)  
>>> y = np.random.rand(100, 1)  # 100 samples, 1 target
>>> 
>>> # Concatenate static and dynamic features for Scikit-Learn compatibility
>>> X = np.concatenate([
...     X_static.reshape(100, -1), 
...     X_dynamic.reshape(100, -1)
... ], axis=1)
>>> 
>>> # Initialize the wrapper
>>> tft = TFTRegressor(
...     static_input_dim=10,
...     dynamic_input_dim=15,
...     num_static_vars=5,
...     num_dynamic_vars=3,
...     hidden_units=64,
...     num_heads=4,
...     dropout_rate=0.1,
...     forecast_horizon=1,
...     quantiles=[0.1, 0.5, 0.9],
...     activation='elu',
...     use_batch_norm=True,
...     num_lstm_layers=2,
...     lstm_units=128,
...     verbose=3
... )
>>> 
>>> # Define parameter distributions for RandomizedSearchCV
>>> param_distributions = {
...     'hidden_units': [64, 128],
...     'num_heads': [4, 8],
...     'dropout_rate': [0.1, 0.2],
...     'forecast_horizon': [1, 3],
...     'num_lstm_layers': [1, 2],
...     'lstm_units': [64, 128]
... }
>>> 
>>> # Initialize RandomizedSearchCV
>>> random_search = RandomizedSearchCV(
...     estimator=tft,
...     param_distributions=param_distributions,
...     n_iter=10,
...     cv=3,
...     verbose=2,
...     random_state=42
... )
>>> 
>>> # Fit RandomizedSearchCV
>>> random_search.fit(X=X, y=y)
>>> 
>>> # Access best parameters and estimator
>>> best_params = random_search.best_params_
>>> best_estimator = random_search.best_estimator_
>>> print(f"Best Parameters: {best_params}")
Best Parameters: {'hidden_units': 128, 'num_heads': 8, 'dropout_rate': 0.2,
                  'forecast_horizon': 3, 'num_lstm_layers': 2, 'lstm_units': 128}
>>> print(f"Best Estimator: {best_estimator}")
Best Estimator: TFTRegressor(...)
"""
    )
)
class TFTRegressor(BaseEstimator, RegressorMixin):
    """
    A Scikit-Learn Compatible Regressor Wrapper for the Temporal Fusion Transformer.

    The ``TFTRegressor`` class provides seamless integration of the Temporal Fusion
    Transformer (TFT) model with Scikit-Learn's API, enabling the use of Scikit-Learn's
    model selection tools such as ``RandomizedSearchCV`` and ``GridSearchCV`` for
    hyperparameter tuning. This wrapper adheres to Scikit-Learn's estimator
    requirements, facilitating easy training, prediction, and evaluation within
    Scikit-Learn workflows.

    {math_f}

    {tft_params}

    epochs : int, optional
        The number of epochs (full passes through the entire dataset) 
        to train the model. Increasing this value can improve the 
        model's performance, but may also lead to longer training 
        times and potential overfitting. Default is ``100``.

    batch_size : int, optional
        The number of samples per batch of computation during training. 
        Adjusting this value can influence training dynamics, 
        memory usage, and convergence behavior. Smaller batch sizes 
        offer more frequent model updates but may be slower due to less 
        efficient computation. Larger batch sizes can speed up 
        computation but might lead to less stable convergence. 
        Default is ``32``.

    verbose : int, default=1
        Controls the level of verbosity for debug information.
        
        - `0`: No output.
        - `1`: Minimal output (e.g., starting messages).
        - `2`: Intermediate output (e.g., model building steps).
        - `3`: Detailed output (e.g., layer-wise information).
        - `4`: Extensive debugging information.
        - `5`: Highly detailed logs for model construction.
        - `6`: Verbose output for input shapes and transformations.
        - `7`: Maximum verbosity with comprehensive debug information.

    Attributes
    ----------
    model_ : tensorflow.keras.Model
        The underlying TensorFlow Keras model after fitting.

    Methods
    -------
    build_model()
        Constructs and compiles the TensorFlow Keras model based on the initialized
        parameters.

    fit(X, y, **fit_params)
        Trains the TFT model on the provided data. `fit_params` accepts all 
        keras API arguments passed to fit method such as callback parameters 
        and others. 

    predict(X)
        Generates predictions using the trained TFT model.

    See Also
    --------
    sklearn.model_selection.RandomizedSearchCV : Randomized hyperparameter search.
    sklearn.model_selection.GridSearchCV : Exhaustive grid hyperparameter search.
    tensorflow.keras.Model : TensorFlow Keras Model API.
    sklearn.base.BaseEstimator : Base class for all estimators in Scikit-Learn.
    sklearn.base.RegressorMixin : Mixin class for all regressors in Scikit-Learn.

    References
    ----------
    .. [1] Lim, B., & Zohdy, M. A. (2019). Temporal Fusion Transformers for interpretable
       multi-horizon time series forecasting. *International Journal of Forecasting*.
    .. [2] McKinney, W. (2017). *Python for Data Analysis: Data Wrangling with Pandas,
       NumPy, and IPython*. O'Reilly Media.
    .. [3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is All You Need.
       *Advances in Neural Information Processing Systems*, 30.
    
    {examples} 
    
    {notes} 
    """
   
    @validate_params({
        "static_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "num_static_vars": [Interval(Integral, 1, None, closed='left')], 
        "num_dynamic_vars": [Interval(Integral, 1, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')], 
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', list,  None],
        "activation": [
            StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", 'gelu'})],
        "use_batch_norm": [bool],
        "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": [list, Interval(Integral, 1, None, closed='left'), None],
        "epochs": [Interval(Integral, 1, None, closed ='left')], 
        "batch_size": [Interval( Integral, 1, None, closed='left')]
        },
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        static_input_dim,
        dynamic_input_dim,
        # num_static_vars,
        # num_dynamic_vars,
        hidden_units,
        num_heads=4,  
        dropout_rate=0.1,
        forecast_horizon=1,
        quantiles=None,
        activation='elu',
        use_batch_norm=False,
        num_lstm_layers=1,
        lstm_units=None, 
        epochs=100,
        batch_size=32,
        verbose=1 
    ):
        self.static_input_dim    = static_input_dim
        self.dynamic_input_dim   = dynamic_input_dim
        self.hidden_units        = hidden_units
        self.num_heads           = num_heads
        self.dropout_rate        = dropout_rate
        self.forecast_horizon    = forecast_horizon
        self.quantiles           = quantiles
        self.activation          = activation
        self.use_batch_norm      = use_batch_norm
        self.num_lstm_layers     = num_lstm_layers
        self.lstm_units          = lstm_units
        self.epochs              = epochs
        self.batch_size          = batch_size
        self.verbose             = verbose

    def build_model(self):
        if self.verbose >= 5:
            print("Building the TFT model...")
        
        static_input = Input(
            shape=(self.static_input_dim, ),
            name='static_input'
        )
        if self.verbose >= 6:
            print(f"Static input shape: {static_input.shape}")
        
        dynamic_input = Input(
            shape=(self.forecast_horizon, 
                   # self.num_dynamic_vars, 
                   self.dynamic_input_dim),
            name='dynamic_input'
        )
        if self.verbose >= 6:
            print(f"Dynamic input shape: {dynamic_input.shape}")
        
        x = dynamic_input
        for layer_num in range(1, self.num_lstm_layers + 1):
            if self.verbose >= 5:
                print(f"Adding LSTM layer {layer_num}...")
            x = LSTM(
                self.lstm_units or self.hidden_units,
                return_sequences=True,
                activation=self.activation,
                name=f'lstm_{layer_num}'
            )(x)
            if self.use_batch_norm:
                if self.verbose >= 5:
                    print(f"Applying LayerNormalization after LSTM layer {layer_num}...")
                x = LayerNormalization(name=f'layer_norm_{layer_num}')(x)
            if self.verbose >= 5:
                print(f"Applying Dropout after LSTM layer {layer_num}...")
            x = Dropout(self.dropout_rate, name=f'dropout_{layer_num}')(x)
            if self.verbose >= 6:
                print(f"LSTM layer {layer_num} output shape: {x.shape}")
        
        if self.verbose >= 5:
            print("Adding Multi-Head Attention layer...")
        attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_units,
            dropout=self.dropout_rate,
            name='multi_head_attention'
        )(x, x)
        if self.verbose >= 6:
            print(f"Attention output shape: {attention.shape}")
        
        if self.verbose >= 5:
            print("Applying Dropout after Attention...")
        attention = Dropout(self.dropout_rate, name='attention_dropout')(attention)
        
        if self.use_batch_norm:
            if self.verbose >= 5:
                print("Applying LayerNormalization after Attention...")
            attention = LayerNormalization(name='attention_layer_norm')(attention)
        
        if self.verbose >= 5:
            print("Concatenating static inputs with attention output...")
        concatenated = Concatenate(axis=-1, name='concatenate')(
            [static_input, attention]
        )
        if self.verbose >= 6:
            print(f"Concatenated output shape: {concatenated.shape}")
        
        if self.verbose >= 5:
            print("Adding Dense output layer...")
        output = Dense(
            self.static_input_dim,
            activation='linear',
            name='output_dense'
        )(concatenated)
        if self.verbose >= 6:
            print(f"Output layer shape: {output.shape}")
        
        model = Model(
            inputs=[static_input, dynamic_input],
            outputs=output,
            name='TFT_Model'
        )
        
        if self.verbose >= 5:
            print("Compiling the model with Adam optimizer and MSE loss...")
        model.compile(
            optimizer=Adam(),
            loss='mse'
        )
        
        if self.verbose >= 5:
            print("Model built successfully.")
        
        return model

    def fit(self, X, y, **fit_params):
        """
        Fit the TFT model according to the given training data.
        """
        if self.verbose >= 3:
            print("Starting fit process...")
        
        # Extract callbacks from fit_params if provided
        callbacks = None
        if fit_params: 
            callbacks, fit_params = extract_callbacks_from(
                fit_params, return_fit_params=True
            )
        # update the epochs and batch_size if they are 
        # explicitly provided in fit_params. This avoid 
        # repeating the same param twice in keras fit method.
        self.epochs = fit_params.pop('epochs', self.epochs) 
        self.batch_size = fit_params.pop('batch_size', self.batch_size) 
        self.verbose= fit_params.pop('verbose', self.verbose) 
        
        if isinstance(X, (list, tuple)):
            if self.verbose >= 4:
                print("Concatenating static and dynamic inputs for compatibility...")
            try:
                X = np.concatenate(
                    [x.reshape(X[0].shape[0], -1) for x in X], axis=1
                )
                if self.verbose >= 6:
                    print(f"Concatenated X shape: {X.shape}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error concatenating inputs: {e}")
                raise ValueError("Error concatenating X inputs.") from e
        elif isinstance(X, np.ndarray):
            if self.verbose >= 6:
                print(f"Input X shape: {X.shape}")
        else:
            raise ValueError("Input X must be a list, tuple, or numpy array.")
        
        if self.verbose >= 4:
            print("Splitting X into static and dynamic components...")
        X_static, X_dynamic = self._split_X(X)
        
        if self.verbose >= 5:
            print(f"X_static shape: {X_static.shape}")
            print(f"X_dynamic shape: {X_dynamic.shape}")
        
        y = y.reshape(
            -1, 
            self.forecast_horizon, 
            self.static_input_dim
        )
        if self.verbose >= 6:
            print(f"Reshaped y shape: {y.shape}")
        
        if self.verbose >= 3:
            print("Building the model...")
        self.model_ = self.build_model()
        
        if self.verbose >= 3:
            print("Starting model training...")
        self.model_.fit(
            [X_static, X_dynamic],
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks = callbacks, 
            verbose=self.verbose >= 1  # Show progress bar if verbose >=1, 
            **fit_params, 
        )
        if self.verbose >= 3:
            print("Model training completed.")
        
        return self

    def predict(self, X):
        """
        Predict using the TFT model.
        """
        check_is_fitted(self, attributes =['model_'])
        
        if self.verbose >= 3:
            print("Starting prediction process...")
        
        if isinstance(X, (list, tuple)):
            if self.verbose >= 4:
                print("Concatenating static and dynamic inputs for compatibility...")
            try:
                X = np.concatenate(
                    [x.reshape(X[0].shape[0], -1) for x in X], axis=1
                )
                if self.verbose >= 6:
                    print(f"Concatenated X shape: {X.shape}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error concatenating inputs: {e}")
                raise ValueError("Error concatenating X inputs.") from e
        elif isinstance(X, np.ndarray):
            if self.verbose >= 6:
                print(f"Input X shape: {X.shape}")
        else:
            raise ValueError("Input X must be a list, tuple, or numpy array.")
        
        if self.verbose >= 4:
            print("Splitting X into static and dynamic components...")
        X_static, X_dynamic = self._split_X(X)
        
        if self.verbose >= 5:
            print(f"X_static shape: {X_static.shape}")
            print(f"X_dynamic shape: {X_dynamic.shape}")
        
        if self.verbose >= 3:
            print("Generating predictions...")
        predictions = self.model_.predict(
            [X_static, X_dynamic],
            batch_size=self.back_size,
            verbose=self.verbose # Show progress bar if verbose >=1
        )
        
        if self.verbose >= 3:
            print("Prediction completed.")
        
        return predictions

    def _split_X(self, X):
        """
        Split the concatenated X into static and dynamic components.
        """
        # static_size = self.num_static_vars * self.static_input_dim
        static_size = self.static_input_dim
        dynamic_size = self.forecast_horizon  * self.dynamic_input_dim
        total_size = static_size + dynamic_size
        
        if self.verbose >= 6:
            print(f"Expected total features: {total_size}, got: {X.shape[1]}")
        
        if X.shape[1] != total_size:
            message = (
                f"Expected X to have {total_size} features, "
                f"got {X.shape[1]}."
            )
            if self.verbose >= 1:
                print(message)
            raise ValueError(message)
        
        if self.verbose >= 6:
            print("Splitting X into static and dynamic parts...")
        
        X_static = X[:, :static_size]
        X_static = X_static.reshape(
            (X_static.shape[0], 
             # self.num_static_vars, 
             self.static_input_dim)
        )
        if self.verbose >= 6:
            print(f"Reshaped X_static to {X_static.shape}")
        
        X_dynamic = X[:, static_size:]
        X_dynamic = X_dynamic.reshape(
            (X_dynamic.shape[0], 
             self.forecast_horizon, 
             # self.num_dynamic_vars, 
             self.dynamic_input_dim)
        )
        if self.verbose >= 6:
            print(f"Reshaped X_dynamic to {X_dynamic.shape}")
        
        return X_static, X_dynamic

@doc(tft_params=dedent(_shared_docs['tft_params_doc']))
class TFTWrapper(BaseEstimator, RegressorMixin):
    """
    TFTWrapper: A Scikit-Learn Compatible Wrapper for the Temporal Fusion
    Transformer.

    This class provides a seamless integration of the Temporal Fusion 
    Transformer (TFT) model with Scikit-Learn's API, enabling the use of 
    Scikit-Learn's model selection tools such as `RandomizedSearchCV` and 
    `GridSearchCV` for hyperparameter tuning. The wrapper ensures compatibility
    by adhering to Scikit-Learn's estimator requirements, facilitating easy 
    training, prediction, and evaluation within Scikit-Learn workflows.
    
    {tft_params}
    
    epochs : int, optional
        The number of epochs (full passes through the entire dataset) 
        to train the model. Increasing this value can improve the 
        model's performance, but may also lead to longer training 
        times and potential overfitting. Default is ``100``.

    batch_size : int, optional
        The number of samples per batch of computation during training. 
        Adjusting this value can influence training dynamics, 
        memory usage, and convergence behavior. Smaller batch sizes 
        offer more frequent model updates but may be slower due to less 
        efficient computation. Larger batch sizes can speed up 
        computation but might lead to less stable convergence. 
        Default is ``32``.
    
    optimizer : `Union[str, tf.keras.optimizers.Optimizer]`, default=`'adam'`
        Optimizer to use for compiling the model. Can be a string identifier 
        or an instance of a Keras Optimizer.
    
    loss : `Union[str, tf.keras.losses.Loss]`, default=`'mse'`
        Loss function to use for training the model. Can be a string identifier 
        or an instance of a Keras Loss.
    
    metrics : `Optional[List[Union[str, tf.keras.metrics.Metric]]]`, default=`None`
        List of metrics to be evaluated by the model during training and testing.
        If `None`, no additional metrics are used.
        
    Attributes
    ----------
    model_ : tensorflow.keras.Model
        The underlying TensorFlow Keras model after fitting.
    
    Methods
    -------
    build_model()
        Constructs and compiles the TensorFlow Keras model based on the initialized
        parameters.
    
    fit(X, y, **fit_params)
        Trains the TFT model on the provided data. `fit_params` accepts all 
        keras API arguments passed to fit method such as callback parameters 
        and others. 
    
    predict(X)
        Generates predictions using the trained TFT model.
    
    See Also
    --------
    sklearn.model_selection.RandomizedSearchCV : Randomized hyperparameter search.
    sklearn.model_selection.GridSearchCV : Exhaustive grid hyperparameter search.
    tensorflow.keras.Model : TensorFlow Keras Model API.
    
    References
    ----------
    .. [1] Lim, B., & Zohdy, M. A. (2019). Temporal Fusion Transformers for interpretable
       multi-horizon time series forecasting. *International Journal of Forecasting*.
    .. [2] McKinney, W. (2017). *Python for Data Analysis: Data Wrangling with Pandas,
       NumPy, and IPython*. O'Reilly Media.
    .. [3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is All You Need.
       *Advances in Neural Information Processing Systems*, 30.
    
    Examples
    --------
    >>> from gofast.nn.wrappers import TFTWrapper
    >>> import numpy as np
    >>> 
    >>> # Sample data
    >>> X_static = np.random.rand(100, 5, 10)  # 100 samples, 5 static vars, 10 dims each
    >>> # 100 samples, 10 time steps, 3 dynamic vars, 15 dims each
    >>> X_dynamic = np.random.rand(100, 10, 3, 15)  
    >>> y = np.random.rand(100, 1)  # 100 samples, 1 target
    >>> 
    >>> # Initialize the wrapper
    >>> tft = TFTWrapper(
    ...     static_input_dim=10,
    ...     dynamic_input_dim=15,
    ...     num_static_vars=5,
    ...     num_dynamic_vars=3,
    ...     hidden_units=64,
    ...     num_heads=4,
    ...     dropout_rate=0.1,
    ...     forecast_horizon=1,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     activation='elu',
    ...     use_batch_norm=True,
    ...     num_lstm_layers=2,
    ...     lstm_units=128
    ... )
    >>> 
    >>> # Fit the model
    >>> tft.fit(X=[X_static, X_dynamic], y=y)
    >>> 
    >>> # Make predictions
    >>> predictions = tft.predict(X=[X_static, X_dynamic])
    >>> print(predictions.shape)
    (100, 1)
    
    Notes
    -----
    - The `TFTWrapper` assumes that the input data `X` is a list containing two
      elements: `[X_static, X_dynamic]`. Ensure that the data is preprocessed to
      match the expected shapes:
      
      - `X_static`: (n_samples, num_static_vars, static_input_dim)
      - `X_dynamic`: (n_samples, forecast_horizon, num_dynamic_vars, dynamic_input_dim)
    
    - The `fit` method trains the underlying Keras model using default parameters.
      For customized training, consider modifying the `build_model` method or extending
      the wrapper.
    
    - Hyperparameter tuning can be performed using Scikit-Learn's model selection
      tools. Ensure that all tunable parameters are exposed in the `__init__` method.
    
    - The `predict` method returns deterministic predictions if `quantiles` is `None`.
      If `quantiles` are specified, the model can be extended to provide probabilistic
      forecasts.
    
    See also
    --------
    sklearn.model_selection.RandomizedSearchCV : Randomized hyperparameter search.
    sklearn.model_selection.GridSearchCV : Exhaustive grid hyperparameter search.
    tensorflow.keras.Model : TensorFlow Keras Model API.
    """
    @validate_params({
        "static_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "hidden_units": [Interval(Integral, 1, None, closed='left')], 
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', list,  None],
        "activation": [
            StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", 'gelu'})],
        "use_batch_norm": [bool],
        "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": [list, Interval(Integral, 1, None, closed='left'), None], 
        "epochs": [Interval(Integral, 1, None, closed ='left')], 
        "batch_size": [Interval( Integral, 1, None, closed='left')]
        },
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        dynamic_input_dim,
        static_input_dim,
        hidden_units=32,
        num_heads=4,
        dropout_rate=0.1,
        forecast_horizon=1,
        quantiles=None,
        activation='elu',
        use_batch_norm=False,
        num_lstm_layers=1,
        lstm_units=None, 
        epochs=100, 
        batch_size= 32, 
        optimizer='adam',
        loss='mse',
        metrics=None,
        verbose=0,
    ):
        self.static_input_dim    = static_input_dim
        self.dynamic_input_dim   = dynamic_input_dim
        self.hidden_units        = hidden_units
        self.num_heads           = num_heads
        self.dropout_rate        = dropout_rate
        self.forecast_horizon   = forecast_horizon
        self.quantiles           = quantiles
        self.activation          = activation
        self.use_batch_norm      = use_batch_norm
        self.num_lstm_layers     = num_lstm_layers
        self.lstm_units          = lstm_units
        self.epochs              = epochs
        self.batch_size          = batch_size 
        self.optimizer           = optimizer
        self.loss                = loss
        self.metrics             = metrics if metrics is not None else ['mae']
        self.verbose             = verbose

    def build_model(self):
        if self.verbose >= 5:
            print("Building the Temporal Fusion Transformer model...")
        
        # Instantiate the Temporal Fusion Transformer model
        model = TemporalFusionTransformer(
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            hidden_units=self.hidden_units,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            forecast_horizon=self.forecast_horizon,
            quantiles=self.quantiles,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            num_lstm_layers=self.num_lstm_layers,
            lstm_units=self.lstm_units,
        )
        
        if self.verbose >= 6:
            print("Model instantiated with the following parameters:")
            print(f"static_input_dim: {self.static_input_dim}")
            print(f"dynamic_input_dim: {self.dynamic_input_dim}")
            print(f"hidden_units: {self.hidden_units}")
            print(f"num_heads: {self.num_heads}")
            print(f"dropout_rate: {self.dropout_rate}")
            print(f"forecast_horizon: {self.forecast_horizon}")
            print(f"quantiles: {self.quantiles}")
            print(f"activation: {self.activation}")
            print(f"use_batch_norm: {self.use_batch_norm}")
            print(f"num_lstm_layers: {self.num_lstm_layers}")
            print(f"lstm_units: {self.lstm_units}")
        
        if self.verbose >= 5:
            print("Compiling the model...")
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )
        
        if self.verbose >= 5:
            print("Model compiled successfully.")
        
        self.model_ = model
        
        if self.verbose >= 5:
            print("Model building completed.")
            
    @_scigofast_set_X_compat(ops ='concat')
    def fit(self, X, y, **fit_params):
        """
        Fit the TFT model according to the given training data.
        """
        if self.verbose >= 3:
            print("Starting the fit process...")
            
        # Extract callbacks instance object if provided 
        callbacks =None 
        if fit_params: 
            callbacks = extract_callbacks_from(
                fit_params, return_fit_params= True
        )
            
        # update the epochs, batch_size and verbose if provided in fit_params 
        self.epochs = fit_params.pop('epochs', self.epochs) 
        self.batch_size = fit_params.pop('batch_size', self.batch_size) 
        self.verbose= fit_params.pop('verbose', self.verbose) 
        
        # Concatenate static and dynamic inputs if provided as list or tuple
        if isinstance(X, (list, tuple)):
            if self.verbose >= 4:
                print("Concatenating static and dynamic inputs for compatibility...")
            try:
                X = np.concatenate([x.reshape(X[0].shape[0], -1) for x in X], axis=1)
                if self.verbose >= 6:
                    print(f"Concatenated X shape: {X.shape}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error concatenating inputs: {e}")
                raise ValueError("Error concatenating X inputs.") from e
        elif isinstance(X, np.ndarray):
            if self.verbose >= 6:
                print(f"Input X shape: {X.shape}")
        else:
            raise ValueError("Input X must be a list, tuple, or numpy array.")
        
        # Split X into static and dynamic components
        if self.verbose >= 4:
            print("Splitting X into static and dynamic components...")
        X_static, X_dynamic = self._split_X(X)
        
        if self.verbose >= 5:
            print(f"X_static shape: {X_static.shape}")
            print(f"X_dynamic shape: {X_dynamic.shape}")
        
        # Reshape y to match model output
        y = y.reshape(
            -1, 
            self.forecast_horizon, 
            self.static_input_dim
        )
        if self.verbose >= 6:
            print(f"Reshaped y shape: {y.shape}")
        
        # Build the model if not already built
        if self.model_ is None:
            if self.verbose >= 3:
                print("Building the model...")
            self.build_model()
        
        # Fit the model
        if self.verbose >= 3:
            print("Starting model training...")
        self.model_.fit(
            [X_static, X_dynamic],
            y,
            epochs=self.epochs,
            batch_size=self.batc_size,
            callbacks = callbacks, 
            verbose=self.verbose, 
            **fit_params,
        )
        if self.verbose >= 3:
            print("Model training completed.")
        
        return self
    
    @_scigofast_set_X_compat(ops ='concat')
    def predict(self, X):
        """
        Predict using the TFT model.
        """
        check_is_fitted(self, attributes =['model_'])
        
        if self.verbose >= 3:
            print("Starting the prediction process...")
        
        # Concatenate static and dynamic inputs if provided as list or tuple
        if isinstance(X, (list, tuple)):
            if self.verbose >= 4:
                print("Concatenating static and dynamic inputs for compatibility...")
            try:
                X = np.concatenate([x.reshape(X[0].shape[0], -1) for x in X], axis=1)
                if self.verbose >= 6:
                    print(f"Concatenated X shape: {X.shape}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error concatenating inputs: {e}")
                raise ValueError("Error concatenating X inputs.") from e
        elif isinstance(X, np.ndarray):
            if self.verbose >= 6:
                print(f"Input X shape: {X.shape}")
        else:
            raise ValueError("Input X must be a list, tuple, or numpy array.")
        
        # Split X into static and dynamic components
        if self.verbose >= 4:
            print("Splitting X into static and dynamic components...")
        X_static, X_dynamic = self._split_X(X)
        
        if self.verbose >= 5:
            print(f"X_static shape: {X_static.shape}")
            print(f"X_dynamic shape: {X_dynamic.shape}")
        
        # Generate predictions
        if self.verbose >= 3:
            print("Generating predictions...")
        predictions = self.model_.predict(
            [X_static, X_dynamic],
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        
        if self.verbose >= 3:
            print("Prediction completed.")
        
        return predictions

    def _split_X(self, X):
        """
        Split the concatenated X into static and dynamic components.
        """
        # static_size = self.num_static_vars * self.static_input_dim
        # dynamic_size = self.forecast_horizon * self.num_dynamic_vars * self.dynamic_input_dim
        static_size = self.static_input_dim
        dynamic_size = self.forecast_horizon * self.dynamic_input_dim
        total_size = static_size + dynamic_size
        
        if self.verbose >= 6:
            print(f"Expected total features: {total_size}, got: {X.shape[1]}")
        
        if X.shape[1] != total_size:
            message = (
                f"Expected X to have {total_size} features, "
                f"got {X.shape[1]}."
            )
            if self.verbose >= 1:
                print(message)
            raise ValueError(message)
        
        if self.verbose >= 6:
            print("Splitting X into static and dynamic parts...")
        
        # Split static features
        X_static = X[:, :static_size]
        X_static = X_static.reshape(
            (X_static.shape[0], 
             # self.num_static_vars, 
             self.static_input_dim)
        )
        if self.verbose >= 6:
            print(f"Reshaped X_static to {X_static.shape}")
        
        # Split dynamic features
        X_dynamic = X[:, static_size:]
        X_dynamic = X_dynamic.reshape(
            (X_dynamic.shape[0], 
             self.forecast_horizon, 
             # self.num_dynamic_vars, 
             self.dynamic_input_dim)
        )
        if self.verbose >= 6:
            print(f"Reshaped X_dynamic to {X_dynamic.shape}")
        
        return X_static, X_dynamic

@doc(xtft_params=dedent(_shared_docs['xtft_params_doc']))
class XTFTWrapper(BaseEstimator, RegressorMixin):
    """
    XTFTWrapper: A Scikit-Learn Compatible Wrapper for the XTFT Model.

    This class provides seamless integration of the extended Temporal 
    Fusion Transformer (XTFT) model with Scikit-Learn's API. It enables 
    the use of Scikit-Learn's model selection tools such as 
    `RandomizedSearchCV` and `GridSearchCV` for hyperparameter tuning. 
    By adhering to Scikit-Learn's estimator requirements, this wrapper 
    facilitates easy training, prediction, and evaluation within 
    Scikit-Learn workflows.

    {xtft_params}

    epochs : int, optional, default=100
        The number of epochs (full passes through the entire dataset) 
        to train the model.

    batch_size : int, optional, default=32
        The number of samples per batch of computation during training.

    optimizer : Union[str, tf.keras.optimizers.Optimizer], default='adam'
        Optimizer to use for compiling the model.

    loss : Union[str, tf.keras.losses.Loss], default='mse'
        Loss function to use for training the model.

    metrics : Optional[List[Union[str, tf.keras.metrics.Metric]]], default=None
        List of metrics to be evaluated by the model during training 
        and testing.

    verbose : int, optional, default=0
        Verbosity mode. Higher values print more logs.


    Attributes
    ----------
    model_ : tensorflow.keras.Model
        The underlying TensorFlow Keras model after fitting.

    Methods
    -------
    build_model()
        Constructs and compiles the XTFT model based on the initialized parameters.

    fit(X, y, **fit_params)
        Trains the XTFT model on the provided data.

    predict(X)
        Generates predictions using the trained XTFT model.

    See Also
    --------
    sklearn.model_selection.RandomizedSearchCV : Randomized hyperparameter search.
    sklearn.model_selection.GridSearchCV : Exhaustive grid hyperparameter search.
    tensorflow.keras.Model : TensorFlow Keras Model API.

    Examples
    --------
    >>> from gofast.nn.wrappers import XTFTWrapper
    >>> import numpy as np
    >>>
    >>> # Sample data
    >>> batch_size = 50
    >>> forecast_horizon = 20
    >>> static_input_dim = 10
    >>> dynamic_input_dim = 45
    >>> future_covariate_dim = 5
    >>> output_dim = 1
    >>>
    >>> # Synthetic inputs:
    >>> static_input = np.random.randn(batch_size, static_input_dim).astype(np.float32)
    >>> dynamic_input = np.random.randn(batch_size, forecast_horizon, dynamic_input_dim).astype(np.float32)
    >>> future_covariate_input = np.random.randn(batch_size, forecast_horizon, future_covariate_dim).astype(np.float32)
    >>>
    >>> # Synthetic target
    >>> y = np.random.randn(batch_size, forecast_horizon, output_dim).astype(np.float32)
    >>>
    >>> # Initialize the XTFTWrapper
    >>> xtft = XTFTWrapper(
    ...     static_input_dim=static_input_dim,
    ...     dynamic_input_dim=dynamic_input_dim,
    ...     future_input_dim=future_covariate_dim,
    ...     embed_dim=32,
    ...     forecast_horizon=forecast_horizon,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     max_window_size=10,
    ...     memory_size=100,
    ...     num_heads=4,
    ...     dropout_rate=0.1,
    ...     output_dim=output_dim,
    ...     attention_units=32,
    ...     hidden_units=64,
    ...     lstm_units=64,
    ...     scales=None,
    ...     multi_scale_agg=None,
    ...     activation='relu',
    ...     use_residuals=True,
    ...     use_batch_norm=False,
    ...     final_agg='last',
    ...     epochs=2,
    ...     batch_size=16,
    ...     optimizer='adam',
    ...     loss='mse',
    ...     metrics=['mae'],
    ...     verbose=2
    ... )
    >>>
    >>> # Fit the model
    >>> xtft.fit(X=[static_input, dynamic_input, future_covariate_input], y=y)
    >>>
    >>> # Make predictions
    >>> predictions = xtft.predict(X=[static_input, dynamic_input, future_covariate_input])
    >>> print(predictions.shape)
    (50, 20, 1)
    """
    @validate_params({
        "static_input_dim": [Interval(Integral, 1, None, closed='left')],
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "future_input_dim": [Interval(Integral, 0, None, closed='left')],
        "embed_dim": [Interval(Integral, 1, None, closed='left')],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like',  None],
        "max_window_size": [Interval(Integral, 1, None, closed='left')],
        "memory_size": [Interval(Integral, 1, None, closed='left')],
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
        "attention_units": [Interval(Integral, 1, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": [Interval(Integral, 1, None, closed='left')],
        "scales": ['array-like', StrOptions({'auto'}), None],
        "multi_scale_agg": [StrOptions({"last", "average",  "flatten", "auto"}), None],
        "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", 'gelu'})],
        "use_residuals": [bool],
        "use_batch_norm": [bool],
        "final_agg": [StrOptions({"last", "average", "flatten"})],
        "epochs": [Interval(Integral, 1, None, closed='left')],
        "batch_size": [Interval(Integral, 1, None, closed='left')],
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        dynamic_input_dim,
        future_input_dim,
        static_input_dim,
        embed_dim=32,
        forecast_horizon=1,
        quantiles=None,
        max_window_size=10,
        memory_size=100,
        num_heads=4,
        dropout_rate=0.1,
        output_dim=1,
        anomaly_config=None,
        attention_units=32,
        hidden_units=64,
        lstm_units=64,
        scales=None,
        multi_scale_agg=None,
        activation='relu',
        use_residuals=True,
        use_batch_norm=False,
        final_agg='last',
        epochs=100,
        batch_size=32,
        optimizer='adam',
        loss='mse',
        metrics=None,
        verbose=0,
    ):
        self.static_input_dim     = static_input_dim
        self.dynamic_input_dim    = dynamic_input_dim
        self.future_input_dim     = future_input_dim
        self.embed_dim            = embed_dim
        self.forecast_horizon    = forecast_horizon
        self.quantiles            = quantiles
        self.max_window_size      = max_window_size
        self.memory_size          = memory_size
        self.num_heads            = num_heads
        self.dropout_rate         = dropout_rate
        self.output_dim           = output_dim
        self.anomaly_config       = anomaly_config
        self.attention_units      = attention_units
        self.hidden_units         = hidden_units
        self.lstm_units           = lstm_units
        self.scales               = scales
        self.multi_scale_agg      = multi_scale_agg
        self.activation           = activation
        self.use_residuals        = use_residuals
        self.use_batch_norm       = use_batch_norm
        self.final_agg            = final_agg
        self.epochs               = epochs
        self.batch_size           = batch_size
        self.optimizer            = optimizer
        self.loss                 = loss
        self.metrics              = metrics if metrics is not None else ['mae']
        self.verbose              = verbose

    def build_model(self):
        if self.verbose >= 1:
            print("Building the XTFT model...")

        model = XTFT(
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_input_dim=self.future_input_dim,
            embed_dim=self.embed_dim,
            forecast_horizon=self.forecast_horizon,
            quantiles=self.quantiles,
            max_window_size=self.max_window_size,
            memory_size=self.memory_size,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            output_dim=self.output_dim,
            anomaly_config=self.anomaly_config,
            attention_units=self.attention_units,
            hidden_units=self.hidden_units,
            lstm_units=self.lstm_units,
            scales=self.scales,
            multi_scale_agg=self.multi_scale_agg,
            activation=self.activation,
            use_residuals=self.use_residuals,
            use_batch_norm=self.use_batch_norm,
            final_agg=self.final_agg,
        )

        if self.verbose >= 2:
            print("Model instantiated with the following parameters:")
            print(f"static_input_dim: {self.static_input_dim}")
            print(f"dynamic_input_dim: {self.dynamic_input_dim}")
            print(f"future_input_dim: {self.future_input_dim}")
            print(f"embed_dim: {self.embed_dim}")
            print(f"forecast_horizon: {self.forecast_horizon}")
            print(f"quantiles: {self.quantiles}")
            print(f"max_window_size: {self.max_window_size}")
            print(f"memory_size: {self.memory_size}")
            print(f"num_heads: {self.num_heads}")
            print(f"dropout_rate: {self.dropout_rate}")
            print(f"output_dim: {self.output_dim}")
            print(f"anomaly_config: {self.anomaly_config}")
            print(f"attention_units: {self.attention_units}")
            print(f"hidden_units: {self.hidden_units}")
            print(f"lstm_units: {self.lstm_units}")
            print(f"scales: {self.scales}")
            print(f"multi_scale_agg: {self.multi_scale_agg}")
            print(f"activation: {self.activation}")
            print(f"use_residuals: {self.use_residuals}")
            print(f"use_batch_norm: {self.use_batch_norm}")
            print(f"final_agg: {self.final_agg}")

        if self.verbose >= 1:
            print("Compiling the model...")
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )

        if self.verbose >= 1:
            print("Model compiled successfully.")

        self.model_ = model

    @_scigofast_set_X_compat('xtft', ops ='concat')
    def fit(self, X, y, **fit_params):
        if self.verbose >= 1:
            print("Starting the fit process...")

        # Extract callbacks if provided
        callbacks = None
        if fit_params:
            callbacks = extract_callbacks_from(
                fit_params, return_fit_params=True
            )

        # Update epochs, batch_size, verbose if provided in fit_params
        self.epochs = fit_params.pop('epochs', self.epochs)
        self.batch_size = fit_params.pop('batch_size', self.batch_size)
        self.verbose = fit_params.pop('verbose', self.verbose)

        # We expect either:
        # 1) X as a single merged numpy array from scikit-learn pipelines
        #    In this case, we need to split it into static_input, dynamic_input, future_covariate_input.
        # 2) X as a list [static_input, dynamic_input, future_covariate_input]
        #    We will concatenate them to a single array for compatibility.
        
        if isinstance(X, (list, tuple)):
            if self.verbose >= 2:
                print(
                    "Concatenating static, dynamic and future_covariate inputs for compatibility...")
            try:
                # Concatenate inputs into a single array
                X = np.concatenate([x.reshape(X[0].shape[0], -1) for x in X], axis=1)
                if self.verbose >= 3:
                    print(f"Concatenated X shape: {X.shape}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error concatenating inputs: {e}")
                raise ValueError("Error concatenating X inputs.") from e
        elif not isinstance(X, np.ndarray):
            raise ValueError("X must be either a single NumPy array or a list/tuple of arrays.")

        # Now X is a single NumPy array. We must split it back into the three components.
        X_static, X_dynamic, X_future = self._split_X(X)

        # Build the model if not already built
        if self.model_ is None:
            if self.verbose >= 1:
                print("Building the model before training...")
            self.build_model()

        if self.verbose >= 1:
            print("Training the model...")

        self.model_.fit(
            [X_static, X_dynamic, X_future],
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose,
            **fit_params,
        )

        if self.verbose >= 1:
            print("Model training completed.")
        return self

    @_scigofast_set_X_compat('xtft', ops ='concat')
    def predict(self, X):
        check_is_fitted(self, attributes=['model_'])

        # Similar logic as in fit:
        if isinstance(X, (list, tuple)):
            if self.verbose >= 2:
                print(
                    "Concatenating static, dynamic and future_covariate"
                    " inputs for prediction compatibility...")
            try:
                X = np.concatenate([x.reshape(X[0].shape[0], -1) for x in X], axis=1)
                if self.verbose >= 3:
                    print(f"Concatenated X shape for prediction: {X.shape}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error concatenating prediction inputs: {e}")
                raise ValueError("Error concatenating X inputs for prediction.") from e
        elif not isinstance(X, np.ndarray):
            raise ValueError(
                "X must be either a single NumPy array or a list/tuple of arrays.")

        X_static, X_dynamic, X_future = self._split_X(X)

        if self.verbose >= 1:
            print("Generating predictions...")
        predictions = self.model_.predict(
            [X_static, X_dynamic, X_future],
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        if self.verbose >= 1:
            print("Prediction completed.")
        return predictions

    def _split_X(self, X: np.ndarray):
        """
        Split the concatenated X into static, dynamic, and future_covariate
        components.

        Expected shapes:
        - static_input: (batch_size, static_input_dim)
        - dynamic_input: (batch_size, forecast_horizon, dynamic_input_dim)
        - future_covariate_input: (batch_size, forecast_horizon, future_input_dim)

        Total features = static_input_dim
                       + forecast_horizon * dynamic_input_dim
                       + forecast_horizon * future_input_dim
        """
        batch_size = X.shape[0]
        static_size = self.static_input_dim
        dynamic_size = self.forecast_horizon * self.dynamic_input_dim
        future_size = self.forecast_horizon * self.future_input_dim
        total_expected = static_size + dynamic_size + future_size

        if X.shape[1] != total_expected:
            message = (
                f"Expected X to have {total_expected} features, "
                f"got {X.shape[1]}."
            )
            if self.verbose >= 1:
                print(message)
            raise ValueError(message)

        # Extract static features
        X_static = X[:, :static_size]
        X_static = X_static.reshape((batch_size, self.static_input_dim))
        # Extract dynamic features
        X_dynamic = X[:, static_size:static_size+dynamic_size]
        X_dynamic = X_dynamic.reshape((batch_size, self.forecast_horizon, self.dynamic_input_dim))
        # Extract future covariate features
        X_future = X[:, static_size+dynamic_size:]
        X_future = X_future.reshape((batch_size, self.forecast_horizon, self.future_input_dim))

        if self.verbose >= 3:
            print(f"X_static shape: {X_static.shape}")
            print(f"X_dynamic shape: {X_dynamic.shape}")
            print(f"X_future shape: {X_future.shape}")

        return X_static, X_dynamic, X_future

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@validate_params({
    "static_input_dim": [Interval(Integral, 1, None, closed='left')], 
    "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
    "hidden_units": [Interval(Integral, 1, None, closed='left')], 
    "num_heads": [Interval(Integral, 1, None, closed='left')],
    "dropout_rate": [Interval(Real, 0, 1, closed="both")],
    "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
    "quantiles": ['array-like', list,  None],
    "activation": [
        StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", 'gelu'})],
    "use_batch_norm": [bool],
    "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
    "lstm_units": [list, Interval(Integral, 1, None, closed='left'), None]
    },
)
@doc(math_f= dedent ( 
    """
    .. math::
        \text{TFT Model} = \text{TemporalFusionTransformer}(
            \text{static\_input\_dim}, 
            \text{dynamic\_input\_dim}, 
            \text{num\_static\_vars}, 
            \text{num\_dynamic\_vars}, 
            \text{hidden\_units}, 
            \text{num\_heads}, 
            \text{dropout\_rate}, 
            \text{forecast\_horizon}, 
            \text{quantiles}, 
            \text{activation}, 
            \text{use\_batch\_norm}, 
            \text{num\_lstm\_layers}, 
            \text{lstm\_units}
        )
    
    """
    ),
    tft_params=dedent(
        _shared_docs['tft_params_doc']
        )
 )
def create_tft_model(
    dynamic_input_dim: int,
    static_input_dim: int,
    hidden_units: int = 64,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    forecast_horizon: int = 1,
    quantiles: Optional[List[float]] = None,
    activation: str = 'relu',
    use_batch_norm: bool = True,
    num_lstm_layers: int = 1,
    lstm_units: int = 64,
    optimizer: Union[str, Optimizer] = 'adam',
    loss: Union[str, Loss] = 'mse',
    metrics: Optional[List[Union[str, Metric]]] = None,
    **kwargs
) -> Model:
    """
    Create and compile a Temporal Fusion Transformer (TFT) model.
    
    The `create_tft_model` function initializes a Temporal Fusion Transformer model 
    with the specified architecture and compiles it with the provided optimizer, 
    loss function, and evaluation metrics. This function is designed to be compatible 
    with scikit-learn estimators like `KerasRegressor`, enabling integration 
    into scikit-learn workflows such as hyperparameter tuning and cross-validation.
    
    {math_f}
    
    {tft_params}
    
    optimizer : `Union[str, tf.keras.optimizers.Optimizer]`, default=`'adam'`
        Optimizer to use for compiling the model. Can be a string identifier 
        or an instance of a Keras Optimizer.
    
    loss : `Union[str, tf.keras.losses.Loss]`, default=`'mse'`
        Loss function to use for training the model. Can be a string identifier 
        or an instance of a Keras Loss.
    
    metrics : `Optional[List[Union[str, tf.keras.metrics.Metric]]]`, default=`None`
        List of metrics to be evaluated by the model during training and testing.
        If `None`, no additional metrics are used.
    
    **kwargs
        Additional keyword arguments to pass to the TemporalFusionTransformer 
        constructor.
    
    Returns
    -------
    tf.keras.Model
        A compiled Temporal Fusion Transformer model ready for training.
    
    Raises
    ------
    ValueError
        If `forecast_horizon` is not a positive integer.
    
    TypeError
        If `quantiles` is provided but is not a list of floats between 0 and 1.
    
    Examples
    --------
    >>> from gofast.nn.wrappers import create_tft_model
    >>> 
    >>> # Define model parameters
    >>> static_input_dim = 1
    >>> dynamic_input_dim = 1
    >>> 
    >>> # Create and compile the TFT model
    >>> model = create_tft_model(
    ...     static_input_dim=static_input_dim,
    ...     dynamic_input_dim=dynamic_input_dim,
    ...     hidden_units=128,
    ...     num_heads=8,
    ...     dropout_rate=0.2,
    ...     forecast_horizon=1,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     activation='tanh',
    ...     use_batch_norm=False,
    ...     num_lstm_layers=2,
    ...     lstm_units=128,
    ...     optimizer='adam',
    ...     loss='mse',
    ...     metrics=['mae']
    ... )
    >>> 
    >>> # Display the model summary
    >>> model.summary()
    
    Notes
    -----
    - **Compatibility with scikit-learn:** This function is designed to work seamlessly 
      with scikit-learn estimators such as `KerasRegressor`, enabling integration 
      into scikit-learn pipelines and facilitating hyperparameter tuning via 
      `RandomizedSearchCV` or `GridSearchCV`.
    
    - **Quantile Forecasting:** Providing a list of quantiles enables the model to 
      perform probabilistic forecasting. Each quantile corresponds to a different 
      prediction, allowing for uncertainty estimation in the forecasts.
    
    - **Customizability:** The function exposes various hyperparameters, allowing users 
      to tailor the model architecture to their specific needs and dataset characteristics.
    
    - **Additional Arguments:** Any additional keyword arguments are forwarded to the 
      `TemporalFusionTransformer` constructor, offering further flexibility in model 
      configuration.
    
    See Also
    --------
    `TemporalFusionTransformer` : The underlying TFT model class.
    `KerasRegressor` : scikit-learn wrapper for Keras models.
    
    References
    ----------
    .. [1] Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., & Cottrell, G. (2017). 
       Temporal fusion transformers for interpretable multi-horizon time series forecasting. 
       *arXiv preprint arXiv:1912.09363*.
    
    .. [2] Brownlee, J. (2018). *Time Series Forecasting with Python: Create accurate 
       models in Python to forecast the future and gain insight from your time series 
       data*. Machine Learning Mastery.
    """
    if not isinstance(forecast_horizon, int) or forecast_horizon < 1:
        raise ValueError("`forecast_horizon` must be a positive integer.")
    
    if quantiles is not None:
        if (not isinstance(quantiles, list) or 
            not all(isinstance(q, float) and 0 < q < 1 for q in quantiles)):
            raise TypeError("`quantiles` must be a list of floats between 0 and 1.")
    
    # Instantiate the Temporal Fusion Transformer model
    model = TemporalFusionTransformer(
        static_input_dim=static_input_dim,
        dynamic_input_dim=dynamic_input_dim,
        hidden_units=hidden_units,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        forecast_horizon=forecast_horizon,
        quantiles=quantiles,
        activation=activation,
        use_batch_norm=use_batch_norm,
        num_lstm_layers=num_lstm_layers,
        lstm_units=lstm_units,
        **kwargs
    )
    
    # Compile the model with the specified optimizer, loss, and metrics
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@validate_params({
    "static_input_dim": [Interval(Integral, 1, None, closed='left')],
    "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
    "future_input_dim": [Interval(Integral, 0, None, closed='left')],
    "embed_dim": [Interval(Integral, 1, None, closed='left')],
    "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
    "quantiles": ['array-like',  None],
    "max_window_size": [Interval(Integral, 1, None, closed='left')],
    "memory_size": [Interval(Integral, 1, None, closed='left')],
    "num_heads": [Interval(Integral, 1, None, closed='left')],
    "dropout_rate": [Interval(Real, 0, 1, closed="both")],
    "output_dim": [Interval(Integral, 1, None, closed='left')],
    "attention_units": [Interval(Integral, 1, None, closed='left')],
    "hidden_units": [Interval(Integral, 1, None, closed='left')],
    "lstm_units": [Interval(Integral, 1, None, closed='left')],
    "scales": ['array-like', StrOptions({'auto'}), None],
    "multi_scale_agg": [StrOptions({"last", "average",  "flatten", "auto"}), None],
    "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", 'gelu'})],
    "use_residuals": [bool],
    "use_batch_norm": [bool],
    "final_agg": [StrOptions({"last", "average", "flatten"})],
    "epochs": [Interval(Integral, 1, None, closed='left')],
    "batch_size": [Interval(Integral, 1, None, closed='left')],
})
@doc(xtft_params=dedent(_shared_docs['xtft_params_doc_minimal']))
def create_xtft_model(
    dynamic_input_dim: int,
    future_input_dim: int,
    static_input_dim: int,
    embed_dim: int = 32,
    forecast_horizon: int = 1,
    quantiles: Optional[List[float]] = None,
    max_window_size: int = 10,
    memory_size: int = 100,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    output_dim: int = 1,
    anomaly_config: Optional[dict] = None,
    attention_units: int = 32,
    hidden_units: int = 64,
    lstm_units: int = 64,
    scales: Union[str, List[int], None] = None,
    multi_scale_agg: Optional[str] = None,
    activation: str = 'relu',
    use_residuals: bool = True,
    use_batch_norm: bool = False,
    final_agg: str = 'last',
    optimizer: Union[str, Optimizer] = 'adam',
    loss: Union[str, Loss] = 'mse',
    metrics: Optional[List[Union[str, Metric]]] = None,
    epochs: int = 100,
    batch_size: int = 32,
    **kwargs
) -> Model:
    """
    Create and compile an XTFT (Extended Temporal Fusion Transformer) model.
    
    The `create_xtft_model` function initializes an XTFT model with the specified 
    architecture and compiles it with the provided optimizer, loss function, and 
    evaluation metrics. Like `create_tft_model`, this function is designed to be 
    compatible with scikit-learn estimators like `KerasRegressor`, enabling integration 
    into scikit-learn workflows (e.g., hyperparameter tuning, cross-validation).

    Parameters
    ----------
    {xtft_params}

    optimizer : `Union[str, tf.keras.optimizers.Optimizer]`, default=`'adam'`
        Optimizer to use for compiling the model.

    loss : `Union[str, tf.keras.losses.Loss]`, default=`'mse'`
        Loss function to use for training the model.

    metrics : `Optional[List[Union[str, tf.keras.metrics.Metric]]]`, default=`None`
        List of metrics to be evaluated by the model during training and testing.
        If `None`, no additional metrics are used.

    epochs : int, optional, default=100
        The number of epochs (full passes through the entire dataset) 
        to train the model.

    batch_size : int, optional, default=32
        The number of samples per batch of computation during training.

    **kwargs
        Additional keyword arguments passed to the XTFT constructor for fine-grained 
        customization of the model.

    Returns
    -------
    tf.keras.Model
        A compiled XTFT model ready for training.

    Raises
    ------
    ValueError
        If `forecast_horizon` is not a positive integer.

    TypeError
        If `quantiles` is provided but is not a list of floats between 0 and 1.

    Examples
    --------
    >>> from gofast.nn.wrappers import create_xtft_model
    >>> import numpy as np
    >>>
    >>> # Define model parameters
    >>> static_input_dim = 10
    >>> dynamic_input_dim = 45
    >>> future_covariate_dim = 5
    >>> forecast_horizon = 20
    >>> output_dim = 1
    >>>
    >>> # Create and compile the XTFT model
    >>> model = create_xtft_model(
    ...     static_input_dim=static_input_dim,
    ...     dynamic_input_dim=dynamic_input_dim,
    ...     future_input_dim=future_covariate_dim,
    ...     embed_dim=32,
    ...     forecast_horizon=forecast_horizon,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     max_window_size=10,
    ...     memory_size=100,
    ...     num_heads=4,
    ...     dropout_rate=0.1,
    ...     output_dim=output_dim,
    ...     attention_units=32,
    ...     hidden_units=64,
    ...     lstm_units=64,
    ...     scales=None,
    ...     multi_scale_agg=None,
    ...     activation='relu',
    ...     use_residuals=True,
    ...     use_batch_norm=False,
    ...     final_agg='last',
    ...     optimizer='adam',
    ...     loss='mse',
    ...     metrics=['mae'],
    ...     epochs=2,
    ...     batch_size=16
    ... )
    >>>
    >>> # Display the model summary
    >>> model.summary()

    Notes
    -----
    - **Compatibility with scikit-learn:** Like `create_tft_model`, this function 
      is designed for seamless integration into scikit-learn workflows using 
      wrappers like `KerasRegressor`.

    - **Quantile Forecasting:** By providing quantiles, the model can produce 
      probabilistic forecasts, helping estimate uncertainty.

    - **Customizability:** Users can customize various aspects of the model, 
      including the number of attention heads, LSTM units, and scaling strategies.

    - **Additional Arguments:** Any additional keyword arguments are forwarded 
      to the `XTFT` constructor for more detailed configuration.

    See Also
    --------
    XTFT : The underlying XTFT model class.
    KerasRegressor : scikit-learn wrapper for Keras models.
    """

    if not isinstance(forecast_horizon, int) or forecast_horizon < 1:
        raise ValueError("`forecast_horizon` must be a positive integer.")

    if quantiles is not None:
        quantiles= validate_quantiles_in( 
            quantiles, round_digits=2, 
            dtype=np.float32 
        )
        # if (not isinstance(quantiles, list) or 
        #     not all(isinstance(q, float) and 0 < q < 1 for q in quantiles)):
        #     raise TypeError("`quantiles` must be a list of floats between 0 and 1.")

    # Instantiate the XTFT model
    model = XTFT(
        static_input_dim=static_input_dim,
        dynamic_input_dim=dynamic_input_dim,
        future_input_dim=future_input_dim,
        embed_dim=embed_dim,
        forecast_horizon=forecast_horizon,
        quantiles=quantiles,
        max_window_size=max_window_size,
        memory_size=memory_size,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        output_dim=output_dim,
        anomaly_config=anomaly_config,
        attention_units=attention_units,
        hidden_units=hidden_units,
        lstm_units=lstm_units,
        scales=scales,
        multi_scale_agg=multi_scale_agg,
        activation=activation,
        use_residuals=use_residuals,
        use_batch_norm=use_batch_norm,
        final_agg=final_agg,
        **kwargs
    )

    # Compile the model with the specified optimizer, loss, and metrics
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model
