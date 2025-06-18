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

from ...api.docs import _shared_docs, doc
from ...core.diagnose_q import validate_quantiles_in 
from ...compat.sklearn import validate_params, Interval, StrOptions
from ...utils.deps_utils import ensure_pkg 
from ...utils.validator import check_is_fitted 

from .. import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .._adapter_utils import compat_X 
from ..models._xtft import XTFT
from ..utils import extract_callbacks_from 
    
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

DEP_MSG = dependency_message('wrappers') 

__all__ = [

    'XTFTWrapper', 
    "create_xtft_model"
]

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
    >>> from fusionlab.nn.wrappers import XTFTWrapper
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

    @compat_X('xtft', ops ='concat')
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

    @compat_X('xtft', ops ='concat')
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
    >>> from fusionlab.nn.wrappers import create_xtft_model
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
