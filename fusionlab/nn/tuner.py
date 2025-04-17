#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Module for NN hyperparameter tuning and optimization.

Includes functions and classes for hyperparameter optimization using methods 
like Hyperband, Population Based Training, and more.
"""
from __future__ import annotations 
import os
import copy 
import datetime
import warnings 
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..api.types import _Tensor, _Dataset, _Optimizer
from ..api.types import  _Callback, _Model, _Sequential 
from ..api.types import List, Union, Dict, Tuple, DataFrame, Series 
from ..api.types import ArrayLike , Callable, Any
from ..api.property import BaseClass
from ..core.checks import is_iterable 
from ..core.utils import type_of_target 
from ..decorators import smartFitRun
from ..utils.deps_utils import ensure_pkg, import_optional_dependency
from ..utils.validator import check_consistent_length, is_frame 

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .keras_validator import validate_keras_model

if KERAS_BACKEND:
    Adam = KERAS_DEPS.Adam
    RMSprop = KERAS_DEPS.RMSprop
    SGD = KERAS_DEPS.SGD
    EarlyStopping=KERAS_DEPS.EarlyStopping
    TensorBoard=KERAS_DEPS.TensorBoard
    LSTM=KERAS_DEPS.LSTM
    load_model = KERAS_DEPS.load_model
    mnist = KERAS_DEPS.mnist
    Loss = KERAS_DEPS.Loss
    Sequential = KERAS_DEPS.Sequential
    Dense = KERAS_DEPS.Dense
    reduce_mean = KERAS_DEPS.reduce_mean
    GradientTape = KERAS_DEPS.GradientTape
    square = KERAS_DEPS.square
    Dataset=KERAS_DEPS.Dataset 
    LearningRateScheduler=KERAS_DEPS.LearningRateScheduler
    clone_model=KERAS_DEPS.clone_model
    
    
__all__= [ 
    'Hyperband', 'PBTTrainer', 'base_tuning', 'custom_loss', 'deep_cv_tuning', 
    'fair_neural_tuning', 'find_best_lr', 'lstm_ts_tuner', 'robust_tuning'
    
]

DEP_MSG=dependency_message('tune')

@smartFitRun 
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
class PBTTrainer(BaseClass):
    """
    Implements Population Based Training (PBT), a hyperparameter optimization
    technique that adapts hyperparameters dynamically during training. 
    
    PBT optimizes a population of models concurrently, utilizing the 
    "exploit and explore" strategy to iteratively refine and discover optimal
    configurations.

    Parameters
    ----------
    model_fn : callable
        Function that constructs and returns a compiled Keras model. This
        function should take no arguments and return a model ready for training.
        Example:
        ```python
        def model_fn():
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            return model
        ```

    param_space : dict
        Defines the hyperparameter space for exploration in the format:
        {'hyperparameter': (min_value, max_value)}. Each hyperparameter's range
        from which values are initially sampled and later perturbed is specified
        by a tuple (min, max).
        Example:
        ```python
        param_space = {'learning_rate': (0.001, 0.01), 'batch_size': (16, 64)}
        ```

    population_size : int, optional
        The size of the model population. Defaults to 10. A larger population
        can increase the diversity of models and potential solutions but requires
        more computational resources.

    exploit_method : str, optional
        The method to use for the exploit phase. Currently, only 'truncation'
        is implemented, which replaces underperforming models with perturbed versions
        of better performers. Default is 'truncation'.

    perturb_factor : float, optional
        The factor by which hyperparameters are perturbed during the explore phase.
        Default is 0.2. 
        This is a fractional change applied to selected hyperparameters.

    num_generations : int, optional
        The number of training/evaluation/generation cycles to perform. Default is 5.
        Each generation involves training all models and applying the exploit and
        explore strategies.

    epochs_per_step : int, optional
        The number of epochs to train each model during each generation before
        applying exploit and explore. Default is 5.

    verbose : int, optional
        Verbosity level of the training output. 0 is silent, while higher values
        increase the logging detail. Default is 0.

    Attributes
    ----------
    best_params_ : dict
        Hyperparameters of the best-performing model at the end of training.

    best_score_ : float
        The highest validation score achieved by any model in the population.

    best_model_ : tf.keras.Model
        The actual Keras model that achieved `best_score_`.

    model_results_ : list
        A list containing the performance and hyperparameters of each model
        at each generation.
        
    Notes
    -----

    Population Based Training (PBT) alternates between two phases:

    - **Exploit**: Underperforming models are replaced by copies of better-performing
      models, often with slight modifications.
    - **Explore**: Hyperparameters of the models are perturbed to encourage
      exploration of the hyperparameter space.

    .. math::

        \text{perturb}(x) = x \times\\
            (1 + \text{uniform}(-\text{perturb_factor}, \text{perturb_factor}))


    PBT dynamically adapts hyperparameters, facilitating discovery of optimal settings
    that might not be reachable through static hyperparameter tuning methods. It is
    especially useful for long-running tasks where hyperparameters may need to change
    as the training progresses.

    Examples
    --------
    >>> from gofast.nn.tune import PBTTrainer
    >>> def model_fn():
    ...     model = tf.keras.Sequential([
    ...         tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    ...         tf.keras.layers.Dense(1, activation='sigmoid')
    ...     ])
    ...     model.compile(optimizer='adam', loss='binary_crossentropy')
    ...     return model
    >>> param_space = {'learning_rate': (0.001, 0.01), 'batch_size': (16, 64)}
    >>> trainer = PBTTrainer(model_fn=model_fn, param_space=param_space, 
    ...                      population_size=5, num_generations=10, 
    ...                      epochs_per_step=2, verbose=1)
    >>> trainer.run(train_data=(X_train, y_train), val_data=(X_val, y_val))

    See Also
    --------
    tf.keras.models.Model : 
        TensorFlow Keras base model class.
    tf.keras.callbacks :
        Callbacks in TensorFlow Keras that can be used within training loops.

    References
    ----------
    .. [1] Jaderberg, Max, et al. "Population based training of neural networks."
           arXiv preprint arXiv:1711.09846 (2017).
    """
    def __init__(
        self, 
        model_fn, 
        param_space, 
        population_size=10, 
        exploit_method='truncation',
        perturb_factor=0.2, 
        num_generations=5, 
        epochs_per_step=5, 
        verbose=0
    ):
        self.model_fn = model_fn
        self.param_space = param_space
        self.population_size = population_size
        self.exploit_method = exploit_method
        self.perturb_factor = perturb_factor
        self.num_generations = num_generations
        self.epochs_per_step = epochs_per_step
        self.verbose = verbose
        
    def run(self, train_data: Tuple[ArrayLike, ArrayLike],
            val_data: Tuple[ArrayLike, ArrayLike])-> 'PBTTrainer':
        """
        Executes the Population Based Training (PBT) optimization cycle across
        multiple generations for the given dataset, applying training, evaluation,
        and the dynamic adjustment of hyperparameters through exploitation 
        and exploration.
    
        Parameters
        ----------
        train_data : Tuple[ArrayLike, ArrayLike]
            A tuple (X_train, y_train) containing the training data and labels,
            where `X_train` is the feature set and `y_train` is the 
            corresponding label set.
        val_data : Tuple[ArrayLike, ArrayLike]
            A tuple (X_val, y_val) containing the validation data and labels, used
            for evaluating the model performance after each training epoch.
    
        Returns
        -------
        self : PBTTrainer
            This method returns the instance of `PBTTrainer` with updated properties
            including the best model, parameters, and scores achieved during 
            the PBT process.
    
        Examples
        --------
        >>> from gofast.nn.tune import PBTTrainer
        >>> def model_fn():
        ...     model = tf.keras.Sequential([
        ...         tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
        ...         tf.keras.layers.Dense(1, activation='sigmoid')
        ...     ])
        ...     model.compile(optimizer='adam', loss='binary_crossentropy',
        ...                     metrics=['accuracy'])
        ...     return model
        >>> train_data = (np.random.rand(100, 10), np.random.rand(100))
        >>> val_data = (np.random.rand(20, 10), np.random.rand(20))
        >>> param_space = {'learning_rate': (0.001, 0.01), 'batch_size': (16, 64)}
        >>> trainer = PBTTrainer(model_fn=model_fn, param_space=param_space, 
        ...                         population_size=5, num_generations=10, 
        ...                         epochs_per_step=2, verbose=1)
        >>> trainer.run(train_data, val_data)
        >>> print(trainer.best_params_)
    
        Notes
        -----
        The PBT process allows models to adapt their hyperparameters dynamically,
        which can significantly enhance model performance over static hyperparameter
        configurations. The exploit and explore mechanism ensures that only the
        most promising configurations are evolved, improving efficiency.
    
        See Also
        --------
        tf.keras.models.Sequential : Frequently used to define a linear stack of layers.
        tf.keras.optimizers.Adam : Popular optimizer with adaptive learning rates.
    
        References
        ----------
        .. [1] Jaderberg, Max, et al. "Population based training of neural networks."
               arXiv preprint arXiv:1711.09846 (2017).
               Describes the foundational PBT approach.
        """
        self.population = self._init_population()
        if self.exploit_method.lower() !="truncation": 
            warnings.warn("Currently, supported only 'truncation' method.")
            self.exploit_method='truncation'
            
        for generation in range(self.num_generations):
            if self.verbose: 
                print(f"Generation {generation + 1}/{self.num_generations}")
            for model, hyperparams in self.population:
                self._train_model(model, train_data, hyperparams, self.epochs_per_step)
                performance = self._evaluate_model(model, val_data)
                self.model_results_.append({'hyperparams': hyperparams, 
                                            'performance': performance})
                if performance > self.best_score_:
                    self.best_score_ = performance
                    self.best_params_ = hyperparams
                    self.best_model_ = copy.deepcopy(model)
            self._exploit_and_explore()

        return self
    
    def _init_population(self):
        """
        Initializes the population with models and random hyperparameters.
        """
        population = []
        for _ in range(self.population_size):
            hyperparams = {k: np.random.uniform(low=v[0], high=v[1]) 
                           for k, v in self.param_space.items()}
            model = self.model_fn()
            population.append((model, hyperparams))
        return population
    
    def _train_model(self, model, train_data, hyperparams, epochs):
        """
        Trains a single model instance using TensorFlow.

        Parameters
        ----------
        model : tf.keras.Model
            The TensorFlow model to train.
        train_data : tuple
            A tuple (X_train, y_train) containing the training data and labels.
        hyperparams : dict
            Hyperparameters to use for training, including 'learning_rate'.
        epochs : int
            Number of epochs to train the model.
        """
        X_train, y_train = train_data
        optimizer = Adam(learning_rate=hyperparams['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=int(
            hyperparams.get('batch_size', 32)), verbose=0)

    def _evaluate_model(self, model, val_data):
        """
        Evaluates a single model instance using TensorFlow.

        Parameters
        ----------
        model : tf.keras.Model
            The TensorFlow model to evaluate.
        val_data : tuple
            A tuple (X_val, y_val) containing the validation data and labels.

        Returns
        -------
        performance : float
            The performance metric of the model, typically accuracy.
        """
        X_val, y_val = val_data
        _, performance = model.evaluate(X_val, y_val, verbose=0)
        return performance
    
    def _exploit_and_explore(self):
        """
        Apply the exploit and explore strategy to evolve the population.
        """
        # Sort models based on performance
        self.population.sort(key=lambda x: x[0].performance, reverse=True)

        # Exploit: Replace bottom half with top performers
        top_performers = self.population[:len(self.population) // 2]
        for i in range(len(self.population) // 2, len(self.population)):
            if self.exploit_method == 'truncation':
                # Clone a top performer's model and hyperparameters
                model, hyperparams = copy.deepcopy(top_performers[i % len(top_performers)])
                self.population[i] = (model, hyperparams)

        # Explore: Perturb the hyperparameters
        for i in range(len(self.population) // 2, len(self.population)):
            _, hyperparams = self.population[i]
            perturbed_hyperparams = {k: v * np.random.uniform(
                1 - self.perturb_factor, 1 + self.perturb_factor) 
                for k, v in hyperparams.items()}
            self.population[i] = (self.model_fn(), perturbed_hyperparams)  
            # Reinitialize model

@smartFitRun 
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
class Hyperband(BaseClass):
    """
    Implements the Hyperband hyperparameter optimization algorithm, utilizing 
    a bandit-based approach combined with successive halving to efficiently
    allocate computational resources to promising model configurations.

    The core idea of Hyperband is to dynamically allocate and prune resources 
    across a spectrum of model configurations, effectively balancing between
    exploration of the hyperparameter space and exploitation of promising 
    configurations through computationally efficient successive halving.

    Parameters
    ----------
    model_fn : Callable[[Dict[str, Any]], tf.keras.Model]
        A function that accepts a dictionary of hyperparameters and returns a
        compiled Keras model. This function is responsible for both the instantiation 
        and compilation of the model, integrating the provided hyperparameters.
    max_resource : int
        The maximum amount of computational resources (typically the number of 
        epochs) that can be allocated to a single model configuration.
    eta : float, optional
        The reduction factor for pruning less promising model configurations in each
        round of successive halving. The default value is 3, where resources 
        are reduced to one-third of the previous round's resources at each step.

    Attributes
    ----------
    best_model_ : tf.keras.Model
        The model instance that achieved the highest validation performance score.
    best_params_ : Dict[str, Any]
        The hyperparameter set associated with `best_model_`.
    best_score_ : float
        The highest validation score achieved by `best_model_`.
    model_results_ : List[Dict[str, Any]]
        Details of all the configurations evaluated, including their hyperparameters 
        and performance scores.

    Examples
    --------
    >>> from gofast.nn.tune import Hyperband
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>> from tensorflow.keras.optimizers import Adam
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.model_selection import train_test_split
    >>> from tensorflow.keras.utils import to_categorical

    >>> def model_fn(params):
    ...     model = Sequential([
    ...         Dense(params['units'], activation='relu', input_shape=(64,)),
    ...         Dense(10, activation='softmax')
    ...     ])
    ...     model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
    ...                   loss='categorical_crossentropy', metrics=['accuracy'])
    ...     return model

    >>> digits = load_digits()
    >>> X = digits.data / 16.0  # Normalize the data
    >>> y = to_categorical(digits.target, num_classes=10)
    >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                          random_state=42)

    >>> hyperband = Hyperband(model_fn=model_fn, max_resource=81, eta=3)
    >>> hyperband.run(train_data=(X_train, y_train), val_data=(X_val, y_val))
    >>> print(f"Best Hyperparameters: {hyperband.best_params_},
              Best Score: {hyperband.best_score_:.2f}")

    Notes
    -----
    Hyperband is particularly effective when combined with models that can 
    benefit from substantial training. The mathematical formulation of the 
    resource allocation in Hyperband is as follows:

    .. math::
        s_{max} = \\left\\lfloor \\log_{\\eta}(\\text{max_resource}) \\right\\rfloor
        B = (s_{max} + 1) \\times \\text{max_resource}

    Where :math:`s_{max}` is the maximum number of iterations, and :math:`B` 
    represents the total budget across all brackets.

    See Also
    --------
    tf.keras.Model : Used for constructing neural networks in TensorFlow.

    References
    ----------
    .. [1] Li, Lisha, et al. "Hyperband: A novel bandit-based approach to 
           hyperparameter optimization." Journal of Machine Learning Research, 2018.
    """

    def __init__(
            self, model_fn: Callable, max_resource: int, eta: float = 3 ):
        self.max_resource = max_resource
        self.eta = eta
        self.model_fn = model_fn
        
    def _train_and_evaluate(
        self, model_config: Dict[str, Any], 
        resource: int, 
        train_data: Tuple, 
        val_data: Tuple
        ) -> float:
        """
        Trains and evaluates a model for a specified configuration and resource.
        
        Parameters
        ----------
        model_config : Dict[str, Any]
            Hyperparameter configuration for the model.
        resource : int
            Allocated resource for the model, typically the number of epochs.
        train_data : Tuple[np.ndarray, np.ndarray]
            Training data and labels.
        val_data : Tuple[np.ndarray, np.ndarray]
            Validation data and labels.

        Returns
        -------
        float
            The performance metric of the model, e.g., validation accuracy.
        """
        model = self.model_fn(model_config)
        X_train, y_train = train_data
        X_val, y_val = val_data
        history = model.fit(X_train, y_train, epochs=resource, 
                            validation_data=(X_val, y_val), verbose=self.verbose )
        val_accuracy = history.history['val_accuracy'][-1]
        return val_accuracy

    def get_hyperparameter_configuration(self, n: int) -> List[Dict[str, Any]]:
        """
        Generates a list of `n` random hyperparameter configurations.
        
        Parameters
        ----------
        n : int
            Number of configurations to generate.
        
        Returns
        -------
        List[Dict[str, Any]]
            A list of hyperparameter configurations.
        """
        configurations = [{'learning_rate': np.random.uniform(1e-4, 1e-2),
                           'units': np.random.randint(50, 500)} 
                          for _ in range(n)]
        return configurations

    def run(self, train_data: Tuple[ArrayLike, ArrayLike],
                val_data: Tuple[ArrayLike, ArrayLike]) -> 'Hyperband':
        """
        Executes the Hyperband optimization process on a given dataset, efficiently
        exploring the hyperparameter space using the adaptive resource allocation and
        early-stopping strategy known as successive halving.
    
        Parameters
        ----------
        train_data : Tuple[ArrayLike, ArrayLike]
            A tuple consisting of the training data and labels (`X_train`, `y_train`).
            These arrays are used to fit the models at each stage of the process.
        val_data : Tuple[ArrayLike, ArrayLike]
            A tuple consisting of the validation data and labels (`X_val`, `y_val`).
            These arrays are used to evaluate the performance of the models and determine
            which configurations proceed to the next round.
    
        Returns
        -------
        self : Hyperband
            This method returns an instance of the `Hyperband` class, providing access
            to the best model, its hyperparameters, and performance metrics after the
            optimization process is completed.
    
        Examples
        --------
        >>> from gofast.nn.tune import Hyperband
        >>> from tensorflow.keras.layers import Dense
        >>> from tensorflow.keras.models import Sequential
        >>> from tensorflow.keras.optimizers import Adam
        >>> from sklearn.datasets import load_digits
        >>> from sklearn.model_selection import train_test_split
        >>> from tensorflow.keras.utils import to_categorical
    
        >>> def model_fn(params):
        ...     model = Sequential([
        ...         Dense(params['units'], activation='relu', input_shape=(64,)),
        ...         Dense(10, activation='softmax')
        ...     ])
        ...     model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
        ...                   loss='categorical_crossentropy', metrics=['accuracy'])
        ...     return model
    
        >>> digits = load_digits()
        >>> X = digits.data / 16.0  # Normalize the data
        >>> y = to_categorical(digits.target, num_classes=10)
        >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                              random_state=42)
    
        >>> hyperband = Hyperband(model_fn=model_fn, max_resource=81, eta=3)
        >>> hyperband.run(train_data=(X_train, y_train), val_data=(X_val, y_val))
        >>> print(f"Best Hyperparameters: {hyperband.best_params_},
                  Best Score: {hyperband.best_score_:.2f}")
    
        Notes
        -----
        Hyperband optimizes hyperparameter configurations in a structured and 
        resource-efficient manner. It employs a geometric progression to 
        systematically eliminate poorer performing configurations and concentrate
        resources on those with more promise. 
        This method is significantly more efficient than random or grid search methods,
        especially when the computational budget is a limiting factor.
    
        Mathematical formulation involves determining the configurations and resources
        in each round:
        
        .. math::
            s_{\\text{max}} = \\left\\lfloor \\log_{\\eta}(\\text{max_resource})\\
                \\right\\rfloor
            B = (s_{\\text{max}} + 1) \\times \\text{max_resource}
            
        where :math:`s_{\\text{max}}` is the maximum number of iterations 
        (depth of configurations),
        and :math:`B` is the total budget across all brackets.
    
        See Also
        --------
        tf.keras.Model :
            The base TensorFlow/Keras model used for constructing neural networks.
    
        References
        ----------
        .. [1] Li, Lisha, et al. "Hyperband: A novel bandit-based approach to 
               hyperparameter optimization." The Journal of Machine Learning Research,
               18.1 (2017): 6765-6816.
        """
        for s in reversed(range(int(np.log(self.max_resource) / np.log(self.eta)) + 1)):
            n = int(np.ceil(self.max_resource / self.eta ** s / (s + 1)))
            resource = self.max_resource * self.eta ** (-s)
            configurations = self.get_hyperparameter_configuration(n)
            for i in range(s + 1):
                n_i = n * self.eta ** (-i)
                r_i = resource * self.eta ** i
                
                val_scores = [self._train_and_evaluate(
                    config, int(np.floor(r_i)), train_data, val_data) 
                    for config in configurations]
                
                if self.verbose:
                   print(f"Generation {i+1}/{s+1}, Configurations evaluated:"
                         f" {len(configurations)}")
                # Select top-k configurations based on validation scores
                if i < s:
                    top_k_indices = np.argsort(val_scores)[-max(int(n_i / self.eta), 1):]
                    configurations = [configurations[j] for j in top_k_indices]
                else:
                    best_index = np.argmax(val_scores)
                    self.best_score_ = val_scores[best_index]
                    self.best_params_ = configurations[best_index]
                    self.best_model_ = self.model_fn(self.best_params_)
                    self.model_results_.append({'config': self.best_params_, 
                                                'score': self.best_score_})
        return self

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def base_tuning(
    model:_Sequential ,
    train_data: Tuple[ArrayLike, ArrayLike],
    val_data: Tuple[ArrayLike, ArrayLike],
    test_data: Tuple[ArrayLike, ArrayLike],
    learning_rates: List[float],
    batch_sizes: List[int],
    epochs: int,
    optimizer: Union[_Optimizer, Callable[[float], _Optimizer]],
    loss: Union[str, Callable],
    metrics: List[str],
    callbacks: List[_Callback] = None
) -> Tuple[_Model, float, float]:
    """
    Conducts hyperparameter tuning for a given neural network model by varying
    learning rates, batch sizes, and potentially other parameters. This method
    systematically explores different combinations of these parameters to find
    the configuration that yields the best performance on the validation set.

    Parameters
    ----------
    model : `_Model`
        The machine learning model to be fine-tuned. This model should be
        an instance of a neural network framework class, such as a Keras Model.
    train_data : Tuple[ArrayLike, ArrayLike]
        A tuple containing the training data and labels. `ArrayLike` can refer
        to any array-like structure (e.g., lists, NumPy arrays, Pandas DataFrame
        columns).
    val_data : Tuple[ArrayLike, ArrayLike]
        A tuple containing the validation data and labels. This data is used to
        evaluate the model performance for each hyperparameter configuration.
    test_data : Tuple[ArrayLike, ArrayLike]
        A tuple containing the test data and labels, used to evaluate the final
        model's performance.
    learning_rates : List[float]
        A list of learning rates to evaluate. The learning rate controls the
        step size during the gradient descent optimization.
    batch_sizes : List[int]
        A list of batch sizes to evaluate. The batch size determines the number
        of samples that will be propagated through the network at once.
    epochs : int
        The total number of passes through the training dataset.
    optimizer : Union[_Optimizer, Callable[[float], _Optimizer]]
        The optimization algorithm or a callable that returns an optimizer instance.
        The optimizer is used to update weights based on the gradients of the loss.
    loss : Union[str, Callable]
        The loss function used to evaluate a candidate solution (model configuration).
    metrics : List[str]
        A list of metrics to be evaluated by the model during training and testing.
    callbacks : List[_Callback], optional
        A list of callbacks to be used during training to view or store intermediate
        results, or halt training based on certain conditions.

    Returns
    -------
    Tuple[_Model, float, float]
        - `best_model`: The best performing model on the validation dataset.
        - `best_accuracy`: Highest accuracy achieved on the validation dataset.
        - `test_accuracy`: Accuracy on the test dataset using the best model.

    Examples
    --------
    >>> from gofast.nn.tune import base_tuning
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>> from tensorflow.keras.optimizers import Adam
    >>> from tensorflow.keras.callbacks import EarlyStopping
    >>> import numpy as np
    >>> model = Sequential([Dense(10, activation='relu', input_shape=(20,)),
    ...                     Dense(1, activation='sigmoid')])
    >>> X_train, y_train = np.random.rand(100, 20), np.random.randint(0, 2, 100)
    >>> X_val, y_val = np.random.rand(20, 20), np.random.randint(0, 2, 20)
    >>> X_test, y_test = np.random.rand(20, 20), np.random.randint(0, 2, 20)
    >>> best_model, best_accuracy, test_accuracy = base_tuning(
    ...     model, (X_train, y_train), (X_val, y_val), (X_test, y_test),
    ...     [0.01, 0.001], [50, 100], 10, Adam, 'binary_crossentropy',
    ...     ['accuracy'], [EarlyStopping(monitor='val_loss', patience=3)])

    Notes
    -----
    Hyperparameter tuning is crucial for optimizing model performance, particularly
    in complex models or datasets with diverse attributes. It is recommended to
    use a validation set which is different from the test set to avoid overfitting.

    See Also
    --------
    Sequential : The Keras model class used to construct neural networks.
    Adam : A commonly used gradient descent optimization algorithm with adaptive
           learning rate capabilities.
    EarlyStopping : A callback to stop training when a monitored metric has stopped
                    improving.

    References
    ----------
    .. [1] Goodfellow, Ian, et al. "Deep Learning." MIT press, 2016. Provides
           a comprehensive overview of deep learning techniques and their
           applications.
    """
    best_accuracy = 0
    best_model = None
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Configure the model with the current hyperparameters
            model.compile(optimizer=optimizer(learning_rate=lr),
                          loss=loss,
                          metrics=metrics)
    
            # Train the model
            model.fit(train_data, batch_size=batch_size,
                      epochs=epochs, validation_data=val_data,
                      callbacks=callbacks)
    
            # Evaluate the model on validation data
            # Assuming accuracy is the second metric
            accuracy = model.evaluate(val_data)[1]  
    
            # Update the best model if current model is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
    
    # Evaluate the best model on the test set
    test_accuracy = best_model.evaluate(test_data)[1]
    
    return best_model, best_accuracy, test_accuracy

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def robust_tuning(
    model_fn: Callable[..., _Model],
    dataset: Tuple[ArrayLike, ArrayLike],
    param_grid: Dict[str, List[Union[float, int]]],
    n_splits: int = 5,
    epochs: int = 50,
    patience: int = 5,
    log_dir: str = "logs/fit",
    loss: Union[str, Callable] = 'sparse_categorical_crossentropy',
    metrics: List[str] = 'accuracy',
    custom_callbacks: List[_Callback] = None
) -> Tuple[_Model, Dict[str, Union[float, int]], float]:
    """
    Implements a robust tuning mechanism by employing cross-validation 
    combined with grid search to find the optimal hyperparameters for a
    neural network model. This approach maximizes model generalizability 
    and performance across different subsets of data.

    Parameters
    ----------
    model_fn : Callable[..., _Model]
        A callable that, when passed hyperparameters, returns a compiled
        machine learning model. This function should have parameters that
        correspond to keys in `param_grid`.
    dataset : Tuple[ArrayLike, ArrayLike]
        A tuple containing training data and labels. The first element, `X`,
        represents the feature matrix, and the second element, `y`, represents
        the target labels.
    param_grid : Dict[str, List[Union[float, int]]]
        A dictionary specifying the hyperparameters to explore. Each key should
        match a parameter name in `model_fn`, with corresponding values being
        lists of settings to try.
    n_splits : int, optional
        Number of folds in the cross-validation process. More splits increase 
        the robustness of the hyperparameter evaluation but also increase
        computational demand. Default is 5.
    epochs : int, optional
        The maximum number of training epochs per model configuration. 
        Default is 50.
    patience : int, optional
        The number of epochs with no improvement after which training will be 
        stopped (used in conjunction with early stopping). Default is 5.
    log_dir : str, optional
        Path to the directory where to save logs, typically for TensorBoard. 
        Default is "logs/fit".
    loss : Union[str, Callable], optional
        The loss function to be used during training. Can be specified as a string
        or a callable object. Default is 'sparse_categorical_crossentropy'.
    metrics : List[str], optional
        Metrics to evaluate during model training. Typically includes metrics
        such as 'accuracy'. Default is ['accuracy'].
    custom_callbacks : List[_Callback], optional
        A list of additional Keras callbacks to use during training, such as 
        learning rate schedulers or custom logging mechanisms.

    Returns
    -------
    Tuple[_Model, Dict[str, Union[float, int]], float]
        - `best_model`: 
            The model instance with the highest cross-validated performance.
        - `best_params`:
            A dictionary of the parameters that yielded the best results.
        - `best_score`: 
            The highest score achieved during the cross-validation.

    Examples
    --------
    >>> from gofast.nn.tune import robust_tuning
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>> import numpy as np
    >>> def model_fn(learning_rate):
    ...     model = Sequential([
    ...         Dense(64, activation='relu', input_shape=(100,)),
    ...         Dense(3, activation='softmax')
    ...     ])
    ...     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    ...                   loss='sparse_categorical_crossentropy',
    ...                   metrics=['accuracy'])
    ...     return model
    >>> X, y = np.random.rand(1000, 100), np.random.randint(0, 3, size=(1000,))
    >>> param_grid = {'learning_rate': [0.001, 0.0001]}
    >>> best_model, best_params, best_score = robust_tuning(
    ...     model_fn=model_fn,
    ...     dataset=(X, y),
    ...     param_grid=param_grid,
    ...     n_splits=5,
    ...     epochs=10,
    ...     patience=3
    ... )

    Notes
    -----
    Using cross-validation in conjunction with grid search provides a
    comprehensive assessment of model stability and effectiveness across
    different subsets of data, thereby helping to mitigate overfitting and
    ensure the model's ability to generalize.

    See Also
    --------
    Sequential : 
        The TensorFlow/Keras model class typically used to construct models.
    Adam :
        An optimizer with adaptive learning rate, commonly used in deep learning.
    EarlyStopping :
        A callback to stop training when a monitored metric has stopped improving.

    References
    ----------
    .. [1] Goodfellow, Ian, et al. "Deep Learning." MIT press, 2016. Provides
           a foundational text on the principles and practices of deep learning,
           including model evaluation techniques such as cross-validation.
    """

    X, y = dataset
    kf = KFold(n_splits=n_splits)
    best_score = 0
    best_params = None
    best_model = None
    
    metrics= is_iterable(metrics, exclude_string=True, transform =True )

    # Prepare grid search
    param_combinations = [dict(zip(param_grid, v))
                          for v in itertools.product(*param_grid.values())]

    for params in param_combinations:
        scores = []
        for train_index, val_index in kf.split(X):
            # Split data
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Create a new instance of the model with the current set of parameters
            model = model_fn(**params)
            model.compile(optimizer=model.optimizer, loss=loss, metrics=metrics)

            # Prepare callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=patience)
            log_path = os.path.join(
                log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
            callbacks = [early_stop, tensorboard_callback] + (
                custom_callbacks if custom_callbacks else [])

            # Train the model
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=params.get('batch_size', 32),
                      callbacks=callbacks)

            # Evaluate the model 
            # Assuming the second metric is the score
            score = model.evaluate(X_val, y_val, verbose=0)[1]  
            scores.append(score)

        # Compute the average score over all folds
        avg_score = np.mean(scores)

        # Update the best model if the current set of parameters is better
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_model = clone_model(model)
            best_model.set_weights(model.get_weights())

    return best_model, best_params, best_score

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def fair_neural_tuning(
    model_fn: Callable[..., _Model],
    train_data: Tuple[ArrayLike, ArrayLike],
    val_data: Tuple[ArrayLike, ArrayLike],
    test_data: Tuple[ArrayLike, ArrayLike],
    param_grid: Dict[str, List[Union[float, int, str]]],
    epochs: int = 50,
    loss: Union[str, Callable] = 'sparse_categorical_crossentropy',
    metrics: List[str] = 'accuracy',
    callbacks: List[_Callback] = None,
    verbose: int = 0
) -> Tuple[_Model, Dict[str, Union[float, int, str]], float]:
    """
    Optimizes hyperparameters of a neural network model by conducting a grid search 
    over specified parameters, evaluating each combination using the provided training,
    validation, and test datasets. This function aims to find the combination that
    maximizes model performance, measured through specified metrics.

    Parameters
    ----------
    model_fn : Callable[..., _Model]
        A callable that creates and compiles a neural network model. It must accept
        hyperparameters as arguments that match keys in `param_grid`.
    train_data : Tuple[ArrayLike, ArrayLike]
        Training data and labels as a tuple (features, labels), where both elements
        are array-like structures.
    val_data : Tuple[ArrayLike, ArrayLike]
        Validation data and labels similar to `train_data`.
    test_data : Tuple[ArrayLike, ArrayLike]
        Test data and labels, used for final evaluation of the tuned model.
    param_grid : Dict[str, List[Union[float, int, str]]]
        A dictionary defining the hyperparameters to explore, with each key being a
        parameter name and each value being a list of settings to try.
    epochs : int, optional
        Total number of epochs to train each model configuration. Default is 50.
    loss : Union[str, Callable], optional
        The loss function to use for training, specified as either a string or a
        callable object. Default is 'sparse_categorical_crossentropy'.
    metrics : List[str], optional
        A list of performance metrics to evaluate during training and testing.
        Default is ['accuracy'].
    callbacks : List[_Callback], optional
        Custom callbacks to be used during training, such as model checkpointing
        or early stopping. Default is None.
    verbose : int, optional
        Verbosity mode. 0 = silent, 1 = progress bar.

    Returns
    -------
    Tuple[_Model, Dict[str, Union[float, int, str]], float]
        - best_model: The model with the best performance on the validation set.
        - best_params: The hyperparameter set that yielded the best results.
        - best_score: The highest score achieved on the validation dataset.

    Examples
    --------
    >>> from gofast.nn.tune import fair_neural_tuning
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>> import numpy as np
    >>> def model_fn(learning_rate):
    ...     model = Sequential([
    ...         Dense(64, activation='relu', input_shape=(10,)),
    ...         Dense(1, activation='sigmoid')
    ...     ])
    ...     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    ...                   loss='binary_crossentropy',
    ...                   metrics=['accuracy'])
    ...     return model
    >>> train_data = (np.random.rand(100, 10), np.random.rand(100))
    >>> val_data = (np.random.rand(20, 10), np.random.rand(20))
    >>> test_data = (np.random.rand(20, 10), np.random.rand(20))
    >>> param_grid = {'learning_rate': [0.01, 0.001]}
    >>> best_model, best_params, best_score = fair_neural_tuning(
    ...     model_fn=model_fn,
    ...     train_data=train_data,
    ...     val_data=val_data,
    ...     test_data=test_data,
    ...     param_grid=param_grid,
    ...     epochs=10,
    ...     loss='binary_crossentropy',
    ...     metrics=['accuracy'],
    ...     verbose=1
    ... )

    Notes
    -----
    Grid search, while thorough, can be computationally expensive, especially
    with a large number of hyperparameter combinations or extensive datasets.
    It's advisable to perform such searches on high-performance computing resources
    to expedite the process.

    See Also
    --------
    Sequential : Keras model class used for building neural networks.
    Adam : Commonly used optimizer with adaptive learning rate.

    References
    ----------
    .. [1] Goodfellow, Ian, et al. "Deep Learning." MIT Press, 2016. This book
           provides foundational machine learning concepts, including the
           details of various optimization techniques.
    """

    best_score = 0
    best_params = None
    best_model = None
    metrics= is_iterable(metrics, exclude_string=True, transform =True )
    param_combinations = [dict(zip(param_grid, v)) for v in itertools.product(
        *param_grid.values())]

    for params in tqdm(param_combinations, desc='Hyperparameter combinations', 
                       ascii=True, ncols=100):
        model = model_fn(**params)
        model.compile(optimizer=model.optimizer, loss=loss, metrics=metrics)
        
        current_callbacks = callbacks if callbacks else []
        
        model.fit(
            train_data[0], train_data[1], 
            batch_size=params.get('batch_size', 32),
            epochs=epochs, 
            validation_data=val_data, 
            verbose=verbose, 
            callbacks=current_callbacks
        )
        
        accuracy = model.evaluate(val_data[0], val_data[1], verbose=0)[1]
        
        if accuracy > best_score:
            best_score = accuracy
            best_params = params
            best_model = clone_model(model)
            best_model.set_weights(model.get_weights())

    best_model.evaluate(test_data[0], test_data[1], verbose=0)[1]
    
    return best_model, best_params, best_score

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def deep_cv_tuning(
    model_fn: Callable[..., _Model],
    dataset: Tuple[ArrayLike, ArrayLike],
    param_grid: Dict[str, List[Union[float, int, str]]],
    n_splits: int = 5,
    epochs: int = 50,
    patience: int = 5,
    log_dir: str = "logs/fit",
    loss: Union[str, Callable] = 'binary_crossentropy',
    metrics: List[str] = 'accuracy',
    verbose: int = 0,
    callbacks: List[_Callback] = None
) -> Tuple[_Model, Dict[str, Union[float, int, str]], float]:
    """
    Optimizes a neural network model's hyperparameters using cross-validation
    and grid search. This method systematically evaluates combinations of
    hyperparameters across multiple data splits, ensuring robustness and
    generalizability of the model's performance.

    Parameters
    ----------
    model_fn : Callable[..., tf.keras.Model]
        A factory function that creates and compiles a neural network model.
        It should accept any hyperparameters defined in `param_grid` as its
        arguments to allow for dynamic model configuration.
    dataset : Tuple[np.ndarray, np.ndarray]
        A tuple (X, y) consisting of the dataset's features (X) and labels (y),
        used for both training and validation during the cross-validation.
    param_grid : Dict[str, List[Union[float, int, str]]]
        Specifies the hyperparameters to be explored in the search. Each key
        represents a hyperparameter name, and the associated list contains
        values to be tested.
    n_splits : int, optional
        Defines the number of partitions the data is split into for
        cross-validation, defaulting to 5.
    epochs : int, optional
        The maximum number of training epochs for each model configuration,
        defaulting to 50.
    patience : int, optional
        The number of epochs to wait for an improvement before terminating
        training early, defaulting to 5.
    log_dir : str, optional
        Directory where to save TensorBoard log files, aiding in visualization
        and tracking of training progress.
    loss : Union[str, Callable], optional
        Specifies the loss function to be used during model compilation and
        training. It can be a string identifier or a callable object.
    metrics : List[str], optional
        A list of performance metrics that the model should be evaluated against
        during training and validation.
    verbose : int, optional
        Controls the verbosity of the training process output 
        (0 = silent, 1 = progress bar).
    callbacks : List[tf.keras.callbacks.Callback], optional
        Custom callbacks to enhance training, such as model checkpointing or
        learning rate adjustments.

    Returns
    -------
    Tuple[tf.keras.Model, Dict[str, Union[float, int, str]], float]
        - `best_model`: The model configuration with the highest performance across
          the cross-validation folds.
        - `best_params`: The set of parameters that resulted in the best performance.
        - `best_score`: The highest scoring metric achieved during cross-validation.

    Examples
    --------
    >>> from gofast.nn.tune import deep_cv_tuning
    >>> import numpy as np
    >>> def model_fn(learning_rate):
    ...     from tensorflow.keras.models import Sequential
    ...     from tensorflow.keras.layers import Dense
    ...     model = Sequential([
    ...         Dense(64, activation='relu', input_shape=(10,)),
    ...         Dense(1, activation='sigmoid')
    ...     ])
    ...     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    ...                   loss='binary_crossentropy', metrics=['accuracy'])
    ...     return model
    >>> dataset = (np.random.rand(1000, 10), np.random.rand(1000))
    >>> param_grid = {'learning_rate': [0.01, 0.001], 'batch_size': [32, 64]}
    >>> best_model, best_params, best_score = deep_cv_tuning(
    ...     model_fn=model_fn,
    ...     dataset=dataset,
    ...     param_grid=param_grid,
    ...     n_splits=5,
    ...     epochs=10,
    ...     patience=3,
    ...     verbose=1
    ... )

    Notes
    -----
    The use of cross-validation is critical in preventing overfitting and ensuring
    that the model's hyperparameters are not just optimal for a particular subset
    of data.

    See Also
    --------
    Sequential : The TensorFlow/Keras model class used to construct models.
    Adam : An optimizer that implements the Adam algorithm.

    References
    ----------
    .. [1] Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization."
           arXiv preprint arXiv:1412.6980 (2014).
    """

    X, y = dataset
    kf = KFold(n_splits=n_splits)
    best_score = -np.inf
    best_params = None
    best_model = None
    metrics= is_iterable(metrics, exclude_string=True, transform =True )

    param_combinations = [dict(zip(param_grid, v)) for v in itertools.product(
        *param_grid.values())]

    for params in tqdm(param_combinations, desc='Hyperparameter Grid', ascii=True, 
                       ncols=100):
        scores = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = model_fn(**params)
            model.compile(optimizer=model.optimizer, loss=loss, metrics=metrics)

            early_stop = EarlyStopping(monitor='val_loss', patience=patience,
                                       verbose=verbose)
            log_path = os.path.join(log_dir,
                                    f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                                    str(params))
            tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)

            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=params.get('batch_size', 32), 
                      verbose=verbose,
                      callbacks=[early_stop, tensorboard_callback] + (
                          callbacks if callbacks else []))
            # Assuming accuracy is the target metric
            score = model.evaluate(X_val, y_val, verbose=verbose)[1] 
            scores.append(score)

        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_model = clone_model(model)
            best_model.set_weights(model.get_weights())

    return best_model, best_params, best_score

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def custom_loss(
    y_true: _Tensor, 
    y_pred: _Tensor, 
    y_estimated: _Tensor, 
    lambda_value: float, 
    reduction: str = 'auto', 
    loss_name: str = 'custom_loss'
    ) -> Callable:
    """
    Computes a custom loss value which is a combination of mean squared 
    error between true and predicted values, and an additional term 
    weighted by a lambda value.

    Parameters
    ----------
    y_true : tf.Tensor
        The ground truth values.
        These are the actual values that the model is trying to predict.
        
    y_pred : tf.Tensor
        The predicted values.
        These are the values predicted by the model.

    y_estimated : tf.Tensor
        An estimated version of the ground truth values, used for an 
        additional term in the loss.
        This tensor allows for incorporating additional domain-specific 
        knowledge into the loss function, which might improve model 
        performance in certain contexts.

    lambda_value : float
        The weight of the additional term in the loss calculation.
        This value determines the importance of the additional term 
        relative to the mean squared error term in the total loss.

    reduction : str, optional
        Type of `tf.keras.losses.Reduction` to apply to loss.
        Default value is 'auto', which means the reduction option will be 
        determined by the current Keras backend. Other possible values 
        include 'sum_over_batch_size', 'sum', and 'none'.

    loss_name : str, optional
        Name to use for the loss.
        This allows the custom loss function to be referred to by a 
        specific name, which can be useful for logging and debugging.

    Returns
    -------
    Callable
        A callable that takes `y_true` and `y_pred` as inputs and returns 
        the loss value as an output.

    Notes
    -----
    The custom loss function is defined as:

    .. math::

        L(y_{\text{true}}, y_{\text{pred}}, y_{\text{estimated}}) = 
        \text{MSE}(y_{\text{true}}, y_{\text{pred}}) + 
        \lambda \cdot \text{MSE}(y_{\text{true}}, y_{\text{estimated}})

    where :math:`\text{MSE}` denotes the mean squared error and 
    :math:`\lambda` is the `lambda_value`.
    
    The `custom_loss` function is designed to be used with TensorFlow and 
    Keras models. The `y_estimated` parameter allows for incorporating 
    additional domain-specific knowledge into the loss, beyond what is 
    captured by comparing `y_true` and `y_pred` alone.

    Examples
    --------
    >>> from gofast.nn.tune import custom_loss
    >>> import tensorflow as tf
    >>> model = tf.keras.models.Sequential([
    ...     tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
    ...     tf.keras.layers.Dense(1)
    ... ])
    >>> model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(
    ...     y_true, y_pred, y_estimated=tf.zeros_like(y_pred), lambda_value=0.5))
    >>> # Assume X_train, y_train are prepared data
    >>> # model.fit(X_train, y_train, epochs=10)


    See Also
    --------
    tf.keras.losses.MeanSquaredError : Mean squared error loss function.
    tf.keras.Model : Base class for Keras models.

    References
    ----------
    .. [1] Chollet, F. et al. "Deep Learning with Python." 
           Manning Publications, 2017.
    """
    check_consistent_length(y_true,y_pred )
    
    def loss(y_true, y_pred):
        mse = reduce_mean(square(y_true - y_pred), axis=-1)
        additional_term = reduce_mean(square(y_true - y_estimated), axis=-1)
        return mse + lambda_value * additional_term
    
    return Loss(name=loss_name, reduction=reduction)(loss)

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def find_best_lr(
    model_fn: Callable, 
    train_data: Union[Tuple[ArrayLike, ArrayLike], _Dataset], 
    epochs: int = 5, 
    initial_lr: float = 1e-6, 
    max_lr: float = 1, 
    loss: str = 'binary_crossentropy', 
    steps_per_epoch: Union[int, None] = None, 
    verbose: int = 0, 
    batch_size: Union[int, None] = None, 
    view: bool = True
) -> float:
    """
    Identifies the optimal learning rate for training a Keras model. 
    
    Function gradually increases the learning rate within a specified range 
    and monitors the loss. The optimal learning rate is estimated 
    programmatically based on the steepest decline observed in the loss curve.

    Parameters
    ----------
    model_fn : Callable
        A function that returns an uncompiled Keras model. The function should
        not compile the model as this method will apply the learning rate
        adjustments dynamically.
        Example:
        ```
        def create_model():
            return tf.keras.models.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(100,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        ```

    train_data : Union[Tuple[np.ndarray, np.ndarray], tf.data.Dataset]
        The training data to be used. This can either be a tuple of NumPy arrays
        `(X_train, y_train)` or a TensorFlow `Dataset` object.

    epochs : int, optional
        The number of epochs to perform during the learning rate range test. 
        Defaults to 5.

    initial_lr : float, optional
        The initial learning rate to start the test from. Defaults to 1e-6.

    max_lr : float, optional
        The maximum learning rate to test up to. Defaults to 1.

    loss : str, optional
        Loss function to be used for compiling the model. Defaults to 
        'binary_crossentropy'.

    steps_per_epoch : int or None, optional
        Specifies the number of steps in each epoch. Required if `train_data` is
        a TensorFlow `Dataset`. Defaults to None, in which case it is calculated
        based on the batch size and the size of `train_data`.

    batch_size : int or None, optional
        The batch size for training. Required if `train_data` is provided as 
        NumPy arrays. Defaults to None.

    verbose : int, default=0 
        Controls the level of verbosity during the training process. 

    view : bool, optional
        If True, plots the loss against the learning rate upon completion of 
        the test, marking the estimated optimal learning rate. Defaults to True.

    Returns
    -------
    float
        The estimated optimal learning rate based on the observed loss curve.
        
    Notes
    -----
    The optimal learning rate is identified by gradually increasing the learning rate 
    from `initial_lr` to `max_lr` and observing the change in loss. The loss curve is 
    plotted, and the optimal learning rate is selected where the steepest decline in 
    loss occurs.

    .. math::
        \text{LR}_{\text{optimal}} = \arg\min_{\text{LR}}\\
            \left( \frac{\partial \text{Loss}}{\partial \text{LR}} \right)


    This function is useful for determining the best learning rate to use for training 
    a neural network. By testing a range of learning rates and observing the corresponding 
    changes in loss, the optimal learning rate that results in the fastest decrease in 
    loss can be identified.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.nn.tune import find_best_lr
    >>> def create_model():
    ...     return tf.keras.models.Sequential([
    ...         tf.keras.layers.Dense(10, activation='relu', input_shape=(100,)),
    ...         tf.keras.layers.Dense(1, activation='sigmoid')])
    >>> X_train, y_train = np.random.rand(1000, 100), np.random.randint(2, size=(1000, 1))
    >>> optimal_lr = find_best_lr(create_model, (X_train, y_train), 
    ...                           epochs=3, batch_size=32)
    >>> print(f"Optimal learning rate: {optimal_lr}")

    See Also
    --------
    tf.keras.optimizers.Adam : Optimizer that implements the Adam algorithm.
    tf.keras.callbacks.LearningRateScheduler : Callback to schedule the learning rate during training.

    References
    ----------
    .. [1] Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks." 
           2017 IEEE Winter Conference on Applications of Computer Vision (WACV), 464-472.
    """

    validate_keras_model(model_fn, raise_exception=True)
    if isinstance(train_data, tuple):
        if batch_size is None:
            raise ValueError("When using NumPy arrays as train_data, batch_size"
                             " must be specified.")
        train_data = Dataset.from_tensor_slices(train_data).shuffle(
            buffer_size=10000).batch(batch_size)
        steps_per_epoch = len(train_data)

    lr_schedule = LearningRateScheduler(
        lambda epoch: initial_lr + (max_lr - initial_lr) * epoch / (epochs - 1)
    )

    model = model_fn()
    model.compile(optimizer=Adam(learning_rate=initial_lr),
                  loss=loss)

    history = model.fit(train_data, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        callbacks=[lr_schedule], verbose=verbose )

    lr_diff = np.diff(learning_rates := np.linspace(initial_lr, max_lr, epochs))
    loss_diff = np.diff(history.history['loss'])
    derivatives = loss_diff / lr_diff
    steepest_decline_index = np.argmin(derivatives)
    optimal_lr = learning_rates[steepest_decline_index]

    if view:
        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates[:-1], history.history['loss'][:-1], label='Loss')
        plt.plot(optimal_lr, history.history['loss'][steepest_decline_index], 'ro',
                 label='Optimal LR')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    return optimal_lr

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def lstm_ts_tuner(
    data: DataFrame,
    target: Union[str, Series, ArrayLike],
    n_lag: int,
    activation: str = 'relu', 
    decompose_ts: bool = True,
    decomposition_model: str = 'additive',
    decomposition_period: int = 12,
    n_splits: int = 5,
    epochs: int = 100,
    metric: str = "auto",
    scale: str = 'minmax',
    learning_rate: float = 0.01
) -> dict:
    """
    Optimizes an LSTM model considering optional decomposition of the time series,
    feature scaling, and cross-validation. 

    Function uses cross-validation for model tuning and supports both mean squared
    error (MSE) and accuracy metrics, and allows for the data to be scaled 
    using either MinMaxScaler or StandardScaler.

    Parameters
    ----------
    data : DataFrame
        Input dataframe containing the time series data and any additional features.
        This dataframe should have a time index for proper time series analysis.

    target : Union[str, Series, ArrayLike]
        The target variable to predict. Can be a column name (str) in `data`, 
        a Series, or an ArrayLike object.
        Example: `'target_column'` if `data` is a dataframe with the target column.

    n_lag : int
        Number of lag observations to include as input features for the model.
        This determines how many past observations are used to predict future values.

    activation : str, default='relu'
        The activation function to use in the LSTM layers. Options include 'relu', 
        'sigmoid', 'tanh', etc.
        Example: `'tanh'` for a hyperbolic tangent activation function.

    decompose_ts : bool, default=True
        Whether to decompose the time series data before modeling.
        Decomposition can help in separating the trend, seasonality, and residuals 
        in the time series.

    decomposition_model : str, default='additive'
        Model to use for seasonal decomposition ('additive' or 'multiplicative').
        Example: `'multiplicative'` if the seasonal variations are proportional to 
        the level of the time series.

    decomposition_period : int, default=12
        Frequency of the time series data for decomposition.
        This is typically set to the number of observations per year, month, etc.

    n_splits : int, default=5
        Number of splits for time series cross-validation.
        This determines how many folds are used in the cross-validation process.

    epochs : int, default=100
        Number of epochs to train the LSTM model.
        This defines how many times the training process will work through the entire 
        training dataset.

    metric : str, default='auto'
        Metric to evaluate the model performance ('mse', 'accuracy', or 'auto' to 
        decide based on target type).
        Example: `'mse'` for mean squared error, which is common in regression tasks.

    scale : str, default='minmax'
        Scaling method for the input features ('minmax' or 'normalize').
        Example: `'normalize'` for standardizing features by removing the mean and 
        scaling to unit variance.

    learning_rate : float, default=0.01
        Learning rate for the optimizer.
        This controls how much to change the model in response to the estimated error 
        each time the model weights are updated.

    Returns
    -------
    dict
        Dictionary containing the best score and LSTM model parameters after tuning.
        The dictionary includes the optimal hyperparameters and the performance metric.
        
    Notes
    -----
    The LSTM model is tuned using cross-validation. The loss function can be 
    mean squared error (MSE) for regression tasks, defined as:

    .. math::

        \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

    where :math:`y_i` are the actual values and :math:`\hat{y}_i` are the predicted values.

    For accuracy in classification tasks, the metric is:

    .. math::

        \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}


    This function is useful for optimizing LSTM models for time series forecasting, 
    taking into account possible seasonality and trends through decomposition. 
    It also allows for feature scaling, which can improve model performance.

    Examples
    --------
    >>> import pandas as pd 
    >>> import numpy as np 
    >>> from gofast.nn.tune import lstm_ts_tuner
    >>> data = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
    >>> best_params = lstm_ts_tuner(data, 'target_column', n_lag=12)
    >>> print(best_params)

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler :
        Transforms features by scaling each feature to a given range.
    sklearn.preprocessing.StandardScaler : 
        Standardize features by removing the mean and scaling to unit variance.

    References
    ----------
    .. [1] Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." 
           Neural Computation, 9(8), 1735-1780.
    .. [2] Hyndman, R. J., & Athanasopoulos, G. (2018). "Forecasting: principles 
           and practice." OTexts.
    """

    is_frame(data, df_only=True, raise_exception= True )
    target = data[target] if isinstance(target, str) else target
    metric = ( 'mse' if metric == 'auto' and type_of_target(target) == 'contineous'
              else 'accuracy') 
    if decompose_ts:
        import_optional_dependency("statsmodels", extra= ( 
            "Need 'statsmodels' for time-series decomposition")
            )
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(target, model=decomposition_model,
                                           period=decomposition_period)
        data['residual'] = decomposition.resid.fillna(method='bfill').fillna(method='ffill')
    
    scaler = MinMaxScaler() if scale == 'minmax' else StandardScaler()
    data['scaled_residual'] = scaler.fit_transform(data[['residual']])

    X, y = _create_sequences(data['scaled_residual'].to_numpy(), n_lag)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = _build_lstm_model(n_lag, learning_rate, activation=activation)

    metric_scores = _cross_validate_lstm(model, X, y, tscv, metric, scaler, epochs)
    best_score = min(metric_scores) if metric == "mse" else max(metric_scores)
    
    return  {
    'best_score': best_score,
    'epochs': epochs,
    'learning_rate': learning_rate,
    'average_cv_score': np.mean(metric_scores),
    'std_cv_score': np.std(metric_scores),
    'model_details': model.get_config(),
    'suggested_next_steps': {
        'adjust_n_lag': 'Try increasing or decreasing n_lag.',
        'try_different_activation': 'Experiment with different activation functions.',
        'alter_architecture': 'Consider adding more LSTM layers or adjusting'
        ' the number of units.'
        }
    }

def _create_sequences(data: ArrayLike, n_lag: int) -> Tuple[ArrayLike, ArrayLike]:
    """
    Creates sequences from time series data for LSTM model input.
    
    Parameters:
    - data: np.ndarray. Time series data.
    - n_lag: int. Number of lag observations per sequence.
    
    Returns:
    - Tuple of np.ndarray: (X, y). X is the sequences for model input, y is 
      the target output.
    """
    X, y = [], []
    for i in range(n_lag, len(data)):
        X.append(data[i-n_lag:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def _build_lstm_model(n_lag: int, learning_rate: float, activation='relu'
                      ) -> _Model:
    """
    Builds and compiles an LSTM model based on the specified input shape 
    and learning rate.
    
    Parameters:
    - n_lag: int. Number of lag observations, defining the input shape.
    - learning_rate: float. Learning rate for the optimizer.
    - activation: str. Activation method, default='relu'.
    
    Returns:
    - Sequential. The compiled LSTM model.
    """
    model = Sequential([
        LSTM(50, activation=activation, input_shape=(n_lag, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='mean_squared_error')
    return model

def _cross_validate_lstm(
        model: _Sequential , 
        X: np.ndarray, y: np.ndarray, 
        tscv: TimeSeriesSplit, metric: str, 
        scaler: Union[MinMaxScaler, StandardScaler], 
        epochs: int) -> list:
    """
    Performs cross-validation on LSTM model with time series data.
    
    Parameters:
    - model: Sequential. The LSTM model to be evaluated.
    - X: np.ndarray. Input sequences.
    - y: np.ndarray. Target outputs.
    - tscv: TimeSeriesSplit. Cross-validator.
    - metric: str. Performance metric ('mse' or 'accuracy').
    - scaler: MinMaxScaler or StandardScaler. Scaler used for inverse transformation.
    - epochs: int. Number of epochs for training.
    
    Returns:
    - list. List of scores for each cross-validation split.
    """
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test, y_train, y_test = (X[train_index], X[test_index],
                                            y[train_index], y[test_index]
                                            )
        model.fit(X_train, y_train, epochs=epochs, verbose=0)
        predictions = model.predict(X_test)
        predictions_original = scaler.inverse_transform(predictions)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        score = mean_squared_error(y_test_original, predictions_original
                                   ) if metric == 'mse' else accuracy_score(
                                       y_test_original, np.round(predictions_original))
        scores.append(score)
    return scores

