#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Module for NN hyperparameter tuning and optimization.
"""
from __future__ import annotations 
import copy 
import warnings 
import numpy as np

from ..api.types import List, Dict, Tuple
from ..api.types import ArrayLike , Callable, Any
from ..api.property import BaseClass
from ..decorators import smartFitRun
from ..utils.deps_utils import ensure_pkg

from . import KERAS_DEPS, KERAS_BACKEND,  dependency_message

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
    'Hyperband', 'PBTTrainer',
    
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
    >>> from fusionlab.nn.tune import PBTTrainer
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
        >>> from fusionlab.nn.tune import PBTTrainer
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
    >>> from fusionlab.nn.tune import Hyperband
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
        >>> from fusionlab.nn.tune import Hyperband
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

