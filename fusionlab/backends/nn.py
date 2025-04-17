# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio (a.k.a. @Daniel) <etanoyau@gmail.com>

"""
The NNBackend module provides a flexible and unified interface for working 
with neural network models using either TensorFlow or PyTorch as the backend. 
This module allows gofast users to perform deep learning tasks seamlessly, 
with the option to switch between backends based on availability or preference. 
It abstracts away backend-specific complexities, enabling users to focus on 
building and training models without worrying about framework-specific syntax.

Setup:
To use NNBackend, ensure that either TensorFlow or PyTorch is installed 
in your environment. The `auto` option will prioritize TensorFlow if both 
frameworks are available; otherwise, it will use PyTorch:
    
    pip install tensorflow
    # or
    pip install torch

Example Usage:

1. Initialize NNBackend:
    >>> from gofast.backends.nn import NNBackend
    >>> nn_backend = NNBackend(backend='auto')  
    # Automatically selects TensorFlow if available, else PyTorch

2. Create and Compile a Model:
    >>> model = nn_backend.sequential_model([
    ...     nn_backend.dense_layer(128, activation='relu'),
    ...     nn_backend.dense_layer(10, activation='softmax')
    ... ])
    >>> optimizer = nn_backend.optimizer(name='adam')
    >>> loss = nn_backend.loss_function(name='mse')
    >>> nn_backend.compile_model(model, optimizer, loss)

3. Perform Training Steps:
    # Assume `x_train` and `y_train` are batches of input and target data
    >>> for epoch in range(epochs):
    ...     loss_value = nn_backend.train_step(x_train, y_train)
    ...     print(f"Epoch {epoch+1}, Loss: {loss_value}")

4. Model Evaluation and Prediction:
    # Evaluate the model
    >>> evaluation_loss = nn_backend.evaluate(x_test, y_test)
    >>> print(f"Evaluation Loss: {evaluation_loss}")

    # Generate predictions
    >>> predictions = nn_backend.predict(x_test)

5. Save and Load the Model:
    # Save the model
    >>> nn_backend.save_model('my_model')

    # Load the model (PyTorch requires the model class)
    >>> model = nn_backend.load_model('my_model', model_class=MyModelClass)

6. Additional Features:
    - `activation`: Retrieve common activation functions such as ReLU, Sigmoid, 
      and Tanh.
    - `early_stopping_callback`: Provides early stopping for TensorFlow models 
      to prevent overfitting.
    - `custom_metric`: Add custom metrics for model performance monitoring.
    - `learning_rate_scheduler`: Set up learning rate schedules 
      (e.g., exponential decay, step decay) for both backends.
    - `log_training`: Enable TensorBoard logging for TensorFlow.

Note:
- The NNBackend module is designed to abstract the complexities associated with 
  TensorFlow and PyTorch, making it easy to switch between them. 
- This backend allows gofast users to implement deep learning workflows 
  flexibly and efficiently, regardless of the framework preference.
- For advanced customization not covered by NNBackend, users can directly 
  utilize TensorFlow or PyTorch APIs as needed.

Remember:
The flexibility of NNBackend is enhanced by its compatibility with two major 
deep learning frameworks, making it a powerful tool for any machine learning 
project within gofast.
"""

from ..api.property import BaseClass 
try:
    import tensorflow as tf 
except: pass 


class BackendNotAvailable(Exception):
    """Exception raised when the requested backend is not available."""
    pass

__all__=["NNBackend"]

class NNBackend( BaseClass):
    """
    Neural Network Backend handler class for managing backend-specific
    neural network operations. It provides flexibility to work with 
    either TensorFlow or PyTorch based on user preference or availability.

    The backend selection (`tensorflow` or `pytorch`) defines the 
    appropriate module for neural network operations, including model 
    layers, optimizers, and loss functions. If `auto` is specified, 
    the class automatically chooses TensorFlow if available; otherwise, 
    it selects PyTorch if installed.

    This flexibility enables users to create and train neural networks 
    without concern for backend-specific syntax differences. The 
    `NNBackend` class abstracts away these differences by providing a 
    unified API.

    Parameters
    ----------
    backend : str, optional
        The backend to use for neural network operations. Options are:
        - `'auto'`: Automatically selects TensorFlow if available, 
          otherwise uses PyTorch.
        - `'tensorflow'`: Forces the usage of TensorFlow backend.
        - `'pytorch'`: Forces the usage of PyTorch backend.
        Default is `'auto'`.
    init: bool, default=False, 
       Initialize the backend for selection. 
    verbose: bool, default=False 
       Output the verbosity messages. 
       
    Raises
    ------
    BackendNotAvailable
        Raised when the specified backend is not installed on the 
        system.

    Notes
    -----
    **Mathematical Formulation**:
    
    Consider a neural network layer function :math:`f(x)` applied on 
    input :math:`x`. For a dense layer, this function is defined as:
    
    .. math:: f(x) = W \cdot x + b

    where :math:`W` and :math:`b` are the weight matrix and bias vector,
    respectively. `NNBackend` facilitates the creation of such layers 
    while managing TensorFlow and PyTorch nuances.

    **Supported Methods**:
    
    - `set_backend`
    - `dense_layer`
    - `sequential_model`
    - `loss_function`
    - `optimizer`

    Examples
    --------
    >>> from gofast.backends.nn import NNBackend
    >>> nn_backend = NNBackend(backend='auto')
    >>> model = nn_backend.sequential_model([
    ...     nn_backend.dense_layer(128, activation='relu'),
    ...     nn_backend.dense_layer(10, activation='softmax')
    ... ])
    >>> optimizer = nn_backend.optimizer(name='adam')
    >>> loss_fn = nn_backend.loss_function(name='mse')

    See Also
    --------
    BaseBackend : The base class for other backend classes.
    TensorFlow and PyTorch Documentation for backend details.

    References
    ----------
    .. [1] Abadi, M., et al., "TensorFlow: Large-scale machine learning 
       on heterogeneous systems", 2015.
    .. [2] Paszke, A., et al., "PyTorch: An Imperative Style, 
       High-Performance Deep Learning Library", 2019.
    """

    def __init__(self, backend='auto', init=True, verbose=False):
        """
        Initializes the `NNBackend` class with the specified neural network 
        backend. The backend determines which deep learning library (TensorFlow 
        or PyTorch) is used for neural network operations.

        Parameters
        ----------
        backend : str, optional
            The backend to use for neural network operations 
            (`'auto'`, `'tensorflow'`, or `'pytorch'`). Default is `'auto'`.
        
        Raises
        ------
        BackendNotAvailable
            If the specified backend is not installed.
        """
        self.backend= backend 
        self.verbose=verbose 
        if init: 
            self._initialize_backend(self.backend)

    def _initialize_backend(self, backend):
        """
        Initializes the backend by setting the appropriate neural network 
        module, either TensorFlow or PyTorch.

        Parameters
        ----------
        backend : str
            The backend to use ('auto', 'tensorflow', or 'pytorch').

        Raises
        ------
        BackendNotAvailable
            If the specified backend is not installed.
        """
        if backend == 'auto':
            try:
                # Attempt to import TensorFlow as the preferred backend
                import tensorflow as tf
                self.nn_module = tf.keras
                self.backend = 'tensorflow'
                if self.verbose:
                    print("TensorFlow backend selected automatically.")
            except ImportError:
                try:
                    # Fallback to PyTorch if TensorFlow is unavailable
                    import torch
                    self.nn_module = torch.nn
                    self.backend = 'pytorch'
                    if self.verbose:
                        print("TensorFlow not found. PyTorch"
                              " backend selected automatically.")
                except ImportError:
                    raise BackendNotAvailable(
                        "Neither TensorFlow nor PyTorch is installed. "
                        "Please install one of them to use NNBackend."
                    )
        elif backend == 'tensorflow':
            try:
                import tensorflow as tf
                self.nn_module = tf.keras
                self.backend = 'tensorflow'
            except ImportError:
                raise BackendNotAvailable(
                    "TensorFlow is not installed. Please install"
                    " it to use the TensorFlow backend."
                )
        elif backend == 'pytorch':
            try:
                import torch
                self.nn_module = torch.nn
                self.backend = 'pytorch'
            except ImportError:
                raise BackendNotAvailable(
                    "PyTorch is not installed. Please install"
                    " it to use the PyTorch backend."
                )
        else:
            raise ValueError("Invalid backend specified. Choose either"
                             " 'auto', 'tensorflow', or 'pytorch'.")

    def set_backend(self, backend):
        """
        Switch to a different backend dynamically.

        Parameters
        ----------
        backend : str
            The backend to switch to ('tensorflow' or 'pytorch').

        Examples
        --------
        >>> nn_backend = NNBackend()
        >>> nn_backend.set_backend('pytorch')
        """
        self._initialize_backend(backend)

    def dense_layer(self, *args, **kwargs):
        """
        Create a dense (fully connected) layer for the neural network 
        using the selected backend.

        Parameters
        ----------
        *args : positional arguments
            Positional arguments passed to the layer constructor.
        **kwargs : keyword arguments
            Keyword arguments passed to the layer constructor.

        Returns
        -------
        layer : tf.keras.layers.Dense or torch.nn.Linear
            Dense layer object for the respective backend.

        Examples
        --------
        >>> nn_backend = NNBackend()
        >>> layer = nn_backend.dense_layer(128, activation='relu')
        """
        if self.backend == 'tensorflow':
            return self.nn_module.layers.Dense(*args, **kwargs)
        elif self.backend == 'pytorch':
            return self.nn_module.Linear(*args, **kwargs)
        else:
            raise RuntimeError("Unsupported backend.")

    def sequential_model(self, layers):
        """
        Create a sequential model using the selected backend.

        Parameters
        ----------
        layers : list
            List of layers to add to the sequential model.

        Returns
        -------
        model : tf.keras.Sequential or torch.nn.Sequential
            A sequential model for the respective backend.

        Examples
        --------
        >>> nn_backend = NNBackend()
        >>> model = nn_backend.sequential_model([
        ...     nn_backend.dense_layer(128, activation='relu'),
        ...     nn_backend.dense_layer(10, activation='softmax')
        ... ])
        """
        if self.backend == 'tensorflow':
            return self.nn_module.Sequential(layers)
        elif self.backend == 'pytorch':
            return self.nn_module.Sequential(*layers)
        else:
            raise RuntimeError("Unsupported backend.")
            
    def loss_function(self, name='mse'):
        """
        Retrieve a loss function based on the specified name and backend.
        Supports a variety of commonly used loss functions for both 
        TensorFlow and PyTorch backends.

        Parameters
        ----------
        name : str
            Name of the loss function to retrieve. Supported loss 
            functions include:
            - `'mse'`: Mean Squared Error
            - `'mae'`: Mean Absolute Error
            - `'cross_entropy'`: Cross Entropy Loss
            - `'binary_crossentropy'`: Binary Cross-Entropy
            - `'categorical_crossentropy'`: Categorical Cross-Entropy
            - `'hinge'`: Hinge Loss (for binary classification)
            - `'huber'`: Huber Loss (smooth L1 loss)
            - `'kld'`: Kullback-Leibler Divergence

        Returns
        -------
        loss : tf.keras.losses.Loss or torch.nn.Module
            The corresponding loss function for the specified backend.

        Raises
        ------
        RuntimeError
            If the specified backend is unsupported.
        ValueError
            If the specified loss function is not found.

        Examples
        --------
        >>> nn_backend = NNBackend()
        >>> loss = nn_backend.loss_function(name='binary_crossentropy')

        Notes
        -----
        - In TensorFlow, loss functions are accessed directly from 
          `tf.keras.losses`.
        - In PyTorch, the loss functions are initialized as instances of 
          `torch.nn.Module`.
        - Additional loss functions may be added as needed.

        """
        # TensorFlow Loss Functions
        if self.backend == 'tensorflow':
            tf_loss_functions = {
                'mse': 'MeanSquaredError',
                'mae': 'MeanAbsoluteError',
                'cross_entropy': 'SparseCategoricalCrossentropy',
                'binary_crossentropy': 'BinaryCrossentropy',
                'categorical_crossentropy': 'CategoricalCrossentropy',
                'hinge': 'Hinge',
                'huber': 'Huber',
                'kld': 'KLDivergence',
            }
            loss_name = tf_loss_functions.get(name.lower())
            if loss_name:
                return getattr(self.nn_module.losses, loss_name)()
            else:
                raise ValueError(
                    f"Loss function '{name}' is not supported in TensorFlow.")

        # PyTorch Loss Functions
        elif self.backend == 'pytorch':
            import torch.nn as nn
            pytorch_loss_functions = {
                'mse': nn.MSELoss(),
                'mae': nn.L1Loss(),
                'cross_entropy': nn.CrossEntropyLoss(),
                'binary_crossentropy': nn.BCEWithLogitsLoss(),
                'categorical_crossentropy': nn.CrossEntropyLoss(),
                'hinge': nn.HingeEmbeddingLoss(),
                'huber': nn.SmoothL1Loss(),
                'kld': nn.KLDivLoss(),
            }
            loss_function = pytorch_loss_functions.get(name.lower())
            if loss_function:
                return loss_function
            else:
                raise ValueError(
                    f"Loss function '{name}' is not supported in PyTorch.")

        else:
            raise RuntimeError("Unsupported backend.")

    def optimizer(self, name='adam', **kwargs):
        """
        Retrieve an optimizer based on the specified name and backend.
        Supports a variety of commonly used optimizers for both TensorFlow 
        and PyTorch backends.

        Parameters
        ----------
        name : str
            The name of the optimizer to retrieve. Supported optimizers include:
            - `'adam'`: Adaptive Moment Estimation
            - `'sgd'`: Stochastic Gradient Descent
            - `'rmsprop'`: RMSProp optimizer
            - `'adagrad'`: Adagrad optimizer
            - `'adamax'`: AdaMax optimizer
            - `'nadam'`: Nesterov-accelerated Adaptive Moment Estimation
        kwargs : dict
            Additional parameters for the optimizer (e.g., `learning_rate`, 
            `momentum` for SGD). Refer to TensorFlow and PyTorch documentation 
            for full options.

        Returns
        -------
        optimizer : tf.keras.optimizers.Optimizer or torch.optim.Optimizer
            Optimizer for the respective backend, initialized with any specified 
            `kwargs`.

        Raises
        ------
        RuntimeError
            If the specified backend is unsupported.
        ValueError
            If the specified optimizer is not found.

        Examples
        --------
        >>> nn_backend = NNBackend()
        >>> optimizer = nn_backend.optimizer(name='adam', learning_rate=0.001)

        Notes
        -----
        - In TensorFlow, optimizers are accessed from `tf.keras.optimizers`.
        - In PyTorch, optimizers are initialized from `torch.optim`, and require 
          `model.parameters()` as input.
        - Additional optimizers can be added as needed.

        """
        # TensorFlow Optimizers
        if self.backend == 'tensorflow':
            tf_optimizers = {
                'adam': 'Adam',
                'sgd': 'SGD',
                'rmsprop': 'RMSprop',
                'adagrad': 'Adagrad',
                'adamax': 'Adamax',
                'nadam': 'Nadam',
            }
            optimizer_name = tf_optimizers.get(name.lower())
            if optimizer_name:
                return getattr(self.nn_module.optimizers, optimizer_name)(**kwargs)
            else:
                raise ValueError(f"Optimizer '{name}' is not supported in TensorFlow.")

        # PyTorch Optimizers
        elif self.backend == 'pytorch':
            import torch.optim as optim
            pytorch_optimizers = {
                'adam': optim.Adam,
                'sgd': optim.SGD,
                'rmsprop': optim.RMSprop,
                'adagrad': optim.Adagrad,
                'adamax': optim.Adamax,
                'nadam': optim.NAdam,  # Available in PyTorch >= 1.10
            }
            optimizer_class = pytorch_optimizers.get(name.lower())
            if optimizer_class:
                return optimizer_class(self.model.parameters(), **kwargs)
            else:
                raise ValueError(f"Optimizer '{name}' is not supported in PyTorch.")

        else:
            raise RuntimeError("Unsupported backend.")

      
    def evaluate(self, x, y):
        """
        Evaluate the model on the provided dataset and return the computed loss.
        This method supports both TensorFlow and PyTorch backends.

        Parameters
        ----------
        x : tf.Tensor or torch.Tensor
            Input data to evaluate the model.
        y : tf.Tensor or torch.Tensor
            Target data corresponding to the input data.

        Returns
        -------
        loss_value : float
            The computed loss value for the evaluation.

        Examples
        --------
        >>> nn_backend = NNBackend(backend='auto')
        >>> loss_value = nn_backend.evaluate(x_test, y_test)
        >>> print(f"Evaluation Loss: {loss_value}")

        Notes
        -----
        Model evaluation allows users to assess the performance of the 
        model on a validation or test dataset. In PyTorch, the model is 
        set to evaluation mode (`eval()`) to disable dropout layers and 
        other training-specific behaviors.
        """
        if self.backend == 'tensorflow':
            return self.model.evaluate(x, y, verbose=0)
        elif self.backend == 'pytorch':
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient computation for evaluation
                predictions = self.model(x)
                loss_value = self.loss_fn(predictions, y)
            return loss_value.item()
        else:
            raise RuntimeError("Unsupported backend.")

    def predict(self, x):
        """
        Generate predictions from the model on the provided input data.
        This method is compatible with both TensorFlow and PyTorch backends.

        Parameters
        ----------
        x : tf.Tensor or torch.Tensor
            Input data on which to generate predictions.

        Returns
        -------
        predictions : tf.Tensor or torch.Tensor
            Model predictions generated from the input data.

        Examples
        --------
        >>> nn_backend = NNBackend(backend='tensorflow')
        >>> predictions = nn_backend.predict(x_test)
        >>> print(predictions)

        Notes
        -----
        In PyTorch, the model is set to evaluation mode and gradient 
        computation is disabled to ensure efficient and accurate inference.
        """
        if self.backend == 'tensorflow':
            return self.model.predict(x)
        elif self.backend == 'pytorch':
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient computation for inference
                predictions = self.model(x)
            return predictions
        else:
            raise RuntimeError("Unsupported backend.")

    def activation(self, name='relu'):
        """
        Retrieve an activation function based on the specified name and 
        backend. Supports a variety of commonly used activation functions 
        for both TensorFlow and PyTorch backends.

        Parameters
        ----------
        name : str
            The name of the activation function to retrieve. Supported 
            activations include:
            - `'relu'`: Rectified Linear Unit
            - `'sigmoid'`: Sigmoid function
            - `'tanh'`: Hyperbolic Tangent
            - `'softmax'`: Softmax function
            - `'softplus'`: Softplus function
            - `'leaky_relu'`: Leaky Rectified Linear Unit
            - `'elu'`: Exponential Linear Unit
            - `'selu'`: Scaled Exponential Linear Unit
            - `'swish'`: Swish function (TensorFlow only)

        Returns
        -------
        activation : function or tf.keras.layers.Layer or torch.nn.Module
            The corresponding activation function for the specified backend.

        Raises
        ------
        RuntimeError
            If the specified backend is unsupported.
        ValueError
            If the specified activation function is not found.

        Examples
        --------
        >>> nn_backend = NNBackend(backend='auto')
        >>> activation_fn = nn_backend.activation(name='leaky_relu')

        Notes
        -----
        - Activation functions introduce non-linearity in neural networks, 
          enabling the model to learn complex patterns.
        - In TensorFlow, activations are accessed from `tf.keras.activations`.
        - In PyTorch, activations are defined as separate modules in `torch.nn`.
        - Additional activations can be added as needed.
        """
        # TensorFlow Activations
        if self.backend == 'tensorflow':
            tf_activations = {
                'relu': 'relu',
                'sigmoid': 'sigmoid',
                'tanh': 'tanh',
                'softmax': 'softmax',
                'softplus': 'softplus',
                'leaky_relu': 'leaky_relu',
                'elu': 'elu',
                'selu': 'selu',
                'swish': 'swish',  # TensorFlow-only activation
            }
            activation_name = tf_activations.get(name.lower())
            if activation_name:
                return getattr(self.nn_module.activations, activation_name)
            else:
                raise ValueError(
                    f"Activation '{name}' is not supported in TensorFlow.")

        # PyTorch Activations
        elif self.backend == 'pytorch':
            import torch.nn as nn
            pytorch_activations = {
                'relu': nn.ReLU(),
                'sigmoid': nn.Sigmoid(),
                'tanh': nn.Tanh(),
                'softmax': nn.Softmax(dim=1),  # Softmax requires dim argument in PyTorch
                'softplus': nn.Softplus(),
                'leaky_relu': nn.LeakyReLU(),
                'elu': nn.ELU(),
                'selu': nn.SELU(),
                # Swish is not natively supported in PyTorch
            }
            activation_fn = pytorch_activations.get(name.lower())
            if activation_fn:
                return activation_fn
            else:
                raise ValueError(f"Activation '{name}' is not supported in PyTorch.")

        else:
            raise RuntimeError("Unsupported backend.")


    def compile_model(self, model, optimizer, loss):
        """
        Compile the model with the specified optimizer and loss function.
        This method stores the optimizer and loss function for both 
        TensorFlow and PyTorch backends.

        Parameters
        ----------
        model : tf.keras.Model or torch.nn.Module
            Model to compile for training.
        optimizer : tf.keras.optimizers.Optimizer or torch.optim.Optimizer
            Optimizer for training the model.
        loss : tf.keras.losses.Loss or torch.nn.Module
            Loss function for model optimization.

        Raises
        ------
        RuntimeError
            If the backend is unsupported.

        Examples
        --------
        >>> nn_backend = NNBackend(backend='tensorflow')
        >>> model = nn_backend.sequential_model([
        ...     nn_backend.dense_layer(128, activation='relu'),
        ...     nn_backend.dense_layer(10, activation='softmax')
        ... ])
        >>> optimizer = nn_backend.optimizer(name='adam')
        >>> loss = nn_backend.loss_function(name='mse')
        >>> nn_backend.compile_model(model, optimizer, loss)

        Notes
        -----
        - In TensorFlow, model compilation finalizes the model with 
          specified `optimizer` and `loss` and prepares it for training.
        - In PyTorch, explicit compilation is not required; hence, 
          the `optimizer` and `loss` are stored as attributes for 
          later use during training.
        """
        self.model = model  # Store model in the backend for compatibility
        if self.backend == 'tensorflow':
            model.compile(optimizer=optimizer, loss=loss)
        elif self.backend == 'pytorch':
            # For PyTorch, model compilation is handled manually, 
            # so we store optimizer and loss function
            self.optimizer = optimizer
            self.loss_fn = loss
        else:
            raise RuntimeError("Unsupported backend.")

    def train_step(self, x, y):
        """
        Perform a single training step, including forward pass, loss 
        computation, backward pass, and optimizer step. This method 
        supports both TensorFlow and PyTorch backends.

        Parameters
        ----------
        x : tf.Tensor or torch.Tensor
            Input data for the model.
        y : tf.Tensor or torch.Tensor
            Target labels corresponding to the input data.

        Returns
        -------
        loss_value : float
            The computed loss for the training step.

        Raises
        ------
        RuntimeError
            If the backend is unsupported.

        Examples
        --------
        >>> nn_backend = NNBackend(backend='pytorch')
        >>> loss_value = nn_backend.train_step(x_train, y_train)
        >>> print(f"Training Loss: {loss_value}")

        Notes
        -----
        In a training step:
        1. **Forward Pass**: Data is passed through the model to compute 
           predictions.
        2. **Loss Calculation**: The difference between predictions and 
           actual values is computed via the specified loss function.
        3. **Backward Pass**: Gradients are calculated with respect to 
           model parameters.
        4. **Optimizer Step**: The optimizer updates model parameters 
           based on gradients to minimize loss.

        In TensorFlow, `GradientTape` is used to record operations for 
        automatic differentiation, while PyTorch handles this via 
        `zero_grad()`, `backward()`, and `step()` calls.
        """
        if self.backend == 'tensorflow':
            # TensorFlow training step with GradientTape
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=True)
                loss_value = self.loss_fn(y, predictions)
            # Calculate gradients and apply optimizer
            gradients = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss_value.numpy()  # Return loss as float for consistency
        elif self.backend == 'pytorch':
            # PyTorch training step
            self.model.train()  # Set model to training mode
            self.optimizer.zero_grad()  # Clear previous gradients
            predictions = self.model(x)  # Forward pass
            loss_value = self.loss_fn(predictions, y)  # Compute loss
            loss_value.backward()  # Backward pass to calculate gradients
            self.optimizer.step()  # Update parameters with optimizer
            return loss_value.item()  # Return loss as float for consistency
        else:
            raise RuntimeError("Unsupported backend.")

    def save_model(self, filepath):
        """
        Save the trained model to the specified file path. This method 
        supports saving for both TensorFlow and PyTorch backends.

        Parameters
        ----------
        filepath : str
            Path where the model should be saved.

        Examples
        --------
        >>> nn_backend = NNBackend(backend='pytorch')
        >>> nn_backend.save_model('my_model.pth')

        Notes
        -----
        In TensorFlow, the entire model architecture, weights, and 
        optimizer state are saved. In PyTorch, only the model state 
        dictionary is saved, which requires re-instantiation of the 
        model class upon loading.
        """
        if self.backend == 'tensorflow':
            self.model.save(filepath)
        elif self.backend == 'pytorch':
            import torch
            torch.save(self.model.state_dict(), filepath)
        else:
            raise RuntimeError("Unsupported backend.")

    def load_model(self, filepath, model_class):
        """
        Load a model from the specified file path. For PyTorch, the 
        `model_class` must be specified to instantiate the model object 
        before loading the state dictionary.

        Parameters
        ----------
        filepath : str
            Path from which the model should be loaded.
        model_class : class
            The model class to instantiate (required for PyTorch). For 
            TensorFlow, this parameter is not required.

        Returns
        -------
        model : tf.keras.Model or torch.nn.Module
            The loaded model object for further use.

        Examples
        --------
        TensorFlow:
        >>> nn_backend = NNBackend(backend='tensorflow')
        >>> model = nn_backend.load_model('my_model.h5')

        PyTorch:
        >>> nn_backend = NNBackend(backend='pytorch')
        >>> model = nn_backend.load_model('my_model.pth', model_class=MyModelClass)

        Notes
        -----
        TensorFlow loads the complete model (architecture, weights, and 
        optimizer state), while PyTorch only loads the model's weights, 
        which requires the model class definition.
        """
        if self.backend == 'tensorflow':
            from tensorflow.keras.models import load_model
            self.model = load_model(filepath)
        elif self.backend == 'pytorch':
            import torch
            self.model = model_class()  # Instantiate model class
            self.model.load_state_dict(torch.load(filepath))
            self.model.eval()  # Set to evaluation mode
        else:
            raise RuntimeError("Unsupported backend.")
        return self.model

    def early_stopping_callback(self, patience=5):
        """
        Get an early stopping callback for TensorFlow, which stops training 
        if validation loss does not improve after a specified number of 
        epochs. This feature is not supported for PyTorch directly.

        Parameters
        ----------
        patience : int, optional
            Number of epochs with no improvement after which training 
            will be stopped. Default is 5.

        Returns
        -------
        callback : tf.keras.callbacks.Callback
            The early stopping callback for TensorFlow.

        Raises
        ------
        RuntimeError
            If the backend is not TensorFlow.

        Examples
        --------
        >>> nn_backend = NNBackend(backend='tensorflow')
        >>> early_stopping = nn_backend.early_stopping_callback(patience=3)
        >>> model.fit(x_train, y_train, epochs=10, callbacks=[early_stopping])

        Notes
        -----
        Early stopping is a regularization technique that prevents 
        overfitting by halting training once the model performance stops 
        improving on the validation dataset.
        """
        if self.backend == 'tensorflow':
            from tensorflow.keras.callbacks import EarlyStopping
            return EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True)
        else:
            raise RuntimeError("Early stopping is only available in TensorFlow.")

    def custom_metric(self, name, func):
        """
        Add a custom metric to the model. The custom metric can be used 
        to monitor model performance during training. This feature is 
        available for both TensorFlow and PyTorch backends.

        Parameters
        ----------
        name : str
            Name of the custom metric.
        func : function
            The function defining the metric. It should take `y_true` and 
            `y_pred` as inputs and return a computed metric value.

        Returns
        -------
        metric : tf.keras.metrics.Metric or function
            The custom metric function or layer.

        Raises
        ------
        RuntimeError
            If the backend is not supported.

        Examples
        --------
        >>> nn_backend = NNBackend(backend='auto')
        >>> def custom_accuracy(y_true, y_pred):
        ...     return np.mean(y_true == np.round(y_pred))
        >>> metric = nn_backend.custom_metric(name='custom_accuracy', func=custom_accuracy)

        Notes
        -----
        In TensorFlow, the custom metric function can be added to the 
        model during compilation. In PyTorch, custom metrics must be 
        calculated manually during training.
        """
        if self.backend == 'tensorflow':
            from tensorflow.keras.metrics import Metric
            return Metric(name=name, metric_fn=func)
        elif self.backend == 'pytorch':
            # In PyTorch, the custom metric function is manually applied 
            # in the training loop.
            return func
        else:
            raise RuntimeError("Unsupported backend.")

    def log_training(self, log_dir='./logs'):
        """
        Set up TensorBoard logging for monitoring training progress. 
        This feature is available only for the TensorFlow backend.

        Parameters
        ----------
        log_dir : str, optional
            Directory where the logs should be saved for TensorBoard 
            visualization. Default is './logs'.

        Returns
        -------
        callback : tf.keras.callbacks.TensorBoard or None
            The TensorBoard callback for TensorFlow. None for PyTorch, 
            as TensorBoard logging is handled differently.

        Raises
        ------
        RuntimeError
            If the backend is not TensorFlow.

        Examples
        --------
        >>> nn_backend = NNBackend()
        >>> tensorboard_logging = nn_backend.log_training(log_dir='./my_logs')
        >>> model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_logging])

        Notes
        -----
        TensorBoard is a visualization tool provided by TensorFlow. It 
        can track metrics like loss and accuracy, as well as provide 
        graphical representation of the model architecture and gradients.
        """
        if self.backend == 'tensorflow':
            from tensorflow.keras.callbacks import TensorBoard
            return TensorBoard(log_dir=log_dir)
        else:
            raise RuntimeError("TensorBoard logging is only supported in TensorFlow.")

    def learning_rate_scheduler(self, schedule_type='exponential', **kwargs):
        """
        Set up a learning rate scheduler to adjust the learning rate 
        during training. This feature is supported for both TensorFlow 
        and PyTorch backends.

        Parameters
        ----------
        schedule_type : str
            The type of scheduler to use ('exponential' or 'step_decay').
        kwargs : dict
            Additional arguments specific to the schedule type.

            For 'exponential':
            - `decay_rate` (float): The factor by which the learning rate 
              is reduced at each epoch.

            For 'step_decay':
            - `drop_rate` (float): The factor by which the learning rate 
              is reduced at each step.
            - `epochs_drop` (int): The number of epochs after which the 
              learning rate is reduced.

        Returns
        -------
        scheduler : tf.keras.callbacks.Callback or torch.optim.lr_scheduler
            The learning rate scheduler for the respective backend.

        Raises
        ------
        RuntimeError
            If the backend is not supported.

        Examples
        --------
        >>> nn_backend = NNBackend()
        >>> scheduler = nn_backend.learning_rate_scheduler(
        ...     schedule_type='step_decay', drop_rate=0.5, epochs_drop=10)
        
        For TensorFlow:
        >>> model.fit(x_train, y_train, epochs=10, callbacks=[scheduler])

        For PyTorch:
        >>> for epoch in range(epochs):
        ...     train(...)  # training loop
        ...     scheduler.step()

        Notes
        -----
        Learning rate scheduling is a common technique to improve 
        model convergence. Gradually reducing the learning rate can 
        lead to better model performance as it allows the model to 
        converge to a better solution over time.
        """
        if self.backend == 'tensorflow':
            from tensorflow.keras.callbacks import LearningRateScheduler
            if schedule_type == 'exponential':
                def exponential_decay(epoch, lr):
                    decay_rate = kwargs.get('decay_rate', 0.96)
                    return lr * decay_rate
                return LearningRateScheduler(exponential_decay)
            elif schedule_type == 'step_decay':
                def step_decay(epoch, lr):
                    drop_rate = kwargs.get('drop_rate', 0.5)
                    epochs_drop = kwargs.get('epochs_drop', 10)
                    return lr * (drop_rate ** (epoch // epochs_drop))
                return LearningRateScheduler(step_decay)
        elif self.backend == 'pytorch':
            import torch.optim.lr_scheduler as lr_scheduler
            if schedule_type == 'exponential':
                return lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=kwargs.get('gamma', 0.96))
            elif schedule_type == 'step_decay':
                return lr_scheduler.StepLR(
                    self.optimizer, step_size=kwargs.get('step_size', 10), 
                    gamma=kwargs.get('gamma', 0.5))
        else:
            raise RuntimeError("Unsupported backend.")

    def checkpoint_callback(self, filepath, monitor='val_loss'):
        """
        Create a model checkpoint callback to save the model during 
        training. This feature is available for TensorFlow, while a 
        manual checkpoint function is provided for PyTorch.

        Parameters
        ----------
        filepath : str
            Path to save the model checkpoint file.
        monitor : str, optional
            Metric to monitor for saving the best model. Default is 
            'val_loss'.

        Returns
        -------
        callback : tf.keras.callbacks.ModelCheckpoint or function
            Checkpoint callback for TensorFlow. For PyTorch, it returns 
            a function that can be called manually to save the model.

        Raises
        ------
        RuntimeError
            If the backend is not supported.

        Examples
        --------
        TensorFlow:
        >>> nn_backend = NNBackend(backend='tensorflow')
        >>> checkpoint_callback = nn_backend.checkpoint_callback(filepath='best_model.h5')
        >>> model.fit(x_train, y_train, epochs=10, callbacks=[checkpoint_callback])

        PyTorch:
        >>> nn_backend = NNBackend(backend='pytorch')
        >>> save_checkpoint = nn_backend.checkpoint_callback(filepath='best_model.pth')
        >>> for epoch in range(epochs):
        ...     # Training code
        ...     save_checkpoint(epoch, nn_backend.model)

        Notes
        -----
        Model checkpointing is a critical practice for saving the best 
        model during training to avoid overfitting. It is especially 
        useful for large models trained over multiple epochs.
        """
        if self.backend == 'tensorflow':
            from tensorflow.keras.callbacks import ModelCheckpoint
            return ModelCheckpoint(filepath=filepath, monitor=monitor, save_best_only=True)
        elif self.backend == 'pytorch':
            def save_checkpoint(epoch, model):
                import torch
                torch.save(model.state_dict(), f"{filepath}_epoch_{epoch}.pth")
            return save_checkpoint
        else:
            raise RuntimeError("Unsupported backend.")

    def summary(self, input_shape=None):
        """
        Display a summary of the model architecture, including each layer's 
        type, output shape, and number of parameters. This method is natively 
        supported in TensorFlow and requires the `torchsummary` package in 
        PyTorch.

        Parameters
        ----------
        input_shape : tuple, optional
            The shape of the input data (only required for PyTorch models). 
            Must be provided in the format (channels, height, width) for 
            PyTorch, or (height, width, channels) for TensorFlow.

        Raises
        ------
        RuntimeError
            If the specified backend is unsupported or if input_shape is 
            not provided for PyTorch.
        ImportError
            If `torchsummary` is not installed for PyTorch backend.

        Examples
        --------
        >>> nn_backend = NNBackend(backend='tensorflow')
        >>> model = nn_backend.sequential_model([
        ...     nn_backend.dense_layer(128, activation='relu'),
        ...     nn_backend.dense_layer(10, activation='softmax')
        ... ])
        >>> nn_backend.compile_model(model, optimizer=nn_backend.optimizer('adam'), 
        ...                          loss=nn_backend.loss_function('mse'))
        >>> nn_backend.summary(input_shape=(28, 28, 1))  # For TensorFlow

        Notes
        -----
        - In TensorFlow, `model.summary()` provides an easy-to-read model 
          summary directly.
        - In PyTorch, `torchsummary` provides a similar summary, and it 
          requires an `input_shape` parameter to compute the layer-wise 
          dimensions.
        - Ensure `torchsummary` is installed by running `pip install torchsummary`.

        """
        if self.backend == 'tensorflow':
            self.model.summary()
        
        elif self.backend == 'pytorch':
            if input_shape is None:
                raise RuntimeError("Input shape must be provided for PyTorch models.")
            try:
                from torchsummary import summary
                summary(self.model, input_shape)
            except ImportError:
                raise ImportError(
                    "To display a summary for PyTorch models, please install "
                    "`torchsummary` using `pip install torchsummary`."
                )

        else:
            raise RuntimeError("Unsupported backend.")

# Example usage in a full training loop
if __name__ == "__main__":
    # Initialize backend
    nn_backend = NNBackend(backend='tensorflow')
    
    # Define a model
    model = nn_backend.sequential_model([
        nn_backend.dense_layer(128, activation='relu'),
        nn_backend.dense_layer(10, activation='softmax')
    ])

    # Set optimizer, loss, and compile
    optimizer = nn_backend.optimizer(name='adam')
    loss = nn_backend.loss_function(name='cross_entropy')
    nn_backend.compile_model(model, optimizer, loss)

    # Add early stopping and tensorboard logging callbacks (TensorFlow only)
    early_stopping = nn_backend.early_stopping_callback(patience=3)
    tensorboard_logging = nn_backend.log_training(log_dir='./logs')

    # Mock training loop with logging and early stopping
    import tensorflow as tf # noqa 
    import numpy as np

    x_train = np.random.random((100, 20))
    y_train = np.random.randint(0, 10, (100,))

    # Convert to TensorFlow datasets for use with fit method
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16)

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=10,
        callbacks=[early_stopping, tensorboard_logging]
    )

    nn_backend = NNBackend(backend='pytorch')
    
    # Define a model
    model = nn_backend.sequential_model([
        nn_backend.dense_layer(128),
        nn_backend.activation('relu'),
        nn_backend.dense_layer(10)
    ])
    
    # Set optimizer and loss
    optimizer = nn_backend.optimizer(name='adam')
    loss = nn_backend.loss_function(name='mse')
    
    # Compile model
    nn_backend.compile_model(model, optimizer, loss)

    # Mock data
    import torch
    x = torch.rand(32, 10)  # example input
    y = torch.rand(32, 10)  # example output

    # Training step
    train_loss = nn_backend.train_step(x, y)
    print(f"Train Loss: {train_loss}")

    # Evaluation
    eval_loss = nn_backend.evaluate(x, y)
    print(f"Eval Loss: {eval_loss}")

    # Prediction
    predictions = nn_backend.predict(x)
    print(predictions)

    nn_backend = NNBackend(backend='tensorflow')
    model = nn_backend.sequential_model([
        nn_backend.dense_layer(128, activation='relu'),
        nn_backend.dense_layer(10, activation='softmax')
    ])
    print(model)
