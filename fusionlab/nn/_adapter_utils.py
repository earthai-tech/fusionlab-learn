# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides compatibility decorators and helper functions to adapt fusionlab
models for use with standard scikit-learn interfaces and pipelines.
"""

import numpy as np
from functools import wraps
from typing import List, Any, Dict

# --- Step 1: Create Helper Functions for Splitting Logic ---

def _split_flat_array_for_tft(
    X_flat: np.ndarray, model_instance: Any
) -> List[np.ndarray]:
    """
    Reshapes a single flat array back into the [static, dynamic] list
    for the legacy TFT model.
    """
    # Extract dimensions from the model instance
    num_static = getattr(model_instance, 'num_static_vars', 0)
    dim_static = getattr(model_instance, 'static_input_dim', 0)
    horizon = getattr(model_instance, 'forecast_horizon', 0)
    num_dynamic = getattr(model_instance, 'num_dynamic_vars', 0)
    dim_dynamic = getattr(model_instance, 'dynamic_input_dim', 0)

    # Calculate the expected split point
    static_size = num_static * dim_static
    expected_total_size = static_size + (horizon * num_dynamic * dim_dynamic)

    if X_flat.shape[1] != expected_total_size:
        raise ValueError(
            f"Input X has {X_flat.shape[1]} features, but TFT model expects"
            f" {expected_total_size} based on its configuration."
        )

    # Split and reshape
    X_static = X_flat[:, :static_size].reshape(
        (-1, num_static, dim_static))
    X_dynamic = X_flat[:, static_size:].reshape(
        (-1, horizon, num_dynamic, dim_dynamic))

    return [X_static, X_dynamic]

def _split_flat_array_for_base_attentive(
    X_flat: np.ndarray, model_instance: Any
) -> List[np.ndarray]:
    """
    Reshapes a single flat array back into the [static, dynamic, future]
    list for BaseAttentive-based models (XTFT, HALNet, PIHALNet, etc.).
    """
    # Extract dimensions from the model instance
    dim_static = getattr(model_instance, 'static_input_dim', 0)
    dim_dynamic = getattr(model_instance, 'dynamic_input_dim', 0)
    dim_future = getattr(model_instance, 'future_input_dim', 0)
    past_steps = getattr(model_instance, 'max_window_size', 0)
    horizon = getattr(model_instance, 'forecast_horizon', 0)

    # Calculate split points based on original, unflattened shapes
    size_dynamic = past_steps * dim_dynamic
    
    # This logic assumes 'tft_like' mode for future features
    size_future = (past_steps + horizon) * dim_future

    expected_total_size = dim_static + size_dynamic + size_future
    if X_flat.shape[1] != expected_total_size:
        raise ValueError(
            f"Input X has {X_flat.shape[1]} features, but model expects"
            f" {expected_total_size} based on its configuration."
        )

    # Split and reshape
    split1 = dim_static
    split2 = dim_static + size_dynamic
    
    X_static = X_flat[:, :split1]
    X_dynamic = X_flat[:, split1:split2].reshape((-1, past_steps, dim_dynamic))
    X_future = X_flat[:, split2:].reshape((-1, past_steps + horizon, dim_future))

    return [X_static, X_dynamic, X_future]

# --- Step 2: Create a Public Utility for Concatenation ---

def concatenate_fusionlab_inputs(
    input_list: List[np.ndarray]
) -> np.ndarray:
    """
    Flattens and concatenates a list of fusionlab inputs into a single
    2D array compatible with scikit-learn pipelines.

    Args:
        input_list (List[np.ndarray]): A list of input arrays, e.g.,
            [static_features, dynamic_features, future_features].

    Returns:
        np.ndarray: A single 2D array of shape (n_samples, n_features_total).
    """
    if not isinstance(input_list, (list, tuple)):
        raise TypeError("`input_list` must be a list or tuple of NumPy arrays.")
        
    flat_arrays = [
        arr.reshape(arr.shape[0], -1) for arr in input_list if arr is not None
    ]
    
    if not flat_arrays:
        return np.array([[]])

    return np.concatenate(flat_arrays, axis=1)

# --- Step 3: Create the New, Cleaner Decorator for Splitting ---

def adapt_sklearn_input(model_type: str):
    """
    A decorator that reshapes a scikit-learn style 2D `X` input array
    into the multi-input list required by fusionlab models before
    calling the decorated method (e.g., `fit` or `predict`).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # args[0] is always 'self' (the class instance)
            model_instance = args[0]
            
            # Find X from the function's arguments
            # This handles both `model.fit(X, y)` and `model.fit(X=X, y=y)`
            if 'X' in kwargs:
                X_input = kwargs.pop('X')
            elif len(args) > 1:
                X_input = args[1]
                # Rebuild args tuple without X to avoid duplication
                args = (model_instance,) + args[2:]
            else:
                # Handle cases where X might be missing entirely
                # Or pass through if no X is expected (e.g., fit())
                return func(*args, **kwargs)

            # If X is already in the correct list format, pass it through
            if isinstance(X_input, (list, tuple)):
                kwargs['X'] = X_input
                return func(*args, **kwargs)
                
            # If X is a single flat array, reshape it
            if isinstance(X_input, np.ndarray):
                if model_type == 'tft':
                    X_reshaped = _split_flat_array_for_tft(X_input, model_instance)
                elif model_type in ['xtft', 'halnet', 'pihalnet', 'transflow', 'base_attentive']:
                    X_reshaped = _split_flat_array_for_base_attentive(X_input, model_instance)
                else:
                    raise ValueError(f"Unsupported model_type for decorator: {model_type}")
                
                # Pass the reshaped X back into kwargs
                kwargs['X'] = X_reshaped
            else:
                # Pass through any other type of X
                kwargs['X'] = X_input

            # Call the original function (fit, predict) with the modified kwargs
            return func(*args, **kwargs)
        return wrapper
    return decorator

def compat_X(model_type='tft', ops='concat'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # args[0] is self if this is a method
            self = args[0]

            X = kwargs.get('X', None)
            y = kwargs.get('y', None)
            # If X is not in kwargs, maybe it's a positional argument
            # Assume the first positional after self is X, second is y
            if X is None and len(args) > 1:
                X = args[1]  
            
            # Extract attributes depending on model type
            if model_type == 'tft':
                # TFT attributes
                num_static_vars = self.num_static_vars
                num_dynamic_vars = self.num_dynamic_vars
                static_input_dim = self.static_input_dim
                dynamic_input_dim = self.dynamic_input_dim
                forecast_horizon = self.forecast_horizon
            elif model_type == 'xtft':
                # XTFT attributes
                forecast_horizons = self.forecast_horizons
                static_input_dim = self.static_input_dim
                dynamic_input_dim = self.dynamic_input_dim
                future_covariate_dim = self.future_covariate_dim

            # Perform concat or split
            if ops == 'concat':
                # For TFT: Expect [X_static, X_dynamic]
                # For XTFT: Expect [X_static, X_dynamic, X_future]
                if isinstance(X, (list, tuple)):
                    # Concat
                    try:
                        if model_type == 'tft':
                            # Expect two inputs
                            # X_static shape: (batch, num_static_vars, static_input_dim)
                            # X_dynamic shape: (batch, forecast_horizon, num_dynamic_vars, dynamic_input_dim)
                            # Flatten and concat
                            # static_size = num_static_vars * static_input_dim
                            # dynamic_size = forecast_horizon * num_dynamic_vars * dynamic_input_dim
                            # Final shape: (batch, static_size + dynamic_size)
                            X_concat = np.concatenate([x.reshape(x.shape[0], -1) for x in X], axis=1)
                            X = X_concat
                        else:
                            # XTFT
                            # Expect three inputs: [X_static, X_dynamic, X_future]
                            # static_input: (batch, static_input_dim)
                            # dynamic_input: (batch, forecast_horizons, dynamic_input_dim)
                            # future_covariate_input: (batch, forecast_horizons, future_covariate_dim)
                            # Flatten and concat
                            X_concat = np.concatenate([xx.reshape(xx.shape[0], -1) for xx in X], axis=1)
                            X = X_concat
                    except Exception as e:
                        raise ValueError(f"Error concatenating inputs: {e}")
                # else X is already a single array, do nothing
            elif ops == 'split':
                # For TFT: single X to [X_static, X_dynamic]
                # For XTFT: single X to [X_static, X_dynamic, X_future]
                if not isinstance(X, np.ndarray):
                    raise ValueError("For 'split', X must be a single numpy array.")

                if model_type == 'tft':
                    # static_size = num_static_vars * static_input_dim
                    # dynamic_size = forecast_horizon * num_dynamic_vars * dynamic_input_dim
                    static_size = num_static_vars * static_input_dim
                    dynamic_size = forecast_horizon * num_dynamic_vars * dynamic_input_dim
                    total_size = static_size + dynamic_size

                    if X.shape[1] != total_size:
                        raise ValueError(f"Expected {total_size} features, got {X.shape[1]}")

                    X_static = X[:, :static_size]
                    X_static = X_static.reshape((X_static.shape[0], num_static_vars, static_input_dim))

                    X_dynamic = X[:, static_size:]
                    X_dynamic = X_dynamic.reshape((X_dynamic.shape[0], forecast_horizon, num_dynamic_vars, dynamic_input_dim))

                    X = [X_static, X_dynamic]

                else:
                    # XTFT
                    # static_size = static_input_dim
                    # dynamic_size = forecast_horizons * dynamic_input_dim
                    # future_size = forecast_horizons * future_covariate_dim
                    static_size = static_input_dim
                    dynamic_size = forecast_horizons * dynamic_input_dim
                    future_size = forecast_horizons * future_covariate_dim
                    total_expected = static_size + dynamic_size + future_size

                    if X.shape[1] != total_expected:
                        raise ValueError(f"Expected {total_expected} features, got {X.shape[1]}")

                    X_static = X[:, :static_size]
                    X_static = X_static.reshape((X_static.shape[0], static_input_dim))

                    X_dynamic = X[:, static_size:static_size+dynamic_size]
                    X_dynamic = X_dynamic.reshape((X_dynamic.shape[0], forecast_horizons, dynamic_input_dim))

                    X_future = X[:, static_size+dynamic_size:]
                    X_future = X_future.reshape((X_future.shape[0], forecast_horizons, future_covariate_dim))

                    X = [X_static, X_dynamic, X_future]

            # Now we have transformed X accordingly, put it back into args or kwargs
            # Original function signature could vary
            # Let's assume the original expects X and y as positional arguments after self
            # If y was passed in kwargs, keep that consistent
            new_args = list(args)
            # We know args[0]=self
            # Let's see if original call had X in args or kwargs
            if 'X' in kwargs:
                kwargs['X'] = X
            else:
                # X was likely in args
                if len(args) > 1:
                    new_args[1] = X
                else:
                    new_args.append(X)
            if y is not None:
                if 'y' in kwargs:
                    # y is already in kwargs
                    pass
                else:
                    # if y was in args
                    if len(args) > 2:
                        # y was in args
                        pass
                    else:
                        new_args.append(y)

            return func(*new_args, **kwargs)
        return wrapper
    return decorator

