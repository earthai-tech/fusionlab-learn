# -*- coding: utf-8 -*-
import numpy as np
from functools import wraps

def _scigofast_set_X_compat(model_type='tft', ops='concat'):
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

