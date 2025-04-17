# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio (a.k.a. @Daniel) <etanoyau@gmail.com>

"""
The DaskBackend module provides a seamless integration with Dask to enable
distributed computing capabilities for gofast's machine learning and data
processing tasks. This backend is especially useful for handling large datasets
that do not fit into the memory of a single machine, allowing for scalable
and parallel computation.

Setup:
To use the DaskBackend, ensure Dask is installed in your environment:
    pip install dask[distributed]

Example Usage:

1. Initialize DaskBackend:
    from gofast.backends.dask import DaskBackend
    dask_backend = DaskBackend()

2. Leverage Dask for Dataframe Operations:
    # Assume `df` is a Dask DataFrame
    df = dask_backend.read_csv('large_dataset.csv')
    result = dask_backend.groupby_aggregate(df, groupby_columns=['category'], aggfunc='mean')

3. Distributed Machine Learning Training:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    X, y = dask_backend.load_data('features.csv', 'targets.csv')
    trained_model = dask_backend.distributed_training(model, X, y)

4. Using Custom Methods for Parallel Operations:
    # Custom method for rechunking a Dask array
    dask_array = dask_backend.create_random_array(shape=(10000, 10000), chunks=(1000, 1000))
    rechunked_array = dask_backend.rechunk(dask_array, chunks=(500, 500))

Note:
- The DaskBackend aims to abstract away the complexity of distributed computing,
  allowing users to focus on their data processing and machine learning tasks.
- For operations not explicitly covered by the DaskBackend, users can directly
  leverage Dask's API for advanced and customized parallel computation.

Remember, the DaskBackend's efficiency and scalability are highly dependent on
the setup of your Dask cluster. For optimal performance, ensure your cluster
configuration matches the computational demands of your tasks.
"""

from .base import BaseBackend

class BackendNotAvailable(Exception):
    pass

__all__=["DaskBackend"]

class DaskBackend(BaseBackend):
    """
    Implements the gofast computational backend using Dask for distributed
    computing capabilities. This backend facilitates the handling of large
    datasets and computationally intensive tasks by leveraging Dask's
    distributed data structures and parallel execution engine.

    Parameters
    ----------
    scheduler_url : str, optional
        URL of the Dask scheduler. If 'local', initializes a local Dask client.
        Otherwise, expects the URL of a remote Dask scheduler. Default is 'local'.

    Attributes
    ----------
    client : dask.distributed.Client
        The Dask client for distributed computing.

    Methods
    -------
    rechunk(array, chunks="auto")
        Rechunk a Dask array for optimal performance.
    persist(array)
        Persist data in memory across Dask workers.
    compute(*args, **kwargs)
        Compute Dask graphs in parallel, returning their results.
    distributed_ml_training(data, target, model, **fit_params)
        Train a machine learning model using Dask for distributed computation.
    delayed_execution(func, *args, **kwargs)
        Apply a function lazily, delaying its computation.
    merge_dataframes(dfs)
        Concatenate multiple Dask DataFrames into a single DataFrame.
    groupby_aggregate(df, by, aggfuncs)
        Perform a groupby operation followed by aggregation on a Dask DataFrame.
    parallelize_function(func, iterable)
        Apply a function over an iterable in parallel using Dask.
    map_partitions(df, func, *args, **kwargs)
        Apply a function to each partition of a Dask DataFrame or Series.
    read_csv(filepath, **kwargs)
        Read a CSV file into a Dask DataFrame, allowing for out-of-core computation.
    parallel_cross_validation(model, X, y, cv=5, scoring='accuracy')
        Perform cross-validation in parallel for a given model and dataset.
    grid_search_cv(model, X, y, param_grid, cv=5, scoring='accuracy')
        Conduct a grid search with cross-validation in parallel.
    distributed_training(model, X, y, **kwargs)
        Train a model using distributed computation with Dask.

    Notes
    -----
    DaskBackend provides a bridge between gofast and Dask, allowing users to
    easily scale their data processing and machine learning tasks to clusters
    of machines. It abstracts away much of the complexity associated with
    distributed computing, offering a familiar and simple API for users.

    Examples
    --------
    >>> from gofast.backends.dask import DaskBackend
    >>> dask_backend = DaskBackend()
    >>> import dask.array as da
    >>> large_array = da.random.random((10000, 10000), chunks=(1000, 1000))
    >>> rechunked_array = dask_backend.rechunk(large_array, chunks=(500, 500))
    >>> persisted_array = dask_backend.persist(rechunked_array)
    >>> result = dask_backend.compute(persisted_array.mean(axis=0))

    Use DaskBackend for distributed machine learning training:

    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression()
    >>> X, y = dask_backend.load_data('features.csv', 'targets.csv')  # Example function
    >>> trained_model = dask_backend.distributed_training(model, X, y)
    """
    def __init__(self, scheduler_url='local'):
        """
        Initializes the Dask backend.
        
        Parameters:
        - scheduler_url: URL of the Dask scheduler. If 'local', a 
        local Dask client is created.
        """
        super().__init__()
        try:
            global da
            global dask 
            import dask.array as da
            import dask 
            from dask.distributed import Client
        except ImportError:
            raise BackendNotAvailable("Dask is not installed. Please install"
                                      " Dask to use the DaskBackend.")
        
        if scheduler_url == 'local':
            self.client = Client()  # Start a local Dask client
        else:
            self.client = Client(scheduler_url)  # Connect to a remote Dask cluster

    def __getattr__(self, name):
        """
        Dynamically delegate attribute calls to custom methods or the Dask module,
        enabling direct use of its functions as if they were part of this class.
        """
        custom_methods = {
            "rechunk": self.rechunk,
            "persist": self.persist,
            "compute": self.compute,
            "array": self.array, 
            "solve": self.solve, 
            "dot": self.dot, 
            "eig": self.eig, 
            "distributed_ml_training": self.distributed_ml_training,
            "delayed_execution": self.delayed_execution,
            "merge_dataframes": self.merge_dataframes,
            "groupby_aggregate": self.groupby_aggregate,
            "parallelize_function": self.parallelize_function,
            "map_partitions": self.map_partitions,
            "read_csv": self.read_csv,
            "parallel_cross_validation": self.parallel_cross_validation,
            "grid_search_cv": self.grid_search_cv,
            "distributed_training": self.distributed_training,
            "delayed_operations": self.delayed_operations
        }
    
        if name in custom_methods:
            return custom_methods[name]
        try:
            # Attempt to delegate to Dask's array module attribute first
            return getattr(da, name)
        except AttributeError:
            # If not found in dask.array, fall back to the main dask module
            try:
                return getattr(dask, name)
            except AttributeError:
                # If still not found, raise an AttributeError
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
                
    def rechunk(self, array, chunks="auto"):
        """
        Rechunk an array to optimize performance for distributed processing.
        
        Parameters:
        - array: Dask array to be rechunked.
        - chunks: New chunk sizes along each dimension. 'auto' lets Dask decide.
        
        Returns:
        - Dask array with new chunk sizes.
        """
        return array.rechunk(chunks)
    
    def persist(self, array):
        """
        Persist data in memory across the cluster. This operation allows future
        computations to be much faster by reducing disk I/O and other overheads.
        
        Parameters:
        - array: Dask array to be persisted in memory.
        
        Returns:
        - A Dask array that is persisted in memory.
        """
        return array.persist()

    def compute(self, *args, **kwargs):
        """
        Compute multiple dask collections at once, bringing their results from distributed
        memory into local process memory. Useful for executing multiple tasks in parallel.
        
        Parameters:
        - args: Dask objects to compute.
        - kwargs: Additional keyword arguments to pass to the compute function.
        
        Returns:
        - Tuple of computed results corresponding to the input Dask objects.
        """
        from dask import compute
        return compute(*args, **kwargs)

    def distributed_ml_training(self, data, target, model, **fit_params):
        """
        Train a machine learning model in a distributed fashion using Dask-ML.
        
        Parameters:
        - data: Dask array or DataFrame representing the features.
        - target: Dask array or Series representing the target variable.
        - model: A scikit-learn estimator or a Dask-ML compatible model.
        - fit_params: Additional parameters to pass to the model's fit method.
        
        Returns:
        - The trained model.
        """
        model.fit(data, target, **fit_params)
        return model

    def delayed_execution(self, func, *args, **kwargs):
        """
        Apply a function in a lazy manner, where actual computations are delayed
        until explicitly triggered. This method is useful for building computation
        graphs with multiple interdependent tasks.
        
        Parameters:
        - func: The function to be applied in a delayed fashion.
        - args: Arguments to pass to the function.
        - kwargs: Keyword arguments to pass to the function.
        
        Returns:
        - A delayed object, which is a task in the computation graph.
        """
        from dask import delayed
        return delayed(func)(*args, **kwargs)
    
    def merge_dataframes(self, dfs):
        """
        Merge multiple Dask DataFrames into a single DataFrame.
        
        Parameters:
        - dfs: List of Dask DataFrames to be merged.
        
        Returns:
        - Merged Dask DataFrame.
        """
        from dask.dataframe import multi
        return multi.concat(dfs, axis=0)
    
    def groupby_aggregate(self, df, by, aggfuncs):
        """
        Perform groupby operation followed by aggregation on a Dask DataFrame.
        
        Parameters:
        - df: Dask DataFrame to perform the operation on.
        - by: Column name or list of column names to group by.
        - aggfuncs: Dictionary mapping column names to aggregation functions.
        
        Returns:
        - Aggregated Dask DataFrame.
        """
        return df.groupby(by).agg(aggfuncs)
    
    def parallelize_function(self, func, iterable):
        """
        Apply a function over all items in an iterable in parallel using Dask.
        
        Parameters:
        - func: Function to apply. It should be serializable by Dask.
        - iterable: Iterable of items to which the function will be applied.
        
        Returns:
        - List of results, with each result corresponding to the function 
          application on an item of the iterable.
        """

        from dask import delayed
        results = [delayed(func)(item) for item in iterable]
        return dask.compute(*results)
    
    def map_partitions(self, df, func, *args, **kwargs):
        """
        Apply a function to each partition of a Dask DataFrame or Series.
        
        Parameters:
        - df: Dask DataFrame or Series to apply the function to.
        - func: Function to apply to each partition. Must return a DataFrame or Series.
        - args: Positional arguments to pass to the function after the partition.
        - kwargs: Keyword arguments to pass to the function.
        
        Returns:
        - A Dask DataFrame or Series with the function applied to each partition.
        """
        return df.map_partitions(func, *args, **kwargs)
    
    def read_csv(self, filepath, **kwargs):
        """
        Read a CSV file into a Dask DataFrame. This method is particularly useful
        for reading large CSV files that do not fit into memory, as it loads the data
        in partitions that can be processed independently.
        
        Parameters:
        - filepath: Path to the CSV file or a pattern string for multiple files.
        - kwargs: Additional keyword arguments to pass to Dask's read_csv function.
        
        Returns:
        - Dask DataFrame representing the CSV data.
        """
        import dask.dataframe as dd
        return dd.read_csv(filepath, **kwargs)

    def parallel_cross_validation(self, model, X, y, cv=5, scoring='accuracy'):
        """
        Perform cross-validation in parallel for a given model and dataset.
        
        Parameters:
        - model: The machine learning model to be evaluated.
        - X: Dask DataFrame or array-like, feature matrix.
        - y: Dask Series or array-like, target values.
        - cv: int, number of cross-validation folds.
        - scoring: str, scoring metric to use.
        
        Returns:
        - Dictionary containing cross-validation scores.
        """
        from dask_ml.model_selection import cross_validate
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        return scores
    
    def grid_search_cv(self, model, X, y, param_grid, cv=5, scoring='accuracy'):
        """
        Conduct a grid search with cross-validation in parallel.
        
        Parameters:
        - model: Estimator for which the grid search is performed.
        - X: Dask DataFrame or array-like, feature matrix.
        - y: Dask Series or array-like, target values.
        - param_grid: Dictionary with parameters names as keys and lists of 
          parameter settings to try as values.
        - cv: int, number of cross-validation folds.
        - scoring: str, scoring metric to use.
        
        Returns:
        - Fitted GridSearchCV object with results.
        """
        from dask_ml.model_selection import GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
        return grid_search
    
    def distributed_training(self, model, X, y, **kwargs):
        """
        Train a model using distributed computation with Dask.

        This method wraps the input model with Dask-ML's ParallelPostFit, which allows
        the model to be trained on a single machine but makes predictions in parallel
        using Dask. This is particularly useful for models that cannot be trained in
        parallel but can benefit from parallel prediction.

        Parameters:
        - model: The machine learning model to be trained. This model should support
                 fit and predict methods similar to scikit-learn models.
        - X: Dask DataFrame or array-like, feature matrix.
        - y: Dask Series or array-like, target values.
        - kwargs: Additional keyword arguments to pass to the model's fit method.

        Returns:
        - The trained model wrapped with ParallelPostFit for distributed prediction.
        
        Note: It's essential to have Dask and Dask-ML installed in your environment
              to use this method. If they are not installed, please install them
              using `pip install dask[complete] dask-ml`.
        """
        # Ensure Dask-ML is available
        try:
            from dask_ml.wrappers import ParallelPostFit
        except ImportError as e:
            raise ImportError(
                "Dask-ML is required for distributed training but is not installed. "
                "Please install it using `pip install dask-ml`.") from e

        # Wrap the model with ParallelPostFit for parallel predictions
        wrapped_model = ParallelPostFit(estimator=model, **kwargs)
        
        # Fit the model using Dask
        wrapped_model.fit(X, y)
        
        return wrapped_model
    
    def delayed_operations(self, func, *args, **kwargs):
        """
        Execute a function in a delayed fashion, allowing for lazy evaluation.
        
        Parameters:
        - func: Function to be executed in a delayed manner.
        - args: Positional arguments to pass to the function.
        - kwargs: Keyword arguments to pass to the function.
        
        Returns:
        - A delayed object which can be computed later.
        """
        from dask import delayed
        return delayed(func)(*args, **kwargs)
    
    def array(self, data, dtype=None, *, chunks="auto", name=None,
              lock=False, asarray=True, fancy=True, getting=None, meta=None):
        """
        Convert input data to a Dask array, allowing for parallel and distributed computation.
        """
        return da.from_array(data, chunks=chunks, dtype=dtype)

    def dot(self, a, b):
        """
        Perform dot product of two arrays using Dask for parallel computation.
        """
        return da.dot(a, b)

    def solve(self, a, b):
        """
        Solve a linear matrix equation, or system of linear scalar equations, leveraging
        Dask's capabilities. Note: Dask does not directly support a 'solve' function
        for linear systems as of its core API, and this operation may need to rely on
        Dask's general functionality for applying NumPy's solve in parallel or other
        specialized packages.
        """
        # May require custom implementation
        # or integration with other libraries like dask-ml or custom dask operations.
        raise NotImplementedError(
            "Dask does not directly support 'solve'. Requires custom implementation.")

    def eig(self, a):
        """
        Compute the eigenvalues and right eigenvectors of a square array using Dask.
        Note: Similar to 'solve', Dask does not directly support an 'eig' function, and
        this operation may require custom implementations or use of other libraries.
        """
        raise NotImplementedError("Dask does not directly support 'eig'."
                                  " Requires custom implementation.")

# Example of using the DaskBackend
if __name__ == "__main__":
    dask_backend = DaskBackend()
    # Example usage of Dask array creation
    x = dask_backend.array([1, 2, 3, 4, 5])
    print(x.compute())  # Compute the result and print
