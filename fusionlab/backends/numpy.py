# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio (a.k.a. @Daniel) <etanoyau@gmail.com>

"""
NumpyBackend module within the gofast computational framework provides 
a comprehensive suite of array operations based on NumPy, the fundamental 
package for scientific computing with Python. This backend serves as the 
default computational backbone for gofast, offering a wide range of array 
and matrix manipulation functionalities that are crucial for data analysis, 
scientific computing, and machine learning tasks.

Setup:
To use the NumpyBackend, ensure NumPy is installed in your environment:

    pip install numpy

Example Usage:

1. Initializing NumpyBackend:
    from gofast.backends.numpy import NumpyBackend
    numpy_backend = NumpyBackend()

2. Performing Array Operations:
    # Creating an array
    array = numpy_backend.array([1, 2, 3, 4, 5])

    # Performing dot product
    vector_a = numpy_backend.array([1, 2, 3])
    vector_b = numpy_backend.array([4, 5, 6])
    dot_product = numpy_backend.dot(vector_a, vector_b)
    print("Dot product:", dot_product)

3. Solving Linear Equations:
    # Solving the system Ax = B
    A = numpy_backend.array([[3, 1], [1, 2]])
    B = numpy_backend.array([9, 8])
    x = numpy_backend.solve(A, B)
    print("Solution of the linear equation Ax = B:", x)

4. Handling Missing Data:
    # Filling NaN values with zeros
    array_with_nan = numpy_backend.array([1, numpy_backend.nan, 3])
    filled_array = numpy_backend.fillna(array_with_nan, fill_value=0)
    print("Array with NaN values filled:", filled_array)

Note:
- The NumpyBackend leverages NumPy's efficiency, broad compatibility, and extensive 
  functionality, making it a versatile and reliable choice for a wide range of tasks.
- This backend offers a familiar set of tools for those already accustomed to NumPy's 
  API, ensuring a smooth integration with gofast's functionalities.
- For tasks requiring GPU acceleration or distributed computing, consider using gofast's 
  CuPyBackend or DaskBackend respectively.

This documentation aims to provide a concise introduction to using the NumpyBackend within 
the gofast framework, highlighting its fundamental role in numerical computing and data 
processing tasks.
"""

from .base import BaseBackend 

__all__=["NumpyBackend"]

class NumpyBackend(BaseBackend):
    """
    The NumpyBackend class provides a comprehensive suite of numerical 
    operations based on NumPy, making it the default computational backbone 
    for the gofast framework. It includes a broad range of array and matrix 
    manipulation tools that are essential for data analysis, scientific 
    computing, and machine learning tasks. This backend seamlessly integrates
    NumPy's high-performance operations with gofast's functionality, offering
    users a familiar and powerful set of tools for numerical computing.

    Attributes
    ----------
    None explicitly declared; relies on NumPy's attributes.

    Methods
    -------
    - All NumPy array creation, manipulation, and querying functions are supported.
    - Custom methods for specific computational needs, enhancing NumPy's capabilities.

    Notes
    -----
    This backend leverages NumPy due to its efficiency, community support, and extensive 
    functionality. NumPy arrays form the core data structure in many scientific computing 
    and data analysis applications, making NumpyBackend a versatile and reliable choice 
    for a wide range of tasks within the gofast ecosystem.

    Examples
    --------
    To utilize the NumpyBackend for common array operations:

    >>> from gofast.backends.numpy import NumpyBackend
    >>> backend = NumpyBackend()
    >>> a = backend.array([1, 2, 3])
    >>> b = backend.random_normal(loc=0, scale=1, size=3)
    >>> c = backend.dot(a, b)

    For matrix operations and linear algebra:

    >>> A = backend.array([[1, 2], [3, 4]])
    >>> inv_A = backend.inv(A)
    >>> eigvals, eigvecs = backend.eig(A)

    Utilizing custom methods for enhanced functionality:

    >>> y = [0, 1, 2, 1]
    >>> num_classes = 3
    >>> y_categorical = backend.to_categorical(y, num_classes=num_classes)

    Note that while NumpyBackend provides a solid foundation for numerical computing, 
    users may switch to other backends like CuPyBackend for GPU acceleration or DaskBackend 
    for distributed computing depending on their specific needs and computational resources.
    """

    def __init__(self):
        super().__init__()
        import numpy as np
        self._np = np  # Directly use the imported NumPy module
        # Define custom methods mapping
        self.custom_methods = {
            'inv': self.inv,
            'eig': self.eig,
            'svd': self.svd,
            'random_uniform': self.random_uniform,
            'random_normal': self.random_normal,
            'to_categorical': self.to_categorical,
            'dot': self.dot, 
            'pinv': self.pinv,
            'solve': self.solve,
            'fillna': self.fillna,
            'dropna': self.dropna,
            'to_datetime': self.to_datetime,
            'array': self.array
        }

    def __getattr__(self, name):
        """
        Dynamically delegate attribute calls either to custom methods or to 
        the underlying NumPy module, enabling direct use of NumPy's array 
        functions and mathematical operations as if they were part of this class.
        """
        if name in self.custom_methods:
            return self.custom_methods[name]
        # Attempt to delegate to NumPy's attribute
        try:
            attr = getattr(self._np, name)
            if callable(attr):
                return attr
        except AttributeError:
            pass
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def array(self, data, dtype=None, *, copy=True, order='K', subok=False, ndmin=0):
        """
        Convert input data to a NumPy array.

        Parameters:
        - data: Data to be converted to an array. Can be a list, tuple, or any array-like object.
        - dtype: Desired data type of the array, optional. If not specified, the data type will
                 be inferred from the data.
        - copy: Whether to copy the input data (default: True).
        - order: Whether to store multi-dimensional data in row-major (C-style) or
                 column-major (Fortran-style) order in memory.
        - subok: Whether to return a subclass of `np.ndarray` if possible.
        - ndmin: Specifies the minimum number of dimensions that the resulting
                 array should have. Ones will be pre-pended to the shape as needed.

        Returns:
        - A NumPy array constructed from the input data.
        """
        return self._np.array(data, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    def inv(self, a):
        """Compute the (multiplicative) inverse of a matrix."""
        return self._np.linalg.inv(a)

    def eig(self, a):
        """Compute the eigenvalues and right eigenvectors of a square array."""
        return self._np.linalg.eig(a)


    def svd(self, a, full_matrices=True, compute_uv=True):
        """
        Singular Value Decomposition.
        """
        return self._np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

    def random_uniform(self, low=0.0, high=1.0, size=None):
        """
        Draw samples from a uniform distribution.
        """
        return self._np.random.uniform(low=low, high=high, size=size)

    def random_normal(self, loc=0.0, scale=1.0, size=None):
        """
        Draw random samples from a normal (Gaussian) distribution.
        """
        return self._np.random.normal(loc=loc, scale=scale, size=size)

    def to_categorical(self, y, num_classes=None):
        """
        Converts a class vector (integers) to binary class matrix.
        """
        return self._np.eye(num_classes)[y]

    def pinv(self, a):
        """
        Compute the Moore-Penrose pseudoinverse of a matrix.
        """
        return self._np.linalg.pinv(a)

    def solve(self, a, b):
        """
        Solve a linear matrix equation, or system of linear scalar equations.
        """
        return self._np.linalg.solve(a, b)

    def fillna(self, a, fill_value=0):
        """
        Replace NaN values with a specified fill value.
        """
        if not self._np.isnan(a).any():
            return a
        return self._np.where(self.isnan(a), fill_value, a)

    def dropna(self, a, axis=0):
        """
        Remove missing values along a given axis.
        """
        if axis == 0:
            return a[~self._np.isnan(a).any(axis=1)]
        elif axis == 1:
            return a[:, ~self._np.isnan(a).any(axis=0)]
        else:
            raise ValueError("axis must be 0 or 1")

    def to_datetime(self, a, format=None):
        """
        Convert argument to datetime.
        """
        import datetime 
        return self._np.array([self._npdatetime64(datetime.strptime(str(x), format)) for x in a])

    def dot(self, a, b, out=None):
        """
        Perform dot product of two arrays.
        
        Parameters:
        - a, b: Input arrays.
        - out: Optional output array, in which to place the result.
        
        Returns:
        - The dot product of a and b.
        """
        return self._np.dot(a, b, out=out)

if __name__=='__main__': 
    # Example usage
    numpy_backend = NumpyBackend()
    array_example = numpy_backend.array([1, 2, 3])
    print(array_example)
    

    from .backends.numpy import NumpyBackend
    # Import other backends as necessary
    
    # Default backend
    _active_backend = NumpyBackend()
    
    def set_backend(backend_name):
        global _active_backend
        if backend_name == 'numpy':
            _active_backend = NumpyBackend()
        # Add other backends as they become available
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
    
    def get_backend():
        return _active_backend
    
    # Example Usage
    # from gofast.config import set_backend, get_backend
    
    # Set the active backend to NumPy
    set_backend('numpy')
    
    # Get the current active backend
    backend = get_backend()
    
    # Perform operations using the active backend
    a = backend.array([1, 2, 3])
    b = backend.array([4, 5, 6])
    dot_product = backend.dot(a, b)
    
    print(dot_product)  # Output will depend on the active backend's implementation
