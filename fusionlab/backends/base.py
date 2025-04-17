# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio (a.k.a. @Daniel) <etanoyau@gmail.com>

"""
The BaseBackend module serves as the foundation for the computational backends
in gofast, offering a unified interface for a wide range of computational tasks 
across different environments (CPU, GPU, distributed systems). It abstracts the 
complexity of directly interacting with lower-level libraries, providing a 
consistent and intuitive API for high-level computations.

The BaseBackend class is designed to be subclassed by specific backend implementations,
such as NumpyBackend for CPU-based computations, CuPyBackend for GPU acceleration,
ScipyBackend for advanced scientific computations, and DaskBackend for distributed
computing. This allows gofast to leverage the strengths of each computational 
framework while offering the user a seamless experience.

Setup:
There is no direct installation required for BaseBackend as it is a part of 
the gofast package. 
However, depending on the specific backend subclass being used, relevant libraries 
(such as NumPy, CuPy, SciPy, Dask) should be installed:: 
    
    pip install numpy scipy cupy-cudaXXX dask[distributed]

Example Usage:

1. Selecting a Backend:
    from gofast.backends.numpy import NumpyBackend
    numpy_backend = NumpyBackend()

2. Performing Array Operations:
    # Creating an array
    array = numpy_backend.array([1, 2, 3, 4, 5])

    # Dot product
    result = numpy_backend.dot(array, array)
    print("Dot product:", result)

3. Linear Algebra Operations:
    # Solving linear equations Ax = B
    A = numpy_backend.array([[3, 1], [1, 2]])
    B = numpy_backend.array([9, 8])
    x = numpy_backend.solve(A, B)
    print("Solution of Ax = B:", x)

Note:
- The BaseBackend and its subclasses aim to simplify the computational aspects of
  data processing and machine learning workflows, allowing users to focus on 
  problem-solving rather than the intricacies of the underlying computational 
  frameworks.
- It is important to select the appropriate backend based on the computational 
  requirements and available resources (e.g., use CuPyBackend for GPU 
                                        acceleration or DaskBackend for
  handling large datasets).

By providing a consistent interface across different computational environments,
gofast ensures that users can easily switch between backends without significant
changes to their code, enhancing productivity and facilitating experimentation
with different computational strategies.
"""
from ..api.property import BaseClass 

__all__=["BaseBackend"]

class BaseBackend(BaseClass):
    """
    Base class for all computational backends in gofast.
    This class defines a common interface for backend operations, ensuring 
    consistency and facilitating ease of use across different computational
    environments.

    Derived classes (NumpyBackend, CupyBackend, ScipyBackend, and DaskBackend)
    are expected to implement these methods, tailored to their specific 
    computational frameworks.
    """

    def array(self, data, dtype =None, *,  copy=True, order='K', 
              subok=False, ndmin =0, like=None,):
        """
        Convert input data to an array.
        
        Parameters:
        - data: Data to be converted to an array. Can be a list, tuple, or any
          array-like object.
        - dtype: Desired data type of the array, optional. If not specified,
          the data type will be inferred.
        
        Returns:
        - An array object.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def dot(self, a, b):
        """
        Perform dot product of two arrays.
        
        Parameters:
        - a, b: Input arrays.
        
        Returns:
        - The dot product of a and b.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def solve(self, a, b):
        """
        Solve a linear matrix equation, or system of linear scalar equations.
        
        Parameters:
        - a: Coefficient matrix.
        - b: Ordinate or dependent variable values.
        
        Returns:
        - Solution to the system a * x = b.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def eig(self, a):
        """
        Compute the eigenvalues and right eigenvectors of a square array.
        
        Parameters:
        - a: Square array from which to compute the eigenvalues and right eigenvectors.
        
        Returns:
        - The eigenvalues and right eigenvectors of a.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def __repr__(self):
        """
        Representation method to display the class name and its identifier.
        
        Returns:
        - String representation of the instance.
        """
        return f"<{self.__class__.__name__} at {hex(id(self))}>"
