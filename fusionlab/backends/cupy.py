# -*- coding: utf-8 -*-
# @author: LKouadio~ @Daniel

"""
CuPyBackend module within the gofast computational framework provides an 
efficient way to utilize GPU acceleration for array operations, leveraging the 
CuPy library as an alternative to NumPy but with support for CUDA. This backend 
is ideal for tasks requiring high computational power and is particularly 
beneficial for operations on large arrays and matrices.

Setup:
To use the CuPyBackend, ensure CuPy is installed in your environment with 
CUDA support:: 
    
    pip install cupy-cudaXXX  # Replace XXX with your CUDA version, e.g., cupy-cuda110 for CUDA 11.0

Example Usage:

1. Initializing CuPyBackend:
    from gofast.backends.cupy_backend import CuPyBackend
    cupy_backend = CuPyBackend()

2. Performing Array Operations:
    # Creating an array on the GPU
    gpu_array = cupy_backend.array([1, 2, 3, 4, 5])

    # Performing matrix multiplication
    matrix_a = cupy_backend.random_uniform(low=0, high=1, size=(5, 5))
    matrix_b = cupy_backend.random_uniform(low=0, high=1, size=(5, 5))
    product = cupy_backend.matmul(matrix_a, matrix_b)
    print("Matrix product:\n", product)

3. Solving Linear Equations:
    # Solving the system Ax = B on the GPU
    A = cupy_backend.array([[3, 1], [1, 2]])
    B = cupy_backend.array([9, 8])
    x = cupy_backend.solve(A, B)
    print("Solution of the linear equation Ax = B:\n", x)

4. Handling Missing Data:
    # Filling NaN values with zeros
    array_with_nan = cupy_backend.array([1, cupy_backend.nan, 3])
    filled_array = cupy_backend.fillna(array_with_nan, fill_value=0)
    print("Array with NaN values filled:\n", filled_array)

Note:
- The CuPyBackend simplifies the transition from CPU to GPU computing, making 
  it accessible even for users unfamiliar with CUDA programming.
- This backend dynamically maps common NumPy functions to their CuPy counterparts,
  ensuring compatibility and ease of use.
- While CuPy offers significant speedups for certain operations, it's important 
  to consider memory constraints on the GPU.

Remember, effective use of CuPyBackend requires a CUDA-compatible NVIDIA GPU. 
For operations not covered by CuPy or for non-GPU environments, consider using
the gofast's NumPy or Dask backends.

This documentation aims to provide a quick start to using the CuPyBackend within
the gofast framework, highlighting its capabilities and ease of use for GPU-accelerated 
scientific computing.
"""

try:
    import cupy as cp
except ImportError:
    cp = None

from .base import BaseBackend

class BackendNotAvailable(Exception):
    pass

class CuPyBackend(BaseBackend):
    """
    A dynamic computational backend leveraging CuPy for GPU-accelerated array 
    operations, designed to be compatible with NumPy's API for seamless 
    integration into the gofast framework.
    This backend is ideal for processing large arrays and matrices with 
    significant speed improvements on CUDA-enabled NVIDIA GPUs.

    Attributes
    ----------
    None

    Methods
    -------
    solve(a, b)
        Solves a linear matrix equation, or system of linear scalar equations on the GPU.
    eig(a)
        Computes the eigenvalues and right eigenvectors of a square array on the GPU.
    svd(a, full_matrices=True, compute_uv=True)
        Singular Value Decomposition on the GPU.
    fillna(a, fill_value=0)
        Replaces NaN values with a specified fill value on the GPU.
    dropna(a, axis=0)
        Removes missing values along a given axis on the GPU.
    ifft(a, n=None, axis=-1)
        Computes the one-dimensional inverse discrete Fourier Transform on the GPU.
    linalg_solve(a, b)
        Solves a linear matrix equation, or system of linear scalar equations on the GPU.
    linalg_inv(a)
        Computes the (multiplicative) inverse of a matrix on the GPU.
    linalg_det(a)
        Computes the determinant of an array on the GPU.
    linalg_eig(a)
        Computes the eigenvalues and right eigenvectors of a square array on the GPU.
    linalg_svd(a, full_matrices=True, compute_uv=True)
        Singular Value Decomposition on the GPU.
    random_normal(loc=0.0, scale=1.0, size=None)
        Draws random samples from a normal (Gaussian) distribution on the GPU.
    random_uniform(low=0.0, high=1.0, size=None)
        Draws samples from a uniform distribution on the GPU.
    random_integers(low, high=None, size=None)
        Returns random integers from `low` (inclusive) to `high` (inclusive) on the GPU.

    Notes
    -----
    The CuPyBackend facilitates easy transition to GPU computing, allowing 
    users to leverage the power of CUDA without needing to deeply understand 
    GPU programming. While designed for interoperability with NumPy, it 
    significantly accelerates computations that are well-suited for parallel 
    processing on GPUs, such as large-scale linear algebra and array operations.

    Example usage to demonstrate the capabilities of CuPyBackend include 
    solving linear equations, performing matrix operations, and handling missing
    data efficiently with GPU acceleration.

    Remember, using CuPyBackend requires a CUDA-compatible NVIDIA GPU. 
    The backend dynamically maps functions to CuPy, providing a flexible and 
    powerful tool for high-performance computing tasks within the gofast framework.

    Examples
    --------
    >>> from gofast.backends.cupy_backend import CuPyBackend
    >>> backend = CuPyBackend()
    >>> a = backend.array([[3, 1], [1, 2]])
    >>> b = backend.array([9, 8])
    >>> x = backend.solve(a, b)
    >>> print(x)
    [ 2.  3.]

    For more advanced operations, such as eigenvalue computation or singular value decomposition,
    the same seamless integration applies, allowing for complex mathematical operations to be
    performed with minimal code changes from a NumPy-based workflow but with the added benefit
    of GPU acceleration.

    """

    def __init__(self):
        super().__init__()
        self._backend = self._import_backend()

        # Define custom methods with specific logic
        self.custom_methods = {
            'solve': self.solve,
            'eig': self.eig,
            'svd': self.svd,
            'fillna': self.fillna,
            'dropna': self.dropna,
            'ifft': self.ifft,
            'array': self.array, 
            'dot': self.dot, 
            'linalg_solve': self.linalg_solve,
            'linalg_inv': self.linalg_inv,
            'linalg_det': self.linalg_det,
            'linalg_eig': self.linalg_eig,
            'linalg_svd': self.linalg_svd,
            'random_normal': self.random_normal,
            'random_uniform': self.random_uniform,
            'random_integers': self.random_integers,
        }
    def _import_backend(self):
        """
        Attempt to import CuPy. Fall back to NumPy if CuPy is not available.
        """
        try:
            import cupy as cp
            return cp
        except ImportError:
            raise BackendNotAvailable(
                "CuPy is not installed. Please install CuPy for GPU acceleration.")

    def __getattr__(self, name):
        """
        Dynamically delegate attribute calls to the backend (CuPy), allowing
        for direct use of CuPy functions as if they were part of this class.
        """
        if name in self.custom_methods:
            return self.custom_methods[name]
        # Attempt to get the attribute from the CuPy backend
        try:
            return getattr(self._backend, name)
        except AttributeError:
            raise AttributeError(f"'CuPyBackend' object has no attribute '{name}'")

    def solve(self, a, b):
        """
        Solve a linear matrix equation, or system of linear scalar equations on the GPU.
        """
        return cp.linalg.solve(a, b)

    def eig(self, a):
        """
        Compute the eigenvalues and right eigenvectors of a square array on the GPU.
        """
        return cp.linalg.eig(a)

    def svd(self, a, full_matrices=True, compute_uv=True):
        """
        Singular Value Decomposition on the GPU.
        """
        return cp.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

    def fillna(self, a, fill_value=0):
        """
        Replace NaN values with a specified fill value on the GPU.
        """
        return cp.where(cp.isnan(a), fill_value, a)

    def dropna(self, a, axis=0):
        """
        Remove missing values along a given axis on the GPU.
        """
        if axis == 0:
            return a[~cp.isnan(a).any(axis=1)]
        elif axis == 1:
            return a[:, ~cp.isnan(a).any(axis=0)]
        else:
            raise ValueError("axis must be 0 or 1")

    def ifft(self, a, n=None, axis=-1):
        """
        Compute the one-dimensional inverse discrete Fourier Transform on the GPU.
        """
        return cp.fft.ifft(a, n=n, axis=axis)

    def linalg_solve(self, a, b):
        """
        Solve a linear matrix equation, or system of linear scalar equations on the GPU.
        Computes the "exact" solution, `x`, of the well-determined, i.e., full 
        rank, linear matrix equation `ax = b`.
        """
        return cp.linalg.solve(a, b)

    def linalg_inv(self, a):
        """
        Compute the (multiplicative) inverse of a matrix on the GPU.
        Given a square matrix `a`, return the matrix `a_inv` satisfying 
        `dot(a, a_inv) = dot(a_inv, a) = eye(a.shape[0])`.
        """
        return cp.linalg.inv(a)

    def linalg_det(self, a):
        """
        Compute the determinant of an array on the GPU.
        """
        return cp.linalg.det(a)

    def linalg_eig(self, a):
        """
        Compute the eigenvalues and right eigenvectors of a square array on the GPU.
        """
        return cp.linalg.eig(a)

    def linalg_svd(self, a, full_matrices=True, compute_uv=True):
        """
        Singular Value Decomposition on the GPU.
        When `a` is a 2D array, it is factorized as `u @ np.diag(s) @ v`, where 
        `u` and `v` are 2D unitary arrays and `s` is a 1D array of `a`'s singular values.
        """
        return cp.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)

    def random_normal(self, loc=0.0, scale=1.0, size=None):
        """
        Draw random samples from a normal (Gaussian) distribution on the GPU.
        """
        return cp.random.normal(loc=loc, scale=scale, size=size)

    def random_uniform(self, low=0.0, high=1.0, size=None):
        """
        Draw samples from a uniform distribution on the GPU.
        """
        return cp.random.uniform(low=low, high=high, size=size)

    def random_integers(self, low, high=None, size=None):
        """
        Return random integers from `low` (inclusive) to `high` (inclusive) on the GPU.
        """
        return cp.random.randint(low, high=high, size=size)

    def array(self, data,**kwargs):
        """
        Convert input data to a CuPy array, utilizing GPU acceleration.
        """
        return cp.array(data, **kwargs,)

    def dot(self, a, b):
        """
        Perform dot product of two arrays using CuPy for GPU acceleration.
        """
        return cp.dot(a, b)


if __name__=='__main__': 
    # Example usage
    try:
        backend = CuPyBackend()
        # Use CuPy's `ones` function through the backend
        ones_array = backend.ones((5, 5))
        print(ones_array)
    except BackendNotAvailable as e:
        print(e)


