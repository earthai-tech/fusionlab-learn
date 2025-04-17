# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio (a.k.a. @Daniel) <etanoyau@gmail.com>

"""
ScipyBackend module enriches the gofast computational framework with SciPy's
scientific computing capabilities. This backend is particularly designed for tasks
requiring advanced mathematical operations such as optimization, root finding,
numerical integration, solving differential equations, and curve fitting.

Setup:
To utilize the ScipyBackend, ensure SciPy is installed in your environment:
    pip install scipy

Example Usage:

1. Initializing ScipyBackend:
    from gofast.backends.scipy import ScipyBackend
    scipy_backend = ScipyBackend()

2. Optimizing a Quadratic Function:
    result = scipy_backend.optimize_quadratic(a=1, b=-3, c=2, x0=[0])
    print('Optimal solution:', result.x)

3. Finding a Root of a Nonlinear Equation:
    def func(x): return x**2 - 4
    root = scipy_backend.find_root(func, x0=[2])
    print('Root of the equation:', root.x)

4. Numerically Integrating a Function:
    from math import sin
    integral, error = scipy_backend.integrate_function(sin, 0, 3.1416)
    print('Integral result:', integral)

5. Solving an Ordinary Differential Equation (ODE):
    def ode_system(t, y): return -2 * y
    sol = scipy_backend.solve_ode(ode_system, y0=[1], t=[0, 0.5, 1])
    print('Solution of the ODE:', sol.y)

6. Fitting a Curve to Data Points:
    def linear_model(x, m, c): return m*x + c
    xdata = np.array([0, 1, 2, 3])
    ydata = np.array([1, 3, 5, 7])
    params, cov = scipy_backend.curve_fit(linear_model, xdata, ydata)
    print('Fitted parameters:', params)

Note:
- ScipyBackend facilitates the integration of SciPy's rich scientific computing
  tools into the gofast ecosystem, enhancing its capabilities for advanced
  mathematical and statistical operations.
- While ScipyBackend covers a wide range of tasks, users are encouraged to
  explore SciPy's documentation for more complex and specific use cases.

The integration of ScipyBackend into gofast exemplifies the framework's flexibility
and its capacity to adapt to various computational requirements, making it a
powerful tool for scientific research and data analysis.
"""

from .base import BaseBackend 

__all__=["ScipyBackend"]

class ScipyBackend(BaseBackend):
    """
    Implements the gofast computational backend using SciPy. This backend
    extends gofast's computational capabilities with a focus on scientific
    computing tasks such as optimization, root finding, integration,
    differential equations, and curve fitting.

    This backend dynamically handles method calls to SciPy, providing
    both standard SciPy functionalities and custom logic for enhanced
    operations.

    Methods
    -------
    optimize_quadratic(*args, **kwargs)
        Optimizes a quadratic function.
    find_root(func, x0, *args, **kwargs)
        Finds a root of a nonlinear equation.
    integrate_function(func, a, b, *args, **kwargs)
        Numerically integrates a function over a given interval.
    solve_ode(func, y0, t, *args, **kwargs)
        Solves an Ordinary Differential Equation (ODE) given an initial value.
    curve_fit(func, xdata, ydata, *args, **kwargs)
        Fits a curve to data points using nonlinear least squares.

    Notes
    -----
    ScipyBackend serves as a bridge between gofast and SciPy, enabling users
    to seamlessly incorporate a wide range of scientific computing tools into
    their workflows. It particularly shines in tasks that require sophisticated
    mathematical computations, offering a rich set of algorithms for
    optimization, root finding, numerical integration, solving differential
    equations, and curve fitting.

    Examples
    --------
    >>> from gofast.backends.scipy_backend import ScipyBackend
    >>> scipy_backend = ScipyBackend()
    
    Optimizing a quadratic function:
    >>> result = scipy_backend.optimize_quadratic(a=1, b=-3, c=2, x0=[0])
    >>> print(result.x)  # Optimal solution

    Finding a root of a nonlinear equation:
    >>> def func(x): return x**2 - 4
    >>> root = scipy_backend.find_root(func, x0=[2])
    >>> print(root.x)  # Root of the equation

    Numerically integrating a function:
    >>> from math import sin
    >>> integral, error = scipy_backend.integrate_function(sin, 0, 3.1416)
    >>> print(integral)  # Integral result

    Solving an ODE:
    >>> def ode_system(t, y): return -2 * y
    >>> sol = scipy_backend.solve_ode(ode_system, y0=[1], t=[0, 0.5, 1])
    >>> print(sol.y)  # Solution of the ODE at specified points

    Curve fitting:
    >>> def linear_model(x, m, c): return m*x + c
    >>> xdata = np.array([0, 1, 2, 3])
    >>> ydata = np.array([1, 3, 5, 7])
    >>> params, cov = scipy_backend.curve_fit(linear_model, xdata, ydata)
    >>> print(params)  # Fitted parameters
    """

    def __init__(self):
        super().__init__()
        self._scipy = __import__('scipy')

    def __getattr__(self, name):
        """
        Delegate attribute calls to the underlying SciPy module, allowing for direct
        use of its functions as if they were part of this class, with special handling
        for methods that have custom logic defined.
        """
        # Define custom methods with specific logic
        custom_methods = {
            'optimize_quadratic': self.optimize_quadratic,
            'find_root': self.find_root, 
            'integrate_function': self.integrate_function, 
            'solve_ode': self.solve_ode, 
            'curve_fit': self.curve_fit, 
            'array': self.array, 
            'dot': self.dot, 
            'solve': self.solve, 
            'eig': self.eig
           }
        if name in custom_methods:
            return custom_methods[name]

        attr = getattr(self._scipy, name, None)
        if attr is not None:
            return attr
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def optimize_quadratic(self, *args, **kwargs):
        """
        Custom method to optimize a simple quadratic function.
        The function is defined as f(x) = ax^2 + bx + c, where a, b, and c are coefficients.
        This method simplifies the process by setting up the quadratic function and
        calling scipy.optimize.minimize with predetermined arguments.
        """
        # Define the quadratic function
        def quadratic_function(x, a=1, b=0, c=0):
            return a*x**2 + b*x + c
        
        # Extract coefficients if provided, else default to a=1, b=0, c=0
        a = kwargs.pop('a', 1)
        b = kwargs.pop('b', 0)
        c = kwargs.pop('c', 0)
    
        # Setup the optimization problem
        initial_guess = kwargs.pop('x0', [0])  # Initial guess for the minimization
        
        # Call scipy.optimize.minimize with the quadratic function
        result = self._scipy.optimize.minimize(
            quadratic_function, x0=initial_guess, args=(a, b, c), **kwargs)
        
        return result
    
    def find_root(self, func, x0, *args, **kwargs):
        """
        Custom method to find a root of a nonlinear equation.
        """
        result = self._scipy.optimize.root(func, x0, args=args, **kwargs)
        return result
    
    def integrate_function(self, func, a, b, *args, **kwargs):
        """
        Custom method to numerically integrate a function over a given interval [a, b].
        """
        result, error = self._scipy.integrate.quad(func, a, b, args=args, **kwargs)
        return result, error
    
    def solve_ode(self, func, y0, t, *args, **kwargs):
        """
        Custom method to solve an Ordinary Differential Equation (ODE) given an initial value.
        """
        sol = self._scipy.integrate.solve_ivp(func, (t[0], t[-1]), y0, t_eval=t, args=args, **kwargs)
        return sol
    
    def curve_fit(self, func, xdata, ydata, *args, **kwargs):
        """
        Custom method to fit a curve to data points using nonlinear least squares.
        """
        params, cov = self._scipy.optimize.curve_fit(func, xdata, ydata, *args, **kwargs)
        return params, cov


    def array(self, data, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None):
        """
        Convert input data to a NumPy array, leveraging NumPy's array creation functionality.
        """
        import numpy as np 
        return np.array(data, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    def dot(self, a, b):
        """
        Perform dot product of two arrays using NumPy.
        """
        import numpy as np 
        return np.dot(a, b)

    def solve(self, a, b):
        """
        Solve a linear matrix equation, or system of linear scalar equations, using SciPy's
        linear algebra solver.
        """
        return self._scipy.linalg.solve(a, b)

    def eig(self, a):
        """
        Compute the eigenvalues and right eigenvectors of a square array using SciPy's
        linear algebra functions.
        """
        return self._scipy.linalg.eig(a )

if __name__=='__main__': 

    scipy_backend = ScipyBackend()
    result = scipy_backend.optimize_quadratic(x0=[0], a=1, b=-3, c=2)
    print(f"Optimization result: {result}")
