# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio (a.k.a. @Daniel) <etanoyau@gmail.com>

"""
Provides an interface to select different computation backends within 
the `gofast` package, ensuring flexibility  between various numerical and 
machine learning frameworks. It includes support for popular backends such 
as NumPy, SciPy, Dask, CuPy, and neural network (NN) libraries.

Classes
-------
BackendSelector
    A class for selecting the appropriate computational backend dynamically 
    based on the user's environment and needs. This class leverages the 
    `EnsureMethod` decorator to prioritize `select_backend` execution.

Dependencies
------------
- `NumpyBackend`: Supports standard numerical operations with NumPy.
- `ScipyBackend`: Adds functionality for scientific and advanced 
  mathematical operations with SciPy.
- `DaskBackend`: Provides parallel computation and scalable data 
  processing using Dask.
- `CuPyBackend`: Allows GPU-accelerated computing with CuPy, compatible 
  with CUDA-enabled devices.
- `NNBackend`: Enables machine learning and deep learning functionality 
  through neural network libraries.
- `EnsureMethod`: A decorator used to ensure that the `select_backend` 
  method is prioritized for execution.

Usage
-----
The `BackendSelector` class is designed to help users dynamically choose 
the backend that best fits their computational environment. It is useful 
in workflows that require seamless transitions between CPU and GPU 
resources or between different data processing libraries.

Examples
--------
>>> from gofast.selector import BackendSelector
>>> backend_selector = BackendSelector()
>>> backend = backend_selector.select_backend("numpy")

Notes
-----
The `select_backend` method, decorated with `EnsureMethod`, operates in 
soft mode with warnings, enabling flexible backend selection and handling 
potentially missing methods gracefully.

__all__
-------
- `BackendSelector`: Exposes the `BackendSelector` class for backend 
  management.

"""
import subprocess
import warnings
from typing import Literal

from .numpy import NumpyBackend
from .scipy import ScipyBackend
from .dask import DaskBackend
from .cupy import CuPyBackend
from .nn import NNBackend 

from ..decorators import EnsureMethod 
from ..api.property import BaseClass 
from ..utils.deps_utils import ensure_pkg 

__all__= ["BackendSelector", "select_backend_n", "safe_cast",
          "check_processor"]

@EnsureMethod("select_backend", error='warn', mode='soft'  )
class BackendSelector(BaseClass):
    """
    A flexible class that manages and selects the most suitable computational 
    backend for `gofast` tasks. The `BackendSelector` allows users to specify 
    their preferred backend for both general computations (e.g., using 
    NumPy, SciPy) and neural network computations (e.g., using TensorFlow, 
    PyTorch), or to enable automatic selection based on system capabilities 
    and installed libraries.

    Parameters
    ----------
    preferred_backend : str, optional
        The preferred backend for general computations. Supported options 
        include `"numpy"`, `"scipy"`, `"dask"`, and `"cupy"`. If `None`, 
        automatic selection is applied.
    preferred_nn_backend : str, optional
        The preferred backend for neural network computations. Supported 
        options are `"tensorflow"` and `"pytorch"`. If `None`, automatic 
        selection is applied.
    verbose : int, optional
        Sets the verbosity level of the class output. `0` means no output, 
        while higher values increase the verbosity of feedback messages 
        regarding backend selection.

    Attributes
    ----------
    backends : dict
        Maps general backend names to their corresponding class implementations.
    nn_backends : dict
        Maps neural network backend names to `NNBackend`, which selects 
        between TensorFlow and PyTorch.
    selected_backend : object
        An instance of the selected backend class for general computations.
    selected_nn_backend : object
        An instance of the selected neural network backend class for deep 
        learning computations.

    Methods
    -------
    select_backend(preferred_backend)
        Selects the general computation backend based on user preference or 
        an automatic fallback if no valid preference is provided.
    select_nn_backend(preferred_nn_backend)
        Selects the neural network backend based on user preference or an 
        automatic fallback if no valid preference is provided.
    is_gpu_available()
        Checks if a CUDA-compatible GPU is available for GPU-accelerated 
        computations.
    is_dask_available()
        Checks if Dask is installed for distributed computations.
    is_scipy_available()
        Checks if SciPy is available for advanced scientific computations.
    auto_select_backend()
        Automatically selects the most suitable general backend based on 
        system resources.
    auto_select_nn_backend()
        Automatically selects the most suitable neural network backend 
        (TensorFlow or PyTorch).
    get_backend()
        Returns the currently selected general backend instance.
    get_nn_backend()
        Returns the currently selected neural network backend instance.

    Examples
    --------
    To select a general computation backend explicitly:

    >>> from gofast.backends.selector import BackendSelector
    >>> backend_selector = BackendSelector(preferred_backend='scipy')
    >>> backend = backend_selector.get_backend()
    >>> array = backend.array([1, 2, 3])
    
    If no backend is specified, an automatic selection is made:

    >>> backend_selector = BackendSelector()
    >>> backend = backend_selector.get_backend()
    # Selected backend based on available resources

    To select a neural network backend:

    >>> backend_selector = BackendSelector(preferred_nn_backend='tensorflow')
    >>> nn_backend = backend_selector.get_nn_backend()

    Notes
    -----
    The `BackendSelector` prioritizes GPU-accelerated backends for general 
    computation tasks by checking for `CuPy` support. If unavailable, it 
    defaults to `Dask`, `SciPy`, or `NumPy` in that order. Neural network 
    backend selection prioritizes TensorFlow if both it and PyTorch are 
    available, providing flexibility based on user configuration and system 
    capabilities.

    See Also
    --------
    numpy_backend, scipy_backend, dask_backend, cupy_backend, nn_backend

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020).
           "Array programming with NumPy." Nature, 585(7825), 357–362.
    .. [2] Abadi, M., Agarwal, A., Barham, P., et al. (2016). "TensorFlow:
           Large-Scale Machine Learning on Heterogeneous Distributed Systems."
           arXiv preprint arXiv:1603.04467.
    .. [3] Paszke, A., Gross, S., Massa, F., et al. (2019). "PyTorch: An 
           Imperative Style, High-Performance Deep Learning Library."
           Advances in Neural Information Processing Systems, 32.
    """

    def __init__(
        self, 
        preferred_backend=None, 
        preferred_nn_backend=None, 
        verbose=0
        ):
        
        self.backends = {
            "numpy": NumpyBackend,
            "scipy": ScipyBackend,
            "dask": DaskBackend,
            "cupy": CuPyBackend
        }
        self.nn_backends = {
            "tensorflow": NNBackend(backend="tensorflow", init=False),
            "pytorch": NNBackend(backend="pytorch", init= False)
        }
        self.verbose = verbose
        self.alias_map = {
            "np": "numpy",
            "sp": "scipy",
            "da": "dask",
            "dk": "dask",
            "cp": "cupy",
            "cy": "cupy",
            "nn": "nn",
            "neural": "nn"
        }
        self.selected_backend = self.select_backend(preferred_backend)
        if preferred_nn_backend is not None:
            self.selected_nn_backend = self.select_nn_backend(preferred_nn_backend)

    def select_backend(self, preferred_backend):
        """
        Selects the general computation backend based on user preference 
        or defaults to automatic selection if the specified preference 
        is unavailable.

        Parameters
        ----------
        preferred_backend : str
            User-specified backend for general computations. Can be one of 
            `"numpy"`, `"scipy"`, `"dask"`, or `"cupy"`. Shortened forms, 
            such as `"np"` for `"numpy"` and `"cp"` for `"cupy"`, are 
            also supported.

        Returns
        -------
        backend_instance : object
            Instance of the selected backend class.

        Notes
        -----
        If `preferred_backend` is not available or is not provided, the 
        `auto_select_backend` method is called to automatically determine 
        the most suitable backend.

        Examples
        --------
        >>> backend_selector = BackendSelector(preferred_backend="dask")
        >>> backend = backend_selector.select_backend("dask")
        DaskBackend selected by user preference.
        """

        normalized_backend = self.alias_map.get(preferred_backend, preferred_backend)
        if normalized_backend in self.backends:
            backend_instance = self.backends[normalized_backend]()
            if self.verbose > 0:
                print(f"{normalized_backend.capitalize()}Backend selected by user preference.")
            return backend_instance
        return self.auto_select_backend()

    def select_nn_backend(self, preferred_nn_backend):
        """
        Selects the neural network backend based on user preference or 
        defaults to automatic selection if the specified preference 
        is unavailable.

        Parameters
        ----------
        preferred_nn_backend : str
            User-specified neural network backend for computations, 
            either `"tensorflow"` or `"pytorch"`.

        Returns
        -------
        nn_backend_instance : object
            Instance of the selected neural network backend class.

        Notes
        -----
        If `preferred_nn_backend` is not available or is not provided, the 
        `auto_select_nn_backend` method is called to determine the optimal 
        backend, prioritizing `"tensorflow"`.

        Examples
        --------
        >>> backend_selector = BackendSelector(preferred_nn_backend="pytorch")
        >>> nn_backend = backend_selector.select_nn_backend("pytorch")
        PyTorch selected for neural network computations.
        """

        if preferred_nn_backend in self.nn_backends:
            if self.verbose > 0:
                print(f"{preferred_nn_backend.capitalize()}"
                      " selected for neural network computations.")
            return self.nn_backends[preferred_nn_backend]._initialize_backend(
                preferred_nn_backend)
        return self.auto_select_nn_backend()

    def is_gpu_available(self):
        """
        Checks whether a CUDA-compatible GPU is accessible on the system.

        Returns
        -------
        bool
            `True` if a CUDA-compatible GPU is available, `False` otherwise.

        Notes
        -----
        This method attempts a simple GPU operation using the `cupy` library. 
        If `cupy` is installed and a GPU is accessible, the function returns 
        `True`.

        Examples
        --------
        >>> backend_selector = BackendSelector()
        >>> backend_selector.is_gpu_available()
        True
        """

        try:
            import cupy
            cupy.array([1])
            return True
        except (ImportError, cupy.cuda.runtime.CUDARuntimeError):
            return False

    def is_dask_available(self):
        """
        Verifies the availability of the Dask library for distributed computing.

        Returns
        -------
        bool
            `True` if Dask is installed, `False` otherwise.

        Notes
        -----
        Dask is used for large-scale computations on distributed systems. 
        This method confirms its installation without initializing any 
        Dask components.

        Examples
        --------
        >>> backend_selector = BackendSelector()
        >>> backend_selector.is_dask_available()
        True
        """

        try:
            import dask #noqa
            return True
        except ImportError:
            return False

    def is_scipy_available(self):
        """
        Checks whether the SciPy library is available for scientific 
        computations.

        Returns
        -------
        bool
            `True` if SciPy is installed, `False` otherwise.

        Notes
        -----
        SciPy provides advanced scientific and numerical computing functions. 
        This method ensures SciPy is available for computations requiring 
        high precision and functionality beyond what is offered by NumPy.

        Examples
        --------
        >>> backend_selector = BackendSelector()
        >>> backend_selector.is_scipy_available()
        True
        """

        try:
            import scipy # # noqa
            return True
        except ImportError:
            return False

    def auto_select_backend(self):
        """
        Automatically selects the most suitable general backend based on 
        system capabilities, prioritizing GPU-accelerated computations 
        where available.

        Returns
        -------
        backend_instance : object
            Instance of the automatically selected general backend class.

        Notes
        -----
        This method follows a priority order:
            - `"cupy"` if a GPU is available,
            - `"dask"` if Dask is installed,
            - `"scipy"` if SciPy is installed,
            - `"numpy"` as the default.

        Examples
        --------
        >>> backend_selector = BackendSelector()
        >>> backend = backend_selector.auto_select_backend()
        CuPyBackend selected for general computations.
        """

        if self.is_gpu_available():
            selected_backend = "cupy"
        elif self.is_dask_available():
            selected_backend = "dask"
        elif self.is_scipy_available():
            selected_backend = "scipy"
        else:
            selected_backend = "numpy"
        if self.verbose > 0:
            print(f"{selected_backend.capitalize()}Backend"
                  " selected for general computations.")
        return self.backends[selected_backend]()

    def auto_select_nn_backend(self):
        """
        Automatically selects the most suitable neural network backend, 
        prioritizing TensorFlow if both TensorFlow and PyTorch are available.

        Returns
        -------
        nn_backend_instance : object
            Instance of the automatically selected neural network backend class.

        Raises
        ------
        ImportError
            If neither TensorFlow nor PyTorch is available.

        Notes
        -----
        This method attempts to import TensorFlow and PyTorch in order. 
        If TensorFlow is found, it is selected as the default neural network 
        backend. If TensorFlow is unavailable, PyTorch is chosen if available.

        Examples
        --------
        >>> backend_selector = BackendSelector()
        >>> nn_backend = backend_selector.auto_select_nn_backend()
        TensorFlow selected for neural network computations.
        """

        try:
            import tensorflow as tf # noqa
            if self.verbose > 0:
                print("TensorFlow selected for neural network computations.")
            return self.nn_backends["tensorflow"]
        except ImportError:
            try:
                import torch # noqa
                if self.verbose > 0:
                    print("PyTorch selected for neural network computations.")
                return self.nn_backends["pytorch"]
            except ImportError:
                raise ImportError(
                    "Neither TensorFlow nor PyTorch is"
                    " available for neural network computations.")

    def get_backend(self):
        """
        Retrieves the selected general backend instance for performing 
        computational tasks.

        Returns
        -------
        backend_instance : object
            The instance of the selected general backend class.

        Examples
        --------
        >>> backend_selector = BackendSelector(preferred_backend="scipy")
        >>> backend = backend_selector.get_backend()
        """

        return self.selected_backend

    def get_nn_backend(self):
        """
        Retrieves the selected neural network backend instance for 
        performing deep learning tasks.

        Returns
        -------
        nn_backend_instance : object
            The instance of the selected neural network backend class.

        Examples
        --------
        >>> backend_selector = BackendSelector(preferred_nn_backend="tensorflow")
        >>> nn_backend = backend_selector.get_nn_backend()
        """

        return self.selected_nn_backend
 

@ensure_pkg(
    "torch", 
    extra="backend is set to ``torch`` while it is not installed.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get("backend") in ("torch", "pytorch")
    )
@ensure_pkg(
    "tensorflow", 
    extra="backend is set to ``tensorflow`` while it is not installed.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get("backend") in ("tensorflow", "tf")
    )

def select_backend_n(
        backend=None, return_module=False, return_both=False):
    """
    Select the backend for computation based on the input string.

    This function maps various input string representations to standardized 
    backend names or their corresponding modules. It is used to choose the 
    appropriate backend for performing computations, such as NumPy, TensorFlow, 
    or PyTorch. This allows the user to provide different variations of backend 
    names, and the function will return the corresponding backend in a 
    consistent format or the actual backend module.

    Parameters
    ----------
    backend : str or None, optional
        The backend to use for computation. Accepts the following values:
        - `None`, `'numpy'`, or `'np'` for NumPy (default).
        - `'torch'`, `'pytorch'` for PyTorch.
        - `'tensorflow'`, `'tf'` for TensorFlow.

        The parameter is case-insensitive, so variations like `'TensorFlow'`, 
        `'TF'`, or `'np'` are also valid. If `None` is provided, the default 
        backend will be NumPy.
    
    return_module : bool, default=False
        If `True`, the function returns the actual backend module (`numpy`, 
        `torch`, or `tensorflow`). If `False`, it returns the standardized 
        backend string (`'numpy'`, `'torch'`, or `'tensorflow'`).
    
    return_both : bool, default=False
        If `True`, the function returns a tuple containing both the standardized 
        backend string and the corresponding backend module. This is useful 
        when both the name and the module are needed for further operations.

        - When `return_both` is `True`, the function returns:
          (`'numpy'`, numpy_module), (`'torch'`, torch_module), or 
          (`'tensorflow'`, tensorflow_module).

        - If `return_both` is `True`, the `return_module` parameter is ignored.

    Returns
    -------
    str or module or tuple of (str, module)
        - If `return_both` is `True`, returns a tuple containing the standardized 
          backend string and the corresponding backend module.
        - If `return_module` is `True`, returns the corresponding backend 
          module:
          - `numpy` module for `'numpy'`.
          - `torch` module for `'torch'`.
          - `tensorflow` module for `'tensorflow'`.
        - If both `return_module` and `return_both` are `False`, returns the 
          standardized backend string, one of `'numpy'`, `'torch'`, or 
          `'tensorflow'`. 

    Raises
    ------
    ValueError
        If the provided `backend` is not recognized, a `ValueError` is raised 
        with a message indicating the valid options.

    Notes
    -----
    - The backend parameter allows flexibility for users to choose between 
      different backend libraries (e.g., NumPy for standard computation, 
      TensorFlow for GPU-accelerated computation, or PyTorch for deep learning 
      tasks).
    - If `None` is passed, the function defaults to NumPy, which is the 
      fallback option for most computations.
    - The backend parameter is case-insensitive, so it will handle different 
      cases of the backend names (e.g., `'TensorFlow'`, `'TF'` will be 
      correctly mapped to `'tensorflow'`).
    - If an unsupported backend is provided, a `ValueError` will be raised.
    - When `return_module` or `return_both` is `True`, ensure that the 
      corresponding backend library is installed in your environment to avoid 
      `ImportError`.

    Examples
    --------
    >>> from gofast.backends.selector import select_backend_n 
    >>> select_backend_n('tf')
    'tensorflow'
    
    >>> select_backend_n('PyTorch')
    'torch'
    
    >>> select_backend_n('np')
    'numpy'
    
    >>> select_backend_n(None)
    'numpy'
    
    >>> select_backend_n('tf', return_module=True)
    <module 'tensorflow' from '...'>
    
    >>> select_backend_n('torch', return_module=True)
    <module 'torch' from '...'>
    
    >>> select_backend_n('numpy', return_module=True)
    <module 'numpy' from '...'>
    
    >>> select_backend_n('tf', return_both=True)
    ('tensorflow', <module 'tensorflow' from '...'>)
    
    >>> select_backend_n('torch', return_both=True)
    ('torch', <module 'torch' from '...'>)
    
    >>> select_backend_n('numpy', return_both=True)
    ('numpy', <module 'numpy' from '...'>)
    
    >>> select_backend_n('invalid_backend')
    Traceback (most recent call last):
        ...
    ValueError: Unsupported backend: invalid_backend. Supported backends ...

    See Also
    --------
    TensorFlow : A powerful open-source machine learning library, often used 
    for large-scale deep learning tasks.
    
    PyTorch : A deep learning framework popular for research and production.
    
    NumPy : A library for numerical computing in Python, providing support 
    for large multi-dimensional arrays and matrices.
    """
    backend_map = {
        None: "numpy",  # Default to numpy if backend is None
        "numpy": "numpy", "np": "numpy",
        "torch": "torch", "pytorch": "torch",
        "tensorflow": "tensorflow", "tf": "tensorflow"
    }

    normalized_backend = (
        backend_map.get(backend.lower()) 
        if isinstance(backend, str) 
        else backend_map.get(backend)
    )

    if normalized_backend is None:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends are "
            "'numpy', 'tensorflow', and 'torch'."
        )
    
    module = None
    if return_module or return_both:
        if normalized_backend == "numpy":
            import numpy as np
            module = np
        elif normalized_backend == "torch":
            import torch
            module = torch
        elif normalized_backend == "tensorflow":
            import tensorflow as tf
            module = tf

    if return_both:
        return normalized_backend, module
    elif return_module:
        return module
    else:
        return normalized_backend

def safe_cast(backend, value, dtype, backend_name=None):
    """
    Helper function to cast values to a specified data type across multiple
    backends.
    
    This function facilitates the conversion of data types for various computational
    backends, including NumPy, TensorFlow, and PyTorch. It allows users to specify
    the backend either by name or by passing the backend module directly. The
    function ensures type safety and compatibility by performing the casting
    operation in a manner specific to the selected backend.
    
    Parameters
    ----------
    backend : str or module
        The backend to use for casting. It can be a string (e.g., ``'numpy'``,
        ``'tf'``, ``'torch'``) or the actual backend module (e.g., ``numpy``,
        ``tensorflow``, ``torch``).
    value : scalar, array-like, TensorFlow tensor, or PyTorch tensor
        The value to be cast. It can be a scalar, NumPy array, TensorFlow tensor,
        or PyTorch tensor.
    dtype : data-type
        The target data type. It should be compatible with the backend's dtype
        system.
    backend_name : str, optional
        Explicitly specify the backend name if ``backend`` is a module. Default
        is ``None``.
    
    Returns
    -------
    casted_value : same type as `value`
        The value cast to the specified ``dtype``, compatible with the active
        backend.
    
    Raises
    ------
    ValueError
        If the backend is unsupported or if the ``dtype`` is incompatible.
    TypeError
        If the input types are incorrect.
    
    Formulation
    ------------
    .. math::
        y = \text{cast}(x, \text{dtype})
    
    Examples
    --------
    Using NumPy backend:
    
    >>> import numpy as np
    >>> from gofast.backends.selector import safe_cast
    >>> value = [1, 2, 3]
    >>> dtype = np.float32
    >>> casted_value = safe_cast('numpy', value, dtype)
    >>> print(casted_value)
    [1. 2. 3.]
    
    Using TensorFlow backend:
    
    >>> import tensorflow as tf
    >>> from gofast.backends.selector import safe_cast
    >>> value = tf.constant([1, 2, 3], dtype=tf.int32)
    >>> dtype = tf.float32
    >>> casted_value = safe_cast('tf', value, dtype)
    >>> print(casted_value)
    tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32)
    
    Using PyTorch backend:
    
    >>> import torch
    >>> from gofast.backends.selector import safe_cast
    >>> value = torch.tensor([1, 2, 3], dtype=torch.int32)
    >>> dtype = torch.float32
    >>> casted_value = safe_cast('torch', value, dtype)
    >>> print(casted_value)
    tensor([1., 2., 3.])
    
    Notes
    -----
    The ``safe_cast`` function dynamically imports the required backend module based on the
    specified backend name or module. If the backend is not installed, an ``ImportError``
    will be raised, prompting the user to install the necessary package.
    
    :math:`y = \text{cast}(x, \text{dtype})` represents the casting operation, where
    :math:`x` is the input value and :math:`y` is the output value after casting.
    
    See also
    --------
    ``numpy.asarray``, ``tensorflow.cast``, ``torch.Tensor.to``
    
    References
    ----------
    .. [1] Waskom, M. (2021). *Seaborn: Statistical Data Visualization*. Journal of
        Open Source Software, 6(60), 3021.
    """

    import types

    # Mapping of backend aliases to standardized names
    BACKEND_ALIASES = {
        'numpy': ['numpy', 'np'],
        'tensorflow': ['tensorflow', 'tf'],
        'torch': ['torch', 'pytorch', 'pt']
    }

    # Mapping of standardized names to import names
    BACKEND_IMPORT_NAMES = {
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'torch': 'torch'
    }

    # Helper function to get standardized backend name
    def get_standard_backend_name(backend_input):
        if isinstance(backend_input, str):
            backend_input_lower = backend_input.lower()
            for std_name, aliases in BACKEND_ALIASES.items():
                if backend_input_lower in aliases:
                    return std_name
            raise ValueError(
                f"Unsupported backend name: '{backend_input}'. Supported "
                f"backends are: {list(BACKEND_ALIASES.keys())}."
            )
        elif isinstance(backend_input, types.ModuleType):
            for std_name, import_name in BACKEND_IMPORT_NAMES.items():
                if backend_input.__name__.startswith(std_name):
                    return std_name
            raise ValueError(
                f"Unsupported backend module: '{backend_input.__name__}'. Supported "
                f"backends are: {list(BACKEND_ALIASES.keys())}."
            )
        else:
            raise TypeError(
                f"Backend must be a string or a module, got {type(backend_input)}."
            )

    # Determine the backend standardized name
    if backend_name is not None:
        if not isinstance(backend_name, str):
            raise TypeError(
                f"'backend_name' must be a string, got {type(backend_name)}."
            )
        backend_std_name = backend_name.lower()
        # Validate backend_name
        if backend_std_name not in BACKEND_ALIASES:
            raise ValueError(
                f"Unsupported backend name: '{backend_name}'. Supported "
                f"backends are: {list(BACKEND_ALIASES.keys())}."
            )
    else:
        # Infer backend name from 'backend'
        backend_std_name = get_standard_backend_name(backend)

    # Import the backend module dynamically
    try:
        if backend_std_name == 'numpy':
            import numpy as np
            backend_module = np
        elif backend_std_name == 'tensorflow':
            import tensorflow as tf
            backend_module = tf
        elif backend_std_name == 'torch':
            import torch
            backend_module = torch
        else:
            # This case should not occur due to earlier validations
            raise ValueError(
                f"Unsupported backend: '{backend_std_name}'. Supported "
                f"backends are: {list(BACKEND_ALIASES.keys())}."
            )
    except ImportError:
        raise ImportError(
            f"The backend '{backend_std_name}' is not installed. "
            f"Please install it to use this functionality."
        )

    # Perform casting based on the backend
    try:
        if backend_std_name == 'numpy':
            # NumPy: Use np.asarray with dtype
            return backend_module.asarray(value, dtype=dtype)

        elif backend_std_name == 'tensorflow':
            # TensorFlow: Use tf.cast
            return backend_module.cast(value, dtype=dtype)

        elif backend_std_name == 'torch':
            # PyTorch:
            # If value is already a PyTorch tensor, use .to(dtype)
            # Otherwise, create a new tensor with the specified dtype
            if isinstance(value, backend_module.Tensor):
                return value.to(dtype)
            else:
                return backend_module.tensor(
                    value, 
                    dtype=dtype
                )

        else:
            # This case should not occur due to earlier validations
            raise ValueError(
                f"Unsupported backend: '{backend_std_name}'."
            )

    except Exception as e:
        raise ValueError(
            f"Error casting value with backend '{backend_std_name}': {e}"
        )

def check_processor(
    authorized: Literal["gpu", "cpu", "auto"] = "auto",
    error: Literal["raise", "warn"] = "raise",
    verbose: int = 0,
    backend: Literal["auto", "pytorch", "tensorflow", "nvidia_smi"] = "auto"
) -> bool:
    """
    Check availability of specified processor and handle errors accordingly.

    Parameters
    ----------
    authorized : {'gpu', 'cpu', 'auto'}, default='auto'
        Target processor type to check:
        - 'gpu': Only allow GPU computation (raise error if unavailable)
        - 'cpu': Use CPU computation (always available)
        - 'auto': Automatically detect and prefer GPU if available

    error : {'raise', 'warn'}, default='raise'
        Error handling strategy when GPU is unavailable but required

    verbose : int, default=0
        Verbosity level (0: silent, 1: informational)

    backend : {'auto', 'pytorch', 'tensorflow', 'nvidia_smi'}, default='auto'
        Backend to use for GPU detection:
        - 'auto': Try all backends in order
        - 'pytorch': Use PyTorch's CUDA check
        - 'tensorflow': Use TensorFlow's GPU device check
        - 'nvidia_smi': Use nvidia-smi command line tool

    Returns
    -------
    bool
        True if authorized processor is available, False otherwise

    Examples
    --------
    >>> from gofast.backends.selector check_processor
    >>> check_processor(authorized='gpu', verbose=1)
    Checking GPU via PyTorch...
    GPU is available.
    True

    >>> check_processor(authorized='cpu')
    True
    """
    authorized = authorized.lower()
    error = error.lower()
    
    if authorized not in ("gpu", "cpu", "auto"):
        raise ValueError(f"Invalid authorized value: {authorized}. "
                         "Choose from ['gpu', 'cpu', 'auto']")
    
    if authorized == "cpu":
        if verbose:
            print("CPU is always available")
        return True

    # GPU check implementations
    def _check_pytorch() -> bool:
        try:
            import torch
            if verbose:
                print("Checking GPU via PyTorch...")
            return torch.cuda.is_available()
        except ImportError:
            if verbose:
                print("PyTorch not installed")
            return False

    def _check_tensorflow() -> bool:
        try:
            from tensorflow.config import list_physical_devices
            if verbose:
                print("Checking GPU via TensorFlow...")
            return len(list_physical_devices("GPU")) > 0
        except ImportError:
            if verbose:
                print("TensorFlow not installed")
            return False

    def _check_nvidia_smi() -> bool:
        try:
            if verbose:
                print("Checking GPU via nvidia-smi...")
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            if verbose:
                print("nvidia-smi check failed")
            return False

    backends = {
        "pytorch": _check_pytorch,
        "tensorflow": _check_tensorflow,
        "nvidia_smi": _check_nvidia_smi
    }

    gpu_available = False
    
    if backend == "auto":
        for name, checker in backends.items():
            if checker():
                gpu_available = True
                if verbose:
                    print(f"GPU detected via {name}")
                break
    else:
        if backend not in backends:
            raise ValueError(f"Invalid backend: {backend}. "
                             "Choose from ['auto', 'pytorch', 'tensorflow', 'nvidia_smi']")
        gpu_available = backends[backend]()

    if authorized == "gpu" and not gpu_available:
        msg = "GPU computation required but no GPU detected"
        if error == "raise":
            raise RuntimeError(msg)
        elif error == "warn":
            warnings.warn(msg)
            return False
        return False

    if verbose:
        status = "available" if gpu_available else "unavailable"
        print(f"GPU {status}" + ("" if authorized == "auto" else " and required"))
    
    return gpu_available if authorized in ("gpu", "auto") else True