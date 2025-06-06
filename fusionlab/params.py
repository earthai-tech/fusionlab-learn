# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Module `fusionlab.params` provides simple, self-documenting classes to
specify how the PINN’s physical coefficient :math:`C` should be handled:

- `LearnableC`: learn :math:`C` (i.e., trainable), initialized via
  `initial_value`.
- `FixedC`: keep :math:`C` fixed (non-trainable) at a specified value.
- `DisabledC`: disable physics (treat :math:`C` as 1.0 internally,
  but unused).

These classes make the model signature clearer than passing bare strings
or floats.  When building the PINN, one checks `isinstance(..., LearnableC)`,
etc., and sets up trainable weights accordingly.
"""


from typing import Union
import importlib

# Attempt to import TensorFlow, else fall back to NumPy
_tf_spec = importlib.util.find_spec("tensorflow")
if _tf_spec is not None:
    import tensorflow as tf
    _BACKEND = "tensorflow"
else:
    import numpy as np
    _BACKEND = "numpy"
    
__all__ = ["LearnableC", "FixedC", "DisabledC"]


class LearnableC:
    """
    Indicates that the PINN’s physical coefficient :math:`C` should be
    learned (trainable).  We actually learn :math:`\log(C)` to ensure
    :math:`C > 0`.  The user supplies an `initial_value`, and the model
    initializes:

    .. math::
       \log C \;=\; \log(\text{initial\_value}).

    Attributes
    ----------
    initial_value : float
        The positive starting value for :math:`C`.  Must be strictly
        positive.

    Examples
    --------
    >>> from fusionlab.params import LearnableC
    >>> # Learn C, starting from C = 0.01
    >>> pinn_coeff = LearnableC(initial_value=0.01)
    >>> # Learn C, starting from C = 0.001
    >>> pinn_coeff_small = LearnableC(initial_value=0.001)
    """
    def __init__(self, initial_value: float = 0.01):
        # Validate type
        if not isinstance(initial_value, (float, int)):
            raise TypeError(
                f"LearnableC.initial_value must be a float, got "
                f"{type(initial_value).__name__}"
            )
        # Validate positivity
        if initial_value <= 0:
            raise ValueError(
                "LearnableC.initial_value must be strictly positive."
            )
        self.initial_value = float(initial_value)


class FixedC:
    """
    Indicates that the PINN’s physical coefficient :math:`C` should be
    held fixed (non-trainable) at a specified `value`.

    .. math::
       C = \text{value}, \qquad \text{non-trainable}.

    Attributes
    ----------
    value : float
        The non-negative, constant value of :math:`C`.

    Examples
    --------
    >>> from fusionlab.params import FixedC
    >>> # Use a fixed C = 0.5
    >>> pinn_coeff = FixedC(value=0.5)
    """
    def __init__(self, value: float):
        # Validate type
        if not isinstance(value, (float, int)):
            raise TypeError(
                f"FixedC.value must be a float, got {type(value).__name__}"
            )
        # Validate non-negativity
        if value < 0:
            raise ValueError("FixedC.value must be non-negative.")
        self.value = float(value)

class DisabledC:
    """
    Indicates that physics should be disabled.  In practice, :math:`C` is
    irrelevant (defaults to 1.0 internally, but is never used if
    `lambda_pde == 0` when compiling).

    Attributes
    ----------
    None

    Examples
    --------
    >>> from fusionlab.params import DisabledC
    >>> pinn_coeff = DisabledC()
    """
    def __init__(self):
        # No parameters needed.  Presence of this class signals “disable”.
        pass


class LearnableK:
    """
    Indicates that the PINN’s hydraulic conductivity :math:`K` should be
    learned (trainable) if TensorFlow is available; otherwise behaves as
    a fixed NumPy‐based parameter. We learn :math:`\log(K)` to ensure
    :math:`K > 0`. The user supplies an `initial_value`, and the object
    initializes:

    .. math::
       \log K \;=\; \log(\text{initial\_value}).

    Attributes
    ----------
    initial_value : float
        The positive starting value for :math:`K`. Must be strictly
        positive.
    log_K : tf.Variable or float
        - If TensorFlow is available: a trainable tf.Variable representing
          \(\log(K)\), initialized to \(\log(\text{initial_value})\).
        - If TensorFlow is not available: a Python float equal to
          \(\log(\text{initial_value})\).

    Examples
    --------
    >>> from fusionlab.params import LearnableK
    >>> # TensorFlow installed: LearnableK uses tf.Variable internally
    >>> learn_K = LearnableK(initial_value=1.0)
    >>> learn_K.get_K()  # returns tf.Tensor
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>
    >>> 
    >>> # If TensorFlow is not installed: LearnableK uses NumPy internally
    >>> learn_K_np = LearnableK(initial_value=2.0)
    >>> print(learn_K_np.get_K())  # returns numpy.float32
    2.0
    """
    def __init__(self, initial_value: float = 1.0):
        # Validate type
        if not isinstance(initial_value, (float, int)):
            raise TypeError(
                f"LearnableK.initial_value must be a float, got "
                f"{type(initial_value).__name__}"
            )
        # Validate positivity
        if initial_value <= 0:
            raise ValueError(
                "LearnableK.initial_value must be strictly positive."
            )
        self.initial_value = float(initial_value)

        if _BACKEND == "tensorflow":
            # Create a trainable variable for log(K)
            self.log_K = tf.Variable(
                initial_value=tf.math.log(self.initial_value),
                trainable=True,
                dtype=tf.float32,
                name="log_K"
            )
        else:
            # Fall back to NumPy: store log(K) as float
            self.log_K = np.log(self.initial_value)  # type: ignore

    def get_K(self) -> Union["tf.Tensor", float]:
        """
        Returns the positive conductivity K by exponentiating log_K.

        Returns
        -------
        tf.Tensor or float
            - If TensorFlow backend: A scalar tf.Tensor representing K > 0.
            - Otherwise: A NumPy float representing K > 0.
        """
        if _BACKEND == "tensorflow":
            return tf.exp(self.log_K)
        else:
            return float(np.exp(self.log_K))  # type: ignore


class LearnableSs:
    """
    Indicates that the PINN’s specific storage coefficient :math:`S_s`
    should be learned (trainable) if TensorFlow is available; otherwise acts
    as a fixed NumPy‐based parameter. We learn :math:`\log(S_s)` to ensure
    :math:`S_s > 0`. The user supplies an `initial_value`, and the object
    initializes:

    .. math::
       \log S_s \;=\; \log(\text{initial\_value}).

    Attributes
    ----------
    initial_value : float
        The positive starting value for :math:`S_s`. Must be strictly
        positive.
    log_Ss : tf.Variable or float
        - If TensorFlow is available: a trainable tf.Variable representing
          \(\log(S_s)\), initialized to \(\log(\text{initial_value})\).
        - If TensorFlow is not available: a Python float equal to
          \(\log(\text{initial_value})\).

    Examples
    --------
    >>> from fusionlab.params import LearnableSs
    >>> # TensorFlow installed: LearnableSs uses tf.Variable
    >>> learn_Ss = LearnableSs(initial_value=1e-4)
    >>> learn_Ss.get_Ss().numpy()
    0.0001
    >>> 
    >>> # If TensorFlow not installed: uses NumPy
    >>> learn_Ss_np = LearnableSs(initial_value=1e-3)
    >>> print(learn_Ss_np.get_Ss())
    0.001
    """
    def __init__(self, initial_value: float = 1e-4):
        # Validate type
        if not isinstance(initial_value, (float, int)):
            raise TypeError(
                f"LearnableSs.initial_value must be a float, got "
                f"{type(initial_value).__name__}"
            )
        # Validate positivity
        if initial_value <= 0:
            raise ValueError(
                "LearnableSs.initial_value must be strictly positive."
            )
        self.initial_value = float(initial_value)

        if _BACKEND == "tensorflow":
            # Create a trainable variable for log(Ss)
            self.log_Ss = tf.Variable(
                initial_value=tf.math.log(self.initial_value),
                trainable=True,
                dtype=tf.float32,
                name="log_Ss"
            )
        else:
            # Fall back to NumPy: store log(Ss) as float
            self.log_Ss = np.log(self.initial_value)  # type: ignore

    def get_Ss(self) -> Union["tf.Tensor", float]:
        """
        Returns the positive storage coefficient Ss by exponentiating log_Ss.

        Returns
        -------
        tf.Tensor or float
            - If TensorFlow backend: A scalar tf.Tensor representing Ss > 0.
            - Otherwise: A NumPy float representing Ss > 0.
        """
        if _BACKEND == "tensorflow":
            return tf.exp(self.log_Ss)
        else:
            return float(np.exp(self.log_Ss))  # type: ignore


class LearnableQ:
    """
    Indicates that the PINN’s source/sink term :math:`Q` should be
    learned (trainable) if TensorFlow is available; otherwise acts as a
    fixed NumPy‐based parameter. Unlike K and Ss, Q may be positive or
    negative, so we learn it directly (no log‐transform). The user supplies
    an `initial_value`, and the object initializes:

    .. math::
       Q \;=\; \text{initial\_value}.

    Attributes
    ----------
    initial_value : float
        The starting value for :math:`Q`. May be positive, negative, or zero.
    Q_var : tf.Variable or float
        - If TensorFlow is available: a trainable tf.Variable representing Q.
        - Otherwise: a Python float equal to the initial value.

    Examples
    --------
    >>> from fusionlab.params import LearnableQ
    >>> # TensorFlow installed: LearnableQ uses tf.Variable
    >>> learn_Q = LearnableQ(initial_value=0.1)
    >>> learn_Q.get_Q().numpy()
    0.1
    >>> 
    >>> # If TensorFlow not installed: uses NumPy
    >>> learn_Q_np = LearnableQ(initial_value=-0.05)
    >>> print(learn_Q_np.get_Q())
    -0.05
    """
    def __init__(self, initial_value: float = 0.0):
        # Validate type
        if not isinstance(initial_value, (float, int)):
            raise TypeError(
                f"LearnableQ.initial_value must be a float, got "
                f"{type(initial_value).__name__}"
            )
        self.initial_value = float(initial_value)

        if _BACKEND == "tensorflow":
            # Create a trainable variable for Q (no log‐transform)
            self.Q_var = tf.Variable(
                initial_value=tf.convert_to_tensor(
                    self.initial_value, dtype=tf.float32
                ),
                trainable=True,
                name="Q"
            )
        else:
            # Fall back to NumPy: store Q as float
            self.Q_var = float(self.initial_value)

    def get_Q(self) -> Union["tf.Tensor", float]:
        """
        Returns the trainable Q coefficient.

        Returns
        -------
        tf.Tensor or float
            - If TensorFlow backend: A scalar tf.Tensor for Q.
            - Otherwise: A NumPy float for Q.
        """
        if _BACKEND == "tensorflow":
            return tf.identity(self.Q_var)
        else:
            return float(self.Q_var)  # type: ignore


def resolve_physical_param(
    param: Union[float, LearnableK, LearnableSs, LearnableQ]
) -> Union["tf.Tensor", float]:
    """
    Helper that returns a scalar (tf.Tensor or float) given either a raw float
    or a learnable parameter instance (LearnableK, LearnableSs, LearnableQ).

    Parameters
    ----------
    param : Union[float, LearnableK, LearnableSs, LearnableQ]
        If float, simply returns as:
          - `tf.constant(param, dtype=tf.float32)` if TensorFlow backend.
          - `float(param)` if NumPy backend.
        If instance, calls its `get_<...>()` method to obtain the tensor/float.

    Returns
    -------
    tf.Tensor or float
        The scalar representing the requested parameter.

    Raises
    ------
    TypeError
        If `param` is not a float or one of the Learnable* classes.
    """
    if isinstance(param, LearnableK):
        return param.get_K()
    if isinstance(param, LearnableSs):
        return param.get_Ss()
    if isinstance(param, LearnableQ):
        return param.get_Q()

    # Plain float or int
    if isinstance(param, (float, int)):
        if _BACKEND == "tensorflow":
            return tf.cast(param, tf.float32)
        else:
            return float(param)

    raise TypeError(
        f"Parameter must be float, LearnableK, LearnableSs, or LearnableQ; "
        f"got {type(param).__name__}"
    )
