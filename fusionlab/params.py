# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Module `fusionlab.params` provides simple, self-documenting classes to
specify how the PINN's physical coefficient :math:`C` should be handled:

- `LearnableC`: learn :math:`C` (i.e., trainable), initialized via
  `initial_value`.
- `FixedC`: keep :math:`C` fixed (non-trainable) at a specified value.
- `DisabledC`: disable physics (treat :math:`C` as 1.0 internally,
  but unused).

These classes make the model signature clearer than passing bare strings
or floats.  When building the PINN, one checks `isinstance(..., LearnableC)`,
etc., and sets up trainable weights accordingly.
"""
from __future__ import annotations
import importlib
from typing import Any, Union, Optional, Dict
from abc import ABC, abstractmethod

# Attempt to import TensorFlow, else fall
# back to NumPy
_tf_spec = importlib.util.find_spec(
    "tensorflow"
)
if _tf_spec is not None:
    import tensorflow as tf
    _BACKEND = "tensorflow"
    Tensor = tf.Tensor
    Variable = tf.Variable
else:
    import numpy as np

    _BACKEND = "numpy"
    class _DummyTF:
        pass

    class tf:
        Tensor = _DummyTF
    # Fallback types for type hinting
    Tensor = Any
    Variable = Any


__all__ = ["LearnableC", "FixedC", "DisabledC", "LearnableK", "LearnableSs", 
           "LearnableQ"
           ]



class BaseLearnable(ABC):
    """
    Abstract base for learnable physical parameters.

    Parameters
    ----------
    initial_value : float
        Initial numeric value for the parameter.
    name : str
        Unique identifier for the variable.
    log_transform : bool, optional
        If True, store in log-space for positivity
        constraint, by default False.
    trainable : bool, optional
        If True, make variable trainable, by
        default True.

    Attributes
    ----------
    initial_value : float
        The original provided value.
    name : str
        Variable name in the computation graph.
    log_transform : bool
        Whether to apply log transform.
    trainable : bool
        Trainable flag for optimization.

    Examples
    --------
    >>> param = LearnableK(initial_value=0.5)
    >>> value = param.get_value()
    """
    def __init__(
        self,
        initial_value: float,
        name: str,
        log_transform: bool = False,
        trainable: bool = True,
    ):
        if not isinstance(
            initial_value, (float, int)
        ):
            raise TypeError(
                f"Initial value for {self.__class__.__name__} "
                f"must be a float, got {type(initial_value).__name__}"
            )
        if log_transform and initial_value <= 0:
            raise ValueError(
                f"{self.__class__.__name__} initial value must be "
                "strictly positive for log transform."
            )
        self.initial_value = float(initial_value)
        self.name = name
        self.log_transform = log_transform
        self.trainable = trainable
        self._variable = self._create_variable()

    def _create_variable(self) -> Union[Variable, Tensor, float]:
        """
        Internal: create tf.Variable or fallback value.

        Returns
        -------
        Union[Variable, Tensor, float]
            Configured variable or numeric.
        """
        if _BACKEND == "tensorflow":
            value = self.initial_value
            if self.log_transform:
                value = tf.math.log(value)
            return tf.Variable(
                initial_value=tf.cast(
                    value, dtype=tf.float32
                ),
                trainable=self.trainable,
                name=self.name
            )
        return (
            np.log(self.initial_value)
            if self.log_transform else
            self.initial_value
        )

    @abstractmethod
    def get_value(
        self
    ) -> Union[Tensor, float]:
        """
        Retrieve parameter value.

        Returns
        -------
        Union[Tensor, float]
            Transformed parameter, e.g.,
            :math:`\exp(log\_param)` if
            log_transform is True.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(initial_value="
            f"{self.initial_value}, trainable={self.trainable})"
        )


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


class LearnableK(BaseLearnable):
    """
    Learnable Hydraulic Conductivity (K).

    Indicates that the PINN’s hydraulic conductivity :math:`K` should be
    learned (trainable) if TensorFlow is available; otherwise behaves as
    a fixed NumPy‐based parameter. We learn :math:`\log(K)` to ensure
    :math:`K > 0`. The user supplies an `initial_value`, and the object
    initializes:

    .. math::
       \log K \;=\; \log(\text{initial\_value}).


    Ensures positivity via log-space.

    See Also
    --------
    BaseLearnableParam

    Examples
    --------
    >>> k = LearnableK(1.2)
    >>> :math:`K = k.get_value()`
    """
    def __init__(
        self,
        initial_value: float = 1.0, 
        name: Optional[str] =None, 
    ):
        super().__init__(
            initial_value=initial_value,
            name= name or "learnable_K",
            log_transform=True
        )

    def get_value(
        self
    ) -> Union[Tensor, float]:
        """
        Return :math:`K = \exp(log\_K)`.

        Returns
        -------
        Union[Tensor, float]
            Positive conductivity.
        """
        if _BACKEND == "tensorflow":
            return tf.exp(self._variable)
        return float(
            __import__("numpy").exp(
                self._variable
            )
        )


class LearnableSs(BaseLearnable):
    """
    Learnable Specific Storage (Ss).
    
    Indicates that the PINN's specific storage coefficient :math:`S_s`
    should be learned (trainable) if TensorFlow is available; otherwise acts
    as a fixed NumPy‐based parameter. We learn :math:`\log(S_s)` to ensure
    :math:`S_s > 0`. The user supplies an `initial_value`, and the object
    initializes:

    .. math::
       \log S_s \;=\; \log(\text{initial\_value}).

    Returns positive values via exp transform.
    

    Examples
    --------
    >>> ss = LearnableSs(1e-3)
    >>> value = ss.get_value()
    """
    def __init__(
        self,
        initial_value: float = 1e-4, 
        name: Optional[str] =None, 
    ):
        super().__init__(
            initial_value=initial_value,
            name= name or "learnable_Ss",
            log_transform=True
        )

    def get_value(
        self
    ) -> Union[Tensor, float]:
        """
        Return :math:`Ss = \exp(log\_Ss)`.

        Returns
        -------
        Union[Tensor, float]
            Positive storage coefficient.
        """
        if _BACKEND == "tensorflow":
            return tf.exp(self._variable)
        return float(
            __import__("numpy").exp(
                self._variable
            )
        )

class LearnableQ(BaseLearnable):
    """
    Learnable Source/Sink term (Q).

    Indicates that the PINN's source/sink term :math:`Q` should be
    learned (trainable) if TensorFlow is available; otherwise acts as a
    fixed NumPy‐based parameter. Unlike K and Ss, Q may be positive or
    negative, so we learn it directly (no log‐transform). The user supplies
    an `initial_value`, and the object initializes:

    .. math::
       Q \;=\; \text{initial\_value}.
       
    Unconstrained: may be positive or
    negative.

    Examples
    --------
    >>> q = LearnableQ(0.0)
    >>> q.get_value()
    0.0
    """
    def __init__(
        self,
        initial_value: float = 0.0, 
        name: Optional[str] =None, 
    ):
        super().__init__(
            initial_value=initial_value,
            name= name or "learnable_Q",
            log_transform=False
        )

    def get_value(
        self
    ) -> Union[Tensor, float]:
        """
        Return raw :math:`Q` value.

        Returns
        -------
        Union[Tensor, float]
            Source/sink strength.
        """
        return self._variable


def resolve_physical_param(
    param: Any,
    name: Optional[str] = None,
    serialize: bool = False
) -> Union[Tensor, float, Dict]:
    """
    Convert parameter to tensor, float, or dict.

    Parameters
    ----------
    param : Any
        Numeric or BaseLearnableParam instance.
    name : str, optional
        Unused: name handled by class.
    serialize : bool, optional
        If True, return serializable dict.

    Returns
    -------
    Union[Tensor, float, Dict]
        Resolved parameter or metadata.

    Raises
    ------
    TypeError
        If param type is unsupported.

    Examples
    --------
    >>> resolve_physical_param(5.0)
    5.0
    >>> resolve_physical_param(
    ...     LearnableK(0.5), serialize=True
    ... )
    {'learnable': True, 'initial_value': 0.5, 'class': 'LearnableK'}
    """
    if serialize:
        if isinstance(param, BaseLearnable):
            return {
                "learnable": param.trainable,
                "initial_value": param.initial_value,
                "class": param.__class__.__name__
            }
        return {
            "learnable": False,
            "initial_value": float(param)
        }

    if isinstance(param, BaseLearnable):
        return param.get_value()

    if isinstance(param, (float, int)):
        if _BACKEND == "tensorflow":
            return tf.constant(
                float(param), dtype=tf.float32
            )
        return float(param)

    raise TypeError(
        "Parameter must be a float, int, or BaseLearnableParam; "
        f"got {type(param).__name__}"
    )
