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
from typing import Any, Union, Optional, Dict, Type
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
        Tensor   = _DummyTF
        Variable = _DummyTF
    # Fallback types for type hinting
    Tensor = Any
    Variable = Any


# Keras serialisable base-class
if _BACKEND =='tensorflow': 
    from tensorflow.keras.saving import register_keras_serializable
else:         # TF missing → no serialisation
    def register_keras_serializable(*_a, **_kw):            # type: ignore
        def decorator(cls):                                 # pragma: no cover
            return cls
        return decorator
    

__all__ = [
    "LearnableC", "FixedC", "DisabledC", 
    "LearnableK", "LearnableSs", "LearnableQ",
    # --- New Parameters for Revised Manuscript ---
    "LearnableMV", "LearnableKappa", "FixedGammaW", "FixedHRef"
]


@register_keras_serializable("fusionlab.params", name="_BaseC")
class _BaseC(ABC):
    r"""
    Parent class for :math:`C` descriptors.

    Each subclass provides :pyattr:`value`
    (``float`` in NumPy mode, ``tf.Variable`` in TF mode)
    and declares whether it is *trainable*.

    The class supports Keras JSON round-trip via
    :py:meth:`get_config` / :py:meth:`from_config`.
    """

    trainable: bool = False      #: overridden by concrete classes

    
    def __init__(self, **kwargs: Any):
        self.value = self._make_value(**kwargs)

    # Keras (de)serialisation 
    def get_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = dict(self._export_kw)          # type: ignore
        cfg["class_name"] = self.__class__.__name__
        return cfg

    @classmethod
    def from_config(cls: Type["_BaseC"], cfg: Dict[str, Any]) -> "_BaseC":
        cfg = dict(cfg)
        cfg.pop("class_name", None)
        return cls(**cfg)

    #  utilities -
    def __repr__(self) -> str:                               # noqa: D401
        nm = self.__class__.__name__
        return f"<{nm} trainable={self.trainable}, value={self.value!r}>"

    # - Implemented by subclasses -
    @abstractmethod
    def _make_value(self, **kwargs: Any) -> Any:             # noqa: D401
        ...

@register_keras_serializable("fusionlab.params", name="LearnableC")
class LearnableC(_BaseC):
    r"""

    Indicates that the PINN’s physical coefficient :math:`C` should be
    learned (trainable).  We actually learn :math:`\log(C)` to ensure
    :math:`C > 0`.  The user supplies an `initial_value`, and the model
    initializes:

    Trainable :math:`C`.

    In TF mode we keep :math:`\log C` as a
    :class:`tf.Variable`, ensuring :math:`C>0`.

    In NumPy mode the coefficient *cannot be trained*,
    so it degrades gracefully to a fixed float.
    
    .. math::
       \log C \;=\; \log(\text{initial\_value}).

    Parameters
    ----------
    initial_value : float
        Strictly positive initial :math:`C`.
    
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
    def __init__(self, initial_value: float = 0.01, **kwargs ): 
        super().__init__(
            initial_value=initial_value, **kwargs
        )

    def _make_value(self,  initial_value: float = 0.01) -> Any:
        if not isinstance(initial_value, (float, int)):
            raise TypeError(
                f"LearnableC.initial_value must be a float, got "
                f"{type(initial_value).__name__}"
            )
        if initial_value <= 0:
            raise ValueError(
                "LearnableC.initial_value must be strictly positive."
            )
        self.initial_value = float(initial_value) 
        self._export_kw = {"initial_value": self.initial_value}  # type: ignore
        
        if _BACKEND == "tensorflow":
            self.trainable = True
            log_c0 = tf.math.log(tf.constant(float(initial_value), tf.float32))
            return tf.Variable(log_c0, dtype=tf.float32,
                               name="log_pinn_coefficient_C")
        # NumPy branch --> behave as a *fixed* coefficient
        self.trainable = False
        return float(initial_value)


@register_keras_serializable("fusionlab.params", name="FixedC")
class FixedC(_BaseC):
    r"""
    Non-trainable, constant :math:`C`.
    
    Indicates that the PINN's physical coefficient :math:`C` should be
    held fixed (non-trainable) at a specified `value`.

    .. math::
       C = \text{value}, \qquad \text{non-trainable}.

    Parameters
    ----------
    value : float
        Constant :math:`C \ge 0`.
        
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
    def __init__(self, value: float, **kwargs):
        super().__init__(value = value, **kwargs)
     
    def _make_value(self,  value: float) -> float:
        if not isinstance(value, (float, int)):
            raise TypeError(
                f"FixedC.value must be a float, got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(
                "LearnableC.initial_value must be strictly positive."
            )
        self._value = float(value) 
        self._export_kw = {"value": self._value}             # type: ignore
        return float(value)


@register_keras_serializable("fusionlab.params", name="DisabledC")
class DisabledC(_BaseC):
    r"""
    Disable physics – :math:`C` is ignored.
    
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
        super().__init__()
    
    def _make_value(self) -> float:                          # noqa: D401
        self._export_kw = {}                                 # type: ignore
        # return 1.0  # No need
        
@register_keras_serializable("fusionlab.params", name ="BaseLearnable")
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
        **kws # for future extension
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
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serialisable dict for tf.keras.

        Notes
        -----
        Keras looks for this method during ``model.save()``
        and ``keras.saving.serialization_lib.serialize_keras_object``.
        """
        return {
            "initial_value": self.initial_value,
            "name":          self.name,
            "log_transform": self.log_transform,
            "trainable":     self.trainable,
            # we also store the concrete subclass path for clarity
            "__class_name__": self.__class__.__name__,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseLearnable":
        """
        Re-instantiate from :py:meth:`get_config`.

        Keras passes *config* exactly as returned above.
        """
        # Guard against stray keys Keras might inject
        kwargs = {
            k: v for k, v in config.items()
            if k in {"initial_value", "name",
                     "log_transform", "trainable"}
        }
        return cls(**kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(initial_value="
            f"{self.initial_value}, trainable={self.trainable}, "
            f"name={self.name})"
        )

@register_keras_serializable("fusionlab.params", name ="LearnableK")
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
        log_transform: bool=True, 
        name: Optional[str] =None,
        trainable: bool=True, 
        **kws
        
    ):
        super().__init__(
            initial_value=initial_value,
            log_transform=log_transform, 
            name= name or "learnable_K",
            trainable= trainable, 
            **kws
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

@register_keras_serializable("fusionlab.params", name ="LearnableSs")
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
        log_transform: bool=True, 
        name: Optional[str] =None,
        trainable: bool=True, 
        **kws
    ):
        super().__init__(
            initial_value=initial_value,
            name= name or "learnable_Ss",
            log_transform=log_transform, 
            trainable= trainable, 
            **kws
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

@register_keras_serializable("fusionlab.params", name ="LearnableQ")
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
        # log_transform: bool=False, # Q should not be log-transformed
        name: Optional[str] =None,
        trainable: bool=True, 
        **kws
    ):
        super().__init__(
            initial_value=initial_value,
            name= name or "learnable_Q",
            log_transform=False, # Explicitly set to False
            trainable= trainable, 
            **kws
            
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
        if _BACKEND == "tensorflow":
            return self._variable # No exp()
        return float(self._variable) # No exp()


@register_keras_serializable("fusionlab.params", name="LearnableMV")
class LearnableMV(BaseLearnable):
    """
    Learnable scalar coefficient of volume compressibility (m_v).
    
    Used in the revised consolidation model:
    :math:`s_{eq}(h) = m_v \gamma_w \Delta h H`
    
    Ensures positivity via log-space transform.

    Parameters
    ----------
    initial_value : float, default=1e-7
        Initial value for :math:`m_v` [Pa^-1]. Must be positive.
    name : str, optional
        Variable name.
    trainable : bool, default=True
        Whether the parameter is trainable.
    """
    def __init__(
        self,
        initial_value: float = 1e-7,
        name: Optional[str] = None,
        trainable: bool = True,
        log_transform: bool=True, # m_v must be positive
        **kws
    ):
        super().__init__(
            initial_value=initial_value,
            name=name or "learnable_mv",
            log_transform=log_transform, 
            trainable=trainable,
            **kws
        )

    def get_value(self) -> Union[Tensor, float]:
        """
        Return :math:`m_v = \exp(\log(m_v))`
        """
        if _BACKEND == "tensorflow":
            return tf.exp(self._variable)
        return float(np.exp(self._variable))

@register_keras_serializable("fusionlab.params", name="LearnableKappa")
class LearnableKappa(BaseLearnable):
    """
    Learnable scalar consistency prior parameter (kappa_bar).
    
    Used in the prior loss term:
    :math:`L_{prior} = ||\log \tau - \log((\bar{\kappa} H^2) / ((\pi^2 K) / S_s))||^2`
    
    Ensures positivity via log-space transform.

    Parameters
    ----------
    initial_value : float, default=1.0
        Initial value for :math:`\bar{\kappa}`. Must be positive.
    name : str, optional
        Variable name.
    trainable : bool, default=True
        Whether the parameter is trainable.
    """
    def __init__(
        self,
        initial_value: float = 1.0,
        name: Optional[str] = None,
        log_transform: bool=True, # kappa_bar must be positive
        trainable: bool = True,
        **kws
    ):
        super().__init__(
            initial_value=initial_value,
            name=name or "learnable_kappa",
            log_transform=log_transform, 
            trainable=trainable,
            **kws
        )

    def get_value(self) -> Union[Tensor, float]:
        """
        Return :math:`\bar{\kappa} = \exp(\log(\bar{\kappa}))`
        """
        if _BACKEND == "tensorflow":
            return tf.exp(self._variable)
        return float(np.exp(self._variable))


@register_keras_serializable("fusionlab.params", name="BaseFixed")
class BaseFixed(ABC):
    """
    Abstract base for fixed physical parameters.

    Parameters
    ----------
    value : float
        Fixed numeric value for the parameter.
    name : str
        Unique identifier for the variable.
    log_transform : bool, optional
        If True, store in log-space for positivity constraint and 
        apply exp() when retrieving value, by default False.
    non_negative : bool, optional
        If True, ensures value cannot be negative, by default True.
        Only enforced when log_transform=False.

    Attributes
    ----------
    value : float
        The fixed parameter value.
    name : str
        Variable name in the computation graph.
    log_transform : bool
        Whether to apply log transform for positivity.
    non_negative : bool
        Whether negative values are allowed.
    trainable : bool
        Always False for fixed parameters.

    Examples
    --------
    >>> param = FixedGammaW(value=9810.0)
    >>> value = param.get_value()
    """
    def __init__(
        self,
        value: float,
        name: str,
        log_transform: bool = False,
        non_negative: bool = True,
        **kws  # for future extension
    ):
        if not isinstance(value, (float, int)):
            raise TypeError(
                f"Value for {self.__class__.__name__} "
                f"must be a float, got {type(value).__name__}"
            )
        
        # Validate constraints
        if log_transform and value <= 0:
            raise ValueError(
                f"{self.__class__.__name__} value must be "
                "strictly positive for log transform."
            )
        if non_negative and value < 0 and not log_transform:
            raise ValueError(
                f"{self.__class__.__name__} value must be "
                "non-negative when non_negative=True."
            )
            
        self.value = float(value)
        self.name = name
        self.log_transform = log_transform
        self.non_negative = non_negative
        self.trainable = False  # Fixed parameters are never trainable
        self._variable = self._create_variable()

    def _create_variable(self) -> Union[Variable, Tensor, float]:
        """
        Internal: create tf.Variable or fallback value for fixed parameter.

        Returns
        -------
        Union[Variable, Tensor, float]
            Configured fixed variable or numeric.
        """
        if _BACKEND == "tensorflow":
            val = self.value
            if self.log_transform:
                val = tf.math.log(val)
            return tf.Variable(
                initial_value=tf.cast(val, dtype=tf.float32),
                trainable=False,  # Explicitly non-trainable
                name=self.name
            )
        # NumPy fallback
        return (
            np.log(self.value) 
            if self.log_transform 
            else self.value
        )

    def get_value(self) -> Union[Tensor, float]:
        """
        Retrieve the fixed parameter value.

        Returns
        -------
        Union[Tensor, float]
            The parameter value, with exp() applied if log_transform=True.
        """
        if _BACKEND == "tensorflow":
            if self.log_transform:
                return tf.exp(self._variable)
            return self._variable
        # NumPy fallback
        if self.log_transform:
            return float(np.exp(self._variable))
        return float(self._variable)

    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serialisable dict for tf.keras serialization.
        """
        return {
            "value": self.value,
            "name": self.name,
            "log_transform": self.log_transform,
            "non_negative": self.non_negative,
            "__class_name__": self.__class__.__name__,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseFixed":
        """
        Re-instantiate from configuration dict.
        """
        kwargs = {
            k: v for k, v in config.items()
            if k in {"value", "name", "log_transform", "non_negative"}
        }
        return cls(**kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(value={self.value}, "
            f"name={self.name}, log_transform={self.log_transform}, "
            f"non_negative={self.non_negative})"
        )


@register_keras_serializable("fusionlab.params", name="FixedGammaW")
class FixedGammaW(BaseFixed):
    """
    Fixed scalar parameter for the unit weight of water (gamma_w).
    
    Used in :math:`s_{eq}(h) = m_v \gamma_w \Delta h H`. This is a
    physical constant and should not be trainable.

    Parameters
    ----------
    value : float, default=9810.0
        Value for :math:`\gamma_w` [N m^-3]. Must be positive.
    name : str, optional
        Variable name.
    non_negative : bool, default=True
        Ensures the value cannot be negative.
    """
    def __init__(
        self,
        value: float = 9810.0,  # Approx. 1000 kg/m^3 * 9.81 m/s^2
        name: Optional[str] = None,
        non_negative: bool = True,
        **kws
    ):
        # gamma_w must be positive, so enforce log_transform for stability
        super().__init__(
            value=value,
            name=name or "fixed_gamma_w",
            log_transform=True,  # gamma_w must always be positive
            non_negative=non_negative,
            **kws
        )


@register_keras_serializable("fusionlab.params", name="FixedHRef")
class FixedHRef(BaseFixed):
    """
    Fixed scalar parameter for the reference head (h_ref).
    
    Used to calculate drawdown: :math:`\Delta h = h_{ref} - h`.
    This is typically a user-defined hyperparameter and is not trainable.

    Parameters
    ----------
    value : float, default=0.0
        Value for :math:`h_{ref}` [m]. Can be negative, zero, or positive.
    name : str, optional
        Variable name.
    non_negative : bool, default=False
        Allow negative values since h_ref can be negative in some contexts.
    """
    def __init__(
        self,
        value: float = 0.0,
        name: Optional[str] = None,
        non_negative: bool = False,  # h_ref can be negative
        **kws
    ):
        super().__init__(
            value=value,
            name=name or "fixed_h_ref", 
            log_transform=False,  # h_ref can be negative, no log transform
            non_negative=non_negative,
            **kws
        )
        

@register_keras_serializable("fusionlab.params", name="resolve_physical_param")
def resolve_physical_param(
    param: Any,
    name: Optional[str] = None,
    *,
    serialize: bool = False,
    status: Optional[str] = None,
    param_type: Optional[str] = None,
    log_transform: Optional[bool] = None,
    non_negative: Optional[bool] = None,
    trainable: Optional[bool] = None,
    **additional_kwargs
) -> Union[Tensor, float, Dict, BaseLearnable, BaseFixed]:
    r"""
    Normalize a physical-parameter descriptor with enhanced flexibility.

    The helper converts *param* into:
        
    - A concrete value (float/tf.Tensor) for runtime use
    - A parameter wrapper (BaseLearnable/BaseFixed) when appropriate
    - A JSON-serializable dict when ``serialize=True``

    Parameters
    ----------
    param : float | int | BaseLearnable | BaseFixed | str | Dict
        Raw descriptor. Can be:
            
        - Plain number: treated as fixed or learnable based on status
        - Wrapped parameter (BaseLearnable/BaseFixed): forwarded as-is
        - String: "learnable" or "fixed" to create wrapper with defaults
        - Dict: configuration for parameter creation
        
    name : str, optional
        Parameter identifier used for:
            
        - Variable naming in TensorFlow backend
        - Type inference when creating wrappers
        
    serialize : bool, default False
        Return configuration dict instead of concrete value.
    status : {'learnable', 'fixed', 'auto', None}, optional
        Global override:
            
        - 'learnable': force creation of learnable wrapper
        - 'fixed': force creation of fixed wrapper  
        - 'auto': infer from param type
        - None: use param's inherent behavior
        
    param_type : str, optional
        Explicit parameter type. Overrides name-based inference.
        Options: 'K', 'Ss', 'Q', 'MV', 'Kappa', 'GammaW', 'HRef'
    log_transform : bool, optional
        Force log-space transformation for positivity.
    non_negative : bool, optional
        Force non-negativity constraint.
    trainable : bool, optional
        Override trainable flag (only for learnable params).
    **additional_kwargs
        Additional parameters passed to wrapper constructors.

    Returns
    -------
    Tensor | float | Dict | BaseLearnable | BaseFixed
        Concrete value, wrapper instance, or serialized configuration.

    Raises
    ------
    TypeError
        If param is of unsupported type.
    ValueError
        If parameter type cannot be inferred or constraints are violated.

    Examples
    --------
    >>> from fusionlab.params import resolve_physical_param
    >>> # Basic usage with type inference from name
    >>> resolve_physical_param(1e-4, name="K", status="learnable")
    LearnableK(initial_value=0.0001, trainable=True)
    
    >>> # Explicit parameter type
    >>> resolve_physical_param(0.5, param_type="MV", status="learnable")
    LearnableMV(initial_value=0.5, trainable=True)
    
    >>> # Fixed parameter with custom constraints
    >>> resolve_physical_param(9810.0, param_type="GammaW", non_negative=True)
    FixedGammaW(value=9810.0, non_negative=True)
    
    >>> # From configuration dict
    >>> config = {"class": "LearnableK", "initial_value": 0.5, "trainable": True}
    >>> resolve_physical_param(config)
    LearnableK(initial_value=0.5, trainable=True)
    
    >>> # Serialization
    >>> k = LearnableK(0.5)
    >>> resolve_physical_param(k, serialize=True)
    {'class': 'LearnableK', 'initial_value': 0.5, ...}
    """

    # 1. Serialization Branch
    if serialize:
        if isinstance(param, (BaseLearnable, BaseFixed)):
            config = param.get_config()
            config["class"] = param.__class__.__name__
            return config
        elif isinstance(param, (float, int)):
            return {
                "class": "float",
                "value": float(param),
                "learnable": False
            }
        elif isinstance(param, dict) and "class" in param:
            return param  # Already serialized
        else:
            raise TypeError(
                f"Cannot serialize parameter of type {type(param).__name__}"
            )

    # 2. Configuration Dict Processing
    if isinstance(param, dict):
        if "class" not in param:
            raise ValueError("Configuration dict must contain 'class' key")
        
        class_name = param["class"]
        config = dict(param)
        config.pop("class", None)
        
        # Map class names to constructors
        wrapper_classes = {
            # Learnable parameters
            "LearnableK": LearnableK, "LearnableSs": LearnableSs, 
            "LearnableQ": LearnableQ, "LearnableMV": LearnableMV,
            "LearnableKappa": LearnableKappa,
            # Fixed parameters  
            "FixedGammaW": FixedGammaW, "FixedHRef": FixedHRef,
            # Legacy parameters
            "LearnableC": LearnableC, "FixedC": FixedC, "DisabledC": DisabledC
        }
        
        if class_name not in wrapper_classes:
            # Handle plain float values
            if class_name == "float":
                return float(config.get("value", 0.0))
            raise ValueError(f"Unknown parameter class: {class_name}")
        
        return wrapper_classes[class_name](**config)

    # 3. String Parameter Processing
    if isinstance(param, str):
        if param.lower() in ("learnable", "fixed"):
            # Use string as status override
            status = param.lower()
            param = 1.0  # Default value for wrapper creation
        else:
            try:
                # Try to parse as numeric string
                param = float(param)
            except ValueError:
                raise ValueError(
                    f"String parameter must be numeric or 'learnable'/'fixed', "
                    f"got '{param}'"
                )

    # 4. Type Inference and Wrapper Mapping
    # Determine parameter type
    resolved_param_type = param_type or _infer_param_type_from_name(name)
    
    # Map parameter types to wrapper classes
    learnable_wrappers = {
        "K": LearnableK, "Ss": LearnableSs, "Q": LearnableQ,
        "MV": LearnableMV, "Kappa": LearnableKappa,
        "C": LearnableC  # Legacy support
    }
    
    fixed_wrappers = {
        "GammaW": FixedGammaW, "HRef": FixedHRef,
        "C": FixedC  # Legacy support
    }

    # 5. Status-Based Processing
    resolved_status = status or "auto"
    
    # Handle already wrapped parameters
    if isinstance(param, (BaseLearnable, BaseFixed)):
        if resolved_status == "auto":
            return param
        elif resolved_status == "learnable" and isinstance(param, BaseFixed):
            # Convert fixed to learnable if requested
            return _convert_fixed_to_learnable(
                param, resolved_param_type, name, **additional_kwargs)
        elif resolved_status == "fixed" and isinstance(param, BaseLearnable):
            # Convert learnable to fixed if requested
            return _convert_learnable_to_fixed(
                param, resolved_param_type, name, **additional_kwargs)
        else:
            return param

    # 6. Numeric Parameter Processing
    if isinstance(param, (float, int)):
        numeric_value = float(param)
        
        # Apply status resolution
        if resolved_status == "learnable":
            return _create_learnable_wrapper(
                numeric_value, resolved_param_type, name, 
                learnable_wrappers, log_transform, non_negative, 
                trainable, **additional_kwargs
            )
        elif resolved_status == "fixed":
            return _create_fixed_wrapper(
                numeric_value, resolved_param_type, name,
                fixed_wrappers, log_transform, non_negative,
                **additional_kwargs
            )
        else:  # auto or None
            # Return as concrete value
            if _BACKEND == "tensorflow":
                return tf.constant(numeric_value, dtype=tf.float32)
            return numeric_value

    # 7. Fallback for Unhandled Types
    raise TypeError(
        f"Parameter must be float, int, BaseLearnable, BaseFixed, dict, or str; "
        f"got {type(param).__name__}"
    )


def _infer_param_type_from_name(name: Optional[str]) -> str:
    """Infer parameter type from name using flexible matching."""
    if not name:
        return "Unknown"
    
    name_upper = name.upper()
    
    # Flexible type matching
    type_patterns = {
        "K": ["K", "CONDUCTIVITY", "HYDRAULIC_CONDUCTIVITY"],
        "Ss": ["SS", "SPECIFIC_STORAGE", "STORAGE"],
        "Q": ["Q", "SOURCE", "SINK", "SOURCE_SINK"],
        "MV": ["MV", "M_V", "COMPRESSIBILITY", "VOLUME_COMPRESSIBILITY"],
        "Kappa": ["KAPPA", "CONSISTENCY", "PRIOR"],
        "GammaW": ["GAMMA_W", "GAMMAW", "UNIT_WEIGHT", "WATER_WEIGHT"],
        "HRef": ["H_REF", "HREF", "REFERENCE_HEAD", "REF_HEAD"],
        "C": ["C", "COEFFICIENT", "PHYSICS_COEFF"]  # Legacy
    }
    
    for param_type, patterns in type_patterns.items():
        if any(pattern in name_upper for pattern in patterns):
            return param_type
    
    return "Unknown"


def _create_learnable_wrapper(
    value: float,
    param_type: str,
    name: Optional[str],
    wrapper_map: Dict[str, Type[BaseLearnable]],
    log_transform: Optional[bool],
    non_negative: Optional[bool], 
    trainable: Optional[bool],
    **kwargs
) -> BaseLearnable:
    """Create a learnable parameter wrapper."""
    if param_type not in wrapper_map:
        raise ValueError(
            f"Cannot create learnable wrapper for parameter type '{param_type}'. "
            f"Available types: {list(wrapper_map.keys())}"
        )
    
    wrapper_class = wrapper_map[param_type]
    
    # Set default parameters based on type
    default_params = {
        "K": {"initial_value": value, "log_transform": True, "trainable": True},
        "Ss": {"initial_value": value, "log_transform": True, "trainable": True},
        "Q": {"initial_value": value, "log_transform": False, "trainable": True},
        "MV": {"initial_value": value, "log_transform": True, "trainable": True},
        "Kappa": {"initial_value": value, "log_transform": True, "trainable": True},
        "C": {"initial_value": value}  # Legacy
    }
    
    params = default_params.get(param_type, {"initial_value": value})
    
    # Apply overrides
    if log_transform is not None:
        params["log_transform"] = log_transform
    if trainable is not None:
        params["trainable"] = trainable
    if name:
        params["name"] = name
    
    params.update(kwargs)
    
    return wrapper_class(**params)


def _create_fixed_wrapper(
    value: float,
    param_type: str, 
    name: Optional[str],
    wrapper_map: Dict[str, Type[BaseFixed]],
    log_transform: Optional[bool],
    non_negative: Optional[bool],
    **kwargs
) -> BaseFixed:
    """Create a fixed parameter wrapper."""
    if param_type not in wrapper_map:
        # For unsupported fixed types, return as concrete value
        if _BACKEND == "tensorflow":
            return tf.constant(value, dtype=tf.float32)
        return value
    
    wrapper_class = wrapper_map[param_type]
    
    # Set default parameters based on type
    default_params = {
        "GammaW": {"value": value, "log_transform": True, "non_negative": True},
        "HRef": {"value": value, "log_transform": False, "non_negative": False},
        "C": {"value": value}  # Legacy
    }
    
    params = default_params.get(param_type, {"value": value})
    
    # Apply overrides
    if log_transform is not None:
        params["log_transform"] = log_transform
    if non_negative is not None:
        params["non_negative"] = non_negative
    if name:
        params["name"] = name
    
    params.update(kwargs)
    
    return wrapper_class(**params)


def _convert_fixed_to_learnable(
    fixed_param: BaseFixed,
    param_type: str,
    name: Optional[str],
    **kwargs
) -> BaseLearnable:
    """Convert a fixed parameter to learnable."""
    learnable_wrappers = {
        "K": LearnableK, "Ss": LearnableSs, "Q": LearnableQ,
        "MV": LearnableMV, "Kappa": LearnableKappa
    }
    
    if param_type not in learnable_wrappers:
        raise ValueError(
            "Cannot convert fixed parameter to"
            f" learnable for type '{param_type}'")
    
    wrapper_class = learnable_wrappers[param_type]
    
    params = {
        "initial_value": fixed_param.value,
        "name": name or fixed_param.name,
        "trainable": True
    }
    params.update(kwargs)
    
    return wrapper_class(**params)


def _convert_learnable_to_fixed(
    learnable_param: BaseLearnable, 
    param_type: str,
    name: Optional[str],
    **kwargs
) -> BaseFixed:
    """Convert a learnable parameter to fixed."""
    fixed_wrappers = {
        "GammaW": FixedGammaW, "HRef": FixedHRef
    }
    
    if param_type not in fixed_wrappers:
        # For unsupported conversions, return as concrete value
        return learnable_param.get_value()
    
    wrapper_class = fixed_wrappers[param_type]
    
    params = {
        "value": learnable_param.initial_value,
        "name": name or learnable_param.name
    }
    params.update(kwargs)
    
    return wrapper_class(**params)

# @register_keras_serializable("fusionlab.params", name="FixedGammaW")
# class FixedGammaW(BaseLearnable):
#     """
#     Fixed scalar parameter for the unit weight of water (gamma_w).
    
#     Used in :math:`s_{eq}(h) = m_v \gamma_w \Delta h H`. This is a
#     physical constant and should not be trainable.

#     Parameters
#     ----------
#     value : float, default=9810.0
#         Value for :math:`\gamma_w` [N m^-3].
#     name : str, optional
#         Variable name.
#     """
#     def __init__(
#         self,
#         value: float = 9810.0, # Approx. 1000 kg/m^3 * 9.81 m/s^2
#         name: Optional[str] = None,
#         log_transform: bool =False,
#         trainable=False,# False: This is a fixed constant
#         **kws
#     ):
#         if 'initial_value' in kws: 
#             kws.pop ('initial_value')
            
#         super().__init__(
#             initial_value=value,
#             name=name or "fixed_gamma_w",
#             log_transform=log_transform,
#             trainable=trainable, 
#             **kws
#         )

#     def get_value(self) -> Union[Tensor, float]:
#         """
#         Return fixed :math:`\gamma_w` value.
#         """
#         if _BACKEND == "tensorflow":
#             return self._variable
#         return float(self._variable)

# @register_keras_serializable("fusionlab.params", name="FixedHRef")
# class FixedHRef(BaseLearnable):
#     """
#     Fixed scalar parameter for the reference head (h_ref).
    
#     Used to calculate drawdown: :math:`\Delta h = h_{ref} - h`.
#     This is typically a user-defined hyperparameter and is not trainable.

#     Parameters
#     ----------
#     value : float, default=0.0
#         Value for :math:`h_{ref}` [m].
#     name : str, optional
#         Variable name.
#     """
#     def __init__(
#         self,
#         value: float = 0.0,
#         name: Optional[str] = None,
#         log_transform: bool =False,
#         trainable:bool =False, # This is a fixed hyperparameter
#         **kws
#     ):
#         if 'initial_value' in kws: 
#             kws.pop ('initial_value')
            
#         super().__init__(
#             initial_value=value,
#             name=name or "fixed_h_ref",
#             log_transform=log_transform,
#             trainable=trainable, # This is a fixed hyperparameter
#             **kws
#         )

#     def get_value(self) -> Union[Tensor, float]:
#         """
#         Return fixed :math:`h_{ref}` value.
#         """
#         if _BACKEND == "tensorflow":
#             return self._variable
#         return float(self._variable)