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
