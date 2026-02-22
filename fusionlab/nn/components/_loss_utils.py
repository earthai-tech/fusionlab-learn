# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Loss utilities for different use-cases, including quantile loss, 
mean squared error, and more. These utilities can be used with 
different models and can be easily extended for future requirements.
"""

from __future__ import annotations
from typing import List

from ...api.property import NNLearner
from ._config import (
    Loss, 
    Tensor,
    tf_abs,
    tf_square,
    tf_reduce_mean,
    tf_reduce_sum,
    tf_constant,
    tf_float32,
    tf_reshape,
    tf_maximum,
    tf_where, 
    register_keras_serializable, 
)

__all__ = [
    "MeanSquaredErrorLoss",
    "QuantileLoss",
    "HuberLoss",
    "WeightedLoss",
    "compute_loss_with_reduction", 
    "compute_quantile_loss"
]

@register_keras_serializable(
    "fusionlab.nn.components", name="MeanSquaredErrorLoss")
class MeanSquaredErrorLoss(Loss, NNLearner):
    """
    Mean Squared Error (MSE) loss function.

    Args:
    - reduction (str): Defines the reduction method for the loss:
      'auto', 'sum', 'mean', or 'none'.
    """

    def __init__(self, reduction: str = "auto", name: str = "MSELoss"):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the Mean Squared Error loss.

        Args:
        - y_true: Ground truth values
        - y_pred: Predicted values

        Returns:
        - Tensor: Computed loss value
        """
        return tf_reduce_mean(tf_square(y_true - y_pred))


@register_keras_serializable(
    "fusionlab.nn.components", name="QuantileLoss")
class QuantileLoss(Loss, NNLearner):
    """
    Adaptive Quantile Loss layer that computes quantile loss for given
    quantiles.

    Args:
    - quantiles (List[float]): List of quantiles to compute the loss.
    """

    def __init__(self, quantiles: List[float], reduction: str = "auto", 
                 name: str = "AdaptiveQuantileLoss"):
        super().__init__(reduction=reduction, name=name)
        self.quantiles = quantiles
        self.q = len(quantiles)
        self._qs = tf_constant(quantiles, dtype=tf_float32)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the Adaptive Quantile Loss.

        Args:
        - y_true: Ground truth values
        - y_pred: Predicted values

        Returns:
        - Tensor: Computed loss value
        """
        y_true_exp = tf_reshape(y_true, [-1, 1, 1, 1])
        err = y_true_exp - y_pred

        # Pinball loss calculation: max(q*e, (q-1)*e)
        pinball_loss = tf_maximum(self._qs * err, (self._qs - 1.0) * err)
        return tf_reduce_mean((2.0 / float(self.q)) * pinball_loss, axis=2)


@register_keras_serializable(
    "fusionlab.nn.components", name="HuberLoss")
class HuberLoss(Loss, NNLearner):
    """
    Huber loss function which is less sensitive to outliers in data.

    Args:
    - delta (float): Threshold for switching between MSE and MAE.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "auto", 
                 name: str = "HuberLoss"):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the Huber loss.

        Args:
        - y_true: Ground truth values
        - y_pred: Predicted values

        Returns:
        - Tensor: Computed loss value
        """
        error = y_true - y_pred
        abs_error = tf_abs(error)

        # Huber loss formula
        condition = abs_error <= self.delta
        squared_loss = tf_square(error) / 2
        linear_loss = self.delta * (abs_error - (self.delta / 2))

        return tf_reduce_mean(
            tf_where(condition, squared_loss, linear_loss))


@register_keras_serializable(
    "fusionlab.nn.components", name="WeightedLoss")
class WeightedLoss(Loss, NNLearner):
    """
    Weighted loss function to apply different weights for each sample.

    Args:
    - weight (float or Tensor): Weighting factor for the loss calculation.
    """

    def __init__(self, weight: float = 1.0, reduction: str = "auto",
                 name: str = "WeightedLoss"):
        super().__init__(reduction=reduction, name=name)
        self.weight = weight

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the weighted loss.

        Args:
        - y_true: Ground truth values
        - y_pred: Predicted values

        Returns:
        - Tensor: Computed weighted loss value
        """
        return self.weight * tf_reduce_mean(tf_square(y_true - y_pred))


### Utility Functions for Losses ###

def compute_loss_with_reduction(
        loss_fn, y_true: Tensor, y_pred: Tensor, reduction: str) -> Tensor:
    """
    Compute the loss using the specified reduction method.
    
    Args:
    - loss_fn: Loss function to compute the loss
    - y_true: Ground truth values
    - y_pred: Predicted values
    - reduction: String indicating reduction method ('sum', 'mean', etc.)
    
    Returns:
    - Tensor: Loss value
    """
    loss_value = loss_fn(y_true, y_pred)
    
    if reduction == "mean":
        return tf_reduce_mean(loss_value)
    elif reduction == "sum":
        return tf_reduce_sum(loss_value)
    elif reduction == "none":
        return loss_value
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")
    

def compute_quantile_loss(
        y_true: Tensor, y_pred: Tensor, quantiles: List[float]) -> Tensor:
    """
    Compute the quantile loss for a given set of quantiles.

    Args:
    - y_true: Ground truth values
    - y_pred: Predicted values
    - quantiles: List of quantiles to compute the loss

    Returns:
    - Tensor: Quantile loss value
    """
    qs = tf_constant(quantiles, dtype=tf_float32)
    error = y_true - y_pred
    pinball_loss = tf_maximum(qs * error, (qs - 1.0) * error)
    return tf_reduce_mean(pinball_loss, axis=2)
