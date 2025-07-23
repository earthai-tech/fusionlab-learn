# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Temporal modules:
    - MultiScaleLSTM
    - aggregate_multiscale
    - aggregate_multiscale_on_3d
    - aggregate_time_window_output
"""

from __future__ import annotations
from typing import List, Union

from ._config import (
    LSTM, Layer, 
    tf_concat, 
    tf_autograph, register_keras_serializable,
    DEP_MSG,
    KERAS_BACKEND,
)

from ..api.property import NNLearner
from ..utils.deps_utils import ensure_pkg
from ..core.checks import validate_nested_param

__all__ = [
    "MultiScaleLSTM",
    "DynamicTimeWindow", 
]


@register_keras_serializable(
    'fusionlab.nn.components', 
    name='MultiScaleLSTM'
)
class MultiScaleLSTM(Layer, NNLearner):
    r"""
    MultiScaleLSTM layer applying multiple LSTMs
    at different sampling scales and concatenating
    their outputs [1]_.

    Each LSTM can either return the full sequence
    or only the last hidden state, controlled by
    `return_sequences`. The user specifies `scales`
    to sub-sample the time dimension. For example,
    a scale of 2 processes every 2nd time step.

    Parameters
    ----------
    lstm_units : int
        Number of units in each LSTM.
    scales : list of int or str or None, optional
        List of scale factors. If `'auto'` or None,
        defaults to `[1]` (no sub-sampling).
    return_sequences : bool, optional
        If True, each LSTM returns the entire
        sequence. Otherwise, it returns only the
        last hidden state. Defaults to False.
    **kwargs
        Additional arguments passed to the parent
        Keras `Layer`.

    Notes
    -----
    - If `return_sequences=False`, the output is
      concatenated along features:
      :math:`(B, \text{units} \times \text{num\_scales})`.
    - If `return_sequences=True`, a list of
      sequence outputs is returned. Each may have
      a different time dimension if scales differ.

    Methods
    -------
    call(`inputs`, training=False)
        Forward pass, applying each LSTM at the
        specified scale.
    get_config()
        Returns the layer's configuration dict.
    from_config(`config`)
        Builds the layer from the config dict.

    Examples
    --------
    >>> from fusionlab.nn.components import MultiScaleLSTM
    >>> import tensorflow as tf
    >>> x = tf.random.normal((32, 20, 16))  # (B, T, D)
    >>> # Instantiating a multi-scale LSTM
    >>> mslstm = MultiScaleLSTM(lstm_units=32,
    ...     scales=[1, 2], return_sequences=False)
    >>> y = mslstm(x)  # shape => (32, 64)
    >>> # because scale=1 and scale=2 each produce 32 units,
    ... # which are concatenated => 64

    See Also
    --------
    DynamicTimeWindow
        For slicing sequences before applying
        multi-scale LSTMs.
    TemporalFusionTransformer
        A complex model that can incorporate
        multi-scale modules.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021).
           "Time-series forecasting with deep
           learning: a survey." *Philosophical
           Transactions of the Royal Society A*,
           379(2194), 20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        lstm_units: int,
        scales: Union[str, List[int], None] = None,
        return_sequences: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if scales is None or scales == 'auto':
            scales = [1]
        # Validate that scales is a list of int
        scales = validate_nested_param(
            scales,
            List[int],
            'scales'
        )

        self.lstm_units = lstm_units
        self.scales = scales
        self.return_sequences = return_sequences

        # Create an LSTM for each scale
        self.lstm_layers = [
            LSTM(
                lstm_units,
                return_sequences=return_sequences
            )
            for _ in scales
        ]

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass that processes the input
        at multiple scales.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            Shape (B, T, D).
        training : bool, optional
            Training mode. Defaults to ``False``.

        Returns
        -------
        tf.Tensor or list of tf.Tensor
            - If `return_sequences=False`, returns
              a single 2D tensor of shape
              (B, lstm_units * len(scales)).
            - If `return_sequences=True`, returns
              a list of 3D tensors, each with shape
              (B, T', lstm_units), where T' depends
              on the scale sub-sampling.
        """
        outputs = []
        for scale, lstm in zip(self.scales, self.lstm_layers):
            scaled_input = inputs[:, ::scale, :]
            lstm_output = lstm(
                scaled_input,
                training=training
            )
            outputs.append(lstm_output)

        # If return_sequences=False:
        #   => (B, units) from each sub-lstm
        #      -> concat => (B, units*len(scales))
        if not self.return_sequences:
            return tf_concat(outputs, axis=-1)
        else:
            # return a list of sequences
            return outputs

    def get_config(self):
        r"""
        Returns a config dictionary containing
        'lstm_units', 'scales', and
        'return_sequences'.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        config = super().get_config().copy()
        config.update({
            'lstm_units': self.lstm_units,
            'scales': self.scales,
            'return_sequences': self.return_sequences
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Builds MultiScaleLSTM from the given
        config dictionary.

        Parameters
        ----------
        ``config`` : dict
            Must include 'lstm_units', 'scales',
            'return_sequences'.

        Returns
        -------
        MultiScaleLSTM
            A new instance of this layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components',
    name="DynamicTimeWindow"
)
class DynamicTimeWindow(Layer, NNLearner):
    r"""
    DynamicTimeWindow layer that slices the last
    `max_window_size` steps from the input sequence.

    This helps in focusing on the most recent time
    steps if the sequence is longer than
    `max_window_size`.

    .. math::
        \mathbf{Z} = \mathbf{X}[:, -W:, :]

    where `W` = `max_window_size`.

    Parameters
    ----------
    max_window_size : int
        Number of time steps to keep from
        the end of the sequence.

    Notes
    -----
    This can be used for models that only need
    the last few time steps instead of the entire
    sequence.

    Methods
    -------
    call(`inputs`, training=False)
        Slice the last `max_window_size` steps.
    get_config()
        Returns configuration dictionary.
    from_config(`config`)
        Recreates the layer from config.

    Examples
    --------
    >>> from fusionlab.nn.components import DynamicTimeWindow
    >>> import tensorflow as tf
    >>> x = tf.random.normal((32, 50, 64))
    >>> # Keep last 10 time steps
    >>> dtw = DynamicTimeWindow(max_window_size=10)
    >>> y = dtw(x)
    >>> y.shape
    TensorShape([32, 10, 64])

    See Also
    --------
    MultiResolutionAttentionFusion
        Another layer that can be used after
        slicing to fuse temporal features.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). 
           "Time-series forecasting with deep
           learning: a survey." 
           *Philosophical Transactions of
           the Royal Society A*, 379(2194),
           20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, max_window_size: int):
        r"""
        Initialize the DynamicTimeWindow layer.

        Parameters
        ----------
        max_window_size : int
            Number of steps to slice from the end
            of the sequence.
        """
        super().__init__()
        self.max_window_size = max_window_size

    def call(self, inputs, training=False):
        r"""
        Forward pass that slices the last
        `max_window_size` steps.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            Tensor of shape :math:`(B, T, D)`.
        training : bool, optional
            Unused. Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            A sliced tensor of shape 
            :math:`(B, W, D)` where W = 
            `max_window_size`.
        """
        return inputs[:, -self.max_window_size:, :]

    def get_config(self):
        r"""
        Returns configuration dictionary.

        Returns
        -------
        dict
            Contains 'max_window_size'.
        """
        config = super().get_config().copy()
        config.update({
            'max_window_size': self.max_window_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Creates a new DynamicTimeWindow layer
        from config.

        Parameters
        ----------
        ``config`` : dict
            Must include 'max_window_size'.

        Returns
        -------
        DynamicTimeWindow
            A new instance of this layer.
        """
        return cls(**config)
