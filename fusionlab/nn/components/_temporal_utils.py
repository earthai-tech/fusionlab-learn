# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Temporal utilities:
    - aggregate_multiscale
    - aggregate_multiscale_on_3d
    - aggregate_time_window_output
"""

from __future__ import annotations

from typing import List, Union, Optional

from ._config import (
    Tensor,
    tf_concat, 
    tf_reduce_mean, 
    tf_reduce_sum, 
    tf_shape, tf_reshape,
    tf_pad, tf_maximum,
    register_keras_serializable,

)

__all__ = [
    "aggregate_multiscale",
    "aggregate_multiscale_on_3d",
    "aggregate_time_window_output",
]

@register_keras_serializable(
    'fusionlab.nn.components', 
    name='aggregate_multiscale'
)
def aggregate_multiscale(lstm_output, mode="auto"):
    r"""Aggregate multi-scale LSTM outputs using 
    specified temporal fusion strategy.

    This function implements multiple strategies for combining outputs from
    multi-scale LSTMs operating at different temporal resolutions. Supports
    six aggregation modes: ``average``, ``sum``, ``flatten``, ``concat``,
    ``last`` (default fallback), and ``auto``[1]_.
    Designed for compatibility with ``MultiScaleLSTM`` layer outputs.
    
    See more in :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    lstm_output : list of tf.Tensor or tf.Tensor
        Input features from multi-scale processing:
        - List of 3D tensors [(B, T', U), ...] when ``mode`` != 'auto'
        - Single 2D tensor (B, U*S) when ``mode=None``
        where:
          B = Batch size
          T' = Variable time dimension (scale-dependent)
          U = LSTM units per scale
          S = Number of scales (len(scales))
    mode : {'auto', 'sum', 'average', 'flatten', 'concat', 'last'}, optional
        Aggregation strategy:
        - ``auto`` : (Default) Concatenate last timesteps from each scale
        - ``sum`` : Temporal summation per scale + feature concatenation
        - ``average`` : Temporal mean per scale + feature concatenation
        - ``flatten`` : Flatten all time-feature dimensions (requires equal T')
        - ``concat`` : Feature concatenation + last global timestep
        - ``last`` : Alias for ``auto`` (backward compatibility)

    Returns
    -------
    tf.Tensor
        Aggregated features with shape:
        - (B, U*S) for modes: ``average``, ``sum``, ``last``
        - (B, T'*U*S) for ``flatten`` mode
        - (B, U*S) for ``concat`` mode (last timestep only)
        - (B, U*S) for ``auto`` mode
        
        In sum: 
        - (B, U*S) for ``auto``/``last``, ``sum``, ``average``, ``concat``
        - (B, T'*U*S) for ``flatten`` mode.

    Notes
    -----
    
    * Mode Comparison Table:

    +------------+---------------------+---------------------+-------------------+
    | Mode       | Temporal Handling   | Requirements        | Typical Use Case  |
    +============+=====================+=====================+===================+
    | ``auto``   | Last step per scale | None                | Default choice    |
    | (last)     |                     |                     | for variable T'   |
    +------------+---------------------+---------------------+-------------------+
    | ``sum``    | Full sequence sum   | None                | Emphasize temporal|
    |            | per scale           |                     | accumulation      |
    +------------+---------------------+---------------------+-------------------+
    | ``average``| Full sequence mean  | None                | Smooth temporal   |
    |            | per scale           |                     | patterns          |
    +------------+---------------------+---------------------+-------------------+
    | ``flatten``| Preserve all time   | Equal T' across     | Fixed-length      |
    |            | steps               | scales              | sequence models   |
    +------------+---------------------+---------------------+-------------------+
    | ``concat`` | Last global step    | Equal T' across     | Specialized       |
    |            | of concatenated     | scales              | architectures     |
    |            | features            |                     | with aligned T'   |
    +------------+---------------------+---------------------+-------------------+

    Mathematical Formulation:

    For S scales with outputs :math:`\{\mathbf{X}_s \in \mathbb{R}^{B \times T'_s 
    \times U}\}_{s=1}^S`:

    .. math::
        \text{auto} &: \bigoplus_{s=1}^S \mathbf{X}_s^{(:, T'_s, :)} 
        \quad \text{(Last step concatenation)}
        
        \text{sum} &: \bigoplus_{s=1}^S \sum_{t=1}^{T'_s} \mathbf{X}_s^{(:, t, :)}
        
        \text{average} &: \bigoplus_{s=1}^S \frac{1}{T'_s} \sum_{t=1}^{T'_s} 
        \mathbf{X}_s^{(:, t, :)}
        
        \text{flatten} &: \text{vec}\left( \bigoplus_{s=1}^S \mathbf{X}_s \right)
        
        \text{concat} &: \left( \bigoplus_{s=1}^S \mathbf{X}_s \right)^{(:, T', :)}

    where :math:`\bigoplus` = feature concatenation, :math:`\text{vec}` = flatten.

    * Critical differences between key modes ``'concat'`` and ``'last'``:

    +------------------+---------------------+-----------------------+
    | Aspect           | ``concat``          | ``last`` (default)    |
    +==================+=====================+=======================+
    | Time alignment   | Requires equal T'   | Handles variable T'   |
    +------------------+---------------------+-----------------------+
    | Feature mixing   | Cross-scale mixing  | Scale-independent     |
    +------------------+---------------------+-----------------------+
    | Scale validity   | Only valid when     | Robust to arbitrary   |
    |                  | scales=[1,1,...]    | scale configurations  |
    +------------------+---------------------+-----------------------+
    
    Examples
    --------
    >>> from fusionlab.nn.components import aggregate_multiscale
    >>> import tensorflow as tf
    
    # Three scales with different time dimensions
    >>> outputs = [
    ...     tf.random.normal((32, 10, 64)),  # Scale 1: T'=10
    ...     tf.random.normal((32, 5, 64)),   # Scale 2: T'=5
    ...     tf.random.normal((32, 2, 64))    # Scale 3: T'=2
    ... ]
    
    # Default auto mode (last timesteps)
    >>> agg_auto = aggregate_multiscale(outputs, mode='auto')
    >>> agg_auto.shape
    (32, 192)  # 64 units * 3 scales

    # Last timestep aggregation (default)
    >>> agg_last = aggregate_multiscale(outputs, mode='last')
    >>> print(agg_last.shape)
    (32, 192)
    
    # Flatten mode (requires manual padding for equal T')
    >>> padded_outputs = [tf.pad(o, [[0,0],[0,3],[0,0]]) for o in outputs[:2]] 
    >>> padded_outputs.append(outputs[2])
    >>> agg_flat = aggregate_multiscale(padded_outputs, mode='flatten')
    >>> agg_flat.shape
    (32, 1280)  # (10+3)*64*3 = 13*192 = 2496? Wait need to check dimensions

    See Also
    --------
    MultiScaleLSTM : Base layer producing multi-scale LSTM outputs
    TemporalFusionTransformer : Advanced temporal fusion architecture
    HierarchicalAttention : Alternative temporal aggregation approach

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). Time-series forecasting with deep
       learning: a survey. Philosophical Transactions of the Royal Society A,
       379(2194), 20200209. https://doi.org/10.1098/rsta.2020.0209
    """
    # "auto", use the last LastStep-First Approach
    if mode is None: 
        # No additional aggregation needed
        lstm_features = lstm_output  # (B, units * len(scales))

    # Apply chosen aggregation to full sequences
    elif mode == "average":
        # Average over time dimension for each scale and then concatenate
        averaged_outputs = [
            tf_reduce_mean(o, axis=1) 
            for o in lstm_output
        ]  # Each is (B, units)
        lstm_features = tf_concat(
            averaged_outputs,
            axis=-1
        )  # (B, units * len(scales))

    elif mode== "flatten":
        # Flatten time and feature dimensions for all scales
        # Assume equal time lengths for all scales
        concatenated = tf_concat(
            lstm_output, 
            axis=-1
        )  # (B, T', units*len(scales))
        shape = tf_shape(concatenated)
        (batch_size,
         time_dim,
         feat_dim) = shape[0], shape[1], shape[2]
        lstm_features = tf_reshape(
            concatenated,
            [batch_size, time_dim * feat_dim]
        )
    elif mode =='sum': 
        # Sum over time dimension for each scale and concatenate
        summed_outputs = [
            tf_reduce_sum(o, axis=1) 
            for o in lstm_output
            ]
        lstm_features = tf_concat(
            summed_outputs, axis=-1)
        
    elif mode=="concat": 
        # Concatenate along the feature dimension for each
        # time step and take the last time step
        concatenated = tf_concat(
            lstm_output, axis=-1)  # (B, T', units * len(scales))
        last_output = concatenated[:, -1, :]  # (B, units * len(scales))
        lstm_features = last_output
        
    else: # "last" or "auto"
        # Default fallback: take the last time step from each scale
        # and concatenate
        last_outputs = [
            o[:, -1, :] 
            for o in lstm_output
        ]  # (B, units)
        lstm_features = tf_concat(
            last_outputs,
            axis=-1
        )  # (B, units * len(scales))
    
    return lstm_features 

def aggregate_multiscale_on_3d(
    lstm_output: Union[Tensor, List[Tensor]],
    mode: str = "auto"
) -> Tensor:
    r"""Aggregate multi-scale LSTM outputs using a specified strategy.

    This function combines outputs from `MultiScaleLSTM`. It is designed
    to either produce a single 3D sequence tensor (for attention
    mechanisms) or a single 2D context vector (by collapsing the
    time dimension).

    Parameters
    ----------
    lstm_output : list of tf.Tensor or tf.Tensor
        The output from `MultiScaleLSTM`.
        - If a list: Expected to be from an LSTM with
          `return_sequences=True`. Each element is a 3D tensor
          `(B, T_scale, U)` where `T_scale` can vary.
        - If a single tensor: Assumed to be from an LSTM with
          `return_sequences=False`, shape `(B, U * num_scales)`.
          In this case, it's returned as is.

    mode : {'auto', 'sum', 'average', 'flatten', 'concat', 'last'}, optional
        Aggregation strategy:
        - **'concat'**: (For 3D output) Pads sequences to the max
          length and concatenates along the feature axis. This is the
          primary mode for creating a rich sequence representation for
          downstream attention layers. Result shape: `(B, T_max, U*S)`.
        - **'last'** or **'auto'**: (For 2D output) Takes the last
          time step from each sequence in the list and concatenates
          them. Result shape: `(B, U*S)`.
        - **'average'**: (For 2D output) Averages each sequence over
          its time dimension and concatenates the results.
        - **'sum'**: (For 2D output) Sums each sequence over its
          time dimension and concatenates the results.
        - **'flatten'**: (For 2D output) Concatenates and flattens all
          dimensions except the batch. Requires sequences to have the
          same length.

    Returns
    -------
    tf.Tensor
        The aggregated feature tensor, either 2D or 3D depending on the mode.
    """
    if not isinstance(lstm_output, list):
        # Input is likely already a 2D tensor, return as is.
        return lstm_output
    
    if not lstm_output:
        raise ValueError("Input `lstm_output` list cannot be empty.")

    # --- New 'concat' behavior to produce a single 3D tensor ---
    if mode == "concat":
        # This mode pads sequences to the same length and concatenates
        # on the feature axis, preserving the time dimension.
        
        # 1. Find the maximum sequence length in the list of tensors.
        max_len = 0
        for tensor in lstm_output:
            if tensor.shape.ndims != 3:
                raise ValueError(
                    "For 'concat' mode, all items in `lstm_output` must be "
                    f"3D tensors, but found shape {tensor.shape}"
                )
            max_len = tf_maximum(max_len, tf_shape(tensor)[1])
            
        # 2. Pad each tensor to the max length.
        padded_tensors = []
        for tensor in lstm_output:
            current_len = tf_shape(tensor)[1]
            # Paddings format: [[dim1_before, dim1_after], [dim2_before, dim2_after], ...]
            paddings = [[0, 0], [0, max_len - current_len], [0, 0]]
            padded_tensors.append(tf_pad(tensor, paddings, "CONSTANT"))
        
        # 3. Concatenate along the feature axis (-1).
        return tf_concat(padded_tensors, axis=-1)

    # --- Existing modes that reduce to a 2D tensor ---
    elif mode == "average":
        averaged_outputs = [
            tf_reduce_mean(o, axis=1) for o in lstm_output
        ]
        return tf_concat(averaged_outputs, axis=-1)

    elif mode == "sum":
        summed_outputs = [
            tf_reduce_sum(o, axis=1) for o in lstm_output
        ]
        return tf_concat(summed_outputs, axis=-1)

    elif mode == "flatten":
        # This mode requires all sequences to have the same length.
        concatenated = tf_concat(lstm_output, axis=-1)
        shape = tf_shape(concatenated)
        batch_size, time_dim, feat_dim = shape[0], shape[1], shape[2]
        return tf_reshape(concatenated, [batch_size, time_dim * feat_dim])
        
    else:  # Default for "last" or "auto"
        # Takes the last time step from each sequence and concatenates.
        last_outputs = [o[:, -1, :] for o in lstm_output]
        return tf_concat(last_outputs, axis=-1)

@register_keras_serializable(
    'fusionlab.nn.components', 
    name='aggregate_time_window_output'
)
def aggregate_time_window_output(
        time_window_output:Tensor,
        mode: Optional[str]=None
    ):
    """
    Aggregates time window output features based on the specified
    aggregation method.

    This function performs the final aggregation on a 3D tensor
    representing temporal features. The aggregation can be done by
    selecting the last time step, computing the average across time,
    or flattening the temporal and feature dimensions into a single
    vector per sample.

    The aggregation methods are defined as follows:

    .. math::
       \text{last: } F = T[:, -1, :]

    .. math::
       \text{average: } F = \frac{1}{T_{dim}} \sum_{i=1}^{T_{dim}}
       T[:, i, :]

    .. math::
       \text{flatten: } F = \text{reshape}(T, (batch\_size,
       time\_dim \times feat\_dim))

    where :math:`T` is the input tensor with shape
    :math:`(batch\_size, time\_dim, feat\_dim)` and :math:`F` is the
    aggregated output.

    Parameters
    ----------
    time_window_output : tf.Tensor
        A 3D tensor of shape :math:`(batch\_size, time\_dim,
        feat\_dim)` representing the output features over time.
    mode : str, optional
        Aggregation method to apply. Supported values are:

        - ``"last"``: Selects the features from the last time step.
        - ``"average"``: Computes the mean of features across
          the time dimension.
        - ``"flatten"``: Flattens the time and feature dimensions
          into a single vector per sample.

        If ``mode`` is `None`, the function falls back to the
        ``flatten`` aggregation method.

    Returns
    -------
    tf.Tensor
        The aggregated features tensor after applying the specified
        aggregation method.

    Raises
    ------
    ValueError
        If an unsupported aggregation method is provided in the
        ``mode`` argument.

    Examples
    --------
    >>> from fusionlab.nn.components import aggregate_time_window_output
    >>> import tensorflow as tf
    >>> # Create a dummy tensor with shape (2, 3, 4)
    >>> dummy = tf.random.uniform((2, 3, 4))
    >>> # Apply average aggregation
    >>> result = aggregate_time_window_output(dummy,
    ...                                      mode="average")

    Notes
    -----
    - The function uses TensorFlow operations to ensure compatibility
      with TensorFlow's computation graph.
    - It is recommended to use this function as part of a larger neural
      network pipeline [1]_.

    See Also
    --------
    tf.reduce_mean
        TensorFlow operation to compute mean along axes.

    References
    ----------
    .. [1] Author Name, "Title of the reference", Journal/Conference,
       Year.

    """
    mode = mode or 'flatten' 
    if mode == "last":
        # Select the features corresponding to the last time step for
        # each sample.
        final_features = time_window_output[:, -1, :]

    elif mode == "average":
        # Compute the mean of the features across the time dimension.
        final_features = tf_reduce_mean(time_window_output, axis=1)

    elif mode == "flatten":
        # Retrieve the dynamic shape of the input tensor.
        shape = tf_shape(time_window_output)
        batch_size, time_dim, feat_dim = (
            shape[0],
            shape[1],
            shape[2]
        )
        # Flatten the time and feature dimensions into a single vector
        # per sample.
        final_features = tf_reshape(
            time_window_output,
            [batch_size, time_dim * feat_dim]
        )

    else:
        # Raise an error if an unsupported aggregation method is provided.
        raise ValueError(
            f"Unsupported mode value: '{mode}'. Supported values are "
            f"'last', 'average', or 'flatten'."
        )

    return final_features