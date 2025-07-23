# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Mask helpers for attention / sequence ops.

- pad_mask_from_lengths: build (B, T) masks from per-seq lengths
- sequence_mask_3d: broadcast a 2D mask to 3D for (B, T, F) tensors
"""

from __future__ import annotations
from typing import Optional

from ._config import (
    Tensor,
    tf_cast, 
    tf_int32, 
    tf_bool, 
    tf_float32,
    tf_range, 
    tf_expand_dims,
    tf_shape, 
    tf_logical_not,
    tf_reduce_max,
)

__all__ = ["pad_mask_from_lengths", "sequence_mask_3d"]


def pad_mask_from_lengths(
    lengths: Tensor,
    max_len: Optional[int] = None,
    *,
    dtype: Tensor = tf_bool,
    invert: bool = False,
) -> Tensor:
    """
    Build a mask of shape (B, T) from integer lengths.

    Parameters
    ----------
    lengths : Tensor[int32/int64], shape (B,)
        Valid lengths per sequence.
    max_len : int, optional
        If None, uses max(lengths).
    dtype : tf.bool or tf.float32
        Output dtype.
    invert : bool
        If True, returns mask==False positions as True.

    Returns
    -------
    Tensor
        Mask of shape (B, T). True/1.0 = keep (unless invert=True).
    """
    lengths = tf_cast(lengths, tf_int32)
    if max_len is None:
        max_len = tf_reduce_max(lengths)
    max_len = tf_cast(max_len, tf_int32)

    rng = tf_range(max_len)                   # (T,)
    rng = tf_expand_dims(rng, 0)              # (1, T)
    len_col = tf_expand_dims(lengths, 1)      # (B, 1)

    mask = tf_cast(rng < len_col, tf_bool)    # (B, T) bool
    if invert:
        mask = tf_logical_not(mask)

    if dtype is tf_float32:
        return tf_cast(mask, tf_float32)
    return mask


def sequence_mask_3d(
    data: Tensor,
    *,
    lengths: Optional[Tensor] = None,
    mask_2d: Optional[Tensor] = None,
    time_axis: int = 1,
    invert: bool = False,
    dtype: Tensor = tf_bool,
) -> Tensor:
    """
    Produce a broadcastable mask for a (B, T, F) tensor.

    One of `lengths` or `mask_2d` must be given.

    Parameters
    ----------
    data : Tensor, shape (B, T, F) or compatible
        Reference tensor to infer batch/time dims.
    lengths : Tensor[int], optional shape (B,)
        Build mask from valid lengths.
    mask_2d : Tensor[bool/0-1], optional shape (B, T)
        Precomputed 2D mask.
    time_axis : int
        Axis of time dim in `data` (default 1).
    invert : bool
        Return the inverse mask.
    dtype : tf.bool or tf.float32
        Output dtype.

    Returns
    -------
    Tensor
        Mask shaped (B, T, 1) to multiply with data or feed to MHA.
    """
    if lengths is None and mask_2d is None:
        raise ValueError("Provide `lengths` or `mask_2d`.")

    # Infer batch/time from data
    b = tf_shape(data)[0] # noqa
    t = tf_shape(data)[time_axis]

    if mask_2d is None:
        mask_2d = pad_mask_from_lengths(
            lengths, max_len=t, dtype=tf_bool
        )
    else:
        mask_2d = tf_cast(mask_2d, tf_bool)

    if invert:
        mask_2d = tf_logical_not(mask_2d)

    # Expand to (B, T, 1) so it broadcasts over feature dim
    mask_3d = tf_expand_dims(mask_2d, -1)

    if dtype is tf_float32:
        return tf_cast(mask_3d, tf_float32)
    return mask_3d
