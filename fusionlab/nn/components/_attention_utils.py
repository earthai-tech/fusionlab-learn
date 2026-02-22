# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Utility helpers for attention masks.

- create_causal_mask(seq_len)
- combine_masks(mask_a, mask_b, mode='and')
"""

from __future__ import annotations
from typing import Union, Optional

from ._config import (
    Tensor,
    tf_cast, tf_int32, tf_range, tf_expand_dims, tf_greater,
    tf_float32, tf_bool, tf_logical_and, tf_logical_or, tf_logical_not,
    tf_linalg, tf_ones, 
)

__all__ = ["create_causal_mask", "combine_masks"]


def create_causal_mask(size: Union[int,Tensor]) -> Tensor:
    """
    Creates a causal attention mask of shape [1,1,seq_len,seq_len]
    where mask[0,0,i,j] = 1.0 if j > i else 0.0.
    """
    
    # Make sure size is a 0-D int32 Tensor
    size = tf_cast(size, tf_int32)

    # Build a vector [0,1,2,...,size-1]
    idxs = tf_range(size)                  # shape: [size]

    # Compare row < col for every pair (i,j)
    #   row_idxs: [size,1], col_idxs: [1,size]
    row_idxs = tf_expand_dims(idxs, 1)     # [size,1]
    col_idxs = tf_expand_dims(idxs, 0)     # [1,size]

    # mask2d[i,j] = True if j > i, else False
    mask2d = tf_greater(col_idxs, row_idxs)  # [size,size], dtype=bool

    # Cast to float (1.0 for masked positions, 0.0 elsewhere)
    mask2d = tf_cast(mask2d, tf_float32)     # [size,size]

    # Expand to [1,1,size,size] so it broadcasts over (batch, heads)
    mask = tf_expand_dims(tf_expand_dims(mask2d, 0), 1)

    return mask

def create_causal_mask_(size: Union[int, Tensor]) -> Tensor:
    """
    Build a causal mask shaped (1, 1, L, L) where entries are 1.0 when
    j > i (future positions) and 0.0 otherwise. Broadcasts over batch
    and heads in Keras MHA.

    Parameters
    ----------
    size : int or Tensor
        Sequence length.

    Returns
    -------
    Tensor
        Float mask (1.0 = masked) of shape (1, 1, L, L).
    """
    size = tf_cast(size, tf_int32)
    idxs = tf_range(size)                 # [L]
    row = tf_expand_dims(idxs, 1)         # [L,1]
    col = tf_expand_dims(idxs, 0)         # [1,L]
    mask2d = tf_greater(col, row)         # True if j > i
    mask2d = tf_cast(mask2d, tf_float32)  # [L,L]
    return tf_expand_dims(tf_expand_dims(mask2d, 0), 1)  # [1,1,L,L]


def combine_masks(
    mask_a: Optional[Tensor],
    mask_b: Optional[Tensor],
    *,
    mode: str = "and",
    invert_b: bool = False,
) -> Optional[Tensor]:
    """
    Combine two boolean/0-1 masks into one.

    Parameters
    ----------
    mask_a, mask_b : Tensor or None
        Any broadcastable masks. If one is None, the other is returned.
        Masks may be bool or 0/1 floats.
    mode : {'and','or','xor'}
        Logical op used to merge masks.
    invert_b : bool, default False
        If True, logical-not is applied to mask_b before combining.

    Returns
    -------
    Tensor or None
        Combined mask (bool). None if both inputs are None.
    """
    if mask_a is None and mask_b is None:
        return None

    def _to_bool(x: Tensor) -> Tensor:
        return tf_cast(x, tf_bool)

    if mask_a is None:
        mb = _to_bool(mask_b)
        return tf_logical_not(mb) if invert_b else mb
    if mask_b is None:
        return _to_bool(mask_a)

    a = _to_bool(mask_a)
    b = _to_bool(mask_b)
    if invert_b:
        b = tf_logical_not(b)

    if mode == "and":
        return tf_logical_and(a, b)
    if mode == "or":
        return tf_logical_or(a, b)
    if mode == "xor":
        # xor = (a or b) and not (a and b)
        return tf_logical_and(tf_logical_or(a, b),
                              tf_logical_not(tf_logical_and(a, b)))
    raise ValueError("mode must be 'and', 'or', or 'xor'.")

# def create_causal_mask(size: Union[int, Tensor]) -> Tensor:
#     """Creates a causal attention mask of shape [1,1,seq_len,seq_len]."""
#     # ensure `size` is an int32 Tensor
#     size = tf_cast(size, tf_int32)

#     # build shape as a Tensor so we don't capture Python tuples of Tensors
#     shape = tf_stack([size, size])
#     ones = tf_ones(shape, dtype=tf_float32)

#     # make the [seq_len, seq_len] causal matrix
#     mask2d = 1.0 - tf_linalg.band_part(ones, -1, 0)

#     # now expand batch dim then head dim → [1,1,seq_len,seq_len]
#     mask = tf_expand_dims(mask2d, 0)  # → [1, seq_len, seq_len]
#     mask = tf_expand_dims(mask, 1)    # → [1, 1, seq_len, seq_len]

#     return mask

def _create_causal_mask(size: Union[int, Tensor]) -> Tensor:
    """Creates a causal attention mask for the decoder."""
    mask = 1 - tf_linalg.band_part(tf_ones((size, size)), -1, 0)
    # Add batch and head dimensions for broadcasting
    return mask[tf_expand_dims(tf_range(size), 0), :] # (1, 1, seq_len, seq_len) -> Keras MHA expects (B, T, T) or (B, N_heads, T, T)
                                                      # TF MHA expects (B, N_heads, T, T)
                                                      # Let's make it (1,1,T,T) for TF MHA layer, it will broadcast
    # # Keras MHA expects mask shape (batch_size, num_heads, query_length, key_length)
    # # or (batch_size, query_length, key_length)
    # # For causal, query_length == key_length == size
    # return tf_expand_dims(tf_expand_dims(
    #     1 - tf_linalg.band_part(tf_ones((size, size)), -1, 0), axis=0), axis=0)


