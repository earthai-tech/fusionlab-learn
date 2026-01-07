# -*- coding: utf-8 -*-
# _geoprior_utils.py
"""
Batch.io 
"""

from __future__ import annotations
from typing import Union, Dict, Sequence, Any 

from collections.abc import Mapping

import numpy as np

from ....utils.generic_utils import rename_dict_keys
from ... import KERAS_DEPS

Tensor = KERAS_DEPS.Tensor

tf_float32 = KERAS_DEPS.float32
tf_int32 = KERAS_DEPS.int32

tf_cast = KERAS_DEPS.cast
tf_constant = KERAS_DEPS.constant
tf_debugging = KERAS_DEPS.debugging
tf_equal = KERAS_DEPS.equal
tf_maximum = KERAS_DEPS.maximum
tf_minimum = KERAS_DEPS.minimum
tf_greater_equal = KERAS_DEPS.greater_equal
tf_rank = KERAS_DEPS.rank
tf_cond = KERAS_DEPS.cond
tf_shape = KERAS_DEPS.shape
tf_zeros_like = KERAS_DEPS.zeros_like
tf_ones = KERAS_DEPS.ones 
tf_greater = KERAS_DEPS.greater 
tf_cond = KERAS_DEPS.cond 
tf_concat = KERAS_DEPS.concat 
tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor 
tf_ones_like = KERAS_DEPS.ones_like 
tf_less_equal =KERAS_DEPS.less_equal 
tf_abs = KERAS_DEPS.abs 
tf_print = KERAS_DEPS.print 
tf_reduce_mean = KERAS_DEPS.reduce_mean 
tf_expand_dims = KERAS_DEPS.expand_dims 
tf_tile = KERAS_DEPS.tile 

def _get_coords(
    inputs: Union[Dict[str, Any], Sequence[Any], Tensor],
    *,
    check_shape: bool = False,
    is_time_dependent: bool = True,
) -> Tensor:
    r"""
    Extract the **coords** tensor from any input layout.

    Parameters
    ----------
    inputs : dict, sequence or tensor
        * **Dict** – must contain a ``"coords"`` key.  
        * **Sequence** – coords are expected at index 0.  
        * **Tensor** – interpreted as coords directly.
    check_shape : bool, default ``False``
        If ``True``, validate that the last dimension equals 3 and that
        the rank matches *is_time_dependent*.
    is_time_dependent : bool, default ``True``
        * ``True``  → expect a 3-D shape ``(B, T, 3)``  
        * ``False`` → allow a 2-D shape ``(B, 3)``; a 3-D shape is also
          accepted (useful when the model ignores time).

    Returns
    -------
    tf.Tensor
        The extracted coords tensor.

    Raises
    ------
    KeyError
        If a dict is missing the ``"coords"`` key.
    TypeError
        If *inputs* is not dict, sequence or tensor.
    ValueError
        If ``check_shape`` is ``True`` and the tensor shape is invalid.

    Notes
    -----
    The function is lightweight and executes outside any
    ``tf.function`` context; use it freely inside the model’s
    ``train_step`` or data-pipe helpers.
    """
    # ── 1. locate the tensor 
    if isinstance(inputs, Tensor):
        coords = inputs
    elif isinstance(inputs, (tuple, list)):
        coords = inputs[0]
    elif isinstance(inputs, dict):
        try:
            coords = inputs["coords"]
        except KeyError as err:
            raise KeyError("Input dict lacks a 'coords' entry.") from err
    else:
        raise TypeError(
            "inputs must be a dict, sequence, or Tensor; "
            f"got {type(inputs).__name__}"
        )

    # ── 2. validate shape if requested 
    if check_shape:
        shape = coords.shape           # static if known, else dynamic
        # last dim must be 3
        if shape.rank is not None:
            if shape[-1] != 3:
                raise ValueError(
                    "coords[..., -1] must equal 3 (t, x, y); "
                    f"found {shape[-1]}"
                )
            # rank check when static
            if is_time_dependent and shape.rank != 3:
                raise ValueError(
                    "Expected coords rank 3 (B, T, 3) for time-dependent "
                    "data; received rank "
                    f"{shape.rank}"
                )
            if not is_time_dependent and shape.rank not in (2, 3):
                raise ValueError(
                    "Expected coords rank 2 or 3 for static data; "
                    f"received rank {shape.rank}"
                )
        else:  # dynamic shapes (inside tf.function)
            tf_debugging.assert_equal(
                tf_shape(coords)[-1],
                3,
                message="coords[..., -1] must equal 3 (t, x, y)",
            )
            if is_time_dependent:
                tf_debugging.assert_rank(coords, 3)
            else:
                tf_debugging.assert_rank_in(coords, (2, 3))

    return coords

def _canonicalize_targets(targets):
    """
    Return targets as a dict matching compiled output names.

    Accepts:
    - dict targets: {"subsidence": ..., "gwl": ...} or already
      {"subs_pred": ..., "gwl_pred": ...}
    - tuple/list targets: (subs, gwl) -> {"subs_pred": subs, "gwl_pred": gwl}

    This avoids Keras 3 structure-mismatch errors when y_pred is a dict.
    """

    if isinstance(targets, Mapping):
        tgt = dict(targets)
        # already canonical?
        if ("subs_pred" in tgt) and ("gwl_pred" in tgt):
            return tgt

        return rename_dict_keys(
            tgt,
            param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"},
        )

    if isinstance(targets, (tuple, list)):
        if len(targets) == 2:
            return {"subs_pred": targets[0], "gwl_pred": targets[1]}
        if len(targets) == 1:
            return {"subs_pred": targets[0]}

    return targets

# ---------------------------------------------------------------------
# Quantile helpers (shape-safe)
# ---------------------------------------------------------------------
def _q_index(quantiles, q=0.5):
    """Return the index of q in quantiles (python int), else None."""
    if quantiles is None:
        return None
    try:
        qs = [float(x) for x in list(quantiles)]
    except Exception:
        return None
    # exact match first
    for i, qi in enumerate(qs):
        if abs(qi - float(q)) < 1e-12:
            return i
    # fallback: closest
    qv = float(q)
    i0 = int(np.argmin([abs(qi - qv) for qi in qs]))
    return i0

def select_q(pred, quantiles=None, q=0.5, fallback="mean"):
    """
    Select q-quantile from pred.

    pred:
      - (B,H,Q,O) -> returns (B,H,O)
      - (B,H,Q,1) -> returns (B,H,1)
      - otherwise returned as-is.
    """
    r = getattr(pred, "shape", None)
    rank = pred.shape.rank if r is not None else None

    if rank != 4:
        return pred

    idx = _q_index(quantiles, q=q)
    if idx is not None:
        return pred[:, :, idx, :]

    # fallback if quantiles list is missing/odd
    if str(fallback).lower() == "mean":
        return tf_reduce_mean(pred, axis=2)
    # "middle" fallback
    qn = tf_shape(pred)[2]
    mid = qn // 2
    return pred[:, :, mid, :]


def tile_true_to_quantiles(y_true, y_pred):
    """
    Make y_true compatible with y_pred when y_pred has quantile axis.

    y_true: (B,H,O) and y_pred: (B,H,Q,O) -> return y_true_q: (B,H,Q,O)
    Else return y_true unchanged.
    """
    rt = y_true.shape.rank
    rp = y_pred.shape.rank
    if (rp == 4) and (rt == 3):
        y4 = tf_expand_dims(y_true, axis=2)  # (B,H,1,O)
        qn = tf_shape(y_pred)[2]
        return tf_tile(y4, [1, 1, qn, 1])
    return y_true


def _align_true_for_loss(y_true, y_pred):
    """
    Align y_true to y_pred for quantile-aware losses/metrics.

    Never tiles y_true across Q. Only reduces y_true if it accidentally
    arrives with a quantile axis.
    """
    y_true = tf_convert_to_tensor(y_true)
    y_pred = tf_convert_to_tensor(y_pred)

    # (B,H) -> (B,H,1)
    if y_true.shape.rank == 2:
        y_true = tf_expand_dims(y_true, axis=-1)

    rt = y_true.shape.rank
    rp = y_pred.shape.rank

    # (B,H,Q,O) -> (B,H,O)
    if rt == 4:
        return y_true[:, :, 0, :]

    # (B,H,Q) -> (B,H,1)  (rare but happens if upstream tiled targets)
    if rt == 3 and rp in (3, 4):
        if rp == 4:
            q_dim = y_pred.shape[2]
            if (q_dim is None) or (y_true.shape[-1] == q_dim):
                return tf_expand_dims(y_true[:, :, 0], axis=-1)
        else:
            return tf_expand_dims(y_true[:, :, 0], axis=-1)

    return y_true