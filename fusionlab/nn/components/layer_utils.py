# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
fusionlab/nn/components/layer_utils.py

Small generic helpers + micro‑layers (residual, gating, etc.).

"""
from __future__ import annotations

from typing import Optional, Tuple

from ...utils.deps_utils import ensure_pkg
from ...api.property import NNLearner

from ._config import (
    Layer, 
    Tensor, 
    Dense, 
    register_keras_serializable,
    tf_shape, 
    tf_rank, 
    tf_bool, 
    tf_stack, 
    tf_expand_dims, 
    tf_cast,
    tf_float32, 
    tf_add, 
    tf_multiply,
    tf_where, 
    tf_tile,
    tf_reduce_mean, 
    KERAS_BACKEND,
    KERAS_DEPS, 
    DEP_MSG, 
    tf_autograph
)

K =KERAS_DEPS 
__all__ = [

    "maybe_expand_time",
    "broadcast_like",
    "ensure_rank_at_least",
    "apply_residual",
    "drop_path",

    "ResidualAdd",
    "LayerScale",
    "StochasticDepth",
    "SqueezeExcite1D",
    "Gate",
]


@register_keras_serializable(
    "fusionlab.nn.components", name="ResidualAdd"
    )
class ResidualAdd(Layer, NNLearner):
    """Y = X + F(X). Assumes shapes match."""

    def call(self, inputs: Tuple[Tensor, Tensor], training=False) -> Tensor:
        x, f = inputs
        return tf_add(x, f)

@register_keras_serializable("fusionlab.nn.components",
                             name="LayerScale")
class LayerScale(Layer, NNLearner):
    """
    Per-channel trainable scale vector (like ConvNeXt).

    Parameters
    ----------
    init_value : float
        Small value to start with (e.g. 1e-4).
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, init_value: float = 1e-4, **kw):
        super().__init__(**kw)
        self.init_value = init_value

    def build(self, input_shape):
        gamma_shape = input_shape[-1:]
        self.gamma = self.add_weight(
            "gamma",
            shape=gamma_shape,
            initializer=lambda s, d=None: tf_float32.as_numpy_dtype(
                self.init_value
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: Tensor, training=False) -> Tensor:
        return x * self.gamma

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"init_value": self.init_value})
        return cfg


@register_keras_serializable("fusionlab.nn.components",
                             name="StochasticDepth")
class StochasticDepth(Layer, NNLearner):
    """
    Wrap a branch with DropPath (stochastic depth).

    Parameters
    ----------
    drop_prob : float
        Probability of dropping the path.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, drop_prob: float = 0.1, **kw):
        super().__init__(**kw)
        self.drop_prob = drop_prob

    def call(self, x: Tensor, training=False) -> Tensor:
        return drop_path(x, self.drop_prob, training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prob": self.drop_prob})
        return cfg


@register_keras_serializable("fusionlab.nn.components",
                             name="SqueezeExcite1D")
class SqueezeExcite1D(Layer, NNLearner):
    """
    Simple SE block for (B,T,C) or (B,C).

    Parameters
    ----------
    ratio : int
        Reduction ratio, e.g. 16.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, ratio: int = 16, **kw):
        super().__init__(**kw)
        self.ratio = ratio

    def build(self, input_shape):
        c = input_shape[-1]
        mid = max(c // self.ratio, 1)
        self.fc1 = Dense(mid, activation="relu", name="se_fc1")
        self.fc2 = Dense(c, activation="sigmoid", name="se_fc2")
        super().build(input_shape)

    def call(self, x: Tensor, training=False) -> Tensor:
        # Global squeeze
        if tf_rank(x) == 3:              # (B,T,C) -> (B,C)
            z = tf_reduce_mean(x, axis=1)
        else:                             # (B,C)
            z = x

        s = self.fc2(self.fc1(z))         # (B,C)
        if tf_rank(x) == 3:
            s = tf_expand_dims(s, 1)      # (B,1,C)
        return x * s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio})
        return cfg


@register_keras_serializable("fusionlab.nn.components", name="Gate")
class Gate(Layer, NNLearner):
    """
    Generic gating layer: y = x * σ(Wx + b).

    Parameters
    ----------
    units : int
        Output units = input dim unless overridden.
    use_bias : bool
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: Optional[int] = None,
                 use_bias: bool = True, **kw):
        super().__init__(**kw)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        feat = input_shape[-1] if self.units is None else self.units
        self.proj = Dense(feat, activation="sigmoid",
                          use_bias=self.use_bias,
                          name="gate_proj")
        super().build(input_shape)

    def call(self, x: Tensor, training=False) -> Tensor:
        g = self.proj(x)
        return tf_multiply(x, g)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "use_bias": self.use_bias})
        return cfg


# ------------------------- Helper functions -----------------------

def maybe_expand_time(x: Tensor, ref: Tensor, axis: int = 1) -> Tensor:
    """
    If `x` lacks a time dim but `ref` has one, expand `x` on that axis.

    Parameters
    ----------
    x   : Tensor  (B, F) or (B,T,F)
    ref : Tensor  reference for time dim (B,T,?)
    axis: int     where to insert time dim

    Returns
    -------
    Tensor
        (B,T,F) if expanded, else original `x`.
    """
    xr = tf_rank(x)
    rr = tf_rank(ref)
    if rr >= 3 and xr == rr - 1:
        return tf_expand_dims(x, axis=axis)
    return x

def broadcast_like(x: Tensor, target: Tensor) -> Tensor:
    """
    Broadcast `x` so it matches `target` along leading dims.

    If last dim already matches, tile/expand others.
    Example: x:(B,1,F) target:(B,T,F) -> repeat T times.

    NOTE: Uses dynamic shapes; avoid for huge dims.
    """
    x_shape = tf_shape(x)
    t_shape = tf_shape(target)

    # Pad ranks by inserting singleton dims until ranks match
    while tf_rank(x) < tf_rank(target):
        x = tf_expand_dims(x, 1)
        x_shape = tf_shape(x)

    # Build repetition vector for tf_tile
    reps = []
    tgt_rank = tf_rank(target)
    for i in range(tgt_rank):
        same = tf_cast(t_shape[i] == x_shape[i], tf_bool)
        # if same -> 1 else -> t_shape[i]
        reps_i = tf_where(same, 1, t_shape[i])
        reps.append(reps_i)

    # Convert list of scalars to int32 tensor
    reps = tf_stack(reps)
    reps_dtype = tf_shape(tf_shape(x)).dtype  # usually int32
    reps = tf_cast(reps, reps_dtype)

    return tf_tile(x, reps)

def _broadcast_like(x: Tensor, target: Tensor) -> Tensor:
    """
    Broadcast `x` so it matches `target` along leading dims.

    If last dim already matches, tile/expand others.
    Example: x:(B,1,F) target:(B,T,F) -> repeat T times.

    NOTE: Uses dynamic shapes; avoid for huge dims.

    Returns
    -------
    Tensor
    """
    x_shape = tf_shape(x)
    t_shape = tf_shape(target)

    # pad ranks
    while tf_rank(x) < tf_rank(target):
        x = tf_expand_dims(x, 1)
        x_shape = tf_shape(x)

    # repeat along mismatching axes
    reps = []
    for i in range(tf_rank(target)):
        reps_i = tf_where(
            t_shape[i] == x_shape[i],
            1,
            t_shape[i]
        )
        reps.append(reps_i)
    reps = tf_cast(reps, tf_int32 := tf_shape(# noqa
        tf_shape(x)).dtype) 
    return tf_tile(x, reps)

def ensure_rank_at_least(x: Tensor, min_rank: int,
                         axis_to_expand: int = -1) -> Tensor:
    """
    Pad dimensions (=1) until rank >= min_rank.

    Parameters
    ----------
    x : Tensor
    min_rank : int
    axis_to_expand : int
        Where to insert singleton dims.

    Returns
    -------
    Tensor
    """
    r = tf_rank(x)
    while r < min_rank:
        x = tf_expand_dims(x, axis=axis_to_expand)
        r += 1
    return x


def apply_residual(x: Tensor, y: Tensor) -> Tensor:
    """
    Add & return residual. Shape check is user's job.
    """
    return tf_add(x, y)


@tf_autograph.experimental.do_not_convert
def drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    """
    Stochastic depth on residual branches. If training, randomly
    zero the entire sample (per-batch item) path with prob p.

    Parameters
    ----------
    x : Tensor (B, ... , F)
    drop_prob : float
    training : bool

    Returns
    -------
    Tensor
    """
    if (not training) or drop_prob <= 0.0:
        return x

    b = tf_shape(x)[0]
    keep_prob = 1.0 - drop_prob
    # mask shape: (B,1,1,...)
    shape = [b] + [1] * (tf_rank(x) - 1)
    # uniform in [0,1)
    if hasattr(K, "random"):
        rnd = K.random.uniform(shape)
    else:  # fallback
        from tensorflow.random import uniform as tf_uniform
        rnd = tf_uniform(shape, dtype=tf_float32)

    mask = tf_cast(rnd < keep_prob, tf_float32)
    # rescale to preserve expected value
    return x * mask / keep_prob
