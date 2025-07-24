# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Prediction heads (point, quantile, probabilistic mixtures, etc.)
and a combinator loss wrapper that aggregates per-head losses.
"""

from __future__ import annotations

from typing import Optional, List, Union,  Dict, Tuple, Mapping
from ...api.property import NNLearner
from ...utils.deps_utils import ensure_pkg

from ._config import KERAS_BACKEND, DEP_MSG
from ._config import (
    Layer, 
    Dense,
    Loss, 
    Softmax,
    Tensor, 
    register_keras_serializable,
    get_loss,
    
    tf_add_n,
    tf_float32,  
    tf_expand_dims,
    tf_stack,
    tf_reduce_mean,
    tf_reshape,
    tf_autograph, 
    tf_shape, 
    tf_cast, 
    tf_square, 
    tf_log, 
    tf_constant, 
    tf_softplus, 
    tf_reduce_logsumexp, 
    tf_tile, 
    tf_newaxis,
    tf_reduce_sum

)

__all__ = [
    "QuantileHead",
    "PointForecastHead",
    "GaussianHead",
    "MixtureDensityHead",
    "CombinedHeadLoss",
    "QuantileDistributionModeling",
]

   
@register_keras_serializable(
    "fusionlab.nn.components", name="GaussianHead"
)
class GaussianHead(Layer, NNLearner):
    """
    Parametric head that predicts a univariate Gaussian per target:
    mean μ and stddev σ > 0.

    Input supports shape (B, F) or (B, H, F).
    Output dict fields (same leading dims):
        - 'mean'  : (B, [H], O)
        - 'scale' : (B, [H], O)    (σ = softplus(raw) + eps)

    Parameters
    ----------
    output_dim : int
        Number of target variables per horizon (O).
    min_scale : float, optional
        Numerical floor added to softplus to keep σ positive.
    """

    @ensure_pkg(
        KERAS_BACKEND or "keras", 
        extra="GaussianHead needs Keras backend."
    )
    def __init__(
        self, output_dim: int, 
        min_scale: float = 1e-4, 
        **kwargs
        ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.min_scale = float(min_scale)
        # Predict 2 * O parameters (μ and raw σ)
        self.proj = Dense(
            2 * output_dim, name="gaussian_head_dense"
            )

    def call(self, features: Tensor, training: bool = False
             ) -> Dict[str, Tensor]:
        params = self.proj(features)                           # (B,[H], 2*O)
        shp = tf_shape(params)
        new_shape = tf_stack(shp[:-1] + [2, self.output_dim])  # (..., 2, O)
        params = tf_reshape(params, new_shape)

        mean  = params[..., 0, :]                              # (..., O)
        raw_s = params[..., 1, :]                              # (..., O)
        # Softplus for strict positivity
        scale = tf_softplus(raw_s) + self.min_scale

        return {"mean": mean, "scale": scale}

    @tf_autograph.experimental.do_not_convert
    def nll(self, y_true: Tensor, mean: Tensor, scale: Tensor
            ) -> Tensor:
        """
        Computes −log p(y | μ, σ) for a factorised Normal.

        Shapes
        ------
        y_true, mean, scale : (B, [H], O)

        Returns
        -------
        scalar Tensor
        """
        two_pi = tf_constant(2.0 * 3.141592653589793, dtype=tf_float32)
        var = tf_square(scale)
        # log σ + (y-μ)^2 / (2 σ^2) + 0.5 log(2π)
        log_prob = (
            tf_log(scale)
            + tf_square(y_true - mean) / (2.0 * var)
            + 0.5 * tf_log(two_pi)
        )
        return tf_reduce_mean(log_prob)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {"output_dim": self.output_dim, "min_scale": self.min_scale})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(
    "fusionlab.nn.components", name="MixtureDensityHead"
)
class MixtureDensityHead(Layer, NNLearner):
    """
    Mixture Density Network head (Gaussian mixtures).

    Predicts K-component mixture for each target:
        - weights π_k  (softmax across K)
        - means   μ_k
        - scales  σ_k  (softplus)

    Output dict:
        'weights': (B,[H], K, O)
        'means'  : (B,[H], K, O)
        'scales' : (B,[H], K, O)

    Parameters
    ----------
    output_dim : int
        Target dimensionality per horizon (O).
    num_components : int
        Number of mixture components K.
    min_scale : float
        Numerical floor added to σ.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", 
                extra="MixtureDensityHead needs Keras backend.")
    def __init__(
        self,
        output_dim: int,
        num_components: int,
        min_scale: float = 1e-4,
        **kwargs
    ):
        super().__init__(**kwargs)
        if num_components < 1:
            raise ValueError("num_components must be >= 1.")
        self.output_dim = output_dim
        self.num_components = num_components
        self.min_scale = float(min_scale)

        # Total params per target = K (weights) + K (means) + K (scales)
        # but weights have to be separate because of softmax over K.
        # We'll predict everything in a single Dense and split.
        self.param_proj = Dense(
            num_components * (2 * output_dim) + num_components,  # w + μ,σ
            name="mdn_dense"
        )
        self.softmax = Softmax(axis=-2)  # softmax across K

    def call(self, features: Tensor, training: bool = False
             ) -> Dict[str, Tensor]:
        raw = self.param_proj(features)  # (B,[H], K*(2*O) + K)
        shp = tf_shape(raw)
        # last = shp[-1] # noqa

        # Split: first K for weights, remaining 2*K*O for μ/σ
        k = self.num_components
        o = self.output_dim
        w_end = k
        w_raw = raw[..., :w_end]                       # (B,[H], K)
        rest = raw[..., w_end:]                        # (B,[H], 2*K*O)

        # Reshape rest → (..., K, 2, O)
        rest_shape = tf_stack(shp[:-1] + [k, 2, o])
        rest = tf_reshape(rest, rest_shape)
        means  = rest[..., 0, :]                       # (..., K, O)
        raw_s  = rest[..., 1, :]                       # (..., K, O)
        scales = tf_softplus(raw_s) + self.min_scale

        # weights softmax across K (axis=-2)
        # Need to align dims: add an O dim for broadcasting if O>1
        # We'll expand to (.., K, 1) then tile to O
        w = self.softmax(tf_expand_dims(w_raw, axis=-1))  # (..., K, 1)
        # broadcast or tile to (..., K, O)
        if o > 1:
            w = tf_tile(w, [1] * (len(w.shape) - 1) + [o])
        return {"weights": w, "means": means, "scales": scales}

    # Negative log-likelihood for mixtures
    @tf_autograph.experimental.do_not_convert
    def nll(
            self, y_true: Tensor, weights: Tensor, means: Tensor,
            scales: Tensor) -> Tensor:
        """
        Compute −log Σ_k π_k N(y|μ_k, σ_k) assuming factorised over O.

        Shapes
        ------
        y_true   : (B,[H], O)
        weights  : (B,[H], K, O)
        means    : (B,[H], K, O)
        scales   : (B,[H], K, O)

        Returns
        -------
        scalar Tensor
        """
        two_pi = tf_constant(2.0 * 3.141592653589793, dtype=tf_float32)
        var = tf_square(scales)

        # log N = -0.5 * [log(2πσ^2) + (y-μ)^2/σ^2]
        log_norm = (
            -0.5 * (tf_log(two_pi * var) + tf_square(
                y_true[..., tf_newaxis, :] - means) / var)
        )  # (B,[H], K, O)

        # log Σ_k π_k exp(log_norm)  (log-sum-exp per O)
        # weights in prob space -> convert to log
        log_w = tf_log(weights)
        log_mix = tf_reduce_logsumexp(log_w + log_norm, axis=-2)  # sum over K

        # Sum across O, then mean over batch/time
        # If you consider independence across O: sum log p(o)
        nll = - tf_reduce_mean(tf_reduce_sum(log_mix, axis=-1))
        return nll

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "output_dim": self.output_dim,
            "num_components": self.num_components,
            "min_scale": self.min_scale,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable(
    "fusionlab.nn.components", name="PointForecastHead"
)
class PointForecastHead(Layer, NNLearner):
    r"""
    Simple dense head that outputs point forecasts.

    Typical use is with MSE/MAE losses. Supports inputs shaped
    (B, F) or (B, H, F); the Dense layer is applied position‑wise.

    Parameters
    ----------
    output_dim : int
        Number of target features per horizon (O).
    """

    @ensure_pkg(KERAS_BACKEND or "keras",
                extra="PointForecastHead needs Keras backend.")
    def __init__(self, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        # Single projection to the final target dimension
        self.proj = Dense(output_dim, name="point_head_dense")

    def call(self,
             features: Tensor,
             training: bool = False) -> Tensor:
        """Forward pass: simple linear projection."""
        return self.proj(features)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"output_dim": self.output_dim})
        return cfg

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)



@register_keras_serializable(
    "fusionlab.nn.components", name="QuantileHead"
)
class QuantileHead(Layer, NNLearner):
    r"""
    Dense head that outputs per‑quantile forecasts.

    Given an input tensor of shape (B, F) or (B, H, F), this head
    returns a tensor of shape (B, H?, Q, O), where Q is the number
    of quantiles and O is the per‑horizon output dimension.

    Parameters
    ----------
    quantiles : List[float]
        e.g. [0.1, 0.5, 0.9]. Must be non‑empty.
    output_dim : int
        Target dimension per horizon (O).
    """

    @ensure_pkg(KERAS_BACKEND or "keras",
                extra="QuantileHead needs Keras backend.")
    def __init__(self,
                 quantiles: List[float],
                 output_dim: int,
                 **kwargs):
        super().__init__(**kwargs)
        if not quantiles:
            raise ValueError("Quantiles list must be non‑empty.")
        self.quantiles   = quantiles
        self.output_dim  = output_dim
        self.q           = len(quantiles)

        # Project to Q * O, then reshape to (..., Q, O)
        self.proj = Dense(self.q * output_dim,
                          name="quantile_head_dense")

    def call(self,
             features: Tensor,
             training: bool = False) -> Tensor:
        """
        Forward pass.

        The Dense layer outputs (..., Q*O). We then reshape to
        (..., Q, O), preserving any leading batch / horizon dims.
        """
        # Supports (B, F) or (B,H,F). Output should insert Q dimension before O.
        out = self.proj(features)                 # (B,[H], Q*O)
        shp = tf_shape(out)                       # dynamic shape
        new_shape = tf_stack(                     # (B,[H], Q, O)
            shp[:-1] + [self.q, self.output_dim]
        )
        out = tf_reshape(out, new_shape)
        return out

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "quantiles": self.quantiles,
            "output_dim": self.output_dim
        })
        return cfg

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


@register_keras_serializable(
    "fusionlab.nn.components", name="CombinedHeadLoss"
)
class CombinedHeadLoss(Loss, NNLearner):
    """
    Aggregates multiple head-specific losses into a single scalar.

    It expects `y_true` and `y_pred` to be *matching nested structures* (dict or
    list) keyed by head names. Each head name must have a corresponding
    (loss_fn, weight) pair supplied at init time.

    Parameters
    ----------
    heads_losses : Mapping[str, Tuple[Loss, float]]
        Dict mapping head_name -> (loss_fn, weight). Weight defaults to 1.0.
        Each `loss_fn` must be a Keras-compatible Loss (callable(y_true, y_pred)).
    reduction : str, optional
        Keep 'sum' (default). (Could add 'mean' later if needed.)

    Example
    -------
    >>> comb_loss = CombinedHeadLoss({
    ...     "point": (tf.keras.losses.MSE(), 1.0),
    ...     "quantile": (AdaptiveQuantileLoss([.1,.5,.9]), 0.5),
    ... })
    >>> # Inside model compile, y_pred/y_true are dicts with those keys.
    """

    @ensure_pkg("keras", extra="CombinedHeadLoss needs Keras backend.")
    def __init__(
        self,
        heads_losses: Mapping[str, Tuple[Loss, float]],
        reduction: str = "sum",
        name: str = "CombinedHeadLoss",
    ):
        super().__init__(name=name, reduction="sum")  # we manage reduction manually
        if not heads_losses:
            raise ValueError("heads_losses cannot be empty.")

        # Normalize to {str: (Loss, weight)}
        norm: Dict[str, Tuple[Loss, float]] = {}
        for k, v in heads_losses.items():
            if isinstance(v, (list, tuple)):
                if len(v) == 1:
                    norm[k] = (v[0], 1.0)
                else:
                    norm[k] = (v[0], float(v[1]))
            else:
                norm[k] = (v, 1.0)
        self.heads_losses = norm
        self._reduction_mode = reduction

    def call(self, y_true, y_pred):
        """
        Assumes y_true and y_pred are structures with same keys as heads_losses.

        For example:
            y_true = {"point": ..., "quantile": ...}
            y_pred = {"point": ..., "quantile": ...}
        """
        total_terms = []
        for head, (loss_fn, w) in self.heads_losses.items():
            if head not in y_true or head not in y_pred:
                raise KeyError(
                    f"Missing key '{head}' in y_true/y_pred for CombinedHeadLoss."
                )
            lt = loss_fn(y_true[head], y_pred[head])
            total_terms.append(tf_cast(w, tf_float32) * lt)

        if self._reduction_mode == "sum":
            return tf_add_n(total_terms)
        elif self._reduction_mode == "mean":
            return tf_reduce_mean(tf_stack(total_terms))
        else:
            raise ValueError(f"Unknown reduction '{self._reduction_mode}'.")

    def get_config(self):
        cfg = super().get_config()
        # For serialization, store sub-loss configs
        sub_cfg = {}
        for k, (loss_fn, w) in self.heads_losses.items():
            sub_cfg[k] = {
                "loss_class": loss_fn.__class__.__name__,
                "config": getattr(loss_fn, "get_config", lambda: {})(),
                "weight": w,
            }
        cfg.update({
            "heads_losses": sub_cfg,
            "reduction_mode": self._reduction_mode,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        # We need to rebuild each loss. We only know class name string; user may
        # prefer to pass already-built object (deserialization logic can be customized).
        sub_cfg = config.pop("heads_losses")
        rebuilt: Dict[str, Tuple[Loss, float]] = {}

        for k, info in sub_cfg.items():
            # Try generic keras get(); if fails, user must patch here
            loss_obj = get_loss(info["config"])
            rebuilt[k] = (loss_obj, info["weight"])
        reduction = config.pop("reduction_mode", "sum")
        obj = cls(rebuilt, reduction=reduction, **config)
        return obj


@register_keras_serializable(
    'fusionlab.nn.components', 
    name="QuantileDistributionModeling"
)
class QuantileDistributionModeling(Layer, NNLearner):
    r"""
    QuantileDistributionModeling layer projects
    deterministic outputs into quantile
    predictions [1]_.

    Depending on whether `quantiles` is specified,
    this layer:
      - Returns (B, H, O) if `quantiles` is None.
      - Returns (B, H, Q, O) otherwise, where Q
        is the number of quantiles.

    .. math::
        \mathbf{Y}_q = \text{Dense}_q(\mathbf{X}),
        \forall q \in \text{quantiles}

    Parameters
    ----------
    quantiles : list of float or str or None
        List of quantiles. If `'auto'`, defaults
        to [0.1, 0.5, 0.9]. If ``None``, no extra
        quantile dimension is added.
    output_dim : int
        Output dimension per quantile or in the
        deterministic case.

    Notes
    -----
    This layer is often used after a decoder
    to provide probabilistic forecasts via
    quantile outputs.

    Methods
    -------
    call(`inputs`, training=False)
        Projects inputs into desired quantile
        shape.
    get_config()
        Returns configuration dictionary.
    from_config(`config`)
        Instantiates from config.

    Examples
    --------
    >>> from fusionlab.nn.components import QuantileDistributionModeling
    >>> import tensorflow as tf
    >>> x = tf.random.normal((32, 10, 64))  # (B, H, O)
    >>> # Instantiate with quantiles
    >>> qdm = QuantileDistributionModeling([0.25, 0.5, 0.75], output_dim=1)
    >>> # Forward pass => (B, H, Q, O) => (32, 10, 3, 1)
    >>> y = qdm(x)

    See Also
    --------
    MultiDecoder
        Outputs multi-horizon predictions that
        can be further turned into quantiles.
    AdaptiveQuantileLoss
        Computes quantile losses for outputs
        generated by this layer.

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
        quantiles: Optional[Union[str, List[float]]],
        output_dim: int
    ):
        r"""
        Initialize the QuantileDistributionModeling
        layer.

        Parameters
        ----------
        quantiles : list of float or str or None
            If `'auto'`, defaults to [0.1, 0.5, 0.9].
            If None, returns deterministic output.
        output_dim : int
            Output dimension for each quantile or
            the deterministic case.
        """
        super().__init__()
        if quantiles == 'auto':
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.output_dim = output_dim

        # Create Dense layers if quantiles specified
        if self.quantiles is not None:
            self.output_layers = [
                Dense(output_dim) for _ in self.quantiles
            ]
        else:
            self.output_layer = Dense(output_dim)

    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        r"""
        Forward pass projecting to quantile outputs
        or deterministic outputs.

        Parameters
        ----------
        ``inputs`` : tf.Tensor
            A 3D tensor of shape (B, H, O).
        training : bool, optional
            Unused in this layer. Defaults to
            ``False``.

        Returns
        -------
        tf.Tensor
            - If `quantiles` is None:
              (B, H, O)
            - Else: (B, H, Q, O)
        """
        # No quantiles => deterministic
        if self.quantiles is None:
            return self.output_layer(inputs)

        # Quantile predictions => (B, H, Q, O)
        outputs = []
        for output_layer in self.output_layers:
            quantile_output = output_layer(inputs)
            outputs.append(quantile_output)
        return tf_stack(outputs, axis=2)

    def get_config(self):
        r"""
        Configuration dictionary for layer
        serialization.

        Returns
        -------
        dict
            Contains 'quantiles' and 'output_dim'.
        """
        config = super().get_config().copy()
        config.update({
            'quantiles': self.quantiles,
            'output_dim': self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Creates a new instance from the given
        config dict.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary with
            'quantiles' and 'output_dim'.

        Returns
        -------
        QuantileDistributionModeling
            A new instance.
        """
        return cls(**config)
