# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Quantile / anomaly / multi-objective / CRPS losses.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np 

from ...api.property import NNLearner  
from ...utils.deps_utils import ensure_pkg 
 
from ._config import (
    Loss, Tensor, 
    Reduction, 
    register_keras_serializable,
    
    tf_abs, 
    tf_unstack, 
    tf_exp,
    tf_expand_dims, 
    tf_constant, 
    tf_float32,
    tf_reshape, tf_maximum, 
    tf_reduce_mean, tf_square, 
    tf_sqrt, 
    tf_erf,
    tf_random, 
    tf_gather, 
    tf_newaxis, 
    tf_shape, 
    tf_cast,
    tf_int32, 
    tf_range, 
    tf_sigmoid,
    tf_equal, 
    tf_cumsum, 
    KERAS_BACKEND, 
    DEP_MSG, 
    tf_autograph
)




__all__ = [
    "AdaptiveQuantileLoss",
    "AnomalyLoss",
    "MultiObjectiveLoss",
    "CRPSLoss"
]


@register_keras_serializable("fusionlab.nn.components", name="CRPSLoss")
class CRPSLoss(Loss, NNLearner):
    r"""
    Continuous Ranked Probability Score (CRPS).

    Modes supported:
      * ``quantile``  – approximate CRPS via pinball avg (Hersbach 2000).
      * ``gaussian``  – closed-form CRPS for N(μ, σ²).
      * ``mixture``   – Gaussian Mixture (or arbitrary discrete mixture)
                        via Monte-Carlo sampling.

    Auto-detection:
      - If ``y_pred`` is a dict containing ``"quantiles"`` -> quantile mode.
      - If it has ``"loc"`` & ``"scale"`` and optional ``"weights"``:
            * with weights -> mixture
            * without      -> gaussian
      - Otherwise rely on ``mode=...`` arg.

    Parameters
    ----------
    mode : {'auto','quantile','gaussian','mixture'}, default 'auto'
        Which CRPS formula to use. 'auto' tries to infer from y_pred.
    quantiles : list[float], optional
        Needed for quantile mode if not provided in y_pred dict.
    mc_samples : int, optional
        Number of Monte-Carlo samples for mixture mode.
        More samples => less variance, more compute. Default 256.
    reduction : str
        Keras loss reduction.
    name : str
        Object name.

    y_true shape   : (B, H, O) or broadcastable.
    y_pred shapes:
      * Quantile: (B, H, Q, O)  or dict{'quantiles': tensor}
      * Gaussian: dict{'loc': (B,H,O), 'scale': (B,H,O)} or tensor(...,2)
      * Mixture : dict{'loc': (B,H,K,O), 'scale': (B,H,K,O),
                       'weights': (B,H,K,1)} (weights softmaxed elsewhere)

    Notes
    -----
    * Quantile approximation:
        CRPS ~ (2/Q) * mean_q Pinball_q(y, \hat{y}_q)

    * Gaussian closed form:
        z    = (y - μ) / σ
        CRPS = σ * [ z (2Φ(z)-1) + 2φ(z) - 1/√π ]

    * Mixture CRPS (MC):
        CRPS = E|X - y| - 0.5 E|X - X'|
        where X, X' ~ predictive CDF. We approximate both expectations
        with Monte-Carlo samples.

    References
    ----------
    Hersbach, H. (2000). Decomposition of the CRPS for ensemble systems.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
            self,
        mode: str = "auto",
        quantiles: Optional[Sequence[float]] = None,
        mc_samples: int = 256,
        reduction: str = Reduction.AUTO,
        name: str = "CRPSLoss"
        ):
        super().__init__(reduction=reduction, name=name)
        self.mode = mode.lower()
        self.quantiles = list(quantiles) if quantiles else None
        self.mc_samples = int(mc_samples)

        # cache qs tensor
        if self.quantiles is not None:
            qs = tf_constant(self.quantiles, dtype=tf_float32)
            self._qs = tf_reshape(qs, [1, 1, len(self.quantiles), 1])

    @tf_autograph.experimental.do_not_convert
    def call(self, y_true: Tensor, y_pred: Any) -> Tensor:
        mode = self._infer_mode(y_pred) if self.mode == "auto" else self.mode
        if mode == "quantile":
            return self._crps_quantile(y_true, y_pred)
        if mode == "gaussian":
            return self._crps_gaussian(y_true, y_pred)
        if mode == "mixture":
            return self._crps_mixture_mc(y_true, y_pred)
        raise ValueError(f"Unsupported mode '{mode}'.")


    def _infer_mode(self, y_pred: Any) -> str:
        if isinstance(y_pred, dict):
            keys = set(y_pred.keys())
            if "quantiles" in keys:
                return "quantile"
            if {"loc", "scale", "weights"}.issubset(keys):
                return "mixture"
            if {"loc", "scale"}.issubset(keys):
                return "gaussian"
        # tensor fallback: look at last dim==2 => gaussian
        if isinstance(y_pred, Tensor):
            if tf_shape(y_pred)[-1] == 2:
                return "gaussian"
        if self.quantiles is not None:
            return "quantile"
        # Default fallback
        return "gaussian"


    def _crps_quantile(self, y_true: Tensor, y_pred: Any) -> Tensor:
        # Accept dict or bare tensor
        if isinstance(y_pred, dict):
            yq = y_pred["quantiles"]  # (B,H,Q,O)
            qs = y_pred.get("q_values", self.quantiles)
        else:
            yq = y_pred
            qs = self.quantiles
        if qs is None:
            raise ValueError("Quantile values needed for quantile CRPS.")

        qs = tf_constant(qs, tf_float32)
        qs = tf_reshape(qs, [1, 1, tf_shape(qs)[0], 1])

        y_true_exp = tf_expand_dims(y_true, axis=2)   # (B,H,1,O)
        err = y_true_exp - yq                         # (B,H,Q,O)
        pinball = tf_maximum(qs * err, (qs - 1.0) * err)
        crps = (2.0 / tf_cast(tf_shape(qs)[2], tf_float32)
                ) * tf_reduce_mean(pinball, axis=2)
        return tf_reduce_mean(crps)


    def _crps_gaussian(self, y_true: Tensor, y_pred: Any) -> Tensor:
        if isinstance(y_pred, dict):
            mu = y_pred["loc"]
            sigma = tf_abs(y_pred["scale"]) + 1e-6
        else:
            mu, sigma = tf_unstack(y_pred, axis=-1)
            sigma = tf_abs(sigma) + 1e-6

        z = (y_true - mu) / sigma
        phi = _std_normal_pdf(z)
        Phi = _std_normal_cdf(z)
        term = z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / tf_sqrt(
            tf_constant(np.pi, tf_float32)
        )
        crps = sigma * term
        return tf_reduce_mean(crps)


    def _crps_mixture_mc(self, y_true: Tensor, y_pred: Dict[str, Tensor]) -> Tensor:
        """
        Monte‑Carlo approx:
           CRPS = E|X - y| - 0.5 E|X - X'|
        X, X' ~ predictive distribution (mixture).

        Expect dict with:
            loc:    (B,H,K,O)
            scale:  (B,H,K,O)
            weights:(B,H,K,1) (sum=1 over K)
        """
        if not isinstance(y_pred, dict):
            raise ValueError("Mixture mode expects dict y_pred.")

        mu   = y_pred["loc"]    # (B,H,K,O)
        sig  = tf_abs(y_pred["scale"]) + 1e-6
        w    = y_pred["weights"]  # (B,H,K,1)
        # Normalize weights just in case
        w = w / tf_reduce_mean(tf_reduce_mean(
            w, axis=2, keepdims=True), axis=3, keepdims=True)

        B, H, K, O = [tf_shape(mu)[i] for i in range(4)]

        # --- Sample indices from categorical weights
        # We sample 'mc_samples' draws per (B,H).
        # Use cumulative weights to sample indices.
        # NOTE: TF doesn't have native categorical for batched, so
        # we do inverse-CDF sampling manually.
        cdf = tf_cumsum(w, axis=2)  # (B,H,K,1)  needs tf_math exposed
        u   = tf_random.normal([B, H, self.mc_samples, 1], mean=0.0, stddev=1.0)
        u   = tf_sigmoid(u)  # uniform-ish [0,1], quick hack (or use stateless_random_uniform)

        # broadcast cdf to compare: (B,H,K,1) vs (B,H,S,1)
        # We'll pick first index where u <= cdf
        # reshape for broadcasting
        cdf_exp = tf_expand_dims(cdf, axis=2)  # (B,H,1,K,1)
        u_exp   = tf_expand_dims(u,   axis=3)  # (B,H,S,1,1)
        mask = tf_cast(u_exp <= cdf_exp, tf_float32)  # True where cdf >= u
        # first True along K dimension:
        # cummax to find first
        # We'll compute argmax over K after applying cumulative max trick
        cum = tf_cumsum(mask, axis=3)
        first = tf_cast(tf_equal(cum, 1.0), tf_float32)
        idxs = tf_reduce_mean(tf_range(K, dtype=tf_float32)[
            tf_newaxis, tf_newaxis, tf_newaxis, :, tf_newaxis] * first, axis=3)
        idxs = tf_cast(tf_reshape(idxs, [B, H, self.mc_samples, 1]), tf_int32)  # (B,H,S,1)

        # Gather μ,σ
        gather_axis = 2  # K
        # XXX TODO:
        # We need a helper to gather along K; easiest: reshape then tf.gather with batch dims.
        # Simpler path: flatten (B*H) then gather.
        BH = B * H
        mu_r = tf_reshape(mu,   [BH, K, O])
        sig_r= tf_reshape(sig,  [BH, K, O])

        idxs_r = tf_reshape(idxs, [BH, self.mc_samples, 1])
        # gather over axis=1
        mu_s  = tf_gather(mu_r,  idxs_r, batch_dims=1)  # (BH,S,1,O)
        sig_s = tf_gather(sig_r, idxs_r, batch_dims=1)  # (BH,S,1,O)

        mu_s  = tf_reshape(mu_s,  [B, H, self.mc_samples, O])
        sig_s = tf_reshape(sig_s, [B, H, self.mc_samples, O])

        # Sample from Gaussian N(mu_s, sig_s)
        eps = tf_random.normal(tf_shape(mu_s), 0.0, 1.0, dtype=tf_float32)
        X   = mu_s + sig_s * eps  # (B,H,S,O)

        # term1 = E|X - y|
        y_exp = tf_expand_dims(y_true, axis=2)  # (B,H,1,O)
        term1 = tf_reduce_mean(tf_abs(X - y_exp), axis=2)  # (B,H,O)

        # term2 = 0.5 E|X - X'|
        # second independent sample
        eps2 = tf_random.normal(tf_shape(mu_s), 0.0, 1.0, dtype=tf_float32)
        Xp   = mu_s + sig_s * eps2  # (B,H,S,O)
        term2 = 0.5 * tf_reduce_mean(tf_abs(X - Xp), axis=2)  # (B,H,O)

        crps = term1 - term2
        return tf_reduce_mean(crps)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "mode": self.mode,
            "quantiles": self.quantiles,
            "mc_samples": self.mc_samples
        })
        return cfg

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)



@register_keras_serializable(
    'fusionlab.nn.components', 
    name="AdaptiveQuantileLoss"
)
class AdaptiveQuantileLoss(Loss, NNLearner):
    r"""
    Adaptive Quantile Loss layer that computes
    quantile loss for given quantiles [1]_.

    The layer expects ``y_true`` of shape
    :math:`(B, H, O)`, where ``B`` is batch size,
    ``H`` is horizon, and ``O`` is output dimension,
    and ``y_pred`` of shape
    :math:`(B, H, Q, O)`, where ``Q`` is the
    number of quantiles if they are specified.

    .. math::
        \text{QuantileLoss}(\hat{y}, y) =
        \max(q \cdot (y - \hat{y}),\,
        (q - 1) \cdot (y - \hat{y}))

    The final loss is the mean across batch, time,
    quantiles, and output dimension.

    Parameters
    ----------
    quantiles : list of float, optional
        A list of quantiles used to compute
        quantile loss. If set to ``'auto'``,
        defaults to [0.1, 0.5, 0.9]. If ``None``,
        the loss returns 0.0 (no quantile loss).

    Notes
    -----
    For quantile regression, each quantile
    penalizes under- and over-estimates
    differently, encouraging a robust modeling
    of the distribution of possible outcomes.

    Methods
    -------
    call(`y_true`, `y_pred`, training=False)
        Compute the quantile loss.
    get_config()
        Returns configuration for serialization.
    from_config(`config`)
        Creates a new instance from config dict.

    Examples
    --------
    >>> from fusionlab.nn.components import AdaptiveQuantileLoss
    >>> import tensorflow as tf
    >>> # Suppose y_true is (B, H, O)
    ... # y_pred is (B, H, Q, O)
    >>> y_true = tf.random.normal((32, 10, 1))
    >>> y_pred = tf.random.normal((32, 10, 3, 1))
    >>> # Instantiate with custom quantiles
    >>> aq_loss = AdaptiveQuantileLoss([0.2, 0.5, 0.8])
    >>> # Forward pass (loss calculation)
    >>> loss_value = aq_loss(y_true, y_pred)

    See Also
    --------
    MultiObjectiveLoss
        Can combine this quantile loss with an
        anomaly loss.
    AnomalyLoss
        Computes anomaly-based loss, complementary
        to quantile loss.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). "Time-series
           forecasting with deep learning: a survey."
           *Philosophical Transactions of the Royal
           Society A*, 379(2194), 20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, quantiles: Optional[List[float]], 
                 name="AdaptiveQuantileLoss"):
        super().__init__(name=name)
        if quantiles == 'auto':
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles

    @tf_autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        r"""
        Compute quantile loss.

        Parameters
        ----------
        ``y_true`` : tf.Tensor
            Ground truth of shape (B, H, O).
        ``y_pred`` : tf.Tensor
            Predicted values of shape (B, H, Q, O)
            if quantiles is not None.
        training : bool, optional
            Unused parameter, included for
            consistency. Defaults to ``False``.

        Returns
        -------
        tf.Tensor
            A scalar representing the mean quantile
            loss. 0.0 if ``quantiles`` is None.
        """
        if self.quantiles is None:
            return 0.0
        # Expand y_true to match y_pred's quantile
        # dimension
        y_true_expanded = tf_expand_dims(
            y_true,
            axis=2
        )  # => (B, H, 1, O)
        error = y_true_expanded - y_pred
        quantiles = tf_constant(
            self.quantiles,
            dtype=tf_float32
        )
        quantiles = tf_reshape(
            quantiles,
            [1, 1, len(self.quantiles), 1]
        )
        # quantile loss
        quantile_loss = tf_maximum(
            quantiles * error,
            (quantiles - 1) * error
        )
        return tf_reduce_mean(quantile_loss)

    def get_config(self):
        r"""
        Configuration for serialization.

        Returns
        -------
        dict
            Dictionary with 'quantiles'.
        """
        config = super().get_config().copy()
        config.update({'quantiles': self.quantiles})
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Creates a new instance from a config dict.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary.

        Returns
        -------
        AdaptiveQuantileLoss
            A new instance of the layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components',
    name="AnomalyLoss"
)
class AnomalyLoss(Loss, NNLearner):
    r"""
    Anomaly Loss layer computing mean squared
    anomaly scores.

    This layer expects anomaly scores of shape
    :math:`(B, H, D)` and multiplies their
    mean squared value by a weight factor.

    .. math::
        \text{AnomalyLoss}(\mathbf{a}) =
        w \cdot \frac{1}{BHD} \sum (\mathbf{a})^2

    where :math:`\mathbf{a}` is the anomaly
    score, and :math:`w` is the weight.

    Parameters
    ----------
    weight : float, optional
        Scalar multiplier for the computed
        mean squared anomaly scores.
        Defaults to 1.0.

    Notes
    -----
    Anomaly loss is often combined with other
    losses in a multi-task setting where
    predictive performance and anomaly detection
    performance are both important.

    Methods
    -------
    call(`anomaly_scores`)
        Compute mean squared anomaly loss.
    get_config()
        Return configuration for serialization.
    from_config(`config`)
        Instantiates a new instance from config.

    Examples
    --------
    >>> from fusionlab.nn.components import AnomalyLoss
    >>> import tensorflow as tf
    >>> # Suppose anomaly_scores is (B, H, D)
    >>> anomaly_scores = tf.random.normal((32, 10, 8))
    >>> # Instantiate anomaly loss
    >>> anomaly_loss_fn = AnomalyLoss(weight=2.0)
    >>> # Compute anomaly loss
    >>> loss_value = anomaly_loss_fn(anomaly_scores)

    See Also
    --------
    AdaptiveQuantileLoss
        Another specialized loss that can be
        combined for multi-objective optimization.
    MultiObjectiveLoss
        Demonstrates how anomaly and quantile loss
        can be merged.

    References
    ----------
    .. [1] Lim, B., & Zohren, S. (2021). "Time-series
           forecasting with deep learning: a survey."
           *Philosophical Transactions of the Royal
           Society A*, 379(2194), 20200209.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, weight: float = 1.0, name="AnomalyLoss"):

        super().__init__(name=name)
        self.weight = weight

    @tf_autograph.experimental.do_not_convert
    def call(self, anomaly_scores: Tensor, y_pred=None): 
        r"""
        Forward pass that computes the mean squared
        anomaly score multiplied by `weight`.

        Parameters
        ----------
        ``anomaly_scores`` : tf.Tensor
            Tensor of shape (B, H, D) representing
            anomaly scores.
        ``y_pred``: Optional 
           Does nothing, just for API consistency.

        Returns
        -------
        tf.Tensor
            A scalar loss value representing the
            weighted mean squared anomaly.
        """
        return self.weight * tf_reduce_mean(
            tf_square(anomaly_scores)
        )

    def get_config(self):
        r"""
        Return configuration dictionary for
        this layer.

        Returns
        -------
        dict
            Includes 'weight'.
        """
        config = super().get_config().copy()
        config.update({'weight': self.weight})
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Recreates an AnomalyLoss layer from a config.

        Parameters
        ----------
        ``config`` : dict
            Configuration containing 'weight'.

        Returns
        -------
        AnomalyLoss
            A new instance of this layer.
        """
        return cls(**config)


@register_keras_serializable(
    'fusionlab.nn.components', 
    name="MultiObjectiveLoss"
)
class MultiObjectiveLoss(Loss, NNLearner):
    r"""
    Multi-Objective Loss layer combining quantile
    loss and anomaly loss [1]_.

    This layer expects:
    1. ``y_true``: :math:`(B, H, O)`
    2. ``y_pred``: :math:`(B, H, Q, O)`, if
       quantiles are used (or (B, H, 1, O)
       for a single quantile).
    3. ``anomaly_scores``: :math:`(B, H, D)`,
       optional.

    .. math::
        \text{Loss} = \text{QuantileLoss} +
                      \text{AnomalyLoss}

    If ``anomaly_scores`` is None, only
    quantile loss is returned.

    Parameters
    ----------
    quantile_loss_fn : Layer
        A callable implementing quantile loss, e.g.
        :class:`AdaptiveQuantileLoss`.
    anomaly_loss_fn : Layer
        A  callable implementing anomaly loss, e.g.
        :class:`AnomalyLoss`.

    Notes
    -----
    This layer allows multi-task learning by
    combining two objectives: forecasting
    accuracy (quantile loss) and anomaly
    detection (anomaly loss).

    Methods
    -------
    call(`y_true`, `y_pred`, `anomaly_scores`=None,
         training=False)
        Compute the combined loss.
    get_config()
        Returns configuration for serialization.
    from_config(`config`)
        Rebuilds the layer from a config dict.

    Examples
    --------
    >>> from fusionlab.nn.components import (
    ...     MultiObjectiveLoss,
    ...     AdaptiveQuantileLoss,
    ...     AnomalyLoss
    ... )
    >>> import tensorflow as tf
    >>> # Suppose y_true is (B, H, O),
    ... # and y_pred is (B, H, Q, O).
    >>> y_true = tf.random.normal((32, 10, 1))
    >>> y_pred = tf.random.normal((32, 10, 3, 1))
    >>> anomaly_scores = tf.random.normal((32, 10, 8))
    >>> # Instantiate loss components
    >>> q_loss_fn = AdaptiveQuantileLoss([0.2, 0.5, 0.8])
    >>> a_loss_fn = AnomalyLoss(weight=2.0)
    >>> # Combine them
    >>> mo_loss = MultiObjectiveLoss(q_loss_fn, a_loss_fn)
    >>> # Compute the combined loss
    >>> total_loss = mo_loss(y_true, y_pred, anomaly_scores)

    See Also
    --------
    AdaptiveQuantileLoss
        Implements quantile loss to handle
        uncertainty in predictions.
    AnomalyLoss
        Computes anomaly-based MSE for
        anomaly detection tasks.

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
        quantile_loss_fn,
        anomaly_loss_fn, 
        anomaly_scores =None, 
        name="MultiObjectiveLoss"
    ):
        super().__init__(name=name)
        
        self.quantile_loss_fn = quantile_loss_fn
        self.anomaly_loss_fn = anomaly_loss_fn
        self.anomaly_scores = anomaly_scores 

    @tf_autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
             
        r"""
        Compute combined quantile and anomaly loss.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth of shape (B, H, O).
        y_pred : tf.Tensor
            Predictions of shape (B, H, Q, O)
            (or (B, H, 1, O) if Q=1).
        anomaly_scores : tf.Tensor or None, optional
            Tensor of shape (B, H, D).
            If None, anomaly loss is omitted.
        training : bool, optional
            Indicates training mode. Defaults to
            ``False``.

        Returns
        -------
        tf.Tensor
            A scalar representing the sum of
            quantile loss and anomaly loss (if
            anomaly_scores is provided).
        """
        quantile_loss = self.quantile_loss_fn(
            y_true,
            y_pred
        )
        
        if self.anomaly_scores  is not None:
            anomaly_loss = self.anomaly_loss_fn(
                # Avoid Keras signature to raise error then 
                # pass y_pred as None,
                self.anomaly_scores, y_pred =None, 
            )
            return quantile_loss + anomaly_loss
        return quantile_loss

    def get_config(self):
        r"""
        Returns configuration dictionary, including
        configs of the sub-layers.

        Returns
        -------
        dict
            Contains serialized configs of
            quantile_loss_fn and anomaly_loss_fn.
        """
        config = super().get_config().copy()
        config.update({
            'quantile_loss_fn': self.quantile_loss_fn.get_config(),
            'anomaly_loss_fn': self.anomaly_loss_fn.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        r"""
        Creates a new MultiObjectiveLoss from
        the config dictionary.

        Parameters
        ----------
        ``config`` : dict
            Configuration dictionary with sub-layer
            configs.

        Returns
        -------
        MultiObjectiveLoss
            A new instance combining quantile
            and anomaly losses.
        """
        # Rebuild sub-layers from their configs
        quantile_loss_fn = AdaptiveQuantileLoss.from_config(
            config['quantile_loss_fn']
        )
        anomaly_loss_fn = AnomalyLoss.from_config(
            config['anomaly_loss_fn']
        )
        return cls(
            quantile_loss_fn=quantile_loss_fn,
            anomaly_loss_fn=anomaly_loss_fn
        )


@register_keras_serializable(
    "fusionlab.nn.components", name="CRPSLossWrapper"
)
class CRPSLossWrapper(Loss, NNLearner):
    r"""
    Continuous Ranked Probability Score (CRPS) wrapper.

    Supports two common cases:
      1) *Quantile form*: y_pred is (B, H, Q, O) and you provide
         the list of quantiles.  We approximate CRPS by averaging
         the pinball loss over Q then multiply by 2/Q (Hersbach,
         2000).
      2) *Gaussian form*: y_pred is dict with 'loc' and 'scale'
         (or a tensor with last dim=2). An analytical closed form
         CRPS is used.

    Parameters
    ----------
    mode : {'quantile', 'gaussian'}
        Select computation style.
    quantiles : List[float], optional
        Needed if mode == 'quantile'.
    reduction : str, optional
        Keras loss reduction ('auto', 'sum', 'none').
    name : str, optional
        Keras object name.

    Notes
    -----
    *Quantile approximation*:

        CRPS  ~=  (2 / Q) * Σ_q Pinball_q(y, \hat{y}_q)

    *Gaussian closed form*:

        z = (y - μ) / σ

        CRPS = σ * [ z * (2Φ(z) - 1) + 2φ(z) - 1/√π ]

    where Φ and φ are std-normal CDF and PDF.

    See Also
    --------
    AdaptiveQuantileLoss : Pinball loss used internally.
    QuantileHead         : Typical producer of quantile outputs.

    References
    ----------
    Hersbach, H. (2000). Decomposition of the continuous ranked
    probability score for ensemble prediction systems. Wea. Forecasting.
    """

    @ensure_pkg(KERAS_BACKEND or "keras",
                extra="CRPSLossWrapper needs Keras backend.")
    def __init__(self,
                 mode: str = "quantile",
                 quantiles: Optional[List[float]] = None,
                 reduction: str = Reduction.AUTO,
                 name: str = "CRPSLossWrapper"):
        super().__init__(reduction=reduction, name=name)
        mode = mode.lower()
        if mode not in {"quantile", "gaussian"}:
            raise ValueError("mode must be 'quantile' or 'gaussian'.")
        if mode == "quantile" and not quantiles:
            raise ValueError("quantiles required when mode='quantile'.")
        self.mode = mode
        self.quantiles = quantiles

        # Pre-cache tensors for quantiles if given
        if self.mode == "quantile":
            qs = tf_constant(quantiles, dtype=tf_float32)
            self._qs = tf_reshape(qs, [1, 1, len(quantiles), 1])  # (1,1,Q,1)

    @tf_autograph.experimental.do_not_convert
    def call(self, y_true: Tensor, y_pred: Any) -> Tensor:
        if self.mode == "quantile":
            return self._crps_quantile(y_true, y_pred)
        else:  # gaussian
            return self._crps_gaussian(y_true, y_pred)

    # ------------------------ Quantile CRPS ------------------------
    def _crps_quantile(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        y_true: (B, H, O)
        y_pred: (B, H, Q, O)
        """
        # Expand true to match quantile dim
        y_true_exp = tf_expand_dims(y_true, axis=2)      # (B,H,1,O)
        err = y_true_exp - y_pred                        # (B,H,Q,O)

        # Pinball / quantile loss
        # max(q*e, (q-1)*e)
        pinball = tf_maximum(self._qs * err,
                             (self._qs - 1.0) * err)

        # Approximate CRPS: 2/Q * mean(pinball across quantiles)
        crps = (2.0 / float(len(self.quantiles))) * tf_reduce_mean(
            pinball, axis=2
        )  # (B,H,O)

        # Mean across remaining dims -> scalar
        return tf_reduce_mean(crps)

    # ------------------------- Gaussian CRPS -----------------------
    def _crps_gaussian(self, y_true: Tensor, y_pred: Any) -> Tensor:
        """
        Expect either:
          y_pred: dict {'loc': Tensor, 'scale': Tensor}
             shapes (B,H,O)
        or a tensor with last dim=2: (..., 2) -> [loc, scale]
        """
        if isinstance(y_pred, dict):
            mu = y_pred["loc"]
            sigma = tf_abs(y_pred["scale"]) + 1e-6
        else:
            # last dim = 2
            mu, sigma = tf_unstack(y_pred, axis=-1)
            sigma = tf_abs(sigma) + 1e-6

        z = (y_true - mu) / sigma

        # Standard normal pdf φ and cdf Φ
        sqrt_2pi = tf_constant(np.sqrt(2.0 * np.pi), dtype=tf_float32)
        phi = tf_exp(-0.5 * tf_square(z)) / sqrt_2pi
        Phi = 0.5 * (1.0 + tf_erf(z / tf_sqrt(tf_constant(2.0, tf_float32))))

        term = z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / tf_sqrt(
            tf_constant(np.pi, tf_float32)
        )
        crps = sigma * term

        return tf_reduce_mean(crps) 
    
    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "mode": self.mode,
            "quantiles": self.quantiles
        })
        return cfg

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
    

def _std_normal_pdf(z: Tensor) -> Tensor:
    sqrt_2pi = tf_constant(np.sqrt(2.0 * np.pi), tf_float32)
    return tf_exp(-0.5 * tf_square(z)) / sqrt_2pi

def _std_normal_cdf(z: Tensor) -> Tensor:
    # Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    return 0.5 * (1.0 + tf_erf(z / tf_sqrt(tf_constant(2.0, tf_float32))))
