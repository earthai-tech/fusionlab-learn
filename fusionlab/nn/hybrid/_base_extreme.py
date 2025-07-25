# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""fusionlab/nn/hybrid/_base_extreme.py
Parent for XTFT-like models.

"""
from __future__ import annotations

from textwrap import dedent 
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.property import NNLearner
from ...api.docs import _shared_docs, doc 
from ...compat.sklearn import Interval, StrOptions, validate_params

from ...utils.deps_utils import ensure_pkg
from .. import KERAS_BACKEND, KERAS_DEPS, dependency_message


if KERAS_BACKEND:
    # internal utils
    from ...compat.tf import optional_tf_function 
    from .._tensor_validation import (
        validate_model_inputs,
        validate_anomaly_config,
    )
    from ..losses import (
        combined_quantile_loss,
        combined_total_loss,
        prediction_based_loss,
    )
    from ..utils import set_default_params
    from ..components import (
        Activation, 
        AnomalyLoss,
        DynamicTimeWindow,
        MultiDecoder,
        QuantileDistributionModeling,
    )
    from ..components import aggregate_time_window_output
    from ..components.utils import ( 
        configure_architecture, 
        resolve_fusion_mode
    )

Tensor = KERAS_DEPS.Tensor
Dense = KERAS_DEPS.Dense 
Model = KERAS_DEPS.Model
Concatenate = KERAS_DEPS.Concatenate
LayerNormalization = KERAS_DEPS.LayerNormalization
register_keras_serializable = KERAS_DEPS.register_keras_serializable

# tensor ops shortcuts
tf_shape = KERAS_DEPS.shape
tf_expand_dims = KERAS_DEPS.expand_dims
tf_tile = KERAS_DEPS.tile
tf_squeeze = KERAS_DEPS.squeeze
tf_zeros_like = KERAS_DEPS.zeros_like
tf_reduce_all = KERAS_DEPS.reduce_all
tf_is_nan = KERAS_DEPS.is_nan
tf_errors = KERAS_DEPS.errors
tf_unstack = KERAS_DEPS.unstack
tf_GradientTape = KERAS_DEPS.GradientTape
tf_autograph = KERAS_DEPS.autograph


DEP_MSG = dependency_message("nn.transformers")
logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

__all__ = ["BaseExtreme"]


_PARAM_SCHEMA: Dict[str, list] = {
    "static_input_dim": [Interval(Integral, 1, None, closed='left')],
    "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
    "future_input_dim": [Interval(Integral, 1, None, closed='left')],
    "embed_dim": [Interval(Integral, 1, None, closed='left')],
    "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
    "quantiles": ['array-like', StrOptions({'auto'}), None],
    "max_window_size": [Interval(Integral, 1, None, closed='left')],
    "memory_size": [Interval(Integral, 1, None, closed='left')],
    "num_heads": [Interval(Integral, 1, None, closed='left')],
    "dropout_rate": [Interval(Real, 0, 1, closed='both')],
    "output_dim": [Interval(Integral, 1, None, closed='left')],
    "attention_units": ['array-like', Interval(Integral, 1, None, closed='left')],
    "hidden_units": ['array-like', Interval(Integral, 1, None, closed='left')],
    "lstm_units": ['array-like', Interval(Integral, 1, None, closed='left'), None],
    "activation": [StrOptions({
        "elu", "relu", "tanh", "sigmoid", "linear", "gelu"
    }), callable],
    "multi_scale_agg": [StrOptions({
        "last", "average", "flatten", "auto", "sum", "concat"
    }), None],
    "scales": ['array-like', StrOptions({'auto'}), None],
    "use_batch_norm": [bool, Interval(Integral, 0, 1, closed='both')],
    "use_residuals": [bool, Interval(Integral, 0, 1, closed='both')],
    "final_agg": [StrOptions({"last", "average", "flatten"})],
    "anomaly_detection_strategy": [
        StrOptions({
            "prediction_based", "feature_based", "from_config"
        }), None
    ],
    "anomaly_loss_weight": [Real],
    "architecture_config": [dict, None], 
}

@register_keras_serializable('fusionlab.nn.hybrid', name='BaseExtreme')
@doc(
    key_parameters = dedent (_shared_docs["xtft_params_doc"]), 
)
class BaseExtreme(Model, NNLearner):

    @validate_params(_PARAM_SCHEMA)
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        *,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        embed_dim: int = 32,
        forecast_horizon: int = 1,
        quantiles: Union[str, List[float], None] = None,
        max_window_size: int = 10,
        memory_size: int = 100,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        output_dim: int = 1,
        attention_units: int = 32,
        hidden_units: int = 64,
        lstm_units: int = 64,
        scales: Union[str, List[int], None] = None,
        multi_scale_agg: Optional[str] = None,
        activation: Union[str, callable] = 'relu',
        use_residuals: bool = True,
        use_batch_norm: bool = False,
        final_agg: str = 'last',
        anomaly_config: Optional[Dict[str, Any]] = None,
        anomaly_detection_strategy: Optional[str] = None,
        anomaly_loss_weight: float = 0.1,
        architecture_config: Optional[Dict] = None,
        fusion_mode: Optional[str] = None, 
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        
        # Set up architecture configuration based on user input
        # stash the raw dict
        self.architecture_config = architecture_config or {}
        self.fusion_mode         = fusion_mode
        # pull everything into instance attributes
        self._sync_architecture()

        logger.debug("Initializing %s with params:")
        logger.debug(
            "static=%s dynamic=%s future=%s embed=%s horizon=%s",
            static_input_dim, dynamic_input_dim, future_input_dim,
            embed_dim, forecast_horizon,
        )
        logger.debug(
            "quantiles=%s scales=%s multi_scale_agg=%s",
            quantiles, scales, multi_scale_agg,
        )

        # Defaults for quantiles/scales/return_sequences etc.
        quantiles, scales, return_sequences = set_default_params(
            quantiles, scales, multi_scale_agg
        )

        # store sanitized params
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.embed_dim = embed_dim
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.max_window_size = max_window_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        self.attention_units = attention_units
        self.hidden_units = hidden_units
        self.lstm_units = lstm_units
        self.scales = scales
        self.multi_scale_agg = multi_scale_agg
        self.activation = activation if isinstance(activation, str) else activation
        self.use_residuals = use_residuals
        self.use_batch_norm = use_batch_norm
        self.final_agg = final_agg
        self.return_sequences = return_sequences
        
        self.activation_fn_str = Activation(self.activation).activation_str

        # anomaly setup
        self.anomaly_detection_strategy = anomaly_detection_strategy
        self.anomaly_loss_weight = anomaly_loss_weight
        (
            self.anomaly_config,
            self.anomaly_detection_strategy,
            self.anomaly_loss_weight,
        ) = validate_anomaly_config(
            anomaly_config=anomaly_config,
            forecast_horizon=self.forecast_horizon,
            strategy=self.anomaly_detection_strategy,
            default_anomaly_loss_weight=self.anomaly_loss_weight,
            return_loss_weight=True,
        )
        self.anomaly_scores = self.anomaly_config.get('anomaly_scores')

        logger.debug("Anomaly strategy=%s, weight=%.3f",
                     self.anomaly_detection_strategy,
                     self.anomaly_loss_weight)

        # shared heads / losses
        self.anomaly_loss_layer = AnomalyLoss(
            weight=self.anomaly_loss_weight
        )
        self.quantile_distribution_modeling = QuantileDistributionModeling(
            quantiles=self.quantiles, output_dim=self.output_dim
        )
        self.dynamic_time_window = DynamicTimeWindow(
            max_window_size=self.max_window_size
        )
        self.multi_decoder = MultiDecoder(
            output_dim=self.output_dim, num_horizons=self.forecast_horizon
        )
        
        # This layer projects the combined decoder context into a
        # consistent feature space (`attention_units`) before it's used
        # in attention mechanisms and residual connections.
        self.decoder_input_projection = Dense(
            self.attention_units,
            activation=self.activation_fn_str,
            name="decoder_input_projection"
        )
        
        # let subclasses register their components
        self._build_components()
        logger.debug("Components built for %s", self.__class__.__name__)

    # Hooks to override
    def _build_components(self) -> None:
        """Create layers/blocks. Must be implemented by subclass."""
        raise NotImplementedError

    def _encode_inputs(
        self,
        static_input: Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        *,
        training: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        """Encode inputs -> (static_feats, dyn_enc, fut_enc, cache)."""
        raise NotImplementedError

    def _temporal_backbone(
        self,
        dynamic_encoded: Tensor,
        future_encoded: Tensor,
        *,
        training: bool,
        cache: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Sequence blocks (LSTM/attention/etc.) → fused features."""
        raise NotImplementedError

    def _aggregate_decode(
        self,
        fused_feats: Tensor,
        *,
        training: bool,
        cache: Dict[str, Any],
    ) -> Tensor:
        """Default: time-window → aggregate → decoder."""
        time_window_output = self.dynamic_time_window(
            fused_feats, training=training
        )
        logger.debug(
            f"Time Window Output Shape: {time_window_output.shape}"
        )
        final_features = aggregate_time_window_output(
            time_window_output, self.final_agg
        )
        decoder_outputs = self.multi_decoder(
            final_features, training=training
        )
        return decoder_outputs

    def _maybe_compute_anomaly_scores(
        self,
        fused_feats: Tensor,
        *,
        training: bool,
        cache: Dict[str, Any],
    ) -> None:
        """Subclasses set self.anomaly_scores here if needed."""
        return None

    def _sync_architecture(self):
        """
        Read self.architecture_config and install:
          - feature selectors (`variable_selection_*` or `dense_*`)
          - which attention blocks (cross, hier, memory)
          - which GRNs to wire up, etc.
        """
        # re‑resolve through your existing util
        self.architecture_config = configure_architecture(
            objective        = self.architecture_config.get('encoder_type'),
            use_vsn          = self.architecture_config.get('use_vsn', True),
            attention_levels = self.architecture_config.get(
                                   'decoder_attention_stack', None),
            architecture_config=self.architecture_config
        )
        self.encoder_type            = self.architecture_config['encoder_type']
        self.decoder_attention_stack = (
            self.architecture_config['decoder_attention_stack']
        )
        self.feature_processing      = (
            self.architecture_config['feature_processing']
        )
        
        logger.debug("Using architecture config: %s", self.architecture_config)
        
        self.fusion_mode = resolve_fusion_mode(self.fusion_mode)
        logger.debug("Fusion mode: %s", self.fusion_mode)
 
    
    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training: bool = False, **kwargs):
        logger.debug("call() on %s, training=%s",
                     self.__class__.__name__, training)

        static_in, dyn_in, fut_in = validate_model_inputs(
            inputs=inputs,
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            forecast_horizon=self.forecast_horizon,
            mode='strict',
            model_name=self.__class__.__name__,
            verbose=1 if logger.level <= 10 else 0,
        )

        static_feats, dyn_enc, fut_enc, cache = self._encode_inputs(
            static_in, dyn_in, fut_in, training=training
        )
        logger.debug("Encoded static=%s dyn=%s fut=%s",
                     static_feats.shape, dyn_enc.shape, fut_enc.shape)

        fused_feats, cache = self._temporal_backbone(
            dyn_enc, fut_enc, training=training, cache=cache
        )
        logger.debug("Fused feats shape=%s", fused_feats.shape)

        # add static context across time dimension
        time_steps = tf_shape(dyn_enc)[1]
        static_expanded = tf_tile(tf_expand_dims(static_feats, 1),
                                  [1, time_steps, 1])
        fused_feats = Concatenate()([static_expanded, fused_feats])

        self._maybe_compute_anomaly_scores(
            fused_feats, training=training, cache=cache
        )

        dec_out = self._aggregate_decode(
            fused_feats, training=training, cache=cache
        )
        preds = self.quantile_distribution_modeling(
            dec_out, training=training
        )

        if self.anomaly_scores is not None:
            anomaly_loss = self.anomaly_loss_layer(
                self.anomaly_scores, tf_zeros_like(self.anomaly_scores)
            )
            self.add_loss(self.anomaly_loss_weight * anomaly_loss)
            logger.debug("Anomaly loss added: %s", anomaly_loss)

        if (self.quantiles is not None and self.output_dim == 1 and
                len(preds.shape) == 4):
            preds = tf_squeeze(preds, axis=-1)
            logger.debug("Squeezed final output to %s", preds.shape)

        return preds


    def compile(self, optimizer, loss=None, **kws):
        if loss is not None:
            logger.info("Using user-provided loss: %s", loss)
            return super().compile(optimizer=optimizer, loss=loss, **kws)

        if self.anomaly_detection_strategy == 'prediction_based':
            logger.info("Compile with prediction_based loss")
            pred_loss = prediction_based_loss(
                quantiles=self.quantiles,
                anomaly_loss_weight=self.anomaly_loss_weight,
            )
            return super().compile(optimizer=optimizer, loss=pred_loss, **kws)

        if self.quantiles is None:
            logger.info("Compile deterministic (MSE)")
            return super().compile(
                optimizer=optimizer, loss="mean_squared_error", **kws
            )

        q_loss = combined_quantile_loss(self.quantiles)
        if (self.anomaly_detection_strategy == 'from_config' and
                self.anomaly_scores is not None):
            logger.info("Compile quantile + anomaly total loss")
            total_loss = combined_total_loss(
                quantiles=self.quantiles,
                anomaly_layer=self.anomaly_loss_layer,
                anomaly_scores=self.anomaly_scores,
            )
            return super().compile(
                optimizer=optimizer, loss=total_loss, **kws)

        logger.info("Compile quantile loss only")
        return super().compile(optimizer=optimizer, loss=q_loss, **kws)

    @optional_tf_function
    def train_step(self, data):
        if self.anomaly_detection_strategy == 'prediction_based':
            try:
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    x, y = data[0], data[1]
                else:
                    x, y = tf_unstack(data, num=2, axis=0)
            except (ValueError, tf_errors.InvalidArgumentError):
                logger.warning(
                    "Prediction-based strategy requires (x, y) pairs. "
                    "Falling back to standard training.")
                return super().train_step(data)

            if y.shape.ndims == 0 or tf_reduce_all(tf_is_nan(y)):
                logger.warning(
                    "Invalid y_true in prediction-based. Fallback.")
                return super().train_step(data)

            with tf_GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred)

            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads,
                                               self.trainable_variables))
            self.compiled_metrics.update_state(y, y_pred)
            return {m.name: m.result() for m in self.metrics}

        return super().train_step(data)


    # (De)serialization
    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config().copy()
        cfg.update({
            'static_input_dim': int(self.static_input_dim),
            'dynamic_input_dim': int(self.dynamic_input_dim),
            'future_input_dim': int(self.future_input_dim),
            'embed_dim': int(self.embed_dim),
            'forecast_horizon': int(self.forecast_horizon),
            'quantiles': (list(self.quantiles)
                          if self.quantiles is not None else None),
            'max_window_size': int(self.max_window_size),
            'memory_size': int(self.memory_size),
            'num_heads': int(self.num_heads),
            'dropout_rate': float(self.dropout_rate),
            'output_dim': int(self.output_dim),
            'attention_units': int(self.attention_units),
            'hidden_units': int(self.hidden_units),
            'lstm_units': (int(self.lstm_units)
                           if self.lstm_units is not None else None),
            'scales': (list(self.scales)
                       if self.scales is not None else None),
            'activation': self.activation_fn_str,
            'use_residuals': bool(self.use_residuals),
            'use_batch_norm': bool(self.use_batch_norm),
            'final_agg': self.final_agg,
            'multi_scale_agg': (str(self.multi_scale_agg)
                                if self.multi_scale_agg is not None else None),
            'anomaly_config': {
                'anomaly_loss_weight': (float(self.anomaly_loss_weight)
                                        if self.anomaly_loss_weight is not None
                                        else 1.0)
            },
            'anomaly_loss_weight': self.anomaly_loss_weight,
            'anomaly_detection_strategy': self.anomaly_detection_strategy,
        })
        logger.debug("get_config called for %s", self.__class__.__name__)
        return cfg

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        if (config.get('anomaly_config') and
                config['anomaly_config'].get('anomaly_scores') is not None):
            config['anomaly_config']['anomaly_scores'] = np.array(
                config['anomaly_config']['anomaly_scores'],
                dtype=np.float32,
            )
        logger.debug("from_config building %s", cls.__name__)
        # Separate architecture_config from the main config
        arch_config = config.pop("architecture_config", None)
        # Re-add it as a keyword argument for __init__
        return cls(**config,  architecture_config=arch_config)
    

    def reconfigure(
        self,
        architecture_config: Dict[str, Any]
    ) -> "BaseExtreme":
        """Creates a new model instance with a modified architecture.
    
        This method takes the configuration of the current model, updates
        the architectural components with the provided dictionary, and
        returns a new, un-trained model instance with the specified
        changes.
    
        Parameters 
        ------------
        architecture_config (Dict[str, Any]):
            A dictionary with new architectural settings, such as
            {'encoder_type': 'transformer'}.
    
        Returns
        ----------
        BaseAttentive:
            A new model instance with the updated architecture.
        """
        # 1. Get the full configuration of the existing model
        config = self.get_config()
        
        # 2. Update the architecture configuration
        # get_config will have stored it as a nested dictionary
        config['architecture_config'].update(architecture_config)
        
        # 3. Create a new model from the modified config
        return self.__class__.from_config(config)
    

BaseExtreme.__doc__ = r"""
A reusable backbone for Extreme Temporal Fusion Transformer
variants (e.g., :class:`XTFT`, :class:`SuperXTFT`). It centralizes
parameter validation, default resolution, anomaly-handling logic,
compile/train overrides, and (de)serialization so subclasses only
implement model-specific pieces through small override hooks.

Motivation
----------
Temporal fusion architectures tend to repeat large chunks of
plumbing code: validating inputs, shaping tensors, aggregating
time windows, decoding horizons, wiring anomaly losses, etc.
`BaseExtreme` enforces *Don't Repeat Yourself* by providing:

* **Validated constructor** with a single schema.
* **Shared losses & heads** (quantile / anomaly).
* **Template hooks**: `_build_components`, `_encode_inputs`,
  `_temporal_backbone`, `_aggregate_decode`,
  `_maybe_compute_anomaly_scores`.
* **Rich logging** to trace shapes and decisions.
* **Consistent compile/train behavior** across descendants.

{key_parameters}

Additional Notes
----------------
return_sequences : bool
    Internal flag computed from scales / aggregation mode. If a
    child needs full sequences from its LSTM stack, this will be
    ``True``; otherwise ``False``. You normally do not set this
    yourself—it's derived in :func:`set_default_params`.

anomaly_detection_strategy : {'prediction_based', 'feature_based',
                              'from_config'} or None
    Strategy for integrating anomaly information. The base class
    wires losses and config parsing; subclasses may implement the
    feature extraction (see `_maybe_compute_anomaly_scores`).

Hooks Overview
--------------
_subclasses MUST implement_:

* `_build_components(self) -> None`  
  Create and assign all layers used later.

* `_encode_inputs(self, static, dyn, fut, training) -> (S, D, F, cache)`  
  Transform raw tensors into encoded representations. Return a
  cache dict for anything you want downstream.

* `_temporal_backbone(self, D, F, training, cache) -> (fused, cache)`  
  Run sequence modules (e.g., LSTMs, attentions) and fuse.

Optional overrides:

* `_aggregate_decode(self, fused, training, cache)`  
  Default uses DynamicTimeWindow → aggregate → MultiDecoder.

* `_maybe_compute_anomaly_scores(self, fused, training, cache)`  
  Populate `self.anomaly_scores` when using 'feature_based'.

Examples
--------
Basic subclass sketch::

    class MyExtreme(BaseExtreme):
        def _build_components(self):
            self.some_dense = Dense(32)

        def _encode_inputs(self, s, d, f, training=False):
            s_enc = self.some_dense(s)
            return s_enc, d, f, {}

        def _temporal_backbone(self, d, f, training=False, cache=None):
            fused = Concatenate()([d, f])
            return fused, cache

        # inherit default _aggregate_decode and anomaly noop

See Also
--------
XTFT :
    Vanilla implementation built on top of this base.

SuperXTFT :
    Variant adding Variable Selection Networks and extra GRNs.

Notes
-----
All logging uses the shared ``fusionlog`` logger. Set its level
to ``DEBUG`` to see tensor shapes and decision branches during
dev/testing.

"""
