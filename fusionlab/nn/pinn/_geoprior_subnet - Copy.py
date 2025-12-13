# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations
from numbers import Integral, Real 
from typing import Optional, Union, Dict, List, Tuple, Any
from collections.abc import Mapping

import numpy as np 
from ..._fusionlog import fusionlog, OncePerMessageFilter
from ...api.docs import DocstringComponents, _halnet_core_params
from ...compat.sklearn import validate_params, Interval, StrOptions 
from ...utils.deps_utils import ensure_pkg 
from ...utils.generic_utils import rename_dict_keys 
from ...params import (
    LearnableMV, LearnableKappa, FixedGammaW, FixedHRef,
    LearnableK, LearnableSs, LearnableQ, LearnableC,
    FixedC, DisabledC
)

from .. import KERAS_BACKEND, KERAS_DEPS, dependency_message 
from .._base_attentive import BaseAttentive

if KERAS_BACKEND:
    from .._tensor_validation import check_inputs, validate_model_inputs
    from .._utils import get_tensor_from 
    from .io import ( 
        _maybe_subsample, 
        default_meta_from_model, 
        save_physics_payload, 
        load_physics_payload, 
        gather_physics_payload
    ) 
    from .op import ( 
        process_pinn_inputs, default_scales, scale_residual, positive 
    )
    from .utils import  process_pde_modes, extract_txy_in, _get_coords
    from ..components import ( 
        aggregate_multiscale_on_3d, 
        aggregate_time_window_output 
    )
    
LSTM = KERAS_DEPS.LSTM
Dense = KERAS_DEPS.Dense
LayerNormalization = KERAS_DEPS.LayerNormalization 
Sequential =KERAS_DEPS.Sequential
InputLayer =KERAS_DEPS.InputLayer
Model= KERAS_DEPS.Model 
Tensor=KERAS_DEPS.Tensor
Variable =KERAS_DEPS.Variable 
Add =KERAS_DEPS.Add
Constant =KERAS_DEPS.Constant 
GradientTape =KERAS_DEPS.GradientTape 
Mean =KERAS_DEPS.Mean 
Dataset = KERAS_DEPS.Dataset

tf_zeros_like= KERAS_DEPS.zeros_like
tf_zeros =KERAS_DEPS.zeros
tf_reduce_mean =KERAS_DEPS.reduce_mean
tf_square =KERAS_DEPS.square
tf_constant =KERAS_DEPS.constant 
tf_log = KERAS_DEPS.log
tf_expand_dims = KERAS_DEPS.expand_dims
tf_tile = KERAS_DEPS.tile
tf_concat = KERAS_DEPS.concat
tf_shape = KERAS_DEPS.shape
tf_float32=KERAS_DEPS.float32
tf_exp =KERAS_DEPS.exp 
tf_rank =KERAS_DEPS.rank 
tf_assert_equal = KERAS_DEPS.assert_equal 
tf_convert_to_tensor =KERAS_DEPS.convert_to_tensor 
tf_split = KERAS_DEPS.split 
tf_sqrt = KERAS_DEPS.sqrt 
tf_stack = KERAS_DEPS.stack
tf_maximum =KERAS_DEPS.maximum 

register_keras_serializable = KERAS_DEPS.register_keras_serializable
deserialize_keras_object= KERAS_DEPS.deserialize_keras_object

tf_autograph=KERAS_DEPS.autograph
tf_autograph.set_verbosity(0)
  
DEP_MSG = dependency_message('nn.pinn.models') 
logger = fusionlog().get_fusionlab_logger(__name__)
logger.addFilter(OncePerMessageFilter())

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_halnet_core_params)
)

# model = GeoPriorSubsNet(
#     ...,
#     scale_pde_residuals=True,
#     scaling_kwargs=dict(
#         # existing stuff for default_scales(...)
#         # ...

#         # NEW: physics bounds (global, city-level ranges)
#         bounds=dict(
#             H_min=5.0,    # m    (lower bound on effective thickness)
#             H_max=80.0,   # m    (upper bound on effective thickness)
#             logK_min=np.log(1e-8),  # m/s
#             logK_max=np.log(1e-3),
#             logSs_min=np.log(1e-7), # Pa^-1
#             logSs_max=np.log(1e-3),
#         ),
#     ),
# )


__all__ = ["GeoPriorSubsNet"]

@register_keras_serializable(
    'fusionlab.nn.pinn', name="GeoPriorSubsNet") 
class GeoPriorSubsNet(BaseAttentive):
    @validate_params({
        'output_subsidence_dim': [Interval(Integral,1, None, closed="left")], 
        'output_gwl_dim': [Interval(Integral,1, None, closed="left"),], 
        "pde_mode": [
            StrOptions({'consolidation', 'gw_flow', 'both', 'none', 'on', 'off'}), 
            'array-like', None 
        ],
        "mv": [LearnableMV, Real],
        "kappa": [LearnableKappa, Real],
        "gamma_w": [FixedGammaW, Real],
        "h_ref": [FixedHRef, Real], 
        "use_effective_h": [bool],
        "hd_factor": [Interval(Real, 0, 1, closed="right")], 
        "kappa_mode": [StrOptions({"bar", "kb"})]
        
    }
   )

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)   
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_subsidence_dim: int = 1,
        output_gwl_dim: int = 1,
        embed_dim: int = 32,
        hidden_units: int = 64,
        lstm_units: int = 64,
        attention_units: int = 32,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        forecast_horizon: int = 1,
        quantiles: Optional[List[float]] = None,
        max_window_size: int = 10,
        memory_size: int = 100,
        scales: Optional[List[int]] = None,
        multi_scale_agg: str = 'last',
        final_agg: str = 'last',
        activation: str = 'relu',
        use_residuals: bool = True,
        use_batch_norm: bool = False,
        pde_mode: Union[str, List[str]] = 'both',
        mv: Union[LearnableMV, float] = LearnableMV(initial_value=1e-7),
        kappa: Union[LearnableKappa, float] = LearnableKappa(initial_value=1.0),
        gamma_w: Union[FixedGammaW, float] = FixedGammaW(value=9810.0),
        h_ref: Union[FixedHRef, float] = FixedHRef(value=0.0),
        use_effective_h: bool = False,
        hd_factor: float = 1.0 ,  # if Hd = Hd_factor * H
        kappa_mode: str = "bar",   # {"bar", "kb"}  # κ̄ vs κ_b
        use_vsn: bool = True,
        vsn_units: Optional[int] = None,
        mode: Optional[str]=None, 
        objective: Optional[str]=None, 
        attention_levels:Optional[Union[str, List[str]]]=None, 
        architecture_config: Optional[Dict] = None,
        scale_pde_residuals: bool = True,
        scaling_kwargs: Optional[Dict[str, Any]] = None,
        name: str = "GeoPriorSubsNet", 
        **kwargs
    ):
        
        self.output_subsidence_dim = output_subsidence_dim
        self.output_gwl_dim = output_gwl_dim
        self._data_output_dim = (
            self.output_subsidence_dim + self.output_gwl_dim
        )
        
        self.output_K_dim = 1      # K(x,y)
        self.output_Ss_dim = 1     # Ss(x,y)
        self.output_tau_dim = 1    # tau(x,y)
        self._phys_output_dim = (
            self.output_K_dim + self.output_Ss_dim + self.output_tau_dim
        )
        
        if 'output_dim' in kwargs: 
            kwargs.pop ('output_dim') 
            
        super().__init__(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            output_dim=self._data_output_dim, 
            forecast_horizon=forecast_horizon,
            mode=mode, 
            quantiles=quantiles,
            embed_dim=embed_dim,
            hidden_units=hidden_units,
            lstm_units=lstm_units,
            attention_units=attention_units,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            max_window_size=max_window_size,
            memory_size=memory_size,
            scales=scales,
            multi_scale_agg=multi_scale_agg,
            final_agg=final_agg,
            activation=activation,
            use_residuals=use_residuals,
            use_vsn=use_vsn,
            use_batch_norm =use_batch_norm, 
            vsn_units=vsn_units,
            attention_levels =attention_levels,
            objective=objective, 
            architecture_config=architecture_config,
            name=name,
            **kwargs
        )
        
        self.pde_modes_active = process_pde_modes(pde_mode)
        self.scale_pde_residuals = bool(scale_pde_residuals)
        self.scaling_kwargs = dict(scaling_kwargs or {})
    
        # --- Process new scalar physics params ---
        if isinstance(mv, (int, float)):
            mv = LearnableMV(initial_value=float(mv))
        if isinstance(kappa, (int, float)):
            kappa = LearnableKappa(initial_value=float(kappa))
        if isinstance(gamma_w, (int, float)):
            gamma_w = FixedGammaW(value=float(gamma_w))
        if isinstance(h_ref, (int, float)):
            h_ref = FixedHRef(value=float(h_ref))

        self.mv_config = mv
        self.kappa_config = kappa
        self.gamma_w_config = gamma_w
        self.h_ref_config = h_ref
        
        self.use_effective_thickness = use_effective_h
        self.Hd_factor = hd_factor   # if Hd = Hd_factor * H
        self.kappa_mode = kappa_mode  # {"bar", "kb"}  # κ̄ vs κ_b
        
        # Sensible defaults before compile() is called
        self.lambda_cons = 1.0
        self.lambda_gw = 1.0
        self.lambda_prior = 1.0
        self.lambda_smooth = 1.0
        self.lambda_mv = 0.0
        self._mv_lr_mult = 1.0
        self._kappa_lr_mult = 1.0
        self.lambda_bounds = 0.0   

        
        logger.info(
            f"Initialized GeoPriorSubsNet with scalar physics params:"
            f" mv_trainable={mv.trainable},"
            f" kappa_trainable={kappa.trainable}"
        )
        
        self._init_coordinate_corrections()
        self._build_pinn_components()

    def _build_attentive_layers(self):
        r"""
        Extend :class:`BaseAttentive` with physics-specific heads.
    
        This method first delegates to
        :meth:`BaseAttentive._build_attentive_layers` to construct the
        standard encoder/decoder stack (VSN blocks, multi-scale LSTMs,
        attention layers, etc.), and then adds the extra components
        required by the PINN:
    
        * a **physics head** that predicts the raw fields
          :math:`K(x,y)`, :math:`S_s(x,y)`, and :math:`\tau(x,y)`,
    
        * scalar **diagnostic metrics** for the physics residuals
          (:math:`\epsilon_\text{prior}`, :math:`\epsilon_\text{cons}`),
    
        * a placeholder for the layer thickness field ``H_field`` that
          is populated during :meth:`call`.
    
        Notes
        -----
        * The physics head operates on the 3D decoder features returned
          by :meth:`run_encoder_decoder_core`. Its outputs are further
          processed by :meth:`split_physics_predictions` (positivity and
          splitting into individual fields).
        * The two ``Mean`` metrics are updated inside
          :meth:`test_step` and :meth:`evaluate_physics`, providing
          lightweight monitoring of the physics misfit during training
          and evaluation.
        """
        # Build the standard attentive stack (encoder + decoder + VSN).
        super()._build_attentive_layers()
    
        # Physics prediction head
        # This head maps decoder features (B, H, U) to three channels:
        #   - K(x,y)
        #   - Ss(x,y)
        #   - tau(x,y)
        # stored as a single tensor, later split in split_physics_predictions.
        self.physics_mean_head = Dense(
            self._phys_output_dim,
            name="physics_mean_head",
        )

        # Runtime placeholders and scalar metrics
        # H_field is populated in call() and reused by test_step /
        # evaluate_physics / export_physics_payload.
        self.H_field: Optional[Tensor] = None
    
        # Physics diagnostics aggregated over batches. These are computed
        # in evaluate_physics() and exposed in logs during evaluation.
        self.eps_prior_metric = Mean(name="epsilon_prior")
        self.eps_cons_metric = Mean(name="epsilon_cons")
        
    
    def _init_coordinate_corrections(
        self,
        gwl_units: Union[int, None] = None,
        subs_units: Union[int, None] = None,
        hidden: Tuple[int, int] = (32, 16),
        act: str = "gelu",
    ) -> None:
        r"""
        Initialize coordinate-based correction MLPs.
    
        Each field (subsidence, GW head, :math:`K`, :math:`S_s`,
        :math:`\tau`) receives an additive correction from a small MLP
        that depends explicitly on coordinates :math:`(t, x, y)`.
    
        The rationale is twofold:
    
        1.  Guarantee a **direct differentiable path** from
            :math:`(t,x,y)` to all physics outputs, so that PDE
            residuals have non-zero gradients (avoiding ``None``
            gradients and NaNs in physics losses).
        2.  Allow flexible, spatially-varying corrections on top of the
            base decoder features, improving expressiveness in complex
            urban settings.
    
        Parameters
        ----------
        gwl_units : int or None, default=None
            Output width for the groundwater-head correction branch.
            Defaults to :attr:`output_gwl_dim`.
    
        subs_units : int or None, default=None
            Output width for the subsidence correction branch.
            Defaults to :attr:`output_subsidence_dim`.
    
        hidden : (int, int), default=(32, 16)
            Hidden layer sizes used for all coordinate MLPs. Each branch
            has the structure ``[Dense(hidden[0]), Dense(hidden[1]),
            Dense(out_units)]``.
    
        act : str, default="gelu"
            Activation function for the hidden layers (shared across all
            branches).
    
        Notes
        -----
        * All branches take an input of shape ``(B, H, 3)`` corresponding
          to :math:`(t, x, y)` (or a reshaped equivalent) and return a
          tensor with the same spatial shape and field-specific channel
          dimension.
        * Corrections are applied on top of the physics head outputs and 
        then passed through positive(·) again to keep 
        :math:`K`, :math:`S_s`, and :math:`\tau` 
        strictly positive..
        """
        gwl_units = gwl_units or self.output_gwl_dim
        subs_units = subs_units or self.output_subsidence_dim
    
        def _branch(out_units: int, name: str) -> Sequential:
            """
            Small helper to create a (t, x, y) -> field-correction MLP.
    
            Input shape is (None, 3), i.e. a per-time-step coordinate
            vector. Keras will treat the leading dimension as time/space
            when used in a time-distributed manner.
            """
            return Sequential(
                [
                    InputLayer(input_shape=(None, 3)),
                    Dense(hidden[0], activation=act),
                    Dense(hidden[1], activation=act),
                    Dense(out_units),
                ],
                name=name,
            )
    
        # Coordinate-based correction for groundwater head
        self.coord_mlp = _branch(gwl_units, "coord_mlp")
    
        # Coordinate-based correction for subsidence
        self.subs_coord_mlp = _branch(subs_units, "subs_coord_mlp")
    
        # Coordinate-based corrections for physics fields K, Ss, tau
        self.K_coord_mlp = _branch(self.output_K_dim, "K_coord_mlp")
        self.Ss_coord_mlp = _branch(self.output_Ss_dim, "Ss_coord_mlp")
        self.tau_coord_mlp = _branch(self.output_tau_dim, "tau_coord_mlp")

    def _build_pinn_components(self):
        r"""
        Instantiate trainable and fixed scalar physics coefficients.
    
        This method builds the scalar parameters used in the PDE
        residuals and geomechanical priors, namely:
    
        * :math:`m_v` (compressibility),
        * :math:`\kappa` (consistency factor),
        * :math:`\gamma_w` (unit weight of water),
        * :math:`h_\text{ref}` (reference hydraulic head).
    
        The first two may be **learnable** (wrapped in
        :class:`LearnableMV` / :class:`LearnableKappa`) or fixed,
        depending on how the model was configured. The scalar values
        are stored in log-space to enforce positivity.
    
        Notes
        -----
        * If ``mv_config`` or ``kappa_config`` is an instance of the
          corresponding ``Learnable*`` wrapper, we create a trainable
          weight ``log_mv`` / ``log_kappa`` which is later exponentiated
          via :meth:`_mv_value` and :meth:`_kappa_value`.
        * If they are fixed wrappers, we create constant tensors
          ``_mv_fixed`` / ``_kappa_fixed`` instead.
        * The raw physics fields ``K_field``, ``Ss_field`` and
          ``tau_field`` are initialized to ``None`` and are populated
          after calls to :meth:`split_physics_predictions` during the
          forward pass.
        """
    
        # Compressibility m_v
        if isinstance(self.mv_config, LearnableMV):
            # Trainable log-parameter for m_v
            self.log_mv = self.add_weight(
                name="log_param_mv",
                shape=(),
                initializer=Constant(
                    tf_log(self.mv_config.initial_value)
                ),
                trainable=self.mv_config.trainable,
            )
        else:
            # Fixed scalar m_v as a constant
            self._mv_fixed = tf_constant(
                float(self.mv_config.initial_value),
                dtype=tf_float32,
            )
    
    
        # Consistency factor κ
        if isinstance(self.kappa_config, LearnableKappa):
            # Trainable log-parameter for κ
            self.log_kappa = self.add_weight(
                name="log_param_kappa",
                shape=(),
                initializer=Constant(
                    tf_log(self.kappa_config.initial_value)
                ),
                trainable=self.kappa_config.trainable,
            )
        else:
            # Fixed scalar κ as a constant
            self._kappa_fixed = tf_constant(
                float(self.kappa_config.initial_value),
                dtype=tf_float32,
            )
    
        # Fixed physical constants (water unit weight, reference head)
        self.gamma_w = tf_constant(
            float(self.gamma_w_config.value),
            dtype=tf_float32,
        )
        self.h_ref = tf_constant(
            float(self.h_ref_config.value),
            dtype=tf_float32,
        )
    
        # Runtime placeholders for physics fields
        # These are filled in train_step / evaluate_physics()
        # *after* coordinate corrections and positivity have been applied,
        # and can be inspected via get_last_physics_fields().
        self.K_field = None
        self.Ss_field = None
        self.tau_field = None


    @tf_autograph.experimental.do_not_convert
    def run_encoder_decoder_core(
        self,
        static_input: Tensor,
        dynamic_input: Tensor,
        future_input: Tensor,
        coords_input: Tensor,
        training: bool,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Core encoder–decoder pass with explicit coordinate injection.
    
        This method overrides
        :meth:`BaseAttentive.run_encoder_decoder_core` to insert the
        coordinate tensor :math:`(t, x, y)` into the decoder path,
        ensuring that all downstream data and physics outputs have a
        differentiable dependence on space and time.
    
        Parameters
        ----------
        static_input : tf.Tensor
            Static covariates with shape ``(B, S)`` (or ``(B, S')`` after
            preprocessing). May be zero-width if no static features
            were configured.
    
        dynamic_input : tf.Tensor
            Dynamic covariates with shape ``(B, T, D)`` where ``T`` is
            the number of encoder time steps and ``D`` is the number of
            dynamic features.
    
        future_input : tf.Tensor
            Future covariates for the decoder, with shape dependent on
            the mode:
    
            * ``"tft_like"``: typically ``(B, T + H, D_f)``; the first
              ``T`` steps may be used by the encoder, the last ``H`` by
              the decoder.
            * ``"pihal_like"``: typically ``(B, H, D_f)``; only the
              forecast horizon is used in the decoder.
            * may be zero-width or ``None`` if no future covariates are
              present.
    
        coords_input : tf.Tensor
            Coordinate tensor to inject into the decoder, with shape
            ``(B, H, 3)`` corresponding to ``(t, x, y)`` across the
            forecast horizon. This is critical for PDE-based physics:
            if this tensor is missing or not connected properly, the
            PDE gradients w.r.t. :math:`(t, x, y)` will be zero or
            ``None``.
    
        training : bool
            Standard Keras training flag for controlling dropout, batch
            normalization, etc.
    
        Returns
        -------
        data_features_2d : tf.Tensor
            Time-aggregated decoder features for the **data path** with
            shape ``(B, U_data)``. These are fed into
            :attr:`multi_decoder` to produce subsidence and GW head.
    
        phys_features_raw_3d : tf.Tensor
            Time-distributed decoder features for the **physics path**
            with shape ``(B, H, U_phys)``. These go into
            :attr:`physics_mean_head` to yield raw
            :math:`K, S_s, \tau` fields.
    
        Notes
        -----
        * Most of the logic mirrors :class:`BaseAttentive`:
    
          - VSN / dense preprocessing for each feature block,
          - hybrid LSTM / transformer encoder,
          - multi-level attention in the decoder.
    
          The main modification is the **coordinate injection** into
          the ``decoder_parts`` list.
    
        * If ``coords_input`` is ``None``, a ``ValueError`` is raised
          immediately, since the PINN cannot function without a valid
          coordinate path.
        """
        # ------------------------------------------------------------------
        # 0. Basic time dimension inference
        # ------------------------------------------------------------------
        time_steps = tf_shape(dynamic_input)[1]
    
        # ------------------------------------------------------------------
        # 1. Initial feature processing (VSN or dense path)
        # ------------------------------------------------------------------
        static_context, dyn_proc, fut_proc = (
            None,
            dynamic_input,
            future_input,
        )
    
        if self.architecture_config.get("feature_processing") == "vsn":
            # Static VSN path
            if self.static_vsn is not None:
                vsn_static_out = self.static_vsn(
                    static_input,
                    training=training,
                )
                static_context = self.static_vsn_grn(
                    vsn_static_out,
                    training=training,
                )
    
            # Dynamic VSN path
            if self.dynamic_vsn is not None:
                dyn_context = self.dynamic_vsn(
                    dynamic_input,
                    training=training,
                )
                dyn_proc = self.dynamic_vsn_grn(
                    dyn_context,
                    training=training,
                )
    
            # Future VSN path
            if self.future_vsn is not None:
                fut_context = self.future_vsn(
                    future_input,
                    training=training,
                )
                fut_proc = self.future_vsn_grn(
                    fut_context,
                    training=training,
                )
        else:
            # Non-VSN dense preprocessing path
            if self.static_dense is not None:
                processed_static = self.static_dense(static_input)
                static_context = self.grn_static_non_vsn(
                    processed_static,
                    training=training,
                )
            if self.dynamic_dense is not None:
                dyn_proc = self.dynamic_dense(dynamic_input)
            if self.future_dense is not None:
                fut_proc = self.future_dense(future_input)
    
        logger.debug(
            "Shape after VSN/initial processing: "
            f"Dynamic={getattr(dyn_proc, 'shape', 'N/A')}, "
            f"Future={getattr(fut_proc, 'shape', 'N/A')}"
        )
    
        # ------------------------------------------------------------------
        # 2. Encoder path (hybrid LSTM/Transformer)
        # ------------------------------------------------------------------
        encoder_input_parts = [dyn_proc]
    
        if self._mode == "tft_like" and self.future_input_dim > 0:
            # For TFT-like mode, the first T steps of future covariates
            # are concatenated with dynamic features in the encoder.
            fut_enc_proc = fut_proc[:, :time_steps, :]
            encoder_input_parts.append(fut_enc_proc)
    
        encoder_raw = tf_concat(encoder_input_parts, axis=-1)
        encoder_input = self.encoder_positional_encoding(encoder_raw)
    
        if self.architecture_config["encoder_type"] == "hybrid":
            # Multi-scale LSTM encoder followed by multiscale aggregation
            lstm_out = self.multi_scale_lstm(
                encoder_input,
                training=training,
            )
            encoder_sequences = aggregate_multiscale_on_3d(
                lstm_out,
                mode="concat",
            )
        else:
            # Pure transformer encoder
            encoder_sequences = encoder_input
            for mha, norm in self.encoder_self_attention:
                attn_out = mha(
                    encoder_sequences,
                    encoder_sequences,
                )
                encoder_sequences = norm(
                    encoder_sequences + attn_out
                )
    
        # Optional dynamic time windowing (DTW)
        if self.apply_dtw and self.dynamic_time_window is not None:
            encoder_sequences = self.dynamic_time_window(
                encoder_sequences,
                training=training,
            )
    
        logger.debug(
            f"Encoder sequences shape: {encoder_sequences.shape}"
        )
    
        # ------------------------------------------------------------------
        # 3. Decoder path (modified to inject coords_input)
        # ------------------------------------------------------------------
        if self._mode == "tft_like" and self.future_input_dim > 0:
            # TFT-like: remaining steps go to decoder
            fut_dec_proc = fut_proc[:, time_steps:, :]
        elif self.future_input_dim > 0:
            # PIHAL-like: decoder sees all future covariates over horizon
            fut_dec_proc = fut_proc
        else:
            fut_dec_proc = None
    
        decoder_parts = []
    
        # Broadcast static context to all horizon steps
        if static_context is not None:
            static_expanded = tf_expand_dims(static_context, 1)
            static_expanded = tf_tile(
                static_expanded,
                [1, self.forecast_horizon, 1],
            )
            decoder_parts.append(static_expanded)
    
        # Decoder future features with positional encoding
        if fut_dec_proc is not None:
            future_with_pos = self.decoder_positional_encoding(
                fut_dec_proc
            )
            decoder_parts.append(future_with_pos)
    
        # Coordinate injection: this is the crucial (t, x, y) signal
        if coords_input is None:
            raise ValueError(
                "GeoPriorSubsNet.run_encoder_decoder_core requires "
                "'coords_input' (B, H, 3) to be provided."
            )
        decoder_parts.append(coords_input)
    
        # If everything is missing (very degenerate case), fall back to
        # a zero tensor so shapes remain valid.
        if not decoder_parts:
            batch_size = tf_shape(dynamic_input)[0]
            raw_decoder_input = tf_zeros(
                (batch_size, self.forecast_horizon, self.attention_units)
            )
        else:
            raw_decoder_input = tf_concat(decoder_parts, axis=-1)
    
        projected_decoder_input = self.decoder_input_projection(
            raw_decoder_input
        )
        logger.debug(
            "Projected decoder input shape: "
            f"{projected_decoder_input.shape}"
        )
    
        # ------------------------------------------------------------------
        # 4. Apply decoder attention levels and aggregate
        # ------------------------------------------------------------------
        # final_features is the 3D tensor (B, H, U) that both data and
        # physics paths will consume.
        final_features = self.apply_attention_levels(
            projected_decoder_input,
            encoder_sequences,
            training=training,
        )
    
        logger.debug(
            f"Shape after final fusion: {final_features.shape}"
        )
    
        # 3D features for physics head
        phys_features_raw_3d = final_features
    
        # Time-aggregated 2D features for data decoder
        data_features_2d = aggregate_time_window_output(
            final_features,
            self.final_agg,
        )
    
        return data_features_2d, phys_features_raw_3d


    @tf_autograph.experimental.do_not_convert
    def call(
        self,
        inputs: Dict[str, Optional[Tensor]],
        training: bool = False,
    ) -> Dict[str, Tensor]:
        r"""
        Forward pass separating data and physics paths.
    
        This override of :meth:`tf.keras.Model.call` does three things:
    
        1.  Unpacks and validates all PINN inputs via
            :func:`process_pinn_inputs`, including coordinates and
            the geomechanical thickness field ``H_field``.
        2.  Constructs a coordinate tensor
            :math:`(t, x, y)` with shape ``(B, H, 3)`` that is injected
            into the decoder path of :class:`BaseAttentive` through
            :meth:`run_encoder_decoder_core`. This guarantees that all
            physics heads depend on space and time.
        3.  Produces separate outputs for:
    
            * the **data path** (subsidence and hydraulic head),
            * the **physics path** (raw :math:`K, S_s, \tau` fields).
    
        Parameters
        ----------
        inputs : dict
            Mapping of input tensors. The exact set of required keys is
            handled by :func:`process_pinn_inputs`, but typically includes:
    
            * ``"coords"`` or equivalent components for time and space,
            * ``"H_field"`` (or ``"soil_thickness"``) for layer thickness,
            * ``"static_features"`` (optional),
            * ``"dynamic_features"`` (required),
            * ``"future_features"`` (optional, depending on mode).
    
            All tensors are expected to be batched with a leading
            dimension ``B``. Time and horizon dimensions must be
            consistent with the configuration passed at initialization
            and enforced via :func:`check_inputs` and
            :func:`validate_model_inputs`.
    
        training : bool, default=False
            Standard Keras ``training`` flag. Controls dropout, batch
            normalization, etc., in both the data encoder/decoder and
            the physics heads.
    
        Returns
        -------
        dict
            Dictionary with three entries:
    
            * ``"data_final"`` : Tensor with the **final data outputs**,
              i.e. subsidence and head after optional quantile expansion.
              Shape typically ``(B, H, Q, D_data)`` or similar.
            * ``"data_mean"`` : Tensor with the **mean data outputs**
              before quantile modeling. Shape typically ``(B, H, D_data)``.
            * ``"phys_mean_raw"`` : Tensor with raw physics features
              (pre-positivity) for :math:`K`, :math:`S_s`, :math:`\tau`,
              with shape ``(B, H, D_phys)`` where ``D_phys = 3`` here.
    
        Notes
        -----
        * The field :attr:`self.H_field` is updated on every call from
          the unpacked ``H_field`` input. This is later used by
          :meth:`evaluate_physics` and the diagnostics helpers.
        * Coordinate injection is **central** to the PINN behaviour.
          The tensor ``coords_for_decoder = concat([t, x, y], axis=-1)``
          is passed to :meth:`run_encoder_decoder_core`, ensuring that
          all physics fields (:math:`K, S_s, \tau`) and data outputs rely
          on :math:`(t, x, y)`. If this path is broken, PDE gradients
          may become ``None`` and will be caught in :meth:`train_step`.
        """
        # ------------------------------------------------------------------
        # 1. Unpack and validate all PINN inputs
        # ------------------------------------------------------------------
        # process_pinn_inputs standardizes the interface and returns
        # tensors with consistent shapes suitable for the encoder/decoder.
        (
            t,
            x,
            y,
            H_field,
            static_features,
            dynamic_features,
            future_features,
        ) = process_pinn_inputs(
            inputs,
            mode="auto",
            model_name="geoprior",
        )
    
        # Build coordinate tensor (B, H, 3) used by the decoder and
        # physics head. This is the explicit (t,x,y) signal that the
        # PINN can differentiate with respect to.
        coords_for_decoder = tf_concat([t, x, y], axis=-1)
    
        # Lightweight shape checks to catch glaring mistakes early:
        #   - consistent feature dims,
        #   - correct forecast horizon length, etc.
        check_inputs(
            dynamic_inputs=dynamic_features,
            static_inputs=static_features,
            future_inputs=future_features,
            dynamic_input_dim=self.dynamic_input_dim,
            static_input_dim=self.static_input_dim,
            future_input_dim=self.future_input_dim,
            forecast_horizon=self.forecast_horizon,
            verbose=0,  # set >0 to debug shape mismatches
        )
        logger.debug(
            "Input shapes after validation:"
            f" S={getattr(static_features, 'shape', 'None')}, "
            f"D={getattr(dynamic_features, 'shape', 'None')}, "
            f"F={getattr(future_features, 'shape', 'None')}"
        )
    
        # Store H_field on the instance so that downstream methods
        # (e.g. evaluate_physics, diagnostic payload exports) can
        # access the thickness field used in this forward pass.
        self.H_field = H_field
    
        # Further strict validation for the encoder/decoder,
        # handling possible 0-width static/future features etc.
        static_p, dynamic_p, future_p = validate_model_inputs(
            inputs=[static_features, dynamic_features, future_features],
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            future_covariate_dim=self.future_input_dim,
            mode="strict",
            verbose=0,
        )
    
        # ------------------------------------------------------------------
        # 2. Run the shared encoder/decoder core
        # ------------------------------------------------------------------
        # The overridden core returns:
        #   - data_features_2d: (B, U_data)   for data decoder
        #   - phys_features_raw_3d: (B, H, U) for physics head
        data_features_2d, phys_features_raw_3d = self.run_encoder_decoder_core(
            static_input=static_p,
            dynamic_input=dynamic_p,
            future_input=future_p,
            coords_input=coords_for_decoder,
            training=training,
        )
    
        # ------------------------------------------------------------------
        # 3. Data path: decode subsidence and head
        # ------------------------------------------------------------------
        decoded_data_means = self.multi_decoder(
            data_features_2d,
            training=training,
        )
        final_data_predictions = decoded_data_means
        if self.quantiles is not None:
            # Expand means into full predictive distributions
            final_data_predictions = self.quantile_distribution_modeling(
                decoded_data_means,
                training=training,
            )
    
        # ------------------------------------------------------------------
        # 4. Physics path: decode K, Ss, tau (pre-positivity)
        # ------------------------------------------------------------------
        decoded_physics_means_raw = self.physics_mean_head(
            phys_features_raw_3d,
            training=training,
        )
    
        # ------------------------------------------------------------------
        # 5. Return all components for downstream use
        # ------------------------------------------------------------------
        # Note: positivity and splitting into individual fields is done
        # by split_physics_predictions(...) inside train_step /
        # evaluate_physics.
        return {
            "data_final": final_data_predictions,
            "data_mean": decoded_data_means,
            "phys_mean_raw": decoded_physics_means_raw,
        }

    def train_step(self, data):
        r"""
        Single optimization step uniting data loss and physics-informed terms.
    
        This method implements the full loss function described in the
        revised manuscript:
    
        .. math::
    
            L_\text{total}
            &= L_\text{data}
             + \lambda_\text{cons} L_\text{cons}
             + \lambda_\text{gw}   L_\text{gw}
             + \lambda_\text{prior} L_\text{prior}
             + \lambda_\text{smooth} L_\text{smooth}
             + \lambda_{m_v} L_{m_v},
    
        where
    
        * :math:`L_\text{data}` is the supervised loss on subsidence and
          head predictions (quantile or mean, depending on the compiled
          loss),
        * :math:`L_\text{gw}` is based on the groundwater-flow residual
          :math:`R_\text{gw}`,
        * :math:`L_\text{cons}` is based on the consolidation residual
          :math:`R_\text{cons}`,
        * :math:`L_\text{prior}` enforces consistency between
          :math:`\tau`, :math:`K`, :math:`S_s` and :math:`H`,
        * :math:`L_\text{smooth}` discourages spatial roughness in
          :math:`K` and :math:`S_s`,
        * :math:`L_{m_v}` is the storage identity penalty linking
          :math:`S_s` and :math:`m_v`.
    
        Parameters
        ----------
        data : tuple
            Tuple ``(inputs, targets)`` as provided by a Keras
            ``tf.data.Dataset``:
    
            * ``inputs`` : mapping accepted by :meth:`call`, and
              **must** include ``"H_field"`` or ``"soil_thickness"`` for
              the consolidation and prior terms.
            * ``targets`` : either
    
              - a dict with keys ``"subsidence"`` and ``"gwl"`` (mapped
                internally to ``"subs_pred"`` / ``"gwl_pred"``), or
              - directly a mapping compatible with the compiled loss.
    
        Returns
        -------
        dict
            Dictionary of scalar training metrics including, at minimum:
    
            * ``"total_loss"`` : full loss including physics terms,
            * ``"data_loss"`` : supervised loss only,
            * ``"physics_loss"`` : weighted sum of physics components,
            * individual physics components:
              ``"consolidation_loss"``, ``"gw_flow_loss"``,
              ``"prior_loss"``, ``"smooth_loss"``, ``"mv_prior_loss"``,
            * all metrics registered in :attr:`self.metrics`.
    
        Notes
        -----
        * All PDE-related derivatives are computed via
          :class:`tf.GradientTape` with explicit ``tape.watch`` calls on
          :math:`t, x, y` **and** the corrected physics fields
          (:math:`K`, :math:`S_s`, :math:`\tau`). If any of those
          gradients are ``None``, a ``ValueError`` is raised with the
          missing keys. This is a crucial guard to prevent silent NaNs.
        * Physics can be fully disabled by including ``"none"`` in
          :attr:`pde_modes_active`. In that case, all residuals are
          forced to zero and the physics loss weights are zeroed in
          :meth:`compile`.
        * The gradients for the scalar physics parameters :math:`m_v` and
          :math:`\kappa` are scaled via :meth:`_scale_param_grads`, which
          applies per-parameter learning-rate multipliers.
        """
        # -----------------------------------------------------------------
        # 0. Unpack and normalize targets
        # -----------------------------------------------------------------
        inputs, targets = data
    
        # Accept targets as {"subsidence", "gwl"} and normalize to the
        # supervised mapping used in compiled_loss.
        if isinstance(targets, dict):
            targets = rename_dict_keys(
                targets.copy(),
                param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"},
            )
    
        # -----------------------------------------------------------------
        # 1. Fetch H_field (required for consolidation & prior)
        # -----------------------------------------------------------------
        # For consolidation and the consistency prior we need an
        # effective layer thickness field H(x, y). The model supports
        # either an explicit "H_field" input or a fallback "soil_thickness".
        H_field_input = get_tensor_from(
            inputs,
            "H_field",
            "soil_thickness",
            auto_convert=True,
        )
    
        if H_field_input is None:
            raise ValueError(
                "Input dictionary must contain 'H_field' or 'soil_thickness' "
                "for train_step. Physics terms (R_cons, R_prior) require a "
                "layer thickness field."
            )
        H_field = tf_convert_to_tensor(H_field_input, dtype=tf_float32)
    
        # -----------------------------------------------------------------
        # 2. GradientTape: forward pass + data loss
        # -----------------------------------------------------------------
        with GradientTape(persistent=True) as tape:
            # 2.1 Extract coordinates and watch them for AD
            coords = _get_coords(inputs)
            t, x, y = extract_txy_in(coords)
            tape.watch([t, x, y])
    
            # 2.2 Full forward pass through the network
            # This call:
            #   - runs the encoder/decoder core,
            #   - injects coordinates into the decoder,
            #   - produces data + physics outputs,
            #   - stores H_field internally.
            outputs_dict = self(inputs, training=True)
    
            # 2.3 Split data predictions (quantile head) for supervised loss
            s_pred_final, gwl_pred_final = self.split_data_predictions(
                outputs_dict["data_final"]
            )
            y_pred_for_loss = {
                "subs_pred": s_pred_final,
                "gwl_pred": gwl_pred_final,
            }
    
            data_loss = self.compiled_loss(
                y_true=targets,
                y_pred=y_pred_for_loss,
                regularization_losses=self.losses,
            )
    
            # -----------------------------------------------------------------
            # 3. PDE preparation: physics fields + coordinate corrections
            # -----------------------------------------------------------------
            (
                s_pred_mean,
                gwl_pred_mean,
                K_field_base,
                Ss_field_base,
                tau_field_base,
            ) = self.split_physics_predictions(outputs_dict)
    
            # Flatten (t, x, y) for the small coordinate MLPs
            coords_flat = tf_concat([t, x, y], axis=-1)
    
            # Apply coordinate-based corrections to s, h, K, Ss, tau.
            # This allows the physics heads to absorb small systematic
            # biases in space/time before positivity.
            mlp_corr = self.coord_mlp(coords_flat, training=True)
            s_corr = self.subs_coord_mlp(coords_flat, training=True)
    
            h_pred_mean_corr = gwl_pred_mean + mlp_corr
            s_pred_mean_corr = s_pred_mean + s_corr
    
            K_corr = self.K_coord_mlp(coords_flat, training=True)
            Ss_corr = self.Ss_coord_mlp(coords_flat, training=True)
            tau_corr = self.tau_coord_mlp(coords_flat, training=True)
    
            # Apply corrections then enforce positivity
            K_field = positive(K_field_base + K_corr)
            Ss_field = positive(Ss_field_base + Ss_corr)
            tau_field = positive(tau_field_base + tau_corr)
            
            # Save latest physical fields for diagnostics / export
            self.K_field = K_field
            self.Ss_field = Ss_field
            self.tau_field = tau_field

            # 3.1 Ensure AD tracks dependence of outputs on coords/fields
            tape.watch(
                [
                    s_pred_mean_corr,
                    h_pred_mean_corr,
                    K_field,
                    Ss_field,
                    tau_field,
                ]
            )
    
            # -----------------------------------------------------------------
            # 4. PDE derivatives via automatic differentiation
            # -----------------------------------------------------------------
            # ds/dt for consolidation residual
            ds_dt = tape.gradient(s_pred_mean_corr, t)
    
            # dh/dt and spatial derivatives for groundwater residual
            dh_dt = tape.gradient(h_pred_mean_corr, t)
            dh_dx = tape.gradient(h_pred_mean_corr, x)
            dh_dy = tape.gradient(h_pred_mean_corr, y)
    
            K_dh_dx = K_field * dh_dx
            K_dh_dy = K_field * dh_dy
    
            d_K_dh_dx_dx = tape.gradient(K_dh_dx, x)
            d_K_dh_dy_dy = tape.gradient(K_dh_dy, y)
    
            # Spatial gradients of K and Ss for smoothness prior
            dK_dx = tape.gradient(K_field, x)
            dK_dy = tape.gradient(K_field, y)
            dSs_dx = tape.gradient(Ss_field, x)
            dSs_dy = tape.gradient(Ss_field, y)
    
            # Guardrail: if any critical derivative is None, fail fast.
            derivs = {
                "ds_dt": ds_dt,
                "dh_dt": dh_dt,
                "d_K_dh_dx_dx": d_K_dh_dx_dx,
                "d_K_dh_dy_dy": d_K_dh_dy_dy,
                "dK_dx": dK_dx,
                "dK_dy": dK_dy,
                "dSs_dx": dSs_dx,
                "dSs_dy": dSs_dy,
            }
            if any(v is None for v in derivs.values()):
                none_keys = [k for k, v in derivs.items() if v is None]
                raise ValueError(
                    "One or more PDE gradients are None: "
                    f"{none_keys}. "
                    "Check that (t, x, y) influence all 5 model outputs "
                    "(s, h, K, Ss, tau) and that coordinate corrections "
                    "are applied correctly."
                )
    
            # -----------------------------------------------------------------
            # 5. Residuals for each physics component
            # -----------------------------------------------------------------
            # Groundwater-flow residual:
            #   R_gw = Ss*dh/dt - div(K*grad(h)) - Q
            # In the revised manuscript Q=0 (no explicit sources/sinks).
            gw_res = self._compute_gw_flow_residual(
                dh_dt,
                d_K_dh_dx_dx,
                d_K_dh_dy_dy,
                Ss_field,
                Q=0.0,
            )
    
            # Consolidation residual:
            #   R_cons = ds/dt - (s_eq - s)/tau
            cons_res = self._compute_consolidation_residual(
                ds_dt,
                s_pred_mean_corr,
                h_pred_mean_corr,
                H_field,
                tau_field,
            )
    
            # Consistency prior residual:
            #   R_prior = log(tau) - log(tau_phys)
            prior_res = self._compute_consistency_prior(
                K_field,
                Ss_field,
                tau_field,
                H_field,
            )
    
            # Smoothness prior:
            #   R_smooth = ||grad(K)||^2 + ||grad(Ss)||^2
            smooth_res = self._compute_smoothness_prior(
                dK_dx,
                dK_dy,
                dSs_dx,
                dSs_dy,
            )
    
            # Storage identity residual:
            #   R_mv = log(Ss) - log(m_v * gamma_w)
            mv_prior_res = self._compute_mv_prior(Ss_field)
            loss_mv = tf_reduce_mean(tf_square(mv_prior_res))
    
            # NEW: soft bounds on H, log K, log S_s
            bounds_res = self._compute_bounds_residual(
                K_field,
                Ss_field,
                H_field,
            )
            loss_bounds = tf_reduce_mean(tf_square(bounds_res))
        
            # If physics is disabled, force all residuals to zero to
            # avoid spurious gradients / NaNs.
            if self._physics_off():
                cons_res = tf_zeros_like(cons_res)
                gw_res = tf_zeros_like(gw_res)
                prior_res = tf_zeros_like(prior_res)
                smooth_res = tf_zeros_like(smooth_res)
                loss_mv = tf_zeros_like(loss_mv)
                bounds_res = tf_zeros_like(bounds_res)
                loss_bounds = tf_zeros_like(loss_bounds)
    
            # -----------------------------------------------------------------
            # 6. Residual scaling (dimensionless normalization)
            # -----------------------------------------------------------------
            if (not self._physics_off()) and self.scale_pde_residuals:
                scales = self._compute_scales(
                    t,
                    s_pred_mean_corr,
                    h_pred_mean_corr,
                    K_field,
                    Ss_field,
                    Q=0.0,
                )
                cons_res = scale_residual(cons_res, scales.get("cons_scale"))
                gw_res = scale_residual(gw_res, scales.get("gw_scale"))
                # By design we do not rescale prior_res or smooth_res here.
    
            # -----------------------------------------------------------------
            # 7. Composite physics loss and total loss
            # -----------------------------------------------------------------
            loss_cons = tf_reduce_mean(tf_square(cons_res))
            loss_gw = tf_reduce_mean(tf_square(gw_res))
            loss_prior = tf_reduce_mean(tf_square(prior_res))
            # smooth_res is already a squared norm, so just average
            loss_smooth = tf_reduce_mean(smooth_res)
    
            total_loss = (
                data_loss
                + self.lambda_cons * loss_cons
                + self.lambda_gw * loss_gw
                + self.lambda_prior * loss_prior
                + self.lambda_smooth * loss_smooth
                + self.lambda_mv * loss_mv
                + self.lambda_bounds * loss_bounds
            )
    
        # -----------------------------------------------------------------
        # 8. Apply gradients (with per-parameter LR multipliers)
        # -----------------------------------------------------------------
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        del tape  # free persistent tape ASAP
    
        # Scale gradients for log_mv and log_kappa before applying.
        self.optimizer.apply_gradients(
            self._scale_param_grads(grads, trainable_vars)
        )
    
        # -----------------------------------------------------------------
        # 9. Update metrics and return logged quantities
        # -----------------------------------------------------------------
        self.compiled_metrics.update_state(targets, y_pred_for_loss)
        # results = {m.name: m.result() for m in self.metrics}
        # filter them out since they are ignore 
        results = {
                m.name: m.result()
                for m in self.metrics
                if m.name not in ("epsilon_prior", "epsilon_cons")
            }

    
        physics_loss = (
            self.lambda_cons * loss_cons
            + self.lambda_gw * loss_gw
            + self.lambda_prior * loss_prior
            + self.lambda_smooth * loss_smooth
            + self.lambda_mv * loss_mv
            + self.lambda_bounds * loss_bounds 
        )
    
        results.update(
            {
                "total_loss": total_loss,
                "data_loss": data_loss,
                "physics_loss": physics_loss,
                "consolidation_loss": loss_cons,
                "gw_flow_loss": loss_gw,
                "prior_loss": loss_prior,
                "smooth_loss": loss_smooth,
                "mv_prior_loss": loss_mv,
                "bounds_loss": loss_bounds, 
            }
        )
        return results

    def test_step(self, data):
        r"""
        Evaluation step mirroring :meth:`train_step` (data-only + physics).
    
        This overrides :meth:`tf.keras.Model.test_step` to ensure that:
    
        * The **same supervised mapping** used in
          :meth:`train_step` is used for evaluation, i.e.
          ``{"subs_pred", "gwl_pred"}`` heads.
        * Data loss and metrics are computed exactly as during training.
        * Optional **physics diagnostics** are logged via
          :meth:`evaluate_physics` **without** computing gradients.
    
        Parameters
        ----------
        data : tuple
            A tuple ``(inputs, targets)`` where:
    
            * ``inputs`` is a mapping accepted by :meth:`call`, and must
              contain at least the coordinate tensor (``"coords"``) and
              ``"H_field"`` / ``"soil_thickness"`` if physics is enabled.
            * ``targets`` is either
    
              - a dict with keys ``"subsidence"`` and ``"gwl"``, or
              - a dict with keys ``"subs_pred"`` and ``"gwl_pred"``, or
              - a tensor-like already matching the compiled loss.
    
            In the dict case, keys are renamed to the canonical
            ``"subs_pred"`` / ``"gwl_pred"`` mapping used during training.
    
        Returns
        -------
        dict
            Dictionary of scalar metrics. At minimum, contains:
    
            * ``"loss"`` : data loss (same definition as in training),
            * ``"epsilon_prior"`` : RMS of the consistency prior
              :math:`R_\text{prior}`,
            * ``"epsilon_cons"`` : RMS of the consolidation residual
              :math:`R_\text{cons}`,
    
            plus all metrics registered in :attr:`self.metrics`
            (e.g. MAE, MSE, coverage, sharpness, etc.).
    
        Notes
        -----
        * Physics diagnostics are computed on the **input batch only**
          (no dataset iteration here) by calling
          :meth:`evaluate_physics(inputs)`. This uses the same internal
          helpers as training (``_compute_consistency_prior``,
          ``_compute_consolidation_residual``).
        * When physics is turned off (``pde_modes_active`` contains
          ``"none"``), we **still** update the epsilon metrics with
          zeros. This keeps the logging interface stable and avoids
          NaN metrics in purely data-driven runs.
        """
        # --- 0. Unpack and normalize targets 
        inputs, targets = data
    
        # For convenience, accept either {"subsidence", "gwl"} or
        # {"subs_pred", "gwl_pred"} as keys and normalize to the
        # supervised mapping expected by compiled_loss.
        if isinstance(targets, dict):
            targets = rename_dict_keys(
                targets.copy(),
                param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"},
            )
    
        # --- 1. Forward pass (no gradients) 
        # This returns the same output structure as during training:
        # {
        #   "data_final": ... (quantile-expanded),
        #   "data_mean":  ...,
        #   "phys_mean_raw": ...
        # }
        outputs = self(inputs, training=False)
    
        # --- 2. Split data predictions for supervised heads 
        # data_final typically has shape (B, H, Q, 2) or similar; we
        # only care about splitting along the last dimension into
        # subsidence and head.
        s_pred_final, gwl_pred_final = self.split_data_predictions(
            outputs["data_final"]
        )
        y_pred_for_eval = {
            "subs_pred": s_pred_final,  # shape ~ (B, H, Q[, ...])
            "gwl_pred": gwl_pred_final,
        }
    
        # --- 3. Data loss + metrics (same mapping as train_step) 
        loss = self.compiled_loss(
            y_true=targets,
            y_pred=y_pred_for_eval,
            regularization_losses=self.losses,
        )
        self.compiled_metrics.update_state(targets, y_pred_for_eval)
    
        # --- 4. Optional physics diagnostics (no GradientTape here) ------
        # This section is deliberately lightweight and purely diagnostic:
        # it probes the current state of the physics without influencing
        # gradients.
        if not self._physics_off():
            # evaluate_physics() returns scalar epsilons and optionally
            # residual maps. Here we only need the scalars.
            phys = self.evaluate_physics(inputs)  # returns tensors
            self.eps_prior_metric.update_state(phys["epsilon_prior"])
            self.eps_cons_metric.update_state(phys["epsilon_cons"])
        else:
            # Physics is disabled: push zeros to keep metrics well-defined
            # and prevent NaNs in logs.
            self.eps_prior_metric.update_state(0.0)
            self.eps_cons_metric.update_state(0.0)
    
        # --- 5. Collect and return metrics 
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {
                "loss": loss,
                "epsilon_prior": self.eps_prior_metric.result(),
                "epsilon_cons": self.eps_cons_metric.result(),
            }
        )
    
        return results

    def _mv_value(self) -> Tensor:
        r"""
        Return the current value of :math:`m_v` in linear space.
    
        If :math:`m_v` is learnable, this is ``exp(log_mv)``; otherwise
        it is the fixed constant ``_mv_fixed``.
    
        Returns
        -------
        tf.Tensor
            Scalar tensor (0D) representing :math:`m_v > 0`.
        """
        return (
            tf_exp(self.log_mv)
            if hasattr(self, "log_mv")
            else self._mv_fixed
        )
    
    
    def _kappa_value(self) -> Tensor:
        r"""
        Return the current value of :math:`\kappa` in linear space.
    
        If :math:`\kappa` is learnable, this is ``exp(log_kappa)``;
        otherwise it is the fixed constant ``_kappa_fixed``.
    
        Returns
        -------
        tf.Tensor
            Scalar tensor (0D) representing :math:`\kappa > 0`.
        """
        return (
            tf_exp(self.log_kappa)
            if hasattr(self, "log_kappa")
            else self._kappa_fixed
        )

    def current_mv(self):
        r"""
        Return the current value of the compressibility :math:`m_v`.
    
        This is a thin convenience wrapper around :meth:`_mv_value`,
        which handles both the trainable (log-parameterized) and
        fixed-scalar cases.
    
        Returns
        -------
        tf.Tensor
            Scalar tensor representing :math:`m_v` in linear space.
        """
        return self._mv_value()
    
    
    def current_kappa(self):
        r"""
        Return the current value of the consistency coefficient
        :math:`\kappa`.
    
        This is a thin convenience wrapper around :meth:`_kappa_value`,
        which handles both the trainable (log-parameterized) and
        fixed-scalar cases.
    
        Returns
        -------
        tf.Tensor
            Scalar tensor representing :math:`\kappa` in linear space.
        """
        return self._kappa_value()

    def _compute_mv_prior(self, Ss_field):
        r"""
        Compute the storage identity prior
    
        .. math::
    
            R_{m_v}
            = \log S_s - \log (m_v \gamma_w),
    
        which encourages the learned specific storage :math:`S_s`
        to remain consistent with the scalar compressibility
        :math:`m_v` and the unit-weight of water :math:`\gamma_w`.
    
        **NaN safety**
    
        - ``S_s`` is clamped from below by a small :math:`\varepsilon`
          to prevent ``\log(0)`` and the resulting ``-inf`` / ``NaN``
          in the loss.
    
        Parameters
        ----------
        Ss_field : tf.Tensor
            Predicted specific storage field :math:`S_s(x,y)`,
            shape ``(B, H, 1)`` (already passed through a positivity
            function earlier in the pipeline).
    
        Returns
        -------
        tf.Tensor
            Residual :math:`R_{m_v}` with the same shape as
            ``Ss_field``.
        """
        eps = tf_constant(1e-6, dtype=tf_float32)
        Ss_safe = tf_maximum(Ss_field, eps)
    
        return tf_log(Ss_safe) - (
            tf_log(self._mv_value()) + tf_log(self.gamma_w)
        )


    def _compute_gw_flow_residual(
        self,
        dh_dt,
        d_K_dh_dx_dx,
        d_K_dh_dy_dy,
        Ss_field,
        Q: float = 0.0,
    ):
        r"""
        Compute groundwater-flow residual
    
        .. math::
    
            R_\text{gw}
            = S_s \frac{\partial h}{\partial t}
              - \nabla \cdot (K \nabla h) - Q,
    
        where :math:`h` is hydraulic head, :math:`K` hydraulic
        conductivity, :math:`S_s` specific storage and :math:`Q`
        represents distributed sources/sinks.
    
        If the mode ``"gw_flow"`` is not active in
        :attr:`pde_modes_active`, this method returns a tensor of
        zeros with the same shape as ``dh_dt`` so that the term is
        effectively disabled in the total loss.
    
        Parameters
        ----------
        dh_dt : tf.Tensor
            Time derivative :math:`\partial h / \partial t`, shape
            ``(B, H, 1)``.
        d_K_dh_dx_dx, d_K_dh_dy_dy : tf.Tensor
            Spatial derivatives of :math:`K \, \partial h / \partial x`
            and :math:`K \, \partial h / \partial y` w.r.t. ``x`` and
            ``y`` respectively, each of shape ``(B, H, 1)``.
            These are usually obtained inside :meth:`train_step` via
            nested :class:`tf.GradientTape` calls.
        Ss_field : tf.Tensor
            Specific storage field :math:`S_s(x,y)`, shape
            ``(B, H, 1)``.
        Q : float, default=0.0
            Source/sink term. For the current GeoPriorSubsNet
            experiments we set :math:`Q = 0`.
    
        Returns
        -------
        tf.Tensor
            Groundwater-flow residual :math:`R_\text{gw}` with the
            same shape as ``dh_dt``.
        """
        if "gw_flow" not in self.pde_modes_active:
            return tf_zeros_like(dh_dt)
    
        div_K_grad_h = d_K_dh_dx_dx + d_K_dh_dy_dy
        storage_term = Ss_field * dh_dt
    
        return storage_term - div_K_grad_h - Q

    
    def _compute_consolidation_residual(
        self,
        ds_dt,
        s_mean,
        h_mean,
        H_field,
        tau_field,
    ):
        r"""
        Compute consolidation residual
    
        .. math::
    
            R_\text{cons}
            = \frac{\partial s}{\partial t}
              - \frac{s_\text{eq}(h) - s}{\tau},
    
        where the equilibrium settlement :math:`s_\text{eq}` is given
        by
    
        .. math::
    
            s_\text{eq}(h)
            = m_v \, \gamma_w \, (h_\text{ref} - h) \, H.
    
        Here :math:`s` is subsidence, :math:`h` hydraulic head,
        :math:`H` the thickness field, :math:`m_v` compressibility,
        :math:`\gamma_w` the unit weight of water, and
        :math:`h_\text{ref}` a reference head.
    
        If the mode ``"consolidation"`` is not active in
        :attr:`pde_modes_active`, a zero tensor is returned so this
        term does not contribute to the loss.
    
        **NaN safety**
    
        - ``tau_field`` is clamped from below by a small
          :math:`\varepsilon` before performing the division to avoid
          accidental division by zero if the network predicts extremely
          small :math:`\tau`.
    
        Parameters
        ----------
        ds_dt : tf.Tensor
            Time derivative :math:`\partial s / \partial t`, shape
            ``(B, H, 1)``.
        s_mean : tf.Tensor
            Mean subsidence field :math:`s(x,y,t)`, shape
            ``(B, H, 1)``.
        h_mean : tf.Tensor
            Mean hydraulic head field :math:`h(x,y,t)`, shape
            ``(B, H, 1)``.
        H_field : tf.Tensor
            Thickness field :math:`H(x,y)`, shape ``(B, H, 1)``.
        tau_field : tf.Tensor
            Relaxation time field :math:`\tau(x,y)`, shape
            ``(B, H, 1)``, usually already passed through a
            positivity function.
    
        Returns
        -------
        tf.Tensor
            Consolidation residual :math:`R_\text{cons}` with the
            same shape as ``ds_dt``.
        """
        if "consolidation" not in self.pde_modes_active:
            return tf_zeros_like(ds_dt)
    
        eps = tf_constant(1e-6, dtype=tf_float32)
        tau_safe = tf_maximum(tau_field, eps)
    
        # Delta_h = h_ref - h
        delta_h = self.h_ref - h_mean
    
        # s_eq(h) = m_v * gamma_w * Delta_h * H
        s_eq = self._mv_value() * self.gamma_w * delta_h * H_field
    
        # Relaxation term (s_eq - s) / tau
        relaxation_term = (s_eq - s_mean) / tau_safe
    
        return ds_dt - relaxation_term


    def _compute_consistency_prior(
        self,
        K_field,
        Ss_field,
        tau_field,
        H_field,
    ):
        r"""
        Compute the geomechanical consistency prior
    
        .. math::
    
            R_\text{prior} = \log(\tau_\text{pred})
                             - \log(\tau_\text{phys})
    
        where :math:`\tau_\text{phys}` is the physical time scale implied
        by :math:`K, S_s, H` and the scalar coefficient :math:`\kappa`.
    
        For ``kappa_mode == "bar"`` we interpret :math:`\kappa` as
        :math:`\bar{\kappa}` acting on an *effective* thickness
        :math:`H_\text{eff} = H_d`:
    
        .. math::
    
            \tau_\text{phys}
            = \frac{\bar{\kappa} \, H_\text{eff}^2 \, S_s}
                   {\pi^2 K}.
    
        For ``kappa_mode == "kb"`` we interpret :math:`\kappa` as a
        basal :math:`\kappa_b` so that the factor
    
        .. math::
    
            \left(\frac{H_d}{H}\right)^2
    
        is handled explicitly in the expression.
    
        **NaN safety**
    
        - All quantities entering ``log(·)`` are shifted by a small
          :math:`\varepsilon` to avoid ``log(0)``.
        - This is *critical* for ``kappa_mode == "kb"`` where the
          original implementation used ``log(H_field)`` directly, which
          produced ``-inf`` and hence ``NaN`` losses when
          :math:`H = 0`.
    
        Parameters
        ----------
        K_field : tf.Tensor
            Predicted hydraulic conductivity field :math:`K(x,y)`,
            shape ``(B, H, 1)``.
        Ss_field : tf.Tensor
            Predicted specific storage field :math:`S_s(x,y)`,
            shape ``(B, H, 1)``.
        tau_field : tf.Tensor
            Predicted relaxation time field :math:`\tau(x,y)`,
            shape ``(B, H, 1)``.
        H_field : tf.Tensor
            Input thickness field :math:`H(x,y)`, in model units,
            shape ``(B, H, 1)``.
    
        Returns
        -------
        tf.Tensor
            Residual :math:`R_\text{prior}` with the same shape
            as ``tau_field``.
        """
        eps = tf_constant(1e-6, dtype=tf_float32)
        pi_sq = tf_constant(np.pi**2, dtype=tf_float32)
    

        # 1) Make all log arguments strictly positive (NaN safety)
        tau_safe = tf_maximum(tau_field, eps)
        K_safe   = tf_maximum(K_field,   eps)
        Ss_safe  = tf_maximum(Ss_field,  eps)
    
        # log(tau_pred)
        log_tau_pred = tf_log(tau_safe)
    
        # H-handling:
        #   - safe_H_eff is the "effective" thickness H_d or H (+eps)
        #   - safe_H is the raw H (+eps) used in the kb-mode formula
        safe_H_eff = (
            H_field * self.Hd_factor
            if self.use_effective_thickness else H_field
        ) + eps
        safe_H = H_field + eps
    
        # 2) Compute log(tau_phys) in a mode-dependent way
        if self.kappa_mode == "bar":
            # Standard formulation using H_eff directly:
            #
            # tau_phys = (kappa * H_eff^2 * Ss) / (pi^2 * K)
            #
            # log(tau_phys)
            #   = log(kappa) + 2 log(H_eff)
            #     - [log(pi^2) + log(K) - log(Ss)]
            log_tau_phys = (
                tf_log(self._kappa_value())
                + 2.0 * tf_log(safe_H_eff)
                - (tf_log(pi_sq) + tf_log(K_safe) - tf_log(Ss_safe))
            )
        else:  # "kb" – kappa is κ_b; incorporate (H_d/H)^2 explicitly.
            #
            # tau_phys
            #   = kappa_b * (H_eff / H)^2 * H^2 * Ss / (pi^2 * K)
            #   = kappa_b * (H_eff^2 * Ss) / (pi^2 * K)
            #     (mathematically equivalent but we keep the ratio explicit)
            #
            # To remain explicit in logs:
            #
            #   log(tau_phys)
            #     = log(kappa_b) + 2 log(H_eff / H)
            #       + 2 log(H) - [log(pi^2) + log(K) - log(Ss)]
            #
            # and we ensure both H_eff and H never hit zero by
            # using safe_H_eff and safe_H above.
            ratio = safe_H_eff / safe_H
            log_tau_phys = (
                tf_log(self._kappa_value())
                + 2.0 * tf_log(ratio)
                + 2.0 * tf_log(safe_H)
                - (tf_log(pi_sq) + tf_log(K_safe) - tf_log(Ss_safe))
            )
    
        # 3) Consistency residual in log-space
        return log_tau_pred - log_tau_phys


    def get_last_physics_fields(self):
        """
        Returns the most recent physics fields and H used by the model call.
        Shapes: (B, H, 1) each, matching the last forward pass.
        """
        return {
            "tau":  self.tau_field,
            "K":    self.K_field,
            "Ss":   self.Ss_field,
            "H_in": self.H_field,   # raw H passed in inputs
        }

    def _tau_phys_from_fields(self, K_field, Ss_field, H_field):
        r"""
        Compute the physical time scale :math:`\tau_\text{phys}` implied
        by :math:`K, S_s, H` and the scalar coefficient :math:`\kappa`.
    
        This is the linear-space counterpart of
        :meth:`_compute_consistency_prior` and follows the same
        ``kappa_mode`` logic.
    
        For ``kappa_mode == "bar"``:
    
        .. math::
    
            \tau_\text{phys}
            = \frac{\bar{\kappa} \, H_\text{eff}^2 \, S_s}
                   {\pi^2 K},
    
        where :math:`H_\text{eff}` is the effective thickness
        (either :math:`H` or :math:`H_d` depending on
        ``use_effective_thickness``).
    
        For ``kappa_mode == "kb"``:
    
        .. math::
    
            \tau_\text{phys}
            = \frac{\kappa_b \, (H_\text{eff}/H)^2 \, H^2 \, S_s}
                   {\pi^2 K}.
    
        **NaN / inf safety**
    
        - ``K_field`` and ``Ss_field`` are clipped to be at least
          :math:`\varepsilon` before entering the denominators.
        - ``H_field`` and the effective thickness are shifted by
          :math:`\varepsilon` to avoid zero divisions in the ratio
          :math:`H_\text{eff} / H`.
    
        Parameters
        ----------
        K_field : tf.Tensor
            Predicted hydraulic conductivity field :math:`K(x,y)`,
            shape ``(B, H, 1)``.
        Ss_field : tf.Tensor
            Predicted specific storage field :math:`S_s(x,y)`,
            shape ``(B, H, 1)``.
        H_field : tf.Tensor
            Input thickness field :math:`H(x,y)`, in model units,
            shape ``(B, H, 1)``.
    
        Returns
        -------
        tau_phys : tf.Tensor
            Physical relaxation time :math:`\tau_\text{phys}(x,y)` in
            linear space, shape ``(B, H, 1)``.
        H_eff : tf.Tensor
            Effective thickness field :math:`H_\text{eff}(x,y)` actually
            used in the computation, shape ``(B, H, 1)``.
        """
        eps = tf_constant(1e-6, dtype=tf_float32)
        pi_sq = tf_constant(np.pi**2, dtype=tf_float32)
    
        # Make sure denominator fields are never exactly zero to avoid
        # inf / NaN when computing tau_phys.
        K_safe  = tf_maximum(K_field,  eps)
        Ss_safe = tf_maximum(Ss_field, eps)
    
        # Same effective thickness logic as in _compute_consistency_prior
        safe_H_eff = (
            H_field * self.Hd_factor
            if self.use_effective_thickness else H_field
        ) + eps
        safe_H = H_field + eps
    
        if self.kappa_mode == "bar":
            # τ_phys = (κ * H_eff^2 * Ss) / (π^2 * K)
            tau_phys = (
                self._kappa_value()
                * (safe_H_eff ** 2)
                * Ss_safe
                / (pi_sq * K_safe)
            )
        else:  # "kb"
            # τ_phys = κ_b * (H_eff/H)^2 * H^2 * Ss / (π^2 * K)
            ratio = safe_H_eff / safe_H
            tau_phys = (
                self._kappa_value()
                * (ratio ** 2)
                * (safe_H ** 2)
                * Ss_safe
                / (pi_sq * K_safe)
            )
    
        # Return tau_phys and the effective thickness actually used.
        return tau_phys, safe_H_eff

    def _compute_smoothness_prior(
        self,
        dK_dx,
        dK_dy,
        dSs_dx,
        dSs_dy,
    ):
        r"""
        Compute spatial smoothness prior on :math:`K` and :math:`S_s`.
    
        The smoothness residual is defined as
    
        .. math::
    
            R_\text{smooth}
            = \|\nabla K\|_2^2 + \|\nabla S_s\|_2^2
            = \left(\frac{\partial K}{\partial x}\right)^2
            + \left(\frac{\partial K}{\partial y}\right)^2
            + \left(\frac{\partial S_s}{\partial x}\right)^2
            + \left(\frac{\partial S_s}{\partial y}\right)^2.
    
        In practice this function returns a tensor with the same spatio-
        temporal shape as the derivative fields. The scalar loss
    
        .. math::
    
            L_\text{smooth}
            = \mathbb{E}\left[R_\text{smooth}\right]
    
        is computed later via :func:`tf.reduce_mean`.
    
        Parameters
        ----------
        dK_dx, dK_dy : tf.Tensor
            Spatial derivatives :math:`\partial K / \partial x` and
            :math:`\partial K / \partial y`, typically of shape
            ``(B, H, 1)``.
        dSs_dx, dSs_dy : tf.Tensor
            Spatial derivatives :math:`\partial S_s / \partial x` and
            :math:`\partial S_s / \partial y`, same shape as above.
    
        Returns
        -------
        tf.Tensor
            Smoothness residual :math:`R_\text{smooth}` with the same
            shape as the input derivative tensors.
    
        Notes
        -----
        * No explicit scaling is applied here; the relative importance
          of this term is controlled by ``lambda_smooth`` in
          :meth:`compile`.
        * Any NaNs in the input derivatives will propagate through;
          those should be prevented earlier (e.g. by ensuring
          gradients are well-defined in :meth:`train_step`).
        """
        grad_K_squared = tf_square(dK_dx) + tf_square(dK_dy)
        grad_Ss_squared = tf_square(dSs_dx) + tf_square(dSs_dy)
    
        return grad_K_squared + grad_Ss_squared

    def _compute_bounds_residual(
        self,
        K_field: Tensor,
        Ss_field: Tensor,
        H_field: Tensor,
    ) -> Tensor:
        """
        Soft bound violations for H, log K, log S_s.

        Bounds are configured via ``self.scaling_kwargs['bounds']`` with
        optional scalar entries:

          - 'H_min', 'H_max'          (m)
          - 'logK_min', 'logK_max'    (natural log of m/s)
          - 'logSs_min', 'logSs_max'  (natural log of Pa^{-1})

        Any missing entry disables the corresponding penalty.

        Returns
        -------
        Tensor
            Non-negative residual with same shape as the fields, whose
            squared mean is added to the loss.
        """
        eps = tf_constant(1e-6, dtype=tf_float32)
        zero = tf_constant(0.0, dtype=tf_float32)

        K_safe  = tf_maximum(K_field, eps)
        Ss_safe = tf_maximum(Ss_field, eps)
        H_safe  = H_field + eps

        bounds_cfg = self.scaling_kwargs.get("bounds", {})

        # --- H bounds ----------------------------------------------------
        H_min = bounds_cfg.get("H_min", None)
        H_max = bounds_cfg.get("H_max", None)
        if H_min is None or H_max is None:
            R_H = tf_zeros_like(H_safe)
        else:
            H_min_t = tf_constant(float(H_min), dtype=tf_float32)
            H_max_t = tf_constant(float(H_max), dtype=tf_float32)
            lower_H = tf_maximum(H_min_t - H_safe, zero)
            upper_H = tf_maximum(H_safe - H_max_t, zero)
            R_H = lower_H + upper_H

        # --- log K bounds -----------------------------------------------
        logK_min = bounds_cfg.get("logK_min", None)
        logK_max = bounds_cfg.get("logK_max", None)
        if logK_min is None or logK_max is None:
            R_K = tf_zeros_like(K_safe)
        else:
            logK = tf_log(K_safe)
            logK_min_t = tf_constant(float(logK_min), dtype=tf_float32)
            logK_max_t = tf_constant(float(logK_max), dtype=tf_float32)
            lower_K = tf_maximum(logK_min_t - logK, zero)
            upper_K = tf_maximum(logK - logK_max_t, zero)
            R_K = lower_K + upper_K

        # --- log S_s bounds ---------------------------------------------
        logSs_min = bounds_cfg.get("logSs_min", None)
        logSs_max = bounds_cfg.get("logSs_max", None)
        if logSs_min is None or logSs_max is None:
            R_Ss = tf_zeros_like(Ss_safe)
        else:
            logSs = tf_log(Ss_safe)
            logSs_min_t = tf_constant(float(logSs_min), dtype=tf_float32)
            logSs_max_t = tf_constant(float(logSs_max), dtype=tf_float32)
            lower_Ss = tf_maximum(logSs_min_t - logSs, zero)
            upper_Ss = tf_maximum(logSs - logSs_max_t, zero)
            R_Ss = lower_Ss + upper_Ss

        return R_H + R_K + R_Ss

    def _compute_scales(
        self,
        t,
        s_mean,
        h_mean,
        K_field,
        Ss_field,
        Q: float = 0.0,
    ):
        r"""
        Build dimensionless scaling factors for PDE residuals.
    
        This helper constructs the arguments required by
        :func:`default_scales` and returns the resulting scaling
        dictionary, typically containing per-batch factors used to
        rescale the PDE residuals (e.g. ``cons_scale``, ``gw_scale``).
    
        Time stepping
        -------------
        The time increment :math:`\Delta t` is estimated via a simple
        forward finite difference on the time coordinate:
    
        .. math::
    
            \Delta t_{b,i} = t_{b,i+1} - t_{b,i}.
    
        If ``t`` does not have at least two time steps, or if no valid
        difference can be computed, we fall back to a tensor of ones
        (unit time step) to keep the scales finite.
    
        Parameters
        ----------
        t : tf.Tensor
            Time coordinate tensor, typically of shape ``(B, H, 1)``.
            Assumed to be *monotonically increasing* in the time axis
            used for the finite differences.
        s_mean : tf.Tensor
            Mean subsidence field :math:`s(x,y,t)`, shape compatible
            with ``t``.
        h_mean : tf.Tensor
            Mean hydraulic head field :math:`h(x,y,t)`, shape compatible
            with ``t``.
        K_field : tf.Tensor
            Predicted hydraulic conductivity field :math:`K(x,y)`,
            shape ``(B, H, 1)``.
        Ss_field : tf.Tensor
            Predicted specific storage field :math:`S_s(x,y)`,
            shape ``(B, H, 1)``.
        Q : float, default=0.0
            Source/sink term in the groundwater-flow PDE. For the
            revised manuscript this is typically set to ``0.0``.
    
        Returns
        -------
        dict
            A dictionary of scaling factors as returned by
            :func:`default_scales`, e.g.::
    
                {
                  "cons_scale": ...,
                  "gw_scale": ...,
                  ...
                }
    
        Notes
        -----
        * The exact non-dimensionalization is delegated to
          :func:`default_scales`, which can be configured via
          ``self.scaling_kwargs``.
        * If you change the signature of :func:`default_scales`,
          remember to keep this adapter in sync.
        """
        # dt_tensor = None
        # if hasattr(t, "shape") and t.shape.rank is not None and t.shape.rank >= 2:
        #     # t shape: (B, H, 1) or (B, T, 1). We assume the second
        #     # axis encodes the temporal dimension used for dt.
        #     if t.shape[1] and t.shape[1] > 1:
        #         dt_tensor = t[:, 1:, :] - t[:, :-1, :]
    
        # # Fallback: if dt cannot be inferred, use a unit time step.
        # if dt_tensor is None:
        #     dt_tensor = tf_zeros_like(s_mean[..., :1]) + 1.0
        dt_tensor = None
        if hasattr(t, "shape") and t.shape.rank is not None and t.shape.rank >= 2:
            second_dim = t.shape[1]
            if (second_dim is not None) and (second_dim > 1):
                dt_tensor = t[:, 1:, :] - t[:, :-1, :]
        
        if dt_tensor is None:
            # shape consistent with (B, H, 1)
            dt_tensor = tf_zeros_like(s_mean[..., :1]) + 1.0

        # --- Pass the predicted K and Ss fields into default_scales -----
        return default_scales(
            h=h_mean,
            s=s_mean,
            dt=dt_tensor,
            K=K_field,   # <-- predicted K field
            Ss=Ss_field, # <-- predicted Ss field
            Q=Q,         # <-- scalar source/sink (0.0 in current setup)
            **self.scaling_kwargs,
        )

    def _evaluate_physics_on_batch(
        self,
        inputs: Dict[str, Optional[Tensor]],
        return_maps: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Core implementation: physics diagnostics on a *single batch*.
        This is basically the old `evaluate_physics` body.
        """
        # --- Validate presence of H_field/soil_thickness and extract coords ---
        H_field_in = get_tensor_from(
            inputs, "H_field", "soil_thickness",
            auto_convert=True,
        )
        if H_field_in is None:
            raise ValueError(
                "evaluate_physics() requires 'H_field' "
                "(or 'soil_thickness') in `inputs`."
            )
        H_field = tf_convert_to_tensor(H_field_in, dtype=tf_float32)
    
        coords = _get_coords(inputs)
        t, x, y = extract_txy_in(coords)
        coords_flat = tf_concat([t, x, y], axis=-1)
    
        # We must compute model outputs under the tape so AD sees dependency on t.
        with GradientTape() as tape:
            tape.watch(t)
    
            # Forward pass (no training); also stores self.H_field internally.
            outputs = self(inputs, training=False)
    
            # Split means and positive physics fields.
            (
                s_mean,
                h_mean,
                K_base,
                Ss_base,
                tau_base,
            )  = self.split_physics_predictions(outputs)
            # (
            #     s_mean,
            #     h_mean,
            #     K_field,
            #     Ss_field,
            #     tau_field,
            # ) = self.split_physics_predictions(outputs)
            
            # Reuse coordinate MLPs (no training, faster)
            mlp_corr = self.coord_mlp(coords_flat, training=False)
            s_corr = self.subs_coord_mlp(coords_flat, training=False)
        
            h_mean_corr = h_mean + mlp_corr
            s_mean_corr = s_mean + s_corr
        
            K_corr = self.K_coord_mlp(coords_flat, training=False)
            Ss_corr = self.Ss_coord_mlp(coords_flat, training=False)
            tau_corr = self.tau_coord_mlp(coords_flat, training=False)
        
            K_field = positive(K_base + K_corr)
            Ss_field = positive(Ss_base + Ss_corr)
            tau_field = positive(tau_base + tau_corr)
            
            # Keep last-physics fields consistent with the training path
            self.K_field = K_field
            self.Ss_field = Ss_field
            self.tau_field = tau_field
                        
            # Bind s_mean to t so the tape tracks s(t,x,y)
            s_bind = s_mean + 0.0 * t
    
        # AD: ds/dt at the same spatiotemporal points
        ds_dt = tape.gradient(s_bind, t)
        if ds_dt is None:
            raise ValueError(
                "Automatic differentiation returned None. "
                "Ensure (t,x,y) influence the subsidence head "
                "via the coordinate injection path."
            )
    
        # --- Residuals using the model's own helpers (exactly as in training) ---
        # Prior: R_prior = log(tau) - log( (kappa * H^2 Ss) / (π^2 K) )
        R_prior = self._compute_consistency_prior(
            K_field, Ss_field, tau_field, H_field)

        # Consolidation: R_cons = ∂s/∂t - (s_eq - s)/tau,
        # with s_eq = m_v γ_w (h_ref - h) H

        R_cons  = self._compute_consolidation_residual(
            ds_dt, s_mean_corr, h_mean_corr, H_field, 
            tau_field
        )
    
        # --- Unscaled RMS errors (paper definitions) ---
        eps_prior = tf_sqrt(tf_reduce_mean(tf_square(R_prior)))
        eps_cons = tf_sqrt(tf_reduce_mean(tf_square(R_cons)))
    
        out = {"epsilon_prior": eps_prior, "epsilon_cons": eps_cons}
        if return_maps:
            # Also return fields needed for Fig.4 payload
            tau_phys, Hd_eff = self._tau_phys_from_fields(
                K_field, Ss_field, H_field
            )
            out.update(
                {
                    "R_prior": R_prior,
                    "R_cons": R_cons,
                    "K": K_field,
                    "Ss": Ss_field,
                    "H": Hd_eff,  # effective H actually used
                    # numerically safe no-op to get a clean tensor
                    "tau": tf_exp(tf_log(tau_field)),
                    "tau_prior": tau_phys,
                }
            )
        return out
    

    def evaluate_physics(
        self,
        inputs: Union[Dict[str, Optional[Tensor]], "Dataset"],
        return_maps: bool = False,
        max_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        r"""
        Evaluate physics diagnostics.
    
        Parameters
        ----------
        inputs : dict or tf.data.Dataset
            - If a mapping (dict), it is treated as **one batch**, exactly
              as in the original implementation.  This is the path used by
              Keras' ``test_step``.
            - If a ``tf.data.Dataset``, the dataset is iterated batch by
              batch (up to ``max_batches``) and the scalar diagnostics are
              aggregated across batches.
    
        return_maps : bool, default=False
            Whether to also return residual maps and physical fields
            (``R_prior``, ``R_cons``, ``K``, ``Ss``, ``H``, ``tau``,
            ``tau_prior``).  When ``inputs`` is a dataset these maps are
            returned **for the last processed batch only**, to avoid
            storing huge tensors in memory.
    
        max_batches : int or None, default=None
            When ``inputs`` is a dataset, limit the number of batches
            that are processed.  If ``None``, the whole dataset is used.
            This is the main knob to avoid out-of-memory situations on
            very large splits.
    
        batch_size : int or None, default=None
            Convenience option when passing a dict of NumPy arrays:
            if not ``None``, a temporary ``tf.data.Dataset`` is built
            and processed using mini-batches of this size.
    
        Returns
        -------
        dict
            Always contains the scalar diagnostics
    
            ``{"epsilon_prior": ..., "epsilon_cons": ...}``
    
            and, if ``return_maps=True``, also residual/field maps.
    
        Notes
        -----
        * Keras' ``test_step`` still passes a **single-batch dict**,
          so behaviour during training/validation is unchanged.
        * For large evaluation splits you should pass a pre-batched
          dataset, for example::
    
              ds = tf.data.Dataset.from_tensor_slices((X_fore, y_fore_fmt))\
                                   .batch(256)
              phys = model.evaluate_physics(ds, max_batches=10)
    
          which only uses the first 10 batches instead of the full
          Zhongshan split.
        """

        # 1) Dataset path: iterate up to `max_batches` and aggregate scalars
        if isinstance(inputs, Dataset):
            eps_prior_vals = []
            eps_cons_vals = []
            last_maps = None
    
            for i, elem in enumerate(inputs):
                # elem may be (xb, yb) or xb
                xb = elem[0] if isinstance(elem, (tuple, list)) else elem
    
                out_b = self._evaluate_physics_on_batch(
                    xb, return_maps=return_maps
                )
                eps_prior_vals.append(out_b["epsilon_prior"])
                eps_cons_vals.append(out_b["epsilon_cons"])
    
                if return_maps:
                    # keep non-scalar maps only from the last batch
                    last_maps = {
                        k: v
                        for k, v in out_b.items()
                        if k not in ("epsilon_prior", "epsilon_cons")
                    }
    
                if max_batches is not None and (i + 1) >= max_batches:
                    break
    
            if not eps_prior_vals:
                raise ValueError("Empty dataset provided to evaluate_physics.")
    
            # eps_* values are already RMS per batch; we aggregate by mean
            eps_prior = tf_reduce_mean(tf_stack(eps_prior_vals))
            eps_cons = tf_reduce_mean(tf_stack(eps_cons_vals))
    
            out = {"epsilon_prior": eps_prior, "epsilon_cons": eps_cons}
            if return_maps and last_maps is not None:
                out.update(last_maps)
            return out
    
        # 2) Dict-of-NumPy arrays convenience path with `batch_size`
        if isinstance(inputs, Mapping) and batch_size is not None:
            # If at least one value is not a Tensor, assume NumPy-like.
            any_tensor = any(
                isinstance(v, Tensor) for v in inputs.values() if v is not None
            )
            if not any_tensor:
                ds = Dataset.from_tensor_slices(inputs).batch(batch_size)
                return self.evaluate_physics(
                    ds,
                    return_maps=return_maps,
                    max_batches=max_batches,
                )
    
        # 3) Backwards-compatible single-batch behaviour
        # Either a dict of Tensors (from test_step) or a small dict of NumPy arrays
        return self._evaluate_physics_on_batch(
            inputs, return_maps=return_maps
        )
    
    def export_physics_payload(
        self,
        dataset,
        max_batches=None,
        save_path=None,
        format: str = "npz",
        overwrite: bool = False,
        metadata=None,
        random_subsample=None,
        float_dtype=np.float32,
        log_fn =None, 
        **tqdm_kws
    ):
        r"""
        Gather a physics payload from a dataset and optionally persist it.
    
        This is a convenience wrapper around
        :func:`gather_physics_payload` that:
    
        1. Iterates over a (pre-batched) dataset and calls
           :meth:`evaluate_physics` with ``return_maps=True`` to collect
           ``tau``, ``tau_prior``, :math:`K`, :math:`S_s`, :math:`H_d`
           and consolidation residuals.
        2. Optionally applies a random subsampling step to reduce the
           payload size for plotting / diagnostics.
        3. Optionally saves the payload + metadata to disk using
           :func:`save_physics_payload`.
    
        Parameters
        ----------
        dataset : iterable
            Typically a ``tf.data.Dataset`` yielding
            ``(inputs, targets)`` or ``inputs`` batches, where
            ``inputs`` is compatible with
            :meth:`evaluate_physics` (i.e. contains ``"coords"`` and
            ``"H_field"`` / ``"soil_thickness"``).
        max_batches : int or None, default=None
            Optional cap on the number of batches processed. If
            ``None``, all batches in ``dataset`` are consumed.
            Use this to limit memory / runtime when the forecast
            split is large (e.g. Zhongshan).
        save_path : str or None, default=None
            If not ``None``, the physics payload and metadata are
            written to this path. The file format is controlled by
            ``format``.
        format : {"npz", "parquet", "csv"}, default="npz"
            Serialization format understood by
            :func:`save_physics_payload`. The default ``"npz"`` is
            compact and preserves dtypes.
        overwrite : bool, default=False
            If ``False`` and a file already exists at ``save_path``,
            :func:`save_physics_payload` will raise. Set to ``True``
            to overwrite existing payloads.
        metadata : dict or None, default=None
            Optional extra metadata to merge into the base metadata
            created by :func:`default_meta_from_model`. This is a good
            place to stash run IDs, dataset descriptors, etc.
        random_subsample : int, float or None, default=None
            Optional subsampling specification forwarded to
            :func:`_maybe_subsample`. Typical values:
    
            * ``int``  → keep at most this many points,
            * ``float`` in ``(0, 1]`` → keep a fixed fraction of
              the payload.
    
            If ``None``, no subsampling is performed.
        float_dtype : numpy dtype, default=np.float32
            Target dtype for the numeric arrays in the payload.
    
        Returns
        -------
        dict
            The physics payload returned by
            :func:`gather_physics_payload`, possibly subsampled.
            Keys include, for example:
    
            ``"tau"``, ``"tau_prior"``, ``"K"``, ``"Ss"``, ``"Hd"``,
            ``"cons_res_vals"``, and a nested ``"metrics"`` dict.
    
        Notes
        -----
        * This method is intentionally high-level and can be called
          from scripts/notebooks as a one-liner after training:
    
          .. code-block:: python
    
              payload = model.export_physics_payload(
                  ds_fore,
                  max_batches=20,
                  save_path="geoprior_phys_zs.npz",
                  random_subsample=500_000,
              )
    
        * For very large splits, choose a moderate ``max_batches`` and
          a subsampling strategy to keep the payload size manageable.
        """
        payload = gather_physics_payload(
            self,
            dataset,
            max_batches=max_batches,
            float_dtype=float_dtype,
            log_fn=log_fn, 
            **tqdm_kws
        )
    
        if random_subsample is not None:
            payload = _maybe_subsample(payload, random_subsample)
    
        if save_path is not None:
            meta = default_meta_from_model(self)
            if metadata:
                meta.update(metadata)
            save_physics_payload(
                payload,
                meta,
                save_path,
                format=format,
                overwrite=overwrite,
                log_fn=log_fn,
            )
    
        return payload
    
    
    @staticmethod
    def load_physics_payload(path):
        r"""
        Load a previously saved physics payload and its metadata.
    
        This is a thin wrapper around :func:`load_physics_payload`
        to keep the API symmetric with
        :meth:`export_physics_payload`.
    
        Parameters
        ----------
        path : str
            File path pointing to a physics payload previously
            written by :func:`save_physics_payload`.
    
        Returns
        -------
        tuple
            ``(payload, metadata)`` where
    
            * ``payload`` is a dict of 1D arrays such as
              ``"tau"``, ``"tau_prior"``, ``"K"``, ``"Ss"``,
              ``"Hd"``, etc.
            * ``metadata`` is a dictionary containing model/config
              information at export time (run ID, city, model name,
              etc.).
        """
        return load_physics_payload(path)
    
    def split_data_predictions(
        self,
        data_tensor: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Split the combined data tensor into subsidence and head parts.
    
        The data decoder produces a single tensor that concatenates
        both data outputs along the last dimension:
    
        .. math::
    
            \text{data} = [s \,\|\, h],
    
        where :math:`s` is subsidence and :math:`h` is hydraulic
        head. This helper slices that tensor into its two components.
    
        Parameters
        ----------
        data_tensor : tf.Tensor
            Combined data predictions with shape
    
            ``(..., output_subsidence_dim + output_gwl_dim)``.
    
            This is typically either:
    
            * the mean predictions (``data_mean``), or
            * the quantile-expanded tensor (``data_final``).
    
        Returns
        -------
        (tf.Tensor, tf.Tensor)
            ``(s_pred, gwl_pred)`` where
    
            * ``s_pred`` has shape ``(..., output_subsidence_dim)``,
            * ``gwl_pred`` has shape ``(..., output_gwl_dim)``.
    
        Notes
        -----
        * This function does **not** apply any activation; it only
          slices the last dimension.
        * The dimensionality is controlled by
          :attr:`output_subsidence_dim` and :attr:`output_gwl_dim`
          set in ``__init__``.
        """
        s_pred = data_tensor[..., : self.output_subsidence_dim]
        gwl_pred = data_tensor[..., self.output_subsidence_dim :]
    
        return s_pred, gwl_pred

    def split_physics_predictions(
        self,
        outputs_dict: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Split mean data predictions and physics *logits*.
    
        The model exposes two streams of features:
    
        * ``"data_mean"``: combined mean predictions for
          :math:`s` and :math:`h`,
        * ``"phys_mean_raw"``: raw physics logits that will be
          mapped to positive fields :math:`K`, :math:`S_s`,
          :math:`\tau`.
    
        This helper:
    
        1. Splits ``data_mean`` into mean subsidence and head,
        2. Slices ``phys_mean_raw`` into three blocks:
    
           * :math:`K(x,y)` (hydraulic conductivity),
           * :math:`S_s(x,y)` (specific storage),
           * :math:`\tau(x,y)` (relaxation time),
    
        3. Applies the positivity function :func:`positive` to each
           physics block, and
        4. Stores the resulting physics fields on the model instance
           for later inspection (e.g. in :meth:`get_last_physics_fields`).
    
        Parameters
        ----------
        outputs_dict : dict
            Output dictionary produced by :meth:`call`, expected to
            contain at least the keys:
    
            * ``"data_mean"`` : combined mean data outputs,
            * ``"phys_mean_raw"`` : raw physics logits.
    
            Shapes are typically ``(B, H, U)`` for both tensors.
    
        Returns
        -------
        tuple of tf.Tensor
            ``(s_pred_mean, gwl_pred_mean, K_field, Ss_field, tau_field)``
            * s_pred_mean : Tensor
                Mean subsidence path \\bar{s}(x,y,t) used for physics.
            * gwl_pred_mean : Tensor
                Mean head path \\bar{h}(x,y,t) used for physics.
            * K_logits, Ss_logits, tau_logits : Tensor
                Raw physics logits from ``physics_mean_head``. Positivity
                is enforced later (after coordinate-based corrections)
                in ``train_step`` / ``evaluate_physics``.
    
        Notes
        -----
        * The lengths of the physics slices are controlled by
          :attr:`output_K_dim`, :attr:`output_Ss_dim` and
          :attr:`output_tau_dim`.
        * By storing ``K_field``, ``Ss_field`` and ``tau_field`` on
          ``self``, we make them accessible to diagnostic routines
          and to :meth:`evaluate_physics`.
        """
        # data_means_tensor = outputs_dict["data_mean"]
        # phys_means_raw_tensor = outputs_dict["phys_mean_raw"]
    
        # # --- Split mean data predictions (s, h) 
        # s_pred_mean = data_means_tensor[..., : self.output_subsidence_dim]
        # gwl_pred_mean = data_means_tensor[
        #     ..., self.output_subsidence_dim :
        # ]
    
        # # --- Slice physics logits: (K, Ss, tau) 
        # start_idx = 0
        # end_idx = self.output_K_dim
        # K_raw = phys_means_raw_tensor[..., start_idx:end_idx]
    
        # start_idx = end_idx
        # end_idx += self.output_Ss_dim
        # Ss_raw = phys_means_raw_tensor[..., start_idx:end_idx]
    
        # start_idx = end_idx
        # tau_raw = phys_means_raw_tensor[..., start_idx:]
    
        # # --- Positivity constraints 
        # K_field = positive(K_raw)
        # Ss_field = positive(Ss_raw)
        # tau_field = positive(tau_raw)
    
        # # Store fields from latest forward pass for debugging/diagnostics
        # self.K_field = K_field
        # self.Ss_field = Ss_field
        # self.tau_field = tau_field
    
        # return s_pred_mean, gwl_pred_mean, K_field, Ss_field, tau_field

        data_means_tensor = outputs_dict["data_mean"]
        phys_means_raw_tensor = outputs_dict["phys_mean_raw"]
    
        # --- Split mean data predictions (s, h)
        s_pred_mean = data_means_tensor[..., : self.output_subsidence_dim]
        gwl_pred_mean = data_means_tensor[..., self.output_subsidence_dim :]
    
        # --- Slice physics logits: (K, Ss, tau)
        start_idx = 0
        end_idx = self.output_K_dim
        K_logits = phys_means_raw_tensor[..., start_idx:end_idx]
    
        start_idx = end_idx
        end_idx += self.output_Ss_dim
        Ss_logits = phys_means_raw_tensor[..., start_idx:end_idx]
    
        start_idx = end_idx
        tau_logits = phys_means_raw_tensor[..., start_idx:]
    
        # NOTE: no positivity transform here by design.
        return s_pred_mean, gwl_pred_mean, K_logits, Ss_logits, tau_logits

    def _scale_param_grads(self, grads, trainable_vars):
        r"""
        Scale gradients for special scalar parameters :math:`m_v` and
        :math:`\kappa`.
    
        The scalar parameters :math:`m_v` and :math:`\kappa` are
        internally represented in log-space via the variables
        ``log_mv`` and ``log_kappa``. To tune their learning speed
        independently of the rest of the network, we apply per-parameter
        multipliers ``mv_lr_mult`` and ``kappa_lr_mult`` to their
        gradients.
    
        Parameters
        ----------
        grads : list of tf.Tensor or None
            Gradients as returned by :func:`tape.gradient` in
            :meth:`train_step`.
        trainable_vars : list of tf.Variable
            The model's trainable variables, in the same order as
            used when calling :func:`tape.gradient`.
    
        Returns
        -------
        list of (tf.Tensor, tf.Variable)
            List of (gradient, variable) pairs ready to be passed to
            :meth:`optimizer.apply_gradients`.
    
        Notes
        -----
        * Any ``None`` gradients (e.g. for unused variables) are
          skipped.
        * Only the gradients associated with ``log_mv`` and
          ``log_kappa`` are scaled; all others are left unchanged.
        * The actual multipliers are stored in
          :attr:`_mv_lr_mult` and :attr:`_kappa_lr_mult`, set in
          :meth:`compile`.
        """
        scaled = []
        mv_var = getattr(self, "log_mv", None)
        kappa_var = getattr(self, "log_kappa", None)
    
        for g, v in zip(grads, trainable_vars):
            if g is None:
                continue
    
            if (mv_var is not None) and (v is mv_var):
                g = g * self._mv_lr_mult
            elif (kappa_var is not None) and (v is kappa_var):
                g = g * self._kappa_lr_mult
    
            scaled.append((g, v))
    
        return scaled

    
    def _physics_off(self) -> bool:
        r"""
        Return ``True`` if physics constraints are effectively disabled.
    
        Physics is considered "off" when ``pde_modes_active`` is a
        list/tuple containing the sentinel value ``"none"``. In that
        case:
    
        * PDE residuals are short-circuited to zero, and
        * physics loss weights are forced to zero in :meth:`compile`.
    
        Returns
        -------
        bool
            ``True`` if PDE constraints should not contribute to the
            loss; ``False`` otherwise.
        """
        return isinstance(self.pde_modes_active, (list, tuple)) and (
            "none" in self.pde_modes_active
        )

    @property
    def mv_lr_mult(self) -> float:
        r"""
        Learning-rate multiplier for :math:`m_v` (via ``log_mv``).
    
        This factor multiplies the gradient of the log-parameter
        ``log_mv`` inside :meth:`_scale_param_grads`, allowing
        :math:`m_v` to learn faster or slower than the rest of the
        network.
    
        Returns
        -------
        float
            Current value of the multiplier for ``log_mv``.
        """
        return self._mv_lr_mult
    
    
    @property
    def kappa_lr_mult(self) -> float:
        r"""
        Learning-rate multiplier for :math:`\kappa` (via ``log_kappa``).
    
        This factor multiplies the gradient of the log-parameter
        ``log_kappa`` inside :meth:`_scale_param_grads`, allowing
        :math:`\kappa` to learn at a different pace than the other
        parameters.
    
        Returns
        -------
        float
            Current value of the multiplier for ``log_kappa``.
        """
        return self._kappa_lr_mult

    def compile(
        self,
        lambda_cons: float = 1.0,
        lambda_gw: float = 1.0,
        lambda_prior: float = 1.0,
        lambda_smooth: float = 1.0,
        lambda_mv: float = 0.0,
        lambda_bounds: float = 0.0,  
        mv_lr_mult: float = 1.0,
        kappa_lr_mult: float = 1.0,
        **kwargs,
    ):
        r"""
        Compile the model and configure physics/data loss weights.
    
        This override extends :meth:`tf.keras.Model.compile` by adding
        weights for the different physics loss components and by
        setting per-parameter learning-rate multipliers for
        :math:`m_v` and :math:`\kappa`.
    
        Parameters
        ----------
        lambda_cons : float, default=1.0
            Weight for the consolidation residual :math:`L_\text{cons}`,
            associated with :math:`R_\text{cons}`.
        lambda_gw : float, default=1.0
            Weight for the groundwater-flow residual :math:`L_\text{gw}`,
            associated with :math:`R_\text{gw}`.
        lambda_prior : float, default=1.0
            Weight for the geomechanical consistency prior
            :math:`L_\text{prior}` (linking :math:`\tau`, :math:`K`,
            :math:`S_s`, :math:`H`).
        lambda_smooth : float, default=1.0
            Weight for the smoothness prior :math:`L_\text{smooth}`,
            based on :math:`\|\nabla K\|^2 + \|\nabla S_s\|^2`.
        lambda_mv : float, default=0.0
            Weight for the storage identity penalty
    
            .. math::
    
                R_{m_v}
                = \log(S_s) - \log(m_v \gamma_w),
    
            which gives :math:`m_v` a direct gradient. Set this to a
            positive value if you want :math:`m_v` to be informed by
            the predicted :math:`S_s`.
        mv_lr_mult : float, default=1.0
            Learning-rate multiplier applied only to the gradient of
            ``log_mv`` (the log-parameter for :math:`m_v`).
        kappa_lr_mult : float, default=1.0
            Learning-rate multiplier applied only to the gradient of
            ``log_kappa`` (the log-parameter for :math:`\kappa`).
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`tf.keras.Model.compile`, e.g. ``optimizer``,
            ``loss``, ``metrics``, etc.
    
        Notes
        -----
        * If :meth:`_physics_off` returns ``True`` (i.e. PDE modes
          contain ``"none"``), the physics weights ``lambda_prior``,
          ``lambda_smooth`` and ``lambda_mv`` are forced to ``0.0``
          regardless of the values passed here. This ensures the
          physics terms do not contribute when physics is disabled.
        * The data loss weight is implicitly ``1.0`` and is defined by
          the loss passed in ``**kwargs``.
        """
        # Let the base class set up optimizer/loss/metrics first
        super().compile(**kwargs)
    
        # Store core physics weights
        self.lambda_cons = float(lambda_cons)
        self.lambda_gw = float(lambda_gw)
    
        if self._physics_off():
            # When physics is off, hard-disable these contributions.
            self.lambda_prior = 0.0
            self.lambda_smooth = 0.0
            self.lambda_mv = 0.0
            self.lambda_bounds = 0.0
        else:
            self.lambda_prior = float(lambda_prior)
            self.lambda_smooth = float(lambda_smooth)
            self.lambda_mv = float(lambda_mv)
            self.lambda_bounds = float(lambda_bounds)
    
        # Per-parameter LR multipliers for log_mv and log_kappa
        self._mv_lr_mult = float(mv_lr_mult)
        self._kappa_lr_mult = float(kappa_lr_mult)


    def get_config(self) -> dict:
        r"""
        Return the full serializable configuration of the model.
    
        This extends :meth:`BaseAttentive.get_config` with additional
        physics-related knobs so that :class:`GeoPriorSubsNet` can be
        saved and restored with :func:`tf.keras.models.clone_model`,
        :func:`tf.keras.models.model_from_json`, or the Keras SavedModel
        machinery.
    
        Returns
        -------
        dict
            A configuration dictionary suitable for reconstruction via
            :meth:`from_config`. It includes:
    
            * all keys from :meth:`BaseAttentive.get_config`,
            * PINN-specific entries:
    
              - ``"output_subsidence_dim"``,
              - ``"output_gwl_dim"``,
              - ``"pde_mode"`` (active PDE modes),
              - ``"mv"`` (possibly a serialized :class:`LearnableMV`),
              - ``"kappa"`` (possibly a serialized :class:`LearnableKappa`),
              - ``"gamma_w"`` and ``"h_ref"`` (fixed physics scalars),
              - ``"scale_pde_residuals"`` and ``"scaling_kwargs"``,
    
            * a lightweight ``"model_version"`` tag for manual inspection.
        """
        base_config = super().get_config()
    
        pinn_config = {
            "output_subsidence_dim": self.output_subsidence_dim,
            "output_gwl_dim": self.output_gwl_dim,
            # Note: pde_modes_active is typically a list; we store it as-is.
            "pde_mode": self.pde_modes_active,
            # These may be wrapper objects (LearnableMV, LearnableKappa,
            # FixedGammaW, FixedHRef) which Keras knows how to serialize.
            "mv": self.mv_config,
            "kappa": self.kappa_config,
            "gamma_w": self.gamma_w_config,
            "h_ref": self.h_ref_config,
            "scale_pde_residuals": self.scale_pde_residuals,
            "scaling_kwargs": self.scaling_kwargs,
            # NEW: keep the effective thickness / κ-mode knobs
            "use_effective_h": self.use_effective_thickness,
            "hd_factor": self.Hd_factor,
            "kappa_mode": self.kappa_mode,
            "model_version": "3.1-GeoPrior",
        }
    
        base_config.update(pinn_config)
        # For backward compatibility with BaseAttentive, keep the original
        # data output dimension visible.
        base_config["output_dim"] = self._data_output_dim
    
        return base_config


    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        r"""
        Reconstruct a :class:`GeoPriorSubsNet` instance from config.
    
        This is the inverse of :meth:`get_config` and is used by the
        Keras serialization utilities. It also handles two important
        concerns:
    
        * Re-hydrating physics parameter wrappers (e.g. ``LearnableMV``)
          from their serialized dict form.
        * Gracefully dropping legacy or intermediate keys that no longer
          belong in the public constructor (older SavedModels / configs).
    
        Parameters
        ----------
        config : dict
            Configuration dictionary produced by :meth:`get_config`,
            possibly extended by Keras when saving/loading. May contain
            serialized versions of ``mv``, ``kappa``, ``gamma_w``,
            ``h_ref`` as nested dicts with ``"class_name"`` and
            ``"config"`` fields.
    
        custom_objects : dict or None, default=None
            Optional mapping of string names to custom classes or
            functions. If ``None``, a fresh dictionary is created and
            populated with the physics parameter wrappers required to
            deserialize the config (e.g. :class:`LearnableMV`,
            :class:`LearnableKappa`, :class:`FixedGammaW`,
            :class:`FixedHRef`, etc.).
    
        Returns
        -------
        GeoPriorSubsNet
            A newly constructed model instance with weights uninitialized
            (to be loaded separately if needed).
    
        Notes
        -----
        * Any legacy keys that refer to older model variants
          (e.g. explicit ``"K"``, ``"Ss"`` or generic
          ``"pinn_coefficient_C"``) are removed to keep the constructor
          signature clean.
        * The ``"output_dim"`` key is also removed because
          :class:`GeoPriorSubsNet` derives it from
          ``output_subsidence_dim`` and ``output_gwl_dim``.
        """
        if custom_objects is None:
            custom_objects = {}
    
        # Register all physics parameter wrappers so that Keras'
        # deserialize_keras_object() can rebuild them from their
        # serialized representation.
        custom_objects.update(
            {
                "LearnableMV": LearnableMV,
                "LearnableKappa": LearnableKappa,
                "FixedGammaW": FixedGammaW,
                "FixedHRef": FixedHRef,
                "LearnableK": LearnableK,
                "LearnableSs": LearnableSs,
                "LearnableQ": LearnableQ,
                "LearnableC": LearnableC,
                "FixedC": FixedC,
                "DisabledC": DisabledC,
            }
        )
    
        # Re-hydrate scalar physics configs if they were serialized as
        # Keras objects (with {"class_name": ..., "config": ...}).
        for key in ("mv", "kappa", "gamma_w", "h_ref"):
            obj = config.get(key)
            if isinstance(obj, dict) and "class_name" in obj:
                config[key] = deserialize_keras_object(
                    obj,
                    custom_objects=custom_objects,
                )
    
        # Drop legacy / internal keys that should not be passed to
        # the public __init__. This keeps backwards compatibility with
        # older SavedModels while avoiding unexpected kwargs.
        config.pop("K", None)
        config.pop("Ss", None)
        config.pop("Q", None)
        config.pop("pinn_coefficient_C", None)
        config.pop("gw_flow_coeffs", None)
        config.pop("output_dim", None)
        config.pop("model_version", None)
    
        return cls(**config)


GeoPriorSubsNet.__doc__ = r""" 
Geomechanical Prior-Informed Subsidence Network (GeoPriorSubsNet)

This model fuses a deep encoder–decoder (from ``BaseAttentive``) with
a physically-sound geomechanical model, addressing key limitations of
simpler PINNs by learning a physically consistent system.

**Model Outputs:**

The network is multi-headed, predicting five distinct fields:
1.  **Subsidence :math:`s(x,y,t)`** (Data, with quantiles)
2.  **Hydraulic Head :math:`h(x,y,t)`** (Data, with quantiles)
3.  **Hydraulic Conductivity :math:`K(x,y)`** (Physics field, positive)
4.  **Specific Storage :math:`S_s(x,y)`** (Physics field, positive)
5.  **Relaxation Time :math:`\tau(x,y)` (Physics field, positive)

**Physics & Loss Function:**

The model is trained by minimizing a composite loss function that
ensures physical consistency, as defined in the revised manuscript:

.. math::
    
    \mathcal{{L}} = \mathcal{{L}}_{{\text{{data}}}} +
                  \lambda_{{gw}} \mathcal{{L}}_{{gw}} +
                  \lambda_{{cons}} \mathcal{{L}}_{{cons}} +
                  \lambda_{{prior}} \mathcal{{L}}_{{prior}} +
                  \lambda_{{smooth}} \mathcal{{L}}_{{smooth}}

Where:

* :math:`\mathcal{{L}}_{{\text{{data}}}}` (Data Fidelity):
  Pinball loss on the predicted quantiles of :math:`s` and :math:`h`.

* :math:`\mathcal{{L}}_{{gw}}` (Groundwater Flow):
  Residual of the 2D transient flow equation. Assumes :math:`Q=0`.
  
  .. math::
      R_{{gw}} = S_s \frac{{\partial h}}{{\partial t}} -
               \nabla \cdot (K \nabla h)

* :math:`\mathcal{{L}}_{{cons}}` (Reduced-Order Consolidation):
  Residual of the 1D relaxation model.
  
  .. math::
      R_{{cons}} = \frac{{\partial s}}{{\partial t}} -
                 \frac{{s_{{eq}}(h) - s}}{{\tau}}
                 
  where:
  .. math::
      s_{{eq}}(h) = m_v \gamma_w (h_{{ref}} - h) H

* :math:`\mathcal{{L}}_{{prior}}` (Geomechanical Consistency Prior):
  Enforces a physical link between the learned fields, addressing
  the non-uniqueness.
  
  .. math::
      R_{{prior}} = \log(\tau) - \log\left(
          \frac{{\bar{{\kappa}} H^2}}{{(\pi^2 K) / S_s}}
      \right)

* :math:`\mathcal{{L}}_{{smooth}}` (Smoothness Prior):
  A regularizer on the spatial gradients of the predicted fields.
  
  .. math::
      R_{{smooth}}= \|\nabla K\|^2_2 + \|\nabla S_s\|^2_2

See :ref:`User Guide <user_guide_geopriorsubsnet>` for a walkthrough.

Parameters
----------
{params.base.static_input_dim}
{params.base.dynamic_input_dim}
{params.base.future_input_dim}

output_subsidence_dim : int, default 1
    Number of subsidence series per horizon step. (Data output :math:`s`)
output_gwl_dim : int, default 1
    Number of hydraulic-head series. (Data output :math:`h`)

forecast_horizon : int, default 1
    Horizon length :math:`H`. The decoder emits :math:`H` steps
    for all data and physics outputs.

quantiles : list[float] | None, default None
    Optional list of quantile levels; enables the Quantile-Distribution
    head for :math:`s` and :math:`h`.

{params.base.embed_dim}
{params.base.hidden_units}
{params.base.lstm_units}
{params.base.attention_units}
{params.base.num_heads}
{params.base.dropout_rate}
{params.base.max_window_size}
{params.base.memory_size}
{params.base.scales}
{params.base.multi_scale_agg}
{params.base.final_agg}
{params.base.activation}
{params.base.use_residuals}
{params.base.use_batch_norm}
{params.base.use_vsn}
{params.base.vsn_units}


pde_mode : {{'consolidation', 'gw_flow', 'both', 'none'}}, default 'both'
    Select which physics residuals participate in the loss.
    (Priors :math:`\mathcal{{L}}_{{prior}}` and 
     :math:`\mathcal{{L}}_{{smooth}}`
    are always active if their lambda weights are > 0).

    ┌─────────────────┬───────────────────────────────────────────────┐
    │ 'consolidation' │ only the **consolidation** term               │
    │                 │ :math:`R_{{cons}} = \dot{{s}} - (s_{{eq}} - s)/\tau` │
    │ 'gw_flow'       │ only the **diffusivity** term                 │
    │                 │ :math:`R_{{gw}} = S_s \partial_t h - \nabla \cdot (K \nabla h)`
    │ 'both'          │ both residuals (recommended)                  │
    │ 'none'          │ pure data-driven (disables :math:`R_{{cons}}`, :math:`R_{{gw}}`) │
    └─────────────────┴───────────────────────────────────────────────┘

mv : float or LearnableMV, default ``LearnableMV(1e-7)``
    Scalar coefficient of volume compressibility :math:`m_v`[Pa⁻¹].
    Used to calculate :math:`s_{{eq}}`.
    Can be a fixed float or a :class:`LearnableMV` instance.

kappa : float or LearnableKappa, default ``LearnableKappa(1.0)``
    Scalar consistency prior parameter :math:`\bar{{\kappa}}` (unitless).
    Used in :math:`\mathcal{{L}}_{{prior}}` to link :math:`\tau, K, S_s, H`.
    Can be a fixed float or a :class:`LearnableKappa` instance.

gamma_w : float or FixedGammaW, default ``FixedGammaW(9810.0)``
    Scalar unit weight of water :math:`\gamma_w` [N m⁻³].
    Used to calculate :math:`s_{{eq}}`.
    Can be a fixed float or a :class:`FixedGammaW` instance.

h_ref : float or FixedHRef, default ``FixedHRef(0.0)``
    Scalar reference head :math:`h_{{ref}}` [m].
    Used to calculate drawdown :math:`\Delta h = h_{{ref}} - h`.
    Can be a fixed float or a :class:`FixedHRef` instance.

use_effective_h : bool, default False
    If ``True``, use an **effective drained thickness** :math:`H_d`
    in place of the physical thickness :math:`H` wherever thickness
    appears in the physics terms (e.g., in :math:`s_{{eq}}` and the
    :math:`\tau` prior). This is useful when only a fraction of the
    layer drains/responds over the forecast horizon.

hd_factor : float, default 1.0
    Multiplicative factor defining the effective thickness
    :math:`H_d = \text{{Hd\_factor}}\, H`. Only used when
    ``use_effective_h=True``. Typical values lie in :math:`(0, 1]`,
    where lower values model partially draining layers.

kappa_mode : {{'bar', 'kb'}}, default 'bar'
    Interpretation of :math:`\kappa` in the geomechanical-consistency
    prior.
    
    - ``'bar'`` – bulk calibration :math:`\bar{{\kappa}}` with thickness
      :math:`H_*` (which equals :math:`H_d` if
      ``use_effective_h=True``, else :math:`H`):
      
      .. math::
         R_{{\text{{prior}}}} = \log \tau - \log \left(
           \frac{{\bar{{\kappa}}\, H_*^{{2}}\, S_s}}{{\pi^{{2}}\, K}} \right)
    
    - ``'kb'`` – boundary factor :math:`\kappa_b` (same functional
      form, different physical meaning). Also pairs :math:`\kappa_b`
      with :math:`H_*` as above:
      
      .. math::
         R_{{\text{{prior}}}} = \log \tau - \log \left(
           \frac{{\kappa_b\, H_*^{{2}}\, S_s}}{{\pi^{{2}}\, K}} \right)

scale_pde_residuals : bool, default True
    If ``True``, non-dimensionalize physics residuals 
    (:math:`R_{{gw}}`, :math:`R_{{cons}}`)
    with simple data-driven scales so the loss terms are
    :math:`\mathcal{{O}}(1)`.
    (See :func:`fusionlab.nn.pinn.op.default_scales`).

scaling_kwargs : dict | None, default None
    Extra keyword arguments forwarded to :func:`default_scales`.

mode : {{'pihal_like', 'tft_like'}}, default ``None``
    Routing for *future_features*:
        
    * **pihal_like** – decoder gets all :math:`H` rows, encoder none.
    * **tft_like** – first *max_window_size* rows to encoder,
      next :math:`H` rows to decoder. ``None`` inherits
      BaseAttentive default ('tft_like').

objective : {{'hybrid', 'transformer'}}, default ``'hybrid'``
    Selects the backbone architecture. (See `BaseAttentive` docs).
    
attention_levels : str | list[str] | None
    Controls the attention layers used in the decoder.
    (See `BaseAttentive` docs).

name : str, default "GeoPriorSubsNet"
    Model scope as registered in Keras.

**kwargs
    Forwarded verbatim to :class:`tf.keras.Model`.

Notes
-----
* **Required Input:** The `inputs` dictionary for `call` and
    `train_step` **must** include an `H_field` (or `soil_thickness`)
    tensor of shape `(B, H, 1)` representing the soil thickness.
* **Loss Weights:** The composite loss is controlled by four
    weights passed to :meth:`compile`:
    `lambda_cons`, `lambda_gw`, `lambda_prior`, `lambda_smooth`.
* **Outputs:** The model's `call` method returns a dictionary with
    keys: `data_final` (for :math:`s, h` quantiles), `data_mean` (for $s, h$
    means), and `phys_mean_raw` (for raw :math:`K, S_s, \tau` logits).

See Also
--------
fusionlab.nn.models.BaseAttentive
    The data-driven backbone for this model.
fusionlab.nn.pinn.models.TransFlowSubsNet
    The previous, simpler PINN model with scalar physics parameters.
fusionlab.params.LearnableMV
    Parameter class for learnable :math:`m_v`.
fusionlab.params.LearnableKappa
    Parameter class for learnable :math:`\bar{{\kappa}}`.
fusionlab.nn.pinn.op.process_pinn_inputs
    Utility for unpacking input dictionaries.

Examples
--------
>>> import tensorflow as tf
>>> from fusionlab.nn.pinn import GeoPriorSubsNet
>>> from fusionlab.params import LearnableMV, LearnableKappa
>>>
>>> B, T, H = 8, 12, 6 # Batch, Timesteps, Horizon
>>>
>>> model = GeoPriorSubsNet(
...     static_input_dim=3, dynamic_input_dim=8, future_input_dim=4,
...     output_subsidence_dim=1, output_gwl_dim=1,
...     forecast_horizon=H, max_window_size=T,
...     mv=LearnableMV(1e-8),       # Pass new scalar params
...     kappa=LearnableKappa(1.2),  # (K, Ss, Q are removed)
...     pde_mode='both',
...     scale_pde_residuals=True
... )
>>>
>>> # Note: future_features length depends on 'mode'
>>> # Default mode 'tft_like' requires T + H = 12 + 6 = 18 steps
>>> batch = {{
...     "static_features":  tf.zeros([B, 3]),
...     "dynamic_features": tf.zeros([B, T, 8]),
...     "future_features":  tf.zeros([B, T + H, 4]),
...     "coords":           tf.zeros([B, H, 3]),
...     "H_field":          tf.ones([B, H, 1]) * 20.0 # Soil thickness
... }}
>>>
>>> # Compile with new loss weights
>>> model.compile(
...     optimizer='adam',
...     loss='mae', # Data loss (will be wrapped)
...     lambda_cons=1.0,
...     lambda_gw=1.0,
...     lambda_prior=0.5,
...     lambda_smooth=0.1
... )
>>>
>>> # Call returns a dictionary of all outputs
>>> out = model(batch, training=False)
>>> sorted(out.keys())
['data_final', 'data_mean', 'phys_mean_raw']
>>> out['data_mean'].shape
TensorShape([8, 6, 2])
>>> out['phys_mean_raw'].shape
TensorShape([8, 6, 3])
""".format(params=_param_docs)