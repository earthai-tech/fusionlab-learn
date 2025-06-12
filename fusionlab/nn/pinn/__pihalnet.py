# --- Example of Refactored PIHALNet ---
# -*- coding: utf-8 -*-
# File: fusionlab/nn/pinn/models.py
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Physics-Informed Hybrid Attentive LSTM Network (PIHALNet).
"""
from __future__ import annotations
from typing import List, Optional, Union, Dict, Any

from ...params import LearnableC, FixedC, DisabledC
from ...utils.generic_utils import select_mode
from ..models._base_attentive import BaseAttentive # Inherit from base
from .op import compute_consolidation_residual
from .utils import process_pde_modes, process_pinn_inputs

from .. import KERAS_BACKEND, KERAS_DEPS
if KERAS_BACKEND:
    import tensorflow as tf
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.optimizers import Adam
    Tensor = KERAS_DEPS.Tensor
    GradientTape = KERAS_DEPS.GradientTape
    tf_exp = tf.exp
    tf_square = tf.square
    tf_reduce_mean = tf.reduce_mean
    tf_zeros_like = tf.zeros_like
else:
    class Model: pass
    Tensor = Any

__all__ = ["PIHALNet"]

@KERAS_DEPS.register_keras_serializable(
    'fusionlab.nn.pinn', name="PIHALNet"
)
class PIHALNet(BaseAttentive):
    """
    Physics-Informed Hybrid Attentive LSTM Network (PIHALNet).

    This model extends the data-driven BaseAttentive architecture by
    incorporating physics-informed constraints from the governing
    equations of land subsidence.
    """
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_subsidence_dim: int = 1,
        output_gwl_dim: int = 1,
        pde_mode: str = 'consolidation',
        pinn_coefficient_C: Any = 'learnable',
        gw_flow_coeffs: Optional[Dict] = None,
        **kwargs
    ):
        # The output_dim for the base attentive model is the sum of
        # the two target dimensions.
        combined_output_dim = output_subsidence_dim + output_gwl_dim

        # Pass all shared parameters to the BaseAttentive parent class.
        super().__init__(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            output_dim=combined_output_dim,
            name="PIHALNet",
            **kwargs
        )
        # Initialize only PINN-specific attributes
        self.output_subsidence_dim = output_subsidence_dim
        self.output_gwl_dim = output_gwl_dim
        self.pde_modes_active = process_pde_modes(
            pde_mode, enforce_consolidation=True)
        self.pinn_coefficient_C_config = self._normalize_C_descriptor(
            pinn_coefficient_C)
        self.gw_flow_coeffs_config = gw_flow_coeffs or {}
        
        # Build the physics-related components
        self._build_pinn_components()

    def _normalize_C_descriptor(self, C_config):
        """Normalizes various C inputs into a consistent class format."""
        if isinstance(C_config, (LearnableC, FixedC, DisabledC)):
            return C_config
        if C_config == 'learnable':
            return LearnableC()
        if isinstance(C_config, (float, int)):
            return FixedC(value=C_config)
        return DisabledC()

    def _build_pinn_components(self):
        """Instantiates trainable/fixed physical coefficients."""
        if isinstance(self.pinn_coefficient_C_config, LearnableC):
            self.log_C_coefficient = self.add_weight(
                name="log_pinn_coefficient_C",
                shape=(),
                initializer=tf.keras.initializers.Constant(
                    tf.math.log(self.pinn_coefficient_C_config.initial_value)
                ),
                trainable=True,
            )
        # Note: FixedC and DisabledC do not require tf.Variables.

    def get_pinn_coefficient_C(self) -> Tensor:
        """Returns the physical coefficient C as a tensor."""
        if isinstance(self.pinn_coefficient_C_config, LearnableC):
            return tf_exp(self.log_C_coefficient)
        elif isinstance(self.pinn_coefficient_C_config, FixedC):
            return tf.constant(
                self.pinn_coefficient_C_config.value, dtype=tf.float32)
        else: # DisabledC or None
            return tf.constant(1.0, dtype=tf.float32)

    def call(self, inputs: Dict[str, Tensor], training: bool = False):
        """Forward pass for PIHALNet."""
        # 1. Get the data-driven predictions from the base model
        # The base call handles validation and the core encoder-decoder
        base_outputs = super().call(inputs, training=training)
        
        # `base_outputs` contains the combined predictions.
        # We need both the final (potentially quantile) predictions for
        # data loss and the mean predictions for the PDE residual.
        # Let's assume the base model's `multi_decoder` provides the means.
        decoded_means = self.multi_decoder(
            base_outputs['final_features'], training=training)
        
        # The final predictions come from the quantile layer
        final_predictions = self.quantile_distribution_modeling(
            decoded_means, training=training)

        # 2. Split outputs into subsidence and GWL for both branches
        (s_pred_final, gwl_pred_final,
          s_pred_mean_for_pde, gwl_pred_mean_for_pde) = self.split_outputs(
              predictions_combined=final_predictions,
              decoded_outputs_for_mean=decoded_means
          )

        # 3. Compute the physics residual
        t, _, _ = process_pinn_inputs(inputs, mode='as_dict')
        
        pde_residual = compute_consolidation_residual(
            s_pred=s_pred_mean_for_pde,
            h_pred=gwl_pred_mean_for_pde,
            time_steps=t, # `t` must be the forecast window coordinates
            C=self.get_pinn_coefficient_C()
        )
        
        # 4. Return the combined dictionary for the loss function
        return {
            "subs_pred": s_pred_final,
            "gwl_pred": gwl_pred_final,
            "pde_residual": pde_residual,
        }

    def train_step(self, data):
        """Custom training step for the composite PINN loss."""
        inputs, targets = data
        with GradientTape() as tape:
            outputs = self(inputs, training=True)
            
            y_pred_for_loss = {
                'subs_pred': outputs['subs_pred'],
                'gwl_pred': outputs['gwl_pred']
            }
            data_loss = self.compute_loss(
                x=inputs, y=targets, y_pred=y_pred_for_loss,
                regularization_losses=self.losses
            )
            
            loss_pde = tf_reduce_mean(tf_square(outputs['pde_residual']))
            total_loss = data_loss + self.lambda_pde * loss_pde

        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        self.compiled_metrics.update_state(targets, y_pred_for_loss)
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "total_loss": total_loss,
            "data_loss": data_loss,
            "physics_loss": loss_pde,
        })
        return results
        
    def test_step(self, data):
        # Similar logic as train_step but without gradient updates
        inputs, targets = data
        outputs = self(inputs, training=False)
        y_pred_for_loss = {
            'subs_pred': outputs['subs_pred'],
            'gwl_pred': outputs['gwl_pred']
        }
        data_loss = self.compute_loss(
            x=inputs, y=targets, y_pred=y_pred_for_loss,
            regularization_losses=self.losses
        )
        loss_pde = tf.reduce_mean(tf.square(outputs['pde_residual']))
        total_loss = data_loss + self.lambda_pde * loss_pde
        
        self.compiled_metrics.update_state(targets, y_pred_for_loss)
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "total_loss": total_loss,
            "data_loss": data_loss,
            "physics_loss": loss_pde,
        })
        return results

    def get_config(self):
        """Returns the full configuration of the PIHALNet model."""
        # Get the config from the base class first
        base_config = super().get_config()
        # Add PIHALNet-specific parameters
        base_config.update({
            "output_subsidence_dim": self.output_subsidence_dim,
            "output_gwl_dim": self.output_gwl_dim,
            "pde_mode": self.pde_modes_active,
            "pinn_coefficient_C": self.pinn_coefficient_C_config,
            "gw_flow_coeffs": self.gw_flow_coeffs_config
        })
        # The base config already contains all other necessary params
        return base_config

