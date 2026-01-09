# -*- coding: utf-8 -*-
from __future__ import annotations
from .. import KERAS_DEPS

tf_clip_by_value = KERAS_DEPS.clip_by_value
tf_math = KERAS_DEPS.math 
tf_where = KERAS_DEPS.where 
tf_stop_gradient = KERAS_DEPS.stop_gradient 
tf_constant = KERAS_DEPS.constant 
tf_minimum = KERAS_DEPS.minimum 
tf_maximum = KERAS_DEPS.maximum 
tf_cast = KERAS_DEPS.cast 
tf_ones_like = KERAS_DEPS.ones_like 
tf_float32 = KERAS_DEPS.float32 
tf_zeros_like = KERAS_DEPS.zeros_like 
tf_IndexedSlices = KERAS_DEPS.IndexedSlices


def clamp_physics_logits(
    K_logits, 
    Ss_logits, 
    dlogtau_logits, 
    Q_logits=None, 
    clip_min=-15.0, 
    clip_max=15.0
):
    """
    (Fix A) Clamps raw logits to prevent exponential explosion/underflow 
    in the physics layer.
    
    Range [-15, 15] corresponds to exp(-15) ~ 3e-7 and exp(15) ~ 3e6.
    """
    K_logits = tf_clip_by_value(K_logits, clip_min, clip_max)
    Ss_logits = tf_clip_by_value(Ss_logits, clip_min, clip_max)
    dlogtau_logits = tf_clip_by_value(dlogtau_logits, clip_min, clip_max)
    
    if Q_logits is not None:
        # Q is linear (source term), usually needs a wider range but still finite
        Q_logits = tf_clip_by_value(Q_logits, -1e5, 1e5)
        
    return K_logits, Ss_logits, dlogtau_logits, Q_logits


def sanitize_scales(scales, min_scale=1e-6, max_scale=1e6):
    """
    (Fix B) Replaces NaN/Inf values in dynamic scaling factors with 1.0 
    and clamps extreme values.
    """
    clean_scales = {}
    for k, v in scales.items():
        # 1. Stop gradient (crucial for scaling factors)
        v_stopped = tf_stop_gradient(v)
        
        # 2. Detect NaN or Inf
        is_bad = tf_math.logical_or(
            tf_math.is_nan(v_stopped), 
            tf_math.is_inf(v_stopped)
        )
        
        # 3. Replace bad values with 1.0 (neutral scaling)
        v_safe = tf_where(is_bad, tf_ones_like(v_stopped), v_stopped)
        
        # 4. Clamp to prevent numerical instability in loss calculation
        v_clamped = tf_clip_by_value(v_safe, min_scale, max_scale)
        
        clean_scales[k] = v_clamped
        
    return clean_scales


def compute_physics_warmup_gate(step_tensor, warmup_steps=500, ramp_steps=500):
    """
    (Fix C) Returns a scalar 0.0 -> 1.0 multiplier based on global step.
    
    Logic:
      - step < warmup: 0.0
      - warmup < step < warmup+ramp: linear ramp 0->1
      - step > warmup+ramp: 1.0
    """
    step = tf_cast(step_tensor, tf_float32)
    w_steps = tf_constant(warmup_steps, tf_float32)
    r_steps = tf_constant(ramp_steps, tf_float32)
    
    # Linear ramp calculation
    gate = (step - w_steps) / r_steps
    
    # Clip between 0.0 and 1.0
    return tf_minimum(tf_maximum(gate, 0.0), 1.0)


def _sanitize_grad(g):
    if g is None:
        return None

    # Handle sparse grads (very common with embeddings)
    if isinstance(g, tf_IndexedSlices):
        vals = g.values
        bad = tf_math.logical_or(tf_math.is_nan(vals), tf_math.is_inf(vals))
        vals = tf_where(bad, tf_zeros_like(vals), vals)
        return tf_IndexedSlices(vals, g.indices, g.dense_shape)

    bad = tf_math.logical_or(tf_math.is_nan(g), tf_math.is_inf(g))
    return tf_where(bad, tf_zeros_like(g), g)


def filter_nan_gradients(grads):
    return [_sanitize_grad(g) for g in grads]

# def filter_nan_gradients(grads):
#     """
#     (Fix D) Replaces None or non-finite gradients with zeros to prevent 
#     weight corruption during a bad batch.
#     """
#     safe_grads = []
#     for g in grads:
#         if g is None:
#             safe_grads.append(g)
#             continue
            
#         # Check finite
#         is_finite = tf_math.is_finite(g)
        
#         # If any element is nan/inf, this might be aggressive, but often 
#         # we want to mask strictly the bad values:
#         g_safe = tf_where(is_finite, g, tf_zeros_like(g))
        
#         safe_grads.append(g_safe)
        
#     return safe_grads