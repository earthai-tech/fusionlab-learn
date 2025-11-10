# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Pytest for the GeoPriorSubsNet model to ensure the computational
graph is correctly connected and the train_step executes without errors.
"""

import pytest
import tensorflow as tf
import numpy as np


from fusionlab.nn.pinn._geoprior_subnet import GeoPriorSubsNet
from fusionlab.params import (
    LearnableMV, LearnableKappa, FixedGammaW, FixedHRef
)
from fusionlab.nn.pinn.op import extract_physical_parameters

# --- Test Configuration ---
B = 4  # Batch size
T = 10 # Time steps (lookback)
H = 5  # Forecast horizon
S_DIM = 3  # Static features dim
D_DIM = 4  # Dynamic features dim
F_DIM = 2  # Future features dim
O_SUBS_DIM = 1 # Output subsidence dim
O_GWL_DIM = 1  # Output GWL dim
MODEL_MODE = 'tft_like' # Match data shape T+H

# --- Define Initial Values for Testing ---
INIT_MV = 1e-7
INIT_KAPPA = 1.0
INIT_GAMMA_W = 9810.0
INIT_H_REF = 0.0

@pytest.fixture(scope="module")
def dummy_data_batch():
    """
    Creates a single batch of dummy data matching the
    exact dictionary structure expected by the model's call and train_step.
    """
    # 1. Inputs Dictionary
    inputs = {
        # Coords (t,x,y) for the forecast horizon
        'coords': tf.random.normal((B, H, 3), dtype=tf.float32),
        
        # Static features (time-invariant)
        'static_features': tf.random.normal((B, S_DIM), dtype=tf.float32),
        
        # Dynamic features (past lookback)
        'dynamic_features': tf.random.normal((B, T, D_DIM), dtype=tf.float32),
        
        # Future features (lookback + horizon for tft_like mode)
        'future_features': tf.random.normal((B, T + H, F_DIM), dtype=tf.float32),
        
        # H_field (soil thickness, static per sample, tiled over horizon)
        'H_field': tf.random.uniform((B, H, 1), minval=10., maxval=50., dtype=tf.float32)
    }
    
    # 2. Targets Dictionary
    targets = {
        # Key 'subs_pred' matches the compile(loss={'subs_pred': ...})
        'subs_pred': tf.random.normal((B, H, O_SUBS_DIM), dtype=tf.float32),
        
        # Key 'gwl_pred' matches the compile(loss={'gwl_pred': ...})
        'gwl_pred': tf.random.normal((B, H, O_GWL_DIM), dtype=tf.float32)
    }
    
    return inputs, targets

@pytest.fixture(scope="module")
def configured_model():
    """
    Instantiates the GeoPriorSubsNet model with parameters
    matching the dummy_data_batch.
    """
    model = GeoPriorSubsNet(
        static_input_dim=S_DIM,
        dynamic_input_dim=D_DIM,
        future_input_dim=F_DIM,
        output_subsidence_dim=O_SUBS_DIM,
        output_gwl_dim=O_GWL_DIM,
        forecast_horizon=H,
        max_window_size=T,
        mode=MODEL_MODE,
        quantiles=None, # Use simple MSE for testing
        pde_mode='both',
        # *** Use specific initial values for testing ***
        mv=LearnableMV(initial_value=INIT_MV),
        kappa=LearnableKappa(initial_value=INIT_KAPPA),
        gamma_w=FixedGammaW(value=INIT_GAMMA_W),
        h_ref=FixedHRef(value=INIT_H_REF)
    )
    return model

def test_model_instantiation(configured_model):
    """Test that the model can be instantiated."""
    assert isinstance(configured_model, GeoPriorSubsNet)
    assert configured_model.name == "GeoPriorSubsNet"

def test_call_method_connects_graph(configured_model, dummy_data_batch):
    """
    Directly tests that the (t,x,y) coordinates have a gradient
    path to the physics outputs (K, Ss, tau).
    
    This test specifically targets the `ValueError: ... gradients are None`.
    """
    model = configured_model
    inputs, _ = dummy_data_batch
    
    # We must watch the 'coords' tensor specifically
    assert 'coords' in inputs, "Dummy data fixture is missing 'coords'"
    
    with tf.GradientTape() as tape:
        # Watch the coordinate input
        tape.watch(inputs['coords'])
        
        # Perform a forward pass
        outputs_dict = model(inputs, training=True)
        
        # Get the raw output from the physics head
        assert 'phys_mean_raw' in outputs_dict, "Model output missing 'phys_mean_raw'"
        physics_output = outputs_dict['phys_mean_raw'] # Shape (B, H, 3)

    # Compute the gradient from physics output back to coords input
    grads = tape.gradient(physics_output, inputs['coords'])
    
    # The test: If the graph is connected, grads will be a Tensor, not None.
    assert grads is not None, \
        "Gradient of 'phys_mean_raw' w.r.t 'coords' is None. " \
        "The computational graph is broken."
    
    assert grads.shape == inputs['coords'].shape, \
        f"Gradient shape {grads.shape} does not match " \
        f"coords shape {inputs['coords'].shape}"

def test_train_step_runs(configured_model, dummy_data_batch):
    """
    Tests that a single train_step executes without raising any errors,
    specifically the ValueError for None gradients.
    """
    model = configured_model
    data = dummy_data_batch # data is (inputs_dict, targets_dict)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss={
            'subs_pred': 'mse',
            'gwl_pred': 'mse'
        },
        lambda_cons=1.0,
        lambda_gw=1.0,
        lambda_prior=0.1,
        lambda_smooth=0.1
    )
    
    # Run the train_step
    # If this raises the ValueError, the test will fail
    try:
        logs = model.train_step(data)
    except Exception as e:
        pytest.fail(f"model.train_step() raised an unexpected error: {e}")
        
    # Check that logs are returned correctly
    assert isinstance(logs, dict)
    assert 'total_loss' in logs
    assert 'physics_loss' in logs
    assert 'consolidation_loss' in logs
    assert 'gw_flow_loss' in logs
    assert 'prior_loss' in logs
    assert 'smooth_loss' in logs
    assert 'data_loss' in logs
    assert 'subs_pred_loss' in logs # From the compiled loss
    assert 'gwl_pred_loss' in logs  # From the compiled loss

def test_physics_parameters_are_learning(configured_model, dummy_data_batch):
    """
    Tests that learnable parameters are updated after a few training steps.
    We check the *field* parameters (K, Ss, tau) which are direct NN
    outputs and should change immediately.
    We also check that *fixed* scalar parameters (gamma_w, h_ref) do not change.
    """
    model = configured_model
    inputs, targets = dummy_data_batch
    data = (inputs, targets)
    
    # Compile the model with all physics losses active
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), # Use a higher LR for test
        loss={'subs_pred': 'mse', 'gwl_pred': 'mse'},
        lambda_cons=1.0,
        lambda_gw=1.0,
        lambda_prior=1.0,
        lambda_smooth=1.0
    )
    
    # --- Check initial fixed scalar values ---
    # We can still check that fixed params have the right initial value
    initial_params = extract_physical_parameters(model)
    assert initial_params['Unit_Weight_Water_gamma_w'] == pytest.approx(INIT_GAMMA_W)
    assert initial_params['Reference_Head_h_ref'] == pytest.approx(INIT_H_REF)
    
    # --- Check initial predicted field values ---
    outputs_before = model(inputs, training=False)
    # Use the raw values before positivity for a cleaner comparison
    K_raw_before = outputs_before["phys_mean_raw"][..., :model.output_K_dim]
    Ss_raw_before = outputs_before["phys_mean_raw"][..., model.output_K_dim:model.output_K_dim+model.output_Ss_dim]
    tau_raw_before = outputs_before["phys_mean_raw"][..., model.output_K_dim+model.output_Ss_dim:]

    initial_K_raw_mean = tf.reduce_mean(K_raw_before).numpy()
    initial_Ss_raw_mean = tf.reduce_mean(Ss_raw_before).numpy()
    initial_tau_raw_mean = tf.reduce_mean(tau_raw_before).numpy()

    # --- Run a few training steps ---
    num_steps = 20 # Keep 20 steps
    print(f"\nRunning {num_steps} training steps to update weights...")
    for i in range(num_steps):
        logs = model.train_step(data)
        print(f"Step {i+1}/{num_steps} - total_loss: {logs['total_loss']:.4f}")
        
    # --- 1. Check Fixed Scalar Parameters ---
    print("Extracting parameters after training...")
    final_params = extract_physical_parameters(model)
    
    # Check that fixed scalar parameters have *not* changed
    assert final_params['Unit_Weight_Water_gamma_w'] == pytest.approx(INIT_GAMMA_W), \
        "Fixed parameter 'gamma_w' changed during training."
    assert final_params['Reference_Head_h_ref'] == pytest.approx(INIT_H_REF), \
        "Fixed parameter 'h_ref' changed during training."

    # --- 2. Check Predicted Fields (K, Ss, tau) ---
    # This is the most important check: are the network's outputs changing?
    print("Checking predicted physics fields after training...")
    outputs_after = model(inputs, training=False)
    
    # Compare the raw logits from the physics head
    K_raw_after = outputs_after["phys_mean_raw"][..., :model.output_K_dim]
    Ss_raw_after = outputs_after["phys_mean_raw"][..., model.output_K_dim:model.output_K_dim+model.output_Ss_dim]
    tau_raw_after = outputs_after["phys_mean_raw"][..., model.output_K_dim+model.output_Ss_dim:]

    # Assert that the mean of the raw fields has changed
    assert not np.isclose(tf.reduce_mean(K_raw_after).numpy(), initial_K_raw_mean), \
        "Predicted field 'K' (raw) did not change from its initial value."
    assert not np.isclose(tf.reduce_mean(Ss_raw_after).numpy(), initial_Ss_raw_mean), \
        "Predicted field 'Ss' (raw) did not change from its initial value."
    assert not np.isclose(tf.reduce_mean(tau_raw_after).numpy(), initial_tau_raw_mean), \
        "Predicted field 'tau' (raw) did not change from its initial value."

    print("All physics parameter update tests passed.")