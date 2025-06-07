import numpy as np
import tensorflow as tf

from fusionlab.nn.pinn.models import PIHALNet
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset, AUTOTUNE

# Toy configuration
batch_size       = 4
time_steps       = 5   # look-back window (max_window_size)
forecast_horizon = 5   # forecast horizon
static_input_dim = 2
dynamic_input_dim= 3
future_input_dim = 1

fixed_model_params = {
    "static_input_dim":      static_input_dim,
    "dynamic_input_dim":     dynamic_input_dim,
    "future_input_dim":      future_input_dim,
    "output_subsidence_dim": 1,
    "output_gwl_dim":        1,
    "forecast_horizon":      forecast_horizon,
    "quantiles":             None,
    "max_window_size":       time_steps,
    "memory_size":           10,
    "scales":                [1],
    "multi_scale_agg":       "last",
    "final_agg":             "last",
    "use_residuals":         True,
    "use_batch_norm":        False,
    "pde_mode":              "consolidation",
    "pinn_coefficient_C":    "learnable",
    "gw_flow_coeffs":        None,
    "use_vsn":               False
}

# Instantiate PIHALNet
model = PIHALNet(
    **fixed_model_params,
    embed_dim=16,
    hidden_units=16,
    lstm_units=16,
    attention_units=8,
    num_heads=2,
    dropout_rate=0.1,
    activation="relu"
)

model.compile(
    optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
    loss={
        "subs_pred": MeanSquaredError(name="subs_data_loss"),
        "gwl_pred":  MeanSquaredError(name="gwl_data_loss"),
    },
    metrics={
        "subs_pred": [MeanAbsoluteError(name="subs_mae")],
        "gwl_pred":  [MeanAbsoluteError(name="gwl_mae")],
    },
    loss_weights={"subs_pred": 1.0, "gwl_pred": 0.8},
    lambda_pde=0.1
)

# Generate random toy data
coords = np.random.rand(batch_size, time_steps, 3).astype("float32")
static_features  = np.random.rand(batch_size, static_input_dim).astype("float32")
dynamic_features = np.random.rand(batch_size, time_steps, dynamic_input_dim).astype("float32")

# Crucial fix: time tensor for future must match forecast_horizon
# For example, assume yearly steps [t0, t0+1, t0+2, …]
# We simulate 'future_time' going from 0…H-1 for each example:
future_time = np.tile(
    np.arange(forecast_horizon, dtype="float32").reshape(1, forecast_horizon, 1),
    (batch_size, 1, 1)
)

future_features = np.random.rand(batch_size, forecast_horizon, future_input_dim).astype("float32")

# Targets (batch, forecast_horizon, 1)
subs_targets = np.random.rand(batch_size, forecast_horizon, 1).astype("float32")
gwl_targets = np.random.rand(batch_size, forecast_horizon, 1).astype("float32")

# Package inputs exactly as PIHALNet expects:
inputs = {
    "coords":             coords,
    "static_features":    static_features,
    "dynamic_features":   dynamic_features,
    "future_features":    future_features,
    "time_steps":         future_time       # <— pass future_time here
}
targets = {
    "subs_pred": subs_targets,
    "gwl_pred":  gwl_targets,
}

dataset = Dataset.from_tensor_slices((inputs, targets))\
                 .batch(batch_size)\
                 .prefetch(AUTOTUNE)

print("Running model.fit() on corrected toy data for 1 epoch …")
model.fit(dataset, epochs=1)
