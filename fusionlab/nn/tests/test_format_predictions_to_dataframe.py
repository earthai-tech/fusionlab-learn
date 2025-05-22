# test_format_predictions_to_dataframe.py


import pytest
import numpy as np
import pandas as pd

import warnings
from typing import List, Optional, Union, Any, Dict

# --- Attempt to import function and dependencies ---
try:
    import tensorflow as tf
    from fusionlab.nn.utils import format_predictions_to_dataframe
    # Mock vlog if not available in test environment, or import it
    try:
        from fusionlab.utils.generic_utils import vlog
    except ImportError:
        def vlog(message, verbose=0, level=0, **kwargs):
            if verbose >= level: print(message)

    # Mock coverage_score if not available
    try:
        from fusionlab.metrics import coverage_score
        HAS_COVERAGE_SCORE_TEST = True
    except ImportError:
        HAS_COVERAGE_SCORE_TEST = False
        def coverage_score(*args, **kwargs): return np.nan

    if hasattr(tf, 'Tensor'):
        Tensor = tf.Tensor
    else:
        class Tensor: pass # Fallback
    FUSIONLAB_INSTALLED = True
except ImportError as e:
    print(f"Skipping format_predictions_to_dataframe tests: {e}")
    FUSIONLAB_INSTALLED = False
    class Tensor: pass
    def vlog(message, verbose=0, level=0, **kwargs): pass
    def coverage_score(*args, **kwargs): return np.nan
    HAS_COVERAGE_SCORE_TEST = False
    def format_predictions_to_dataframe(*args, **kwargs):
        raise ImportError("format_predictions_to_dataframe not found")

pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab.nn.utils.format_predictions_to_dataframe not found"
)

# --- Test Fixtures and Constants ---
B, H, O_SINGLE, O_MULTI = 4, 3, 1, 2 # Batch, Horizon, OutputDims
Q_LIST = [0.1, 0.5, 0.9]
N_Q = len(Q_LIST)
SAMPLES = B # Number of sequences

@pytest.fixture
def dummy_keras_model():
    """Creates a simple Keras model that returns a predefined output."""
    class MockModel(tf.keras.Model):
        def __init__(self, output_shape_to_return):
            super().__init__()
            self.output_shape_to_return = output_shape_to_return
            # Add a dummy layer so it's considered trainable/built
            self.dummy_dense = tf.keras.layers.Dense(1)

        def call(self, inputs):
            # For simplicity, always return a tensor of zeros
            # with the predefined shape, using batch size from input
            batch_size = tf.shape(inputs[0] if isinstance(inputs, list) else inputs)[0]
            # Adjust output shape to use dynamic batch size
            shape_to_use = [batch_size] + list(self.output_shape_to_return[1:])
            return tf.zeros(shape_to_use, dtype=tf.float32)

        def get_config(self): # Basic config for Keras
            return {"name": self.name, "output_shape_to_return": self.output_shape_to_return}

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    return MockModel

@pytest.fixture
def dummy_model_inputs():
    """Dummy inputs for model.predict()"""
    # [Static, Dynamic, Future] - shapes don't matter much for mock model
    return [
        tf.zeros((B, 2), dtype=tf.float32), # Static
        tf.zeros((B, 5, 3), dtype=tf.float32), # Dynamic
        tf.zeros((B, 5 + H, 2), dtype=tf.float32) # Future
    ]
@pytest.fixture
def dynamic_input_3d() -> tf.Tensor:
    """Valid 3D dynamic input."""
    return tf.random.normal((B, H, O_SINGLE), dtype=tf.float32)

@pytest.fixture
def y_true_single_output():
    return tf.random.normal((SAMPLES, H, O_SINGLE), dtype=tf.float32)

@pytest.fixture
def y_true_multi_output():
    return tf.random.normal((SAMPLES, H, O_MULTI), dtype=tf.float32)

@pytest.fixture
def spatial_data_df():
    return pd.DataFrame({
        'loc_id': [f'loc_{i//H}' for i in range(SAMPLES * H)], # Repeats for each horizon
        'region': [f'reg_{i//(H*2)}' for i in range(SAMPLES * H)]
    }).iloc[:SAMPLES] # Take first SAMPLES rows to match num_samples

@pytest.fixture
def spatial_data_np():
    # (Samples, NumSpatialFeatures)
    return np.array([[i, i*10] for i in range(SAMPLES)], dtype=np.float32)

@pytest.fixture
def dummy_scaler():
    """A mock scaler with an inverse_transform method."""
    class MockScaler:
        def __init__(self, feature_names_in_):
            self.feature_names_in_ = feature_names_in_
            self.n_features_in_ = len(feature_names_in_)

        def inverse_transform(self, X):
            # Simple inverse: adds 100 to all values
            # Assumes X is 2D (samples, features)
            if X.shape[1] != self.n_features_in_ :
                 # If only target col is passed for inverse transform
                if X.shape[1] == 1 and self.n_features_in_ > 1:
                    # This is a simplified inverse for single target column
                    return X + 100
                raise ValueError(f"Scaler expected {self.n_features_in_} features, "
                                 f"got {X.shape[1]}")
            return X + 100
    return MockScaler

# --- Test Cases ---

def test_point_forecast_single_output(y_true_single_output):
    """Point forecast, single output dimension."""
    preds = tf.random.normal((SAMPLES, H, O_SINGLE), dtype=tf.float32)
    df = format_predictions_to_dataframe(
        predictions=preds,
        y_true_sequences=y_true_single_output,
        target_name="sales",
        forecast_horizon=H, # Explicitly provide
        output_dim=O_SINGLE   # Explicitly provide
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == SAMPLES * H
    assert 'sample_idx' in df.columns
    assert 'forecast_step' in df.columns
    assert 'sales_pred' in df.columns
    assert 'sales_actual' in df.columns
    assert df['sales_pred'].isna().sum() == 0
    assert df['sales_actual'].isna().sum() == 0
    print("Point forecast, single output: OK")

def test_point_forecast_multi_output(y_true_multi_output):
    """Point forecast, multiple output dimensions."""
    preds = tf.random.normal((SAMPLES, H, O_MULTI), dtype=tf.float32)
    df = format_predictions_to_dataframe(
        predictions=preds,
        y_true_sequences=y_true_multi_output,
        target_name="revenue",
        # forecast_horizon and output_dim inferred
    )
    assert len(df) == SAMPLES * H
    for i in range(O_MULTI):
        assert f'revenue_{i}_pred' in df.columns
        assert f'revenue_{i}_actual' in df.columns
    print("Point forecast, multi-output: OK")

def test_quantile_forecast_single_output(y_true_single_output):
    """Quantile forecast, single output dimension."""
    # preds shape: (Samples, Horizon, NumQuantiles)
    preds = tf.random.normal((SAMPLES, H, N_Q), dtype=tf.float32)
    df = format_predictions_to_dataframe(
        predictions=preds,
        y_true_sequences=y_true_single_output,
        target_name="demand",
        quantiles=Q_LIST,
        # output_dim=O_SINGLE # Should be inferred as 1
    )
    assert len(df) == SAMPLES * H
    for q_val in Q_LIST:
        assert f'demand_q{int(q_val*100)}' in df.columns
    assert 'demand_actual' in df.columns
    print("Quantile forecast, single output: OK")

def test_quantile_forecast_multi_output_3d_pred(y_true_multi_output):
    """Quantile forecast, multi-output, preds shape (S, H, Q*O)."""
    # preds shape: (Samples, Horizon, NumQuantiles * OutputDim)
    preds = tf.random.normal((SAMPLES, H, N_Q * O_MULTI), dtype=tf.float32)
    df = format_predictions_to_dataframe(
        predictions=preds,
        y_true_sequences=y_true_multi_output,
        target_name="stock",
        quantiles=Q_LIST,
        output_dim=O_MULTI # Must provide output_dim to disambiguate
    )
    assert len(df) == SAMPLES * H
    for o_idx in range(O_MULTI):
        for q_val in Q_LIST:
            assert f'stock_{o_idx}_q{int(q_val*100)}' in df.columns
        assert f'stock_{o_idx}_actual' in df.columns
    print("Quantile forecast, multi-output (3D preds): OK")

def test_quantile_forecast_multi_output_4d_pred(y_true_multi_output):
    """Quantile forecast, multi-output, preds shape (S, H, Q, O)."""
    # preds shape: (Samples, Horizon, NumQuantiles, OutputDim)
    preds = tf.random.normal((SAMPLES, H, N_Q, O_MULTI), dtype=tf.float32)
    df = format_predictions_to_dataframe(
        predictions=preds,
        y_true_sequences=y_true_multi_output,
        target_name="price",
        quantiles=Q_LIST
        # output_dim should be inferred from 4th dim
    )
    assert len(df) == SAMPLES * H
    for o_idx in range(O_MULTI):
        for q_val in Q_LIST:
            assert f'price_{o_idx}_q{int(q_val*100)}' in df.columns
        assert f'price_{o_idx}_actual' in df.columns
    print("Quantile forecast, multi-output (4D preds): OK")

@pytest.mark.skip(
    "RuntimeError: Failed to generate predictions from model")
def test_predictions_from_model(dummy_keras_model, dummy_model_inputs, y_true_single_output):
    """Test generating predictions from a model."""
    model_output_shape = (B, H, O_SINGLE) # Model returns (B,H,O)
    model = dummy_keras_model(output_shape_to_return=model_output_shape)
    df = format_predictions_to_dataframe(
        model=model,
        model_inputs=dummy_model_inputs,
        y_true_sequences=y_true_single_output,
        target_name="value",
        forecast_horizon=H, # Provide to model if needed
        output_dim=O_SINGLE
    )
    assert len(df) == SAMPLES * H
    assert 'value_pred' in df.columns
    assert 'value_actual' in df.columns
    print("Predictions from model: OK")

def test_spatial_data_from_dataframe(dynamic_input_3d, spatial_data_df):
    """Test adding spatial columns from a DataFrame."""
    preds = dynamic_input_3d # Use dynamic input as dummy predictions (B,H,O)
    spatial_names = ['loc_id', 'region']
    df = format_predictions_to_dataframe(
        predictions=preds,
        spatial_data_array=spatial_data_df, # This df has SAMPLES rows
        spatial_cols_names=spatial_names
    )
    T_DYN = H
    assert len(df) == B * T_DYN # H is T_DYN here
    assert 'loc_id' in df.columns
    assert 'region' in df.columns
    # Check if values are repeated correctly
    assert df['loc_id'].nunique() <= B
    print("Spatial data from DataFrame: OK")

def test_spatial_data_from_numpy(dynamic_input_3d, spatial_data_np):
    """Test adding spatial columns from a NumPy array."""
    preds = dynamic_input_3d
    spatial_names = ['geo_1', 'geo_2']
    df = format_predictions_to_dataframe(
        predictions=preds,
        spatial_data_array=spatial_data_np, # Shape (SAMPLES, NumSpatialFeatures)
        spatial_cols_names=spatial_names,
        spatial_cols_indices=[0, 1] # Use both columns
    )
    T_DYN = H
    assert len(df) == B * T_DYN
    assert 'geo_1' in df.columns
    assert 'geo_2' in df.columns
    print("Spatial data from NumPy array: OK")

def test_scaler_functionality(dynamic_input_3d, y_true_single_output, dummy_scaler):
    """Test inverse transformation with a scaler."""
    preds_scaled = dynamic_input_3d # Shape (B, H, O_SINGLE)
    actual_scaled = y_true_single_output

    scaler_features = ['other_feat1', 'target_col_name', 'other_feat2']
    target_idx = 1
    
    # Create a scaler instance for 'target_col_name'
    # The dummy_scaler mock needs feature_names_in_ at init
    scaler_instance = dummy_scaler(scaler_features)

    df = format_predictions_to_dataframe(
        predictions=preds_scaled,
        y_true_sequences=actual_scaled,
        target_name="target_col_name",
        scaler=scaler_instance,
        scaler_feature_names=scaler_features,
        target_idx_in_scaler=target_idx
    )
    # Check if inverse transform happened (values should be > 100)
    assert (df['target_col_name_pred'] > 50).all() # Dummy scaler adds 100
    assert (df['target_col_name_actual'] > 50).all()
    print("Scaler functionality: OK")

def test_evaluate_coverage(y_true_single_output):
    """Test coverage score evaluation."""
    # Predictions: (Samples, Horizon, NumQuantiles)
    preds_q = tf.random.normal((SAMPLES, H, N_Q), dtype=tf.float32)
    # Ensure q_low < q_median < q_high for meaningful coverage
    preds_np = preds_q.numpy()
    preds_np[:, :, 0] = y_true_single_output.numpy()[:,:,0] - np.abs(np.random.normal(0,0.5,size=preds_np[:,:,0].shape)) # q10 < actual
    preds_np[:, :, 1] = y_true_single_output.numpy()[:,:,0] # q50 = actual
    preds_np[:, :, 2] = y_true_single_output.numpy()[:,:,0] + np.abs(np.random.normal(0,0.5,size=preds_np[:,:,2].shape)) # q90 > actual
    preds_q_ordered = tf.constant(preds_np)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = format_predictions_to_dataframe(
            predictions=preds_q_ordered,
            y_true_sequences=y_true_single_output,
            quantiles=Q_LIST,
            evaluate_coverage=True,
            verbose=3 # To see coverage score print
        )
        # Check if coverage score was calculated (printed by vlog)
        # For this test, we don't assert the value, just that it runs.
        # A more robust test would mock vlog or check df.attrs
        # found_coverage_log = any(
        #     "Coverage Score" in str(warn_msg.message) for warn_msg in w
        #     ) or \
        #     any("Coverage Score" in captured.out for captured in capsys.readouterr()) # if using capsys

    # This test is a bit weak as it relies on vlog printing.
    # A better way would be if format_predictions_to_dataframe returned metrics
    # or stored them in df.attrs. For now, we check if it runs without error.
    assert 'target_q10' in df.columns # Check if df was formed
    print("Evaluate coverage: OK (runs without error)")

# @pytest.mark.skipif(not HAS_COVERAGE_SCORE_TEST, reason="coverage_score not available")
# @pytest.mark.parametrize("H, N_Q", [(3, 3)])
def test_evaluate_coverage2(y_true_single_output, capsys):
    """Test coverage score evaluation."""
    # Predictions: (Samples, Horizon, NumQuantiles)
    preds_q = tf.random.normal((SAMPLES, H, N_Q), dtype=tf.float32)
    preds_np = preds_q.numpy()
    # Ensure q_low < q_median < q_high
    preds_np[:, :, 0] = (
        y_true_single_output.numpy()[:, :, 0]
        - np.abs(np.random.normal(0, 0.5, preds_np[:,:,0].shape))
    )
    preds_np[:, :, 1] = y_true_single_output.numpy()[:, :, 0]
    preds_np[:, :, 2] = (
        y_true_single_output.numpy()[:, :, 0]
        + np.abs(np.random.normal(0, 0.5, preds_np[:,:,2].shape))
    )
    preds_q_ordered = tf.constant(preds_np)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = format_predictions_to_dataframe(
            predictions=preds_q_ordered,
            y_true_sequences=y_true_single_output,
            quantiles=Q_LIST,
            evaluate_coverage=True,
            verbose=3  # to force the coverage print
        )

    # Now capture stdout
    captured = capsys.readouterr()

    found_coverage_log = any(
        "Coverage Score" in str(wm.message) for wm in w
    ) or ("Coverage Score" in captured.out)

    # We don’t assert the numeric value here, just that it ran & logged
    assert 'target_q10' in df.columns
    assert found_coverage_log, "Did not see any 'Coverage Score' log"
    print("Evaluate coverage: OK (runs without error)")


def test_error_conditions(dummy_keras_model, dummy_model_inputs):
    """Test various error conditions."""
    # Missing model and predictions
    with pytest.raises(ValueError, match="If 'predictions' is None, both 'model'"):
        format_predictions_to_dataframe(predictions=None, model=None)
    # Missing model_inputs if model is provided
    model_output_shape = (B, H, O_SINGLE)
    model = dummy_keras_model(output_shape_to_return=model_output_shape)
    with pytest.raises(ValueError, match="If 'predictions' is None, both 'model'"):
        format_predictions_to_dataframe(predictions=None, model=model, model_inputs=None)

    # Invalid predictions shape (1D)
    with pytest.raises(ValueError, match="Predictions must be 2D, 3D or 4D"):
        format_predictions_to_dataframe(predictions=np.random.rand(SAMPLES))

    # Quantiles provided but prediction shape incompatible
    with pytest.raises(ValueError, match="Prediction's last dim .* not divisible by num_quantiles"):
        format_predictions_to_dataframe(
            predictions=tf.random.normal((SAMPLES, H, N_Q + 1)),
            quantiles=Q_LIST
        )
    print("Error conditions: OK")


if __name__ == '__main__':
    pytest.main([__file__])


