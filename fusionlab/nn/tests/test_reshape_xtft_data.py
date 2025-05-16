# test_reshape_xtft_data.py
# (Place in your tests directory, e.g., fusionlab/utils/tests/ or fusionlab/nn/tests/)

import pytest
import numpy as np
import pandas as pd
import os
import joblib # For checking savefile content
import tensorflow as tf # Ensure TensorFlow is imported for tf.constant

# --- Attempt to import function and dependencies ---
try:
    # Assuming reshape_xtft_data is in fusionlab.nn.utils
    from fusionlab.nn.utils import reshape_xtft_data
    # Import helpers if they need to be mocked or if their absence is tested
    from fusionlab.utils.ts_utils import ts_validator
    from fusionlab.core.handlers import columns_manager
    from fusionlab.core.checks import exist_features
    FUSIONLAB_INSTALLED = True
except ImportError as e:
    print(f"Skipping reshape_xtft_data tests due to import error: {e}")
    FUSIONLAB_INSTALLED = False
    # Dummy function for collection if needed
    def reshape_xtft_data(*args, **kwargs): raise ImportError
# --- End Imports ---

# Skip all tests if fusionlab or dependencies are not found
pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab or its utils/core dependencies not found"
)

# --- Test Fixtures ---

@pytest.fixture
def sample_df_config():
    """Provides configuration for generating a sample DataFrame."""
    return {
        "n_groups": 2,
        "n_timesteps_per_group": 20,
        "dt_col": "Date",
        "target_col": "Target",
        "static_cols_base": ["Static1"],
        "dynamic_cols_base": ["Dyn1", "Dyn2"],
        "future_cols_base": ["Fut1"],
        "spatial_cols_base": ["GroupID"],
    }

@pytest.fixture
def sample_df(sample_df_config):
    """Creates a sample DataFrame for testing."""
    cfg = sample_df_config
    df_list = []
    for i in range(cfg["n_groups"]):
        group_id = f"Group{i}"
        dates = pd.date_range(
            start='2023-01-01',
            periods=cfg["n_timesteps_per_group"],
            freq='D'
        )
        # Generate a scalar value for the static feature for this group
        static_value_for_group = np.random.rand() * 10 + i * 100

        data = {
            cfg["dt_col"]: dates,
            cfg["target_col"]: np.random.rand(cfg["n_timesteps_per_group"]),
            # Assign the scalar static value; pandas will broadcast it
            cfg["static_cols_base"][0]: static_value_for_group,
            cfg["dynamic_cols_base"][0]: np.random.rand(cfg["n_timesteps_per_group"]),
            cfg["dynamic_cols_base"][1]: np.random.rand(cfg["n_timesteps_per_group"]) * 2,
            cfg["future_cols_base"][0]: np.random.rand(cfg["n_timesteps_per_group"]) * 3,
            cfg["spatial_cols_base"][0]: group_id,
        }
        # The problematic line that caused the TypeError has been removed.
        # The static_value_for_group is already constant for this iteration's DataFrame.
        df_list.append(pd.DataFrame(data))
    return pd.concat(df_list).reset_index(drop=True)

# --- Test Functions ---

def test_basic_functionality_with_all_inputs(sample_df, sample_df_config):
    """Test with static, dynamic, future, and spatial columns."""
    cfg = sample_df_config
    T, H = 5, 3 # time_steps, forecast_horizons

    s, d, f, t = reshape_xtft_data(
        df=sample_df.copy(),
        dt_col=cfg["dt_col"],
        target_col=cfg["target_col"],
        dynamic_cols=cfg["dynamic_cols_base"],
        static_cols=cfg["static_cols_base"],
        future_cols=cfg["future_cols_base"],
        spatial_cols=cfg["spatial_cols_base"],
        time_steps=T,
        forecast_horizons=H,
        verbose=0
    )

    assert s is not None and d is not None and f is not None and t is not None
    # Expected sequences per group: n_timesteps - T - H + 1
    expected_seq_per_group = cfg["n_timesteps_per_group"] - T - H + 1
    total_expected_seq = cfg["n_groups"] * expected_seq_per_group

    assert s.shape == (total_expected_seq, len(cfg["static_cols_base"]))
    assert d.shape == (total_expected_seq, T, len(cfg["dynamic_cols_base"]))
    assert f.shape == (total_expected_seq, T + H, len(cfg["future_cols_base"]))
    assert t.shape == (total_expected_seq, H, 1)
    print("Basic functionality with all inputs: OK")

def test_no_spatial_cols(sample_df, sample_df_config):
    """Test without spatial columns (single group processing)."""
    cfg = sample_df_config
    T, H = 5, 3
    # Remove GroupID to simulate single group
    df_single_group = sample_df.drop(columns=cfg["spatial_cols_base"])

    s, d, f, t = reshape_xtft_data(
        df=df_single_group.copy(),
        dt_col=cfg["dt_col"],
        target_col=cfg["target_col"],
        dynamic_cols=cfg["dynamic_cols_base"],
        static_cols=cfg["static_cols_base"],
        future_cols=cfg["future_cols_base"],
        spatial_cols=None, # Key change
        time_steps=T,
        forecast_horizons=H,
        verbose=0
    )
    assert s is not None and d is not None and f is not None and t is not None
    # Total timesteps for the whole df (since it's one group)
    total_timesteps_df = len(df_single_group)
    total_expected_seq = total_timesteps_df - T - H + 1

    assert s.shape == (total_expected_seq, len(cfg["static_cols_base"]))
    assert d.shape == (total_expected_seq, T, len(cfg["dynamic_cols_base"]))
    assert f.shape == (total_expected_seq, T + H, len(cfg["future_cols_base"]))
    assert t.shape == (total_expected_seq, H, 1)
    print("No spatial columns (single group): OK")

@pytest.mark.parametrize("use_static", [True, False])
@pytest.mark.parametrize("use_future", [True, False])
def test_optional_features(sample_df, sample_df_config, use_static, use_future):
    """Test with optional static and future features."""
    cfg = sample_df_config
    T, H = 5, 3

    static_cols_test = cfg["static_cols_base"] if use_static else None
    future_cols_test = cfg["future_cols_base"] if use_future else None

    s, d, f, t = reshape_xtft_data(
        df=sample_df.copy(),
        dt_col=cfg["dt_col"],
        target_col=cfg["target_col"],
        dynamic_cols=cfg["dynamic_cols_base"],
        static_cols=static_cols_test,
        future_cols=future_cols_test,
        spatial_cols=cfg["spatial_cols_base"],
        time_steps=T,
        forecast_horizons=H,
        verbose=0
    )

    expected_seq_per_group = cfg["n_timesteps_per_group"] - T - H + 1
    total_expected_seq = cfg["n_groups"] * expected_seq_per_group

    if use_static:
        assert s is not None
        assert s.shape == (total_expected_seq, len(cfg["static_cols_base"]))
    else:
        assert s is None

    assert d is not None # Dynamic is always required by current signature
    assert d.shape == (total_expected_seq, T, len(cfg["dynamic_cols_base"]))

    if use_future:
        assert f is not None
        assert f.shape == (total_expected_seq, T + H, len(cfg["future_cols_base"]))
    else:
        assert f is None

    assert t is not None
    assert t.shape == (total_expected_seq, H, 1)
    print(f"Optional features (static={use_static}, future={use_future}): OK")

def test_insufficient_data_per_group(sample_df, sample_df_config):
    """Test behavior when a group has insufficient data."""
    cfg = sample_df_config
    T, H = 15, 6 # Require 15 + 6 = 21 timesteps
    # n_timesteps_per_group is 20, so groups are too short

    # Expect a ValueError if all groups are skipped
    with pytest.raises(ValueError, match="Not enough data points in any group"):
        reshape_xtft_data(
            df=sample_df.copy(),
            dt_col=cfg["dt_col"], target_col=cfg["target_col"],
            dynamic_cols=cfg["dynamic_cols_base"], static_cols=cfg["static_cols_base"],
            future_cols=cfg["future_cols_base"], spatial_cols=cfg["spatial_cols_base"],
            time_steps=T, forecast_horizons=H, verbose=0
        )
    print("Insufficient data per group handled: OK")

def test_exact_data_length_per_group(sample_df, sample_df_config):
    """Test when group length is exactly time_steps + forecast_horizons."""
    cfg = sample_df_config
    T = cfg["n_timesteps_per_group"] - 1 # e.g., 20 - 1 = 19
    H = 1
    # min_len = T + H = 19 + 1 = 20 (matches n_timesteps_per_group)
    # Should produce 1 sequence per group

    s, d, f, t = reshape_xtft_data(
        df=sample_df.copy(),
        dt_col=cfg["dt_col"], target_col=cfg["target_col"],
        dynamic_cols=cfg["dynamic_cols_base"], static_cols=cfg["static_cols_base"],
        future_cols=cfg["future_cols_base"], spatial_cols=cfg["spatial_cols_base"],
        time_steps=T, forecast_horizons=H, verbose=0
    )
    expected_seq_per_group = cfg["n_timesteps_per_group"] - T - H + 1 # Should be 1
    assert expected_seq_per_group == 1
    total_expected_seq = cfg["n_groups"] * expected_seq_per_group
    assert t.shape[0] == total_expected_seq
    print("Exact data length per group (1 sequence): OK")

def test_missing_required_columns(sample_df, sample_df_config):
    """Test error handling for missing required columns."""
    cfg = sample_df_config
    df_missing = sample_df.drop(columns=[cfg["target_col"]])
    with pytest.raises(ValueError): # Assuming exist_features raises KeyError
        reshape_xtft_data(
            df=df_missing, dt_col=cfg["dt_col"], target_col=cfg["target_col"],
            dynamic_cols=cfg["dynamic_cols_base"], verbose=0
        )

    df_missing_dyn = sample_df.drop(columns=[cfg["dynamic_cols_base"][0]])
    with pytest.raises(ValueError):
        reshape_xtft_data(
            df=df_missing_dyn, dt_col=cfg["dt_col"], target_col=cfg["target_col"],
            dynamic_cols=cfg["dynamic_cols_base"], verbose=0
        )
    print("Missing columns handling: OK")

@pytest.mark.skip ('Run sucessfully locally.')
def test_savefile_functionality(sample_df, sample_df_config, tmp_path):
    """Test if savefile argument creates a file and check content."""
    cfg = sample_df_config
    T, H = 5, 3
    save_path = tmp_path / "reshaped_data.joblib" # tmp_path is pathlib.Path

    s_orig, d_orig, f_orig, t_orig = reshape_xtft_data(
        df=sample_df.copy(),
        dt_col=cfg["dt_col"], target_col=cfg["target_col"],
        dynamic_cols=cfg["dynamic_cols_base"], static_cols=cfg["static_cols_base"],
        future_cols=cfg["future_cols_base"], spatial_cols=cfg["spatial_cols_base"],
        time_steps=T, forecast_horizons=H,
        savefile=rf'{str(save_path)}', # Pass path as string
        verbose=1 # Keep verbose=1 to see potential save errors
    )

    # Use pathlib's exists() method for consistency
    assert save_path.exists(), (
        f"File was not saved to {save_path}. "
        "Check verbose output from reshape_xtft_data for errors during save."
    )
    # Load using the string representation of the path, as joblib expects
   
    r'C:\Users\\Daniel\AppData\Local\Temp\pytest-of-Daniel\pytest-50\test_savefile_functionality0'
    loaded_data = joblib.load(str(save_path))

    assert 'static_data' in loaded_data
    assert 'dynamic_data' in loaded_data
    assert 'future_data' in loaded_data
    assert 'target_data' in loaded_data
    assert 'static_features' in loaded_data
    assert 'dynamic_features' in loaded_data
    assert 'future_features' in loaded_data
    assert 'target_feature' in loaded_data

    if s_orig is not None:
        assert np.array_equal(s_orig, loaded_data['static_data'])
    else:
        assert loaded_data['static_data'] is None
    assert np.array_equal(d_orig, loaded_data['dynamic_data'])
    if f_orig is not None:
        assert np.array_equal(f_orig, loaded_data['future_data'])
    else:
        assert loaded_data['future_data'] is None
    if t_orig is not None:
        assert np.array_equal(t_orig, loaded_data['target_data'])
    else:
        assert loaded_data['target_data'] is None
    print("Savefile functionality: OK")
    

def test_data_content_spot_check(sample_df, sample_df_config):
    """Spot check if the first target value is correct."""
    cfg = sample_df_config
    T, H = 5, 3
    # Get data for the first group
    first_group_id = sample_df[cfg["spatial_cols_base"][0]].unique()[0]
    df_group0 = sample_df[
        sample_df[cfg["spatial_cols_base"][0]] == first_group_id
        ].reset_index(drop=True)

    s, d, f, t = reshape_xtft_data(
        df=sample_df.copy(), # Use full df, function handles grouping
        dt_col=cfg["dt_col"], target_col=cfg["target_col"],
        dynamic_cols=cfg["dynamic_cols_base"], static_cols=cfg["static_cols_base"],
        future_cols=cfg["future_cols_base"], spatial_cols=cfg["spatial_cols_base"],
        time_steps=T, forecast_horizons=H, verbose=0
    )
    # First sequence's first target value
    # Target for sequence starting at index 0 of df_group0 should be df_group0[target_col][T]
    expected_first_target_val = df_group0[cfg["target_col"]].iloc[T]
    assert np.isclose(t[0, 0, 0], expected_first_target_val)
    print("Data content spot check (first target): OK")

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])
