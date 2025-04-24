# test_make.py

import pytest
import numpy as np
import pandas as pd

# --- Attempt Imports ---
try:
    from fusionlab.datasets.make import (
        make_multi_feature_time_series,
        make_quantile_prediction_data,
        make_anomaly_data,
        make_trend_seasonal_data,
        make_multivariate_target_data
    )
    from fusionlab.api.bunch import XBunch
    FUSIONLAB_INSTALLED = True
except ImportError as e:
    print(f"Skipping make tests due to import error: {e}")
    FUSIONLAB_INSTALLED = False
    class XBunch(dict): pass # Dummy for collection

# --- Skip Conditions ---
pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab.datasets.make or fusionlab.api.bunch not found"
)

# --- Constants ---
N_SAMPLES = 50 # Smaller number for faster tests
N_SERIES = 3
N_TIMESTEPS = 24
N_HORIZONS = 6
SEED = 42

# === Tests for make_multi_feature_time_series ===

def test_make_multi_feature_return_types():
    """Test return types of make_multi_feature_time_series."""
    bunch_out = make_multi_feature_time_series(
        n_series=N_SERIES, n_timesteps=N_TIMESTEPS, as_frame=False, seed=SEED
    )
    df_out = make_multi_feature_time_series(
        n_series=N_SERIES, n_timesteps=N_TIMESTEPS, as_frame=True, seed=SEED
    )
    assert isinstance(bunch_out, XBunch)
    assert isinstance(df_out, pd.DataFrame)

def test_make_multi_feature_bunch_structure():
    """Test Bunch structure from make_multi_feature_time_series."""
    bunch = make_multi_feature_time_series(
        n_series=N_SERIES, n_timesteps=N_TIMESTEPS, as_frame=False, seed=SEED
    )
    assert hasattr(bunch, 'frame')
    assert hasattr(bunch, 'static_features')
    assert hasattr(bunch, 'dynamic_features')
    assert hasattr(bunch, 'future_features')
    assert hasattr(bunch, 'target_col')
    assert hasattr(bunch, 'dt_col')
    assert hasattr(bunch, 'spatial_id_col')
    assert hasattr(bunch, 'feature_names')
    assert hasattr(bunch, 'DESCR')
    assert isinstance(bunch.frame, pd.DataFrame)
    assert len(bunch.frame) == N_SERIES * N_TIMESTEPS
    # Check if column lists match frame columns (excluding target/id/date)
    expected_cols = (
        set(bunch.static_features[1:]) | # Exclude series_id
        set(bunch.dynamic_features) |
        set(bunch.future_features)
        )
    assert set(bunch.feature_names) == expected_cols
    assert bunch.target_col in bunch.frame.columns
    assert bunch.dt_col in bunch.frame.columns
    assert bunch.spatial_id_col in bunch.frame.columns

def test_make_multi_feature_params():
    """Test parameters of make_multi_feature_time_series."""
    df = make_multi_feature_time_series(
        n_series=2, n_timesteps=15, freq='MS', as_frame=True, seed=SEED
    )
    assert len(df) == 2 * 15
    assert df['series_id'].nunique() == 2
    assert df['date'].dtype == 'datetime64[ns]'

def test_make_multi_feature_reproducibility():
    """Test seed parameter for make_multi_feature_time_series."""
    df1 = make_multi_feature_time_series(
        n_series=N_SERIES, n_timesteps=N_TIMESTEPS, as_frame=True, seed=SEED
    )
    df2 = make_multi_feature_time_series(
        n_series=N_SERIES, n_timesteps=N_TIMESTEPS, as_frame=True, seed=SEED
    )
    df3 = make_multi_feature_time_series(
        n_series=N_SERIES, n_timesteps=N_TIMESTEPS, as_frame=True, seed=SEED + 1
    )
    pd.testing.assert_frame_equal(df1, df2)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df3)

# === Tests for make_quantile_prediction_data ===

def test_make_quantile_pred_return_types():
    """Test return types of make_quantile_prediction_data."""
    bunch_out = make_quantile_prediction_data(
        n_samples=N_SAMPLES, n_horizons=N_HORIZONS, as_frame=False, seed=SEED
    )
    df_out = make_quantile_prediction_data(
        n_samples=N_SAMPLES, n_horizons=N_HORIZONS, as_frame=True, seed=SEED
    )
    assert isinstance(bunch_out, XBunch)
    assert isinstance(df_out, pd.DataFrame)

def test_make_quantile_pred_bunch_structure():
    """Test Bunch structure from make_quantile_prediction_data."""
    quantiles = [0.2, 0.5, 0.8]
    bunch = make_quantile_prediction_data(
        n_samples=N_SAMPLES, n_horizons=N_HORIZONS, quantiles=quantiles,
        as_frame=False, seed=SEED
    )
    assert hasattr(bunch, 'frame')
    assert hasattr(bunch, 'quantiles')
    assert hasattr(bunch, 'horizons')
    assert hasattr(bunch, 'target_cols')
    assert hasattr(bunch, 'prediction_cols')
    assert hasattr(bunch, 'DESCR')
    assert len(bunch.frame) == N_SAMPLES
    assert len(bunch.target_cols) == N_HORIZONS
    assert len(bunch.prediction_cols) == len(quantiles)
    assert len(bunch.prediction_cols['q0.2']) == N_HORIZONS
    assert bunch.quantiles == quantiles

def test_make_quantile_pred_params():
    """Test parameters of make_quantile_prediction_data."""
    df = make_quantile_prediction_data(
        n_samples=10, n_horizons=3, quantiles=[0.5], add_coords=False,
        as_frame=True, seed=SEED
    )
    assert len(df) == 10
    assert len(df.columns) == 3 + 3 # target_h1..3, pred_q5_h1..3
    assert 'longitude' not in df.columns
    assert 'target_h3' in df.columns
    assert 'pred_q0.5_h3' in df.columns # Check generated name format

# === Tests for make_anomaly_data ===

def test_make_anomaly_data_return_types():
    """Test return types of make_anomaly_data."""
    # Default return is tuple
    sequences, labels = make_anomaly_data(
        n_sequences=N_SAMPLES, sequence_length=N_TIMESTEPS, as_frame=False, seed=SEED
    )
    assert isinstance(sequences, np.ndarray)
    assert isinstance(labels, np.ndarray)
    # Test frame return
    bunch_out = make_anomaly_data(
        n_sequences=N_SAMPLES, sequence_length=N_TIMESTEPS, as_frame=True, seed=SEED
    )
    assert isinstance(bunch_out, XBunch)
    assert isinstance(bunch_out.frame, pd.DataFrame)

def test_make_anomaly_data_shapes_labels():
    """Test output shapes and label consistency for make_anomaly_data."""
    n_seq, T, F = 50, 20, 1
    anomaly_frac = 0.2
    sequences, labels = make_anomaly_data(
        n_sequences=n_seq, sequence_length=T, n_features=F,
        anomaly_fraction=anomaly_frac, as_frame=False, seed=SEED
    )
    assert sequences.shape == (n_seq, T, F)
    assert labels.shape == (n_seq,)
    assert labels.dtype == int
    assert np.all((labels == 0) | (labels == 1))
    expected_anomalies = int(n_seq * anomaly_frac)
    assert np.sum(labels) == expected_anomalies

@pytest.mark.parametrize("anomaly_type", ['spike', 'level_shift'])
def test_make_anomaly_data_types(anomaly_type):
    """Test different anomaly types run without error."""
    try:
        sequences, labels = make_anomaly_data(
            n_sequences=10, sequence_length=20, anomaly_type=anomaly_type,
            as_frame=False, seed=SEED
        )
        assert sequences is not None
        assert labels is not None
    except Exception as e:
        pytest.fail(f"make_anomaly_data failed for type='{anomaly_type}'. Error: {e}")

def test_make_anomaly_data_errors():
    """Test error handling for make_anomaly_data."""
    with pytest.raises(ValueError, match="Currently only supports n_features=1"):
        make_anomaly_data(n_features=2)
    with pytest.raises(ValueError, match="'anomaly_fraction' must be between 0 and 1"):
        make_anomaly_data(anomaly_fraction=1.1)
    with pytest.raises(ValueError, match="anomaly_type must be 'spike' or 'level_shift'"):
        make_anomaly_data(anomaly_type='invalid')

# === Tests for make_trend_seasonal_data ===

def test_make_trend_seasonal_return_types():
    """Test return types of make_trend_seasonal_data."""
    bunch_out = make_trend_seasonal_data(
        n_timesteps=N_TIMESTEPS, as_frame=False, seed=SEED
    )
    df_out = make_trend_seasonal_data(
        n_timesteps=N_TIMESTEPS, as_frame=True, seed=SEED
    )
    assert isinstance(bunch_out, XBunch)
    assert isinstance(df_out, pd.DataFrame)

def test_make_trend_seasonal_bunch_structure():
    """Test Bunch structure from make_trend_seasonal_data."""
    bunch = make_trend_seasonal_data(
        n_timesteps=N_TIMESTEPS, as_frame=False, seed=SEED
    )
    assert hasattr(bunch, 'frame')
    assert hasattr(bunch, 'data')
    assert hasattr(bunch, 'target_names')
    assert hasattr(bunch, 'target')
    assert hasattr(bunch, 'dt_col')
    assert hasattr(bunch, 'DESCR')
    assert len(bunch.frame) == N_TIMESTEPS
    assert bunch.target_names == ['value']
    assert bunch.dt_col == 'date'
    assert bunch.data.shape == (N_TIMESTEPS,)
    assert bunch.target.shape == (N_TIMESTEPS,)

@pytest.mark.parametrize("trend_order", [0, 1, 2])
def test_make_trend_seasonal_trend_order(trend_order):
    """Test different trend orders run without error."""
    try:
        df = make_trend_seasonal_data(
            n_timesteps=30, trend_order=trend_order, as_frame=True, seed=SEED
        )
        assert isinstance(df, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"make_trend_seasonal_data failed for trend_order={trend_order}. Error: {e}")

def test_make_trend_seasonal_errors():
    """Test error handling for make_trend_seasonal_data."""
    with pytest.raises(ValueError, match="Lengths of 'seasonal_periods' and"):
        make_trend_seasonal_data(seasonal_periods=[7], seasonal_amplitudes=[5, 10])
    with pytest.raises(ValueError, match="Length of 'trend_coeffs'"):
        make_trend_seasonal_data(trend_order=1, trend_coeffs=[10, 0.1, 0.01])
    with pytest.raises(ValueError, match="'trend_order' must be >= 0"):
        make_trend_seasonal_data(trend_order=-1)


def test_make_multivariate_target_structure():
    n_targets = 3
    bunch = make_multivariate_target_data(
        n_series=2, n_timesteps=20, n_targets=n_targets, seed=SEED
        )
    assert isinstance(bunch, XBunch)
    assert len(bunch.target_names) == n_targets
    assert bunch.target.shape == (2 * 20, n_targets)
    assert all(c in bunch.frame.columns for c in bunch.target_names)


# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])
