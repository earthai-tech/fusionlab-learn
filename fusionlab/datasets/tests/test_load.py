# test_load.py

import pytest
import os
import numpy as np
import pandas as pd
import joblib # # Noqa D:504
from pathlib import Path # For tmp_path # Noqa D:504

# --- Attempt Imports ---
try:
    from fusionlab.datasets.load import (
        fetch_zhongshan_data,
        fetch_nansha_data,
        load_processed_subsidence_data
    )
    from fusionlab.api.bunch import XBunch
    # Attempt to import a function needed by load_processed_subsidence_data
    # to check if dependencies are met. Actual execution handles missing helpers.
    from fusionlab.utils.spatial_utils import spatial_sampling # Noqa D:504
    FUSIONLAB_INSTALLED = True
except ImportError:
    FUSIONLAB_INSTALLED = False
    # Define dummy classes/functions if needed for pytest collection
    class XBunch(dict): pass
    def fetch_zhongshan_data(*args, **kwargs): raise ImportError
    def fetch_nansha_data(*args, **kwargs): raise ImportError
    def load_processed_subsidence_data(*args, **kwargs): raise ImportError

# --- Skip Conditions ---
pytestmark = pytest.mark.skipif(
    not FUSIONLAB_INSTALLED,
    reason="fusionlab or its dependencies not found"
)

# Helper to check if data exists locally (avoids download in tests)
# This is basic; a robust check might involve download_file_if with download=False
def data_file_exists(dataset_name, data_home=None):
    from fusionlab.datasets._property import get_data
    from fusionlab.datasets.load import _ZHONGSHAN_METADATA, _NANSHA_METADATA
    try:
        data_dir = get_data(data_home)
        meta = _ZHONGSHAN_METADATA if dataset_name == 'zhongshan' else _NANSHA_METADATA
        filepath = os.path.join(data_dir, meta.file)
        return os.path.exists(filepath)
    except Exception:
        return False # Cannot determine existence

skip_if_zhongshan_missing = pytest.mark.skipif(
    not data_file_exists('zhongshan'), reason="Zhongshan data file missing"
)
skip_if_nansha_missing = pytest.mark.skipif(
    not data_file_exists('nansha'), reason="Nansha data file missing"
)

# --- Tests for fetch_ functions ---

@pytest.mark.parametrize("dataset_name", ["zhongshan", "nansha"])
def test_fetch_data_as_frame(dataset_name):
    """Test fetching data returns a DataFrame."""
    fetch_func = fetch_zhongshan_data if dataset_name == 'zhongshan' \
                 else fetch_nansha_data
    # Skip if specific file missing
    if (dataset_name == 'zhongshan' and not data_file_exists('zhongshan')) or \
       (dataset_name == 'nansha' and not data_file_exists('nansha')):
        pytest.skip(f"{dataset_name} data file missing")

    df = fetch_func(as_frame=True, verbose=False, download_if_missing=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000 # Check it loaded a reasonable amount of data
    assert 'longitude' in df.columns
    assert 'latitude' in df.columns
    assert 'subsidence' in df.columns
    print(f"Fetch OK (as_frame=True) for {dataset_name}")

@pytest.mark.parametrize("dataset_name", ["zhongshan", "nansha"])
def test_fetch_data_as_bunch(dataset_name):
    """Test fetching data returns a Bunch object."""
    fetch_func = fetch_zhongshan_data if dataset_name == 'zhongshan' \
                 else fetch_nansha_data
    if (dataset_name == 'zhongshan' and not data_file_exists('zhongshan')) or \
       (dataset_name == 'nansha' and not data_file_exists('nansha')):
        pytest.skip(f"{dataset_name} data file missing")

    data_bunch = fetch_func(as_frame=False, verbose=False,
                              download_if_missing=False)
    assert isinstance(data_bunch, XBunch)
    assert hasattr(data_bunch, 'frame')
    assert hasattr(data_bunch, 'data')
    assert hasattr(data_bunch, 'feature_names')
    assert hasattr(data_bunch, 'target_names')
    assert hasattr(data_bunch, 'target')
    assert hasattr(data_bunch, 'longitude')
    assert hasattr(data_bunch, 'latitude')
    assert hasattr(data_bunch, 'DESCR')
    assert isinstance(data_bunch.frame, pd.DataFrame)
    assert len(data_bunch.frame) > 1000
    assert data_bunch.target_names == ['subsidence']
    print(f"Fetch OK (as_frame=False) for {dataset_name}")

@pytest.mark.parametrize("dataset_name", ["zhongshan", "nansha"])
def test_fetch_data_sampling(dataset_name):
    """Test n_samples parameter."""
    fetch_func = fetch_zhongshan_data if dataset_name == 'zhongshan' \
                 else fetch_nansha_data
    if (dataset_name == 'zhongshan' and not data_file_exists('zhongshan')) or \
       (dataset_name == 'nansha' and not data_file_exists('nansha')):
        pytest.skip(f"{dataset_name} data file missing")

    n_request = 100
    df_sampled = fetch_func(as_frame=True, n_samples=n_request,
                            random_state=42, verbose=False,
                            download_if_missing=False)
    assert isinstance(df_sampled, pd.DataFrame)
    # spatial_sampling might return slightly less than requested
    assert abs(len(df_sampled) - n_request) <= 5

    # Test requesting too many samples (should warn and return full)
    with pytest.warns(UserWarning, match="Requested n_samples .* larger than"):
        df_full = fetch_func(as_frame=True, n_samples=5000, verbose=False,
                             download_if_missing=False)
    assert len(df_full) > 1000 # Should be full dataset size (~2000)

    # Test invalid n_samples
    with pytest.raises(ValueError):
        fetch_func(as_frame=True, n_samples=-10)
    with pytest.raises(ValueError):
        fetch_func(as_frame=True, n_samples=0)
    with pytest.raises(ValueError): # Or ValueError depending on validation
        fetch_func(as_frame=True, n_samples='abc')
    print(f"Sampling OK for {dataset_name}")

@pytest.mark.parametrize("dataset_name", ["zhongshan", "nansha"])
def test_fetch_data_includes(dataset_name):
    """Test include_coords and include_target parameters."""
    fetch_func = fetch_zhongshan_data if dataset_name == 'zhongshan' \
                 else fetch_nansha_data
    if (dataset_name == 'zhongshan' and not data_file_exists('zhongshan')) or \
       (dataset_name == 'nansha' and not data_file_exists('nansha')):
        pytest.skip(f"{dataset_name} data file missing")

    # Test excluding coords
    df_no_coords = fetch_func(as_frame=True, include_coords=False, verbose=False)
    assert 'longitude' not in df_no_coords.columns
    assert 'latitude' not in df_no_coords.columns
    assert 'subsidence' in df_no_coords.columns # Target included by default

    # Test excluding target
    df_no_target = fetch_func(as_frame=True, include_target=False, verbose=False)
    assert 'longitude' in df_no_target.columns
    assert 'latitude' in df_no_target.columns
    assert 'subsidence' not in df_no_target.columns

    # Test excluding both
    df_only_features = fetch_func(as_frame=True, include_coords=False,
                                  include_target=False, verbose=False)
    assert 'longitude' not in df_only_features.columns
    assert 'latitude' not in df_only_features.columns
    assert 'subsidence' not in df_only_features.columns
    print(f"Include flags OK for {dataset_name}")


# --- Tests for load_processed_subsidence_data ---

@pytest.mark.parametrize("dataset_name", ["zhongshan", "nansha"])
@pytest.mark.parametrize("as_frame", [True, False])
def test_load_processed_frame_bunch(dataset_name, as_frame):
    """Test loading processed data as DataFrame or Bunch."""
    if (dataset_name == 'zhongshan' and not data_file_exists('zhongshan')) or \
       (dataset_name == 'nansha' and not data_file_exists('nansha')):
        pytest.skip(f"{dataset_name} data file missing")

    try:
        result = load_processed_subsidence_data(
            dataset_name=dataset_name,
            return_sequences=False,
            as_frame=as_frame,
            # Disable caching for this basic test
            use_processed_cache=False,
            save_processed_frame=False,
            verbose=False,
            download_if_missing=False # Assume file exists
        )
    except Exception as e:
        pytest.fail(f"load_processed failed (as_frame={as_frame}, "
                    f"return_seq=False, dataset={dataset_name}). Error: {e}")

    if as_frame:
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'geology' not in result.columns # Should be encoded/dropped
        assert any(c.startswith('geology_') for c in result.columns)
        assert 'subsidence' in result.columns # Target should be present
    else: # Bunch
        assert isinstance(result, XBunch)
        assert hasattr(result, 'frame')
        assert hasattr(result, 'data')
        assert len(result.frame) > 0
        assert result.target_names == ['subsidence']
        assert result.target is not None
        assert result.data is not None
    print(f"Load Processed OK: dataset={dataset_name}, as_frame={as_frame}")

@pytest.mark.parametrize("dataset_name", ["zhongshan", "nansha"])
def test_load_processed_sequences(dataset_name):
    """Test loading processed data as sequences."""
    if (dataset_name == 'zhongshan' and not data_file_exists('zhongshan')) or \
       (dataset_name == 'nansha' and not data_file_exists('nansha')):
        pytest.skip(f"{dataset_name} data file missing")

    T, H = 7, 3 # Example sequence params

    try:
        sequences = load_processed_subsidence_data(
            dataset_name=dataset_name,
            return_sequences=True, # To check sequences 
            time_steps=T,
            forecast_horizon=H,
            use_sequence_cache=False, # Disable caching for this test
            save_sequences=False,
            verbose=False,
            download_if_missing=False, # Assume file exists, 
            group_by_cols =None, # No enough samples, so set to None 
        )
    except Exception as e:
        pytest.fail(f"load_processed failed (return_sequences=True, "
                    f"dataset={dataset_name}). Error: {e}")

    assert isinstance(sequences, tuple)
    assert len(sequences) == 4 # static, dynamic, future, target
    s, d, f, t = sequences
    assert isinstance(s, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(f, np.ndarray)
    assert isinstance(t, np.ndarray)

    # Check shapes (batch size N depends on data after processing/dropna)
    # for one long sequences 
    N = s.shape[0]
    assert N > 0
    assert s.shape[0] == d.shape[0] == f.shape[0] == t.shape[0]
    assert d.shape[1] == T
    assert f.shape[1] == T+ H # reshape_xtft_data output for future; H+T spanned
    assert t.shape[1] == H
    assert s.ndim >= 2 # (N, StaticFeatures)
    assert d.ndim == 3 # (N, T, DynFeatures)
    assert f.ndim == 3 # (N, H+T, FutFeatures)
    assert t.ndim == 3 # (N, H, TargetFeatures=1)
    print(f"Load Sequences OK: dataset={dataset_name}")

@pytest.mark.parametrize("dataset_name", ["zhongshan", "nansha"])
def test_load_processed_cache(dataset_name, tmp_path):
    """Test caching logic for processed frame and sequences."""
    if (dataset_name == 'zhongshan' and not data_file_exists('zhongshan')) or \
       (dataset_name == 'nansha' and not data_file_exists('nansha')):
        pytest.skip(f"{dataset_name} data file missing")

    cache_dir = tmp_path / f"fusionlab_cache_{dataset_name}"
    T, H = 6, 2
    suffix = "_test_cache"

    # 1. Run once, saving processed frame
    _ = load_processed_subsidence_data(
        dataset_name=dataset_name, data_home=str(cache_dir),
        return_sequences=False, as_frame=True,
        save_processed_frame=True, use_processed_cache=False,
        cache_suffix=suffix, verbose=False, download_if_missing=False
    )
    proc_cache_file = cache_dir / f"{dataset_name}_processed{suffix}.joblib"
    assert proc_cache_file.exists()

    # 2. Run again, should load from processed frame cache
    with pytest.warns(None) as record: # Check no reprocessing warnings
        df_loaded = load_processed_subsidence_data(
            dataset_name=dataset_name, data_home=str(cache_dir),
            return_sequences=False, as_frame=True,
            save_processed_frame=False, use_processed_cache=True, # Use cache
            cache_suffix=suffix, verbose=False, download_if_missing=False
        )
    assert len(record) == 0 # Should not warn about reprocessing
    assert isinstance(df_loaded, pd.DataFrame)

    # 3. Run once, saving sequences
    _ = load_processed_subsidence_data(
        dataset_name=dataset_name, data_home=str(cache_dir),
        return_sequences=True, time_steps=T, forecast_horizon=H,
        save_sequences=True, use_sequence_cache=False,
        cache_suffix=suffix, verbose=False, group_by_cols=None, 
        download_if_missing=False
    )
    seq_cache_file = cache_dir / (f"{dataset_name}_sequences_T{T}_H{H}"
                                  f"{suffix}.joblib")
    assert seq_cache_file.exists()

    # 4. Run again, should load from sequence cache
    with pytest.warns(None) as record_seq:
        seq_loaded = load_processed_subsidence_data(
            dataset_name=dataset_name, data_home=str(cache_dir),
            return_sequences=True, time_steps=T, forecast_horizon=H,
            save_sequences=False, use_sequence_cache=True, # Use cache
            cache_suffix=suffix, verbose=False, download_if_missing=False
        )
    assert len(record_seq) == 0
    assert isinstance(seq_loaded, tuple) and len(seq_loaded) == 4

    print(f"Caching OK for {dataset_name}")

# Allows running the tests directly if needed
if __name__=='__main__':
     pytest.main([__file__])