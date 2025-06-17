
import pytest
import pandas as pd
import numpy as np


try:
    from fusionlab.datasets.load import load_subsidence_pinn_data, _ZHONGSHAN_METADATA
    from fusionlab.utils.geo_utils import generate_dummy_pinn_data
    from fusionlab.api.bunch import XBunch
    FUSIONLAB_AVAILABLE = True
except ImportError:
    FUSIONLAB_AVAILABLE = False

# Skip all tests if fusionlab-learn is not installed
pytestmark = pytest.mark.skipif(
    not FUSIONLAB_AVAILABLE, reason="fusionlab-learn is not installed"
)

# --- Fixtures ---

@pytest.fixture(scope="function")
def mock_data_home(tmp_path):
    """
    Creates a temporary data home directory and places the expected
    dataset file inside it.
    """
    # The function looks for a specific filename defined in its metadata
    expected_filename = _ZHONGSHAN_METADATA.file # e.g., 'zhongshan_2000.csv'
    file_path = tmp_path / expected_filename
    
    # Generate data with known characteristics
    dummy_dict = generate_dummy_pinn_data(n_samples=200)
    dummy_dict['geology'] = np.random.choice(['Clay', 'Sand', 'Gravel'], 200)
    pd.DataFrame(dummy_dict).to_csv(file_path, index=False)
    
    # The function expects data_home to be the root directory
    yield str(tmp_path)

# --- Test Cases ---

def test_load_strategy_success(mock_data_home):
    """Tests the 'load' strategy when the file exists in data_home."""
    bunch = load_subsidence_pinn_data(
        data_name='zhongshan',
        strategy='load',
        data_home=mock_data_home,
        use_cache=False
    )
    assert isinstance(bunch, XBunch)
    assert isinstance(bunch.frame, pd.DataFrame)
    assert len(bunch.frame) > 0
    # Check if preprocessing (one-hot encoding) was applied
    assert 'geology_Clay' in bunch.frame.columns

def test_load_strategy_failure():
    """Tests that 'load' strategy raises FileNotFoundError if file is missing."""
    with pytest.raises(FileNotFoundError):
        load_subsidence_pinn_data(
            data_name='zhongshan',
            strategy='load',
            data_home='./non_existent_directory',
            use_cache=False
        )

def test_generate_strategy():
    """Tests the 'generate' strategy to create dummy data."""
    n_samples = 150
    bunch = load_subsidence_pinn_data(
        data_name='nansha', # Test with a different city config
        strategy='generate',
        n_samples=n_samples,
        use_cache=False
    )
    assert isinstance(bunch, XBunch)
    assert len(bunch.frame) == n_samples
    # Check for a nansha-specific column
    assert 'soil_thickness' in bunch.frame.columns

def test_fallback_strategy(mock_data_home):
    """Tests the 'fallback' strategy for both success and failure cases."""
    # Case 1: File exists, should load it from mock_data_home.
    bunch_load = load_subsidence_pinn_data(
        data_name='zhongshan',
        strategy='fallback',
        data_home=mock_data_home,
        use_cache=False
    )
    assert len(bunch_load.frame) == 200 # Should match the file

    # Case 2: File does not exist, should fall back to generating data.
    bunch_generate = load_subsidence_pinn_data(
        data_name='zhongshan',
        strategy='fallback',
        n_samples=50, # Specify n_samples for fallback generation
        data_home='./non_existent_directory',
        use_cache=False
    )
    assert len(bunch_generate.frame) == 50

@pytest.mark.parametrize("encode", [True, False])
@pytest.mark.parametrize("scale", [True, False])
def test_preprocessing_flags(mock_data_home, encode, scale):
    """Tests the encode_categoricals and scale_numericals flags."""
    bunch = load_subsidence_pinn_data(
        data_name='zhongshan',
        data_home=mock_data_home,
        encode_categoricals=encode,
        scale_numericals=scale,
        use_cache=False
    )
    df = bunch.frame
    if encode:
        assert 'geology' not in df.columns and 'geology_Clay' in df.columns
    else:
        assert 'geology' in df.columns and 'geology_Clay' not in df.columns

    if scale:
        print(df['rainfall_mm'].min(), df['rainfall_mm'].max())
        # Check if a numerical column is scaled (values should be ~0-1)
        assert df['rainfall_mm'].min() >= 0 and df['rainfall_mm'].max() <= 1
    else:
        # Check if it remains unscaled (values will likely be > 1)
        assert df['rainfall_mm'].max() > 1

def test_caching(tmp_path):
    """Tests if caching saves and reloads data correctly."""
    # Run 1: Generate data and save to cache. We use 'generate' strategy
    # to avoid needing a source file for this test.
    bunch1 = load_subsidence_pinn_data(
        data_name='zhongshan',
        strategy='generate',
        n_samples=100,
        data_home=str(tmp_path), # Use tmp_path as the cache
        use_cache=True,
        save_cache=True,
    )
    
    # Check that a cache file was created in the specified data_home
    expected_cache_file = tmp_path / "zhongshan_zhongshan_2000_processed.joblib"
    assert expected_cache_file.exists()
    
    # Run 2: Load from cache. This call should not need to generate data.
    # We can prove this by setting n_samples to a different value. If the
    # cache is used, the returned frame will have 100 samples, not 10.
    bunch2 = load_subsidence_pinn_data(
        data_name='zhongshan',
        strategy='generate', # This would generate 10 samples if not for cache
        n_samples=10,
        data_home=str(tmp_path),
        use_cache=True,
        save_cache=False,
    )
    
    # Assert that the loaded data came from the cache (100 rows)
    assert len(bunch2.frame) == 100
    pd.testing.assert_frame_equal(bunch1.frame, bunch2.frame)

def test_return_types(mock_data_home):
    """Tests the `as_frame` flag."""
    # Case 1: Return DataFrame
    result_df = load_subsidence_pinn_data(
        data_name='zhongshan',
        data_home=mock_data_home,
        as_frame=True,
        use_cache=False
    )
    assert isinstance(result_df, pd.DataFrame)
    
    # Case 2: Return XBunch
    result_bunch = load_subsidence_pinn_data(
        data_name='zhongshan',
        data_home=mock_data_home,
        as_frame=False,
        use_cache=False
    )
    assert isinstance(result_bunch, XBunch)
    assert hasattr(result_bunch, 'frame')
    assert hasattr(result_bunch, 'DESCR')

if __name__ =='__main__': # pragma : no cover 
    pytest.main( [__file__,  "--maxfail=1 ", "--disable-warnings",  "-q"])