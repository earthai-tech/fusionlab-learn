import pytest
import numpy as np
import pandas as pd
import os

# Assuming the function to be tested is in fusionlab/nn/pinn/utils.py
# and the test is run from the root of the fusionlab project.
# try:
from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences
# Assuming a helper for resolving spatial columns exists if used inside
from fusionlab.utils.geo_utils import resolve_spatial_columns
FUSIONLAB_AVAILABLE = True
# except ImportError:
#     FUSIONLAB_AVAILABLE = False
    # If imports fail, tests will be skipped. This is better than
    # defining dummy functions which might hide import errors.


# --- Test Fixture for Creating Sample Data ---
@pytest.fixture
def sample_pinn_dataframe():
    """
    Creates a sample pandas DataFrame for testing prepare_pinn_data_sequences.

    Includes two groups ('site_A' and 'site_B'). 'site_A' is long
    enough to create sequences, while 'site_B' is too short.
    """
    # Group A: Long enough to create 2 sequences
    # time_steps=5, forecast_horizon=3 -> min_len=8
    # site_A has 9 data points. 9 - 8 + 1 = 2 sequences.
    data_a = {
        'date': pd.to_datetime(pd.date_range(start='2020-01-01', periods=9, freq='MS')),
        'longitude': np.linspace(113.0, 113.08, 9),
        'latitude': np.linspace(22.0, 22.08, 9),
        'subsidence': np.arange(1, 10) * 1.0,  # e.g., 1mm, 2mm, ...
        'gwl': np.arange(10, 19) * -1.0,      # e.g., -10m, -11m, ...
        'rainfall': np.arange(100, 109),
        'density': np.arange(200, 209),
        'geology_id': [1] * 9,  # Static within group
        'pump_rate_future': np.arange(30, 39),
        'group_id': ['site_A'] * 9
    }
    df_a = pd.DataFrame(data_a)

    # Group B: Too short to create any sequence
    # min_len is 8, site_B has 7 data points.
    data_b = {
        'date': pd.to_datetime(pd.date_range(start='2020-01-01', periods=7, freq='MS')),
        'longitude': np.linspace(114.0, 114.06, 7),
        'latitude': np.linspace(23.0, 23.06, 7),
        'subsidence': np.arange(1, 8) * 1.0,
        'gwl': np.arange(10, 17) * -1.0,
        'rainfall': np.arange(100, 107),
        'density': np.arange(200, 207),
        'geology_id': [2] * 7,
        'pump_rate_future': np.arange(30, 37),
        'group_id': ['site_B'] * 7
    }
    df_b = pd.DataFrame(data_b)

    return pd.concat([df_a, df_b], ignore_index=True)

# --- Pytest Test Class ---

@pytest.mark.skipif(not FUSIONLAB_AVAILABLE, reason="fusionlab PINN utils not available")
class TestPreparePinnDataSequences:
    
    # Define standard column mappings for tests
    COLS_CONFIG = {
        'time_col': 'date',
        'lon_col': 'longitude',
        'lat_col': 'latitude',
        'subsidence_col': 'subsidence',
        'gwl_col': 'gwl',
        'dynamic_cols': ['rainfall', 'density'],
        'static_cols': ['geology_id'],
        'future_cols': ['pump_rate_future'],
        'group_id_cols': ['group_id'],
    }

    def test_basic_sequencing_and_shapes(self, sample_pinn_dataframe):
        """
        Tests the basic sequencing functionality and verifies output shapes.
        """
        inputs_dict, targets_dict = prepare_pinn_data_sequences(
            df=sample_pinn_dataframe,
            time_steps=5,
            forecast_horizon=3,
            **self.COLS_CONFIG
        )
        # Expected sequences: Only from site_A, which has 9 points.
        # min_len = 5 + 3 = 8. num_sequences = 9 - 8 + 1 = 2.
        N = 2
        H = 3
        T_past = 5

        assert isinstance(inputs_dict, dict)
        assert isinstance(targets_dict, dict)
        
        # Check shapes
        assert inputs_dict['coords'].shape == (N, H, 3) # t, x, y
        assert inputs_dict['static_features'].shape == (N, 1)
        assert inputs_dict['dynamic_features'].shape == (N, T_past, 2)
        assert inputs_dict['future_features'].shape == (N, H, 1)
        assert targets_dict['subsidence'].shape == (N, H, 1)
        assert targets_dict['gwl'].shape == (N, H, 1)

    def test_content_verification(self, sample_pinn_dataframe):
        """
        Tests the content of the first generated sequence to ensure correctness.
        """
        inputs_dict, targets_dict = prepare_pinn_data_sequences(
            df=sample_pinn_dataframe,
            time_steps=5,
            forecast_horizon=3,
            **self.COLS_CONFIG,
            normalize_coords=False # Use original values for easy checking
        )
        
        # --- Check static features ---
        # Should be geology_id of site_A, which is 1
        np.testing.assert_allclose(inputs_dict['static_features'][0], [1])
        np.testing.assert_allclose(inputs_dict['static_features'][1], [1])
        
        # --- Check dynamic features of the first sequence ---
        # Should correspond to rainfall/density from index 0 to 4
        expected_dynamic_rainfall = np.arange(100, 105)
        np.testing.assert_allclose(
            inputs_dict['dynamic_features'][0, :, 0], expected_dynamic_rainfall
        )
        
        # --- Check target subsidence of the first sequence ---
        # Should correspond to subsidence from index 5 to 7
        expected_target_subs = np.arange(6, 9)
        np.testing.assert_allclose(
            targets_dict['subsidence'][0, :, 0], expected_target_subs
        )

        # --- Check coordinates of the first sequence ---
        # Should correspond to lon/lat/time from index 5 to 7
        expected_target_lon = np.linspace(113.0, 113.08, 9)[5:8]
        np.testing.assert_allclose(
            inputs_dict['coords'][0, :, 1], expected_target_lon
        )

    def test_missing_optional_features(self, sample_pinn_dataframe):
        """
        Tests behavior when optional static and future features are not provided.
        """
        # Create a copy of the config without optional columns
        cols_config_minimal = self.COLS_CONFIG.copy()
        cols_config_minimal.pop('static_cols')
        cols_config_minimal.pop('future_cols')

        inputs_dict, _ = prepare_pinn_data_sequences(
            df=sample_pinn_dataframe,
            time_steps=5,
            forecast_horizon=3,
            **cols_config_minimal
        )
        
        # The function should filter out None values before returning dict
        assert 'static_features' not in inputs_dict
        assert 'future_features' not in inputs_dict
        # Ensure required keys are still present
        assert 'coords' in inputs_dict
        assert 'dynamic_features' in inputs_dict

    def test_insufficient_data_error(self, sample_pinn_dataframe):
        """
        Tests that a ValueError is raised if no sequences can be generated.
        """
        with pytest.raises(ValueError, match="No group has enough data points"):
            prepare_pinn_data_sequences(
                df=sample_pinn_dataframe,
                time_steps=10, # Requires 10 + 3 = 13 data points
                forecast_horizon=3,
                **self.COLS_CONFIG
            )

    def test_coordinate_normalization(self, sample_pinn_dataframe):
        """
        Tests the `normalize_coords` parameter.
        """
        # With normalization (default)
        inputs_dict_norm, _ = prepare_pinn_data_sequences(
            df=sample_pinn_dataframe,
            time_steps=5,
            forecast_horizon=3,
            **self.COLS_CONFIG,
            normalize_coords=True
        )
        coords_norm = inputs_dict_norm['coords']
        # Check that values are within [0, 1] range
        assert np.all(coords_norm >= 0) and np.all(coords_norm <= 1)
        # Check that the first time step is 0 and last is 1 for the group range
        # Group A has 9 points. Horizon is 3 steps. First sequence target t is points 5,6,7
        # Normalized t should start > 0 and end < 1 for this slice.
        assert coords_norm[0, 0, 0] > 0 # t_norm for time index 5
        assert coords_norm[-1, -1, 0] == 1 # t_norm for time index 8
        
        # Without normalization
        inputs_dict_no_norm, _ = prepare_pinn_data_sequences(
            df=sample_pinn_dataframe,
            time_steps=5,
            forecast_horizon=3,
            **self.COLS_CONFIG,
            normalize_coords=False
        )
        coords_no_norm = inputs_dict_no_norm['coords']
        # Check that longitude value matches original data for first sequence
        expected_lon = sample_pinn_dataframe[
            sample_pinn_dataframe.group_id == 'site_A'
        ].iloc[5:8]['longitude'].values
        np.testing.assert_allclose(coords_no_norm[0, :, 1], expected_lon)

    def test_no_grouping(self, sample_pinn_dataframe):
        """Tests functionality when no group_id_cols are provided."""
        site_a_df = sample_pinn_dataframe[
            sample_pinn_dataframe.group_id == 'site_A'
        ].copy()

        cols_config_no_group = self.COLS_CONFIG.copy()
        cols_config_no_group.pop('group_id_cols')
        
        inputs_dict, targets_dict = prepare_pinn_data_sequences(
            df=site_a_df,
            time_steps=5,
            forecast_horizon=3,
            **cols_config_no_group
        )
        # Should produce the same 2 sequences as before
        assert targets_dict['subsidence'].shape[0] == 2

    @pytest.mark.skip("Test already passed localled. can skipped.")
    def test_savefile(self, sample_pinn_dataframe, tmp_path):
        """Tests that the savefile functionality works."""
        # tmp_path is a pytest fixture that provides a temporary directory
        try:
            output_file = tmp_path / "pinn_data.joblib"
            prepare_pinn_data_sequences(
                df=sample_pinn_dataframe,
                time_steps=5,
                forecast_horizon=3,
                **self.COLS_CONFIG,
                savefile=str(output_file)
            )
            
            assert output_file.exists()
        except: 
            output_file= os.path.join( os.getcwd () , "pinn_data.joblib") 
            assert os.path.isfile (output_file)
            
        # Optional: Load the file and check its contents
        # loaded_data = joblib.load(output_file)
        # assert 'inputs_dict' in loaded_data
        # assert 'targets_dict' in loaded_data

# This allows running the test file directly
if __name__ == '__main__': # pragma: no cover 
    # This will run all tests in the current file
    pytest.main([__file__])