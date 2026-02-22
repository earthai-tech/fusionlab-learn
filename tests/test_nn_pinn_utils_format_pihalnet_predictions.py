import pytest
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from fusionlab.nn.pinn.utils import format_pihalnet_predictions, _inverse_transform_array


@pytest.fixture
def scaler_setup():
    """
    Creates a fitted MinMaxScaler and the corresponding scaler_info dict.
    
    The scaler is fit on a 2-column array where 'subsidence' (index 0) 
    ranges from -100 to 100, and 'gwl' (index 1) ranges from 10 to 90.
    """
    # Original data range for fitting
    original_data = np.array([
        [-100, 10],  # feature 0 (subsidence), feature 1 (gwl)
        [0, 50],
        [100, 90]
    ])
    
    scaler = MinMaxScaler()
    scaler.fit(original_data)
    
    # This scaler_info dict mocks what 'main_NATCOM_GEOPRIOR.py' generates
    feature_list = ['subsidence', 'gwl']
    scaler_info = {
        'subsidence': {
            'scaler': scaler,
            'all_features': feature_list,
            'idx': 0 
        },
        'gwl': {
            'scaler': scaler,
            'all_features': feature_list,
            'idx': 1
        }
    }
    return scaler_info, scaler

def test_inverse_transform_array_fix(scaler_setup):
    """
    Directly tests the _inverse_transform_array helper to ensure its
    mathematical logic is correct.
    """
    scaler_info, scaler = scaler_setup
    base_name = 'subsidence' # Test on the first feature
    
    # These are the *original*, unscaled values
    original_values = np.array([-50., 0., 75., 100.])
    
    # These are the *scaled* values (what the model would output)
    # We create a dummy array to use the scaler's transform method
    dummy_transform = np.zeros((len(original_values), 2))
    dummy_transform[:, 0] = original_values
    scaled_values = scaler.transform(dummy_transform)[:, 0]
    
    # Now, call our fixed helper function
    inverted_result = _inverse_transform_array(
        scaled_values, 
        base_name, 
        scaler_info, 
        verbose=0
    )
    
    # The result should be the original values
    np.testing.assert_allclose(
        inverted_result, 
        original_values, 
        atol=1e-6,
        err_msg="Inverse transform helper failed to reconstruct original data."
    )

def test_format_pihalnet_predictions_e2e_scaling(scaler_setup):
    """
    Tests the end-to-end functionality of `format_pihalnet_predictions`
    to ensure it correctly uses the fixed helper for both point
    and quantile predictions.
    """
    scaler_info, scaler = scaler_setup
    
    B, H, Q, O = 2, 3, 2, 1 # Batch, Horizon, Quantiles, OutputDim
    QUANTILES = [0.1, 0.9]
    
    # 1. Define ORIGINAL (unscaled) data
    original_subs_data = np.array([
        [-10, -12, -14], # Sample 0, H=3
        [-50, -52, -54]  # Sample 1, H=3
    ]).reshape((B, H, O))
    
    original_subs_quantiles = np.array([
        # Sample 0
        [[-15, -5], [-17, -7], [-19, -9]], # H=3, Q=2
        # Sample 1
        [[-55, -45], [-57, -47], [-59, -49]] # H=3, Q=2
    ]).reshape((B, H, Q, O))

    # 2. Manually SCALE the data (this is what the model outputs)
    
    # Function to scale a (B, H, ...) array using the 'subsidence' (idx 0) scaler
    def scale_data(data, col_idx=0):
        shape = data.shape
        flat_data = data.flatten()
        dummy = np.zeros((len(flat_data), 2)) # 2 features
        dummy[:, col_idx] = flat_data
        scaled_flat = scaler.transform(dummy)[:, col_idx]
        return scaled_flat.reshape(shape)

    scaled_subs_point = scale_data(original_subs_data, col_idx=0)
    scaled_subs_quantiles = scale_data(original_subs_quantiles, col_idx=0)
    
    # 3. Setup Mocks for the formatter function
    
    # Mock for point forecast
    pihalnet_outputs_point = {
        'subs_pred': scaled_subs_point # Shape (B, H, O)
    }
    y_true_dict_point = {
        'subs_pred': scaled_subs_point # Use same data for actuals
    }
    
    # Mock for quantile forecast
    pihalnet_outputs_q = {
        'subs_pred': scaled_subs_quantiles # Shape (B, H, Q, O)
    }
    y_true_dict_q = {
        'subs_pred': scaled_subs_point # Actuals are point values
    }

    # 4. Test Point Forecast Inversion
    df_point = format_pihalnet_predictions(
        pihalnet_outputs=pihalnet_outputs_point,
        y_true_dict=y_true_dict_point,
        scaler_info=scaler_info,
        output_dims={'subs_pred': O},
        verbose=0
    )
    
    # Check if the 'subsidence_pred' column matches the original unscaled data
    np.testing.assert_allclose(
        df_point['subsidence_pred'].values,
        original_subs_data.flatten(),
        atol=1e-6,
        err_msg="E2E test failed: 'subsidence_pred' was not correctly inverted."
    )
    # Check if the 'subsidence_actual' column was also inverted
    np.testing.assert_allclose(
        df_point['subsidence_actual'].values,
        original_subs_data.flatten(),
        atol=1e-6,
        err_msg="E2E test failed: 'subsidence_actual' was not correctly inverted."
    )

    # 5. Test Quantile Forecast Inversion
    df_q = format_pihalnet_predictions(
        pihalnet_outputs=pihalnet_outputs_q,
        y_true_dict=y_true_dict_q,
        scaler_info=scaler_info,
        quantiles=QUANTILES,
        output_dims={'subs_pred': O},
        verbose=0
    )
    
    # Need to flatten in the same order as the formatter
    # Formatter does: (B, H, Q, O) -> (B*H, Q*O)
    # Our helper _add_quantiles_to_df iterates q_idx first, then o_idx
    # Original shape (B, H, Q, O) = (2, 3, 2, 1)
    
    # q10 values: original_subs_quantiles[..., 0, 0] (all samples, all steps, 0th quantile, 0th output)
    original_q10_flat = original_subs_quantiles[..., 0, 0].flatten()
    # q90 values: original_subs_quantiles[..., 1, 0] (all samples, all steps, 1st quantile, 0th output)
    original_q90_flat = original_subs_quantiles[..., 1, 0].flatten()

    np.testing.assert_allclose(
        df_q['subsidence_q10'].values,
        original_q10_flat,
        atol=1e-6,
        err_msg="E2E test failed: 'subsidence_q10' was not correctly inverted."
    )
    np.testing.assert_allclose(
        df_q['subsidence_q90'].values,
        original_q90_flat,
        atol=1e-6,
        err_msg="E2E test failed: 'subsidence_q90' was not correctly inverted."
    )
    np.testing.assert_allclose(
        df_q['subsidence_actual'].values,
        original_subs_data.flatten(), # Actuals are still the point values
        atol=1e-6,
        err_msg="E2E test failed: 'subsidence_actual' (quantile) was not correctly inverted."
    )