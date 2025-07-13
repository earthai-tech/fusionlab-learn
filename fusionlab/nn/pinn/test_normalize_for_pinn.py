import pytest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from fusionlab.utils.forecast_utils import normalize_for_pinn

# Sample test data
@pytest.fixture
def sample_df():
    data = {
        'year': [2019, 2020, 2021, 2022],
        'coord_x': [113.151405, 113.151405, 113.151405, 113.151405],
        'coord_y': [22.632114, 22.632114, 22.632114, 22.632114],
        'subsidence': [0.064863, 0.077552, 0.090091, 0.102047]
    }
    df = pd.DataFrame(data)
    return df


def test_forecast_horizon_added_before_scaling(sample_df):
    forecast_horizon = 4  # Adding 4 years

    # Use the normalize_for_pinn function with scale_coords set to True
    df_scaled, coord_scaler, _ = normalize_for_pinn(
        sample_df,
        time_col='year',
        coord_x='coord_x',
        coord_y='coord_y',
        scale_coords=True,
        forecast_horizon=forecast_horizon,
        verbose=2
    )

    # Check if the forecast horizon was correctly added before scaling
    expected_time_after_horizon = [2019 + forecast_horizon, 2020 + forecast_horizon, 
                                   2021 + forecast_horizon, 2022 + forecast_horizon]
    
    # Assert that the time column was correctly adjusted
    print(df_scaled['year'].iloc[0], expected_time_after_horizon[0])
    print(df_scaled['year'].iloc[1], expected_time_after_horizon[1])
    print(df_scaled['year'].iloc[2], expected_time_after_horizon[2])
    print(df_scaled['year'].iloc[3], expected_time_after_horizon[3])

    # Apply inverse transformation to check if the dates align
    time_col_scaled = df_scaled[['year', 'coord_x', 'coord_y']].values  # Extract full scaled data

    # Inverse transform only the 'year' column
    time_col_inversed = coord_scaler.inverse_transform(time_col_scaled)[:, 0]  # Only take the first column (year)
    df_inv_scaled = df_scaled.copy()
    df_inv_scaled['year'] = time_col_inversed.round(1).astype(int)

    # Print for checking
    print(df_inv_scaled['year'].iloc[0], expected_time_after_horizon[0])
    print(df_inv_scaled['year'].iloc[1], expected_time_after_horizon[1])
    print(df_inv_scaled['year'].iloc[2], expected_time_after_horizon[2])
    print(df_inv_scaled['year'].iloc[3], expected_time_after_horizon[3])
    
    # Assert the inverse transformation gives the correct result
    assert df_inv_scaled['year'].iloc[0] == expected_time_after_horizon[0]
    assert df_inv_scaled['year'].iloc[1] == expected_time_after_horizon[1]
    assert df_inv_scaled['year'].iloc[2] == expected_time_after_horizon[2]
    assert df_inv_scaled['year'].iloc[3] == expected_time_after_horizon[3]


# Test that scaling was performed correctly
def test_scaling_correct(sample_df):
    forecast_horizon = 4  # Adding 4 years

    # Use the normalize_for_pinn function
    df_scaled, coord_scaler, _ = normalize_for_pinn(
        sample_df,
        time_col='year',
        coord_x='coord_x',
        coord_y='coord_y',
        scale_coords=True,
        forecast_horizon=forecast_horizon,
        verbose=2
    )

    # Ensure that the scaled values of the time column are in [0, 1] range
    assert df_scaled['year'].min() >= 0
    assert df_scaled['year'].max() <= 1

    # Check if the scaling correctly normalized the year after adjustment
    assert df_scaled['year'].iloc[0] != df_scaled['year'].iloc[1]  # Ensure there are differences

# Test the inverse scaling to check if forecasted dates are correct
def test_inverse_scaling_correct(sample_df):
    forecast_horizon = 4  # Adding 4 years

    # Apply scaling
    df_scaled, coord_scaler, _ = normalize_for_pinn(
        sample_df,
        time_col='year',
        coord_x='coord_x',
        coord_y='coord_y',
        scale_coords=True,
        forecast_horizon=forecast_horizon,
        verbose=2
    )

    # Apply inverse transformation to check if the dates align
    # Pass the entire scaled data (time_col, coord_x, coord_y)
    time_col_scaled = df_scaled[['year', 'coord_x', 'coord_y']].values  # Extract full scaled data
    time_col_inversed = coord_scaler.inverse_transform(time_col_scaled)[:, 0]  # Only take the 'year' column

    # The inverse-transformed values should align with the adjusted time values
    expected_time_after_inverse = [2019 + forecast_horizon, 2020 + forecast_horizon,
                                   2021 + forecast_horizon, 2022 + forecast_horizon]

    for i in range(4):
        assert abs(time_col_inversed[i] - expected_time_after_inverse[i]) < 1e-6  # Allow some small error

# Test the case where no forecast_horizon is provided
def test_no_forecast_horizon(sample_df):
    # No forecast horizon provided, so scaling will happen normally without adjustment
    df_scaled, coord_scaler, _ = normalize_for_pinn(
        sample_df,
        time_col='year',
        coord_x='coord_x',
        coord_y='coord_y',
        scale_coords=True,
        verbose=2
    )

    # Ensure that the year column is still scaled (no forecast horizon adjustment)
    assert df_scaled['year'].iloc[0] != 2019  # It should be normalized, not equal to 2019


if __name__ == '__main__':
    pytest.main([__file__])
