

# --- City-Specific Default Configurations ---
CITY_CONFIGS = {
    "zhongshan": {
        "metadata": '{}',  # Use the 500k metadata
        "time_col": "year",
        "dt_col_name": "datetime_temp_zs",
        "lon_col": "longitude",
        "lat_col": "latitude",
        "subsidence_col": "subsidence",
        "gwl_col": "GWL",
        "categorical_cols": [
            "geology",
            "density_tier",
            "rainfall_category"
        ],  
        "numerical_main": [
            # Features for main scaling pass
            "rainfall_mm",
            "normalized_density",
            "normalized_seismic_risk_score"
        ],
        "default_value_cols_interpolate": [
            # For augmentation
            "GWL",
            "rainfall_mm",
            "normalized_density",
            "normalized_seismic_risk_score"
        ],
        "default_feature_cols_augment": [
            # For augmentation
            "GWL",
            "rainfall_mm",
            "normalized_density",
            "normalized_seismic_risk_score"
        ],
        "known_other_cols": [
            "subsidence_intensity",
            "density_concentration"
        ]
    },
    "nansha": {
        "metadata": '{}',  # Or specific Nansha metadata
        "time_col": "year",
        "dt_col_name": "datetime_temp_ns",
        "lon_col": "longitude",
        "lat_col": "latitude",
        "subsidence_col": "subsidence",
        "gwl_col": "GWL",
        "categorical_cols": [
            "geology",
            "building_concentration"
        ],
        "numerical_main": [
            "rainfall_mm",
            "normalized_seismic_risk_score",
            "soil_thickness"
        ],
        "default_value_cols_interpolate": [
            "GWL",
            "rainfall_mm",
            "normalized_seismic_risk_score",
            "soil_thickness"
        ],
        "default_feature_cols_augment": [
            "GWL",
            "rainfall_mm",
            "normalized_seismic_risk_score",
            "soil_thickness"
        ],
        "known_other_cols": []  # For future extension
    }
}
