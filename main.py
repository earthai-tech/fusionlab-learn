# -*- coding: utf-8 -*-
"""
zhongshan_forecast_2023_2026.py

This script performs quantile-based subsidence prediction for the next 
four years (2023-2026) using the XTFT deep learning model. The data for 
Zhongshan spans from 2015 to 2023. For training, data from 2015 to 2022 
is used, while 2023 is reserved for validation. The model forecasts 
subsidence for 2023-2026 with a forecast horizon of 4 and a rolling 
window (time_steps) of 3.

Import note: This script main.py used for reproducibitlity prupose takes only 
2,000  over 4,449,321 samples  of zhongshan subsidence original data load 
directly from the datasets. 
Full complete data should be available upon requests. However, the 2,000 samples 
is uysed for reproducility and how prediction can be made. 


Workflow:
----------
1. Data preprocessing and feature engineering.
2. Encoding categorical features and normalizing numerical data.
3. Splitting the dataset into training (2015-2022) and testing (2023).
4. Reshaping data for XTFT model input using `reshape_xtft_data`.
5. Train-validation split for model training.
6. Training and evaluation of the XTFT model.
7. Generating forecasts for 2023-2026 and validating the 2023 
   prediction against actual observations.
8. Visualization of actual versus predicted subsidence.

Author: LKouadio
"""

# ==========================================
#  SECTION 1: LIBRARY IMPORTS & DATA LOADING
# ==========================================
import os
# import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
# import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score

from fusionlab.nn.transformers import XTFT, SuperXTFT
from fusionlab.utils.data_utils import nan_ops
from fusionlab.nn.utils import reshape_xtft_data
# from fusionlab.metrics_special import coverage_score
from fusionlab.nn.utils import visualize_forecasts#, generate_forecast,
from fusionlab.nn.utils import forecast_multi_step
from fusionlab.nn.losses import combined_quantile_loss
from fusionlab.utils.io_utils import fetch_joblib_data   
#%
# main_path = r'D:\nature_data\zhongshan_data\another_tests'
from fusionlab.datasets._property import get_data
# from fusionlab.datasets.load import _ZHONGSHAN_METADATA, _NANSHA_METADATA
from fusionlab.datasets import fetch_zhongshan_data, fetch_nansha_data 

# =================== CONFIG PARAMS ========================================

USE_SUPER =False 

super_ext = '' if not USE_SUPER else '_super'
TRANSF_= SuperXTFT if USE_SUPER else XTFT  # TRANSF_ MEANS Fusion transformer 
forecast_years = [2022, 2023, 2024, 2025]  # for NANSHA WHEN [2023, 2024, 2025, 2026]
time_steps =3 
# IMPORTANT NOTE: Spatial columns is important for runing, however need enough data 
# for reproducityky and testting pupose wen turn of too None 
spatial_cols=None#("longitude", "latitude")
quantiles = [ 0.1, 0.5, 0.9 ] # quantiles =None, for point prediction 

# ===========================================================================

# ------------------------------------------
# ** Step 1: Define Data Paths**
# ------------------------------------------

data_path = get_data()
     
# data_path = os.path.join(main_path,'qt_forecast_2023_2026' )

# # Load dataset
# zhongshan_file = os.path.join(main_path, 'zhongshan_filtered_final_data.csv')

# zhongshan_data = fetch_zhongshan_data().frame
# zhongshan_data = fetch_nansha_data().frame

zhongshan_data = pd.read_csv(r'J:\nature_data\final\nansha_200_0000.csv')
# # Rename geological category column for consistency
zhongshan_data.rename(columns={"geological_category": "geology"}, inplace=True)

# Backup original dataset
zhongshan_data_original = zhongshan_data.copy()

# ------------------------------------------
# ** Step 2: Feature Selection**
# ------------------------------------------
selected_features = [
    'longitude', 'latitude', 'year', 
    'GWL','rainfall_mm', 'geology',
    
    # --> density_tier and normalized_density are only valid for zhongshan_data 
    #'density_tier',  'normalized_density', # (only in zhongshan)'
    
    #-->  Now soil_thickness and building_concentration are  only valid for nanshan_data, 
     'soil_thickness', 'building_concentration', # (only for Nansha)
     
    'normalized_seismic_risk_score', 'subsidence'
]
zhongshan_data = zhongshan_data[selected_features].copy()

# Check and fix NaN values
print(f" NaN exists before processing? "
      f"{zhongshan_data.isna().any().any()}")
zhongshan_data = nan_ops(zhongshan_data, ops='sanitize', 
                          action='fill', process="do_anyway")
print(f" NaN exists after processing? "
      f"{zhongshan_data.isna().any().any()}")

# =================================================
#  SECTION 2: FEATURE ENGINEERING & NORMALIZATION
# =================================================

#  Step 3: Encoding Categorical Features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Encode 'geology'
geology_encoded = encoder.fit_transform(zhongshan_data[['geology']])
geology_columns = [f'geology_{cat}' for cat in encoder.categories_[0]]


# Encode 'density_tier'

if 'density_tier' in selected_features: 
    density_tier_encoded = encoder.fit_transform(zhongshan_data[['density_tier']])
    density_tier_columns = [f'density_tier_{cat}' for cat in encoder.categories_[0]]
  
else: # use building concentration # for Nansha 
    density_tier_encoded = encoder.fit_transform(zhongshan_data[['building_concentration']])
    density_tier_columns = [f'building_concentration_{cat}' for cat in encoder.categories_[0]]
    
# Convert encoded arrays to DataFrames
geology_df = pd.DataFrame(geology_encoded, columns=geology_columns)
density_df = pd.DataFrame(density_tier_encoded, columns=density_tier_columns)

#  Step 4: Normalize Numerical Features (excluding longitude & latitude, year,  subsidence)
scaler = MinMaxScaler()
columns_to_normalize= ['GWL', 'rainfall_mm',] # skip renormalizing normalized_density and 
# normalized_seismic_risk_score since they are already normalized. 
zhongshan_data[columns_to_normalize] = scaler.fit_transform(
    zhongshan_data[columns_to_normalize]
)
joblib.dump(scaler, os.path.join(data_path, 'zhongshan_scaler_at.joblib'))

print("Columns before dropping geology/density_tier:\n",
      list(zhongshan_data.columns))
#  Step 5: Merge Encoded Features & Clean Data
zhongshan_data = pd.concat([zhongshan_data, geology_df, density_df],
                            axis=1)
if 'density_tier' in selected_features: # when use zhongshan data 
    zhongshan_data.drop(columns=['geology', 'density_tier'], inplace=True)
else: # when use Nansha data 
    zhongshan_data.drop(columns=['geology', 'building_concentration'], inplace=True)
    
print("Columns after dropping geology/density_tier:\n",
      list(zhongshan_data.columns))

# ==========================================
# SECTION 3: DATA SPLITTING & SEQUENCING
# ==========================================

# For ZHONGSHAN: 
if 'density_tier' in selected_features: 
    # Step 6: Split Data for Training and Testing
    # Training data: years 2015 to 2022; Testing data: year 2023
    train_data = zhongshan_data[zhongshan_data['year'] <= 2022].copy()
    train_data.sort_values('year', inplace=True)
    test_data  = zhongshan_data[zhongshan_data['year'] == 2023].copy()
else:
    # FOR NANSHA : 
        # Training data: years 2015 to 2021; Testing data: year 2022
    train_data = zhongshan_data[zhongshan_data['year'] <= 2021].copy()
    train_data.sort_values('year', inplace=True)
    test_data  = zhongshan_data[zhongshan_data['year'] == 2022].copy()
    
# Step 7: Define Feature Sets
static_features  = ['longitude', 'latitude'] + \
                    list(geology_df.columns) + list(density_df.columns)
dynamic_features = ['GWL', 'rainfall_mm', 
                    'normalized_seismic_risk_score',
                    # 'normalized_density'# valid only for zhongshan
                    ]
future_features  = ['rainfall_mm']

print("Actual subsidence for 2023 (validation):\n")
print(test_data["subsidence"].head())

# # Step 8: Generate Sequences Using `reshape_xtft_data`
# # Set time_steps = 3 and forecast_horizons = 4 to forecast 2023-2026
# from fusionlab.datasets import load_processed_subsidence_data 
forecast_horizon = len(forecast_years)
time_steps = forecast_horizon -1  if time_steps is None else time_steps 

is_valid_time_steps = time_steps <= forecast_horizon
time_steps       = time_steps if is_valid_time_steps else ( 
    forecast_horizon - 1  if forecast_horizon > 1 else 1) 


# X_static, X_dynamic, X_future, y_train_seq = load_processed_subsidence_data (
#     'zhongshan',return_sequences =True, 
#     time_steps= time_steps, 
#     forecast_horizons= forecast_horizon, 
#     save_processed_frame= True, 
#     save_sequences=True, 
#     verbose = 7 
#     )

# 
# #%
print("Reshaping training data for XTFT input...\n")
if os.path.isfile (os.path.join(data_path, 'qt.2023_2026.train_data_v3.joblib')): 
   X_static, X_dynamic, X_future, y_train_seq  = fetch_joblib_data ( 
       os.path.join(data_path, 'qt.2023_2026.train_data_v3.joblib'), 
       'static_data', 'dynamic_data', 'future_data', 'target_data', 
       verbose=7 , 
       ) 
else: 
   
    # if not then reshape again 
    X_static, X_dynamic, X_future, y_train_seq = reshape_xtft_data(
        train_data,
        dt_col            = "year",
        target_col        = "subsidence",
        static_cols       = static_features,
        dynamic_cols      = dynamic_features,
        future_cols       = future_features,
        time_steps        = time_steps,
        forecast_horizons = forecast_horizon,
        spatial_cols      = spatial_cols, 
        savefile          = os.path.join(data_path, 'qt.2023_2026.train_data_v3.joblib'), 
        verbose           = 7 
    )
    print("yes")
#%
print("test_data=", test_data)
print("train_data", train_data)
print("train_data.year", train_data['year'].unique())


print(f"Static input shape:  {X_static.shape}")
print(f"Dynamic input shape: {X_dynamic.shape}")
print(f"Future input shape:  {X_future.shape}")
print(f"Target sequence shape: {y_train_seq.shape}")

# Step 9: Train-Validation Split (80-20)
print("Splitting training sequences into train and validation sets...\n")

X_static_train, X_static_val, \
X_dynamic_train, X_dynamic_val, \
X_future_train, X_future_val, \
y_train, y_val = train_test_split(
    X_static, X_dynamic, X_future, y_train_seq,
    test_size    = 0.2,
    random_state = 42
)



print("Training set shapes:")
print(f"  Static:  {X_static_train.shape}")
print(f"  Dynamic: {X_dynamic_train.shape}")
print(f"  Future:  {X_future_train.shape}")
print(f"  Target:  {y_train.shape}\n")

print("Validation set shapes:")
print(f"  Static:  {X_static_val.shape}")
print(f"  Dynamic: {X_dynamic_val.shape}")
print(f"  Future:  {X_future_val.shape}")
print(f"  Target:  {y_val.shape}")
#%
# # Save prepared data for reproducibility
if os.path.isfile(os.path.join(data_path, 'qt.prepared_data_2023_2026_v3.joblib')): 
    test_data = fetch_joblib_data (
        os.path.join(data_path, 'qt.prepared_data_2023_2026_v3.joblib'), 
        'test_data'
        )
else: 
    
    data_prepared = {
        "data"             : zhongshan_data,
        "train_data"       : train_data,
        "test_data"        : test_data,
        "selected_features": selected_features,
    }
    joblib.dump(data_prepared, os.path.join(data_path, 'qt.prepared_data_2023_2026_v3.joblib'))
    print(f"Prepared data saved at "
          f"{os.path.join(data_path, 'qt.prepared_data_2023_2026_v3.joblib')}.")

if not os.path.isfile ( os.path.join(data_path, 'qt.xtft_data_2023_2026_v3.joblib')):
    data_dict = {
        'X_static_train': X_static_train,
        'X_dynamic_train': X_dynamic_train,
        'X_future_train': X_future_train,
        'y_train': y_train,
        'X_static_val': X_static_val,
        'X_dynamic_val': X_dynamic_val,
        'X_future_val': X_future_val,
        'y_val': y_val,
        'y':y_train_seq, 
        'X_static': X_static,
        'X_dynamic': X_dynamic,
        'X_future': X_future,
        'y_train_seq': y_train_seq
    }
    joblib.dump(data_dict, os.path.join(data_path, 'qt.xtft_data_2023_2026_v3.joblib'))
    print(f"Processed data saved at "
          f"{os.path.join(data_path, 'qt.xtft_data_2023_2026_v3.joblib')}.")

# ==========================================
#  SECTION 4: TRAINING XTFT MODEL
# ==========================================
best_params = {
    'embed_dim'       : 32,
    'max_window_size' : 3,
    'memory_size'     : 100,
    'num_heads'       : 4,
    'dropout_rate'    : 0.1,
    'lstm_units'      : 64,
    'attention_units' : 64,
    'hidden_units'    : 32, 
    'multi_scale_agg' : 'auto', 
   #  'anomaly_detection_strategy': 'feature_based', 
    
}

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor              = 'val_loss',
    patience             = 5,
    restore_best_weights = True
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(data_path, f'qt.xtft_best_model_2023_2026_v3{super_ext}'),
    monitor           = 'val_loss',
    save_best_only    = True,
    save_weights_only = False,  # Save entire model
    verbose           = 1, 
    save_format='tf', 
)
#%



xtft_model = TRANSF_(
    static_input_dim  = X_static_train.shape[1],
    dynamic_input_dim = X_dynamic_train.shape[2],
    future_input_dim  = X_future_train.shape[2],
    
    forecast_horizon = forecast_horizon,
    quantiles         = quantiles,
    **best_params
)
if quantiles is not None: 
    loss_fn = combined_quantile_loss (quantiles)
else: 
    loss_fn ='mse'
    
xtft_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss=loss_fn 
)

print("Training the XTFT model...\n")
xtft_model.fit(
    x              = [X_static_train, X_dynamic_train, X_future_train],
    y              = y_train,
    epochs         = 50,
    batch_size     = 32,
    validation_data= ([X_static_val, X_dynamic_val, X_future_val], y_val),
    callbacks      = [early_stopping, model_checkpoint]
)
# try: 
#     print("Trying saving the model direct way")
#     xtft_model.save(
#         os.path.join(data_path, 'qt.xtft_best_model_fixed_super'),
#         save_format='tf',
#         include_optimizer=False,
#         custom_objects={"combined_quantile_loss": combined_quantile_loss}
#     )
# except :
#     pass 

# %
# import gofast 
# import tensorflow as tf
# from fusionlab.nn.losses import combined_quantile_loss
#%
print("Try to fetch model ... ")

try : 
    print("Fetch_model  from direct way")
    xtft_model = tf.keras.models.load_model(
        os.path.join(data_path, f'qt.xtft_best_model_fixed{super_ext}')
    )
except: 
    print("Failed: Cannot restored model from direct ")
    try: 
        # tf.keras.utils.get_custom_objects()["combined_quantile_loss"] = combined_quantile_loss
        print("trying recovering the model saved during training ")
       
        from tensorflow.keras.utils import custom_object_scope
        with custom_object_scope({"combined_quantile_loss": combined_quantile_loss}):
            xtft_model = tf.keras.models.load_model(
                os.path.join(data_path, f'qt.xtft_best_model_2023_2026_v3{super_ext}')
            )
    except: 
        print("Failed: d restored model.")
    else: 
        print("TRAINING way fetching model successfully completed. ")
else: 
    print("DIRECT way fetching model successfully completed. ")
    

# from tensorflow.keras.utils import custom_object_scope

# with custom_object_scope({"combined_quantile_loss": combined_quantile_loss}):
#     xtft_model = tf.keras.models.load_model(
#         os.path.join(data_path, 'qt.xtft_best_model_2023_2026_v3')
#     )

# xtft_model = tf.keras.models.load_model (
#     os.path.join(data_path, 'qt.xtft_best_model_2023_2026_v3'), 
#     custom_objects={"combined_quantile_loss": combined_quantile_loss }
# )
#%
# ==========================================
#  SECTION 5: FORECASTING & EVALUATION
# ==========================================
print("Generating Super quantile forecast for 2023-2026...\n")

# Get all unique location coordinates
#unique_locations = train_data[['longitude', 'latitude']].drop_duplicates()

# Prepare empty lists for forecast results
# forecast_results = []
#%


# from fusionlab.nn.utils import generate_forecast 

# forecast_df0 = generate_forecast(
#     xtft_model, 
#     train_data, 
#     dt_col='year' , 
#     dynamic_features= dynamic_features, 
#     static_features= static_features, 
#     future_features = future_features, 
#     test_data = test_data, 
#     forecast_dt= forecast_years, 
#     spatial_cols = ('longitude', 'latitude'), 
#     forecast_horizon= forecast_horizon, 
#     time_steps = 2, 
#     q= quantiles if quantiles is not None else None, 
#     tname ='subsidence', 
#     savefile = os.path.join(data_path, f"qt.forecast_results_2023_2026_v3{super_ext}_2.csv"), 
#     verbose=7 
#   )
#%
forecast_df2 = forecast_multi_step(
     xtft_model=xtft_model,
     inputs=[X_static, X_dynamic, X_future],
     forecast_horizon=forecast_horizon,
     y=y_train_seq,
     dt_col="year",
     mode="quantile" if quantiles else "point",
     #spatial_cols=["longitude", "latitude"],
     q= quantiles, 
     tname="subsidence",
     forecast_dt=forecast_years,
     #apply_mask=True,
     mask_values= 0, 
     mask_fill_value=0, 
     savefile = os.path.join(data_path, f"qt.forecast_results_2023_2026_v3{super_ext}.csv"), 
     verbose=7
 )
#%
# get manaully the longitude latitude since data is not enought then grouped coordinates 
# are not made.we can get the coordinate manually from X_static 
 
ll_df = pd.DataFrame() 
ll_df[['longitude', 'latitude']] = X_static[:, :2]

ll_dupl = [ll_df[['longitude', 'latitude']] for _ in range(len(forecast_years))]

ll_df= pd.concat(ll_dupl, ignore_index= True, axis =0 ) 
#
forecast_df3 = pd.concat ([ll_df, forecast_df2, ], axis =1) 
#%
visualize_forecasts(
    forecast_df3, 
    dt_col="year", 
    tname="subsidence", 
    # actual_name ='subsidence_q50', # use when generate_forecast is triggered.
    eval_periods= forecast_years, 
    test_data = test_data, 
    mode="quantile" if quantiles is not None else 'point', 
    kind= "spatial", 
    x="longitude", 
    y="latitude",
    max_cols=2, 
    axis="off", 
    verbose=7 ,
)

