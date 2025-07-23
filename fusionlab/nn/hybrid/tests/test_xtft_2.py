# tests/test_xtft.py
# -*- coding: utf-8 -*-
import pytest
import numpy as np

tf = pytest.importorskip("tensorflow", reason="TF required")

from fusionlab.nn.hybrid.xtft import XTFT  # adjust import path
from fusionlab.nn.hybrid._base_extreme import KERAS_BACKEND


@pytest.mark.skipif(not KERAS_BACKEND, reason="No Keras backend")
def test_xtft_forward_point_forecast():
    b, t_past, t_fut = 4, 5, 3
    x_static = np.random.rand(b, 2).astype("float32")
    x_dyn = np.random.rand(b, t_past, 3).astype("float32")
    x_fut = np.random.rand(b, t_fut, 1).astype("float32")

    model = XTFT(
        static_input_dim=2,
        dynamic_input_dim=3,
        future_input_dim=1,
        forecast_horizon=t_fut,
        quantiles=None,        # point forecast
    )
    # build once
    _ = model([x_static, x_dyn, x_fut], training=False)

    # compile fallback -> mse
    model.compile(optimizer="adam")
    y = np.random.rand(b, t_fut, 1).astype("float32")
    hist = model.fit(
        [x_static, x_dyn, x_fut], y,
        epochs=1, batch_size=2, verbose=0
    )
    assert hist.history, "Training history empty"

    y_pred = model.predict([x_static, x_dyn, x_fut], verbose=0)
    assert y_pred.shape == (b, t_fut, 1)


# @pytest.mark.skipif(not KERAS_BACKEND, reason="No Keras backend")
# def test_xtft_forward_quantiles():
#     b, t_past, t_fut = 2, 4, 2
#     q = [0.1, 0.5, 0.9]
#     x_static = np.random.rand(b, 1).astype("float32")
#     x_dyn = np.random.rand(b, t_past, 2).astype("float32")
#     x_fut = np.random.rand(b, t_fut, 1).astype("float32")

#     model = XTFT(
#         static_input_dim=1,
#         dynamic_input_dim=2,
#         future_input_dim=1,
#         forecast_horizon=t_fut,
#         quantiles=q,
#     )
#     _ = model([x_static, x_dyn, x_fut], training=False)
#     model.compile(optimizer="adam")  # quantile loss auto
#     y = np.random.rand(b, t_fut, 1).astype("float32")
#     model.fit([x_static, x_dyn, x_fut], y,
#               epochs=1, batch_size=1, verbose=0)
#     y_pred = model.predict([x_static, x_dyn, x_fut], verbose=0)
#     # squeeze -> (B, H, Q) because output_dim==1
#     assert y_pred.shape == (b, t_fut, len(q))


# @pytest.mark.skipif(not KERAS_BACKEND, reason="No Keras backend")
# def test_xtft_feature_based_anomaly_scores():
#     b, t_past, t_fut = 3, 6, 2
#     x_static = np.random.rand(b, 2).astype("float32")
#     x_dyn = np.random.rand(b, t_past, 3).astype("float32")
#     x_fut = np.random.rand(b, t_fut, 1).astype("float32")

#     model = XTFT(
#         static_input_dim=2,
#         dynamic_input_dim=3,
#         future_input_dim=1,
#         forecast_horizon=t_fut,
#         quantiles=[0.5],
#         anomaly_detection_strategy="feature_based",
#         anomaly_loss_weight=0.5,
#     )
#     _ = model([x_static, x_dyn, x_fut], training=True)
#     # anomaly_scores should be set during call
#     assert model.anomaly_scores is not None
#     # compile will use quantile loss (feature_based handled by add_loss)
#     model.compile(optimizer="adam")
#     y = np.random.rand(b, t_fut, 1).astype("float32")
#     model.fit([x_static, x_dyn, x_fut], y,
#               epochs=1, batch_size=1, verbose=0)
#     y_pred = model.predict([x_static, x_dyn, x_fut], verbose=0)
#     assert y_pred.shape[0] == b


# @pytest.mark.skipif(not KERAS_BACKEND, reason="No Keras backend")
# def test_xtft_from_config_anomaly_scores():
#     b, t_past, t_fut = 2, 5, 3
#     x_static = np.random.rand(b, 1).astype("float32")
#     x_dyn = np.random.rand(b, t_past, 2).astype("float32")
#     x_fut = np.random.rand(b, t_fut, 1).astype("float32")

#     scores = np.random.rand(b, t_fut, 1).astype("float32")
#     model = XTFT(
#         static_input_dim=1,
#         dynamic_input_dim=2,
#         future_input_dim=1,
#         forecast_horizon=t_fut,
#         quantiles=[0.1, 0.9],
#         anomaly_detection_strategy="from_config",
#         anomaly_config={"anomaly_scores": scores},
#     )
#     _ = model([x_static, x_dyn, x_fut], training=False)
#     assert model.anomaly_scores is not None

#     # combined_total_loss should be selected
#     model.compile(optimizer="adam")
#     y = np.random.rand(b, t_fut, 1).astype("float32")
#     model.fit([x_static, x_dyn, x_fut], y,
#               epochs=1, batch_size=1, verbose=0)


# @pytest.mark.skipif(not KERAS_BACKEND, reason="No Keras backend")
# def test_xtft_prediction_based_train_step():
#     b, t_past, t_fut = 2, 4, 2
#     x_static = np.random.rand(b, 1).astype("float32")
#     x_dyn = np.random.rand(b, t_past, 2).astype("float32")
#     x_fut = np.random.rand(b, t_fut, 1).astype("float32")
#     y = np.random.rand(b, t_fut, 1).astype("float32")

#     model = XTFT(
#         static_input_dim=1,
#         dynamic_input_dim=2,
#         future_input_dim=1,
#         forecast_horizon=t_fut,
#         quantiles=[0.5],
#         anomaly_detection_strategy="prediction_based",
#         anomaly_loss_weight=0.3,
#     )
#     _ = model([x_static, x_dyn, x_fut], training=True)
#     model.compile(optimizer="adam")  # custom pred_based loss
#     model.fit([x_static, x_dyn, x_fut], y,
#               epochs=1, batch_size=1, verbose=0)

if __name__=='__main__': 
    pytest.main([__file__])
