# tests/test_format_and_forecast.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from fusionlab.utils.forecast_utils import format_and_forecast


class DummyScaler:
    """Minimal sklearn-like scaler used to test inverse_transform."""
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)
        self.n_features_in_ = int(self.mean_.shape[0])

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        return X * self.scale_ + self.mean_


def _make_scaled_targets_point(physical, mean=100.0, scale=10.0):
    """Return (scaled, scaler_info) for a single-column target.

    physical: (B, H, O)
    """
    scaled = (physical - mean) / scale
    scaler = DummyScaler([mean], [scale])
    scaler_info = {"subsidence_cum": {"scaler": scaler, "columns": ["subsidence_cum"]}}
    return scaled, scaler_info


def _make_scaled_targets(physical, mean=100.0, scale=10.0):
    """Return (scaled, scaler_info) for a single-column target.

    Works for predictions too (e.g. physical can be (B,H,Q,O)).
    """
    scaled = (physical - mean) / scale
    scaler = DummyScaler([mean], [scale])
    scaler_info = {
        "subsidence_cum": {"scaler": scaler, "columns": ["subsidence_cum"]}
    }
    return scaled, scaler_info


def test_auto_strip_cum_suffix_and_inverse_transform_quantiles():
    """
    If target_name ends with '_cum' and output_target_name is None,
    output columns must use the stripped prefix (e.g. 'subsidence_*'),
    while inverse scaling still uses the scaler key ('subsidence_cum').
    """
    B, H, Q, O = 2, 5, 3, 1
    quantiles = [0.1, 0.5, 0.9]

    # predictions: (B, H, Q, O)
    phys_pred = (np.arange(B * H * Q * O).reshape(B, H, Q, O).astype(float) + 1.0)

    # ground truth MUST be (B, H, O) (point target, not quantiles)
    phys_true_point = phys_pred[:, :, 0, :] + 5.0  # use q10 slice just for determinism

    scaled_pred, scaler_info = _make_scaled_targets(phys_pred)
    scaled_true_point, _ = _make_scaled_targets_point(phys_true_point)

    df_eval, df_future = format_and_forecast(
        y_pred={"subs_pred": scaled_pred},
        y_true={"subsidence_cum": scaled_true_point},
        quantiles=quantiles,
        target_name="subsidence_cum",
        output_target_name=None,            # auto-strip expected
        scaler_target_name=None,            # uses target_name -> "subsidence_cum"
        scaler_info=scaler_info,
        train_end_time=2022,
        forecast_start_time=2023,
        eval_forecast_step=3,
        eval_export="last",
        eval_metrics=False,
    )

    # --- column naming: must be stripped in outputs ---
    assert "subsidence_q10" in df_eval.columns
    assert "subsidence_q50" in df_eval.columns
    assert "subsidence_q90" in df_eval.columns
    assert "subsidence_actual" in df_eval.columns
    assert not any(c.startswith("subsidence_cum_") for c in df_eval.columns)

    # --- inverse scaling correctness (eval step = 3 -> index 2) ---
    s_idx = 2
    # np.testing.assert_allclose(df_eval["subsidence_q10"].to_numpy(), phys_pred[:, s_idx, 0, 0])
    # np.testing.assert_allclose(df_eval["subsidence_q50"].to_numpy(), phys_pred[:, s_idx, 1, 0])
    # np.testing.assert_allclose(df_eval["subsidence_q90"].to_numpy(), phys_pred[:, s_idx, 2, 0])
    # np.testing.assert_allclose(df_eval["subsidence_actual"].to_numpy(), phys_true_point[:, s_idx, 0])
    np.testing.assert_allclose(
        df_eval["subsidence_q10"].to_numpy(), phys_pred[:, s_idx, 0, 0],
        rtol=1e-6, atol=1e-5
    )
    np.testing.assert_allclose(
        df_eval["subsidence_q50"].to_numpy(), phys_pred[:, s_idx, 1, 0],
        rtol=1e-6, atol=1e-5
    )
    np.testing.assert_allclose(
        df_eval["subsidence_q90"].to_numpy(), phys_pred[:, s_idx, 2, 0],
        rtol=1e-6, atol=1e-5
    )
    np.testing.assert_allclose(
        df_eval["subsidence_actual"].to_numpy(), phys_true_point[:, s_idx, 0],
        rtol=1e-6, atol=1e-5
    )
    # --- future df has no actuals and uses stripped prefix too ---
    assert "subsidence_actual" not in df_future.columns
    assert "subsidence_q10" in df_future.columns
    assert not any(c.startswith("subsidence_cum_") for c in df_future.columns)


def test_preserve_output_target_name_when_provided():
    """
    If output_target_name is explicitly set, keep it verbatim
    (no auto-strip).
    """
    B, H, Q, O = 2, 5, 3, 1
    quantiles = [0.1, 0.5, 0.9]

    phys_pred = (np.arange(B * H * Q * O).reshape(B, H, Q, O).astype(float) + 1.0)
    phys_true_point = phys_pred[:, :, 0, :] + 2.0

    scaled_pred, scaler_info = _make_scaled_targets(phys_pred)
    scaled_true_point, _ = _make_scaled_targets_point(phys_true_point)

    df_eval, _ = format_and_forecast(
        y_pred={"subs_pred": scaled_pred},
        y_true={"subsidence_cum": scaled_true_point},
        quantiles=quantiles,
        target_name="subsidence_cum",
        output_target_name="subsidence_cum",   # force keeping suffix
        scaler_target_name="subsidence_cum",
        scaler_info=scaler_info,
        train_end_time=2022,
        forecast_start_time=2023,
        eval_forecast_step=5,
        eval_export="last",
        eval_metrics=False,
    )

    assert "subsidence_cum_q10" in df_eval.columns
    assert "subsidence_cum_q50" in df_eval.columns
    assert "subsidence_cum_q90" in df_eval.columns
    assert "subsidence_cum_actual" in df_eval.columns
    assert "subsidence_q10" not in df_eval.columns


def test_eval_time_value_matches_eval_forecast_step_not_always_train_end():
    """
    Regression test:
    eval time must be eval_time_grid_full[s_idx], not always train_end_time.
    """
    B, H, Q, O = 2, 5, 3, 1
    quantiles = [0.1, 0.5, 0.9]
    eval_step = 3  # s_idx=2 -> expected time = 2020 if train_end_time=2022 and H=5

    phys_pred = np.ones((B, H, Q, O), dtype=float) * 10.0
    phys_true_point = np.ones((B, H, O), dtype=float) * 12.0

    scaled_pred, scaler_info = _make_scaled_targets(phys_pred)
    scaled_true_point, _ = _make_scaled_targets_point(phys_true_point)

    df_eval, _ = format_and_forecast(
        y_pred={"subs_pred": scaled_pred},
        y_true={"subsidence_cum": scaled_true_point},
        quantiles=quantiles,
        target_name="subsidence_cum",
        output_target_name=None,
        scaler_target_name=None,
        scaler_info=scaler_info,
        train_end_time=2022,
        forecast_start_time=2023,
        eval_forecast_step=eval_step,
        eval_export="last",
        eval_metrics=False,
    )

    # train_end_time=2022, H=5 => full grid: [2018, 2019, 2020, 2021, 2022]
    expected_time = 2020.0
    got = pd.to_numeric(df_eval["coord_t"], errors="coerce").unique()
    assert got.size == 1
    assert float(got[0]) == expected_time


def test_eval_export_all_returns_multi_horizon_eval_df():
    """
    When eval_export='all' and y_true is provided, returned df_eval should be
    the multi-horizon evaluation DF (B*H rows) with correct time grid tiling.
    """
    B, H, Q, O = 2, 5, 3, 1
    quantiles = [0.1, 0.5, 0.9]

    phys_pred = (np.arange(B * H * Q * O).reshape(B, H, Q, O).astype(float) + 1.0)
    phys_true_point = phys_pred[:, :, 0, :] + 1.0

    scaled_pred, scaler_info = _make_scaled_targets(phys_pred)
    scaled_true_point, _ = _make_scaled_targets_point(phys_true_point)

    df_eval_all, _ = format_and_forecast(
        y_pred={"subs_pred": scaled_pred},
        y_true={"subsidence_cum": scaled_true_point},
        quantiles=quantiles,
        target_name="subsidence_cum",
        output_target_name=None,
        scaler_target_name=None,
        scaler_info=scaler_info,
        train_end_time=2022,
        forecast_start_time=2023,
        eval_forecast_step=5,
        eval_export="all",
        eval_metrics=False,
    )

    assert len(df_eval_all) == B * H
    assert set(df_eval_all["forecast_step"].unique()) == set(range(1, H + 1))

    expected = np.array([2018, 2019, 2020, 2021, 2022], dtype=float)
    for _, g in df_eval_all.groupby("sample_idx", sort=False):
        got = pd.to_numeric(g.sort_values("forecast_step")["coord_t"], errors="coerce").to_numpy()
        np.testing.assert_allclose(got, expected)
