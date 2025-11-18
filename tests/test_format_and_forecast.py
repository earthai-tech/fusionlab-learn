import numpy as np
import pandas as pd
import pytest

from fusionlab.nn.pinn import utils as pinn_utils


def _make_stub_full_df_years() -> pd.DataFrame:
    # Two samples, horizon H=3
    # coord_t are "future" years
    data = {
        "sample_idx": [0, 0, 0, 1, 1, 1],
        "forecast_step": [1, 2, 3, 1, 2, 3],
        "subsidence_q10": [1.0, 2.0, 3.0, 4.0,
                           5.0, 6.0],
        "subsidence_q50": [1.5, 2.5, 3.5, 4.5,
                           5.5, 6.5],
        "subsidence_q90": [2.0, 3.0, 4.0, 5.0,
                           6.0, 7.0],
        "subsidence_actual": [10.0, 11.0, 12.0,
                              13.0, 14.0, 15.0],
        "coord_t": [2023.0, 2024.0, 2025.0,
                    2023.0, 2024.0, 2025.0],
        "coord_x": [113.0] * 6,
        "coord_y": [22.0] * 6,
    }
    return pd.DataFrame(data)


def _make_stub_full_df_mixed_years() -> pd.DataFrame:
    # Two samples, horizon H=3
    # Mixed years for eval vs future
    data = {
        "sample_idx": [0, 0, 0, 1, 1, 1],
        "forecast_step": [1, 2, 3, 1, 2, 3],
        "subsidence_q10": [1.0, 2.0, 3.0, 4.0,
                           5.0, 6.0],
        "subsidence_q50": [1.5, 2.5, 3.5, 4.5,
                           5.5, 6.5],
        "subsidence_q90": [2.0, 3.0, 4.0, 5.0,
                           6.0, 7.0],
        "subsidence_actual": [10.0, 11.0, 12.0,
                              13.0, 14.0, 15.0],
        # years: first step < split, others >= split
        "coord_t": [2019.0, 2020.0, 2021.0,
                    2019.0, 2020.0, 2021.0],
        "coord_x": [113.0] * 6,
        "coord_y": [22.0] * 6,
    }
    return pd.DataFrame(data)


def _make_stub_full_df_datetimes() -> pd.DataFrame:
    # Two samples, horizon H=3, datetime coord_t
    dates = pd.date_range(
        "2018-01-01", periods=3, freq="YS"
    )
    coord_t = list(dates) + list(dates)
    data = {
        "sample_idx": [0, 0, 0, 1, 1, 1],
        "forecast_step": [1, 2, 3, 1, 2, 3],
        "subsidence_q10": [1.0, 2.0, 3.0, 4.0,
                           5.0, 6.0],
        "subsidence_q50": [1.5, 2.5, 3.5, 4.5,
                           5.5, 6.5],
        "subsidence_q90": [2.0, 3.0, 4.0, 5.0,
                           6.0, 7.0],
        "subsidence_actual": [10.0, 11.0, 12.0,
                              13.0, 14.0, 15.0],
        "coord_t": coord_t,
        "coord_x": [113.0] * 6,
        "coord_y": [22.0] * 6,
    }
    return pd.DataFrame(data)


@pytest.fixture
def monkeypatch_formatter(monkeypatch):
    # Helper to monkeypatch format_pihalnet_predictions
    # in this module
    def _patch_with_df(df: pd.DataFrame):
        def _fake_formatter(*args, **kwargs):
            return df.copy()

        monkeypatch.setattr(
            pinn_utils,
            "format_pihalnet_predictions",
            _fake_formatter,
        )

    return _patch_with_df


def test_format_and_forecast_future_only_overrides_time_and_drops_actuals(  # noqa: E501
    monkeypatch_formatter, tmp_path
):
    # All rows are future according to split, so eval is
    # empty and future gets its coord_t overridden and
    # actuals removed.
    stub_df = _make_stub_full_df_years()
    monkeypatch_formatter(stub_df)

    eval_path = tmp_path / "eval.csv"
    future_path = tmp_path / "future.csv"

    future_grid = np.array(
        [2030.0, 2031.0, 2032.0], dtype=float
    )

    df_eval, df_future = pinn_utils.format_and_forecast(
        predictions={"subs_pred": None},
        model=None,
        model_inputs=None,
        y_true_dict=None,
        target_mapping={
            "subs_pred": "subsidence"
        },
        include_gwl=False,
        include_coords=True,
        quantiles=[0.1, 0.5, 0.9],
        forecast_horizon=3,
        output_dims=None,
        scaler_info=None,
        coord_scaler=None,
        evaluate_coverage=False,
        savefile_eval=str(eval_path),
        savefile_forecast=str(future_path),
        time_split_value=2020.0,
        time_col="coord_t",
        primary_target="subsidence",
        future_time_grid=future_grid,
        verbose=0,
    )

    # All rows go to future, eval is empty
    assert df_eval.empty
    assert not df_future.empty
    assert len(df_future) == len(stub_df)

    # subsidence_actual should be removed from future
    assert "subsidence_actual" not in df_future

    # coord_t must be overridden with future_grid per
    # sample and horizon
    n_samples = df_future["sample_idx"].nunique()
    expected_t = np.tile(future_grid, n_samples)
    assert np.allclose(
        df_future["coord_t"].to_numpy(),
        expected_t,
    )

    # Only future CSV should exist (eval is empty)
    assert not eval_path.exists()
    assert future_path.exists()

def test_format_and_forecast_splits_eval_and_future_years(  # noqa: E501
    monkeypatch_formatter, tmp_path
):
    # Mixed years -> some rows eval, some future.
    stub_df = _make_stub_full_df_mixed_years()
    monkeypatch_formatter(stub_df)

    eval_path = tmp_path / "eval.csv"
    future_path = tmp_path / "future.csv"

    df_eval, df_future = pinn_utils.format_and_forecast(
        predictions={"subs_pred": None},
        model=None,
        model_inputs=None,
        y_true_dict=None,
        target_mapping={
            "subs_pred": "subsidence"
        },
        include_gwl=False,
        include_coords=True,
        quantiles=[0.1, 0.5, 0.9],
        forecast_horizon=3,
        output_dims=None,
        scaler_info=None,
        coord_scaler=None,
        evaluate_coverage=False,
        savefile_eval=str(eval_path),
        savefile_forecast=str(future_path),
        time_split_value=2020.0,
        time_col="coord_t",
        primary_target="subsidence",
        future_time_grid=None,
        verbose=0,
    )

    # For each sample: coord_t 2019 (< split) is eval,
    # 2020, 2021 are future.
    assert not df_eval.empty
    assert not df_future.empty

    # There are 2 samples, 1 eval row per sample
    assert len(df_eval) == 2
    assert set(df_eval["coord_t"]) == {2019.0}

    # Future is the remaining 4 rows
    assert len(df_future) == 4
    assert set(df_future["coord_t"]) == {
        2020.0,
        2021.0,
    }

    # Eval keeps actuals, future drops them
    assert "subsidence_actual" in df_eval
    assert "subsidence_actual" not in df_future

    assert eval_path.exists()
    assert future_path.exists()


def test_format_and_forecast_handles_datetime_time_axis(  # noqa: E501
    monkeypatch_formatter
):
    # coord_t is datetime64, split and override also
    # in datetime.
    stub_df = _make_stub_full_df_datetimes()
    monkeypatch_formatter(stub_df)

    dates_future = pd.date_range(
        "2020-01-01", periods=3, freq="YS"
    )

    df_eval, df_future = pinn_utils.format_and_forecast(
        predictions={"subs_pred": None},
        model=None,
        model_inputs=None,
        y_true_dict=None,
        target_mapping={
            "subs_pred": "subsidence"
        },
        include_gwl=False,
        include_coords=True,
        quantiles=[0.1, 0.5, 0.9],
        forecast_horizon=3,
        output_dims=None,
        scaler_info=None,
        coord_scaler=None,
        evaluate_coverage=False,
        savefile_eval=None,
        savefile_forecast=None,
        time_split_value="2019-01-01",
        time_col="coord_t",
        primary_target="subsidence",
        future_time_grid=dates_future,
        verbose=0,
    )

    # Split: coord_t < 2019-01-01 -> only 2018 rows
    # (one per sample)
    assert not df_eval.empty
    assert len(df_eval) == 2

    # Future rows are the remaining 4 rows
    assert not df_future.empty
    assert len(df_future) == 4

    # coord_t dtype should stay datetime64
    assert np.issubdtype(
        df_future["coord_t"].dtype,
        np.datetime64,
    )

    # Future coord_t must be overridden based on
    # forecast_step
    steps = (
        df_future["forecast_step"]
        .astype(int)
        .to_numpy()
    )
    expected = dates_future.values[steps - 1]
    np.testing.assert_array_equal(
        df_future["coord_t"].to_numpy(),
        expected,
    )
