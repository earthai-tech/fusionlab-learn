# -*- coding: utf-8 -*-
# test_prepare_pinn_data_sequences.py
import numpy as np
import pandas as pd
import pytest

from fusionlab.nn.pinn.utils import prepare_pinn_data_sequences


# ------------------------------------------------------------------ #
# Helper to build a minimal toy DataFrame
# ------------------------------------------------------------------ #
def _make_toy_df(n_dates: int = 20, n_sites: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

    df = pd.DataFrame({
        "date": dates.tolist() * n_sites,
        "site": np.repeat(list("AB")[:n_sites], n_dates),
        "lon":  rng.uniform(110, 111, n_dates * n_sites),
        "lat":  rng.uniform(22,  23, n_dates * n_sites),
        "subs": rng.normal(size=n_dates * n_sites),
        "gwl":  rng.normal(size=n_dates * n_sites),
        "rain": rng.random(n_dates * n_sites),
        "evap": rng.random(n_dates * n_sites),
        "forecast_rain": rng.random(n_dates * n_sites),
        "soil_type": np.repeat(["sand", "clay"], n_dates)[:n_dates*n_sites],
    })
    return df


# ------------------------------------------------------------------ #
# 1) Parametrised test for both preparation modes
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("mode", ["pihal_like", "tft_like"])
def test_future_window_length(mode):
    """future_features must have correct temporal span for each mode."""
    time_steps, horizon = 5, 3
    df = _make_toy_df()

    # One‑hot encode soil_type so static features are numeric
    df_enc = pd.get_dummies(df, columns=["soil_type"])

    inputs, targets = prepare_pinn_data_sequences(
        df=df_enc,
        time_col="date",
        subsidence_col="subs",
        gwl_col="gwl",
        dynamic_cols=["rain", "evap"],
        static_cols=[c for c in df_enc.columns if c.startswith("soil_type_")],
        future_cols=["forecast_rain"],
        spatial_cols=("lon", "lat"),
        group_id_cols=["site"],
        time_steps=time_steps,
        forecast_horizon=horizon,
        mode=mode,
        verbose=0,
    )

    future_len = inputs["future_features"].shape[1]
    expected_len = horizon if mode == "pihal_like" else time_steps + horizon
    assert future_len == expected_len
    # quick shape sanity
    assert inputs["dynamic_features"].shape[1] == time_steps
    assert targets["subsidence"].shape[1] == horizon


# ------------------------------------------------------------------ #
# 2) Test that coordinate scaler is returned and ranges in [0, 1]
# ------------------------------------------------------------------ #
def test_coord_scaler_return():
    df = _make_toy_df()
    df_enc = pd.get_dummies(df, columns=["soil_type"])

    inputs, _, scaler = prepare_pinn_data_sequences(
        df=df_enc,
        time_col="date",
        subsidence_col="subs",
        gwl_col="gwl",
        dynamic_cols=["rain", "evap"],
        static_cols=[c for c in df_enc.columns if c.startswith("soil_type_")],
        future_cols=None,
        spatial_cols=("lon", "lat"),
        group_id_cols=["site"],
        time_steps=4,
        forecast_horizon=2,
        mode="pihal_like",
        normalize_coords=True,
        return_coord_scaler=True,
        verbose=0,
    )

    # scaler must be MinMax‑like: transformed coords in [0,1]
    coords = inputs["coords"].reshape(-1, 3)
    assert np.all(coords >= 0.0) and np.all(coords <= 1.0)
    assert scaler is not None


# ------------------------------------------------------------------ #
# 3) Test behaviour with normalize_coords=False
# ------------------------------------------------------------------ #
def test_no_coord_normalisation():
    df = _make_toy_df()
    df_enc = pd.get_dummies(df, columns=["soil_type"])

    inputs, _ = prepare_pinn_data_sequences(
        df=df_enc,
        time_col="date",
        subsidence_col="subs",
        gwl_col="gwl",
        dynamic_cols=["rain", "evap"],
        static_cols=[c for c in df_enc.columns if c.startswith("soil_type_")],
        future_cols=None,
        spatial_cols=("lon", "lat"),
        group_id_cols=["site"],
        time_steps=4,
        forecast_horizon=2,
        mode="pihal_like",
        normalize_coords=False,
        verbose=0,
    )

    coords = inputs["coords"].reshape(-1, 3)
    # lon / lat are around 110–111 / 22–23, so max > 1 when not scaled
    assert coords[:, 1].max() > 1.0
    assert coords[:, 2].max() > 1.0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
