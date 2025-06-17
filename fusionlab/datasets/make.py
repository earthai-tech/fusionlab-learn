# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause 
# -------------------------------------------------------------------
# Provides functions for generating synthetic datasets suitable for
# demonstrating and testing fusionlab models.
# -------------------------------------------------------------------
"""
Synthetic Dataset Generation Utilities (:mod:`fusionlab.datasets.make`)
======================================================================

This module provides functions to create synthetic datasets tailored
for demonstrating and testing the various models and utilities within
the `fusionlab` package, particularly those expecting static, dynamic,
and future features (like TFT and XTFT).
"""
from __future__ import annotations

import textwrap
import warnings
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple

from ..api.bunch import XBunch


__all__ = [
    "make_multi_feature_time_series", 
    "make_quantile_prediction_data",
    "make_anomaly_data", 
    "make_trend_seasonal_data",
    "make_multivariate_target_data"
    ]

def make_multi_feature_time_series(
    n_series: int = 3,
    n_timesteps: int = 100,
    freq: str = 'D', 
    static_noise_level: float = 0.1,
    trend_base: float = 10,
    trend_factor: float = 0.1,
    seasonality_period: float = 7, 
    seasonality_amplitude: float = 5,
    dynamic_cov_amplitude: float = 2,
    future_cov_amplitude: float = 1,
    noise_level: float = 1.0,
    as_frame: bool = False,
    seed: Optional[int] = None,
) -> Union[XBunch, pd.DataFrame]:
    r"""Generate multi-variate time series with static, dynamic, and future features.

    Creates a synthetic dataset suitable for models like TFT/XTFT. It
    simulates data for multiple independent series (e.g., items, locations)
    over a specified number of time steps.

    Each series includes:
    - Static features (unique ID, a noisy base value).
    - Dynamic features (time index features like month/dayofweek,
      a simulated covariate like temperature, lagged target).
    - Known future features (time index features, a simulated binary
      event like promotion).
    - A target variable generated from trend, seasonality, covariates,
      static base, and noise.

    Parameters
    ----------
    n_series : int, default=3
        Number of independent time series (e.g., items, sensors)
        to generate.
    n_timesteps : int, default=100
        Number of time steps (rows) per series.
    freq : str, default='D'
        Pandas frequency string for generating the datetime index
        (e.g., 'D' for daily, 'MS' for month start, 'H' for hourly).
    static_noise_level : float, default=0.1
        Amount of noise added to the static 'base_level' feature.
    trend_base : float, default=10
        Base value for the linear trend component.
    trend_factor : float, default=0.1
        Slope factor for the linear trend component.
    seasonality_period : float, default=7
        Periodicity for the main seasonal component (e.g., 7 for weekly
        pattern with daily data, 12 for yearly pattern with monthly data).
    seasonality_amplitude : float, default=5
        Amplitude of the main seasonal sinusoidal component.
    dynamic_cov_amplitude : float, default=2
        Amplitude of the simulated dynamic covariate (e.g., temperature).
    future_cov_amplitude : float, default=1
        Magnitude of the effect of the simulated future binary event.
    noise_level : float, default=1.0
        Standard deviation of the Gaussian noise added to the final
        target signal.
    as_frame : bool, default=False
        Determines the return type:
        - If ``False`` (default): Returns a Bunch object containing the
          DataFrame and metadata (column names grouped by type).
        - If ``True``: Returns only the pandas DataFrame.
    seed : int, optional
        Seed for NumPy's random number generator for reproducibility.
        Default is None.

    Returns
    -------
    data : :class:`~fusionlab.api.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        A Bunch object with attributes like ``frame`` (DataFrame),
        ``static_features`` (list of col names), ``dynamic_features``,
        ``future_features``, ``target_col``, ``dt_col``, ``spatial_id_col``,
        and ``DESCR``.
        If ``as_frame=True``:
        The generated data solely as a pandas DataFrame.

    Examples
    --------
    >>> from fusionlab.datasets.make import make_multi_feature_time_series
    >>> # Generate daily data for 5 series
    >>> data_bunch = make_multi_feature_time_series(n_series=5, n_timesteps=100,
    ...                                           freq='D', seasonality_period=7,
    ...                                           seed=42)
    >>> print(data_bunch.frame.head())
    >>> print("Static Features:", data_bunch.static_features)
    >>> print("Dynamic Features:", data_bunch.dynamic_features)
    >>> print("Future Features:", data_bunch.future_features)

    >>> # Generate monthly data as DataFrame
    >>> df_monthly = make_multi_feature_time_series(n_series=2, n_timesteps=36,
    ...                                           freq='MS', seasonality_period=12,
    ...                                           as_frame=True, seed=123)
    >>> print(df_monthly.info())
    """
    # if seed is not None:
    rng = np.random.RandomState(seed)


    all_series_df = []
    start_date = '2020-01-01' # Arbitrary start date

    for i in range(n_series):
        # --- Time Index ---
        date_rng = pd.date_range(
            start=start_date, periods=n_timesteps, freq=freq)
        time_idx = np.arange(n_timesteps)

        # --- Static Features ---
        series_id = i
        # Each series gets a slightly different noisy base level
        base_level = 50 + i * 20 + rng.normal(0, static_noise_level)

        # --- Dynamic Features ---
        month = date_rng.month
        dayofweek = date_rng.dayofweek
        # Simulated dynamic covariate (e.g., temperature-like)
        dynamic_cov = dynamic_cov_amplitude * np.sin(
            2 * np.pi * time_idx / (seasonality_period * 2) + i * np.pi / 3 # Phase shift per series
            ) + rng.normal(0, noise_level * 0.5, n_timesteps)

        # --- Future Features ---
        # Known future event (e.g., promotion flag)
        future_event = rng.randint(0, 2, n_timesteps)
        # Time features known in advance
        future_month = month
        future_dayofweek = dayofweek

        # --- Target Variable ---
        trend = trend_base + trend_factor * time_idx * (1 + i * 0.1)
        seasonality = seasonality_amplitude * np.sin(
            2 * np.pi * time_idx / seasonality_period + i * np.pi / 4 # Phase shift
            )
        event_effect = future_event * future_cov_amplitude * (5 + i) # Event impact
        noise = rng.normal(0, noise_level, n_timesteps)

        target = base_level + trend + seasonality + event_effect + \
                 0.5 * dynamic_cov + noise # Combine components

        # --- Lagged Target (as Dynamic Input) ---
        # Create after calculating target
        lagged_target = pd.Series(target).shift(1).fillna(method='bfill') # Backfill first NaN

        # --- Assemble DataFrame for this series ---
        series_df = pd.DataFrame({
            'date': date_rng,
            'series_id': series_id, # Static identifier
            'base_level': base_level, # Static numerical
            'month': future_month, # Dynamic and Future: month
            'dayofweek': future_dayofweek, # Dynamic and Future:dayofweek
            'dynamic_cov': dynamic_cov, # Dynamic only
            'target_lag1': lagged_target, # Dynamic only
            'future_event': future_event, # Future only
            'target': target # Target variable
        })
        all_series_df.append(series_df)

    # --- Combine all series ---
    df = pd.concat(all_series_df).reset_index(drop=True)

    # --- Define Column Roles ---
    dt_col = 'date'
    target_col = 'target'
    spatial_id_col = 'series_id'
    static_features = ['series_id', 'base_level']
    dynamic_features = ['month', 'dayofweek', 'dynamic_cov', 'target_lag1']
    future_features = ['month', 'dayofweek', 'future_event']
    # Exclude target and ID from features list passed to Bunch
    dynamic_and_future_features = list(set (dynamic_features + future_features))
    feature_names = static_features[1:] + dynamic_and_future_features 
    # dynamic_features + future_features

    # --- Return based on as_frame ---
    if as_frame:
        # Return DataFrame with logical column order
        ordered_cols = (
            [dt_col, spatial_id_col] + static_features[1:] +
            dynamic_and_future_features + [target_col]
            )
        # Ensure columns exist before ordering
        ordered_cols = [c for c in ordered_cols if c in df.columns]
        return df[ordered_cols]
    else:
        # Create Bunch object
        descr = textwrap.dedent(f"""\
        Synthetic Multi-Feature Time Series Data

        **Description:**
        Simulates data for {n_series} independent series over {n_timesteps}
        time steps with frequency '{freq}'. Includes static, dynamic, and
        known future features suitable for TFT/XTFT models.

        **Generation Parameters:** (Approximate)
        - n_series: {n_series}
        - n_timesteps: {n_timesteps}
        - freq: '{freq}'
        - seasonality_period: {seasonality_period}
        - noise_level: {noise_level:.2f}
        - trend/seasonality/covariates included.

        **Data Structure (Bunch object):**
        - frame            : Complete pandas DataFrame.
        - static_features  : List of static column names.
        - dynamic_features : List of dynamic column names.
        - future_features  : List of future column names.
        - target_col       : Name of the target column ('{target_col}').
        - dt_col           : Name of the datetime column ('{dt_col}').
        - spatial_id_col   : Name of the series identifier column ('{spatial_id_col}').
        - feature_names    : Combined list of static (excl. ID), dynamic, future features.
        - DESCR            : This description.
        """)

        # Order frame columns for Bunch frame attribute
        frame_cols = (
             [dt_col, spatial_id_col] + static_features[1:] +
             dynamic_and_future_features + [target_col]
             )
        frame_cols = [c for c in frame_cols if c in df.columns]

        return XBunch(
            frame=df[frame_cols],
            static_features=static_features,
            dynamic_features=dynamic_features,
            future_features=future_features,
            target_col=target_col,
            dt_col=dt_col,
            spatial_id_col=spatial_id_col,
            feature_names=feature_names, # Combined list
            DESCR=descr
        )


def make_quantile_prediction_data(
    n_samples: int = 100,
    n_horizons: int = 6,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    target_mean: float = 50.0,
    target_stddev: float = 10.0,
    pred_bias: float = 1.0,
    pred_spread_factor: float = 1.5,
    add_coords: bool = True,
    coord_scale: float = 10.0,
    as_frame: bool = False,
    seed: Optional[int] = None,
) -> Union[XBunch, pd.DataFrame]:
    r"""Generate synthetic actuals and corresponding quantile predictions.

    Creates a dataset simulating the output of a multi-horizon quantile
    forecasting model. It includes actual target values and predicted
    values for specified quantiles across multiple forecast horizons for
    a set of samples (e.g., locations).

    This data is useful for demonstrating and testing functions that evaluate
    or visualize probabilistic forecasts, such as those comparing prediction
    intervals to actual outcomes.

    Parameters
    ----------
    n_samples : int, default=100
        Number of independent samples (e.g., locations) to generate.
    n_horizons : int, default=6
        Number of future time steps (forecast horizon) per sample.
    quantiles : list of float, default=[0.1, 0.5, 0.9]
        List of quantile levels (between 0 and 1) for which to generate
        predictions.
    target_mean : float, default=50.0
        Mean value around which the 'actual' target values are generated.
    target_stddev : float, default=10.0
        Standard deviation for generating the 'actual' target values
        (using a normal distribution).
    pred_bias : float, default=1.0
        Systematic bias added to the median (0.5 quantile) prediction
        relative to the generated actual value.
    pred_spread_factor : float, default=1.5
        Factor controlling the width of the prediction intervals. A higher
        value creates wider intervals between quantiles. Specifically, it
        scales the offsets added/subtracted from the biased median.
    add_coords : bool, default=True
        If ``True``, add 'longitude' and 'latitude' columns with random
        coordinates.
    coord_scale : float, default=10.0
        Scaling factor for the random coordinates if `add_coords` is True.
    as_frame : bool, default=False
        Determines the return type:
        - ``False`` (default): Returns a Bunch object.
        - ``True``: Returns only the pandas DataFrame.
    seed : int, optional
        Seed for NumPy's random number generator for reproducibility.
        Default is None.

    Returns
    -------
    data : :class:`~fusionlab.api.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        A Bunch object with attributes like ``frame`` (DataFrame),
        ``quantiles`` (list), ``horizons`` (list), ``target_cols``,
        ``prediction_cols`` (nested dict), `longitude`, `latitude`
        (if generated), and ``DESCR``.
        If ``as_frame=True``:
        The generated data solely as a pandas DataFrame in wide format
        (e.g., columns 'target_h1', 'pred_q10_h1', 'pred_q50_h1', ...).

    Examples
    --------
    >>> from fusionlab.datasets import make_quantile_prediction_data
    >>> # Generate data as Bunch
    >>> pred_bunch = make_quantile_prediction_data(n_samples=5, n_horizons=3, seed=1)
    >>> print(pred_bunch.frame.head())
    >>> print("Quantile columns for q=0.1:", pred_bunch.prediction_cols['q0.1'])

    >>> # Generate data as DataFrame
    >>> pred_df = make_quantile_prediction_data(as_frame=True, seed=2)
    >>> print(pred_df.info())
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if not quantiles or not isinstance(quantiles, list):
        raise ValueError("'quantiles' must be a non-empty list of floats.")

    # Generate base actuals and coordinates
    actuals = rng.normal(
        target_mean, target_stddev, size=(n_samples, n_horizons)
    )
    data_dict = {}
    if add_coords:
        # Simulate coordinates (e.g., centered around 0)
        longitude = rng.uniform(-coord_scale, coord_scale, n_samples)
        latitude = rng.uniform(-coord_scale/2, coord_scale/2, n_samples)
        data_dict['longitude'] = longitude
        data_dict['latitude'] = latitude

    target_cols = []
    prediction_cols = {f"q{q:.1f}".replace("0.", ""): [] for q in quantiles}
    all_pred_cols_flat = []

    # Generate predictions for each horizon step and quantile
    for h in range(n_horizons):
        step = h + 1
        # Add actual column for this step
        target_col_name = f"target_h{step}"
        data_dict[target_col_name] = actuals[:, h]
        target_cols.append(target_col_name)

        # Generate biased median prediction for this step
        median_pred = actuals[:, h] + pred_bias + rng.normal(
            0, target_stddev * 0.5, n_samples) # Add some noise to median

        # Generate other quantiles around the biased median
        for q in quantiles:
            # Calculate offset based on quantile distance from median
            # Scaled by spread factor and target stddev
            quantile_offset = (q - 0.5) * pred_spread_factor * target_stddev
            # Add noise specific to this quantile/step
            q_noise = rng.normal(0, target_stddev * 0.2, n_samples)
            pred_val = median_pred + quantile_offset + q_noise

            # Add prediction column
            q_key = f"q{q:.1f}".replace("0.", "") # e.g., q0.1 -> q1
            pred_col_name = f"pred_{q_key}_h{step}"
            data_dict[pred_col_name] = pred_val
            prediction_cols[q_key].append(pred_col_name)
            all_pred_cols_flat.append(pred_col_name)

    # Create DataFrame
    df = pd.DataFrame(data_dict)

    # Define column categories for Bunch
    feature_names = [c for c in df.columns if c in ['longitude', 'latitude']]
    target_names = target_cols

    if as_frame:
        # Order columns logically
        ordered_cols = feature_names + target_names + sorted(all_pred_cols_flat)
        return df[[c for c in ordered_cols if c in df.columns]]
    else:
        # Create Bunch description
        descr = textwrap.dedent(f"""\
        Synthetic Quantile Prediction Data

        **Description:**
        Simulates {n_samples} samples (e.g., locations) with actual
        target values and corresponding quantile predictions for
        {n_horizons} future horizons. Target values are drawn from a
        normal distribution. Predictions are generated around a biased
        median, with spread controlled by `pred_spread_factor`.

        **Generation Parameters:**
        - n_samples: {n_samples}
        - n_horizons: {n_horizons}
        - quantiles: {quantiles}
        - target_mean: {target_mean:.2f}
        - target_stddev: {target_stddev:.2f}
        - pred_bias: {pred_bias:.2f}
        - pred_spread_factor: {pred_spread_factor:.2f}
        - seed: {seed}

        **Data Structure (Bunch object):**
        - frame           : Complete pandas DataFrame in wide format.
        - quantiles       : List of quantiles generated.
        - horizons        : List of horizon steps [1, ..., {n_horizons}].
        - feature_names   : List of coordinate columns (if generated).
        - target_cols     : List of target column names ('target_hX').
        - prediction_cols : Dict mapping quantile keys ('qX') to lists
                            of corresponding prediction column names.
        - longitude       : NumPy array of longitude values (if generated).
        - latitude        : NumPy array of latitude values (if generated).
        - DESCR           : This description.
        """)

        bunch_dict = {
            "frame": df,
            "quantiles": quantiles,
            "horizons": list(range(1, n_horizons + 1)),
            "feature_names": feature_names,
            "target_cols": target_names,
            "prediction_cols": prediction_cols,
            "DESCR": descr,
        }
        if add_coords:
            if 'longitude' in df: bunch_dict['longitude'] = df['longitude'].values
            if 'latitude' in df: bunch_dict['latitude'] = df['latitude'].values

        return XBunch(**bunch_dict)


def make_anomaly_data(
    n_sequences: int = 200,
    sequence_length: int = 50,
    n_features: int = 1,
    anomaly_fraction: float = 0.1,
    anomaly_type: str = 'spike', # 'spike' or 'level_shift'
    anomaly_magnitude: float = 5.0,
    noise_level: float = 0.2,
    as_frame: bool = False, 
    seed: Optional[int] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], XBunch, pd.DataFrame]:
    r"""Generate sequence data with injected anomalies.

    Creates a dataset of time series sequences, where a specified
    fraction contains synthetically generated anomalies (spikes or
    level shifts). It returns the sequences and corresponding binary
    labels (0 for normal, 1 for anomaly).

    This data is useful for testing and evaluating anomaly detection
    algorithms like :class:`~fusionlab.nn.anomaly_detection.LSTMAutoencoderAnomaly`
    or anomaly-aware training strategies.

    Parameters
    ----------
    n_sequences : int, default=200
        Total number of sequences to generate.
    sequence_length : int, default=50
        Number of time steps in each sequence.
    n_features : int, default=1
        Number of features for each time step. Currently supports 1.
    anomaly_fraction : float, default=0.1
        Fraction of sequences that should contain anomalies (between 0 and 1).
    anomaly_type : {'spike', 'level_shift'}, default='spike'
        Type of anomaly to inject:
        - ``'spike'``: Adds/subtracts `anomaly_magnitude` at a random single point.
        - ``'level_shift'``: Adds/subtracts `anomaly_magnitude` to all points
          after a random point in the sequence.
    anomaly_magnitude : float, default=5.0
        The magnitude (absolute value) of the injected anomaly. The sign
        (add or subtract) is chosen randomly.
    noise_level : float, default=0.2
        Standard deviation of Gaussian noise added to the base signal.
    as_frame : bool, default=False
        Determines return type:
        - If ``False`` (default): Returns a tuple `(sequences, labels)`
          where `sequences` is a NumPy array `(N, T, F)` and `labels`
          is `(N,)`.
        - If ``True``: Attempts to create a DataFrame and returns a Bunch
          object (less standard for sequence data).
    seed : int, optional
        Seed for NumPy's random number generator for reproducibility.
        Default is None.

    Returns
    -------
    data : tuple or :class:`~fusionlab.api.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        Tuple `(sequences, labels)`:
            - sequences : ndarray of shape (n_sequences, sequence_length, n_features)
            - labels : ndarray of shape (n_sequences,) with 0 (normal) or 1 (anomaly).
        If ``as_frame=True``:
        A Bunch object containing a DataFrame (`frame` - potentially very wide
        if sequences flattened), `labels`, `feature_names`, etc. Or just the
        DataFrame if preferred (structure TBD). *Note: Returning sequences
        as a DataFrame can be awkward.*

    Raises
    ------
    ValueError
        If `n_features` is not 1 (currently only supports univariate).
        If `anomaly_fraction` is not between 0 and 1.
        If `anomaly_type` is invalid.

    Examples
    --------
    >>> from fusionlab.datasets import make_anomaly_data
    >>> # Generate sequences and labels as NumPy arrays
    >>> sequences, labels = make_anomaly_data(n_sequences=50, anomaly_fraction=0.2, seed=42)
    >>> print(f"Generated sequences shape: {sequences.shape}")
    >>> print(f"Generated labels shape: {labels.shape}")
    >>> print(f"Number of anomalies: {np.sum(labels)}")

    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if n_features != 1:
        # TODO: Extend to multivariate sequences if needed
        raise ValueError("Currently only supports n_features=1")
    if not 0 <= anomaly_fraction <= 1:
        raise ValueError("'anomaly_fraction' must be between 0 and 1.")
    if anomaly_type not in ['spike', 'level_shift']:
        raise ValueError("anomaly_type must be 'spike' or 'level_shift'")

    n_anomalies = int(n_sequences * anomaly_fraction)
    n_normal = n_sequences - n_anomalies

    sequences = []
    labels = []

    # Generate Normal Sequences (e.g., sine wave + noise)
    for _ in range(n_normal):
        time = np.arange(sequence_length)
        signal = np.sin(time * 0.2 + rng.uniform(0, np.pi)) # Add random phase
        noise = rng.normal(0, noise_level, sequence_length)
        sequences.append((signal + noise).reshape(sequence_length, 1))
        labels.append(0)

    # Generate Anomalous Sequences
    for _ in range(n_anomalies):
        time = np.arange(sequence_length)
        signal = np.sin(time * 0.2 + rng.uniform(0, np.pi))
        noise = rng.normal(0, noise_level, sequence_length)
        sequence = signal + noise

        # Inject anomaly
        anomaly_point = rng.integers(1, sequence_length - 1) # Avoid edges
        direction = rng.choice([-1, 1])
        magnitude = anomaly_magnitude * direction

        if anomaly_type == 'spike':
            sequence[anomaly_point] += magnitude
        elif anomaly_type == 'level_shift':
            sequence[anomaly_point:] += magnitude

        sequences.append(sequence.reshape(sequence_length, 1))
        labels.append(1)

    # Shuffle sequences and labels together
    sequences = np.array(sequences).astype(np.float32)
    labels = np.array(labels).astype(int)
    indices = np.arange(n_sequences)
    rng.shuffle(indices)
    sequences = sequences[indices]
    labels = labels[indices]

    if as_frame:
        # Create DataFrame (less standard for sequences, might flatten)
        # Example: Flattening each sequence - creates many columns
        warnings.warn("Returning sequence data as a DataFrame can lead"
                      " to a very wide table. Tuple (sequences, labels)"
                      " is generally preferred.")
        seq_flat = sequences.reshape(n_sequences, -1)
        col_names = [f"t_{i}" for i in range(sequence_length * n_features)]
        df = pd.DataFrame(seq_flat, columns=col_names)
        df['label'] = labels
        df['sequence_id'] = np.arange(n_sequences)

        descr = textwrap.dedent(f"""\
        Synthetic Anomaly Sequence Data (DataFrame Format)

        **Description:**
        Contains {n_sequences} sequences, each of length {sequence_length}
        with {n_features} feature(s). {n_anomalies} sequences contain
        '{anomaly_type}' anomalies of magnitude ~{anomaly_magnitude}.
        Data is flattened in the 'frame'.

        **Data Structure (Bunch object):**
        - frame         : Flattened sequences + label pandas DataFrame.
        - labels        : NumPy array of labels (0=normal, 1=anomaly).
        - feature_names : List of time step column names.
        - target_names  : ['label'].
        - DESCR         : This description.
        """)
        return XBunch(frame=df, labels=labels, feature_names=col_names,
                     target_names=['label'], DESCR=descr)
    else:
        # Return standard NumPy arrays
        return sequences, labels


def make_trend_seasonal_data(
    n_timesteps: int = 365 * 2, # Default 2 years of daily data
    freq: str = 'D',
    trend_order: int = 1, # 0: constant, 1: linear, 2: quadratic
    trend_coeffs: Optional[List[float]] = None, # Specify if order > 0
    seasonal_periods: List[float] = [7, 365.25], # Weekly, Yearly
    seasonal_amplitudes: List[float] = [5, 15], # Amplitudes for each period
    noise_level: float = 1.0,
    base_level: float = 50.0,
    as_frame: bool = False,
    seed: Optional[int] = None,
) -> Union[XBunch, pd.DataFrame]:
    r"""Generate synthetic time series with specified trend and seasonality.

    Creates a univariate time series containing a configurable polynomial
    trend, multiple sinusoidal seasonal components, and Gaussian noise.

    This is useful for testing decomposition methods or how well models
    capture specific trend and seasonal patterns.

    Parameters
    ----------
    n_timesteps : int, default=730
        Number of time steps (rows) to generate.
    freq : str, default='D'
        Pandas frequency string for generating the datetime index.
    trend_order : int, default=1
        Order of the polynomial trend (0=constant, 1=linear, 2=quadratic).
    trend_coeffs : list of float, optional
        Coefficients for the polynomial trend, starting with the constant term.
        Length should be `trend_order + 1`. If None, default coefficients
        are used (e.g., [base_level, 0.1] for order 1). Default is None.
    seasonal_periods : list of float, default=[7, 365.25]
        List of periods for the sinusoidal seasonal components (in number
        of time steps).
    seasonal_amplitudes : list of float, default=[5, 15]
        List of amplitudes corresponding to each period in
        `seasonal_periods`. Length must match `seasonal_periods`.
    noise_level : float, default=1.0
        Standard deviation of the Gaussian noise added to the signal.
    base_level : float, default=50.0
        The constant term (offset) if `trend_order` is 0, or the intercept
        used in default trend coefficients.
    as_frame : bool, default=False
        Return type: ``False`` for Bunch, ``True`` for DataFrame.
    seed : int, optional
        Seed for NumPy's random number generator for reproducibility.

    Returns
    -------
    data : :class:`~fusionlab.api.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        A Bunch object with ``frame``, ``data`` (values), ``target_names``
        (['value']), ``target`` (values array), ``dt_col`` ('date'),
        and ``DESCR``.
        If ``as_frame=True``:
        The generated data as a pandas DataFrame with 'date' and 'value'.

    Raises
    ------
    ValueError
        If lengths of `seasonal_periods` and `seasonal_amplitudes` mismatch,
        or if `trend_coeffs` length doesn't match `trend_order`.

    Examples
    --------
    >>> from fusionlab.datasets import make_trend_seasonal_data
    >>> # Generate data with linear trend and two seasonalities
    >>> data_bunch = make_trend_seasonal_data(n_timesteps=100, freq='D', seed=1)
    >>> print(data_bunch.frame.head())
    >>> data_bunch.frame.plot(x='date', y='value', figsize=(10, 3)) # Quick plot
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if len(seasonal_periods) != len(seasonal_amplitudes):
        raise ValueError("Lengths of 'seasonal_periods' and "
                         "'seasonal_amplitudes' must match.")

    # --- Time Index ---
    date_rng = pd.date_range(
        start='2020-01-01', periods=n_timesteps, freq=freq)
    time_idx = np.arange(n_timesteps) # Simple index for trend calc

    # --- Trend Component ---
    if trend_order < 0:
        raise ValueError("'trend_order' must be >= 0.")
    if trend_coeffs is None:
        # Create default coefficients
        if trend_order == 0: trend_coeffs = [base_level]
        elif trend_order == 1: trend_coeffs = [base_level, 0.1] # Slope 0.1
        elif trend_order == 2: trend_coeffs = [base_level, 0.1, 0.01] # Quadratic term
        else: trend_coeffs = [base_level] + [0.01] * trend_order # Small higher orders
    elif len(trend_coeffs) != trend_order + 1:
        raise ValueError(f"Length of 'trend_coeffs' ({len(trend_coeffs)}) must be "
                         f"'trend_order' + 1 ({trend_order + 1}).")

    # Calculate polynomial trend
    trend_component = np.polynomial.polynomial.polyval(time_idx, trend_coeffs)

    # --- Seasonal Component ---
    seasonal_component = np.zeros(n_timesteps)
    for period, amplitude in zip(seasonal_periods, seasonal_amplitudes):
        if period <= 0: continue # Skip invalid periods
        omega = 2 * np.pi / period
        # Add phase shift to make multiple components distinct
        phase_shift = rng.uniform(0, np.pi / 2)
        seasonal_component += amplitude * np.sin(omega * time_idx + phase_shift)

    # --- Noise Component ---
    noise_component = rng.normal(0, noise_level, n_timesteps)

    # --- Combine Components ---
    value = trend_component + seasonal_component + noise_component

    # --- Create DataFrame ---
    df = pd.DataFrame({'date': date_rng, 'value': value})
    target_col = 'value'
    dt_col = 'date'

    if as_frame:
        return df
    else:
        descr = textwrap.dedent(f"""\
        Synthetic Time Series with Trend and Seasonality

        **Description:**
        A univariate time series generated with {n_timesteps} steps
        (frequency '{freq}'). Includes a polynomial trend of order
        {trend_order}, {len(seasonal_periods)} seasonal component(s) with
        periods {seasonal_periods}, and Gaussian noise with standard
        deviation {noise_level:.2f}.

        **Generation Parameters:**
        - trend_coeffs: {trend_coeffs}
        - seasonal_periods: {seasonal_periods}
        - seasonal_amplitudes: {seasonal_amplitudes}
        - noise_level: {noise_level:.2f}
        - seed: {seed}

        **Data Structure (Bunch object):**
        - frame         : pandas DataFrame with 'date' and 'value'.
        - data          : NumPy array of 'value'.
        - target_names  : ['value'].
        - target        : NumPy array of 'value'.
        - dt_col        : 'date'.
        - DESCR         : This description.
        """)

        return XBunch(
            frame=df,
            data=df[target_col].values,
            target_names=[target_col],
            target=df[target_col].values,
            dt_col=dt_col,
            DESCR=descr
        )

def make_multivariate_target_data(
    n_series: int = 2,
    n_timesteps: int = 100,
    n_targets: int = 2, # Number of target variables
    freq: str = 'D',
    trend_factor: float = 0.1,
    seasonality_period: float = 7,
    seasonality_amplitude: float = 5,
    noise_level: float = 0.5,
    # Control relationship between targets
    cross_target_lag: int = 1,
    cross_target_factor: float = 0.3,
    as_frame: bool = False,
    seed: Optional[int] = None,
) -> Union[XBunch, pd.DataFrame]:
    r"""Generate multi-series data with multiple related target variables.

    Creates a dataset suitable for demonstrating multivariate forecasting.
    It simulates data for multiple independent series (e.g., items) where
    each series has several features (static, dynamic, future) and
    multiple target variables.

    The target variables are generated with some interdependence (e.g.,
    target 2 depends on the lagged value of target 1).

    Parameters
    ----------
    n_series : int, default=2
        Number of independent time series (e.g., items).
    n_timesteps : int, default=100
        Number of time steps (rows) per series.
    n_targets : int, default=2
        Number of related target variables to generate (e.g., 'target_1',
        'target_2', ...).
    freq : str, default='D'
        Pandas frequency string for the datetime index.
    trend_factor : float, default=0.1
        Slope factor for the linear trend component in targets.
    seasonality_period : float, default=7
        Periodicity for the main seasonal component in targets.
    seasonality_amplitude : float, default=5
        Amplitude of the main seasonal component in targets.
    noise_level : float, default=0.5
        Standard deviation of Gaussian noise added to each target.
    cross_target_lag : int, default=1
        Lag used for the dependency between targets (target N depends on
        target N-1 lagged by this amount).
    cross_target_factor : float, default=0.3
        Coefficient determining the strength of dependence between lagged
        targets.
    as_frame : bool, default=False
        Return type: ``False`` for Bunch, ``True`` for DataFrame.
    seed : int, optional
        Seed for NumPy's random number generator.

    Returns
    -------
    data : :class:`~fusionlab.api.bunch.Bunch` or pandas.DataFrame
        If ``as_frame=False`` (default):
        A Bunch object including ``frame`` (DataFrame), lists of
        ``static_features``, ``dynamic_features``, ``future_features``,
        ``target_names`` (list of target columns), ``target`` (NumPy
        array of shape (N_rows, n_targets)), and ``DESCR``.
        If ``as_frame=True``:
        The generated data solely as a pandas DataFrame.

    Examples
    --------
    >>> from fusionlab.datasets import make_multivariate_target_data
    >>> # Generate data with 3 targets for 4 series
    >>> data_bunch = make_multivariate_target_data(n_series=4, n_targets=3, seed=1)
    >>> print(data_bunch.frame.head())
    >>> print("Target names:", data_bunch.target_names)
    >>> print("Target array shape:", data_bunch.target.shape)

    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if n_targets <= 0:
        raise ValueError("'n_targets' must be >= 1.")

    all_series_df = []
    start_date = '2021-01-01'

    for i in range(n_series):
        # --- Time Index ---
        date_rng = pd.date_range(
            start=start_date, periods=n_timesteps, freq=freq)
        time_idx = np.arange(n_timesteps)

        # --- Static Features ---
        series_id = i
        base_level_factor = 1 + rng.uniform(-0.2, 0.2) # Static variation

        # --- Shared Components for Targets ---
        trend = (50 + i * 10) + trend_factor * time_idx
        seasonality = seasonality_amplitude * np.sin(
            2 * np.pi * time_idx / seasonality_period + rng.uniform(0, np.pi)
            )
        base_signal = trend + seasonality

        # --- Generate Multiple Targets ---
        targets = {}
        target_names_list = [f"target_{j+1}" for j in range(n_targets)]
        previous_target_lagged = None

        for j in range(n_targets):
            target_name = target_names_list[j]
            # Base target value
            target_j = base_signal * (base_level_factor + j * 0.1)
            # Add dependency on previous target's lag (if not the first target)
            if j > 0 and previous_target_lagged is not None:
                target_j += cross_target_factor * previous_target_lagged
            # Add noise
            target_j += rng.normal(0, noise_level * (1 + j*0.1), n_timesteps)
            targets[target_name] = target_j
            # Prepare lagged version for the *next* target's calculation
            previous_target_lagged = pd.Series(target_j).shift(
                cross_target_lag).fillna(method='bfill')

        # --- Other Features (Dynamic/Future) ---
        month = date_rng.month
        dayofweek = date_rng.dayofweek
        # Dynamic covariate (example)
        dynamic_cov = rng.normal(5, 1, n_timesteps)
        # Future covariate (example)
        future_event = rng.choice([0, 0, 0, 1], n_timesteps) # Sparse event

        # --- Assemble DataFrame ---
        series_df = pd.DataFrame({
            'date': date_rng,
            'series_id': series_id,
            'base_level_factor': base_level_factor, # Static
            'month': month, # Dynamic/Future
            'dayofweek': dayofweek, # Dynamic/Future
            'dynamic_cov': dynamic_cov, # Dynamic
            'future_event': future_event, # Future
            **targets # Add all target columns
        })
        all_series_df.append(series_df)

    # --- Combine and Define Roles ---
    df = pd.concat(all_series_df).reset_index(drop=True)
    dt_col = 'date'
    target_names = target_names_list # List of generated target names
    spatial_id_col = 'series_id'
    static_features = ['series_id', 'base_level_factor']
    dynamic_features = ['month', 'dayofweek', 'dynamic_cov']
    future_features = ['month', 'dayofweek', 'future_event']
    # Combined feature list for Bunch.feature_names
    feature_names = static_features[1:] + dynamic_features + future_features

    # --- Return ---
    if as_frame:
        ordered_cols = (
             [dt_col, spatial_id_col] + static_features[1:] +
             dynamic_features + future_features + target_names
             )
        ordered_cols = [c for c in ordered_cols if c in df.columns]
        return df[ordered_cols]
    else:
        descr = textwrap.dedent(f"""\
        Synthetic Multi-Series, Multi-Target Data

        **Description:**
        Simulates data for {n_series} independent series over {n_timesteps}
        time steps (frequency '{freq}'). Each series has static, dynamic,
        and future features, along with {n_targets} related target variables.
        Targets exhibit trend, seasonality, noise, and lagged cross-target
        dependencies. Suitable for multivariate forecasting.

        **Generation Parameters:** (Approximate)
        - n_series: {n_series}
        - n_timesteps: {n_timesteps}
        - n_targets: {n_targets}
        - freq: '{freq}'
        - seed: {seed}

        **Data Structure (Bunch object):**
        - frame            : Complete pandas DataFrame.
        - static_features  : List of static column names.
        - dynamic_features : List of dynamic column names.
        - future_features  : List of future column names.
        - target_names     : List of target column names {target_names}.
        - target           : NumPy array of target values shape (N_rows, {n_targets}).
        - dt_col           : Name of datetime column ('{dt_col}').
        - spatial_id_col   : Name of series identifier column ('{spatial_id_col}').
        - feature_names    : Combined list of non-ID/non-target features.
        - DESCR            : This description.
        """)

        target_array = df[target_names].values
        # Extract numerical features for Bunch.data
        try:
            data_cols = [c for c in feature_names if c != spatial_id_col]
            data_array = df[data_cols].select_dtypes(include=np.number).values
        except:
             data_array = None

        return XBunch(
            frame=df,
            static_features=static_features,
            dynamic_features=dynamic_features,
            future_features=future_features,
            target_names=target_names,
            target=target_array,
            dt_col=dt_col,
            spatial_id_col=spatial_id_col,
            feature_names=feature_names,
            data=data_array,
            DESCR=descr
        )