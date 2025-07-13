# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
geospatial_utils - A collection of utilities for geospatial and positional 
data analysis, filtering, and transformations.
"""
from __future__ import print_function, annotations 

import warnings
from numbers import Real
from scipy.spatial import cKDTree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
 
from ..api.types import (
    DataFrame, 
    Optional,
    Tuple, 
    Union, 
    List,
)
from ..compat.sklearn import validate_params, StrOptions, Interval 
from ..core.array_manager import extract_array_from 
from ..core.checks import ( 
    exist_features, 
    assert_ratio, 
    are_all_frames_valid, 
    check_spatial_columns
    )
from ..core.handlers import columns_manager, resolve_label  
from ..core.io import SaveFile, is_data_readable 
# from ..core.plot_manager import set_axis_grid  
from ..decorators import Deprecated, isdf 
from .generic_utils import find_id_column 
from .validator import ( 
    validate_positive_integer, 
    validate_length_range , 
    filter_valid_kwargs, 
    parameter_validator
    )

HAS_TQDM=True 
try: 
    from tqdm import tqdm 
except: 
    HAS_TQDM = False 
    
__all__ = [
     'spatial_sampling', 
     'extract_coordinates', 
     'batch_spatial_sampling', 
     'extract_zones_from', 
     'filter_position', 
     'create_spatial_clusters', 
     'gen_negative_samples', 
     'gen_buffered_negative_samples', 
     'gen_negative_samples_plus', 
     'extract_spatial_roi', 
 ]

@SaveFile
@isdf 
def extract_spatial_roi(
    df: pd.DataFrame,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    x_col: str = 'longitude',
    y_col: str = 'latitude',
    snap_to_closest: bool = True,
    savefile: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Extracts a spatial Region of Interest (ROI) from a DataFrame.

    This function filters a DataFrame to include only the data points
    that fall within a specified rectangular bounding box defined by
    x and y coordinate ranges.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the spatial data.
    x_range : tuple of (float, float)
        A tuple containing the minimum and maximum desired values for
        the x-coordinate (e.g., longitude). The order does not matter.
    y_range : tuple of (float, float)
        A tuple containing the minimum and maximum desired values for
        the y-coordinate (e.g., latitude). The order does not matter.
    x_col : str, default='longitude'
        The name of the column in `df` that contains the x-coordinates.
    y_col : str, default='latitude'
        The name of the column in `df` that contains the y-coordinates.
    snap_to_closest : bool, default=True
        If True, and a value in `x_range` or `y_range` does not
        exist in the data, the function will "snap" to the nearest
        available coordinate in the dataset. If False, it will use
        the exact boundaries provided.
    savefile : str, optional
        The path to save the resulting DataFrame as a CSV file.
    Returns
    -------
    pd.DataFrame
        A new DataFrame containing only the rows that fall within the
        specified spatial bounding box.

    Raises
    ------
    ValueError
        If `x_col` or `y_col` are not found in the DataFrame, or if the
        range tuples are not provided correctly.
    """
    if not all(c in df.columns for c in [x_col, y_col]):
        raise ValueError(
            f"Columns '{x_col}' and/or '{y_col}' not found in DataFrame."
        )
    from fusionlab.core.checks import check_numeric_dtype 
    # this check the whether the xcol and y_col are numeric valid 
    check_numeric_dtype(
        df[[x_col, y_col]], 
        param_names={"X": "df[x_col, y_col]]"}
    )
    # Ensure ranges are correctly ordered (min, max)
    x_min, x_max = min(x_range), max(x_range)
    y_min, y_max = min(y_range), max(y_range)

    # Optionally snap the boundaries to the closest available data points
    if snap_to_closest:
        x_min = _find_closest_value(df[x_col], x_min)
        x_max = _find_closest_value(df[x_col], x_max)
        y_min = _find_closest_value(df[y_col], y_min)
        y_max = _find_closest_value(df[y_col], y_max)

    # Create boolean masks for the x and y ranges
    x_mask = (df[x_col] >= x_min) & (df[x_col] <= x_max)
    y_mask = (df[y_col] >= y_min) & (df[y_col] <= y_max)

    # Filter the DataFrame using the combined mask
    roi_df = df[x_mask & y_mask].copy()

    return roi_df

def _find_closest_value(series: pd.Series, value: float) -> float:
    """Finds the value in a Series closest to a target value."""
    return series.iloc[(series - value).abs().argmin()]

@SaveFile 
@isdf  
def gen_negative_samples_plus(
    df: pd.DataFrame,
    target_col: str,
    spatial_cols: Tuple[str, str] = (
        'longitude',
        'latitude'
    ),
    feature_cols: Optional[List[str]] = None,
    buffer_km: float = 10,
    neg_feature_range: Tuple[float, float] = (
        0, 5
    ),
    num_neg_per_pos: int = 1,
    strategy: str = 'landslide', 
    gauge_data: Optional[pd.DataFrame] = None,
    elevation_data: Optional[pd.DataFrame] = None,
    similarity_features: Optional[
        List[str]
    ] = None,
    time_col: Optional[str] = None,
    cluster_method: str = 'kmeans',  
    use_gpd: Union[bool, str] = 'auto',
    id_col='auto',
    view: bool = False,
    savefile: Optional[str] = None,
    verbose: int = 1,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generates negative samples for modeling in spatial scenarios,
    offering multiple strategies. The function calls
    `gen_buffered_negative_samples` when the ``strategy`` argument
    is ``'landslide'``, ``'event'``, or ``'gauge'``. It also calls
    `generate_negative_samples` partially in the ``'hybrid'``
    strategy. Internally, each sample is augmented to produce
    negative instances according to a chosen method.
    
    .. math::
       \\text{buffer_deg} = \\frac{\\text{buffer_km}}{111.0}
    
    The above formula approximates degrees from kilometers near
    the equator. The output is a combined dataset containing
    original positives and generated negatives. The ratio of
    negatives per positive is controlled by
    ``num_neg_per_pos``.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing spatial coordinates and features.
    target_col : str
        Name of the classification target column. Positive
        samples in `df` are labeled, negatives will be generated.
    spatial_cols : tuple of str, optional
        Columns representing longitude and latitude in `df`.
    feature_cols : list of str, optional
        Additional feature columns used to drive or constrain
        negative sampling processes.
    buffer_km : float, optional
        Radius in kilometers for local negative sampling. Used
        to compute buffer degrees.
    neg_feature_range : tuple of float, optional
        Lower and upper range for continuous features in random
        negative generation.
    num_neg_per_pos : int, optional
        Number of negatives to generate per positive instance.
    strategy : str, optional
        Defines the sampling approach. Options include
        ``'landslide'``, ``'gauge'``, ``'random_global'``,
        ``'temporal_shift'``, ``'clustered_negatives'``,
        ``'environmental_similarity'``, ``'elevation_based'``,
        and ``'hybrid'``.
        
        See more in :ref:`User Guide <user_guide>`.
        
    gauge_data : pandas.DataFrame, optional
        Reference data for gauge-based or hybrid strategies.
    elevation_data : pandas.DataFrame, optional
        Elevation records, used if ``strategy='elevation_based'``.
    similarity_features : list of str, optional
        Columns for nearest neighbor computation in
        ``'environmental_similarity'``.
    time_col : str, optional
        Name of the time column for ``'temporal_shift'``. Required
        if using that strategy.
    cluster_method : str, optional
        Clustering algorithm for ``'clustered_negatives'``. Default
        is ``'kmeans'``.
    use_gpd : bool or str, optional
        Indicator for whether geopandas is used in buffer-based
        processes.
    id_col : str, optional
        Column name used as an identifier. If `'auto'`, a default is
        used.
    view : bool, optional
        Flag for visualizing or previewing the results.
    savefile : str, optional
        Path to save the output dataset. If None, no file is saved.
    verbose : int, optional
        Verbosity level. Higher values yield more logs.
    seed : int, optional
        Random seed for reproducibility. If None, randomness is not
        fixed.
    
    Returns
    -------
    pandas.DataFrame
        A combined DataFrame containing the original positive samples
        labeled as 1 and newly generated negative samples labeled 0.
    
    Notes
    -----
    When ``strategy='hybrid'``, partial sets of negatives come from
    two distinct calls to `generate_negative_samples` for
    ``'landslide'`` and ``'gauge'`` sub-strategies, then merged.
    
    Examples
    --------
    >>> from fusionlab.utils.spatial_utils import gen_negative_samples_plus
    >>> import pandas as pd
    >>> df_example = pd.DataFrame({{
    ...     "longitude": [10.1, 10.2, 10.3],
    ...     "latitude":  [45.1, 45.2, 45.3],
    ...     "feature":   [3.4, 2.1, 6.7],
    ...     "target":    [1, 1, 1]
    ... }})
    >>> gauge_data = pd.DataFrame({{
    ...     'gauge_id': ['G1', 'G2', 'G3'],
    ...     'latitude': np.random.uniform(24.0, 25.0, 3),
    ...     'longitude': np.random.uniform(113.0, 114.0, 3)
    ... }})
    >>> # Generate random global negatives
    >>> result = gen_negative_samples_plus(
    ...     df_example,
    ...     target_col="target",
    ...     strategy="random_global"
    ... )
    >>> print(result.head())
    
    See Also
    --------
    gen_buffered_negative_samples : Generates negative samples
        within a buffer region around reference events or gauges.
    generate_negative_samples : A simpler negative sampling
        utility for certain strategies.
    
    References
    ----------
    .. [1] P. Goovaerts, "Geostatistics for Natural Resources
       Evaluation," Oxford University Press, 1997.
    """

    # If the strategy is one of the recognized
    # types (landslide, event, gauge), we
    # directly call gen_buffered_negative_samples.
    if str(strategy).lower() in (
        'landslide',
        'event',
        'gauge'
    ):
        return gen_buffered_negative_samples(
            df=df,
            target_col=target_col,
            spatial_cols=spatial_cols,
            feature_cols=feature_cols,
            buffer_km=buffer_km,
            neg_feature_range=neg_feature_range,
            num_neg_per_pos=num_neg_per_pos,
            strategy=strategy,
            gauge_data=gauge_data,
            use_gpd=use_gpd,
            view=view,
            id_col=id_col,
            savefile=savefile,
            seed=seed, 
            verbose=verbose
        )

    # If not in the above categories, we proceed
    # with advanced strategies.
    np.random.seed(seed)

    # Validate columns and parameters via the
    # _validate_negative_sampling utility.
    checked_values = _validate_negative_sampling(
        df=df,
        target_col=target_col,
        spatial_cols=spatial_cols,
        feature_cols=feature_cols,
        neg_feature_range=neg_feature_range,
        num_neg_per_pos=num_neg_per_pos,
        verbose=verbose,
        id_col=id_col
    )
    (
        spatial_cols,
        feature_cols,
        neg_feature_range,
        num_neg_per_pos
    ) = checked_values

    # Extract column names for lon/lat.
    lon_col, lat_col = spatial_cols

    # Convert buffer kilometers to degrees,
    # approximating 1 deg ~ 111 km.
    buffer_deg = buffer_km / 111.0

    # Additional advanced strategy options.
    add_strategies = [
        'random_global',
        'temporal_shift',
        'clustered_negatives',
        'environmental_similarity',
        'elevation_based',
        'hybrid'
    ]

    # Validate the selected strategy from
    # the advanced list.
    strategy = parameter_validator(
        'strategy',
        target_strs=add_strategies,
        deep=True
    )(strategy)

    # Collect negative samples in a list.
    negatives = []

    # -----------------------
    # RANDOM_GLOBAL strategy:
    # Generate uniform random coordinates
    # across the bounding box of df.
    # -----------------------
    if strategy == 'random_global':
        lon_min, lon_max = (
            df[lon_col].min(),
            df[lon_col].max()
        )
        lat_min, lat_max = (
            df[lat_col].min(),
            df[lat_col].max()
        )
        for _ in range(len(df) * num_neg_per_pos):
            sample = {
                lon_col: np.random.uniform(
                    lon_min,
                    lon_max
                ),
                lat_col: np.random.uniform(
                    lat_min,
                    lat_max
                ),
                target_col: 0
            }
            for col in feature_cols:
                sample[col] = np.random.uniform(
                    *neg_feature_range
                )
            negatives.append(sample)

    # -----------------------
    # TEMPORAL_SHIFT strategy:
    # Shift time backward (e.g. 30 days)
    # and mark as negative samples.
    # -----------------------
    elif strategy == 'temporal_shift':
        if time_col is None:
            raise ValueError(
                "Time column must be provided "
                "for 'temporal_shift'."
            )
        df_shifted = df.copy()
        df_shifted[time_col] = pd.to_datetime(
            df_shifted[time_col]
        ) - pd.Timedelta(days=30)
        for _, row in df_shifted.iterrows():
            sample = row.to_dict()
            sample[target_col] = 0
            negatives.append(sample)

    # ---------------------------
    # CLUSTERED_NEGATIVES:
    # Cluster positive points and
    # sample negatives around the
    # resulting centroids. Two
    # clustering options are
    # supported: KMeans and DBSCAN.
    # ---------------------------
    elif strategy == 'clustered_negatives':
        coords = df[[lon_col, lat_col]].values
    
        # --- choose clustering backend
        if cluster_method.lower() == 'kmeans':
            clustering = KMeans(
                n_clusters=max(2, len(df) // 10),
                random_state=seed
            ).fit(coords)
            centers = clustering.cluster_centers_
    
        elif cluster_method.lower() == 'dbscan':
            clustering = DBSCAN(
                eps=buffer_deg * 2,     # search radius (deg)
                min_samples=5
            ).fit(coords)
            labels  = clustering.labels_
            centers = []
            for lbl in np.unique(labels):
                if lbl == -1:           # DBSCAN noise
                    continue
                pts = coords[labels == lbl]
                centers.append(pts.mean(axis=0))
            centers = np.asarray(centers)
    
        else:
            raise ValueError(
                "cluster_method must be either "
                "'KMeans' or 'DBSCAN'"
            )
        # --- generate negatives near each centroid
        for center in centers:
            for _ in range(num_neg_per_pos):
                lat_offset = np.random.uniform(
                    -buffer_deg, buffer_deg
                )
                lon_offset = np.random.uniform(
                    -buffer_deg, buffer_deg
                )
                sample = {
                    lat_col: center[1] + lat_offset,
                    lon_col: center[0] + lon_offset,
                    target_col: 0
                }
                for col in feature_cols:
                    sample[col] = np.random.uniform(
                        *neg_feature_range
                    )
                negatives.append(sample)
    
    # -----------------------
    # ENVIRONMENTAL_SIMILARITY:
    # Nearest neighbor approach on
    # specific feature vectors.
    # -----------------------
    elif strategy == 'environmental_similarity':
        if similarity_features is None:
            raise ValueError(
                "Specify similarity_features for "
                "'environmental_similarity'."
            )
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(df[similarity_features])
        for _ in range(len(df) * num_neg_per_pos):
            random_vector = np.random.uniform(
                0, 1,
                size=len(similarity_features)
            )
            dist, idx = nn.kneighbors([
                random_vector
            ])
            ref = df.iloc[idx[0][0]]
            sample = ref.copy()
            sample[target_col] = 0
            sample[lon_col] += np.random.uniform(
                -buffer_deg,
                buffer_deg
            )
            sample[lat_col] += np.random.uniform(
                -buffer_deg,
                buffer_deg
            )
            negatives.append(
                sample.to_dict()
            )

    # -----------------------
    # ELEVATION_BASED:
    # Pick negative samples from
    # low slope areas, etc.
    # -----------------------
    elif strategy == 'elevation_based':
        if elevation_data is None:
            raise ValueError(
                "Provide elevation data for "
                "'elevation_based' strategy."
            )
        low_slope_areas = elevation_data[
            elevation_data['slope'] < buffer_km
        ]
        sample_count = min(
            len(df) * num_neg_per_pos,
            len(low_slope_areas)
        )
        for _, row in low_slope_areas.sample(
            n=sample_count
        ).iterrows():
            sample = row.to_dict()
            sample[target_col] = 0
            negatives.append(sample)

    # -----------------------
    # HYBRID:
    # Combine partial sets of
    # negative samples from
    # different strategies.
    # -----------------------
    elif strategy == 'hybrid':
        part1 = gen_buffered_negative_samples(
            df,
            target_col,
            spatial_cols,
            feature_cols,
            buffer_km,
            neg_feature_range,
            num_neg_per_pos // 2,
            strategy='landslide',
            gauge_data=None,
            seed=seed,
            verbose=0
        )
        part2 = gen_buffered_negative_samples(
            df,
            target_col,
            spatial_cols,
            feature_cols,
            buffer_km,
            neg_feature_range,
            num_neg_per_pos // 2,
            strategy='gauge',
            gauge_data=gauge_data,
            seed=seed,
            verbose=0
        )
        combined = pd.concat([part1, part2], ignore_index=True)

        # Deduplicate based on spatial + feature + target columns
        dedup_cols = spatial_cols + (feature_cols or []) + [target_col]
        combined = combined.drop_duplicates(subset=dedup_cols)
    
        if verbose >= 1:
            print(
                "[INFO] Final dataset after deduplication:"
                f" {len(combined)} samples"
            )
            
        if view:
            _visualize_negative_sampling(
                df_combined=combined,
                base_points= df if strategy == 'landslide' else gauge_data,
                strategy=strategy,
                spatial_cols=spatial_cols,
                target_col=target_col,
                buffer_km=buffer_km, 
                title="Sample Generation via Hybrid Strategy",
            )
        return combined

    # Create DataFrame of negative samples.
    df_neg = pd.DataFrame(negatives)

    # Copy df for positive samples.
    df_pos = df.copy()
    df_pos[target_col] = 1

    # Combine positives and negatives.
    combined = pd.concat(
        [df_pos, df_neg],
        ignore_index=True
    )
    
    if view:
        _visualize_negative_sampling(
            df_combined=combined,
            base_points= df,
            strategy=strategy,
            spatial_cols=spatial_cols,
            target_col=target_col,
            buffer_km=buffer_km, 
        )
        
    # Optionally print info about the results.
    if verbose >= 1:
        print(
            f"[INFO] Generated {len(df_neg)} negative "
            f"samples using strategy '{strategy}'"
        )
        print(
            f"[INFO] Total dataset: {len(combined)} "
            "samples"
        )

    return combined

@SaveFile 
@isdf          
def gen_buffered_negative_samples(
    df: pd.DataFrame,
    target_col: str,
    spatial_cols: Tuple[str, str] = ('longitude', 'latitude'),
    feature_cols: Optional[List[str]] = None,
    buffer_km: float = 10,
    neg_feature_range: Tuple[float, float] = (0, 5),
    num_neg_per_pos: int = 1,
    strategy: str = 'landslide',  
    gauge_data: Optional[pd.DataFrame] = None,
    use_gpd: Union[bool, str] = 'auto',
    id_col='auto', 
    view: bool = False,
    savefile: Optional[str] = None,
    seed: Optional[int] = None, 
    verbose: int = 1,
) -> pd.DataFrame:
    
    """
    Generate buffer-based negative samples around existing
    points or gauge stations.
    
    This function creates additional negative samples
    for binary spatial events. It either takes an existing
    landslide dataset (when `strategy` is `'landslide'`)
    or a separate gauge dataset (if `strategy` is
    `'gauge'`) to serve as the base points for generating
    negatives within a circular buffer. The function
    validates input columns and parameters via
    `_validate_negatives_sampling` before constructing
    synthetic samples [1]_.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing positive event samples
        (e.g., landslides). Must include `<target_col>`
        and `<spatial_cols>`.
    target_col : str
        Name of the binary target column (1 for event,
        0 for no event).
    spatial_cols : tuple of str, default ('longitude', 'latitude')
        Indicates which columns hold `<longitude>` and
        `latitude` in `df`.
    feature_cols : list of str, optional
        Additional feature columns to simulate or copy
        for negatives. If ``None``, all columns except
        `<spatial_cols>` and `target_col` are used.
    buffer_km : float, default 10
        The radial distance in kilometers for sampling
        negative points around each base point.
    neg_feature_range : tuple of float, default (0, 5)
        A numeric range from which feature values are
        drawn for negative samples if the column
        is numeric.
    num_neg_per_pos : int, default 1
        Number of negatives to generate per positive
        (landslide) or gauge point.
    strategy : str, default 'landslide'
        Defines the base from which negative samples
        are generated:
          - `'landslide'` or ``'event'``: Use rows from `df`
            as base. 
          - `'gauge'`: Use rows from `<gauge_data>` as base.
    gauge_data : pandas.DataFrame, optional
        Required if `<strategy>` is `'gauge'`. Must
        contain `<spatial_cols>`.
    use_gpd : bool or 'auto', default 'auto'
        If `'auto'`, attempts to use GeoPandas for
        visualization if installed. Otherwise,
        falls back to Matplotlib. This parameter
        is forwarded to the underlying visualization
        function.
    id_col : str or list of str, default 'auto'
        Column(s) representing IDs in `<df>`. If
        `'auto'`, the function tries to detect
        possible ID columns. Used by
        `_validate_negatives_sampling`.
    view : bool, default False
        Whether to visualize the sampled negatives
        around the base points.
    savefile : str, optional
        If provided, saves the final combined dataset
        (positives and negatives) to a CSV file at the
        specified path.
    seed : int, optional
        Seed for NumPy's random generator, ensuring
        reproducible offsets in negative sampling.
    verbose : int, default 1
        Controls console messages: `1` for minimal,
        `2` for more detailed logs.

    Returns
    -------
    pandas.DataFrame
        The combined dataset containing both the original
        (positive) rows, labeled with `target_col`=1,
        and the newly generated negative rows, labeled
        `target_col`=0.
    
    Methods
    -------
    `_validate_negatives_sampling`
        Validates required columns and parameters,
        including `<num_neg_per_pos>` and
        `<neg_feature_range>`.
    `visualize_negative_sampling`
        Generates a plot showing the negative samples
        around the base points if `<view>` is True.
    
    Notes
    -----
    - If `strategy` is `'gauge'`, `gauge_data` must
      be provided and contain columns `longitude` and
      `latitude`.
    - When `<view>` is True, circles are drawn to
      illustrate the buffer radius.
    - The ratio of 1° ~ 111 km is approximate and can
      vary slightly by latitude [1]_.
      
    Formally, a buffer in degrees
    :math:`\\Delta` is computed by:
    
    .. math::
       \\Delta = \\frac{\\text{buffer\\_km}}{111},
    
    where :math:`111` is an approximate km-per-degree
    conversion factor. Each base point
    :math:`(lat, lon)` spawns :math:`n` negatives, each
    offset by :math:`\\delta_{lat}`, :math:`\\delta_{lon}`
    drawn from :math:`U(-\\Delta, \\Delta)`.
    
    
    Examples
    --------
    Below is an illustration of how to generate negative samples
    around both existing event locations (strategy=`landslide`)
    and separate gauge stations (strategy=`gauge`) using
    ``gen_buffered_negative_samples``.
    
    First, we simulate a small DataFrame of positive
    landslide samples with rainfall attributes, as well
    as a separate DataFrame for gauge stations:
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> np.random.seed(42)
    
    >>> positive_samples = pd.DataFrame({
    ...     'id': [1, 2, 3, 4, 5],
    ...     'latitude': np.random.uniform(24.0, 25.0, 5),
    ...     'longitude': np.random.uniform(113.0, 114.0, 5),
    ...     'rainfall_day_1': np.random.randint(10, 30, 5),
    ...     'rainfall_day_2': np.random.randint(10, 30, 5),
    ...     'rainfall_day_3': np.random.randint(10, 30, 5),
    ...     'rainfall_day_4': np.random.randint(10, 30, 5),
    ...     'rainfall_day_5': np.random.randint(10, 30, 5),
    ...     'landslide': [1]*5
    ... })
    
    >>> gauge_data = pd.DataFrame({
    ...     'gauge_id': ['G1', 'G2', 'G3'],
    ...     'latitude': np.random.uniform(24.0, 25.0, 3),
    ...     'longitude': np.random.uniform(113.0, 114.0, 3)
    ... })
    
    We then call ``gen_buffered_negative_samples`` to
    produce negatives around these data using two
    different strategies:
    
    >>> from fusionlab.utils.spatial_utils import gen_buffered_negative_samples
    
    >>> # Generate negatives around landslide points
    >>> results_landslide = generate_negative_samples_with(
    ...     df=positive_samples,
    ...     target_col='landslide',
    ...     spatial_cols=('longitude', 'latitude'),
    ...     feature_cols=[f'rainfall_day_{i+1}' for i in range(5)],
    ...     buffer_km=10,
    ...     num_neg_per_pos=1,
    ...     strategy='landslide',
    ...     verbose=1
    ... )
    
    >>> # Generate negatives around the gauge stations
    >>> results_gauge = gen_buffered_negative_samples(
    ...     df=positive_samples,
    ...     target_col='landslide',
    ...     spatial_cols=('longitude', 'latitude'),
    ...     feature_cols=[f'rainfall_day_{i+1}' for i in range(5)],
    ...     buffer_km=10,
    ...     num_neg_per_pos=1,
    ...     strategy='gauge',
    ...     gauge_data=gauge_data,
    ...     verbose=1
    ... )

    
    See Also
    --------
    generate_negative_samples: 
        Generate synthetic negative samples for spatial 
        binary classification tasks.
    _validate_negatives_sampling: 
        Ensures inputs and parameters are correct.
    visualize_negative_sampling : 
        Plots the positive and negative points for inspection.
    
    References
    ----------
    .. [1] "What is a degree of Latitude/Longitude?"
           US National Geodetic Survey (NGS),
           https://www.ngs.noaa.gov/.
    """
    # See for reproducibility 
    strategy = ( 
        'landslide' if strategy.lower() in ('landslide','event')
        else strategy
    )
    np.random.seed(seed)

    # Validate input and columns
    checked_values = _validate_negative_sampling (
        df=df, 
        target_col=target_col, 
        spatial_cols= spatial_cols, 
        feature_cols= feature_cols, 
        neg_feature_range=neg_feature_range, 
        num_neg_per_pos= num_neg_per_pos, 
        verbose=verbose, 
        id_col=id_col, 
        )
    ( 
     spatial_cols, feature_cols,
     neg_feature_range, num_neg_per_pos
     ) = checked_values  
    
    lon_col, lat_col = spatial_cols 

    if verbose >= 1:
        print(f"[INFO] Generating negative samples using strategy: {strategy}")

    buffer_deg = buffer_km / 111.0
    negatives = []
    
    if strategy == 'gauge':
        if gauge_data is None:
            raise ValueError(
                "Gauge data must be provided for 'gauge' strategy."
                )

    base_points = df if strategy == 'landslide' else gauge_data

    if (base_points is None
       or not all(c in base_points.columns for c in spatial_cols)):
        raise ValueError(
            "Missing gauge data or invalid spatial columns."
        )

    for _, point in base_points.iterrows():
        for _ in range(num_neg_per_pos):
            lat_offset = np.random.uniform(
                -buffer_deg,
                buffer_deg
            )
            lon_offset = np.random.uniform(
                -buffer_deg,
                buffer_deg
            )
            new_lat = point[lat_col] + lat_offset
            new_lon = point[lon_col] + lon_offset

            sample = {
                lat_col: new_lat,
                lon_col: new_lon,
                target_col: 0
            }

            for col in feature_cols:
                sample[col] = np.random.uniform(
                    *neg_feature_range
                )

            negatives.append(sample)

    df_neg = pd.DataFrame(negatives)
    df_pos = df.copy()
    df_pos[target_col] = 1

    combined = pd.concat(
        [df_pos, df_neg],
        ignore_index=True
    )

    if verbose >= 1:
        print(f"[INFO] Generated {len(df_neg)} negative samples")
        print(f"[INFO] Total dataset: {len(combined)} samples")

    if view:
        _visualize_negative_sampling(
            df_combined=combined,
            base_points=base_points,
            strategy=strategy,
            spatial_cols=spatial_cols,
            target_col=target_col,
            buffer_km=buffer_km
        )

    return combined

def _visualize_negative_sampling(
        df_combined, base_points, strategy, spatial_cols, 
        target_col, buffer_km=10, s=50, title=None, 
        ):
    

    lon_col, lat_col = spatial_cols
    buffer_deg = buffer_km / 111.0

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {1: 'red', 0: 'green'}
    labels = {1: 'Positive', 0: 'Negative'}

    for val in [1, 0]:
        subset = df_combined[df_combined[target_col] == val]
        ax.scatter(subset[lon_col], subset[lat_col],
                   color=colors[val], label=f"{labels[val]} Samples", s=s)

    for _, row in base_points.iterrows():
        circle = Circle(
            (row[lon_col], row[lat_col]),
            buffer_deg,
            color='blue',
            alpha=0.2
        )
        ax.add_patch(circle)

    if strategy == 'gauge':
        ax.scatter(base_points[lon_col], base_points[lat_col], 
                   color='blue', label='Gauges',
                   marker='x', s=+20
                   )
    if strategy == 'clustered_negatives':
        ax.scatter(base_points[lon_col], base_points[lat_col],
                   color='blue', label='Cluster Centers',
                   marker='P', s=s +30
                   )
        
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"Negative Sample Generation Strategy: {strategy}")
    ax.legend(handles=[
        Patch(color='red', label='Positive Samples'),
        Patch(color='green', label='Negative Samples'),
        Patch(color='blue', alpha=0.2, label=f'{buffer_km} km Buffer')
    ])
    
    ax.grid (True, **{"linestyle":":", "alpha": 0.7})

    plt.tight_layout()
    plt.show()

@SaveFile
@isdf
def gen_negative_samples(
    df: DataFrame,
    target_col: str,
    spatial_cols: Tuple [str, str]=('longitude', 'latitude'),
    feature_cols: Optional[List[str]]=None,
    buffer_km: float=10,
    neg_feature_range: Tuple[float, float]=(0, 5),
    num_neg_per_pos: int=1,
    use_gpd: Union [bool, str]='auto',
    view: bool=False,
    savefile: Optional[str] = None, 
    verbose: int=1
):
    r"""
    Generate synthetic negative samples for spatial binary
    classification tasks.
    
    This function creates additional samples labeled as
    non-events within a specified spatial buffer around
    the positive (event) observations. The main idea is to
    generate negative examples that reflect realistic
    conditions but have not triggered an event, thereby
    assisting models in distinguishing occurrences from
    non-occurrences [1]_.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the positive samples
        (events). Must include both the target column
        and the specified spatial columns.
    target_col : str
        Column name for the binary target (e.g.
        `landslide`). Rows where this column is
        1 (or True) are considered positive samples.
    spatial_cols : tuple of str, default ('longitude', 'latitude')
        Tuple specifying the longitude> and latitude
        column names in `df`.
    feature_cols : list of str or None, default None
        List of feature columns to use or to simulate
        for generated negatives. If ``None``, the
        function automatically detects numeric and
        categorical columns excluding `spatial_cols`
        and `target_col`.
    buffer_km : float, default 10
        Spatial buffer in kilometers used to define
        the radius around each positive sample
        within which negative samples are created.
    neg_feature_range : tuple of int, default (0, 5)
        Value range (minimum, maximum) used for
        simulating numeric feature values in negative
        samples if the corresponding feature column
        does not exist in `<df>`.
    num_neg_per_pos : int, default 1
        Number of negative samples to generate
        per positive sample. For instance, if
        ``num_neg_per_pos=2``, each positive sample
        spawns two negatives.
    use_gpd : str, default 'auto'
        If set to 'auto', the function tries to
        import GeoPandas for visualization. If
        `'none'`, no GeoPandas usage will occur.
    view : bool, default False
        Whether to visualize the generated samples
        on a map. Attempts to use `geopandas` if
        installed; falls back to `matplotlib` if
        `'auto'` is chosen and GeoPandas is not
        available.
    savefile : str or None, default None
        Path to which the resulting DataFrame is
        saved if provided. Handled by the decorator
        that wraps this function.
    verbose : int, default 1
        Verbosity level. `0` for silent,
        `1` for progress indication, `2` for
        more messages, `3` for debugging output.
    
    Returns
    -------
    pandas.DataFrame
        Combined DataFrame with both original positive
        samples and newly generated negative samples. The
        `<target_col>` is 1 for positives and 0 for negatives.
    
    Methods
    -------
    `columns_manager`
        This internal function is used to handle the
        processing of columns for features and
        spatial parameters.
    
    Notes
    -----
    - If a feature column exists in `df`, the negative
      samples will copy or randomly select categories for
      categorical columns, and sample integers within
      ``neg_feature_range`` for numeric columns.
    - If `feature_cols` is empty or does not exist
      in `df`, the function simulates all values for
      negative samples.
    - When `view=True`, circles depicting the buffer
      zone around each positive sample are drawn for
      visualization.
    - The approximation of 1° ~ 111 km varies slightly
      depending on latitude [1]_.
    
    Mathematically, we define the spatial buffer in degrees
    as:
    
    .. math::
       \\Delta = \\frac{\\text{buffer_km}}{111.0},
    
    where :math:`111.0` km approximates the distance of
    one degree of latitude or longitude [1]_. For each
    positive sample at location :math:`(lat, lon)`,
    we generate :math:`n` new points with offsets
    :math:`\\delta_{lat}` and :math:`\\delta_{lon}`, each
    drawn from a uniform distribution
    :math:`U(-\\Delta, \\Delta)`:
    
    .. math::
       \\begin{aligned}
       &lat_{new} = lat + \\delta_{lat},\\\\
       &lon_{new} = lon + \\delta_{lon}.
       \\end{aligned}
    
    Combined with randomly sampled or inferred feature
    values, these new samples serve as negative examples
    for modeling tasks such as landslide prediction.
    
    Examples
    --------
    >>> from fusionlab.utils.spatial_utils import gen_negative_samples
    >>> import pandas as pd
    >>> import numpy as np
    >>> df_pos = pd.DataFrame({
    ...        'latitude': np.random.uniform(24.0, 25.0, 5),
    ...        'longitude': np.random.uniform(113.0, 114.0, 5),
    ...        'rainfall_day_1': np.random.randint(10, 30, 5),
    ...        'rainfall_day_2': np.random.randint(10, 30, 5),
    ...        'rainfall_day_3': np.random.randint(10, 30, 5),
    ...        'rainfall_day_4': np.random.randint(10, 30, 5),
    ...        'rainfall_day_5': np.random.randint(10, 30, 5),
    ...        'landslide': 1
    ...    })
    >>> combined = gen_negative_samples(
    ...     df=df_pos,
    ...     target_col='landslide',
    ...     buffer_km=10,
    ...     num_neg_per_pos=2,
    ...     view=False,
    ...     verbose=2
    ... )
    >>> print(combined.head())
    
    See Also
    --------
    check_spatial_columns : Ensures the existence of
                              required spatial columns.
    exist_features : Verifies the presence of
                       specified features in `<df>`.
    columns_manager : Handles both feature and spatial
                        columns for processing.
    
    References
    ----------
    .. [1] US National Geodetic Survey (NGS). "What is a
           degree of Latitude/Longitude?"
           (https://www.ngs.noaa.gov/).
    """
    # Validate input and columns
    checked_values = _validate_negative_sampling (
        df=df, 
        target_col=target_col, 
        spatial_cols= spatial_cols, 
        feature_cols= feature_cols, 
        neg_feature_range=neg_feature_range, 
        num_neg_per_pos= num_neg_per_pos, 
        verbose=verbose 
        )
    ( 
     spatial_cols, feature_cols,
     neg_feature_range, num_neg_per_pos
     ) = checked_values  
    
    lon_col, lat_col = spatial_cols 
    
    # Check GeoPandas
    HAS_GPD = False
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        HAS_GPD = True
    except ImportError:
        HAS_GPD = False

    # Buffer conversion (km -> deg)
    buffer_deg = buffer_km / 111.0

    if verbose >= 2:
        print(f"[INFO] Generating negative samples within "
              f"{buffer_km} km buffer zones.")
    if verbose >= 3:
        print(f"[DEBUG] Buffer in degrees: {buffer_deg:.4f}")

    # Generate negative samples per row
    negative_samples = []
    iterator = df.iterrows()
    if verbose >= 1 and HAS_TQDM:
        iterator = tqdm(
            iterator,
            total=len(df),
            desc="Generating negatives"
        )

    for _, row in iterator:
        for _ in range(num_neg_per_pos):
            lat_offset = np.random.uniform(-buffer_deg,
                                           buffer_deg)
            lon_offset = np.random.uniform(-buffer_deg,
                                           buffer_deg)
            new_lat = row[lat_col] + lat_offset
            new_lon = row[lon_col] + lon_offset

            sample = {
                lat_col: new_lat,
                lon_col: new_lon,
                target_col: 0
            }

            # Simulate or copy feature columns for negs
            for col in feature_cols:
                # Categorical features
                if (df[col].dtype == 'object'
                   or df[col].dtype.name == 'category'):
                    unique_vals = (df[col].dropna()
                                      .unique())
                    if len(unique_vals) > 0:
                        sample[col] = np.random.choice(
                                          unique_vals
                                       )
                    else:
                        sample[col] = None
                # Numeric features
                elif np.issubdtype(df[col].dtype,
                                   np.number):
                    sample[col] = np.random.randint(
                        neg_feature_range[0],
                        neg_feature_range[1] + 1
                    )
                else:
                    sample[col] = None

            negative_samples.append(sample)

    df_negative = pd.DataFrame(negative_samples)

    # Label positive set and combine
    df_positive = df.copy()
    df_positive[target_col] = 1

    combined = pd.concat([df_positive, df_negative],
                         ignore_index=True)

    if verbose >= 2:
        print(f"\n[INFO] Final dataset: {len(df_positive)} "
              f"positive and {len(df_negative)} negative "
              f"samples.")


    # Optional View/Plot
    if view:
        if (use_gpd == 'auto') and HAS_GPD:
            if verbose >= 2:
                print("[INFO] Visualizing with GeoPandas.")

            geometry = [
                Point(xy) for xy in zip(
                    combined[lon_col],
                    combined[lat_col]
                )
            ]
            gdf = gpd.GeoDataFrame(combined,
                                   geometry=geometry)

            fig, ax = plt.subplots(figsize=(10, 8))
            for label, color, name in zip(
                [1, 0],
                ['red', 'green'],
                ['Positive', 'Negative']
            ):
                gdf[gdf[target_col] == label].plot(
                    ax=ax,
                    color=color,
                    label=f"{name} Sample",
                    markersize=50
                )

            for _, row in df_positive.iterrows():
                circle = plt.Circle(
                    (row[lon_col], row[lat_col]),
                    buffer_deg,
                    color='blue',
                    alpha=0.2
                )
                ax.add_patch(circle)

        else:
            if verbose >= 2:
                print("[INFO] Visualizing with Matplotlib.")

            fig, ax = plt.subplots(figsize=(10, 8))
            for label, color, name in zip(
                [1, 0],
                ['red', 'green'],
                ['Positive', 'Negative']
            ):
                subset = combined[combined[target_col]
                                  == label]
                ax.scatter(subset[lon_col],
                           subset[lat_col],
                           color=color,
                           label=f"{name} Sample",
                           s=50)

            for _, row in df_positive.iterrows():
                circle = plt.Circle(
                    (row[lon_col], row[lat_col]),
                    buffer_deg,
                    color='blue',
                    alpha=0.2
                )
                ax.add_patch(circle)

        ax.set_title("Negative Sample Generation with "
                     "Spatial Buffers")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(handles=[
            Patch(color='red', label='Positive Samples'),
            Patch(color='green', label='Negative Samples'),
            Patch(color='blue', alpha=0.2,
                  label=f"{buffer_km} km Buffer")
        ])
        ax.grid (True, **{"linestyle":":", "alpha": 0.7})
        plt.tight_layout()
        plt.show()

    return combined

def _validate_negative_sampling(
    df: DataFrame,
    target_col: str,
    spatial_cols: Tuple[str, str] = ('longitude', 'latitude'),
    feature_cols: Optional[List[str]] = None,
    neg_feature_range: Tuple[float, float] = (0, 5),
    num_neg_per_pos: int = 1,
    id_col: Optional [str]="auto",
    verbose: int=0
):
    """
    Helper that validates input parameters and column configurations
    for the generation of negative samples. Ensures required columns exist,
    processes feature and spatial columns, and checks numeric parameters
    like `neg_feature_range` and `num_neg_per_pos`.
    """
    # Validate input and columns
    exist_features(df,features=target_col,
                   name=f"Target '{target_col}'")

    feature_cols = columns_manager(feature_cols)
    spatial_cols = columns_manager(spatial_cols)
 
    check_spatial_columns(df, spatial_cols=spatial_cols)
    exist_features(df,features=spatial_cols,
                   name="Spatial columns")
    neg_feature_range= validate_length_range(
        neg_feature_range, 
        param_name="neg_feature_range"
    )
    num_neg_per_pos= validate_positive_integer(
        num_neg_per_pos, 
        "num_neg_per_pos", 
    )
    # detect id columns if auto 
    if str(id_col).lower() =='auto': 
        id_col = find_id_column(
            df, strategy= 'naive', 
            errors='ignore'
        )
    
    id_col= columns_manager( id_col, empty_as_none= False ) 
    # If no feature_cols provided, auto-detect from remaining columns
    # including 
    if feature_cols is None:
        excluded_cols = list(spatial_cols) + [target_col] + id_col
        feature_cols = [
            col for col in df.columns
            if col not in excluded_cols
        ]
        if verbose >= 2:
            print(f"[INFO] Auto-detected feature_cols:"
                  f" {feature_cols}")

    feature_cols = columns_manager(
        feature_cols,
        empty_as_none=False
    )

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Please "
                         "specify `feature_cols` explicitly.")

    # Warn if a feature column does not exist in df
    for col in feature_cols:
        if col not in df.columns and verbose >= 2:
            print(f"[WARN] Feature column '{col}' not "
                  "found in df; it will be simulated.")
            
    return ( 
        spatial_cols, 
        feature_cols,
        neg_feature_range, 
        num_neg_per_pos, 
     )

@SaveFile 
@isdf 
def create_spatial_clusters(
    df: pd.DataFrame,
    spatial_cols: Optional[List[str]] = None ,
    cluster_col: str = 'region',
    n_clusters: Optional[int] = None,
    algorithm: str = 'kmeans',
    view: bool = True,
    figsize: tuple = (14, 10),
    s: int=60, 
    plot_style: str = 'seaborn',
    cmap: str = 'tab20',
    show_grid: bool=True, 
    grid_props: dict =None, 
    auto_scale: bool = True,
    savefile: Optional[str]=None, 
    verbose: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Cluster 2D spatial data in ``df`` using `<algorithm>`
    and optionally plot the results.

    This function, `<create_spatial_clusters>`, extracts
    two coordinate columns from `<df>` to form clusters
    via methods such as 'kmeans', 'dbscan', or 'agglo'
    (agglomerative). It uses the function
    `filter_valid_kwargs` (when relevant) to strip out
    invalid parameters for certain estimators, and
    writes cluster labels into `<cluster_col>`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame holding spatial coordinates
        and optional other fields.
    spatial_cols : list of str, optional
        Two-column list for x and y coordinates.
        Defaults to ``['longitude','latitude']`` if
        None.
    cluster_col : str, default='region'
        Name of the column to store the assigned
        cluster labels.
    n_clusters : int, optional
        Number of clusters to form. If not provided
        for KMeans, it is auto-detected. For DBSCAN
        or Agglomerative, a warning is issued if not
        set.
    algorithm : str, default='kmeans'
        Choice of clustering algorithm among
        ``['kmeans','dbscan','agglo']``.
    view : bool, default=True
        If True, displays a scatterplot of the final
        clusters.
    figsize : tuple, default=(14, 10)
        Size of the displayed figure for the
        cluster plot.
    s : int, default=60
        Marker size in the scatterplot.
    plot_style : str, default='seaborn'
        Matplotlib style used for the plot.
    cmap : str, default='tab20'
        Colormap name used to differentiate clusters.
    show_grid : bool, default=True
        Toggles grid lines on or off.
    grid_props : dict, optional
        Additional keyword arguments controlling
        the grid style.
    auto_scale : bool, default=True
        If True, standardize coordinates before
        clustering.
    savefile : str, optional
        File path to save the data with an additional
        `<cluster_col>` storing the assigned
        cluster labels if desired.
    verbose : int, default=1
        Controls console logs. Higher values yield
        more details about scaling and cluster
        detection.
    **kwargs
        Additional keyword arguments passed to the
        chosen algorithm (filtered by
        `filter_valid_kwargs` for KMeans, DBSCAN,
        AgglomerativeClustering ).

    Returns
    -------
    pandas.DataFrame
        A copy of `<df>` with an additional
        `<cluster_col>` storing the assigned
        cluster labels.

    Notes
    -----
    If `<auto_scale>` is True, it uses a standard
    scaler to normalize the coordinate columns. The
    scatterplot is generated using the library
    seaborn for enhanced styling.
    
    By default, for `<algorithm>` = "kmeans", the model
    attempts to minimize:

    .. math::
       J = \\sum_{i=1}^{N} \\min_{\\mu_j} \\lVert x_i
       - \\mu_j \\rVert^2

    where :math:`x_i` are the scaled or raw 2D
    coordinates in `<df>`. The function can optionally
    auto-detect ``n_clusters`` using a silhouette and
    elbow analysis if not provided.

    Examples
    --------
    >>> from fusionlab.utils.spatial_utils import create_spatial_clusters
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "longitude": [0.1, 0.2, 2.2, 2.3],
    ...     "latitude": [1.0, 1.1, 2.1, 2.2]
    ... })
    >>> # KMeans with auto scale and auto-detect k
    >>> result = create_spatial_clusters(
    ...     df=df,
    ...     algorithm="kmeans",
    ...     view=True
    ... )
    >>> # DBSCAN with custom arguments
    >>> result_db = create_spatial_clusters(
    ...     df=df,
    ...     algorithm="dbscan",
    ...     eps=0.5,
    ...     min_samples=2
    ... )

    See Also
    --------
    filter_valid_kwargs : Helps discard unsupported
        keyword arguments for chosen estimators.

    References
    ----------
    .. [1] Pedregosa et al. *Scikit-learn:
       Machine Learning in Python*, JMLR 12,
       pp. 2825-2830, 2011.
    """
    # Confirm required columns exist in DataFrame
    # This prevents missing key data issues
    if spatial_cols is None: 
        spatial_cols = ['longitude', 'latitude']
        
    assert all(col in df.columns for col in spatial_cols), (
        "Missing spatial columns"
    )
    assert len(spatial_cols) == 2, (
        f"Need exactly 2 spatial columns. Got {len(spatial_cols)}"
    )
    assert algorithm in ['kmeans', 'dbscan', 'agglo'], (
        "Invalid algorithm. Expect one of ['kmeans', 'dbscan', 'agglo']."
    )

    # Use requested plotting style
    plt.style.use(plot_style)

    # Copy DataFrame to avoid modification
    local_df = df.copy()

    # Extract coordinates from the spatial columns
    coords = local_df[spatial_cols].values

    # Debug info about data shape if verbosity is high
    if verbose >= 2:
        print(f"DataFrame shape: {local_df.shape}")
        print("Initial coords sample:", coords[:5])

    # Scale coordinates to standardize range, if requested
    if auto_scale:
        if verbose >= 1:
            print("Scaling coordinates...")
        scaler = StandardScaler()
        coords = scaler.fit_transform(coords)

        # Debug info about scaled coords if verbosity is high
        if verbose >= 2:
            print("Scaled coords sample:", coords[:5])

    # Determine an optimal number of clusters if none provided
    if n_clusters is None:
        if algorithm == 'kmeans':
            if verbose >= 1:
                print("Auto-detecting optimal number of clusters...")
            n_clusters = _auto_detect_k(
                coords=coords,
                verbose=verbose,
                show_grid=show_grid, 
                grid_props=grid_props 
            )
        else:
            # Warn if user expects auto-detection with non-kmeans
            warnings.warn(
                "Auto-cluster detection only supported for KMeans"
            )

    # Notify user about the clustering approach
    if verbose >= 1:
        print(f"Clustering with {algorithm.upper()}...")

    # Initialize the selected clustering algorithm
    if algorithm == 'kmeans':
        kwargs = filter_valid_kwargs(KMeans, kwargs)
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            **kwargs
        )
    elif algorithm == 'dbscan':
        kwargs = filter_valid_kwargs (DBSCAN, kwargs)
        clusterer = DBSCAN(**kwargs)
    else:  # algorithm == 'agglo'
        kwargs = filter_valid_kwargs (AgglomerativeClustering, kwargs)
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            **kwargs
        )

    # Fit the model and predict cluster labels
    labels = clusterer.fit_predict(coords)
    local_df[cluster_col] = labels

    # Debug info about clustering output if verbosity is high
    if verbose >= 2:
        unique_labels = np.unique(labels)
        print(f"Unique cluster labels: {unique_labels}")

    # Plot the clusters if requested
    if view:
        _plot_clusters(
            df=local_df,
            spatial_cols=spatial_cols,
            cluster_col=cluster_col,
            figsize=figsize,
            cmap=cmap,
            algorithm=algorithm, 
            s=s, 
            show_grid=show_grid, 
            grid_props=grid_props, 
        )

    # Return the DataFrame with assigned cluster labels
    return local_df

def _auto_detect_k(
    coords: np.ndarray,
    verbose: int,
    max_k: int = 10, 
    show_grid: bool=True, 
    grid_props: dict=None, 
) -> int:
    # Evaluate multiple k values using
    # elbow (distortion) and silhouette scores
    distortions = []
    silhouettes = []
    K_range = range(2, max_k + 1)

    if verbose >= 1:
        print(f"Evaluating k from 2 to {max_k}...")

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(coords)
        distortions.append(kmeans.inertia_)
        silhouettes.append(
            silhouette_score(coords, kmeans.labels_)
        )

        # Detailed iteration log if verbosity is high
        if verbose >= 2:
            output = "k={0:<5} | Distortion={1:^28} | Silhouette={2:^20}".format(
                k, f'{distortions[-1]:.3f}', f'{silhouettes[-1]:.3f}')
            print(output)
            
    # Plot elbow and silhouette analyses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(K_range, distortions, 'bo-')
    ax1.set(
        xlabel='Number of clusters',
        ylabel='Distortion',
        title='Elbow Method'
    )

    ax2.plot(K_range, silhouettes, 'go-')
    ax2.set(
        xlabel='Number of clusters',
        ylabel='Silhouette Score',
        title='Silhouette Analysis'
    )

    # Enable gridlines
    if grid_props is None: 
        grid_props= {"linestyle": ':', 'alpha': .7}
        
    if show_grid: 
        ax1.grid(show_grid, **grid_props)
    else: 
        ax1.grid(show_grid)
        
    if show_grid: 
        ax2.grid(show_grid, **grid_props)
    else: 
        ax2.grid(show_grid)
        
    plt.tight_layout()
    plt.show()

    # Use silhouette score peak to suggest optimal k
    optimal_k = np.argmax(silhouettes) + 2
    if verbose >= 1:
        print(f"Suggested optimal k: {optimal_k}")

    return optimal_k

def _plot_clusters(
    df: pd.DataFrame,
    spatial_cols: List[str],
    cluster_col: str,
    figsize: tuple,
    cmap: str,
    s: int, 
    algorithm: str, 
    show_grid: bool=True, 
    grid_props: dict=None, 
) -> None:
    # Create a scatterplot to visualize clustered data
    plt.figure(figsize=figsize)

    ax = sns.scatterplot(
        x=spatial_cols[0],
        y=spatial_cols[1],
        hue=cluster_col,
        palette=cmap,
        data=df,
        s=s,
        edgecolor='k',
        linewidth=0.5,
        alpha=0.8,
        legend='auto'
    )

    # Professional labeling and presentation
    plt.title(
        f"{algorithm.upper()} Clustering - "
        f"{df[cluster_col].nunique()} Clusters",
        fontsize=14, pad=20
    )
    plt.xlabel(spatial_cols[0], fontsize=12)
    plt.ylabel(spatial_cols[1], fontsize=12)

    # Enable gridlines
    if show_grid: 
        plt.grid(show_grid, **( grid_props or {"linestyle": ':', 'alpha': .7}))
    else: 
        plt.grid(show_grid)

    # Adjust and position legend
    plt.legend(
        title='Cluster',
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    # Annotate cluster labels near their median positions
    for cluster in df[cluster_col].unique():
        # Skip noise cluster (-1) for DBSCAN
        if cluster == -1:
            continue
        median_position = df[df[cluster_col] == cluster][
            spatial_cols
        ].median()
        plt.text(
            median_position[0],
            median_position[1],
            str(cluster),
            fontdict=dict(weight='bold', size=10),
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='none'
            )
        )

    # Hide top/right spines, tighten layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.figure.tight_layout()

    # Display the final clustering plot
    plt.show()
    
    
@validate_params ({ 
    'threshold':[Interval(Real, 0, 1, closed='neither')], 
    'error': [StrOptions({'raise', 'warn', 'ignore'})], 
    'pos': [Real, 'array-like']
    })
@isdf
def filter_position(
    df,
    pos,
    pos_cols=None,
    find_closest=True,
    threshold=0.01,
    error='raise',
    verbose=0
   ):
    """
    filter_position is a utility that filters a
    pandas.DataFrame based on user-specified spatial
    positions. It can match positions exactly or compute
    distances to find the closest points within a threshold.
    
    For a single dimension, the distance is computed as:
    
    .. math::
       d = |x - p|
    
    For multi-dimensional data with n coordinates, the
    Euclidean distance is computed as:
    
    .. math::
       d = \\sqrt{\\sum_{i=1}^n (x_i - p_i)^2}
    
    Parameters
    ------------
    df : pandas.DataFrame
        The DataFrame that will be filtered. This parameter
        is essential and must contain columns referenced by
        `pos_cols` if ``pos_cols`` is not None.
    
    pos : float or tuple of floats
        The reference position(s) to match or approximate.
        When `pos_cols` is None, `pos` is interpreted as an
        index value. Otherwise, each value in `pos` aligns
        with a specific column in `pos_cols`.
    
    pos_cols : str or tuple of str, optional
        Name(s) of the column(s) in `df` to match against
        `pos`. If ``pos_cols=None``, then `pos` is treated
        as an index filter. If multiple columns are given
        (e.g., latitude and longitude), each coordinate in
        `pos` should correspond to one column in `pos_cols`.
    
    find_closest : bool, optional
        If True, nearest-neighbor filtering is performed
        within the distance `threshold`. If False, exact
        matches are used.
    
    threshold : float, optional
        The maximum distance within which points are
        considered a match if `find_closest` is True. The
        unit corresponds to the column data (e.g., degrees
        for geographic lat/lon).
    
    error : {'raise', 'warn', 'ignore'}, optional
        Specifies how to handle dimension mismatches or
        missing values. If ``'raise'``, a ValueError will be
        raised. If ``'warn'``, a warning is printed and extra
        values are ignored. If ``'ignore'``, mismatches are
        silently ignored.
    
    verbose : int, optional
        Controls the level of output messages:
        - 0: No output
        - 1: Basic info
        - 2: Additional details
        - >=3: Comprehensive summary
    
    Returns
    -------
    pandas.DataFrame
        A new DataFrame that contains only rows matching or
        approximating the specified position(s) within the
        given threshold if `find_closest` is True.
    
    Notes
    -----
    When `pos_cols` is None, the function attempts to filter
    by DataFrame index using the first element of `pos`. This
    approach may fail for multi-level indexes unless
    ``error='warn'`` or ``error='ignore'`` is used to bypass
    the dimension mismatch. See [1]_ for further discussion
    on multi-dimensional indexing.
    
    Examples
    --------
    >>> from fusionlab.utils.spatial_utils import filter_position
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'lat': [113.309998, 113.310001],
    ...     'lon': [22.831362, 22.831364]
    ... })
    >>> # Exact match
    >>> result_exact = filter_position(df, pos=(113.309998,
    ...                                         22.831362),
    ...                                pos_cols=('lat', 'lon'),
    ...                                find_closest=False)
    >>> # Nearest match with threshold
    >>> result_close = filter_position(df,
    ...                                pos=(113.31,
    ...                                     22.83),
    ...                                pos_cols=('lat',
    ...                                          'lon'),
    ...                                find_closest=True,
    ...                                threshold=0.01)
    
    See Also
    --------
    fusionlab.utils.data_utils.truncate_data: 
        Truncate multiple DataFrames based on spatial 
        coordinates or index alignment with a base DataFrame.
    
    References
    ----------
    .. [1] Smith, J., & Doe, A. (2020). Multi-dimensional
       indexing in big data. Journal of Spatial Computing,
       15(3), 200-210.
    """
    # Initialize filtered_df as the original DataFrame
    filtered_df = df.copy()

    # Convert pos to tuple if it's not already
    if not isinstance(pos, (tuple, list)):
        pos = (pos,)

    # Helper function to handle errors
    def _handle_error(msg, ex_msg=''):
        if error == 'raise':
            raise ValueError(msg)
        elif error == 'warn':
            warnings.warn(f"{msg}{ex_msg}")
        # if error == 'ignore', do nothing

    # If no position columns specified,
    # treat pos as DataFrame index
    if pos_cols is None:
        if len(pos) > 1:
            _handle_error(
                "Multiple position values provided but no pos_cols. "
                "Unable to match multi-level index. ",
                "Using the first position value only."
            )
        idx = pos[0]  # Consider only the first if multiple are given
        if idx not in df.index:
            _handle_error(
                f"Index '{idx}' not found in DataFrame. ",
                "Returning empty DataFrame."
            )
            return filtered_df.iloc[0:0]  # Return empty DF if not found
        # Filter by index
        filtered_df = filtered_df.loc[[idx]]
        return filtered_df

    # Ensure pos_cols is tuple for consistency
    if not isinstance(pos_cols, (tuple, list)):
        pos_cols = (pos_cols,)
    
    pos_cols = columns_manager(pos_cols) 
    
    # Check dimension match between pos and pos_cols
    if len(pos) != len(pos_cols):
        # If mismatch, handle based on error param
        _handle_error(
            f"pos ({len(pos)}) and pos_cols ({len(pos_cols)}) lengths "
            "do not match.", " Extra values or columns will be ignored."
        )
        # If ignoring or warning, align up to the shortest length
        min_len = min(len(pos), len(pos_cols))
        pos = pos[:min_len]
        pos_cols = pos_cols[:min_len]

    # If find_closest is disabled, do exact matching
    if not find_closest:
        # Build query for exact match
        mask = True
        for p_val, col in zip(pos, pos_cols):
            mask &= (filtered_df[col] == p_val)
        filtered_df = filtered_df[mask]
        return filtered_df

    # If find_closest is enabled, use threshold-based filtering
    # Compute distance (Euclidean if multiple columns, absolute if single)
    if len(pos_cols) == 1:
        # Single dimension: absolute difference
        col = pos_cols[0]
        distance = (filtered_df[col] - pos[0]).abs()
        filtered_df = filtered_df[distance <= threshold]
    else:
        # Multi-dimensional: Euclidean distance
        # Summation of squared diffs for each column
        squared_diff = 0
        for p_val, col in zip(pos, pos_cols):
            squared_diff += (filtered_df[col] - p_val) ** 2
        distance = squared_diff ** 0.5
        filtered_df = filtered_df[distance <= threshold]

    # If verbose, provide basic info
    if verbose >= 1:
        print(f"Filtered {len(df)} rows to {len(filtered_df)} rows "
              f"within threshold {threshold}.")

    # If verbose >= 2, show sample of distance
    if verbose >= 2 and not filtered_df.empty:
        print("Example distance values within threshold:")
        print(distance[distance <= threshold].head())

    return filtered_df

@is_data_readable(data_to_read='data')
@SaveFile
@validate_params ({ 
    'z': ['array-like', str], 
    'threshold': ['array-like', StrOptions({'auto'}), Real], 
    'condition': [StrOptions({'auto', 'above', 'below', 'between'})], 
    'use_negative_criteria': [bool], 
    'percentile': [Real], 
    'x': ['array-like', str, None], 
    'y': ['array-like', str, None], 
    'data': ['array-like', None], 
    })
def extract_zones_from(
    z,
    threshold='auto',        
    condition='auto',        
    use_negative_criteria=True,  
    percentile=10,           
    x=None,                  
    y=None,                 
    data=None,              
    view=False,              
    plot_type='scatter',    
    figsize=(8, 6),  
    savefile=None,  
    axis_off=False, 
    show_grid=True,        
    **kwargs                 
):
    r"""
    Extracts specific zones by filtering an input array or arrays
    using a threshold criterion. This function applies a logical
    mask to the values and retains those which satisfy a chosen
    condition (e.g. ``'above'``, ``'below'``, or ``'between'`` a
    specific threshold or thresholds). The threshold can be
    automatically derived using percentiles if ``'auto'`` is
    specified.

    .. math::
       \text{mask}(z) \;=\;
       \begin{cases}
         1 & \text{if } z \,\in\, \Omega \\
         0 & \text{otherwise}
       \end{cases}

    where :math:`\Omega` is the region of acceptance determined
    by the threshold mechanism.

    Parameters
    ----------
    z : array-like, Series, or string
        The input data to be filtered. If <z> is a string,
        it is treated as a column name from the provided
        <data> (see below). If it is an array or Series, it
        is used directly.
    threshold : { ``'auto'``, float, int, tuple }
        The criterion for filtering. If ``'auto'``, the
        function computes a percentile-based threshold
        driven by <percentile>. If a float or int is
        given, the function will filter values above or
        below that single threshold. If a tuple of length 2
        is provided, the function will filter between those
        bounds.
    condition : { ``'auto'``, ``'above'``, ``'below'``,
                     ``'between'`` }
        Defines how the filter is applied relative to the
        given or derived threshold. If ``'auto'``, the
        function decides based on <use_negative_criteria>.
        If ``'above'``, all values satisfying
        :math:`z > \text{threshold}` are kept. If
        ``'below'``, then :math:`z < \text{threshold}`
        are retained. If ``'between'``, a range
        (low, high) is respected.
    use_negative_criteria : bool
        When ``True``, automatically interprets
        ``'auto'`` condition as filtering from below, akin
        to negative-based risk. When ``False``, filters
        from above.
    percentile : int or float
        Used only if ``threshold='auto'``. Determines which
        percentile is used to compute the threshold. For
        example, if <use_negative_criteria> is ``True``,
        the <percentile>th percentile is chosen, otherwise
        the (100 - <percentile>)th percentile is used.
    x : array-like, Series, or string, optional
        The x-axis data. If <x> is a string and <data> is
        provided, the function extracts the relevant
        column. If <x> is an array or Series, it is used
        directly.
    y : array-like, Series, or string, optional
        Similar to <x>, representing the y-axis data.
    data : pandas.DataFrame, optional
        The DataFrame source if <x>, <y>, or <z> are
        provided as strings referencing column names.
    view : bool
        If ``True``, displays a plot of the filtered data.
    plot_type : { ``'scatter'``, ``'line'``, ``'hist'``,
                      ... }
        Determines how the data are visualized when <view>
        is ``True``.
    figsize : tuple of int
        The size of the generated figure for plotting. E.g.
        ``(8,6)`` is typical.
    axis_off: bool, default=False 
        Remove the axis if set to ``True``. 
    show_grid: bool, default=True 
        Display the plot grid or make it invisible if ``False``. 

    Methods
    -------
    The function itself does not expose methods starting with
    letters (excluding `_`), as it is a single operation. All
    steps are executed internally with no additional public
    methods.

    Notes
    ------
    .. math::
       \mathbf{z}_{\text{filtered}}
       \;=\; \{ z_i \mid \text{condition}(z_i) \}

    where :math:`\text{condition}(z_i)` is derived from
    ``threshold`` and ``condition``. For instance, if
    ``condition='below'`` and :math:`\tau =
    \text{threshold}`, then

    .. math::
       \text{condition}(z_i)
       \;=\; [\, z_i < \tau \,].
       
    This function relies on ``extract_array_from`` (from
    the fusionlab.core.array_manager) if <z>, <x>, or <y>
    are passed as strings and a <data> DataFrame is
    supplied. The user has the option to visualize the
    retained data points by enabling <view> and
    customizing <plot_type>.
    
    Examples
    --------
    >>> from fusionlab.utils.spatial_utils import extract_zones_from
    >>> import numpy as np
    >>> z_data = np.array([0, 2, 5, 10, 15, 20])
    >>> result = extract_zones_from(z=z_data, threshold=10,
    ...                        condition='above')
    >>> print(result)

    See Also
    --------
    fusionlab.core.array_manager.extract_array_from`` :
        The array extraction utility used for retrieving
        arrays from DataFrame columns.

    References
    ----------
    .. [1] Smith, J. & Doe, A. "Advanced Filtering
       Techniques", Journal of Data Science, 2022.
    .. [2] Brown, K. "Data Visualization Best
       Practices", Data Analytics Press, 2021.
    """

    # 1) If z, x, y are strings, extract from DataFrame
    if data is not None:
        z, x, y = extract_array_from(
            data,
            z, x, y,
            handle_unknown='raise',  # raise if columns not found
            error='raise',
            check_size=True,         # ensure x,y,z have same length
        )

    # 2) Ensure z, x, y are arrays
    z = np.asarray(z) if not isinstance(z, np.ndarray) else z
    x = np.asarray(x) if (x is not None and not isinstance(x, np.ndarray)) else x
    y = np.asarray(y) if (y is not None and not isinstance(y, np.ndarray)) else y

    # 3) If threshold='auto', compute it using percentiles
    #    and decide condition based on use_negative_criteria
    if isinstance(threshold, str) and threshold.lower() == 'auto':
        if use_negative_criteria:
            thr_value = np.percentile(z, percentile)  # e.g., lower percentile
            condition = 'below'
        else:
            thr_value = np.percentile(z, 100 - percentile)
            condition = 'above'
        threshold = thr_value

    # 4) Build a mask depending on single-value or range threshold
    if isinstance(threshold, (tuple, list)) and len(threshold) == 2:
        threshold = validate_length_range(threshold, param_name ='Threshold')
        
        if condition == 'auto':
            condition = 'between'  # default if user didn't specify
        if condition.lower() == 'between':
            mask = (z >= threshold[0]) & (z <= threshold[1])
        else:
            raise ValueError(
                "For tuple/list threshold, specify condition='between'"
                " or handle logic manually."
            )
    else:
        # Single numeric threshold
        if condition == 'auto':
            condition = 'below' if use_negative_criteria else 'above'
        if condition.lower() == 'above':
            mask = (z > threshold)
        elif condition.lower() == 'below':
            mask = (z < threshold)
        else:
            raise ValueError(
                f"condition={condition} not valid for a single numeric threshold."
            )

    # 5) Filter the arrays
    z_filtered = z[mask]
    x_filtered = x[mask] if x is not None else None
    y_filtered = y[mask] if y is not None else None

    # 6) Prepare the output
    #    If x & y exist, return a DataFrame with columns [x, y, z].
    #    If only x, return [x, z].
    #    Otherwise, return just the filtered z as a Series.
    x_name = resolve_label(x, default_name ='x')
    y_name = resolve_label(y, default_name ='y')
    z_name = resolve_label(z, default_name ='z')
    if x_filtered is not None and y_filtered is not None:
        result = pd.DataFrame({x_name: x_filtered, y_name: y_filtered, z_name: z_filtered})
    elif x_filtered is not None:
        result = pd.DataFrame({x_name: x_filtered, z_name: z_filtered})
    else:
        result = pd.Series(z_filtered)

    # 7) If view=True, build a plot according to plot_type and data availability
    if view:
        plt.figure(figsize=figsize)
        #  - If x & y provided, do a 2D scatter of x vs y (colored by z or sized by z).
        #  - If only x provided, do 1D plot with z as the values.
        #  - If neither x nor y, do a histogram of z.
        if x_filtered is not None and y_filtered is not None:
            if plot_type == 'scatter':
                plt.scatter(x_filtered, y_filtered, c=z_filtered, **kwargs)
                plt.colorbar(label=z_name)
                plt.xlabel(x_name)
                plt.ylabel(y_name)
                plt.title('Filtered Scatter Plot')
            elif plot_type == 'line':
                # Plot lines in x-y plane, ignoring z
                plt.plot(x_filtered, y_filtered, **kwargs)
                plt.xlabel(x_name)
                plt.ylabel(y_name)
                plt.title('Filtered Line Plot')
            else:
                # fallback
                plt.scatter(x_filtered, y_filtered, **kwargs)
                plt.xlabel(x_name)
                plt.ylabel(y_name)
                plt.title(f'Fallback to Scatter: {plot_type}')
        elif x_filtered is not None: # only x is given
            if plot_type == 'scatter':
                plt.scatter(x_filtered, z_filtered, **kwargs)
                plt.xlabel(x_name)
                plt.ylabel(z_name)
                plt.title('Filtered Scatter (x vs z)')
            elif plot_type == 'line':
                plt.plot(x_filtered, z_filtered, **kwargs)
                plt.xlabel(x_name)
                plt.ylabel(z_name)
                plt.title('Filtered Line (x vs z)')
            elif plot_type == 'hist':
                plt.hist(z_filtered, **kwargs)
                plt.xlabel(z_name)
                plt.title('Filtered Histogram (z)')
            else:
                # fallback
                plt.scatter(x_filtered, z_filtered, **kwargs)
                plt.xlabel(x_name)
                plt.ylabel(z_name)
                plt.title(f'Fallback to Scatter: {plot_type}')
        else:
            # No x or y => just do a histogram of z
            plt.hist(z_filtered, **kwargs)
            plt.xlabel(z_name)
            plt.title('Filtered Histogram (z)')
            
        if axis_off : 
            plt.axis('off')
        if not show_grid: 
            plt.grid(False )
            
        plt.tight_layout()
        plt.show()

    return result

@SaveFile            
def dual_merge(
    df1: pd.DataFrame, 
    df2: pd.DataFrame,
    feature_cols: Union[list, tuple] = ('longitude', 'latitude'),
    find_closest: bool = False, 
    force_coords: bool = False,  
    threshold: float = 0.01,  
    how: str = 'inner', 
    savefile: Optional[str]=None, 
) -> pd.DataFrame:
    """
    Merge two DataFrames based on specified feature columns. The function 
    can match the features exactly or find the closest matches within a 
    specified distance threshold. It also allows for overwriting coordinates 
    or feature values from one DataFrame to another when a close match is found.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame to be merged. It contains the primary data 
        along with the feature columns (e.g., longitude, latitude) to be 
        merged on.

    df2 : pd.DataFrame
        The second DataFrame to be merged. It contains the data to be 
        matched with `df1` based on the specified feature columns.

    feature_cols : tuple or list, default ``('longitude', 'latitude')``
        The names of the columns in each DataFrame to merge on. It should 
        contain two columns representing features, such as coordinates 
        (longitude, latitude) or other relevant attributes. These columns 
        will be used to match the rows from `df1` to `df2`.

    find_closest : bool, default ``False``
        If ``True``, the function will attempt to find the closest points 
        in ``df2`` for each point in ``df1`` within the specified distance 
        threshold (`threshold`). If no exact match is found, the closest 
        point within the threshold will be considered.

    force_coords : bool, default ``False``
        If ``True``, when the closest points are found, the coordinates 
        of `df1` will overwrite those of `df2` for the matched points.

    threshold : float, default ``0.01``
        The maximum distance threshold within which points will be considered 
        as "close" for the closest point matching. The value is in the same 
        unit as the feature columns (e.g., degrees for latitude/longitude).

    how : str, default ``'inner'``
        The type of merge to perform. Options include:
        - ``'inner'``: Only includes points that appear in both DataFrames.
        - ``'left'``: All points from `df1` are included; unmatched points 
          from `df2` are excluded.
        - ``'right'``: All points from `df2` are included; unmatched points 
          from `df1` are excluded.
        - ``'outer'``: Includes all points from both DataFrames, with NaN 
          for unmatched points.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the merged data based on the specified 
        feature columns and merge type. If ``find_closest=True``, it will 
        contain the closest matches within the specified threshold, with 
        coordinates from ``df1`` overwritten if ``force_coordinates=True``.

    Notes
    -----
    - This function uses a KDTree for efficient nearest-neighbor searching 
      when ``find_closest=True``. This is useful when dealing with large 
      datasets that may not have exact coordinate matches.
    - When ``force_coordinates=True``, the coordinates from ``df1`` will 
      overwrite those from ``df2`` for the closest points. However, other 
      feature values will be kept from ``df2``.

    Examples
    --------
    >>> import pandas as pd
    >>> from fusionlab.utils.datautils import dual_merge
    >>> df1 = pd.DataFrame({
    >>>     'longitude': [1.1, 1.2, 1.3],
    >>>     'latitude': [2.1, 2.2, 2.3],
    >>>     'value1': [10, 20, 30]
    >>> })
    >>> df2 = pd.DataFrame({
    >>>     'longitude': [1.1, 1.4],
    >>>     'latitude': [2.1, 2.4],
    >>>     'value2': [100, 200]
    >>> })
    >>> result = dual_merge(df1, df2, feature_cols=('longitude', 'latitude'), 
    >>>                     find_closest=True, threshold=0.05)
    >>> print(result)
       longitude  latitude  value1  value2
    0        1.1       2.1      10     100

    See Also
    --------
    scipy.spatial.cKDTree: Used for finding the closest points in `df2` 
        when ``find_closest=True``. 
    pandas.merge: pandas.DataFrame.merge
        The pandas merge function, used to merge DataFrames based on columns.

    References
    ----------
    .. [1] Scipy Documentation, cKDTree
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
    """

    # Ensure feature_cols are valid and both contain two elements
    feature_cols = columns_manager(feature_cols , empty_as_none= False )
    if len(feature_cols) != 2:
        raise ValueError(
            "feature_cols must contain exactly two features (e.g., longitude, latitude)")
    
    # check wether df1 and df2  are both dataframes.
    are_all_frames_valid(df1, df2 )
    
    # Extract columns from feature_cols for both DataFrames
    feature1_1, feature1_2 = feature_cols
    feature2_1, feature2_2 = feature_cols
    
    # Filter for relevant columns in both DataFrames
    df1_coords = df1[[feature1_1, feature1_2]]
    df2_coords = df2[[feature2_1, feature2_2]]

    if find_closest:
        # Use KDTree for fast nearest-neighbor search
        tree = cKDTree(df2_coords.values)
        
        # Query for the closest points in df2 for each point in df1
        dist, indices = tree.query(
            df1_coords.values, distance_upper_bound=threshold)
        
        # Filter out points that couldn't find a close match
        valid_idx = dist != np.inf
        df1_coords_closest = df1_coords.iloc[valid_idx]
        df2_coords_closest = df2_coords.iloc[indices[valid_idx]]

        if force_coords:
            # Force coordinates of df1 to overwrite df2
            df2_coords_closest[feature2_1] = df1_coords_closest[feature1_1].values
            df2_coords_closest[feature2_2] = df1_coords_closest[feature1_2].values
        
        # Update df1 with the closest matches from df2
        df1 = df1.iloc[valid_idx]
        df2 = df2.iloc[indices[valid_idx]]

    # Perform the merge based on the feature columns
    merged_data = pd.merge(
        df1,
        df2,
        how=how,
        left_on=[feature1_1, feature1_2],
        right_on=[feature2_1, feature2_2]
    )

    return merged_data

@isdf 
def extract_coordinates(
    df: pd.DataFrame,
    as_frame: bool = False,
    drop_xy: bool = False,
    error: Union[bool, str] = 'raise',
    verbose: int = 0
) -> Tuple[Union[Tuple[float, float], pd.DataFrame, None], pd.DataFrame, Tuple[str, str]]:
    """
    Identifies coordinate columns (longitude/latitude or easting/northing) 
    in a DataFrame, returns the coordinates or their central values, and 
    optionally removes the coordinate columns from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame expected to contain coordinates (`longitude` and 
        `latitude` or `easting` and `northing`). If both types are present, 
        `longitude` and `latitude` are prioritized.

    as_frame : bool, default=False
        If True, returns the coordinate columns as a DataFrame. If False, 
        computes and returns the midpoint values.

    drop_xy : bool, default=False
        If True, removes coordinate columns (`longitude`/`latitude` or 
        `easting`/`northing`) from the DataFrame after extracting them.

    error : Union[bool, str], {'raise', 'warn', 'ignore'} default='raise'
        If True, raises an error if `df` is not a DataFrame. If set to False, 
        converts errors to warnings. If set to ``"ignore"``, suppresses 
        warnings.

    verbose : int, default=0
        If greater than 0, outputs messages about coordinate detection.

    Returns
    -------
    Tuple[Union[Tuple[float, float], pd.DataFrame, None], pd.DataFrame, Tuple[str, str]]
        - A tuple containing either the midpoint (longitude, latitude) or 
          (easting, northing) if `as_frame=False` or the coordinate columns 
          as a DataFrame if `as_frame=True`.
        - The original DataFrame, optionally with coordinates removed if 
          `drop_xy=True`.
        - A tuple of detected coordinate column names, or an empty tuple if 
          none are detected.

    Notes
    -----
    - This function searches for either `longitude`/`latitude` or 
      `easting`/`northing` columns and returns them as coordinates. If both 
      are found, `longitude`/`latitude` is prioritized.
      
    - To calculate the midpoint of the coordinates, the function averages 
      the values in the columns:

      .. math::
          \text{midpoint} = \left(\frac{\text{longitude}_{min} + \text{longitude}_{max}}{2}, 
          \frac{\text{latitude}_{min} + \text{latitude}_{max}}{2}\right)

    Examples
    --------
    >>> import gofast as gf
    >>> from fusionlab.utils.spatial_utils import extract_coordinates
    >>> testdata = gf.datasets.make_erp(samples=7, seed=42, as_frame=True)

    # Extract midpoint coordinates
    >>> xy, d, xynames = extract_coordinates(testdata)
    >>> xy, xynames
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))

    # Extract coordinates as a DataFrame without removing columns
    >>> xy, d, xynames = extract_coordinates(testdata, as_frame=True)
    >>> xy.head(2)
       longitude   latitude
    0  110.485833  26.051389
    1  110.485982  26.051577

    # Drop coordinate columns from the DataFrame
    >>> xy, d, xynames = extract_coordinates(testdata, drop_xy=True)
    >>> xy, xynames
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5

    References
    ----------
    .. [1] Fotheringham, A. Stewart, *Geographically Weighted Regression: 
           The Analysis of Spatially Varying Relationships*, Wiley, 2002.

    See Also
    --------
    pd.DataFrame : Main pandas data structure for handling tabular data.
    np.nanmean : Computes the mean along specified axis, ignoring NaNs.
    """
    
    def rename_if_exists(val: str, col: pd.Index, default: str) -> pd.DataFrame:
        """Rename column in `d` if `val` is found in column names."""
        match = list(filter(lambda x: val in x.lower(), col))
        if match:
            df.rename(columns={match[0]: default}, inplace=True)
        return df

    # Validate input is a DataFrame
    if not (hasattr(df, 'columns') and hasattr(df, '__array__')):
        emsg = ("Expected a DataFrame containing coordinates (`longitude`/"
                "`latitude` or `easting`/`northing`). Got type: "
                f"{type(df).__name__!r}")
        
        error = str(error).lower().strip()
        if error == 'raise':
            raise TypeError(emsg)
        if error =='warn':
            warnings.warn(emsg)
        return None, df, ()

    # Rename columns to standardized names if they contain coordinate values
    for name, std_name in zip(['lat', 'lon', 'east', 'north'], 
                              ['latitude', 'longitude', 'easting', 'northing']):
        df = rename_if_exists(name, df.columns, std_name)

    # Check for and prioritize coordinate columns
    coord_columns = []
    for x, y in [('longitude', 'latitude'), ('easting', 'northing')]:
        if x in df.columns and y in df.columns:
            coord_columns = [x, y]
            break

    # Extract coordinates as DataFrame or midpoint
    if coord_columns:
        xy = df[coord_columns] if as_frame else tuple(
            np.nanmean(df[coord_columns].values, axis=0))
    else:
        xy = None
    
    # Drop coordinates if `drop_xy=True`
    if drop_xy and coord_columns:
        df.drop(columns=coord_columns, inplace=True)

    # Verbose messaging
    if verbose > 0:
        print("###", "No" if not coord_columns else coord_columns, "coordinates found.")
    
    return xy, df, tuple(coord_columns)

@Deprecated(reason=( 
    "This function is deprecated and will be removed in future versions. "
    "Please use `extract_coordinates` instead, which provides enhanced "
    "flexibility and robustness for coordinates extraction.")
)
@isdf 
def get_xy_coordinates(
        df, as_frame=False, drop_xy=False, raise_exception=True, verbose=0
    ):
    """Check whether the coordinate values x, y exist in the data.
    
    Parameters 
    ------------
    df: Dataframe 
       Frame that is expected to contain the longitude/latitude or 
       easting/northing coordinates.  Note if all types of coordinates are
       included in the data frame, the longitude/latitude takes the 
       priority. 
    as_frame: bool, default= False, 
       Returns the coordinates values if included in the data as a frame rather 
       than computing the middle points of the line 
    drop_xy: bool, default=False, 
       Drop the coordinates in the data and return the data transformed inplace 
       
    raise_exception: bool, default=True 
       raise error message if data is not a dataframe. If set to ``False``, 
       exception is converted to a warning instead. To mute the warning set 
       `raise_exception` to ``mute``
       
    verbose: int, default=0 
      Send message whether coordinates are detected. 
         
    returns 
    --------
    xy, d, xynames: Tuple 
      xy : tuple of float ( longitude, latitude) or (easting/northing ) 
         if `as_frame` is set to ``True``. 
      d: Dataframe transformed (coordinated removed )  or not
      xynames: str, the name of coordinates detected. 
      
    Examples 
    ----------
    >>> import gofast as gf 
    >>> from fusionlab.utils.spatial_utils import get_xy_coordinates 
    >>> testdata = gf.datasets.make_erp ( samples =7, seed =42 , as_frame=True)
    >>> xy, d, xynames = get_xy_coordinates ( testdata,  )
    >>> xy , xynames 
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))
    >>> xy, d, xynames = get_xy_coordinates ( testdata, as_frame =True  )
    >>> xy.head(2) 
        longitude   latitude        easting      northing
    0  110.485833  26.051389  448565.380621  2.881476e+06
    1  110.485982  26.051577  448580.339199  2.881497e+06
    >>> # remove longitude and  lat in data 
    >>> testdata = testdata.drop (columns =['longitude', 'latitude']) 
    >>> xy, d, xynames = get_xy_coordinates ( testdata, as_frame =True  )
    >>> xy.head(2) 
             easting      northing
    0  448565.380621  2.881476e+06
    1  448580.339199  2.881497e+06
    >>> # note testdata should be transformed inplace when drop_xy is set to True
    >>> xy, d, xynames = get_xy_coordinates ( testdata, drop_xy =True)
    >>> xy, xynames 
    ((448610.25612032827, 2881538.4380570543), ('easting', 'northing'))
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5
    >>> testdata.head(2) # coordinates are henceforth been dropped 
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5
    >>> xy, d, xynames = get_xy_coordinates ( testdata, drop_xy =True)
    >>> xy, xynames 
    (None, ())
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5

    """   
    
    def get_value_in ( val,  col , default): 
        """ Get the value in the frame columns if `val` exists in """
        x = list( filter ( lambda x: x.find (val)>=0 , col)
                   )
        if len(x) !=0: 
            # now rename col  
            df.rename (columns = {x[0]: str(default) }, inplace = True ) 
            
        return df

    if not (
            hasattr ( df, 'columns') and hasattr ( df, '__array__') 
            ) : 
        emsg = ("Expect dataframe containing coordinates longitude/latitude"
                f" or easting/northing. Got {type (df).__name__!r}")
        
        raise_exception = str(raise_exception).lower().strip() 
        if raise_exception=='true': 
            raise TypeError ( emsg )
        
        if raise_exception  not in ('mute', 'silence'):  
            warnings.warn( emsg )
       
        return df 
    
    # check whether coordinates exists in the data columns 
    for name, tname in zip ( ('lat', 'lon', 'east', 'north'), 
                     ( 'latitude', 'longitude', 'easting', 'northing')
                     ) : 
        df = get_value_in(name, col = df.columns , default = tname )
       
    # get the exist coodinates 
    coord_columns  = []
    for x, y in zip ( ( 'longitude', 'easting' ), ( 'latitude', 'northing')):
        if ( x  in df.columns and y in df.columns ): 
            coord_columns.extend  ( [x, y] )

    xy  = df[ coord_columns] if len(coord_columns)!=0 else None 

    if ( not as_frame 
        and xy is not None ) : 
        # take the middle of the line and if both types of 
        # coordinates are supplied , take longitude and latitude 
        # and drop easting and northing  
        xy = tuple ( np.nanmean ( np.array ( xy ) , axis =0 )) [:2]

    xynames = tuple ( coord_columns)[:2]
    if ( 
            drop_xy  and len( coord_columns) !=0
            ): 
        # modifie the data inplace 
        df.drop ( columns=coord_columns, inplace = True  )

    if verbose: 
        print("###", "No" if len(xynames)==0 else ( 
            tuple (xy.columns) if as_frame else xy), "coordinates found.")
        
    return  xy , df , xynames 

@is_data_readable 
@validate_params ({ 
    'data': ['array-like'], 
    'method': [StrOptions({"abs", "absolute",  "relative"}), None], 
    })
@isdf
def batch_spatial_sampling(
    data,
    sample_size=0.1,
    n_batches=10,
    stratify_by=None,
    spatial_bins=10,
    spatial_cols=None,
    method="abs", 
    min_relative_ratio=.01, 
    random_state=42, 
    verbose=1, 
):
    """
    Batch resample spatial data with stratification over spatial and
    specified columns.

    This function divides the dataset into `n_batches` batches, each
    being a stratified sample of the data. It ensures that samples in
    the first batch are not present in subsequent batches, and so on.
    This is particularly useful when dealing with very large datasets
    that cannot be processed at once, allowing for batch processing in
    machine learning algorithms.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame from which samples are to be drawn. It must
        contain the spatial coordinate columns specified in
        `spatial_cols`, and any additional columns specified in
        `stratify_by`.

    sample_size : float or int, optional
        The total number of samples to draw from `data`. If `sample_size`
        is a float between 0.0 and 1.0, it represents the fraction of the
        dataset to include in the sample (e.g., `sample_size=0.1` selects
        10% of the data). If `sample_size` is an integer, it represents
        the absolute number of samples to select. The default is ``0.1``.

    n_batches : int, optional
        The number of batches to divide the total samples into. The
        samples are divided as evenly as possible among the batches. The
        default is ``10``.

    stratify_by : str, list of str, optional
        A list of column names in `data` to use for stratification. The
        sampling will ensure that the distribution of these columns in
        each batch matches the distribution in the original dataset.

    spatial_bins : int or tuple/list of int, optional
        The number of bins to divide the spatial coordinates into for
        stratification. If an integer, the same number of bins is used
        for all spatial dimensions. If a tuple or list, its length must
        match the number of spatial columns specified in `spatial_cols`,
        and each element specifies the number of bins for that spatial
        dimension. The default is ``10``.

    spatial_cols : list or tuple of str, optional
        A list of column names in `data` representing spatial coordinates.
        The function can accept one or two columns (e.g., longitude and
        latitude). If ``None``, the function will look for columns named
        `'longitude'` and/or `'latitude'` in `data`. If only one spatial
        column is provided or found, a warning is issued, suggesting that
        providing both spatial columns is recommended for more accurate
        sampling. If more than two columns are provided, an error is
        raised.
    
    method : str, {'abs', 'relative'}, default='abs'
        Defines how the sample size is determined:
        - ``'abs'`` or ``'absolute'``: Uses a **fixed** sampling proportion
          based on `sample_size`.
        - ``'relative'``: Dynamically **scales** sampling based on dataset
          stratification, ensuring that all stratification groups receive
          a proportional sample while maintaining a minimum sampling ratio
          (controlled by `min_relative_ratio`).
        
        When ``method='relative'``, the function ensures that even small
        stratification groups receive a sufficient sample by applying
        `min_relative_ratio`.

    min_relative_ratio : float, default=0.01
        Controls the **minimum allowable fraction** of records that 
        must be sampled when ``method='relative'``.

        - Ensures that no group is **undersampled** to zero, even if
          its natural proportion in the dataset is very small.
        - Must be a value between ``0`` and ``1``.
        - The default value (``0.01``) means that at **least 1% of the
          total dataset** will be sampled from each stratification group,
          regardless of its relative size.
        
        **Example Scenarios:**
        
        - If `min_relative_ratio=0.05`, then each group **must** 
          contribute **at least 5%** of the total dataset size (if possible).
        - If a group is too small to reach this minimum, its entire
          subset is sampled instead.
        - This ensures that no group receives **less than
        ``min_relative_ratio × total samples**``.
        
    random_state : int, optional
        Controls the randomness of the sampling for reproducibility. This
        integer seed is used to initialize the random number generator.
        The default is ``42``.
    
    verbose: bool, default=False, 
       If `True`, displays a progress bar and detailed status messages
       during execution. Useful for monitoring the process, especially
       when working with large datasets.

    Returns
    -------
    batches : list of pandas.DataFrame
        A list of DataFrames, each representing a batch of the stratified
        sampled data.

    Notes
    -----
    The function performs stratified sampling based on spatial bins and
    other specified stratification columns. Spatial coordinates are
    binned using quantile-based discretization (:func:`pandas.qcut`),
    ensuring each bin has approximately the same number of observations.

    The total number of samples, :math:`n`, is divided among the batches,
    and within each batch, samples are drawn in a stratified manner. The
    sample size for each batch is calculated as:

    .. math::

        n_{\text{batch}} = \left\lfloor \frac{n}{n_{\text{batches}}} \right\rfloor

    The remaining samples are distributed among the first few batches:

    .. math::

        n_{\text{leftover}} = n \mod n_{\text{batches}}

    For each batch, the number of samples per stratification group is
    calculated based on the proportion of the group size to the remaining
    data size:

    .. math::

        n_{i} = \left\lceil \frac{N_{i}}{N_{\text{remaining}}}\\
            \times n_{\text{batch}} \right\rceil

    where:

    - :math:`N_{i}` is the size of group :math:`i`.
    - :math:`N_{\text{remaining}}` is the total number of samples
      remaining in the data.
    - :math:`n_{i}` is the number of samples to draw from group
      :math:`i`.

    After sampling, the selected samples are removed from the remaining
    data to ensure that they are not selected again in subsequent batches.

    Examples
    --------
    Examples
    --------
    **Case 1: Stratified Sampling (Using 'year' and 'geological_category')**
    
    >>> from fusionlab.utils.spatial_utils import batch_spatial_sampling
    >>> import pandas as pd
    >>> import numpy as np
    
    >>> # Create a sample dataset
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     "id": np.arange(10_000),
    ...     "longitude": np.random.uniform(-180, 180, 10_000),  # Geographic range
    ...     "latitude": np.random.uniform(-90, 90, 10_000),     # Geographic range
    ...     "year": np.random.randint(1990, 2025, 10_000),  # Temporal feature
    ...     "geological_category": np.random.choice(
    ...         ["Sedimentary", "Metamorphic", "Igneous"], 10_000
    ...     ),
    ...     "value": np.random.randn(10_000)  # Random numerical data
    ... })
    
    >>> # Perform stratified batch sampling
    >>> sampled_batches = batch_spatial_sampling(
    ...     data=df,
    ...     sample_size=0.05,  # 5% of total data
    ...     n_batches=5,
    ...     stratify_by=['year', 'geological_category'],  # Stratify by year & geology type
    ...     spatial_bins=(10, 15),
    ...     spatial_cols=['longitude', 'latitude'],
    ...     random_state=42
    ... )
    
    >>> for i, batch in enumerate(sampled_batches):
    ...     print(f"Batch {i+1}: {batch.shape}")
    
    Creating 5 stratified batches with a total of 500 samples...
    Batch Sampling Progress: 100%|██████████| 5/5 [00:01<00:00,  4.43it/s]
    Batch sampling completed. 5 batches created.
    
    **Stratified Sampling Results:**
    Batch 1: (100, 6)
    Batch 2: (100, 6)
    Batch 3: (100, 6)
    Batch 4: (100, 6)
    Batch 5: (100, 6)
    
    **Case 2: Random Sampling (Without Stratification)**
    
    >>> sampled_batches = batch_spatial_sampling(
    ...     data=df,
    ...     sample_size=0.05,
    ...     n_batches=5,
    ...     stratify_by=None,  # No stratification
    ...     spatial_bins=(10, 15),
    ...     spatial_cols=['longitude', 'latitude'],
    ...     random_state=42
    ... )
    
    >>> for i, batch in enumerate(sampled_batches):
    ...     print(f"Batch {i+1}: {batch.shape}")
    
    Creating 5 random batches with a total of 500 samples...
    Batch Sampling Progress: 100%|██████████| 5/5 [00:00<00:00, 247.27it/s]
    Batch sampling completed. 5 batches created.
    
    **Random Sampling Results:**
    Batch 1: (100, 6)
    Batch 2: (100, 6)
    Batch 3: (100, 6)
    Batch 4: (100, 6)
    Batch 5: (100, 6)


    See Also
    --------
    spatial_sampling : Perform stratified sampling without batching.

    References
    ----------
    .. [1] Kotsiantis, S., Kanellopoulos, D., & Pintelas, P. (2006).
           "Data preprocessing for supervised learning." *International
           Journal of Computer Science*, 1(2), 111-117.

    """
    data = data.copy()
    total_samples = sample_size
    if isinstance(sample_size, float):
        if not 0 < sample_size < 1:
            raise ValueError("When sample_size is a float, it must be between 0 and 1.")
        total_samples = int(len(data) * sample_size)
    elif isinstance(sample_size, int):
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
    else:
        raise ValueError("sample_size must be a float or int.")

    if total_samples > len(data):
        raise ValueError("sample_size is larger than the dataset.")

    if n_batches <= 0:
        raise ValueError("n_batches must be a positive integer.")

    sample_size_per_batch = total_samples // n_batches
    leftover = total_samples % n_batches

    batches = []
    remaining_data = data.copy()
    sampled_indices = set()
    rng = np.random.RandomState(random_state)

    # Set default spatial columns if not specified
    spatial_cols = columns_manager(spatial_cols) 
    if spatial_cols is None:
        spatial_cols = []
        if 'longitude' in data.columns:
            spatial_cols.append('longitude')
        if 'latitude' in data.columns:
            spatial_cols.append('latitude')
        if not spatial_cols:
            raise ValueError(
                "No spatial columns specified and "
                "'longitude' and 'latitude' not found in data."
            )
        if len(spatial_cols) == 1:
            warnings.warn(
                f"Only one spatial column '{spatial_cols[0]}' found. "
                "Using it for spatial stratification. "
                "For more accurate sampling, providing both spatial "
                "columns is recommended.",
                UserWarning
            )
    else:
        if not isinstance(spatial_cols, (list, tuple)):
            raise ValueError(
                "spatial_cols must be a list or tuple of column names."
            )
        if len(spatial_cols) > 2:
            raise ValueError(
                "spatial_cols can have at most two columns."
            )
        for col in spatial_cols:
            if col not in data.columns:
                raise ValueError(
                    f"Spatial column '{col}' not found in data."
                )
        if len(spatial_cols) == 1:
            warnings.warn(
                f"Only one spatial column '{spatial_cols[0]}' specified. "
                "For more accurate sampling, providing two spatial columns "
                "is recommended.",
                UserWarning
            )
    # Validate spatial_bins
    if isinstance(spatial_bins, int):
        n_bins_list = [spatial_bins] * len(spatial_cols)
    elif isinstance(spatial_bins, (tuple, list)):
        if len(spatial_bins) != len(spatial_cols):
            raise ValueError(
                "Length of spatial_bins must match number of spatial_cols."
            )
        n_bins_list = list(spatial_bins)
    else:
        raise ValueError(
            "spatial_bins must be int or tuple/list of int."
        )
    # Create spatial bins in the original data
    for col, n_bins, axis in zip(
        spatial_cols, n_bins_list, ['x_bin', 'y_bin']
    ):
        data[axis] = pd.qcut(
            data[col],
            q=n_bins,
            duplicates='drop'
        )
    # Create combined stratification key in original data
    # Create combined stratification key
    if stratify_by is not None:
        strat_columns = stratify_by + [
            axis for axis in ['x_bin', 'y_bin'][:len(spatial_cols)]
        ]
        if verbose and len(data) > 10_000:
            print(f"\nGenerating stratification keys for {len(data):,}"
                  " records...")
            print(" This may take some time. Please be patient...")

        # Optimized vectorized concatenation
        data['strat_key'] = data[strat_columns].astype(str).agg('_'.join, axis=1)

        if verbose and len(data) > 10_000:
            print("Stratification keys generated"
                  f" successfully for {len(data):,} records.")
    else:
        data['strat_key'] = 'all_data'  # Single group for random sampling
        
    # Initialize remaining data
    remaining_data = data.copy()
    batches = []

    # Set initial random state
    rng = np.random.RandomState(random_state)

    if verbose:
        print(f"\nCreating {n_batches} stratified batches with"
              f" a total of {total_samples:,} samples...")

    # TQDM progress bar
    if verbose and HAS_TQDM:
        progbar = tqdm(
            range(n_batches),
            total=n_batches,
            ascii=True,
            ncols=80,
            desc="Batch Sampling Progress"
        )
    
    if method=="relative": 
        min_relative_ratio= assert_ratio(
            min_relative_ratio, bounds=(0, 1), 
            excludes = (0, 1), 
            name="`min_relative_ratio`"
        )
    for batch_idx in range(n_batches):
        # Adjust sample size for batches if total_samples
        # is not divisible by n_batches
        if batch_idx < leftover:
            batch_sample_size = sample_size_per_batch + 1
        else:
            batch_sample_size = sample_size_per_batch

        # if batch_sample_size > len(remaining_data):
        #     batch_sample_size = len(remaining_data)
        batch_sample_size = min(batch_sample_size, len(remaining_data))
        
        # Group remaining data by stratification key
        grouped = remaining_data.groupby('strat_key')
        # Calculate number of samples per group
        group_sizes = grouped.size()
        total_size = group_sizes.sum()
        
        if method in ["abs", "absolute"]:
            group_sample_sizes = (
                (group_sizes / total_size * batch_sample_size)
                .round()
                .astype(int)
            )
        else:  # "relative"
            relative_scale = np.clip(batch_sample_size / len(
                remaining_data), min_relative_ratio, 1)
            group_sample_sizes = (
                (group_sizes * relative_scale)
                .round()
                .astype(int)
            )
    
        # Sample data from each group
        sampled_indices = []
        for strat_value, group in grouped:
            n = group_sample_sizes.loc[strat_value]
            if n > 0 and len(group) > 0:
                sampled_group = group.sample(
                    n=min(n, len(group)),
                    random_state=rng.randint(0, 10000)
                )
                sampled_indices.extend(sampled_group.index)
        # Create the sampled DataFrame
        batch_sampled_data = remaining_data.loc[sampled_indices]
        batches.append(batch_sampled_data.drop(
            columns=['strat_key'] + [axis for axis in [
                'x_bin', 'y_bin'][:len(spatial_cols)]]))
        # Remove sampled data from remaining_data
        remaining_data = remaining_data.drop(index=sampled_indices)
        if len(remaining_data) == 0:
            break  # No more data to sample
        
        if verbose and HAS_TQDM:
            progbar.update(1)

    if verbose and HAS_TQDM:
        progbar.close()
    
    if verbose:
        print(f"\nBatch sampling completed. {len(batches)} batches created.")
    
    has_empty_batches = any([ b.empty for b in batches])
    
    if verbose and has_empty_batches:
        warnings.warn(
            "\nNo records were sampled. This is likely due to"
            " insufficient data for the specified stratification"
            f" columns {stratify_by}. To resolve this, consider:\n"
            " • Using a different stratification method.\n"
            " • Increasing the dataset size to include more"
            " representative data.\n"
            " • Adjusting the sample size to ensure sufficient"
            " records per group.\n"
            " • Or, setting `stratify_by=None` to perform"
            " random sampling instead."
        )
        
    return batches

@SaveFile
@is_data_readable 
@validate_params ({ 
    'data': ['array-like'], 
    'method': [StrOptions({"abs", "absolute",  "relative"}), None], 
    })
@isdf
def spatial_sampling(
    data,
    sample_size=0.01,
    stratify_by=None,
    spatial_bins=10,
    spatial_cols=None,
    method='abs', 
    min_relative_ratio=.01, 
    random_state=42, 
    savefile=None, 
    verbose=1,
    
):
    """
    Sample spatial data intelligently to represent the distribution
    of the whole area and include different years.

    This function performs stratified sampling on spatial data,
    ensuring that the sample reflects both spatial distribution
    and temporal aspects of the entire dataset [1]_. It combines spatial
    stratification based on coordinates and additional stratification
    columns specified by the user.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame to sample from. Must contain spatial
        coordinate columns (e.g., `'longitude'`, `'latitude'`) and
        any columns specified in ``stratify_by``.
    sample_size : float or int, optional
        The proportion or absolute number of samples to select.
        If float, should be between 0.0 and 1.0 and represents the
        fraction of the dataset to include in the sample.
        If int, represents the absolute number of samples to select.
        Default is ``0.01`` (1% of the data).
    stratify_by : list of str, optional
        List of column names to stratify by.
    spatial_bins : int or tuple/list of int, optional
        Number of bins to divide the spatial coordinates into.
        If an integer, the same number of bins is used for all spatial
        dimensions. If a tuple or list, its length must match the number
        of spatial columns, specifying the number of bins for each spatial
        dimension. Default is ``10``.
    spatial_cols : list or tuple of str, optional
        List of spatial coordinate column names. Can accept one or two
        columns. If ``None``, the function checks for columns named
        `'longitude'` and/or `'latitude'` in ``data``. If only one spatial
        column is provided or found, a warning is issued, suggesting that
        providing both spatial columns is recommended for more accurate
        sampling. If more than two columns are provided, an error is raised.
        
    method : str, {'abs', 'relative'}, default='abs'
        Defines how the sample size is determined:
        - ``'abs'`` or ``'absolute'``: Uses a **fixed** sampling proportion
          based on `sample_size`.
        - ``'relative'``: Dynamically **scales** sampling based on dataset
          stratification, ensuring that all stratification groups receive
          a proportional sample while maintaining a minimum sampling ratio
          (controlled by `min_relative_ratio`).
        
        When ``method='relative'``, the function ensures that even small
        stratification groups receive a sufficient sample by applying
        `min_relative_ratio`.

    min_relative_ratio : float, default=0.01
        Controls the **minimum allowable fraction** of records that 
        must be sampled when ``method='relative'``.

        - Ensures that no group is **undersampled** to zero, even if
          its natural proportion in the dataset is very small.
        - Must be a value between ``0`` and ``1``.
        - The default value (``0.01``) means that at **least 1% of the
          total dataset** will be sampled from each stratification group,
          regardless of its relative size.
        
        **Example Scenarios:**
        
        - If `min_relative_ratio=0.05`, then each group **must** 
          contribute **at least 5%** of the total dataset size (if possible).
        - If a group is too small to reach this minimum, its entire
          subset is sampled instead.
        - This ensures that no group receives **less than
        ``min_relative_ratio × total samples**``.

    random_state : int, optional
        Random seed for reproducibility. Default is ``42``.
        
    verbose: bool, default=False, 
       If `True`, displays a progress bar and detailed status messages
       during execution. Useful for monitoring the process, especially
       when working with large datasets.

    Returns
    -------
    sampled_data : pandas.DataFrame
        A sampled DataFrame representing the distribution of the whole
        area and including different years.

    Notes
    -----
    The function performs stratified sampling based on spatial bins
    and other specified stratification columns. Spatial coordinates
    are binned using quantile-based discretization (:func:`pandas.qcut`),
    ensuring each bin has approximately the same number of observations.

    Let :math:`N` be the total number of samples in ``data``, and
    :math:`n` be the desired sample size. The function calculates the
    number of samples to draw from each stratification group based on
    the proportion of the group size to the total dataset size:

    .. math::

        n_i = \left\lceil \frac{N_i}{N} \times n \right\rceil

    where :math:`N_i` is the size of group :math:`i`, and :math:`n_i`
    is the number of samples to draw from group :math:`i`.

    The function ensures that:

    - All specified spatial and stratification columns exist in ``data``.
    - The number of spatial bins matches the number of spatial columns.
    - The sample size is valid (positive float between 0 and 1, or
      positive integer).

    Warnings are issued if:

    - Only one spatial column is used, suggesting that using two spatial
      columns is recommended for more accurate sampling.

    Examples
    --------
    >>> from fusionlab.utils.spatial_utils import spatial_sampling
    >>> import pandas as pd
    >>> # Assume 'df' is a pandas DataFrame with columns
    >>> # 'longitude', 'latitude', 'year', and other data.
    >>> sampled_df = spatial_sampling(
    ...     data=df,
    ...     sample_size=0.05,
    ...     stratify_by=['year', 'geological_category'],
    ...     spatial_bins=(10, 15),
    ...     spatial_cols=['longitude', 'latitude'],
    ...     random_state=42
    ... )
    >>> print(sampled_df.shape)

    See Also
    --------
    pandas.qcut : Quantile-based discretization function used for binning.
    sklearn.model_selection.StratifiedShuffleSplit : For stratified sampling.
    batch_spatial_sampling: Resample spatial data with batching. 

    References
    ----------
    .. [1] Kotsiantis, S., Kanellopoulos, D., & Pintelas, P. (2006).
           "Data preprocessing for supervised learning." *International
           Journal of Computer Science*, 1(2), 111-117.

    """
    data = data.copy()
    # Set default spatial columns if not specified
    spatial_cols= columns_manager(spatial_cols)

    if spatial_cols is None:
        spatial_cols = []
        if 'longitude' in data.columns:
            spatial_cols.append('longitude')
        if 'latitude' in data.columns:
            spatial_cols.append('latitude')
        if not spatial_cols:
            raise ValueError(
                "No spatial columns specified and "
                "'longitude' and 'latitude' not found in data."
            )
        if len(spatial_cols) == 1:
            warnings.warn(
                f"Only one spatial column '{spatial_cols[0]}' found. "
                "Using it for spatial stratification. "
                "For more accurate sampling, providing both spatial "
                "columns is recommended.",
                UserWarning
            )
    else:
        if not isinstance(spatial_cols, (list, tuple)):
            raise ValueError(
                "spatial_cols must be a list or tuple of column names."
            )
        if len(spatial_cols) > 2:
            raise ValueError(
                "spatial_cols can have at most two columns."
            )
        for col in spatial_cols:
            if col not in data.columns:
                raise ValueError(
                    f"Spatial column '{col}' not found in data."
                )
        if len(spatial_cols) == 1:
            warnings.warn(
                f"Only one spatial column '{spatial_cols[0]}' specified. "
                "For more accurate sampling, providing two spatial columns "
                "is recommended.",
                UserWarning
            )
    # Validate spatial_bins
    if isinstance(spatial_bins, int):
        n_bins = validate_positive_integer(
            spatial_bins,
            'spatial_bins'
        )
        n_bins_list = [n_bins] * len(spatial_cols)
    elif isinstance(spatial_bins, (tuple, list)):
        if len(spatial_bins) != len(spatial_cols):
            raise ValueError(
                "Length of spatial_bins must match number of spatial_cols."
            )
        n_bins_list = [
            validate_positive_integer(
                n, 'spatial_bins'
            ) for n in spatial_bins
        ]
    else:
        raise ValueError(
            "spatial_bins must be int or tuple/list of int."
        )
    
    # if verbose and HAS_TQDM are True.
    if verbose and HAS_TQDM:
        progbar = tqdm(
            zip(spatial_cols, n_bins_list, ['x_bin', 'y_bin']),
            total=len(spatial_cols),
            ascii=True,
            ncols=77,
            desc=f"{'Creating spat. bins: ' + str(len(spatial_cols)):<20}"
        )
    # Create spatial bins
    for col, n_bins, axis in zip(
            spatial_cols, n_bins_list, ['x_bin', 'y_bin']):
        data[axis] = pd.qcut(
            data[col],
            q=n_bins,
            duplicates='drop'
        )
        if verbose and HAS_TQDM:
            progbar.update(1)
    
    if verbose and HAS_TQDM:
        progbar.close()

    stratify_by= columns_manager(stratify_by, empty_as_none=False )
    # Create combined stratification key
    strat_columns = stratify_by + [
        axis for axis in ['x_bin', 'y_bin'][:len(spatial_cols)]
    ]
    if verbose and len(data) > 10_000:
        print("\nGenerating stratification keys...") 
        print(f"This may take some time for {len(data):,}"
              " records. Please be patient...")
              
    # data['strat_key'] = data[strat_columns].apply(
    #     lambda row: '_'.join(row.values.astype(str)),
    #     axis=1
    # )
    # Using .agg and .astype(str) for vectorized string concatenation
    data['strat_key'] = data[strat_columns].astype(str).agg('_'.join, axis=1)

    # Verbose message when done
    if verbose and len(data)> 10_000:
        print("Stratification keys generated successfully"
              f" for {len(data):,} records.")
    # Determine total number of samples
    if isinstance(sample_size, float):
        if not 0 < sample_size < 1:
            raise ValueError(
                "When sample_size is a float, it must be between 0 and 1."
            )
        n_samples = int(len(data) * sample_size)
    elif isinstance(sample_size, int):
        n_samples = validate_positive_integer(
            sample_size,
            'sample_size'
        )
    else:
        raise ValueError(
            "sample_size must be a positive float or int."
        )
    # Group data by stratification key
    grouped = data.groupby('strat_key')
    if verbose:
        print(f"Data grouped into {len(grouped):,} stratified bins.")
        
    # Apply stratification if stratify_by is provided
    if stratify_by is not None:
        # Calculate number of samples per group
        group_sizes = grouped.size()
        total_size = group_sizes.sum()
        
        if method in ["abs", "absolute"]:
            group_sample_sizes = (
                (group_sizes / total_size * n_samples)
                .round()
                .astype(int)
            )
        else:  # "relative"
            min_relative_ratio = assert_ratio(
                min_relative_ratio, bounds=(0, 1), 
                exclude_values= [0, 1], 
                name="`min_relative_ratio`"
            ) 
            relative_scale = np.clip(
                n_samples / len(data), min_relative_ratio, 1)  
            group_sample_sizes = (
                (group_sizes * relative_scale)
                .round()
                .astype(int)
            )
        
        # Sample data from each group
        sampled_indices = []
        np.random.seed(random_state)
        # Use tqdm to wrap the grouped iterator 
        # if verbose and HAS_TQM are True.
        if verbose and HAS_TQDM:
            progbar = tqdm(
                grouped,
                total=len(grouped),
                ascii=True,
                ncols=77,
                desc=f"Sampling {n_samples:,} records"
            )
        
        for strat_value, group in grouped:
            n = group_sample_sizes.loc[strat_value]
            if n > 0 and len(group) > 0:
                sampled_group = group.sample(
                    n=min(n, len(group)),
                    random_state=np.random.randint(
                        0, 10_000
                    )
                )
                sampled_indices.extend(
                    sampled_group.index
                )
            if verbose and HAS_TQDM:
                progbar.update(1)
                
        if verbose and HAS_TQDM:
            progbar.close() 
    else: 
        sampled_indices = np.random.choice(
            data.index, size=n_samples,
            replace=False
        )
        
    # Create the sampled DataFrame
    sampled_data = data.loc[
        sampled_indices
    ]
    # Drop helper columns
    cols_to_drop = ['strat_key'] + [
        axis for axis in ['x_bin', 'y_bin'][:len(spatial_cols)]
    ]
    sampled_data = sampled_data.drop(
        columns=cols_to_drop
    )
    if verbose:
        print(f"\nSampling completed: {len(sampled_indices):,}"
              " records selected.")
    
    if verbose and sampled_data.empty:
        warnings.warn(
            "\nNo records were sampled. This is likely due to"
            " insufficient data for the specified stratification"
            f" columns {stratify_by}. To resolve this, consider:\n"
            " • Using a different stratification method.\n"
            " • Increasing the dataset size to include more"
            " representative data.\n"
            " • Adjusting the sample size to ensure sufficient"
            " records per group.\n"
            " • Or, setting `stratify_by=None` to perform"
            " random sampling instead."
        )

    return sampled_data.reset_index(
        drop=True
    )

