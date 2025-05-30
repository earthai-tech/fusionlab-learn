
import warnings 

from ..core.checks import ( 
    exist_features, 
    check_spatial_columns 
    )
from ..core.handlers import columns_manager

__all__ = ["resolve_spatial_columns"]

def resolve_spatial_columns(
    df,
    spatial_cols=None,
    lon_col=None,
    lat_col=None
):
    """
    Helper to validate and resolve spatial columns.

    Accepts either explicit lon/lat columns or a
    list of spatial_cols. Returns (lon_col, lat_col).

    - If lon_col and lat_col are both provided, they
      take precedence (warn if spatial_cols also set).
    - Else if spatial_cols is provided, it must yield
      exactly two column names.
    - Otherwise, error is raised.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame for feature checks.
    spatial_cols : list[str] or None
        Two-element list of [lon_col, lat_col].
    lon_col : str or None
        Name of longitude column.
    lat_col : str or None
        Name of latitude column.

    Returns
    -------
    (lon_col, lat_col) : tuple of str
        Validated column names for longitude and
        latitude.

    Raises
    ------
    ValueError
        If neither lon/lat nor valid spatial_cols is
        provided, or if spatial_cols len != 2.
    """
    # Case 1: explicit lon/lat
    if lon_col is not None and lat_col is not None:
        if spatial_cols:
            warnings.warn(
                "Both lon_col/lat_col and spatial_cols set;"
                " spatial_cols will be ignored.",
                UserWarning
            )
        exist_features(
            df,
            features=[lon_col, lat_col],
            name="Longitude/Latitude"
        )
        return lon_col, lat_col

    # Case 2: spatial_cols provided
    if spatial_cols:
        spatial_cols = columns_manager(
            spatial_cols,
            empty_as_none=False
        )
        check_spatial_columns(
            df,
            spatial_cols=spatial_cols
        )
        exist_features(
            df,
            features=spatial_cols,
            name="Spatial columns"
        )
        if len(spatial_cols) != 2:
            raise ValueError(
                "spatial_cols must contain exactly two"
                " column names"
            )
        lon, lat = spatial_cols
        return lon, lat

    # Neither provided
    raise ValueError(
        "Either lon_col & lat_col, or spatial_cols"
        " must be provided."
    )
