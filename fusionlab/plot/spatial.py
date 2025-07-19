# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

"""
Spatial plots
"""
import os
import logging
from typing import List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon

import numpy as np 
import pandas as pd
from ..api.docs import DocstringComponents, _spatial_params
from ..core.checks import check_spatial_columns, exist_features, check_empty
from ..core.handlers import columns_manager
from ..decorators import isdf
from ..utils.generic_utils import vlog
from ..utils.spatial_utils import extract_spatial_roi 

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_spatial_params), 
)

__all__= [ 
    'plot_spatial', 
    'plot_spatial_roi', 
    'plot_spatial_contours', 
    'plot_hotspots', 
    'plot_spatial_voronoi', 
    'plot_spatial_heatmap'
    ]

@check_empty(['df'])
@isdf
def plot_spatial(
    df: pd.DataFrame,
    value_col: str,
    spatial_cols: Optional[Tuple[str, str]] = None,
    dt_col: Optional[str] = None,
    dt_values: Optional[List[Union[int, str]]] = None,
    *,
    cmap: str = "viridis",
    marker_size: int = 50,
    alpha: float = 0.8,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_grid: bool = True,
    grid_props: Optional[dict] = None,
    savefig: Optional[str] = None,
    save_fmts: Union[str, List[str]] = ".png",
    max_cols: int = 3,
    prefix: str = "",
    verbose: int = 1,
    cbar: str = "uniform",
    show_axis: Union[str, bool] = "on",
    _logger: Optional[logging.Logger] = None,
    **plot_kws
) -> List[plt.Figure]:
    
    def _v(msg: str, level: int = 1):
        vlog(message=msg, verbose=verbose, level=level, logger=_logger)

    exist_features(df, features=value_col, error='raise')
    spatial_cols = spatial_cols or ("coord_x", "coord_y")
    spatial_cols = columns_manager(spatial_cols, empty_as_none=False)
    check_spatial_columns(df, spatial_cols=spatial_cols)
    x_col, y_col = spatial_cols

    if dt_col:
        if dt_col not in df.columns:
            raise ValueError(f"'{dt_col}' not in DataFrame")
        dt_values = dt_values or sorted(df[dt_col].dropna().unique())
    elif dt_values is None:
        raise ValueError("Either 'dt_col' or 'dt_values' must be provided")

    if show_grid and grid_props is None:
        grid_props = dict(linestyle=':', alpha=0.7)

    n = len(dt_values)
    ncols = min(max_cols, n)
    figures: List[plt.Figure] = []

    for start in range(0, n, ncols):
        row_dts = dt_values[start : start + ncols]
        fig, axes = plt.subplots(
            1,
            len(row_dts),
            figsize=(4 * len(row_dts), 4),
            constrained_layout=True
        )
        axes = axes if isinstance(axes, np.ndarray) else [axes]
        scatters: List = []

        for ax, dt in zip(axes, row_dts):
            slice_df = df[df[dt_col] == dt] if dt_col else df
            sc = ax.scatter(
                slice_df[x_col], slice_df[y_col],
                c=slice_df[value_col],
                cmap=cmap,
                s=marker_size,
                alpha=alpha,
                vmin=vmin if vmin is not None else slice_df[value_col].min(),
                vmax=vmax if vmax is not None else slice_df[value_col].max(),
                **plot_kws
            )
            scatters.append(sc)
            if show_grid:
                ax.grid(**grid_props)

            flag = (show_axis.lower() not in ('off','false')) if isinstance(
                show_axis, str) else bool(show_axis)
            if not flag:
                ax.set_axis_off()

            ax.set_title(f"{value_col} @ {dt}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        if cbar == 'uniform':
            fig.colorbar(
                scatters[0],
                ax=axes,
                orientation='vertical',
                pad=0.02,
                label=value_col
            )
        else:
            for ax_i, sc_i in zip(axes, scatters):
                fig.colorbar(
                    sc_i,
                    ax=ax_i,
                    shrink=0.6,
                    pad=0.02,
                    label=value_col
                )

        if savefig:
            fmts = [savefig] if isinstance(savefig, str) else save_fmts
            base, _ = os.path.splitext(savefig)
            for fmt in (fmts if isinstance(fmts, list) else [fmts]):
                ext = fmt if fmt.startswith('.') else '.' + fmt
                out = f"{base}{prefix}{ext}"
                os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
                _v(f"Saving figure to {out}")
                fig.savefig(out, dpi=300, bbox_inches='tight')

        figures.append(fig)

    return figures

plot_spatial.__doc__ = r"""
Plot spatial scatter maps for multiple time slices.

{params.base.spatial_cols}
{params.base.dt_col}
{params.base.dt_values}

value_col : str
    Column name containing the numeric values to colour the scatter
    points on the map. This must be a column in `df` with numeric
    data (e.g. 'subsidence_q50').

{params.base.cmap}
{params.base.marker_size}
{params.base.alpha}
{params.base.vmin}
{params.base.vmax}
{params.base.show_grid}
{params.base.grid_props}
{params.base.savefig}
{params.base.save_fmts}
{params.base.max_cols}
{params.base.prefix}
{params.base.verbose}
{params.base.cbar}
{params.base.show_axis}
{params.base._logger}

Returns
-------
List[matplotlib.figure.Figure]
    One figure per row of time-slices, each containing up to
    `max_cols` subplots.

Examples
--------
>>> from fusionlab.plot.spatial import plot_spatial
>>> figs = plot_spatial(
...     df=my_df,
...     value_col='subsidence_q50',
...     dt_col='coord_t',
...     dt_values=[2020, 2021, 2022],
...     cmap='viridis',
...     cbar='uniform'
... )
>>> for fig in figs:
...     fig.show()

Notes
-----
- When `cbar='uniform'`, a single colorbar is shared across all
  subplots in a row.  Otherwise, each subplot gets its own.
- To hide axes and ticks, use `show_axis='off'`.
- Lays out with `constrained_layout=True` to minimize overlap.

See Also
--------
plot_spatial_roi : Plot the same maps but limited to a
                   specified region of interest.
""".format(params=_param_docs)

@check_empty(['df'])
@isdf
def plot_spatial_roi(
    df: pd.DataFrame,
    value_cols: List[str],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    spatial_cols: Optional[Tuple[str, str]] = None,
    dt_col: Optional[str] = None,
    dt_values: Optional[List[Union[int, str]]] = None,
    *,
    cmap: str = "viridis",
    marker_size: int = 50,
    alpha: float = 0.8,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_grid: bool = True,
    grid_props: Optional[dict] = None,
    savefig: Optional[str] = None,
    save_fmts: Union[str, List[str]] = ".png",
    max_cols: int = 3,
    prefix: str = "",
    verbose: int = 1,
    cbar: str = "uniform",
    show_axis: Union[str, bool] = "on",
    _logger: Optional[logging.Logger] = None,
    **plot_kws
) -> List[plt.Figure]:
    def _v(msg: str, level: int = 1):
        vlog(message=msg, verbose=verbose, level=level, logger=_logger)

    for col in value_cols:
        exist_features(df, features=col, error='raise')

    spatial_cols = spatial_cols or ('coord_x', 'coord_y')
    spatial_cols = columns_manager(spatial_cols, empty_as_none=False)
    check_spatial_columns(df, spatial_cols=spatial_cols)
    x_col, y_col = spatial_cols

    if dt_col:
        dt_values = dt_values or sorted(df[dt_col].dropna().unique())
    elif dt_values is None:
        raise ValueError("Either 'dt_col' or 'dt_values' must be provided")

    if show_grid and grid_props is None:
        grid_props = dict(linestyle=':', alpha=0.7)

    # grid of dt rows x value_cols columns
    n_dt = len(dt_values)
    n_val = len(value_cols)
    # figures: List[plt.Figure] = []
    fig, axes = plt.subplots(
        n_dt, n_val,
        figsize=(4 * n_val, 4 * n_dt),
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    scatters: List = []

    for i, dt in enumerate(dt_values):
        df_dt = df[df[dt_col] == dt] if dt_col else df
        roi_df = extract_spatial_roi(
            df_dt, x_range=x_range, y_range=y_range,
            x_col=x_col, y_col=y_col, snap_to_closest=True
        )
        for j, val in enumerate(value_cols):
            ax = axes[i, j]
            sc = ax.scatter(
                roi_df[x_col], roi_df[y_col],
                c=roi_df[val], cmap=cmap,
                s=marker_size, alpha=alpha,
                vmin=vmin if vmin is not None else roi_df[val].min(),
                vmax=vmax if vmax is not None else roi_df[val].max(),
                **plot_kws
            )
            scatters.append(sc)
            if show_grid:
                ax.grid(**grid_props)
            flag = (show_axis.lower() not in ('off','false')) if isinstance(
                show_axis, str) else bool(show_axis)
            if not flag:
                ax.set_axis_off()
            ax.set_title(f"{val} @ {dt}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

    if cbar == 'uniform':
        fig.colorbar(
            scatters[0], ax=axes.flatten(), pad=0.02,
            label="value"
        )
    else:
        for ax_i, sc_i in zip(axes.flatten(), scatters):
            fig.colorbar(sc_i, ax=ax_i, shrink=0.6,
                         pad=0.02, label="value")

    if savefig:
        fmts = [savefig] if isinstance(savefig, str) else save_fmts
        base, _ = os.path.splitext(savefig)
        for fmt in (fmts if isinstance(fmts, list) else [fmts]):
            ext = fmt if fmt.startswith('.') else '.' + fmt
            out = f"{base}{prefix}{ext}"
            os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
            _v(f"Saving figure to {out}")
            fig.savefig(out, dpi=300, bbox_inches='tight')

    return [fig]

plot_spatial_roi.__doc__ = r"""
Plot spatial scatter maps over a Region of Interest (ROI) for
multiple time slices and multiple value columns.

{params.base.spatial_cols}
{params.base.dt_col}
{params.base.dt_values}

x_range, y_range : tuple(float, float)
    Bounding box for the ROI as (xmin, xmax) and (ymin, ymax).
    Defines the rectangular area to extract via extract_spatial_roi.

value_cols : list of str
    List of column names to colour each subplot by.  Each
    column will be plotted in its own column of the grid.

{params.base.cmap}
{params.base.marker_size}
{params.base.alpha}
{params.base.vmin}
{params.base.vmax}
{params.base.show_grid}
{params.base.grid_props}
{params.base.savefig}
{params.base.save_fmts}
{params.base.max_cols}
{params.base.prefix}
{params.base.verbose}
{params.base.cbar}
{params.base.show_axis}
{params.base._logger}

Returns
-------
List[matplotlib.figure.Figure]
    A single Figure laid out as n_time_slices × n_value_cols,
    containing one subplot per combination.

Examples
--------
>>> from fusionlab.plot.spatial import plot_spatial_roi
>>> figs = plot_spatial_roi(
...     df=my_df,
...     value_cols=['subsidence_actual','subsidence_q50'],
...     dt_col='coord_t',
...     dt_values=[2024,2025],
...     x_range=(113.6,113.7),
...     y_range=(22.55,22.65),
...     cbar='individual'
... )
>>> figs[0].show()

Notes
-----
- Uses extract_spatial_roi to filter to the bounding box.
- Plots a grid of time-slices (rows) by value columns
  (columns) in a single figure.
- Colorbars follow the `cbar` setting.

See Also
--------
plot_spatial : Plot full-domain maps without ROI filtering.
""".format(params=_param_docs)

@check_empty(['df'])
@isdf
@check_empty(['df'])
@isdf
def plot_spatial_contours(
    df: pd.DataFrame,
    value_col: str,
    spatial_cols: Optional[Tuple[str, str]] = None,
    dt_col: Optional[str] = None,
    dt_values: Optional[List[Union[int, str]]] = None,
    *,
    grid_res: int = 100,
    levels: List[float] = [0.1, 0.5, 0.9],
    method: str = 'linear',
    cmap: str = 'viridis',
    show_points: bool = True,
    show_grid: bool = True,
    grid_props: Optional[dict]=None, 
    max_cols: int = 3,
    savefig: Optional[str] = None,
    save_fmts: Union[str, List[str]] = '.png',
    prefix: str = '',
    verbose: int = 1,
    cbar: str = 'uniform',
    show_axis: Union[str, bool] = 'on',
    _logger: Optional[logging.Logger] = None,
    **contour_kws
) -> List[plt.Figure]:
    # Create one figure of subplots for all dt_values
    def _v(msg: str, level: int = 1):
        vlog(message=msg, verbose=verbose, level=level, logger=_logger)

    exist_features(df, features=value_col, error='raise')
    spatial_cols = spatial_cols or ('coord_x', 'coord_y')
    spatial_cols = columns_manager(spatial_cols, empty_as_none=False)
    check_spatial_columns(df, spatial_cols=spatial_cols)
    x_col, y_col = spatial_cols

    if dt_col:
        dt_values = dt_values or sorted(df[dt_col].dropna().unique())
    elif dt_values is None:
        raise ValueError("Either 'dt_col' or 'dt_values' must be provided")

    n = len(dt_values)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        constrained_layout=True
    )
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    scatters = []
    for idx, dt in enumerate(dt_values):
        ax = axes_flat[idx]
        df_dt = df[df[dt_col] == dt] if dt_col else df
        x = df_dt[x_col].values
        y = df_dt[y_col].values
        z = df_dt[value_col].values

        # grid & interpolate
        xi = np.linspace(x.min(), x.max(), grid_res)
        yi = np.linspace(y.min(), y.max(), grid_res)
        Xi, Yi = np.meshgrid(xi, yi)
        from scipy.interpolate import griddata
        Zi = griddata((x, y), z, (Xi, Yi), method=method)

        clevels = np.quantile(z, levels)
        cf = ax.contourf(Xi, Yi, Zi, levels=clevels, cmap=cmap, alpha=0.6, **contour_kws)
        cs = ax.contour(Xi, Yi, Zi, levels=clevels, colors='k', linewidths=1.0)
        ax.clabel(cs, fmt='%.2f', fontsize=8)

        if show_points:
            ax.scatter(x, y, c='k', s=5)
        if show_grid:
            grid_props = grid_props or dict(linestyle=':', alpha=0.7)
            ax.grid(**grid_props)

        flag = (show_axis.lower() not in ('off','false')) if isinstance(
            show_axis, str) else bool(show_axis)
        if not flag:
            ax.set_axis_off()

        ax.set_title(f"{value_col} @ {dt}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        scatters.append(cf)

    # hide unused axes
    for j in range(idx+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # shared colorbar
    if cbar == 'uniform':
        fig.colorbar(scatters[0], ax=axes_flat[:idx+1], shrink=0.7, label=value_col)
    else:
        for ax_i, cf_i in zip(axes_flat[:idx+1], scatters):
            fig.colorbar(cf_i, ax=ax_i, shrink=0.6, pad=0.02, label=value_col)

    # save figure
    if savefig:
        fmts = [savefig] if isinstance(savefig, str) else save_fmts
        base, _ = os.path.splitext(savefig)
        for fmt in (fmts if isinstance(fmts, list) else [fmts]):
            ext = fmt if fmt.startswith('.') else '.' + fmt
            out = f"{base}{prefix}{ext}"
            os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
            _v(f"Saving contour plot to {out}")
            fig.savefig(out, dpi=300, bbox_inches='tight')

    return [fig]

plot_spatial_contours.__doc__ = r"""
Plot contour lines at specified quantiles over a spatial region.

{params.base.spatial_cols}
{params.base.dt_col}
{params.base.dt_values}

value_col : str
    Column name containing the numeric values to contour
    (e.g. 'subsidence_q50').

grid_res : int
    Number of points along each axis for the interpolation grid.
levels : list of float
    Quantile levels (0–1) at which to draw contour lines
    (e.g. [0.1, 0.5, 0.9]).
method : str
    Interpolation method passed to `scipy.interpolate.griddata`
    (choices: 'linear', 'cubic', 'nearest').
show_points : bool
    If True, overlay the raw data points on top of the contours.

{params.base.cmap}
{params.base.show_grid}
{params.base.savefig}
{params.base.save_fmts}
{params.base.prefix}
{params.base.verbose}
{params.base._logger}

Returns
-------
List[matplotlib.figure.Figure]
    One figure per time slice, each showing filled contours,
    overlaid lines at the specified quantiles, and optional
    scatter points.

Examples
--------
>>> from fusionlab.plot.spatial import plot_spatial_contours
>>> figs = plot_spatial_contours(
...     df=my_df,
...     value_col='subsidence_q50',
...     dt_col='coord_t',
...     dt_values=[2024, 2025],
...     grid_res=200,
...     levels=[0.1, 0.5, 0.9],
...     show_points=False,
...     cmap='plasma'
... )
>>> for fig in figs:
...     fig.show()

Notes
-----
- Internally uses `scipy.interpolate.griddata` to rasterize
  the scattered z-values onto a regular mesh before contouring.
- Uses `constrained_layout=True` to prevent label/colorbar overlap.
- Any extra keyword args passed via `contour_kws` are forwarded to
  both `contourf` and `contour` calls.

See Also
--------
plot_spatial     : Discrete scatter-map faceting by time‐slice.
plot_spatial_roi : Region-of-interest scatter‐map faceting.
""".format(params=_param_docs)

@check_empty(['df'])
@isdf
def plot_hotspots(
    df: pd.DataFrame,
    value_col: str,
    spatial_cols: Optional[Tuple[str, str]] = None,
    dt_col: Optional[str] = None,
    dt_values: Optional[List[Union[int, str]]] = None,
    *,
    grid_size: Union[int, Tuple[int,int]] = 50,
    threshold: Optional[float] = None,
    percentile: float = 90.0,
    cmap: str = 'Reds',
    marker_size: int = 50,
    alpha: float = 0.6,
    show_grid: bool = True,
    grid_props: Optional[dict] = None,
    max_cols: int = 3,
    savefig: Optional[str] = None,
    save_fmts: Union[str, List[str]] = '.png',
    prefix: str = '',
    verbose: int = 1,
    cbar: str = 'uniform',
    show_axis: Union[str,bool] = 'on',
    _logger: Optional[logging.Logger] = None,
    **hexbin_kws
) -> List[plt.Figure]:
    """
    Plot spatial hotspots by highlighting areas where `value_col`
    exceeds a threshold (or top percentile) using a hexbin heatmap
    and overlaying hotspot points.
    """
    def _v(msg: str, level: int = 1):
        vlog(message=msg, verbose=verbose, level=level, logger=_logger)

    exist_features(df, features=value_col, error='raise')
    spatial_cols = spatial_cols or ('coord_x', 'coord_y')
    spatial_cols = columns_manager(spatial_cols, empty_as_none=False)
    check_spatial_columns(df, spatial_cols=spatial_cols)
    x_col, y_col = spatial_cols

    if dt_col:
        dt_values = dt_values or sorted(df[dt_col].dropna().unique())
    elif dt_values is None:
        raise ValueError("Either 'dt_col' or 'dt_values' must be provided")

    # Normalize grid_size
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    n = len(dt_values)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        constrained_layout=True
    )
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    hexbs = []
    for idx, dt in enumerate(dt_values):
        ax = axes_flat[idx]
        df_dt = df[df[dt_col] == dt] if dt_col else df
        x = df_dt[x_col]
        y = df_dt[y_col]
        z = df_dt[value_col]

        thr = threshold if threshold is not None else np.percentile(z, percentile)
        df_hot = df_dt[z >= thr]

        hb = ax.hexbin(
            x, y,
            C=z,
            gridsize=grid_size,
            cmap=cmap,
            mincnt=1,
            **hexbin_kws
        )
        hexbs.append(hb)

        ax.scatter(
            df_hot[x_col], df_hot[y_col],
            color='black', s=marker_size,
            alpha=alpha, label=f'Hotspots ≥ {thr:.2f}'
        )
        if show_grid:
            gp = grid_props or dict(linestyle=':', alpha=0.7)
            ax.grid(**gp)

        flag = (show_axis.lower() not in ('off','false')) if isinstance(
            show_axis, str) else bool(show_axis)
        if not flag:
            ax.set_axis_off()

        ax.set_title(f"Hotspots for {value_col} @ {dt}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()

    # hide unused axes
    for j in range(idx+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # colorbars
    if cbar == 'uniform':
        fig.colorbar(hexbs[0], ax=axes_flat[:idx+1],
                     shrink=0.7, label=value_col)
    else:
        for ax_i, hb_i in zip(axes_flat[:idx+1], hexbs):
            fig.colorbar(hb_i, ax=ax_i, shrink=0.6,
                         pad=0.02, label=value_col)

    if savefig:
        fmts = [savefig] if isinstance(savefig, str) else save_fmts
        base, _ = os.path.splitext(savefig)
        for fmt in (fmts if isinstance(fmts, list) else [fmts]):
            ext = fmt if fmt.startswith('.') else '.' + fmt
            out = f"{base}{prefix}{ext}"
            os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
            _v(f"Saving hotspot plot to {out}")
            fig.savefig(out, dpi=300, bbox_inches='tight')

    return [fig]

plot_hotspots.__doc__ += r"""

{params.base.spatial_cols}
{params.base.dt_col}
{params.base.dt_values}

grid_size : Union[int, Tuple[int,int]]
    Resolution of the hexagonal binning grid.  Can be a single int or
    a (nx, ny) tuple for hexbin `gridsize`.
threshold : Optional[float]
    Absolute cutoff value; points with `value_col >= threshold`
    will be marked as hotspots.  If None, `percentile` is used.
percentile : float
    Percentile (0–100) threshold for hotspot selection if `threshold`
    is None (e.g. 90.0 for top 10%).

cmap : str
    Matplotlib colormap for the hexbin aggregation.
marker_size : int
    Size of the hotspot scatter markers.
alpha : float
    Transparency for the hotspot overlay points.
{params.base.show_grid}
{params.base.grid_props}
max_cols : int
    Maximum number of columns per row when laying out subplots.
savefig : Optional[str]
    Base filename for saving figures.
save_fmts : Union[str, List[str]]
    File extension(s) for saving.
prefix : str
    Suffix inserted before the file extension when saving.
verbose : int
    Verbosity level for `vlog()` messages.
cbar : str
    'uniform' shares one colorbar per figure; otherwise each
    subplot gets its own colorbar.
show_axis : Union[str,bool]
    If 'off' or False, hide axes via `ax.set_axis_off()`.
_logger : Optional[logging.Logger]
    Logger for verbosity messages.

Returns
-------
List[matplotlib.figure.Figure]
    One Figure object with hotspot maps arranged by time-slice.

Examples
--------
>>> figs = plot_hotspots(
...     df=my_df,
...     value_col='subsidence_q90',
...     dt_col='coord_t',
...     dt_values=[2020,2021,2022],
...     percentile=95.0,
...     cmap='Reds'
... )
>>> plt.show()

See Also
--------
plot_spatial          : Discrete scatter maps by time-slice.
plot_spatial_contours : Contour maps at specified quantiles.
plot_spatial_roi      : Region-of-interest scatter maps.
plot_spatial_voronoi  : Voronoi tessellation maps.
""".format(params=_param_docs)


@check_empty(['df'])
@isdf
def plot_spatial_voronoi(
    df: pd.DataFrame,
    value_col: str,
    spatial_cols: Optional[Tuple[str, str]] = None,
    dt_col: Optional[str] = None,
    dt_values: Optional[List[Union[int, str]]] = None,
    *,
    cmap: str = 'viridis',
    show_grid: bool = True,
    grid_props: Optional[dict] = None,
    max_cols: int = 3,
    savefig: Optional[str] = None,
    save_fmts: Union[str, List[str]] = '.png',
    prefix: str = '',
    verbose: int = 1,
    cbar: str = 'uniform',
    show_axis: Union[str, bool] = 'on',
    _logger: Optional[logging.Logger] = None
) -> List[plt.Figure]:
    def _v(msg: str, level: int = 1):
        vlog(message=msg, verbose=verbose, level=level, logger=_logger)

    exist_features(df, features=value_col, error='raise')
    spatial_cols = spatial_cols or ('coord_x', 'coord_y')
    spatial_cols = columns_manager(spatial_cols, empty_as_none=False)
    check_spatial_columns(df, spatial_cols=spatial_cols)
    x_col, y_col = spatial_cols

    if dt_col:
        dt_values = dt_values or sorted(df[dt_col].dropna().unique())
    elif dt_values is None:
        raise ValueError("Either 'dt_col' or 'dt_values' must be provided")



    n = len(dt_values)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        constrained_layout=True
    )
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    mosaics = []
    for idx, dt in enumerate(dt_values):
        ax = axes_flat[idx]
        df_dt = df[df[dt_col] == dt] if dt_col else df
        pts = np.vstack([df_dt[x_col], df_dt[y_col]]).T
        vals = df_dt[value_col].values

        # Compute Voronoi
        vor = Voronoi(pts)
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

        # Draw polygons
        for pt_idx, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue
            poly_pts = vor.vertices[region]
            color = mapper.to_rgba(vals[pt_idx])
            patch = Polygon(poly_pts, facecolor=color, edgecolor='k', 
                            linewidth=0.5)
            ax.add_patch(patch)

        # Overlay points
        ax.scatter(pts[:,0], pts[:,1], c='k', s=5)
        if show_grid:
            gp = grid_props or dict(linestyle=':', alpha=0.7)
            ax.grid(**gp)
        flag = (show_axis.lower() not in ('off','false')) if isinstance(
            show_axis, str) else bool(show_axis)
        if not flag:
            ax.set_axis_off()
        ax.set_title(f"Voronoi {value_col} @ {dt}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        mosaics.append(mapper)

    # hide unused axes
    for j in range(idx+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # colorbar
    if cbar == 'uniform':
        fig.colorbar(mosaics[0], ax=axes_flat[:idx+1], label=value_col)
    else:
        for ax_i, m_i in zip(axes_flat[:idx+1], mosaics):
            fig.colorbar(m_i, ax=ax_i, shrink=0.6, pad=0.02, label=value_col)

    # Save
    if savefig:
        fmts = [savefig] if isinstance(savefig, str) else save_fmts
        base, _ = os.path.splitext(savefig)
        for fmt in (fmts if isinstance(fmts, list) else [fmts]):
            ext = fmt if fmt.startswith('.') else '.' + fmt
            out = f"{base}{prefix}{ext}"
            os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
            _v(f"Saving voronoi plot to {out}")
            fig.savefig(out, dpi=300, bbox_inches='tight')

    return [fig]


plot_spatial_voronoi.__doc__ = r"""
Plot Voronoi (Thiessen) tessellation colored by 
sample values over time slices.

{params.base.spatial_cols}
{params.base.dt_col}
{params.base.dt_values}

value_col : str
    Column name containing the numeric values used to colour each
    Voronoi cell (e.g. 'subsidence_q50').

cmap : str
    Matplotlib colormap for filling the polygons.
show_grid : bool
    Whether to draw grid lines behind each subplot.
grid_props : Optional[dict]
    Custom properties for the grid lines (linestyle, alpha, etc.).
max_cols : int
    Maximum number of columns per row when laying out subplots.
prefix : str
    Suffix inserted before the file extension when saving.
verbose : int
    Verbosity level for `vlog()` messages.
cbar : str
    'uniform' shares one colorbar per figure; otherwise each
    subplot gets its own colorbar.
show_axis : Union[str,bool]
    If 'off' or False, hide axes via `ax.set_axis_off()`.
_logger : Optional[logging.Logger]
    Logger for verbosity messages.

Returns
-------
List[matplotlib.figure.Figure]
    One Figure object laid out as time-slices × cells.

Examples
--------
>>> figs = plot_spatial_voronoi(
...     df=my_df,
...     value_col='subsidence_q50',
...     dt_col='coord_t',
...     dt_values=[2024,2025],
...     cmap='plasma',
...     cbar='individual'
... )
>>> figs[0].show()

See Also
--------
plot_spatial          : Discrete scatter maps by time-slice.
plot_spatial_contours : Contour maps at specified quantiles.
plot_spatial_roi      : Region-of-interest scatter maps.
""".format(params=_param_docs)


@check_empty(['df'])
@isdf
def plot_spatial_heatmap(
    df: pd.DataFrame,
    value_col: str,
    spatial_cols: Optional[Tuple[str, str]] = None,
    dt_col: Optional[str] = None,
    dt_values: Optional[List[Union[int, str]]] = None,
    *,
    grid_res: int = 100,
    method: str = 'grid',  
    cmap: str = 'viridis',
    alpha: float = 0.8,
    show_points: bool = False,
    show_grid: bool = True,
    grid_props: Optional[dict] = None,
    max_cols: int = 3,
    savefig: Optional[str] = None,
    save_fmts: Union[str, List[str]] = '.png',
    prefix: str = '',
    verbose: int = 1,
    cbar: str = 'uniform',
    show_axis: Union[str, bool] = 'on',
    _logger: Optional[logging.Logger] = None
) -> List[plt.Figure]:
    """
    Plot a smooth spatial heatmap of `value_col` over a grid, faceted by time.
    """
    def _v(msg: str, level: int = 1):
        vlog(message=msg, verbose=verbose, level=level, logger=_logger)

    exist_features(df, features=value_col, error='raise')
    spatial_cols = spatial_cols or ('coord_x', 'coord_y')
    spatial_cols = columns_manager(spatial_cols, empty_as_none=False)
    check_spatial_columns(df, spatial_cols=spatial_cols)
    x_col, y_col = spatial_cols

    if dt_col:
        dt_values = dt_values or sorted(df[dt_col].dropna().unique())
    elif dt_values is None:
        raise ValueError("Either 'dt_col' or 'dt_values' must be provided")

    n = len(dt_values)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        constrained_layout=True
    )
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    imaps = []

    for idx, dt in enumerate(dt_values):
        ax = axes_flat[idx]
        df_dt = df[df[dt_col] == dt] if dt_col else df
        x = df_dt[x_col].values
        y = df_dt[y_col].values
        z = df_dt[value_col].values

        # define grid
        xi = np.linspace(x.min(), x.max(), grid_res)
        yi = np.linspace(y.min(), y.max(), grid_res)
        Xi, Yi = np.meshgrid(xi, yi)

        if method == 'grid':
            from scipy.interpolate import griddata
            Zi = griddata((x, y), z, (Xi, Yi), method='linear')
        elif method == 'kde':
            from scipy.stats import gaussian_kde
            # Shift weights to be non-negative
            weights = z.copy().astype(float)
            w_min = weights.min()
            if w_min < 0:
                weights = weights - w_min + 1e-6
            kde = gaussian_kde(np.vstack([x, y]), weights=weights)
            Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        else:
            raise ValueError("Unsupported method: choose 'grid' or 'kde'.")

        # plot heatmap
        im = ax.pcolormesh(
            Xi, Yi, Zi,
            cmap=cmap,
            shading='auto',
            alpha=alpha
        )
        imaps.append(im)

        if show_points:
            ax.scatter(x, y, c='k', s=5)
        if show_grid:
            gp = grid_props or dict(linestyle=':', alpha=0.7)
            ax.grid(**gp)

        flag = (show_axis.lower() not in ('off','false')) if isinstance(
            show_axis, str) else bool(show_axis)
        if not flag:
            ax.set_axis_off()

        ax.set_title(f"{value_col} heatmap @ {dt}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    # hide unused axes
    for j in range(idx+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # colorbar
    if cbar == 'uniform':
        fig.colorbar(imaps[0], ax=axes_flat[:idx+1], label=value_col)
    else:
        for ax_i, im_i in zip(axes_flat[:idx+1], imaps):
            fig.colorbar(im_i, ax=ax_i, shrink=0.6, pad=0.02, label=value_col)

    # save
    if savefig:
        fmts = [savefig] if isinstance(savefig, str) else save_fmts
        base, _ = os.path.splitext(savefig)
        for fmt in (fmts if isinstance(fmts, list) else [fmts]):
            ext = fmt if fmt.startswith('.') else '.' + fmt
            out = f"{base}{prefix}{ext}"
            os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
            _v(f"Saving heatmap plot to {out}")
            fig.savefig(out, dpi=300, bbox_inches='tight')

    return [fig]
