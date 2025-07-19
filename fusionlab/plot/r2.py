# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
r2 plots
"""

from __future__ import annotations 
import warnings 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import  r2_score 
from ..api.types import Optional, Tuple, Any, List
from ..api.types import Dict, ArrayLike

from ..core.array_manager import drop_nan_in
from ..core.checks import is_iterable
from ..metrics import get_scorer 
from ..utils.validator import validate_yy
from ..utils.validator import process_y_pairs


__all__= [ 
    'plot_r2', 'plot_r2_in', ]

def plot_r2_in(
    *ys: ArrayLike,
    titles: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fig_size: Optional[Tuple[int, int]] = None,
    scatter_colors: Optional[List[str]] = None,
    line_colors: Optional[List[str]] = None,
    line_styles: Optional[List[str]] = None,
    other_metrics: Optional[List[str]] = None,
    annotate: bool = True,
    show_grid: bool = True,
    grid_props: Dict[str, Any]=None, 
    max_cols: int = 3,
    fit_eq: bool = True,
    fit_line_color: str = 'k',
    **r2_score_kws: Any
) -> plt.Figure:
    r""" 
    Plot R² diagnostics for multiple (``y_true``, ``y_pred``) pairs with 
    advanced annotations.
    
    This function creates a grid of scatter plots comparing actual vs predicted 
    values across multiple datasets. Each subplot displays:
    
    1. Scatter plot of ``y_true`` vs ``y_pred``
    2. Perfect fit line (``y = x``)
    3. Linear regression fit (optional)
    4. Annotated metrics (R², RMSE, MAE, etc.)
    
    Parameters
    ----------
    *ys : ArrayLike
        Alternating sequence of (``y_true``, ``y_pred``) pairs. Requires even 
        number of inputs. Processed through ``process_y_pairs`` for validation 
        and NaN removal.
    titles : list of str, optional
        Subplot titles corresponding to each pair. Length should match number 
        of pairs. Default generates "Pair 1", "Pair 2", etc.
    xlabel : str, optional
        X-axis label for all subplots. Default: ``'Actual Values'``.
    ylabel : str, optional
        Y-axis label for all subplots. Default: ``'Predicted Values'``.
    fig_size : tuple of (int, int), optional
        Figure dimensions in inches. Auto-calculated based on grid dimensions 
        if ``None``.
    scatter_colors : list of str, optional
        Colors for scatter points in each subplot. Cycles last color if 
        insufficient colors provided.
    line_colors : list of str, optional
        Colors for perfect fit lines. Default: ``['red'] * n_pairs``.
    line_styles : list of str, optional
        Linestyles for perfect fit lines. Default: ``['--'] * n_pairs``.
    other_metrics : list of str, optional
        Additional metrics to display. Valid options: ``'rmse'``, ``'mae'``, or 
        any scikit-learn scorer name.
    annotate : bool, default=True
        Whether to display metrics annotations on subplots.
    show_grid : bool, default=True
        Toggle grid display in subplots.
    max_cols : int, default=3
        Maximum columns in subplot grid. Rows auto-adjusted accordingly.
    fit_eq : bool, default=True
        Display linear regression equation for each pair.
    fit_line_color : str, default='k'
        Color for regression line when ``line_eq=True``.
    **r2_score_kws : dict
        Additional arguments for ``sklearn.metrics.r2_score`` calculation.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object containing all subplots.
    
    Raises
    ------
    ValueError
        - If processed pairs return empty after validation
        - Invalid color/list length mismatches with warnings suppressed
    
    Examples
    --------
    Basic usage with synthetic data:
    
    >>> from fusionlab.plot.r2 import plot_r2_in
    >>> import numpy as np
    >>> y_true = np.random.rand(100)
    >>> y_pred1 = y_true + np.random.normal(0, 0.1, 100)
    >>> y_pred2 = y_true * 1.1 + np.random.normal(0, 0.2, 100)
    >>> fig = plot_r2_in(
    ...     y_true, y_pred1, y_true, y_pred2,
    ...     titles=['Model A', 'Model B'],
    ...     other_metrics=['rmse', 'mae'],
    ...     fit_line_color='navy'
    ... )
    >>> fig.savefig('model_comparison.png')
    
    Advanced validation scenario:
    
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, noise=10)
    >>> train_X, test_X = X[:150], X[150:]
    >>> train_y, test_y = y[:150], y[150:]
    >>> # Simulate two different models
    >>> preds1 = train_X.dot(np.random.randn(train_X.shape[1]))
    >>> preds2 = test_X.dot(np.random.randn(test_X.shape[1]) * 0.8
    >>> plot_r2_in(
    ...     train_y, preds1, test_y, preds2,
    ...     ops='validate',  # Enable data cleaning
    ...     fit_eq=False    # Disable regression lines
    ... )
    
    Notes
    -----
    1. Underlying data validation uses:
       - ``process_y_pairs`` for pair alignment and NaN handling
       - ``r2_score`` for coefficient calculation [1]_
    2. Regression lines use NumPy's polyfit with degree=1
    3. For large datasets (>10⁴ points), consider setting ``annotate=False``
       for better rendering performance
       
    .. math::
        R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}
        
        \text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|
        
        \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}
    
    See Also
    --------
    process_y_pairs : Core validation and pairing utility
    plot_residuals : Diagnostic plots for regression residuals
    sklearn.metrics.r2_score : Scikit-learn's R² implementation
    
    References
    ----------
    .. [1] Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, 
       pp. 2825-2830, 2011.
    .. [2] Hunter, J. D. Matplotlib: A 2D Graphics Environment. Computing in 
       Science & Engineering 9.3 (2007): 90-95.
    """
    # Validate input lengths
    y_trues, y_preds = process_y_pairs(
        *ys, 
        ops="validate", 
        error='warn', 
        
        )
    if len(y_trues) == 0 or len(y_trues)==0:
        raise ValueError("No valid data pairs to plot.")
        
    # Determine how many pairs of (y_true, y_pred) to plot
    n_pairs = min(len(y_trues), len(y_preds))

    # Prepare subplots arrangement based on max_cols
    ncols = min(max_cols, n_pairs) if n_pairs > 0 else 1
    nrows = int(np.ceil(n_pairs / ncols)) if n_pairs > 0 else 1

    # Convert titles to a list if provided
    if titles is not None:
        # Ensure it's iterable if not already
        titles = is_iterable(
            titles, exclude_string=True, 
            transform=True
            )

    # Build default scatter colors if needed
    if scatter_colors is None:
        scatter_colors = ['blue'] * n_pairs
    else:
        # Ensure scatter_colors is at 
        # least as long as the number of pairs
        scatter_colors = is_iterable(
            scatter_colors, exclude_string=True, 
            transform=True
            )
        if len(scatter_colors) < n_pairs:
            scatter_colors += [scatter_colors[-1]] * (
                n_pairs - len(scatter_colors))

    # Build default line colors if needed
    if line_colors is None:
        line_colors = ['red'] * n_pairs
    else:
        line_colors = is_iterable(
            line_colors, 
            exclude_string=True, 
            transform=True
            )
        if len(line_colors) < n_pairs:
            line_colors += [line_colors[-1]] * (
                n_pairs - len(line_colors))

    # Build default line styles if needed
    if line_styles is None:
        line_styles = ['--'] * n_pairs
    else:
        line_styles = is_iterable(
            line_styles, exclude_string=True, 
            transform=True
            )
        if len(line_styles) < n_pairs:
            line_styles += [line_styles[-1]] * (
                n_pairs - len(line_styles))

    # Determine figure size if none given
    if fig_size is None:
        base_width = 5
        base_height = 4
        fig_width = base_width * ncols
        fig_height = base_height * nrows
        fig_size = (fig_width, fig_height)

    # Create subplots
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=fig_size,
        squeeze=False
    )
    axes_flat = axes.flatten()

    # Initialize empty lists for metrics
    metrics_values = []
    valid_metrics = []

    for idx in range(n_pairs):
        # Access the current y_true and y_pred
        y_true = y_trues[idx]
        y_pred = y_preds[idx]

        # Current axis
        ax = axes_flat[idx]

        # Compute r2
        r_squared = r2_score(y_true, y_pred, **r2_score_kws)

        # Compute other metrics if requested
        if other_metrics is not None:
            for metric in other_metrics:
                try:
                    value = get_scorer(metric)(y_true, y_pred)
                except Exception as e:
                    warnings.warn(str(e))
                    continue
                metrics_values.append(value)
                valid_metrics.append(metric)

        # Scatter plot
        ax.scatter(y_true, y_pred, color=scatter_colors[idx],
                   label='Predictions vs Actual data')

        # Draw the perfect fit line
        perfect_min = min(y_true.min(), y_pred.min())
        perfect_max = max(y_true.max(), y_pred.max())
        perfect_line = [perfect_min, perfect_max]
        ax.plot(perfect_line, perfect_line,
                color=line_colors[idx],
                linestyle=line_styles[idx],
                label='Perfect fit')


        # Annotate R^2 and other metrics if needed
        i_m=0
        if annotate:
            ax.text(0.95, 0.05,
                    f'$R^2 = {r_squared:.2f}$',
                    fontsize=12,
                    ha='right',
                    va='bottom',
                    transform=ax.transAxes)
            if other_metrics and valid_metrics:
                for i_m, metric in enumerate(valid_metrics):
                    ax.text(0.95, 0.05 + (i_m + 1) * 0.1,
                            f'${metric} = {metrics_values[i_m]:.2f}$',
                            transform=ax.transAxes,
                            fontsize=12,
                            va='bottom',
                            ha='right',
                            color='black')
                # Reset metrics for next subplot
                metrics_values = []
                valid_metrics = []
                
        # Optionally compute and annotate the best-fit line equation
        if fit_eq:
            # Perform a linear fit
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            # Plot it (optional line style - reusing line_colors)
            ax.plot([perfect_min, perfect_max],
                    [slope*perfect_min + intercept,
                     slope*perfect_max + intercept],
                    color=fit_line_color,
                    label='Fitted line'
                    )
            # Add text with line equation
            eq_str = f'$y = {slope:.4f}x + {intercept:.4f}$'
            ax.text(0.95, 0.05  + (i_m + 2) * 0.1, 
                    eq_str,
                    transform=ax.transAxes,
                    fontsize=12,
                    va='bottom', #'top',
                    ha='right', # 'left',
                    color=fit_line_color
                )
            
        # Apply axis labels
        ax.set_xlabel(xlabel or 'Actual Values')
        ax.set_ylabel(ylabel or 'Predicted Values')

        # Subplot title if any
        try:
            sub_title = titles[idx] if titles else f"Pair {idx+1}"
            ax.set_title(sub_title)
        except:
            ax.set_title(f"Pair {idx+1}")

        # Grid if requested
        if show_grid: 
            ax.grid(
                True , **(grid_props or {'linestyle':':', 'alpha': 0.7}))
        else: 
            ax.grid(False) 

        # Legend
        ax.legend(loc='upper left')

    # Hide unused subplots, if any
    for idx_unused in range(n_pairs, len(axes_flat)):
        axes_flat[idx_unused].axis('off')

    # Adjust layout
    fig.tight_layout()

    # Show and return
    plt.show()
    
    return fig

def plot_r2(
    y_true: ArrayLike, 
    *y_preds: ArrayLike, 
    titles: Optional[str] = None,  
    xlabel: Optional[str] = None, 
    ylabel: Optional[str] = None,  
    fig_size: Optional[Tuple[int, int]] = None,
    scatter_colors: Optional[List[str]] = None, 
    line_colors: Optional[List[str]] = None, 
    line_styles: Optional[List[str]] = None, 
    other_metrics: List[str]=None, 
    annotate: bool = True, 
    show_grid: bool = True, 
    grid_props: Dict[str, Any]=None, 
    max_cols: int = 3,
    **r2_score_kws: Any
) -> plt.Figure:
    """
    Plot R-squared scatter plots comparing actual values with 
    multiple predictions.
    
    This function generates a series of scatter plots comparing the true values
    (`y_true`) with multiple predicted values (`y_pred`). Each prediction is
    plotted against the true values in its own subplot, arranged in a grid layout
    based on the `max_cols` parameter. The function calculates and annotates the
    R-squared value for each prediction to assess model performance.
    
    .. math::
        R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}
    
    where :math:`SS_{res}` is the sum of squares of residuals and
    :math:`SS_{tot}` is the total sum of squares.
    
    Parameters
    ----------
    y_true : ArrayLike
        The true target values. Should be an array-like structure such as 
        a list, NumPy array, or pandas Series.
    
    *y_preds : ArrayLike
        One or more predicted target values. Each should be an array-like 
        structure corresponding to `y_true`.
    
    titles : Optional[str], default=None
        The overall titles for the each figure. If ``None``, no overarching
         title is set.
    
    xlabel : Optional[str], default=None
        The label for the x-axis of each subplot. Defaults to ``'Actual Values'``
        if ``None``.
    
    ylabel : Optional[str], default=None
        The label for the y-axis of each subplot. Defaults to ``'Predicted Values'``
        if ``None``.
    
    fig_size : Optional[Tuple[int, int]], default=None
        The size of the entire figure in inches. If ``None``, the figure size is
        automatically determined based on the number of subplots to ensure an
        aesthetically pleasing layout.
    
    scatter_colors : Optional[List[str]], default=None
        A list of colors for the scatter points in each subplot. If ``None``,
        defaults to ``'blue'`` for all scatter plots. If fewer colors are provided
        than the number of predictions, the last color is repeated.
    
    line_colors : Optional[List[str]], default=None
        A list of colors for the perfect fit lines in each subplot. If ``None``,
        defaults to ``'red'`` for all lines. If fewer colors are provided
        than the number of predictions, the last color is repeated.
    
    line_styles : Optional[List[str]], default=None
        A list of line styles for the perfect fit lines in each subplot. If ``None``,
        defaults to dashed lines (``'--'``) for all lines. If fewer styles are
        provided than the number of predictions, the last style is repeated.
    view_other_metrics: bool, default=False 
       Display others metrics like Root-Mean Squared Error (RMSE) and 
       Mean Absolute Error (MAE) on the figure. 
       
    annotate : bool, default=True
        Whether to annotate each subplot with its corresponding R-squared value.
        Annotations are positioned at the bottom right corner of each subplot to
        avoid overlapping with the legend.
    
    show_grid : bool, default=True
        Whether to display grid lines on each subplot.
    
    max_cols : int, default=3
        The maximum number of columns in the subplot grid. Determines how many
        subplots are placed in each row before moving to a new row.
    
    **r2_score_kws : Any
        Additional keyword arguments to pass to the ``sklearn.metrics.r2_score``
        function.
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object containing all the subplots.
    
    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.plot.r2  import plot_r2
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Generate synthetic true values and predictions
    >>> y_true = np.linspace(0, 100, 50)
    >>> y_pred1 = y_true + np.random.normal(0, 10, 50)
    >>> y_pred2 = y_true + np.random.normal(0, 20, 50)
    >>> y_pred3 = y_true + np.random.normal(0, 5, 50)
    >>> y_pred4 = y_true + np.random.normal(0, 15, 50)
    >>> 
    >>> # Plot R-squared for multiple predictions
    >>> plot_r2(
    ...     y_true, y_pred1, y_pred2, y_pred3, y_pred4,
    ...     title='Model Performance Comparison',
    ...     xlabel='Actual Values',
    ...     ylabel='Predicted Values',
    ...     scatter_colors=['green', 'orange', 'purple', 'cyan'],
    ...     line_colors=['black', 'black', 'black', 'black'],
    ...     line_styles=['-', '-', '-', '-'],
    ...     annotate=True,
    ...     show_grid=True,
    ...     max_cols=2
    ... )
    >>> plt.show()
    
    Notes
    -----
    - **Multiple Predictions Visualization**: This function allows for the 
      simultaneous visualization of multiple model predictions against actual 
      values, facilitating comparative analysis of different models or 
      configurations.
    
    - **Dynamic Layout Adjustment**: By specifying the ``max_cols`` parameter,
      users can control the number of columns in the subplot grid, ensuring 
      that the plots are organized neatly irrespective of the number of 
      predictions.
    
    - **Customization Flexibility**: The ability to customize scatter colors, 
      line colors, and line styles for each subplot allows for clear 
      differentiation between multiple predictions, especially when dealing 
      with a large number of plots.
    
    See Also
    --------
    - :func:`sklearn.metrics.r2_score` : 
        Function to compute the R-squared, or
      coefficient of determination.
    - :func:`matplotlib.pyplot.subplots` : 
        Function to create multiple subplots.
    - :func:`gofast.plot.utils.plot_r_squared` : 
        Utility function to plot R-squared in multiple kind format.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
           Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A.,
           Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: 
           Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
    
    .. [2] Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. 
           *Computing in Science & Engineering*, 9(3), 90-95.
    """

    # Remove NaN values from y_true and all y_pred arrays
    y_true, *y_preds = drop_nan_in(
        y_true, *y_preds, error='raise', reset_index=True)
    
    # Validate y_true and each y_pred to ensure consistency and continuity
    y_preds = [
        validate_yy(y_true, pred, expected_type="continuous",
                    flatten="auto")[1] 
        for pred in y_preds
    ]
    
    # Determine the number of predictions to plot
    num_preds = len(y_preds)

    # Calculate the number of columns and rows based on max_cols and num_preds
    ncols = min(max_cols, num_preds) if num_preds > 0 else 1
    nrows = int(np.ceil(num_preds / ncols)) if num_preds > 0 else 1

    # Set default scatter colors if not provided
    if scatter_colors is None:
        scatter_colors = ['blue'] * num_preds
    else:
        # If fewer colors are provided than predictions, repeat the last color
        scatter_colors= is_iterable(scatter_colors, exclude_string=True, transform=True)
        if len(scatter_colors) < num_preds:
            scatter_colors += [scatter_colors[-1]] * (num_preds - len(scatter_colors))
    
    # Set default line colors if not provided
    if line_colors is None:
        line_colors = ['red'] * num_preds
    else:
        line_colors= is_iterable(line_colors, exclude_string=True, transform=True)
        # If fewer colors are provided than predictions, repeat the last color
        if len(line_colors) < num_preds:
            line_colors += [line_colors[-1]] * (num_preds - len(line_colors))
    
    # Set default line styles if not provided
    if line_styles is None:
        line_styles = ['--'] * num_preds
    else:
        # If fewer styles are provided than predictions, repeat the last style
        if len(line_styles) < num_preds:
            line_styles += [line_styles[-1]] * (num_preds - len(line_styles))
    
    # Determine figure size if not provided
    if fig_size is None:
        # Define base size for each subplot
        base_width = 5  # inches per subplot width
        base_height = 4  # inches per subplot height
        # Calculate total figure size based on number of rows and cols
        fig_width = base_width * ncols
        fig_height = base_height * nrows
        fig_size = (fig_width, fig_height)
    
    # Create subplots with the calculated number of rows and columns
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=fig_size,
        # Automatically adjust subplot params for a neat layout
        # constrained_layout=True, 
        # Ensures axes is always a 2D array for consistent indexing
        squeeze=False  
    )
 
    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    # list the titles if not None: 
    if titles is not None: 
        titles = is_iterable(titles, exclude_string= True, transform=True )
        
    metrics_values =[] 
    valid_metrics =[] 
    
    for idx, pred in enumerate(y_preds):
        # Determine the current subplot's row and column
        ax = axes_flat[idx]
        
        # Calculate R-squared value for the current prediction
        r_squared = r2_score(y_true, pred, **r2_score_kws)
        
        if other_metrics is not None:
            for metric in other_metrics : 
                try: 
                    value = get_scorer(metric)(y_true, pred)
                except Exception as e : 
                    warnings.warn(str(e))
                    continue 
                
                metrics_values.append(value)
                valid_metrics.append (metric)
        # Plot actual vs predicted values as a scatter plot
        ax.scatter(
            y_true, pred, 
            color=scatter_colors[idx], 
            label='Predictions vs Actual data'
        )
        
        # Determine the range for the perfect fit line
        perfect_min = min(y_true.min(), pred.min())
        perfect_max = max(y_true.max(), pred.max())
        perfect_line = [perfect_min, perfect_max]
        
        # Plot the perfect fit line (diagonal line)
        ax.plot(
            perfect_line, 
            perfect_line, 
            color=line_colors[idx],
            linestyle=line_styles[idx],
            label='Perfect fit'
        )
        
        # Annotate the R-squared value on the plot if requested
        if annotate:
            # Position the annotation at the bottom 
            # right to avoid overlapping with the legend
            ax.text(
                0.95, 0.05, f'$R^2 = {r_squared:.2f}$', 
                fontsize=12, ha='right', va='bottom', 
                transform=ax.transAxes
            )
            if other_metrics and valid_metrics: 
                for ii, metric in enumerate (valid_metrics): 
                    # Add text with others metrics on the plot
                    ax.text( 0.95 , 0.05 + ( ii + 1 ) * 0.1,
                            f'${metric} = {metrics_values[ii]:.2f}$', 
                             transform=ax.transAxes, 
                             fontsize=12,
                             va='bottom',
                             ha='right', 
                             color='black'
                    )
                # Initialize the list 
                metrics_values =[] 
                valid_metrics =[] 
                
        # Set axis labels; use provided labels or default ones
        ax.set_xlabel(xlabel or 'Actual Values')
        ax.set_ylabel(ylabel or 'Predicted Values')
        
        # Set subplot title, optionally including the overall title
        try: 
            subplot_title = ( 
                f"{titles[idx]}" if titles else f"Prediction {idx + 1}"
                )
            ax.set_title(subplot_title)
        except : 
            # when title is less than numberpred 
            ax.set_title('Prediction {idx + 1}')
        
        if show_grid: 
            ax.grid(
                True , **(grid_props or {'linestyle':':', 'alpha': 0.7}))
        else: 
            ax.grid(False) 
            
        # Enable grid lines if requested
        ax.grid(show_grid)
        
        # Add legend to the subplot
        ax.legend(loc='upper left')
    
    # Hide Any Unused Subplots
    # If the number of predictions is less than total
    # subplots, hide the unused axes
    for idx in range(num_preds, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    # Adjust layout to prevent overlapping elements
    fig.tight_layout()
    
    # Display the plot
    plt.show()
    
    # Return the figure object 
    # for further manipulation if needed
    return fig 


