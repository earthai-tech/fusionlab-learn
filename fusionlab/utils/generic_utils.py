# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides common helper functions and for validation, 
comparison, and other generic operations
"""
import os
import re
import warnings
import inspect
import textwrap
from numbers import Real 

import matplotlib.pyplot as plt
from datetime import datetime
from typing import ( 
    Union, Optional, 
    Dict, Any, List,
    Literal, Tuple
)
import numpy as np 
import pandas as pd 

__all__ =['verify_identical_items', 'vlog', 'detect_dt_format',
          'get_actual_column_name', 'transform_contributions', 
          'exclude_duplicate_kwargs', 'reorder_columns',
          'find_id_column', 'check_group_column_validity', 
          'save_all_figures']

def check_group_column_validity(
    df: pd.DataFrame,
    group_col: str,
    ops: str = 'check_only',  
    max_unique: int = 10,
    auto_bin: bool = False,
    bins: int = 4,
    error: str = "warn",      
    bin_labels: Optional[List[str]] = None,
    verbose: bool = True
) -> Union[pd.DataFrame, bool]:
    """
    Checks and optionally transforms a numeric group column,
    providing flexibility for categorical plots or group-based
    operations. Depending on ``ops``, it may simply validate,
    apply binning, or return a boolean status of validity.

    Internally, the function compares the number of unique
    values in `group_col` to ``max_unique``. If the column is
    numeric and exceeds this threshold, it may require binning
    or triggers a warning/error based on ``error``.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame holding the data to be examined.
    group_col : str
        Name of the column in ``df`` that may be treated as
        a grouping or categorical variable in plots.
    ops : {'check_only', 'binning', 'validate'}, optional
        Defines the operation mode:
        - ``"check_only"`` : Returns a boolean indicating
          whether `group_col` is valid as categorical.
        - ``"binning"`` : Bins `group_col` if necessary,
          returning a modified DataFrame.
        - ``"validate"`` : Acts like ``"check_only"``, but
          raises or warns if invalid, depending on
          ``error``.
    max_unique : int
        Maximum allowable unique numeric values to consider
        `group_col` categorical.
    auto_bin : bool
        Whether to auto-bin if `group_col` is invalid and
        ``ops='binning'``. If False, a warning/error may
        appear.
    bins : int
        Number of quantile bins to create if binning is
        used. Must be an integer >= 1.
    error : {'warn', 'raise', 'ignore'}, optional
        Determines how validation issues are handled:
        - ``"warn"`` : Prints a warning message.
        - ``"raise"`` : Raises a ValueError.
        - ``"ignore"`` : Does nothing.
    bin_labels : list of str, optional
        Custom labels for the binned categories if used.
        If None, default labels like "Q1", "Q2", etc. are
        generated.
    verbose : bool, optional
        Whether to print informational messages when
        transformations or warnings occur.

    Returns
    -------
    bool or pandas.DataFrame
        - If ``ops="check_only"``, a boolean indicating
          whether `group_col` can be used as a category.
        - Otherwise, a pandas DataFrame (possibly with
          a transformed `group_col`).

    Notes
    -----
    If ``ops="validate"`` and `group_col` is numeric with
    many unique values, the function may raise or warn,
    based on the ``error`` argument. If ``auto_bin=True``,
    it automatically switches to ``"binning"``.
    
    Mathematically, if the user chooses quantile binning, then
    :math:`bins` equally spaced quantiles are computed:

    .. math::
       Q_i = \\text{quantile}\\bigl(
       \\frac{i}{\\text{bins}}\\bigr),
       \\quad i = 1,2,\\ldots,\\text{bins}

    where each :math:`Q_i` is the i-th quantile boundary of
    the distribution of `group_col`.

    Examples
    --------
    >>> from gofast.utils.generic_utils import check_group_column_validity
    >>> import pandas as pd
    >>> data = {
    ...     "value": [10.5, 11.2, 9.8, 15.6, 12.0],
    ...     "category": ["A", "B", "B", "A", "C"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> # Simple check if 'category' can be used
    >>> # as a grouping column
    >>> is_valid = check_group_column_validity(
    ...     df, "category", ops="check_only"
    ... )
    >>> print(is_valid)
    True

    >>> # Binning a numeric column with auto_bin
    >>> result_df = check_group_column_validity(
    ...     df, "value", ops="binning", auto_bin=True
    ... )
    >>> print(result_df["value"])

    See Also
    --------
    pd.qcut : Pandas method used internally for creating
        quantile-based bins from numeric data.

    References
    ----------
    .. [1] T. Hastie, R. Tibshirani, and J. Friedman.
       "The Elements of Statistical Learning:
       Data Mining, Inference, and Prediction."
       Springer Series in Statistics, 2009.
    """

    # Check if group_col is in the DataFrame.
    if group_col not in df.columns:
        raise ValueError(
            f"[ERROR] Column '{group_col}' not found "
            f"in DataFrame."
        )

    # Extract data from the column and check if numeric.
    col_data = df[group_col]
    is_numeric = pd.api.types.is_numeric_dtype(
        col_data
    )
    # If numeric, ensure we haven't exceeded max_unique.
    is_valid_group = (
        not is_numeric
        or col_data.nunique()
          <= max_unique
    )

    # Validate ops argument.
    if ops not in {
        'check_only',
        'binning',
        'validate'
    }:
        raise ValueError(
            f"Unknown 'ops' value: {ops}. Choose from "
            f"'check_only', 'binning', 'validate'."
        )

    # If user only wants a check, return boolean.
    if ops == 'check_only':
        return is_valid_group

    # If 'validate', decide whether to warn/raise
    # or fallback to binning if auto_bin is True.
    elif ops == 'validate':
        if is_valid_group:
            return df
        msg = (
            f"Column '{group_col}' is numeric with "
            f"{col_data.nunique()} unique values. "
            "Not suitable for grouping in plots."
        )
        if auto_bin:
            ops = 'binning'  # fallback
        else:
            if error == 'raise':
                raise ValueError(msg)
            elif error == 'warn':
                warnings.warn(f"{msg}")
            return df

    # If 'binning', we attempt to transform the column.
    if ops == 'binning':
        # If already valid, no change needed.
        if is_valid_group:
            return df

        # If auto_bin is True, we do quantile binning.
        if auto_bin:
            if bin_labels is None:
                bin_labels = [
                    f"Q{i+1}"
                    for i in range(bins)
                ]
            df[group_col] = pd.qcut(
                col_data,
                q=bins,
                labels=bin_labels
            )
            if verbose:
                print(
                    f"[INFO] Auto-binned '{group_col}' "
                    f"into {bins} quantile-based categories."
                )
            return df
        else:
            # If auto_bin is False and data is invalid,
            # we handle per 'error'.
            if error == 'raise':
                raise ValueError(
                    "Auto-binning disabled, and group "
                    "column is not suitable."
                )
            elif error == 'warn':
                warnings.warn(
                    "Group column is not categorical "
                    "and was not binned."
                )
            return df

def find_id_column(
    df: pd.DataFrame,
    strategy: Literal[
        'naive',
        'exact',
        'dtype',
        'regex',
        'prefix_suffix'
    ] = 'naive',
    regex_pattern: Optional[str] = None,
    uniqueness_threshold: float = 0.95,
    errors: Literal['raise', 'warn', 'ignore'] = 'raise',
    empty_as_none: bool = True,
    as_list: bool = False,
    case_sensitive: bool = False,
    as_frame: bool = False
) -> Union[str, List[str], pd.DataFrame, None]:
    """
    Identify potential ID column(s) in a pandas DataFrame
    using multiple heuristic strategies.
    
    The function examines column names and/or data properties
    to detect columns likely to serve as unique identifiers.
    This is particularly useful for large datasets where the
    ID field is not explicitly labeled, and for quick scanning
    of possible key columns [1]_.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame in which to search for
        potential ID columns.
    strategy : {'naive', 'exact', 'dtype', 'regex',
                'prefix_suffix'}, default 'naive'
        Defines the logic for detecting ID columns:
        - `exact`: Checks for a column name that exactly
          matches `id` (case sensitivity controlled by
          ``case_sensitive``).
        - `naive`: Searches for columns where `id` is
          part of the name (e.g., `location_id`) subject
          to case sensitivity.
        - `prefix_suffix`: Considers columns prefixed
          or suffixed with `id` or `_id`.
        - `dtype`: Examines columns having data types
          commonly used for IDs (integer, string, or
          object) and checks if they show high uniqueness
          via :math:`\\text{uniqueness\\_ratio}
          \\geq \\text{uniqueness\\_threshold}`.
        - `<regex>`: Uses a custom regular expression
          ``<regex_pattern>`` to find matches in column
          names.
    regex_pattern : str, optional
        Required if `strategy` is `'regex'`. The
        pattern is compiled via `re.compile`, with case
        sensitivity determined by `<case_sensitive>`.
    uniqueness_threshold : float, default 0.95
        For `<dtype>` strategy, columns are flagged as ID
        candidates if the ratio:
    
        .. math::
           r = \\frac{
                  \\text{unique\\_values}
              }{
                  \\text{non\\_NA\\_rows}
              }
    
        satisfies :math:`r \\geq \\text{uniqueness\\_threshold}`,
        or if the number of unique values equals the number
        of non-null rows.
    errors : {'raise', 'warn', 'ignore'}, default 'raise'
        How to handle no-match cases:
          - `raise`: Raises a `ValueError`.
          - `warn`: Issues a `UserWarning` and returns
            based on `<as_frame>` or `<empty_as_none>`.
          - `ignore`: Returns an empty result based on
            the same parameters without warning.
    empty_as_none : bool, default True
        *Applies only if `as_frame` is False.* Defines
        whether to return `None` (if True) or an empty list
        (if False) when no ID column is found and
        `<errors>` is `'warn'` or `'ignore'`.
    as_list : bool, default False
        If True, return all matched columns. If False,
        return only the first match. Affects both name
        returns and DataFrame returns.
    case_sensitive : bool, default False
        If False, comparisons (including regex) are
        performed in a case-insensitive manner.
    as_frame : bool, default False
        If True, return the matched columns as a
        pandas DataFrame. If `as_list` is True,
        it may include multiple columns. If no column
        is found, returns an empty DataFrame (if
        `<errors>` is `'warn'` or `'ignore'`).
    
    Returns
    -------
    str or List[str] or pandas.DataFrame or None
        Depends on `as_frame`, `as_list`, and the
        number of matching columns:
        - `<as_frame>`=False, `as_list`=False:
          returns the first match as a string,
          or `None`/`[]`.
        - `as_frame`=False, `as_list`=True:
          returns all matching column names as
          a list of strings.
        - `as_frame`=True, `as_list`=False:
          returns a DataFrame with the first matched
          column. If no match is found, an empty
          DataFrame may be returned.
        - `as_frame`=True, `as_list`=True:
          returns a DataFrame with all matched columns
          included.
    
    Notes
    -----
    - For `<dtype>` strategy, integer, string, and
      object columns are inspected. The function
      calculates a uniqueness ratio and compares it
      against `<uniqueness_threshold>`.
    - Negative or zero thresholds are invalid, as are
      values above 1.
    - If the DataFrame has no columns or is empty, the
      behavior is determined by `<errors>`.
    
    Examples
    --------
    >>> from gofast.utils.generic_utils import find_id_column
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'ID_code': [101, 102, 103],
    ...     'Name': ['Alice', 'Bob', 'Charlie'],
    ...     'value': [10, 20, 30]
    ... })
    >>> # Example using the 'naive' strategy
    >>> col = find_id_column(data, strategy='naive')
    >>> print(col)  # Might return 'ID_code'
    >>> # Example with as_list=True
    >>> cols = find_id_column(data, strategy='naive',
    ...                       as_list=True)
    >>> print(cols)  # ['ID_code']
    
    See Also
    --------
    re.compile : The regex compilation method used
                   when `strategy`='regex'.
    pandas.api.types.is_integer_dtype : Checks integer type.
    pandas.api.types.is_string_dtype : Checksstring type.
    pandas.api.types.is_object_dtype : Checksobject type.
    
    References
    ----------
    .. [1] E. F. Codd (1970). "A Relational Model of Data
           for Large Shared Data Banks."
    """
    # Validate that df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Input 'df' must be a pandas DataFrame."
        )

    valid_strategies = [
        'naive',
        'exact',
        'dtype',
        'regex',
        'prefix_suffix'
    ]
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Must be one "
            f"of {valid_strategies}."
        )

    valid_errors = ['raise', 'warn', 'ignore']
    if errors not in valid_errors:
        raise ValueError(
            f"Invalid errors value '{errors}'. Must be one "
            f"of {valid_errors}."
        )

    # If strategy='regex', ensure regex_pattern is valid
    if strategy == 'regex':
        if not regex_pattern or not isinstance(
           regex_pattern, str):
            raise ValueError(
                "Parameter 'regex_pattern' must be a "
                "non-empty string when strategy is "
                "'regex'."
            )

    # If strategy='dtype', ensure threshold in [0,1]
    if strategy == 'dtype':
        if not (0.0 <= uniqueness_threshold <= 1.0):
            raise ValueError(
                "Parameter 'uniqueness_threshold' must be "
                "between 0.0 and 1.0."
            )

    # If DataFrame is empty, handle accordingly
    if df.empty or len(df.columns) == 0:
        msg = ("DataFrame is empty or has no columns. "
               "Cannot find ID column.")
        if errors == 'raise':
            raise ValueError(msg)
        elif errors == 'warn':
            warnings.warn(msg, UserWarning)
        if as_frame:
            return pd.DataFrame()
        else:
            return [] if not empty_as_none else None

    # Prepare columns for matching
    original_columns = df.columns.tolist()
    if not case_sensitive:
        col_map = {}
        seen_lower = set()
        for c in original_columns:
            lc = c.lower()
            if lc not in seen_lower:
                col_map[lc] = c
                seen_lower.add(lc)
        match_columns_keys = list(col_map.keys())
    else:
        col_map = {c: c for c in original_columns}
        match_columns_keys = original_columns

    found_matches_keys = []

    # -------------- Strategy: 'exact' --------------
    if strategy == 'exact':
        target = 'id' if not case_sensitive else 'id'
        if target in match_columns_keys:
            found_matches_keys.append(target)

    # -------------- Strategy: 'naive' --------------
    elif strategy == 'naive':
        target = 'id' if not case_sensitive else 'id'
        found_matches_keys = [
            mc_key for mc_key in match_columns_keys
            if target in mc_key
        ]

    # -------------- Strategy: 'prefix_suffix' -------
    elif strategy == 'prefix_suffix':
        targets = ['id', '_id']
        if not case_sensitive:
            targets = [t.lower() for t in targets]
        for mc_key in match_columns_keys:
            match = False
            for t in targets:
                if (mc_key.startswith(t)
                   or mc_key.endswith(t)):
                    match = True
                    break
            if match:
                found_matches_keys.append(mc_key)

    # -------------- Strategy: 'regex' --------------
    elif strategy == 'regex':
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(regex_pattern, flags)
            original_regex_matches = [
                c for c in original_columns
                if pattern.search(c)
            ]
            if not case_sensitive:
                found_matches_keys = [
                    k for k, v in col_map.items()
                    if v in original_regex_matches
                ]
            else:
                found_matches_keys = [
                    k for k in original_regex_matches
                    if k in match_columns_keys
                ]
        except re.error as e:
            raise ValueError(
                f"Invalid regex pattern provided: {e}"
            ) from e

    # -------------- Strategy: 'dtype' --------------
    elif strategy == 'dtype':
        for col_name in original_columns:
            col_series = df[col_name]
            # Check if type is integer, string, or object
            is_potential_type = (
                pd.api.types.is_integer_dtype(col_series)
                or pd.api.types.is_string_dtype(col_series)
                or pd.api.types.is_object_dtype(col_series)
            )
            if is_potential_type:
                non_na = col_series.dropna()
                if len(non_na) > 0:
                    if pd.api.types.is_object_dtype(
                       non_na):
                        num_unique = (
                            non_na.astype(str)
                            .nunique()
                        )
                    else:
                        num_unique = non_na.nunique()
                    uniqueness_ratio = (
                        num_unique / len(non_na)
                    )
                    is_perfectly_unique = (
                        num_unique == len(non_na)
                    )
                    if (uniqueness_ratio
                       >= uniqueness_threshold
                       or is_perfectly_unique):
                        key_to_add = (
                            col_name.lower()
                            if not case_sensitive
                            else col_name
                        )
                        if key_to_add in col_map:
                            if key_to_add not in (
                               found_matches_keys):
                                found_matches_keys.append(
                                    key_to_add
                                )

    # Remove duplicates while preserving order
    ordered_unique_matches_keys = sorted(
        list(set(found_matches_keys)),
        key=found_matches_keys.index
    )
    original_matches = [
        col_map[mk] for mk in ordered_unique_matches_keys
    ]

    # -------------- Handle Results --------------
    if original_matches:
        if as_frame:
            # Decide columns to select
            cols_to_select = (
                original_matches if as_list else
                [original_matches[0]]
            )
            valid_cols = [
                c for c in cols_to_select
                if c in df.columns
            ]
            if not valid_cols:
                msg = ("Internal error: Matched columns "
                       "not found in DataFrame.")
                if errors == 'raise':
                    raise ValueError(msg)
                elif errors == 'warn':
                    warnings.warn(
                        f"{msg} Returning empty DataFrame.",
                        UserWarning
                    )
                return pd.DataFrame()
            return df[valid_cols]
        elif as_list:
            return original_matches
        else:
            return original_matches[0]
    else:
        msg = (f"No ID column found in DataFrame using "
               f"strategy='{strategy}'")
        if regex_pattern and strategy == 'regex':
            msg += f" with pattern='{regex_pattern}'"
        if strategy == 'dtype':
            msg += f" and threshold={uniqueness_threshold}"
        if not case_sensitive:
            msg += " (case-insensitive)."
        else:
            msg += " (case-sensitive)."

        if errors == 'raise':
            raise ValueError(msg)
        elif errors == 'warn':
            warnings.warn(msg, UserWarning)

        # Return empty or None if no match
        if as_frame:
            return pd.DataFrame()
        else:
            return [] if not empty_as_none else None

    
def verify_identical_items(
    list1, 
    list2, 
    mode: str = "unique", 
    ops: str = "check_only", 
    error: str = "raise", 
    objname: str = None, 
) -> Union[bool, list]:
    """
    Check if two lists contain identical elements according 
    to the specified mode.

    In "unique" mode, the function compares the unique elements
    in each list.
    In "ascending" mode, it compares elements pairwise in order.

    Parameters
    ----------
    list1     : list
        The first list of items.
    list2`     : list
        The second list of items.
    mode      : {'unique', 'ascending'}, default="unique"
        The mode of comparison:
          - "unique": Compare unique elements (order-insensitive).
          - "ascending": Compare each element pairwise in order.
    ops       : {'check_only', 'validate'}, default="check_only"
        If "check_only", returns True/False indicating a match.
        If "validate", returns the validated list.
    error     : {'raise', 'warn', 'ignore'}, default="raise"
        Specifies how to handle mismatches.
    objname   : str, optional
        A name to include in error messages.

    Returns
    -------
    bool or list
        Depending on `ops`, returns True/False or the validated list.

    Examples
    --------
    >>> from gofast.utils.generic_utils import verify_identical_items 
    >>> list1 = [0.1, 0.5, 0.9]
    >>> list2 = [0.1, 0.5, 0.9]
    >>> verify_identical_items(list1, list2, mode="unique", ops="validate")
    [0.1, 0.5, 0.9]
    >>> verify_identical_items(list1, list2, mode="ascending", ops="check_only")
    True

    Notes
    -----
    In "ascending" mode, both lists must have the same length, and the
    function compares each corresponding pair of elements.
    In "unique" mode, the function uses the set of unique values for
    comparison. If the lists contain mixed types, the function attempts
    to compare their string representations.
    """
    # Validate mode.
    if mode not in ("unique", "ascending"):
        raise ValueError("mode must be either 'unique' or 'ascending'")
    if ops not in ("check_only", "validate"):
        raise ValueError("ops must be either 'check_only' or 'validate'")
    if error not in ("raise", "warn", "ignore"):
        raise ValueError(
            "error must be one of 'raise', 'warn', or 'ignore'")

    # Ascending mode: compare each element in order.
    if mode == "ascending":
        if len(list1) != len(list2):
            msg = (
                f"Length mismatch in {objname or 'object lists'}: "
                f"{len(list1)} vs {len(list2)}."
            )
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                import warnings
                warnings.warn(msg, UserWarning)
            return False
        
        differences = []
        for idx, (a, b) in enumerate(zip(list1, list2)):
            if a != b:
                differences.append((idx, a, b))
        if differences:
            msg = (
                f"Differences in {objname or 'object lists'}: {differences}."
            )
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                import warnings
                warnings.warn(msg, UserWarning)
            return False
        return True if ops == "check_only" else list1

    # Unique mode: compare the unique elements of each list.
    else:
        try:
            unique1 = sorted(set(list1))
            unique2 = sorted(set(list2))
        except Exception:
            unique1 = sorted({str(x) for x in list1})
            unique2 = sorted({str(x) for x in list2})
        if unique1 != unique2:
            msg = (
                f"Inconsistent unique elements in {objname or 'object lists'}: "
                f"{unique1} vs {unique2}."
            )
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                import warnings
                warnings.warn(msg, UserWarning)
            return False
        return True if ops == "check_only" else unique1


def vlog(
    message,
    verbose=None,
    level: int = 3,
    depth: Union[int, str] = "auto",
    mode:Optional[str]=None,
    vp: bool=True
):
    """
    Log or naive messages with optional indentation and
    bracketed tags.

    This function, `vlog`, allows conditional logging
    or printing of messages based on a global or passed
    in <parameter inline> `verbose` level. By default,
    it behaves differently depending on whether
    ``mode`` is ``'log'`` or ``'naive'``. When
    :math:`mode = 'log'`, the message is printed only if
    :math:`\\text{verbose} \\geq \\text{level}`. Otherwise,
    for :math:`mode` in [``None``, ``'naive'``], the
    verbosity threshold leads to various bracketed
    prefixes (e.g. [INFO], [DEBUG], [TRACE]) unless the
    message already contains such a prefix.

    .. math::
       \\text{indentation} = 2 \\times \\text{depth}

    where :math:`\\text{depth}` is either manually
    specified or auto-derived based on `<parameter inline>`
    `level` (1 = ERROR, 2 = WARNING, 3 = INFO, 4/5 =
    DEBUG, 6/7 = TRACE).

    Parameters
    ----------
    message : str
        The text to be printed or logged.
    verbose : int, optional
        Overall verbosity threshold. If ``None``, it
        looks for a global variable named ``verbose``.
        Default is ``None``.
    level : int, default=3
        Severity or importance level of the message.
        Commonly:
          * 1 = ERROR
          * 2 = WARNING
          * 3 = INFO
          * 4,5 = DEBUG
          * 6,7 = TRACE
    depth : int or str, default="auto"
        Indentation level used for the printed message.
        If ``"auto"``, the depth is computed from
        `<parameter inline> level`.
    mode : str, optional
        Determines logging mode. If set to ``'log'``,
        prints messages only if
        :math:`\\text{verbose} \\geq \\text{level}`.
        Otherwise (if ``None`` or ``'naive'``), it
        follows a custom logic driven by `<parameter
        inline> verbose`.
    vp : bool, default=True
        If ``True``, the function automatically prepends
        bracketed tags (e.g. [INFO]) unless the message
        already contains one of [INFO], [DEBUG], [ERROR],
        [WARNING], or [TRACE].

    Returns
    -------
    None
        This function does not return anything. It either
        prints the message to stdout or omits it,
        depending on `<parameter inline> verbose`, `<parameter
        inline> level`, and ``mode``.

    Notes
    -----
    This function is helpful for selectively displaying
    or logging messages in applications that adapt to
    the user's required verbosity. By default, each
    level has a specific bracketed tag and an auto
    indentation depth.

    Examples
    --------
    >>> from gofast.utils.generic_utils import vlog
    >>> # Example with mode='log'
    >>> # This prints only if global or passed-in
    >>> # verbose >= 4.
    >>> vlog("Check debugging details.", verbose=3, 
    ...      level=4, mode='log')
    >>> # Example with mode='naive'
    >>> # If verbose=2, it displays as [INFO] prefixed.
    >>> vlog("Loading data...", verbose=2, mode='naive')

    See Also
    --------
    globals : Used to retrieve the fallback `verbose`
        value if not explicitly passed.

    """
    verbosity_labels = {
        1: "[ERROR]",
        2: "[WARNING]",
        3: "[INFO]",
        4: "[DEBUG]",
        5: "[DEBUG]",
        6: "[TRACE]",
        7: "[TRACE]"
    }

    # When depth="auto", assign indentation based on `level`.
    if depth == "auto":
        if level == 1:
            depth = 0
        elif level == 2:
            depth = 2
        elif level == 3:
            depth = 0
        elif level in (4, 5):
            depth = 2
        elif level in (6, 7):
            depth = 4
        else:
            depth = 0

    # Fallback to a global 'verbose' if none given.
    actual_verbose = verbose if verbose is not None else globals().get('verbose', 0)

    # If mode == 'log', keep original approach:
    if mode == 'log':
        if actual_verbose >= level:
            # Indent and prefix with the label from `level`.
            indent = " " * (depth * 2)
            print(f"{indent}{verbosity_labels[level]} {message}")
        # Nothing else for mode='log' if verbosity is too low.
        return

    # If mode == 'naive' or None, override with the new rules:
    if mode in [None, 'naive']:
        # If actual_verbose < 1, do not print anything.
        if actual_verbose < 1:
            return

        # Otherwise, figure out the prefix based on the threshold:
        # We check if the message already includes [INFO], [DEBUG], [WARNING],
        # [ERROR], or [TRACE]. If so, skip adding any prefix.
        prefix_tags = ("[INFO]", "[DEBUG]", "[ERROR]", "[WARNING]", "[TRACE]")
        already_tagged = any(tag in message for tag in prefix_tags)

        # indent as per the computed `depth`
        indent = " " * (depth * 2)
        
        # If >=3 => prefix with [INFO] if vp is True and not already tagged
        if actual_verbose <=3:
            if vp and not already_tagged:
                print(f"{indent}[INFO] {message}")
            else:
                print(f"{indent}{message}")
            return 

        # If 3 < verbose < 5 => prefix with [DEBUG] if vp is True and not already tagged
        if 3 < actual_verbose < 5:
            if vp and not already_tagged:
                print(f"{indent}[DEBUG] {message}")
            else:
                print(f"{indent}{message}")
            return

        # If verbose >= 5 => prefix with [TRACE] if vp is True and not already tagged
        if actual_verbose >= 5:
            if vp and not already_tagged:
                print(f"{indent}[TRACE] {message}")
            else:
                print(f"{indent}{message}")
            return

def get_actual_column_name(
    df: pd.DataFrame, 
    tname: str = None, 
    actual_name: str = None, 
    error: str = 'raise',  
    default_to=None, 
) -> str:
    """
    Determines the actual target column name in the given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the target column.
    tname : str, optional
        The base target name (e.g., "subsidence"). If not found in the DataFrame,
        it will attempt to find a matching column using "<tname>_actual" format.
    actual_name : str, optional
        If provided, this name will be returned as the actual target column name.
    error : {'raise', 'warn', 'ignore'}, default='raise'
        Specifies how to handle the case when no valid column is found:
        - 'raise': Raises a `ValueError`.
        - 'warn': Issues a warning and returns `None`.
        - 'ignore': Silently returns `None`.

    Returns
    -------
    str or None
        The determined actual column name, or None if no match is found 
        and `error='warn'` or `error='ignore'`.

    Raises
    ------
    ValueError
        If no valid target column is found and `error='raise'`.

    Examples
    --------
    >>> from gofast.utils.generic_utils import get_actual_column_name
    >>> df = pd.DataFrame({'subsidence_actual': [1, 2, 3]})
    >>> get_actual_column_name(df, tname="subsidence")
    'subsidence_actual'

    >>> df = pd.DataFrame({'subsidence': [1, 2, 3]})
    >>> get_actual_column_name(df, tname="subsidence")
    'subsidence'

    >>> df = pd.DataFrame({'actual': [1, 2, 3]})
    >>> get_actual_column_name(df)
    'actual'
    
    >>> df = pd.DataFrame({'measurement': [1, 2, 3]})
    >>> get_actual_column_name(df, tname="subsidence", error="warn")
    Warning: Could not determine the actual target column in the DataFrame.
    None
    """

    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    # If `actual_name` is provided, check if it exists in df
    if actual_name and actual_name in df.columns:
        return actual_name

    # If `tname` exists in df, return it
    if tname and tname in df.columns:
        return tname

    # If `<tname>_actual` exists, return it
    if tname and f"{tname}_actual" in df.columns:
        return f"{tname}_actual"

    # If "actual" column exists, return it
    if "actual" in df.columns:
        return "actual"

    # Handle the case when no valid column is found
    msg = "Could not determine the actual target column in the DataFrame."
    if error == 'raise':
        raise ValueError(msg)
    elif error == 'warn':
        warnings.warn(msg, UserWarning)
    
    if default_to=='tname': 
        return tname 
    
    return None  # If `error='ignore'`, return None silently


def detect_dt_format(series: pd.Series) -> str:
    r"""
    Detect the datetime format of a pandas Series containing datetime values.

    This function inspects a non-null sample from the datetime Series and
    infers the format string based on its components (year, month, day, hour,
    minute, and second). It returns a format string that can be used with 
    ``strftime``. For example, if the sample indicates only a year is relevant, 
    it returns ``"%Y"``; if full date information is present, it returns 
    ``"%Y-%m-%d"``; and if time details are also present, it extends the format 
    accordingly.

    Parameters
    ----------
    series : pandas.Series
        A Series containing datetime values (dtype datetime64).

    Returns
    -------
    str
        A datetime format string (e.g., ``"%Y"``, ``"%Y-%m-%d"``, or 
        ``"%Y-%m-%d %H:%M:%S"``) that represents the resolution of the data.

    Examples
    --------
    >>> from gofast.utils.generic_utils import detect_dt_format
    >>> import pandas as pd
    >>> dates = pd.to_datetime(['2023-01-01', '2024-01-01', '2025-01-01'])
    >>> fmt = detect_dt_format(pd.Series(dates))
    >>> print(fmt)
    %Y

    Notes
    -----
    The detection logic checks if month, day, hour, minute, and second are 
    all default values (e.g., month == 1, day == 1, hour == 0, etc.) and infers
    the most compact format that still represents the data accurately.

    """
    # Validate input DataFrame
    if not isinstance(series, pd.Series):
        raise TypeError("`series` must be a pandas Series.")

    # Drop null values and pick a sample for analysis.
    sample = series.dropna().iloc[0]
    
    # Always include year.
    fmt = "%Y"
    
    # Include month if not January.
    if sample.month != 1 or sample.day != 1 or sample.hour != 0 or \
       sample.minute != 0 or sample.second != 0:
        fmt += "-%m"
    
    # Include day if not the first day.
    if sample.day != 1 or sample.hour != 0 or sample.minute != 0 or \
       sample.second != 0:
        fmt += "-%d"
    
    # Include time details if any are non-zero.
    if sample.hour != 0 or sample.minute != 0 or sample.second != 0:
        fmt += " %H"
        if sample.minute != 0 or sample.second != 0:
            fmt += ":%M"
        if sample.second != 0:
            fmt += ":%S"
    
    return fmt

def transform_contributions(
    contributions, 
    to_percent=True, 
    normalize=False, 
    norm_range=(0, 1), 
    scale_type=None, 
    zero_division='warn', 
    epsilon=1e-6, 
    log_transform=False
):
    """
    Converts the feature contributions either to a direct percentage, 
    normalizes them to a custom range, or applies a scaling strategy 
    based on the chosen parameters.

    Parameters
    ----------
    contributions : dict
        A dictionary where keys are feature names and values are the 
        feature contributions. Each value is expected to be a numerical 
        value representing the contribution of the respective feature.

    to_percent : bool, optional, default=True
        Whether to convert the contributions to percentages. If `True`, 
        each value in `contributions` will be multiplied by 100. This is 
        useful when contributions are given in decimal form but are expected 
        as percentages.

    normalize : bool, optional, default=False
        Whether to normalize the contributions using min-max scaling. If 
        `True`, the values will be scaled to the range defined in 
        ``norm_range``.

    norm_range : tuple, optional, default=(0, 1)
        A tuple specifying the range (min, max) for normalization. This range 
        is applied when `normalize` is set to `True`. The contributions will 
        be rescaled so that the minimum value maps to `norm_range[0]` and the 
        maximum value maps to `norm_range[1]`.

    scale_type : str, optional, default=None
        The scaling strategy. Options include:
        - ``'zscore'``: Performs Z-score normalization.
        - ``'log'``: Applies a logarithmic transformation to the data.
        If `None`, no scaling is applied.

    zero_division : str, optional, default='warn'
        Defines how to handle zero or missing values in the contributions. 
        Options include:
        - ``'skip'``: Skips zero values (no modification).
        - ``'warn'``: Issues a warning if zero values are found.
        - ``'replace'``: Replaces zeros with a small value defined by 
          ``epsilon`` to avoid division by zero or undefined results.

    epsilon : float, optional, default=1e-6
        A small value used to replace zeros when `zero_division` is set to 
        ``'replace'``. This prevents division by zero errors during 
        transformations like Z-score or log transformation.

    log_transform : bool, optional, default=False
        Whether to apply a logarithmic transformation to the contributions. 
        If `True`, it applies the natural logarithm to each value in the 
        `contributions` dictionary. Only positive values are valid for log 
        transformation, and zero values are either skipped or replaced 
        based on the ``zero_division`` parameter.

    Returns
    -------
    dict
        A dictionary with feature names as keys and the transformed feature 
        contributions as values. The transformation is applied according to 
        the chosen parameters.

    Notes
    -----
    - When ``normalize=True``, if the minimum and maximum values in the 
      `contributions` are the same, normalization is skipped with a warning.
    - If ``scale_type='zscore'``, the function applies Z-score normalization:
      
      .. math::
          Z = \frac{X - \mu}{\sigma}
      
      where :math:`X` is the contribution, :math:`\mu` is the mean of the 
      contributions, and :math:`\sigma` is the standard deviation of the 
      contributions.
      
    - If ``log_transform=True``, the function applies the natural logarithm:
      
      .. math::
          \text{log}(X) \text{ for } X > 0
          
    - The ``zero_division`` parameter handles zero values by either skipping, 
      warning, or replacing them with a small value (`epsilon`).

    Examples
    --------
    >>> from gofast.utils.generic_utils import transform_contributions
    >>> contributions = {
    >>>     'GWL': 2.226836617133828,
    >>>     'rainfall_mm': 12.398293851061492,
    >>>     'normalized_seismic_risk_score': 0.9402759347406523,
    >>>     'normalized_density': 4.806074194258057,
    >>>     'density_concentration': 5.666943330566496e-06,
    >>>     'geology': 1.2798872011280326e-05,
    >>>     'density_tier': 1.044039559604414e-05,
    >>>     'rainfall_category': 0.0
    >>> }
    >>> transform_contributions(contributions, to_percent=True, normalize=True)
    >>> transform_contributions(contributions, to_percent=False, scale_type='zscore')
    
    See Also
    --------
    `numpy.mean`: Compute the arithmetic mean of an array.
    `numpy.std`: Compute the standard deviation of an array.

    References
    ----------
    [1]_ "Statistical Methods for Data Transformation" by J. Smith, 
         Springer, 2020.
    """
    
    # Handle zero values based on user preference
    if zero_division == 'replace':
        contributions = {
            feature: (contribution if contribution != 0 else epsilon)
            for feature, contribution in contributions.items()
        }
    elif zero_division == 'warn' and any(
        contribution == 0 for contribution in contributions.values()
    ):
        warnings.warn(
            "Some contribution values are zero. Consider replacing them.",
            UserWarning
        )

    # Convert contributions to percentage if specified
    if to_percent:
        contributions = {
            feature: contribution * 100 
            for feature, contribution in contributions.items()
        }
    
    # Apply normalization to the specified range
    if normalize:
        min_val = min(contributions.values())
        max_val = max(contributions.values())
        
        # Check if min and max values are the
        # same to avoid division by zero
        if min_val == max_val:
            warnings.warn(
                "All contribution values are the same,"
                " cannot normalize; Skipped.",
                UserWarning
            )
        else:
            norm_range = 100 * np.asarray(
                norm_range) if to_percent else norm_range 
            
            contributions = {
                feature: (
                    ((contribution - min_val) / (max_val - min_val)) * 
                    (norm_range[1] - norm_range[0]) + norm_range[0]
                ) 
                for feature, contribution in contributions.items()
            }

    # Apply scaling (Z-score or log)
    if scale_type == 'zscore':
        mean_val = np.mean(list(contributions.values()))
        std_val = np.std(list(contributions.values()))
        
        contributions = {
            feature: (contribution - mean_val) / std_val 
            if std_val != 0 else contribution
            for feature, contribution in contributions.items()
        }
    
    elif log_transform:
        contributions = {
            feature: np.log(contribution) if contribution > 0 else 0
            for feature, contribution in contributions.items()
        }
    
    return contributions


def exclude_duplicate_kwargs(
    func: callable,
    existing_kwargs: Union[
        Dict[str, Any],
        List[str]
    ],
    user_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prevents the user from overriding existing parameters
    in a target function. The method `exclude_duplicate_kwargs`
    checks both developer-specified and function-level
    parameter names to exclude them from `user_kwargs`.
    
    .. math::
       \text{final\_kwargs} =
       \{\,(k, v) \in \text{user\_kwargs} \,\mid\,
       k \notin \text{protected\_params}\,\}
    
    Parameters
    ----------
    func : callable
        The target function whose valid parameters are
        checked. It uses Python's introspection to gather
        the acceptable parameter names.
    
    existing_kwargs : dict or list
        Developer-defined parameters to protect. Can be:
        * A dictionary of parameter-value pairs (e.g.,
          ``{'ax': ax_obj, 'data': df}``) whose keys
          are excluded from user overrides.
        * A list of parameter names (e.g., ``['ax',
          'data']``) to protect from user overrides.
    
    user_kwargs : dict
        The user-supplied keyword arguments that are
        candidates for merging with `existing_kwargs`.
        This dictionary is filtered to remove collisions
        with protected parameters.
    
    Returns
    -------
    dict
        A filtered dictionary of user-defined arguments
        that do not overlap with protected parameters.
    
    Examples
    --------
    >>> from gofast.utils.generic_utils import exclude_duplicate_kwargs
    >>> import seaborn as sns
    >>> # Developer has some base kwargs
    ... base_kwargs = {
    ...     'x': 'species',
    ...     'y': 'sepal_length',
    ...     'palette': 'viridis'
    ... }
    >>> # User tries to override 'x' with new param
    ... user_args = {
    ...     'x': 'petal_width',
    ...     'color': 'red'
    ... }
    >>> # Filter out duplicates
    ... safe_args = exclude_duplicate_kwargs(
    ...     sns.scatterplot,
    ...     base_kwargs,
    ...     user_args
    ... )
    >>> safe_args
    {'color': 'red'}
    
    Notes
    -----
    By default, if `existing_kwargs` is a dictionary,
    its keys are treated as protected parameter names.
    If it's a list, those items are protected. The
    function signature of `func` is also used to
    verify that only recognized parameters are
    protected.
    
    See Also
    --------
    inspect.signature : Used to introspect function
        parameters.
    filter_valid_kwargs : Another inline function that
        discards user params not valid for a given
        function.
    
    References
    ----------
    .. [1] David Beazley and Brian K. Jones.
       *Python Cookbook, 3rd Edition.* O'Reilly Media, 2013.
    """

    # Validate existing_kwargs
    if not isinstance(existing_kwargs, (dict, list)):
        raise TypeError(
            "existing_kwargs must be a dict or list"
        )

    # Build set of protected parameters
    protected_params = (
        existing_kwargs.keys()
        if isinstance(existing_kwargs, dict)
        else existing_kwargs
    )

    # Inspect func signature to get valid names
    sig = inspect.signature(func)
    valid_params = {
        name: param
        for name, param in sig.parameters.items()
        if param.kind not in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD
        )
    }

    # Restrict protection to valid parameters only
    exclude = [
        param
        for param in protected_params
        if param in valid_params
    ]

    # Filter out excluded params from user_kwargs
    safe_kwargs = {
        k: v
        for k, v in user_kwargs.items()
        if k not in exclude
    }

    return safe_kwargs

def reorder_columns(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    pos: Union[str, int, float] = "end"
):
    """
    Reorder columns in a DataFrame by moving specified
    columns to a chosen position.

    This function locates `<columns>` in the original
    DataFrame `<df>` and rearranges them based on the
    parameter ``pos``. If ``pos`` is `"end"`, columns
    are appended to the end. If `"begin"` or `"start"`,
    they are placed at the front. If `"center"`, they
    are inserted at the midpoint:

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be modified.

    columns : str or iterable of str
        A single column name or multiple column
        names to reposition. If a single string is
        given, it is converted to a list with one
        element.

    pos : str, int, or float, default ``"end"``
        Determines the target placement:
          - ``"end"``: Append after all other
            columns.
          - ``"begin"`` or ``"start"``: Prepend at
            the start.
          - ``"center"``: Insert at the midpoint of
            remaining columns.
          - integer or float: Insert at zero-based
            index among the remaining columns. If
            out of bounds, the original DataFrame is
            returned unchanged.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with `<columns>` moved as
        specified by ``pos``.

    Methods
    -------
    `reorder_columns_in`
        This method rearranges columns without
        altering values or data order beyond column
        placement.

    Notes
    -----
    - The function checks if `<columns>` exist in
      `<df>`, ignoring columns not present.
    - A warning is issued if the position is beyond
      the range of valid indices.
    - Negative indices for integer ``pos`` are
      converted to positive by adding the total
      number of remaining columns.
      
    .. math::
       i_{\\text{center}} = \\left\\lfloor
           \\frac{|R|}{2}
       \\right\\rfloor,

    where :math:`|R|` is the number of remaining
    columns after removing the target columns [1]_.
    For integer or float ``pos``, the target columns
    are inserted at index :math:`\\lfloor pos
    \\rfloor` among the remaining columns.

    Examples
    --------
    >>> from gofast.utils.generic_utils import reorder_columns
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'latitude': [10.1, 10.2, 10.3],
    ...     'landslide': [0, 1, 0],
    ...     'longitude': [20.1, 20.2, 20.3]
    ... })
    >>> # Move 'landslide' to the end (default)
    >>> reorder_columns(data, 'landslide', pos="end")
       id  latitude  longitude  landslide
    0   1      10.1       20.1          0
    1   2      10.2       20.2          1
    2   3      10.3       20.3          0

    See Also
    --------
    pandas.DataFrame.reindex : Pandas method for
                                 reindexing or
                                 reordering columns
                                 more generally.

    References
    ----------
    .. [1] Wes McKinney. "Python for Data Analysis,"
           2nd Edition, O'Reilly Media.
    """
    # Validate that the input 'df' is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expect dataframe. Got {type(df).__name__!r}"
        )

    # Normalize 'columns' to a list of strings
    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, (list, tuple, set)):
        raise TypeError(
            "columns must be a string or an "
            "iterable of strings."
        )

    # Filter out columns that are actually in df
    valid_targets = [
        col for col in columns if col in df.columns
    ]
    if not valid_targets:
        raise ValueError(
            "None of the specified columns were "
            "found in the DataFrame."
        )

    # Create a list of remaining columns while
    # preserving their original order
    remaining_columns = [
        col for col in df.columns if col not in
        valid_targets
    ]

    # Maintain the original target columns' order
    target_order = [
        col for col in df.columns if col in
        valid_targets
    ]

    # Compute new order based on 'pos'
    if isinstance(pos, str):
        pos_lower = pos.lower()
        if pos_lower == "end":
            new_order = remaining_columns + target_order
        elif pos_lower in ("begin", "start"):
            new_order = target_order + remaining_columns
        elif pos_lower == "center":
            center_index = len(remaining_columns) // 2
            new_order = (
                remaining_columns[:center_index]
                + target_order
                + remaining_columns[center_index:]
            )
        else:
            try:
                pos = int(pos)
            except Exception as e:
                raise ValueError(
                    f"Unrecognized string value for pos:"
                    f" {pos}"
                ) from e

    # If 'pos' is numeric, insert the columns at
    # that position
    if isinstance(pos, (int, float)):
        pos_int = int(pos)
        # Check if pos_int is within bounds
        if (pos_int > len(remaining_columns)
           or pos_int < -len(remaining_columns)):
            warnings.warn(
                "Position is out of bounds. Skipping "
                "moving columns."
            )
            return df
        # Convert negative index to positive
        if pos_int < 0:
            pos_int = len(remaining_columns) + pos_int
        new_order = (
            remaining_columns[:pos_int]
            + target_order
            + remaining_columns[pos_int:]
        )
    else:
        # If it's not a recognized type, raise an error
        if not isinstance(pos, str):
            raise TypeError(
                "pos must be a string, integer, or float."
            )

    # Return the DataFrame with the new column order
    return df[new_order]


def map_scales_choice(
    scales_choice_str: str
) -> Optional[List[int]]:
    """
    Maps a string choice for scales to an actual list of scale values 
    or None if no scales are provided.

    This function interprets specific string inputs and converts them 
    to corresponding lists of integers. If no matching string is found, 
    it returns None as a default fallback.

    Parameters
    ----------
    scales_choice_str : str
        A string representing the choice of scale. It can be one of 
        'default_scales', 'alt_scales', or 'no_scales'.

    Returns
    -------
    Optional[List[int]]
        A list of integers corresponding to the chosen scales, or None 
        if no scales are provided.

    Examples
    --------
    >>> from fusionlab.utils.generic_utils import map_scales_choice
    >>> map_scales_choice('default_scales')
    [1, 3, 7]
    
    >>> map_scales_choice('alt_scales')
    [1, 5, 10]
    
    >>> map_scales_choice('no_scales')
    None

    >>> map_scales_choice('unknown_scales')
    None
    """
    if not isinstance(scales_choice_str, str):
        raise ValueError("scales_choice_str must be a string.")
    
    if scales_choice_str == 'default_scales':
        return [1, 3, 7]
    elif scales_choice_str == 'alt_scales':
        return [1, 5, 10]
    elif scales_choice_str == 'no_scales':
        return None
    return None  # Default fallback for unrecognized inputs


def cast_hp_to_bool(
    params: Dict[str, Any],
    param_name: str,
    default_value: bool = False  
) -> None:
    """
    Casts a hyperparameter value in the `params` dictionary to a boolean.

    The function ensures that hyperparameters that are supposed to be 
    booleans are correctly cast to Python booleans. This is particularly 
    useful in cases where Keras Tuner or other libraries might return 
    0 or 1 as the value for boolean choices.

    Parameters
    ----------
    params : Dict[str, Any]
        A dictionary of hyperparameters to be validated and updated in place.
    
    param_name : str
        The key of the hyperparameter to be checked and casted to boolean.
    
    default_value : bool, optional
        The default value to assign if the parameter is not found or 
        has an invalid value. The default is False.

    Returns
    -------
    None
        This function modifies the `params` dictionary in-place.

    Examples
    --------
    >>> from fusionlab.utils.generic_utils import cast_hp_to_bool
    >>> params = {'use_batch_norm': 1}
    >>> cast_hp_to_bool(params, 'use_batch_norm', default_value=False)
    >>> print(params['use_batch_norm'])
    True
    
    >>> params = {'use_residuals': 0}
    >>> cast_hp_to_bool(params, 'use_residuals', default_value=True)
    >>> print(params['use_residuals'])
    False
    
    >>> params = {'use_dropout': 'yes'}
    >>> cast_hp_to_bool(params, 'use_dropout', default_value=False)
    >>> print(params['use_dropout'])
    False
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary.")
    
    if param_name in params:
        value = params[param_name]
        if isinstance(value, (int, float)):  # Handles 0, 1, 0.0, 1.0
            params[param_name] = bool(value)
        elif not isinstance(value, bool):
            # Warn if the value is not a valid boolean type (int, float, bool)
            warnings.warn(
                f"Hyperparameter '{param_name}' received unexpected value "
                f"'{value}' (type: {type(value)}). Expected bool or 0/1. "
                f"Defaulting to {default_value}. Please check param_space definition."
            )
            params[param_name] = default_value

def cast_multiple_bool_params(
    params: Dict[str, Any],
    bool_params_to_cast: List[Tuple[str, bool]]
) -> None:
    """
    Casts a list of boolean hyperparameters to ensure they are Python booleans.

    This function iterates over a list of parameter names and default values,
    ensuring that each parameter is properly cast to a boolean type. If a 
    parameter does not exist or has an invalid value, it is set to the 
    provided default.

    Parameters
    ----------
    params : Dict[str, Any]
        A dictionary of hyperparameters to be validated and updated in place.
    
    bool_params_to_cast : List[Tuple[str, bool]]
        A list of tuples where each tuple consists of a hyperparameter name 
        and its default boolean value. The function will cast the corresponding
        parameter to a boolean.

    Returns
    -------
    None
        This function modifies the `params` dictionary in-place.

    Examples
    --------
    >>> from fusionlab.utils.generic_utils import cast_multiple_bool_params
    >>> params = {'use_batch_norm': 1, 'use_residuals': 0}
    >>> cast_multiple_bool_params(params, [('use_batch_norm', False), ('use_residuals', True)])
    >>> print(params)
    {'use_batch_norm': True, 'use_residuals': False}
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary.")
    
    if not isinstance(bool_params_to_cast, list):
        raise ValueError("bool_params_to_cast must be a list of tuples.")
    
    for param_name, default_value in bool_params_to_cast:
        cast_hp_to_bool(params, param_name, default_value)

def save_all_figures(
    output_dir: str = "figures",
    prefix: str = "figure",
    fmts: Union[List[str], tuple] = ("png",),
    close: bool = True,
    dpi: Union[int, None] = 150,
    transparent: bool = False,
    timestamp: bool = True,
    verbose: bool = True
) -> List[str]:
    """
    Save all currently open Matplotlib figures to disk in specified formats.

    Parameters
    ----------
    output_dir : str
        Directory where figures will be saved. Created if not exists.
    prefix : str
        Filename prefix for each figure.
    formats : list or tuple of str
        File formats/extensions to use (e.g., ('png','pdf')).
    close : bool
        Whether to close each figure after saving. Default is True.
    dpi : int or None
        Resolution in dots per inch. None uses Matplotlib default.
    transparent : bool
        Whether to save figures with transparent background.
    timestamp : bool
        Append current timestamp (YYYYmmddTHHMMSS) to filenames.
    verbose : bool
        Print progress messages.

    Returns
    -------
    List[str]
        List of saved file paths.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(); plt.plot([1, 2, 3])
    >>> from fusionlab.utils.generic_utils import save_all_figures
    >>> paths = save_all_figures(output_dir="plots", formats=("png",))
    >>> print(paths)
    ['plots/figure_1_20250521T153045.png']
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    fig_nums = plt.get_fignums()
    
    for num in fig_nums:
        fig = plt.figure(num)
        # Build base filename
        name_parts = [prefix, str(num)]
        if timestamp:
            name_parts.append(datetime.now().strftime("%Y%m%dT%H%M%S"))
        base_name = "_".join(name_parts)
        
        # Save in each requested format
        for ext in fmts:
            filename = f"{base_name}.{ext.lstrip('.')}"
            path = os.path.join(output_dir, filename)
            try:
                fig.savefig(path, dpi=dpi, transparent=transparent)
                saved_paths.append(path)
                if verbose:
                    print(f"Saved Figure {num} as: {path}")
            except Exception as e:
                if verbose:
                    print(f"ERROR saving Figure {num} to {path}: {e}")
        
        if close:
            plt.close(fig)
    
    return saved_paths

def print_box(
    msg: Union[str, List[str]],
    width: int = 80,
    align: str = 'center',
    border_char: str = '+',
    horizontal_char: str = '-',
    vertical_char: str = '|',
    padding: int = 1
) -> None:
    """
    Print a boxed message with customizable styling.

    Parameters
    ----------
    msg : str or list of str
        The message to display. If a list is provided, each element
        is treated as a separate line.
    width : int, default=80
        Total width of the box including borders.
    align : {'center','left','right'}, default='center'
        Text alignment within the box.
    border_char : str, default='+'
        Character used for the four corners of the box.
    horizontal_char : str, default='-'
        Character used for the top/bottom border lines.
    vertical_char : str, default='|'
        Character used for the left/right border lines.
    padding : int, default=1
        Number of spaces between text and vertical borders.

    Returns
    -------
    None
        Prints the styled box directly to stdout.

    Examples
    --------
    >>> from fusionlab.utils.generic_utils import print_box 
    >>> print_box("Hello, world!", width=40)
    +--------------------------------------+
    |             Hello, world!           |
    +--------------------------------------+

    >>> print_box(
    ...     ["Line one", "This is a longer line that will wrap"],
    ...     width=50, align='left', border_char='*'
    ... )
    **************************************************
    * Line one                                      *
    * This is a longer line that will                *
    * wrap                                          *
    **************************************************
    """
    # Ensure msg is a list of lines
    lines = msg if isinstance(msg, list) else msg.split('\n')

    # Calculate inner width for text (excluding borders & padding)
    inner_width = width - 2 - 2 * padding
    if inner_width < 10:
        raise ValueError("Width too small for the given padding.")

    # Build top and bottom border
    border_line = (
        border_char
        + horizontal_char * (width - 2)
        + border_char
    )

    # Function to align a single line
    def align_line(text: str) -> str:
        if len(text) > inner_width:
            text = text[:inner_width]
        if align == 'left':
            return text.ljust(inner_width)
        elif align == 'right':
            return text.rjust(inner_width)
        else:  # center
            return text.center(inner_width)

    # Print the box
    print(border_line)
    for raw_line in lines:
        # Wrap long lines
        wrapped = textwrap.wrap(raw_line, inner_width) or ['']
        for wline in wrapped:
            print(
                vertical_char
                + ' ' * padding
                + align_line(wline)
                + ' ' * padding
                + vertical_char
            )
    print(border_line)

def handle_emptiness(
    obj: Any,
    ops: str = 'validate',
    empty_as_none: bool = True
) -> Union[Any, bool]:
    """
    Smart helper to check or normalize empty/None values.

    Parameters
    ----------
    obj : Any
        The object to inspect. Can be None, numpy array,
        pandas Series/DataFrame, list, tuple, etc.
    ops : {'validate','check_only'}, default='validate'
        - 'check_only': return True if obj is None or empty.
        - 'validate': return normalized obj or placeholder.
    empty_as_none : bool, default=True
        Only used when ops='validate':
        - If True, empty or None -> return None
        - If False, empty or None -> return [] (empty list)

    Returns
    -------
    obj or bool
        - If `ops=='check_only'`, returns a bool indicating if
          `obj` is None or empty.
        - If `ops=='validate'`,
          - Non-empty object -> returned unchanged
          - Empty/None -> None or [] based on `empty_as_none`

    Examples
    --------
    >>> import numpy as np
    >>> from fusionlab.utils.generic_utils import \
            handle_emptiness
    >>> handle_emptiness(None, ops='check_only')
    True
    >>> handle_emptiness([], ops='check_only')
    True
    >>> handle_emptiness(np.array([]), ops='check_only')
    True
    >>> handle_emptiness(pd.DataFrame(), ops='check_only')
    True
    >>> handle_emptiness([1,2,3], ops='check_only')
    False
    >>> handle_emptiness([], ops='validate', empty_as_none=True)
    None
    >>> handle_emptiness([], ops='validate', empty_as_none=False)
    []

    """
    def _is_empty(o: Any) -> bool:
        """Determine if o is None or contains no elements."""
        if o is None:
            return True
        if isinstance(o, np.ndarray):
            return o.size == 0
        if isinstance(o, (pd.Series, pd.DataFrame)):
            return o.empty
        try:
            return len(o) == 0
        except Exception:
            return False

    if ops == 'check_only':
        return _is_empty(obj)

    if ops == 'validate':
        if _is_empty(obj):
            return None if empty_as_none else []
        return obj

    raise ValueError("`ops` must be 'validate' or 'check_only'")


def _report_condition(
    policy: Literal['raise', 'warn', 'ignore'],
    default_message: str,
    custom_message: Optional[str],
    exception_type: type = ValueError,
    warning_type: type = UserWarning
) -> None:
    """Helper to raise error, issue warning, or ignore based on policy."""
    message_to_use = custom_message if custom_message is not None \
        else default_message
    
    if policy == 'raise':
        raise exception_type(message_to_use)
    elif policy == 'warn':
        warnings.warn(message_to_use, warning_type)
    # If 'ignore', do nothing.

def are_all_values_in_bounds(
    values: Any,
    bounds: Union[Tuple[Real, Real], List[Real]] = (0, 1),
    closed: Literal['both', 'left', 'right', 'neither'] = 'neither',
    nan_policy: Literal['raise', 'propagate', 'omit'] = 'propagate',
    empty_policy: Literal['allow_true', 'treat_as_false'] = 'allow_true',
    error: Literal['raise', 'warn', 'ignore'] = 'raise',
    message: Optional[str] = None
) -> bool:
    """
    Check if all evaluable input values are within specified numeric bounds.

    Supports various input types including scalars, lists, tuples,
    NumPy arrays, and pandas Series/Index. It attempts to convert
    inputs to a numeric format (float) for comparison. Non-numeric
    entries that cannot be converted will typically result in False
    or trigger the `error` policy if conversion itself fails.

    Parameters
    ----------
    values : Any
        Input data to check.
    bounds : tuple[Real, Real] or list[Real, Real], default=(0, 1)
        A sequence of two numeric values (lower_bound, upper_bound).
        `bounds[0]` must be less than or equal to `bounds[1]`.
    closed : {'both', 'left', 'right', 'neither'}, default='neither'
        Defines whether the interval is closed or open:
        - ``'both'``: `lower_bound <= value <= upper_bound`
        - ``'left'``: `lower_bound <= value < upper_bound`
        - ``'right'``: `lower_bound < value <= upper_bound`
        - ``'neither'``: `lower_bound < value < upper_bound` (strict)
    nan_policy : {'raise', 'propagate', 'omit'}, default='propagate'
        How to handle NaN (Not a Number) values:
        - ``'raise'``: Behavior depends on `error` policy. If `error`
          is 'raise', a ValueError is raised. If 'warn', a warning
          is issued and False is returned. If 'ignore', False is
          returned (as NaNs are not in bounds).
        - ``'propagate'``: If any NaN is found, return False.
        - ``'omit'``: NaNs are removed. The behavior for an array
          that becomes empty after omission is governed by
          `empty_array_policy`.
    empty_array_policy : {'allow_true', 'treat_as_false'}, default='allow_true'
        Policy for handling an empty array of values to check. An array
        can be empty if the original input was empty (e.g., `[]`) or
        if all values were NaNs and `nan_policy='omit'` was used.
        - ``'allow_true'``: An empty array is considered to have all
          its (zero) elements within bounds (vacuously true).
        - ``'treat_as_false'``: An empty array results in False.
    error : {'raise', 'warn', 'ignore'}, default='raise'
        Policy for handling invalid parameters (e.g., `bounds`,
        `closed`), non-convertible inputs, NaNs when
        `nan_policy='raise'`, or when values are out of bounds:
        - ``'raise'``: Raise a ValueError.
        - ``'warn'``: Issue a UserWarning. The function will then
          return False if the condition (e.g., out of bounds, NaN
          present with nan_policy='raise') makes the check fail.
        - ``'ignore'``: Suppress the error/warning. The function will
          return False if the condition makes the check fail.
    message : str, optional
        Custom message to use when `error` policy is 'raise' or 'warn'.
        If None, a default message is used for the specific condition.

    Returns
    -------
    bool
        True if all evaluable values are within bounds (considering
        `nan_policy` and `empty_array_policy`), False otherwise.

    Raises
    ------
    ValueError
        If `error='raise'` and an invalid parameter is provided,
        input is non-convertible, NaNs are present with
        `nan_policy='raise'`, or values are out of bounds.
    UserWarning
        If `error='warn'` for the same conditions as ValueError.

    Examples
    --------
    >>> from fusionlab.utils.generic_utils import are_all_values_in_bounds
    >>> are_all_values_in_bounds(0.5)
    True
    >>> are_all_values_in_bounds([]) # Empty list, default empty_policy='allow_true'
    True
    >>> are_all_values_in_bounds([], empty_policy='treat_as_false')
    False
    >>> are_all_values_in_bounds(pd.Series([np.nan]), nan_policy='omit',
                                 empty_array_policy='treat_as_false')
    False
    >>> try:
    ...     are_all_values_in_bounds([0, 2], bounds=(0,1),
    ...                                 error='raise', closed='neither')
    ... except ValueError as e:
    ...     print(e)
    One or more values are out of the specified bounds.
    """
    arr: np.ndarray

    # --- 1. Input Conversion to NumPy float array ---
    if np.isscalar(values):
        if not isinstance(values, (int, float, np.number)):
            try:
                val_float = float(values)
                arr = np.array([val_float])
            except (ValueError, TypeError):
                 _report_condition(
                    error,
                    f"Scalar input '{values}' of type {type(values).__name__} "
                    "is not convertible to a numeric value.",
                    message,
                )
                 return False
        else:
            arr = np.array([float(values)])
            
    elif isinstance(values, (pd.Series, pd.Index)):
        temp_arr = values.to_numpy()
        try:
            if temp_arr.dtype == object or \
               not np.issubdtype(temp_arr.dtype, np.number):
                arr = pd.to_numeric(temp_arr, errors='raise').astype(float)
            else:
                arr = temp_arr.astype(float)
        except (ValueError, TypeError):
            _report_condition(
                error,
                "Pandas input contains non-convertible non-numeric values.",
                message,
            )
            return False
    else: 
        try:
            arr = np.asarray(values, dtype=float)
        except (ValueError, TypeError):
            _report_condition(
                error,
                "Input sequence contains non-convertible non-numeric values.",
                message,
            )
            return False
    
    # --- 2. Validate `bounds` and `closed` Parameters ---
    # These checks happen early. If they fail and error is 'raise'/'warn',
    # the function returns False or raises, so subsequent logic is safe.
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
        _report_condition(
            error,
            "`bounds` must be a list or tuple of two numbers.",
            message
        )
        return False 
    
    try:
        lower_b = float(bounds[0])
        upper_b = float(bounds[1])
    except (IndexError, ValueError, TypeError):
        _report_condition(
            error,
            "Elements of `bounds` must be numeric and `bounds` must "
            "have length 2.",
            message
        )
        return False

    if lower_b > upper_b:
        _report_condition(
            error,
            f"Lower bound {lower_b} cannot be greater than upper "
            f"bound {upper_b}.",
            message
        )
        return False

    valid_closed_options = {'both', 'left', 'right', 'neither'}
    if closed not in valid_closed_options:
        _report_condition(
            error,
            f"Invalid 'closed' parameter: {closed}. Must be one of "
            f"{valid_closed_options}.",
            message
        )
        return False
        
    # --- 3. Handle NaNs based on `nan_policy` ---
    # This section modifies `arr` if nan_policy is 'omit'.
    # For other policies, it might return early.
    has_nans = np.isnan(arr).any()

    if has_nans:
        if nan_policy == 'raise':
            _report_condition(
                error,
                "Input contains NaNs.",
                message
            )
            return False # Returns False if error policy is 'warn' or 'ignore'
        elif nan_policy == 'propagate':
            return False 
        elif nan_policy == 'omit':
            arr = arr[~np.isnan(arr)]
            # `arr` might be empty now. This is handled in step 4.
    
    # --- 4. Handle Empty Array (original or after NaN omission) ---
    if arr.size == 0:
        if empty_policy == 'allow_true':
            return True
        elif empty_policy == 'treat_as_false':
            return False
        else: # Should be caught by Literal type hint
            _report_condition(
                error,
                f"Internal error: Invalid 'empty_policy' "
                f"value '{empty_policy}'.",
                message
            )
            return False # Fallback for invalid policy if not raised

    # --- 5. Perform Bounds Check on (Potentially Modified) `arr` ---
    # At this point, `arr` contains only non-NaN numeric values,
    # and is guaranteed to be non-empty.
    in_bounds_mask: np.ndarray
    if closed == 'both':
        in_bounds_mask = (arr >= lower_b) & (arr <= upper_b)
    elif closed == 'left':
        in_bounds_mask = (arr >= lower_b) & (arr < upper_b)
    elif closed == 'right':
        in_bounds_mask = (arr > lower_b) & (arr <= upper_b)
    elif closed == 'neither': 
        in_bounds_mask = (arr > lower_b) & (arr < upper_b)
    else:
        # This case should have been caught by parameter validation.
        # If error='ignore' for invalid 'closed', this is a fallback.
        _report_condition(
            error,
            f"Internal error: Invalid 'closed' value '{closed}' "
            "reached bounds check.",
            message
        )
        return False

    all_values_are_in_bounds = bool(np.all(in_bounds_mask))

    if not all_values_are_in_bounds:
        _report_condition(
            error,
            "One or more values are out of the specified bounds.",
            message
        )
        return False # Returns False if error policy is 'warn' or 'ignore'
    else:
        return True

