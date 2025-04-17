#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import warnings 
from typing import Any, Optional, List, Union
import numpy as np 
import pandas as pd 

from ..core.io import is_data_readable, SaveFile 
from ..decorators import Dataify 

@SaveFile  
@is_data_readable 
@Dataify(auto_columns=True, fail_silently=True) 
def mask_by_reference(
    data: pd.DataFrame,
    ref_col: str,
    values: Optional[Union[Any, List[Any]]] = None,
    find_closest: bool = False,
    fill_value: Any = 0,
    mask_columns: Optional[Union[str, List[str]]] = None,
    error: str = "raise",
    verbose: int = 0,
    inplace: bool = False,
    savefile:Optional[str]=None, 
) -> pd.DataFrame:
    r"""
    Masks (replaces) values in columns other than the reference column
    for rows in which the reference column matches (or is closest to) the
    specified value(s).

    If a row's reference-column value is matched, that row's values in
    the *other* columns are overwritten by ``fill_value``. The reference
    column itself is not modified.

    This function supports both exact and approximate matching:
      - **Exact** matching is used if ``find_closest=False``.
      - **Approximate** (closest) matching is used if
        ``find_closest=True`` and the reference column is numeric.

    By default, if the reference column does not exist or if the
    given ``values`` cannot be found (or approximated) in the reference
    column, an exception is raised. This behavior can be adjusted with
    the ``error`` parameter.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data to be masked.

    ref_col : str
        The column in ``data`` serving as the reference for matching
        or finding the closest values.

    values : Any or sequence of Any, optional
        The reference values to look for in ``ref_col``. This can be:
          - A single value (e.g., ``0`` or ``"apple"``).
          - A list/tuple of values (e.g., ``[0, 10, 25]``).
          - If ``values`` is None, **all rows** are masked 
            (i.e. all rows match), effectively overwriting the entire
            DataFrame (except the reference column) with ``fill_value``.
        
        Note that if ``find_closest=False``, these values must appear
        in the reference column; otherwise, an error or warning is
        triggered (depending on the ``error`` setting).

    find_closest : bool, default=False
        If True, performs an approximate match for numeric reference
        columns. For each entry in ``values``, the function locates
        the row(s) in ``ref_col`` whose value is numerically closest.
        Non-numeric reference columns will revert to exact matching
        regardless.

    fill_value : Any, default=0
        The value used to fill/mask the non-reference columns wherever
        the condition (exact or approximate match) is met. This can
        be any valid type, e.g., integer, float, string, np.nan, etc.
        If ``fill_value='auto'`` and multiple values
        are given, each row matched by a particular reference
        value is filled with **that same reference value**.

        **Examples**:
          - If ``values=9`` and ``fill_value='auto'``, the fill
            value is **9** for matched rows.
          - If ``values=['a', 10]`` and ``fill_value='auto'``,
            then rows matching `'a'` are filled with `'a'`, and
            rows matching `10` are filled with `10`.
            
    mask_columns : str or list of str, optional
        If specified, *only* these columns are masked. If None,
        all columns except ``ref_col`` are masked. If any column in
        ``mask_columns`` does not exist in the DataFrame and
        ``error='raise'``, a KeyError is raised; otherwise, a warning
        may be issued or ignored.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        Controls how to handle errors:
          - 'raise': raise an error if the reference column does not
            exist or if any of the given values cannot be matched (or
            approximated).
          - 'warn': only issue a warning instead of raising an error.
          - 'ignore': silently ignore any issues.

    verbose : int, default=0
        Verbosity level:
          - 0: silent (no messages).
          - 1: minimal feedback.
          - 2 or 3: more detailed messages for debugging.

    inplace : bool, default=False
        If True, performs the operation in place and returns the 
        original DataFrame with modifications. If False, returns a
        modified copy, leaving the original unaltered.
        
    savefile : str or None, optional
        File path where the DataFrame is saved if the
        decorator-based saving is active. If `None`, no saving
        occurs.

    Returns
    -------
    pd.DataFrame
        A DataFrame where rows matching the specified condition (exact
        or approximate) have had their non-reference columns replaced by
        ``fill_value``.

    Raises
    ------
    KeyError
        If ``error='raise'`` and ``ref_col`` is not in ``data.columns``.
    ValueError
        If ``error='raise'`` and no exact/approx match can be found
        for one or more entries in ``values``.

    Notes
    -----
    - If ``values`` is None, **all** rows are masked in the non-ref
      columns, effectively overwriting them with ``fill_value``.
    - When ``find_closest=True``, approximate matching is performed only
      if the reference column is numeric. For non-numeric data, it falls
      back to exact matching.
    - When multiple reference values are provided, each is
      processed in turn. If `fill_value='auto'`, each matched row
      is filled with that specific reference value.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.data_utils import mask_by_reference
    >>>
    >>> df = pd.DataFrame({
    ...     "A": [10, 0, 8, 0],
    ...     "B": [2, 0.5, 18, 85],
    ...     "C": [34, 0.8, 12, 4.5],
    ...     "D": [0, 78, 25, 3.2]
    ... })
    >>>
    >>> # Example 1: Exact matching, replace all columns except 'A' with 0
    >>> masked_df = mask_by_reference(
    ...     data=df,
    ...     ref_col="A",
    ...     values=0,
    ...     fill_value=0,
    ...     find_closest=False,
    ...     error="raise"
    ... )
    >>> print(masked_df)
    >>> # 'B', 'C', 'D' for rows where A=0 are replaced with 0.
    >>>
    >>> # Example 2: Approximate matching for numeric
    >>> # If 'A' has values [0, 10, 8] and we search for 9, then 'A=8' or 'A=10'
    >>> # are the closest, so those rows get masked in non-ref columns.
    >>> masked_df2 = mask_by_reference(
    ...     data=df,
    ...     ref_col="A",
    ...     values=9,
    ...     find_closest=True,
    ...     fill_value=-999
    ... )
    >>> print(masked_df2)
    
    >>>
    >>> # Example 2: Approx. match for numeric ref_col
    >>> # 9 is between 8 and 10, so rows with A=8 and A=10 are masked
    >>> res2 = mask_by_reference(df, "A", 9, find_closest=True, fill_value=-999)
    >>> print(res2)
    ... # Rows 0 (A=10) and 2 (A=8) are replaced with -999 in columns B,C,D
    >>>
    >>> # Example 3: fill_value='auto' with multiple values
    >>> # Rows matching A=0 => fill with 0; rows matching A=8 => fill with 8
    >>> res3 = mask_by_reference(df, "A", [0, 8], fill_value='auto')
    >>> print(res3)
    ... # => rows with A=0 => B,C,D replaced by 0
    ... # => rows with A=8 => B,C,D replaced by 8
    >>> 
    >>> # 2) mask_columns=['C','D'] => only columns C and D are masked
    >>> res2 = mask_by_reference(df, "A", values=0, fill_value=999,
    ...                         mask_columns=["C","D"])
    >>> print(res2)
    ... # Rows where A=0 => columns C,D replaced by 999, while B remains unchanged
    >>>
    """
    # --- Preliminary checks --- #
    if ref_col not in data.columns:
        msg = (f"[mask_by_reference] Column '{ref_col}' not found "
               f"in the DataFrame.")
        if error == "raise":
            raise KeyError(msg)
        elif error == "warn":
            warnings.warn(msg)
            return data  # return as is
        else:
            return data  # error=='ignore'

    # Decide whether to operate on a copy or in place
    df = data if inplace else data.copy()

    # Determine which columns we'll mask
    if mask_columns is None:
        # mask all except ref_col
        mask_cols = [c for c in df.columns if c != ref_col]
    else:
        # Convert a single string to list
        if isinstance(mask_columns, str):
            mask_columns = [mask_columns]

        # Check that columns exist
        not_found = [col for col in mask_columns if col not in df.columns]
        if len(not_found) > 0:
            msg_cols = (f"[mask_by_reference] The following columns were "
                        f"not found in DataFrame: {not_found}.")
            if error == "raise":
                raise KeyError(msg_cols)
            elif error == "warn":
                warnings.warn(msg_cols)
                # Remove them from mask list if ignoring/warning
                mask_columns = [c for c in mask_columns if c in df.columns]
            else:
                pass  # silently ignore
        mask_cols = [c for c in mask_columns if c != ref_col]

    if verbose > 1:
        print(f"[mask_by_reference] Columns to be masked: {mask_cols}")

    # If values is None => mask all rows in mask_cols
    if values is None:
        if verbose > 0:
            print("[mask_by_reference] 'values' is None. Masking ALL rows.")
        if fill_value == 'auto':
            # 'auto' doesn't make sense with None => fill with None
            if verbose > 0:
                print("[mask_by_reference] 'fill_value=auto' but no values "
                      "specified. Will use None for fill.")
            df[mask_cols] = None
        else:
            df[mask_cols] = fill_value
        return df

    # Convert single value to a list
    if not isinstance(values, (list, tuple, set)):
        values = [values]

    ref_series = df[ref_col]
    is_numeric = pd.api.types.is_numeric_dtype(ref_series)

    # If find_closest and ref_series isn't numeric => revert to exact
    if find_closest and not is_numeric:
        if verbose > 0:
            print("[mask_by_reference] 'find_closest=True' but reference "
                  "column is not numeric. Reverting to exact matching.")
        find_closest = False

    total_matched_rows = set()  # track distinct row indices matched

    # Loop over each value and find matched rows
    for val in values:
        if find_closest:
            # Approximate match for numeric
            distances = (ref_series - val).abs()
            min_dist = distances.min()
            # If min_dist is inf, no numeric interpretation possible
            if min_dist == np.inf:
                matched_idx = []
            else:
                matched_idx = distances[distances == min_dist].index
        else:
            # Exact match
            matched_idx = ref_series[ref_series == val].index

        if len(matched_idx) == 0:
            # No match found for val
            msg_val = (
                f"[mask_by_reference] No matching value found for '{val}'"
                f" in column '{ref_col}'. Ensure '{val}' exists in "
                f"'{ref_col}' before applying the mask, or set"
                " ``find_closest=True`` to select the closest match."
            )
            if find_closest:
                msg_val = (f"[mask_by_reference] Could not approximate '{val}' "
                           f"in numeric column '{ref_col}'.")
            if error == "raise":
                raise ValueError(msg_val)
            elif error == "warn":
                warnings.warn(msg_val)
                continue  # skip
            else:
                continue  # error=='ignore'
        else:
            # Decide the actual fill we use for these matches
            if fill_value == 'auto':
                fill = val
            else:
                fill = fill_value

            # Mask these matched rows
            df.loc[matched_idx, mask_cols] = fill

            # Accumulate matched indices
            total_matched_rows.update(matched_idx)

    if verbose > 0:
        distinct_count = len(total_matched_rows)
        print(f"[mask_by_reference] Distinct matched rows: {distinct_count}")

    return df