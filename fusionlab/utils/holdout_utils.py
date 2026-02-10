
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


HoldoutStrategy = Literal["random", "spatial_block"]


@dataclass(frozen=True)
class GroupMasks:
    """Group-level validity masks for early filtering."""

    required_train_years: list[int]
    required_forecast_years: list[int]
    valid_for_train: pd.DataFrame
    valid_for_forecast: pd.DataFrame

    @property
    def keep_for_processing(self) -> pd.DataFrame:
        """Union(valid_for_train, valid_for_forecast)."""
        if self.valid_for_train.empty:
            return self.valid_for_forecast.copy()
        if self.valid_for_forecast.empty:
            return self.valid_for_train.copy()
        a = self.valid_for_train.copy()
        b = self.valid_for_forecast.copy()
        out = pd.concat([a, b], axis=0, ignore_index=True)
        out = out.drop_duplicates()
        return out


def _year_span(end_year: int, n_years: int) -> list[int]:
    end_year = int(end_year)
    n_years = int(n_years)
    start = end_year - n_years + 1
    return list(range(start, end_year + 1))


def _groups_with_all_years(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    time_col: str,
    required_years: list[int],
) -> pd.DataFrame:
    """
    Return unique group rows (group_cols) that contain ALL required_years.

    Efficient approach:
      - drop duplicate (group,time)
      - filter to required years
      - count unique years per group
    """
    if not required_years:
        return df[group_cols].drop_duplicates().copy()

    d0 = df[group_cols + [time_col]].drop_duplicates()
    d0 = d0[d0[time_col].isin(required_years)]
    if d0.empty:
        return d0[group_cols].drop_duplicates().copy()

    need = len(required_years)
    cnt = d0.groupby(group_cols, sort=False)[time_col].nunique()
    ok = cnt[cnt == need].index

    if isinstance(ok, pd.MultiIndex):
        out = ok.to_frame(index=False)
        out.columns = group_cols
        return out.reset_index(drop=True)

    # single group col
    return pd.DataFrame({group_cols[0]: ok}).reset_index(drop=True)


def compute_group_masks(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    time_col: str,
    train_end_year: int,
    time_steps: int,
    horizon: int,
) -> GroupMasks:
    """
    Build:
      - valid_for_train: groups containing all years for last (T+H)
      - valid_for_forecast: groups containing all years for last T

    This assumes annual steps and integer years in time_col.
    """
    req_train = _year_span(train_end_year, time_steps + horizon)
    req_fcst = _year_span(train_end_year, time_steps)

    v_train = _groups_with_all_years(
        df,
        group_cols=group_cols,
        time_col=time_col,
        required_years=req_train,
    )
    v_fcst = _groups_with_all_years(
        df,
        group_cols=group_cols,
        time_col=time_col,
        required_years=req_fcst,
    )

    return GroupMasks(
        required_train_years=req_train,
        required_forecast_years=req_fcst,
        valid_for_train=v_train,
        valid_for_forecast=v_fcst,
    )


def _hash_groups(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    """
    Stable-ish 64-bit hash per row for group membership filtering.

    Collisions are theoretically possible but extremely unlikely.
    """
    return pd.util.hash_pandas_object(
        df[group_cols],
        index=False,
    ).astype("uint64")


def filter_df_by_groups(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only rows in df whose (group_cols) exist in groups DataFrame.
    """
    if groups is None or groups.empty:
        return df.iloc[0:0].copy()

    h_df = _hash_groups(df, group_cols)
    h_g = _hash_groups(groups.drop_duplicates(), group_cols)
    keep = h_df.isin(h_g.to_numpy())
    return df.loc[keep].copy()


@dataclass(frozen=True)
class HoldoutSplit:
    """Pixel holdout split (disjoint groups)."""

    train_groups: pd.DataFrame
    val_groups: pd.DataFrame
    test_groups: pd.DataFrame

    def check_disjoint(self) -> None:
        g = ["__h__"]
        a = self.train_groups.copy()
        b = self.val_groups.copy()
        c = self.test_groups.copy()
        a[g[0]] = _hash_groups(a, list(a.columns))
        b[g[0]] = _hash_groups(b, list(b.columns))
        c[g[0]] = _hash_groups(c, list(c.columns))
        sa = set(a[g[0]].to_numpy())
        sb = set(b[g[0]].to_numpy())
        sc = set(c[g[0]].to_numpy())
        if sa & sb or sa & sc or sb & sc:
            raise RuntimeError("Holdout groups are not disjoint.")


def split_groups_holdout(
    groups: pd.DataFrame,
    *,
    seed: int = 42,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    strategy: HoldoutStrategy = "random",
    x_col: str | None = None,
    y_col: str | None = None,
    block_size: float | None = None,
) -> HoldoutSplit:
    """
    Split unique groups into train/val/test (pixel-level holdout).

    strategy:
      - "random": shuffle groups
      - "spatial_block": shuffle spatial blocks (needs x_col,y_col,block)
    """
    if groups is None or groups.empty:
        z = groups.iloc[0:0].copy()
        return HoldoutSplit(z, z, z)

    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError("Bad val/test fractions.")

    g = groups.drop_duplicates().reset_index(drop=True)

    rng = np.random.default_rng(int(seed))

    if strategy == "spatial_block":
        if x_col is None or y_col is None:
            raise ValueError("spatial_block needs x_col and y_col.")
        if block_size is None or float(block_size) <= 0:
            raise ValueError("spatial_block needs block_size > 0.")

        bx = np.floor(g[x_col].to_numpy(float) / float(block_size))
        by = np.floor(g[y_col].to_numpy(float) / float(block_size))
        blocks = pd.DataFrame({"bx": bx.astype(int), "by": by.astype(int)})

        b_id = pd.util.hash_pandas_object(blocks, index=False)
        uniq = pd.DataFrame({"b": b_id}).drop_duplicates()
        perm = rng.permutation(len(uniq))

        n = len(uniq)
        n_test = max(1, int(round(test_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        n_train = max(1, n - n_val - n_test)

        b_train = set(uniq.iloc[perm[:n_train]]["b"].to_numpy())
        b_val = set(
            uniq.iloc[perm[n_train : n_train + n_val]]["b"].to_numpy()
        )
        b_test = set(uniq.iloc[perm[n_train + n_val :]]["b"].to_numpy())

        in_train = b_id.isin(b_train).to_numpy()
        in_val = b_id.isin(b_val).to_numpy()
        in_test = b_id.isin(b_test).to_numpy()

        tr = g.loc[in_train].copy()
        va = g.loc[in_val].copy()
        te = g.loc[in_test].copy()

    else:
        perm = rng.permutation(len(g))
        n = len(g)
        n_test = max(1, int(round(test_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        n_train = max(1, n - n_val - n_test)

        tr = g.iloc[perm[:n_train]].copy()
        va = g.iloc[perm[n_train : n_train + n_val]].copy()
        te = g.iloc[perm[n_train + n_val :]].copy()

    # hard safety: ensure disjoint (by hash)
    out = HoldoutSplit(tr, va, te)
    out.check_disjoint()
    return out
