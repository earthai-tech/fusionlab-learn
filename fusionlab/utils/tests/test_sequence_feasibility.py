"""
tests/test_sequence_feasibility.py
==================================

Unit tests for `check_sequence_feasibility` and its helper machinery.

The tests cover:

* Correct counting with all three engines (`vectorized`, `loop`,
  `pyarrow` → fallback when PyArrow is absent).
* Equality of results across engines.
* Proper error / warning paths when no sequence is possible.
"""

from __future__ import annotations

import sys
from typing import Dict, Tuple, Union
from packaging.version import parse

import numpy as np
import pandas as pd
import pytest

from fusionlab.utils.forecast_utils import (
    check_sequence_feasibility,
    SequenceGeneratorError,
)

def _make_df() -> pd.DataFrame:
    """
    Two trajectories (“A” and “B”) with 10 consecutive points each.

    With `time_steps=5` and `forecast_horizon=4` the minimum length is 9,
    so each group yields exactly 2 sequences.
    """
    rng = np.random.default_rng(42)
    t = pd.date_range("2020-01-01", periods=10, freq="M")

    return pd.DataFrame(
        {
            "date": np.tile(t, 2),
            "site": ["A"] * 10 + ["B"] * 10,
            "val": rng.normal(size=20),
        }
    )


_EXPECTED_COUNTS: Dict[Union[str, Tuple], int] = {"A": 2, "B": 2}
_EXPECTED_TOTAL = 4
_ARGS = dict(
    time_col="date",
    group_id_cols=["site"],
    time_steps=5,
    forecast_horizon=4,
)


@pytest.mark.parametrize("engine", ["vectorized", "native"])
def test_counts_per_engine(engine):
    """vectorized & loop engines return identical correct results."""
    ok, counts = check_sequence_feasibility(
        _make_df(), engine=engine, verbose=0, **_ARGS
    )
    assert ok
    assert counts == _EXPECTED_COUNTS

@pytest.mark.skipif(
    parse(pd.__version__) < parse("2.0"),
    reason="PyArrow backend requires pandas ≥ 2.0",
)
def test_pyarrow_engine_fallback(monkeypatch):
    """
    When ``engine='pyarrow'`` but PyArrow is *not* installed,
    the function must fall back to the vectorised path and still
    produce correct counts.
    """
    # 1 Ensure 'pyarrow' is not already imported
    if "pyarrow" in sys.modules:
        monkeypatch.delitem(sys.modules, "pyarrow")  # no 'raising=' kw

    # 2 Pretend that importing pyarrow fails
    class _Sentinel:  # any non-module sentinel will work
        pass

    monkeypatch.setitem(sys.modules, "pyarrow", _Sentinel)

    ok, counts = check_sequence_feasibility(
        _make_df(),
        engine="pyarrow",   # triggers fallback inside the function
        verbose=0,
        **_ARGS,
    )

    assert ok
    assert counts == _EXPECTED_COUNTS


def test_total_sequences():
    """Total number of sequences equals the sum of the per-group counts."""
    ok, counts = check_sequence_feasibility(
        _make_df(), engine="vectorized", verbose=0, **_ARGS
    )
    assert ok
    assert sum(counts.values()) == _EXPECTED_TOTAL


def test_raise_on_insufficient_data():
    """`error='raise'` triggers a custom exception when no sequence fits."""
    df_short = _make_df().iloc[:3]  # 3 rows < min_len=9
    with pytest.raises(SequenceGeneratorError):
        check_sequence_feasibility(
            df_short,
            error="raise",
            engine="vectorized",
            **_ARGS,
        )

def test_warn_on_insufficient_data(recwarn):
    """`error='warn'` emits a UserWarning and returns (False, {})."""
    df_short = _make_df().iloc[:3]
    ok, counts = check_sequence_feasibility(
        df_short,
        error="warn",
        engine="vectorized",
        **_ARGS,
    )
    assert not ok
    print(counts)
    assert counts == {'A': 0} or counts == {}  # depending on implementation
    assert any(issubclass(w.category, UserWarning) for w in recwarn)


if __name__ =='__main__': # pragma : no cover 
    pytest.main( [__file__,  "--maxfail=1 ", "--disable-warnings",  "-q"])