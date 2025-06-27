import os
import json
import pytest
import numpy as np
import pandas as pd

from fusionlab.metrics import utils


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, tmp_path):
    """
    Stub out all external dependencies inside compute_quantile_diagnostics
    so we can control their behavior and side‐effects.
    """
    # 1. are_all_frames_valid should just return its inputs
    monkeypatch.setattr(utils, 'are_all_frames_valid', lambda *dfs, ops=None: dfs)

    # 2. validate_quantiles → identity
    monkeypatch.setattr(utils, 'validate_quantiles', lambda q, dtype=None: q)

    # 3. Stub the three metric functions to return known constants
    monkeypatch.setattr(utils, 'coverage_score', lambda actual, lo, hi: 0.123)
    monkeypatch.setattr(utils, 'prediction_stability_score', lambda y_pred: 0.456)
    monkeypatch.setattr(utils, 'quantile_calibration_error', lambda y_true, y_pred, quantiles: 0.789)

    # 4. Stub generic_utils.vlog and insert_affix_in
    import fusionlab.utils.generic_utils as gu
    monkeypatch.setattr(gu, 'vlog', lambda *args, **kwargs: None)
    monkeypatch.setattr(gu, 'insert_affix_in', lambda filename, affix, separator: filename)

    # 5. Stub io_utils.to_txt to actually write out JSON so we can assert on it
    import fusionlab.utils.io_utils as iu
    def fake_to_txt(results, format, indent, filename, savepath):
        path = os.path.join(savepath, filename)
        with open(path, 'w') as f:
            json.dump(results, f, indent=indent)
    monkeypatch.setattr(iu, 'to_txt', fake_to_txt)

    return str(tmp_path)


def test_stack_quantile_predictions_1d():
    lower = [0, 1, 2]
    median = [0.5, 1.5, 2.5]
    upper = [1, 2, 3]
    y_pred = utils.stack_quantile_predictions(lower, median, upper)

    # should promote 1D → shape (1, 3, 3)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (1, 3, 3)
    np.testing.assert_allclose(y_pred[0, 0, :], lower)
    np.testing.assert_allclose(y_pred[0, 1, :], median)
    np.testing.assert_allclose(y_pred[0, 2, :], upper)


def test_stack_quantile_predictions_2d():
    lower = [[0, 1, 2], [3, 4, 5]]
    median = [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]
    upper = [[1, 2, 3], [4, 5, 6]]
    y_pred = utils.stack_quantile_predictions(np.array(lower),
                                              np.array(median),
                                              np.array(upper))

    # should keep batch dimension: shape (2,3,3)
    assert y_pred.shape == (2, 3, 3)
    for i in range(2):
        np.testing.assert_allclose(y_pred[i, 0, :], lower[i])
        np.testing.assert_allclose(y_pred[i, 1, :], median[i])
        np.testing.assert_allclose(y_pred[i, 2, :], upper[i])


def test_stack_quantile_predictions_mismatched_raises():
    # lower and upper lengths differ → ValueError
    with pytest.raises(ValueError):
        utils.stack_quantile_predictions([0,1],
                                          [0.5,1.5,2.5],
                                          [1,2])


def test_compute_quantile_diagnostics_success(patch_dependencies):
    tmp_dir = patch_dependencies

    # Build a minimal DataFrame with the required columns:
    # - test_actual
    # - test_q10  (lower)
    # - test_q50  (median for odd count)
    # - test_q90  (upper)
    df = pd.DataFrame({
        'test_actual': [1, 2],
        'test_q10':    [0, 1],
        'test_q50':    [1, 2],
        'test_q90':    [2, 3],
    })

    results = utils.compute_quantile_diagnostics(
        df,
        base_name='test',
        quantiles=[0.1, 0.5, 0.9],
        coverage_quantile_indices=(0, -1),
        savefile=None,
        savepath=tmp_dir,
        filename='diagnostics.json',
        name='runA',
        verbose=0,
        logger=None,
    )

    # Should return our stubbed values
    assert results == {'coverage': 0.123, 'pss': 0.456, 'qce': 0.789}

    # And it should have written out diagnostics.json
    out_path = os.path.join(tmp_dir, 'diagnostics.json')
    assert os.path.exists(out_path)
    saved = json.loads(open(out_path).read())
    assert saved == results


def test_compute_quantile_diagnostics_invalid_index(patch_dependencies):
    # Only two quantiles → indices (0,2) is out of range
    df = pd.DataFrame({
        'test_actual': [0],
        'test_q10':    [0],
        'test_q90':    [1],
    })
    with pytest.raises(ValueError):
        utils.compute_quantile_diagnostics(
            df,
            base_name='test',
            quantiles=[0.1, 0.9],
            coverage_quantile_indices=(0, 2),
        )


def test_compute_quantile_diagnostics_missing_column(patch_dependencies):
    # Missing the upper‐quantile column → ValueError
    df = pd.DataFrame({
        'test_actual': [0],
        'test_q10':    [0],
        # 'test_q90' is absent
    })
    with pytest.raises(ValueError):
        utils.compute_quantile_diagnostics(
            df,
            base_name='test',
            quantiles=[0.1, 0.9],
            coverage_quantile_indices=(0, -1),
        )

if __name__ =='__main__': 
    pytest.main( [__file__])
    