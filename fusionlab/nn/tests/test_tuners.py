# tests/test_tuners.py
#
# Minimal integration tests for XTFTTuner and TFTTuner.
# Run with:  pytest -q tests/test_tuners.py
#
# The suite is intentionally lightweight: one trial, one epoch,
# tiny tensors – just enough to verify the end‑to‑end contract
# *returns (best_hps, best_model, tuner)* without error.

import numpy as np
import pytest

# --- Attempt to import tuner functions and dependencies ---
try:
    from fusionlab.nn.forecast_tuner import XTFTTuner, TFTTuner
    # from fusionlab.core.io import _get_valid_kwargs # Not used in this test file
    import keras_tuner as kt
    FUSIONLAB_INSTALLED = True
    HAS_KT = True
except ImportError as e:
    print(f"Skipping forecast_tuner tests due to import error: {e}")
    FUSIONLAB_INSTALLED = False
    HAS_KT = False
    class XTFT: pass
    class SuperXTFT: pass
    class TFTFlexible: pass # Use alias
    class TFTStricter: pass # Use alias
    def xtft_tuner(*args, **kwargs): raise ImportError("xtft_tuner not found")
    def tft_tuner(*args, **kwargs): raise ImportError("tft_tuner not found")
    class kt:
        class Tuner: pass
    
# --- End Imports ---
# XXX TO OPTIMIZE later : SKIP for Now; 
# HAS_KT =False 

pytestmark = pytest.mark.skipif(
    not (FUSIONLAB_INSTALLED and HAS_KT),
    reason="Keras Tuner or fusionlab components not found"
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_data():
    """Create tiny but valid tensors for tuner smoke‑tests."""
    B, F, Ns, Nd, Nf, O = 8, 2, 2, 3, 1, 1
    rng = np.random.default_rng(123)

    X_static = rng.normal(size=(B, Ns)).astype("float32")
    X_dynamic = rng.normal(size=(B, F, Nd)).astype("float32")
    X_future = rng.normal(size=(B, F, Nf)).astype("float32")
    y = rng.normal(size=(B, F, O)).astype("float32")

    return X_static, X_dynamic, X_future, y, F


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
def _run_quick_tune(tuner_cls, model_name, inputs, y, fh):
    tuner = tuner_cls(
        model_name=model_name,
        max_trials=1,
        epochs=1,               # refit epochs
        batch_sizes=[2],
        search_epochs=1,        # speed‑up search phase
        validation_split=0.25,
        verbose=3,
    )

    best_hps, best_model, tuner_obj = tuner.fit(
        inputs=inputs,
        y=y,
        forecast_horizon=fh,
    )

    # --- Assertions ------------------------------------------------
    assert isinstance(best_hps, dict)
    assert best_model is not None
    assert tuner_obj is not None
    assert best_hps["batch_size"] == 2
    assert "learning_rate" in best_hps


# ------------------------------------------------------------------
# Parametrised smoke tests
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "tuner_cls, model_name, needs_static_future",
    [
        (XTFTTuner, "xtft", True),
        (XTFTTuner, "super_xtft", True),
        (TFTTuner, "tft", True),
        (TFTTuner, "tft_flex", False),  # flexible variant works w/o static/future
    ],
)
def test_tuner_smoke(
    synthetic_data,
    tuner_cls,
    model_name,
    needs_static_future,
):
    X_s, X_d, X_f, y, fh = synthetic_data

    if needs_static_future:
        inputs = [X_s, X_d, X_f]
    else:
        inputs = [None, X_d, None]

    _run_quick_tune(tuner_cls, model_name, inputs, y, fh)


if __name__=='__main__': 
    pytest.main([__file__])