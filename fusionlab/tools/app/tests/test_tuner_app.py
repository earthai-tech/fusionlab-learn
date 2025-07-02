# tests/test_tuner_app.py
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

FUSIONLAB_INSTALLED =True 
HAS_KT = True
try : 
    import tensorflow as tf 
    import numpy as np
    import pandas as pd 
    from fusionlab.tools.app.tuner import TunerApp 
except: 
   FUSIONLAB_INSTALLED =False 
   HAS_KT = False 

pytestmark = pytest.mark.skipif(
    not ( FUSIONLAB_INSTALLED and HAS_KT),
    reason="Keras Tuner or fusionlab components not found"
)

@pytest.fixture(autouse=True)
def _patch_registry(monkeypatch, tmp_path):
    """Replace ManifestRegistry with a no-op stub for the duration of a test."""
    class _StubRegistry:
        def __init__(self, *a, **k):
            self.root_dir = tmp_path / "runs"
            self.root_dir.mkdir(exist_ok=True)
            self._run_registry_path = None
        # API used by TunerApp ↓
        def new_run_dir(self, **_):
            run = self.root_dir / "run"
            run.mkdir(exist_ok=True)
            self._run_registry_path = run
            return run
        def import_manifest(self, p): return Path(p)
        @property
        def _manifest_filename(self): return "tuner_run_manifest.json"

    monkeypatch.setattr(
        "fusionlab.registry._manifest_registry.ManifestRegistry",
        _StubRegistry,
        raising=True,
    )

# ---------------------------------------------------------------------
# helpers ­– ultra-lightweight fakes for the heavy runtime objects
# ---------------------------------------------------------------------
@pytest.fixture
def dummy_cfg(tmp_path, monkeypatch):
    """Return a minimal SubsConfig stub & monkey-patch ctor used by TunerApp."""
    # 1) create a throw-away output dir
    out   = tmp_path / "run_dir"
    csv_path = tmp_path / "dummy.csv"
    csv_path.write_text("col\n1\n")          # tiny throw-awa
    out.mkdir()

    # 2) build a namespace with only the attrs that TunerApp uses
    cfg = SimpleNamespace(
        data_dir         = str(tmp_path),    # <-  NOT None
        data_filename    = csv_path.name,    # <-  NOT None
        # meta
        city_name          = "pytest_city",
        model_name         = "TransFlowSubsNet",
        registry_path      = None,
        # hyper-params
        epochs             = 1,
        fit_verbose        = 0,
        quantiles          = [0.5],
        pde_mode           = "both",
        pinn_coeff_c       = "learnable",
        lambda_cons        = 1.0,
        lambda_gw          = 1.0,
        save_format        = "keras",
        # sequence params
        forecast_horizon   = 3,
        save_intermediate =True,  # save artefacts 
        run_output_path = out, 
        # stub methods
        to_json=lambda *a, **k: None,          # no file writing
        
    )

    # DataProcessor → returns a DataFrame-like stub
    class _FakeProc:
        def __init__(self, cfg, log): pass
        def run(self, *a, **k):

            return pd.DataFrame({"year": [2000, 2001], "subsidence": [0.0, 0.1]})
    monkeypatch.setattr("fusionlab.tools.app.processing.DataProcessor", _FakeProc)

    # SequenceGenerator → returns two tf.data.Datasets with trivial shapes
    class _FakeSeq:
        def __init__(self, cfg, log):

            self.inputs_train  = {
                "coords":            np.zeros((1, 2)),
                "dynamic_features":  np.zeros((1, 2, 1)),
            }
            self.targets_train = {
                "subsidence": np.zeros((1, 3, 1)),
                "gwl":        np.zeros((1, 3, 1)),
            }
        def run(self, *a, **k):

            dummy = tf.data.Dataset.from_tensor_slices(({"x": [0]}, {"y": [0]}))
            return dummy, dummy               # train_ds , val_ds
    monkeypatch.setattr("fusionlab.tools.app.processing.SequenceGenerator", _FakeSeq)

    # HydroTuner → no real search, but emulates API used later
    class _FakeTuner:
        def __init__(self, *a, **k):
            self.objective = "val_loss"
            self._best_hp  = {"embed_dim": 32}
 
            self._best_model = tf.keras.Sequential(
                [tf.keras.layers.Dense(1, input_shape=[2])])
        def search(self, *a, **k): pass
        def get_best_hyperparameters(self, n): return [SimpleNamespace(values=self._best_hp)]
        def get_best_models(self, n):         return [self._best_model]
    monkeypatch.setattr("fusionlab.nn.forecast_tuner.HydroTuner", _FakeTuner)

    yield cfg

@pytest.fixture(autouse=True)
def patch_heavy_parts(monkeypatch):
    # 1) cheap pre-processing
    monkeypatch.setattr(
        "fusionlab.tools.app.processing.DataProcessor.run",
        lambda self, *a, **k: pd.DataFrame({"dummy": [0]})
    )

    # 2) lightweight SequenceGenerator.run
    def _fake_run(self, *a, **k):
        # minimal tensors with the right ranks
        dyn   = np.zeros((1, 1, 3), dtype="float32")   # (batch, t, dyn_feat)
        subs  = np.zeros((1, 1, 1), dtype="float32")
        gwl   = np.zeros((1, 1, 1), dtype="float32")

        # what the Tuner needs later:
        self.inputs_train = {
            "dynamic_features": dyn,
            "coords": np.zeros((1, 1, 2), dtype="float32"),
        }
        self.targets_train = {
            "subsidence": subs,
            "gwl":        gwl,
        }

        ds = tf.data.Dataset.from_tensor_slices((dyn, subs))  # dummy
        return ds, ds   # train_ds, val_ds

    monkeypatch.setattr(
        "fusionlab.tools.app.processing.SequenceGenerator.run",
        _fake_run
    )

def test_tuner_app_creates_manifest(dummy_cfg, tmp_path, monkeypatch):
    """Running .run() must finish and write tuner_run_manifest.json."""

    run_dir  = tmp_path / "smith_city_run"
    run_dir.mkdir()
    # force ManifestRegistry to use the tmp dir to stay fully isolated
    monkeypatch.setenv("FUSIONLAB_RUN_DIR", str(tmp_path))

    # --- run the tuner -------------------------------------------------
    app = TunerApp(
        cfg           = dummy_cfg,
        search_space  = {"embed_dim": [32, 64]},
        tuner_kwargs  = {"tuner_type": "randomsearch", "max_trials": 1},
        log_callback  = lambda *_: None,      # silence
    )
    model, best_hps, tuner = app.run()

    # --- assertions ----------------------------------------------------
    # 1) best artefacts returned
    assert best_hps.values["embed_dim"] == 32
    assert model is not None

    # 2) manifest exists & carries tuner_results
    mani_path = app._run_dir / "tuner_run_manifest.json"
    assert mani_path.exists(), "manifest not written"

    data = json.loads(mani_path.read_text())
    assert "tuner_results" in data
    assert data["tuner_results"]["best_hyperparameters"]["embed_dim"] == 32

if __name__=='__main__': 
    pytest.main([__file__])