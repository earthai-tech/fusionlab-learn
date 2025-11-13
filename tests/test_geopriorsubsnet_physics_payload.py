# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pytest

from fusionlab.nn.pinn.io import (
    default_meta_from_model,
    _maybe_subsample,
    save_physics_payload,
    load_physics_payload,
    gather_physics_payload,
)


class FakeGeoPriorSubsNet:
    """
    Lightweight stand-in for GeoPriorSubsNet to test payload export/load
    without invoking TensorFlow.

    - Exposes attributes read by default_meta_from_model
    - Implements evaluate_physics(..., return_maps=True) returning the
      required dict with arrays
    - Provides wrappers export_physics_payload/load_physics_payload that
      delegate to io.py (matching the intended API on GeoPriorSubsNet)
    """

    # --- attributes that default_meta_from_model inspects ---
    def __init__(self, use_effective_thickness=True, Hd_factor=0.8):
        self.pde_modes_active = ["consolidation", "gw_flow"]
        self.kappa_mode = "bar"
        self.use_effective_thickness = use_effective_thickness
        self.Hd_factor = Hd_factor
        self.lambda_cons = 1.0
        self.lambda_gw = 1.0
        self.lambda_prior = 0.5
        self.lambda_smooth = 1.0
        self.lambda_mv = 0.0
        self.quantiles = [0.1, 0.5, 0.9]

    def get_config(self):
        # default_meta_from_model will try get("model_version")
        return {"model_version": "3.0-GeoPrior"}

    # --- the only runtime method gather_physics_payload needs ---
    def evaluate_physics(self, inputs, return_maps=True):
        """
        Return synthetic but consistent physics maps. Shape convention:
        (B, H, 1). We ignore 'inputs' since this is a unit test.
        """
        B, H = 2, 4
        # make distinct numeric ranges so the test can assert lengths/values
        K = np.full((B, H, 1), 10.0, dtype=np.float32)
        Ss = np.full((B, H, 1), 0.005, dtype=np.float32)
        H_eff = np.full((B, H, 1), 25.0 * self.Hd_factor, dtype=np.float32)
        tau = np.full((B, H, 1), 2.0, dtype=np.float32)
        tau_prior = np.full((B, H, 1), 1.8, dtype=np.float32)
        R_cons = np.linspace(-0.1, 0.1, B * H, dtype=np.float32).reshape(B, H, 1)

        # metrics (not used by gather..., but realistic)
        eps_prior = np.array(0.05, dtype=np.float32)
        eps_cons = np.array(0.02, dtype=np.float32)

        out = {
            "epsilon_prior": eps_prior,
            "epsilon_cons": eps_cons,
        }
        if return_maps:
            out.update(
                {
                    "R_prior": np.zeros_like(R_cons),
                    "R_cons": R_cons,
                    "K": K,
                    "Ss": Ss,
                    "H": H_eff,
                    "tau": tau,
                    "tau_prior": tau_prior,
                }
            )
        return out

    # ---- thin wrappers that mirror the class methods you added ----
    def export_physics_payload(
        self,
        dataset,
        max_batches=None,
        save_path=None,
        format="npz",
        overwrite=False,
        metadata=None,
        random_subsample=None,
        float_dtype=np.float32,
    ):
        payload = gather_physics_payload(
            self, dataset, max_batches=max_batches, float_dtype=float_dtype
        )
        if random_subsample is not None:
            payload = _maybe_subsample(payload, random_subsample)

        if save_path is not None:
            meta = default_meta_from_model(self)
            if metadata:
                meta.update(metadata)
            save_physics_payload(
                payload, meta, save_path, format=format, overwrite=overwrite
            )
        return payload

    @staticmethod
    def load_physics_payload(path):
        return load_physics_payload(path)


# ----------------------------- fixtures -------------------------------------


@pytest.fixture
def fake_model():
    return FakeGeoPriorSubsNet(use_effective_thickness=True, Hd_factor=0.75)


@pytest.fixture
def tiny_dataset():
    """
    Minimal iterable that yields 3 batches. Each batch is either an inputs
    dict or (inputs, targets). Our Fake model ignores contents anyway.
    """
    dummy_inputs = {"H_field": np.ones((2, 4, 1), dtype=np.float32)}
    batches = [(dummy_inputs, None), (dummy_inputs, None), (dummy_inputs, None)]
    return batches


# ------------------------------ tests ---------------------------------------


def test_default_meta_from_model(fake_model):
    meta = default_meta_from_model(fake_model)
    # core keys present
    for key in [
        "created_utc",
        "model_name",
        "model_version",
        "pde_modes_active",
        "kappa_mode",
        "use_effective_thickness",
        "Hd_factor",
        "lambda_cons",
        "lambda_gw",
        "lambda_prior",
        "lambda_smooth",
        "lambda_mv",
        "quantiles",
    ]:
        assert key in meta
    # a few semantic checks
    assert meta["model_name"] == "FakeGeoPriorSubsNet"
    assert meta["model_version"] == "3.0-GeoPrior"
    assert meta["use_effective_thickness"] is True
    assert math.isclose(meta["Hd_factor"], 0.75, rel_tol=1e-6)
    assert meta["pde_modes_active"] == ["consolidation", "gw_flow"]


def test__maybe_subsample_shapes_and_fraction():
    N = 1000
    payload = {
        "tau": np.arange(N, dtype=np.float32),
        "tau_prior": np.arange(N, dtype=np.float32) * 0.9,
        "K": np.full(N, 10.0, dtype=np.float32),
        "Ss": np.full(N, 0.01, dtype=np.float32),
        "Hd": np.full(N, 25.0, dtype=np.float32),
        "cons_res_vals": np.zeros(N, dtype=np.float32),
        "misc": "kept as-is (not an array)",  # non-array should pass through
    }
    out = _maybe_subsample(payload, frac=0.2)
    # length should be close to ceil(0.2 * 1000) = 200
    assert out["tau"].shape[0] == int(math.ceil(0.2 * N))
    # non-array key preserved
    assert out["misc"] == "kept as-is (not an array)"
    # value correspondence: selected subset must be from original set
    assert set(out["K"]).issubset({10.0})


def test_save_and_load_npz_roundtrip(tmp_path):
    path = tmp_path / "phys_payload.npz"
    payload = {
        "tau": np.array([1.0, 2.0], dtype=np.float32),
        "tau_prior": np.array([0.9, 1.8], dtype=np.float32),
        "K": np.array([10.0, 11.0], dtype=np.float32),
        "Ss": np.array([0.01, 0.02], dtype=np.float32),
        "Hd": np.array([20.0, 21.0], dtype=np.float32),
        "cons_res_vals": np.array([0.0, 0.1], dtype=np.float32),
        "log10_tau": np.log10(np.array([1.0, 2.0], dtype=np.float32)),
        "log10_tau_prior": np.log10(np.array([0.9, 1.8], dtype=np.float32)),
    }
    meta = {"city": "Nansha", "split": "val"}

    # save once
    saved_path = save_physics_payload(payload, meta, str(path), format="npz", overwrite=True)
    assert os.path.exists(saved_path)
    assert os.path.exists(saved_path + ".meta.json")

    # load back
    loaded_payload, loaded_meta = load_physics_payload(str(path))
    for k in ["tau", "tau_prior", "K", "Ss", "Hd", "cons_res_vals", "log10_tau", "log10_tau_prior"]:
        assert np.allclose(payload[k], loaded_payload[k])

    # meta roundtrip (sidecar JSON)
    assert loaded_meta["city"] == "Nansha"
    assert loaded_meta["split"] == "val"
    assert "saved_utc" in loaded_meta

    # saving again without overwrite should fail
    with pytest.raises(FileExistsError):
        save_physics_payload(payload, meta, str(path), format="npz", overwrite=False)


def test_gather_and_export_roundtrip_with_fake_model(fake_model, tiny_dataset, tmp_path):
    # 1) gather (no save)
    gathered = gather_physics_payload(fake_model, tiny_dataset, max_batches=None)
    # Our fake model returns B=2, H=4 per batch -> per batch 8 samples; 3 batches => 24
    assert gathered["tau"].shape[0] == 24
    assert gathered["tau_prior"].shape[0] == 24
    assert "metrics" in gathered and "r2_logtau" in gathered["metrics"]

    # 2) export with save_path
    outpath = tmp_path / "exported_phys_payload.npz"
    payload = fake_model.export_physics_payload(
        tiny_dataset,
        save_path=str(outpath),
        format="npz",
        overwrite=True,
        metadata={"city": "Zhongshan", "split": "test"},
    )
    # file was written and sidecar meta exists
    assert os.path.exists(outpath)
    assert os.path.exists(str(outpath) + ".meta.json")

    # payload returned matches the on-disk content
    loaded_payload, loaded_meta = fake_model.load_physics_payload(str(outpath))
    for key in ["tau", "tau_prior", "K", "Ss", "Hd", "cons_res_vals"]:
        assert np.allclose(payload[key], loaded_payload[key])

    # metadata contains defaults + our overrides
    assert loaded_meta["city"] == "Zhongshan"
    assert loaded_meta["split"] == "test"
    assert loaded_meta["model_name"] == "FakeGeoPriorSubsNet"
    assert loaded_meta["model_version"] == "3.0-GeoPrior"


def test_export_with_random_subsample(fake_model, tiny_dataset, tmp_path):
    outpath = tmp_path / "payload_subsampled.npz"
    payload = fake_model.export_physics_payload(
        tiny_dataset,
        save_path=str(outpath),
        format="npz",
        overwrite=True,
        random_subsample=0.25,  # keep ~25%
    )
    # Expect ~24 * 0.25 samples (ceil in helper)
    assert payload["tau"].shape[0] == int(math.ceil(24 * 0.25))
    # ensure the file was still saved and reloadable
    loaded_payload, _ = fake_model.load_physics_payload(str(outpath))
    assert loaded_payload["tau"].ndim == 1
