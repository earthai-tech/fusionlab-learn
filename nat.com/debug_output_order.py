# -*- coding: utf-8 -*-
"""
Debug output ordering between:
  (A) in-memory model (same process)
  (B) loaded inference model (compat.keras)

It prints:
- call() output keys and shapes
- predict() output structure (dict vs list)
- inferred mapping (predict list index -> key)
- stage2-like mapping behavior
- quantile monotonicity check

Run:
  python debug_output_order.py --outdir _tmp_order_dbg

Optional:
  python debug_output_order.py --outdir _tmp_order_dbg --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf


# -----------------------------
# Compat imports (best-effort)
# -----------------------------
def _import_compat_loader():
    tried = []
    for mod in (
        "fusionlab.compat.keras",
        "compat.keras",
        "geoprior.compat.keras",
    ):
        try:
            m = __import__(mod, fromlist=["*"])
            return m
        except Exception as e:
            tried.append((mod, str(e)))
    raise ImportError(
        "Could not import compat.keras. Tried:\n"
        + "\n".join([f"- {a}: {b}" for a, b in tried])
    )


# -----------------------------
# Synthetic batch generation
# -----------------------------
def _try_make_windows_batch(
    *,
    lookback: int,
    horizon: int,
    step: int,
    n_years: int,
    n_static: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Try to use your sm3_synthetic_identifiability.make_windows().
    If not available, raise to fallback generator.
    """
    try:
        from sm3_synthetic_identifiability import make_windows
    except Exception as e:
        raise RuntimeError(str(e))

    years = np.arange(2000, 2000 + n_years).astype(np.float32)

    # Simple identifiability-like driver:
    # dh = drawdown (positive)
    t = np.linspace(0.0, 1.0, n_years).astype(np.float32)
    dh = (0.6 * t + 0.15 * np.sin(2 * np.pi * t)).astype(np.float32)

    # settlement cumulative proxy
    s_cum = np.cumsum(dh).astype(np.float32)

    static_vec = np.linspace(0.1, 1.0, n_static).astype(np.float32)

    
    X, y, t_rng = make_windows(
        years,
        dh,
        s_cum,
        static_vec,
        time_steps=lookback,              # was lookback=
        forecast_horizon=horizon,         # was horizon=
        H_field_value=1.0,                # any float is fine for this test
        z_surf_static_index=0,            # REQUIRED kw-only arg
    )

    # Force target keys to the model contract if needed.
    # (Some generators use y_subs/y_gwl; we normalize below.)
    y_out: Dict[str, np.ndarray] = {}

    if isinstance(y, dict):
        if "subs_pred" in y:
            y_out["subs_pred"] = y["subs_pred"]
        elif "subs" in y:
            y_out["subs_pred"] = y["subs"]
        elif "y_subs" in y:
            y_out["subs_pred"] = y["y_subs"]

        if "gwl_pred" in y:
            y_out["gwl_pred"] = y["gwl_pred"]
        elif "gwl" in y:
            y_out["gwl_pred"] = y["gwl"]
        elif "y_gwl" in y:
            y_out["gwl_pred"] = y["y_gwl"]

    if "subs_pred" not in y_out:
        # best-effort fallback: take first item
        k0 = list(y.keys())[0]
        y_out["subs_pred"] = y[k0]
    if "gwl_pred" not in y_out:
        # fallback: zeros (not used for order test)
        shp = y_out["subs_pred"].shape
        y_out["gwl_pred"] = np.zeros(shp, dtype=np.float32)

    return X, y_out


def _fallback_batch(
    *,
    lookback: int,
    horizon: int,
    n_static: int,
    n_samples: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Minimal synthetic batch if make_windows isn't importable.
    """
    rng = np.random.default_rng(0)

    x_dyn = rng.normal(size=(n_samples, lookback, 2)).astype(np.float32)
    x_static = rng.normal(size=(n_samples, n_static)).astype(np.float32)

    # Targets: (B, H) (we'll broadcast to quantiles in loss)
    y_subs = rng.normal(size=(n_samples, horizon)).astype(np.float32)
    y_gwl = rng.normal(size=(n_samples, horizon)).astype(np.float32)

    X = {"x_dyn": x_dyn, "x_static": x_static}
    y = {"subs_pred": y_subs, "gwl_pred": y_gwl}
    return X, y


# -----------------------------
# Toy dict-output model
# (works even if your real model
# import is painful)
# -----------------------------
@tf.keras.utils.register_keras_serializable()
class ToyDictModel(tf.keras.Model):
    def __init__(self, *, horizon: int, quantiles: List[float]):
        super().__init__()
        self.h = int(horizon)
        self.qs = list(quantiles)
        self.q = len(self.qs)

        self.d1 = tf.keras.layers.Dense(64, activation="relu")
        self.d2 = tf.keras.layers.Dense(64, activation="relu")

        self.subs_head = tf.keras.layers.Dense(self.h * self.q)
        self.gwl_head = tf.keras.layers.Dense(self.h * self.q)

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False):
        flat = []
        for k in sorted(inputs.keys()):
            v = tf.cast(inputs[k], tf.float32)
            flat.append(tf.reshape(v, [tf.shape(v)[0], -1]))
        x = tf.concat(flat, axis=1)

        x = self.d1(x, training=training)
        x = self.d2(x, training=training)

        s = self.subs_head(x)
        g = self.gwl_head(x)

        s = tf.reshape(s, [-1, self.h, self.q, 1])
        g = tf.reshape(g, [-1, self.h, self.q, 1])

        # IMPORTANT: dict output (your real model does this too)
        return {"subs_pred": s, "gwl_pred": g}
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "horizon": self.h,
                "quantiles": self.qs,
            }
        )
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(
            horizon=config["horizon"],
            quantiles=config["quantiles"],
        )

def _as_rank4(y: tf.Tensor) -> tf.Tensor:
    y = tf.cast(y, tf.float32)

    if len(y.shape) == 4:
        return y
    if len(y.shape) == 3:
        # (B,H,Q) or (B,H,1) -> (B,H,Q,1)
        return y[..., None]
    if len(y.shape) == 2:
        # (B,H) -> (B,H,1,1)
        return y[:, :, None, None]

    raise ValueError(f"bad rank: {len(y.shape)}")


def _loss_quantile_mse(y_true, y_pred):
    yp = _as_rank4(y_pred)
    yt = _as_rank4(y_true)
    return tf.reduce_mean(tf.square(yp - yt))


# -----------------------------
# Output diagnostics
# -----------------------------
def _np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _shape_str(x: Any) -> str:
    try:
        return str(tuple(x.shape))
    except Exception:
        return str(x)


def _quantile_monotonicity(arr: np.ndarray, q_axis: int = 2) -> float:
    a = _np(arr)
    if a.ndim <= q_axis:
        return 0.0
    q = a.shape[q_axis]
    if q < 2:
        return 1.0

    ok = np.ones(a.shape[:q_axis] + a.shape[q_axis+1:], dtype=bool)
    for i in range(q - 1):
        qi = np.take(a, i, axis=q_axis)
        qj = np.take(a, i + 1, axis=q_axis)
        ok &= (qi <= qj)
    return float(np.mean(ok))


def _infer_list_mapping(
    call_out: Dict[str, np.ndarray],
    pred_list: List[np.ndarray],
) -> Dict[int, str]:
    """
    Find which pred_list[i] matches which call_out[key]
    by smallest RMS difference.
    """
    mapping: Dict[int, str] = {}
    keys = list(call_out.keys())

    for i, a in enumerate(pred_list):
        best_k = None
        best = None
        for k in keys:
            b = call_out[k]
            if a.shape != b.shape:
                continue
            d = float(np.sqrt(np.mean((a - b) ** 2)))
            if best is None or d < best:
                best = d
                best_k = k
        mapping[i] = best_k if best_k is not None else "<?>"
    return mapping


def _stage2_like_pred_dict(
    model: tf.keras.Model,
    pred_out: Any,
) -> Dict[str, np.ndarray]:
    """
    Mimic the stage2 logic you currently use:
    - if dict: keep it
    - elif list/tuple and output_names exist: zip
    - else: assume ['subs_pred','gwl_pred']
    """
    if isinstance(pred_out, dict):
        return {k: _np(v) for k, v in pred_out.items()}

    if isinstance(pred_out, (list, tuple)):
        out_names = getattr(model, "output_names", []) or []
        if len(out_names) == len(pred_out) and len(out_names) > 0:
            return {k: _np(v) for k, v in zip(out_names, pred_out)}

        if len(pred_out) >= 2:
            return {
                "subs_pred": _np(pred_out[0]),
                "gwl_pred": _np(pred_out[1]),
            }

    raise TypeError(f"Unsupported predict() type: {type(pred_out)}")


def _run_one(
    *,
    tag: str,
    model: tf.keras.Model,
    x_batch: Dict[str, tf.Tensor],
):
    print("\n" + "=" * 72)
    print(f"[{tag}] call() outputs")
    out_call = model(x_batch, training=False)

    if not isinstance(out_call, dict):
        raise TypeError(
            f"{tag}: call() did not return dict. "
            f"Got: {type(out_call)}"
        )

    call_np = {k: _np(v) for k, v in out_call.items()}

    for k in sorted(call_np.keys()):
        print(f"  - {k:10s} shape={call_np[k].shape}")

    # Quantile monotonicity sanity
    if "subs_pred" in call_np:
        frac = _quantile_monotonicity(call_np["subs_pred"])
        print(f"  subs q-monotone frac: {frac:.3f}")

    print("\n" + "-" * 72)
    print(f"[{tag}] predict() outputs")
    try:
        pred_out = model.predict(x_batch, verbose=0)
    except TypeError:
        # Some Keras versions dislike dict-of-tensors in predict
        xb = {k: _np(v) for k, v in x_batch.items()}
        pred_out = model.predict(xb, verbose=0)

    print(f"  predict type: {type(pred_out)}")
    if isinstance(pred_out, dict):
        for k in sorted(pred_out.keys()):
            print(f"  - {k:10s} shape={_shape_str(pred_out[k])}")
    elif isinstance(pred_out, (list, tuple)):
        print(f"  predict list len={len(pred_out)}")
        for i, a in enumerate(pred_out):
            print(f"  - [{i}] shape={_shape_str(a)}")
    else:
        print(f"  predict value={pred_out}")

    # Infer mapping if list/tuple
    if isinstance(pred_out, (list, tuple)):
        mapping = _infer_list_mapping(call_np, list(pred_out))
        print("\n  inferred mapping (predict idx -> key):")
        for i in range(len(pred_out)):
            print(f"    [{i}] -> {mapping[i]}")

    # Stage2-like mapping test
    pred_dict = _stage2_like_pred_dict(model, pred_out)

    print("\n" + "-" * 72)
    print(f"[{tag}] stage2-like pred_dict keys")
    for k in sorted(pred_dict.keys()):
        print(f"  - {k:10s} shape={pred_dict[k].shape}")

    # Compare stage2-like mapping vs call()
    print("\n" + "-" * 72)
    print(f"[{tag}] compare stage2-like predict vs call()")
    for k in sorted(call_np.keys()):
        if k not in pred_dict:
            print(f"  - {k:10s} missing in pred_dict")
            continue
        a = pred_dict[k]
        b = call_np[k]
        if a.shape != b.shape:
            print(f"  - {k:10s} SHAPE MISMATCH {a.shape} {b.shape}")
            continue
        rms = float(np.sqrt(np.mean((a - b) ** 2)))
        print(f"  - {k:10s} rms_diff={rms:.6e}")


# -----------------------------
# Bundle save/load helpers
# -----------------------------
def _write_min_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

def _toy_builder(manifest: dict) -> tf.keras.Model:
    horizon = int(manifest.get("horizon", 3))
    quantiles = manifest.get("quantiles", [0.1, 0.5, 0.9])
    return ToyDictModel(horizon=horizon, quantiles=quantiles)

# -----------------------------
# Main
# -----------------------------
@dataclass(frozen=True)
class Args:
    outdir: str
    epochs: int
    batch_size: int
    lookback: int
    horizon: int
    step: int
    n_years: int
    n_static: int


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="_tmp_order_dbg")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lookback", type=int, default=6)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument(
        "--quantiles",
        default="0.1,0.5,0.9",
        help="comma-separated quantiles, e.g. 0.05,0.5,0.95",
    )
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--n-years", type=int, default=18)
    p.add_argument("--n-static", type=int, default=8)
    ns = p.parse_args()
    
    args = Args(
        outdir=str(ns.outdir),
        epochs=int(ns.epochs),
        batch_size=int(ns.batch_size),
        lookback=int(ns.lookback),
        horizon=int(ns.horizon),
        step=int(ns.step),
        n_years=int(ns.n_years),
        n_static=int(ns.n_static),
    )

    tf.keras.utils.set_random_seed(7)

    # --------- make one batch
    try:
        X_np, y_np = _try_make_windows_batch(
            lookback=args.lookback,
            horizon=args.horizon,
            step=args.step,
            n_years=args.n_years,
            n_static=args.n_static,
        )
        src = "sm3.make_windows"
    except Exception as e:
        X_np, y_np = _fallback_batch(
            lookback=args.lookback,
            horizon=args.horizon,
            n_static=args.n_static,
            n_samples=64,
        )
        src = f"fallback ({e})"

    print(f"\n[batch] source = {src}")
    print("[batch] X keys =", sorted(X_np.keys()))
    print("[batch] y keys =", sorted(y_np.keys()))

    ds = tf.data.Dataset.from_tensor_slices(
        (
            {k: tf.constant(v) for k, v in X_np.items()},
            {k: tf.constant(v) for k, v in y_np.items()},
        )
    )
    ds = ds.batch(args.batch_size).take(1)
    x_batch, y_batch = next(iter(ds))

    # --------- train in-memory model
    quantiles = [float(x) for x in ns.quantiles.split(",")]
    model_mem = ToyDictModel(
        horizon=args.horizon,
        quantiles=quantiles,
    )

    # Build (ensures weights exist)
    _ = model_mem(x_batch, training=False)

    model_mem.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={
            "subs_pred": _loss_quantile_mse,
            "gwl_pred": _loss_quantile_mse,
        },
    )

    # Train a tiny bit so weights are non-trivial
    model_mem.fit(
        x_batch,
        y_batch,
        epochs=args.epochs,
        verbose=0,
    )

    # --------- save bundle
    os.makedirs(args.outdir, exist_ok=True)

    # Minimal manifest for compat loader
    manifest_path = os.path.join(args.outdir, "manifest.json")
    _write_min_manifest(
        manifest_path,
        {
            "horizon": args.horizon,
            "quantiles": quantiles,
            "note": "toy manifest for output-order debug",
        },
    )

    # Try to save via compat.keras if available.
    # If not, we save a SavedModel + weights.
    # --- save weights-only bundle (robust for subclassed models)
    compat = _import_compat_loader()
    
    saved_ok =False 
    keras_path = None
    weights_path = os.path.join(args.outdir, "model.weights.h5")
    manifest_path = os.path.join(args.outdir, "manifest.json")

    if hasattr(compat, "save_bundle"):
        try:
            compat.save_bundle(
                model=model_mem,
                keras_path=keras_path,
                weights_path=weights_path,
                manifest_path=manifest_path,
                manifest={
                    "horizon": args.horizon,
                    "quantiles": quantiles,
                    "note": "toy manifest for output-order debug",
                },
                overwrite=True,
            )
            saved_ok = True
            print("\n[bundle] saved via compat.save_bundle()")
        except Exception as e:
            print("\n[bundle] compat.save_bundle failed:", e)

    if not saved_ok:
        # fallback: save weights + SavedModel
        wpath = os.path.join(args.outdir, "weights.weights.h5")
        spath = os.path.join(args.outdir, "saved_model")
        model_mem.save_weights(wpath)
        tf.saved_model.save(model_mem, spath)
        print("\n[bundle] saved fallback:")
        print("  - weights:", wpath)
        print("  - saved_model:", spath)

    # --------- load inference model via compat
    if hasattr(compat, "load_bundle_for_inference"):
        try:
            model_load = compat.load_bundle_for_inference(
                keras_path=None,
                weights_path=weights_path,
                manifest_path=manifest_path,
                builder=_toy_builder,
                build_inputs=x_batch,
                prefer_full_model=False,
            )
            print("\n[load] loaded via compat.load_bundle_for_inference()")
        except Exception as e:
            raise RuntimeError(
                "compat.load_bundle_for_inference failed: " + str(e)
            )
    else:
        raise RuntimeError(
            "compat.keras missing load_bundle_for_inference"
        )

    # Ensure loaded model is built
    _ = model_load(x_batch, training=False)

    # --------- run diagnostics
    _run_one(tag="IN_MEMORY", model=model_mem, x_batch=x_batch)
    _run_one(tag="LOADED", model=model_load, x_batch=x_batch)

    print("\nDONE.")


if __name__ == "__main__":
    main()
