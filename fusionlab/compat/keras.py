# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author : LKouadio <etanoyau@gmail.com>

"""
fusionlab.compat.keras

Small Keras compatibility helpers:
- Prefer standalone `keras`, fallback to `tf.keras`.
- Save/load portable bundles: model + weights + manifest.
- Robust inference loader with fallbacks:
  (1) load_model() on .keras/.h5 (and SavedModel in tf.keras)
  (2) rebuild via builder(manifest) + load_weights()
  (3) (Keras 3) TF SavedModel dir -> TFSMLayer wrapper
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional


__all__ = [
    "keras_major",
    "is_keras3",
    "save_manifest",
    "load_manifest",
    "save_bundle",
    "load_bundle_for_inference",
    "save_model",
    "load_model_from_tf",
]

CustomObjects = Optional[Dict[str, Any]]
Builder = Optional[Callable[[Dict[str, Any]], Any]]
LogFn = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------
# Keras import + version helpers
# ---------------------------------------------------------------------
def _import_keras():
    """
    Prefer standalone `keras` (Keras 3 style), then fallback
    to `tensorflow.keras` for TF-only runtimes.
    """
    try:
        import keras  # Keras 3 (or keras==2.15)

        return keras
    except Exception:
        from tensorflow import keras  # TF2.x fallback

        return keras


def _keras_version_str(keras_mod) -> str:
    v = getattr(keras_mod, "__version__", None)
    return str(v or "2.0.0")


def keras_major() -> int:
    """Return major Keras version as an int."""
    keras = _import_keras()
    v = _keras_version_str(keras)
    try:
        return int(v.split(".", 1)[0])
    except Exception:
        return 2


def is_keras3() -> bool:
    """Return True if Keras major version is >= 3."""
    return keras_major() >= 3


# ---------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------
def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _json_dump(obj: Any, path: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )


def _json_load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _custom_object_scope(keras_mod, custom_objects: CustomObjects):
    """
    Keras 3 uses: keras.saving.custom_object_scope
    Keras 2 uses: keras.utils.custom_object_scope
    """
    co = custom_objects or {}

    saving = getattr(keras_mod, "saving", None)
    if saving is not None:
        scope = getattr(saving, "custom_object_scope", None)
        if scope is not None:
            return scope(co)

    utils = getattr(keras_mod, "utils", None)
    if utils is not None:
        scope = getattr(utils, "custom_object_scope", None)
        if scope is not None:
            return scope(co)

    # Very defensive fallback: no-op context manager
    class _NoScope:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    return _NoScope()


def _extract_x(build_inputs: Any) -> Any:
    """
    Best-effort extractor for x from (x, y[, w]) batches.
    """
    if isinstance(build_inputs, (tuple, list)):
        if not build_inputs:
            return build_inputs
        return build_inputs[0]
    return build_inputs


def _log_default(_: str) -> None:
    return None

def _to_savedmodel_dir(path: str) -> str:
    """
    Convert a file-like path to a SavedModel directory.
    """
    base, ext = os.path.splitext(path)
    if ext in (".keras", ".h5"):
        return base + "_savedmodel"
    return path


def _clear_path(path: str, overwrite: bool) -> None:
    """
    Remove an existing directory if overwrite is True.
    """
    if not overwrite:
        return
    if not os.path.exists(path):
        return

    import shutil

    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            os.remove(path)
        except OSError:
            return

# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------
def save_manifest(path: str, payload: Dict[str, Any]) -> None:
    """Save JSON manifest (pretty + sorted keys)."""
    _json_dump(payload, path)


def load_manifest(path: str) -> Dict[str, Any]:
    """Load JSON manifest from disk."""
    obj = _json_load(path)
    return obj if isinstance(obj, dict) else {}


# ---------------------------------------------------------------------
# Bundle save/load
# ---------------------------------------------------------------------
def save_bundle(
    *,
    model: Any,
    keras_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    manifest: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> None:
    """
    Save a portable model bundle (any subset is allowed):

    1) Full model  (.keras or .h5)         -> keras_path
    2) Weights only (.weights.h5)          -> weights_path
    3) Manifest JSON (init params/meta)    -> manifest_path
    """
    _import_keras()  # ensure keras import side-effects

    if keras_path:
        _ensure_parent_dir(keras_path)
        model.save(
            keras_path,
            overwrite=overwrite,
        )

    if weights_path:
        _ensure_parent_dir(weights_path)
        model.save_weights(
            weights_path,
            overwrite=overwrite,
        )

    if manifest_path and manifest is not None:
        save_manifest(manifest_path, manifest)

def save_model(
    model: Any,
    keras_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    manifest: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
    use_tf_format: bool = False,
) -> None:
    """
    Save model artifacts.

    - Default: save_bundle() (native .keras/.h5 + weights + manifest)
    - TF format:
        * Keras 3  -> model.export(dir)
        * Keras 2  -> model.save(dir, save_format="tf")
    """
    keras = _import_keras()

    if manifest_path and manifest is not None:
        save_manifest(manifest_path, manifest)

    if not use_tf_format:
        save_bundle(
            model=model,
            keras_path=keras_path,
            weights_path=weights_path,
            manifest_path=None,
            manifest=None,
            overwrite=overwrite,
        )
        return

    if not keras_path:
        raise ValueError(
            "keras_path is required when use_tf_format=True."
        )

    tf_dir = _to_savedmodel_dir(keras_path)
    _ensure_parent_dir(tf_dir)
    _clear_path(tf_dir, overwrite=overwrite)

    # Optional: keep weights as a fallback artifact.
    if weights_path:
        _ensure_parent_dir(weights_path)
        try:
            model.save_weights(
                weights_path,
                overwrite=overwrite,
            )
        except Exception:
            pass

    # --- Keras 3: export() writes a SavedModel directory.
    if is_keras3() and hasattr(model, "export"):
        model.export(tf_dir)
        print(
            "[INFO] Exported TF SavedModel -> "
            f"{tf_dir}"
        )
        return

    # --- Keras 2 / tf.keras: save(..., save_format="tf")
    try:
        model.save(
            tf_dir,
            overwrite=overwrite,
            save_format="tf",
        )
        print(
            "[INFO] Saved TF SavedModel -> "
            f"{tf_dir}"
        )
        return
    except TypeError:
        # Some builds infer from directory path.
        model.save(tf_dir, overwrite=overwrite)
        print(
            "[INFO] Saved TF SavedModel -> "
            f"{tf_dir}"
        )
        return

def load_model_from_tfv2(
    saved_model_dir: str,
    endpoint: str = "serve",
    custom_objects: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Minimal TF SavedModel loader that supports dict inputs/outputs.

    - Keras 2 / tf.keras: keras.models.load_model(saved_model_dir)
    - Keras 3: wrap SavedModel as inference-only keras.Model via TFSMLayer,
      building dict Inputs from the SavedModel signature (supports positional
      dict arg 'args_0' like your export).
    """
    keras = _import_keras()

    if not os.path.isdir(saved_model_dir):
        raise ValueError(
            f"SavedModel directory not found: {saved_model_dir!r}"
        )

    # --- Keras 2 / tf.keras can load SavedModel directly
    if not is_keras3():
        with _custom_object_scope(keras, custom_objects):
            return keras.models.load_model(saved_model_dir, compile=False)

    # --- Keras 3: TFSMLayer + build dict Inputs from signature
    import tensorflow as tf

    obj = tf.saved_model.load(saved_model_dir)
    sigs = getattr(obj, "signatures", {}) or {}

    # pick signature: requested endpoint -> fallback to serving_default -> any
    fn = sigs.get(endpoint) or sigs.get("serving_default")
    used_ep = endpoint if sigs.get(endpoint) is not None else "serving_default"
    if fn is None and sigs:
        used_ep, fn = next(iter(sigs.items()))

    if fn is None:
        raise ValueError(
            f"No callable signatures found in SavedModel: {saved_model_dir!r}"
        )

    # Your export uses a single positional dict argument (args_0) -> parse args[0]
    args, kwargs = fn.structured_input_signature

    specs = None
    if isinstance(kwargs, dict) and kwargs:
        # keyword-input SavedModel
        specs = kwargs
    elif isinstance(args, (tuple, list)) and args:
        # positional-input SavedModel: args[0] may be dict of TensorSpecs
        if isinstance(args[0], dict) and args[0]:
            specs = args[0]

    if not isinstance(specs, dict) or not specs:
        raise ValueError(
            "Cannot infer dict input specs from SavedModel signature. "
            f"Available signatures: {list(sigs.keys())}"
        )

    # Build a TFSMLayer for the chosen endpoint
    layer = keras.layers.TFSMLayer(
        saved_model_dir,
        call_endpoint=used_ep,
    )

    # Create one keras.Input per dict key
    inputs = {}
    for name, spec in specs.items():
        if not hasattr(spec, "shape"):
            raise ValueError(
                f"Invalid TensorSpec for key {name!r}: {spec!r}"
            )
        shape = tuple(spec.shape)[1:]  # drop batch dim
        inputs[name] = keras.Input(
            shape=shape,
            dtype=spec.dtype,
            name=name,
        )

    outputs = layer(inputs)  # dict -> dict
    return keras.Model(inputs=inputs, outputs=outputs, name="tfsm_inference")

def load_model_from_tf(
    saved_model_path: str,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Load a TF SavedModel directory for inference.

    - Keras 2: keras.models.load_model(dir)
    - Keras 3: TFSMLayer(dir) wrapped as keras.Model
    """
    keras = _import_keras()

    if not is_keras3():
        with _custom_object_scope(keras, custom_objects):
            return keras.models.load_model(
                saved_model_path,
                compile=False,
            )

    # Keras 3: inference-only wrapper.
    endpoints = ("serve", "serving_default")
    layer = None
    last_err = None
    used_ep = None

    for ep in endpoints:
        try:
            layer = keras.layers.TFSMLayer(
                saved_model_path,
                call_endpoint=ep,
            )
            used_ep = ep
            last_err = None
            break
        except Exception as e:
            last_err = e

    if layer is None:
        raise ValueError(
            "Cannot create TFSMLayer from SavedModel: "
            f"{last_err!r}"
        )

    # Try to infer inputs from TF signature (best).
    specs = None
    try:
        import tensorflow as tf

        obj = tf.saved_model.load(saved_model_path)
        sigs = getattr(obj, "signatures", {}) or {}
        fn = sigs.get(used_ep) or sigs.get("serving_default")

        if fn is not None:
            _, kw = fn.structured_input_signature
            if isinstance(kw, dict) and kw:
                specs = kw
    except Exception:
        specs = None

    if specs:
        inputs = {}
        for name, spec in specs.items():
            shp = tuple(spec.shape)[1:]
            inputs[name] = keras.Input(
                shape=shp,
                dtype=spec.dtype,
                name=name,
            )

        outputs = layer(inputs)
        return keras.Model(
            inputs,
            outputs,
            name="tfsm_inference",
        )

    # Fallback: single-input case only.
    inp_shape = getattr(layer, "input_shape", None)
    if not inp_shape or len(inp_shape) < 2:
        raise ValueError(
            "Cannot infer inputs for TFSMLayer wrapper."
        )

    inp = keras.Input(shape=tuple(inp_shape[1:]))
    out = layer(inp)
    return keras.Model(inp, out, name="tfsm_inference")

def load_inference_model(
    keras_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    custom_objects: Optional[Dict[str, Any]] = None,
    compile: bool = False,
    builder: Optional[Callable[[Dict[str, Any]], Any]] = None,
    build_inputs: Optional[Any] = None,
    prefer_full_model: bool = True,
    log_fn=print,
    use_in_memory_model: bool = False,
):
    """Load the model for inference based on the configuration."""

    if use_in_memory_model:
        print("[INFO] Using in-memory model.")
        # Use the in-memory model (no need to load from disk)
        model = builder(load_manifest(manifest_path))
        model.build(build_inputs)  # Ensure the model is built for inference
    else:
        # Default behavior: Load the model from disk
        model = load_bundle_for_inference(
            keras_path=keras_path,
            weights_path=weights_path,
            manifest_path=manifest_path,
            custom_objects=custom_objects,
            compile=compile,
            builder=builder,
            build_inputs=build_inputs,
            prefer_full_model=prefer_full_model,
            log_fn=log_fn,
        )
    return model

def load_bundle_for_inference(
    *,
    keras_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    custom_objects: CustomObjects = None,
    compile: bool = False,
    builder: Builder = None,
    build_inputs: Optional[Any] = None,
    prefer_full_model: bool = True,
    log_fn: LogFn = None,
) -> Any:
    """
    Load an inference model with compatibility fallbacks.

    Strategy:
    1) If prefer_full_model and keras_path is given:
       - try load_model() under custom_object_scope.
    2) If it fails (or is disabled):
       - rebuild via builder(manifest) then load_weights().
    3) (Keras 3 only) if keras_path is a TF SavedModel dir:
       - wrap it with keras.layers.TFSMLayer.

    Notes:
    - builder must return an *unbuilt* model instance.
    - if build_inputs is given, we build subclassed models
      by running a forward pass before load_weights().
    """
    keras = _import_keras()
    log = log_fn or _log_default

    # ------------------------------------------------------------
    # (1) Full-model load (.keras / .h5, and SavedModel in tf.keras)
    # ------------------------------------------------------------
    if prefer_full_model and keras_path:
        try:
            with _custom_object_scope(keras, custom_objects):
                return keras.models.load_model(
                    keras_path,
                    compile=compile,
                )
        except Exception as e:
            log(
                "[compat.keras] load_model failed: "
                f"{e!r}"
            )

        # --------------------------------------------------------
        # (3) Keras 3: TF SavedModel dir -> TFSMLayer wrapper
        # --------------------------------------------------------
        if is_keras3() and os.path.isdir(keras_path):
            try:
                return load_model_from_tf(
                    keras_path,
                    custom_objects=custom_objects,
                )
                # layer = keras.layers.TFSMLayer(
                #     keras_path,
                #     call_endpoint="serving_default",
                # )

                # x = _extract_x(build_inputs) if build_inputs else None
                # inp_shape = getattr(layer, "input_shape", None)

                # if isinstance(inp_shape, (list, tuple)):
                #     if inp_shape and isinstance(
                #         inp_shape[0],
                #         (list, tuple),
                #     ):
                #         # Multi-input SavedModel
                #         raise ValueError(
                #             "TFSMLayer multi-input is "
                #             "not supported here."
                #         )

                # if isinstance(inp_shape, (list, tuple)) and len(
                #     inp_shape
                # ) >= 2:
                #     shape = tuple(inp_shape[1:])
                # elif x is not None and hasattr(x, "shape"):
                #     shape = tuple(x.shape[1:])
                # else:
                #     raise ValueError(
                #         "Cannot infer input shape for "
                #         "TFSMLayer wrapper."
                #     )

                # inp = keras.Input(shape=shape)
                # out = layer(inp)
                # return keras.Model(
                #     inp,
                #     out,
                #     name="tfsm_inference",
                # )
            except Exception as e:
                log(
                    "[compat.keras] TFSMLayer failed: "
                    f"{e!r}"
                )

    # ------------------------------------------------------------
    # (2) Rebuild + weights (most robust for subclassed models)
    # ------------------------------------------------------------
    if builder is None:
        raise ValueError(
            "builder is required when full-model load "
            "fails or is disabled."
        )

    manifest = load_manifest(manifest_path) if manifest_path else {}
    model = builder(manifest)

    # Build variables before load_weights (critical for subclassed)
    if build_inputs is not None:
        x = _extract_x(build_inputs)
        _ = model(x, training=False)

    if not weights_path:
        raise ValueError(
            "weights_path is required for weights "
            "fallback."
        )

    status = model.load_weights(weights_path)

    # TF may return a status object with these methods
    if hasattr(status, "expect_partial"):
        status.expect_partial()
    elif hasattr(status, "assert_consumed"):
        status.assert_consumed()

    return model
