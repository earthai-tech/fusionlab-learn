# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides high-level utility functions for the GUI application,
such as a robust model loader that can handle multiple Keras/TF
saving formats.
"""
from __future__ import annotations 

import os
import json
from typing import Callable, Optional, Any, Dict, List 
from pathlib import Path
import math 


try:
    from fusionlab.params import (
        LearnableK, LearnableSs, LearnableQ,
        LearnableC, FixedC, DisabledC,
    )
    from fusionlab.utils._manifest_registry import ManifestRegistry
    from fusionlab.nn import KERAS_DEPS
    from fusionlab.nn.models import TransFlowSubsNet, PIHALNet
except ImportError as e:
    raise ImportError(
        "This utility requires the `fusionlab` library and its"
        f" dependencies to be installed. Error: {e}"
    )

# Define custom objects needed for model deserialization
_CUSTOM_OBJECTS = {
    "LearnableK": LearnableK, "LearnableSs": LearnableSs,
    "LearnableQ": LearnableQ, "LearnableC":  LearnableC,
    "FixedC":     FixedC,     "DisabledC":   DisabledC,
}

_LEARNABLE_TYPES = (LearnableK, LearnableSs, LearnableQ)

Callback =KERAS_DEPS.Callback 
load_model = KERAS_DEPS.load_model
Model = KERAS_DEPS.Model            # type alias
custom_object_scope = KERAS_DEPS.custom_object_scope
deserialize_keras_object = KERAS_DEPS.deserialize_keras_object
serialize_keras_object =KERAS_DEPS.serialize_keras_object


class GuiProgress(Callback):
    """
    Emit percentage updates while `model.fit` runs.

    Parameters
    ----------
    total_epochs : int
        Epochs you pass to `model.fit`.
    batches_per_epoch : int
        Length of the training dataset (`len(ds)`).
        Needed only for *batch-level* granularity.
    update_fn : Callable[[int], None]
        Function that receives an **int 0-100**.
        Examples: `my_qprogressbar.setValue`, `signal.emit`.
    epoch_level : bool, default=True
        If True, update once per epoch; otherwise per batch.
    """
    def __init__(
        self,
        total_epochs: int,
        batches_per_epoch: int,
        update_fn: Callable[[int], None],
        *,
        epoch_level: bool = True,
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.update_fn = update_fn
        self.epoch_level = epoch_level
        self._seen_batches = 0

    # epoch-level -
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_level:
            pct = int((epoch + 1) / self.total_epochs * 100)
            self.update_fn(pct)

    # batch-level 
    def on_train_batch_end(self, batch, logs=None):
        if not self.epoch_level:
            self._seen_batches += 1
            total_batches = self.total_epochs * self.batches_per_epoch
            pct = math.floor(self._seen_batches / total_batches * 100)
            self.update_fn(pct)

def locate_and_load_manifest(
    manifest_path: Optional[str | os.PathLike] = None,
    validation_data_path: Optional[str | os.PathLike] = None,
    log: Callable[[str], None] = print
) -> Dict[str, Any]:
    """
    Locates and loads a run_manifest.json file.

    This function provides a robust way to find the correct manifest
    file for an inference run. It can be given a direct path to the
    manifest or a path to a new data file, from which it will
    heuristically search for the most recent, relevant manifest.

    Parameters
    ----------
    manifest_path : str or pathlib.Path, optional
        A direct path to the `run_manifest.json` file. If provided,
        this path is used directly.
    validation_data_path : str or pathlib.Path, optional
        A path to the new data file for prediction. This is used as
        the starting point for a heuristic search if `manifest_path`
        is not provided.
    log : callable, default=print
        A logging function to output status messages during the search.

    Returns
    -------
    dict
        The parsed content of the found JSON manifest file.

    Raises
    ------
    ValueError
        If neither `manifest_path` nor `validation_data_path` is provided.
    FileNotFoundError
        If no manifest file can be found from the given paths.
    """
    if not manifest_path and not validation_data_path:
        raise ValueError(
            "Must provide either a `manifest_path` or a "
            "`validation_data_path` to find the run manifest."
        )

    if manifest_path:
        found_manifest_path = Path(manifest_path)
    else:
        # If no direct path, search heuristically from the data path
        found_manifest_path = _locate_manifest(
            Path(validation_data_path), log=log
        )

    if not found_manifest_path or not found_manifest_path.exists():
        raise FileNotFoundError(
            f"Could not find a valid `run_manifest.json` at or near the "
            f"provided path: {manifest_path or validation_data_path}"
        )

    log(f"Loading configuration from manifest: {found_manifest_path}")
    return json.loads(found_manifest_path.read_text("utf-8"))

def _find_manifest_in(dir_: Path) -> List[Path]:
    """Return every run_manifest.json inside *dir_/**_run/ sub-folders."""
    return list(dir_.glob("*_run/run_manifest.json"))

def _locate_manifest(csv_path: Path, max_up: int = 3) -> Optional[Path]:
    """
    Heuristic search for the `run_manifest.json` that matches *csv_path*.

    *   check the CSV folder itself;
    *   check *_run/ sub-folders right under it;
    *   walk **up to `max_up` parent levels**, at each step:
          – look for a `run_manifest.json` next to the folder,
          – look in any *_run/ sub-folder,
          – look inside a sibling *results_pinn/* directory (common default).
    Return the most **recent** hit (mtime) or *None*.
    """
    here = csv_path.parent

    for lvl in range(max_up + 1):
        probe = here if lvl == 0 else csv_path.parents[lvl]

        # 1) same folder
        direct = probe / "run_manifest.json"
        if direct.exists():
            return direct

        # 2) any *_run/ below this folder
        hits = _find_manifest_in(probe)
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)

        # 3) common results dir (e.g.  <probe>/results_pinn/*_run/)
        res_dir = probe / "results_pinn"
        if res_dir.is_dir():
            hits = _find_manifest_in(res_dir)
            if hits:
                return max(hits, key=lambda p: p.stat().st_mtime)

    # nothing found
    return None


def _rebuild_from_arch_cfg(arch_cfg: dict) -> Model:
    """
    Rebuilds an un-compiled model instance from an architecture config dict.
    
    This helper de-serializes any custom `Learnable` parameter objects
    and instantiates the correct model class (`TransFlowSubsNet` or
    `PIHALNet`) from the configuration.
    """
    # 1. De-serialize nested Learnable parameter objects
    for key in ("K", "Ss", "Q", "pinn_coefficient_C"):
        param_config = arch_cfg.get(key)
        if isinstance(param_config, dict) and "class_name" in param_config:
            arch_cfg[key] = deserialize_keras_object(
                param_config, custom_objects=_CUSTOM_OBJECTS
            )

    # 2. Decide which concrete model class to instantiate
    cls_name = arch_cfg.get("name", "TransFlowSubsNet")
    ModelCls = {
        "TransFlowSubsNet": TransFlowSubsNet,
        "PIHALNet": PIHALNet
    }.get(cls_name)
    
    if ModelCls is None:
        raise ValueError(f"Unknown model class '{cls_name}' in manifest config.")

    # 3. Build and return the un-compiled model from its config
    return ModelCls.from_config(arch_cfg)

def safe_model_loader(
    model_path: str | os.PathLike,
    *,
    build_fn: Optional[Callable[[], Model]] = None,
    custom_objects: Optional[dict] = None,
    log: Callable[[str], None] = print,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Model:
    """Loads a Keras/TF model saved in various formats.

    This function provides a single, robust entry point for loading models
    saved as:
    - The modern Keras v3 format (``.keras``)
    - The legacy HDF5 format (``.h5``)
    - The TensorFlow SavedModel directory format
    - Weights-only files (e.g., ``.weights.h5``)

    For weights-only files, this function requires a build function or an
    architecture configuration from a manifest to reconstruct the model
    before loading the weights.

    Parameters
    ----------
    model_path : str or pathlib.Path
        The path to the model file or directory.
    build_fn : callable, optional
        A callable function that returns a fresh, un-compiled model
        instance. Required if `model_path` points to a weights-only
        file and `arch_cfg` is not provided.
    custom_objects : dict, optional
        A dictionary mapping names to custom classes or functions required
        for loading the model. Merged with default PINN custom objects.
    log : callable, default=print
        A logging function to output status messages.
    model_cfg : dict, optional
        The trained model configuration. It encompasses the architecture 
         configuration dictionary, typically from a manifest
        file. Used with `build_fn` to reconstruct a model for
        weights-only loading.

    Returns
    -------
    keras.Model
        The loaded Keras model.

    Raises
    ------
    IOError
        If the model path does not exist or if loading fails for other reasons.
    ValueError
        If a weights-only file is provided without a way to reconstruct the
        model (i.e., missing `build_fn` or `arch_cfg`).
    """
    path = Path(model_path)
    if not path.exists():
        raise IOError(f"[safe_model_loader] Path does not exist: {path}")

    # Combine user custom objects with the default ones for PINN params
    final_custom_objects = _CUSTOM_OBJECTS.copy()
    if custom_objects:
        final_custom_objects.update(custom_objects)
    
    arch_cfg = model_cfg.get("config")
    try:
        # --- Case 1: Full Model (directory or .keras/.h5 file) ---
        if path.is_dir() or (path.suffix in {".keras", ".h5"} and not
                             path.name.endswith(".weights.h5")):
            log(f"[loader] Reading full model from: {path.name}")
            with custom_object_scope(final_custom_objects):
                model = load_model(path)
            log("  Model loaded successfully.")
            return model

        # --- Case 2: Weights-only file ---
        elif path.name.endswith((".weights.h5", ".weights.keras")):
            log(f"[loader] Loading weights from: {path.name}")
            
            # Rebuild the model architecture first
            if build_fn:
                model = build_fn()
                log("  Rebuilding model from provided build_fn...")
            elif arch_cfg:
                model = _rebuild_from_arch_cfg(arch_cfg)
                log("  Rebuilding model from manifest architecture config...")
            else:
                raise ValueError(
                    "A `build_fn` or `arch_cfg` must be provided to load a "
                    "weights-only file."
                )

            # Build the model to create its weights before loading
            if not model.built and arch_cfg and "input_shapes" in model_cfg:
                log("  Building model with input shapes from manifest...")
                try:
                    # The `input_shapes` from the manifest will be a dict
                    model.build(model_cfg["input_shapes"])
                except Exception as e:
                    log(f"  [Warning] Failed to build model with manifest"
                        f" shapes: {e}. Model will build on first call.")

            # Load the weights into the reconstructed architecture
            model.load_weights(str(path))
            log("  Weights loaded successfully into reconstructed model.")
            return model

        else:
            raise ValueError(f"Unknown model format for path: {path}")

    except Exception as err:
        raise IOError(
            f"[safe_model_loader] Failed to load model from {path}. "
            f"Ensure custom objects are registered or provided. Error: {err}"
        ) from err

def json_ready(obj: Any, *, mode: str = "literal") -> Any:
    """
    Recursively walk *obj* and return a clone that **json.dumps**
    can handle.

    Parameters
    ----------
    obj   : any Python object (dict / list / scalar / …)
    mode  : "literal" | "config"
        * literal – replace Learnable… objects with the same value you
          originally passed to the model (i.e. ``'learnable'`` /
          ``'fixed'`` **or** the float) so the JSON is compact.
        * config  – call ``.get_config()`` on each Learnable… object and
          store that dict instead (useful if you want to rebuild them
          later with ``.from_config``).

    Returns
    -------
    jsonable_obj : an isomorphic structure containing only JSON-safe
                   data types (dict / list / str / int / float / bool /
                   None).

    Notes
    -----
    •  If *obj* already is JSON-safe it is returned unchanged.  
    •  Unknown custom classes will raise ``TypeError`` – add your own
       handler if you need more.
    """
    # 1) primitives — nothing to do
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # 2) Learnable…  --------------------------------------------
    if isinstance(obj, _LEARNABLE_TYPES):
        if mode == "literal":
            # Re-use the original “initial value”
            return obj.initial_value
        elif mode == "config":
            return serialize_keras_object(obj)# obj.get_config()
        else:
            raise ValueError("mode must be 'literal' or 'config'")

    # 3) collections — walk recursively
    if isinstance(obj, dict):
        return {k: json_ready(v, mode=mode) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v, mode=mode) for v in obj]

    # 4) anything else → unsupported
    raise TypeError(f"Object of type {type(obj).__name__} "
                    "is not JSON serialisable")
