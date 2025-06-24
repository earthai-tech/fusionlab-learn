import os
import json
from typing import Callable, Optional, Any, Dict, List 
from pathlib import Path
import math 

from fusionlab.params      import (
    LearnableK, LearnableSs, LearnableQ,
    LearnableC, FixedC, DisabledC,
)
from fusionlab.nn.models import TransFlowSubsNet, PIHALNet
from fusionlab.nn import KERAS_DEPS 

_LEARNABLE_TYPES = (LearnableK, LearnableSs, LearnableQ)

Callback =KERAS_DEPS.Callback 
load_model = KERAS_DEPS.load_model
Model      = KERAS_DEPS.Model            # type alias
custom_object_scope = KERAS_DEPS.custom_object_scope
deserialize_keras_object = KERAS_DEPS.deserialize_keras_object
serialize_keras_object =KERAS_DEPS.serialize_keras_object


_CUSTOM_OBJECTS = {
    "LearnableK": LearnableK, "LearnableSs": LearnableSs,
    "LearnableQ": LearnableQ, "LearnableC":  LearnableC,
    "FixedC":     FixedC,     "DisabledC":   DisabledC,
}

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

def _rebuild_from_arch_cfg(arch_cfg: dict):
    """Turn the manifest’s JSON back into a live Keras model."""
    # 1. de-serialise nested Learnable… objects
    for key in ("K", "Ss", "Q", "pinn_coefficient_C"):
        if isinstance(arch_cfg.get(key), dict) and "class_name" in arch_cfg[key]:
            arch_cfg[key] = deserialize_keras_object(
                arch_cfg[key],  custom_objects=_CUSTOM_OBJECTS
            )

    # 2. decide which concrete model class to instantiate
    cls_name = arch_cfg.get("name", "TransFlowSubsNet")
    ModelCls = TransFlowSubsNet if cls_name == "TransFlowSubsNet" else PIHALNet

    # 3. build **un-compiled** model
    return ModelCls.from_config(arch_cfg)

# def _rebuild_from_arch_cfg(arch_cfg: dict):
#     """Return a *compiled* model recreated from the JSON architecture dict."""
#     cls_name = arch_cfg.get("name")               # 'TransFlowSubsNet' / 'PIHALNet'
#     model_cls = {"TransFlowSubsNet": TransFlowSubsNet,
#                  "PIHALNet":        PIHALNet}.get(cls_name)
#     if model_cls is None:
#         raise ValueError(f"Unknown model class '{cls_name}' in manifest")

#     # re-hydrate possible Learnable-objects stored as {"class_name": …}
#     for key in ("K", "Ss", "Q", "pinn_coefficient_C"):
#         obj = arch_cfg.get(key)
#         if isinstance(obj, dict) and ( "class_name" in obj or "__class_name__" in obj):
#             arch_cfg[key] = deserialize_keras_object(obj, _CUSTOM_OBJECTS)

#     return model_cls.from_config(arch_cfg)

def safe_model_loader(
    model_path: str | os.PathLike,
    *,
    build_fn: Optional[Callable[[], Model]] = None,
    custom_objects: Optional[dict] = None,
    log: Callable[[str], None] = print,
    arch_cfg=None, 
) -> Model:
    """
    Load a Keras / TF model saved as **.keras**, **SavedModel dir**,
    legacy **.h5**, or **weights-only .weights.h5**.

    If the file is weights-only you *must* pass `build_fn` that returns
    a fresh, **un-compiled** model with the *same* architecture.
    """
    path = Path(model_path)
    if not path.exists():
        raise IOError(f"[safe_model_loader] Path does not exist: {path}")

    ext = path.suffix.lower()
    # try:
    # ---- full-graph formats --
    # if ext in {".keras", ".h5"} and not path.name.endswith(".weights.h5"):
    #     log(f"[loader] reading full model from {path.name}")
    #     try: 
    #         model = load_model(path, custom_objects=custom_objects)
    #     except: 
    #         with custom_object_scope(custom_objects):
    #             model = load_model(path)
                
    #     log("  Model loaded successfully.")
        
    #     return model
    
    # ---- SavedModel directory ------------------------------------
    # if path.is_dir():
    #     log(f"[loader] reading TensorFlow SavedModel at {path}")
    #     model =  load_model(path, custom_objects=custom_objects)
    #     log("  Best model loaded successfully.")
    #     return model 

    # ---- weights-only case ---------------------------------------
    if path.name.endswith(".weights.h5"):
        if build_fn is None:
            raise ValueError("weights file requires a `build_fn`.")
        log(f"[loader] rebuilding architecture then "
            f"loading weights from {path.name}")
        
        model = build_fn()          # un-compiled model
        
        # ----------  NEW  ---------------------------------------
        # force variable creation so the weight names exist
        if not model.built:
            #  a) if the model implements `.build(input_shape)`:
            try:
                # retrieve the input shapes stored in the manifest
                ish = arch_cfg.get("input_shapes")  # save this during training!
                if ish is not None:
                    model.build(ish)
            except Exception:
                pass
    
            # #  b) fall back to a dummy forward-pass
            # if not model.built:
            #     import tensorflow as tf
            #     dummy = {
            #         "coords":           tf.zeros(
            #             (1, arch_cfg["max_window_size"], 3)),
            #         "dynamic_features": tf.zeros(
            #             (1, arch_cfg["max_window_size"], arch_cfg["dynamic_input_dim"])),
            #         "static_features":  tf.zeros(
            #             (1, arch_cfg["static_input_dim"])),
            #         "future_features":  tf.zeros(
            #             (1, arch_cfg["forecast_horizon"], arch_cfg["future_input_dim"])),
            #     }
            #     _ = model(dummy, training=False)
        # --------------------------------------------------------
    
        # model.load_weights(path)                # (2) now succeeds
        # return model

        
        model.load_weights(path)
        return model

    #     raise ValueError(f"unknown model format: {path}")

    # except Exception as err:
    #     raise IOError(f"[safe_model_loader] failed to load: {err}") from err


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
