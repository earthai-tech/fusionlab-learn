# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
fusionlab.compat.fit_compat

Keras 2/3 helpers for:
- filling missing targets for multi-output models
- updating compiled metrics safely
- reading compiled metrics results safely
- optional warning suppression for noisy Keras deprecations
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

# from .keras import _import_keras  # internal reuse is OK

#  unified deps (tf.keras / keras ops)
from ._config import import_keras_dependencies


# Custom message for missing dependencies
EXTRA_MSG = ( 
    "`fit-compat` module expects the `tensorflow` or"
    " `keras` library to be installed."
    )
# Configure and install dependencies if needed

# Lazy-load Keras dependencies
KERAS_DEPS = import_keras_dependencies(
    extra_msg=EXTRA_MSG, error='ignore')

Tensor = KERAS_DEPS.Tensor
tf_float32 = KERAS_DEPS.float32
tf_convert = KERAS_DEPS.convert_to_tensor
tf_stop_grad = KERAS_DEPS.stop_gradient
tf_convert_to_tensor = KERAS_DEPS.convert_to_tensor
tf_squeeze = KERAS_DEPS.squeeze
tf_cast = KERAS_DEPS.cast 
tf_expand_dims = KERAS_DEPS.expand_dims 

LogFn = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------
# Warnings (optional)
# ---------------------------------------------------------------------
@contextlib.contextmanager
def suppress_compiled_metrics_warning():
    """
    Silence the Keras warning about `model.compiled_metrics()`.

    This warning may be emitted by Keras internals in some
    TF/Keras combos. It is safe to ignore.
    """
    msg = "`model.compiled_metrics()` is deprecated"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=msg,
            category=UserWarning,
        )
        yield


def _as_BHO(y_true: Tensor, y_pred: Tensor | None = None):
    """Normalize y_true to (B,H,O) float32."""
    y = tf_convert_to_tensor(y_true)

    # (B,H) -> (B,H,1)
    if y.shape.rank == 2:
        y = tf_expand_dims(y, axis=-1)

    # (B,H,O,1) -> (B,H,O)
    if (y.shape.rank == 4) and (y.shape[-1] == 1):
        y = tf_squeeze(y, axis=-1)

    # If targets got tiled to (B,H,Q,O), drop Q.
    if (y.shape.rank == 4) and (y_pred is not None):
        yp_shape = getattr(y_pred, "shape", None)
        yp_rank = getattr(yp_shape, "rank", None)

        # Dev note: in this codebase rank-4 preds mean (B,H,Q,O)
        # so any rank-4 y_true is suspicious and should be untiled.
        if yp_rank == 4:
            y = y[:, :, 0, :]

    return tf_cast(y, tf_float32)
# ---------------------------------------------------------------------
# Targets (missing-output placeholder)
# ---------------------------------------------------------------------
def ensure_targets_for_outputs(
    *,
    output_names: Sequence[str],
    targets: Any,
    y_pred: Any,
    log_fn: LogFn = None,
) -> Any:
    """
    Ensure `targets` contains keys for all outputs.

    If targets is a dict and a key is missing/None, we inject a
    loss-only placeholder:
        y_true := stop_gradient(y_pred)

    This keeps compiled multi-output loss dicts happy while
    producing zero loss and no gradients for that head.
    """
    log = log_fn or (lambda *_: None)

    if not isinstance(targets, dict):
        return targets
    if not isinstance(y_pred, dict):
        return targets

    out = dict(targets)
    missing = []

    for k in output_names:
        v = out.get(k, None)
        if v is None and k in y_pred:
            out[k] = tf_stop_grad(y_pred[k])
            missing.append(k)

    if missing:
        s = ", ".join(missing)
        log(
            "Missing targets for outputs: "
            f"{s}. Using stop_gradient(y_pred)."
        )

    return out


def supervised_output_keys(
    *,
    output_names: Sequence[str],
    targets: Any,
    y_pred: Any,
) -> list[str]:
    """
    Return output keys that are truly supervised.

    Used to avoid updating metrics on placeholder targets.
    """
    if not isinstance(targets, dict):
        return list(output_names)
    if not isinstance(y_pred, dict):
        return list(output_names)

    keys: list[str] = []
    for k in output_names:
        if k in y_pred and targets.get(k, None) is not None:
            keys.append(k)
    return keys


# ---------------------------------------------------------------------
# Compiled metrics container access (Keras 2/3)
# ---------------------------------------------------------------------
def get_compile_metrics(model: Any):
    """
    Return the underlying compiled-metrics container.

    Keras 3: model._compile_metrics
    Keras 2 variants: _compiled_metrics / _metrics_container
    Fallback: model.compiled_metrics
    """
    cm = getattr(model, "_compile_metrics", None)
    if cm is not None:
        return cm

    for name in ("_compiled_metrics", "_metrics_container"):
        cm = getattr(model, name, None)
        if cm is not None:
            return cm

    return getattr(model, "compiled_metrics", None)


def _as_list_by_outputs(obj: Any, keys: Sequence[str]):
    if isinstance(obj, Mapping):
        return [obj[k] for k in keys]
    return obj


def update_compiled_metrics(
    model: Any,
    *,
    targets: Any,
    y_pred: Any,
    keys: Optional[Sequence[str]] = None,
) -> None:
    """
    Update compiled metrics in a version-safe way.

    - Updates only supervised outputs by default.
    - Tries list path first, then dict path, then manual fallback.
    """
    cm = get_compile_metrics(model)
    if cm is None:
        return

    out_names = (
        list(getattr(model, "output_names", None) or [])
    )
    if not out_names:
        return

    if keys is None:
        keys = supervised_output_keys(
            output_names=out_names,
            targets=targets,
            y_pred=y_pred,
        )

    if not keys:
        return

    t = {k: _as_BHO(targets[k], y_pred=y_pred[k])for k in keys} if isinstance(
        targets, dict
    ) else targets
    p = {k: y_pred[k] for k in keys} if isinstance(
        y_pred, dict
    ) else y_pred

    yt_list = _as_list_by_outputs(t, keys)
    yp_list = _as_list_by_outputs(p, keys)

    try:
        cm.update_state(yt_list, yp_list)
        return
    except Exception:
        pass

    try:
        cm.update_state(t, p)
        return
    except Exception:
        pass

    # Last fallback: per-metric update by prefix
    for out in keys:
        yt = t[out]
        yp = p[out]
        pref = out + "_"
        for m in getattr(model, "metrics", []):
            nm = getattr(m, "name", "") or ""
            if nm.startswith(pref) and "loss" not in nm:
                try:
                    m.update_state(yt, yp)
                except Exception:
                    continue

def compiled_metrics_dict(
    model: Any,
    *,
    dtype: Any = tf_float32,
) -> Dict[str, Tensor]:
    """
    Read compiled metrics into a dict safely (Keras 2/3).
    """
    cm = get_compile_metrics(model)
    if cm is None:
        return {}

    try:
        d = cm.result()
    except Exception:
        return {}

    if not isinstance(d, dict):
        return {}

    out: Dict[str, Tensor] = {}
    for k, v in d.items():
        if not k:
            continue
        out[k] = tf_convert(v, dtype=dtype)
    return out
