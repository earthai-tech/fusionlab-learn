# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
fusionlab.compat.keras_fit

Keras 2/3 helpers for:
- filling missing targets for multi-output models
- updating compiled metrics safely
- reading compiled metrics results safely
- optional warning suppression for noisy Keras deprecations
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Mapping
import itertools
from typing import ( 
    Any, 
    Callable, 
    Dict, 
    Optional, 
    Sequence
)
import numpy as np

from ._config import import_keras_dependencies


# Custom message for missing dependencies
EXTRA_MSG = ( 
    "`keras-fit` module expects the `tensorflow` or"
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

def _slice_first_batch(x: Any, n: int) -> Any:

    if x is None:
        return None

    if isinstance(x, Mapping):
        return {k: _slice_first_batch(v, n) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return type(x)(_slice_first_batch(v, n) for v in x)

    try:
        a = np.asarray(x)
    except Exception:
        return x

    if a.ndim == 0:
        return x

    try:
        return x[:n]
    except Exception:
        return x


def _match_score(a: Any, b: Any) -> float:


    aa = np.asarray(a)
    bb = np.asarray(b)

    if aa.shape != bb.shape:
        return float("inf")

    af = aa.reshape(-1)
    bf = bb.reshape(-1)

    m = int(min(256, af.size))
    if m <= 0:
        return 0.0

    return float(np.mean(np.abs(af[:m] - bf[:m])))


def normalize_predict_output(
    model: Any,
    *,
    x: Any,
    pred_out: Any,
    required: Sequence[str] = ("subs_pred", "gwl_pred"),
    batch_n: int = 8,
    log_fn: LogFn = None,
) -> Dict[str, Any]:
    """
    Normalize model.predict() output to a dict with required keys.

    Why:
    - Keras predict() may return a list/tuple even if call()
      returns a dict.
    - List order can be ambiguous across TF/Keras/compat paths.
    - This function uses a one-batch call() to robustly map
      list outputs to dict keys.

    Returns:
    - dict containing at least the `required` keys.
    """

    log = log_fn or (lambda *_: None)

    req = list(required)

    if isinstance(pred_out, Mapping):
        d = dict(pred_out)
        missing = [k for k in req if k not in d]
        if not missing:
            return d

        raise KeyError(
            "predict() dict missing keys: "
            f"{missing}. got={list(d.keys())}"
        )

    if not isinstance(pred_out, (list, tuple)):
        raise TypeError(
            "Unexpected predict() output type: "
            f"{type(pred_out)}"
        )

    pred_list = list(pred_out)
    if len(pred_list) < len(req):
        raise ValueError(
            "predict() returned too few outputs: "
            f"{len(pred_list)} < {len(req)}"
        )

    # Candidate mapping via output_names (best-effort)
    names = list(getattr(model, "output_names", []) or [])
    cand = None
    if names:
        tmp = {}
        for i in range(min(len(names), len(pred_list))):
            tmp[names[i]] = pred_list[i]
        if all(k in tmp for k in req):
            cand = tmp

    # Strong mapping via one-batch call()
    xb = _slice_first_batch(x, int(batch_n))
    try:
        out_call = model(xb, training=False)
    except Exception:
        out_call = None

    if isinstance(out_call, Mapping) and all(k in out_call for k in req):
        ref = {k: out_call[k] for k in req}

        idxs = list(range(len(pred_list)))
        k = len(req)

        best = None
        best_perm = None

        for perm in itertools.permutations(idxs, k):
            tot = 0.0
            ok = True
            for j, pi in enumerate(perm):
                s = _match_score(pred_list[pi], ref[req[j]])
                if not (s < float("inf")):
                    ok = False
                    break
                tot += s

            if not ok:
                continue

            if best is None or tot < best:
                best = tot
                best_perm = perm

        if best_perm is None:
            if cand is not None:
                log(
                    "normalize_predict_output: "
                    "call()-mapping failed; "
                    "using output_names mapping."
                )
                return cand
            raise RuntimeError(
                "Could not map predict() list outputs "
                "to required keys using call()."
            )

        out = {}
        for j, pi in enumerate(best_perm):
            out[req[j]] = pred_list[pi]

        if cand is not None:
            for k2, v2 in cand.items():
                if k2 not in out:
                    out[k2] = v2

        log(
            "normalize_predict_output: "
            f"mapped via call(): {best_perm} "
            f"score={best:.6g}"
        )
        return out

    # Fallbacks
    if cand is not None:
        log(
            "normalize_predict_output: "
            "call() not usable; using output_names."
        )
        return cand

    # Last-resort positional mapping (avoid if possible)
    log(
        "normalize_predict_output: "
        "no call()/output_names; "
        "using positional fallback."
    )
    out = {req[i]: pred_list[i] for i in range(len(req))}
    return out

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
