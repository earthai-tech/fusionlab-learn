from __future__ import annotations 

from collections.abc import Mapping, Sequence
from .generic_utils import rename_dict_keys


_DEFAULT_EXCLUDE_KEYS = {
    "data_final", "data_mean_raw",
    "phys_final", "phys_mean_raw",
    "aux", "physics", "maps",
}

def _metric_key_from_name(name: str):
    # Keras names: "subs_pred_mae", "subs_pred_coverage80", "gwl_pred_mse", ...
    # Skip Keras loss trackers
    if name in ("loss",) or name.endswith("_loss"):
        return None

    if name.startswith("subs_pred_"):
        return "subs_pred"
    if name.startswith("gwl_pred_"):
        return "gwl_pred"
    return None

def _safe_to_numpy(x, *, mode="auto"):
    """
    Convert to numpy only when it is safe.

    mode:
      - "never": never convert
      - "always": try convert (will fail in graph mode)
      - "auto": convert only if .numpy() works (eager)
    """
    if mode == "never":
        return x
    if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
        if mode == "always":
            return x.numpy()
        # auto: try, but don't crash (graph/XLA will throw)
        try:
            return x.numpy()
        except Exception:
            return x
    return x


def get_output_names(
        model=None, y=None, y_pred=None, *, 
        exclude_keys=_DEFAULT_EXCLUDE_KEYS
    ):
    """
    Try to obtain stable output names (best-effort, Keras-3-safe).
    Priority:
      1) model._output_keys / model._output_names (your own, recommended)
      2) model.output_names (functional models typically)
      3) keys from y_pred dict
      4) keys from y dict
    """
    if model is not None:
        for attr in ("_output_keys", "_output_names", "output_names"):
            names = getattr(model, attr, None)
            if names:
                return [str(n) for n in list(names)]

    for obj in (y_pred, y):
        if isinstance(obj, Mapping):
            keys = [k for k in obj.keys() if k not in exclude_keys]
            if keys:
                return list(keys)

    return None


def as_tuple(
    obj,
    *,
    names=None,
    model=None,
    ctx="value",
    strict=True,
    exclude_keys=_DEFAULT_EXCLUDE_KEYS,
    to_numpy="never",  # "never" | "auto" | "always"
):
    """
    Convert obj to an ordered tuple of outputs.

    Supports:
      - dict: ordered by `names` (or inferred)
      - list/tuple: returns tuple(obj)
      - tensor/ndarray/scalar: returns (obj,)

    Parameters
    ----------
    obj : Any
        Targets/predictions container.
    names : list[str] or None
        Desired order of outputs.
    model : Any or None
        Model used to infer output names (via _output_keys/output_names).
    strict : bool
        If True, missing keys in dict raises KeyError.
    to_numpy : {"never","auto","always"}
        Optional conversion of individual leaves to numpy (only safe in eager).

    Returns
    -------
    tuple
        Ordered outputs.
    """
    if isinstance(obj, Mapping):
        if names is None:
            names = get_output_names(
                model=model, y=obj, y_pred=None, exclude_keys=exclude_keys
            )

        if names:
            out = []
            missing = []
            for n in names:
                if n in obj:
                    out.append(_safe_to_numpy(obj[n], mode=to_numpy))
                else:
                    missing.append(n)
            if missing and strict:
                raise KeyError(
                    f"{ctx}: missing keys {missing}. Available keys={list(obj.keys())}")
            return tuple(out)

        # No names: keep insertion order excluding known aux keys
        return tuple(_safe_to_numpy(v, mode=to_numpy)
                     for k, v in obj.items() if k not in exclude_keys)

    # Avoid treating strings as sequences
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return tuple(_safe_to_numpy(v, mode=to_numpy) for v in obj)

    return (_safe_to_numpy(obj, mode=to_numpy),)


def _prune_dict(d, names):
    if not isinstance(d, Mapping) or not names:
        return d
    return {k: d[k] for k in names if k in d}


def update_compiled_metrics(
    model,
    y_true,
    y_pred,
    *,
    output_names=None,
    to_numpy="never",   # keep "never" for real training
):
    """
    Keras-3-safe compiled metrics updater.

    - Prefers dict structure (since you compiled with dict loss/metrics).
    - Ensures deterministic output order via output_names/_output_keys.
    - Falls back to list/tuple update_state if needed.
    - Final fallback: manual per-metric update (won't crash training).

    Note: converting to numpy inside train_step is generally NOT safe.
    """
    cm = getattr(model, "compiled_metrics", None)
    if cm is None:
        return

    names = output_names or get_output_names(model=model, y=y_true, y_pred=y_pred)
    # Prefer dicts if available (matches your compile(loss=dict, metrics=dict))
    yt = _prune_dict(y_true, names) if isinstance(y_true, Mapping) else y_true
    yp = _prune_dict(y_pred, names) if isinstance(y_pred, Mapping) else y_pred

    # Optional numpy conversion (debug only; keep default "never")
    if to_numpy != "never":
        if isinstance(yt, Mapping):
            yt = {k: _safe_to_numpy(v, mode=to_numpy) for k, v in yt.items()}
        if isinstance(yp, Mapping):
            yp = {k: _safe_to_numpy(v, mode=to_numpy) for k, v in yp.items()}

    # Attempt 1: dict update (best match for your compile config)
    try:
        cm.update_state(yt, yp)
        return
    except Exception:
        pass

    # Attempt 2: tuple/list update
    try:
        yt_list = as_tuple(y_true, names=names, model=model, ctx="y_true", to_numpy=to_numpy)
        yp_list = as_tuple(y_pred, names=names, model=model, ctx="y_pred", to_numpy=to_numpy)
        cm.update_state(list(yt_list), list(yp_list))
        return
    except Exception:
        pass

    # Attempt 3: manual per-metric update (last resort, never crash)
    try:
        metrics = getattr(cm, "metrics", []) or []
        if isinstance(y_true, Mapping) and isinstance(y_pred, Mapping):
            for m in metrics:
                key = _metric_key_from_name(getattr(m, "name", "") or "")
                if not key:
                    continue
                if key in y_true and key in y_pred:
                    try:
                        m.update_state(y_true[key], y_pred[key])
                    except Exception:
                        continue
    except Exception:
        return

def _canonicalize_targets(targets):
    """
    Return targets as a dict matching compiled output names.

    Accepts:
    - dict targets: {"subsidence": ..., "gwl": ...} or already
      {"subs_pred": ..., "gwl_pred": ...}
    - tuple/list targets: (subs, gwl) -> {"subs_pred": subs, "gwl_pred": gwl}

    This avoids Keras 3 structure-mismatch errors when y_pred is a dict.
    """

    if isinstance(targets, Mapping):
        tgt = dict(targets)
        # already canonical?
        if ("subs_pred" in tgt) or ("gwl_pred" in tgt):
            return tgt
        return rename_dict_keys(
            tgt,
            param_to_rename={"subsidence": "subs_pred", "gwl": "gwl_pred"},
        )

    if isinstance(targets, (tuple, list)):
        if len(targets) == 2:
            return {"subs_pred": targets[0], "gwl_pred": targets[1]}
        if len(targets) == 1:
            return {"subs_pred": targets[0]}

    return targets