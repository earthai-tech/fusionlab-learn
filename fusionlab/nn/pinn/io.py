# -*- coding: utf-8 -*-
"""
I/O helpers for physics diagnostics payloads.

This module centralizes data collection from a trained model for
physics sanity plots (e.g., Fig.4) and provides robust persistence
to disk with simple provenance metadata.
"""

from __future__ import annotations
import os, json, time
from typing import Dict, Iterable, Tuple, Optional
import numpy as np
import pandas as pd  

from ..._optdeps import with_progress


def _iso_now() -> str:
    """Return current UTC time in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _to_1d(x, dtype=np.float32) -> np.ndarray:
    """Flatten to 1D and cast."""
    arr = np.asarray(x)
    return np.ravel(arr).astype(dtype, copy=False)


def _r2_from_logs(x: np.ndarray, y: np.ndarray) -> float:
    """R^2 between log-transformed arrays (already logs)."""
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm**2).sum() * (ym**2).sum()
    if denom <= 0:
        return float("nan")
    return float(((xm * ym).sum() ** 2) / denom)


def _maybe_subsample(payload: Dict[str, np.ndarray],
                     frac: Optional[float]) -> Dict[str, np.ndarray]:
    """Randomly subsample rows from a payload (for speed/size)."""
    if frac is None:
        return payload
    f = float(frac)
    if not (0.0 < f <= 1.0):
        raise ValueError("random_subsample must be in (0, 1].")
    n = payload["tau"].shape[0]
    keep = np.random.choice(n, size=int(np.ceil(f * n)), replace=False)
    out = {}
    for k, v in payload.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] == n:
            out[k] = v[keep]
        else:
            out[k] = v
    return out


def default_meta_from_model(model) -> Dict:
    """
    Build lightweight, JSON-serializable provenance from a model.
    """
    return {
        "created_utc": _iso_now(),
        "model_name": type(model).__name__,
        "model_version": getattr(model, "get_config", lambda: {})()
            .get("model_version", None),
        "pde_modes_active": getattr(model, "pde_modes_active", None),
        "kappa_mode": getattr(model, "kappa_mode", None),
        "use_effective_thickness":
            bool(getattr(model, "use_effective_thickness", False)),
        "Hd_factor": float(getattr(model, "Hd_factor", 1.0)),
        "lambda_cons": float(getattr(model, "lambda_cons", 0.0)),
        "lambda_gw": float(getattr(model, "lambda_gw", 0.0)),
        "lambda_prior": float(getattr(model, "lambda_prior", 0.0)),
        "lambda_smooth": float(getattr(model, "lambda_smooth", 0.0)),
        "lambda_mv": float(getattr(model, "lambda_mv", 0.0)),
        "quantiles": list(getattr(model, "quantiles", []) or []),
    }


# ---------------------------- core routines -------------------------------- #

def gather_physics_payload(
    model,
    dataset: Iterable,
    max_batches: Optional[int] = None,
    float_dtype=np.float32,
    log_fn =None, 
    **tqdm_kws
) -> Dict[str, np.ndarray]:
    """
    Iterate a dataset and collect flattened arrays for physics plots.

    Parameters
    ----------
    model : tf.keras.Model-like
        Must expose `evaluate_physics(inputs, return_maps=True)` returning
        a dict with keys: "K", "Ss", "H", "tau", "tau_prior", "R_cons".
    dataset : iterable
        Yields (inputs, targets) or inputs that `evaluate_physics` accepts.
    max_batches : int or None
        Optional cap on the number of batches processed.
    float_dtype : numpy dtype
        Casting dtype for returned arrays (e.g., np.float32).

    Returns
    -------
    dict
        1D arrays with keys: "tau", "tau_prior", "K", "Ss",
        "Hd", "cons_res_vals", "log10_tau", "log10_tau_prior",
        and a small "metrics" sub-dict with "eps_prior_rms", "r2_logtau".
    """
    taus, tau_ps, Ks, Sss, Hds, cons_vals = [], [], [], [], [], []
    n = 0
    # Optional tqdm progress bar
    iterable = dataset
    try:
        total = max_batches if max_batches is not  None else len(dataset) 
    except: 
        # Use len(dataset) if available; otherwise tqdm will show unknown total
        total =None 
    
    iterable = with_progress(
        dataset,
        total=total,
        desc="Gathering physics payload",
        ascii=True,
        leave=False, 
        log_fn= log_fn, 
        **tqdm_kws
    )

    for batch in iterable:
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        phys = model.evaluate_physics(inputs, return_maps=True)

        taus.append(_to_1d(phys["tau"], dtype=float_dtype))
        tau_ps.append(_to_1d(phys["tau_prior"], dtype=float_dtype))
        Ks.append(_to_1d(phys["K"], dtype=float_dtype))
        Sss.append(_to_1d(phys["Ss"], dtype=float_dtype))
        Hds.append(_to_1d(phys["H"], dtype=float_dtype))
        cons_vals.append(_to_1d(phys["R_cons"], dtype=float_dtype))

        n += 1
        if (max_batches is not None) and (n >= max_batches):
            break

    if not taus:
        raise ValueError("gather_physics_payload: dataset yielded no batches.")

    payload = {
        "tau": np.concatenate(taus),
        "tau_prior": np.concatenate(tau_ps),
        "K": np.concatenate(Ks),
        "Ss": np.concatenate(Sss),
        "Hd": np.concatenate(Hds),
        "cons_res_vals": np.concatenate(cons_vals),
    }

    # derived logs + metrics used in Fig.4 annotations
    tau_clip = np.clip(payload["tau"], 1e-12, None)
    tp_clip = np.clip(payload["tau_prior"], 1e-12, None)
    payload["log10_tau"] = np.log10(tau_clip)
    payload["log10_tau_prior"] = np.log10(tp_clip)

    eps_prior_rms = float(
        np.sqrt(np.mean((np.log(tau_clip) - np.log(tp_clip)) ** 2))
    )
    r2_logtau = _r2_from_logs(
        payload["log10_tau_prior"], payload["log10_tau"]
    )
    payload["metrics"] = {
        "eps_prior_rms": eps_prior_rms,
        "r2_logtau": r2_logtau,
    }
    return payload


def save_physics_payload(
    payload: Dict[str, np.ndarray],
    meta: Dict,
    path: str | None =None,
    format: str = "npz",
    overwrite: bool = False,
    log_fn =None, 
) -> str:
    """
    Save payload + sidecar metadata to disk.

    Parameters
    ----------
    payload : dict
        Output of `gather_physics_payload`.
    meta : dict
        Provenance dictionary. Will be JSON-serialized.
    path : str or Nonr
        File path. If extension missing, inferred from `format`.
        If not provided, then get the current directory instead. 
    format : {"npz","csv","parquet"}
        Storage format. "npz" is compact and dependency-free.
    overwrite : bool
        If False, raise if the file already exists.

    Returns
    -------
    str
        The resolved data file path that was written.
    """
    log = log_fn if log_fn is not None else print 
    
    if path is None: 
        path = os.gcwd() 
        
    root, ext = os.path.splitext(path)
    if ext == "":
        ext = "." + format.lower()
        path = root + ext
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"File exists: {path}")

    meta = dict(meta or {})
    meta.setdefault("saved_utc", _iso_now())

    if format.lower() == "npz":
        np.savez_compressed(
            path,
            tau=payload["tau"],
            tau_prior=payload["tau_prior"],
            K=payload["K"],
            Ss=payload["Ss"],
            Hd=payload["Hd"],
            cons_res_vals=payload["cons_res_vals"],
            log10_tau=payload["log10_tau"],
            log10_tau_prior=payload["log10_tau_prior"],
        )
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path

    df = pd.DataFrame({
        "tau": payload["tau"],
        "tau_prior": payload["tau_prior"],
        "K": payload["K"],
        "Ss": payload["Ss"],
        "Hd": payload["Hd"],
        "cons_res_vals": payload["cons_res_vals"],
        "log10_tau": payload["log10_tau"],
        "log10_tau_prior": payload["log10_tau_prior"],
    })
    if format.lower() == "csv":
        df.to_csv(path, index=False)
    elif format.lower() == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError("format must be one of {'npz','csv','parquet'}")

    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    log(f"Physics payload sucessfully saved to {path}")
    return path


def load_physics_payload(path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load a previously saved physics payload and its metadata.

    Parameters
    ----------
    path : str
        Data file path. Supports .npz, .csv, .parquet.

    Returns
    -------
    (payload, meta) : (dict, dict)
        Payload dict with arrays and metadata dict (if found).
    """
    root, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == ".npz":
        arrs = np.load(path)
        payload = {k: arrs[k] for k in arrs.files}
        meta_path = path + ".meta.json"
    elif ext in (".csv", ".parquet"):
        # if pd is None:
        #     raise RuntimeError(
        #         "CSV/Parquet requires pandas. Install pandas/pyarrow."
        #     )
        df = pd.read_csv(path) if ext == ".csv" else pd.read_parquet(path)
        payload = {c: df[c].to_numpy() for c in df.columns}
        meta_path = path + ".meta.json"
    else:
        raise ValueError("Unsupported extension. Use .npz, .csv or .parquet.")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return payload, meta
