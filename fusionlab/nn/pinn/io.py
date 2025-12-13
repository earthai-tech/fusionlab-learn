# -*- coding: utf-8 -*-
"""
I/O helpers for physics diagnostics payloads.

This module centralizes data collection from a trained model for
physics sanity plots (e.g., Fig.4) and provides robust persistence
to disk with simple provenance metadata.
"""

from __future__ import annotations
import os, json, time
from typing import Dict, Iterable, Tuple, Optional, Any 
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
        "lambda_bounds": float(getattr(model, "lambda_bounds", 0.0)),
        "lambda_offsets": float(getattr(model, "lambda_offsets", 0.0)),

    }

# ---------------------------- core routines -------------------------------- #

def identifiability_diagnostics_from_payload(
    payload: Dict[str, np.ndarray],
    tau_true: float,
    K_true: float,
    Ss_true: float,
    Hd_true: float,
    K_prior: float,
    Ss_prior: float,
    Hd_prior: float,
    quantiles: Tuple[float, ...] = (0.5, 0.75, 0.9, 0.95),
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute synthetic identifiability diagnostics from a physics payload.

    This implements the three diagnostics described in
    Supplementary Methods 3:

    1. Relative error in the effective relaxation time tau.
    2. Discrepancy between the composite timescale closure
       H_d^2 S_s / (kappa K) (stored as tau_prior) and the true
       effective timescale tau_eff,true, via a log-timescale residual.
    3. Marginal log-offsets of K, S_s and H_d relative to their
       true effective values and lithology-based priors.

    Parameters
    ----------
    payload : dict
        Physics payload returned by :func:`gather_physics_payload`
        or :meth:`GeoPriorSubsNet.export_physics_payload`.
        Must contain 1D arrays with keys:
        "tau", "tau_prior", "K", "Ss", "Hd".
    tau_true : float
        True effective relaxation time tau_eff,true from the
        1D consolidation column.
    K_true, Ss_true, Hd_true : float
        True effective closures (K_eff, Ss_eff, H_d,eff) at the
        column scale.
    K_prior, Ss_prior, Hd_prior : float
        Lithology-based priors used to construct the GeoPrior head
        for this synthetic column.
    quantiles : tuple of float, default (0.5, 0.75, 0.9, 0.95)
        Quantile levels used for summary statistics of the
        distributions.
    eps : float, default 1e-12
        Lower bound used to clip strictly positive quantities
        before taking logarithms.

    Returns
    -------
    dict
        A dictionary with three blocks:

        - "tau_rel_error": statistics of the relative error
          |tau - tau_true| / tau_true.
        - "closure_log_resid": statistics of the log-timescale
          residual log(tau_prior) - log(tau_true).
        - "offsets": nested dict with "vs_true" and "vs_prior",
          each containing summary stats for the log-offsets
          delta_K, delta_Ss, delta_Hd.
    """

    # from fusionlab.nn.pinn.io import (
    #     load_physics_payload,
    #     identifiability_diagnostics_from_payload,
    # )
    
    # # 1) Load payload saved by model.export_physics_payload(...)
    # payload, meta = load_physics_payload("synthetic_column_run01.npz")
    
    # # 2) Your synthetic "truth" & priors (computed from the column)
    # tau_eff_true = tau_eff_from_column   # scalar
    # K_eff_true   = K_eff_from_column
    # Ss_eff_true  = Ss_eff_from_column
    # Hd_eff_true  = Hd_eff_from_column
    
    # K_prior = K_prior_from_lithology
    # Ss_prior = Ss_prior_from_lithology
    # Hd_prior = Hd_prior_from_lithology
    
    # # 3) Compute diagnostics
    # diag = identifiability_diagnostics_from_payload(
    #     payload,
    #     tau_true=tau_eff_true,
    #     K_true=K_eff_true,
    #     Ss_true=Ss_eff_true,
    #     Hd_true=Hd_eff_true,
    #     K_prior=K_prior,
    #     Ss_prior=Ss_prior,
    #     Hd_prior=Hd_prior,
    # )
    
    # # Then you can access e.g.
    # diag["tau_rel_error"]["median"]
    # diag["tau_rel_error"]["q95"]
    # diag["closure_log_resid"]["mean"]
    # diag["offsets"]["vs_true"]["delta_K"]["q90"]

    tau = np.asarray(payload["tau"], dtype=float)
    tau_prior = np.asarray(payload["tau_prior"], dtype=float)
    K = np.asarray(payload["K"], dtype=float)
    Ss = np.asarray(payload["Ss"], dtype=float)
    Hd = np.asarray(payload["Hd"], dtype=float)

    # --- 1. Relative error in tau --------------------------------------
    rel_err_tau = np.abs(tau - tau_true) / max(tau_true, eps)

    def _summ_stats(x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        out = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
        }
        for q in quantiles:
            out[f"q{int(q*100):02d}"] = float(np.quantile(x, q))
        return out

    tau_rel_stats = _summ_stats(rel_err_tau)

    # --- 2. Log-timescale residual for the closure ---------------------
    tau_prior_safe = np.clip(tau_prior, eps, None)
    log_resid = np.log(tau_prior_safe) - np.log(max(tau_true, eps))
    closure_stats = _summ_stats(log_resid)

    # --- 3. Log-offsets for K, Ss, Hd ----------------------------------
    K_safe = np.clip(K, eps, None)
    Ss_safe = np.clip(Ss, eps, None)
    Hd_safe = np.clip(Hd, eps, None)

    logK = np.log(K_safe)
    logSs = np.log(Ss_safe)
    logHd = np.log(Hd_safe)

    logK_true = np.log(max(K_true, eps))
    logSs_true = np.log(max(Ss_true, eps))
    logHd_true = np.log(max(Hd_true, eps))

    logK_prior = np.log(max(K_prior, eps))
    logSs_prior = np.log(max(Ss_prior, eps))
    logHd_prior = np.log(max(Hd_prior, eps))

    # Offsets vs true effective closures
    dK_true = logK - logK_true
    dSs_true = logSs - logSs_true
    dHd_true = logHd - logHd_true

    # Offsets vs lithology-based priors
    dK_prior = logK - logK_prior
    dSs_prior = logSs - logSs_prior
    dHd_prior = logHd - logHd_prior

    offsets = {
        "vs_true": {
            "delta_K": _summ_stats(dK_true),
            "delta_Ss": _summ_stats(dSs_true),
            "delta_Hd": _summ_stats(dHd_true),
        },
        "vs_prior": {
            "delta_K": _summ_stats(dK_prior),
            "delta_Ss": _summ_stats(dSs_prior),
            "delta_Hd": _summ_stats(dHd_prior),
        },
    }

    return {
        "tau_rel_error": tau_rel_stats,
        "closure_log_resid": closure_stats,
        "offsets": offsets,
    }

def summarise_effective_params(payload: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Collapse 1D arrays to scalar effective parameters.

    Intended for 1D synthetic-column experiments where model
    outputs are spatially constant and we only need a single
    representative value per run.
    """
    # payload = model.export_physics_payload(dataset, save_path=None)
    # eff = summarise_effective_params(payload)
    
    # tau_est       = eff["tau"]
    # tau_prior_est = eff["tau_prior"]
    # K_est         = eff["K"]
    # Ss_est        = eff["Ss"]
    # Hd_est        = eff["Hd"]
    
    out = {}
    for key in ("tau", "tau_prior", "K", "Ss", "Hd"):
        arr = np.asarray(payload[key])
        mask = np.isfinite(arr)
        if not mask.any():
            out[key] = float("nan")
        else:
            out[key] = float(np.median(arr[mask]))
    return out


def compute_identifiability_summary(
    eff_params: Dict[str, float],
    true_params: Dict[str, float],
    prior_params: Dict[str, float],
    kappa_b: float = 1.0,
) -> Dict[str, float]:
    """
    Compute identifiability diagnostics for Supp. Methods 3.

    See Supplementary Methods 3 for definitions of the
    quantities returned.
    """
    # summaries = []
    # for run_seed in seeds:
    #     # train 1D model, build dataset, etc…
    
    #     payload = model.export_physics_payload(dataset, save_path=None)
    #     eff = summarise_effective_params(payload)
    
    #     true_params = {
    #         "tau": tau_eff_true,
    #         "K": K_eff_true,
    #         "Ss": Ss_eff_true,
    #         "Hd": Hd_eff_true,
    #     }
    #     prior_params = {
    #         "K": K_prior,
    #         "Ss": Ss_prior,
    #         "Hd": Hd_prior,
    #     }
    
    #     s = compute_identifiability_summary(eff, true_params, prior_params, kappa_b)
    #     summaries.append(s)
    
    # # Example: median relative error in tau across runs
    # rel_err_tau_all = np.array([s["rel_err_tau"] for s in summaries])
    # median_rel_err_tau = np.median(rel_err_tau_all)
    # q90_rel_err_tau = np.quantile(rel_err_tau_all, 0.9)


    tau_est = eff_params["tau"]
    K_est   = eff_params["K"]
    Ss_est  = eff_params["Ss"]
    Hd_est  = eff_params["Hd"]

    tau_true = true_params["tau"]

    rel_err_tau = abs(tau_est - tau_true) / tau_true

    closure_est = Hd_est**2 * Ss_est / (kappa_b * K_est)
    log_closure_resid = float(np.log(closure_est) - np.log(tau_true))

    K_prior  = prior_params["K"]
    Ss_prior = prior_params["Ss"]
    Hd_prior = prior_params["Hd"]

    delta_K_prior  = float(np.log(K_est)  - np.log(K_prior))
    delta_Ss_prior = float(np.log(Ss_est) - np.log(Ss_prior))
    delta_Hd_prior = float(Hd_est - Hd_prior)

    delta_K_true  = float(np.log(K_est)  - np.log(true_params["K"]))
    delta_Ss_true = float(np.log(Ss_est) - np.log(true_params["Ss"]))
    delta_Hd_true = float(Hd_est - true_params["Hd"])

    return {
        "rel_err_tau": float(rel_err_tau),
        "log_closure_resid": log_closure_resid,
        "delta_K_prior": delta_K_prior,
        "delta_Ss_prior": delta_Ss_prior,
        "delta_Hd_prior": delta_Hd_prior,
        "delta_K_true": delta_K_true,
        "delta_Ss_true": delta_Ss_true,
        "delta_Hd_true": delta_Hd_true,
    }


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
        "Hd": np.concatenate(Hds), #  # note: from phys["H"]
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
        path = os.getcwd()
        
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
