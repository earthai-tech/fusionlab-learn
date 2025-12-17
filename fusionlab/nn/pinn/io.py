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
    # Try to fetch time_units from model/scaling_kwargs if available
    skw = getattr(model, "scaling_kwargs", None) or {}
    time_units = skw.get("time_units", getattr(model, "time_units", None))

    # Prefer your real attribute names, fallback to old ones
    use_eff_h = bool(getattr(model, "use_effective_h",
                     getattr(model, "use_effective_thickness", False)))
    hd_factor = float(getattr(model, "hd_factor",
                      getattr(model, "Hd_factor", 1.0)))
    lambda_offset = float(getattr(model, "lambda_offset",
                          getattr(model, "lambda_offsets", 0.0)))

    meta = {
        "created_utc": _iso_now(),
        "model_name": type(model).__name__,
        "pde_modes_active": getattr(model, "pde_modes_active", None),
        "kappa_mode": getattr(model, "kappa_mode", None),

        # fixed / clarified
        "use_effective_h": use_eff_h,
        "hd_factor": hd_factor,
        "lambda_offset": lambda_offset,

        # optional: keep old keys for backward compat
        "use_effective_thickness": use_eff_h,
        "Hd_factor": hd_factor,
        "lambda_offsets": lambda_offset,

        "time_units": time_units,
        "units": {
            "Hd": "m",
            "Ss": "1/m",
            "K": f"m/{time_units}" if time_units else "unknown",
            "tau": str(time_units) if time_units else "unknown",
            "tau_prior": str(time_units) if time_units else "unknown",
            "cons_res_vals": "see model scaling",  # residual may be scaled
        },

        # your naming clarity block is good:
        "tau_prior_definition": "tau_closure_from_learned_fields",
        "tau_prior_human_name": "tau_closure",
        "tau_prior_source": "model.evaluate_physics() closure head",
        "tau_closure_formula": "Hd^2 * Ss / (pi^2 * kappa_b * K)",
    }
    return meta

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
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Compute identifiability diagnostics for Supp. Methods 3.

    See Supplementary Methods 3 for definitions of the
    quantities returned.
    """

    tau_est = eff_params["tau"]
    K_est   = eff_params["K"]
    Ss_est  = eff_params["Ss"]
    Hd_est  = eff_params["Hd"]

    tau_true = true_params["tau"]

    rel_err_tau = abs(tau_est - tau_true) / tau_true

    # closure_est = Hd_est**2 * Ss_est / (kappa_b * K_est)
    # log_closure_resid = float(np.log(closure_est) - np.log(tau_true))
    # FIX: include pi^2
    closure_est = (Hd_est**2) * Ss_est / (
        np.pi**2 * max(kappa_b, eps) * max(K_est, eps))
    log_closure_resid = float(
        np.log(max(closure_est, eps)) - np.log(max(tau_true, eps)))
    
    K_prior  = prior_params["K"]
    Ss_prior = prior_params["Ss"]
    Hd_prior = prior_params["Hd"]


    #  use log offsets for Hd to match the main diagnostics
    delta_K_prior  = float(np.log(max(K_est, eps))  - np.log(max(K_prior, eps)))
    delta_Ss_prior = float(np.log(max(Ss_est, eps)) - np.log(max(Ss_prior, eps)))
    delta_Hd_prior = float(np.log(max(Hd_est, eps)) - np.log(max(Hd_prior, eps)))

    delta_K_true  = float(np.log(max(K_est, eps))  - np.log(max(true_params["K"], eps)))
    delta_Ss_true = float(np.log(max(Ss_est, eps)) - np.log(max(true_params["Ss"], eps)))
    delta_Hd_true = float(np.log(max(Hd_est, eps)) - np.log(max(true_params["Hd"], eps)))
    
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
    log_fn=None,
    eps: float = 1e-12,
    **tqdm_kws,
) -> Dict[str, np.ndarray]:
    """
    Collect a *flat* physics payload from a batched dataset for diagnostics.

    This function iterates over a `tf.data.Dataset` (or any iterable) and
    calls `model.evaluate_physics(inputs, return_maps=True)` on each batch.
    The returned per-batch tensors are flattened and concatenated into
    1D arrays suitable for scatter plots, histograms, and summary stats.

    Important
    ---------
    **No unit conversion is performed here.**
    The payload is exported in *whatever units* `evaluate_physics(...)`
    returns. Unit consistency is therefore a responsibility of the model's
    physics implementation (and its `scaling_kwargs`), not this I/O layer.

    Expected model output
    ---------------------
    `evaluate_physics(..., return_maps=True)` must return a dict with:
      - "tau"       : relaxation time (effective timescale)
      - "tau_prior" : closure timescale implied by (Hd, Ss, K, kappa)
      - "K"         : hydraulic conductivity (effective)
      - "Ss"        : specific storage (effective)
      - "Hd" or "H" : drainage thickness (effective)
      - "R_cons"    : consolidation residual values (diagnostic)

    Parameters
    ----------
    model : tf.keras.Model-like
        A model exposing `evaluate_physics(inputs, return_maps=True)`.
    dataset : iterable
        Yields either `inputs` or `(inputs, targets)`. If targets are present,
        they are ignored (physics evaluation uses inputs only).
    max_batches : int or None, default=None
        If provided, stop after this many batches.
    float_dtype : numpy dtype, default=np.float32
        Dtype used when casting flattened arrays (keeps files compact).
    log_fn : callable or None, default=None
        Optional logger passed to the progress helper (e.g., `print`).
    eps : float, default=1e-12
        Small positive constant used to clip strictly-positive quantities
        before applying log/log10. Prevents `-inf` and NaNs.
    **tqdm_kws :
        Extra keyword arguments forwarded to the progress helper.

    Returns
    -------
    payload : dict
        Dictionary with 1D arrays:
          - "tau", "tau_prior", "K", "Ss", "Hd", "cons_res_vals"
          - "log10_tau", "log10_tau_prior"
          - aliases: "tau_closure" == "tau_prior",
                     "log10_tau_closure" == "log10_tau_prior"
        and a nested dict `payload["metrics"]` with:
          - "eps_prior_rms" : RMS of log(tau) - log(tau_prior)
          - "r2_logtau"     : R² between log10(tau_prior) and log10(tau)
          - "closure_consistency_rms" : (optional) RMS between log(tau_prior)
            and log(tau_closure_calc) computed from (Hd, Ss, K) and a scalar
            kappa extracted from the model (NaN if unavailable)
        If kappa is available, also includes:
          - "tau_closure_calc" : closure computed from Hd² Ss / (π² kappa K)
    """
    # ----------------------------- setup ---------------------------------
    taus, tau_priors, Ks, Sss, Hds, cons_vals = [], [], [], [], [], []

    # Progress reporting: use `len(dataset)` if available, else unknown total.
    try:
        total = max_batches if max_batches is not None else len(dataset)
    except Exception:
        total = max_batches  # may still be None

    iterable = with_progress(
        dataset,
        total=total,
        desc="Gathering physics payload",
        ascii=True,
        leave=False,
        log_fn=log_fn,
        **tqdm_kws,
    )

    # ------------------------- batch iteration ---------------------------
    n_seen = 0
    for batch in iterable:
        # Support datasets that yield either `inputs` or `(inputs, targets)`.
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch

        phys = model.evaluate_physics(inputs, return_maps=True)

        # Flatten batch maps -> 1D arrays; concatenate at the end.
        taus.append(_to_1d(phys["tau"], dtype=float_dtype))
        tau_priors.append(_to_1d(phys["tau_prior"], dtype=float_dtype))
        Ks.append(_to_1d(phys["K"], dtype=float_dtype))
        Sss.append(_to_1d(phys["Ss"], dtype=float_dtype))

        # Historical compatibility: some implementations return "H" not "Hd".
        Hd_key = "Hd" if "Hd" in phys else "H"
        Hds.append(_to_1d(phys[Hd_key], dtype=float_dtype))

        cons_vals.append(_to_1d(phys["R_cons"], dtype=float_dtype))

        n_seen += 1
        if max_batches is not None and n_seen >= max_batches:
            break

    if not taus:
        raise ValueError("gather_physics_payload: dataset yielded no batches.")

    # ------------------------ concatenate arrays -------------------------
    payload: Dict[str, np.ndarray] = {
        "tau": np.concatenate(taus, axis=0),
        "tau_prior": np.concatenate(tau_priors, axis=0),
        "K": np.concatenate(Ks, axis=0),
        "Ss": np.concatenate(Sss, axis=0),
        "Hd": np.concatenate(Hds, axis=0),
        "cons_res_vals": np.concatenate(cons_vals, axis=0),
    }

    # -------------------------- safe logs --------------------------------
    tau_clip = np.clip(payload["tau"], eps, None)
    tp_clip = np.clip(payload["tau_prior"], eps, None)

    payload["log10_tau"] = np.log10(tau_clip)
    payload["log10_tau_prior"] = np.log10(tp_clip)

    # Back/forward compatible naming: "tau_prior" is the closure timescale.
    payload["tau_closure"] = payload["tau_prior"]
    payload["log10_tau_closure"] = payload["log10_tau_prior"]

    # ---------------------- primary summary metrics ----------------------
    eps_prior_rms = float(
        np.sqrt(np.mean((np.log(tau_clip) - np.log(tp_clip)) ** 2))
    )
    r2_logtau = _r2_from_logs(payload["log10_tau_prior"], payload["log10_tau"])

    payload["metrics"] = {
        "eps_prior_rms": eps_prior_rms,
        "r2_logtau": r2_logtau,
    }

    # ----------------- optional closure consistency check ----------------
    # If the model exposes a usable scalar kappa, recompute tau_closure from
    # (Hd, Ss, K) and compare to the reported tau_prior in log-space.
    kappa_val = None
    for attr in ("kappa_value", "kappa", "kappa_b", "kappa_bar"):
        if hasattr(model, attr):
            try:
                cand = getattr(model, attr)
                # handle keras Variables / params with `.numpy()`
                if hasattr(cand, "numpy"):
                    cand = cand.numpy()
                kappa_val = float(np.asarray(cand).reshape(()))
                break
            except Exception:
                kappa_val = None

    if kappa_val is not None and np.isfinite(kappa_val) and kappa_val > 0:
        K = np.clip(payload["K"], eps, None)
        Ss = np.clip(payload["Ss"], eps, None)
        Hd = np.clip(payload["Hd"], eps, None)

        tau_closure_calc = (Hd ** 2) * Ss / (np.pi ** 2 * kappa_val * K)
        payload["tau_closure_calc"] = tau_closure_calc

        tp = np.clip(payload["tau_prior"], eps, None)
        tc = np.clip(tau_closure_calc, eps, None)

        payload["metrics"]["closure_consistency_rms"] = float(
            np.sqrt(np.mean((np.log(tp) - np.log(tc)) ** 2))
        )
        payload["metrics"]["kappa_used_for_closure"] = float(kappa_val)
    else:
        payload["metrics"]["closure_consistency_rms"] = float("nan")
        payload["metrics"]["kappa_used_for_closure"] = None

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
        path = os.path.join(os.getcwd(), "physics_payload.npz")

    if os.path.isdir(path):
        path = os.path.join(path, "physics_payload.npz")

        
    root, ext = os.path.splitext(path)
    if ext == "":
        ext = "." + format.lower()
        path = root + ext
    if (not overwrite) and os.path.exists(path):
        raise FileExistsError(f"File exists: {path}")

    meta = dict(meta or {})
    meta.setdefault("saved_utc", _iso_now())
    
    # Ensure definition always exists even if caller bypassed default_meta
    meta.setdefault("tau_prior_definition", "tau_closure_from_learned_fields")
    meta.setdefault("tau_prior_human_name", "tau_closure")
    meta.setdefault("tau_prior_source", "model.evaluate_physics() closure head")
    
    tau_prior = payload["tau_prior"]
    
    if format.lower() == "npz":
        np.savez_compressed(
            path,
            tau=payload["tau"],
            tau_prior=tau_prior,                    # legacy key
            tau_closure=tau_prior,                  # alias
            K=payload["K"],
            Ss=payload["Ss"],
            Hd=payload["Hd"],
            cons_res_vals=payload["cons_res_vals"],
            log10_tau=payload["log10_tau"],
            log10_tau_prior=payload["log10_tau_prior"],
            log10_tau_closure=payload["log10_tau_prior"],  # alias
        )
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path

    df = pd.DataFrame({
        "tau": payload["tau"],
        "tau_prior": payload["tau_prior"],
        "tau_closure": payload.get("tau_closure", payload["tau_prior"]),
        "K": payload["K"],
        "Ss": payload["Ss"],
        "Hd": payload["Hd"],
        "cons_res_vals": payload["cons_res_vals"],
        "log10_tau": payload["log10_tau"],
        "log10_tau_prior": payload["log10_tau_prior"],
        "log10_tau_closure": payload.get(
            "log10_tau_closure", payload["log10_tau_prior"]
        ),
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
            
    # Back/forward compatible aliases
    if "tau_prior" not in payload and "tau_closure" in payload:
        payload["tau_prior"] = payload["tau_closure"]
    if "tau_closure" not in payload and "tau_prior" in payload:
        payload["tau_closure"] = payload["tau_prior"]
    
    if "log10_tau_prior" not in payload and "log10_tau_closure" in payload:
        payload["log10_tau_prior"] = payload["log10_tau_closure"]
    if "log10_tau_closure" not in payload and "log10_tau_prior" in payload:
        payload["log10_tau_closure"] = payload["log10_tau_prior"]

    return payload, meta
