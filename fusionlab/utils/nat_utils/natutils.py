from __future__ import annotations

import json
import os
import glob
from typing import Any 
import datetime as dt 
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd 

# --- Optional TensorFlow import for GeoPrior helpers -----------------------
try:  # pragma: no cover - defensive import
    import tensorflow as tf # noqa
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except Exception:  # pragma: no cover
    TF_AVAILABLE = False
    tf = None  # type: ignore[assignment]

    class _AdamStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "TensorFlow is required for NATCOM GeoPrior helpers "
                "(e.g. compile_geoprior_for_eval). Please install "
                "`tensorflow>=2.12`."
            )

    Adam = _AdamStub  # type: ignore[assignment]
    
def save_ablation_record(
    outdir: str,
    city: str,
    model_name: str,
    cfg: dict,
    eval_dict: dict | None,
    phys_diag: dict | None = None,
    per_h_mae: dict | None = None,
    per_h_r2: dict | None = None,
    log_fn = None, 
) -> None:
    """
    Append a single ablation record to ``ablation_record.jsonl``.

    Each training run (e.g., different physics toggles or weights)
    writes one JSON line containing:

    - Basic run identifiers (city, model, timestamp).
    - Physics configuration (``PDE_MODE_CONFIG``, lambda weights,
      effective head flags, etc.).
    - Key performance metrics (R², MSE, MAE, coverage, sharpness).
    - Optional physics diagnostics (``epsilon_prior``,
      ``epsilon_cons``).
    - Optional per-horizon MAE/R² for more detailed analysis.

    Parameters
    ----------
    outdir : str
        Base output directory for the current run. The ablation
        file is created under ``outdir / "ablation_records"``.
    city : str
        City name (e.g., ``"nansha"`` or ``"zhongshan"``).
    model_name : str
        Model identifier (e.g., ``"GeoPriorSubsNet"``).
    cfg : dict
        Lightweight configuration dictionary containing at least
        the physics-related keys used below.
    eval_dict : dict or None
        Dictionary of evaluation metrics (R², MSE, MAE,
        coverage80, sharpness80). If ``None``, metrics fields
        default to ``None``.
    phys_diag : dict or None, optional
        Physics diagnostics (e.g., from ``evaluate()``) with keys
        such as ``"epsilon_prior"`` and ``"epsilon_cons"``.
    per_h_mae : dict or None, optional
        Per-horizon MAE values (e.g., keyed by year/step).
    per_h_r2 : dict or None, optional
        Per-horizon R² values.

    Notes
    -----
    The output file is a JSON-Lines file, so it can be loaded
    with :func:`load_ablation_jsonl`.
    """
    if log_fn is None: 
        log_fn =print 
        
    eval_dict = eval_dict or {}

    rec = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "city": city,
        "model": model_name,
        # Physics toggles / weights
        "pde_mode": cfg.get("PDE_MODE_CONFIG"),
        "use_effective_h": bool(cfg.get("GEOPRIOR_USE_EFFECTIVE_H", True)),
        "kappa_mode": cfg.get("GEOPRIOR_KAPPA_MODE", "bar"),
        "hd_factor": cfg.get("GEOPRIOR_HD_FACTOR", 0.6),
        "lambda_cons": cfg.get("LAMBDA_CONS"),
        "lambda_gw": cfg.get("LAMBDA_GW"),
        "lambda_prior": cfg.get("LAMBDA_PRIOR"),
        "lambda_smooth": cfg.get("LAMBDA_SMOOTH"),
        "lambda_mv": cfg.get("LAMBDA_MV"),
        # Key metrics
        "r2": eval_dict.get("r2"),
        "mse": eval_dict.get("mse"),
        "mae": eval_dict.get("mae"),
        "coverage80": eval_dict.get("coverage80"),
        "sharpness80": eval_dict.get("sharpness80"),
    }

    if phys_diag:
        rec["epsilon_prior"] = phys_diag.get("epsilon_prior")
        rec["epsilon_cons"] = phys_diag.get("epsilon_cons")

    if per_h_mae is not None:
        rec["per_horizon_mae"] = per_h_mae
    if per_h_r2 is not None:
        rec["per_horizon_r2"] = per_h_r2

    abl_dir = os.path.join(outdir, "ablation_records")
    os.makedirs(abl_dir, exist_ok=True)

    jpath = os.path.join(abl_dir, "ablation_record.jsonl")
    with open(jpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    log_fn(f"[Ablation] appended -> {jpath}")


def load_ablation_jsonl(path: str) -> pd.DataFrame:
    """
    Load an ablation JSON-Lines file into a :class:`pandas.DataFrame`.

    This is the companion to :func:`save_ablation_record`. Each
    line is parsed as JSON and turned into one row.

    Parameters
    ----------
    path : str
        Path to ``ablation_record.jsonl``.

    Returns
    -------
    pandas.DataFrame
        DataFrame where each row corresponds to one ablation
        record.

    Examples
    --------
    >>> df_abl = load_ablation_jsonl(
    ...     "ablation_records/ablation_record.jsonl"
    ... )
    >>> df_abl.head()
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def name_of(obj: object) -> str:
    """
    Return a human-readable name for an object.

    This utility is handy when serialising compile configurations
    (e.g., turning metric callables into simple strings for JSON
    logs).

    Parameters
    ----------
    obj : object
        Any Python object (function, class instance, etc.).

    Returns
    -------
    str
        ``obj.__name__`` if present, otherwise the class name, and
        finally ``str(obj)`` as a last resort.
    """
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "__class__"):
        return obj.__class__.__name__
    return str(obj)


def serialize_subs_params(
    params: dict,
    cfg: dict | None = None,
) -> dict:
    """
    Make GeoPrior subnet parameters JSON-friendly.

    The training scripts typically pass a dictionary of model
    construction arguments, e.g. ``subsmodel_params``, which
    contains objects such as ``LearnableMV`` or ``FixedGammaW``
    that are not directly JSON-serialisable.

    This helper replaces those objects by small dictionaries
    describing their type and scalar value, optionally using
    values from the NATCOM config dictionary.

    Parameters
    ----------
    params : dict
        Dictionary of model init parameters (e.g.
        ``subsmodel_params`` in ``training_NATCOM_GEOPRIOR.py``).
    cfg : dict, optional
        NATCOM config dictionary. If provided, scalar values are
        taken from:

        - ``GEOPRIOR_INIT_MV``
        - ``GEOPRIOR_INIT_KAPPA``
        - ``GEOPRIOR_GAMMA_W``
        - ``GEOPRIOR_H_REF``

        and used as the authoritative numbers.

    Returns
    -------
    dict
        Copy of ``params`` where scalar GeoPrior parameters are
        replaced by JSON-friendly dictionaries.

    Notes
    -----
    This function does **not** import any of the GeoPrior classes.
    It only introspects attributes like ``initial_value`` or
    ``value`` when the corresponding config entry is missing.
    """
    out = dict(params)
    cfg = cfg or {}

    # Helper to extract a scalar from either the config or the
    # original object (Learnable*/Fixed*).
    def _extract_scalar(obj, cfg_key: str) -> float | None:
        if cfg_key in cfg and cfg[cfg_key] is not None:
            try:
                return float(cfg[cfg_key])
            except Exception:
                pass
        # Fallback: try to read a typical attribute name.
        for attr in ("initial_value", "value"):
            if hasattr(obj, attr):
                try:
                    return float(getattr(obj, attr))
                except Exception:
                    continue
        return None

    if "mv" in out:
        mv_val = _extract_scalar(out["mv"], "GEOPRIOR_INIT_MV")
        out["mv"] = {
            "type": "LearnableMV",
            "initial_value": mv_val,
        }

    if "kappa" in out:
        kap_val = _extract_scalar(out["kappa"], "GEOPRIOR_INIT_KAPPA")
        out["kappa"] = {
            "type": "LearnableKappa",
            "initial_value": kap_val,
        }

    if "gamma_w" in out:
        gw_val = _extract_scalar(out["gamma_w"], "GEOPRIOR_GAMMA_W")
        out["gamma_w"] = {
            "type": "FixedGammaW",
            "value": gw_val,
        }

    if "h_ref" in out:
        href_val = _extract_scalar(out["h_ref"], "GEOPRIOR_H_REF")
        out["h_ref"] = {
            "type": "FixedHRef",
            "value": href_val,
        }

    return out

def best_epoch_and_metrics(
    history: dict,
    monitor: str = "val_loss",
) -> tuple[int | None, dict]:
    """
    Return the best epoch and metrics at that epoch.

    Given a ``History.history`` dictionary produced by
    ``model.fit(...)``, this helper identifies the index of the
    minimum value for the monitored quantity (by default
    ``"val_loss"``) and returns:

    - The epoch index (0-based).
    - A dictionary mapping each metric name to its value at that
      epoch.

    Parameters
    ----------
    history : dict
        The ``history.history`` attribute from Keras training.
    monitor : str, default="val_loss"
        Name of the metric to minimise.

    Returns
    -------
    best_epoch : int or None
        Index of the best epoch, or ``None`` if ``monitor`` is
        not present.
    metrics_at_best : dict
        Mapping from metric name to its value at the best epoch.
        Empty if ``monitor`` is not present.
    """
    if not history or monitor not in history:
        return None, {}

    # nanargmin makes sure NaNs are ignored when searching for the
    # best epoch.
    be = int(np.nanargmin(history[monitor]))
    metrics_at_best = {
        k: float(v[be])
        for k, v in history.items()
        if len(v) > be
    }
    return be, metrics_at_best
    
def load_or_rebuild_geoprior_model(
    model_path: str,
    manifest: dict,
    X_sample: dict,
    out_s_dim: int,
    out_g_dim: int,
    mode: str,
    horizon: int,
    quantiles: list[float] | None,
    city_name: str | None = None,
    compile_on_load: bool = True,
    verbose: int = 1,
):
    """
    Load a tuned *or trained* GeoPriorSubsNet, with robust rebuild fallback.

    Strategy
    --------
    1. Try ``tf.keras.models.load_model(model_path)`` with all required
       custom objects registered.

    2. If that fails:

       2a) Try tuned-model reconstruction::

              best_hps = load_best_hps_near_model(...)
              model = build_geoprior_from_hps(...)

       2b) If no best_hps JSON is found, assume a plain *trained* model
           and fall back to the ``*_training_summary.json`` recorded next
           to the checkpoint::

              training_summary = load_training_summary_near_model(...)
              model = build_geoprior_from_training_summary(...)

       In both cases, a minimal ``best_hps``-like dict is returned so
       :func:`compile_for_eval` can recreate the physics weights and
       learning rate.

    3. Try to load the best weights checkpoint via
       :func:`infer_best_weights_path(model_path)`.

    Returns
    -------
    model :
        A GeoPriorSubsNet instance ready to be recompiled for evaluation.

    best_hps : dict or None
        Tuned hyperparameters if present, otherwise a small dict
        containing at least ``learning_rate`` and the lambda weights
        from the training summary, or ``None``.
    """
    label_city = city_name or "GeoPrior"

    # --- Lazy imports so nat_utils can be imported without TF/fusionlab ---
    try:
        import tensorflow as tf  # type: ignore # noqa
        from tensorflow.keras.models import load_model  # type: ignore
        from tensorflow.keras.utils import custom_object_scope  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "load_or_rebuild_geoprior_model requires TensorFlow. "
            "Please install 'tensorflow>=2.12' to use this helper."
        ) from e

    try:
        from fusionlab.nn.pinn.models import GeoPriorSubsNet  # type: ignore
        from fusionlab.params import (  # type: ignore
            LearnableMV,
            LearnableKappa,
            FixedGammaW,
            FixedHRef,
        )
        from fusionlab.nn.losses import make_weighted_pinball  # type: ignore
        from fusionlab.nn.keras_metrics import (  # type: ignore
            coverage80_fn,
            sharpness80_fn,
        )
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "load_or_rebuild_geoprior_model requires fusionlab components "
            "(GeoPriorSubsNet, LearnableMV, etc.). Ensure fusionlab is "
            "installed and importable."
        ) from e

    custom_objects = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        "make_weighted_pinball": make_weighted_pinball,
        "coverage80_fn": coverage80_fn,
        "sharpness80_fn": sharpness80_fn,
    }

    best_hps: dict | None = None

    # ------------------- 1) Try direct load_model -------------------------
    with custom_object_scope(custom_objects):
        if verbose:
            print(f"[Model] Attempting to load model from: {model_path}")

        try:
            model = load_model(model_path, compile=compile_on_load)
            if verbose:
                print(
                    f"[Model] Successfully loaded model for {label_city} "
                    f"from: {model_path}"
                )
            return model, best_hps
        except Exception as e_load:
            if verbose:
                print(
                    f"[Warn] load_model('{model_path}') failed: {e_load}\n"
                    "[Warn] Falling back to config-based reconstruction."
                )

    # ------------------- 2) Fallback: tuned HPs OR training summary -------
    try:
        # 2a) Tuned model path
        best_hps = load_best_hps_near_model(model_path)
        model = build_geoprior_from_hps(
            manifest=manifest,
            X_sample=X_sample,
            best_hps=best_hps,
            out_s_dim=out_s_dim,
            out_g_dim=out_g_dim,
            mode=mode,
            horizon=horizon,
            quantiles=quantiles,
        )
    except Exception as e_hps:
        # 2b) No best_hps JSON -> treat this as a plain trained model.
        if verbose:
            print(
                "[Fallback] No best_hps JSON found next to model_path; "
                "assuming a plain trained model.\n"
                f"          Reason: {e_hps}"
            )

        training_summary = load_training_summary_near_model(
            model_path, city_name=city_name
        )
        if training_summary is None:
            raise RuntimeError(
                "Failed to reconstruct GeoPriorSubsNet: neither tuned "
                "hyperparameters nor *_training_summary.json were found "
                f"near model_path={model_path!r}."
            ) from e_hps

        model = build_geoprior_from_training_summary(
            manifest=manifest,
            X_sample=X_sample,
            training_summary=training_summary,
            out_s_dim=out_s_dim,
            out_g_dim=out_g_dim,
            mode=mode,
            horizon=horizon,
            quantiles=quantiles,
        )

        # Build a minimal best_hps dict so compile_for_eval can recover the
        # training-time physics weights and learning rate.
        compile_block = training_summary.get("compile", {}) or {}
        phys = compile_block.get("physics_loss_weights", {}) or {}
        lr = compile_block.get("learning_rate", None)

        hps_from_train: dict[str, float] = {}
        if lr is not None:
            try:
                hps_from_train["learning_rate"] = float(lr)
            except Exception:
                pass
        for k, v in phys.items():
            try:
                hps_from_train[k] = float(v)
            except Exception:
                continue

        best_hps = hps_from_train or None

    # ------------------- 3) Load weights if checkpoint exists -------------
    weights_path = infer_best_weights_path(model_path)
    if weights_path is not None:
        try:
            model.load_weights(weights_path)
            if verbose:
                print(
                    "[Fallback] Loaded weights into reconstructed "
                    f"GeoPriorSubsNet from: {weights_path}"
                )
        except Exception as e_w:
            if verbose:
                print(
                    "[Warn] Could not load weights from checkpoint:\n"
                    f"       {weights_path}\n"
                    f"       Error: {e_w}\n"
                    "       The rebuilt model is using freshly-initialised "
                    "weights. Predictions will NOT match the original run."
                )
    else:
        if verbose:
            print(
                "[Warn] No weights checkpoint found near model.\n"
                "       Using rebuilt model with freshly-initialised "
                "weights. Predictions will NOT match the original run."
            )

    return model, best_hps


def build_geoprior_from_training_summary(
    manifest: dict,
    X_sample: dict,
    training_summary: dict,
    out_s_dim: int,
    out_g_dim: int,
    mode: str,
    horizon: int,
    quantiles: list[float] | None,
) -> Any:
    """
    Reconstruct a GeoPriorSubsNet from a training_summary JSON.

    This is the fallback path for plain *trained* models (no tuning),
    using the architecture recorded under ``hp_init['model_init_params']``.

    Parameters
    ----------
    manifest : dict
        Stage-1 manifest dictionary (for some defaults).

    X_sample : dict
        One NPZ inputs dictionary already passed through
        :func:`ensure_input_shapes`. Only shapes are used.

    training_summary : dict
        Parsed ``*_training_summary.json`` for this run.

    out_s_dim, out_g_dim, mode, horizon, quantiles :
        Same semantics as in :func:`build_geoprior_from_hps`.

    Returns
    -------
    model : GeoPriorSubsNet
        Reconstructed model (uncompiled).
    """
    try:
        from fusionlab.nn.pinn.models import GeoPriorSubsNet  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "build_geoprior_from_training_summary requires "
            "'fusionlab.nn.pinn.models.GeoPriorSubsNet'. "
            "Ensure fusionlab is installed and importable."
        ) from e

    cfg = manifest.get("config", {}) or {}

    # Infer input dims from the NPZ sample
    static_dim, dynamic_dim, future_dim = infer_input_dims_from_X(X_sample)

    hp_init = training_summary.get("hp_init", {}) or {}
    model_init = hp_init.get("model_init_params", {}) or {}

    # Quantiles: prefer explicit argument, then the training summary
    q = quantiles or hp_init.get("quantiles")

    # Attention stack
    attention_levels = model_init.get(
        "attention_levels",
        hp_init.get(
            "attention_levels",
            cfg.get("ATTENTION_LEVELS", ["cross", "hierarchical", "memory"]),
        ),
    )

    # Physics toggles
    censor_cfg = cfg.get("censoring", {}) or {}
    use_effective_h = bool(
        model_init.get(
            "use_effective_h",
            hp_init.get(
                "use_effective_h",
                censor_cfg.get("use_effective_h_field", True),
            ),
        )
    )
    pde_mode = hp_init.get("pde_mode", cfg.get("PDE_MODE_CONFIG", "both"))
    kappa_mode = model_init.get(
        "kappa_mode",
        cfg.get("GEOPRIOR_KAPPA_MODE", "bar"),
    )
    scale_pde_residuals = bool(
        model_init.get("scale_pde_residuals", True)
    )

    # Architecture hyperparameters (as used at training time)
    embed_dim = int(model_init.get("embed_dim", 32))
    hidden_units = int(model_init.get("hidden_units", 96))
    lstm_units = int(model_init.get("lstm_units", 96))
    attention_units = int(model_init.get("attention_units", 32))
    num_heads = int(model_init.get("num_heads", 4))
    dropout_rate = float(model_init.get("dropout_rate", 0.1))

    use_vsn = bool(
        model_init.get("use_vsn", hp_init.get("use_vsn", True))
    )
    vsn_units = int(
        model_init.get("vsn_units", hp_init.get("vsn_units", 32))
    )
    use_batch_norm = bool(
        model_init.get(
            "use_batch_norm",
            hp_init.get("use_batch_norm", cfg.get("USE_BATCH_NORM", True)),
        )
    )

    # Geomechanical parameters were serialised via `serialize_subs_params`
    # so we need to extract scalar initial values again.
    def _extract_initial(
        spec: Any, cfg_key: str, cfg_default: float
    ) -> float:
        if isinstance(spec, dict):
            if "initial_value" in spec:
                try:
                    return float(spec["initial_value"])
                except Exception:
                    pass
            if "value" in spec:
                try:
                    return float(spec["value"])
                except Exception:
                    pass
        if cfg_key in cfg and cfg[cfg_key] is not None:
            try:
                return float(cfg[cfg_key])
            except Exception:
                pass
        return float(cfg_default)

    mv_spec = model_init.get("mv", {})
    kappa_spec = model_init.get("kappa", {})

    mv_init = _extract_initial(mv_spec, "GEOPRIOR_INIT_MV", 5e-7)
    kappa_init = _extract_initial(kappa_spec, "GEOPRIOR_INIT_KAPPA", 1.0)

    # Pack the remaining architectural knobs into `architecture_config`
    known_keys = {
        "embed_dim",
        "hidden_units",
        "lstm_units",
        "attention_units",
        "num_heads",
        "dropout_rate",
        "use_vsn",
        "vsn_units",
        "use_batch_norm",
        "mv",
        "kappa",
        "gamma_w",
        "h_ref",
        "kappa_mode",
        "use_effective_h",
        "scale_pde_residuals",
        "attention_levels",
        "mode",
        "time_steps",
    }
    architecture_config = {
        k: v for k, v in model_init.items() if k not in known_keys
    }

    model = GeoPriorSubsNet(
        static_input_dim=static_dim,
        dynamic_input_dim=dynamic_dim,
        future_input_dim=future_dim,
        output_subsidence_dim=out_s_dim,
        output_gwl_dim=out_g_dim,
        forecast_horizon=horizon,
        mode=mode,
        attention_levels=attention_levels,
        quantiles=q,
        # physics switches
        pde_mode=pde_mode,
        scale_pde_residuals=scale_pde_residuals,
        kappa_mode=kappa_mode,
        use_effective_h=use_effective_h,
        # architecture hyperparameters
        embed_dim=embed_dim,
        hidden_units=hidden_units,
        lstm_units=lstm_units,
        attention_units=attention_units,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        use_vsn=use_vsn,
        vsn_units=vsn_units,
        use_batch_norm=use_batch_norm,
        # geomechanical priors
        mv=float(mv_init),
        kappa=float(kappa_init),
        architecture_config=architecture_config,
    )

    print(
        "[Fallback] Reconstructed GeoPriorSubsNet from training_summary with "
        f"static_dim={static_dim}, dynamic_dim={dynamic_dim}, "
        f"future_dim={future_dim}, horizon={horizon}, mode={mode}"
    )
    return model

def load_geoprior_for_inference(
    model_path: str,
    manifest: dict,
    X_sample: dict,
    out_s_dim: int,
    out_g_dim: int,
    mode: str,
    horizon: int,
    quantiles: list[float] | None,
    city_name: str | None = None,
    include_metrics: bool = True,
    verbose: int = 1,
):
    """
    Convenience wrapper: load (tuned or trained) GeoPriorSubsNet and
    compile it for evaluation/inference.

    Returns
    -------
    model :
        Compiled model ready for ``predict`` / diagnostics.

    info : dict
        Small dict with the ``best_hps`` (if any) and the resolved
        quantiles, useful for logging.
    """
    model, best_hps = load_or_rebuild_geoprior_model(
        model_path=model_path,
        manifest=manifest,
        X_sample=X_sample,
        out_s_dim=out_s_dim,
        out_g_dim=out_g_dim,
        mode=mode,
        horizon=horizon,
        quantiles=quantiles,
        city_name=city_name,
        compile_on_load=False,
        verbose=verbose,
    )

    model = compile_for_eval(
        model=model,
        manifest=manifest,
        best_hps=best_hps,
        quantiles=quantiles,
        include_metrics=include_metrics,
    )

    info = {
        "best_hps": best_hps,
        "quantiles": quantiles,
    }
    return model, info


def load_training_summary_near_model(
    model_path: str,
    city_name: str | None = None,
) -> dict | None:
    """
    Locate and load a ``*_training_summary.json`` next to a trained model.

    Strategy
    --------
    1. Prefer ``<city>_GeoPriorSubsNet_training_summary.json`` if
       ``city_name`` is given.
    2. Fallback: first file in the run directory that ends with
       ``'_training_summary.json'``.

    Parameters
    ----------
    model_path : str
        Path to the `.keras` archive or any checkpoint inside a
        ``train_YYYYMMDD-HHMMSS`` directory.

    city_name : str or None
        Optional city name to build a more specific candidate filename.

    Returns
    -------
    dict or None
        Parsed JSON dict if found and loadable, otherwise ``None``.
    """
    run_dir = os.path.dirname(os.path.abspath(model_path))
    candidates: list[str] = []

    if city_name:
        candidates.append(
            os.path.join(
                run_dir,
                f"{city_name}_GeoPriorSubsNet_training_summary.json",
            )
        )

    # Generic fallback: any *_training_summary.json in the run dir
    try:
        for fname in os.listdir(run_dir):
            if fname.endswith("_training_summary.json"):
                candidates.append(os.path.join(run_dir, fname))
    except FileNotFoundError:
        return None

    seen: set[str] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    ts = json.load(f)
                print(f"[TrainSummary] Loaded training_summary from: {path}")
                return ts
            except Exception as e:  # pragma: no cover - defensive
                print(
                    f"[Warn] Could not read training_summary JSON at {path!r}: {e}"
                )
                # Try the next candidate
                continue

    return None


def extract_preds(
    model: Any,
    out: Any,
    *,
    strict: bool = True,
    output_names: Sequence[str] | None = None,
) -> tuple[Any, Any]:
    r"""
    Extract (subs_pred, gwl_pred) from GeoPrior outputs.

    Supports:
      1) v3.2+ call(): {"subs_pred","gwl_pred"}
      2) forward_with_aux(): (y_pred, aux)
      3) legacy: {"data_final"} + model.split_data_predictions
      4) predict(): list/tuple mapped via output names

    If `strict=True`, list/tuple outputs *must* be mappable via
    output names; otherwise we raise to avoid silent swaps.
    

    This helper normalizes the output interface across two
    GeoPrior generation families:

    1. New interface (preferred)
       ``model(inputs) -> {"subs_pred": ..., "gwl_pred": ...}``

    2. Legacy interface (backward compatible)
       ``model(inputs) -> {"data_final": ...}``, where the caller
       must split the tensor using ``model.split_data_predictions``.

    Parameters
    ----------
    model : object
        A Keras-like model instance that may expose
        ``split_data_predictions(data_final)``.

        The splitter must return a tuple:

        - ``subs_pred`` with shape ``(B, H, 1)`` or ``(B, H, Q, 1)``
        - ``gwl_pred``  with shape ``(B, H, 1)`` or ``(B, H, Q, 1)``

    out : dict
        Output returned by the model call, typically
        ``model(inputs, training=False)``.

        Supported keys are either:

        - ``{"subs_pred", "gwl_pred"}`` (new interface), or
        - ``{"data_final"}`` (legacy interface).

    Returns
    -------
    subs_pred : Tensor
        Predicted subsidence in model space.

        Expected shapes:

        - Point mode: ``(B, H, 1)``
        - Quantile mode: ``(B, H, Q, 1)``

    gwl_pred : Tensor
        Predicted groundwater/head variable in model space.

        Expected shapes:

        - Point mode: ``(B, H, 1)``
        - Quantile mode: ``(B, H, Q, 1)``

    Raises
    ------
    KeyError
        If ``out`` does not contain a supported key set.
    TypeError
        If ``out`` is not a mapping/dict-like object.

    Notes
    -----
    This function is intended for Stage-2 and Stage-3
    scripts where you may load checkpoints from older
    experiments. It avoids fragile code that slices
    ``data_final`` manually.

    The function does not validate tensor dtypes or
    numerical finiteness. Upstream code should handle
    ``NaN`` and ``Inf`` checks as needed.

    Examples
    --------
    New interface::

        out = model_inf(xb, training=False)
        s_pred, h_pred = extract_stage_outputs(
            model_inf,
            out,
        )

    Legacy interface::

        out = model_inf(xb, training=False)
        s_pred, h_pred = extract_stage_outputs(
            model_inf,
            out,
        )

    See Also
    --------
    subs_point_from_stage_out :
        Convert subsidence predictions to a point forecast.

    References
    ----------
    .. [1] Chollet, F. et al. Keras: Deep Learning for Humans.
           (Software documentation).
    """
    # ---------------------------------------------------------
    # 0) forward_with_aux() style: (y_pred, aux)
    # ---------------------------------------------------------
    if isinstance(out, tuple) and len(out) == 2:
        y_pred, aux = out
        if isinstance(y_pred, Mapping):
            out = y_pred
        elif isinstance(aux, Mapping):
            # fallback: sometimes callers pass aux by mistake
            out = aux

    # ---------------------------------------------------------
    # 1) Mapping outputs
    # ---------------------------------------------------------
    if isinstance(out, Mapping):
        has_new = ("subs_pred" in out) and ("gwl_pred" in out)
        if has_new:
            return out["subs_pred"], out["gwl_pred"]

        # Legacy: data_final -> split
        if "data_final" in out and hasattr(
            model, "split_data_predictions"
        ):
            return model.split_data_predictions(out["data_final"])

        # Single-key wrapper: unwrap one level and retry
        if len(out) == 1:
            only_val = next(iter(out.values()))
            return extract_preds(
                model,
                only_val,
                strict=strict,
                output_names=output_names,
            )

        raise KeyError(
            "Unsupported model output keys. Expected "
            "{'subs_pred','gwl_pred'} or {'data_final'} "
            "or a single-key wrapper. "
            f"Got keys={list(out.keys())!r}."
        )

    # ---------------------------------------------------------
    # 2) predict() outputs as list/tuple
    # ---------------------------------------------------------
    if isinstance(out, (list, tuple)):
        names = None
        if output_names is not None:
            names = list(output_names)
        else:
            names = getattr(model, "output_names", None)

        if names and len(names) == len(out):
            mapped = dict(zip(names, out))
            return extract_preds(
                model,
                mapped,
                strict=strict,
                output_names=names,
            )

        if not strict and len(out) >= 2:
            # last-resort, opt-in only
            return out[0], out[1]

        raise TypeError(
            "Model output is a list/tuple but cannot be mapped "
            "to names. Provide `output_names=...` or set "
            "`strict=False` to assume order."
        )

    raise TypeError(
        "Expected `out` as Mapping, (y_pred, aux), "
        "or list/tuple. "
        f"Got type={type(out)!r}."
    )


def subs_point_from_out(model, out, quantiles=None, med_idx=None):
    r"""
    Convert model output into a subsidence point forecast.

    This helper produces a subsidence tensor shaped ``(B, H, 1)``
    in model space, regardless of whether the model emits
    quantiles or a point prediction.

    - If quantiles are present and the subsidence prediction
      is shaped ``(B, H, Q, 1)``, the function selects the
      median quantile slice.
    - Otherwise, it returns the point prediction directly.

    Parameters
    ----------
    model : object
        A Keras-like model instance passed to
        :func:`extract_stage_outputs`.

    out : dict
        Output returned by the model call.

        This can be either the new interface with keys
        ``"subs_pred"`` and ``"gwl_pred"``, or the legacy
        interface with key ``"data_final"``.

    quantiles : sequence of float or None, default=None
        Quantile levels used by the model, such as
        ``[0.1, 0.5, 0.9]``.

        If provided, the function may use it to interpret
        the rank-4 quantile output and select the median.

        If ``None``, quantile selection is disabled unless
        ``med_idx`` is explicitly provided and the tensor
        rank indicates quantiles.

    med_idx : int or None, default=None
        Index along the quantile axis to use as the
        "point" forecast when quantiles are available.

        If ``None`` and ``quantiles`` is provided, the
        function selects the index closest to ``0.5``.

    Returns
    -------
    subs_point : Tensor
        Subsidence point prediction in model space with
        shape ``(B, H, 1)``.

    Raises
    ------
    ValueError
        If subsidence prediction is missing or ``None``.
    ValueError
        If a quantile tensor is detected but a valid
        median index cannot be resolved.

    Notes
    -----
    Quantile outputs are assumed to be shaped
    ``(B, H, Q, 1)`` where the quantile axis is the
    third dimension (axis=2).

    If the model returns point predictions already,
    the function is effectively a no-op.

    Examples
    --------
    Quantile model::

        out = model_inf(xb, training=False)
        s_point = subs_point_from_stage_out(
            model_inf,
            out,
            quantiles=[0.1, 0.5, 0.9],
        )

    Point model::

        out = model_inf(xb, training=False)
        s_point = subs_point_from_stage_out(
            model_inf,
            out,
        )

    See Also
    --------
    extract_stage_outputs :
        Normalize outputs across new and legacy checkpoints.

    References
    ----------
    .. [1] Koenker, R. and Bassett, G. Regression Quantiles.
           Econometrica, 1978.
    """
    subs_pred, _ = extract_preds(model, out)

    if subs_pred is None:
        raise ValueError(
            "Model output 'subs_pred' is None."
        )

    # has_rank = hasattr(subs_pred, "shape") and (
    #     getattr(subs_pred.shape, "rank", None) is not None
    # )
    # is_quantile_tensor = has_rank and (subs_pred.shape.rank == 4)
    rank = None
    if hasattr(subs_pred, "shape"):
        rank = getattr(subs_pred.shape, "rank", None)  # TF tensors
        if rank is None:
            try:
                rank = len(subs_pred.shape)            # NumPy arrays / tuples
            except Exception:
                rank = None
    
    is_quantile_tensor = (rank == 4)

    if not is_quantile_tensor:
        return subs_pred

    if med_idx is None:
        if not quantiles:
            raise ValueError(
                "Quantile tensor detected but `med_idx` "
                "is None and `quantiles` is not provided."
            )

        q = np.asarray(quantiles, dtype=float)
        med_idx = int(np.argmin(np.abs(q - 0.5)))

    if med_idx is None or int(med_idx) < 0:
        raise ValueError(
            "Invalid `med_idx` resolved for quantiles."
        )

    # return subs_pred[..., int(med_idx), :]
    # Quantile outputs assumed (B, H, Q, 1)
    return subs_pred[:, :, int(med_idx), :]

def _extract_allowed_hps(
    obj: object,
    *,
    allowed: set[str],
) -> dict:
    out: dict = {}

    def _walk(x: object) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k in allowed:
                    out[k] = v
                _walk(v)
        elif isinstance(x, (list, tuple)):
            for it in x:
                _walk(it)

    _walk(obj)
    return {k: out[k] for k in allowed if k in out}

def load_tuned_hps_near_model(
    model_path: str,
    *,
    prefer: str = "keras",
    required: bool = True,
    log_fn=None,
) -> dict:

    log = log_fn if callable(log_fn) else None

    def _msg(s: str) -> None:
        if log is not None:
            log(s)

    def _load_json(p: str) -> dict:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    mp = os.path.abspath(str(model_path))
    run_dir = mp if os.path.isdir(mp) else os.path.dirname(mp)
    base = "" if os.path.isdir(mp) else os.path.basename(mp)

    stem = None
    if prefer == "keras":
        if base.endswith("_best.keras"):
            stem = base[: -len("_best.keras")]
        elif base.endswith(".keras"):
            stem = base[: -len(".keras")]
    else:
        if base.endswith("_best.weights.h5"):
            stem = base[: -len("_best.weights.h5")]
        elif base.endswith(".weights.h5"):
            stem = base[: -len(".weights.h5")]

    cands: list[str] = []
    if stem:
        cands.append(os.path.join(run_dir, stem + "_best_hps.json"))
    cands.append(os.path.join(run_dir, "tuning_summary.json"))
    cands.extend(glob.glob(os.path.join(run_dir, "*tuning_summary*.json")))

    for p in cands:
        if not os.path.exists(p):
            continue
        data = _load_json(p)
        hps = data.get("best_hps") or data.get("hps") or {}
        if isinstance(hps, dict) and hps:
            _msg(f"[HP] tuned: {p}")
            return hps

    if required:
        raise FileNotFoundError(
            "No tuned hyperparameters found near:\n"
            f"  model_path={model_path!r}\n"
            f"  run_dir={run_dir!r}\n"
        )
    return {}

def load_trained_hps_near_model(
    model_path: str,
    *,
    allowed: set[str],
    required: bool = False,
    log_fn=None,
) -> dict:

    log = log_fn if callable(log_fn) else None

    def _msg(s: str) -> None:
        if log is not None:
            log(s)

    def _load_json(p: str) -> object:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    mp = os.path.abspath(str(model_path))
    run_dir = mp if os.path.isdir(mp) else os.path.dirname(mp)

    pats = [
        os.path.join(run_dir, "model_init_manifest.json"),
        os.path.join(run_dir, "*training_summary*.json"),
        os.path.join(run_dir, "*architecture*.json"),
    ]

    files: list[str] = []
    for pat in pats:
        files.extend(glob.glob(pat))

    for p in files:
        if not os.path.exists(p):
            continue
        data = _load_json(p)
        hps = _extract_allowed_hps(data, allowed=allowed)
        if hps:
            _msg(f"[HP] trained: {p}")
            return hps

    if required:
        raise FileNotFoundError(
            "No trained hyperparameters found near:\n"
            f"  model_path={model_path!r}\n"
            f"  run_dir={run_dir!r}\n"
        )
    return {}

def load_hps_auto_near_model(
    model_path: str,
    *,
    allowed: set[str],
    prefer: str = "keras",
    required: bool = False,
    log_fn=None,
) -> dict:

    mp = os.path.abspath(str(model_path))
    run_dir = mp if os.path.isdir(mp) else os.path.dirname(mp)

    tuned_hits = []
    tuned_hits.extend(
        glob.glob(os.path.join(run_dir, "*_best_hps.json"))
    )
    tuned_hits.extend(
        glob.glob(os.path.join(run_dir, "*tuning_summary*.json"))
    )
    is_tuned = bool(tuned_hits) or ("tuning" in run_dir)

    if is_tuned:
        return load_tuned_hps_near_model(
            model_path,
            prefer=prefer,
            required=required,
            log_fn=log_fn,
        )

    return load_trained_hps_near_model(
        model_path,
        allowed=allowed,
        required=required,
        log_fn=log_fn,
    )

def load_best_hps_near_model(
    model_path: str,
    *,
    model_name: str | None = "GeoPriorSubsNet",
    prefer: str = "keras",
    log_fn=None,
) -> dict:
    """
    Load best hyperparameters saved near a model artifact.

    Supports model names like:
    <city>_<model_name>_H{H}_best.keras
    <city>_<model_name>_H{H}_best.weights.h5

    Parameters
    ----------
    model_path : str
        Path to a model file or its run directory.
    model_name : str or None, default="GeoPriorSubsNet"
        Model name token in filenames.
    prefer : {"keras", "weights"}, default="keras"
        Which artifact type to infer the prefix from.
    log_fn : callable or None, default=None
        Logger (e.g. print). None disables logs.

    Returns
    -------
    best_hps : dict
        Non-empty hyperparameter dictionary.

    Raises
    ------
    FileNotFoundError
        If no hyperparameter JSON is found.
    ValueError
        If a candidate JSON exists but is empty/invalid.
    """

    log = log_fn if callable(log_fn) else None

    if prefer not in ("keras", "weights"):
        raise ValueError(
            "prefer must be 'keras' or 'weights'."
        )

    mp = os.path.abspath(str(model_path))
    run_dir = mp if os.path.isdir(mp) else os.path.dirname(mp)
    base = "" if os.path.isdir(mp) else os.path.basename(mp)

    def _msg(s: str) -> None:
        if log is not None:
            log(s)

    def _load_json(p: str) -> dict:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _newest(paths: list[str]) -> str | None:
        c = []
        for p in paths:
            try:
                c.append((os.path.getmtime(p), p))
            except Exception:
                pass
        if not c:
            return None
        c.sort(reverse=True)
        return c[0][1]

    # -------------------------------------------------
    # If a directory is provided, infer "base" by scan.
    # -------------------------------------------------
    if not base:
        pats = []
        if prefer == "keras":
            if model_name:
                pats.append(
                    os.path.join(
                        run_dir,
                        f"*_{model_name}_H*_best.keras",
                    )
                )
            pats.append(
                os.path.join(run_dir, "*_H*_best.keras")
            )
            pats.append(
                os.path.join(run_dir, "*_best.keras")
            )
        else:
            if model_name:
                pats.append(
                    os.path.join(
                        run_dir,
                        f"*_{model_name}_H*_best.weights.h5",
                    )
                )
            pats.append(
                os.path.join(
                    run_dir,
                    "*_H*_best.weights.h5",
                )
            )
            pats.append(
                os.path.join(run_dir, "*_best.weights.h5")
            )

        hits = []
        for pat in pats:
            hits.extend(glob.glob(pat))
        best = _newest(hits)
        if best:
            base = os.path.basename(best)

    # -------------------------------------------------
    # Infer stem/prefix from the artifact filename.
    # -------------------------------------------------
    stem = None
    if prefer == "keras":
        if base.endswith("_best.keras"):
            stem = base[: -len("_best.keras")]
        elif base.endswith(".keras"):
            stem = base[: -len(".keras")]
    else:
        if base.endswith("_best.weights.h5"):
            stem = base[: -len("_best.weights.h5")]
        elif base.endswith(".weights.h5"):
            stem = base[: -len(".weights.h5")]

    # Try to parse city / model / horizon from stem.
    city = None
    mname = None
    horizon = None
    city_model = None

    if stem:
        left = stem
        if "_H" in left:
            a, b = left.rsplit("_H", 1)
            digs = []
            for ch in b:
                if ch.isdigit():
                    digs.append(ch)
                else:
                    break
            if digs:
                horizon = int("".join(digs))
            left = a

        city_model = left

        if model_name:
            tok = "_" + str(model_name)
            if city_model.endswith(tok):
                city = city_model[: -len(tok)]
                mname = str(model_name)

        if city is None:
            parts = city_model.split("_")
            if len(parts) >= 2:
                city = parts[0]
                mname = "_".join(parts[1:])

    # -------------------------------------------------
    # 1) Near-model explicit JSONs.
    # -------------------------------------------------
    cands = []
    if stem:
        cands.append(
            os.path.join(run_dir, stem + "_best_hps.json")
        )

    if city and mname:
        cands.append(
            os.path.join(
                run_dir,
                f"{city}_{mname}_best_hps.json",
            )
        )
        if horizon is not None:
            cands.append(
                os.path.join(
                    run_dir,
                    f"{city}_{mname}_H{horizon}_best_hps.json",
                )
            )

    if city:
        cands.append(
            os.path.join(
                run_dir,
                f"{city}_{mname}_best_hps.json",
            )
        )

    seen = set()
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            best_hps = _load_json(p)
            if isinstance(best_hps, dict) and best_hps:
                _msg(f"[HP] Loaded best_hps: {p}")
                return best_hps
            raise ValueError(
                f"{p!r} exists but is empty/invalid."
            )

    # -------------------------------------------------
    # 2) tuning summaries.
    # -------------------------------------------------
    sum_pats = [
        os.path.join(run_dir, "tuning_summary.json"),
        os.path.join(run_dir, "*tuning_summary*.json"),
    ]
    sums = []
    for pat in sum_pats:
        sums.extend(glob.glob(pat))
    for p in sums:
        if not os.path.exists(p):
            continue
        s = _load_json(p)
        best_hps = s.get("best_hps") or s.get("hps") or {}
        if isinstance(best_hps, dict) and best_hps:
            _msg(f"[HP] Loaded best_hps: {p}")
            return best_hps

    # -------------------------------------------------
    # 3) training summaries.
    # -------------------------------------------------
    for p in glob.glob(
        os.path.join(run_dir, "*training_summary*.json")
    ):
        s = _load_json(p)
        best_hps = (
            s.get("best_hps")
            or s.get("hps")
            or s.get("params")
            or {}
        )
        if isinstance(best_hps, dict) and best_hps:
            _msg(f"[HP] Loaded best_hps: {p}")
            return best_hps

    # -------------------------------------------------
    # 4) architecture dumps.
    # -------------------------------------------------
    for p in glob.glob(
        os.path.join(run_dir, "*architecture*.json")
    ):
        a = _load_json(p)
        best_hps = (
            a.get("best_hps")
            or a.get("hps")
            or a.get("params")
            or {}
        )
        if isinstance(best_hps, dict) and best_hps:
            _msg(f"[HP] Loaded best_hps: {p}")
            return best_hps

    raise FileNotFoundError(
        "Could not find best hyperparameters near:\n"
        f"  model_path={model_path!r}\n"
        f"  run_dir={run_dir!r}\n"
        f"  prefer={prefer!r}\n"
        "Looked for *_best_hps.json + summaries."
    )

def coerce_quantile_weights(
    d: dict | None,
    default: dict,
) -> dict:
    """
    Normalize a quantile-weight mapping to have float keys and float values.

    This helper is useful when reading JSON configs where the quantile
    keys are stored as strings (e.g. ``{'0.1': 3.0, '0.5': 1.0}``).

    Parameters
    ----------
    d : dict or None
        Original dictionary mapping quantile-like keys (str or float) to
        numeric weights. If ``None`` or empty, ``default`` is returned.

    default : dict
        Fallback dictionary to use when ``d`` is ``None`` or empty.

    Returns
    -------
    out : dict
        Dictionary with the same keys and values, but with:

        - keys coerced to float when possible (otherwise left as-is),
        - values coerced to ``float``.
    """
    if not d:
        return default

    out: dict[Any, float] = {}
    for k, v in d.items():
        try:
            q = float(k)
        except (TypeError, ValueError):
            # Non-numeric key (rare): keep as-is
            q = k
        out[q] = float(v)
    return out

def compile_for_eval(
    model: Any,
    manifest: dict,
    best_hps: dict | None,
    quantiles: list[float] | None,
    *,
    include_metrics: bool = True,
) -> Any:
    """
    Recompile a GeoPriorSubsNet instance for evaluation / diagnostics.

    This is intended for:
    - tuned models loaded from a `.keras` archive, or
    - models rebuilt from best_hps.

    It does NOT change the architecture or weights, only the compile
    configuration (optimizer, losses, and physics loss weights).

    Parameters
    ----------
    model : GeoPriorSubsNet
        Loaded or freshly-built GeoPriorSubsNet instance.
    manifest : dict
        Stage-1 manifest; training config is taken from
        ``manifest['config']``.
    best_hps : dict or None
        Dictionary of tuned hyperparameters. If empty/None, reasonable
        defaults are inferred from the manifest.
    quantiles : list of float or None
        Quantiles used for probabilistic subsidence/GWL outputs.
    include_metrics : bool, default=True
        If True, attach MAE/MSE + coverage/sharpness metrics to match
        the training script; if False, only losses are configured.

    Returns
    -------
    model :
        The same model instance, compiled in-place.
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to compile GeoPriorSubsNet. "
            "Install `tensorflow>=2.12` to use "
            "`compile_geoprior_for_eval`."
        )

    # Local imports so nat_utils.py itself stays lightweight
    from fusionlab.nn.losses import make_weighted_pinball
    if include_metrics:
        from fusionlab.nn.keras_metrics import coverage80_fn, sharpness80_fn

    cfg = manifest.get("config", {}) or {}
    best_hps = best_hps or {}

    # ---- 1. Data loss weights / quantile weights -------------------------
    subs_raw = cfg.get(
        "SUBS_WEIGHTS",
        {0.1: 3.0, 0.5: 1.0, 0.9: 3.0},
    )
    gwl_raw = cfg.get(
        "GWL_WEIGHTS",
        {0.1: 1.5, 0.5: 1.0, 0.9: 1.5},
    )

    subs_w = _coerce_quantile_weights(
        subs_raw, {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
    )
    gwl_w = _coerce_quantile_weights(
        gwl_raw, {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}
    )

    if quantiles:
        loss_dict = {
            "subs_pred": make_weighted_pinball(quantiles, subs_w),
            "gwl_pred": make_weighted_pinball(quantiles, gwl_w),
        }
    else:
        mse = tf.keras.losses.MeanSquaredError()
        loss_dict = {"subs_pred": mse, "gwl_pred": mse}

    loss_weights = {"subs_pred": 1.0, "gwl_pred": 0.5}

    # ---- 2. Physics weights: prefer best_hps, fall back to config --------
    def _hp_or_cfg(hp_key: str, cfg_key: str, default: float) -> float:
        if hp_key in best_hps and best_hps[hp_key] is not None:
            return float(best_hps[hp_key])
        if cfg_key in cfg and cfg[cfg_key] is not None:
            return float(cfg[cfg_key])
        return float(default)

    lr = _hp_or_cfg("learning_rate", "LEARNING_RATE", 1e-4)

    physics_kwargs = {
        "lambda_gw": _hp_or_cfg("lambda_gw", "LAMBDA_GW", 1.0),
        "lambda_cons": _hp_or_cfg("lambda_cons", "LAMBDA_CONS", 1.0),
        "lambda_prior": _hp_or_cfg("lambda_prior", "LAMBDA_PRIOR", 0.1),
        "lambda_smooth": _hp_or_cfg("lambda_smooth", "LAMBDA_SMOOTH", 0.01),
        "lambda_mv": _hp_or_cfg("lambda_mv", "LAMBDA_MV", 0.0),
        "mv_lr_mult": _hp_or_cfg("mv_lr_mult", "MV_LR_MULT", 1.0),
        "kappa_lr_mult": _hp_or_cfg(
            "kappa_lr_mult", "KAPPA_LR_MULT", 1.0
        ),
    }

    compile_kwargs: dict[str, Any] = {
        "optimizer": Adam(learning_rate=lr),
        "loss": loss_dict,
        "loss_weights": loss_weights,
        **physics_kwargs,
    }

    if include_metrics:
        metrics_dict = {
            "subs_pred": ["mae", "mse"]
            + ([coverage80_fn, sharpness80_fn] if quantiles else []),
            "gwl_pred": ["mae", "mse"],
        }
        compile_kwargs["metrics"] = metrics_dict

    model.compile(**compile_kwargs)
    return model

def compile_geoprior_for_eval(
    model: Any,  # type: ignore[override]
    manifest: dict,
    best_hps: dict,
    quantiles: list[float] | None,
) -> Any:
    """
    (Re)compile a GeoPriorSubsNet-like model for evaluation.

    This helper uses the Stage-1 manifest and tuned hyperparameters to
    configure:

    - the pinball losses for subsidence and GWL outputs,
    - loss weights for the two heads,
    - physics loss weights (lambda_*),
    - learning rate and LR multipliers.

    TensorFlow and fusionlab are imported lazily inside this function so
    that ``nat_utils`` can be imported even in non-TF environments.

    Parameters
    ----------
    model : GeoPriorSubsNet-like
        An instance of the GeoPriorSubsNet model (or any model exposing
        the same compile signature).

    manifest : dict
        Stage-1 manifest dictionary. The ``config`` entry is used to
        retrieve default loss weights and physics settings.

    best_hps : dict
        Hyperparameters loaded from the tuning run
        (e.g. via :func:`load_best_hps_near_model`).

    quantiles : list of float or None
        Quantile levels used for probabilistic outputs. If ``None``,
        mean-squared error is used instead of pinball loss.

    Returns
    -------
    model
        The same model instance, compiled in-place.

    Raises
    ------
    ImportError
        If TensorFlow or fusionlab's ``make_weighted_pinball`` cannot be
        imported.
    """
    cfg = manifest.get("config", {}) or {}

    # Lazy imports so nat_utils.py is importable without TensorFlow
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.optimizers import Adam  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "compile_geoprior_for_eval requires TensorFlow. "
            "Please install 'tensorflow>=2.12' to use this helper."
        ) from e

    try:
        from fusionlab.nn.losses import make_weighted_pinball  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "compile_geoprior_for_eval requires "
            "'fusionlab.nn.losses.make_weighted_pinball'. "
            "Ensure fusionlab is installed and importable."
        ) from e

    # Base loss weights between subsidence and GWL heads
    loss_weights = {"subs_pred": 1.0, "gwl_pred": 0.5}

    # Quantile-specific weights from config (with robust defaults)
    subs_raw = cfg.get("SUBS_WEIGHTS", {0.1: 3.0, 0.5: 1.0, 0.9: 3.0})
    gwl_raw = cfg.get("GWL_WEIGHTS", {0.1: 1.5, 0.5: 1.0, 0.9: 1.5})

    subs_weights = coerce_quantile_weights(
        subs_raw, {0.1: 3.0, 0.5: 1.0, 0.9: 3.0}
    )
    gwl_weights = coerce_quantile_weights(
        gwl_raw, {0.1: 1.5, 0.5: 1.0, 0.9: 1.5}
    )

    if quantiles:
        loss_subs = make_weighted_pinball(quantiles, subs_weights)
        loss_gwl = make_weighted_pinball(quantiles, gwl_weights)
    else:
        loss_subs = tf.keras.losses.MSE
        loss_gwl = tf.keras.losses.MSE

    loss_dict = {"subs_pred": loss_subs, "gwl_pred": loss_gwl}

    # Learning rate: tuned value or config fallback
    lr_default = cfg.get("LEARNING_RATE", 5e-5)
    lr = float(best_hps.get("learning_rate", lr_default))
    optimizer = Adam(learning_rate=lr)

    # Physics loss weights and LR multipliers
    lambda_gw = float(best_hps.get("lambda_gw", cfg.get("LAMBDA_GW", 1.0)))
    lambda_cons = float(best_hps.get("lambda_cons", cfg.get("LAMBDA_CONS", 1.0)))
    lambda_prior = float(best_hps.get("lambda_prior", cfg.get("LAMBDA_PRIOR", 1.0)))
    lambda_smooth = float(
        best_hps.get("lambda_smooth", cfg.get("LAMBDA_SMOOTH", 1.0))
    )
    lambda_mv = float(best_hps.get("lambda_mv", cfg.get("LAMBDA_MV", 0.0)))
    mv_lr_mult = float(best_hps.get("mv_lr_mult", cfg.get("MV_LR_MULT", 1.0)))
    kappa_lr_mult = float(
        best_hps.get("kappa_lr_mult", cfg.get("KAPPA_LR_MULT", 1.0))
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        loss_weights=loss_weights,
        # physics loss weights + LR multipliers
        lambda_gw=lambda_gw,
        lambda_cons=lambda_cons,
        lambda_prior=lambda_prior,
        lambda_smooth=lambda_smooth,
        lambda_mv=lambda_mv,
        mv_lr_mult=mv_lr_mult,
        kappa_lr_mult=kappa_lr_mult,
    )
    return model


def build_geoprior_from_hps(
    manifest: dict,
    X_sample: dict,
    best_hps: dict,
    out_s_dim: int,
    out_g_dim: int,
    mode: str,
    horizon: int,
    quantiles: list[float] | None,
) -> Any:
    """
    Reconstruct a GeoPriorSubsNet from Stage-1 metadata + tuned HPs.

    This function is primarily intended as a **robust fallback** when
    ``tf.keras.models.load_model`` cannot deserialize a tuned model.
    It reconstructs the network geometry and physics settings from:

    - Stage-1 ``manifest['config']`` (for fixed architecture / physics),
    - tuned hyperparameters (for variable architecture / physics),
    - the Stage-1 input NPZ (for input dimensions).

    Parameters
    ----------
    manifest : dict
        Stage-1 manifest dictionary.

    X_sample : dict
        Inputs NPZ dictionary (already sanitized and passed through
        :func:`ensure_input_shapes`). Only shapes are used.

    best_hps : dict
        Hyperparameters loaded via :func:`load_best_hps_near_model`.

    out_s_dim : int
        Output dimension for the subsidence head.

    out_g_dim : int
        Output dimension for the GWL head.

    mode : str
        Sequence mode, e.g. ``"tft_like"`` or ``"pihal_like"``.

    horizon : int
        Forecast horizon (number of time steps).

    quantiles : list of float or None
        Quantile levels for probabilistic outputs.

    Returns
    -------
    model : GeoPriorSubsNet
        A freshly instantiated and compiled GeoPriorSubsNet instance.

    Raises
    ------
    ImportError
        If GeoPriorSubsNet cannot be imported from fusionlab.
    """
    try:
        from fusionlab.nn.pinn.models import GeoPriorSubsNet  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "build_geoprior_from_hps requires "
            "'fusionlab.nn.pinn.models.GeoPriorSubsNet'. "
            "Ensure fusionlab is installed and importable."
        ) from e

    cfg = manifest.get("config", {}) or {}

    # Infer input dims directly from NPZ
    static_dim, dynamic_dim, future_dim = infer_input_dims_from_X(X_sample)

    # Attention stack: fall back to a sensible default if not present
    attention_levels = cfg.get(
        "ATTENTION_LEVELS",
        ["cross", "hierarchical", "memory"],
    )

    # Whether we used effective H during Stage-2
    censor_cfg = cfg.get("censoring", {}) or {}
    use_effective_h = censor_cfg.get("use_effective_h_field", True)

    # Feature-processing mode controlled by tuned HPs
    use_vsn = bool(best_hps.get("use_vsn", True))
    feature_processing = "vsn" if use_vsn else "dense"

    architecture_config = {
        "encoder_type": "hybrid",
        "decoder_attention_stack": attention_levels,
        "feature_processing": feature_processing,
    }

    # Instantiate the model core with tuned settings
    model = GeoPriorSubsNet(
        static_input_dim=static_dim,
        dynamic_input_dim=dynamic_dim,
        future_input_dim=future_dim,
        output_subsidence_dim=out_s_dim,
        output_gwl_dim=out_g_dim,
        forecast_horizon=horizon,
        mode=mode,
        attention_levels=attention_levels,
        quantiles=quantiles,
        # physics switches from best_hps
        pde_mode=best_hps.get("pde_mode", "both"),
        scale_pde_residuals=bool(best_hps.get("scale_pde_residuals", True)),
        kappa_mode=best_hps.get("kappa_mode", "bar"),
        use_effective_h=use_effective_h,
        # architecture hyperparameters
        embed_dim=int(best_hps.get("embed_dim", 32)),
        hidden_units=int(best_hps.get("hidden_units", 96)),
        lstm_units=int(best_hps.get("lstm_units", 96)),
        attention_units=int(best_hps.get("attention_units", 32)),
        num_heads=int(best_hps.get("num_heads", 4)),
        dropout_rate=float(best_hps.get("dropout_rate", 0.1)),
        use_vsn=use_vsn,
        vsn_units=int(best_hps.get("vsn_units", 32)),
        use_batch_norm=bool(best_hps.get("use_batch_norm", True)),
        # geomechanical priors (floats interpreted internally by the model)
        mv=float(best_hps.get("mv", 5e-7)),
        kappa=float(best_hps.get("kappa", 1.0)),
        architecture_config=architecture_config,
    )

    # Compile using the shared helper (losses + physics weights)
    compile_geoprior_for_eval(
        model=model,
        manifest=manifest,
        best_hps=best_hps,
        quantiles=quantiles,
    )

    print(
        "[Fallback] Reconstructed GeoPriorSubsNet from best_hps with "
        f"static_dim={static_dim}, dynamic_dim={dynamic_dim}, "
        f"future_dim={future_dim}, horizon={horizon}, mode={mode}"
    )
    return model

def build_geoprior_from_cfg(
    manifest: dict,
    X_sample: dict,
    out_s_dim: int,
    out_g_dim: int,
    mode: str,
    horizon: int,
    quantiles: list[float] | None,
) -> Any:
    """
    Reconstruct a GeoPriorSubsNet from the NATCOM config only.

    This is intended as a fallback for *trained* models (no tuning JSON)
    or when no best_hps can be found next to `model_path`.

    Parameters
    ----------
    manifest : dict
        Stage-1 manifest dictionary. The ``"config"`` entry is used as
        the single source of truth for architecture + physics settings.
    X_sample : dict
        NPZ inputs dict (already sanitised and passed through
        :func:`ensure_input_shapes` or equivalent). Only shapes are used.
    out_s_dim, out_g_dim : int
        Output dims for subsidence and GWL heads.
    mode : str
        Sequence mode, e.g. ``"tft_like"`` or ``"pihal_like"``.
    horizon : int
        Forecast horizon (number of time steps).
    quantiles : list of float or None
        Quantiles for probabilistic outputs.

    Returns
    -------
    model : GeoPriorSubsNet
        Compiled model instance ready for prediction/eval.
    """
    try:
        from fusionlab.nn.pinn.models import GeoPriorSubsNet  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "build_geoprior_from_cfg requires "
            "'fusionlab.nn.pinn.models.GeoPriorSubsNet'. "
            "Ensure fusionlab is installed and importable."
        ) from e

    cfg = manifest.get("config", {}) or {}

    # --- Input dims inferred from X_sample ----------------------------
    static_dim, dynamic_dim, future_dim = infer_input_dims_from_X(X_sample)

    # --- Attention stack / effective-H flag ---------------------------
    attention_levels = cfg.get(
        "ATTENTION_LEVELS",
        ["cross", "hierarchical", "memory"],
    )

    censor_cfg = cfg.get("censoring", {}) or {}
    use_effective_h = censor_cfg.get(
        "use_effective_h_field",
        bool(cfg.get("GEOPRIOR_USE_EFFECTIVE_H", True)),
    )

    # --- Physics switches ---------------------------------------------
    pde_mode = cfg.get("PDE_MODE_CONFIG", cfg.get("PDE_MODE", "both"))
    scale_pde_residuals = bool(
        cfg.get("SCALE_PDE_RESIDUALS", cfg.get("SCALE_PDE_RES", True))
    )
    kappa_mode = cfg.get(
        "GEOPRIOR_KAPPA_MODE",
        cfg.get("KAPPA_MODE", "bar"),
    )

    # --- Small helpers to read ints/floats/bools from cfg ------------
    def _cfg_int(default: int, *keys: str) -> int:
        for k in keys:
            if k in cfg and cfg[k] is not None:
                try:
                    return int(cfg[k])
                except Exception:
                    pass
        return int(default)

    def _cfg_float(default: float, *keys: str) -> float:
        for k in keys:
            if k in cfg and cfg[k] is not None:
                try:
                    return float(cfg[k])
                except Exception:
                    pass
        return float(default)

    def _cfg_bool(default: bool, *keys: str) -> bool:
        for k in keys:
            if k in cfg and cfg[k] is not None:
                return bool(cfg[k])
        return bool(default)

    # --- Architecture hyperparams (config-side defaults) --------------
    embed_dim = _cfg_int(32, "EMBED_DIM", "GEOPRIOR_EMBED_DIM")
    hidden_units = _cfg_int(96, "HIDDEN_UNITS", "GEOPRIOR_HIDDEN_UNITS")
    lstm_units = _cfg_int(96, "LSTM_UNITS", "GEOPRIOR_LSTM_UNITS")
    attention_units = _cfg_int(
        32, "ATTENTION_UNITS", "GEOPRIOR_ATTENTION_UNITS"
    )
    num_heads = _cfg_int(
        4, "NUM_HEADS", "NUMBER_HEADS", "GEOPRIOR_NUM_HEADS"
    )
    dropout_rate = _cfg_float(
        0.1, "DROPOUT_RATE", "GEOPRIOR_DROPOUT_RATE"
    )
    use_vsn = _cfg_bool(True, "USE_VSN", "GEOPRIOR_USE_VSN")
    vsn_units = _cfg_int(32, "VSN_UNITS", "GEOPRIOR_VSN_UNITS")
    use_batch_norm = _cfg_bool(
        True, "USE_BATCH_NORM", "GEOPRIOR_USE_BATCH_NORM"
    )

    # --- Geomechanical priors (Terzaghi-ish) --------------------------
    mv = _cfg_float(5e-7, "GEOPRIOR_INIT_MV")
    kappa = _cfg_float(1.0, "GEOPRIOR_INIT_KAPPA")

    architecture_config = {
        "encoder_type": "hybrid",
        "decoder_attention_stack": attention_levels,
        "feature_processing": "vsn" if use_vsn else "dense",
    }

    model = GeoPriorSubsNet(
        static_input_dim=static_dim,
        dynamic_input_dim=dynamic_dim,
        future_input_dim=future_dim,
        output_subsidence_dim=out_s_dim,
        output_gwl_dim=out_g_dim,
        forecast_horizon=horizon,
        mode=mode,
        attention_levels=attention_levels,
        quantiles=quantiles,
        # physics switches
        pde_mode=pde_mode,
        scale_pde_residuals=scale_pde_residuals,
        kappa_mode=kappa_mode,
        use_effective_h=use_effective_h,
        # architecture hyperparameters
        embed_dim=embed_dim,
        hidden_units=hidden_units,
        lstm_units=lstm_units,
        attention_units=attention_units,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        use_vsn=use_vsn,
        vsn_units=vsn_units,
        use_batch_norm=use_batch_norm,
        # priors
        mv=mv,
        kappa=kappa,
        architecture_config=architecture_config,
    )

    # Compile using config-only settings
    compile_for_eval(
        model=model,
        manifest=manifest,
        best_hps=None,
        quantiles=quantiles,
        include_metrics=True,
    )

    print(
        "[Fallback] Reconstructed GeoPriorSubsNet from manifest config with "
        f"static_dim={static_dim}, dynamic_dim={dynamic_dim}, "
        f"future_dim={future_dim}, horizon={horizon}, mode={mode}"
    )
    return model

def infer_best_weights_path(model_path: str) -> str | None:
    """
    Infer the best-weights checkpoint path for a tuned GeoPrior model.

    Strategy
    --------
    1. Look for ``tuning_summary.json`` in the same folder as
       ``model_path`` and return the stored ``\"best_weights_path\"``
       if it exists and the file is present on disk.
    2. Fallback: replace the ``.keras`` suffix of ``model_path`` by
       ``.weights.h5``, assuming the convention::

           <CITY>_GeoPrior_best.keras
           -> <CITY>_GeoPrior_best.weights.h5

    Parameters
    ----------
    model_path : str
        Path to the tuned model archive (usually ``.keras``).

    Returns
    -------
    weights_path : str or None
        Absolute path to the weights file if found, otherwise ``None``.
    """
    run_dir = os.path.dirname(os.path.abspath(model_path))

    # 1) Preferred: tuning_summary.json
    summary_path = os.path.join(run_dir, "tuning_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            w = summary.get("best_weights_path")
            if w and os.path.exists(w):
                return w
        except Exception as e:  # pragma: no cover - defensive
            print(f"[Warn] Could not read tuning_summary.json for weights: {e}")

    # 2) Name-based guess from the .keras path
    root, ext = os.path.splitext(model_path)
    guess = root + ".weights.h5"
    if os.path.exists(guess):
        return guess

    return None


def _load_or_rebuild_geoprior_model(
    model_path: str,
    manifest: dict,
    X_sample: dict,
    out_s_dim: int,
    out_g_dim: int,
    mode: str,
    horizon: int,
    quantiles: list[float] | None,
    city_name: str | None = None,
    compile_on_load: bool = True,
    verbose: int = 1,
):
    """
    Load a tuned GeoPriorSubsNet from disk, with robust rebuild fallback.

    This helper centralizes the logic:

    1. Try to load the model from ``model_path`` via
       :func:`tf.keras.models.load_model`, with all required custom
       objects registered (GeoPriorSubsNet, LearnableMV, etc.).

    2. If loading fails (e.g. due to environment or serialization
       changes), attempt a robust fallback:
       - Load the tuned hyperparameters via
         :func:`load_best_hps_near_model`.
       - Rebuild a compatible GeoPriorSubsNet instance using
         :func:`build_geoprior_from_hps`, based on Stage-1
         ``manifest['config']`` and an input sample ``X_sample``.
       - Find the best weights checkpoint via
         :func:`infer_best_weights_path` and load them into the
         rebuilt model, if available.

    Parameters
    ----------
    model_path : str
        Path to the tuned model archive (usually ``.keras``) produced
        by the tuner, e.g.::

            .../tuning/run_YYYYMMDD-HHMMSS/nansha_GeoPrior_best.keras

    manifest : dict
        Stage-1 manifest dictionary; its ``"config"`` entry is used to
        reconstruct the compile/physics configuration when rebuilding.

    X_sample : dict
        One NPZ inputs dictionary (e.g. validation or train NPZ) that
        has already been sanitized and passed through
        :func:`ensure_input_shapes`. Only its shapes are used to infer
        input dimensions.

    out_s_dim : int
        Output dimension for the subsidence head
        (usually from ``M['artifacts']['sequences']['dims']``).

    out_g_dim : int
        Output dimension for the GWL head.

    mode : str
        Sequence mode, e.g. ``"tft_like"`` or ``"pihal_like"``.

    horizon : int
        Forecast horizon (number of time steps).

    quantiles : list of float or None
        Quantile levels used for probabilistic outputs. If ``None``,
        the model is treated as a point-forecast model.

    city_name : str or None, optional
        City name for log messages. If ``None``, a neutral label is
        used in logs.

    compile_on_load : bool, default=True
        Whether to pass ``compile=True`` to :func:`load_model`. If
        ``False``, the model is loaded uncompiled, and only the
        rebuilt branch is compiled via
        :func:`build_geoprior_from_hps`.

    verbose : int, default=1
        Verbosity level for log messages (0 = silent, 1 = info).

    Returns
    -------
    model :
        A GeoPriorSubsNet instance (or compatible model) ready for
        evaluation/prediction.

    best_hps : dict or None
        Dictionary of tuned hyperparameters if they were loaded during
        the fallback path; otherwise ``None``.

    Raises
    ------
    ImportError
        If TensorFlow or required fusionlab components cannot be
        imported.

    RuntimeError
        If both direct loading and fallback reconstruction fail.
    """
    label_city = city_name or "GeoPrior"

    # --- Lazy imports so nat_utils can be imported without TF/fusionlab ---
    try:
        import tensorflow as tf  # type: ignore # noqa
        from tensorflow.keras.models import load_model  # type: ignore
        from tensorflow.keras.utils import custom_object_scope  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "load_or_rebuild_geoprior_model requires TensorFlow. "
            "Please install 'tensorflow>=2.12' to use this helper."
        ) from e

    try:
        from fusionlab.nn.pinn.models import GeoPriorSubsNet  # type: ignore
        from fusionlab.params import (  # type: ignore
            LearnableMV,
            LearnableKappa,
            FixedGammaW,
            FixedHRef,
        )
        from fusionlab.nn.losses import make_weighted_pinball  # type: ignore
        from fusionlab.nn.keras_metrics import (  # type: ignore
            coverage80_fn,
            sharpness80_fn,
        )
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "load_or_rebuild_geoprior_model requires fusionlab components "
            "(GeoPriorSubsNet, LearnableMV, etc.). Ensure fusionlab is "
            "installed and importable."
        ) from e

    # ------------------- 1) Try direct load_model -------------------------
    custom_objects = {
        "GeoPriorSubsNet": GeoPriorSubsNet,
        "LearnableMV": LearnableMV,
        "LearnableKappa": LearnableKappa,
        "FixedGammaW": FixedGammaW,
        "FixedHRef": FixedHRef,
        # custom loss factory / class
        "make_weighted_pinball": make_weighted_pinball,
        # custom metrics used in compile
        "coverage80_fn": coverage80_fn,
        "sharpness80_fn": sharpness80_fn,
    }

    best_hps: dict | None = None

    with custom_object_scope(custom_objects):
        if verbose:
            print(f"[Model] Attempting to load tuned model from: {model_path}")
        
        # try:
        model = load_model(model_path, compile=compile_on_load)
        if verbose:
            print(f"[Model] Successfully loaded tuned model for {label_city} "
                  f"from: {model_path}")
        return model, best_hps
        # except Exception as e_load:
        #     if verbose:
        #         print(
        #             f"[Warn] load_model('{model_path}') failed: {e_load}\n"
        #             "[Warn] Attempting robust fallback: rebuild GeoPriorSubsNet "
        #             "from tuned hyperparameters."
        #         )

    # ------------------- 2) Fallback: rebuild + load weights --------------
    # 2.1 Hyperparameters near the tuned model
    try:
        best_hps = load_best_hps_near_model(model_path)
    except Exception as e_hps:
        raise RuntimeError(
            "Failed to load tuned hyperparameters for fallback model "
            f"reconstruction near model_path={model_path!r}: {e_hps}"
        ) from e_hps

    # 2.2 Rebuild architecture + compile using Stage-1 manifest + best_hps
    try:
        model = build_geoprior_from_hps(
            manifest=manifest,
            X_sample=X_sample,
            best_hps=best_hps,
            out_s_dim=out_s_dim,
            out_g_dim=out_g_dim,
            mode=mode,
            horizon=horizon,
            quantiles=quantiles,
        )
    except Exception as e_build:
        raise RuntimeError(
            "Failed to reconstruct GeoPriorSubsNet from best_hps. "
            f"Error: {e_build}"
        ) from e_build

    # 2.3 Load weights into the rebuilt model, if a checkpoint is found
    weights_path = infer_best_weights_path(model_path)
    if weights_path is not None:
        try:
            model.load_weights(weights_path)
            if verbose:
                print(
                    "[Fallback] Loaded weights into rebuilt GeoPriorSubsNet "
                    f"from: {weights_path}"
                )
        except Exception as e_w:
            # We still return the rebuilt model, but warn that it is not
            # weight-identical to the tuned run.
            if verbose:
                print(
                    "[Warn] Could not load weights from checkpoint:\n"
                    f"       {weights_path}\n"
                    f"       Error: {e_w}\n"
                    "       The rebuilt model is using freshly-initialized "
                    "weights. Predictions will NOT match the tuned model."
                )
    else:
        if verbose:
            print(
                "[Warn] No weights checkpoint found near tuned model.\n"
                "       Using rebuilt model with freshly-initialized "
                "weights. Predictions will NOT match the tuned model."
            )

    return model, best_hps

def infer_input_dims_from_X(X: dict) -> tuple[int, int, int]:
    """
    Infer (static_input_dim, dynamic_input_dim, future_input_dim) 
    from NPZ inputs.

    This is a public, defensive version of the former
    ``_infer_input_dims_from_X`` helper.

    Parameters
    ----------
    X : dict
        Dictionary with keys:

        - ``'dynamic_features'`` (required, shape (N, T, D_dyn))
        - ``'static_features'`` (optional, shape (N, D_static) or None)
        - ``'future_features'`` (optional, shape (N, T_future, D_future) or None)

    Returns
    -------
    static_dim : int
        Last-dimension size of ``static_features`` (0 if missing or None).

    dynamic_dim : int
        Last-dimension size of ``dynamic_features``. Raises if missing.

    future_dim : int
        Last-dimension size of ``future_features`` (0 if missing or None).

    Raises
    ------
    KeyError
        If ``'dynamic_features'`` is missing in ``X``.
    """
    if "dynamic_features" not in X:
        raise KeyError(
            "X must contain key 'dynamic_features' with shape (N, T, D_dyn)."
        )

    dyn = np.asarray(X["dynamic_features"])
    dynamic_dim = int(dyn.shape[-1])

    static = X.get("static_features", None)
    static_dim = int(np.asarray(static).shape[-1]) if static is not None else 0

    fut = X.get("future_features", None)
    future_dim = int(np.asarray(fut).shape[-1]) if fut is not None else 0

    return static_dim, dynamic_dim, future_dim
# -------------------------------------------------------------------------
# Backward-compatible aliases for old private helper names
# -------------------------------------------------------------------------
safe_compile = compile_for_eval 

_infer_input_dims_from_X = infer_input_dims_from_X
_load_best_hps_near_model = load_best_hps_near_model
_coerce_quantile_weights = coerce_quantile_weights
_compile_geoprior_for_eval = compile_geoprior_for_eval
_build_geoprior_from_hps = build_geoprior_from_hps
_infer_best_weights_path = infer_best_weights_path

_build_geoprior_from_cfg = build_geoprior_from_cfg

# Back-compat alias (docstrings still mention it)
extract_stage_outputs = extract_preds
