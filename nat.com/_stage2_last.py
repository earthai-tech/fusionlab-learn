
# =============================================================================
# Evaluate metrics & physics on the forecasting split (+ optional censoring)
# =============================================================================
eval_results = {}
phys = {}

# Better: reuse make_tf_dataset so keys/shapes match training exactly
# (instead of tf.data.Dataset.from_tensor_slices)
ds_eval = make_tf_dataset(
    X_fore, y_fore,                 
    batch_size=BATCH_SIZE,
    shuffle=False,
    mode=MODE,
    forecast_horizon=FORECAST_HORIZON_YEARS,
)

if DEBUG:
    xb, yb = next(iter(ds_eval))
    out = model_inf(xb, training=False)
    sp = out["subs_pred"]
    print("subs_pred shape:", sp.shape)
    print("subs_pred static:", sp.shape)
    print("subs_pred dyn   :", tf.shape(sp).numpy())
    
    # Try both interpretations explicitly:
    sp_fix = canonicalize_BHQO(
        sp,
        y_true=yb["subs_pred"],
        q_values=QUANTILES,
        n_q=(len(QUANTILES) if QUANTILES else None),
        enforce_monotone=False,
        verbose=1,
        log_fn=(lambda *_: None),
    )
    
    print("canonicalized shape:", sp_fix.shape)

# 1) Dataset debug (safe)
_ = debug_val_interval(
    model_inf,
    ds_eval,
    n_q=len(QUANTILES),
    max_batches=2,
    verbose=1,
)
# after loading with compile=False
if not USE_IN_MEMORY_MODEL:
    model_inf.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0),  # dummy is fine
        loss=loss_arg,
        loss_weights=lossw_arg,
        metrics=metrics_compile,
        **physics_loss_weights,
    )

# --- 2.1 Standard Keras evaluate() + physics metrics ---
try:# Use model_inf instead of subs_model_inst
    eval_raw = model_inf.evaluate(
        ds_eval,
        return_dict=True,
        verbose=1,
    )
    
    eval_results = _logs_to_py(eval_raw)
    print("Evaluation:", eval_results)

    # v3.2: include epsilon_gw if available
    phys_keys = ("epsilon_prior", "epsilon_cons", "epsilon_gw")
    phys = {
        k: float(_to_py(eval_raw[k]))
        for k in phys_keys
        if k in eval_results
    }
    if phys:
        print("Physics diagnostics (from evaluate):", phys)

except Exception as e:
    print(f"[Warn] Evaluation failed (metrics + physics): {e}")
    eval_results, phys = {}, {}

# --- 2.2 Save physics payload (use the same model & same ds split you evaluated) ---
phys_npz_path = os.path.join(
    RUN_OUTPUT_PATH,
    f"{CITY_NAME}_phys_payload_run_val.npz",
    # f"{CITY_NAME}_phys_payload_{dataset_name_for_forecast.lower()}.npz"
 )

try:
    _ = model_inf.export_physics_payload(
        ds_eval,
        max_batches=None,
        save_path=phys_npz_path,
        format="npz",
        overwrite=True,
        metadata={
            "city": CITY_NAME,
            "split": dataset_name_for_forecast,
            "time_units": TIME_UNITS,
            "gwl_kind": GWL_KIND,
            "gwl_sign": GWL_SIGN,
            "use_head_proxy": USE_HEAD_PROXY,
        },
    )
    print(f"[OK] Saved physics payload -> {phys_npz_path}")
except Exception as e : 
    print(f"Failed to saved physic payload: {e}")
     
#%
# -------------------------------------------------------------------------
# SM3: log-offset diagnostics (δ_K, δ_Ss, δ_Hd, δ_tau)
# -------------------------------------------------------------------------
def ensure_subs_bhq(
    s_pred_b,
    *,
    y_true_b,
    q_values,
    enforce_monotone=True,
):
    # Expect rank-4 in quantile mode.
    if getattr(s_pred_b, "shape", None) is None:
        s_pred_b = tf.convert_to_tensor(s_pred_b)

    if s_pred_b.shape.rank != 4:
        return s_pred_b

    n_q = len(q_values) if q_values else 0
    if n_q <= 0:
        return s_pred_b

    # Easy cases when dims differ.
    d1 = int(s_pred_b.shape[1])
    d2 = int(s_pred_b.shape[2])

    if (d2 == n_q) and (d1 != n_q):
        out = s_pred_b
    elif (d1 == n_q) and (d2 != n_q):
        out = tf.transpose(s_pred_b, [0, 2, 1, 3])
    else:
        # Ambiguous (often H==Q): choose by median MAE vs y_true.
        q = np.asarray(q_values, dtype=float)
        med = int(np.argmin(np.abs(q - 0.5)))

        keep = s_pred_b
        swap = tf.transpose(s_pred_b, [0, 2, 1, 3])

        mae_keep = tf.reduce_mean(
            tf.abs(keep[:, :, med, :] - y_true_b)
        )
        mae_swap = tf.reduce_mean(
            tf.abs(swap[:, :, med, :] - y_true_b)
        )

        out = tf.cond(
            mae_swap < mae_keep,
            lambda: swap,
            lambda: keep,
        )

    if enforce_monotone:
        out = tf.sort(out, axis=2)

    return out
#%
# --- 2.3 Interval diagnostics + optional censor-stratified MAE ---
cov80_uncal = cov80_cal = sharp80_uncal = sharp80_cal = None
censor_metrics = None   # will become a dict if we have a flag

y_true_list, s_q_list, mask_list = [], [], []


for xb, yb in with_progress(ds_eval, desc="Interval-Censoring Diagnostics"):
    out = model_inf(xb, training=False)
    s_pred_b, _ = extract_preds(model_inf, out)   # <- (B,H,1) or (B,H,Q,1)
    # s_pred_b is already (B, H, Q, 1) by design
    # just enforce monotone quantiles
    # s_pred_b = tf.sort(s_pred_b, axis=2)

    # s_pred_b = ensure_subs_bhq(
    #     s_pred_b,
    #     y_true_b=yb["subs_pred"],
    #     q_values=QUANTILES,
    #     enforce_monotone=True,
    # )


    s_pred_b = canonicalize_BHQO(
        s_pred_b,
        y_true=yb["subs_pred"],
        q_values=QUANTILES,
        n_q=(len(QUANTILES) if QUANTILES else None),
        enforce_monotone=True,
        layout="BHQO",
        verbose=0,
        log_fn=(lambda *_: None),
    )
    
    # ----------------------------NEW---------------
    # s_pred_b = canonicalize_BHQO_quantiles_np(
    #     s_pred_b,
    #     n_q=len(QUANTILES),
    #     verbose=0,
    #     log_fn=print,
    # )
    
    # # If you still want monotone quantiles (optional):
    # s_pred_b = np.sort(s_pred_b, axis=2)
    
    # ------------------------------
    y_true_b = yb["subs_pred"]                    # (B,H,1)
    y_true_list.append(y_true_b)

    if QUANTILES:
        # s_pred_b is already (B,H,Q,1)
        s_q_list.append(s_pred_b)

    if CENSOR_FLAG_IDX is not None:
        H = tf.shape(y_true_b)[1]
        mask_b = build_censor_mask(
            xb, H, CENSOR_FLAG_IDX, CENSOR_THRESH,
            source=CENSOR_MASK_SOURCE or "dynamic",
            reduce_time="any",
            align="broadcast",
        )
        mask_list.append(mask_b)
#
# # Stack what we collected
y_true = tf.concat(y_true_list, axis=0) if y_true_list else None  # (N,H,1)
s_q = tf.concat(s_q_list, axis=0) if s_q_list else None           # (N,H,Q,1)
mask = tf.concat(mask_list, axis=0) if mask_list else None        # (N,H,1) booleans
#%
#=========================test of axis shape 
# s_q is (N,H,Q,1), y_true is (N,H,1)
q10 = s_q[..., 0, :]      # (N,H,1)
q90 = s_q[..., -1, :]     # (N,H,1)

cov_manual = tf.reduce_mean(
    tf.cast((y_true >= q10) & (y_true <= q90), tf.float32)
).numpy()

print("cov_manual:", cov_manual)
print("cov_fn    :", float(coverage80_fn(y_true, s_q).numpy()))

sharp_manual = tf.reduce_mean((q90 - q10)).numpy()
print("sharp_manual:", sharp_manual)
print("sharp_fn    :", float(sharpness80_fn(y_true, s_q).numpy()))

hit = tf.cast((y_true >= q10) & (y_true <= q90), tf.float32)  # (N,H,1)
cov_per_h = tf.reduce_mean(hit, axis=[0, 2])  # (H,)

print("cov_per_h:", cov_per_h.numpy())

#%
# --- 2.3.a Interval coverage/sharpness (scaled + physical) ---------------
cov80_uncal_phys = cov80_cal_phys = None
sharp80_uncal_phys = sharp80_cal_phys = None
s_q_cal = None

if QUANTILES and (y_true is not None) and (s_q is not None):
    # ---------- SCALED metrics (as before) ----------
    cov80_uncal   = float(coverage80_fn(y_true, s_q).numpy())
    sharp80_uncal = float(sharpness80_fn(y_true, s_q).numpy())
    # Calibrated (apply same calibrator to the whole tensor)
    s_q_cal = apply_calibrator_to_subs(cal80, s_q)  # (N, H, Q, 1) # or keeps (N, H, 3, 1)
    cov80_cal   = float(coverage80_fn(y_true, s_q_cal).numpy())
    sharp80_cal = float(sharpness80_fn(y_true, s_q_cal).numpy())

    # # ---------- PHYSICAL metrics (inverse-scaled) ----------
    # # 1) inverse-transform y_true and quantiles to physical units
    # y_true_phys_np = inverse_scale_target(
    #     y_true,
    #     scaler_info=scaler_info_dict,
    #     target_name=SUBSIDENCE_COL,
    # )
    # s_q_phys_np = inverse_scale_target(
    #     s_q,
    #     scaler_info=scaler_info_dict,
    #     target_name=SUBSIDENCE_COL,
    # )

    # y_true_phys_tf = tf.convert_to_tensor(y_true_phys_np, dtype=tf.float32)
    # s_q_phys_tf    = tf.convert_to_tensor(s_q_phys_np,    dtype=tf.float32)

    # cov80_uncal_phys   = float(coverage80_fn(y_true_phys_tf, s_q_phys_tf).numpy())
    # sharp80_uncal_phys = float(sharpness80_fn(y_true_phys_tf, s_q_phys_tf).numpy())

    # if s_q_cal is not None:
    #     s_q_cal_phys_np = inverse_scale_target(
    #         s_q_cal,
    #         scaler_info=scaler_info_dict,
    #         target_name=SUBSIDENCE_COL,
    #     )
    #     s_q_cal_phys_tf = tf.convert_to_tensor(s_q_cal_phys_np, dtype=tf.float32)

    #     cov80_cal_phys   = float(coverage80_fn(y_true_phys_tf, s_q_cal_phys_tf).numpy())
    #     sharp80_cal_phys = float(sharpness80_fn(y_true_phys_tf, s_q_cal_phys_tf).numpy())

    # ---------- PHYSICAL metrics (inverse-scaled) ----------
    # IMPORTANT:
    #   Stage-1 scaler_info is keyed by SUBS_SCALER_KEY (scaler entry name),
    #   not by SUBSIDENCE_COL (df/output column name). Using SUBSIDENCE_COL
    #   can silently skip or mis-apply inverse scaling.
    _subs_scale_key = SUBS_SCALER_KEY
    
    # 1) inverse-transform y_true and quantiles to physical units
    y_true_phys_np = inverse_scale_target(
        y_true.numpy() if hasattr(y_true, "numpy") else y_true,
        scaler_info=scaler_info_dict,
        target_name=_subs_scale_key,
    )
    s_q_phys_np = inverse_scale_target(
        s_q.numpy() if hasattr(s_q, "numpy") else s_q,
        scaler_info=scaler_info_dict,
        target_name=_subs_scale_key,
    )
    
    y_true_phys_tf = tf.convert_to_tensor(y_true_phys_np, dtype=tf.float32)
    s_q_phys_tf    = tf.convert_to_tensor(s_q_phys_np,    dtype=tf.float32)
    
    cov80_uncal_phys   = float(coverage80_fn(y_true_phys_tf, s_q_phys_tf).numpy())
    sharp80_uncal_phys = float(sharpness80_fn(y_true_phys_tf, s_q_phys_tf).numpy())
    
    if s_q_cal is not None:
        s_q_cal_phys_np = inverse_scale_target(
            s_q_cal.numpy() if hasattr(s_q_cal, "numpy") else s_q_cal,
            scaler_info=scaler_info_dict,
            target_name=_subs_scale_key,
        )
        s_q_cal_phys_tf = tf.convert_to_tensor(s_q_cal_phys_np, dtype=tf.float32)
    
        cov80_cal_phys   = float(coverage80_fn(y_true_phys_tf, s_q_cal_phys_tf).numpy())
        sharp80_cal_phys = float(sharpness80_fn(y_true_phys_tf, s_q_cal_phys_tf).numpy())
#%
# ---- Debug: scaling should NOT change coverage (only sharpness) ----
if DEBUG:
    print("[SCALEDBG] subs scaler key:", _subs_scale_key)
    print("[SCALEDBG] cov scaled vs phys:",
          cov80_uncal, cov80_uncal_phys,
          "| cal:", cov80_cal, cov80_cal_phys)
    print("[SCALEDBG] sharp scaled vs phys:",
          sharp80_uncal, sharp80_uncal_phys,
          "| cal:", sharp80_cal, sharp80_cal_phys)

    # Detect silent 'no-op' inverse scaling
    yt_np = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true)
    if np.allclose(y_true_phys_np, yt_np, atol=1e-12, rtol=0):
        print("[WARN] inverse_scale_target() did not change y_true "
              "(likely wrong target_name / scaler entry not resolved).")


# 2) Tensor debug (safe)

    _ = debug_tensor_interval(
        y_true,
        s_q,
        n_q=len(QUANTILES),
        name="RAW",
        verbose=1,
    )
    
    _ = debug_tensor_interval(
        y_true,
        s_q_cal,
        n_q=len(QUANTILES),
        name="CAL",
        verbose=1,
    )

# # Interpret axis=2 as quantiles (what stage2 assumes)
# w_axis2 = np.mean(s_q[:, :, 2, 0] - s_q[:, :, 0, 0])
# c_axis2 = np.mean(
#     (y_true[:, :, 0] >= s_q[:, :, 0, 0]) &
#     (y_true[:, :, 0] <= s_q[:, :, 2, 0])
# )

# # Interpret axis=1 as quantiles (the “other” possibility)
# w_axis1 = np.mean(s_q[:, 2, :, 0] - s_q[:, 0, :, 0])
# c_axis1 = np.mean(
#     (y_true[:, :, 0] >= s_q[:, 0, :, 0]) &
#     (y_true[:, :, 0] <= s_q[:, 2, :, 0])
# )

# print("width axis2:", w_axis2, "coverage axis2:", c_axis2)
# print("width axis1:", w_axis1, "coverage axis1:", c_axis1)
#%
# --- 2.3.b Optional censor-stratified MAE on the same loop products ---
# Works for both quantile mode (use median) and point-forecast mode (fallback).

# Median quantile index (robust)
_med_idx = None
if QUANTILES:
    _med_idx = int(np.argmin(np.abs(np.asarray(QUANTILES, dtype=float) - 0.5)))
    
if (y_true is not None) and (mask is not None):

    if QUANTILES and (s_q is not None):
        # s_q: (N,H,Q,1) -> median: (N,H,1)
        med_idx = _med_idx
        s_med = s_q[..., med_idx, :]
    else:
        # point-forecast: run model and collect (B,H,1) batches
        s_pred_list = []
        for xb2, yb2 in with_progress(ds_eval,
                              desc="Point preds for censor-MAE"):
            out2 = model_inf(xb2, training=False)
            s2 = subs_point_from_out(model_inf, out2, QUANTILES, _med_idx)  # (B,H,1)
            s_pred = out2["subs_pred"]
            s_pred = canonicalize_BHQO(
                s_pred,
                y_true=yb2["subs_pred"],
                q_values=QUANTILES,
                n_q=(len(QUANTILES) if QUANTILES else None),
                enforce_monotone=True,
                layout="BHQO",
                verbose=0,
                log_fn=(lambda *_: None),
            )
            s2 = s_pred[:, :, int(_med_idx), :]
            s_pred_list.append(s2)

        if not s_pred_list:
            raise RuntimeError("No batches collected for point preds (censor-MAE).")

        s_med = tf.concat(s_pred_list, axis=0)  # (N,H,1) scaled/model space

    # # Convert both y_true and s_med to physical units using Stage-1 scaler_info
    # y_true_phys_np = inverse_scale_target(
    #     y_true,
    #     scaler_info=scaler_info_dict,
    #     target_name=SUBSIDENCE_COL,
    # )
    # s_med_phys_np = inverse_scale_target(
    #     s_med,
    #     scaler_info=scaler_info_dict,
    #     target_name=SUBSIDENCE_COL,
    # )

    # y_true_phys = tf.convert_to_tensor(y_true_phys_np, dtype=tf.float32)
    # s_med_phys  = tf.convert_to_tensor(s_med_phys_np,  dtype=tf.float32)
    
    # Convert both y_true and s_med to physical units using Stage-1 scaler_info
    # IMPORTANT: use SUBS_SCALER_KEY, not SUBSIDENCE_COL
    _subs_scale_key = SUBS_SCALER_KEY
    
    y_true_phys_np = inverse_scale_target(
        y_true.numpy() if hasattr(y_true, "numpy") else y_true,
        scaler_info=scaler_info_dict,
        target_name=_subs_scale_key,
    )
    s_med_phys_np = inverse_scale_target(
        s_med.numpy() if hasattr(s_med, "numpy") else s_med,
        scaler_info=scaler_info_dict,
        target_name=_subs_scale_key,
    )
    
    y_true_phys = tf.convert_to_tensor(y_true_phys_np, dtype=tf.float32)
    s_med_phys  = tf.convert_to_tensor(s_med_phys_np,  dtype=tf.float32)
    
    if DEBUG:
        yt_np = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true)
        if np.allclose(y_true_phys_np, yt_np, atol=1e-12, rtol=0):
            print("[WARN] censor-MAE inverse_scale_target() no-op on y_true "
                  "(check SUBS_SCALER_KEY/scaler_info).")

    mask_f = tf.cast(mask, tf.float32)  # (N,H,1)
    num_cens = tf.reduce_sum(mask_f) + 1e-8
    num_unc  = tf.reduce_sum(1.0 - mask_f) + 1e-8

    abs_err = tf.abs(y_true_phys - s_med_phys)
    mae_cens = tf.reduce_sum(abs_err * mask_f) / num_cens
    mae_unc  = tf.reduce_sum(abs_err * (1.0 - mask_f)) / num_unc

    censor_metrics = {
        "flag_name": CENSOR_FLAG_NAME,
        "threshold": float(CENSOR_THRESH),
        "mae_censored": float(mae_cens.numpy()),
        "mae_uncensored": float(mae_unc.numpy()),
    }

    print(
        f"[CENSOR] MAE censored={censor_metrics['mae_censored']:.4f} | "
        f"uncensored={censor_metrics['mae_uncensored']:.4f}"
    )
    
if DEBUG:
    # Minimal sanity checks: If that var() is no longer tiny,  
    # R² will stop being absurdly negative

    yt_phys = inverse_scale_target(
        y_true.numpy() if hasattr(y_true, "numpy") else y_true,
        scaler_info=scaler_info_dict,
        target_name=SUBS_SCALER_KEY,
    )
    print(
        "[DEBUG] y_true_phys stats:",
        float(np.min(yt_phys)),
        float(np.max(yt_phys)),
        float(np.mean(yt_phys)),
        float(np.var(yt_phys)),
    )


# Normalize coverage/sharpness choices for ablation record (prefer calibrated)
coverage80_for_abl = (
    cov80_cal_phys if (cov80_cal_phys is not None)
    else cov80_cal if (cov80_cal is not None)
    else cov80_uncal_phys if (cov80_uncal_phys is not None)
    else cov80_uncal
)

sharpness80_for_abl = (
    sharp80_cal_phys if (sharp80_cal_phys is not None)
    else sharp80_uncal_phys if (sharp80_uncal_phys is not None)
    else sharp80_cal if (sharp80_cal is not None)
    else sharp80_uncal
)

# Save summary JSON
stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
payload = {
    "timestamp": stamp,
    "tf_version": tf.__version__,
    "numpy_version": np.__version__,
    "quantiles": QUANTILES,
    "horizon": FORECAST_HORIZON_YEARS,
    "batch_size": BATCH_SIZE,
    "metrics_evaluate": {k: _to_py(v) for k, v in (eval_results or {}).items()},
    "physics_diagnostics": phys,
}

if QUANTILES:
    payload["interval_calibration"] = {
        "target": 0.80,
        "factors_per_horizon": getattr(cal80, "factors_", None).tolist()
        if hasattr(cal80, "factors_") else None,

        # scaled-space metrics (backward compatible)
        "coverage80_uncalibrated": cov80_uncal,
        "coverage80_calibrated":   cov80_cal,
        "sharpness80_uncalibrated": sharp80_uncal,
        "sharpness80_calibrated":   sharp80_cal,

        # physical-space metrics (new, recommended for the paper)
        "coverage80_uncalibrated_phys": cov80_uncal_phys,
        "coverage80_calibrated_phys":   cov80_cal_phys,
        "sharpness80_uncalibrated_phys": sharp80_uncal_phys,
        "sharpness80_calibrated_phys":   sharp80_cal_phys,
    }

if censor_metrics is not None:
    payload["censor_stratified"] = censor_metrics

# Attach point metrics & per-horizon into payload
if metrics_point:
    payload["point_metrics"] = {
        "mae": metrics_point.get("mae"),
        "mse": metrics_point.get("mse"),
        "r2":  metrics_point.get("r2"),
    }
if per_h_mae_dict:
    payload.setdefault("per_horizon", {})
    payload["per_horizon"]["mae"] = per_h_mae_dict
if per_h_r2_dict:
    payload.setdefault("per_horizon", {})
    payload["per_horizon"]["r2"] = per_h_r2_dict

# -------------------------------------------------------------------------
# Unit post-processing for evaluation JSON (controlled by config).
# - EVAL_JSON_UNITS_MODE  : 'si' (default) or 'interpretable'
# - EVAL_JSON_UNITS_SCOPE : 'subsidence', 'physics', or 'all'
# -------------------------------------------------------------------------
# XXX DO NOT CONVERT #: FOR DEBUG
try:
    payload = convert_eval_payload_units(
        payload,
        cfg,
        mode=_units_mode,
        scope=_units_scope,
    )
except Exception as e:
    print(f"[Warn] unit conversion skipped (mode={_units_mode}, scope={_units_scope}): {e}")

json_out = os.path.join(RUN_OUTPUT_PATH, f"geoprior_eval_phys_{stamp}.json")
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"Saved metrics + physics JSON -> {json_out}")
#%
ABLCFG = {
    "PDE_MODE_CONFIG": PDE_MODE_CONFIG,
    "GEOPRIOR_USE_EFFECTIVE_H": GEOPRIOR_USE_EFFECTIVE_H,
    "GEOPRIOR_KAPPA_MODE": GEOPRIOR_KAPPA_MODE,
    "GEOPRIOR_HD_FACTOR": GEOPRIOR_HD_FACTOR,
    "LAMBDA_CONS": LAMBDA_CONS,
    "LAMBDA_GW": LAMBDA_GW,
    "LAMBDA_PRIOR": LAMBDA_PRIOR,
    "LAMBDA_SMOOTH": LAMBDA_SMOOTH,
    "LAMBDA_BOUNDS": LAMBDA_BOUNDS,
    "LAMBDA_MV": LAMBDA_MV,
    "LAMBDA_Q": LAMBDA_Q,
}

# Prefer MAE/MSE from the *unit-consistent* payload (already post-processed);
# fall back to locally computed metrics when missing.
_m_eval = payload.get("metrics_evaluate", {}) if isinstance(payload, dict) else {}
_p_point = payload.get("point_metrics", {}) if isinstance(payload, dict) else {}
_p_hor = payload.get("per_horizon", {}) if isinstance(payload, dict) else {}

eval_mae = _m_eval.get("subs_pred_mae", _p_point.get("mae"))
eval_mse = _m_eval.get("subs_pred_mse", _p_point.get("mse"))

# Coverage/sharpness for ablation: prefer evaluate() values if present.
abl_coverage80 = _m_eval.get("subs_pred_coverage80", coverage80_for_abl)
abl_sharpness80 = _m_eval.get("subs_pred_sharpness80", sharpness80_for_abl)

# Per-horizon: prefer payload copies (unit-consistent) else local.
per_h_mae_for_abl = (_p_hor.get("mae") if isinstance(_p_hor, dict) else None) or per_h_mae_dict
per_h_r2_for_abl = (_p_hor.get("r2") if isinstance(_p_hor, dict) else None) or per_h_r2_dict

save_ablation_record(
    outdir=RUN_OUTPUT_PATH,
    city=CITY_NAME,
    model_name=MODEL_NAME,
    cfg=ABLCFG,
    eval_dict={
        "r2": (_p_point or {}).get("r2"),
        "mse": float(eval_mse) if eval_mse is not None else None,
        "mae": float(eval_mae) if eval_mae is not None else None,
        "coverage80": float(abl_coverage80) if abl_coverage80 is not None else None,
        "sharpness80": float(abl_sharpness80) if abl_sharpness80 is not None else None,
    },
    phys_diag=(phys or {}),
    per_h_mae=per_h_mae_for_abl,   # unit-consistent when available
    per_h_r2=per_h_r2_for_abl      # unit-consistent when available
)
print("Ablation record saved.")
#
# Convert an existing SI JSON to interpretable and write a new file:
json_out_interp = os.path.join(
    RUN_OUTPUT_PATH, f"geoprior_eval_phys_{stamp}_interpretable.json"
 )
out = postprocess_eval_json(
    json_out,
    scope="all",
    out_path=json_out_interp,
    overwrite=True,
    add_rmse=True,
)
# "mm" (if SUBS_UNIT_TO_SI ~ 1e-3)
print("Interpretable EVAL JSON file to", out["units"]["subs_metrics_unit"])  


try:
    # payload is what you saved via export_physics_payload(...)
    phys_payload, _ = load_physics_payload(phys_npz_path)
    
    # 1) Spatial maps (needs coords from dataset)
    plot_physics_values_in(
        phys_payload,
        dataset=ds_eval,
        keys=[
            "cons_res_vals",
            "log10_tau",
            "log10_tau_prior",
            "K",
            "Ss",
            "Hd",
        ],
        mode="map",
        transform=None,
        savefig=os.path.join(RUN_OUTPUT_PATH, "phys_maps.png"),
    )
    
    # 2) Residual distribution (no coords needed)
    plot_physics_values_in(
        phys_payload,
        keys=["cons_res_vals"],
        mode="hist",
        transform="signed_log10",
        savefig=os.path.join(RUN_OUTPUT_PATH, "cons_res_hist.png"),
    )
except: 
    print("Failed to plot physic values in...")
    
# =============================================================================
# Visualization (optional)
# =============================================================================

print("\nPlotting forecast views...")

try:
    plot_eval_future(
        df_eval=df_eval,
        df_future=df_future,
        target_name=SUBSIDENCE_COL,
        quantiles=QUANTILES,
        spatial_cols=("coord_x", "coord_y"),
        time_col="coord_t",
        # Eval: show last eval year (e.g. 2022)
        eval_years=[FORECAST_START_YEAR - 1],
        # Future: use the same grid you passed to format_and_forecast
        future_years=future_grid,
        # For eval: compare [actual] vs [q50] only
        eval_view_quantiles=[0.5],
        # For future: show full [q10, q50, q90]
        future_view_quantiles=QUANTILES,
        spatial_mode="hexbin",      # hotspot view
        hexbin_gridsize=40,
        savefig_prefix=os.path.join(
            RUN_OUTPUT_PATH,
            f"{CITY_NAME}_subsidence_view",
        ),
        save_fmts=[".png", ".pdf"],
        show=False,
        verbose=1,
    )
except Exception as e:
    print(f"[Warn] plot_eval_future failed: {e}")

try:
    save_all_figures(
        output_dir=RUN_OUTPUT_PATH,
        prefix=f"{CITY_NAME}_{MODEL_NAME}_plot_",
        fmts=[".png", ".pdf"],
    )
    print(f"Saved all open Matplotlib figures in: {RUN_OUTPUT_PATH}")
except Exception as e:
    print(f"[Warn] save_all_figures failed: {e}")

print(f"\n---- {CITY_NAME.upper()} {MODEL_NAME} TRAINING COMPLETE ----\n"
      f"Artifacts -> {RUN_OUTPUT_PATH}\n")

tf.keras.backend.clear_session()
gc.collect()

