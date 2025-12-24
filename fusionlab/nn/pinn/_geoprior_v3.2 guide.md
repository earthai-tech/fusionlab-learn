Perfect — for **GeoPrior v3.2** (physics-driven mean subsidence + quantile residuals + latent head + forcing (Q)), your current `_geoprior_utils.py` / `_geoprior_maths.py` are *close*, but a few functions must be **changed / added** so the new architecture is clean and robust.

Below is exactly what I would revise **before touching `geoprior_subnet.py`**.

---

## A) `_geoprior_utils.py` — what to change / add

### 1) `validate_scaling_kwargs()` is currently wrong for degrees mode

Right now it **always raises** if `coords_in_degrees=True`, which prevents the degree→meter workflow you already support via `deg_to_m()`.

✅ **Change**:

* If `coords_in_degrees=True`, don’t raise. Instead require `deg_to_m_lon/lat` exist and are valid.
* If `coords_normalized=True`, keep requiring `coord_ranges`.
* Keep requiring `time_units`.

### 2) `gwl_to_head_m()` needs to be able to get `z_surf` from more places

In v3.2 you’ll often store `z_surf` in **static_features** (or some prepacked input), not always as a direct key in `inputs`.

✅ **Change**: upgrade `gwl_to_head_m(..., inputs=...)` to search:

* `inputs[z_surf_col]` (current behavior)
* **OR** `inputs["static_features"][..., z_surf_static_index]` if you provide `z_surf_static_index` in `scaling_kwargs`
* fallback: proxy `-depth_m` if enabled

This makes head conversion consistent no matter how the dataset is packed.

### 3) Add a dedicated helper to get **head history in SI**

In v3.2 we will optionally add a **weak head anchoring loss on history**, so we need a clean function:

✅ **Add** `get_h_hist_si(model, inputs, *, want_head=True)`
Returns `(B,T,1)` head in SI meters computed from the GWL channel (and `z_surf` if available).
This avoids duplicating:

* resolve index
* slice channel
* `to_si_head`
* `gwl_to_head_m`

### 4) Add a dedicated helper to get **subsidence baseline** (s_T) in SI

In v3.2, the physics mean path needs an initial condition:
[
\bar s_T = s^{obs}(T)
]
So we want a standardized way to retrieve it from `inputs` (or a fallback).

✅ **Add** `get_s_init_si(model, inputs, like)`

* Look for keys like: `s_init_si`, `subs_hist_last_si`, `s_ref_si`, etc.
* Broadcast to `like`
* Fallback to zeros if nothing exists (but ideally your stage-1 packs it)

This replaces the “baseline” handling being hidden inside `settlement_state_for_pde()`.

### 5) Keep `get_h_ref_si()` (it’s still needed)

But we should tweak semantics:

* v3.2 uses `h_ref` mainly to compute drawdown for compaction.
* default “auto” mode should still use **last historical head**.

✅ Small change: make `get_h_ref_si()` call the new `get_h_hist_si()` internally (less duplicated logic).

---

## B) `_geoprior_maths.py` — what to change / add

### 1) `compute_gw_flow_residual()` must accept **tensor Q**, not float

v3.2 requires:
[
S_s \partial_t h = \nabla\cdot(K\nabla h) + Q
]
Your function currently uses `Q: float = 0.0` and broadcasts a scalar. That prevents learning a forcing field.

✅ **Change signature**

* `Q: Optional[Tensor] = None` (preferred)
* If `Q is None`, treat as zero
* If `Q` is provided but rank is 0/1/2, broadcast to `dh_dt` shape.

Also: decide **units** of `Q`. Since you already compute `dh_dt` in **1/s**, then `Q` must be **1/s** too. So either:

* model outputs `Q` directly in 1/s, or
* model outputs in 1/time_unit and you convert using `rate_to_per_second()` before residual.

(We’ll lock that choice when we implement the subnet.)

### 2) Consolidation residual must become **step-based**, not derivative-based

In v3.2 the subsidence mean path (\bar s) is computed by integrating the ODE:
[
\partial_t \bar s = (s_{eq}-\bar s)/\tau
]
So instead of needing `ds_dt` (autodiff), we should penalize a **discrete step residual** (much more stable).

✅ **Add** two functions:

**(i) `integrate_consolidation_mean(...)`**
Returns (\bar s(t)) over the forecast horizon using the **exact stable update**:
[
\bar s_{n+1}=\bar s_n e^{-\Delta t/\tau}+s_{eq,n}(1-e^{-\Delta t/\tau})
]

Inputs:

* `h_mean_si (B,H,1)`
* `Ss_field (B,H,1)`
* `H_field_si (B,H,1)`
* `tau_field (B,H,1)`
* `h_ref_si (B,H,1)` (broadcast ok)
* `s_init_si (B,1,1)`
* `dt` (scalar or (B,H,1)) in **time_units**

**(ii) `compute_consolidation_step_residual(...)`**
Compute:
[
R_{\text{cons,step}} = \bar s_{n+1} - \Big(\bar s_n e^{-\Delta t/\tau}+s_{eq,n}(1-e^{-\Delta t/\tau})\Big)
]
This becomes `consolidation_loss`.

➡️ With this, your current `compute_consolidation_residual(ds_dt,...)` becomes “legacy” and is no longer used by v3.2.

### 3) `settlement_state_for_pde()` becomes optional / simplified

In v3.2, the **network is not directly outputting cumulative (s) as a free function** for physics; physics produces (\bar s).
So `settlement_state_for_pde()` is no longer the core piece.

✅ **Change**:

* keep it for compatibility (increment/rate modes)
* but v3.2 will mostly use `integrate_consolidation_mean()` for the state.

### 4) `compute_scales()` should accept tensor Q and use v3.2 signals

Right now `compute_scales(..., Q: float=0.0)` is scalar-only.

✅ **Change**:

* allow `Q: Optional[Tensor]=None`
* use `s_mean = s_bar` (physics mean) not “model raw output”
* keep your auto-scaling idea, but ensure GW residual isn’t always ~1e-9.

(We can improve scaling later; first we just make the API consistent.)

---

## Summary checklist (what you’ll edit right now)

### `_geoprior_utils.py`

* [ ] Fix `validate_scaling_kwargs()` (don’t reject degrees; validate conversion factors)
* [ ] Upgrade `gwl_to_head_m()` to read `z_surf` from static_features too
* [ ] Add `get_h_hist_si()`
* [ ] Add `get_s_init_si()`
* [ ] Refactor `get_h_ref_si()` to reuse `get_h_hist_si()`

### `_geoprior_maths.py`

* [ ] Update `compute_gw_flow_residual(..., Q)` to accept tensor Q
* [ ] Add `integrate_consolidation_mean()` (exact-step integrator)
* [ ] Add `compute_consolidation_step_residual()` and switch v3.2 to it
* [ ] Adjust `compute_scales()` for tensor Q and physics-mean `s_bar`

---

If you want, paste your current **input packing convention** (what keys exist in `inputs` during `train_step`: do you already pass `subs_hist_last` / `s_init` / `z_surf`?), and I’ll propose the exact key names so utilities line up perfectly with your pipeline.

#################################################################################
Absolutely — I’ve started revising `_geoprior_subnet.py` to implement the **v3.2 architecture** 
you described:

## ✅ **DONE so far (in `_geoprior_subnet.py`)**

### ✅ 1) Added a learnable **Q-head**

* `self.output_Q_dim = 1`
* `self.Q_head = Dense(1, name="Q_head", ...)`
* `phys_mean_raw` now concatenates:
  [
  [K,; Ss,; \Delta \log \tau,; Q]
  ]

### ✅ 2) `call()` now uses **physics-driven mean subsidence** via `integrate_consolidation_mean()`

Inside `call()` we now:

1. predict head (GWL mean)
2. convert to SI + head (`gwl_to_head_m`)
3. compose bounded fields (`compose_physics_fields`)
4. compute `h_ref`, `s_init`
5. compute `s_eq`
6. integrate:
   [
   \bar{s}(t)=\texttt{integrate_consolidation_mean}(\hat{h}, Ss, H, \tau)
   ]
7. add this physics mean to the predicted subsidence channel (broadcast-safe even when quantiles exist)

So subsidence prediction is now structurally:
[
\hat{s} = \bar{s}*{phys} + r*{net}
]

### ✅ 3) `train_step()` updated to use physics-driven settlement for PDE

* Removed dependence on `settlement_state_for_pde()` for physics state.
* Uses:

  * `s_eq = compute_equilibrium_settlement(...)`
  * `s_pde = integrate_consolidation_mean(...)`

### ✅ 4) GW residual now receives learnable **Q**

* `Q_base` from subnet output
* converted to 1/s via `rate_to_per_second(Q_base, time_units)`
* passed into:

```python
compute_gw_flow_residual(..., Q=Q_si)
```

### ✅ 5) Consolidation loss now uses **step-residual**

* replaced derivative residual with:

```python
cons_step = compute_consolidation_step_residual(...)
cons_res  = rate_to_per_second(cons_step, time_units)
```

---

## ⚠️ INCOMPLETE (but I know exactly what remains)

---

### 🚧 A) `_evaluate_physics_on_batch()` (used by `test_step`) is still on old logic

It still:

* calls `settlement_state_for_pde()`
* computes `ds_dt`
* uses `compute_consolidation_residual()`
* uses `Q=0.0`
* references `ds_dt` in scaling

✅ It needs the *same refactor* as `train_step()`:

* include `Q_base`
* compute `Q_si`
* compute `s_eq`
* compute `s_pde = integrate_consolidation_mean(...)`
* compute `cons_step` + `rate_to_per_second`
* remove all `ds_dt` usage
* update `compute_scales(ds_dt=None, Q=Q_si, ...)`

### 🚧 B) `split_physics_predictions()` updated correctly — but call sites must match

I updated it to return:

```python
(s_mean, gwl_mean, K_base, Ss_base, tau_base, Q_base)
```

✅ `train_step()` was updated to unpack 6 terms,
but evaluation still unpacks 5 terms → must fix.

### 🚧 C) minor cleanup needed

In some places the subnet still references variables that no longer exist (like `ds_dt`) — but only inside evaluation code.

---

## 🟦 About `compute_drawdown`

You’re right: we don’t strictly need `compute_drawdown()` anymore because:

* `compute_equilibrium_settlement()` already implements drawdown internally with clamping
* v3.2 workflow uses `compute_equilibrium_settlement()` directly

So `compute_drawdown` is optional.

---

# ✅ Next step (what I will do immediately if you confirm)

I will finish the refactor by rewriting `_evaluate_physics_on_batch()` to match the new physics pipeline:

✅ physics mean path via `integrate_consolidation_mean`
✅ Q forcing passed into GW residual
✅ consolidation residual becomes step-based
✅ remove ds_dt autodiff entirely

---

If you want, I can paste a **full patch block** for `_evaluate_physics_on_batch()` next (it’s about ~80 lines changed, but very clean).

###############################################################
   # def _evaluate_physics_on_batch(
   #     self,
   #     inputs: Dict[str, Optional[Tensor]],
   #     return_maps: bool = False,
   # ) -> Dict[str, Tensor]:
   #     """
   #     Physics diagnostics on a *single batch* (v3.2):
   #     - uses physics-driven mean settlement path via stable consolidation
   #       integration (exact-step),
   #     - supports a learnable Q-head fed into the GW residual.
   #     """
   #     # --- Validate presence of H_field/soil_thickness and extract coords ---
   #     H_field_in = get_tensor_from(
   #         inputs, "H_field", "soil_thickness",
   #         auto_convert=True,
   #     )
   #     if H_field_in is None:
   #         raise ValueError(
   #             "_evaluate_physics_on_batch() requires 'H_field' "
   #             "(or 'soil_thickness') in `inputs`."
   #         )
   #     H_field = tf_convert_to_tensor(H_field_in, dtype=tf_float32)
   
   #     coords = _get_coords(inputs)
   #     t, x, y = extract_txy_in(coords)
   #     coords_flat = tf_concat([t, x, y], axis=-1)
   
   #     # ------------------------------------------------------------------
   #     # Small local helpers (keep this method standalone & robust)
   #     # ------------------------------------------------------------------
   #     eps = tf_constant(1e-12, dtype=tf_float32)
   
   #     def _seconds_per_unit(time_units: Optional[str]) -> Tensor:
   #         # seconds_per_time_unit isn't imported here; use rate_to_per_second(1)
   #         one = tf_constant(1.0, dtype=tf_float32)
   #         inv = rate_to_per_second(one, time_units)  # (1/unit) -> (1/s)
   #         return one / (inv + eps)
   
   #     def _as_B11(v: Tensor) -> Tensor:
   #         v = tf_cast(v, tf_float32)
   #         r = tf_rank(v)
   #         return tf_cond(
   #             tf_equal(r, 1),
   #             lambda: v[:, None, None],
   #             lambda: tf_cond(
   #                 tf_equal(r, 2),
   #                 lambda: v[:, :, None],
   #                 lambda: v,
   #             ),
   #         )
   
   #     def _get_s_init_si(like_BH1: Tensor) -> Tensor:
   #         """Best-effort initial condition s_T in SI meters (B,1,1)."""
   #         if inputs is None:
   #             return tf_zeros_like(like_BH1[:, :1, :])
   
   #         # Prefer explicit SI keys (no extra scaling)
   #         for k in (
   #             "s_init_si", "subs_init_si", "subs_hist_last_si",
   #             "subsidence_cum_last_si", "subsidence_cum_hist_last_si",
   #         ):
   #             if (k in inputs) and (inputs[k] is not None):
   #                 return _as_B11(inputs[k])
   
   #         # Non-SI / model-scale keys: convert using model scaling
   #         for k in (
   #             "s_init", "subs_init", "subs_hist_last",
   #             "subsidence_cum_last", "subsidence_cum_hist_last",
   #         ):
   #             if (k in inputs) and (inputs[k] is not None):
   #                 return self._to_si_subsidence(_as_B11(inputs[k]))
   
   #         # If you later decide to pack last historical subsidence inside
   #         # dynamic_features, you can set scaling_kwargs["subs_dyn_index"]
   #         # and we will pick it up here.
   #         sk = self.scaling_kwargs or {}
   #         if (
   #             ("dynamic_features" in inputs)
   #             and (inputs["dynamic_features"] is not None)
   #             and ("subs_dyn_index" in sk)
   #         ):
   #             Xh = tf_cast(inputs["dynamic_features"], tf_float32)
   #             idx = int(sk["subs_dyn_index"])
   #             F = tf_shape(Xh)[-1]
   #             tf_debugging.assert_less(
   #                 tf_cast(idx, tf_int32), F,
   #                 message="subs_dyn_index out of range for inputs['dynamic_features']",
   #             )
   #             s_last = Xh[:, -1:, idx:idx + 1]  # (B,1,1)
   #             return self._to_si_subsidence(s_last)
   
   #         return tf_zeros_like(like_BH1[:, :1, :])
   
   #     def _infer_dt_units(t_BH1: Tensor) -> Tensor:
   #         """
   #         dt per forecast step in *time_units* with shape (B,H,1).
   #         - If t is absolute (e.g., year), we use diffs.
   #         - First step dt is approximated by the first diff.
   #         - If coords are normalized, we de-normalize using coord_ranges['t'].
   #         """
   #         B = tf_shape(t_BH1)[0]
   #         H = tf_shape(t_BH1)[1]
   
   #         # Default dt = 1 (in time_units)
   #         dt0 = tf_ones([B, 1, 1], dtype=tf_float32)
   
   #         # If H==1: keep dt=1
   #         def _dt_from_diffs():
   #             diffs = t_BH1[:, 1:, :] - t_BH1[:, :-1, :]          # (B,H-1,1)
   #             dt_first = diffs[:, :1, :]                           # (B,1,1)
   #             dt_full = tf_concat([dt_first, diffs], axis=1)       # (B,H,1)
   #             return dt_full
   
   #         dt = tf_cond(tf_greater(H, 1), _dt_from_diffs, lambda: dt0)
   
   #         # De-normalize time if needed
   #         sk = self.scaling_kwargs or {}
   #         if bool(sk.get("coords_normalized", False)):
   #             cr = self._coord_ranges()
   #             if cr is not None and ("t" in cr):
   #                 tR = tf_cast(cr["t"][1] - cr["t"][0], tf_float32)
   #                 dt = dt * tR
   
   #         return tf_maximum(dt, eps)
   
   #     def _integrate_consolidation_exact(
   #         h_si_BH1: Tensor,
   #         Ss_BH1: Tensor,
   #         H_si_BH1: Tensor,
   #         tau_sec_BH1: Tensor,
   #         h_ref_si_BH1: Tensor,
   #         s_init_si_B11: Tensor,
   #         dt_units_BH1: Tensor,
   #     ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
   #         """
   #         Exact-step integration:
   #             s_{n+1} = s_n * exp(-dt/tau) + s_eq_n * (1 - exp(-dt/tau))
   #         Returns: s_bar(B,H,1), s_eq(B,H,1), dt_sec(B,H,1), alpha(B,H,1).
   #         """
   #         sec_per_unit = _seconds_per_unit(self.time_units)
   #         dt_sec = dt_units_BH1 * sec_per_unit                           # (B,H,1)
   #         tau_safe = tf_maximum(tau_sec_BH1, eps)
   #         alpha = tf_exp(-dt_sec / tau_safe)                               # (B,H,1)
   
   #         delta_h = tf_maximum(h_ref_si_BH1 - h_si_BH1, 0.0)
   #         s_eq = Ss_BH1 * delta_h * H_si_BH1                               # (B,H,1)
   
   #         # tf.scan expects time-major
   #         tf_transpose = getattr(KERAS_DEPS, "transpose", None)
   #         tf_scan = getattr(KERAS_DEPS, "scan", None)
   
   #         if tf_transpose is None or tf_scan is None:
   #             # Eager fallback (rare): python loop
   #             s_prev = s_init_si_B11[:, 0:1, :]                            # (B,1,1)
   #             outs = []
   #             for i in range(int(h_si_BH1.shape[1] or 1)):
   #                 a = alpha[:, i:i + 1, :]
   #                 se = s_eq[:, i:i + 1, :]
   #                 s_prev = a * s_prev + (1.0 - a) * se
   #                 outs.append(s_prev)
   #             s_bar = tf_concat(outs, axis=1) if outs else tf_zeros_like(h_si_BH1)
   #             return s_bar, s_eq, dt_sec, alpha
   
   #         alpha_T = tf_transpose(alpha, perm=[1, 0, 2])                    # (H,B,1)
   #         s_eq_T = tf_transpose(s_eq, perm=[1, 0, 2])                      # (H,B,1)
   #         s0 = tf_cast(s_init_si_B11[:, 0, :], tf_float32)                 # (B,1)
   
   #         def step(s_prev, elems):
   #             a_t, s_eq_t = elems
   #             return a_t * s_prev + (1.0 - a_t) * s_eq_t
   
   #         s_T = tf_scan(step, (alpha_T, s_eq_T), initializer=s0)           # (H,B,1)
   #         s_bar = tf_transpose(s_T, perm=[1, 0, 2])                        # (B,H,1)
   #         return s_bar, s_eq, dt_sec, alpha
   
   #     # ------------------------------------------------------------------
   #     # Physics OFF path
   #     # ------------------------------------------------------------------
   #     if "none" in self.pde_modes_active:
   #         zeros = tf_constant(0.0, dtype=tf_float32)
   #         out = {
   #             "loss_physics": zeros,
   #             "loss_consolidation": zeros,
   #             "loss_gw_flow": zeros,
   #             "loss_prior": zeros,
   #             "loss_smooth": zeros,
   #             "loss_mv": zeros,
   #             "loss_bounds": zeros,
   #         }
   #         if return_maps:
   #             out.update(
   #                 {
   #                     "R_cons": tf_zeros_like(t),
   #                     "R_gw": tf_zeros_like(t),
   #                     "K_field": tf_zeros_like(t),
   #                     "Ss_field": tf_zeros_like(t),
   #                     "tau_field": tf_zeros_like(t),
   #                 }
   #             )
   #         return out
   
   #     # --- Need derivatives wrt t,x,y (and 2nd derivatives like in train_step)
   #     with GradientTape(persistent=True) as tape:
   #         tape.watch(coords_flat)
   
   #         outputs = self(inputs, training=False)
   
   #         # ---- Data predictions: (subs, gwl/head) mean ----
   #         data_mean_raw = outputs.get("data_mean_raw", None)
   #         if data_mean_raw is None:
   #             raise ValueError("Model outputs missing 'data_mean_raw'.")
   
   #         subs_mean_raw, gwl_mean_raw = self.split_data_predictions(data_mean_raw)
   
   #         # Latent head: convert to SI and then to head if gwl is depth-bgs
   #         h_model = self._to_si_head(gwl_mean_raw)
   #         h_si = self._gwl_to_head_m(h_model, inputs=inputs)
   
   #         # ---- Physics predictions: K, Ss, tau (+ optional Q) ----
   #         phys_mean_raw = outputs.get("phys_mean_raw", None)
   #         if phys_mean_raw is None:
   #             raise ValueError("Model outputs missing 'phys_mean_raw'.")
   
   #         split = self.split_physics_predictions(phys_mean_raw)
   #         if isinstance(split, (tuple, list)) and len(split) == 6:
   #             _s_ignore, _h_ignore, K_base, Ss_base, tau_base, Q_base = split
   #         elif isinstance(split, (tuple, list)) and len(split) == 5:
   #             _s_ignore, _h_ignore, K_base, Ss_base, tau_base = split
   #             Q_base = None
   #         else:
   #             # Very defensive fallback
   #             K_base = phys_mean_raw[:, :, :1]
   #             Ss_base = phys_mean_raw[:, :, 1:2]
   #             tau_base = phys_mean_raw[:, :, 2:3]
   #             Q_base = phys_mean_raw[:, :, 3:4] if tf_shape(phys_mean_raw)[-1] >= 4 else None
   
   #         # Physical fields in SI (K, Ss, tau, etc.)
   #         H_si = self._to_si_thickness(H_field)
   #         (
   #             K_field, Ss_field, tau_field, tau_phys, Hd_eff,
   #             delta_log_tau, log_tau_prior,
   #         ) = self._compose_physics_fields(
   #             coords=coords_flat,
   #             H_field=H_si,
   #             K_base=K_base,
   #             Ss_base=Ss_base,
   #             tau_base=tau_base,
   #         )
   
   #         # --- First derivatives of head ---
   #         dh_dcoords = tape.gradient(h_si, coords_flat)
   #         dh_dt, dh_dx, dh_dy = extract_txy_in(dh_dcoords)
   
   #         # Build K * grad(h) and second derivatives (divergence)
   #         K_dh_dx = K_field * dh_dx
   #         K_dh_dy = K_field * dh_dy
   
   #         dKdh_dx_dcoords = tape.gradient(K_dh_dx, coords_flat)
   #         dKdh_dy_dcoords = tape.gradient(K_dh_dy, coords_flat)
   
   #         _, d_K_dh_dx_dx, _ = extract_txy_in(dKdh_dx_dcoords)
   #         _, _, d_K_dh_dy_dy = extract_txy_in(dKdh_dy_dcoords)
   
   #         # Smoothness (spatial grads of K and Ss)
   #         dK_dcoords = tape.gradient(K_field, coords_flat)
   #         dSs_dcoords = tape.gradient(Ss_field, coords_flat)
   #         _, dK_dx, dK_dy = extract_txy_in(dK_dcoords)
   #         _, dSs_dx, dSs_dy = extract_txy_in(dSs_dcoords)
   
   #     # no longer needed
   #     del tape
   
   #     # ------------------------------------------------------------------
   #     # Chain-rule scaling (normalized coords and/or degrees)
   #     # ------------------------------------------------------------------
   #     sk = self.scaling_kwargs or {}
   #     coords_norm = bool(sk.get("coords_normalized", False))
   #     coords_deg = bool(sk.get("coords_in_degrees", False))
   
   #     if coords_norm:
   #         ranges = self._coord_ranges()
   #         if ranges is None:
   #             raise ValueError(
   #                 "coords_normalized=True but coord_ranges missing "
   #                 "in scaling_kwargs."
   #             )
   #         tR = tf_cast(ranges["t"][1] - ranges["t"][0], tf_float32)
   #         xR = tf_cast(ranges["x"][1] - ranges["x"][0], tf_float32)
   #         yR = tf_cast(ranges["y"][1] - ranges["y"][0], tf_float32)
   
   #         dh_dt = dh_dt / (tR + eps)
   #         dh_dx = dh_dx / (xR + eps)
   #         dh_dy = dh_dy / (yR + eps)
   
   #         d_K_dh_dx_dx = d_K_dh_dx_dx / ((xR * xR) + eps)
   #         d_K_dh_dy_dy = d_K_dh_dy_dy / ((yR * yR) + eps)
   
   #         dK_dx = dK_dx / (xR + eps)
   #         dK_dy = dK_dy / (yR + eps)
   #         dSs_dx = dSs_dx / (xR + eps)
   #         dSs_dy = dSs_dy / (yR + eps)
   
   #     if coords_deg:
   #         lon_m = tf_cast(sk.get("deg_to_m_lon", 1.0), tf_float32)
   #         lat_m = tf_cast(sk.get("deg_to_m_lat", 1.0), tf_float32)
   
   #         dh_dx = dh_dx / (lon_m + eps)
   #         dh_dy = dh_dy / (lat_m + eps)
   
   #         d_K_dh_dx_dx = d_K_dh_dx_dx / ((lon_m * lon_m) + eps)
   #         d_K_dh_dy_dy = d_K_dh_dy_dy / ((lat_m * lat_m) + eps)
   
   #         dK_dx = dK_dx / (lon_m + eps)
   #         dK_dy = dK_dy / (lat_m + eps)
   #         dSs_dx = dSs_dx / (lon_m + eps)
   #         dSs_dy = dSs_dy / (lat_m + eps)
   
   #     # time-derivative per second
   #     dh_dt = rate_to_per_second(dh_dt, self.time_units)
   
   #     # ------------------------------------------------------------------
   #     # Q-head -> SI (1/s)
   #     # ------------------------------------------------------------------
   #     if Q_base is None:
   #         Q_si = tf_zeros_like(d_K_dh_dx_dx)
   #     else:
   #         Q_base = tf_cast(Q_base, tf_float32)
   
   #         # Optional: if user declares Q is w.r.t normalized time, de-normalize
   #         if coords_norm and bool(sk.get("Q_wrt_normalized_time", False)):
   #             ranges = self._coord_ranges()
   #             tR = tf_cast(ranges["t"][1] - ranges["t"][0], tf_float32)
   #             Q_base = Q_base / (tR + eps)
   
   #         # Convert from 1/time_unit to 1/s (default)
   #         if bool(sk.get("Q_in_per_second", False)) or bool(sk.get("Q_in_si", False)):
   #             Q_si = Q_base
   #         else:
   #             Q_si = rate_to_per_second(Q_base, self.time_units)
   
   #         # Broadcast if needed
   #         if getattr(Q_si, "shape", None) is not None and Q_si.shape.rank == 2:
   #             Q_si = Q_si[:, :, None]
   #         Q_si = Q_si + tf_zeros_like(d_K_dh_dx_dx)
   
   #     # ------------------------------------------------------------------
   #     # Physics-driven mean settlement path (exact-step integration)
   #     # ------------------------------------------------------------------
   #     dt_units = _infer_dt_units(t)
   #     h_ref_si = get_h_ref_si(self, inputs, like=h_si)
   #     s_init_si = _get_s_init_si(h_si)
   
   #     s_bar, s_eq, dt_sec, alpha = _integrate_consolidation_exact(
   #         h_si_BH1=h_si,
   #         Ss_BH1=Ss_field,
   #         H_si_BH1=H_si,
   #         tau_sec_BH1=tau_field,
   #         h_ref_si_BH1=h_ref_si,
   #         s_init_si_B11=s_init_si,
   #         dt_units_BH1=dt_units,
   #     )
   
   #     # Consolidation step residual (m/s) using exact-step consistency
   #     # (will be ~0 if s_bar is generated with the same exact stepper)
   #     s_prev_seq = tf_concat([s_init_si, s_bar[:, :-1, :]], axis=1)
   #     s_pred_step = alpha * s_prev_seq + (1.0 - alpha) * s_eq
   #     cons_step_m = s_bar - s_pred_step
   #     cons_res = cons_step_m / (tf_maximum(dt_sec, eps))  # m/s
   
   #     # GW flow residual with forcing Q (1/s)
   #     gw_res = self._compute_gw_flow_residual(
   #         dh_dt=dh_dt,
   #         d_K_dh_dx_dx=d_K_dh_dx_dx,
   #         d_K_dh_dy_dy=d_K_dh_dy_dy,
   #         Ss_field=Ss_field,
   #         Q=Q_si,
   #     )
   
   #     # Prior residual (dimensionless in log-space)
   #     prior_res = delta_log_tau
   
   #     # Smoothness residual (units depend on field scales; keep as diagnostic)
   #     smooth_res = tf_square(dK_dx) + tf_square(
   #         dK_dy) + tf_square(dSs_dx) + tf_square(dSs_dy)
   
   #     # MV prior residual
   #     mv_res = self._compute_mv_prior_residual(Ss_field)
   
   #     # Bounds residual (if enabled)
   #     bounds_res = self._compute_bounds_residual(K_field, Ss_field, Hd_eff, tau_field)
   
   #     # ------------------------------------------------------------------
   #     # Optional nondimensionalization (recommended)
   #     # ------------------------------------------------------------------
   #     scales = self._compute_scales(
   #         h_mean=h_si,
   #         s_mean=s_bar,
   #         dt=dt_units,
   #         K=K_field,
   #         Ss=Ss_field,
   #         Q=Q_si,
   #         ds_dt=None,  # v3.2: no autodiff ds/dt
   #     )
   
   #     cons_res_scaled = scale_residual(cons_res, scales["cons_scale"])
   #     gw_res_scaled = scale_residual(gw_res, scales["gw_scale"])
   
   #     # ------------------------------------------------------------------
   #     # Losses + eps diagnostics
   #     # ------------------------------------------------------------------
   #     loss_cons = tf_reduce_mean(tf_square(cons_res_scaled))
   #     loss_gw = tf_reduce_mean(tf_square(gw_res_scaled))
   #     loss_physics = loss_cons + loss_gw
   
   #     loss_prior = tf_reduce_mean(tf_square(prior_res))
   #     loss_smooth = tf_reduce_mean(smooth_res)
   #     loss_mv = tf_reduce_mean(tf_square(mv_res))
   #     loss_bounds = tf_reduce_mean(tf_square(bounds_res))
   
   #     eps_prior = self._rms(prior_res)
   #     eps_cons = self._rms(cons_res)
   #     eps_gw = self._rms(gw_res)
   
   #     out = {
   #         "loss_physics": loss_physics,
   #         "loss_consolidation": loss_cons,
   #         "loss_gw_flow": loss_gw,
   #         "loss_prior": loss_prior,
   #         "loss_smooth": loss_smooth,
   #         "loss_mv": loss_mv,
   #         "loss_bounds": loss_bounds,
   #         "eps_prior": eps_prior,
   #         "eps_cons": eps_cons,
   #         "eps_gw": eps_gw,
   #     }
   
   #     if return_maps:
   #         out.update(
   #             {
   #                 # physics states
   #                 "h_si": h_si,
   #                 "h_ref_si": h_ref_si,
   #                 "s_bar": s_bar,
   #                 "s_eq": s_eq,
   #                 "s_init_si": s_init_si,
   #                 "dt_units": dt_units,
   #                 "Q_si": Q_si,
   #                 # fields
   #                 "K_field": K_field,
   #                 "Ss_field": Ss_field,
   #                 "tau_field": tau_field,
   #                 "tau_phys": tau_phys,
   #                 "Hd_eff": Hd_eff,
   #                 # residuals (raw + scaled)
   #                 "R_cons": cons_res,
   #                 "R_gw": gw_res,
   #                 "R_prior": prior_res,
   #                 "R_smooth": smooth_res,
   #                 "R_mv": mv_res,
   #                 "R_bounds": bounds_res,
   #                 "R_cons_scaled": cons_res_scaled,
   #                 "R_gw_scaled": gw_res_scaled,
   #                 # extras (useful for debugging)
   #                 "subs_mean_raw": subs_mean_raw,
   #                 "gwl_mean_raw": gwl_mean_raw,
   #             }
   #         )
   
   #     return out

You’re right to lock in **Option 1** for the Nature-paper story: *physics produces the central (mean) subsidence path*, and the network learns *uncertainty/residual structure around it*. That’s the only option where physics **cannot be ignored**.

### Why I “revised Option 1 → Option 2” earlier

I did that for **backward compatibility** with your existing v3.x contract:

* your current `call()` returns `"data_final" / "data_mean" / "phys_mean_raw"` and downstream code expects `split_data_predictions()` to produce `(s, gwl)`;
* Option 2 preserves “model predicts mean subsidence + mean gwl” so all old dashboards/metrics keep working;
* Option 2 also keeps a non-trivial `cons_res` (so people don’t panic when it’s ~0).

But since you explicitly want **a new architecture** where physics structurally drives subsidence (Option 1), we should **stop trying to align** and redesign the output contract.

---

## 1) Option 1 architecture (what the network must output)

### Network outputs (forecast window only, shape (B,H,…))

1. **Latent mean head**
   `h_raw`  → later converted to `h_si` (meters)

2. **GW forcing**
   `Q_raw`  → later converted to `Q_si` (1/s)

3. **Physics fields (logits / raw)**
   `K_raw`, `Ss_raw`, `delta_log_tau_raw`
   (You can keep Hd implicit as `Hd_eff = Hd_factor * H_field` for now, or add an `Hd_head` later.)

4. **Subsidence residual quantiles** (NOT mean subsidence)
   `r_q_raw` with shape `(B,H,Q,subs_dim)` (or `(B,H,subs_dim)` if `quantiles is None`)

### Physics-driven final prediction

Compute physics mean path in SI:

* `s_bar_si = IntegratorExact(h_si, K, Ss, tau, Hd_eff, h_ref, s_init, dt)`

Then **final subsidence quantiles** are:

* `s_q_si = s_bar_si + r_q_si`

And the **data loss** is pinball on `s_q` (either in model space or SI; both are affine-equivalent—see below).

---

## 2) Where the physics lives (recommended wiring)

### Recommended: compute `s_bar` **inside `call()`**

Because you want inference mode A (fully model-driven) to output subsidence without requiring the training loop.

So `call()` should do:

* decode latent states: `h_raw`, `Q_raw`, `K_raw`, `Ss_raw`, `delta_log_tau_raw`, `r_q_raw`
* convert to SI: `h_si`, `Q_si`
* build positive fields: `K_field`, `Ss_field`, `tau_field` via `compose_physics_fields(...)`
* compute `s_bar_si` via `integrate_consolidation_mean(..., method="exact")`
* output `subs_final` (quantiles) as `s_q_si` (or as model-space if you add inverse affine helper)

**Key point about “cons residual ≈ 0”:**

* If you integrate with the *exact step* and then compute the “step residual” using the *same exact step*, it will be ~0 by construction.
* That’s fine in Option 1 because **you are not using `L_cons` to enforce consolidation anymore**; consolidation enforces itself by *defining the mean path used in the forecast*.
* Practically: set `lambda_cons = 0` (or keep a diagnostic-only term computed from a *different* discretization, e.g., Euler residual vs exact-integrated path).

---

## 3) Minimal rewrite plan (concrete)

### A) Replace output contract

Stop returning `"data_final"` as `(subs,gwl)`.

Instead return (example):

```python
return {
    # main supervised output
    "subs_final_si": s_q_si,          # (B,H,Q,subs_dim) or (B,H,subs_dim)

    # optional debugging/diagnostics
    "subs_bar_si": s_bar_si,          # (B,H,subs_dim)
    "h_si": h_si,                     # (B,H,1)
    "Q_si": Q_si,                     # (B,H,1)

    # raw physics for inspection
    "phys_raw": tf_concat([K_raw, Ss_raw, delta_log_tau_raw, Q_raw], axis=-1),
}
```

And **compile/train only against** `"subs_final_si"` (or `"subs_final"` if you keep it in model space).

### B) New heads you must add

In `_build_physics_layers()` (Option 1 mode):

* `self.h_head = Dense(1, name="h_head")`
* `self.Q_head = Dense(1, name="Q_head")`
* keep `K_head`, `Ss_head`, `tau_head` (tau_head outputs `delta_log_tau`)
* new residual quantile head:

  * if `quantiles`: Dense(`subs_dim * nQ`) and reshape to `(B,H,nQ,subs_dim)`
  * else: Dense(`subs_dim`) giving `(B,H,subs_dim)`

### C) Train step becomes simpler

Now the supervised part is just:

* `y_pred = {"subs_pred": outputs["subs_final_(si or model)"]}`
* data loss is only subsidence pinball.

Physics loss remains:

* GW PDE residual (still needs tape + coords)
* prior/bounds/smooth/mv terms (same as now)
* consolidation loss is optional / diagnostic-only in Option 1.

---

## 4) RMS helper you asked for (`to_rms`) to move into `_geoprior_maths.py`

Add this in `_geoprior_maths.py` (near other small scalar helpers):

```python
def to_rms(
    x: Tensor,
    axis: Optional[Sequence[int]] = None,
    keepdims: bool = False,
    eps: float = 1e-12,
) -> Tensor:
    """
    Root-mean-square of a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis : sequence of int, optional
        Axes to reduce. If None, reduce over all axes.
    keepdims : bool
        Keep reduced dimensions.
    eps : float
        Numerical epsilon.

    Returns
    -------
    Tensor
        RMS value(s).
    """
    x = tf_cast(x, tf_float32)
    e = tf_constant(float(eps), tf_float32)
    return tf_sqrt(tf_reduce_mean(tf_square(x), axis=axis, keepdims=keepdims) + e)
```

Then replace:

```python
eps_prior = self._rms(prior_res)
```

with:

```python
eps_prior = to_rms(prior_res)
```

---

## 5) One important practical note (loss space: SI vs normalized)

Option 1 says “training stable in normalized space but physics in SI”.

You have two valid implementations:

### Preferred (clean): **compute pinball loss in model space**

To do that you need inverse affine helpers:

* `from_si_subsidence(s_si) = (s_si - b)/a`
* same for head if needed.

Then:

* `s_bar_model = from_si_subsidence(s_bar_si)`
* `r_q_model = r_q_raw` (already model space)
* `s_q_model = s_bar_model + r_q_model`
* pinball compares `targets (model)` vs `s_q_model`

### Also OK: compute pinball in SI

* convert `targets` to SI (`to_si_subsidence`)
* use `s_q_si` directly

This only changes gradient scale by a constant factor (`a`), but if your SI scale is huge it can make tuning harder. That’s why I recommend adding the inverse affine helper and keeping the pinball term in model space.

---

If you want, I can write the **exact Option-1 `call()` skeleton** (heads + reshape + `compose_physics_fields()` + `integrate_consolidation_mean()` + residual addition) using *your existing helpers* exactly as they exist in `_geoprior_utils.py/_geoprior_maths.py`, so you can paste it in and iterate.
