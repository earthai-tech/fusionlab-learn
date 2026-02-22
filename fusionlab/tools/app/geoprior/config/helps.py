# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Dict, Optional, Tuple


HelpKey = Tuple[str, Optional[str]]


_PHYS_HELP: Dict[HelpKey, str] = {
    ("bounds_mode", None): (
        "How parameter bounds are enforced.\n"
        "\n"
        "soft:\n"
        "- adds a smooth penalty when a learned field exceeds\n"
        "  configured bounds.\n"
        "\n"
        "hard:\n"
        "- clamps / hard-enforces bounds (stronger constraint),\n"
        "  which can hurt optimization if bounds are too tight.\n"
    ),
    ("lambda_bounds", None): (
        "Strength of the bounds penalty term.\n"
        "Higher values enforce bounds more aggressively.\n"
    ),
    ("physics_bounds", "K_min"): (
        "Lower bound for hydraulic conductivity K.\n"
        "Typical SI unit: m/s.\n"
    ),
    ("physics_bounds", "K_max"): (
        "Upper bound for hydraulic conductivity K.\n"
        "Typical SI unit: m/s.\n"
    ),
    ("physics_bounds", "Ss_min"): (
        "Lower bound for specific storage Ss.\n"
        "Typical SI unit: 1/m.\n"
    ),
    ("physics_bounds", "Ss_max"): (
        "Upper bound for specific storage Ss.\n"
        "Typical SI unit: 1/m.\n"
    ),
    ("physics_bounds", "tau_min"): (
        "Lower bound for consolidation time scale tau.\n"
        "Unit follows your time_units (e.g. days or years).\n"
    ),
    ("physics_bounds", "tau_max"): (
        "Upper bound for consolidation time scale tau.\n"
        "Unit follows your time_units (e.g. days or years).\n"
    ),
    ("training_strategy", None): (
        "Controls how training phases weight data vs physics.\n"
        "\n"
        "physics_first:\n"
        "- stabilize physics residual terms early, then blend in\n"
        "  data-driven losses.\n"
        "\n"
        "data_first:\n"
        "- fit data first, then gradually increase physics.\n"
    ),
    ("q_policy_physics_first", None): (
        "Schedule for activating groundwater forcing Q.\n"
        "\n"
        "up_off: keep Q off initially.\n"
        "up_on:  Q starts on and can ramp.\n"
        "always: Q always active.\n"
        "\n"
        "Warmup/ramp control when Q becomes influential.\n"
    ),
    ("q_warmup_epochs_physics_first", None): (
        "Number of epochs where Q influence stays minimal.\n"
        "After warmup, ramp begins.\n"
    ),
    ("q_ramp_epochs_physics_first", None): (
        "Number of epochs used to ramp Q influence from low to\n"
        "target strength.\n"
    ),
    ("lambda_q_physics_first", None): (
        "Target weight for Q-related physics contribution.\n"
        "Higher values force the model to use Q more strongly.\n"
    ),
    ("eval_json_units_mode", None): (
        "Units used in exported evaluation JSON.\n"
        "\n"
        "interpretable:\n"
        "- exports values in human-friendly domain units.\n"
        "\n"
        "si:\n"
        "- exports in strict SI units (can look small/large).\n"
    ),
}

# ---------------------------------------------------------------------
# Auto-generated physics help (filled, missing-only).
# Paste this block under your existing _PHYS_HELP = {...}.
# ---------------------------------------------------------------------

_PHYS_HELP.update({

    # =============================================================
    # Physics inputs & conventions
    # =============================================================
    ("gwl_kind", None): (
        "How groundwater level (GWL) is represented in the\n"
        "dataset.\n"
        "\n"
        "depth_bgs:\n"
        "- depth below ground surface (positive magnitude).\n"
        "\n"
        "head:\n"
        "- hydraulic head (elevation) already provided.\n"
        "\n"
        "This choice affects head conversion and residuals.\n"
    ),
    ("gwl_sign", None): (
        "Sign convention for the GWL feature.\n"
        "\n"
        "down_positive:\n"
        "- depth below surface increases downward.\n"
        "\n"
        "up_positive:\n"
        "- elevation-like convention (increases upward).\n"
        "\n"
        "Use a convention consistent with your Stage-1 data.\n"
    ),
    ("use_head_proxy", None): (
        "Whether to compute a head proxy from GWL inputs.\n"
        "\n"
        "When enabled, the pipeline derives an approximate\n"
        "head using z_surf_col and the configured sign.\n"
        "This is useful when GWL is given as depth_bgs.\n"
    ),
    ("z_surf_col", None): (
        "Name of the surface elevation column used for head\n"
        "conversion.\n"
        "\n"
        "Example: z_surf_m or dem_m.\n"
        "Set to None if surface elevation is unavailable.\n"
    ),
    ("include_z_surf_as_static", None): (
        "If True, include surface elevation (z_surf_col) as a\n"
        "static input feature.\n"
        "\n"
        "This can help the model learn spatial structure when\n"
        "topography correlates with groundwater dynamics.\n"
    ),
    ("head_col", None): (
        "Column name to use for hydraulic head.\n"
        "\n"
        "If the dataset already contains head, set head_col to\n"
        "that column name. If using head proxy, head_col may\n"
        "be the derived output name.\n"
    ),
    ("gwl_dyn_index", None): (
        "Index of the GWL variable inside the dynamic feature\n"
        "tensor.\n"
        "\n"
        "Use this when the model expects a fixed ordering of\n"
        "dynamic channels.\n"
    ),
    ("geoprior_use_effective_h", None): (
        "If True, use the effective thickness (H_eff) field\n"
        "for physics closures instead of a raw thickness proxy.\n"
        "\n"
        "This can stabilize consolidation terms when thickness\n"
        "is heterogeneous.\n"
    ),

    # =============================================================
    # PDE / residual engine
    # =============================================================
    ("pde_mode", None): (
        "Which physics residuals are active during training.\n"
        "\n"
        "off/none:\n"
        "- no physics residuals.\n"
        "\n"
        "consolidation:\n"
        "- consolidation (subsidence) residual only.\n"
        "\n"
        "gw_flow:\n"
        "- groundwater flow residual only.\n"
        "\n"
        "both/on:\n"
        "- enable both residuals.\n"
    ),
    ("residual_method", None): (
        "How residual derivatives are computed.\n"
        "\n"
        "exact:\n"
        "- automatic differentiation (preferred when stable).\n"
        "\n"
        "fd:\n"
        "- finite differences (useful as a fallback).\n"
    ),
    ("physics_warmup_steps", None): (
        "Number of optimizer steps to keep physics influence\n"
        "minimal at the start.\n"
        "\n"
        "Used to avoid early training instability when physics\n"
        "terms are noisy.\n"
    ),
    ("physics_ramp_steps", None): (
        "Number of steps used to ramp physics influence from\n"
        "low to target strength.\n"
        "\n"
        "A larger value makes the transition smoother.\n"
    ),
    ("scale_pde_residuals", None): (
        "If True, rescale PDE residual terms using internal\n"
        "normalization so magnitudes are comparable.\n"
        "\n"
        "Recommended when residual units differ strongly.\n"
    ),
    ("allow_subs_residual", None): (
        "If True, enable the subsidence (consolidation) residual\n"
        "path when available.\n"
        "\n"
        "Disable to focus on groundwater flow only.\n"
    ),
    ("dt_min_units", None): (
        "Minimum time step (in time_units) used in residual\n"
        "computations.\n"
        "\n"
        "Acts as a safety floor to avoid division by very small\n"
        "dt when time spacing is irregular.\n"
    ),

    # =============================================================
    # Base physics loss weights
    # =============================================================
    ("lambda_cons", None): (
        "Base weight for the consolidation residual term.\n"
        "\n"
        "Higher values enforce subsidence physics more strongly.\n"
    ),
    ("lambda_gw", None): (
        "Base weight for the groundwater flow residual term.\n"
        "\n"
        "Higher values enforce Darcy-style consistency.\n"
    ),
    ("lambda_prior", None): (
        "Base weight for prior consistency penalties.\n"
        "\n"
        "Used to keep learned effective fields close to priors.\n"
    ),
    ("lambda_smooth", None): (
        "Base weight for spatial smoothness regularization.\n"
        "\n"
        "Higher values encourage smoother learned fields.\n"
    ),
    ("lambda_mv", None): (
        "Base weight for MV / kappa related penalties.\n"
        "\n"
        "Controls strength of MV prior and schedule effects.\n"
    ),
    ("lambda_q", None): (
        "Base weight for Q (source-term) related physics.\n"
        "\n"
        "Used when Q forcing is enabled by the active policy.\n"
    ),

    # =============================================================
    # Global physics-loss offset (v3.2)
    # =============================================================
    ("offset_mode", None): (
        "How the global physics offset is applied.\n"
        "\n"
        "mul:\n"
        "- multiply physics weights by lambda_offset.\n"
        "\n"
        "log10:\n"
        "- apply an exponent shift (10**lambda_offset).\n"
        "\n"
        "Use log10 when you want decade-level scaling.\n"
    ),
    ("lambda_offset", None): (
        "Global offset applied to physics loss weights.\n"
        "\n"
        "This is a single knob to make physics overall stronger\n"
        "or weaker without changing each lambda_* term.\n"
    ),
    ("use_lambda_offset_scheduler", None): (
        "If True, schedule the global offset over epochs or\n"
        "steps.\n"
        "\n"
        "Use this to gradually introduce physics strength.\n"
    ),
    ("lambda_offset_unit", None): (
        "Unit for the offset scheduler.\n"
        "\n"
        "epoch: schedule is keyed by epoch.\n"
        "step:  schedule is keyed by optimizer step.\n"
    ),
    ("lambda_offset_when", None): (
        "When to apply the scheduler update inside the loop.\n"
        "\n"
        "begin: update at the start of the unit.\n"
        "end:   update at the end of the unit.\n"
    ),
    ("lambda_offset_warmup", None): (
        "Warmup length (in lambda_offset_unit) during which the\n"
        "offset stays near lambda_offset_start.\n"
    ),
    ("lambda_offset_start", None): (
        "Starting value for the offset when scheduling.\n"
        "\n"
        "Used during warmup and as the ramp start.\n"
    ),
    ("lambda_offset_end", None): (
        "Final value for the offset when scheduling.\n"
        "\n"
        "Used as the ramp end or default steady value.\n"
    ),
    ("lambda_offset_schedule", None): (
        "Explicit schedule map for the offset.\n"
        "\n"
        "Format: {unit_index: value} where unit_index is an\n"
        "epoch or step depending on lambda_offset_unit.\n"
        "Set to None to use a simple warmup+linear ramp.\n"
    ),

    # =============================================================
    # MV / kappa priors & schedules
    # =============================================================
    ("mv_lr_mult", None): (
        "Learning-rate multiplier for MV-related parameters.\n"
        "\n"
        "Use values < 1.0 to make MV updates more conservative.\n"
    ),
    ("kappa_lr_mult", None): (
        "Learning-rate multiplier for kappa-related parameters.\n"
        "\n"
        "Lower values can stabilize training when kappa is noisy.\n"
    ),
    ("geoprior_init_mv", None): (
        "Initial MV value used to initialize the MV pathway.\n"
        "\n"
        "Unit depends on mv_prior_units and mv_prior_mode.\n"
    ),
    ("geoprior_init_kappa", None): (
        "Initial kappa value used to initialize the kappa path.\n"
        "\n"
        "Choose a physically reasonable starting point.\n"
    ),
    ("geoprior_gamma_w", None): (
        "Weighting factor used in MV / kappa coupling.\n"
        "\n"
        "This controls how strongly the model trusts MV-derived\n"
        "structure during optimization.\n"
    ),
    ("geoprior_h_ref", None): (
        "Reference head used to compute drawdown or head change.\n"
        "\n"
        "Typical choice: a stable baseline year or mean head.\n"
    ),
    ("geoprior_kappa_mode", None): (
        "Mode used to interpret kappa in the physics closure.\n"
        "\n"
        "This is an advanced control for mapping kappa into the\n"
        "effective parameterization.\n"
    ),
    ("geoprior_hd_factor", None): (
        "Multiplier applied to the drainage thickness scale used\n"
        "in kappa and time-scale priors.\n"
        "\n"
        "Higher values imply thicker drainage influence.\n"
    ),
    ("mv_prior_units", None): (
        "Units assumed for the MV prior.\n"
        "\n"
        "Keep this consistent with your MV source and the\n"
        "Stage-1 export conventions.\n"
    ),
    ("mv_alpha_disp", None): (
        "Dispersion parameter for robust MV penalties.\n"
        "\n"
        "Higher values reduce sensitivity to MV outliers.\n"
    ),
    ("mv_huber_delta", None): (
        "Huber delta for MV robust loss.\n"
        "\n"
        "Controls the transition between L2 and L1 behavior.\n"
    ),
    ("mv_prior_mode", None): (
        "How the MV prior is applied.\n"
        "\n"
        "Examples: additive, multiplicative, or log-domain modes.\n"
        "Exact meaning depends on the chosen MV implementation.\n"
    ),
    ("mv_weight", None): (
        "Overall weight for the MV prior term.\n"
        "\n"
        "Higher values enforce MV consistency more strongly.\n"
    ),
    ("mv_schedule_unit", None): (
        "Unit used for MV scheduling controls.\n"
        "\n"
        "Common choices: epoch or step.\n"
    ),
    ("mv_delay_epochs", None): (
        "Delay (epochs) before MV penalties start to increase.\n"
        "\n"
        "Used when mv_schedule_unit is epoch-based.\n"
    ),
    ("mv_warmup_epochs", None): (
        "Warmup length (epochs) for MV penalties to ramp up.\n"
        "\n"
        "Used when mv_schedule_unit is epoch-based.\n"
    ),
    ("mv_delay_steps", None): (
        "Delay (steps) before MV penalties start to increase.\n"
        "\n"
        "Used when mv_schedule_unit is step-based.\n"
    ),
    ("mv_warmup_steps", None): (
        "Warmup length (steps) for MV penalties to ramp up.\n"
        "\n"
        "Used when mv_schedule_unit is step-based.\n"
    ),

    # =============================================================
    # Bounds & constraints (v3.2)
    # =============================================================
    ("physics_bounds", "H_min"): (
        "Lower bound for the effective thickness H field.\n"
        "Unit: meters.\n"
        "\n"
        "Set a positive minimum to avoid degenerate physics.\n"
    ),
    ("physics_bounds", "H_max"): (
        "Upper bound for the effective thickness H field.\n"
        "Unit: meters.\n"
        "\n"
        "Choose a physically plausible maximum for the domain.\n"
    ),
    ("h_field_min_si", None): (
        "Safety floor applied to the head or thickness-related\n"
        "field used inside physics computations (SI units).\n"
        "\n"
        "Prevents divisions by near-zero values in closures.\n"
    ),

    # =============================================================
    # Units, SI mapping, scaling floors
    # =============================================================
    ("time_units", None): (
        "Time unit used by the physics engine.\n"
        "\n"
        "year/day/second affects dt_min_units, tau bounds, and\n"
        "any conversion to per-second quantities.\n"
    ),
    ("subs_unit_to_si", None): (
        "Multiplier that converts subsidence units to meters.\n"
        "\n"
        "Example: subsidence in mm -> 0.001.\n"
    ),
    ("subs_scale_si", None): (
        "Additional scale applied to subsidence in SI.\n"
        "\n"
        "Useful when Stage-1 scaling differs from raw units.\n"
    ),
    ("subs_bias_si", None): (
        "Additive bias applied to subsidence in SI.\n"
        "\n"
        "Used for affine mapping: x_si = x*scale + bias.\n"
    ),
    ("head_unit_to_si", None): (
        "Multiplier that converts head units to meters.\n"
        "\n"
        "Example: head in cm -> 0.01.\n"
    ),
    ("head_scale_si", None): (
        "Additional scale applied to head in SI.\n"
        "\n"
        "Used for affine mapping to match Stage-1 conventions.\n"
    ),
    ("head_bias_si", None): (
        "Additive bias applied to head in SI.\n"
        "\n"
        "Used for affine mapping: h_si = h*scale + bias.\n"
    ),
    ("thickness_unit_to_si", None): (
        "Multiplier that converts thickness units to meters.\n"
        "\n"
        "Use when thickness is provided in non-meter units.\n"
    ),
    ("z_surf_unit_to_si", None): (
        "Multiplier that converts surface elevation (z_surf_col)\n"
        "to meters.\n"
    ),
    ("auto_si_affine_from_stage1", None): (
        "If True, infer SI affine mapping from Stage-1 manifest\n"
        "when available.\n"
        "\n"
        "This reduces manual unit configuration mistakes.\n"
    ),
    ("coord_mode", None): (
        "Coordinate mode used by the model.\n"
        "\n"
        "Examples: normalized, utm, lonlat.\n"
        "Used to interpret coord scalers and epsg settings.\n"
    ),
    ("utm_epsg", None): (
        "EPSG code for the UTM coordinate system used by the\n"
        "dataset.\n"
        "\n"
        "Only relevant when coord_mode uses UTM.\n"
    ),
    ("coord_src_epsg", None): (
        "EPSG code for the source coordinates before conversion.\n"
        "\n"
        "Example: 4326 for lon/lat.\n"
    ),
    ("cons_residual_units", None): (
        "Label for consolidation residual units (for diagnostics\n"
        "and exports).\n"
        "\n"
        "This does not change math, but improves interpretability.\n"
    ),
    ("gw_residual_units", None): (
        "Label for groundwater residual units (for diagnostics\n"
        "and exports).\n"
        "\n"
        "This does not change math, but improves interpretability.\n"
    ),
    ("cons_scale_floor", None): (
        "Lower bound on the consolidation residual scale used\n"
        "when normalizing residual magnitudes.\n"
        "\n"
        "Helps avoid huge normalized losses when scale is tiny.\n"
    ),
    ("gw_scale_floor", None): (
        "Lower bound on the groundwater residual scale used when\n"
        "normalizing residual magnitudes.\n"
        "\n"
        "Helps avoid huge normalized losses when scale is tiny.\n"
    ),

    # =============================================================
    # Q / source-term controls
    # =============================================================
    ("q_wrt_normalized_time", None): (
        "If True, interpret Q schedules with respect to the\n"
        "normalized time coordinate (0..1).\n"
        "\n"
        "If False, interpret Q schedules in physical time_units.\n"
    ),
    ("q_in_si", None): (
        "If True, Q values are already provided in SI units.\n"
        "\n"
        "If False, Q is converted using the configured mapping.\n"
    ),
    ("q_in_per_second", None): (
        "If True, Q is interpreted as a per-second quantity.\n"
        "\n"
        "If your time_units is year/day, conversion may apply.\n"
    ),
    ("q_kind", None): (
        "Type of Q (source term) used by the physics engine.\n"
        "\n"
        "This selects the functional form or parameterization\n"
        "used to inject forcing into the groundwater equation.\n"
    ),
    ("q_length_in_si", None): (
        "If True, length scales used in Q computations are\n"
        "interpreted in meters (SI).\n"
        "\n"
        "If False, length scales follow the dataset units.\n"
    ),
    ("drainage_mode", None): (
        "Controls how drainage / recharge is modeled in the\n"
        "source-term pathway.\n"
        "\n"
        "This is an advanced control used with q_kind.\n"
    ),
    ("log_q_diagnostics", None): (
        "If True, log extra diagnostics for Q during training.\n"
        "\n"
        "Useful for debugging sign, units, and schedule behavior.\n"
    ),

    # =============================================================
    # Training strategy (v3.2)
    # =============================================================
    ("loss_weight_gwl_physics_first", None): (
        "Weight applied to the GWL data loss during physics_first.\n"
        "\n"
        "Lower values prioritize physics stabilization early.\n"
    ),
    ("subs_resid_policy_physics_first", None): (
        "Schedule policy for enabling the subsidence residual\n"
        "during physics_first.\n"
        "\n"
        "Use warmup/ramp controls below to shape activation.\n"
    ),
    ("subs_resid_warmup_epochs_physics_first", None): (
        "Warmup epochs before the subsidence residual becomes\n"
        "influential in physics_first.\n"
    ),
    ("subs_resid_ramp_epochs_physics_first", None): (
        "Ramp epochs over which the subsidence residual increases\n"
        "from low to target influence in physics_first.\n"
    ),
    ("loss_weight_gwl_data_first", None): (
        "Weight applied to the GWL data loss during data_first.\n"
        "\n"
        "Higher values prioritize matching observed GWL first.\n"
    ),
    ("lambda_q_data_first", None): (
        "Target weight for Q-related physics contribution in\n"
        "data_first.\n"
    ),
    ("q_policy_data_first", None): (
        "Schedule for activating groundwater forcing Q during\n"
        "data_first.\n"
        "\n"
        "Use warmup/ramp controls to delay Q until data fit is\n"
        "reasonable.\n"
    ),
    ("q_warmup_epochs_data_first", None): (
        "Warmup epochs in data_first before Q becomes influential.\n"
    ),
    ("q_ramp_epochs_data_first", None): (
        "Ramp epochs in data_first to increase Q influence to its\n"
        "target strength.\n"
    ),
    ("subs_resid_policy_data_first", None): (
        "Schedule policy for enabling the subsidence residual\n"
        "during data_first.\n"
    ),
    ("subs_resid_warmup_epochs_data_first", None): (
        "Warmup epochs in data_first before the subsidence residual\n"
        "becomes influential.\n"
    ),
    ("subs_resid_ramp_epochs_data_first", None): (
        "Ramp epochs in data_first to increase subsidence residual\n"
        "influence to its target strength.\n"
    ),

    # =============================================================
    # Diagnostics & safeguards
    # =============================================================
    ("track_aux_metrics", None): (
        "If True, compute and log auxiliary metrics in addition\n"
        "to the main training losses.\n"
        "\n"
        "This can slightly increase overhead but improves insight.\n"
    ),
    ("physics_baseline_mode", None): (
        "Baseline used for physics diagnostics and comparisons.\n"
        "\n"
        "none:   no baseline.\n"
        "data:   baseline from data-driven pathway.\n"
        "physics: baseline from physics-only pathway.\n"
    ),
    ("debug_physics_grads", None): (
        "If True, log extra information about physics gradients.\n"
        "\n"
        "Useful for diagnosing NaNs, exploding gradients, or dead\n"
        "physics pathways.\n"
    ),
    ("scaling_error_policy", None): (
        "What to do when scaling / SI conversion information is\n"
        "missing or inconsistent.\n"
        "\n"
        "raise: stop with an error.\n"
        "warn:  continue but emit a warning.\n"
        "ignore: continue silently (not recommended).\n"
    ),
    ("clip_global_norm", None): (
        "Global gradient norm clipping threshold.\n"
        "\n"
        "Set to 0 or None to disable. Clipping can stabilize\n"
        "training when physics gradients spike.\n"
    ),
    ("scaling_kwargs_json_path", None): (
        "Optional path to a JSON file that overrides scaling\n"
        "parameters used by the physics engine.\n"
        "\n"
        "Common use: adjust SI mapping, floors, or time ranges\n"
        "without rebuilding Stage-1 artifacts.\n"
    ),
    ("eval_json_units_scope", None): (
        "Which outputs are exported with eval_json_units_mode.\n"
        "\n"
        "Examples: subs_only, gwl_only, both, physics_only.\n"
        "This is an advanced export control.\n"
    ),
})


def help_text(
    name: str,
    subkey: Optional[str] = None,
) -> str:
    txt = _PHYS_HELP.get((name, subkey))
    if txt:
        return txt
    txt = _PHYS_HELP.get((name, None))
    return txt or ""


def first_line(txt: str) -> str:
    s = (txt or "").strip()
    if not s:
        return ""
    return s.splitlines()[0].strip()
