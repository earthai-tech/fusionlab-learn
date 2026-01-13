# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""prior_schema

UI schema / registry for GeoPriorConfig physics parameters.

This module is *data only*:
- a stable list of keys to render in dialogs
- a FieldSpec registry (type, limits, choices, tooltip)
- a guardrail validator so new config fields are not missed

It is intentionally independent from Qt widgets.
Dialogs should
consume this schema through GeoConfigStore.get_value() and
GeoConfigStore.set_value_by_key().

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    # Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    # Sequence,
    Set,
    Tuple,
    Iterator, 
)

from .geoprior_config import GeoPriorConfig
from .helps import help_text as _help_text
# ------------------------------------------------------------
# Keys
# ------------------------------------------------------------


@dataclass(frozen=True)
class FieldKey:
    """Address a config value.

    name
        Top-level GeoPriorConfig attribute.
    subkey
        Optional sub-key for dict attributes
        (e.g. physics_bounds).
    """

    name: str
    subkey: Optional[str] = None

    def is_dict_item(self) -> bool:
        return self.subkey is not None


# ------------------------------------------------------------
# Specs
# ------------------------------------------------------------


Kind = str


@dataclass(frozen=True)
class FieldSpec:
    """Schema entry for a FieldKey."""

    key: FieldKey
    group: str
    label: str

    kind: Kind = "text"
    tooltip: str = ""

    source: str = "geoprior_config"
    advanced: bool = False

    # Choice fields
    choices: Optional[Tuple[str, ...]] = None
    editable: bool = False

    # Numeric widgets
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    decimals: Optional[int] = None


# ------------------------------------------------------------
# Groups (ordered)
# ------------------------------------------------------------


PHYSICS_GROUP_TITLES: Dict[str, str] = {
    "physics.inputs": "Physics inputs & conventions",
    "physics.engine": "PDE / residual engine",
    "physics.loss_weights": "Base physics loss weights",
    "physics.offset": "Global physics-loss offset (v3.2)",
    "physics.mv": "MV / kappa priors & schedules",
    "physics.bounds": "Bounds & constraints (v3.2)",
    "physics.units": "Units, SI mapping, scaling floors",
    "physics.q": "Q / source-term controls",
    "physics.strategy": "Training strategy (v3.2)",
    "physics.diagnostics": "Diagnostics & safeguards",
}


def K(name: str, subkey: Optional[str] = None) -> FieldKey:
    return FieldKey(name=name, subkey=subkey)


PHYSICS_GROUPS: Dict[str, List[FieldKey]] = {
    # 1) Physics inputs & conventions
    "physics.inputs": [
        K("gwl_kind"),
        K("gwl_sign"),
        K("use_head_proxy"),
        K("z_surf_col"),
        K("include_z_surf_as_static"),
        K("head_col"),
        K("gwl_dyn_index"),
        K("geoprior_use_effective_h"),
    ],
    # 2) PDE / residual engine
    "physics.engine": [
        K("pde_mode"),
        K("residual_method"),
        K("physics_warmup_steps"),
        K("physics_ramp_steps"),
        K("scale_pde_residuals"),
        K("allow_subs_residual"),
        K("dt_min_units"),
    ],
    # 3) Base physics loss weights
    "physics.loss_weights": [
        K("lambda_cons"),
        K("lambda_gw"),
        K("lambda_prior"),
        K("lambda_smooth"),
        K("lambda_mv"),
        K("lambda_bounds"),
        K("lambda_q"),
    ],
    # 4) Global physics-loss offset (v3.2)
    "physics.offset": [
        K("offset_mode"),
        K("lambda_offset"),
        K("use_lambda_offset_scheduler"),
        K("lambda_offset_unit"),
        K("lambda_offset_when"),
        K("lambda_offset_warmup"),
        K("lambda_offset_start"),
        K("lambda_offset_end"),
        K("lambda_offset_schedule"),
    ],
    # 5) MV / kappa priors + schedules
    "physics.mv": [
        K("mv_lr_mult"),
        K("kappa_lr_mult"),
        K("geoprior_init_mv"),
        K("geoprior_init_kappa"),
        K("geoprior_gamma_w"),
        K("geoprior_h_ref"),
        K("geoprior_kappa_mode"),
        K("geoprior_hd_factor"),
        K("mv_prior_units"),
        K("mv_alpha_disp"),
        K("mv_huber_delta"),
        K("mv_prior_mode"),
        K("mv_weight"),
        K("mv_schedule_unit"),
        K("mv_delay_epochs"),
        K("mv_warmup_epochs"),
        K("mv_delay_steps"),
        K("mv_warmup_steps"),
    ],
    # 6) Bounds & constraints (v3.2)
    "physics.bounds": [
        K("physics_bounds", "K_min"),
        K("physics_bounds", "K_max"),
        K("physics_bounds", "Ss_min"),
        K("physics_bounds", "Ss_max"),
        K("physics_bounds", "tau_min"),
        K("physics_bounds", "tau_max"),
        K("physics_bounds", "H_min"),
        K("physics_bounds", "H_max"),
        K("bounds_mode"),
        K("h_field_min_si"),
    ],
    # 7) Units, SI mapping, scaling floors
    "physics.units": [
        K("time_units"),
        K("subs_unit_to_si"),
        K("subs_scale_si"),
        K("subs_bias_si"),
        K("head_unit_to_si"),
        K("head_scale_si"),
        K("head_bias_si"),
        K("thickness_unit_to_si"),
        K("z_surf_unit_to_si"),
        K("auto_si_affine_from_stage1"),
        K("coord_mode"),
        K("utm_epsg"),
        K("coord_src_epsg"),
        K("cons_residual_units"),
        K("gw_residual_units"),
        K("cons_scale_floor"),
        K("gw_scale_floor"),
    ],
    # 8) Q / source-term controls
    "physics.q": [
        K("q_wrt_normalized_time"),
        K("q_in_si"),
        K("q_in_per_second"),
        K("q_kind"),
        K("q_length_in_si"),
        K("drainage_mode"),
        K("log_q_diagnostics"),
    ],
    # 9) Training strategy (physics-first vs data-first)
    "physics.strategy": [
        K("training_strategy"),
        K("q_policy_physics_first"),
        K("q_warmup_epochs_physics_first"),
        K("q_ramp_epochs_physics_first"),
        K("lambda_q_physics_first"),
        K("loss_weight_gwl_physics_first"),
        K("subs_resid_policy_physics_first"),
        K("subs_resid_warmup_epochs_physics_first"),
        K("subs_resid_ramp_epochs_physics_first"),
        K("loss_weight_gwl_data_first"),
        K("lambda_q_data_first"),
        K("q_policy_data_first"),
        K("q_warmup_epochs_data_first"),
        K("q_ramp_epochs_data_first"),
        K("subs_resid_policy_data_first"),
        K("subs_resid_warmup_epochs_data_first"),
        K("subs_resid_ramp_epochs_data_first"),
    ],
    # 10) Diagnostics & stability safeguards
    "physics.diagnostics": [
        K("track_aux_metrics"),
        K("physics_baseline_mode"),
        K("debug_physics_grads"),
        K("scaling_error_policy"),
        K("clip_global_norm"),
        K("scaling_kwargs_json_path"),
        K("eval_json_units_mode"),
        K("eval_json_units_scope"),
    ],
}


def iter_physics_keys() -> Iterable[FieldKey]:
    for group in PHYSICS_GROUP_TITLES:
        for key in PHYSICS_GROUPS.get(group, []):
            yield key


# ------------------------------------------------------------
# Overrides
# ------------------------------------------------------------


CHOICE_SPECS: Dict[str, Tuple[str, ...]] = {
    "gwl_kind": ("depth_bgs", "head"),
    "gwl_sign": ("down_positive", "up_positive"),
    "pde_mode": (
        "off",
        "none",
        "on",
        "both",
        "consolidation",
        "gw_flow",
    ),
    "residual_method": ("exact", "fd"),
    "offset_mode": ("mul", "log10"),
    "lambda_offset_unit": ("epoch", "step"),
    "lambda_offset_when": ("begin", "end"),
    "bounds_mode": ("soft", "hard"),
    "time_units": ("year", "day", "second"),
    "training_strategy": ("physics_first", "data_first"),
    "eval_json_units_mode": ("interpretable", "si"),
    "scaling_error_policy": ("raise", "warn", "ignore"),
    "physics_baseline_mode": ("none", "data", "physics"),
}
CHOICE_SPECS["calibration_mode"] = (
    "none",
    "temperature",
    "isotonic",
    "conformal",
)


# Choice-like, but keep editable for forward-compat.
EDITABLE_CHOICE_NAMES: Set[str] = {
    "coord_mode",
    "cons_residual_units",
    "gw_residual_units",
    "geoprior_kappa_mode",
    "mv_prior_units",
    "mv_prior_mode",
    "mv_schedule_unit",
    "q_kind",
    "drainage_mode",
    "q_policy_physics_first",
    "q_policy_data_first",
    "subs_resid_policy_physics_first",
    "subs_resid_policy_data_first",
    "eval_json_units_scope",
}


BOOL_NAMES: Set[str] = {
    "use_head_proxy",
    "include_z_surf_as_static",
    "geoprior_use_effective_h",
    "scale_pde_residuals",
    "allow_subs_residual",
    "use_lambda_offset_scheduler",
    "auto_si_affine_from_stage1",
    "q_wrt_normalized_time",
    "q_in_si",
    "q_in_per_second",
    "q_length_in_si",
    "log_q_diagnostics",
    "track_aux_metrics",
    "debug_physics_grads",
}


INT_NAMES: Set[str] = {
    "gwl_dyn_index",
    "physics_warmup_steps",
    "physics_ramp_steps",
    "mv_delay_epochs",
    "mv_warmup_epochs",
    "mv_delay_steps",
    "mv_warmup_steps",
    "utm_epsg",
    "coord_src_epsg",
    "q_warmup_epochs_physics_first",
    "q_ramp_epochs_physics_first",
    "q_warmup_epochs_data_first",
    "q_ramp_epochs_data_first",
    "subs_resid_warmup_epochs_physics_first",
    "subs_resid_ramp_epochs_physics_first",
    "subs_resid_warmup_epochs_data_first",
    "subs_resid_ramp_epochs_data_first",
    "lambda_offset_warmup",
}


PATH_NAMES: Set[str] = {"scaling_kwargs_json_path"}
JSON_NAMES: Set[str] = {"lambda_offset_schedule"}


MIN_ZERO_NAMES: Set[str] = {
    "dt_min_units",
    "lambda_cons",
    "lambda_gw",
    "lambda_prior",
    "lambda_smooth",
    "lambda_mv",
    "lambda_bounds",
    "lambda_q",
    "lambda_offset",
    "lambda_offset_start",
    "lambda_offset_end",
    "mv_lr_mult",
    "kappa_lr_mult",
    "geoprior_init_mv",
    "geoprior_init_kappa",
    "geoprior_gamma_w",
    "geoprior_h_ref",
    "geoprior_hd_factor",
    "mv_alpha_disp",
    "mv_huber_delta",
    "mv_weight",
    "h_field_min_si",
    "subs_unit_to_si",
    "subs_scale_si",
    "head_unit_to_si",
    "head_scale_si",
    "thickness_unit_to_si",
    "z_surf_unit_to_si",
    "cons_scale_floor",
    "gw_scale_floor",
    "lambda_q_physics_first",
    "lambda_q_data_first",
    "loss_weight_gwl_physics_first",
    "loss_weight_gwl_data_first",
    "clip_global_norm",
}


LABEL_OVERRIDES: Dict[FieldKey, str] = {
    K("gwl_kind"): "GWL representation",
    K("gwl_sign"): "GWL sign convention",
    K("use_head_proxy"): "Use head proxy",
    K("pde_mode"): "PDE mode",
    K("residual_method"): "Residual method",
    K("scale_pde_residuals"): "Scale PDE residuals",
    K("allow_subs_residual"): "Enable subsidence residual",
    K("dt_min_units"): "Min dt in time_units",
    K("bounds_mode"): "Bounds penalty mode",
    K("h_field_min_si"): "Min H field (SI)",
    K("q_in_si"): "Q in SI",
    K("q_in_per_second"): "Q per second",
    K("q_wrt_normalized_time"): "Q wrt normalized time",
    K("scaling_kwargs_json_path"): "Scaling kwargs JSON",
}

LABEL_OVERRIDES.update(
    {
        K("interval_level"): "Interval level",
        K("crossing_penalty"): "Crossing penalty",
        K("crossing_margin"): "Crossing margin",
        K("calibration_mode"): "Calibration mode",
        K("calibration_temperature"): "Calibration temperature",
    }
)

TOOLTIP_OVERRIDES: Dict[FieldKey, str] = {
    K("pde_mode"): "Which physics residuals are active.",
    K("offset_mode"): "How loss weights are offset.",
    K("lambda_offset_schedule"): "Epoch/step -> offset map.",
    K("physics_bounds", "K_min"): "Lower K bound (SI).",
    K("physics_bounds", "K_max"): "Upper K bound (SI).",
    K("physics_bounds", "Ss_min"): "Lower Ss bound (SI).",
    K("physics_bounds", "Ss_max"): "Upper Ss bound (SI).",
    K("physics_bounds", "tau_min"): "Lower tau bound (SI).",
    K("physics_bounds", "tau_max"): "Upper tau bound (SI).",
    K("physics_bounds", "H_min"): "Lower H bound (m).",
    K("physics_bounds", "H_max"): "Upper H bound (m).",
}


def _titleize(name: str) -> str:
    parts = name.replace("__", "_").split("_")
    return " ".join(p[:1].upper() + p[1:] for p in parts if p)


def _infer_kind(key: FieldKey) -> Kind:
    if key.is_dict_item():
        return "float"
    n = key.name

    if n in {
        "interval_level",
        "crossing_penalty",
        "crossing_margin",
        "calibration_temperature",
    }:
        return "float"

    if n in PATH_NAMES:
        return "path"
    if n in JSON_NAMES:
        return "json"
    if n in BOOL_NAMES:
        return "bool"
    if n in INT_NAMES:
        return "int"
    if n in CHOICE_SPECS:
        return "choice"
    if n in EDITABLE_CHOICE_NAMES:
        return "choice"
    if n.startswith("lambda_"):
        return "float"
    if n.startswith("mv_"):
        return "float"
    if n.endswith("_mult"):
        return "float"
    if n.endswith("_to_si"):
        return "float"
    if n.endswith("_scale_si"):
        return "float"
    if n.endswith("_bias_si"):
        return "float"
    if n.endswith("_floor"):
        return "float"
    if n.endswith("_si"):
        return "float"
    return "text"


def _label_for(key: FieldKey) -> str:
    if key in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[key]
    if key.is_dict_item() and key.subkey:
        return f"{key.subkey} bound"
    return _titleize(key.name)


def _tooltip_for(key: FieldKey) -> str:
    return TOOLTIP_OVERRIDES.get(key, "")


def build_physics_schema() -> Dict[FieldKey, FieldSpec]:
    """Build the physics registry from PHYSICS_GROUPS."""

    out: Dict[FieldKey, FieldSpec] = {}

    for group_id in PHYSICS_GROUP_TITLES:
        keys = PHYSICS_GROUPS.get(group_id, [])
        for key in keys:
            kind = _infer_kind(key)

            choices = None
            editable = False

            if not key.is_dict_item():
                if key.name in CHOICE_SPECS:
                    choices = CHOICE_SPECS[key.name]
                if key.name in EDITABLE_CHOICE_NAMES:
                    editable = True
                if kind == "choice" and choices is None:
                    editable = True

            min_value = None
            if (not key.is_dict_item()) and (
                key.name in MIN_ZERO_NAMES
            ):
                min_value = 0.0

            out[key] = FieldSpec(
                key=key,
                group=group_id,
                label=_label_for(key),
                kind=kind,
                tooltip=_tooltip_for(key),
                choices=choices,
                editable=editable,
                min_value=min_value,
            )

    return out


PHYSICS_SCHEMA: Dict[FieldKey, FieldSpec] = (
    build_physics_schema()
)
# “hydrate” the schema once
# so schema-driven tooltips everywhere
# for k, spec in PHYSICS_SCHEMA.items():
#     if not (spec.tooltip or "").strip():
#         # spec.tooltip = help_text(k.name, k.subkey)
#         object.__setattr__(spec, "tooltip", help_text(k.name, k.subkey))

# ------------------------------------------------------------
# Validation guardrail
# ------------------------------------------------------------


_PHYSICS_LIKE_PREFIX: Tuple[str, ...] = (
    "lambda_",
    "mv_",
    "kappa_",
    "gwl_",
    "q_",
    "physics_",
    "subs_resid_",
    "subs_",
    "head_",
    "thickness_",
    "z_surf_",
    "coord_",
    "cons_",
    "gw_",
)


_PHYSICS_LIKE_NAMES: Set[str] = {
    "pde_mode",
    "residual_method",
    "offset_mode",
    "bounds_mode",
    "physics_bounds",
    "dt_min_units",
    "time_units",
    "coord_mode",
    "utm_epsg",
    "coord_src_epsg",
    "cons_residual_units",
    "gw_residual_units",
    "cons_scale_floor",
    "gw_scale_floor",
    "scaling_kwargs_json_path",
    "scaling_error_policy",
    "clip_global_norm",
    "track_aux_metrics",
    "debug_physics_grads",
    "physics_baseline_mode",
    "eval_json_units_mode",
    "eval_json_units_scope",
    "use_head_proxy",
    "include_z_surf_as_static",
    "geoprior_use_effective_h",
    "z_surf_col",
    "head_col",
}


def validate_schema_against_config(
    *,
    cfg: Optional[GeoPriorConfig] = None,
    strict: bool = True,
) -> None:
    """Validate that schema keys exist on GeoPriorConfig.

    strict=True raises when a physics-like config field is
    present in GeoPriorConfig but not indexed
    in PHYSICS_GROUPS.
    """

    if cfg is None:
        cfg = GeoPriorConfig()

    # 1) Duplicates
    keys = list(iter_physics_keys())
    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate FieldKey in PHYSICS_GROUPS."
        )

    # 2) Existence checks
    cfg_fields = set(cfg.__dataclass_fields__.keys())
    top_names = {k.name for k in keys}

    missing_top = sorted(
        n for n in top_names if n not in cfg_fields
    )
    if missing_top:
        raise ValueError(
            "Schema references missing config fields: "
            f"{missing_top}"
        )

    # 3) Dict-subkey checks
    for key in keys:
        if not key.is_dict_item():
            continue
        val = getattr(cfg, key.name, None)
        if not isinstance(val, Mapping):
            raise ValueError(
                f"{key.name!r} is not a mapping in config."
            )
        if key.subkey not in val:
            raise ValueError(
                f"Missing {key.subkey!r} in {key.name!r}."
            )

    # 4) Guardrail for new physics-like fields
    physics_like = set()
    for name in cfg_fields:
        if name.startswith(_PHYSICS_LIKE_PREFIX):
            physics_like.add(name)
        if name in _PHYSICS_LIKE_NAMES:
            physics_like.add(name)

    not_indexed = sorted(
        n for n in physics_like if n not in top_names
    )
    if not_indexed and strict:
        raise ValueError(
            "Unindexed physics-like config fields: "
            f"{not_indexed}"
        )


UNCERTAINTY_GROUP_TITLES = {
    "uncertainty": "Uncertainty & calibration",
}

UNCERTAINTY_GROUPS = {
    "uncertainty": [
        "interval_level",
        "crossing_penalty",
        "crossing_margin",
        "calibration_mode",
        "calibration_temperature",
    ],
}


def iter_uncertainty_keys() -> Iterator[FieldKey]:
    for names in UNCERTAINTY_GROUPS.values():
        for name in names:
            yield FieldKey(name)


def build_uncertainty_schema(
    *,
    help_text: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Dict[FieldKey, FieldSpec]:
    help_text = help_text or {}
    labels = labels or {}

    schema: Dict[FieldKey, FieldSpec] = {}

    for group_id, names in UNCERTAINTY_GROUPS.items():
        for name in names:
            key = FieldKey(name)
            kind = _infer_kind(key)

            choices = None
            editable = False

            if key.name in CHOICE_SPECS:
                choices = CHOICE_SPECS[key.name]

            if (kind == "choice") and (choices is None):
                editable = True

            # min/max hints (nice for numeric widgets)
            min_value = None
            max_value = None

            if key.name in {
                "crossing_penalty",
                "crossing_margin",
            }:
                min_value = 0.0

            if key.name == "interval_level":
                min_value = 0.001
                max_value = 0.999

            if key.name == "calibration_temperature":
                min_value = 1e-6

            # tooltip: prefer overrides, then helps
            tooltip = _tooltip_for(key)
            if not (tooltip or "").strip():
                try:
                    tooltip = str(
                        _help_text(key.name, key.subkey)
                        or ""
                    )
                except Exception:
                    tooltip = ""

            schema[key] = FieldSpec(
                key=key,
                group=group_id,
                label=labels.get(
                    key.name,
                    _label_for(key),
                ),
                kind=kind,
                tooltip=tooltip,
                choices=choices,
                editable=editable,
                min_value=min_value,
                max_value=max_value,
            )

    return schema



UNCERTAINTY_SCHEMA = build_uncertainty_schema()

ALL_SCHEMA = {**PHYSICS_SCHEMA, **UNCERTAINTY_SCHEMA}