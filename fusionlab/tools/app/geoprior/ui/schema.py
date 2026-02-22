# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


ComboItems = List[Tuple[str, Any]]


@dataclass(frozen=True)
class SectionSpec:
    sec_id: str
    title: str
    description: str = ""


@dataclass(frozen=True)
class FieldSpec:
    """
    UI metadata for a GeoPriorConfig field.

    kind values (convention)
    ------------------------
    - "line": QLineEdit
    - "check": QCheckBox
    - "spin": QSpinBox
    - "dspin": QDoubleSpinBox
    - "combo": QComboBox
    - "json": QTextEdit JSON editor (later)
    - "list": list editor (later)
    """
    key: str
    label: str
    sec_id: str
    kind: str
    tooltip: str = ""
    placeholder: str = ""
    editable: bool = False
    advanced: bool = False

    # numeric hints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None

    # combo hints
    items: Optional[ComboItems] = None
    none_text: Optional[str] = None


def default_sections() -> List[SectionSpec]:
    return [
        SectionSpec(
            "summary",
            "Summary",
            "Preview of current run setup.",
        ),
        SectionSpec(
            "paths",
            "Project & paths",
            "City, dataset path, results root.",
        ),
        SectionSpec(
            "time",
            "Time window & forecast",
            "Training end year, forecast start, horizon.",
        ),
        SectionSpec(
            "data_semantics",
            "Data columns & semantics",
            "Map dataset columns to model semantics.",
        ),
        SectionSpec(
            "coords",
            "Coordinates & CRS",
            "Coordinate units and EPSG settings.",
        ),
        # keep placeholders for later expansion
        SectionSpec("features", "Feature registry"),
        SectionSpec("censoring", "Censoring & H-field"),
        SectionSpec("scaling", "Scaling & units"),
        SectionSpec("arch", "Model architecture"),
        SectionSpec("train", "Training basics"),
        SectionSpec("physics", "Physics & constraints"),
        SectionSpec("prob", "Probabilistic outputs"),
        SectionSpec("tuning", "Tuning"),
        SectionSpec("device", "Device & runtime"),
        SectionSpec("ui", "UI preferences"),
    ]


def _combo_items() -> Dict[str, ComboItems]:
    return {
        "mode": [
            ("tft_like", "tft_like"),
            ("pihal_like", "pihal_like"),
        ],
        "gwl_kind": [
            ("depth_bgs", "depth_bgs"),
            ("head", "head"),
        ],
        "gwl_sign": [
            ("down_positive", "down_positive"),
            ("up_positive", "up_positive"),
        ],
        "subsidence_kind": [
            ("cumulative", "cumulative"),
            ("rate", "rate"),
        ],
        "coord_mode": [
            ("degrees", "degrees"),
            ("meters", "meters"),
        ],
    }


def default_fields() -> Dict[str, FieldSpec]:
    items = _combo_items()

    specs = [
        # ---------------- paths ----------------
        FieldSpec(
            key="city",
            label="City",
            sec_id="paths",
            kind="line",
            tooltip="Experiment city / dataset name.",
        ),
        FieldSpec(
            key="dataset_path",
            label="Dataset path",
            sec_id="paths",
            kind="line",
            tooltip="Canonical dataset path used by the GUI.",
        ),
        FieldSpec(
            key="results_root",
            label="Results root",
            sec_id="paths",
            kind="line",
            tooltip="Root folder where runs and artifacts are saved.",
        ),
        FieldSpec(
            key="clean_stage1_dir",
            label="Clean Stage-1 dir",
            sec_id="paths",
            kind="check",
            advanced=True,
        ),
        FieldSpec(
            key="stage1_auto_reuse_if_match",
            label="Auto reuse if match",
            sec_id="paths",
            kind="check",
            advanced=True,
        ),
        FieldSpec(
            key="stage1_force_rebuild_if_mismatch",
            label="Force rebuild if mismatch",
            sec_id="paths",
            kind="check",
            advanced=True,
        ),
        FieldSpec(
            key="audit_stages",
            label="Audit stages",
            sec_id="paths",
            kind="line",
            advanced=True,
            placeholder="* or stage1,stage2,stage3",
        ),
        # ---------------- time ----------------
        FieldSpec(
            key="train_end_year",
            label="Train end year",
            sec_id="time",
            kind="spin",
            min_value=1900,
            max_value=2200,
        ),
        FieldSpec(
            key="forecast_start_year",
            label="Forecast start year",
            sec_id="time",
            kind="spin",
            min_value=1900,
            max_value=2200,
        ),
        FieldSpec(
            key="forecast_horizon_years",
            label="Horizon (years)",
            sec_id="time",
            kind="spin",
            min_value=1,
            max_value=100,
        ),
        FieldSpec(
            key="time_steps",
            label="Time steps",
            sec_id="time",
            kind="spin",
            min_value=1,
            max_value=500,
        ),
        FieldSpec(
            key="build_future_npz",
            label="Build future NPZ",
            sec_id="time",
            kind="check",
        ),
        # -------- data columns & semantics -----
        FieldSpec(
            key="mode",
            label="Mode",
            sec_id="data_semantics",
            kind="combo",
            items=items["mode"],
        ),
        FieldSpec(
            key="time_col",
            label="Time column",
            sec_id="data_semantics",
            kind="combo",
            editable=True,
            tooltip="Dataset column representing time index.",
        ),
        FieldSpec(
            key="lon_col",
            label="Lon column",
            sec_id="data_semantics",
            kind="combo",
            editable=True,
        ),
        FieldSpec(
            key="lat_col",
            label="Lat column",
            sec_id="data_semantics",
            kind="combo",
            editable=True,
        ),
        FieldSpec(
            key="subs_col",
            label="Subsidence column",
            sec_id="data_semantics",
            kind="combo",
            editable=True,
        ),
        FieldSpec(
            key="gwl_col",
            label="GWL column",
            sec_id="data_semantics",
            kind="combo",
            editable=True,
        ),
        FieldSpec(
            key="h_field_col",
            label="H-field column",
            sec_id="data_semantics",
            kind="combo",
            editable=True,
        ),
        FieldSpec(
            key="gwl_kind",
            label="GWL kind",
            sec_id="data_semantics",
            kind="combo",
            items=items["gwl_kind"],
        ),
        FieldSpec(
            key="gwl_sign",
            label="GWL sign",
            sec_id="data_semantics",
            kind="combo",
            items=items["gwl_sign"],
        ),
        FieldSpec(
            key="use_head_proxy",
            label="Use head proxy",
            sec_id="data_semantics",
            kind="check",
            tooltip="Derive head from depth-to-water and z_surf.",
        ),
        FieldSpec(
            key="z_surf_col",
            label="z_surf column",
            sec_id="data_semantics",
            kind="combo",
            editable=True,
            none_text="(None)",
            tooltip="Surface elevation column if available.",
        ),
        FieldSpec(
            key="include_z_surf_as_static",
            label="Include z_surf as static",
            sec_id="data_semantics",
            kind="check",
        ),
        FieldSpec(
            key="head_col",
            label="Head output name",
            sec_id="data_semantics",
            kind="line",
            tooltip="Target column name for head after conversion.",
        ),
        FieldSpec(
            key="gwl_dyn_index",
            label="GWL dynamic index",
            sec_id="data_semantics",
            kind="opt_spin",
            advanced=True,
            tooltip="Index of GWL inside dynamic arrays (if used).",
        ),
        FieldSpec(
            key="subsidence_kind",
            label="Subsidence kind",
            sec_id="data_semantics",
            kind="combo",
            items=items["subsidence_kind"],
        ),
        # ------------- coords & CRS ------------
        FieldSpec(
            key="coord_mode",
            label="Coordinate mode",
            sec_id="coords",
            kind="combo",
            items=items["coord_mode"],
        ),
        FieldSpec(
            key="coord_src_epsg",
            label="Source EPSG",
            sec_id="coords",
            kind="spin",
            min_value=0,
            max_value=999999,
        ),
        FieldSpec(
            key="utm_epsg",
            label="UTM EPSG",
            sec_id="coords",
            kind="spin",
            min_value=0,
            max_value=999999,
        ),
        FieldSpec(
            key="normalize_coords",
            label="Normalize coords",
            sec_id="coords",
            kind="check",
            advanced=True,
        ),
        FieldSpec(
            key="keep_coords_raw",
            label="Keep raw coords",
            sec_id="coords",
            kind="check",
            advanced=True,
        ),
        FieldSpec(
            key="shift_raw_coords",
            label="Shift raw coords",
            sec_id="coords",
            kind="check",
            advanced=True,
        ),
    ]

    return {s.key: s for s in specs}


def fields_for_section(
    sec_id: str,
    *,
    include_advanced: bool = True,
) -> List[FieldSpec]:
    out: List[FieldSpec] = []
    for f in default_fields().values():
        if f.sec_id != sec_id:
            continue
        if not include_advanced and f.advanced:
            continue
        out.append(f)
    return out