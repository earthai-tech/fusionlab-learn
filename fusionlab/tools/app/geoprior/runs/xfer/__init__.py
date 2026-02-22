# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.runs.xfer

Exports only.
"""

from .xfer_core import (
    WarmStartConfig,
    XferCase,
    XferPlan,
    build_plan,
    iter_cases,
    run_case,
    run_plan,
)

from .xfer_utils import (
    LogFn,
    align_static_to_source,
    best_model_artifact,
    check_transfer_schema,
    choose_warm_idx,
    ensure_dir,
    find_stage1_manifest,
    infer_input_dims,
    load_calibrator_near,
    load_stage1_manifest,
    manifest_run_dir,
    now_tag,
    pick_npz,
    reproject_dynamic_to_source,
    resolve_bundle_paths,
    safe_load_json,
    slice_npz_dict,
)

__all__ = [
    "LogFn",
    "WarmStartConfig",
    "XferCase",
    "XferPlan",
    "align_static_to_source",
    "best_model_artifact",
    "build_plan",
    "check_transfer_schema",
    "choose_warm_idx",
    "ensure_dir",
    "find_stage1_manifest",
    "infer_input_dims",
    "iter_cases",
    "load_calibrator_near",
    "load_stage1_manifest",
    "manifest_run_dir",
    "now_tag",
    "pick_npz",
    "reproject_dynamic_to_source",
    "resolve_bundle_paths",
    "run_case",
    "run_plan",
    "safe_load_json",
    "slice_npz_dict",
]
