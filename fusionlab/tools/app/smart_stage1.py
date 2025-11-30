from __future__ import annotations

import json
import os
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Stage1Summary:
    city: str
    model: str
    stage: str
    timestamp: str

    manifest_path: Path
    run_dir: Path

    time_steps: int
    horizon_years: int
    train_end_year: int
    forecast_start_year: int

    n_train: int
    n_val: int

    is_complete: bool
    completeness_errors: List[str]

    config_match: bool
    config_diffs: Dict[str, Tuple[Any, Any]] 
    diff_fields: List[str] = field(default_factory=list)

    @property
    def stage1_dir(self) -> Path:
        """Compatibility alias for older code."""
        return self.run_dir
    
REQUIRED_NUMPY_KEYS = [
    "train_inputs_npz",
    "train_targets_npz",
    "val_inputs_npz",
    "val_targets_npz",
]

REQUIRED_CSV_KEYS = [
    "raw",
    "clean",
    "scaled",
]


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_files_exist(paths: Dict[str, str],
                       required_keys: List[str],
                       section: str) -> List[str]:
    errors: List[str] = []
    for key in required_keys:
        p = paths.get(key)
        if not p or not os.path.exists(p):
            errors.append(f"[{section}] Missing or not found: {key} -> {p}")
    return errors


def validate_stage1_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Return list of problems; empty list => manifest is OK."""
    errs: List[str] = []

    artifacts = manifest.get("artifacts", {})
    csv_paths = artifacts.get("csv", {})
    npy_paths = artifacts.get("numpy", {})

    errs.extend(_check_files_exist(csv_paths, REQUIRED_CSV_KEYS, "csv"))
    errs.extend(_check_files_exist(npy_paths, REQUIRED_NUMPY_KEYS, "numpy"))

    # Check basic shapes > 0
    shapes = artifacts.get("shapes", {})
    train_targets = shapes.get("train_targets", {})
    subs_shape = train_targets.get("subs_pred") or []
    if not subs_shape or subs_shape[0] <= 0:
        errs.append("train_targets.subs_pred has zero samples or missing shape.")

    return errs

def compare_stage1_configs(
    manifest_cfg: Dict[str, Any],
    current_cfg: Optional[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Tuple[Any, Any]]]:
    """
    Return (config_match, diffs) between manifest and current GUI config.

    diffs maps key -> (value_in_manifest, value_in_current_cfg).
    Keys are e.g. "TIME_STEPS", "features.dynamic", "cols", "censoring".
    """
    if not current_cfg:
        # No GUI snapshot – treat as "no comparison requested".
        return True, {}

    diffs: Dict[str, Tuple[Any, Any]] = {}

    # 1) Simple scalar keys
    for k in (
        "TIME_STEPS",
        "FORECAST_HORIZON_YEARS",
        "TRAIN_END_YEAR",
        "FORECAST_START_YEAR",
        "MODE",
    ):
        v_m = manifest_cfg.get(k)
        v_c = current_cfg.get(k)
        if v_m != v_c:
            diffs[k] = (v_m, v_c)

    # 2) Feature sets (compare as sets to ignore ordering)
    feat_m = manifest_cfg.get("features") or {}
    feat_c = current_cfg.get("features") or {}
    for sub in ("static", "dynamic", "future", "group_id_cols"):
        v_m = list(feat_m.get(sub) or [])
        v_c = list(feat_c.get(sub) or [])
        if sorted(v_m) != sorted(v_c):
            diffs[f"features.{sub}"] = (v_m, v_c)

    # 3) Cols mapping
    cols_m = manifest_cfg.get("cols") or {}
    cols_c = current_cfg.get("cols") or {}
    if cols_m and cols_c and cols_m != cols_c:
        diffs["cols"] = (cols_m, cols_c)

    # 4) Censoring – compare specs
    c_m = manifest_cfg.get("censoring") or {}
    c_c = current_cfg.get("censoring") or {}
    if c_m.get("specs") != c_c.get("specs"):
        diffs["censoring"] = (c_m, c_c)

    return (len(diffs) == 0), diffs


def make_stage1_summary(
    manifest_path: Path,
    current_cfg: Optional[Dict[str, Any]] = None,
) -> Stage1Summary:
    m = _load_manifest(manifest_path)

    cfg = m.get("config", {})
    artifacts = m.get("artifacts", {})
    shapes = artifacts.get("shapes", {})

    train_targets = shapes.get("train_targets", {})
    val_targets = shapes.get("val_targets", {})

    n_train = (train_targets.get("subs_pred") or [0])[0]
    n_val = (val_targets.get("subs_pred") or [0])[0]

    completeness_errors = validate_stage1_manifest(m)
    is_complete = len(completeness_errors) == 0

    if current_cfg is not None:
        config_match, config_diffs = compare_stage1_configs(cfg, current_cfg)
        diff_fields = list(config_diffs.keys())
    else:
        config_match, config_diffs, diff_fields = True, {}, []

    return Stage1Summary(
        city=m.get("city", "unknown"),
        model=m.get("model", "unknown"),
        stage=m.get("stage", "unknown"),
        timestamp=m.get("timestamp", ""),
        manifest_path=manifest_path,
        run_dir=Path(m.get("paths", {}).get("run_dir", manifest_path.parent)),
        time_steps=cfg.get("TIME_STEPS", -1),
        horizon_years=cfg.get("FORECAST_HORIZON_YEARS", -1),
        train_end_year=cfg.get("TRAIN_END_YEAR", -1),
        forecast_start_year=cfg.get("FORECAST_START_YEAR", -1),
        n_train=n_train,
        n_val=n_val,
        is_complete=is_complete,
        completeness_errors=completeness_errors,
        config_match=config_match,
        config_diffs=config_diffs,
        diff_fields=diff_fields,
    )


def discover_stage1_runs(
    results_root: Path,
    current_cfg: Optional[Dict[str, Any]] = None,
) -> List[Stage1Summary]:
    results_root = Path(results_root)
    summaries: List[Stage1Summary] = []

    for manifest_path in results_root.rglob("manifest.json"):
        try:
            m = _load_manifest(manifest_path)
        except Exception:
            continue

        if m.get("stage") != "stage1":
            continue

        try:
            s = make_stage1_summary(manifest_path, current_cfg=current_cfg)
        except Exception:
            continue

        summaries.append(s)

    summaries.sort(key=lambda s: s.timestamp)
    return summaries


def find_stage1_for_city(
    city: str,
    results_root: Path,
    current_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Stage1Summary], List[Stage1Summary]]:
    """
    Return (matching_city, all_summaries) for UI.

    matching_city = Stage-1 runs for this city (complete/incomplete).
    """
    all_runs = discover_stage1_runs(results_root, current_cfg=current_cfg)
    matching_city = [s for s in all_runs if s.city.lower() == city.lower()]
    return matching_city, all_runs


def build_stage1_cfg_from_nat(
    *,
    base_cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    feature_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve the minimal Stage-1 config used for compatibility checks.

    This function is GUI-friendly: it takes the nat.com-style base
    config and optional overrides (from GeoPriorConfig.to_cfg_overrides)
    and returns only the fields that Stage-1 cares about.

    Parameters
    ----------
    base_cfg : dict
        nat.com config as loaded from config.json.
    overrides : dict, optional
        Flat overrides (e.g. from GeoPriorConfig.to_cfg_overrides).
        These are applied on top of base_cfg.
    feature_overrides : dict, optional
        Optional feature overrides from GeoPriorConfig.feature_overrides.

    Returns
    -------
    cfg : dict
        Sub-config with keys:
        TIME_STEPS, FORECAST_HORIZON_YEARS, TRAIN_END_YEAR,
        FORECAST_START_YEAR, MODE, censoring, features, cols.
    """
    cfg = copy.deepcopy(base_cfg) if base_cfg else {}
    if overrides:
        # Simple shallow override is enough for the pieces we care about.
        cfg.update(overrides)

    out: Dict[str, Any] = {
        "TIME_STEPS": cfg.get("TIME_STEPS"),
        "FORECAST_HORIZON_YEARS": cfg.get("FORECAST_HORIZON_YEARS"),
        "TRAIN_END_YEAR": cfg.get("TRAIN_END_YEAR"),
        "FORECAST_START_YEAR": cfg.get("FORECAST_START_YEAR"),
        "MODE": cfg.get("MODE"),
        "censoring": cfg.get("censoring", {}) or {},
    }

    # ---- features ----
    features = copy.deepcopy(cfg.get("features", {})) or {}
    if feature_overrides:
        # naive merge: override per-section (static/dynamic/future/…)
        for key, val in feature_overrides.items():
            features[key] = val
    if features:
        out["features"] = features

    # ---- cols ----
    cols = cfg.get("cols")
    if cols:
        out["cols"] = cols

    return out


