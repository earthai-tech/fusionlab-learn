from __future__ import annotations

import json
import os
import copy
import hashlib

from dataclasses import dataclass, field
from pathlib import Path
from typing import ( 
    Any, 
    Dict, 
    List, 
    Optional, 
    Tuple, 
    Union, 
    Mapping, 
    Iterable, 
)

PathLike = Union[str, Path]

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


@dataclass(frozen=True)
class Stage1Bundle:
    run_dir: Path
    manifest_path: Path
    audit_path: Optional[Path]
    artifacts_dir: Optional[Path]


def _norm_stage(v: Any) -> str:
    s = str(v or "").strip().lower()
    s = s.replace("-", "").replace("_", "").replace(" ", "")
    return s


def _looks_like_stage1_dir(d: Path) -> bool:
    mf = d / "manifest.json"
    if not mf.is_file():
        return False
    if (d / "stage1_scaling_audit.json").is_file():
        return True
    if (d / "artifacts").is_dir():
        return True
    return "stage1" in d.name.lower()


# -------------------------------
# Canonicalization helpers
# -------------------------------

def _is_primitive(x: Any) -> bool:
    return x is None or isinstance(x, (bool, int, float, str))


def _to_str(x: Any) -> str:
    # Avoid nondeterministic reprs where possible
    if isinstance(x, Path):
        return str(x)
    return str(x)


def _canonicalize(obj: Any) -> Any:
    """
    Convert obj to a JSON-stable, deterministic structure.

    - dict: keys sorted, values canonicalized
    - set/frozenset: sorted list (canonicalized elements)
    - list/tuple: list (canonicalized elements)
    - Path: str
    - unknown objects: str(obj)
    """
    if _is_primitive(obj):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    # dict-like
    if isinstance(obj, Mapping):
        # stringify keys to avoid non-JSON keys
        items = []
        for k, v in obj.items():
            ks = k if isinstance(k, str) else _to_str(k)
            items.append((ks, _canonicalize(v)))
        items.sort(key=lambda kv: kv[0])
        return {k: v for k, v in items}

    # set-like: order independent
    if isinstance(obj, (set, frozenset)):
        lst = [_canonicalize(x) for x in obj]
        # stable sort on JSON representation
        lst.sort(key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
        return lst

    # sequence-like
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]

    # fallback: stringify
    return _to_str(obj)


def _sorted_unique(seq: Iterable[Any]) -> list:
    """
    Convert a feature list/set to a sorted unique list of strings.
    """
    out = []
    seen = set()
    for x in seq:
        xs = x if isinstance(x, str) else _to_str(x)
        xs = xs.strip()
        if not xs:
            continue
        if xs in seen:
            continue
        seen.add(xs)
        out.append(xs)
    out.sort()
    return out


# -------------------------------
# Reduced Stage-1 signature
# -------------------------------

def stage1_cfg_signature(cfg: Any) -> Dict[str, Any]:
    """
    Build a reduced, stable signature for Stage-1 compatibility / caching.

    Goal: same signature => same Stage-1 meaning (so caching is safe).

    We intentionally keep only the fields that affect Stage-1 outputs
    and normalize order-insensitive structures (like feature lists).
    """
    if not isinstance(cfg, Mapping):
        return {}

    # Common top-level keys your Stage-1 pipeline cares about
    core_keys = (
        "schema_version",
        "model",
        "stage",
        "MODE",
        "TIME_STEPS",
        "FORECAST_HORIZON_YEARS",
        "TRAIN_END_YEAR",
        "FORECAST_START_YEAR",
        "seed",
    )

    sig: Dict[str, Any] = {}
    for k in core_keys:
        if k in cfg:
            sig[k] = cfg.get(k)

    # Column mapping is important
    cols = cfg.get("cols", {})
    if isinstance(cols, Mapping):
        # keep it fully (it’s small) but canonicalized
        sig["cols"] = cols

    # Features: normalize to sorted lists (order-insensitive)
    feats = cfg.get("features", {})
    if isinstance(feats, Mapping):
        norm_feats: Dict[str, Any] = {}
        for group, items in feats.items():
            g = group if isinstance(group, str) else _to_str(group)
            if items is None:
                norm_feats[g] = []
            elif isinstance(items, (list, tuple, set, frozenset)):
                norm_feats[g] = _sorted_unique(items)
            else:
                # sometimes a single string sneaks in
                norm_feats[g] = _sorted_unique([items])
        # stable order of groups
        sig["features"] = {k: norm_feats[k] for k in sorted(norm_feats.keys())}

    # Optional: include ONLY if they affect Stage-1 artifacts in your pipeline
    # (keep these if you know Stage-1 uses them)
    optional_keys = (
        "stage1_options",
        "scaling",
        "scaler",
        "feature_scaling",
        "target_scaling",
        "censoring",
        "clip",
        "imputation",
        "nan_policy",
    )
    for k in optional_keys:
        if k in cfg:
            sig[k] = cfg.get(k)

    return sig


# -------------------------------
# Canonical hash function
# -------------------------------

def canonical_hash_cfg(cfg: Any) -> str:
    """
    Stable hash for caching. Uses reduced Stage-1 signature + canonicalization.

    Returns "" if cfg is None/unhashable in a meaningful way.
    """
    if cfg is None:
        return ""

    try:
        sig = stage1_cfg_signature(cfg)
        canon = _canonicalize(sig)
        s = json.dumps(
            canon,
            sort_keys=True,
            separators=(",", ":"),  # remove whitespace for stability
            ensure_ascii=False,
        )
        # sha256 is safer than md5; still fast
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    except Exception:
        # last resort: still be stable-ish
        try:
            s = json.dumps(_canonicalize(cfg), sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(s.encode("utf-8")).hexdigest()
        except Exception:
            return ""

def _is_stage1_manifest(m: dict, mf_path: Path) -> bool:
    tag = _norm_stage(m.get("stage"))
    if tag in ("stage1", "1"):
        return True
    if tag not in ("", "unknown"):
        return False
    return _looks_like_stage1_dir(mf_path.parent)


def resolve_stage1_bundle(
    *,
    results_root: Path,
    city: str = "",
    model: str = "GeoPriorSubsNet",
) -> Optional[Stage1Bundle]:
    rr = results_root.expanduser()

    # Case A: rr itself is the stage1 dir
    if _looks_like_stage1_dir(rr):
        d = rr
        return Stage1Bundle(
            run_dir=d,
            manifest_path=d / "manifest.json",
            audit_path=(
                d / "stage1_scaling_audit.json"
                if (d / "stage1_scaling_audit.json").is_file()
                else None
            ),
            artifacts_dir=(
                d / "artifacts"
                if (d / "artifacts").is_dir()
                else None
            ),
        )

    # Case B: typical layout under global results root
    if city:
        cand = rr / f"{city}_{model}_stage1"
        if _looks_like_stage1_dir(cand):
            d = cand
            return Stage1Bundle(
                run_dir=d,
                manifest_path=d / "manifest.json",
                audit_path=(
                    d / "stage1_scaling_audit.json"
                    if (d / "stage1_scaling_audit.json").is_file()
                    else None
                ),
                artifacts_dir=(
                    d / "artifacts"
                    if (d / "artifacts").is_dir()
                    else None
                ),
            )

    # Fallback: search (bounded)
    for mf in rr.rglob("manifest.json"):
        d = mf.parent
        if not _looks_like_stage1_dir(d):
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _is_stage1_manifest(m, mf):
            continue
        if city and city.lower() not in d.name.lower():
            continue
        return Stage1Bundle(
            run_dir=d,
            manifest_path=mf,
            audit_path=(
                d / "stage1_scaling_audit.json"
                if (d / "stage1_scaling_audit.json").is_file()
                else None
            ),
            artifacts_dir=(
                d / "artifacts"
                if (d / "artifacts").is_dir()
                else None
            ),
        )

    return None


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
    
    structured_keys = ("static", "dynamic", "future", "group_id_cols")
    has_structured = any(
        k in feat_m or k in feat_c for k in structured_keys
    )
    
    if has_structured:
        # NAT-style features: compare per section
        for sub in structured_keys:
            v_m = list(feat_m.get(sub) or [])
            v_c = list(feat_c.get(sub) or [])
            if sorted(v_m) != sorted(v_c):
                diffs[f"features.{sub}"] = (v_m, v_c)
    else:
        # Pure-style (OPTIONAL_* etc.) – fall back to direct dict equality
        if feat_m != feat_c:
            diffs["features"] = (feat_m, feat_c)
    
    # 3) Cols mapping
    cols_m = manifest_cfg.get("cols") or {}
    cols_c = current_cfg.get("cols") or {}
    if cols_m and cols_c and cols_m != cols_c:
        diffs["cols"] = (cols_m, cols_c)
        
    def _extract_censor_specs(c: Dict[str, Any]) -> Any:
        if not isinstance(c, dict):
            return None
        # NAT-style
        if "specs" in c:
            return c["specs"]
        # Pure GeoPrior-style
        if "CENSORING_SPECS" in c:
            return c["CENSORING_SPECS"]
        return None
    
    # 4) Censoring – compare specs
    c_m = manifest_cfg.get("censoring") or {}
    c_c = current_cfg.get("censoring") or {}
    
    specs_m = _extract_censor_specs(c_m)
    specs_c = _extract_censor_specs(c_c)
    if specs_m != specs_c:
        diffs["censoring"] = (c_m, c_c)

    return (len(diffs) == 0), diffs


def pick_best_stage1_run(
    runs: List["Stage1Summary"],
) -> Optional["Stage1Summary"]:
    if not runs:
        return None
    ordered = sorted(runs, key=lambda s: s.timestamp)
    for s in reversed(ordered):
        if s.is_complete and s.config_match:
            return s
    return ordered[-1]


def find_stage1_for_city_root(
    *,
    city_root: Path,
    current_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List["Stage1Summary"], List["Stage1Summary"]]:
    """
    Like find_stage1_for_city, but searches only inside <city_root>
    (fast; avoids scanning the whole results_root).
    """
    city_root = Path(city_root)
    if not city_root.is_dir():
        return [], []

    manifests: List[Path] = []

    # Preferred layout: <city_root>/artifacts/manifest.json
    mf0 = city_root / "manifest.json" # "artifacts" / 
    if mf0.is_file():
        manifests.append(mf0)

    # Also pick up nested stage1 dirs if any (bounded search)
    for mf in city_root.rglob("manifest.json"):
        if mf == mf0:
            continue
        try:
            m = _load_manifest(mf)
        except Exception:
            continue
        if not _is_stage1_manifest(m, mf):
            continue
        manifests.append(mf)

    summaries: List[Stage1Summary] = []
    for mf in manifests:
        try:
            summaries.append(
                make_stage1_summary(mf, current_cfg=current_cfg)
            )
        except Exception:
            pass

    summaries.sort(key=lambda s: s.timestamp)
    return summaries, summaries

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

        if not _is_stage1_manifest(m, manifest_path):
            continue

        try:
            s = make_stage1_summary(
                manifest_path,
                current_cfg=current_cfg,
            )
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
    b = resolve_stage1_bundle(
        results_root=results_root,
        city=city,
    )
    if b is not None:
        try:
            s = make_stage1_summary(
                b.manifest_path,
                current_cfg=current_cfg,
            )
            return [s], [s]
        except Exception:
            pass

    all_runs = discover_stage1_runs(
        results_root,
        current_cfg=current_cfg,
    )
    matching_city = [
        s for s in all_runs
        if s.city.lower() == city.lower()
    ]
    return matching_city, all_runs

# def find_stage1_for_city(
#     city: str,
#     results_root: Path,
#     current_cfg: Optional[Dict[str, Any]] = None,
# ) -> Tuple[List[Stage1Summary], List[Stage1Summary]]:
#     """
#     Return (matching_city, all_summaries) for UI.

#     matching_city = Stage-1 runs for this city (complete/incomplete).
#     """
#     all_runs = discover_stage1_runs(results_root, current_cfg=current_cfg)
#     matching_city = [s for s in all_runs if s.city.lower() == city.lower()]
#     return matching_city, all_runs


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

def load_json(
    path: Path | str | None,
    *,
    default: Any = None,
    encoding: str = "utf-8",
) -> Any:
    """
    Robust JSON loader for GUI discovery.

    - Accepts Path/str/None
    - Expands "~" and environment variables
    - Returns `default` if missing/unreadable/invalid JSON

    Parameters
    ----------
    path : Path | str | None
        JSON file path.
    default : Any, optional
        Value returned on failure (missing, invalid, etc.).
    encoding : str, default="utf-8"
        File encoding.

    Returns
    -------
    Any
        Parsed JSON on success, otherwise `default`.
    """
    if path is None:
        return default

    try:
        p = Path(path)
    except Exception:
        return default

    try:
        p = Path(os.path.expandvars(str(p))).expanduser()
    except Exception:
        # If expandvars fails, try plain expanduser
        try:
            p = p.expanduser()
        except Exception:
            return default

    if not p.is_file():
        return default

    try:
        with p.open("r", encoding=encoding) as f:
            return json.load(f)
    except Exception:
        return default


# def load_json(
#     path: PathLike,
#     *,
#     default: Any = None,
# ) -> Any:
#     p = Path(path).expanduser()

#     if not p.exists() or not p.is_file():
#         return default

#     try:
#         txt = p.read_text(encoding="utf-8-sig")
#     except Exception:
#         try:
#             txt = p.read_text(encoding="utf-8")
#         except Exception:
#             return default

#     if not (txt or "").strip():
#         return default

#     try:
#         return json.loads(txt)
#     except Exception:
#         return default
