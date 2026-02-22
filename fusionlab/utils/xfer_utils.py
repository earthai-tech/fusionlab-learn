# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
fusionlab.utils.xfer_utils

Stage-5 transfer helpers (I/O centric):

- Locate Stage-1 run folders under a shared results root.
- Load Stage-1 manifest.json and stage1_scaling_audit.json.
- Resolve artifact paths robustly (portable across machines).
- Load Stage-1 scalers/encoders persisted as joblib artifacts.
- Infer canonical target column names (subsidence + gwl/head).

These utilities are intentionally lightweight and avoid
business logic; Stage-5 orchestration should live in stage5.py.

Notes
-----
Manifests often store absolute Windows paths. For portability,
``resolve_artifact_path`` attempts to fall back to:

* ``<run_dir>/artifacts/<basename>``
* ``<run_dir>/<basename>``

Examples
--------
>>> from pathlib import Path
>>> from fusionlab.utils.xfer_utils import load_stage1_bundle
>>> b = load_stage1_bundle(
...     results_root=Path('results'),
...     city='nansha',
...     model='GeoPriorSubsNet',
... )
>>> b.target_cols['subs']
'subsidence_cum__si'

"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


__all__ = [
    "find_stage1_run_dir",
    "load_json",
    "load_stage1_manifest",
    "load_stage1_scaling_audit",
    "resolve_artifact_path",
    "infer_target_cols",
    "load_main_scaler",
    "load_coord_scaler",
    "load_ohe_encoders",
    "get_stage1_scaler_info",
    "get_scaled_ml_numeric_cols",
    "scale_ml_numeric_frame",
    "load_stage1_csv_path",
    "Stage1Bundle",
    "load_stage1_bundle",
]


def _as_path(p: Any) -> Path:
    if isinstance(p, Path):
        return p
    return Path(str(p))


def load_json(path: Any) -> Dict[str, Any]:
    p = _as_path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_stage1_run_dir(
    results_root: Any,
    city: str,
    model: str,
    stage: str = "stage1",
) -> Path:
    """
    Locate the Stage-1 run directory for (city, model).

    Expected folder name:
    ``<city>_<model>_<stage>``
    (e.g. ``nansha_GeoPriorSubsNet_stage1``)

    Falls back to a case-insensitive scan under results_root.
    """
    root = _as_path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"results_root not found: {root}")

    city_s = str(city).strip()
    model_s = str(model).strip()
    stage_s = str(stage).strip()

    name = f"{city_s}_{model_s}_{stage_s}"
    d = root / name
    if d.exists():
        return d

    # Case-insensitive scan
    target = name.lower()
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name.lower() == target:
            return child

    raise FileNotFoundError(
        "Stage-1 run_dir not found. expected="
        f"{name!r} under {root}"
    )


def load_stage1_manifest(run_dir: Any) -> Dict[str, Any]:
    """
    Load manifest.json from Stage-1 run directory.

    We accept either:
      - <run_dir>/manifest.json
      - <run_dir>/artifacts/manifest.json  (fallback)
    """
    d = _as_path(run_dir)
    p1 = d / "manifest.json"
    if p1.exists():
        return load_json(p1)

    p2 = d / "artifacts" / "manifest.json"
    if p2.exists():
        return load_json(p2)

    raise FileNotFoundError(
        "manifest.json not found in run_dir or artifacts: "
        f"{d}"
    )


def load_stage1_scaling_audit(
    run_dir: Any,
    allow_missing: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Load stage1_scaling_audit.json from Stage-1 run directory.

    Accepts either:
      - <run_dir>/stage1_scaling_audit.json
      - <run_dir>/artifacts/stage1_scaling_audit.json
    """
    d = _as_path(run_dir)
    p1 = d / "stage1_scaling_audit.json"
    if p1.exists():
        return load_json(p1)

    p2 = d / "artifacts" / "stage1_scaling_audit.json"
    if p2.exists():
        return load_json(p2)

    if allow_missing:
        return None

    raise FileNotFoundError(
        "stage1_scaling_audit.json not found in run_dir "
        f"or artifacts: {d}"
    )


def resolve_artifact_path(
    run_dir: Any,
    stored_path: Any,
    artifacts_dir: Optional[Any] = None,
    strict: bool = True,
) -> Path:
    """
    Resolve an artifact path stored in manifest to an actual path.

    Strategy:
    1) If stored_path exists as-is -> return it.
    2) Try basename under artifacts_dir (default: run_dir/artifacts).
    3) Try basename under run_dir.
    """
    d = _as_path(run_dir)

    if stored_path is None:
        if strict:
            raise FileNotFoundError("stored_path is None")
        return d

    p = _as_path(stored_path)

    # 1) As-is
    if p.exists():
        return p

    # 2) Basename under artifacts
    if artifacts_dir:
        art = _as_path(artifacts_dir)
    else:
        art = d / "artifacts"

    cand = art / p.name
    if cand.exists():
        return cand

    # 3) Basename under run_dir
    cand2 = d / p.name
    if cand2.exists():
        return cand2

    if strict:
        raise FileNotFoundError(
            "Artifact file not found. stored="
            f"{str(stored_path)!r} run_dir={d}"
        )

    return cand


def infer_target_cols(
    manifest: Mapping[str, Any],
    scaling_audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Infer target column names from Stage-1.

    (subsidence + groundwater target: depth/head).

    Returns:
      dict with keys:
        - "subs": SI subsidence column used by the model
        - "gwl":  SI groundwater target column used by the model
    """
    cfg = manifest.get("config", {})
    cols = cfg.get("cols", {})

    subs = cols.get("subs_model") or cols.get("subs_raw")
    if not subs:
        raise KeyError("subs cols missing in manifest.config.cols")

    # Prefer explicit target-kind from conventions; fall back to audit.
    conv = cfg.get("conventions", {})
    tgt_kind = conv.get("gwl_target_kind")

    # scaling_audit may explicitly say "targets.gwl(head)"
    aud_has_head = False
    if scaling_audit:
        ts = scaling_audit.get("targets_stats", {})
        aud_has_head = any("gwl(head" in str(k) for k in ts.keys())

    use_head = False
    if str(tgt_kind).lower().strip() == "head":
        use_head = True
    elif aud_has_head:
        use_head = True

    if use_head:
        gwl = cols.get("head_model") or cols.get("head_raw")
    else:
        gwl = cols.get("depth_model") or cols.get("depth_raw")

    if not gwl:
        raise KeyError("gwl target cols missing in manifest.config.cols")

    return {"subs": str(subs), "gwl": str(gwl)}


def get_stage1_scaler_info(
    manifest: Mapping[str, Any],
    scaling_audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract Stage-1 scaler_info mapping.

    Preferred sources:
      1) manifest['config']['scaler_info']
      2) scaling_audit['scalers']['scaler_info']
    """
    cfg = manifest.get("config", {})
    s = cfg.get("scaler_info")
    if isinstance(s, dict) and s:
        return dict(s)

    if scaling_audit:
        sc = scaling_audit.get("scalers", {})
        s2 = sc.get("scaler_info")
        if isinstance(s2, dict) and s2:
            return dict(s2)

    return {}


def get_scaled_ml_numeric_cols(
    manifest: Mapping[str, Any],
    scaling_audit: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Return ML numeric columns scaled by Stage-1."""
    art = manifest.get("artifacts", {})
    enc = art.get("encoders", {})

    cols = enc.get("scaled_ml_numeric_cols")
    if isinstance(cols, list) and cols:
        return [str(c) for c in cols]

    if scaling_audit:
        cols2 = scaling_audit.get("scaled_ml_numeric_cols")
        if isinstance(cols2, list) and cols2:
            return [str(c) for c in cols2]

    return []


def load_stage1_csv_path(
    manifest: Mapping[str, Any],
    run_dir: Any,
    kind: str = "scaled",
    strict: bool = True,
) -> Optional[Path]:
    """Return a Stage-1 CSV artifact path.

    kind: 'raw', 'clean', or 'scaled'.
    """
    kind = str(kind).strip().lower()
    csv = manifest.get("artifacts", {}).get("csv", {})
    p = csv.get(kind)

    if p is None:
        if strict:
            msg = f"manifest.artifacts.csv[{kind!r}] missing"
            raise KeyError(msg)
        return None

    return resolve_artifact_path(run_dir, p, strict=strict)


def scale_ml_numeric_frame(
    df: Any,
    main_scaler: Any,
    cols: List[str],
    copy: bool = True,
) -> Any:
    """
    Apply Stage-1 main_scaler to a DataFrame subset of columns.

    Notes:
    - This is intended for transfer runs where you want to
      transform target-city data using the source-city scaler.
    - Requires pandas at runtime.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("pandas is required") from e

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not cols:
        return df.copy() if copy else df

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    out = df.copy() if copy else df

    x = out.loc[:, cols].to_numpy()
    x2 = main_scaler.transform(x)
    out.loc[:, cols] = x2
    return out


def _joblib_load(path: Path) -> Any:
    try:
        import joblib  # type: ignore
    except Exception as e:
        raise ImportError(
            "joblib is required to load Stage-1 scalers"
        ) from e
    return joblib.load(str(path))


def load_main_scaler(
    manifest: Mapping[str, Any],
    run_dir: Any,
    strict: bool = True,
) -> Any:
    art = manifest.get("artifacts", {})
    enc = art.get("encoders", {})

    p = enc.get("main_scaler")
    if p is None:
        if strict:
            msg = "manifest missing main_scaler path"
            raise KeyError(msg)
        return None

    ap = resolve_artifact_path(
        run_dir,
        p,
        strict=strict,
    )
    return _joblib_load(ap)


def load_coord_scaler(
    manifest: Mapping[str, Any],
    run_dir: Any,
    strict: bool = True,
) -> Any:
    art = manifest.get("artifacts", {})
    enc = art.get("encoders", {})

    p = enc.get("coord_scaler")
    if p is None:
        if strict:
            msg = "manifest missing coord_scaler path"
            raise KeyError(msg)
        return None

    ap = resolve_artifact_path(
        run_dir,
        p,
        strict=strict,
    )
    return _joblib_load(ap)


def load_ohe_encoders(
    manifest: Mapping[str, Any],
    run_dir: Any,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load OHE encoders from manifest if present.

    Returns:
        dict mapping feature_name -> encoder_object
    """
    enc = manifest.get("artifacts", {}).get("encoders", {})
    ohe = enc.get("ohe", {})
    if not isinstance(ohe, dict) or not ohe:
        return {}

    out: Dict[str, Any] = {}
    for k, p in ohe.items():
        if not p:
            continue
        try:
            ap = resolve_artifact_path(
                run_dir,
                p,
                strict=True,
            )
            out[str(k)] = _joblib_load(ap)
        except Exception:
            if strict:
                raise
            continue
    return out


def load_stage1_scalers(
    manifest: Mapping[str, Any],
    run_dir: Any,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load Stage-1 scalers/encoders referenced by the manifest.
    """
    out: Dict[str, Any] = {}
    out["main_scaler"] = load_main_scaler(
        manifest,
        run_dir,
        strict=strict,
    )
    out["coord_scaler"] = load_coord_scaler(
        manifest,
        run_dir,
        strict=strict,
    )
    out["ohe"] = load_ohe_encoders(
        manifest,
        run_dir,
        strict=False,
    )
    return out


@dataclass
class Stage1Bundle:
    results_root: Path
    city: str
    model: str
    run_dir: Path

    manifest: Dict[str, Any]
    scaling_audit: Optional[Dict[str, Any]] = None

    target_cols: Dict[str, str] = field(default_factory=dict)
    scalers: Dict[str, Any] = field(default_factory=dict)
    scaler_info: Dict[str, Any] = field(default_factory=dict)
    scaled_ml_numeric_cols: Optional[List[str]] = None


def load_stage1_bundle(
    results_root: Any,
    city: str,
    model: str,
    stage: str = "stage1",
    load_scalers_flag: bool = True,
    strict: bool = True,
) -> Stage1Bundle:
    """
    Build a Stage1Bundle (manifest + audit + scalers + derived info).
    """
    run_dir = find_stage1_run_dir(
        results_root=results_root,
        city=city,
        model=model,
        stage=stage,
    )
    manifest = load_stage1_manifest(run_dir)

    aud = load_stage1_scaling_audit(run_dir, allow_missing=True)

    targets = infer_target_cols(
        manifest=manifest,
        scaling_audit=aud,
    )
    scaler_info = get_stage1_scaler_info(
        manifest=manifest,
        scaling_audit=aud,
    )
    cols = get_scaled_ml_numeric_cols(
        manifest=manifest,
        scaling_audit=aud,
    )

    scalers: Dict[str, Any] = {}
    if load_scalers_flag:
        scalers = load_stage1_scalers(
            manifest=manifest,
            run_dir=run_dir,
            strict=strict,
        )

    return Stage1Bundle(
        results_root=_as_path(results_root),
        city=str(city),
        model=str(model),
        run_dir=_as_path(run_dir),
        manifest=manifest,
        scaling_audit=aud,
        target_cols=targets,
        scalers=scalers,
        scaler_info=scaler_info,
        scaled_ml_numeric_cols=cols,
    )

