# geoprior/ui/tools/identifiability_core.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from fusionlab.nn.pinn.geoprior.payloads import (
    load_physics_payload,
    identifiability_diagnostics_from_payload,
)

SEC_PER_YEAR = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class PayloadBundle:
    payload: Dict[str, np.ndarray]
    meta: Dict[str, Any]
    payload_units: str
    report_units: str
    sec_per_year: float
    units_reason: str


def infer_payload_time_units(
    meta: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Infer the *units of payload arrays* (tau, K).

    v3.2 rule of thumb:
    - If nothing explicit is present, assume SI:
      tau in sec, K in m/s.

    Priority:
      1) meta["units"]["tau"]
      2) meta["payload_time_units"]
      3) meta["payload_units"] (if you add it later)
      4) meta["time_units"] (weak hint)
      5) default -> "sec"
    """
    units = meta.get("units") or {}
    tau_u = str(units.get("tau", "")).strip().lower()

    if "year" in tau_u:
        return "year", "meta.units.tau"
    if tau_u in ("s", "sec", "second", "seconds"):
        return "sec", "meta.units.tau"
    if "sec" in tau_u:
        return "sec", "meta.units.tau"

    pu = str(meta.get("payload_time_units", "")).strip().lower()
    if pu.startswith("y"):
        return "year", "meta.payload_time_units"
    if pu.startswith("s"):
        return "sec", "meta.payload_time_units"

    pu2 = str(meta.get("payload_units", "")).strip().lower()
    if pu2.startswith("y"):
        return "year", "meta.payload_units"
    if pu2.startswith("s"):
        return "sec", "meta.payload_units"

    tu = str(meta.get("time_units", "")).strip().lower()
    if tu.startswith("y"):
        return "year", "meta.time_units (weak)"
    if tu.startswith("s"):
        return "sec", "meta.time_units (weak)"

    return "sec", "default(v3.2 SI)"


def infer_report_time_units(
    meta: Dict[str, Any],
    ui_choice: str,
    *,
    default: str = "year",
) -> str:
    """
    Decide what the *user wants to see* (reporting units).
    """
    c = (ui_choice or "auto").strip().lower()
    if c in ("sec", "s"):
        return "sec"
    if c in ("year", "yr", "y"):
        return "year"

    ru = str(meta.get("report_time_units", "")).strip().lower()
    if ru.startswith("s"):
        return "sec"
    if ru.startswith("y"):
        return "year"

    return default


def convert_payload_time_units(
    payload: Dict[str, np.ndarray],
    *,
    from_units: str,
    to_units: str,
    sec_per_year: float,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Convert tau/K fields in a payload between:
      - sec  : tau in seconds, K in m/s
      - year : tau in years,   K in m/year

    Converts:
      - tau, tau_prior, tau_closure (if present)
      - K
    Refreshes log10_* fields if they exist.
    """
    fu = (from_units or "sec").strip().lower()
    tu = (to_units or "sec").strip().lower()
    fu = "year" if fu.startswith("y") else "sec"
    tu = "year" if tu.startswith("y") else "sec"

    out = dict(payload)
    if fu == tu:
        return _refresh_logs(out, eps=eps)

    spy = float(sec_per_year)

    def _a(x):
        return np.asarray(x, float)

    if fu == "sec" and tu == "year":
        if "tau" in out:
            out["tau"] = _a(out["tau"]) / spy
        if "tau_prior" in out:
            out["tau_prior"] = _a(out["tau_prior"]) / spy
        if "tau_closure" in out:
            out["tau_closure"] = _a(out["tau_closure"]) / spy
        if "K" in out:
            out["K"] = _a(out["K"]) * spy

    if fu == "year" and tu == "sec":
        if "tau" in out:
            out["tau"] = _a(out["tau"]) * spy
        if "tau_prior" in out:
            out["tau_prior"] = _a(out["tau_prior"]) * spy
        if "tau_closure" in out:
            out["tau_closure"] = _a(out["tau_closure"]) * spy
        if "K" in out:
            out["K"] = _a(out["K"]) / spy

    for k in ("tau", "tau_prior", "tau_closure", "K"):
        if k in out:
            out[k] = np.clip(_a(out[k]), eps, None)

    return _refresh_logs(out, eps=eps)


def _refresh_logs(
    payload: Dict[str, np.ndarray],
    *,
    eps: float,
) -> Dict[str, np.ndarray]:
    out = dict(payload)

    def _safe_log10(key, src):
        if key in out and src in out:
            out[key] = np.log10(
                np.clip(np.asarray(out[src], float), eps, None)
            )

    _safe_log10("log10_tau", "tau")
    _safe_log10("log10_tau_prior", "tau_prior")
    _safe_log10("log10_tau_closure", "tau_closure")
    return out


def try_truth_prior_in_units(
    meta: Dict[str, Any],
    *,
    report_units: str,
    sec_per_year: float,
) -> Optional[Dict[str, float]]:
    """
    For SM3 synthetic payloads: extract truth/prior in report units.
    Returns None if SM3 keys not present.
    """
    ru = (report_units or "sec").strip().lower()
    ru = "year" if ru.startswith("y") else "sec"
    spy = float(sec_per_year)

    need = (
        "tau_true_year",
        "tau_prior_year",
        "tau_true_sec",
        "tau_prior_sec",
        "K_true_mps",
        "K_prior_mps",
        "Ss_true",
        "Ss_prior",
        "Hd_true",
        "Hd_prior",
    )
    if any(k not in meta for k in need):
        return None

    if ru == "year":
        return {
            "tau_true": float(meta["tau_true_year"]),
            "tau_prior": float(meta["tau_prior_year"]),
            "K_true": float(meta["K_true_mps"]) * spy,
            "K_prior": float(meta["K_prior_mps"]) * spy,
            "Ss_true": float(meta["Ss_true"]),
            "Ss_prior": float(meta["Ss_prior"]),
            "Hd_true": float(meta["Hd_true"]),
            "Hd_prior": float(meta["Hd_prior"]),
        }

    return {
        "tau_true": float(meta["tau_true_sec"]),
        "tau_prior": float(meta["tau_prior_sec"]),
        "K_true": float(meta["K_true_mps"]),
        "K_prior": float(meta["K_prior_mps"]),
        "Ss_true": float(meta["Ss_true"]),
        "Ss_prior": float(meta["Ss_prior"]),
        "Hd_true": float(meta["Hd_true"]),
        "Hd_prior": float(meta["Hd_prior"]),
    }


def load_bundle_from_npz(
    npz_path: str,
    *,
    ui_report_units: str = "auto",
    default_report: str = "year",
) -> PayloadBundle:
    payload_raw, meta = load_physics_payload(npz_path)

    payload_units, reason = infer_payload_time_units(meta)
    spy = float(meta.get("sec_per_year", SEC_PER_YEAR))
    report_units = infer_report_time_units(
        meta,
        ui_report_units,
        default=default_report,
    )

    payload = convert_payload_time_units(
        payload_raw,
        from_units=payload_units,
        to_units=report_units,
        sec_per_year=spy,
    )

    return PayloadBundle(
        payload=payload,
        meta=meta,
        payload_units=payload_units,
        report_units=report_units,
        sec_per_year=spy,
        units_reason=reason,
    )


def run_identifiability_from_npz(
    npz_path: str,
    *,
    ui_report_units: str = "auto",
) -> Tuple[PayloadBundle, Optional[Dict[str, Any]]]:
    bundle = load_bundle_from_npz(
        npz_path,
        ui_report_units=ui_report_units,
    )

    tp = try_truth_prior_in_units(
        bundle.meta,
        report_units=bundle.report_units,
        sec_per_year=bundle.sec_per_year,
    )
    if tp is None:
        return bundle, None

    diag = identifiability_diagnostics_from_payload(
        bundle.payload,
        tau_true=tp["tau_true"],
        K_true=tp["K_true"],
        Ss_true=tp["Ss_true"],
        Hd_true=tp["Hd_true"],
        K_prior=tp["K_prior"],
        Ss_prior=tp["Ss_prior"],
        Hd_prior=tp["Hd_prior"],
    )
    return bundle, diag
