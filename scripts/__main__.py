# scripts/__main__.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class _CmdSpec:
    mod: str
    fn: str
    desc: str
    mode: str = "argv"  # "argv" | "sysargv"


# ---------------------------------------------------------------------
# Registry (primary names are hyphenated)
# ---------------------------------------------------------------------

_CMD: Dict[str, _CmdSpec] = {
    # Main paper figures
    "plot-driver-response": _CmdSpec(
        "plot_driver_response",
        "plot_driver_response_main",
        "Driver-response figure.",
    ),
    "plot-core-ablation": _CmdSpec(
        "plot_core_ablation",
        "plot_fig3_core_ablation_main",
        "Core ablation figure.",
    ),
    "plot-litho-parity": _CmdSpec(
        "plot_litho_parity",
        "figS1_lithology_parity_main",
        "Lithology parity figure.",
    ),
    "plot-uncertainty": _CmdSpec(
        "plot_uncertainty",
        "plot_fig5_uncertainty_main",
        "Forecast uncertainty figure.",
    ),
    "plot-spatial-forecasts": _CmdSpec(
        "plot_spatial_forecasts",
        "plot_fig6_spatial_forecasts_main",
        "Spatial forecast maps.",
    ),
    # Supplementary / appendix
    "plot-physics-sanity": _CmdSpec(
        "plot_physics_sanity",
        "plot_physics_sanity_main",
        "Physics sanity plots.",
    ),
    "plot-physics-maps": _CmdSpec(
        "plot_physics_maps",
        "plot_physics_maps_main",
        "Physics maps plots.",
    ),
    "plot-physics-fields": _CmdSpec(
        "plot_physics_fields",
        "plot_physics_fields_main",
        "Physics fields plots.",
    ),
    "plot-physics-profiles": _CmdSpec(
        "plot_physics_profiles",
        "figA1_phys_profiles_main",
        "Physics profiles (Appendix).",
    ),
    "plot-uncertainty-extras": _CmdSpec(
        "plot_uncertainty_extras",
        "supp_figS5_uncertainty_extras_main",
        "Extra uncertainty panels.",
    ),
    "plot-ablations-sensitivity": _CmdSpec(
        "plot_ablations_sensitivity",
        "plot_ablations_sensivity_main",
        "Ablations sensitivity.",
    ),
    "plot-physics-sensitivity": _CmdSpec(
        "plot_physics_sensitivity",
        "plot_physics_sensitivity_main",
        "Physics sensitivity.",
    ),
    "plot-sm3-identifiability": _CmdSpec(
        "plot_sm3_identifiability",
        "plot_sm3_identifiability_main",
        "SM3 identifiability figure.",
    ),
    "plot-sm3-bounds-ridge-summary": _CmdSpec(
        "plot_sm3_bounds_ridge_summary",
        "plot_sm3_bounds_ridge_summary_main",
        "SM3 bounds vs ridge summary.",
    ),
    "plot-sm3-log-offsets": _CmdSpec(
        "plot_sm3_log_offsets",
        "plot_sm3_log_offsets_main",
        "SM3 log-offset diagnostics.",
    ),
    "plot-xfer-transferability": _CmdSpec(
        "plot_xfer_transferability",
        "figSx_xfer_transferability_main",
        "Cross-city transferability.",
    ),
    "plot-xfer-impact": _CmdSpec(
        "plot_xfer_impact",
        "figSx_xfer_impact_main",
        "Transfer impact (retention + risk).",
    ),
    "plot-transfer": _CmdSpec(
        "plot_transfer",
        "figSx_xfer_transferability_main",
        "Alias of transferability plot.",
    ),
    "plot-transfer-impact": _CmdSpec(
        "plot_xfer_impact",
        "figSx_xfer_impact_main",
        "Alias of transfer impact plot.",
    ),
    "plot-geo-cumulative": _CmdSpec(
        "plot_geo_cumulative",
        "main",
        "Cumulative geo curves.",
        mode="sysargv",
    ),
    # Tables / summaries
    "compute-brier-exceedance": _CmdSpec(
        "compute_brier_exceedance",
        "brier_exceedance_main",
        "Compute exceedance Brier table.",
    ),
    "summarize-hotspots": _CmdSpec(
        "summarize_hotspots",
        "summarize_hotspots_main",
        "Summarize hotspot outputs.",
    ),
    "update-ablation-records": _CmdSpec(
        "update_ablation_records",
        "update_ablation_records_main",
        "Patch ablation_record.jsonl with post-hoc metrics.",
    ),

    "build-model-metrics": _CmdSpec(
        "build_model_metrics",
        "build_model_metrics_main",
        "Build unified metrics tables (CSV/JSON).",
    ),
 
}


# Backward-compatible aliases
_ALIASES: Dict[str, str] = {
    "plot_uncertainty_extras": "plot-uncertainty-extras",
    "plot-sm3-bounds-ridge": "plot-sm3-bounds-ridge-summary",
}


# Help groups (only presentation; does not affect CLI)
_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "Figures",
        (
            "plot-driver-response",
            "plot-core-ablation",
            "plot-litho-parity",
            "plot-uncertainty",
            "plot-spatial-forecasts",
            "plot-transfer-impact",
        ),
    ),
    (
        "Supplementary",
        (
            "plot-physics-sanity",
            "plot-physics-profiles",
            "plot-uncertainty-extras",
            "plot-ablations-sensitivity",
            "plot-physics-sensitivity",
            "plot-sm3-identifiability",
            "plot-sm3-bounds-ridge-summary",
            "plot-sm3-log-offsets",
            "plot-xfer-transferability",
            "plot-transfer",
            "plot-xfer-impact",
            "plot-geo-cumulative",
        ),
    ),
    (
        "Tables & summaries",
        (
            "compute-brier-exceedance",
            "summarize-hotspots",
            "update-ablation-records", 
            "build-model-metrics"
        ),
    ),
)


def _load_callable(spec: _CmdSpec) -> Callable[..., None]:
    mod = importlib.import_module(
        f".{spec.mod}",
        package=__package__,
    )
    fn = getattr(mod, spec.fn, None)
    if fn is None:
        raise AttributeError(
            f"Missing '{spec.fn}' in {spec.mod}.py"
        )
    return fn


def _call_with_sysargv(
    fn: Callable[[], None],
    cmd: str,
    argv: Optional[List[str]],
) -> None:
    old = list(sys.argv)
    sys.argv = [f"python -m scripts {cmd}"]
    if argv:
        sys.argv += list(argv)
    try:
        fn()
    finally:
        sys.argv = old


def _print_group(title: str, cmds: Tuple[str, ...]) -> None:
    print(f"{title}:")

    width = 30
    for c in cmds:
        if c not in _CMD:
            continue
        d = _CMD[c].desc
        left = ("  " + c).ljust(width)
        print(f"{left}{d}")

    print("")


def _print_help() -> None:
    print("Usage:")
    print("  python -m scripts <command> [args]")
    print("")
    print("Commands:")
    print("")

    for title, cmds in _GROUPS:
        _print_group(title, cmds)

    # Any commands not listed in groups
    grouped = {c for _, cs in _GROUPS for c in cs}
    extra = sorted(set(_CMD) - grouped)
    if extra:
        _print_group("Other", tuple(extra))

    if _ALIASES:
        print("Aliases:")
        for k in sorted(_ALIASES):
            print(f"  {k} -> {_ALIASES[k]}")
        print("")

    print("Tip:")
    print("  python -m scripts <command> -h")


def main(argv: Optional[List[str]] = None) -> None:
    args = list(argv) if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return

    cmd = args[0]
    rest = args[1:]

    cmd = _ALIASES.get(cmd, cmd)
    spec = _CMD.get(cmd)

    if spec is None:
        print(f"[ERR] Unknown command: {cmd}")
        _print_help()
        raise SystemExit(2)

    fn = _load_callable(spec)

    if spec.mode == "sysargv":
        _call_with_sysargv(fn, cmd, rest)
        return

    fn(rest)


if __name__ == "__main__":
    main()
