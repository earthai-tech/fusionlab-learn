# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

r"""Column-role mapping utilities for GeoPrior GUI.

This helper is UI-agnostic and is used by the Data tab to:

- assign each input column a role (Longitude, Time, etc.)
- enforce uniqueness (a role can be assigned only once)
- auto-detect common names and aliases
- build patches for:
  - GeoPriorConfig fields (``lon_col``, ``time_col`` ...)
  - UI/store ``feature_overrides`` dict keys (``LON_COL``, ``TIME_COL`` ...)
- build a rename map used when saving (normalize to canonical names)

Compatibility notes
-------------------
Your current DataTab expects the legacy methods:

- ``auto_assign()``
- ``missing_required_roles()``
- ``to_feature_overrides_patch()``

This module provides these methods as thin wrappers, while keeping
the newer API methods:

- ``missing_required(columns)``
- ``to_config_patch(reset_required=True)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class ColumnRoleSpec:
    role: str
    label: str
    cfg_field: str
    canonical: str = ""
    required: bool = False
    aliases: Tuple[str, ...] = ()


def _default_specs() -> List[ColumnRoleSpec]:
    return [
        ColumnRoleSpec(
            role="time",
            label="Time (year)",
            cfg_field="time_col",
            canonical="year",
            required=True,
            aliases=("Year", "time", "t", "date"),
        ),
        ColumnRoleSpec(
            role="lon",
            label="Longitude",
            cfg_field="lon_col",
            canonical="longitude",
            required=True,
            aliases=("lon", "x", "X", "LONG"),
        ),
        ColumnRoleSpec(
            role="lat",
            label="Latitude",
            cfg_field="lat_col",
            canonical="latitude",
            required=True,
            aliases=("lat", "y", "Y", "LAT"),
        ),
        ColumnRoleSpec(
            role="subs",
            label="Subsidence (cumulative)",
            cfg_field="subs_col",
            canonical="subsidence_cum",
            required=True,
            aliases=("subsidence", "subs", "cum_subs"),
        ),
        ColumnRoleSpec(
            role="h_field",
            label="Soil thickness (H field)",
            cfg_field="h_field_col",
            canonical="soil_thickness",
            required=True,
            aliases=("H", "thickness", "soil_thick"),
        ),
        ColumnRoleSpec(
            role="gwl",
            label="GWL depth (bgs)",
            cfg_field="gwl_col",
            canonical="GWL_depth_bgs_m",
            required=False,
            aliases=("GWL", "gwl", "water_level", "gwl_depth"),
        ),
        ColumnRoleSpec(
            role="z_surf",
            label="Surface elevation (z_surf)",
            cfg_field="z_surf_col",
            canonical="z_surf_m",
            required=False,
            aliases=("z_surf", "elevation", "z"),
        ),
        ColumnRoleSpec(
            role="head",
            label="Hydraulic head",
            cfg_field="head_col",
            canonical="head_m",
            required=False,
            aliases=("head", "h", "hydraulic_head"),
        ),
        ColumnRoleSpec(
            role="easting",
            label="Easting (projected X)",
            cfg_field="easting_col",
            canonical="",
            required=False,
            aliases=("easting", "x_m", "x"),
        ),
        ColumnRoleSpec(
            role="northing",
            label="Northing (projected Y)",
            cfg_field="northing_col",
            canonical="",
            required=False,
            aliases=("northing", "y_m", "y"),
        ),
    ]


class ColumnRoleMapper:
    def __init__(
        self,
        specs: Optional[Sequence[ColumnRoleSpec]] = None,
    ) -> None:
        self._spec_list = list(specs) if specs else _default_specs()
        self._specs = {s.role: s for s in self._spec_list}

        self._columns: List[str] = []
        self._col_to_role: Dict[str, str] = {}
        self._role_to_col: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Core mapping API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._col_to_role.clear()
        self._role_to_col.clear()

    def specs(self) -> List[ColumnRoleSpec]:
        return list(self._spec_list)

    def spec_for(self, role: str) -> Optional[ColumnRoleSpec]:
        return self._specs.get(role)

    def role_for(self, column: str) -> Optional[str]:
        return self._col_to_role.get(column)

    def column_for(self, role: str) -> Optional[str]:
        return self._role_to_col.get(role)

    def reset(self, columns: Iterable[str]) -> None:
        """Clear mapping and auto-detect obvious matches."""
        self.clear()
        cols = [str(c) for c in columns]
        self._columns = cols

        lower_map = {c.lower(): c for c in cols}

        for spec in self._spec_list:
            # prefer canonical match (if any)
            if spec.canonical:
                hit = lower_map.get(spec.canonical.lower())
                if hit is not None:
                    self.assign(hit, spec.role)
                    continue

            # then aliases
            for a in spec.aliases:
                hit = lower_map.get(str(a).lower())
                if hit is not None:
                    self.assign(hit, spec.role)
                    break

    def available_roles_for(self, column: str) -> List[ColumnRoleSpec]:
        """Roles not already taken by another column (plus current role)."""
        cur_role = self.role_for(column)
        out: List[ColumnRoleSpec] = []
        for spec in self._spec_list:
            taken_by = self._role_to_col.get(spec.role)
            if taken_by is None:
                out.append(spec)
                continue
            if cur_role == spec.role:
                out.append(spec)

        out.sort(key=lambda s: (not s.required, s.label))
        return out

    def assign(self, column: str, role: str) -> None:
        """Assign role to column (role uniqueness enforced)."""
        if role not in self._specs:
            return

        # unassign role from previous column
        prev_col = self._role_to_col.get(role)
        if prev_col is not None and prev_col != column:
            self._col_to_role.pop(prev_col, None)

        # unassign any role currently on this column
        prev_role = self._col_to_role.get(column)
        if prev_role is not None and prev_role != role:
            self._role_to_col.pop(prev_role, None)

        self._col_to_role[column] = role
        self._role_to_col[role] = column

    def unassign(self, column: str) -> None:
        role = self._col_to_role.pop(column, None)
        if role is None:
            return
        self._role_to_col.pop(role, None)

    # ------------------------------------------------------------------
    # New API: config patch + required role checks
    # ------------------------------------------------------------------
    def missing_required(self, columns: Iterable[str]) -> List[ColumnRoleSpec]:
        """Required roles that are not mapped to an existing column."""
        colset = {str(c) for c in columns}
        missing: List[ColumnRoleSpec] = []
        for spec in self._spec_list:
            if not spec.required:
                continue
            col = self._role_to_col.get(spec.role)
            if not col or col not in colset:
                missing.append(spec)

        missing.sort(key=lambda s: s.label)
        return missing

    def to_config_patch(
        self,
        *,
        reset_required: bool = True,
    ) -> Dict[str, str]:
        """Build a GeoPriorConfig patch (field names like ``lon_col``)."""
        patch: Dict[str, str] = {}
        for spec in self._spec_list:
            if not spec.cfg_field:
                continue

            col = self._role_to_col.get(spec.role)
            if col is not None:
                patch[spec.cfg_field] = col
                continue

            if reset_required and spec.required and spec.canonical:
                patch[spec.cfg_field] = spec.canonical

        return patch

    def rename_map(self) -> Dict[str, str]:
        """Map {original_name: canonical_name} for assigned roles only."""
        ren: Dict[str, str] = {}
        for role, col in self._role_to_col.items():
            spec = self._specs.get(role)
            if not spec:
                continue
            if spec.canonical and col != spec.canonical:
                ren[col] = spec.canonical
        return ren

    # ------------------------------------------------------------------
    # Legacy API: DataTab compatibility (feature_overrides patch)
    # ------------------------------------------------------------------
    def auto_assign(self) -> None:
        """Legacy API: fill unmapped roles without clearing."""
        alias_map = {
            spec.role: {_norm(a) for a in spec.aliases}
            for spec in self._spec_list
        }

        for col in self._columns:
            n = _norm(col)

            for spec in self._spec_list:
                if spec.role in self._role_to_col:
                    continue

                if spec.canonical and _norm(spec.canonical) == n:
                    self.assign(col, spec.role)
                    break

                if n in alias_map.get(spec.role, set()):
                    self.assign(col, spec.role)
                    break

    def missing_required_roles(self) -> List[ColumnRoleSpec]:
        """Legacy API: required roles missing for current dataset."""
        return self.missing_required(self._columns)

    def to_feature_overrides_patch(self) -> Dict[str, str]:
        """Legacy API: emit feature_overrides keys (TIME_COL, LON_COL, ...)."""
        role_to_key = {
            "time": "TIME_COL",
            "lon": "LON_COL",
            "lat": "LAT_COL",
            "subs": "SUBSIDENCE_COL",
            "subs_cum": "SUBSIDENCE_COL",
            "h_field": "H_FIELD_COL_NAME",
            "gwl": "GWL_COL",
            "z_surf": "Z_SURF_COL",
            "head": "HEAD_COL",
            "easting": "EASTING_COL",
            "northing": "NORTHING_COL",
        }

        patch: Dict[str, str] = {}
        for role, col in self._role_to_col.items():
            key = role_to_key.get(role)
            if key:
                patch[key] = col
        return patch
