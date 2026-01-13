# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .prior_schema import (
    PHYSICS_GROUPS,
    PHYSICS_GROUP_TITLES,
    FieldKey,
)
from . import helps as _helps


HelpKey = Tuple[str, Optional[str]]


def _help_map() -> Dict[HelpKey, str]:
    m = getattr(_helps, "_PHYS_HELP", None)
    if isinstance(m, dict):
        return m
    return {}


def _fmt_key(name: str, sub: Optional[str]) -> str:
    if sub is None:
        return f'("{name}", None)'
    return f'("{name}", "{sub}")'


def _missing_entries() -> List[Tuple[str, FieldKey, bool]]:
    hm = _help_map()
    out: List[Tuple[str, FieldKey, bool]] = []

    for gid in PHYSICS_GROUP_TITLES:
        keys = PHYSICS_GROUPS.get(gid, [])
        for k in keys:
            name = k.name
            sub = k.subkey

            generic = (name, None) in hm
            specific = (name, sub) in hm

            if sub is None:
                if generic:
                    continue
                out.append((gid, k, False))
                continue

            if specific:
                continue
            out.append((gid, k, generic))

    return out


def emit_missing_skeleton() -> str:
    missing = _missing_entries()

    lines: List[str] = []
    lines.append("# Auto-generated help skeleton (missing).")
    lines.append("# Fill in values, then paste into helps.py.")
    lines.append("")
    lines.append("_PHYS_HELP.update({")

    cur_gid = None

    for gid, k, has_generic in missing:
        if gid != cur_gid:
            title = PHYSICS_GROUP_TITLES.get(gid, gid)
            lines.append("")
            lines.append(f"    # {title}")
            cur_gid = gid

        name = k.name
        sub = k.subkey

        note = ""
        if (sub is not None) and has_generic:
            note = "  # inherits generic"

        kk = _fmt_key(name, sub)
        lines.append(f"    {kk}: \"\",{note}")

    lines.append("})")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print(emit_missing_skeleton())


if __name__ == "__main__":
    main()
