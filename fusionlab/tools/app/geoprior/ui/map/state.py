# geoprior/ui/map/state.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.map.state

Derived MapState from the store.

This is a small dataclass that lets the map layer
code avoid querying the store everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.store import GeoConfigStore


@dataclass(frozen=True)
class MapState:
    engine: str
    coord_mode: str
    x_col: str
    y_col: str
    z_col: str
    focus_mode: bool
    show_analytics: bool

    @classmethod
    def from_store(cls, store: GeoConfigStore) -> "MapState":
        return cls(
            engine=str(store.get("map.engine", "leaflet")),
            coord_mode=str(
                store.get("map.coord_mode", "lonlat"),
            ),
            x_col=str(store.get("map.x_col", "")),
            y_col=str(store.get("map.y_col", "")),
            z_col=str(store.get("map.z_col", "")),
            focus_mode=bool(
                store.get("map.focus_mode", False),
            ),
            show_analytics=bool(
                store.get("map.show_analytics", False),
            ),
        )
