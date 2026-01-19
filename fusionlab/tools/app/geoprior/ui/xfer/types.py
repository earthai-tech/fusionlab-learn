# geoprior/ui/xfer/types.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
geoprior.ui.xfer.types

Small shared dataclasses / Protocols for Xfer UI.
Keep this file dependency-light (no pandas).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol


@dataclass(frozen=True)
class CityJobRef:
    city: str
    model: str
    stage: str
    job_kind: str
    job_id: str


@dataclass(frozen=True)
class XferFileRef:
    """
    One forecast-like CSV we can load for mapping.
    """
    owner_city: str
    peer_city: str
    job: CityJobRef
    split: str
    path: Path


@dataclass(frozen=True)
class MapPoint:
    """
    Leaflet-friendly point.
    """
    lat: float
    lon: float
    v: float
    sid: Optional[int] = None


@dataclass(frozen=True)
class LayerOpts:
    """
    Rendering options (kept minimal for now).
    """
    # stroke: str = "A"
    stroke: str = "#2E3191"
    radius: int = 6
    opacity: float = 0.9
    name: str = ""


@dataclass
class CityPoint:
    name: str
    lat: float
    lon: float


class MapApi(Protocol):
    """
    Contract required by XferMapController.
    Implemented by ui/xfer/map/view.py MapView.
    """

    def clear_layers(self) -> None: ...
    def clear_layer(self, layer_id: str) -> None: ...

    def set_layer(
        self,
        layer_id: str,
        name: str,
        points: Iterable[MapPoint],
        opts: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def fit_layers(
        self,
        layer_ids: Optional[Iterable[str]] = None,
    ) -> None: ...

    def set_legend(self, opts: Dict[str, Any]) -> None: ...

    def set_centroids(
        self,
        src_name: str,
        src_lat: float,
        src_lon: float,
        tgt_name: str,
        tgt_lat: float,
        tgt_lon: float,
    ) -> None: ...

    def clear(self) -> None: ...
