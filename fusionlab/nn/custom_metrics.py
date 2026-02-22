# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author : LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from collections import OrderedDict

from . import KERAS_DEPS, dependency_message

from .keras_metrics import (
    Coverage80,
    MAEQ50,
    MSEQ50,
    Sharpness80,
)

Metric = KERAS_DEPS.Metric

DEP_MSG = dependency_message("nn.custom_metrics")


class _BaseTrackPack:
    """Small helper to group metrics + update them."""

    def __init__(self) -> None:
        self._metrics = OrderedDict()

    @property
    def metrics(self) -> list[Metric]:
        # Keras expects a list of Metric objects.
        return list(self._metrics.values())

    @property
    def as_dict(self) -> dict[str, Metric]:
        # Helpful if you still pass manual_trackers.
        return dict(self._metrics)

    def reset_state(self) -> None:
        # Useful if you want explicit reset in tests.
        for m in self._metrics.values():
            m.reset_state()


class Subs80(_BaseTrackPack):
    """Subsidence add-on metrics (q50 + interval)."""

    def __init__(
        self,
        *,
        key: str = "subs_pred",
        quantiles: bool = True,
        q_axis: int = 2,
        n_q: int = 3,
    ) -> None:
        super().__init__()

        self.key = str(key)
        self.quantiles = bool(quantiles)

        # Developer note:
        # - names are "subs_*" to avoid collisions
        #   with compiled metrics like "subs_pred_mae_q50".
        self._metrics["subs_mae"] = MAEQ50(
            name="subs_mae",
            q_axis=q_axis,
            n_q=n_q,
        )
        self._metrics["subs_mse"] = MSEQ50(
            name="subs_mse",
            q_axis=q_axis,
            n_q=n_q,
        )

        if self.quantiles:
            self._metrics["subs_cov80"] = Coverage80(
                name="subs_cov80",
                q_axis=q_axis,
                n_q=n_q,
            )
            self._metrics["subs_sharp80"] = Sharpness80(
                name="subs_sharp80",
                q_axis=q_axis,
                n_q=n_q,
            )

    def update_from_outputs(
        self,
        targets: dict,
        preds: dict,
        *,
        sample_weight=None,
    ) -> None:
        # Developer note:
        # - each Metric class is shape-safe (via _as_BHO).
        yt = targets.get(self.key, None)
        yp = preds.get(self.key, None)

        if yt is None or yp is None:
            return

        for m in self._metrics.values():
            m.update_state(
                yt,
                yp,
                sample_weight=sample_weight,
            )


class GWL80(_BaseTrackPack):
    """Groundwater add-on metrics (q50 only)."""

    def __init__(
        self,
        *,
        key: str = "gwl_pred",
        q_axis: int = 2,
        n_q: int = 3,
    ) -> None:
        super().__init__()

        self.key = str(key)

        self._metrics["gwl_mae"] = MAEQ50(
            name="gwl_mae",
            q_axis=q_axis,
            n_q=n_q,
        )
        self._metrics["gwl_mse"] = MSEQ50(
            name="gwl_mse",
            q_axis=q_axis,
            n_q=n_q,
        )

    def update_from_outputs(
        self,
        targets: dict,
        preds: dict,
        *,
        sample_weight=None,
    ) -> None:
        yt = targets.get(self.key, None)
        yp = preds.get(self.key, None)

        if yt is None or yp is None:
            return

        for m in self._metrics.values():
            m.update_state(
                yt,
                yp,
                sample_weight=sample_weight,
            )


class GeoPriorTrackers:
    """Top-level container for add-on trackers."""

    def __init__(
        self,
        *,
        quantiles: bool,
        subs_key: str = "subs_pred",
        gwl_key: str = "gwl_pred",
        q_axis: int = 2,
        n_q: int = 3,
    ) -> None:
        self.subs = Subs80(
            key=subs_key,
            quantiles=quantiles,
            q_axis=q_axis,
            n_q=n_q,
        )
        self.gwl = GWL80(
            key=gwl_key,
            q_axis=q_axis,
            n_q=n_q,
        )

    @property
    def metrics(self) -> list[Metric]:
        return self.subs.metrics + self.gwl.metrics

    @property
    def as_dict(self) -> dict[str, Metric]:
        d = {}
        d.update(self.subs.as_dict)
        d.update(self.gwl.as_dict)
        return d

    def reset_state(self) -> None:
        self.subs.reset_state()
        self.gwl.reset_state()

    def update_state(
        self,
        targets: dict,
        preds: dict,
        *,
        sample_weight=None,
    ) -> None:
        self.subs.update_from_outputs(
            targets,
            preds,
            sample_weight=sample_weight,
        )
        self.gwl.update_from_outputs(
            targets,
            preds,
            sample_weight=sample_weight,
        )
