# geoprior/ui/tune/center_panel.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Callable, Tuple 

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from ...config.store import GeoConfigStore

from .cards.search_space_card import TuneSearchSpaceCard
from .cards.physics_card import TunePhysicsCard
from .cards.algo_search_card import TuneAlgoSearchCard
from .cards.trial_template_card import TrialTemplateCard
from .cards.compute_card import TuneComputeCard
from .cards.advanced_card import TuneAdvancedCard


MakeCardFn = Callable[[str], Tuple[QWidget, QVBoxLayout]]

__all__ = ["TuneCenterPanel"]


class TuneCenterPanel(QWidget):
    """
    Center panel (Tune tab).

    Hosts the full set of Tune cards:
    1) Search space
    2) Physics switches
    3) Algorithm & objective
    4) Trial template
    5) Compute & parallelism
    6) Advanced

    Notes
    -----
    - Each card expands inline (inside the same card).
    - This panel only assembles + wires card signals.
    - The store remains the single source of truth.
    """

    # Keep a simple "section opened" signal so tab.py can
    # scroll / sync chips if desired.
    edit_requested = pyqtSignal(str)

    # Convenience shortcuts (tab.py can hook these)
    arch_hp_clicked = pyqtSignal()
    phys_hp_clicked = pyqtSignal()

    reset_space_requested = pyqtSignal()

    algo_changed = pyqtSignal()
    trial_changed = pyqtSignal()
    compute_changed = pyqtSignal()
    advanced_clicked = pyqtSignal()
    

    def __init__(
        self,
        *,
        store: GeoConfigStore,
        make_card: Optional[MakeCardFn] =None,  # kept for compat; unused now
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._store = store
        self._filter_q = ""
        self._filter_on = False
        self._make_card = make_card 

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )

        self._build_ui()
        self._wire()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        # Card 1: Search space
        self.card_space = TuneSearchSpaceCard(
            store=self._store,
            make_card=self._make_card,
            parent=self,
        )
        root.addWidget(self.card_space)
        
        # Card 2: Physics switches
        self.card_phys = TunePhysicsCard(
            store=self._store,
            make_card=self._make_card,
            parent=self,
        )
        root.addWidget(self.card_phys)

        # Card 3: Algorithm & objective
        self.card_algo = TuneAlgoSearchCard(
            store=self._store,
            make_card=self._make_card,
            parent=self,
        )
        root.addWidget(self.card_algo)

        # Card 4: Trial template
        self.card_trial = TrialTemplateCard(
            store=self._store,
            make_card=self._make_card,
            parent=self,
        )
        root.addWidget(self.card_trial)

        # Card 5: Compute & parallelism
        self.card_compute = TuneComputeCard(
            store=self._store,
            make_card=self._make_card,
            parent=self,
        )
        root.addWidget(self.card_compute)

        # Card 6: Advanced
        self.card_adv = TuneAdvancedCard(
            store=self._store,
            make_card = self._make_card, 
            parent=self,
        )
        root.addWidget(self.card_adv)

        root.addStretch(1)

    # -------------------------
    # Wiring
    # -------------------------
    def _wire(self) -> None:
        # Search space
        self.card_space.edit_toggled.connect(
            lambda on: self._on_edit("space", on)
        )
        self.card_space.reset_requested.connect(
            self.reset_space_requested.emit
        )
        self.card_space.changed.connect(
            lambda: self._emit_changed("space")
        )

        # Physics
        if hasattr(self.card_phys, "edit_toggled"):
            self.card_phys.edit_toggled.connect(
                lambda on: self._on_edit("physics", on)
            )
        if hasattr(self.card_phys, "changed"):
            self.card_phys.changed.connect(
                lambda: self._emit_changed("physics")
            )

        # Algo
        if hasattr(self.card_algo, "edit_toggled"):
            self.card_algo.edit_toggled.connect(
                lambda on: self._on_edit("algo", on)
            )
        if hasattr(self.card_algo, "changed"):
            self.card_algo.changed.connect(self.algo_changed.emit)

        # Trial
        self.card_trial.changed.connect(self.trial_changed.emit)

        # Compute
        if hasattr(self.card_compute, "edit_toggled"):
            self.card_compute.edit_toggled.connect(
                lambda on: self._on_edit("compute", on)
            )
        if hasattr(self.card_compute, "changed"):
            self.card_compute.changed.connect(
                self.compute_changed.emit
            )

        # Advanced
        if hasattr(self.card_adv, "edit_toggled"):
            self.card_adv.edit_toggled.connect(
                lambda on: self._on_edit("adv", on)
            )
        if hasattr(self.card_adv, "advanced_clicked"):
            self.card_adv.advanced_clicked.connect(
                self.advanced_clicked.emit
            )

    def _on_edit(self, key: str, on: bool) -> None:
        if on:
            self.edit_requested.emit(str(key))

    def _emit_changed(self, key: str) -> None:
        # keep a simple "changed" fanout for now
        if key == "algo":
            self.algo_changed.emit()
        elif key == "compute":
            self.compute_changed.emit()
        elif key == "trial":
            self.trial_changed.emit()

    # -------------------------
    # Public API used by tab.py
    # -------------------------
    def refresh_from_store(self) -> None:
        """
        Refresh all cards.

        Cards are store-driven; this is safe to call often.
        """
        for w in (
            self.card_space,
            self.card_phys,
            self.card_algo,
            self.card_trial,
            self.card_compute,
            self.card_adv,
        ):
            fn = getattr(w, "refresh_from_store", None)
            if callable(fn):
                fn()

        self._apply_filter_now()

    def card_for(self, key: str) -> Optional[QWidget]:
        k = str(key or "").strip().lower()
        if k in {"space", "search", "search_space"}:
            return self.card_space
        if k in {"physics", "phys"}:
            return self.card_phys
        if k in {"algo", "algorithm"}:
            return self.card_algo
        if k in {"trial", "template"}:
            return self.card_trial
        if k in {"compute", "parallel"}:
            return self.card_compute
        if k in {"adv", "advanced"}:
            return self.card_adv
        return None

    def apply_filter(self, text: str, on: bool) -> None:
        self._filter_q = str(text or "").strip().lower()
        self._filter_on = bool(on)
        self._apply_filter_now()

    # -------------------------
    # Filter (simple + robust)
    # -------------------------
    def _apply_filter_now(self) -> None:
        if not self._filter_on:
            self._set_card_visible(self.card_space, True)
            self._set_card_visible(self.card_phys, True)
            self._set_card_visible(self.card_algo, True)
            self._set_card_visible(self.card_trial, True)
            self._set_card_visible(self.card_compute, True)
            self._set_card_visible(self.card_adv, True)
            return

        q = self._filter_q
        if not q:
            self._set_card_visible(self.card_space, True)
            self._set_card_visible(self.card_phys, True)
            self._set_card_visible(self.card_algo, True)
            self._set_card_visible(self.card_trial, True)
            self._set_card_visible(self.card_compute, True)
            self._set_card_visible(self.card_adv, True)
            return

        def match(card: QWidget, *extra: str) -> bool:
            # Match on title/summary labels if present
            hay = " ".join(extra).strip().lower()
            if not hay:
                # best-effort: read common label objects
                for nm in ("lbl_title", "lbl_sum", "lbl_help"):
                    lab = getattr(card, nm, None)
                    txt = getattr(lab, "text", None)
                    if callable(txt):
                        hay += " " + str(txt() or "")
                hay = hay.strip().lower()
            return q in hay

        self._set_card_visible(
            self.card_space,
            match(
                self.card_space,
                "search space architecture loss weights",
            ),
        )
        self._set_card_visible(
            self.card_phys,
            match(self.card_phys, "physics switches pde"),
        )
        self._set_card_visible(
            self.card_algo,
            match(self.card_algo, "algorithm objective"),
        )
        self._set_card_visible(
            self.card_trial,
            match(self.card_trial, "trial template"),
        )
        self._set_card_visible(
            self.card_compute,
            match(
                self.card_compute,
                "compute parallelism device cpu gpu",
            ),
        )
        self._set_card_visible(
            self.card_adv,
            match(self.card_adv, "advanced dialogs"),
        )

    @staticmethod
    def _set_card_visible(w: QWidget, on: bool) -> None:
        if w.isVisible() != bool(on):
            w.setVisible(bool(on))
