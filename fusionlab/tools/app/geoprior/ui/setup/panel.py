# geoprior/ui/setup/panel.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...config.geoprior_config import GeoPriorConfig
from ...config.store import GeoConfigStore

from .bindings import Binder
from .header import SetupHeader
from .nav import SetupNav
from .schema import SectionSpec, default_sections
from .lock import SetupLockController

from .cards.summary import SummaryCard
from .cards.paths import ProjectPathsCard
from .cards.time import TimeWindowCard
from .cards.data_semantics import DataSemanticsCard
from .cards.coords import CoordsCard
from .cards.feature_registry import FeatureRegistryCard
from .cards.censoring import CensoringCard
from .cards.scaling import ScalingCard
from .cards.model_architecture import ModelArchitectureCard
from .cards.training_basics import TrainingBasicsCard
from .cards.phys_constraints import PhysicsConstraintsCard
from .cards.prob import ProbabilisticOutputsCard
from .cards.tune import TuningCard
from .cards.device import DeviceRuntimeCard
from .cards.ui_prefs import UiPreferencesCard


class ConfigCenterPanel(QWidget):
    """
    Experiment Setup panel (Config Center).

    Layout
    ------
    - SetupHeader (sticky)
    - Left SetupNav
    - Right scroll: cards in section order
    """

    def __init__(
        self,
        store: GeoConfigStore,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.binder = Binder(store)

        self._sections: List[SectionSpec] = (
            list(default_sections())
        )
        self._sec_meta: Dict[str, SectionSpec] = {
            s.sec_id: s for s in self._sections
        }
        self._cards: Dict[str, QWidget] = {}
        self._dataset_cols: List[str] = []

        self._snap_path: Optional[Path] = None

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(
            self._refresh_cards
        )

        self._build_ui()
        
        self.lock_ctl = SetupLockController(
            self._scroll_body,
            keep_copy=True,
        )
        self._wire_store()
        
        self._pull_lock_state()

        self._apply_filter("")
        self._queue_refresh()
        
    def _pull_lock_state(self) -> None:
        locked = bool(
            self.store.get("setup.locked", False)
        )
        self.header.set_locked(locked)

        if hasattr(self, "lock_ctl"):
            self.lock_ctl.set_locked(locked)


    # -----------------------------------------------------------------
    # UI build
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.header = self._build_header()
        root.addWidget(self.header, 0)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        self.nav = self._build_nav()
        self.scroll = self._build_scroll()

        splitter.addWidget(self.nav)
        splitter.addWidget(self.scroll)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 9999])

        root.addWidget(splitter, 1)

    def _build_header(self) -> SetupHeader:
        h = SetupHeader(self)

        cfg = self.store.cfg
        h.set_context(
            city=str(cfg.city),
            model=str(cfg.model_name),
            stage="",
        )
        h.set_dirty_count(
            int(self.store.overrides_count())
        )
        h.set_locked(
            bool(self.store.get("setup.locked", False))
        )

        h.request_load.connect(self._on_load)
        h.request_save.connect(self._on_save)
        h.request_save_as.connect(self._on_save_as)

        h.request_reset.connect(self._on_reset)
        h.request_apply.connect(self._on_apply)
        h.request_diff.connect(self._on_show_diff)

        h.request_export_json.connect(self._on_save_as)
        h.request_import_json.connect(self._on_load)

        h.request_show_snapshot.connect(
            self._on_show_snapshot
        )
        h.request_show_overrides.connect(
            self._on_show_diff
        )

        h.request_copy_snapshot.connect(
            self._on_copy_snapshot
        )
        h.request_copy_overrides.connect(
            self._on_copy_overrides
        )

        h.search_changed.connect(self._apply_filter)
        h.lock_changed.connect(self._on_lock_changed)

        return h

    def _build_nav(self) -> SetupNav:
        nav = SetupNav(
            self,
            with_search=False,
            show_descriptions=True,
        )
        nav.set_sections(self._sections)

        nav.section_changed.connect(self._on_nav)
        nav.filter_changed.connect(self._apply_filter)

        return nav

    def _build_scroll(self) -> QScrollArea:
        scr = QScrollArea(self)
        scr.setWidgetResizable(True)

        body = QWidget(scr)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self._scroll_body = body
        self._scroll_lay = lay

        scr.setWidget(body)

        self._build_cards()
        lay.addStretch(1)

        return scr

    def _build_cards(self) -> None:
        for s in self._sections:
            sec_id = str(s.sec_id)
            w = self._make_card(sec_id)
            if w is None:
                continue
            self._register_card(sec_id, w)

    def _register_card(self, sec_id: str, w: QWidget) -> None:
        self._cards[str(sec_id)] = w
        self._scroll_lay.addWidget(w)

        if self._dataset_cols:
            self._push_cols_to_card(w)

    def _make_card(self, sec_id: str) -> Optional[QWidget]:
        mk = self._card_factory(str(sec_id))
        if mk is None:
            return None

        try:
            return mk()
        except Exception as exc:
            self.store.error_raised.emit(str(exc))
            return self._make_error_card(
                sec_id=str(sec_id),
                err=str(exc),
            )

    def _card_factory(self, sec_id: str):
        if sec_id == "summary":
            return lambda: SummaryCard(
                store=self.store,
                parent=self._scroll_body,
            )

        if sec_id == "paths":
            return lambda: ProjectPathsCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "time":
            return lambda: TimeWindowCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "data_semantics":
            return lambda: DataSemanticsCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "coords":
            return lambda: CoordsCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "features":
            return lambda: FeatureRegistryCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "censoring":
            return lambda: CensoringCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "scaling":
            return lambda: ScalingCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "arch":
            return lambda: ModelArchitectureCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "train":
            return lambda: TrainingBasicsCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "physics":
            return lambda: PhysicsConstraintsCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "prob":
            return lambda: ProbabilisticOutputsCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "tuning":
            return lambda: TuningCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "device":
            return lambda: DeviceRuntimeCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        if sec_id == "ui":
            return lambda: UiPreferencesCard(
                store=self.store,
                binder=self.binder,
                parent=self._scroll_body,
            )

        return None

    def _make_error_card(
        self,
        *,
        sec_id: str,
        err: str,
    ) -> QWidget:
        from PyQt5.QtWidgets import (
            QGroupBox,
            QLabel,
            QVBoxLayout,
        )

        meta = self._sec_meta.get(sec_id)
        title = sec_id if meta is None else meta.title

        box = QGroupBox(str(title), self._scroll_body)
        box.setObjectName(f"cfgCard_{sec_id}")

        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        lab = QLabel(box)
        lab.setWordWrap(True)
        lab.setText(
            "Card failed to load.\n\n"
            f"{err}"
        )
        lay.addWidget(lab, 0)

        return box

    # -----------------------------------------------------------------
    # Store wiring / refresh
    # -----------------------------------------------------------------
    def _wire_store(self) -> None:
        self.store.dirty_changed.connect(
            self.header.set_dirty_count
        )
        self.store.config_changed.connect(
            self._on_config_changed
        )
        self.store.config_replaced.connect(
            self._on_config_replaced
        )

        self.header.set_dirty_count(
            int(self.store.overrides_count())
        )
        self._on_config_replaced(self.store.cfg)

    def _on_config_replaced(self, cfg: GeoPriorConfig) -> None:
        self.header.set_context(
            city=str(cfg.city),
            model=str(cfg.model_name),
            stage="",
        )
        #Yes, it calls set_locked twice, but that’s fine.
        # self.header.set_locked(
        #     bool(self.store.get("setup.locked", False))
        # )
        self._pull_lock_state()
        self._queue_refresh()

    def _on_config_changed(self, _keys: object) -> None:
        self._queue_refresh()

    def _queue_refresh(self) -> None:
        if self._refresh_timer.isActive():
            return
        self._refresh_timer.start(0)

    def _refresh_cards(self) -> None:
        for w in self._cards.values():
            fn = getattr(w, "refresh", None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:
                    self.store.error_raised.emit(str(exc))

    # -----------------------------------------------------------------
    # Filtering / navigation
    # -----------------------------------------------------------------
    def _apply_filter(self, text: str) -> None:
        q = (text or "").strip().lower()

        self.nav.apply_filter(text)

        for sec_id, w in self._cards.items():
            meta = self._sec_meta.get(sec_id)
            hay = ""
            if meta is not None:
                hay = (
                    f"{meta.title} {meta.description}"
                ).lower()

            hit = (not q) or (q in hay)
            w.setVisible(bool(hit))

        cur = self.nav.current_section_id()
        if cur is None:
            self.nav.select_first_visible()
            cur = self.nav.current_section_id()

        if cur is not None:
            self._scroll_to(cur)

    def _on_nav(self, sec_id: str) -> None:
        self._scroll_to(str(sec_id))

    def _scroll_to(self, sec_id: str) -> None:
        w = self._cards.get(str(sec_id))
        if w is None:
            return
        if not w.isVisible():
            return
        self.scroll.ensureWidgetVisible(w, 0, 24)

    # -----------------------------------------------------------------
    # Public: dataset columns
    # -----------------------------------------------------------------
    def set_dataset_columns(self, cols: List[str]) -> None:
        self._dataset_cols = [str(c) for c in (cols or [])]

        for w in self._cards.values():
            self._push_cols_to_card(w)

    def _push_cols_to_card(self, w: QWidget) -> None:
        fn = getattr(w, "set_dataset_columns", None)
        if not callable(fn):
            return
        try:
            fn(list(self._dataset_cols))
        except Exception as exc:
            self.store.error_raised.emit(str(exc))

    # -----------------------------------------------------------------
    # Header actions
    # -----------------------------------------------------------------
    def _on_lock_changed(self, locked: bool) -> None:
        on = bool(locked)
        self.store.set("setup.locked", on)
        self.lock_ctl.set_locked(on)
        self.header.set_locked(on)

    def _on_apply(self) -> None:
        keys = set(self.store.cfg.__dataclass_fields__.keys())
        self.store.config_changed.emit(keys)

    def _on_show_snapshot(self) -> None:
        self._show_json_dialog(
            title="Config snapshot",
            payload=self.store.cfg.as_dict(),
        )

    def _on_show_diff(self) -> None:
        self._show_json_dialog(
            title="Config overrides (diff)",
            payload=self.store.snapshot_overrides(),
        )

    def _on_copy_snapshot(self) -> None:
        self._copy_json(self.store.cfg.as_dict())

    def _on_copy_overrides(self) -> None:
        self._copy_json(self.store.snapshot_overrides())

    def _on_save(self) -> None:
        if self._snap_path is None:
            self._on_save_as()
            return
        self._save_to_path(self._snap_path)

    def _on_save_as(self) -> None:
        path, _flt = QFileDialog.getSaveFileName(
            self,
            "Save config snapshot",
            "",
            "JSON (*.json)",
        )
        if not path:
            return
        p = Path(path).expanduser()
        self._snap_path = p
        self._save_to_path(p)

    def _save_to_path(self, path: Path) -> None:
        payload = self.store.cfg.as_dict()
        try:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            self.store.error_raised.emit(str(exc))

    def _on_load(self) -> None:
        path, _flt = QFileDialog.getOpenFileName(
            self,
            "Load config snapshot",
            "",
            "JSON (*.json)",
        )
        if not path:
            return
        p = Path(path).expanduser()
        self._snap_path = p
        self._load_from_path(p)

    def _load_from_path(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            payload = self._extract_cfg_payload(raw)
            if not isinstance(payload, dict):
                raise ValueError("Snapshot must be a dict.")

            with self.store.batch():
                self.store.patch(payload)

        except Exception as exc:
            self.store.error_raised.emit(str(exc))

    def _extract_cfg_payload(self, raw: Any) -> Any:
        if isinstance(raw, dict):
            cfg = raw.get("config", None)
            if isinstance(cfg, dict):
                return cfg
            cfg2 = raw.get("cfg", None)
            if isinstance(cfg2, dict):
                return cfg2
        return raw

    def _on_reset(self) -> None:
        try:
            cfg = GeoPriorConfig.from_defaults()
            self.store.replace_config(cfg)
        except Exception as exc:
            self.store.error_raised.emit(str(exc))

    # -----------------------------------------------------------------
    # Dialog + clipboard helpers
    # -----------------------------------------------------------------
    def _copy_json(self, payload: Dict[str, Any]) -> None:
        try:
            from PyQt5.QtWidgets import QApplication

            txt = json.dumps(payload, indent=2)
            QApplication.clipboard().setText(txt)
        except Exception as exc:
            self.store.error_raised.emit(str(exc))

    def _show_json_dialog(
        self,
        *,
        title: str,
        payload: Dict[str, Any],
    ) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(str(title))
        dlg.resize(820, 520)

        lay = QVBoxLayout(dlg)

        txt = QTextEdit(dlg)
        txt.setReadOnly(True)

        try:
            pretty = json.dumps(payload, indent=2)
        except Exception:
            pretty = str(payload)

        txt.setPlainText(pretty)
        lay.addWidget(txt, 1)

        row = QHBoxLayout()
        row.addStretch(1)

        btn = QPushButton("Close", dlg)
        btn.clicked.connect(dlg.accept)
        row.addWidget(btn)

        lay.addLayout(row)
        dlg.exec_()
