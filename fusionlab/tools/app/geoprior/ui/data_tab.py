# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union 

import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTableView,
    QStackedWidget,
    QFrame,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QMenu, 
    QCheckBox, 
    QToolButton,
    QStyle,
)
from ..dialogs.csv_dialog import _PandasModel
from ..services.column_mapping import ColumnRoleMapper
from ..services.dataset_summary import build_dataset_summary_text
from ..services.selection_viz import (
    SelectionInsightsPane,
    SelectionVizController
)
from .icon_utils import try_icon

class RolePandasModel(_PandasModel):
    def __init__(self, df, mapper: ColumnRoleMapper, parent=None):
        super().__init__(df, parent=parent)
        self._mapper = mapper

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        txt = super().headerData(section, orientation, role)
        if role != Qt.DisplayRole:
            return txt
        if orientation != Qt.Horizontal:
            return txt

        col = str(txt)
        r = self._mapper.role_for(col)
        if not r:
            return f"{col} \u25BE"
        spec = self._mapper.spec_for(r)
        if not spec:
            return f"{col} \u25BE"
        return f"{col} \u25BE [{spec.label}]"



class DataTab(QWidget):
    """
    Data tab with a left dataset library + right preview/editor actions.
    """

    # Existing requests (handled by MainWindow)
    # Compat + new name (emit both)
    request_open = pyqtSignal()
    request_open_new = pyqtSignal()

    request_edit = pyqtSignal()
    request_save = pyqtSignal()
    request_save_as = pyqtSignal()
    request_reload = pyqtSignal()

    request_load_saved = pyqtSignal(str)
    request_duplicate_saved = pyqtSignal(str)
    
    dataset_changed = pyqtSignal(object)
    # Emits a feature_overrides patch (dict) for GeoPriorConfig:
    # {"TIME_COL": "...", "LON_COL": "...", ...}
    column_overrides_changed = pyqtSignal(dict)
    
    request_browse_results_root = pyqtSignal()
    request_open_results_root = pyqtSignal()



    def __init__(self, parent=None):
        super().__init__(parent)

        self._datasets_root: Optional[Path] = None

        self._csv_path: Optional[Path] = None
        self._df: Optional[pd.DataFrame] = None
        self._city: str = ""
        self._dirty: bool = False
        
        self._preview_df: Optional[pd.DataFrame] = None
        self._colmap = ColumnRoleMapper()
        
        self.cb_auto_insights: Optional[QCheckBox] = None
        self._viz_split: Optional[QSplitter] = None
        self._viz_pane: Optional[SelectionInsightsPane] = None
        self._sel_viz: Optional[SelectionVizController] = None
        self._results_root: Optional[Path] = None

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        split = QSplitter(Qt.Horizontal, self)
        root.addWidget(split, 1)

        # ------------------------------------------------------------
        # Left: Dataset library
        # ------------------------------------------------------------
   
        split.addWidget(self._build_left_panel())

        # ------------------------------------------------------------
        # Right: stack (empty vs loaded)
        # ------------------------------------------------------------
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        self._stack = QStackedWidget()
        right_lay.addWidget(self._stack, 1)

        self._build_empty_page()
        self._build_loaded_page()
        self._show_empty()

        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        split.setSizes([280, 720])
        
    def _build_left_panel(self) -> QWidget:
        left = QFrame()
        left.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(left)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._left_split = QSplitter(Qt.Vertical, left)
        self._left_split.setChildrenCollapsible(False)

        self._lib_widget = self._build_library_widget()
        self._details_widget = self._build_details_widget()

        self._left_split.addWidget(self._lib_widget)
        self._left_split.addWidget(self._details_widget)

        self._left_split.setStretchFactor(0, 1)
        self._left_split.setStretchFactor(1, 0)

        lay.addWidget(self._left_split, 1)

        self._set_details_visible(False)
        return left
    
    def _build_library_widget(self) -> QWidget:
        def _icon_or_std(
            svg: str,
            std: QStyle.StandardPixmap,
        ):
            ico = try_icon(svg)
            if ico is None:
                ico = self.style().standardIcon(std)
            return ico
    
        w = QFrame()
        w.setFrameShape(QFrame.NoFrame)
    
        left_lay = QVBoxLayout(w)
        left_lay.setContentsMargins(10, 10, 10, 10)
        left_lay.setSpacing(8)
    
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Datasets"), 0)
    
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_refresh.clicked.connect(self.refresh_library)
        title_row.addWidget(self.btn_refresh, 0)
    
        title_row.addStretch(1)
        left_lay.addLayout(title_row)
    
        self.edt_lib_search = QLineEdit()
        self.edt_lib_search.setPlaceholderText("Search…")
        self.edt_lib_search.textChanged.connect(self.refresh_library)
    
        # -------------------------------------------------
        # Leading search icon (SVG first, fallback to Qt std)
        # -------------------------------------------------
        ico = _icon_or_std(
            "search.svg",
            QStyle.SP_FileDialogContentsView,
        )
        act = self.edt_lib_search.addAction(
            ico,
            QLineEdit.LeadingPosition,
        )
        act.setToolTip("Search datasets")
        act.triggered.connect(self.edt_lib_search.setFocus)
    
        left_lay.addWidget(self.edt_lib_search)
    
        self.list_datasets = QListWidget()
        self.list_datasets.itemDoubleClicked.connect(
            self._on_library_item_activated
        )
        left_lay.addWidget(self.list_datasets, 1)
    
        lib_btns = QHBoxLayout()
    
        self.btn_use_selected = QPushButton("Load selected")
        self.btn_use_selected.setCursor(Qt.PointingHandCursor)
        self.btn_use_selected.clicked.connect(
            self._load_selected_clicked
        )
        lib_btns.addWidget(self.btn_use_selected)
    
        self.btn_duplicate = QPushButton("Duplicate")
        self.btn_duplicate.setCursor(Qt.PointingHandCursor)
        self.btn_duplicate.clicked.connect(
            self._duplicate_selected_clicked
        )
        lib_btns.addWidget(self.btn_duplicate)
    
        left_lay.addLayout(lib_btns)
    
        hint = QLabel("Tip: double-click to load")
        hint.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hint.setStyleSheet("font-size:11px; opacity:0.8;")
        left_lay.addWidget(hint)
    
        return w

    def _build_details_widget(self) -> QWidget:
        w = QFrame()
        w.setFrameShape(QFrame.StyledPanel)
    
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)
    
        title = QLabel("Dataset details")
        title.setStyleSheet("font-weight:600;")
        lay.addWidget(title)
    
        self.lbl_d_summary = QLabel("")
        self.lbl_d_summary.setWordWrap(True)
        lay.addWidget(self.lbl_d_summary)
    
        self.lbl_d_required = QLabel("")
        self.lbl_d_required.setWordWrap(True)
        lay.addWidget(self.lbl_d_required)
    
        self.lbl_d_mapping = QLabel("")
        self.lbl_d_mapping.setWordWrap(True)
        lay.addWidget(self.lbl_d_mapping, 1)
    
        return w


    def _set_details_visible(self, visible: bool) -> None:
        self._details_widget.setVisible(bool(visible))
        if not visible:
            self._left_split.setSizes([9999, 0])
            return
        self._left_split.setSizes([750, 260])

    def _refresh_details_panel(self) -> None:
        if self._df is None:
            self._set_details_visible(False)
            return
    
        self._set_details_visible(True)
    
        try:
            summary = build_dataset_summary_text(
                self._df,
                mapper=self._colmap,
                city=self._city,
                csv_path=self._csv_path,
            )
        except TypeError:
            summary = build_dataset_summary_text(self._df)
    
        self.lbl_d_summary.setText(summary)
    
        missing = self._colmap.missing_required_roles()
        if missing:
            names = ", ".join(s.label for s in missing)
            self.lbl_d_required.setText(
                f"Missing required roles: {names}"
            )
        else:
            self.lbl_d_required.setText("All required roles mapped.")
    
        patch = self._colmap.to_feature_overrides_patch()
        if not patch:
            self.lbl_d_mapping.setText("Mapping: (none)")
            return
    
        lines = ["Mapping:"]
        for k, v in patch.items():
            lines.append(f"- {k}: {v}")
        self.lbl_d_mapping.setText("\n".join(lines))

 
    # -----------------------------------------------------------------
    # Public API (MainWindow calls these)
    # -----------------------------------------------------------------
    def set_datasets_root(self, root: str | Path) -> None:
        self._datasets_root = Path(root)
        self.refresh_library()
        
    def set_dataset(
        self,
        csv_path: Optional[str | Path],
        df: Optional[pd.DataFrame],
        *,
        city: str = "",
        dirty: bool = False,
    ) -> None:
        self._csv_path = Path(csv_path) if csv_path else None
        self._df = df
        self._city = city or ""
        self._dirty = bool(dirty)
    
        if self._df is None:
            self._show_empty()
            self.dataset_changed.emit([])
            self._colmap.reset([])
            self._preview_df = None
            self._refresh_details_panel()
            if self._sel_viz is not None:
                self._sel_viz.set_enabled(False)

            return
    
        try:
            cols = [str(c) for c in self._df.columns]
        except Exception:
             cols = []
 
        self._colmap.reset(cols)
        
        try:
            self._colmap.auto_assign()
        except Exception:
            pass
        
        self._emit_colmap_patch()

        self._show_loaded()
        self._refresh_loaded_labels()
        self._refresh_preview()
        
        if self.cb_auto_insights is not None:
            self._on_auto_insights_toggled(
                self.cb_auto_insights.isChecked()
            )

        self._refresh_details_panel()
    
        # highlight in library
        if self._csv_path is not None:
            self.refresh_library(select_path=self._csv_path)
    
        if self.cb_auto_insights is not None:
            self._on_auto_insights_toggled(
                self.cb_auto_insights.isChecked()
            )
            
        self.dataset_changed.emit(cols)
        

    def refresh_library(self, select_path: Optional[str | Path] = None) -> None:
        root = self._datasets_root
        self.list_datasets.clear()

        if root is None or not root.exists():
            return

        term = self.edt_lib_search.text().strip().lower()
        paths = list(root.glob("*.csv"))

        # Sort newest first
        try:
            paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            paths.sort()

        selected_norm = None
        if select_path:
            try:
                selected_norm = str(Path(select_path).resolve())
            except Exception:
                selected_norm = str(select_path)

        for p in paths:
            name = p.stem
            if term and term not in name.lower():
                continue

            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, str(p))

            self.list_datasets.addItem(item)

            if self._csv_path is not None:
                try:
                    cur = str(self._csv_path.resolve())
                except Exception:
                    cur = str(self._csv_path)
                if str(p) == str(self._csv_path) or cur == str(p):
                    item.setSelected(True)

            if selected_norm:
                try:
                    if str(p.resolve()) == selected_norm:
                        item.setSelected(True)
                        self.list_datasets.scrollToItem(item)
                except Exception:
                    pass

    def set_dirty(self, dirty: bool) -> None:
        self._dirty = bool(dirty)
        self._refresh_loaded_labels()

    def current_csv_path(self) -> Optional[Path]:
        return self._csv_path

    def current_df(self) -> Optional[pd.DataFrame]:
        return self._df

    # -----------------------------------------------------------------
    # Empty / loaded pages
    # -----------------------------------------------------------------
    def _build_empty_page(self) -> None:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(12)

        lay.addStretch(2)

        title = QLabel("No dataset loaded")
        title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        title.setStyleSheet("font-size:18px; font-weight:600;")
        lay.addWidget(title)

        msg = QLabel(
            "Load a dataset to preview, edit and save it.\n"
            "Or pick one from the library on the left."
        )
        msg.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        msg.setWordWrap(True)
        lay.addWidget(msg)

        btn = QPushButton("Load new dataset…")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedWidth(240)
        btn.clicked.connect(self._emit_open)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

        lay.addStretch(3)

        self._stack.addWidget(page)
        self._empty_page = page

    def _build_loaded_page(self) -> None:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # Top action row
        top = QHBoxLayout()
        top.setSpacing(8)

        self.btn_load = QPushButton("Load…")
        self.btn_edit = QPushButton("Edit…")
        self.btn_save = QPushButton("Save")
        self.btn_save_as = QPushButton("Save as…")
        self.btn_reload = QPushButton("Reload")

        for b in (
            self.btn_load,
            self.btn_edit,
            self.btn_save,
            self.btn_save_as,
            self.btn_reload,
        ):
            b.setCursor(Qt.PointingHandCursor)

        self.btn_load.clicked.connect(self._emit_open)
        self.btn_edit.clicked.connect(self.request_edit.emit)
        self.btn_save.clicked.connect(self.request_save.emit)
        self.btn_save_as.clicked.connect(self.request_save_as.emit)
        self.btn_reload.clicked.connect(self.request_reload.emit)
        
        self.cb_auto_insights = QCheckBox("Insights")
        self.cb_auto_insights.setChecked(True)
        self.cb_auto_insights.toggled.connect(
            self._on_auto_insights_toggled
        )


        top.addWidget(self.btn_load)
        top.addWidget(self.btn_edit)
        top.addWidget(self.btn_save)
        top.addWidget(self.btn_save_as)
        top.addWidget(self.btn_reload)
        top.addWidget(self.cb_auto_insights)
        top.addStretch(1)

        self.lbl_state = QLabel("")
        self.lbl_state.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.lbl_state)

        outer.addLayout(top)

        # -------------------------------------------------
        # Context bar: Results root + Filter + Rows (single row)
        # -------------------------------------------------
        bar = QHBoxLayout()
        bar.setSpacing(8)

        def _mini_btn(
            std_icon: QStyle.StandardPixmap,
            tip: str,
            *,
            svg: Optional[ Union [str, Tuple[str, ...]]] = None,
        ) -> QToolButton:
            b = QToolButton()
            b.setObjectName("miniAction")
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedSize(28, 28)
            ico = None
            if isinstance (svg, str): 
                svg = [svg]
            if svg:
                for nm in svg:
                    ico = try_icon(nm)
                    if ico is not None:
                        break
        
            if ico is None:
                ico = self.style().standardIcon(std_icon)
            b.setIcon(ico)
            return b

        # Results root (global context)
        self.btn_results_root = _mini_btn(
            QStyle.SP_DialogOpenButton,
            "Change results root…"
        )
        self.btn_results_root.clicked.connect(
            self.request_browse_results_root.emit
        )

        self.edt_results_root = QLineEdit()
        self.edt_results_root.setReadOnly(True)
        self.edt_results_root.setObjectName("resultsRootEdit")
        self.edt_results_root.setPlaceholderText("Results root…")

        self.btn_open_results_root = _mini_btn(
            QStyle.SP_DirOpenIcon,
            "Open results root"
        )
        self.btn_open_results_root.setEnabled(False)
        self.btn_open_results_root.clicked.connect(
            self.request_open_results_root.emit
        )

        # Filter columns (local control)
        self.btn_filter_icon = _mini_btn(
            QStyle.SP_FileDialogContentsView, # neutral icon; acts as "filter/search"
            "Filter columns",
            svg=("filter.svg", "filter2.svg"),
        )

        self.edt_col_filter = QLineEdit()
        self.edt_col_filter.setPlaceholderText("Filter columns…")
        self.edt_col_filter.textChanged.connect(self._refresh_preview)

        self.btn_filter_clear = _mini_btn(
            QStyle.SP_DialogResetButton,
            "Clear filter"
        )
        self.btn_filter_clear.clicked.connect(
            lambda: self.edt_col_filter.setText("")
        )

        def _focus_filter() -> None:
            self.edt_col_filter.setFocus()
            self.edt_col_filter.selectAll()

        self.btn_filter_icon.clicked.connect(_focus_filter)

        # Rows (local control)
        self.spin_preview = QSpinBox()
        self.spin_preview.setMinimum(50)
        self.spin_preview.setMaximum(50_000)
        self.spin_preview.setSingleStep(50)
        self.spin_preview.setValue(500)
        self.spin_preview.valueChanged.connect(self._refresh_preview)

        # Layout: [Root icon][Root field][Open]  |  [Filter icon][Filter][Clear]  Rows:[spin]
        bar.addWidget(self.btn_results_root, 0)
        bar.addWidget(self.edt_results_root, 2)
        bar.addWidget(self.btn_open_results_root, 0)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        bar.addWidget(sep, 0)

        bar.addWidget(self.btn_filter_icon, 0)
        bar.addWidget(self.edt_col_filter, 3)
        bar.addWidget(self.btn_filter_clear, 0)
        bar.addWidget(QLabel("Rows:"), 0)
        bar.addWidget(self.spin_preview, 0)

        outer.addLayout(bar)

        self._viz_split = QSplitter(Qt.Vertical, page)
        
        self.table = QTableView()
        self.table.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QTableView.NoEditTriggers)
        self.table.horizontalHeader().sectionClicked.connect(
            self._on_preview_header_clicked
        )
        
        self._viz_pane = SelectionInsightsPane(parent=page)
        self._viz_pane.setVisible(True)
        
        self._viz_split.addWidget(self.table)
        self._viz_split.addWidget(self._viz_pane)
        
        self._viz_split.setStretchFactor(0, 1)
        self._viz_split.setStretchFactor(1, 0)
        # self._viz_split.setSizes([650, 250])
        self._viz_split.setSizes([650, 350])

        outer.addWidget(self._viz_split, 1)


        self._stack.addWidget(page)
        self._loaded_page = page
        

        self._sel_viz = SelectionVizController(
            table=self.table,
            pane=self._viz_pane,
            df_getter=self._get_viz_df,
            debounce_ms=180,
            max_rows=3000,
            max_cols=12,
            parent=self,
        )
        
        if self.cb_auto_insights is not None:
            self._on_auto_insights_toggled(
                self.cb_auto_insights.isChecked()
            )


    # -----------------------------------------------------------------
    # Library interactions
    # -----------------------------------------------------------------


    def _on_preview_header_clicked(self, idx: int) -> None:
        if self._df is None:
            return
        if self._preview_df is None:
            return
    
        col = str(self._preview_df.columns[idx])
    
        menu = QMenu(self)
        act_none = menu.addAction("Unassigned (keep original)")
        act_none.setCheckable(True)
        act_none.setChecked(self._colmap.role_for(col) is None)
    
        menu.addSeparator()
    
        acts = {}
        for spec in self._colmap.available_roles_for(col):
            a = menu.addAction(spec.label)
            a.setCheckable(True)
            a.setChecked(self._colmap.role_for(col) == spec.role)
            acts[a] = spec
    
        chosen = menu.exec_(QCursor.pos())
        if not chosen:
            return
    
        if chosen == act_none:
            self._colmap.unassign(col)
        else:
            self._colmap.assign(col, acts[chosen].role)
    
        self._emit_colmap_patch()

        self._refresh_preview()
        self._refresh_details_panel()


    def _emit_open(self) -> None:
        # Old app.py expects request_open
        self.request_open.emit()
        # New name (if you later switch app.py)
        self.request_open_new.emit()

    def _selected_library_path(self) -> Optional[str]:
        item = self.list_datasets.currentItem()
        if item is None:
            return None
        val = item.data(Qt.UserRole)
        return str(val) if val else None

    def _on_library_item_activated(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.UserRole)
        if p:
            self.request_load_saved.emit(str(p))

    def _load_selected_clicked(self) -> None:
        p = self._selected_library_path()
        if p:
            self.request_load_saved.emit(p)

    def _duplicate_selected_clicked(self) -> None:
        p = self._selected_library_path()
        if p:
            self.request_duplicate_saved.emit(p)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _show_empty(self) -> None:
        self._stack.setCurrentWidget(self._empty_page)

    def _show_loaded(self) -> None:
        self._stack.setCurrentWidget(self._loaded_page)

    def _refresh_loaded_labels(self) -> None:
        df = self._df
        if df is None:
            return

        self.lbl_state.setText(
            "Unsaved changes" if self._dirty else "Saved"
        )
        self.btn_save.setEnabled(self._dirty)

    def _refresh_preview(self) -> None:
        if self._df is None:
            return

        df = self._df
        term = self.edt_col_filter.text().strip().lower()

        if term:
            cols = [c for c in df.columns if term in str(c).lower()]
            if cols:
                df = df.loc[:, cols]

        n = int(self.spin_preview.value())
        view_df = df.head(n)

        # model = _PandasModel(view_df, parent=self)
        # self.table.setModel(model)
        # self.table.resizeColumnsToContents()
        self._preview_df = view_df
        # model = RolePandasModel(view_df, self._colmap, parent=self)
        # self.tbl_preview.setModel(model)
        model = RolePandasModel(
             view_df,
             self._colmap,
             parent=self,
         )
        self.table.setModel(model)
        self.table.resizeColumnsToContents()
        if self._sel_viz is not None:
            self._sel_viz.set_table(self.table)
            self._sel_viz.schedule_refresh()


    def dataframe_for_save(self) -> Optional[pd.DataFrame]:
        if self._df is None:
            return None
        ren = self._colmap.rename_map()
        if not ren:
            return self._df
        return self._df.rename(columns=ren)

    def _emit_colmap_patch(self) -> None:
        patch = self._colmap.to_feature_overrides_patch()
        self.column_overrides_changed.emit(patch)

    def _get_viz_df(self) -> Optional[pd.DataFrame]:
        return self._preview_df

    def _on_auto_insights_toggled(self, checked: bool) -> None:
        if self._sel_viz is not None:
            self._sel_viz.set_enabled(bool(checked))
    
        if self._viz_pane is not None:
            self._viz_pane.setVisible(bool(checked))
    
        if self._viz_split is None:
            return
    
        if checked:
            self._viz_split.setSizes([650, 350])
        else:
            self._viz_split.setSizes([9999, 0])
            
    def set_results_root(self, root: Optional[str | Path]) -> None:
        self._results_root = Path(root) if root else None

        # UI exists only after _build_loaded_page()
        if not hasattr(self, "edt_results_root"):
            return

        txt = str(self._results_root) if self._results_root else ""
        self.edt_results_root.setText(txt)
        self.edt_results_root.setToolTip(
            txt
            or "Results root: base folder for all runs/outputs "
               "(Stage-1/Train/Tune/Infer/Xfer)."
        )

        has = bool(txt)
        self.btn_open_results_root.setEnabled(has)

        # show end of path (more useful than start)
        if has:
            self.edt_results_root.setCursorPosition(len(txt))
