# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Results & Downloads tab for GeoPrior GUI.

Provides a compact, scrollable view over all available results
(artifacts, train runs, tuning runs, inference runs, xfer runs)
under a given results root, and lets the user download any job
as a ZIP archive with a simple progress dialog.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional, Callable
from html import escape

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QGroupBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QProgressDialog,
    QLineEdit,        
    QMessageBox,      
)

from .results_index import (
    discover_results_for_root,
    ResultsIndex,
    CityResults,
)
from .styles import PRIMARY  


# ---------------------------------------------------------------------
# Zip worker thread
# ---------------------------------------------------------------------
class ZipWorker(QThread):
    """Background worker that zips a directory with progress."""

    progress_changed = pyqtSignal(int, int)  # done, total
    finished_ok = pyqtSignal(str)           # target path
    failed = pyqtSignal(str)                # error message

    def __init__(
        self,
        source_dir: Path,
        target_zip: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.source_dir = Path(source_dir)
        self.target_zip = Path(target_zip)

    def run(self) -> None:
        try:
            files: list[Path] = []
            for root, _, filenames in os.walk(self.source_dir):
                for fn in filenames:
                    files.append(Path(root) / fn)

            total = max(len(files), 1)
            done = 0

            self.target_zip.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(
                self.target_zip,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as zf:
                for fp in files:
                    if self.isInterruptionRequested():
                        raise RuntimeError("Zipping cancelled by user")

                    rel = fp.relative_to(self.source_dir)
                    zf.write(str(fp), str(rel))
                    done += 1
                    self.progress_changed.emit(done, total)

            self.finished_ok.emit(str(self.target_zip))
        except Exception as e:
            self.failed.emit(str(e))

# ---------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------
class ResultsDownloadTab(QWidget):
    """
    Main tab widget for browsing GeoPrior results and downloading jobs.

    Parameters
    ----------
    results_root : str or Path
        Initial results root (e.g. ``~/.fusionlab_runs`` or ``tests``).
    get_results_root : callable or None
        Optional callback returning the current results root from the
        main window. If provided, it is used when refreshing.
    """

    def __init__(
        self,
        *,
        results_root: Path | str,
        get_results_root: Optional[Callable[[], Path | str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._get_results_root = get_results_root
        self._results_root = Path(results_root).expanduser().resolve()
        self._index: Optional[ResultsIndex] = None

        self._current_city: Optional[str] = None
        self._current_kind: Optional[str] = None  # "artifacts"/"train"/"tune"/"inference"
        
        self._zip_worker: ZipWorker | None = None
        self._zip_progress: QProgressDialog | None = None

        self._build_ui()
        self.refresh_index()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
    
        # Top row: root selector
        top = QHBoxLayout()
        top.addWidget(QLabel("Results root:"))
    
        self.root_edit = QLineEdit(str(self._results_root))
        self.root_edit.setReadOnly(True)
        top.addWidget(self.root_edit, 1)
    
        self.browse_root_btn = QPushButton("Browse…")
        top.addWidget(self.browse_root_btn)
    
        self.refresh_btn = QPushButton("Refresh")
        top.addWidget(self.refresh_btn)
    
        layout.addLayout(top)
    
        # ------------------------------------------------------------------
        # Middle: vertical splitter
        #   - top: [cities/workflows] | [xfer runs]
        #   - bottom: [details]
        # ------------------------------------------------------------------
        main_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(main_splitter, 1)
    
        # ----- top: horizontal splitter -----
        top_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(top_splitter)
    
        # Left: cities & workflows
        left_box = QGroupBox("Cities & workflows")
        left_layout = QVBoxLayout(left_box)
    
        self.cities_table = QTableWidget()
        self.cities_table.setColumnCount(5)
        self.cities_table.setHorizontalHeaderLabels(
            ["City", "Artifacts", "Train", "Tune", "Inference"]
        )
        self.cities_table.verticalHeader().setVisible(False)
        self.cities_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cities_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cities_table.setSelectionMode(QTableWidget.SingleSelection)
        self.cities_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.cities_table.setAlternatingRowColors(True)
        for col in range(1, 5):
            self.cities_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
    
        left_layout.addWidget(self.cities_table)
        top_splitter.addWidget(left_box)
    
        # Right: transferability runs
        xfer_box = QGroupBox("Transferability runs")
        xfer_layout = QVBoxLayout(xfer_box)
    
        self.xfer_table = QTableWidget()
        self.xfer_table.setColumnCount(4)
        self.xfer_table.setHorizontalHeaderLabels(
            ["City A", "City B", "Timestamp", "Download"]
        )
        self.xfer_table.verticalHeader().setVisible(False)
        self.xfer_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.xfer_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.xfer_table.setSelectionMode(QTableWidget.SingleSelection)
        self.xfer_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.xfer_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.xfer_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.xfer_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents
        )
        self.xfer_table.setAlternatingRowColors(True)
    
        xfer_layout.addWidget(self.xfer_table)
        top_splitter.addWidget(xfer_box)
    
        top_splitter.setStretchFactor(0, 1)  # cities
        top_splitter.setStretchFactor(1, 1)  # xfer
    
        # ----- bottom: details (full width) -----
        details_box = QGroupBox("Details")
        right_layout = QVBoxLayout(details_box)
    
        self.details_label = QLabel("Select a city + workflow to see jobs.")
        self.details_label.setTextFormat(Qt.RichText)  # allow HTML
        right_layout.addWidget(self.details_label)
    
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(4)
        self.details_table.setHorizontalHeaderLabels(
            ["Type", "Job", "Path", "Download"]
        )
        self.details_table.verticalHeader().setVisible(False)
        self.details_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.details_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.details_table.setSelectionMode(QTableWidget.SingleSelection)
        self.details_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.details_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.details_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch
        )
        self.details_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents
        )
        self.details_table.setAlternatingRowColors(True)
    
        right_layout.addWidget(self.details_table)
        main_splitter.addWidget(details_box)
    
        # Details should get more height
        main_splitter.setStretchFactor(0, 1)  # top (cities + xfer)
        main_splitter.setStretchFactor(1, 2)  # bottom (details)
    
        # Connections
        self.browse_root_btn.clicked.connect(self._on_browse_root)
        self.refresh_btn.clicked.connect(self.refresh_index)

    # ------------------------------------------------------------------
    # Root handling
    # ------------------------------------------------------------------
    def _current_root(self) -> Path:
        if self._get_results_root is not None:
            try:
                root = Path(self._get_results_root()).expanduser()
            except Exception:
                root = self._results_root
        else:
            root = self._results_root
        return root.resolve()

    def _on_browse_root(self) -> None:
        start_dir = str(self._current_root())
        path = QFileDialog.getExistingDirectory(
            self,
            "Select results root",
            start_dir,
        )
        if not path:
            return
        self._results_root = Path(path).expanduser().resolve()
        self.root_edit.setText(str(self._results_root))
        self.refresh_index()

    # ------------------------------------------------------------------
    # Discovery + table population
    # ------------------------------------------------------------------
    def refresh_index(self) -> None:
        root = self._current_root()
        self.root_edit.setText(str(root))
        self._index = discover_results_for_root(root)
        self._populate_cities_table()
        self._populate_xfer_table()
        self._clear_details()

    def _populate_cities_table(self) -> None:
        self.cities_table.clearContents()
        if self._index is None:
            self.cities_table.setRowCount(0)
            return

        cities = sorted(self._index.cities.values(), key=lambda cr: cr.city.lower())
        self.cities_table.setRowCount(len(cities))

        for row, city_res in enumerate(cities):
            # City label
            item_city = QTableWidgetItem(city_res.city)
            self.cities_table.setItem(row, 0, item_city)

            self._add_workflow_cell(
                row=row,
                col=1,
                city_res=city_res,
                kind="artifacts",
                count=1 if city_res.artifacts_dir else 0,
            )
            self._add_workflow_cell(
                row=row,
                col=2,
                city_res=city_res,
                kind="train",
                count=len(city_res.train_runs),
            )
            self._add_workflow_cell(
                row=row,
                col=3,
                city_res=city_res,
                kind="tune",
                count=len(city_res.tune_runs),
            )
            self._add_workflow_cell(
                row=row,
                col=4,
                city_res=city_res,
                kind="inference",
                count=len(city_res.inference_runs),
            )
        
        self.cities_table.resizeRowsToContents()

    def _add_workflow_cell(
        self,
        *,
        row: int,
        col: int,
        city_res: CityResults,
        kind: str,
        count: int,
    ) -> None:
        """Populate a workflow cell with either a dash or a Browse button.
    
        If ``count`` is zero, we show a non-interactive em dash. Otherwise a
        small pill-style button with primary background and white text that
        supports hover / pressed states.
        """
        if count <= 0:
            item = QTableWidgetItem("—")
            item.setFlags(Qt.ItemIsEnabled)
            self.cities_table.setItem(row, col, item)
            return
    
        text = f"Browse ({count})"
        btn = QPushButton(text)
        btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {PRIMARY};
                color: white;
                border-radius: 10px;
                padding: 2px 8px;
            }}
            QPushButton:hover {{
                background-color: #4047C8;  /* slightly lighter than PRIMARY */
                color: white;
            }}
            QPushButton:pressed {{
                background-color: #30369F;  /* slightly darker than PRIMARY */
                color: white;
            }}
            """
        )
    
        city = city_res.city
    
        def on_click(_checked: bool = False, city=city, kind=kind) -> None:
            self._show_city_details(city, kind)
    
        btn.clicked.connect(on_click)
        self.cities_table.setCellWidget(row, col, btn)

    def _populate_xfer_table(self) -> None:
        self.xfer_table.clearContents()
        if self._index is None:
            self.xfer_table.setRowCount(0)
            return

        runs = list(self._index.xfer_runs)
        self.xfer_table.setRowCount(len(runs))

        for row, r in enumerate(runs):
            self.xfer_table.setItem(row, 0, QTableWidgetItem(r.city_a))
            self.xfer_table.setItem(row, 1, QTableWidgetItem(r.city_b))
            self.xfer_table.setItem(row, 2, QTableWidgetItem(r.stamp))

            btn = QPushButton("Download…")

            def on_click(_checked: bool = False, run=r) -> None:
                label = f"{run.city_a}_to_{run.city_b}_{run.stamp}"
                self._download_directory(run.run_dir, label)

            btn.clicked.connect(on_click)
            self.xfer_table.setCellWidget(row, 3, btn)
        
        self.xfer_table.resizeRowsToContents()
        
    # ------------------------------------------------------------------
    # Details view
    # ------------------------------------------------------------------
    def _clear_details(self) -> None:
        self._current_city = None
        self._current_kind = None
        self.details_label.setText("Select a city + workflow to see jobs.")
        self.details_table.setRowCount(0)
        self.details_table.clearContents()

    def _show_city_details(self, city: str, kind: str) -> None:
        if self._index is None:
            return
        city_res = self._index.cities.get(city)
        if city_res is None:
            return

        self._current_city = city
        self._current_kind = kind
        
        city_html = escape(city)

        rows: list[tuple[str, str, Path]] = [] 
        
        if kind == "artifacts":
            if city_res.artifacts_dir is not None:
                rows.append(
                    ("artifacts", "Stage-1 artifacts", city_res.artifacts_dir)
                )
        elif kind == "train":
            rows = [
                (
                    "train",
                    f"{r.stamp}",
                    r.run_dir,
                )
                for r in city_res.train_runs
            ]
        elif kind == "tune":
            rows = [
                (
                    "tune",
                    f"{r.stamp}",
                    r.run_dir,
                )
                for r in city_res.tune_runs
            ]
        elif kind == "inference":
            rows = []
            for r in city_res.inference_runs:
                ds = r.dataset or "?"
                label = f"{r.stamp} (dataset={ds})"
                rows.append(("inference", label, r.run_dir))
 
        self.details_table.setRowCount(len(rows))
        self.details_table.clearContents()

        if kind == "artifacts":
            title = f"Stage-1 artifacts for city: <b>{city_html}</b>"
        else:
            title = f"{kind.capitalize()} jobs for <b>{city_html}</b>"
        self.details_label.setText(title)

        for row_idx, (typ, label, path) in enumerate(rows):
            self.details_table.setItem(row_idx, 0, QTableWidgetItem(typ))
            self.details_table.setItem(row_idx, 1, QTableWidgetItem(label))
            self.details_table.setItem(row_idx, 2, QTableWidgetItem(str(path)))

            btn = QPushButton("Download…")

            def on_click(
                _checked: bool = False,
                city=city,
                typ=typ,
                label=label,
                path=path,
            ) -> None:
                safe_label = label.replace(" ", "_").replace("[", "").replace("]", "")
                base_name = f"{city}_{typ}_{safe_label}"
                self._download_directory(path, base_name)

            btn.clicked.connect(on_click)
            self.details_table.setCellWidget(row_idx, 3, btn)
            
        self.details_table.resizeRowsToContents()
    # ------------------------------------------------------------------
    # Zipping + progress dialog
    # ------------------------------------------------------------------

    def _download_directory(self, run_dir: Path, base_label: str) -> None:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            QMessageBox.warning(
                self,
                "Missing directory",
                f"The selected job directory does not exist:\n{run_dir}",
            )
            return

        # Suggest a default file name in the user's home folder
        default_name = f"{base_label}.zip"
        suggested = str(Path.home() / default_name)

        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save job as ZIP",
            suggested,
            "ZIP archives (*.zip)",
        )
        if not target_path:
            return

        target = Path(target_path)
        if target.suffix.lower() != ".zip":
            target = target.with_suffix(".zip")

        # Progress dialog
        progress = QProgressDialog(
            "Creating ZIP archive…",
            "Cancel",
            0,
            100,
            self,
        )
        progress.setWindowTitle("Zipping job")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setValue(0)

        worker = ZipWorker(run_dir, target, self)
        self._zip_worker = worker
        self._zip_progress = progress

        def on_progress(done: int, total: int) -> None:
            if total <= 0:
                progress.setRange(0, 0)  # indeterminate
            else:
                progress.setRange(0, total)
                progress.setValue(done)

        def on_finished_ok(path_str: str) -> None:
            progress.setValue(progress.maximum())
            progress.close()
            self._zip_worker = None
            self._zip_progress = None
            QMessageBox.information(
                self,
                "Download ready",
                f"ZIP archive saved to:\n{path_str}",
            )

        def on_failed(msg: str) -> None:
            progress.close()
            self._zip_worker = None
            self._zip_progress = None
            QMessageBox.critical(
                self,
                "Error while zipping",
                f"Could not create archive:\n{msg}",
            )

        def on_cancel() -> None:
            if self._zip_worker is not None:
                self._zip_worker.requestInterruption()

        worker.progress_changed.connect(on_progress)
        worker.finished_ok.connect(on_finished_ok)
        worker.failed.connect(on_failed)
        progress.canceled.connect(on_cancel)

        worker.start()
        progress.show()


