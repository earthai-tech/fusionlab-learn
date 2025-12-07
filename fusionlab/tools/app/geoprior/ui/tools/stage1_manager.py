
from __future__ import annotations 
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
    QGroupBox,
    QHeaderView,
    QComboBox,
)

class Stage1ManagerTool(QWidget):
    """
    Inspect and select Stage-1 manifests per city.

    This tool scans one or more ``results_root`` directories for
    folders matching ``*_stage1`` that contain the ``manifest.json``
    file written by :func:`run_stage1`.

    For each manifest it extracts a compact summary:
    - city, model, timestamp
    - temporal window (TIME_STEPS, FORECAST_HORIZON_YEARS,
      TRAIN_END_YEAR, FORECAST_START_YEAR)
    - basic feature counts (#static, #dynamic, #future)
    - main artifact paths (scaled CSV, sequence joblib, NPZs)

    The upper table shows one row per Stage-1 run. Selecting a row
    populates a detailed key/value summary in the lower panel.

    If the main window exposes a method
    ``set_preferred_stage1_manifest(city, manifest_path)``, clicking
    "Use for this city in GUI" will call it. Otherwise, the selected
    manifest is stored in
    ``app_ctx.preferred_stage1_manifest = {'city', 'manifest_path'}``
    so that the rest of the GUI can pick it up if desired.
    """

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._app_ctx: Optional[object] = app_ctx

        # All discovered Stage-1 runs
        self._runs: List[Dict[str, Any]] = []
        # Currently filtered view (per city)
        self._filtered_runs: List[Dict[str, Any]] = []
        # Roots we actually scanned
        self._scan_roots: List[Path] = []
        self._current_city: Optional[str] = None

        self._init_ui()
        self._refresh()

    # ==================================================================
    # UI construction
    # ==================================================================
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ---------------- Header row -----------------------------------
        header = QHBoxLayout()
        header.setSpacing(8)

        title_lbl = QLabel("<b>Stage-1 manager</b>", self)

        self._summary_lbl = QLabel(
            "Scanning results directories for Stage-1 manifests…", self
        )
        self._summary_lbl.setWordWrap(True)
        self._summary_lbl.setStyleSheet(
            "color: palette(mid); font-style: italic;"
        )

        btn_refresh = QPushButton("Refresh", self)
        btn_refresh.clicked.connect(self._refresh)

        header.addWidget(title_lbl)
        header.addSpacing(12)
        header.addWidget(self._summary_lbl, stretch=1)
        header.addWidget(btn_refresh)
        layout.addLayout(header)

        # ---------------- Filter + action row --------------------------
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)

        filter_row.addWidget(QLabel("Filter by city:", self))
        self._city_combo = QComboBox(self)
        self._city_combo.addItem("All cities", userData=None)
        self._city_combo.currentIndexChanged.connect(self._on_city_changed)
        filter_row.addWidget(self._city_combo)

        filter_row.addStretch(1)

        self._btn_use_for_city = QPushButton(
            "Use for this city in GUI", self
        )
        self._btn_use_for_city.setEnabled(False)
        self._btn_use_for_city.clicked.connect(self._on_use_for_city)
        filter_row.addWidget(self._btn_use_for_city)

        layout.addLayout(filter_row)

        # ---------------- Main table (runs) ----------------------------
        runs_group = QGroupBox("Available Stage-1 runs", self)
        runs_layout = QVBoxLayout(runs_group)
        runs_layout.setContentsMargins(6, 6, 6, 6)
        runs_layout.setSpacing(4)

        self._table = QTableWidget(self)
        self._table.setColumnCount(11)
        self._table.setHorizontalHeaderLabels(
            [
                "City",
                "Model",
                "Timestamp",
                "T",
                "H (yrs)",
                "Train end",
                "Forecast start",
                "#Static",
                "#Dynamic",
                "#Future",
                "Run directory",
            ]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        header_view = self._table.horizontalHeader()
        header_view.setStretchLastSection(True)
        header_view.setSectionResizeMode(QHeaderView.ResizeToContents)

        self._table.itemSelectionChanged.connect(
            self._on_table_selection_changed
        )

        runs_layout.addWidget(self._table)
        layout.addWidget(runs_group, stretch=3)

        # ---------------- Details panel --------------------------------
        details_group = QGroupBox(
            "Selected manifest (config summary)", self
        )
        details_layout = QVBoxLayout(details_group)
        details_layout.setContentsMargins(6, 6, 6, 6)
        details_layout.setSpacing(4)

        self._details_label = QLabel(
            "Select a Stage-1 run above to see key configuration "
            "details and artifact paths.",
            self,
        )
        self._details_label.setWordWrap(True)
        self._details_label.setStyleSheet("color: palette(mid);")

        self._details_table = QTableWidget(self)
        self._details_table.setColumnCount(2)
        self._details_table.setHorizontalHeaderLabels(["Key", "Value"])
        self._details_table.verticalHeader().setVisible(False)
        self._details_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._details_table.horizontalHeader().setStretchLastSection(True)
        self._details_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )

        details_layout.addWidget(self._details_label)
        details_layout.addWidget(self._details_table, stretch=1)

        layout.addWidget(details_group, stretch=2)

    # ==================================================================
    # Discovery helpers
    # ==================================================================
    def _candidate_roots(self) -> List[Path]:
        """
        Infer plausible results roots from the app context.

        We look at a few common attributes on the main window
        (``results_root``, ``runs_root``, etc.) and also fall back
        to ``./results`` and ``~/.fusionlab_runs``.
        """
        roots: List[Path] = []

        ctx = self._app_ctx
        if ctx is not None:
            # Main window attributes
            for name in (
                "results_root",
                "runs_root",
                "_results_root",
                "_runs_root",
            ):
                val = getattr(ctx, name, None)
                if val:
                    roots.append(Path(val))

            # GeoPriorConfig-style attributes, if exposed
            geo = getattr(ctx, "geo_cfg", None)
            if geo is not None:
                for name in ("results_root", "runs_root", "BASE_OUTPUT_DIR"):
                    val = getattr(geo, name, None)
                    if val:
                        roots.append(Path(val))

        # Reasonable fallbacks
        roots.append(Path.cwd() / "results")
        roots.append(Path.home() / ".fusionlab_runs")

        # Deduplicate while preserving order
        uniq: List[Path] = []
        seen = set()
        for r in roots:
            try:
                rp = r.expanduser().resolve()
            except Exception:
                continue
            if rp not in seen:
                seen.add(rp)
                uniq.append(rp)
        return uniq

    def _discover_runs(self) -> List[Dict[str, Any]]:
        """
        Scan candidate roots for ``*_stage1/manifest.json``.
        """
        runs: List[Dict[str, Any]] = []
        scan_roots: List[Path] = []

        for root in self._candidate_roots():
            if not root.is_dir():
                continue

            scan_roots.append(root)

            # Convention from run_stage1: {city}_{model}_stage1
            for stage1_dir in root.glob("*_stage1"):
                if not stage1_dir.is_dir():
                    continue

                manifest_path = stage1_dir / "manifest.json"
                if not manifest_path.is_file():
                    continue

                try:
                    with manifest_path.open("r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except Exception:
                    continue

                # Be strict: this is a Stage-1 manifest
                if manifest.get("stage") not in (None, "stage1"):
                    continue

                runs.append(
                    self._extract_run_info(
                        manifest=manifest,
                        manifest_path=manifest_path,
                        root=root,
                    )
                )

        self._scan_roots = scan_roots
        return runs

    @staticmethod
    def _extract_run_info(
        manifest: Dict[str, Any],
        manifest_path: Path,
        root: Path,
    ) -> Dict[str, Any]:
        """
        Pull out the key pieces we care about from a Stage-1 manifest.

        This follows the exact structure written by ``run_stage1``:
        - ``config.TIME_STEPS``
        - ``config.FORECAST_HORIZON_YEARS``
        - ``config.TRAIN_END_YEAR``, ``config.FORECAST_START_YEAR``
        - columns in ``config.cols``
        - feature lists in ``config.features``
        - CSV / NPZ / joblib paths in ``artifacts``
        """
        cfg = manifest.get("config", {}) or {}
        cols = cfg.get("cols", {}) or {}
        feats = cfg.get("features", {}) or {}
        arts = manifest.get("artifacts", {}) or {}
        csvs = arts.get("csv", {}) or {}
        seqs = arts.get("sequences", {}) or {}
        numpy_section = arts.get("numpy", {}) or {}
        shapes = arts.get("shapes", {}) or {}

        static_feats = feats.get("static", []) or []
        dynamic_feats = feats.get("dynamic", []) or []
        future_feats = feats.get("future", []) or []

        train_shapes = shapes.get("train_inputs", {}) or {}
        dyn_shape = train_shapes.get("dynamic_features")

        return {
            "city": manifest.get("city", "?"),
            "model": manifest.get("model", "?"),
            "timestamp": manifest.get("timestamp", ""),
            "manifest_path": manifest_path,
            "run_dir": manifest_path.parent,
            "results_root": root,
            "time_steps": cfg.get("TIME_STEPS"),
            "horizon_years": cfg.get("FORECAST_HORIZON_YEARS"),
            "train_end_year": cfg.get("TRAIN_END_YEAR"),
            "forecast_start_year": cfg.get("FORECAST_START_YEAR"),
            "time_col": cols.get("time"),
            "lon_col": cols.get("lon"),
            "lat_col": cols.get("lat"),
            "subs_col": cols.get("subsidence"),
            "gwl_col": cols.get("gwl"),
            "h_field_col": cols.get("h_field"),
            "static_features": static_feats,
            "dynamic_features": dynamic_feats,
            "future_features": future_feats,
            "group_id_cols": feats.get("group_id_cols", []) or [],
            "raw_csv": csvs.get("raw"),
            "clean_csv": csvs.get("clean"),
            "scaled_csv": csvs.get("scaled"),
            "train_sequences": seqs.get("joblib_train_sequences"),
            "train_inputs_npz": numpy_section.get("train_inputs_npz"),
            "train_targets_npz": numpy_section.get("train_targets_npz"),
            "val_inputs_npz": numpy_section.get("val_inputs_npz"),
            "val_targets_npz": numpy_section.get("val_targets_npz"),
            "dynamic_shape": dyn_shape,
        }

    # ==================================================================
    # Refresh / filtering
    # ==================================================================
    def _refresh(self) -> None:
        """
        Rescan roots and repopulate the table + city filter.
        """
        self._runs = self._discover_runs()
        cities = sorted({r["city"] for r in self._runs}) if self._runs else []

        # Summary label
        if not self._runs:
            self._summary_lbl.setText(
                "No Stage-1 manifests found yet. "
                "Run Stage-1 at least once from the Train tab."
            )
        else:
            roots_str = ", ".join(str(r) for r in self._scan_roots)
            self._summary_lbl.setText(
                f"Found {len(self._runs)} Stage-1 run(s) across "
                f"{len(cities)} citie(s) under: {roots_str}"
            )

        # City filter combo
        self._city_combo.blockSignals(True)
        self._city_combo.clear()
        self._city_combo.addItem("All cities", userData=None)
        for city in cities:
            self._city_combo.addItem(city, userData=city)
        self._city_combo.blockSignals(False)

        self._current_city = None
        self._apply_filter()

    def _on_city_changed(self, idx: int) -> None:
        city = self._city_combo.itemData(idx)
        self._current_city = city
        self._apply_filter()

    def _apply_filter(self) -> None:
        """
        Filter runs according to the selected city and update the table.
        """
        if self._current_city:
            self._filtered_runs = [
                r for r in self._runs if r["city"] == self._current_city
            ]
        else:
            self._filtered_runs = list(self._runs)

        self._populate_table()
        self._details_table.setRowCount(0)
        self._btn_use_for_city.setEnabled(False)

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._filtered_runs))

        for row, run in enumerate(self._filtered_runs):
            values = [
                run["city"],
                run["model"],
                run["timestamp"],
                run.get("time_steps"),
                run.get("horizon_years"),
                run.get("train_end_year"),
                run.get("forecast_start_year"),
                len(run["static_features"]),
                len(run["dynamic_features"]),
                len(run["future_features"]),
                str(run["run_dir"]),
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem("" if val is None else str(val))
                if col in (3, 4, 5, 6, 7, 8, 9):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._table.setItem(row, col, item)


    # ==================================================================
    # Selection / details
    # ==================================================================
    def _current_run(self) -> Optional[Dict[str, Any]]:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._filtered_runs):
            return None
        return self._filtered_runs[row]

    def _on_table_selection_changed(self) -> None:
        run = self._current_run()
        self._btn_use_for_city.setEnabled(run is not None)
        self._update_details(run)

    def _update_details(self, run: Optional[Dict[str, Any]]) -> None:
        """
        Populate the lower key/value table with the selected run.
        """
        if run is None:
            self._details_label.setText(
                "Select a Stage-1 run above to see key configuration "
                "details and artifact paths."
            )
            self._details_table.setRowCount(0)
            return

        self._details_label.setText(
            f"Manifest for <b>{run['city']}</b> / "
            f"<code>{run['model']}</code> in:<br>"
            f"{run['run_dir']}"
        )

        rows: List[tuple[str, str]] = []

        rows.append(("Timestamp", run["timestamp"]))
        rows.append(("Manifest path", str(run["manifest_path"])))
        rows.append(("Results root", str(run["results_root"])))

        rows.append(
            (
                "Temporal window",
                f"T={run.get('time_steps')} | "
                f"H={run.get('horizon_years')} year(s); "
                f"train ≤ {run.get('train_end_year')}, "
                f"forecast from {run.get('forecast_start_year')}",
            )
        )

        rows.append(
            (
                "Columns",
                "time={time}, lon/lat={lon}/{lat}, subs={subs}, "
                "gwl={gwl}, h_field={h_field}".format(
                    time=run.get("time_col"),
                    lon=run.get("lon_col"),
                    lat=run.get("lat_col"),
                    subs=run.get("subs_col"),
                    gwl=run.get("gwl_col"),
                    h_field=run.get("h_field_col"),
                ),
            )
        )

        rows.append(
            (
                "Static features",
                f"{len(run['static_features'])} → "
                + ", ".join(run["static_features"]),
            )
        )
        rows.append(
            (
                "Dynamic features",
                f"{len(run['dynamic_features'])} → "
                + ", ".join(run["dynamic_features"]),
            )
        )
        rows.append(
            (
                "Future drivers",
                f"{len(run['future_features'])} → "
                + (", ".join(run["future_features"]) or "—"),
            )
        )
        rows.append(
            ("Group id cols", ", ".join(run["group_id_cols"]) or "—")
        )

        rows.append(("Raw CSV", run.get("raw_csv") or "—"))
        rows.append(("Clean CSV", run.get("clean_csv") or "—"))
        rows.append(("Scaled CSV", run.get("scaled_csv") or "—"))

        rows.append(
            (
                "Train sequences joblib",
                run.get("train_sequences") or "—",
            )
        )

        dyn_shape = run.get("dynamic_shape")
        if dyn_shape:
            rows.append(
                (
                    "Sequence shape (dynamic_features)",
                    str(dyn_shape),
                )
            )

        rows.append(
            ("train_inputs.npz", run.get("train_inputs_npz") or "—")
        )
        rows.append(
            ("train_targets.npz", run.get("train_targets_npz") or "—")
        )
        rows.append(("val_inputs.npz", run.get("val_inputs_npz") or "—"))
        rows.append(("val_targets.npz", run.get("val_targets_npz") or "—"))

        self._details_table.setRowCount(len(rows))
        for row_idx, (key, value) in enumerate(rows):
            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(Qt.ItemIsEnabled)
            val_item = QTableWidgetItem(str(value))
            val_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self._details_table.setItem(row_idx, 0, key_item)
            self._details_table.setItem(row_idx, 1, val_item)

    # ==================================================================
    # Integration with main window
    # ==================================================================
    def _on_use_for_city(self) -> None:
        """
        Mark the selected Stage-1 run as the preferred one for its city.

        If the main window exposes
        ``set_preferred_stage1_manifest(city, manifest_path)``, that
        will be called. Otherwise, a simple attribute
        ``preferred_stage1_manifest`` is set on the app context.
        """
        run = self._current_run()
        if run is None:
            return

        city = run["city"]
        manifest_path = str(run["manifest_path"])

        handled = False
        ctx = self._app_ctx

        if ctx is not None and hasattr(
            ctx, "set_preferred_stage1_manifest"
        ):
            try:
                ctx.set_preferred_stage1_manifest(city, manifest_path)
                handled = True
            except Exception:
                handled = False

        if not handled and ctx is not None:
            # Simple fallback: stash it on the context
            setattr(
                ctx,
                "preferred_stage1_manifest",
                {"city": city, "manifest_path": manifest_path},
            )

        self._summary_lbl.setText(
            f"Preferred Stage-1 for '{city}' set to:\n{manifest_path}"
        )
