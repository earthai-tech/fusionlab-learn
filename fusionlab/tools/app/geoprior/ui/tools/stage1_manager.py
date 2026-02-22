
from __future__ import annotations 
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

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
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
    QStyle,
    QTabWidget
)


from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

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
    COL_STATUS = 0
    COL_CITY = 1
    COL_MODEL = 2
    COL_TS = 3
    COL_T = 4
    COL_H = 5
    COL_TEND = 6
    COL_FSTART = 7
    COL_NSTAT = 8
    COL_NDYN = 9
    COL_NFUT = 10
    COL_RUNDIR = 11


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
        self._run_by_key: Dict[str, Dict[str, Any]] = {}
        self._ov_cbar = None

        self._init_ui()
        self._refresh()


    # ==================================================================
    # UI construction
    # ==================================================================
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
    
        # --------------------------------------------------------------
        # Summary (no duplicate title; ToolPageFrame already has it)
        # --------------------------------------------------------------
        self._summary_lbl = QLabel(
            "Scanning results directories for Stage-1 manifests…",
            self,
        )
        self._summary_lbl.setWordWrap(True)
        self._summary_lbl.setStyleSheet(
            "color: palette(mid); font-style: italic;"
        )
        layout.addWidget(self._summary_lbl)
    
        # --------------------------------------------------------------
        # Filter + actions (Refresh + Use button on SAME row)
        # --------------------------------------------------------------
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
    
        filter_row.addWidget(QLabel("Filter by city:", self))
    
        self._city_combo = QComboBox(self)
        self._city_combo.addItem("All cities", userData=None)
        self._city_combo.currentIndexChanged.connect(
            self._on_city_changed
        )
        filter_row.addWidget(self._city_combo)
    
        filter_row.addStretch(1)
    
        self._btn_refresh = QPushButton("Refresh", self)
        self._btn_refresh.clicked.connect(self._refresh)
        filter_row.addWidget(self._btn_refresh)
    
        self._btn_use_for_city = QPushButton(
            "Use for this city in GUI",
            self,
        )
        self._btn_use_for_city.setEnabled(False)
        self._btn_use_for_city.clicked.connect(
            self._on_use_for_city
        )
        filter_row.addWidget(self._btn_use_for_city)
    
        layout.addLayout(filter_row)
    
        # --------------------------------------------------------------
        # Runs table
        # --------------------------------------------------------------
        runs_group = QGroupBox("Available Stage-1 runs", self)
        runs_layout = QVBoxLayout(runs_group)
        runs_layout.setContentsMargins(6, 6, 6, 6)
        runs_layout.setSpacing(4)
    
        self._table = QTableWidget(self)
        self._table.setColumnCount(12)
        self._table.setHorizontalHeaderLabels(
            [
                "Status",
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
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
    
        hv = self._table.horizontalHeader()
        for c in range(self._table.columnCount()):
            hv.setSectionResizeMode(
                c,
                QHeaderView.ResizeToContents,
            )
        hv.setSectionResizeMode(
            self.COL_RUNDIR,
            QHeaderView.Stretch,
        )
    
        self._table.itemSelectionChanged.connect(
            self._on_table_selection_changed
        )
    
        runs_layout.addWidget(self._table)
    
        # --------------------------------------------------------------
        # Details panel (tabs: Inspector + Overview)
        # --------------------------------------------------------------
        details_group = QGroupBox(
            "Selected manifest (config summary)",
            self,
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
    
        self._details_tabs = QTabWidget(details_group)
        details_layout.addWidget(self._details_tabs, 1)
    
        # --- Inspector tab (tree)
        tab_ins = QWidget(self._details_tabs)
        ins_l = QVBoxLayout(tab_ins)
        ins_l.setContentsMargins(0, 0, 0, 0)
        ins_l.setSpacing(6)
    
        ins_l.addWidget(self._details_label)
    
        self._details_tree = QTreeWidget(tab_ins)
        self._details_tree.setColumnCount(2)
        self._details_tree.setHeaderLabels(["Item", "Value"])
        self._details_tree.setAlternatingRowColors(True)
        self._details_tree.header().setStretchLastSection(True)
        ins_l.addWidget(self._details_tree, 1)
    
        self._details_tabs.addTab(tab_ins, "Inspector")
    
        # --- Overview tab (chart)
        tab_ov = QWidget(self._details_tabs)
        ov_l = QVBoxLayout(tab_ov)
        ov_l.setContentsMargins(0, 0, 0, 0)
        ov_l.setSpacing(6)
    
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Graphic:", tab_ov))
    
        self._cmb_overview = QComboBox(tab_ov)
        self._cmb_overview.addItem(
            "Feature footprint (stacked bars)",
            userData="footprint",
        )
        self._cmb_overview.addItem(
            "City similarity (Jaccard heatmap)",
            userData="similarity",
        )
        self._cmb_overview.currentIndexChanged.connect(
            self._render_overview
        )
    
        row.addWidget(self._cmb_overview)
        row.addStretch(1)
        ov_l.addLayout(row)
    
        self._ov_fig = Figure()
        self._ov_ax = self._ov_fig.add_subplot(111)
        self._ov_canvas = FigureCanvas(self._ov_fig)
        ov_l.addWidget(self._ov_canvas, 1)
    
        self._details_tabs.addTab(tab_ov, "Overview")
    
        # --------------------------------------------------------------
        # Splitter
        # --------------------------------------------------------------
        split = QSplitter(Qt.Vertical, self)
        split.setChildrenCollapsible(False)
        split.addWidget(runs_group)
        split.addWidget(details_group)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
    
        layout.addWidget(split, 1)


    # ==================================================================
    # Discovery helpers
    # ==================================================================
    def refresh(self) -> None:
        self._refresh()

    def _run_key(self, run: Dict[str, Any]) -> str:
        return str(run.get("manifest_path") or "")
    
    def _run_status(self, run: Dict[str, Any]) -> tuple[str, str]:
        req = [
            "scaled_csv",
            "train_sequences",
            "train_inputs_npz",
            "train_targets_npz",
            "val_inputs_npz",
            "val_targets_npz",
        ]
        ok = 0
        missing: List[str] = []
        for k in req:
            p = run.get(k)
            if self._path_ok(p):
                ok += 1
            else:
                missing.append(k)
    
        if ok == len(req):
            return ("OK", "All key artifacts found.")
        if ok == 0:
            return ("MISSING", "No key artifacts found.")
        miss = ", ".join(missing)
        return ("PARTIAL", f"Missing: {miss}")
    
    def _elide_mid(self, text: str, px: int) -> str:
        from PyQt5.QtGui import QFontMetrics
        fm = QFontMetrics(self._table.font())
        return fm.elidedText(text, Qt.ElideMiddle, px)
    
    def _status_icon(self, label: str):
        st = self.style()
        if label == "OK":
            return st.standardIcon(QStyle.SP_DialogApplyButton)
        if label == "PARTIAL":
            return st.standardIcon(QStyle.SP_MessageBoxWarning)
        return st.standardIcon(QStyle.SP_MessageBoxCritical)

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
    def _pick(
        d: Dict[str, Any],
        *keys: str,
    ) -> Optional[Any]:
        for k in keys:
            if k in d:
                return d.get(k)
        return None
    
    @staticmethod
    def _extract_run_info(
        manifest: Dict[str, Any],
        manifest_path: Path,
        root: Path,
    ) -> Dict[str, Any]:
        cfg = manifest.get("config") or {}
        cols = cfg.get("cols") or {}
        feats = cfg.get("features") or {}
    
        arts = manifest.get("artifacts") or {}
        csvs = arts.get("csv") or {}
        seqs = arts.get("sequences") or {}
        numpy_sec = arts.get("numpy") or {}
        shapes = arts.get("shapes") or {}
    
        static_f = feats.get("static") or []
        dynamic_f = feats.get("dynamic") or []
        future_f = feats.get("future") or []
    
        subs_col = Stage1ManagerTool._pick(
            cols,
            "subs_model",
            "subs_raw",
            "subsidence",
            "subs",
        )
        gwl_col = Stage1ManagerTool._pick(
            cols,
            "depth_model",
            "head_model",
            "depth_raw",
            "head_raw",
            "gwl",
        )
        h_field_col = Stage1ManagerTool._pick(
            cols,
            "h_field_model",
            "h_field_raw",
            "h_field",
        )
    
        train_shapes = (shapes.get("train_inputs") or {})
        dyn_shape = train_shapes.get("dynamic_features")
    
        return {
            "schema_version": manifest.get("schema_version", ""),
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
            "subs_col": subs_col,
            "gwl_col": gwl_col,
            "h_field_col": h_field_col,
    
            "static_features": static_f,
            "dynamic_features": dynamic_f,
            "future_features": future_f,
            "group_id_cols": feats.get("group_id_cols") or [],
    
            "raw_csv": csvs.get("raw"),
            "clean_csv": csvs.get("clean"),
            "scaled_csv": csvs.get("scaled"),
    
            "train_sequences": seqs.get("joblib_train_sequences"),
            "train_inputs_npz": numpy_sec.get("train_inputs_npz"),
            "train_targets_npz": numpy_sec.get("train_targets_npz"),
            "val_inputs_npz": numpy_sec.get("val_inputs_npz"),
            "val_targets_npz": numpy_sec.get("val_targets_npz"),
    
            "dynamic_shape": dyn_shape,
    
            # keep these for Inspector sections
            "conventions": cfg.get("conventions") or {},
            "indices": cfg.get("indices") or {},
            "feature_registry": cfg.get("feature_registry") or {},
            "censoring": cfg.get("censoring") or {},
            "scaler_info": cfg.get("scaler_info") or {},
            "scaling_kwargs": cfg.get("scaling_kwargs") or {},
            "units_provenance": cfg.get("units_provenance") or {},
            "versions": manifest.get("versions") or {},
            "paths": manifest.get("paths") or {},
            "artifacts": arts,
            "shapes": shapes,
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
        self._details_tree.clear()
        self._render_overview()
        self._btn_use_for_city.setEnabled(False)

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._filtered_runs))
        self._run_by_key = {}
    
        for r_i, run in enumerate(self._filtered_runs):
            key = self._run_key(run)
            self._run_by_key[key] = run
    
            st, tip = self._run_status(run)
            st_item = QTableWidgetItem(st)
            st_item.setToolTip(tip)
            st_item.setData(Qt.UserRole, key)
            st_item.setIcon(self._status_icon(st))
            st_item.setTextAlignment(
                Qt.AlignCenter | Qt.AlignVCenter
            )
            self._table.setItem(r_i, self.COL_STATUS, st_item)
    
            vals = [
                (self.COL_CITY, run["city"]),
                (self.COL_MODEL, run["model"]),
                (self.COL_TS, run["timestamp"]),
                (self.COL_T, run.get("time_steps")),
                (self.COL_H, run.get("horizon_years")),
                (self.COL_TEND, run.get("train_end_year")),
                (self.COL_FSTART, run.get("forecast_start_year")),
                (self.COL_NSTAT, len(run["static_features"])),
                (self.COL_NDYN, len(run["dynamic_features"])),
                (self.COL_NFUT, len(run["future_features"])),
            ]
    
            for col, v in vals:
                it = QTableWidgetItem("" if v is None else str(v))
                if col >= self.COL_T and col <= self.COL_NFUT:
                    it.setTextAlignment(
                        Qt.AlignRight | Qt.AlignVCenter
                    )
                self._table.setItem(r_i, col, it)
    
            full = str(run.get("run_dir") or "")
            el = self._elide_mid(full, 420)
            p_it = QTableWidgetItem(el)
            p_it.setToolTip(full)
            p_it.setData(Qt.UserRole, full)
            self._table.setItem(r_i, self.COL_RUNDIR, p_it)

    # ==================================================================
    # Selection / details
    # ==================================================================
    def _current_run(self) -> Optional[Dict[str, Any]]:
        row = self._table.currentRow()
        if row < 0:
            return None
        it = self._table.item(row, self.COL_STATUS)
        if it is None:
            return None
        key = it.data(Qt.UserRole)
        if not key:
            return None
        return self._run_by_key.get(str(key))

    def _on_table_selection_changed(self) -> None:
        run = self._current_run()
        self._btn_use_for_city.setEnabled(run is not None)
        self._update_details(run)
        
    def _tgroup(
        self,
        title: str,
    ) -> QTreeWidgetItem:
        g = QTreeWidgetItem(self._details_tree)
        g.setText(0, title)
        g.setFirstColumnSpanned(True)
        g.setExpanded(True)
        return g
    
    def _tkv(
        self,
        parent: QTreeWidgetItem,
        k: str,
        v: str,
    ) -> None:
        it = QTreeWidgetItem(parent)
        it.setText(0, k)
        it.setText(1, v)
        it.setToolTip(1, v)
    
    def _tlist(
        self,
        parent: QTreeWidgetItem,
        title: str,
        items: List[str],
    ) -> None:
        g = QTreeWidgetItem(parent)
        g.setText(0, f"{title} ({len(items)})")
        g.setFirstColumnSpanned(True)
        g.setExpanded(False)
        for x in items:
            it = QTreeWidgetItem(g)
            it.setText(0, x)
    
    def _path_ok(
        self,
        p: Optional[str],
    ) -> bool:
        if not p:
            return False
        try:
            return Path(p).exists()
        except Exception:
            return False

    def _update_details(
        self,
        run: Optional[Dict[str, Any]],
    ) -> None:
        self._details_tree.clear()
    
        if run is None:
            self._details_label.setText(
                "Select a Stage-1 run above to see key "
                "configuration details and artifact paths."
            )
            return
    
        self._details_label.setText(
            f"Manifest for <b>{run['city']}</b> / "
            f"<code>{run['model']}</code> in:<br>"
            f"{run['run_dir']}"
        )
    
        g_run = self._tgroup("Run")
        self._tkv(g_run, "Schema", run.get("schema_version", ""))
        self._tkv(g_run, "Timestamp", run.get("timestamp", ""))
        self._tkv(g_run, "Manifest", str(run.get("manifest_path")))
    
        g_time = self._tgroup("Temporal window")
        self._tkv(g_time, "T", str(run.get("time_steps")))
        self._tkv(g_time, "H (yrs)", str(run.get("horizon_years")))
        self._tkv(
            g_time,
            "Train end",
            str(run.get("train_end_year")),
        )
        self._tkv(
            g_time,
            "Forecast start",
            str(run.get("forecast_start_year")),
        )
    
        g_cols = self._tgroup("Columns (resolved)")
        self._tkv(g_cols, "time", str(run.get("time_col")))
        self._tkv(g_cols, "lon", str(run.get("lon_col")))
        self._tkv(g_cols, "lat", str(run.get("lat_col")))
        self._tkv(g_cols, "subs", str(run.get("subs_col")))
        self._tkv(g_cols, "gwl/head", str(run.get("gwl_col")))
        self._tkv(g_cols, "H_field", str(run.get("h_field_col")))
    
        g_feat = self._tgroup("Features")
        self._tlist(
            g_feat,
            "Static",
            run.get("static_features") or [],
        )
        self._tlist(
            g_feat,
            "Dynamic",
            run.get("dynamic_features") or [],
        )
        self._tlist(
            g_feat,
            "Future",
            run.get("future_features") or [],
        )
        self._tkv(
            g_feat,
            "Group id cols",
            ", ".join(run.get("group_id_cols") or []) or "—",
        )
    
        g_ready = self._tgroup("Artifacts readiness")
        for k in (
            "raw_csv",
            "clean_csv",
            "scaled_csv",
            "train_sequences",
            "train_inputs_npz",
            "train_targets_npz",
            "val_inputs_npz",
            "val_targets_npz",
        ):
            p = run.get(k)
            ok = self._path_ok(p)
            tag = "OK" if ok else "MISSING"
            self._tkv(
                g_ready,
                k,
                f"{tag}  {p or '—'}",
            )
    
        g_phys = self._tgroup("Conventions & indices")
        conv = run.get("conventions") or {}
        idx = run.get("indices") or {}
        self._tkv(g_phys, "gwl_kind", str(conv.get("gwl_kind")))
        self._tkv(g_phys, "gwl_sign", str(conv.get("gwl_sign")))
        self._tkv(
            g_phys,
            "gwl_dyn_name",
            str(idx.get("gwl_dyn_name")),
        )
    
        g_cens = self._tgroup("Censoring")
        cens = run.get("censoring") or {}
        self._tkv(
            g_cens,
            "use_effective_h_field",
            str(cens.get("use_effective_h_field")),
        )
        rep = (cens.get("report") or {})
        for name, info in rep.items():
            it = QTreeWidgetItem(g_cens)
            it.setText(0, f"{name}")
            it.setExpanded(False)
            for kk in ("cap", "direction", "censored_rate"):
                self._tkv(it, kk, str(info.get(kk)))
    
        g_ver = self._tgroup("Versions")
        ver = run.get("versions") or {}
        for kk in ("python", "tensorflow", "numpy", "pandas", "sklearn"):
            if kk in ver:
                self._tkv(g_ver, kk, str(ver.get(kk)))

    def _ts_key(
        self,
        ts: str,
    ):
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.min
    
    def _latest_per_city(
        self,
        runs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        for r in runs:
            city = str(r.get("city") or "")
            cur = best.get(city)
            if cur is None:
                best[city] = r
                continue
            a = self._ts_key(str(r.get("timestamp") or ""))
            b = self._ts_key(str(cur.get("timestamp") or ""))
            if a >= b:
                best[city] = r
        return [best[k] for k in sorted(best.keys())]
    
    def _render_overview(self) -> None:
        # Clear the *entire figure* so old colorbar axes
        # never accumulate.
        self._ov_fig.clf()
        self._ov_ax = self._ov_fig.add_subplot(111)
        self._ov_cbar = None
    
        runs = self._latest_per_city(self._filtered_runs)
        mode = self._cmb_overview.currentData()
    
        if not runs:
            self._ov_ax.text(
                0.5,
                0.5,
                "No runs to plot",
                ha="center",
                va="center",
                transform=self._ov_ax.transAxes,
            )
            self._ov_canvas.draw_idle()
            return
    
        if mode == "similarity":
            self._plot_similarity(runs)
        else:
            self._plot_footprint(runs)
    
        self._ov_fig.tight_layout()
        self._ov_canvas.draw_idle()

    def _plot_footprint(
        self,
        runs: List[Dict[str, Any]],
    ) -> None:
        cities = [r["city"] for r in runs]
        stat = [len(r.get("static_features") or []) for r in runs]
        dyn = [len(r.get("dynamic_features") or []) for r in runs]
        fut = [len(r.get("future_features") or []) for r in runs]
        bot = [a + b for a, b in zip(stat, dyn)]
        x = list(range(len(cities)))
    
        self._ov_ax.bar(x, stat, label="static")
        self._ov_ax.bar(x, dyn, bottom=stat, label="dynamic")
        self._ov_ax.bar(x, fut, bottom=bot, label="future")
    
        self._ov_ax.set_xticks(x)
        self._ov_ax.set_xticklabels(cities, rotation=0)
        self._ov_ax.set_ylabel("feature count")
        self._ov_ax.set_title("Feature footprint (latest per city)")
        self._ov_ax.legend()
    
    def _plot_similarity(self, runs: List[Dict[str, Any]]) -> None:
        cities = [r["city"] for r in runs]
        n = len(cities)
    
        if n < 2:
            self._ov_ax.text(
                0.5,
                0.5,
                "Need ≥ 2 cities for similarity",
                ha="center",
                va="center",
                transform=self._ov_ax.transAxes,
            )
            self._ov_ax.set_title("City similarity")
            return
    
        sets: List[set] = []
        for r in runs:
            s = set(r.get("static_features") or [])
            s |= set(r.get("dynamic_features") or [])
            s |= set(r.get("future_features") or [])
            sets.append(s)
    
        m = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                a = sets[i]
                b = sets[j]
                den = len(a | b)
                m[i][j] = (len(a & b) / den) if den else 0.0
    
        im = self._ov_ax.imshow(m, aspect="auto")
    
        # Only ONE colorbar because fig was cleared in _render_overview
        self._ov_cbar = self._ov_fig.colorbar(
            im,
            ax=self._ov_ax,
            fraction=0.046,
            pad=0.04,
        )
    
        self._ov_ax.set_xticks(range(n))
        self._ov_ax.set_yticks(range(n))
        self._ov_ax.set_xticklabels(cities, rotation=45, ha="right")
        self._ov_ax.set_yticklabels(cities)
        self._ov_ax.set_title("City similarity (Jaccard of features)")


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
