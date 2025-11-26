from typing import Any, Dict, List, Optional

import os
import json
import csv

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QGridLayout,
    QDialog,
    QDialogButtonBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView,
    # QFileDialog,
    QMessageBox,
    QPushButton
    
)


class XferAdvancedDialog(QDialog):
    """
    Simple dialog for advanced cross-city transfer settings:
    - quantiles override
    - write_json / write_csv switches
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        quantiles: Any = None,
        write_json: bool = True,
        write_csv: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Transfer matrix – advanced options")

        layout = QVBoxLayout(self)

        lbl = QLabel(
            "Optional settings for the cross-city transfer matrix.\n"
            "Leave quantiles blank to use the configuration stored in "
            "each city's manifest."
        )
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        form = QGridLayout()
        row = 0

        form.addWidget(QLabel("Quantiles override:"), row, 0)
        self.quantiles_edit = QLineEdit()
        if quantiles:
            try:
                self.quantiles_edit.setText(
                    ", ".join(f"{float(q):g}" for q in quantiles)
                )
            except Exception:
                # Fallback to string representation
                self.quantiles_edit.setText(str(quantiles))
        else:
            self.quantiles_edit.setPlaceholderText(
                "e.g. 0.1, 0.5, 0.9 (optional)"
            )
        form.addWidget(self.quantiles_edit, row, 1)
        row += 1

        self.chk_write_json = QCheckBox("Write transfer results on JSON")
        self.chk_write_json.setChecked(write_json)
        form.addWidget(self.chk_write_json, row, 0, 1, 2)
        row += 1

        self.chk_write_csv = QCheckBox("Write transfer results on CSV")
        self.chk_write_csv.setChecked(write_csv)
        form.addWidget(self.chk_write_csv, row, 0, 1, 2)
        row += 1

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_quantiles(self) -> list[float] | None:
        text = self.quantiles_edit.text().strip()
        if not text:
            return None

        vals: list[float] = []
        for tok in text.replace(";", ",").split(","):
            t = tok.strip()
            if not t:
                continue
            try:
                vals.append(float(t))
            except Exception:
                # silently ignore malformed tokens
                continue
        return vals or None

    def write_json(self) -> bool:
        return self.chk_write_json.isChecked()

    def write_csv(self) -> bool:
        return self.chk_write_csv.isChecked()

class XferResultsDialog(QDialog):
    """
    Pretty dialog to inspect cross-city transfer (xfer) results.

    It consumes:
    - xfer_results.json (preferred)
    - or xfer_results.csv (fallback)

    and summarises metrics for:
    - A_to_B and B_to_A directions
    - all calibration modes (none / source / target)
    """

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------
    @classmethod
    def show_for_xfer_result(
        cls,
        parent: QWidget,
        result: Dict[str, Any],
        *,
        split: str = "val",
        title: Optional[str] = None,
    ) -> None:
        """
        Open a dialog for a finished XferMatrixJob / XferMatrixThread.

        Parameters
        ----------
        parent :
            Parent widget (usually the main GeoPriorForecaster).
        result :
            Dict emitted by XferMatrixThread.xfer_finished, i.e.

            {
              "out_dir": ...,
              "results": [...],
              "json_path": "xfer_results.json" or None,
              "csv_path": "xfer_results.csv" or None,
            }
        split :
            Which split to focus on (e.g. 'val' or 'test').
        title :
            Optional window title override.
        """
        out_dir = result.get("out_dir")
        json_path = result.get("json_path")
        csv_path = result.get("csv_path")

        # Best effort: infer paths from out_dir if missing
        if out_dir:
            if not json_path:
                cand = os.path.join(out_dir, "xfer_results.json")
                if os.path.exists(cand):
                    json_path = cand
            if not csv_path:
                cand = os.path.join(out_dir, "xfer_results.csv")
                if os.path.exists(cand):
                    csv_path = cand

        # If both missing, nothing to show
        if not (json_path and os.path.exists(json_path)) and not (
            csv_path and os.path.exists(csv_path)
        ):
            QMessageBox.information(
                parent,
                "No transfer results",
                "No xfer_results.json or xfer_results.csv found.\n"
                "Run the transfer matrix first.",
            )
            return

        dlg = cls(
            json_path=json_path if json_path and os.path.exists(json_path) else None,
            csv_path=csv_path if csv_path and os.path.exists(csv_path) else None,
            split=split,
            parent=parent,
            title=title,
        )
        dlg.exec_()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        json_path: Optional[str],
        csv_path: Optional[str],
        split: str = "val",
        parent: Optional[QWidget] = None,
        title: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._json_path = json_path
        self._csv_path = csv_path
        self._split = split

        self._records: List[Dict[str, Any]] = self._load_records()

        if not self._records:
            QMessageBox.information(
                self,
                "No rows for split",
                f"No transfer rows found for split={split!r}.",
            )
            self.close()
            return

        self.setWindowTitle(title or "Cross-city transfer results")
        self.setModal(True)
        self.resize(900, 520)

        main_layout = QVBoxLayout(self)

        header = QLabel(self._build_header_text())
        header.setWordWrap(True)
        header.setObjectName("xferHeaderLabel")
        main_layout.addWidget(header)

        tabs = QTabWidget(self)
        mae_keys, r2_keys = self._collect_horizon_keys()

        for direction in self._unique_directions():
            dir_records = [
                r for r in self._records if r.get("direction") == direction
            ]
            if not dir_records:
                continue
            tab_widget, tab_label = self._build_direction_tab(
                direction, dir_records, mae_keys, r2_keys
            )
            tabs.addTab(tab_widget, tab_label)

        main_layout.addWidget(tabs, 1)

        # Buttons: Close / Open folder / Open CSV / Open JSON
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_open_folder = QPushButton("Open folder")
        btn_box.addButton(btn_open_folder, QDialogButtonBox.ActionRole)

        if self._csv_path:
            btn_open_csv = QPushButton("Open CSV")
            btn_box.addButton(btn_open_csv, QDialogButtonBox.ActionRole)
            btn_open_csv.clicked.connect(self._on_open_csv)

        if self._json_path:
            btn_open_json = QPushButton("Open JSON")
            btn_box.addButton(btn_open_json, QDialogButtonBox.ActionRole)
            btn_open_json.clicked.connect(self._on_open_json)

        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)

        btn_open_folder.clicked.connect(self._on_open_folder)

    # ------------------------------------------------------------------
    # Data loading / normalisation
    # ------------------------------------------------------------------
    def _load_records(self) -> List[Dict[str, Any]]:
        # Prefer JSON – richer structure
        if self._json_path and os.path.exists(self._json_path):
            try:
                with open(self._json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
            if not isinstance(data, list):
                data = []

            recs = [r for r in data if r.get("split") == self._split]
            return recs or data  # fall back to all if no match

        # Fallback: CSV
        if self._csv_path and os.path.exists(self._csv_path):
            rows: List[Dict[str, Any]] = []
            try:
                with open(self._csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("split") and row["split"] != self._split:
                            continue
                        rows.append(self._normalize_csv_row(row))
            except Exception:
                rows = []
            return rows

        return []

    @staticmethod
    def _to_float(val: Any) -> Optional[float]:
        if val in (None, "", "NA"):
            return None
        try:
            return float(val)
        except Exception:
            return None

    def _normalize_csv_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "direction": row.get("direction"),
            "source_city": row.get("source_city"),
            "target_city": row.get("target_city"),
            "split": row.get("split"),
            "calibration": row.get("calibration"),
            "coverage80": self._to_float(row.get("coverage80")),
            "sharpness80": self._to_float(row.get("sharpness80")),
            "overall_mae": self._to_float(row.get("overall_mae")),
            "overall_mse": self._to_float(row.get("overall_mse")),
            "overall_r2": self._to_float(row.get("overall_r2")),
            "quantiles": None,
            "per_horizon_mae": {},
            "per_horizon_r2": {},
        }

        for k, v in row.items():
            if k.startswith("per_horizon_mae."):
                h = k.split(".", 1)[1]
                rec["per_horizon_mae"][h] = self._to_float(v)
            elif k.startswith("per_horizon_r2."):
                h = k.split(".", 1)[1]
                rec["per_horizon_r2"][h] = self._to_float(v)
        return rec

    # ------------------------------------------------------------------
    # Header + horizon keys
    # ------------------------------------------------------------------
    def _unique_directions(self) -> List[str]:
        seen = []
        for r in self._records:
            d = r.get("direction")
            if d and d not in seen:
                seen.append(d)
        return seen or ["A_to_B", "B_to_A"]

    def _collect_horizon_keys(self) -> tuple[List[str], List[str]]:
        mae_keys = set()
        r2_keys = set()
        for r in self._records:
            mae_keys |= set((r.get("per_horizon_mae") or {}).keys())
            r2_keys |= set((r.get("per_horizon_r2") or {}).keys())

        def sort_key(name: str) -> int:
            # Expect 'H1', 'H2', ...
            try:
                return int(str(name).strip().split("H")[-1])
            except Exception:
                return 9999

        mae_sorted = sorted(mae_keys, key=sort_key)
        r2_sorted = sorted(r2_keys, key=sort_key)
        return mae_sorted, r2_sorted

    def _build_header_text(self) -> str:
        src_cities = {r.get("source_city") for r in self._records if r.get("source_city")}
        tgt_cities = {r.get("target_city") for r in self._records if r.get("target_city")}
        cities_str = " ↔ ".join(sorted(src_cities | tgt_cities)) or "unknown cities"

        # Quantiles: try to infer from first record
        q = None
        for r in self._records:
            q = r.get("quantiles")
            if q:
                break
        if isinstance(q, (list, tuple)):
            q_str = ", ".join(f"{float(x):g}" for x in q)
        else:
            q_str = "n/a"

        backend = "JSON" if self._json_path else "CSV"
        return (
            f"Cross-city transfer matrix for {cities_str}  |  "
            f"split={self._split}  |  quantiles={q_str}  "
            f"(source={backend})"
        )

    # ------------------------------------------------------------------
    # Per-direction tabs
    # ------------------------------------------------------------------
    def _build_direction_tab(
        self,
        direction: str,
        records: List[Dict[str, Any]],
        mae_keys: List[str],
        r2_keys: List[str],
    ) -> tuple[QWidget, str]:
        w = QWidget(self)
        layout = QVBoxLayout(w)

        if not records:
            layout.addWidget(QLabel("No rows for this direction."))
            return w, direction

        first = records[0]
        src = first.get("source_city") or "source"
        tgt = first.get("target_city") or "target"

        title = QLabel(
            f"{src} → {tgt}  (direction={direction}, n={len(records)})"
        )
        title.setObjectName("xferDirectionLabel")
        layout.addWidget(title)

        # Build table
        base_cols = [
            "Calibration",
            "Coverage80",
            "Sharpness80",
            "Overall MAE",
            "Overall MSE",
            "Overall R²",
        ]
        mae_cols = [f"{k} MAE" for k in mae_keys]
        r2_cols = [f"{k} R²" for k in r2_keys]
        all_cols = base_cols + mae_cols + r2_cols

        table = QTableWidget(len(records), len(all_cols), w)
        table.setHorizontalHeaderLabels(all_cols)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)

        # Order by calibration: none → source → target → others
        order_map = {"none": 0, "source": 1, "target": 2}
        records_sorted = sorted(
            records,
            key=lambda r: order_map.get(r.get("calibration"), 9),
        )

        for row_idx, rec in enumerate(records_sorted):
            c = 0
            table.setItem(
                row_idx, c, QTableWidgetItem(str(rec.get("calibration")))
            ); c += 1
            table.setItem(
                row_idx, c, QTableWidgetItem(self._fmt(rec.get("coverage80")))
            ); c += 1
            table.setItem(
                row_idx, c, QTableWidgetItem(self._fmt(rec.get("sharpness80")))
            ); c += 1
            table.setItem(
                row_idx, c, QTableWidgetItem(self._fmt(rec.get("overall_mae")))
            ); c += 1
            table.setItem(
                row_idx, c, QTableWidgetItem(self._fmt(rec.get("overall_mse")))
            ); c += 1
            table.setItem(
                row_idx, c, QTableWidgetItem(self._fmt(rec.get("overall_r2")))
            ); c += 1

            ph_mae = rec.get("per_horizon_mae") or {}
            ph_r2 = rec.get("per_horizon_r2") or {}

            for k in mae_keys:
                table.setItem(
                    row_idx, c, QTableWidgetItem(self._fmt(ph_mae.get(k)))
                )
                c += 1

            for k in r2_keys:
                table.setItem(
                    row_idx, c, QTableWidgetItem(self._fmt(ph_r2.get(k)))
                )
                c += 1

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, len(all_cols)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        layout.addWidget(table)
        return w, f"{src} → {tgt}"

    # ------------------------------------------------------------------
    # Buttons handlers
    # ------------------------------------------------------------------
    def _on_open_folder(self) -> None:
        base = self._json_path or self._csv_path
        if not base:
            return
        folder = os.path.dirname(base)
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _on_open_csv(self) -> None:
        if not self._csv_path:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._csv_path))

    def _on_open_json(self) -> None:
        if not self._json_path:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._json_path))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(val: Any) -> str:
        if val is None:
            return "-"
        if isinstance(val, float):
            return f"{val:.4g}"
        return str(val)
