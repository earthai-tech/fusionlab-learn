# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import json 
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QGroupBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QTableWidget,
    QAbstractItemView,
    QTableWidgetItem,
    QHeaderView,
)

from ..smart_stage1 import Stage1Summary

class Stage1ChoiceDialog(QDialog):
    """
    Let the user decide how to handle Stage-1 before training.

    API
    ---
    decision, summary = Stage1ChoiceDialog.ask(
        parent, city, runs_for_city, all_runs, clean_stage1=False
    )

    decision in {"reuse", "rebuild", "cancel"}.
    summary is the selected Stage1Summary (for reuse) or None.
    """

    def __init__(
        self,
        parent,
        city,
        runs_for_city,
        all_runs,
        clean_stage1: bool = False,
    ):
        super().__init__(parent)
        self._city = city
        self._runs = runs_for_city
        self._all_runs = all_runs
        self._clean_stage1 = bool(clean_stage1)

        self.decision = "cancel"
        self.selected_summary = None

        self.setWindowTitle(f"Stage-1 runs for {city}")
        self.resize(700, 420)

        layout = QVBoxLayout(self)

        # --- Optional danger zone banner when cleanup is enabled ---------
        if self._clean_stage1:
            danger = QLabel(
                "<b>Danger zone:</b> "
                "'Clean Stage-1 run dir' is enabled. "
                "If you choose to rebuild, the entire "
                "<code>&lt;city&gt;_GeoPriorSubsNet_stage1/</code> "
                "directory for this city will be removed "
                "before rebuilding."
            )
            danger.setTextFormat(Qt.RichText)
            danger.setWordWrap(True)
            danger.setStyleSheet("color:#b03030; font-size:9pt;")
            layout.addWidget(danger)

        # --- table with existing runs ------------------------------------
        self.table = QTableWidget(len(self._runs), 5, self)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.table.setHorizontalHeaderLabels(
            ["City", "Timestamp", "T", "H (years)", "Status"]
        )

        for row, s in enumerate(self._runs):
            self.table.setItem(row, 0, QTableWidgetItem(s.city))
            self.table.setItem(row, 1, QTableWidgetItem(s.timestamp))
            self.table.setItem(row, 2, QTableWidgetItem(str(s.time_steps)))
            self.table.setItem(row, 3, QTableWidgetItem(str(s.horizon_years)))
            status = "OK" if s.is_complete else "Incomplete"
            if not s.config_match:
                status += " (config mismatch)"
            self.table.setItem(row, 4, QTableWidgetItem(status))

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        layout.addWidget(self.table)

        # --- diff label ---------------------------------------------------
        self.diff_label = QLabel("")
        self.diff_label.setObjectName("diffLabel")
        self.diff_label.setStyleSheet("color:#d98a00; font-weight:600;")
        layout.addWidget(self.diff_label)

        # --- buttons ------------------------------------------------------
        btn_row = QHBoxLayout()
        self.reuse_btn = QPushButton("Reuse selected Stage-1")

        if self._clean_stage1:
            self.rebuild_btn = QPushButton("Rebuild (clean Stage-1 dir)")
            self.rebuild_btn.setStyleSheet("color:#b03030;")
        else:
            self.rebuild_btn = QPushButton("Rebuild Stage-1")

        self.cancel_btn = QPushButton("Cancel")

        btn_row.addWidget(self.reuse_btn)
        btn_row.addWidget(self.rebuild_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.cancel_btn)

        layout.addLayout(btn_row)

        # connections
        self.reuse_btn.clicked.connect(self._accept_reuse)
        self.rebuild_btn.clicked.connect(self._accept_rebuild)
        self.cancel_btn.clicked.connect(self.reject)

        self.table.selectionModel().currentRowChanged.connect(
            self._on_row_changed
        )

        if self._runs:
            self.table.selectRow(len(self._runs) - 1)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @classmethod
    def ask(
        cls,
        parent,
        city,
        runs_for_city,
        all_runs,
        clean_stage1: bool = False,
    ):
        """
        Open the dialog and return (decision, selected_summary).

        Parameters
        ----------
        clean_stage1 : bool, default=False
            If True, the dialog highlights that rebuilding will clean
            the Stage-1 directory for this city and shows an extra
            destructive-action confirmation when the user clicks
            "Rebuild".
        """
        dlg = cls(
            parent=parent,
            city=city,
            runs_for_city=runs_for_city,
            all_runs=all_runs,
            clean_stage1=clean_stage1,
        )
        result = dlg.exec_()
        if result != QDialog.Accepted:
            return "cancel", None
        return dlg.decision, dlg.selected_summary

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _on_row_changed(self, current, _previous):
        row = current.row()
        if row < 0 or row >= len(self._runs):
            self.diff_label.setText("")
            return

        summary = self._runs[row]
        if summary.diff_fields:
            self.diff_label.setText(
                "⚠ changed: " + ", ".join(summary.diff_fields)
            )
        else:
            self.diff_label.setText(
                "✓ config matches current GUI setup."
            )

    def _accept_reuse(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self._runs):
            QMessageBox.warning(
                self,
                "No selection",
                "Please select a Stage-1 run to reuse.",
            )
            return

        self.decision = "reuse"
        self.selected_summary = self._runs[row]
        self.accept()

    def _accept_rebuild(self):
        # If clean_stage1 is True, ask one more time with rich text
        if self._clean_stage1:
            msg_html = (
                "<p><span style='color:#b03030; font-weight:bold;'>"
                "Rebuild Stage-1 with cleanup</span></p>"
                "<p>You have enabled "
                "<b>'Clean Stage-1 run dir before running Stage-1'</b> "
                "in the Training options.</p>"
                "<p>Rebuilding Stage-1 for "
                f"city <b>{self._city}</b> will first delete the "
                "<code>&lt;city&gt;_GeoPriorSubsNet_stage1/</code> "
                "directory for this city, including:</p>"
                "<ul>"
                "<li><code>artifacts/</code> "
                "(NPZ sequences, <code>manifest.json</code>)</li>"
                "<li><code>train_*/</code> (all past training runs)</li>"
                "<li><code>tuning/</code> (all tuning runs)</li>"
                "<li><code>inference/</code> "
                "(diagnostic results and summaries)</li>"
                "</ul>"
                "<p><b>This cannot be undone.</b></p>"
                "<p>Do you still want to rebuild Stage-1 for this city?</p>"
            )

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Confirm Stage-1 rebuild (with cleanup)")
            box.setTextFormat(Qt.RichText)
            box.setText(msg_html)
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setDefaultButton(QMessageBox.No)

            if box.exec_() != QMessageBox.Yes:
                return  # user cancelled rebuild

        self.decision = "rebuild"
        self.selected_summary = None
        self.accept()

class Stage1DetailsDialog(QDialog):
    """
    Simple read-only view of a Stage-1 manifest for a given city.
    """

    def __init__(self, summary: Stage1Summary, parent=None) -> None:
        super().__init__(parent)
        self.summary = summary

        self.setWindowTitle(f"Stage-1 details — {summary.city}")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        # Header
        header = QLabel(
            f"<b>{summary.city}</b> — {summary.timestamp}"
        )
        header.setTextFormat(Qt.RichText)
        layout.addWidget(header)

        run_dir_lbl = QLabel(f"Run dir: {summary.run_dir}")
        run_dir_lbl.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        run_dir_lbl.setWordWrap(True)
        layout.addWidget(run_dir_lbl)

        # Load manifest config for extra details
        cfg = {}
        features = {}
        censoring = {}
        mode = "?"
        try:
            with summary.manifest_path.open(
                "r", encoding="utf-8"
            ) as f:
                manifest = json.load(f)
            cfg = manifest.get("config", {}) or {}
            features = cfg.get("features", {}) or {}
            censoring = cfg.get("censoring", {}) or {}
            mode = cfg.get("MODE", "?")
        except Exception:
            # Keep things robust: just don't show extended info
            pass

        form = QFormLayout()
        form.addRow("Timestamp:", QLabel(summary.timestamp))
        form.addRow(
            "Time steps (T):", QLabel(str(summary.time_steps))
        )
        form.addRow(
            "Horizon (H, years):",
            QLabel(str(summary.horizon_years)),
        )
        form.addRow(
            "Train / Val samples:",
            QLabel(f"{summary.n_train} / {summary.n_val}"),
        )
        form.addRow(
            "Train end year:",
            QLabel(str(summary.train_end_year)),
        )
        form.addRow(
            "Forecast start year:",
            QLabel(str(summary.forecast_start_year)),
        )
        form.addRow("Mode:", QLabel(str(mode)))

        complete_lbl = QLabel(
            "Yes" if summary.is_complete else "No"
        )
        form.addRow("Complete:", complete_lbl)

        match_text = (
            "Yes" if summary.config_match else "No"
        )
        if summary.diff_fields:
            match_text += " (diff: " + ", ".join(summary.diff_fields) + ")"
        form.addRow(
            "Config matches GUI:",
            QLabel(match_text),
        )

        layout.addLayout(form)

        # Features summary (static / dynamic / future)
        if features:
            feats_box = QGroupBox("Features", self)
            v = QVBoxLayout(feats_box)

            def _fmt_list(name: str, seq) -> str:
                seq = [str(s) for s in (seq or [])]
                if not seq:
                    return f"{name}: (none)"
                if len(seq) > 10:
                    head = ", ".join(seq[:8])
                    return f"{name}: {len(seq)} ({head}, …)"
                return f"{name}: {', '.join(seq)}"

            v.addWidget(
                QLabel(
                    _fmt_list("Static", features.get("static"))
                )
            )
            v.addWidget(
                QLabel(
                    _fmt_list("Dynamic", features.get("dynamic"))
                )
            )
            v.addWidget(
                QLabel(
                    _fmt_list("Future", features.get("future"))
                )
            )
            feats_box.setLayout(v)
            layout.addWidget(feats_box)

        # Censoring
        if censoring:
            c_box = QGroupBox("Censoring", self)
            v = QVBoxLayout(c_box)
            specs = censoring.get("specs")
            enabled = bool(specs)
            txt = "Enabled" if enabled else "Disabled"
            if isinstance(specs, dict) and specs:
                txt += " (" + ", ".join(sorted(specs.keys())) + ")"
            v.addWidget(QLabel(txt))
            c_box.setLayout(v)
            layout.addWidget(c_box)

        # Completeness issues, if any
        if (
            not summary.is_complete
            and summary.completeness_errors
        ):
            err_box = QGroupBox("Completeness issues", self)
            v = QVBoxLayout(err_box)
            for err in summary.completeness_errors:
                v.addWidget(QLabel("• " + err))
            err_box.setLayout(v)
            layout.addWidget(err_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

