
import json 
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QGroupBox,
    QDialogButtonBox,
    QFormLayout,
)

from ..smart_stage1 import Stage1Summary


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

