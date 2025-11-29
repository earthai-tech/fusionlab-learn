
from typing import Dict, Any

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QWidget,
    QPushButton,
    QHBoxLayout,
)

from .components import RangeListEditor 
from .geoprior_config import default_tuner_search_space

class ScalarsLossDialog(QDialog):
    """
    Popup to configure scalar hyperparameters and loss weights used in
    TUNER_SEARCH_SPACE.

    Uses RangeListEditor widgets internally (range or discrete list).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scalars & loss weights")
        self.setModal(True)

        main = QVBoxLayout(self)
        info = QLabel(
            "Configure physical scalars and loss weights used during "
            "Stage-2 hyperparameter tuning."
        )
        info.setWordWrap(True)
        main.addWidget(info)

        grid = QGridLayout()
        row = 0

        def add_row(label: str, editor: QWidget) -> None:
            nonlocal row
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(editor, row, 1)
            row += 1

        # --- editors -------------------------------------------------

        spin_w = 130  # or 120, tweak to taste
        self.ed_mv = RangeListEditor(
            min_allowed=1e-9,
            max_allowed=1e-5,
            decimals=8,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_kappa = RangeListEditor(
            min_allowed=0.0,
            max_allowed=10.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lr = RangeListEditor(
            min_allowed=1e-6,
            max_allowed=1e-3,
            decimals=8,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lgw = RangeListEditor(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lcons = RangeListEditor(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lprior = RangeListEditor(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lsmooth = RangeListEditor(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_lmv = RangeListEditor(
            min_allowed=0.0,
            max_allowed=1.0,
            decimals=6,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_mv_lr = RangeListEditor(
            min_allowed=0.0,
            max_allowed=20.0,
            decimals=3,
            show_sampling=True,
            spin_width=spin_w,
        )
        self.ed_kappa_lr = RangeListEditor(
            min_allowed=0.0,
            max_allowed=20.0,
            decimals=3,
            show_sampling=True,
            spin_width=spin_w,
        )

        # --- layout --------------------------------------------------
        add_row("Storage coefficient mᵥ:", self.ed_mv)
        add_row("Consolidation factor κ:", self.ed_kappa)
        add_row("Learning rate:", self.ed_lr)
        add_row("λ (GW loss):", self.ed_lgw)
        add_row("λ (consolidation loss):", self.ed_lcons)
        add_row("λ (prior term):", self.ed_lprior)
        add_row("λ (smoothness):", self.ed_lsmooth)
        add_row("λ (mᵥ regularizer):", self.ed_lmv)
        add_row("LR multiplier for mᵥ:", self.ed_mv_lr)
        add_row("LR multiplier for κ:", self.ed_kappa_lr)

        main.addLayout(grid)

        # --- buttons -------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_reset = QPushButton("Reset to defaults")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok = QPushButton("OK")
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_ok)
        main.addLayout(btn_row)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_reset.clicked.connect(self._on_reset_clicked)

        # Initialize with defaults
        self._on_reset_clicked()

    # ------------------------------------------------------------------
    # Helpers to sync with TUNER_SEARCH_SPACE
    # ------------------------------------------------------------------
    def _on_reset_clicked(self) -> None:
        defaults = default_tuner_search_space()
        self.load_from_space(defaults, defaults)

    def load_from_space(
        self,
        space: Dict[str, Any],
        defaults: Dict[str, Any],
    ) -> None:
        """Fill editors from a tuner search-space dict."""
        def _get(name: str):
            return space.get(name, defaults.get(name))

        self.ed_mv.from_search_space_value(_get("mv"), defaults["mv"])
        self.ed_kappa.from_search_space_value(
            _get("kappa"), defaults["kappa"]
        )
        self.ed_lr.from_search_space_value(
            _get("learning_rate"), defaults["learning_rate"]
        )
        self.ed_lgw.from_search_space_value(
            _get("lambda_gw"), defaults["lambda_gw"]
        )
        self.ed_lcons.from_search_space_value(
            _get("lambda_cons"), defaults["lambda_cons"]
        )
        self.ed_lprior.from_search_space_value(
            _get("lambda_prior"), defaults["lambda_prior"]
        )
        self.ed_lsmooth.from_search_space_value(
            _get("lambda_smooth"), defaults["lambda_smooth"]
        )
        self.ed_lmv.from_search_space_value(
            _get("lambda_mv"), defaults["lambda_mv"]
        )
        self.ed_mv_lr.from_search_space_value(
            _get("mv_lr_mult"), defaults["mv_lr_mult"]
        )
        self.ed_kappa_lr.from_search_space_value(
            _get("kappa_lr_mult"), defaults["kappa_lr_mult"]
        )

    def to_search_space_fragment(self) -> Dict[str, Any]:
        """Return just the scalar / loss entries for TUNER_SEARCH_SPACE."""
        return {
            "mv": self.ed_mv.to_search_space_value(),
            "kappa": self.ed_kappa.to_search_space_value(),
            "learning_rate": self.ed_lr.to_search_space_value(),
            "lambda_gw": self.ed_lgw.to_search_space_value(),
            "lambda_cons": self.ed_lcons.to_search_space_value(),
            "lambda_prior": self.ed_lprior.to_search_space_value(),
            "lambda_smooth": self.ed_lsmooth.to_search_space_value(),
            "lambda_mv": self.ed_lmv.to_search_space_value(),
            "mv_lr_mult": self.ed_mv_lr.to_search_space_value(),
            "kappa_lr_mult": self.ed_kappa_lr.to_search_space_value(),
        }
