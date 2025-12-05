# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
"""
Inference advanced options dialog.

This module provides a Qt dialog to help the user pick a trained or
tuned GeoPriorSubsNet model for the Inference tab.

Features
--------
- Scan a results root (e.g. ~/.fusionlab_runs or "tests/") for
  <city>_GeoPriorSubsNet_stage1/ folders.
- For each city, discover:
    * Training runs under "train_*/" that contain a Keras model and
      optional training_summary + physical parameters CSV.
    * Tuning runs under "tuning/run_*/" that contain a best tuned
      Keras model and tuning_summary + best_hps JSON.
- GUI layout:
    * Left: list of cities (with counts of train / tuned workflows).
    * Right: two tabs, "Trained models" and "Tuned models",
      each listing the available runs.
    * For each run, the user can:
        - View details (metrics, physics, λ-weights) in a read-only
          dialog.
        - Select the model as the one to use for Inference.
- API:
    accepted, new_root, choice = InferenceOptionsDialog.run(
        parent,
        results_root,
    )

  where `choice` is an InferenceModelChoice dataclass instance or
  None if the user closed the dialog without selecting a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import ( 
    GeoPriorConfig, 
    discover_stage1_runs
    )

# ----------------------------------------------------------------------
# Data structures & helpers
# ----------------------------------------------------------------------

@dataclass
class InferenceModelChoice:
    """
    Selected model description returned to the main GUI.

    Attributes
    ----------
    city : str
        City / dataset name (e.g. "nansha", "zhongshan").
    run_type : {"train", "tune"}
        Source of the model: a plain training run or a tuned workflow.
    run_name : str
        Timestamp-like identifier of the run, e.g. "20251110-122128"
        or "run_20251121-141933".
    run_dir : Path
        Directory that holds the run artifacts.
    model_path : Path
        Path to the chosen Keras model file (.keras or .h5).
    stage1_manifest_path : Path
        Stage-1 manifest.json for this city, required by run_inference.
    training_summary_json : Optional[Path]
        Path to "<city>_GeoPriorSubsNet_training_summary.json", if
        available.
    tuning_summary_json : Optional[Path]
        Path to "tuning_summary.json" for tuned runs, if available.
    best_hps_json : Optional[Path]
        Path to "<city>_GeoPrior_best_hps.json" for tuned runs, if
        available.
    physical_params_csv : Optional[Path]
        Path to "<city>_geopriorsubsnet_physical_parameters.csv", if
        available (only meaningful when physics is enabled).
    """

    city: str
    run_type: str
    run_name: str
    run_dir: Path
    model_path: Path
    stage1_manifest_path: Path
    training_summary_json: Optional[Path] = None
    tuning_summary_json: Optional[Path] = None
    best_hps_json: Optional[Path] = None
    physical_params_csv: Optional[Path] = None


def _safe_load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file defensively."""
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_load_phys_params(path: Path) -> Dict[str, Any]:
    """
    Load physical parameters CSV of the form:

        Parameter,Value
        Compressibility_mv,1.64E-06
        Consistency_Kappa,1.158208847
        Unit_Weight_Water_gamma_w,9810
        Reference_Head_h_ref,0.0
    """
    if not path or not path.exists():
        return {}

    out: Dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param = (row.get("Parameter") or "").strip()
                val = (row.get("Value") or "").strip()
                if param:
                    out[param] = val
    except Exception:
        return {}
    return out


def _is_physics_enabled(summary: Dict[str, Any]) -> bool:
    """
    Decide if physics was actually enabled in this run.

    Uses hp_init["pde_mode"] from the training_summary. If it is
    "none" or "off", physics is considered disabled.

    As a fallback, also inspects metrics_at_best["physics_loss"].
    """
    if not summary:
        return False

    hp_init = summary.get("hp_init", {}) or {}
    pde_mode = hp_init.get("pde_mode")

    if isinstance(pde_mode, str):
        pm = pde_mode.strip().lower()
        if pm not in ("", "none", "off", "disabled"):
            return True

    # Fallback: look at physics_loss magnitude
    metrics = summary.get("metrics_at_best", {}) or {}
    physics_loss = metrics.get("physics_loss")
    if isinstance(physics_loss, (int, float)):
        return physics_loss > 0.0

    return False

def _discover_inference_models(
    results_root: Path,
) -> Dict[str, List[InferenceModelChoice]]:
    """
    Discover available models (train / tuned) under a results root.

    This mirrors the Stage-1 discovery used elsewhere in the GUI:
    we call :func:`discover_stage1_runs`, which recursively searches
    for ``manifest.json`` files with ``stage == "stage1"`` and builds
    Stage1Summary objects. From each Stage-1 run, we then look for
    training / tuning workflows living under that Stage-1 directory.

    Parameters
    ----------
    results_root : Path
        Base directory (e.g. ~/.fusionlab_runs or a "tests/" folder).

    Returns
    -------
    models_by_city : dict
        Mapping "city" -> list of InferenceModelChoice instances.
    """
    models_by_city: Dict[str, List[InferenceModelChoice]] = {}

    root = Path(results_root).expanduser().resolve()
    if not root.exists():
        return models_by_city

    # Use the same Stage-1 discovery as training dialogs.
    # try:
    stage1_runs = discover_stage1_runs(root, current_cfg=None)
    # except Exception:
    #     stage1_runs = []

    for s in stage1_runs:
        city = s.city or "unknown"
        stage_dir = s.stage1_dir
        manifest_path = s.manifest_path

        if not stage_dir or not stage_dir.is_dir():
            continue
        if not manifest_path.exists():
            # Incomplete Stage-1; skip to avoid confusion
            continue

        # Accumulate entries per city (there *could* be multiple Stage-1
        # runs per city, although typically there is only one).
        city_models = models_by_city.setdefault(city, [])

        # ------------------------------------------------------------------
        # 1) Training runs under <stage_dir>/train_*/
        # ------------------------------------------------------------------
        for run_dir in sorted(stage_dir.glob("train_*")):
            if not run_dir.is_dir():
                continue

            keras_files = sorted(run_dir.glob("*.keras")) + sorted(
                run_dir.glob("*.h5")
            )
            if not keras_files:
                continue

            # Prefer a city-specific model name when present.
            model_path: Optional[Path] = None
            preferred_prefix = f"{city}_GeoPriorSubsNet"
            for kf in keras_files:
                if kf.name.startswith(preferred_prefix):
                    model_path = kf
                    break
            if model_path is None:
                model_path = keras_files[0]

            # Optional training summary + physics CSV
            summary_json: Optional[Path] = None
            phys_csv: Optional[Path] = None

            for cand in run_dir.glob("*.json"):
                if "training_summary" in cand.name:
                    summary_json = cand
                    break

            for cand in run_dir.glob("*.csv"):
                if "geopriorsubsnet_physical_parameters" in cand.name:
                    phys_csv = cand
                    break

            run_name = run_dir.name.replace("train_", "", 1)

            city_models.append(
                InferenceModelChoice(
                    city=city,
                    run_type="train",
                    run_name=run_name,
                    run_dir=run_dir,
                    model_path=model_path,
                    stage1_manifest_path=manifest_path,
                    training_summary_json=summary_json,
                    tuning_summary_json=None,
                    best_hps_json=None,
                    physical_params_csv=phys_csv,
                )
            )

        # ------------------------------------------------------------------
        # 2) Tuning runs under <stage_dir>/tuning/run_*/
        # ------------------------------------------------------------------
        tuning_root = stage_dir / "tuning"
        if tuning_root.is_dir():
            for tune_dir in sorted(tuning_root.glob("run_*")):
                if not tune_dir.is_dir():
                    continue

                keras_files = sorted(
                    tune_dir.glob(f"{city}_GeoPrior_best.keras")
                )
                if not keras_files:
                    keras_files = sorted(tune_dir.glob("*.keras"))
                if not keras_files:
                    continue

                model_path = keras_files[0]

                tuning_summary = tune_dir / "tuning_summary.json"
                best_hps_json = tune_dir / f"{city}_GeoPrior_best_hps.json"
                if not tuning_summary.exists():
                    tuning_summary = None
                if not best_hps_json.exists():
                    best_hps_json = None

                run_name = tune_dir.name.replace("run_", "", 1)

                city_models.append(
                    InferenceModelChoice(
                        city=city,
                        run_type="tune",
                        run_name=run_name,
                        run_dir=tune_dir,
                        model_path=model_path,
                        stage1_manifest_path=manifest_path,
                        training_summary_json=None,
                        tuning_summary_json=tuning_summary,
                        best_hps_json=best_hps_json,
                        physical_params_csv=None,
                    )
                )

    return models_by_city


# ----------------------------------------------------------------------
# Detail dialogs (training / tuning)
# ----------------------------------------------------------------------


def _add_metric_rows(
    form: QFormLayout,
    metrics: Dict[str, Any],
    keys: List[str],
    labels: Dict[str, str],
    highlight: Optional[List[str]] = None,
) -> None:
    """
    Convenience helper to add metrics rows to a QFormLayout.
    """
    highlight = highlight or []
    for key in keys:
        nice_label = labels.get(key, key)
        value = metrics.get(key, "NA")
        if isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)

        lbl_val = QLabel(text)
        if key in highlight:
            # Emphasise important diagnostics: coverage, sharpness,
            # epsilons, etc.
            lbl_val.setStyleSheet("font-weight:600; color:#0055aa;")
        form.addRow(f"{nice_label}:", lbl_val)


def _show_training_details(parent: QWidget, choice: InferenceModelChoice) -> None:
    """
    Show details for a training run: metrics_at_best + physical parameters.
    Handles older runs that do not have ``metrics_at_best`` by either
    falling back to ``final_epoch_metrics`` or showing a short message
    instead of an empty panel.
    """
    summary = _safe_load_json(choice.training_summary_json or Path())

    # Prefer metrics_at_best, then fall back to final_epoch_metrics.
    metrics_best = summary.get("metrics_at_best") or {}
    metrics_final = summary.get("final_epoch_metrics") or {}

    if isinstance(metrics_best, dict) and metrics_best:
        metrics = metrics_best
        metrics_title = "Metrics at best epoch"
    elif isinstance(metrics_final, dict) and metrics_final:
        metrics = metrics_final
        metrics_title = "Metrics at final epoch"
    else:
        # No detailed metrics stored for this run.
        metrics = {}
        metrics_title = None

    physics_enabled = _is_physics_enabled(summary)
    phys_params = (
        _safe_load_phys_params(choice.physical_params_csv or Path())
        if physics_enabled
        else {}
    )

    dlg = QDialog(parent)
    dlg.setWindowTitle(
        f"Training details — {choice.city} / {choice.run_name}"
    )
    dlg.setModal(True)
    dlg.setMinimumWidth(720)

    layout = QVBoxLayout(dlg)

    header = QLabel(
        f"<b>{choice.city}</b> — training run {choice.run_name}"
    )
    header.setTextFormat(Qt.RichText)
    layout.addWidget(header)

    run_dir_lbl = QLabel(f"Run directory: {choice.run_dir}")
    run_dir_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    run_dir_lbl.setWordWrap(True)
    layout.addWidget(run_dir_lbl)

    model_lbl = QLabel(f"Model: {choice.model_path}")
    model_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    model_lbl.setWordWrap(True)
    layout.addWidget(model_lbl)

    layout.addSpacing(8)

    # ------------------------------------------------------------------
    # Metrics block
    # ------------------------------------------------------------------
    if metrics_title and metrics:
        metrics_box = QGroupBox(metrics_title, dlg)
        grid = QGridLayout(metrics_box)

        # Column 1: data / reconstruction losses
        data_box = QGroupBox("Data losses", metrics_box)
        data_form = QFormLayout(data_box)

        data_labels = {
            "total_loss": "Total loss",
            "data_loss": "Data loss (subs + GWL)",
            "gw_flow_loss": "GW flow loss",
            "consolidation_loss": "Consolidation loss",
            "val_loss": "Validation loss (total)",
            "val_gwl_pred_loss": "Val GWL loss",
            "val_subs_pred_loss": "Val subsidence loss",
        }
        data_keys = [
            "total_loss",
            "data_loss",
            "gw_flow_loss",
            "consolidation_loss",
            "val_loss",
            "val_gwl_pred_loss",
            "val_subs_pred_loss",
        ]
        _add_metric_rows(data_form, metrics, data_keys, data_labels)
        grid.addWidget(data_box, 0, 0)

        # Column 2: physics-related losses (only if physics enabled)
        physics_box = QGroupBox("Physics losses", metrics_box)
        physics_form = QFormLayout(physics_box)

        phys_labels = {
            "physics_loss": "Physics loss (total)",
            "prior_loss": "Prior loss",
            "smooth_loss": "Smoothness loss",
            "mv_prior_loss": "mᵥ prior loss",
        }
        phys_keys = [
            "physics_loss",
            "prior_loss",
            "smooth_loss",
            "mv_prior_loss",
        ]

        if physics_enabled:
            _add_metric_rows(physics_form, metrics, phys_keys, phys_labels)
            grid.addWidget(physics_box, 0, 1)

        # Column 3: validation / coverage / epsilons
        val_box = QGroupBox("Validation & intervals", metrics_box)
        val_form = QFormLayout(val_box)

        val_labels = {
            "val_subs_pred_mae": "Val subsidence MAE (norm.)",
            "val_subs_pred_mse": "Val subsidence MSE (norm.)",
            "val_subs_pred_coverage80": "Coverage 80%",
            "val_subs_pred_sharpness80": "Sharpness 80% (norm.)",
            "epsilon_prior": "ε_prior (training)",
            "epsilon_cons": "ε_cons (training)",
            "val_epsilon_prior": "ε_prior (validation)",
            "val_epsilon_cons": "ε_cons (validation)",
        }

        val_keys = [
            "val_subs_pred_mae",
            "val_subs_pred_mse",
            "val_subs_pred_coverage80",
            "val_subs_pred_sharpness80",
        ]
        if physics_enabled:
            val_keys.extend(
                [
                    "epsilon_prior",
                    "epsilon_cons",
                    "val_epsilon_prior",
                    "val_epsilon_cons",
                ]
            )

        highlight_keys = [
            "val_subs_pred_coverage80",
            "val_subs_pred_sharpness80",
            "val_epsilon_prior",
            "val_epsilon_cons",
        ]
        _add_metric_rows(
            val_form,
            metrics,
            val_keys,
            val_labels,
            highlight=highlight_keys,
        )
        grid.addWidget(val_box, 0, 2)

        layout.addWidget(metrics_box)
    else:
        # No metrics dict at all → small, non-overlapping message.
        msg = QLabel(
            "No detailed training metrics are available for this run.\n\n"
            "This can happen if the model was trained with an older GeoPrior "
            "version that did not record per-epoch metrics, or if the training "
            "workflow was interrupted before metrics could be written. The "
            "trained model may still be usable; only this summary panel is empty."
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)

    # ------------------------------------------------------------------
    # Physical parameters (unchanged)
    # ------------------------------------------------------------------
    if physics_enabled and phys_params:
        phys_box = QGroupBox("Physical parameters", dlg)
        phys_form = QFormLayout(phys_box)

        mv = phys_params.get("Compressibility_mv")
        kappa = phys_params.get("Consistency_Kappa")
        gamma_w = phys_params.get("Unit_Weight_Water_gamma_w")
        h_ref = phys_params.get("Reference_Head_h_ref")

        if mv is not None:
            phys_form.addRow(
                "mᵥ (compressibility, 1/Pa):", QLabel(str(mv))
            )
        if kappa is not None:
            phys_form.addRow("κ (consistency factor):", QLabel(str(kappa)))
        if gamma_w is not None:
            phys_form.addRow(
                "γ_w (unit weight of water, N·m⁻³):",
                QLabel(str(gamma_w)),
            )
        if h_ref is not None:
            phys_form.addRow("h_ref (reference head, m):", QLabel(str(h_ref)))

        layout.addWidget(phys_box)

    # Buttons
    btn_box = QDialogButtonBox(QDialogButtonBox.Close, parent=dlg)
    btn_box.rejected.connect(dlg.reject)
    layout.addWidget(btn_box)

    dlg.exec_()


# def _show_training_details(parent: QWidget, choice: InferenceModelChoice) -> None:
#     """
#     Show details for a training run: metrics_at_best + physical parameters.
#     """
#     summary = _safe_load_json(choice.training_summary_json or Path())
#     metrics = summary.get("metrics_at_best", summary) or {}

#     physics_enabled = _is_physics_enabled(summary)
#     phys_params = (
#         _safe_load_phys_params(choice.physical_params_csv or Path())
#         if physics_enabled
#         else {}
#     )

#     dlg = QDialog(parent)
#     dlg.setWindowTitle(
#         f"Training details — {choice.city} / {choice.run_name}"
#     )
#     dlg.setModal(True)
#     dlg.setMinimumWidth(720)

#     layout = QVBoxLayout(dlg)

#     header = QLabel(
#         f"<b>{choice.city}</b> — training run {choice.run_name}"
#     )
#     header.setTextFormat(Qt.RichText)
#     layout.addWidget(header)

#     run_dir_lbl = QLabel(f"Run directory: {choice.run_dir}")
#     run_dir_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
#     run_dir_lbl.setWordWrap(True)
#     layout.addWidget(run_dir_lbl)

#     model_lbl = QLabel(f"Model: {choice.model_path}")
#     model_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
#     model_lbl.setWordWrap(True)
#     layout.addWidget(model_lbl)

#     layout.addSpacing(8)

#     # --- Metrics block with 3 columns ---------------------------------
#     metrics_box = QGroupBox("Metrics at best epoch", dlg)
#     grid = QGridLayout(metrics_box)

#     # Column 1: data / reconstruction losses
#     data_box = QGroupBox("Data losses", metrics_box)
#     data_form = QFormLayout(data_box)

#     data_labels = {
#         "total_loss": "Total loss",
#         "data_loss": "Data loss (subs + GWL)",
#         "gw_flow_loss": "GW flow loss",
#         "consolidation_loss": "Consolidation loss",
#         "val_loss": "Validation loss (total)",
#         "val_gwl_pred_loss": "Val GWL loss",
#         "val_subs_pred_loss": "Val subsidence loss",
#     }
#     data_keys = [
#         "total_loss",
#         "data_loss",
#         "gw_flow_loss",
#         "consolidation_loss",
#         "val_loss",
#         "val_gwl_pred_loss",
#         "val_subs_pred_loss",
#     ]
#     _add_metric_rows(data_form, metrics, data_keys, data_labels)
#     grid.addWidget(data_box, 0, 0)

#     # Column 2: physics-related losses (shown only if physics enabled)
#     physics_box = QGroupBox("Physics losses", metrics_box)
#     physics_form = QFormLayout(physics_box)

#     phys_labels = {
#         "physics_loss": "Physics loss (total)",
#         "prior_loss": "Prior loss",
#         "smooth_loss": "Smoothness loss",
#         "mv_prior_loss": "mᵥ prior loss",
#     }
#     phys_keys = [
#         "physics_loss",
#         "prior_loss",
#         "smooth_loss",
#         "mv_prior_loss",
#     ]

#     if physics_enabled:
#         _add_metric_rows(physics_form, metrics, phys_keys, phys_labels)
#         grid.addWidget(physics_box, 0, 1)

#     # Column 3: validation / coverage / epsilons
#     val_box = QGroupBox("Validation & intervals", metrics_box)
#     val_form = QFormLayout(val_box)

#     val_labels = {
#         "val_subs_pred_mae": "Val subsidence MAE (norm.)",
#         "val_subs_pred_mse": "Val subsidence MSE (norm.)",
#         "val_subs_pred_coverage80": "Coverage 80%",
#         "val_subs_pred_sharpness80": "Sharpness 80% (norm.)",
#         "epsilon_prior": "ε_prior (training)",
#         "epsilon_cons": "ε_cons (training)",
#         "val_epsilon_prior": "ε_prior (validation)",
#         "val_epsilon_cons": "ε_cons (validation)",
#     }

#     # Always show the pure data/interval metrics
#     val_keys = [
#         "val_subs_pred_mae",
#         "val_subs_pred_mse",
#         "val_subs_pred_coverage80",
#         "val_subs_pred_sharpness80",
#     ]

#     # Only show epsilons when physics is actually on
#     if physics_enabled:
#         val_keys.extend(
#             [
#                 "epsilon_prior",
#                 "epsilon_cons",
#                 "val_epsilon_prior",
#                 "val_epsilon_cons",
#             ]
#         )

#     highlight_keys = [
#         "val_subs_pred_coverage80",
#         "val_subs_pred_sharpness80",
#         "val_epsilon_prior",
#         "val_epsilon_cons",
#     ]
#     _add_metric_rows(
#         val_form,
#         metrics,
#         val_keys,
#         val_labels,
#         highlight=highlight_keys,
#     )
#     grid.addWidget(val_box, 0, 2)

#     layout.addWidget(metrics_box)

#     # --- Physical parameters -------------------------------------------
#     if physics_enabled and phys_params:
#         phys_box = QGroupBox("Physical parameters", dlg)
#         phys_form = QFormLayout(phys_box)

#         mv = phys_params.get("Compressibility_mv")
#         kappa = phys_params.get("Consistency_Kappa")
#         gamma_w = phys_params.get("Unit_Weight_Water_gamma_w")
#         h_ref = phys_params.get("Reference_Head_h_ref")

#         if mv is not None:
#             phys_form.addRow(
#                 "mᵥ (compressibility, 1/Pa):", QLabel(str(mv))
#             )
#         if kappa is not None:
#             phys_form.addRow("κ (consistency factor):", QLabel(str(kappa)))
#         if gamma_w is not None:
#             phys_form.addRow(
#                 "γ_w (unit weight of water, N·m⁻³):",
#                 QLabel(str(gamma_w)),
#             )
#         if h_ref is not None:
#             phys_form.addRow("h_ref (reference head, m):", QLabel(str(h_ref)))

#         layout.addWidget(phys_box)

#     # Buttons
#     btn_box = QDialogButtonBox(QDialogButtonBox.Close, parent=dlg)
#     btn_box.rejected.connect(dlg.reject)
#     layout.addWidget(btn_box)

#     dlg.exec_()
def _show_tuning_details(parent: QWidget, choice: InferenceModelChoice) -> None:
    """
    Show details for a tuning run: tuning_summary + best_hps.
    """
    tuning_summary = _safe_load_json(
        choice.tuning_summary_json or Path()
    )
    best_hps = tuning_summary.get("best_hps", {}) or _safe_load_json(
        choice.best_hps_json or Path()
    )

    dlg = QDialog(parent)
    dlg.setWindowTitle(
        f"Tuning details — {choice.city} / {choice.run_name}"
    )
    dlg.setModal(True)
    dlg.setMinimumWidth(720)

    layout = QVBoxLayout(dlg)

    header = QLabel(
        f"<b>{choice.city}</b> — tuned run {choice.run_name}"
    )
    header.setTextFormat(Qt.RichText)
    layout.addWidget(header)

    run_dir_lbl = QLabel(f"Tuning directory: {choice.run_dir}")
    run_dir_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    run_dir_lbl.setWordWrap(True)
    layout.addWidget(run_dir_lbl)

    model_lbl = QLabel(f"Best model: {choice.model_path}")
    model_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    model_lbl.setWordWrap(True)
    layout.addWidget(model_lbl)

    layout.addSpacing(8)

    # Basic meta from tuning_summary
    meta_box = QGroupBox("Tuning summary", dlg)
    meta_form = QFormLayout(meta_box)

    if tuning_summary:
        meta_form.addRow(
            "Timestamp:",
            QLabel(str(tuning_summary.get("timestamp", ""))),
        )
        meta_form.addRow(
            "City:",
            QLabel(str(tuning_summary.get("city", ""))),
        )
        meta_form.addRow(
            "Model:",
            QLabel(str(tuning_summary.get("model", ""))),
        )
        meta_form.addRow(
            "Mode:",
            QLabel(str(tuning_summary.get("mode", ""))),
        )
        if "epochs" in tuning_summary:
            meta_form.addRow(
                "Epochs:",
                QLabel(str(tuning_summary.get("epochs"))),
            )
        if "batch_size" in tuning_summary:
            meta_form.addRow(
                "Batch size:",
                QLabel(str(tuning_summary.get("batch_size"))),
            )

    layout.addWidget(meta_box)

    if not best_hps:
        btn_box = QDialogButtonBox(QDialogButtonBox.Close, parent=dlg)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)
        dlg.exec_()
        return

    # HP blocks: architecture / physics / λ-weights
    hp_box = QGroupBox("Best hyperparameters", dlg)
    hp_grid = QGridLayout(hp_box)

    # Architecture
    arch_box = QGroupBox("Architecture", hp_box)
    arch_form = QFormLayout(arch_box)

    def _add_hp(form: QFormLayout, label: str, key: str) -> None:
        if key not in best_hps:
            return
        form.addRow(label + ":", QLabel(str(best_hps.get(key))))

    _add_hp(arch_form, "Embedding dim (dₑ)", "embed_dim")
    _add_hp(arch_form, "Hidden units", "hidden_units")
    _add_hp(arch_form, "LSTM units", "lstm_units")
    _add_hp(arch_form, "Attention units", "attention_units")
    _add_hp(arch_form, "Number of heads", "num_heads")
    _add_hp(arch_form, "Dropout rate", "dropout_rate")
    _add_hp(arch_form, "Use VSN", "use_vsn")
    _add_hp(arch_form, "VSN units", "vsn_units")
    _add_hp(arch_form, "Use batch norm", "use_batch_norm")
    hp_grid.addWidget(arch_box, 0, 0)

    # Physics settings
    phys_box = QGroupBox("Physics configuration", hp_box)
    phys_form = QFormLayout(phys_box)

    _add_hp(phys_form, "PDE mode", "pde_mode")
    _add_hp(phys_form, "Scale PDE residuals", "scale_pde_residuals")
    _add_hp(phys_form, "κ mode", "kappa_mode")
    _add_hp(phys_form, "h_d factor", "hd_factor")
    _add_hp(phys_form, "mᵥ (compressibility)", "mv")
    _add_hp(phys_form, "κ (consistency)", "kappa")
    _add_hp(phys_form, "Learning rate", "learning_rate")
    hp_grid.addWidget(phys_box, 0, 1)

    # λ-weights
    lam_box = QGroupBox("Loss weights (λ)", hp_box)
    lam_form = QFormLayout(lam_box)

    lambda_mapping = [
        ("λ (Groundwater flow)", "lambda_gw"),
        ("λ (Consolidation)", "lambda_cons"),
        ("λ (Prior)", "lambda_prior"),
        ("λ (Smoothness)", "lambda_smooth"),
        ("λ (mᵥ prior)", "lambda_mv"),
        ("mᵥ LR multiplier", "mv_lr_mult"),
        ("κ LR multiplier", "kappa_lr_mult"),
    ]
    for label, key in lambda_mapping:
        if key in best_hps:
            lam_form.addRow(label + ":", QLabel(str(best_hps.get(key))))

    hp_grid.addWidget(lam_box, 0, 2)

    layout.addWidget(hp_box)

    # Buttons
    btn_box = QDialogButtonBox(QDialogButtonBox.Close, parent=dlg)
    btn_box.rejected.connect(dlg.reject)
    layout.addWidget(btn_box)

    dlg.exec_()

# ----------------------------------------------------------------------
# Main advanced dialog for Inference tab
# ----------------------------------------------------------------------

class InferenceOptionsDialog(QDialog):
    """
    Advanced model selection for the Inference tab.

    This dialog lets the user:
    - Choose a results root (base directory for Stage-1 / Stage-2 runs).
    - Browse cities discovered under that root.
    - For each city, inspect trained and tuned models.
    - Select one model so that the Inference tab can auto-fill:
        * Model file path (.keras)
        * Stage-1 manifest.json for that city
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        results_root: Path,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Inference – Advanced model selection")
        self.setModal(True)
        # self.setMinimumSize(900, 520)
        self.setMinimumSize(0, 0)  # let layouts decide; we’ll adjust later

        self._default_root = Path(results_root).expanduser().resolve()
        self._results_root = Path(self._default_root)
        self._models_by_city: Dict[str, List[InferenceModelChoice]] = {}
        self._selected_entry: Optional[InferenceModelChoice] = None
        self._chosen_entry: Optional[InferenceModelChoice] = None

        main = QVBoxLayout(self)
        
        # Label used when no models are found under the current root.
        self.info_label = QLabel(self)
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.info_label.hide()  # only visible in "no models" state
        main.addWidget(self.info_label)

        # --- Root selector ------------------------------------------------
        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Results root:"))
        self.root_edit = QLineEdit(str(self._results_root))
        self.root_edit.setReadOnly(True)
        root_row.addWidget(self.root_edit, stretch=1)

        self.btn_browse_root = QPushButton("Browse…")
        self.btn_reset_root = QPushButton("Reset")
        root_row.addWidget(self.btn_browse_root)
        root_row.addWidget(self.btn_reset_root)
        main.addLayout(root_row)

        # --- Central split: cities list + tabs (train/tune) ---------------
        central = QHBoxLayout()

        # Left: cities list
        self.city_list = QListWidget(self)
        self.city_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.city_list.currentItemChanged.connect(self._on_city_changed)
        self.city_list.setMinimumWidth(220)
        central.addWidget(self.city_list)

        # Right: tabs
        self.tabs = QTabWidget(self)

        # Trained models tab
        self.train_table = QTableWidget(self)
        self.train_table.setColumnCount(3)
        self.train_table.setHorizontalHeaderLabels(
            ["Run", "Model", "Notes"]
        )
        self.train_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.train_table.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self.train_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.train_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.train_table.cellClicked.connect(
            self._on_train_row_selected
        )
        self.tabs.addTab(self.train_table, "Trained models")

        # Tuned models tab
        self.tune_table = QTableWidget(self)
        self.tune_table.setColumnCount(3)
        self.tune_table.setHorizontalHeaderLabels(
            ["Run", "Model", "Notes"]
        )
        self.tune_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.tune_table.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self.tune_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
        self.tune_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.tune_table.cellClicked.connect(
            self._on_tune_row_selected
        )
        self.tabs.addTab(self.tune_table, "Tuned models")

        central.addWidget(self.tabs, stretch=1)
        main.addLayout(central)

        # --- Bottom row: details + use / cancel ---------------------------
        bottom_row = QHBoxLayout()
        self.btn_details = QPushButton("Details…")
        self.btn_details.setEnabled(False)
        bottom_row.addWidget(self.btn_details)

        bottom_row.addStretch(1)

        btn_box = QDialogButtonBox(self)
        self.btn_use = btn_box.addButton(
            "Use selected model", QDialogButtonBox.AcceptRole
        )
        self.btn_use.setEnabled(False)
        self.btn_cancel = btn_box.addButton(
            QDialogButtonBox.Cancel
        )

        bottom_row.addWidget(btn_box)
        main.addLayout(bottom_row)

        # Connections
        self.btn_browse_root.clicked.connect(self._on_browse_root)
        self.btn_reset_root.clicked.connect(self._on_reset_root)
        self.btn_details.clicked.connect(self._on_show_details)
        self.btn_use.clicked.connect(self._on_accept)
        self.btn_cancel.clicked.connect(self.reject)

        # Initial scan
        self._refresh_models()

    def _update_dialog_size(self, has_models: bool) -> None:
        """
        Compact dialog when no models are found; larger when showing tables.
        """
        if has_models:
            # big enough for tables
            min_w, min_h = 900, 520
            self.setMinimumSize(min_w, min_h)
            # don't shrink if user already resized bigger
            w = max(self.width(), min_w)
            h = max(self.height(), min_h)
            self.resize(w, h)
        else:
            # small, nice dialog for "no models" state
            self.setMinimumSize(0, 0)
            # some reasonable compact size; adjust as you like
            self.resize(480, 220)
            self.adjustSize()

    # ------------------------------------------------------------------
    # Properties & API
    # ------------------------------------------------------------------

    @property
    def results_root(self) -> Path:
        return self._results_root

    @property
    def chosen_entry(self) -> Optional[InferenceModelChoice]:
        return self._chosen_entry

    # ------------------------------------------------------------------
    # Root handling
    # ------------------------------------------------------------------

    def _on_browse_root(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select results root",
            str(self._results_root),
        )
        if not path:
            return

        self._results_root = Path(path).expanduser().resolve()
        self.root_edit.setText(str(self._results_root))
        self._refresh_models()

    def _on_reset_root(self) -> None:
        self._results_root = Path(self._default_root)
        self.root_edit.setText(str(self._results_root))
        self._refresh_models()

    # ------------------------------------------------------------------
    # Discovery & city list
    # ------------------------------------------------------------------
    def _refresh_models(self) -> None:
        self._models_by_city = _discover_inference_models(self._results_root)
    
        self.city_list.clear()
        self._selected_entry = None
        self._chosen_entry = None
        self.btn_details.setEnabled(False)
        self.btn_use.setEnabled(False)
        self.train_table.setRowCount(0)
        self.tune_table.setRowCount(0)
    
        has_models = bool(self._models_by_city)
    
        # Central widgets + bottom buttons only make sense when models exist.
        self.city_list.setVisible(has_models)
        self.tabs.setVisible(has_models)
        self.btn_details.setVisible(has_models)
        self.btn_use.setVisible(has_models)
        self.btn_reset_root.setVisible(has_models)

        if not has_models:
            # Compact in-dialog message + root + Browse/Reset/Cancel.
            self.info_label.setText(
                "No trained or tuned GeoPrior models were found under "
                f"the selected root:\n\n{self._results_root}\n\n"
                "You can browse to another results root or close this dialog."
            )
            self.info_label.show()
            self._update_dialog_size(False)
            return
    
        # If we get here, models exist → hide message and populate tables.
        self.info_label.hide()
    
        for city in sorted(self._models_by_city.keys()):
            entries = self._models_by_city[city]
            n_train = sum(e.run_type == "train" for e in entries)
            n_tune = sum(e.run_type == "tune" for e in entries)
    
            text = f"{city}  (train: {n_train}, tuned: {n_tune})"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, city)
            self.city_list.addItem(item)
    
        if self.city_list.count() > 0:
            self.city_list.setCurrentRow(0)

        # now we’re in “tables” mode → big dialog
        self._update_dialog_size(True)
    
    def _on_city_changed(
        self,
        current: QListWidgetItem,
        _previous: QListWidgetItem,
    ) -> None:
        if not current:
            return
        city = str(current.data(Qt.UserRole))
        self._populate_tables_for_city(city)

    def _populate_tables_for_city(self, city: str) -> None:
        entries = self._models_by_city.get(city, [])

        train_entries = [e for e in entries if e.run_type == "train"]
        tune_entries = [e for e in entries if e.run_type == "tune"]

        # Trained table
        self.train_table.setRowCount(len(train_entries))
        for row, e in enumerate(train_entries):
            run_item = QTableWidgetItem(e.run_name)
            model_item = QTableWidgetItem(e.model_path.name)
            notes = []
            if e.training_summary_json and e.training_summary_json.exists():
                notes.append("summary")
            if e.physical_params_csv and e.physical_params_csv.exists():
                notes.append("phys.")
            notes_item = QTableWidgetItem(", ".join(notes) or "-")

            # Attach the entry
            run_item.setData(Qt.UserRole, e)

            self.train_table.setItem(row, 0, run_item)
            self.train_table.setItem(row, 1, model_item)
            self.train_table.setItem(row, 2, notes_item)

        # Tuned table
        self.tune_table.setRowCount(len(tune_entries))
        for row, e in enumerate(tune_entries):
            run_item = QTableWidgetItem(e.run_name)
            model_item = QTableWidgetItem(e.model_path.name)
            notes = []
            if e.tuning_summary_json and e.tuning_summary_json.exists():
                notes.append("tuning_summary")
            if e.best_hps_json and e.best_hps_json.exists():
                notes.append("best_hps")
            notes_item = QTableWidgetItem(", ".join(notes) or "-")

            run_item.setData(Qt.UserRole, e)

            self.tune_table.setItem(row, 0, run_item)
            self.tune_table.setItem(row, 1, model_item)
            self.tune_table.setItem(row, 2, notes_item)

        # Clear selection and buttons
        self._selected_entry = None
        self.btn_details.setEnabled(False)
        self.btn_use.setEnabled(False)

    # ------------------------------------------------------------------
    # Selection handlers
    # ------------------------------------------------------------------

    def _on_train_row_selected(self, row: int, _col: int) -> None:
        item = self.train_table.item(row, 0)
        if not item:
            return
        entry = item.data(Qt.UserRole)
        if not isinstance(entry, InferenceModelChoice):
            return
        self._selected_entry = entry
        self.btn_details.setEnabled(True)
        self.btn_use.setEnabled(True)
        # Clear tuned-table selection to avoid ambiguity
        self.tune_table.clearSelection()

    def _on_tune_row_selected(self, row: int, _col: int) -> None:
        item = self.tune_table.item(row, 0)
        if not item:
            return
        entry = item.data(Qt.UserRole)
        if not isinstance(entry, InferenceModelChoice):
            return
        self._selected_entry = entry
        self.btn_details.setEnabled(True)
        self.btn_use.setEnabled(True)
        # Clear train-table selection to avoid ambiguity
        self.train_table.clearSelection()

    def _on_show_details(self) -> None:
        if not self._selected_entry:
            return

        if self._selected_entry.run_type == "train":
            _show_training_details(self, self._selected_entry)
        else:
            _show_tuning_details(self, self._selected_entry)

    def _on_accept(self) -> None:
        if not self._selected_entry:
            QMessageBox.warning(
                self,
                "No model selected",
                "Please select a model from the table first.",
            )
            return

        self._chosen_entry = self._selected_entry
        self.accept()

    # ------------------------------------------------------------------
    # Static / class API
    # ------------------------------------------------------------------
    @classmethod
    def run(
        cls,
        parent: QWidget,
        *,
        geo_cfg: GeoPriorConfig,
        results_root: Path,
    ) -> Tuple[bool, Path, Optional[InferenceModelChoice]]:
        """
        Open the dialog and return the user's selection.

        Parameters
        ----------
        parent : QWidget
            Parent widget (typically the main GeoPriorForecaster).
        geo_cfg : GeoPriorConfig
            Current GeoPrior configuration. Accepted for API symmetry
            with TrainOptionsDialog.run and kept for future use
            (e.g. filtering models by configuration). Currently not
            used inside the dialog.
        results_root : Path
            Initial results root to scan (e.g. ~/.fusionlab_runs or
            tests/).

        Returns
        -------
        accepted : bool
            True if the user clicked "Use selected model", otherwise
            False.
        new_root : Path
            Possibly updated results root (if user changed it).
        choice : InferenceModelChoice or None
            Selected model information, or None if cancelled.
        """
        dlg = cls(
            parent=parent,
            results_root=results_root,
        )
        accepted = dlg.exec_() == QDialog.Accepted
        return accepted, dlg.results_root, dlg.chosen_entry

