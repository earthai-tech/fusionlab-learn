# geoprior/ui/file_browse.py
# -*- coding: utf-8 -*-
# License: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtWidgets import QWidget, QFileDialog, QLineEdit


class FileBrowseHelper:
    """
    Small helper to centralise 'browse...' dialogs.

    Parameters
    ----------
    parent : QWidget
        Parent widget for the QFileDialog.
    root_getter : callable
        Function returning the current results root as a Path.
        Typically something like ``lambda: self.gui_runs_root``.
    """

    def __init__(
        self,
        parent: QWidget,
        root_getter: Callable[[], Path],
    ) -> None:
        self._parent = parent
        self._root_getter = root_getter

    # ------------------------------------------------------------------
    # Core primitive
    # ------------------------------------------------------------------
    def _browse_to_line_edit(
        self,
        *,
        title: str,
        filters: str,
        line_edit: QLineEdit,
        start_dir: Optional[Path] = None,
    ) -> None:
        """
        Generic 'open file and push path into line_edit' helper.
        """
        root = start_dir or self._root_getter() or Path(".")
        path, _ = QFileDialog.getOpenFileName(
            self._parent,
            title,
            str(root),
            filters,
        )
        if path:
            line_edit.setText(path)

    # ------------------------------------------------------------------
    # Specialised helpers for the Inference tab
    # ------------------------------------------------------------------
    def browse_model(self, line_edit: QLineEdit) -> None:
        self._browse_to_line_edit(
            title="Select trained/tuned model",
            filters="Keras models (*.keras *.h5);;All files (*)",
            line_edit=line_edit,
        )

    def browse_manifest(self, line_edit: QLineEdit) -> None:
        self._browse_to_line_edit(
            title="Select Stage-1 manifest.json",
            filters="JSON files (*.json);;All files (*)",
            line_edit=line_edit,
        )

    def browse_inputs_npz(self, line_edit: QLineEdit) -> None:
        self._browse_to_line_edit(
            title="Select custom inputs NPZ",
            filters="NumPy archives (*.npz);;All files (*)",
            line_edit=line_edit,
        )

    def browse_targets_npz(self, line_edit: QLineEdit) -> None:
        self._browse_to_line_edit(
            title="Select custom targets NPZ (optional)",
            filters="NumPy archives (*.npz);;All files (*)",
            line_edit=line_edit,
        )

    def browse_calibrator(self, line_edit: QLineEdit) -> None:
        self._browse_to_line_edit(
            title="Select calibrator .npy (optional)",
            filters="NumPy arrays (*.npy);;All files (*)",
            line_edit=line_edit,
        )

