# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Qt threads wrapping GeoPrior AppJob classes.

Each thread owns one AppJob from backend.py and exposes
Qt signals for logging, status, progress and results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from .backend import (
    AppJob,
    Stage1Job,
    TrainingJob,
    TuningJob,
    InferenceJob,
    XferMatrixJob,
    XferViewJob,
)


class BaseJobThread(QThread):
    """Generic QThread running a single AppJob."""

    log_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    progress_changed = pyqtSignal(float, str)
    error_occurred = pyqtSignal(str)
    results_ready = pyqtSignal(dict)

    def __init__(
        self,
        job: AppJob,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self._job = job

        # Rewire job callbacks to emit Qt signals.
        self._job._logger = self._log
        self._job._stop_check = self._should_stop
        self._job._progress_hook = self._progress

    # ---- callbacks wired into AppJob ---------------------------------
    def _log(self, msg: str) -> None:
        self.log_updated.emit(msg)

    def _should_stop(self) -> bool:
        return self.isInterruptionRequested()

    def _progress(
        self,
        value: float,
        message: Optional[str] = None,
    ) -> None:
        self.progress_changed.emit(value, message or "")

    # ---- main run ----------------------------------------------------
    def run(self) -> None:
        try:
            self.status_updated.emit("Running …")
            result = self._job.run() or {}
            self.results_ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))


class Stage1Thread(BaseJobThread):
    """Thread wrapper around Stage1Job."""

    def __init__(
        self,
        city: str,
        cfg_overrides: Optional[
            Dict[str, Any]
        ] = None,
        *,
        clean_run_dir: bool = True,
        parent: Optional[object] = None,
    ) -> None:
        job = Stage1Job(
            city=city,
            cfg_overrides=cfg_overrides,
            clean_run_dir=clean_run_dir,
        )
        super().__init__(job=job, parent=parent)


class TrainingThread(BaseJobThread):
    """Thread wrapper around TrainingJob."""

    training_finished = pyqtSignal(dict)

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        cfg_overrides: Optional[
            Dict[str, Any]
        ] = None,
        *,
        evaluate_training: bool = True,
        parent: Optional[object] = None,
    ) -> None:
        job = TrainingJob(
            manifest_path=manifest_path,
            cfg_overrides=cfg_overrides,
            evaluate_training=evaluate_training,
        )
        super().__init__(job=job, parent=parent)

    def run(self) -> None:
        super().run()
        result = self._job.last_result or {}
        self.training_finished.emit(result)


class TuningThread(BaseJobThread):
    """Thread wrapper around TuningJob."""

    tuning_finished = pyqtSignal(dict)

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        cfg_overrides: Optional[
            Dict[str, Any]
        ] = None,
        *,
        evaluate_tuned: bool = False,
        parent: Optional[object] = None,
    ) -> None:
        job = TuningJob(
            manifest_path=manifest_path,
            cfg_overrides=cfg_overrides,
            evaluate_tuned=evaluate_tuned,
        )
        super().__init__(job=job, parent=parent)

    def run(self) -> None:
        super().run()
        result = self._job.last_result or {}
        self.tuning_finished.emit(result)


class InferenceThread(BaseJobThread):
    """Thread wrapper around InferenceJob."""

    inference_finished = pyqtSignal(dict)

    def __init__(
        self,
        model_path: str,
        dataset: str = "test",
        *,
        use_stage1_future_npz: bool = False,          
        manifest_path: Optional[str] = None,
        stage1_dir: Optional[str] = None,
        inputs_npz: Optional[str] = None,
        targets_npz: Optional[str] = None,
        use_source_calibrator: bool = False,
        calibrator_path: Optional[str] = None,
        fit_calibrator: bool = False,
        cov_target: float = 0.80,
        include_gwl: bool = False,
        batch_size: int = 32,
        make_plots: bool = True,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        parent: Optional[object] = None,
    ) -> None:
        job = InferenceJob(
            model_path=model_path,
            dataset=dataset,
            use_stage1_future_npz=use_stage1_future_npz,  
            manifest_path=manifest_path,
            stage1_dir=stage1_dir,
            inputs_npz=inputs_npz,
            targets_npz=targets_npz,
            use_source_calibrator=use_source_calibrator,
            calibrator_path=calibrator_path,
            fit_calibrator=fit_calibrator,
            cov_target=cov_target,
            include_gwl=include_gwl,
            batch_size=batch_size,
            make_plots=make_plots,
            cfg_overrides=cfg_overrides,
        )
        super().__init__(job=job, parent=parent)


    def run(self) -> None:
        super().run()
        result = self._job.last_result or {}
        self.inference_finished.emit(result)


class XferMatrixThread(BaseJobThread):
    """Thread wrapper around XferMatrixJob."""

    xfer_finished = pyqtSignal(dict)

    def __init__(
        self,
        city_a: str,
        city_b: str,
        *,
        results_dir: str = "results",
        splits: Optional[
            Any
        ] = None,
        calib_modes: Optional[
            Any
        ] = None,
        rescale_to_source: bool = False,
        batch_size: int = 32,
        quantiles_override: Optional[
            Any
        ] = None,
        out_dir: Optional[str] = None,
        write_json: bool = True,
        write_csv: bool = True,
        parent: Optional[object] = None,
    ) -> None:
        if splits is None:
            splits = ("val", "test")
        if calib_modes is None:
            calib_modes = (
                "none",
                "source",
                "target",
            )

        job = XferMatrixJob(
            city_a=city_a,
            city_b=city_b,
            results_dir=results_dir,
            splits=splits,
            calib_modes=calib_modes,
            rescale_to_source=rescale_to_source,
            batch_size=batch_size,
            quantiles_override=quantiles_override,
            out_dir=out_dir,
            write_json=write_json,
            write_csv=write_csv,
        )
        super().__init__(job=job, parent=parent)

    def run(self) -> None:
        super().run()
        result = self._job.last_result or {}
        self.xfer_finished.emit(result)


class XferViewThread(BaseJobThread):
    """Thread wrapper around XferViewJob."""

    xfer_view_finished = pyqtSignal(dict)

    def __init__(
        self,
        *,
        view_kind: str,
        results_root: str,
        xfer_out_dir: Optional[str] = None,
        xfer_csv: Optional[str] = None,
        xfer_json: Optional[str] = None,
        split: str = "val",
        prefer_split: Optional[str] = None,
        prefer_calibration: Optional[str] = None,
        show_overall: bool = True,
        dpi: int = 150,
        fontsize: int = 8,
        parent: Optional[object] = None,
    ) -> None:
        job = XferViewJob(
            view_kind=view_kind,
            results_root=results_root,
            xfer_out_dir=xfer_out_dir,
            xfer_csv=xfer_csv,
            xfer_json=xfer_json,
            split=split,
            prefer_split=prefer_split,
            prefer_calibration=prefer_calibration,
            show_overall=show_overall,
            dpi=dpi,
            fontsize=fontsize,
            logger=None,
            stop_check=None,
        )
        super().__init__(job=job, parent=parent)

    def run(self) -> None:
        super().run()
        result = self._job.last_result or {}
        self.xfer_view_finished.emit(result)


__all__ = [
    "BaseJobThread",
    "Stage1Thread",
    "TrainingThread",
    "TuningThread",
    "InferenceThread",
    "XferMatrixThread",
    "XferViewThread"
]

