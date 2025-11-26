# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Backend job wrappers for GeoPrior GUI.

Each job is a small, stateful wrapper around the
run_* helpers in this package. They accept logger,
stop_check and an optional progress_hook, so the
GUI can stream logs and later wire progress bars.
"""

from __future__ import annotations

import os 
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
)

from .run_stage1 import run_stage1
from .run_training import run_training
from .run_tuning import run_tuning
from .run_inference import run_inference
from .run_xfer_matrix import run_xfer_matrix
from .xfer_view import (
    latest_xfer_csv,
    latest_xfer_json,
    make_transferability_panel_from_csv,
    make_cross_transfer_from_json,
)


LogFn = Callable[[str], None]
StopCheckFn = Callable[[], bool]
ProgressHook = Callable[[float, Optional[str]], None]


class AppJob(ABC):
    """Small base class for long-running GUI jobs."""

    def __init__(
        self,
        logger: Optional[LogFn] = None,
        stop_check: Optional[StopCheckFn] = None,
        progress_hook: Optional[ProgressHook] = None,
    ) -> None:
        self._logger: LogFn = logger or print
        self._stop_check: Optional[StopCheckFn] = stop_check
        self._progress_hook: Optional[
            ProgressHook
        ] = progress_hook
        self.last_result: Optional[
            Dict[str, Any]
        ] = None

    def log(self, msg: str) -> None:
        """Emit a log message to the GUI."""
        self._logger(msg)

    def should_stop(self) -> bool:
        """Return True when the GUI asked to abort."""
        if self._stop_check is None:
            return False
        return bool(self._stop_check())

    def update_progress(
        self,
        value: float,
        message: Optional[str] = None,
    ) -> None:
        """Notify GUI about progress in [0, 1]."""
        if self._progress_hook is None:
            return
        self._progress_hook(value, message)

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the job and return a result dict."""
        raise NotImplementedError


class Stage1Job(AppJob):
    """Run Stage-1 preprocessing as a job."""

    def __init__(
        self,
        city: str,
        cfg_overrides: Optional[
            Dict[str, Any]
        ] = None,
        *,
        clean_run_dir: bool = True,
        logger: Optional[LogFn] = None,
        stop_check: Optional[
            StopCheckFn
        ] = None,
        progress_hook: Optional[
            ProgressHook
        ] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            stop_check=stop_check,
            progress_hook=progress_hook,
        )
        self.city = city
        self.clean_run_dir = clean_run_dir

        overrides: Dict[str, Any] = dict(
            cfg_overrides or {}
        )
        if city:
            overrides.setdefault("CITY_NAME", city)
        self.cfg_overrides = overrides

    def run(self) -> Dict[str, Any]:
        self.log(
            f"[Stage1Job] Stage-1 for city={self.city!r}"
        )
        if self.should_stop():
            self.log(
                "[Stage1Job] stop requested before "
                "start."
            )
            self.last_result = {}
            return self.last_result

        result = run_stage1(
            cfg_overrides=self.cfg_overrides,
            logger=self.log,
            clean_run_dir=self.clean_run_dir,
            stop_check=self.should_stop,
            progress_callback=self.update_progress, 
        )
        self.last_result = result
        return result


class TrainingJob(AppJob):
    """Run Stage-2 training as a job."""

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        cfg_overrides: Optional[
            Dict[str, Any]
        ] = None,
        *,
        evaluate_training: bool = True,
        logger: Optional[LogFn] = None,
        stop_check: Optional[
            StopCheckFn
        ] = None,
        progress_hook: Optional[
            ProgressHook
        ] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            stop_check=stop_check,
            progress_hook=progress_hook,
        )
        self.manifest_path = manifest_path
        self.cfg_overrides = dict(cfg_overrides or {})
        self.evaluate_training = evaluate_training

    def run(self) -> Dict[str, Any]:
        self.log("[TrainingJob] Stage-2 training.")
        if self.should_stop():
            self.log(
                "[TrainingJob] stop requested before "
                "start."
            )
            self.last_result = {}
            return self.last_result

        result = run_training(
            manifest_path=self.manifest_path,
            cfg_overrides=self.cfg_overrides,
            logger=self.log,
            stop_check=self.should_stop,
            evaluate_training=self.evaluate_training,
            progress_callback=self.update_progress, 
        )
        self.last_result = result
        return result


class TuningJob(AppJob):
    """Run hyperparameter tuning as a job."""

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        cfg_overrides: Optional[
            Dict[str, Any]
        ] = None,
        *,
        evaluate_tuned: bool = False,
        logger: Optional[LogFn] = None,
        stop_check: Optional[
            StopCheckFn
        ] = None,
        progress_hook: Optional[
            ProgressHook
        ] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            stop_check=stop_check,
            progress_hook=progress_hook,
        )
        self.manifest_path = manifest_path
        self.cfg_overrides = dict(cfg_overrides or {})
        self.evaluate_tuned = evaluate_tuned

    def run(self) -> Dict[str, Any]:
        self.log("[TuningJob] Hyperparameter search.")
        if self.should_stop():
            self.log(
                "[TuningJob] stop requested before "
                "start."
            )
            self.last_result = {}
            return self.last_result

        result = run_tuning(
            manifest_path=self.manifest_path,
            cfg_overrides=self.cfg_overrides,
            logger=self.log,
            stop_check=self.should_stop,
            evaluate_tuned=self.evaluate_tuned,
            progress_callback=self.update_progress, 
        )
        self.last_result = result
        return result


class InferenceJob(AppJob):
    """Run Stage-3 inference as a job."""

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
        cfg_overrides: Optional[
            Dict[str, Any]
        ] = None,
        logger: Optional[LogFn] = None,
        stop_check: Optional[
            StopCheckFn
        ] = None,
        progress_hook: Optional[
            ProgressHook
        ] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            stop_check=stop_check,
            progress_hook=progress_hook,
        )
        self.model_path = model_path
        self.dataset = dataset
        self.use_stage1_future_npz = use_stage1_future_npz
        self.manifest_path = manifest_path
        self.stage1_dir = stage1_dir
        self.inputs_npz = inputs_npz
        self.targets_npz = targets_npz
        self.use_source_calibrator = (
            use_source_calibrator
        )
        self.calibrator_path = calibrator_path
        self.fit_calibrator = fit_calibrator
        self.cov_target = cov_target
        self.include_gwl = include_gwl
        self.batch_size = batch_size
        self.make_plots = make_plots
        self.cfg_overrides = dict(cfg_overrides or {})

    def run(self) -> Dict[str, Any]:
        self.log("[InferenceJob] Stage-3 inference.")
        if self.should_stop():
            self.log(
                "[InferenceJob] stop requested before start."
            )
            self.last_result = {}
            return self.last_result

        result = run_inference(
            model_path=self.model_path,
            dataset=self.dataset,
            use_stage1_future_npz=self.use_stage1_future_npz,  
            manifest_path=self.manifest_path,
            stage1_dir=self.stage1_dir,
            inputs_npz=self.inputs_npz,
            targets_npz=self.targets_npz,
            use_source_calibrator=self.use_source_calibrator,
            calibrator_path=self.calibrator_path,
            fit_calibrator=self.fit_calibrator,
            cov_target=self.cov_target,
            include_gwl=self.include_gwl,
            batch_size=self.batch_size,
            make_plots=self.make_plots,
            cfg_overrides=self.cfg_overrides,
            logger=self.log,
            stop_check=self.should_stop,
            progress_callback=self.update_progress, 
        )
        self.last_result = result
        return result



class XferMatrixJob(AppJob):
    """Run cross-city transfer matrix as a job."""

    def __init__(
        self,
        city_a: str,
        city_b: str,
        *,
        results_dir: str = "results",
        splits: Sequence[str] = ("val", "test"),
        calib_modes: Sequence[str] = (
            "none",
            "source",
            "target",
        ),
        rescale_to_source: bool = False,
        batch_size: int = 32,
        quantiles_override: Optional[
            Sequence[float]
        ] = None,
        out_dir: Optional[str] = None,
        write_json: bool = True,
        write_csv: bool = True,
        logger: Optional[LogFn] = None,
        stop_check: Optional[
            StopCheckFn
        ] = None,
        progress_hook: Optional[
            ProgressHook
        ] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            stop_check=stop_check,
            progress_hook=progress_hook,
        )
        self.city_a = city_a
        self.city_b = city_b
        self.results_dir = results_dir
        self.splits = tuple(splits)
        self.calib_modes = tuple(calib_modes)
        self.rescale_to_source = rescale_to_source
        self.batch_size = batch_size
        self.quantiles_override = (
            tuple(quantiles_override)
            if quantiles_override
            else None
        )
        self.out_dir = out_dir
        self.write_json = write_json
        self.write_csv = write_csv

    def run(self) -> Dict[str, Any]:
        self.log(
            "[XferMatrixJob] Transfer matrix "
            f"{self.city_a} -> {self.city_b}"
        )
        if self.should_stop():
            self.log(
                "[XferMatrixJob] stop requested "
                "before start."
            )
            self.last_result = {}
            return self.last_result

        result = run_xfer_matrix(
            city_a=self.city_a,
            city_b=self.city_b,
            results_dir=self.results_dir,
            splits=self.splits,
            calib_modes=self.calib_modes,
            rescale_to_source=self.rescale_to_source,
            batch_size=self.batch_size,
            quantiles_override=self.quantiles_override,
            out_dir=self.out_dir,
            write_json=self.write_json,
            write_csv=self.write_csv,
            logger=self.log,
            stop_check=self.should_stop,
            progress_callback=self.update_progress, 
        )
        self.last_result = result
        return result

class XferViewJob(AppJob):
    """Render transferability figure(s) from xfer_results.* files."""

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
        logger: Optional[LogFn] = None,
        stop_check: Optional[StopCheckFn] = None,
    ) -> None:
        """
        Parameters
        ----------
        view_kind :
            'calib_panel' or 'summary_panel'.
        results_root :
            Base results directory (e.g. ~/.fusionlab_runs).
        xfer_out_dir :
            Last xfer output folder (optional hint).
        xfer_csv, xfer_json :
            Explicit paths (optional).  If None, fall back to latest
            under results_root.
        """
        super().__init__(logger=logger, stop_check=stop_check)
        self.view_kind = view_kind
        self.results_root = results_root
        self.xfer_out_dir = xfer_out_dir
        self.xfer_csv = xfer_csv
        self.xfer_json = xfer_json
        self.split = split
        self.prefer_split = prefer_split
        self.prefer_calibration = prefer_calibration
        self.show_overall = show_overall
        self.dpi = dpi
        self.fontsize = fontsize

    def run(self) -> Dict[str, Any]:
        self.log(
            f"[XferViewJob] view_kind={self.view_kind} "
            f"split={self.split!r}"
        )
        if self.should_stop():
            self.log("[XferViewJob] stop requested before start.")
            self.last_result = {}
            return self.last_result

        out: Dict[str, Any]

        if self.view_kind == "calib_panel":
            csv_path = self.xfer_csv
            if not csv_path or not os.path.exists(csv_path):
                # Prefer last out_dir if given
                if self.xfer_out_dir:
                    cand = os.path.join(self.xfer_out_dir, "xfer_results.csv")
                    if os.path.exists(cand):
                        csv_path = cand
                if not csv_path or not os.path.exists(csv_path):
                    csv_path = latest_xfer_csv(self.results_root)
            if not csv_path:
                msg = (
                    "No xfer_results.csv found – cannot build view "
                    "(run transfer matrix first)."
                )
                self.log(msg)
                self.last_result = {}
                return self.last_result

            self.log(f"[XferViewJob] Using CSV: {csv_path}")
            base = os.path.join(
                os.path.dirname(csv_path),
                "xfer_transferability",
            )
            out = make_transferability_panel_from_csv(
                csv_path,
                split=self.split,
                out_base=base,
                fontsize=self.fontsize,
                dpi=self.dpi,
                add_legend=True,
                add_suptitle=False,
            )

        else:  # 'summary_panel'
            json_path = self.xfer_json
            if not json_path or not os.path.exists(json_path):
                if self.xfer_out_dir:
                    cand = os.path.join(
                        self.xfer_out_dir, "xfer_results.json"
                    )
                    if os.path.exists(cand):
                        json_path = cand
                if not json_path or not os.path.exists(json_path):
                    json_path = latest_xfer_json(self.results_root)
            if not json_path:
                msg = (
                    "No xfer_results.json found – cannot build view "
                    "(run transfer matrix first)."
                )
                self.log(msg)
                self.last_result = {}
                return self.last_result

            self.log(f"[XferViewJob] Using JSON: {json_path}")
            out = make_cross_transfer_from_json(
                json_path,
                out_dir=os.path.dirname(json_path),
                prefer_split=self.prefer_split,
                prefer_calibration=self.prefer_calibration,
                show_overall=self.show_overall,
                fontsize=self.fontsize,
                dpi=self.dpi,
                add_legend=True,
            )

        self.last_result = out
        return out

__all__ = [
    "AppJob",
    "Stage1Job",
    "TrainingJob",
    "TuningJob",
    "InferenceJob",
    "XferMatrixJob",
    "XferViewJob"
]

