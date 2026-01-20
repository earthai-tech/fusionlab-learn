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
from typing import TYPE_CHECKING

from .config.store import GeoConfigStore
from .runs.run_stage1 import run_stage1
from .runs.run_training import run_training
from .runs.run_tuning import run_tuning
from .runs.run_inference import run_inference
from .runs.run_xfer_matrix import run_xfer_matrix
from .services.xfer_view import (
    latest_xfer_csv,
    latest_xfer_json,
    make_transferability_panel_from_csv,
    make_cross_transfer_from_json,
)
from .config.overrides import _merge_overrides

if TYPE_CHECKING:
    import pandas as pd

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
        cfg_overrides: Optional[Dict[str, Any]] = None,
        *,
        store: Optional["GeoConfigStore"] = None,
        config_overwrite: Optional[Dict[str, Any]] = None,
        clean_run_dir: bool = True,
        base_cfg: Optional[Dict[str, Any]] = None,
        results_root: Optional[os.PathLike | str] = None,
        edited_df: Optional["pd.DataFrame"] = None,
        logger: Optional[LogFn] = None,
        stop_check: Optional[StopCheckFn] = None,
        progress_hook: Optional[ProgressHook] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            stop_check=stop_check,
            progress_hook=progress_hook,
        )
        self.city = city
        self.clean_run_dir = clean_run_dir
        self.base_cfg = base_cfg
        self.results_root = results_root
        self.edited_df = edited_df

        self.store = store
        self.config_overwrite = config_overwrite

        city_ov = {"CITY_NAME": city} if city else {}
        self.cfg_overrides = _merge_overrides(
            cfg_overrides,
            city_ov,
        )


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
            store=self.store,
            config_overwrite=self.config_overwrite,
            logger=self.log,
            clean_run_dir=self.clean_run_dir,
            stop_check=self.should_stop,
            progress_callback=self.update_progress,
            base_cfg=self.base_cfg,
            results_root=self.results_root,
            edited_df=self.edited_df,
        )
        self.last_result = result
        return result

class TrainingJob(AppJob):
    """Run Stage-2 training as a job."""

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        *,
        store: Optional["GeoConfigStore"] = None,
        config_overwrite: Optional[Dict[str, Any]] = None,
        evaluate_training: bool = True,
        base_cfg: Optional[Dict[str, Any]] = None,
        results_root: Optional[os.PathLike | str] = None,
        logger: Optional[LogFn] = None,
        stop_check: Optional[StopCheckFn] = None,
        progress_hook: Optional[ProgressHook] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            stop_check=stop_check,
            progress_hook=progress_hook,
        )
        self.manifest_path = manifest_path
        self.evaluate_training = evaluate_training
        self.base_cfg = base_cfg
        self.results_root = results_root

        self.store = store
        self.config_overwrite = config_overwrite
        self.cfg_overrides = cfg_overrides
    

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
            store=self.store,
            config_overwrite=self.config_overwrite,
            logger=self.log,
            stop_check=self.should_stop,
            evaluate_training=self.evaluate_training,
            progress_callback=self.update_progress,
            base_cfg=self.base_cfg,
            results_root=self.results_root,
        )
        self.last_result = result
        return result


class TuningJob(AppJob):
    """Run hyperparameter tuning as a job."""

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        *,
        store: Optional["GeoConfigStore"] = None,
        config_overwrite: Optional[Dict[str, Any]] = None,
        evaluate_tuned: bool = False,
        base_cfg: Optional[Dict[str, Any]] = None,
        results_root: Optional[os.PathLike | str] = None,
        logger: Optional[LogFn] = None,
        stop_check: Optional[StopCheckFn] = None,
        progress_hook: Optional[ProgressHook] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            stop_check=stop_check,
            progress_hook=progress_hook,
        )
        self.manifest_path = manifest_path
        self.evaluate_tuned = evaluate_tuned
        self.base_cfg = base_cfg
        self.results_root = results_root

        self.store = store
        self.config_overwrite = config_overwrite
        self.cfg_overrides = cfg_overrides


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
            store=self.store,
            config_overwrite=self.config_overwrite,
            logger=self.log,
            stop_check=self.should_stop,
            evaluate_tuned=self.evaluate_tuned,
            progress_callback=self.update_progress,
            base_cfg=self.base_cfg,
            results_root=self.results_root,
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
        store: Optional["GeoConfigStore"] = None,
        config_overwrite: Optional[Dict[str, Any]] = None,
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
        self.store = store
        self.cfg_overrides = _merge_overrides(
            cfg_overrides,
            config_overwrite,  # deprecated compat
        )

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
            store=self.store,
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
        store: Optional["GeoConfigStore"] = None,
        results_dir: str = "results",
        results_root: Optional[str] = None,
        splits: Sequence[str] = ("val", "test"),
        calib_modes: Sequence[str] = (
            "none",
            "source",
            "target",
        ),
        rescale_to_source: bool = False,
        rescale_modes: Optional[Sequence[str]] = None,
        strategies: Optional[Sequence[str]] = None,
        batch_size: int = 32,
        quantiles_override: Optional[Sequence[float]] = None,
        out_dir: Optional[str] = None,
        write_json: bool = True,
        write_csv: bool = True,
        model_name: str = "GeoPriorSubsNet",
        prefer_tuned: bool = True,
        align_policy: str = "align_by_name_pad",
        allow_reorder_dynamic: Optional[bool] = None,
        allow_reorder_future: Optional[bool] = None,
        warm_split: Optional[str] = None,
        warm_samples: Optional[int] = None,
        warm_frac: Optional[float] = None,
        warm_epochs: Optional[int] = None,
        warm_lr: Optional[float] = None,
        warm_seed: Optional[int] = None,
        logger: Optional[LogFn] = None,
        stop_check: Optional[StopCheckFn] = None,
        progress_hook: Optional[ProgressHook] = None,
    ) -> None:
        super().__init__(
            logger=logger,
            stop_check=stop_check,
            progress_hook=progress_hook,
        )
        self.city_a = city_a
        self.city_b = city_b

        self.store = store
        self.results_dir = results_dir
        self.results_root = results_root

        self.splits = tuple(splits)
        self.calib_modes = tuple(calib_modes)

        self.rescale_to_source = bool(rescale_to_source)
        self.rescale_modes = (
            tuple(rescale_modes)
            if rescale_modes is not None
            else None
        )
        self.strategies = (
            tuple(strategies)
            if strategies is not None
            else None
        )

        self.batch_size = int(batch_size)
        self.quantiles_override = (
            tuple(quantiles_override)
            if quantiles_override
            else None
        )

        self.out_dir = out_dir
        self.write_json = bool(write_json)
        self.write_csv = bool(write_csv)

        self.model_name = model_name
        self.prefer_tuned = bool(prefer_tuned)
        self.align_policy = align_policy
        self.allow_reorder_dynamic = allow_reorder_dynamic
        self.allow_reorder_future = allow_reorder_future

        self.warm_split = warm_split
        self.warm_samples = warm_samples
        self.warm_frac = warm_frac
        self.warm_epochs = warm_epochs
        self.warm_lr = warm_lr
        self.warm_seed = warm_seed

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
            store=self.store,
            results_dir=self.results_dir,
            results_root=self.results_root,
            splits=self.splits,
            calib_modes=self.calib_modes,
            rescale_to_source=self.rescale_to_source,
            rescale_modes=self.rescale_modes,
            strategies=self.strategies,
            batch_size=self.batch_size,
            quantiles_override=self.quantiles_override,
            out_dir=self.out_dir,
            write_json=self.write_json,
            write_csv=self.write_csv,
            logger=self.log,
            stop_check=self.should_stop,
            progress_callback=self.update_progress,
            model_name=self.model_name,
            prefer_tuned=self.prefer_tuned,
            align_policy=self.align_policy,
            allow_reorder_dynamic=self.allow_reorder_dynamic,
            allow_reorder_future=self.allow_reorder_future,
            warm_split=self.warm_split,
            warm_samples=self.warm_samples,
            warm_frac=self.warm_frac,
            warm_epochs=self.warm_epochs,
            warm_lr=self.warm_lr,
            warm_seed=self.warm_seed,
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

