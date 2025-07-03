# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""Threads
"""

from __future__ import annotations 
import os
import json 
from typing import Optional, Tuple, Dict  

import pandas as pd  

from PyQt5.QtCore    import ( 
    QThread, 
    pyqtSignal, 
)

from .config      import SubsConfig
from .processing  import DataProcessor, SequenceGenerator
from .modeling    import ModelTrainer, Forecaster
from .view        import ResultsVisualizer
from .inference import PredictionPipeline
from .tuner import TunerApp 
from .components import ProgressManager

__all__= ["TrainingThread", "InferenceThread", "TunerThread"]


class TrainingThread(QThread):
    """Run the complete *training* workflow in a background thread.

    Each major stage runs sequentially; after every stage the thread checks
    :py:meth:`isInterruptionRequested` so the GUI can stop the run quickly.
 
    This QThread subclass is designed to run the entire data
    processing and model training pipeline without freezing the main
    GUI thread. It orchestrates the instantiation and execution of
    the various processing classes (`DataProcessor`, `SequenceGenerator`,
    etc.) and emits signals to update the GUI with progress, logs,
    and final results.
    """
  

    status_updated = pyqtSignal(str)     # one‑line status text
    log_updated = pyqtSignal(str)        # console text
    coverage_ready = pyqtSignal(float)   # final coverage score (if any)
    error_occurred = pyqtSignal(str)     # emitted on any exception

    def __init__(
        self,
        cfg: SubsConfig,
        progress_manager: ProgressManager,
        edited_df: Optional[pd.DataFrame] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.progress_manager = progress_manager
        self.edited_df = edited_df
        # helper so we can pass *percent* directly to the Manager
        self._pct = lambda p: self.progress_manager.update(p, 100)

    def run(self) -> None:  # noqa: C901 – complexity fine for thread body
        # clean bar at the very beginning
        self.progress_manager.reset()

        try:
            proc = self._run_preprocessing()
            if self.isInterruptionRequested():
                return

            seq_gen, train_ds, val_ds = self._run_sequencing(proc)
            if self.isInterruptionRequested():
                return

            trainer, best_model = self._run_training(train_ds, val_ds)
            if self.isInterruptionRequested():
                return

            _, forecast_df = self._run_forecasting(
                best_model, seq_gen, val_ds, proc
            )
            if self.isInterruptionRequested():
                return

            self._run_visualisation(forecast_df)
            self._write_coverage_result()

            self.status_updated.emit("✔ Forecast finished.")
        except InterruptedError:
            self.status_updated.emit("⏹️ Workflow stopped by user.")
            self.log_updated.emit("Workflow was cancelled.")
            self.progress_manager.reset()
        except Exception as exc:  # pragma: no‑cover – bubble up
            self.error_occurred.emit(str(exc))
            self.progress_manager.reset()


    def _run_preprocessing(self) -> DataProcessor:
        self.status_updated.emit("📊 Pre‑processing…")
        self.progress_manager.start_step("Pre‑processing")
        self.cfg.progress_callback = self._pct

        processor = DataProcessor(
            self.cfg,
            self.log_updated.emit,
            raw_df=self.edited_df,
        )
        processor.run(stop_check=self.isInterruptionRequested)

        self.progress_manager.finish_step("Pre‑processing ✓")
        return processor

    def _run_sequencing(
        self, processor: DataProcessor
    ) -> Tuple[SequenceGenerator, object, object]:
        self.status_updated.emit("🌀 Generating sequences…")
        self.progress_manager.start_step("Sequencing")
        self.cfg.progress_callback = self._pct

        seq_gen = SequenceGenerator(self.cfg, self.log_updated.emit)
        train_ds, val_ds = seq_gen.run(
            processor.processed_df,
            processor.static_features_encoded,
            stop_check=self.isInterruptionRequested,
        )

        self.progress_manager.finish_step("Sequencing ✓")
        return seq_gen, train_ds, val_ds

    def __run_training(self, train_ds, val_ds):
        self.status_updated.emit("🔧 Training model…")
        self.progress_manager.start_step("Training")
        # Pass bar updates through GuiProgress (inside ModelTrainer)
        self.cfg.progress_callback = self._pct

        sample_inputs, _ = next(iter(train_ds))
        input_shapes = {k: v.shape for k, v in sample_inputs.items()}

        trainer = ModelTrainer(self.cfg, self.log_updated.emit)
        best_model = trainer.run(
            train_ds,
            val_ds,
            input_shapes,
            stop_check=self.isInterruptionRequested,
        )

        self.progress_manager.finish_step("Training ✓")
        return trainer, best_model


    def _run_training(self, train_ds, val_ds):
        """Runs the model-training stage with live *Epoch x/N* prefix."""
        self.status_updated.emit("🔧 Training model…")
        self.progress_manager.start_step("Training")

        # ------------------------------------------------------------------
        # ⟵ build a *dedicated* callback only for the training phase
        #     • updates the bar (normal behaviour)
        #     • derives the epoch index from the percentage
        #       and tells ProgressManager to prepend it.
        # ------------------------------------------------------------------
        total_epochs = self.cfg.epochs

        def _pct_training(percent: int) -> None:
            # 1) regular bar update
            self.progress_manager.update(percent, 100)
            # 2) convert % → epoch number (1-based, clamped)
            ep = max(1, min(total_epochs,
                            int(round((percent / 100) * total_epochs))))
            self.progress_manager.set_epoch_context(
                epoch=ep, total=total_epochs)

        # Tell ModelTrainer / GuiProgress to use our enriched callback
        self.cfg.progress_callback = _pct_training

        # everything else is unchanged
        sample_inputs, _ = next(iter(train_ds))
        input_shapes = {k: v.shape for k, v in sample_inputs.items()}

        trainer = ModelTrainer(self.cfg, self.log_updated.emit)
        best_model = trainer.run(
            train_ds, val_ds, input_shapes,
            stop_check=self.isInterruptionRequested,
        )

        self.progress_manager.finish_step("Training ✓")
        return trainer, best_model
    

    def _run_forecasting(
        self,
        model,
        seq_gen,
        val_ds,
        processor,
    ):  # -> Tuple[Forecaster, pd.DataFrame]
        self.status_updated.emit("🔮 Forecasting…")
        self.progress_manager.start_step("Forecasting")
        self.cfg.progress_callback = self._pct

        forecaster = Forecaster(self.cfg, self.log_updated.emit)
        forecast_df = forecaster.run(
            model=model,
            test_df=seq_gen.test_df,
            val_dataset=val_ds,
            static_features_encoded=processor.static_features_encoded,
            coord_scaler=seq_gen.coord_scaler,
            stop_check=self.isInterruptionRequested,
        )

        self.progress_manager.finish_step("Forecasting ✓")
        return forecaster, forecast_df

    def _run_visualisation(self, forecast_df):  # noqa: D401 – keep verb
        self.status_updated.emit("🎨 Visualising…")
        self.progress_manager.start_step("Visualising")
        # Visualisation is fast → no fine‑grained callback

        visualiser = ResultsVisualizer(self.cfg, self.log_updated.emit)
        visualiser.run(forecast_df, stop_check=self.isInterruptionRequested)

        self.progress_manager.finish_step("Visualising ✓")


    def _write_coverage_result(self):
        if not (self.cfg.evaluate_coverage and self.cfg.quantiles):
            return
        json_path = os.path.join(
            self.cfg.run_output_path, "diagnostics_results.json"
        )
        try:
            with open(json_path, "r", encoding="utf-8") as fp:
                cv = json.load(fp)["coverage"]
            self.coverage_ready.emit(float(cv))
        except Exception as err:
            self.log_updated.emit(f"[WARN] Could not read coverage file: {err}")

        # bar back to idle (useful when TrainingThread was run stand‑alone)
        self.progress_manager.reset()


class InferenceThread(QThread):
    """Background worker that runs the *inference* pipeline.

    Unlike training, inference is typically short, but still benefits from an
    ETA and a responsive GUI.  The new :class:`ProgressManager` is used to
    drive the single progress bar.
    
    Run the end-to-end *inference* workflow in a background
    thread so the Qt main-loop (GUI) remains responsive.

    """

    log_msg = pyqtSignal(str)
    status_msg = pyqtSignal(str)
    progress_val = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    finished_with_results = pyqtSignal()
    def __init__(
        self,
        manifest_path: str,
        edited_df: pd.DataFrame,
        progress_manager: ProgressManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.manifest_path = manifest_path
        self.edited_df = edited_df
        self.pm = progress_manager
        # helper – send % to ProgressManager
        self._pct = lambda p: self.pm.update(p, 100)


    def run(self):  # noqa: C901 – single high‑level method
        self.pm.reset()
        self.pm.start_step("Inference")

        try:
            self.status_msg.emit("🤖 Inferring…")
            self.log_msg.emit("⏳ Prediction pipeline triggered…")

            if self.isInterruptionRequested():
                raise InterruptedError

            # Build pipeline from manifest
            pipe = PredictionPipeline(
                manifest_path=self.manifest_path,
                log_callback=self.log_msg.emit,
                kind="inference",
            )
            cfg: SubsConfig = pipe.config
            cfg.progress_callback = self._pct  # 0‑100 from internal steps

            pipe.run(
                validation_data=self.edited_df,
                stop_check=self.isInterruptionRequested,
            )

            if self.isInterruptionRequested():
                raise InterruptedError

            self.pm.finish_step("Inference ✓")
            self.progress_val.emit(100)
            self.status_msg.emit("✔ Inference finished.")
            self.log_msg.emit("✔ Inference finished.")
            self.finished_with_results.emit()

        except InterruptedError:
            self.status_msg.emit("⏹️ Inference cancelled.")
            self.log_msg.emit("Workflow was cancelled.")
            self.pm.reset()
        except Exception as exc:
            self.log_msg.emit(f"❌ {exc}")
            self.error_occurred.emit(str(exc))
            self.pm.reset()

class TunerThread(QThread):
    """
    Background worker that drives a HydroTuner hyper-parameter search
    while keeping the Qt event-loop responsive.

    The single ProgressBar shown in the GUI is entirely controlled by
    `ProgressManager`; per-trial / per-epoch updates are emitted by the
    `TunerProgress` callback created inside ``TunerApp``.
    """

    # Qt ➜ GUI signals 
    log_updated      = pyqtSignal(str)
    status_updated   = pyqtSignal(str)
    tuning_finished  = pyqtSignal()          # emitted in *finally*
    error_occurred   = pyqtSignal(str)

    def __init__(
        self,
        cfg: SubsConfig,
        search_space: Dict,
        tuner_kwargs: Dict,
        *,
        progress_manager: ProgressManager=None,
        edited_df: Optional[pd.DataFrame] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)

        # immutable state 
        self.cfg           = cfg
        self.search_space  = search_space or {}
        self.tuner_kwargs  = tuner_kwargs or {}
        self.edited_df     = edited_df

        # ProgressManager 
        # Progress manager hooks 
        self.pm = progress_manager
        self._pct = lambda p: self.pm.update(p, 100)

        # Wire cfg callbacks so *all* internal steps reuse the single bar
        self.cfg.log = self.log_updated.emit
        self.cfg.progress_callback = self._pct
        

    def run(self) -> None:  # noqa: C901
        # initial bar state
        self.pm.reset()
        self.pm.start_step("Tuning")

        self.status_updated.emit("🔍 Tuning…")

        try:
            # ── build & execute the tuner application ----------------
            tuner_app = TunerApp(
                cfg             = self.cfg,
                search_space    = self.search_space,
                log_callback    = self.log_updated.emit,
                tuner_kwargs    = self.tuner_kwargs,
                trial_info      = self._on_trial_update,
                progress_manager= self.pm,         
                edited_df       = self.edited_df,   
            )

            tuner_app.run(stop_check=self.isInterruptionRequested)

            # On success the ProgressManager is already at 100 %
            self.pm.finish_step("Tuning ✓")
            self.status_updated.emit("✅ Tuning finished")

        except Exception as exc:
            self.pm.reset()
            self.status_updated.emit("❌ Tuning failed")
            self.error_occurred.emit(str(exc))

        finally:
            self.tuning_finished.emit()

    def _on_trial_update(self, cur: int, total: int, _eta: str) -> None:
        """
        Received once at the *start of every trial*.

        We simply change the prefix so the label reads  
        “Trial 3/20 – ETA: 02:13”.
        Percentage updates are handled by ``TunerProgress``.
        """
        self.pm.set_trial_context(trial=cur, total=total)
        self.pm.set_label(f"Trial {cur}/{total} - ETA: {_eta}")
        
        # self.pm.set_trial_context(trial=cur, total=total)

    # external request to stop the run
    def requestInterruption(self) -> None:
        self.log_updated.emit(
            "⏹ Stop requested — will interrupt at next safe point"
        )
        super().requestInterruption()

