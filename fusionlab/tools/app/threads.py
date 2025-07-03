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

        # ------------------------------------------------------------------
        # everything else is unchanged
        # ------------------------------------------------------------------
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


class _TrainingThread(QThread):
    """Executes the forecasting workflow in a background thread.

    This QThread subclass is designed to run the entire data
    processing and model training pipeline without freezing the main
    GUI thread. It orchestrates the instantiation and execution of
    the various processing classes (`DataProcessor`, `SequenceGenerator`,
    etc.) and emits signals to update the GUI with progress, logs,
    and final results.

    Parameters
    ----------
    cfg : SubsConfig
        A configuration object containing all parameters required for
        the workflow, gathered from the GUI.
    edited_df : pd.DataFrame, optional
        If the user has edited the data in the preview dialog, this
        DataFrame is passed to bypass the initial file loading step.
        If None, the workflow will load data from the file path
        specified in the `cfg` object.
    parent : QObject, optional
        The parent Qt object, by default None.

    Attributes
    ----------
    status_msg : pyqtSignal(str)
        Emits status updates for the main status label in the GUI.
    progress_val : pyqtSignal(int)
        Emits progress updates (0-100) for the progress bar.
    log_msg : pyqtSignal(str)
        Emits detailed log messages to be displayed in the log panel.
    coverage_val : pyqtSignal(float)
        Emits the final calculated coverage score to be displayed in
        the status bar.

    Methods
    -------
    run()
        The main entry point for the thread. Executes the entire
        data processing and forecasting pipeline sequentially.
    _write_coverage_result()
        A helper method to read the saved coverage score from a JSON
        file and emit it via the `coverage_val` signal.

    See Also
    --------
    DataProcessor : Handles the data loading and preprocessing stage.
    SequenceGenerator : Handles sequence generation and dataset creation.
    ModelTrainer : Handles model definition, compilation, and training.
    Forecaster : Handles prediction on new data.
    ResultsVisualizer : Handles the final visualization of results.
    """
    status_msg   = pyqtSignal(str)
    progress_val = pyqtSignal(int)
    log_msg      = pyqtSignal(str)
    coverage_val = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, cfg, edited_df=None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.edited_df  = edited_df
        self._p = lambda frac, lo, hi: int(lo + (hi - lo) * frac)

    def run(self):
        try:
            self.status_msg.emit("📊 Pre-processing…")
            self.cfg.progress_callback = lambda p: self.progress_val.emit(
                self._p(p / 100, 0, 10) 
            )
            processor = DataProcessor(
                self.cfg, self.log_msg.emit, 
                raw_df=self.edited_df  
            )
            df_proc   = processor.run(
                stop_check=self.isInterruptionRequested) 
            self.progress_val.emit(10)

            if self.isInterruptionRequested():          # ← CHECK #1
                return
            self.status_msg.emit("🌀 Generating sequences…")
            self.cfg.progress_callback = lambda p: self.progress_val.emit(
               self._p(p / 100, 10, 30)           # ← divide by 100!
            )      
            seq_gen   = SequenceGenerator(self.cfg, self.log_msg.emit)
            try: 
                train_ds, val_ds = seq_gen.run(
                    df_proc, processor.static_features_encoded, 
                    stop_check=self.isInterruptionRequested
                )
            except InterruptedError:
                return
              
            self.progress_val.emit(30)
            
            if self.isInterruptionRequested():          # ← CHECK #2
                return
            
            self.status_msg.emit("🔧 Training…")
            train_range = (30, 90)
            self.cfg.progress_callback = lambda p: self.progress_val.emit(
                self._p(p/100, *train_range))    # ← same here
            
            sample_inputs, _ = next(iter(train_ds))
            shapes = {k: v.shape for k, v in sample_inputs.items()}
            model  = ModelTrainer(self.cfg, self.log_msg.emit).run(
                train_ds, val_ds, shapes, 
                stop_check = self.isInterruptionRequested
            )
            self.progress_val.emit(train_range[1])          # 90 %

            if self.isInterruptionRequested():          # ← CHECK #3
                return
            self.status_msg.emit("🔮 Forecasting…")
            forecast_range = (90, 100)
            self.cfg.progress_callback = lambda p: self.progress_val.emit(
                self._p(p / 100, *forecast_range)          
            )
            forecast_df = Forecaster(self.cfg, self.log_msg.emit).run(
                model=model,
                test_df=seq_gen.test_df,
                val_dataset=val_ds,
                static_features_encoded=processor.static_features_encoded,
                coord_scaler=seq_gen.coord_scaler,
                stop_check= self.isInterruptionRequested
            )
            self._write_coverage_result () 
            self.status_msg.emit("✔ Forecast finished.")
            self.progress_val.emit(100)
            
            ResultsVisualizer(self.cfg, self.log_msg.emit).run(
                forecast_df, stop_check= self.isInterruptionRequested 
            )
            
        except Exception as e:
            self.log_msg.emit(f"❌ {e}")
            self.error_occurred.emit(str(e))
            
    def _write_coverage_result(self) :
        if self.cfg.evaluate_coverage and self.cfg.quantiles:
            json_path = os.path.join(self.cfg.run_output_path,
                                     "diagnostics_results.json")
            try:
                with open(json_path, "r", encoding="utf-8") as fp:
                    cv = json.load(fp)["coverage"]
                    self.coverage_val.emit(float(cv))
            except Exception as e:
                self.log_msg.emit(
                    f"[WARN] Could not read coverage file: {e}")
     


class InferenceThread(QThread):
    """Background worker that runs the *inference* pipeline.

    Unlike training, inference is typically short, but still benefits from an
    ETA and a responsive GUI.  The new :class:`ProgressManager` is used to
    drive the single progress bar.
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

class _InferenceThread(QThread):
    """
    Run the end-to-end *inference* workflow in a background
    thread so the Qt main-loop (GUI) remains responsive.

    Why a separate thread?
    ----------------------
    • Prediction can still take a few seconds (model + I/O);  
      moving it off the GUI thread prevents the application
      from “freezing” and keeps the progress-bar animating.  
    • All status / log / progress signals are emitted back to
      the main window, which updates widgets in a thread-safe
      way (Qt’s signal/slot mechanism).

    Parameters
    ----------
    manifest_path : str
        Absolute path to the *run_manifest.json* that contains
        all artefact locations of a previously-trained run
        (model weights, encoders, scalers, …).
    edited_df : pandas.DataFrame
        The pre-processed **validation** dataframe that the user
        may have edited in the CSV preview dialog.  
        Passing it directly avoids re-loading the CSV from disk.
    parent : QObject, optional
        Parent widget; used only so that
        ``QMessageBox.critical()`` has the correct owner.

    Signals
    -------
    log_msg(str)
        One log line ready to be appended to the GUI console.
    progress_val(int)
        0 – 100 progress updates for the progress-bar.
    finished_with_results
        Can be connected if the caller needs to know when the
        full pipeline finished successfully (no payload here;
        extend as needed).

    Notes
    -----
    *Never* create or touch Qt widgets inside the thread itself;
    Qt objects must live in the GUI thread.  Here we only emit
    signals.
    """
    
    log_msg      = pyqtSignal(str)
    status_msg   = pyqtSignal(str)
    progress_val = pyqtSignal(int)
    error_occurred = pyqtSignal(str) 
    finished_with_results = pyqtSignal() 

    def __init__(
        self,
        manifest_path: str,
        edited_df: pd.DataFrame, 
        progress_manager=None,  
        parent=None
    ):
        super().__init__(parent)
        self.manifest_path = manifest_path
        self.edited_df     = edited_df 
        

    def run(self):
        """
        Spin up :class:`PredictionPipeline`, wire its callbacks to
        our Qt signals, and execute the inference pass.

        Workflow
        --------
        1.  Build the pipeline from *run_manifest.json*  
            → all artefact paths are resolved automatically.
        2.  Redirect pipeline logs to ``log_msg`` so the GUI gets
            real-time messages.
        3.  Attach the GUI’s progress-bar callback to the pipeline’s
            config so every internal step reports its percentage.
        4.  Call ``pipe.run(validation_data=edited_df)``, which
            returns only when the full prediction + visualisation
            logic is done.
        5.  On success: emit *100 %* and log a final line.  
            On failure: emit an error line **and** show a critical
            message-box on the GUI thread.

        Any uncaught exception is converted to a user-visible
        message while keeping the application alive.
        """
        msg = "✔ Inference finished."
        try:
            # Create the pipeline using the manifest path.
            # The log callback is now passed during initialization.
            self.status_msg.emit("🤖 Inferring...")
            self.log_msg.emit("⏳ Prediction Pipeline triggered...")
            if self.isInterruptionRequested():          # ← CHECK #1
                return
            
            pipe = PredictionPipeline(
                manifest_path=self.manifest_path,
                log_callback=self.log_msg.emit,
                kind ='inference', 
            )
            # The config object's progress callback can be linked
            # to the GUI's progress bar.
            cfg: SubsConfig = pipe.config
            cfg.progress_callback = self.progress_val.emit

            # Execute the pipeline with the in-memory DataFrame.
            # We assume the `run` method is updated to accept a DataFrame.
            pipe.run(
                validation_data=self.edited_df,
                stop_check = self.isInterruptionRequested 
                )
            
            if self.isInterruptionRequested():          # ← CHECK #2
                return
            
            self.status_msg.emit(msg)
            self.log_msg.emit(msg)
            self.progress_val.emit(100)

        except Exception as err:
            self.log_msg.emit(f"❌ {err}")
            self.error_occurred.emit(str(err))

class TunerThread(QThread):
    """
    Background worker that drives a HydroTuner hyper-parameter search
    while keeping the Qt event-loop responsive.

    The single ProgressBar shown in the GUI is entirely controlled by
    `ProgressManager`; per-trial / per-epoch updates are emitted by the
    `TunerProgress` callback created inside ``TunerApp``.
    """

    # Qt ➜ GUI signals -------------------------------------------------
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

        # ------------ immutable state ---------------------------------
        self.cfg           = cfg
        self.search_space  = search_space or {}
        self.tuner_kwargs  = tuner_kwargs or {}
        self.edited_df     = edited_df

        # ------------ ProgressManager -------------------------------
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

    # external request to stop the run
    def requestInterruption(self) -> None:
        self.log_updated.emit(
            "⏹ Stop requested — will interrupt at next safe point"
        )
        super().requestInterruption()


class _TunerThread(QThread):
    """
    Background worker that runs an HydroTuner hyper-parameter search
    through the higher-level `TunerApp`, keeping the GUI responsive.
    """
    # ----- Qt signals poked back into the main GUI ---------------------------
    log_updated      = pyqtSignal(str)     # console lines
    status_updated   = pyqtSignal(str)     # one-line status bar
    progress_updated = pyqtSignal(int)     # 0-100%
    tuning_finished  = pyqtSignal()        # emitted on graceful completion
    finished         = pyqtSignal() 
    error_occurred   = pyqtSignal(str)     # emitted on any exception

    def __init__(
        self,
        cfg,                                 # SubsConfig already prepared
        search_space: dict,                  # user-defined HP search space
        tuner_kwargs: dict,                  # algorithm, max_trials, …
        *,                                   # keyword-only after here ↓
        progress_manager =None, 
        edited_df=None,                      # optional GUI-edited CSV
        parent=None
    ):
        super().__init__(parent)
        self.cfg           = cfg
        self.search_space  = search_space
        self.tuner_kwargs  = tuner_kwargs or {}
        self.edited_df     = edited_df      # None ⇒ DataProcessor loads from disk

        # give SubsConfig live hooks so DataProcessor & friends update the GUI
        self.progress_val = 0   
        self.cfg.log               = self.log_updated.emit
        self.cfg.progress_callback = self.progress_updated.emit
        self.progress_updated.connect(lambda v: setattr(self, "progress_val", v))

    def run(self):
        try:
            self.status_updated.emit("🔍 Tuning…")
            # Instantiate & run the high-level TunerApp 
            tuner_app = TunerApp(
                cfg=self.cfg,
                search_space=self.search_space,
                log_callback=self.log_updated.emit,
                tuner_kwargs=self.tuner_kwargs,
                trial_info = self.parent().trial_updated,   
            )
            # Wrap our GUI logger so we get live trial / epoch updates
            # max_trials = self.tuner_kwargs.get("max_trials", 10)
            # # add trial info 
            # gui_cb = GUILoggerCallback(
            #         log_sig   = self.log_updated,
            #         prog_sig  = self.progress_updated,
            #         trial_sig = self.parent().trial_updated,   
            #         max_trials= max_trials,
            #         epochs    = self.cfg.epochs,
            #     )
            tuner_app.run(
                stop_check=self.isInterruptionRequested,
                # callbacks=[gui_cb],
            )

            self.status_updated.emit("✅ tuning finished")
            self.progress_updated.emit(100)
            # self.tuning_finished.emit()

        except Exception as exc:
            self.status_updated.emit("❌ tuning failed")
            self.error_occurred.emit(str(exc))
        finally: 
            # let MiniForecaster know we're done
            self.tuning_finished.emit()
            # # also trigger the standard QThread signal
            # self.finished.emit()          
            
    def requestInterruption(self):
        """
        Overridden just to emit a log line; real stop-check flag is read
        inside `TunerApp.run(stop_check=…)`.
        """
        self.log_updated.emit(
            "⏹ stop requested — will interrupt at next safe point")
        super().requestInterruption()

