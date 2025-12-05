Got it — I have the whole beast in my head now ??
Let me propose a refactor that:

* **Keeps behaviour identical** (Stage-1 smartness, QuickTrain/Tune, dry-run, etc.)
* But **moves the logic out of the giant `GeoPriorForecaster`** into small, testable pieces.

---

## 1. High-level target architecture

Right now `GeoPriorForecaster` is doing *everything*:

* UI layout + styling
* Config<->GUI sync
* Stage-1 smart handshake
* Building cfg_overrides
* Starting/stopping threads
* Dry-run previews
* Global mode + Stop button orchestration
* Results root syncing, xfer discovery, etc.

Refactor goal:

> **Keep `GeoPriorForecaster` mostly as a *thin Qt shell*** that delegates to
> small “controller” objects which know how to:
>
> * Plan a workflow (train / tune / infer / xfer)
> * Optionally do a dry-run preview
> * Actually start threads with a clean API

Concretely, I’d split into **three layers**:

1. **Domain / services (pure Python, no Qt)**

   * Smart Stage-1 handshake, config bridging, run planning, run roots, etc.
2. **Workflow controllers (still pure Python, no widgets)**

   * `TrainController`, `TuneController`, `InferController`, `XferController`, `DryRunController`.
3. **GUI (Qt)**

   * `GeoPriorForecaster` only wires widgets ? controllers, and pipes logs/status/progress.

---

## 2. Proposed modules & classes

I’ll keep names close to your current folder:

### 2.1 `geoprior/workflows/base.py`

A base helper for all flows, to centralise shared things:

```python
@dataclass
class GUIHooks:
    log: Callable[[str], None]
    status: Callable[[str], None]
    update_progress: Callable[[float], None]
    ask_yes_no: Callable[[str, str], bool]        # title, question
    warn: Callable[[str, str], None]              # title, msg
    error: Callable[[str, str], None]
```

```python
@dataclass
class RunEnv:
    gui_runs_root: Path
    geo_cfg: GeoPriorConfig
    device_overrides: Dict[str, Any]
    dry_mode: bool
```

```python
class BaseWorkflowController:
    def __init__(self, env: RunEnv, hooks: GUIHooks):
        self.env = env
        self.hooks = hooks

    def build_cfg_overrides(self) -> Dict[str, Any]:
        overrides = self.env.geo_cfg.to_cfg_overrides()
        overrides.update(self.env.device_overrides)
        return overrides
```

All other controllers subclass this.

---

### 2.2 `geoprior/services/stage1_service.py`

Move `_smart_stage1_handshake` and related bits into a **pure service**.

```python
@dataclass
class Stage1Decision:
    need_stage1: bool
    manifest_hint: Optional[str]    # path if reuse, None if build
    cancelled: bool                 # user pressed Cancel
    messages: list[str]             # log lines
```

```python
class Stage1Service:
    def __init__(self, env: RunEnv, hooks: GUIHooks):
        self.env = env
        self.hooks = hooks

    def decide(
        self,
        city: str,
        clean_stage1_dir: bool,
    ) -> Stage1Decision:
        """
        Encapsulates your current logic:

        - clean_stage1_dir -> force rebuild
        - build current cfg via geo_cfg.to_stage1_config()
        - find_stage1_for_city(...)
        - auto_reuse / force_rebuild_mismatch
        - Stage1ChoiceDialog.ask(...) (called via hooks)
        """
```

Key point: `Stage1Service.decide()` returns a **data object** that can be:

* Used by real training (to actually run Stage-1 or reuse).
* Used by **dry-run** to just *log* what would happen.

Your existing `_smart_stage1_handshake` becomes the implementation of `Stage1Service.decide`, but with side-effects (logging, QMessageBox) replaced by `hooks.log` and `hooks.ask_yes_no`.

---

### 2.3 `geoprior/workflows/train.py`

Encapsulate everything around training (real + dry).

```python
@dataclass
class TrainPlan:
    city: str
    csv_path: str
    cfg_overrides: Dict[str, Any]
    stage1_decision: Stage1Decision
    experiment_name: str
```

```python
class TrainController(BaseWorkflowController):
    def __init__(self, env: RunEnv, hooks: GUIHooks, stage1_svc: Stage1Service):
        super().__init__(env, hooks)
        self.stage1_svc = stage1_svc

    def plan_from_gui(
        self,
        city: str,
        csv_path: Optional[Path],
        experiment_name: str | None,
    ) -> Optional[TrainPlan]:
        """
        - Validate csv + city
        - sync GUI ? geo_cfg
        - cfg.ensure_valid()
        - compute Stage1Decision via self.stage1_svc.decide(...)
        - fill cfg_overrides
        """
```

Two main entrypoints:

```python
    def dry_preview(self, gui_state: TrainGuiState) -> None:
        """
        - Build TrainPlan via plan_from_gui(...)
        - If cancelled in handshake ? log & return
        - Use hooks.log / hooks.status / hooks.update_progress
          to mirror your existing _run_train_dry_preview
        """

    def start_real_run(
        self,
        gui_state: TrainGuiState,
        start_stage1_cb: Callable[[str, Dict[str, Any]], None],
        start_training_cb: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        """
        - Build TrainPlan
        - If cancelled ? just status 'Training cancelled...'
        - If plan.stage1_decision.need_stage1:
            start_stage1_cb(city, plan.cfg_overrides)
          else:
            start_training_cb(manifest_hint, plan.cfg_overrides)
        """
```

Where `TrainGuiState` is just a small dataclass capturing what the GUI provides:

```python
@dataclass
class TrainGuiState:
    city_text: str
    csv_path: Optional[Path]
    experiment_name: Optional[str]
```

Inside `GeoPriorForecaster`, `_on_train_clicked` becomes:

```python
@pyqtSlot()
def _on_train_clicked(self) -> None:
    if self._any_job_running():
        QMessageBox.information(...); return

    state = TrainGuiState(
        city_text=self.city_edit.text(),
        csv_path=self.csv_path,
        experiment_name=self.edit_experiment_name.text().strip() or None,
    )

    if self._is_dry_mode():
        self.train_controller.dry_preview(state)
        return

    # Real run
    def start_stage1(city, cfg_overrides):
        self._cfg_overrides = cfg_overrides
        self._start_stage1(city)

    def start_training(manifest_path, cfg_overrides):
        self._cfg_overrides = cfg_overrides
        self._start_training(manifest_path)

    self.train_controller.start_real_run(
        state, start_stage1, start_training
    )
```

All the complicated “smart” logic disappears from the GUI method.

---

### 2.4 `geoprior/workflows/tune.py`

Same pattern for tuning.

```python
@dataclass
class TunePlan:
    city: str
    manifest_path: Optional[str]
    cfg_overrides: Dict[str, Any]
    max_trials: int
```

```python
class TuneController(BaseWorkflowController):
    def resolve_job(
        self,
        queued_job: Optional[TuneJobSpec],
        city_text: str,
        quick_dialog_runner: Callable[[], tuple[bool, Optional[TuneJobSpec]]],
    ) -> Optional[TunePlan]:
        """
        Encapsulates:

        - priority of queued_job
        - QuickTuneDialog.run if no city/job
        - building tuner_search_space, cfg_overrides, TUNER_MAX_TRIALS
        """
    
    def dry_preview(self, plan: TunePlan, stage1_root: str) -> None:
        """
        Implement your _run_tune_dry_preview using hooks.
        """
```

Then GUI’s `_on_tune_clicked` just becomes:

```python
@pyqtSlot()
def _on_tune_clicked(self) -> None:
    if self.tuning_thread and self.tuning_thread.isRunning():
        QMessageBox.information(...); return

    if hasattr(self, "log_mgr"):
        self.log_mgr.clear()

    def run_quick_dialog():
        return QuickTuneDialog.run(
            results_root=self.gui_runs_root, parent=self
        )

    plan = self.tune_controller.resolve_job(
        queued_job=self._queued_tune_job,
        city_text=self.city_edit.text().strip(),
        quick_dialog_runner=run_quick_dialog,
    )
    self._queued_tune_job = None

    if plan is None:
        return

    if self._is_dry_mode():
        self.tune_controller.dry_preview(plan, stage1_root=plan.cfg_overrides.get("BASE_OUTPUT_DIR", ""))
        return

    # real run: create TuningThread from plan
    th = TuningThread(
        manifest_path=plan.manifest_path,
        cfg_overrides=plan.cfg_overrides,
        evaluate_tuned=self.chk_eval_tuned.isChecked(),
        parent=self,
    )
    ...
```

Again, *logic* lives in controller, GUI just glues plan ? thread.

---

### 2.5 `geoprior/workflows/infer.py` & `geoprior/workflows/xfer.py`

Same idea:

* Small dataclasses for GUI state and plans:

```python
@dataclass
class InferGuiState:
    model_path: str
    dataset_key: str
    use_future: bool
    manifest_path: Optional[str]
    inputs_npz: Optional[str]
    targets_npz: Optional[str]
    use_source_calib: bool
    calibrator_path: Optional[str]
    fit_calibrator: bool
    cov_target: float
    include_gwl: bool
    batch_size: int
    make_plots: bool
```

```python
class InferController(BaseWorkflowController):
    def validate_from_gui(self, state: InferGuiState) -> Optional[InferGuiState]:
        """
        Move all QMessageBox warnings into hooks.warn(...) and
        return None on invalid config.
        """

    def dry_preview(self, state: InferGuiState) -> None:
        """
        Your _run_infer_dry_preview, but backend-agnostic.
        """
```

Then `_on_infer_clicked` is just:

```python
@pyqtSlot()
def _on_infer_clicked(self) -> None:
    if self.inference_thread is not None:
        QMessageBox.information(...); return

    state = InferGuiState(... read from widgets ...)

    state = self.infer_controller.validate_from_gui(state)
    if state is None:
        return

    if self._is_dry_mode():
        self.infer_controller.dry_preview(state)
        return

    # build thread directly from `state`
    th = InferenceThread(...params from state...)
    ...
```

For transferability (`xfer`), same pattern as your `_run_xfer_dry_preview` / `_on_xfer_clicked`.

---

### 2.6 `geoprior/ui/mode_manager.py`

Extract all Mode button & Stop button logic:

* `_update_mode_button`
* `_update_global_running_state`
* `_on_stop_pulse_tick`
* `_on_stop_clicked`

into a dedicated helper:

```python
class ModeManager(QObject):
    def __init__(self, mode_btn, btn_stop, make_play_icon, timer, parent=None):
        ...

    def set_dry_mode(self, is_dry: bool) -> None: ...
    def set_active_job_kind(self, kind: Optional[str]) -> None: ...
    def update_for_tab(self, index: int, tabs, help_texts: dict) -> None: ...
    def update_running_state(self, any_running: bool) -> None: ...
```

`GeoPriorForecaster` will just forward calls (`mode_mgr.update_for_tab(index)`, etc) instead of embedding the styling logic.

That makes it very easy to tweak mode behaviour without hunting across the main file.

---

### 2.7 `geoprior/services/results_service.py`

Small functions for:

* `_discover_last_xfer_for_root`
* `_save_gui_log_for_result`

Turn them into pure helpers:

```python
class ResultsService:
    def __init__(self, log: Callable[[str], None]):
        self.log = log

    def discover_last_xfer(self, results_root: Path) -> dict: ...
    def save_gui_log(self, log_mgr, result: dict) -> None: ...
```

That removes the repeated “if hasattr(self, 'log_mgr')” noise from the main window.

---

## 3. What stays in `GeoPriorForecaster`

After refactor, your main class would be responsible for:

* Building widgets/layout (all the UI you already have).

* Creating controllers and services:

  ```python
  env = RunEnv(
      gui_runs_root=self.gui_runs_root,
      geo_cfg=self.geo_cfg,
      device_overrides=self._device_cfg_overrides,
      dry_mode=self._is_dry_mode(),
  )
  hooks = GUIHooks(
      log=self.log,
      status=lambda s: self.status_updated.emit(s),
      update_progress=self._update_progress,
      ask_yes_no=self._ask_yes_no_dialog,
      warn=self._warn_dialog,
      error=self._error_dialog,
  )

  self.stage1_service = Stage1Service(env, hooks)
  self.train_controller = TrainController(env, hooks, self.stage1_service)
  self.tune_controller = TuneController(env, hooks)
  self.infer_controller = InferController(env, hooks)
  self.xfer_controller = XferController(env, hooks)
  ```

* Wiring signals:

  ```python
  self.train_btn.clicked.connect(self._on_train_clicked)
  self.btn_run_tune.clicked.connect(self._on_tune_clicked)
  self.btn_run_infer.clicked.connect(self._on_infer_clicked)
  self.btn_run_xfer.clicked.connect(self._on_xfer_clicked)
  self.chk_dry_run.toggled.connect(self._on_dry_run_toggled)
  self.tabs.currentChanged.connect(self._on_tab_changed)
  ```

* Starting concrete threads when controllers tell it “what to run”.

This means debugging a bug in Stage-1 logic is:

* Go to `Stage1Service.decide` and log/print/test it in isolation
  rather than digging through 10 GUI methods.

---

## 4. Incremental migration plan

Because this is a lot, I’d refactor in **small, safe steps**:

1. **Extract Stage-1 service**

   * Move `_smart_stage1_handshake` into `Stage1Service`.
   * Update `_on_train_clicked` and `_run_train_dry_preview` to call it.
   * Behaviour must remain bit-for-bit the same.

2. **Extract TrainController**

   * Move `_run_train_dry_preview` logic into `TrainController.dry_preview`.
   * Move the “normal run” logic from `_on_train_clicked` into `TrainController.start_real_run`.
   * The GUI method becomes ~10–15 lines.

3. **Do the same for Tune** (controller + dry-run).

4. **Do the same for Inference & Xfer**.

5. Extract `ModeManager` & `ResultsService` to shrink the GUI further.

At any step you can still run the GUI and check “smart Stage-1, quick train, dry-run, transferability etc.” behave as before.

---

If you’d like, next step I can:

* Draft **concrete code** for `Stage1Service` + `TrainController` and
* Show the new, much smaller `_on_train_clicked` method,
  so you can actually drop it into your repo and iterate from there.
