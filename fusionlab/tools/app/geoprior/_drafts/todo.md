Short answer: I’d treat **GeoPriorForecaster** as a thin, beautiful GUI shell around the NATCOM GeoPrior pipeline you already wrote, with the *Stage-1 manifest* as the central “contract” between everything.

Below is a concrete plan and some design advice.

---

## 1. What we already have (backend)

From the files:

* **Stage-1**: `main_NATCOM_GEOPRIOR_stage1_prepare.py` (pre-processing + sequence export, manifest, scalers, NPZs). 
* **Stage-2 (tune)**: `tune_NATCOM_GEOPRIOR.py` (loads Stage-1 manifest, defines HP space, runs `GeoPriorTuner`, saves best model + best_hps). 
* **Stage-2 (train)**: `training_NATCOM_GEOPRIOR.py` (training with physics losses, interval calibration, exports eval/future plots & CSVs). 
* **Stage-3 (inference)**: `inference_NATCOM_GEOPRIOR.py` (val/test/custom inference, metrics, physics diagnostics, forecast plots). 
* **Future forecasts**: `make_future_forecasts.py` (seed from val/train, pick tuned or trained model, apply calibrator, export multi-year q10/q50/q90 CSVs). 
* **Cross-city transfer**: `xfer_matrix.py` (A?B/B?A, val/test, different calibration modes, optional re-scaling of dynamics). 
* **Single source of truth config**: `nat.com/config.py` ? `load_nat_config()`, giving city, time windows, feature registry, censoring, physics weights, etc. 
* **Model**: `GeoPriorSubsNet` with coordinate injection, physics head (K, Ss, t), scalar mv/?/?w/h_ref, and PDE residual machinery. 

Your existing **MiniForecaster** GUI already solved:

* CSV selection + preview/editor
* Config panels (model, training, physics)
* A generic progress bar + text log using your `TunerProgress` / `ProgressManager` components
* Running tuning / inference in background threads and dumping plots & CSVs in `fusionlab_runs`

So for GeoPriorForecaster we don’t need to reinvent the backend; we just need to:

1. Wrap each NATCOM stage in a callable “job” object.
2. Drive those jobs from a nicer PyQt GUI with better structure and theming.
3. Expose *new* GeoPrior-specific features (physics diagnostics, cross-city transfer, future forecasts) in the UI.

---

## 2. Overall architecture for **GeoPriorForecaster**

### 2.1 Layers

**A. Core pipeline layer (nat.com, already done)**
Keep your current scripts as the authoritative research pipeline (for CLI reproducibility and the paper).

**B. “Service” layer for the GUI (new, thin)**
Create a small module, e.g. `fusionlab/tools/app/geoprior/backend.py`, that wraps each stage in Python functions/classes that do *not* parse CLI arguments:

* `run_stage1(city: str, config_overrides: dict | None) -> ManifestPath`
* `run_tuning(manifest_path: str, tuner_config: dict) -> BestModelInfo`
* `run_training(manifest_path: str, train_config: dict) -> TrainedModelInfo`
* `run_inference(manifest_path: str, model_path: str, dataset: str, options: dict) -> InferenceResult`
* `run_future_forecasts(manifest_path: str, model_path: str, options: dict) -> ForecastResult`
* `run_xfer_matrix(city_a: str, city_b: str, options: dict) -> XferMatrixResult`

Internally these can just call the existing scripts’ `main()` functions (or refactored helpers) but with clear Python signatures instead of `argparse`.

**C. GUI layer (new): `GeopriorForecaster`**

* QMainWindow with rich layout and custom QSS theme (you already have `style.qss` and logo assets).
* Uses your shared helpers: `qt_utils`, `components` (ProgressManager), `notifications`, `tables`, `gui_popups`, etc.
* Each long-running job is executed via a `QThread`/`QRunnable` worker that calls the backend service functions.

This keeps things DRY:

* Any CLI script update automatically propagates to the app (since the backend just calls it).
* The GUI only knows about high-level objects: “run Stage-1 for city Zhongshan”, “tune model for Nansha”, etc.

---

## 3. GUI design proposal

Think of the app as a **wizard with power-user shortcuts**.

### 3.1 Main layout

Top bar:

* **City / Dataset** selector
* Buttons: **Stage-1**, **Train**, **Tune**, **Inference**, **Future forecasts**, **Transferability**
* “Mode” toggle: *Quick demo* vs *Research mode* (the latter exposes all configs, HPs, seeds).

Central area: a **tabbed** or **stacked** layout:

1. **Data & Stage-1 tab**

   * Inputs: city (Nansha/Zhongshan/Other), data root, BIG/SMALL CSV dropdown (auto from config).
   * Show parsed `config.py` values (TIME_STEPS, FORECAST_HORIZON_YEARS, features, censoring specs) in a pretty table (you already have `print_config_table`, we can reformat for Qt). 
   * CSV preview/editor using your existing table dialog (like screenshot 3).
   * “Run Stage-1” button ? builds everything and surfaces:

     * Path to `manifest.json`
     * Sizes of train/val/test splits
     * Feature summary (dynamic/static/future dims)

2. **Model & Physics tab**

   * Read from `config.json` + manifest:

     * architecture params (embed_dim, hidden_units, attention_levels, VSN, etc.) 
     * physics flags (PDE_MODE_CONFIG, ?_cons, ?_gw, ?_prior, ?_smooth, ?_mv, mv/kappa modes, Hd_factor, use_effective_h). 
   * Allow toggling:

     * “Physics off / data-only baseline”
     * “Consolidation only / GW only / both”
   * Offer presets: “Safe physics”, “Aggressive physics”, “No physics”.

3. **Training & Tuning tab**

   * Reuse the layout of the Mini Forecaster but cleaner:

     * Left: training hyperparameters (epochs, batch size, LR, early stopping)
     * Middle: tuning hyperparameters (search space summary from `tune_NATCOM_GEOPRIOR.py`) 
     * Right: run controls + progress bar + ETA.
   * Two big buttons:

     * **Train only** (no HP search, just use config) ? `training_NATCOM_GEOPRIOR.py`. 
     * **Tune & train** (start GeoPriorTuner, then retrain best model if desired). 
   * Log console at bottom (stream stdout from subprocess / worker).

4. **Inference & Evaluation tab**

   * Dataset selector: train/val/test/custom NPZ.
   * Options checkboxes:

     * “Compute MAE/MSE/R²”
     * “Compute physics diagnostics (e_prior, e_cons)” 
     * “Use source calibrator / Fit calibrator / Raw quantiles”.
   * Run ? call `inference_NATCOM_GEOPRIOR.py`, capture:

     * Metrics table
     * Coverage–sharpness plot (reuse `forecast_view` / `plot_forecasts`). 
   * Embed matplotlib canvas to show:

     * Actual vs q50 maps (like your screenshot)
     * Coverage–sharpness scatter, histogram of residuals.

5. **Future Forecasts tab**

   * Parameters: seed split (train/val), preferred model (tuned/trained), calibration mode, years to export, “cumulative q50” checkbox. 
   * Output panel:

     * Table of years with mean MAE / coverage.
     * Button: “Open map for year YYYY” (loads CSV slice, displays map with q10/q50/q90 overlays).

6. **Cross-city Transfer tab**

   * Controls mapped directly to `xfer_matrix.py`: cities A/B, splits (val/test), calibration modes, `rescale_to_source`. 
   * Results:

     * 2×2 grid: A?B(val/test), B?A(val/test) metrics (MAE, R², coverage, sharpness).
     * Optional physical parameter comparison (mv, ?, Hd_factor) extracted via `extract_physical_parameters`. 

7. **Config & Reproducibility tab**

   * Pretty view of `config.py` + auto-generated `config.json`. 
   * “Export experiment card” (Markdown/JSON with:

     * city, data paths, model version, physics settings, metrics, links to CSVs/figures).

---

## 4. Step-by-step development plan

### Step 0 – Freeze and test the pipeline

* Make sure the 5 NATCOM scripts run cleanly via CLI for Nansha & Zhongshan with the current `config.py`.
* Fix any remaining small mismatches (e.g. manifest city checks) *before* integrating into the GUI.

### Step 1 – Refactor minimal helpers out of scripts

In each NATCOM script, add a *library entry point*, e.g.:

```python
# nat.com/training_NATCOM_GEOPRIOR.py
def run_training(manifest_path: str | None = None, overrides: dict | None = None):
    # factor the current main body into a function
    ...
    return {
        "run_dir": RUN_OUTPUT_PATH,
        "best_model_path": best_model_path,
        "metrics": final_metrics_dict,
    }

if __name__ == "__main__":
    run_training()
```

Do the same for Stage-1, tuning, inference, future forecasts, and xfer matrix. Then your backend can just `from nat.com.training_NATCOM_GEOPRIOR import run_training`.

### Step 2 – Backend “job” classes

Create something like:

```python
class Stage1Job(AppJob):
    def __init__(self, city: str, overrides: dict | None = None): ...
    def run(self):
        return run_stage1(...)

class TuneJob(AppJob):
    def __init__(self, manifest_path, tuner_config): ...
    def run(self):
        return run_tuning(...)
```

Each job:

* Accepts a logger callback for streaming messages to the GUI.
* Returns a simple dataclass / dict with paths & metrics.

Your existing `ProgressManager` and `TunerProgress` can be reused to update the status bar, just as in MiniForecaster.

### Step 3 – New `GeopriorForecaster` UI

* Start from `mini_forecaster_gui.py` as a template (window creation, toolbar, basic layout).
* Replace hard-coded TransFlow-specific knobs by bindings to the NATCOM config keys (e.g., EPOCHS, BATCH_SIZE, QUANTILES, LAMBDA_* from `cfg`). 
* Add tabs as listed above; each tab knows which job type to create.

### Step 4 – Results viewers

* Wrap a simple matplotlib canvas widget for:

  * spatial maps (lon/lat scatter/hexbin coloured by subsidence/quantile).
  * coverage vs sharpness scatter.
* For CSVs, reuse the existing “CSV preview & editing” dialog.
* For physics diagnostics, show `epsilon_prior`, `epsilon_cons`, and maybe small text summaries.

### Step 5 – Polish & impact

To make it **impactful**:

1. **One-click reproduction of the paper**

   * Preset buttons like “Reproduce Nansha experiment” / “Reproduce Zhongshan experiment” that load the exact config used in the NatCom revision (city, time windows, physics weights).

2. **Educational mode**

   * Small tooltips explaining what each physics weight does (?_cons, ?_gw, ?_prior, ...).
   * Plots that visually compare *physics on vs off* for the same city.

3. **Exportable artefacts**

   * Export a simple `.zip` “capsule”: config.json, manifest.json, key CSVs, and a README summarizing metrics.
   * This aligns with Code Ocean / reproducible science expectations.

4. **Hooks for future cities**

   * Don’t hard-code Nansha/Zhongshan in the GUI logic; treat them as examples.
   * Any city with a harmonized CSV and the right columns should be runnable via `config.py`.

---

## 5. How I’d proceed with you

Very concretely, next coding steps could be:

1. Add `run_stage1`, `run_training`, `run_tuning`, `run_inference`, `run_future_forecasts`, `run_xfer_matrix` functions to the NATCOM scripts.
2. Create `fusionlab/tools/app/geoprior/backend.py` with thin job wrappers.
3. Copy `mini_forecaster_gui.py` ? `geoprior_forecaster_gui.py`, strip TransFlow-specific code, keep the infrastructure (threads, logging, progress bar).
4. Implement the new tabs one by one, starting with *Stage-1* and *Training* (since you already use those daily).
5. Only then add “luxury” features: cross-city panel, physics diagnostics visualisation, future forecast explorer.

If you want, in the next step I can help you sketch the skeleton of `geoprior_forecaster_gui.py` (class names, signals/slots, and how to call the Stage-1 / training jobs).
