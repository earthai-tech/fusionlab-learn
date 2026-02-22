# -*- coding: utf-8 -*-
# License: BSD-3-Clause

from __future__ import annotations

import platform

from importlib import metadata
from importlib.util import find_spec
from ...styles import PRIMARY  

APP_NAME = "GeoPrior Forecaster"
APP_VERSION = "v3.2"

TAGLINE = (
    "Physics-informed subsidence forecasting with "
    "GeoPriorSubsNet, transferability, and maps."
)

DOCS_URL = (
    "https://fusion-lab.readthedocs.io/en/latest/"
    "user_guide/apps/geoprior_v3/index.html"
)
GITHUB_URL = "https://github.com/earthai-tech"
PORTFOLIO_URL = "https://lkouadio.com/"

COPYRIGHT_LINE = "© 2025 EarthAi-tech"

DEFAULT_AVATAR_PATH = "geoprior_conceptor.png"


def system_info_text() -> str:
    tfv, dev = _tf_info()
    return (
        f"Python {platform.python_version()}, "
        f"TensorFlow {tfv} ({dev}), "
        f"OS: {platform.system()} {platform.release()}."
    )


def _tf_info() -> tuple[str, str]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return ("not installed", "n/a")

    try:
        tfv = getattr(tf, "__version__", "unknown")
        gpus = tf.config.list_physical_devices("GPU")
        dev = "GPU" if gpus else "CPU"
        return (tfv, dev)
    except Exception:
        return ("unknown", "unknown")

def fusionlab_version() -> str:
    names = [
        "fusionlab-learn",
        "fusionlab_learn",
        "fusionlab",
    ]
    for n in names:
        try:
            return metadata.version(n)
        except Exception:
            continue
    return "dev"


def has_kdiagram() -> bool:
    return find_spec("kdiagram") is not None


def has_qtwebengine() -> bool:
    return (
        find_spec("PyQt5.QtWebEngineWidgets")
        is not None
    )


def install_profile() -> str:
    kd = has_kdiagram()
    web = has_qtwebengine()
    if kd and web:
        return "full"
    if kd or web:
        return "partial"
    return "base"

# ---------------------------------------------------------------------
# Cite / citation blocks (Qt-safe HTML + brace-safe BibTeX handling)
# ---------------------------------------------------------------------

# 1) Primary paper BibTeX (TEXT ONLY). Never run .format() on this.
CITATION_BIBTEX_TEXT = (
    "@unpublished{kouadio_geopriorsubsnet_nature_2025,\n"
    "  author  = {Kouadio, Kouao Laurent and Liu, Rong and Jiang, Shiyu and\n"
    "             Liu, Zhuo and Kouamelan, Serge and Liu, Wenxiang and\n"
    "             Qing, Zhanhui and Zheng, Zhiwen},\n"
    "  title   = {Physics-Informed Deep Learning Reveals Divergent Urban Land Subsidence Regimes},\n"
    "  journal = {Nature Communications},\n"
    "  note    = {Submitted; update this entry when the official citation is published},\n"
    "  year    = {2025}\n"
    "}\n"
)

# 2) Primary paper NOTE (HTML ONLY). No BibTeX here (avoid duplication).
CITATION_NOTE_HTML = (
    f"<span style='font-weight:700; color:{PRIMARY};'>Primary paper</span>"
    "<br><br>"
    "<span>Please cite this manuscript when reporting scientific "
    "results produced with GeoPrior/GeoPriorSubsNet. Update the "
    "entry once the article is published (volume/pages/DOI).</span>"
    "<br><br>"
    "<span><span style='font-weight:600;'>Manuscript:</span> "
    "Physics-Informed Deep Learning Reveals Divergent Urban Land "
    "Subsidence Regimes.</span><br>"
    "<span><span style='font-weight:600;'>Venue:</span> "
    "Nature Communications.</span><br>"
    "<span><span style='font-weight:600;'>Status:</span> "
    "Submitted (2025).</span><br><br>"
    f"<span style='font-weight:700; color:{PRIMARY};'>Note</span>"
    f"<span style='color:{PRIMARY};'>:</span> "
    "<span>The BibTeX entry is provided in the BibTeX section. "
    "Please update it after publication (volume/pages/DOI).</span>"
)

# Backward-compatible alias if other pages still import CITATION_HTML
CITATION_HTML = CITATION_NOTE_HTML

# 3) General cite text blocks (HTML)
CITE_HERO_HTML = (
    f"<span style='font-weight:700; color:{PRIMARY};'>Citing this work</span>"
    "<br><br>"
    "<span>If you use GeoPrior Forecaster or FusionLab in a paper, "
    "report, or public benchmark, please cite the software and the "
    "corresponding scientific references (when applicable).</span>"
    "<br><br>"
    f"<span style='font-weight:700; color:{PRIMARY};'>Tip</span>"
    f"<span style='color:{PRIMARY};'>:</span> "
    "<span>include the version number and (if possible) the Git "
    "commit or release tag used for your experiments.</span>"
)

CITE_WHEN_HTML = (
    f"<span style='font-weight:700; color:{PRIMARY};'>When to cite</span>"
    "<br><br>"
    f"<span style='font-weight:700; color:{PRIMARY};'>•</span> "
    "<span>Using GeoPrior Forecaster to run experiments and produce "
    "figures/tables.</span><br>"
    f"<span style='font-weight:700; color:{PRIMARY};'>•</span> "
    "<span>Using FusionLab models, utilities, or pipelines in "
    "reproducible research.</span><br>"
    f"<span style='font-weight:700; color:{PRIMARY};'>•</span> "
    "<span>Reporting transferability matrices, reliability/sharpness "
    "diagnostics, or Stage-1/Stage-2 artifacts.</span>"
)

# 4) Software BibTeX entries (TEXT)
CITE_BIBTEX_FUSIONLAB = (
    "@software{fusionlab_learn,\n"
    "  title        = {FusionLab: Next-Gen Temporal Fusion Architectures for Time-Series Forecasting},\n"
    "  author       = {Kouadio, Laurent},\n"
    "  year         = {2025},\n"
    "  license      = {BSD-3-Clause},\n"
    "  url          = {https://github.com/earthai-tech/fusionlab-learn}\n"
    "}\n"
)

CITE_BIBTEX_GEOPRIOR = (
    "@software{geoprior_forecaster,\n"
    "  title        = {GeoPrior Forecaster: Physics-Informed Subsidence Forecasting GUI},\n"
    "  author       = {Kouadio, Laurent},\n"
    "  year         = {2025},\n"
    "  license      = {BSD-3-Clause},\n"
    "  url          = {https://fusion-lab.readthedocs.io}\n"
    "}\n"
)

# 5) Big BibTeX card (HTML). Use f-strings only (brace-safe).
CITE_BIBTEX_HTML = (
    f"<span style='font-weight:700; color:{PRIMARY};'>BibTeX</span>"
    "<br><br>"
    "<span>Copy the entries below (and update year/version and "
    "the paper DOI once published):</span><br><br>"
    "<pre style=\"font-family:Consolas,'DejaVu Sans Mono',monospace;"
    " font-size:11px; margin:0;\">"
    f"{CITE_BIBTEX_FUSIONLAB}\n"
    f"{CITE_BIBTEX_GEOPRIOR}\n"
    f"{CITATION_BIBTEX_TEXT}"
    "</pre>"
)

CITE_ACK_HTML = (
    f"<span style='font-weight:700; color:{PRIMARY};'>Acknowledgement</span>"
    "<br><br>"
    "<span>If you cite GeoPrior/FusionLab, consider adding a brief "
    "acknowledgement describing the dataset source and the computing "
    "environment (CPU/GPU, TF/Keras versions). This improves "
    "reproducibility.</span>"
)

# 6) Clipboard text (all three entries)
CITE_COPY_TEXT = (
    CITE_BIBTEX_FUSIONLAB
    + "\n"
    + CITE_BIBTEX_GEOPRIOR
    + "\n"
    + CITATION_BIBTEX_TEXT
)

# --- page body stubs (fill progressively) -------------------

OVERVIEW_HTML = (
    "GeoPrior is a physics-informed forecasting toolkit "
    "for urban land subsidence.<br><br>"
    "<b>v3.2</b> introduces dedicated tabs for "
    "<b>Data</b>, <b>Setup</b>, <b>Preprocess</b>, "
    "and <b>Map</b> to make workflows structured, "
    "repeatable, and easier to inspect."
)

QUICKSTART_HTML = (
    "<ol>"
    "<li><b>Data</b>: select dataset + map columns.</li>"
    "<li><b>Setup</b>: define experiment context.</li>"
    "<li><b>Preprocess</b>: Stage-1 manifests/NPZ.</li>"
    "<li><b>Train/Tune</b>: Stage-2 train or search.</li>"
    "<li><b>Inference/Xfer</b>: evaluate + export.</li>"
    "<li><b>Map</b>: spatial inspection + analytics.</li>"
    "<li><b>Results</b>: browse + archive runs.</li>"
    "</ol>"
)

TABS_GUIDE_HTML = (
    "This page will become a complete tab-by-tab guide.<br>"
    "For now, it is a placeholder you can fill gradually."
)

TROUBLESHOOT_HTML = (
    "This page will host common issues and fixes:<br>"
    "results root selection, manifest mismatch, "
    "missing columns, export locations, etc."
)

CITATION_HTML = (
    f"<span style='font-weight:700; color:{PRIMARY};'>"
    "Primary paper</span>"
    "<br><br>"
    "<span><span style='font-weight:600;'>Manuscript:</span> "
    "Physics-Informed Deep Learning Reveals Divergent Urban Land "
    "Subsidence Regimes.</span><br>"
    "<span><span style='font-weight:600;'>Venue:</span> "
    "Nature Communications.</span><br>"
    "<span><span style='font-weight:600;'>Status:</span> "
    "Submitted (2025).</span><br><br>"
    f"<span style='font-weight:700; color:{PRIMARY};'>Note</span>"
    f"<span style='color:{PRIMARY};'>:</span> "
    "<span>the BibTeX entry is provided in the section below. "
    "Please update it after publication (volume/pages/DOI).</span>"
)



OVERVIEW_INTRO_HTML = (
    "GeoPrior Forecaster is the GUI companion of "
    "<b>FusionLab</b>, designed to run structured "
    "subsidence forecasting experiments with "
    "<b>GeoPriorSubsNet</b>.<br><br>"
    "It emphasizes <b>reproducibility</b> "
    "(manifests, saved configs, result folders) "
    "and <b>inspectability</b> "
    "(maps, analytics, transferability)."
)

OVERVIEW_WHATS_NEW_HTML = (
    "<b>v3.2 highlights</b><br>"
    "<ul style='margin-top:6px;'>"
    "<li><b>Data</b>: dataset library + role mapping.</li>"
    "<li><b>Setup</b>: experiment context, presets, scope.</li>"
    "<li><b>Preprocess</b>: Stage-1 manifests/NPZ reuse.</li>"
    "<li><b>Map</b>: spatial inspection + analytics panels.</li>"
    "</ul>"
)

OVERVIEW_FLOW_HTML = (
    "<b>Workflow at a glance</b><br><br>"
    "<code>"
    "Data → Setup → Preprocess → "
    "Train/Tune → Inference/Xfer → "
    "Map → Results"
    "</code><br><br>"
    "Start by preparing a stable Stage-1 "
    "<b>manifest</b>, then iterate quickly on "
    "training, tuning, inference, and transfer."
)

OVERVIEW_PROJECT_HTML = (
    "<b>Project</b><br><br>"
    f"<b>Package</b>: fusionlab-learn "
    f"{fusionlab_version()}<br>"
    "<b>License</b>: BSD-3-Clause<br>"
    "<b>Python</b>: ≥ 3.9<br>"
    "<b>GUI</b>: PyQt5<br><br>"
    "<b>Links</b><br>"
    f"<a href='{DOCS_URL}'>User guide</a><br>"
    f"<a href='{GITHUB_URL}'>GitHub</a><br>"
    f"<a href='{PORTFOLIO_URL}'>Author</a>"
)

OVERVIEW_ADDONS_HTML = (
    "<b>Optional add-ons</b><br><br>"
    f"<b>Install profile</b>: {install_profile()}<br><br>"
    "<ul style='margin-top:6px;'>"
    f"<li>k-diagram: "
    f"{'installed' if has_kdiagram() else 'not installed'}</li>"
    f"<li>QtWebEngine: "
    f"{'installed' if has_qtwebengine() else 'not installed'}</li>"
    "</ul>"
)

OVERVIEW_NEXT_HTML = (
    "<b>Next step</b><br><br>"
    "Open <b>Quickstart</b> for the recommended "
    "end-to-end workflow, then use <b>Tabs guide</b> "
    "as a reference while exploring each section."
)

# --- Quickstart page blocks ---------------------------------

QUICKSTART_HERO_HTML = (
    "Use this page as the shortest path from "
    "<b>no dataset</b> to a fully reproducible run.<br><br>"
    "Key idea: build a stable <b>Stage-1 manifest</b>, then "
    "iterate quickly on Train/Tune/Inference/Transfer."
)

QS_STEP1_HTML = (
    "<span style='font-weight:700; color:{c};'>"
    "Step 1 — Data</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Do</span>"
    "<span style='color:{c};'>:</span> "
    "<span>choose a <span style='font-weight:600;'>results root</span>, "
    "load a CSV (or select one from the dataset library), then map "
    "the column roles (time, lon/lat, subsidence, GWL, rainfall, …) "
    "and save the dataset entry.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Why</span>"
    "<span style='color:{c};'>:</span> "
    "<span>consistent role mapping prevents silent feature mismatches "
    "and ensures downstream preprocessing and inference remain "
    "compatible.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Output</span>"
    "<span style='color:{c};'>:</span> "
    "<span>a reusable dataset entry with stable column roles.</span>"
).format(c=PRIMARY)


QS_STEP2_HTML = (
    "<span style='font-weight:700; color:{c};'>"
    "Step 2 — Setup</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Do</span>"
    "<span style='color:{c};'>:</span> "
    "<span>select a preset/scope and define the temporal window "
    "(train end year, forecast start year, forecast horizon, and "
    "look-back/time steps). Review core run options before moving on."
    "</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Why</span>"
    "<span style='color:{c};'>:</span> "
    "<span>Setup establishes the experiment context used throughout "
    "Stage-1 and Stage-2, making runs repeatable and auditable.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Output</span>"
    "<span style='color:{c};'>:</span> "
    "<span>a coherent experiment configuration ready for Stage-1.</span>"
).format(c=PRIMARY)


QS_STEP3_HTML = (
    "<span style='font-weight:700; color:{c};'>"
    "Step 3 — Preprocess (Stage-1)</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Do</span>"
    "<span style='color:{c};'>:</span> "
    "<span>run Stage-1 to generate sequences/NPZ and write a "
    "<span style='font-weight:600;'>manifest</span>. Optionally build "
    "a <span style='font-weight:600;'>future NPZ</span> for "
    "forecast-only inference.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Why</span>"
    "<span style='color:{c};'>:</span> "
    "<span>the manifest is the contract that defines exactly what the "
    "model expects (roles, window, scaling/normalization, shapes), "
    "and it is required for correct inference.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Output</span>"
    "<span style='color:{c};'>:</span> "
    "<span>a Stage-1 folder, a manifest file, and optionally a future "
    "NPZ for fast inference.</span>"
).format(c=PRIMARY)


QS_STEP4_HTML = (
    "<span style='font-weight:700; color:{c};'>"
    "Step 4 — Train (Stage-2)</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Do</span>"
    "<span style='color:{c};'>:</span> "
    "<span>set training parameters (epochs, batch size, learning rate) "
    "and physics weights, then run Stage-2 training. Enable extra "
    "training diagnostics if you need deeper evaluation.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Why</span>"
    "<span style='color:{c};'>:</span> "
    "<span>physics weights control the trade-off between predictive "
    "fit and physical consistency; start with conservative values and "
    "iterate based on diagnostics.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Output</span>"
    "<span style='color:{c};'>:</span> "
    "<span>a trained <code>.keras</code> model plus logs, plots, and "
    "saved run artifacts.</span>"
).format(c=PRIMARY)


QS_STEP5_HTML = (
    "<span style='font-weight:700; color:{c};'>"
    "Step 5 — Tune (optional)</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Do</span>"
    "<span style='color:{c};'>:</span> "
    "<span>define hyper-parameter search ranges (architecture, "
    "regularization, physics switches) and run a capped number of "
    "trials around a base configuration.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Why</span>"
    "<span style='color:{c};'>:</span> "
    "<span>tuning improves robustness without manual guesswork and "
    "helps identify stable configurations across datasets.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Output</span>"
    "<span style='color:{c};'>:</span> "
    "<span>the best trial configuration and its corresponding model.</span>"
).format(c=PRIMARY)


QS_STEP6_HTML = (
    "<span style='font-weight:700; color:{c};'>"
    "Step 6 — Inference</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Do</span>"
    "<span style='color:{c};'>:</span> "
    "<span>load a <code>.keras</code> model and the Stage-1 "
    "<span style='font-weight:600;'>manifest</span>, select a dataset "
    "split (train/val/test/future), run prediction, and export plots "
    "and CSV outputs. Apply calibration if required.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Why</span>"
    "<span style='color:{c};'>:</span> "
    "<span>inference is only reliable when the model and manifest "
    "match exactly; otherwise feature ordering/scaling can invalidate "
    "results.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Output</span>"
    "<span style='color:{c};'>:</span> "
    "<span>predictions, evaluation summaries, and exported artifacts.</span>"
).format(c=PRIMARY)


QS_STEP7_HTML = (
    "<span style='font-weight:700; color:{c};'>"
    "Step 7 — Transferability (optional)</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Do</span>"
    "<span style='color:{c};'>:</span> "
    "<span>choose a source city (A) and target city (B), select "
    "strategies/splits, configure calibration and rescaling mode, "
    "then run the transferability evaluation.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Why</span>"
    "<span style='color:{c};'>:</span> "
    "<span>transferability quantifies cross-city generalization and "
    "helps detect semantic feature mismatches that can compromise "
    "deployment.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Output</span>"
    "<span style='color:{c};'>:</span> "
    "<span>a cross-city report (matrix) with summary artifacts.</span>"
).format(c=PRIMARY)


QS_MAP_RESULTS_HTML = (
    "<span style='font-weight:700; color:{c};'>Validate & share</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Map</span>"
    "<span style='color:{c};'>:</span> "
    "<span>sanity-check spatial patterns, hotspots, and reliability "
    "diagnostics.</span><br>"
    "<span style='font-weight:700; color:{c};'>Results</span>"
    "<span style='color:{c};'>:</span> "
    "<span>browse run folders, inspect manifests, and download ZIP "
    "archives for sharing or archiving.</span>"
).format(c=PRIMARY)


QS_TIPS_HTML = (
    "<span style='font-weight:700; color:{c};'>Pro tips</span>"
    "<br><br>"
    "<ul style='margin-top:6px;'>"
    "<li><span>Reuse <span style='font-weight:600;'>Stage-1</span> "
    "whenever possible — it is the fastest way to iterate while "
    "preserving data semantics.</span></li>"
    "<li><span>When results look suspicious, first verify the "
    "<span style='font-weight:600;'>results root</span>, then verify "
    "the <span style='font-weight:600;'>manifest</span> and column "
    "role mapping.</span></li>"
    "<li><span>Start with a short run (few epochs) to validate the "
    "pipeline, then scale up once everything is consistent.</span></li>"
    "</ul>"
).format(c=PRIMARY)

TABS_GUIDE_HERO_HTML = (
    "This guide explains what each tab controls, what it "
    "produces, and where to look when something feels off."
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Rule of "
    "thumb</span><span style='color:{c};'>:</span> "
    "<span>build a stable Stage-1 </span>"
    "<span style='font-weight:600;'>manifest</span>"
    "<span>, then iterate quickly on Train/Tune/Inference/"
    "Transferability.</span>"
).format(c=PRIMARY)

TABS_GUIDE_LEGEND_HTML = (
    "<span style='font-weight:700; color:{c};'>Key artifacts"
    "</span><span style='color:{c};'>:</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Manifest</span>"
    "<span style='color:{c};'>:</span> "
    "<span>Stage-1 contract (roles, window, scaling, shapes)."
    "</span><br>"
    "<span style='font-weight:700; color:{c};'>NPZ</span>"
    "<span style='color:{c};'>:</span> "
    "<span>packed arrays used by training/inference.</span><br>"
    "<span style='font-weight:700; color:{c};'>Model</span>"
    "<span style='color:{c};'>:</span> "
    "<span><code>.keras</code> artifact produced by Stage-2."
    "</span><br>"
    "<span style='font-weight:700; color:{c};'>Results root"
    "</span><span style='color:{c};'>:</span> "
    "<span>the parent folder that stores all runs.</span>"
).format(c=PRIMARY)

TAB_DATA_HTML = (
    "<span style='font-weight:700; color:{c};'>Purpose</span>"
    "<span style='color:{c};'>:</span> "
    "<span>select a dataset and map column roles used across "
    "the entire pipeline.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Key actions</span>"
    "<span style='color:{c};'>:</span> "
    "<span>load CSV / choose from library, map roles (time, "
    "lon/lat, subsidence, GWL, rainfall, …), save dataset "
    "entry.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Outputs</span>"
    "<span style='color:{c};'>:</span> "
    "<span>a reusable dataset entry and consistent role mapping."
    "</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Pitfalls</span>"
    "<span style='color:{c};'>:</span> "
    "<span>incorrect role mapping or missing required columns "
    "causes later manifest mismatches.</span>"
).format(c=PRIMARY)

TAB_SETUP_HTML = (
    "<span style='font-weight:700; color:{c};'>Purpose</span>"
    "<span style='color:{c};'>:</span> "
    "<span>define experiment identity and the temporal window "
    "used by Stage-1 and Stage-2.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Key actions</span>"
    "<span style='color:{c};'>:</span> "
    "<span>preset/scope selection, names, train end year, "
    "forecast start year, horizon, look-back/time steps.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Outputs</span>"
    "<span style='color:{c};'>:</span> "
    "<span>a coherent experiment configuration (store-backed)."
    "</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Pitfalls</span>"
    "<span style='color:{c};'>:</span> "
    "<span>changing the temporal window requires rebuilding "
    "Stage-1 (new manifest/NPZ).</span>"
).format(c=PRIMARY)

TAB_PREPROCESS_HTML = (
    "<span style='font-weight:700; color:{c};'>Purpose</span>"
    "<span style='color:{c};'>:</span> "
    "<span>run Stage-1 to produce sequences/NPZ plus a manifest."
    "</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Key actions</span>"
    "<span style='color:{c};'>:</span> "
    "<span>build Stage-1 workspace, reuse or rebuild, optionally "
    "build future NPZ for forecast-only inference.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Outputs</span>"
    "<span style='color:{c};'>:</span> "
    "<span>Stage-1 folder, manifest, NPZ files, optional future "
    "NPZ.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Pitfalls</span>"
    "<span style='color:{c};'>:</span> "
    "<span>if roles/features/window change, reusing an old "
    "manifest will invalidate inference.</span>"
).format(c=PRIMARY)

TAB_TRAIN_HTML = (
    "<span style='font-weight:700; color:{c};'>Purpose</span>"
    "<span style='color:{c};'>:</span> "
    "<span>Stage-2 training of GeoPriorSubsNet under physics-"
    "informed loss constraints.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Key actions</span>"
    "<span style='color:{c};'>:</span> "
    "<span>epochs/batch/LR, physics weights, optional extra "
    "metrics, run training.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Outputs</span>"
    "<span style='color:{c};'>:</span> "
    "<span><code>.keras</code> model, logs, plots, run summary."
    "</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Pitfalls</span>"
    "<span style='color:{c};'>:</span> "
    "<span>physics weights too large can dominate learning; "
    "too small can reduce physical consistency.</span>"
).format(c=PRIMARY)

TAB_TUNE_HTML = (
    "<span style='font-weight:700; color:{c};'>Purpose</span>"
    "<span style='color:{c};'>:</span> "
    "<span>hyper-parameter search around a base configuration."
    "</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Key actions</span>"
    "<span style='color:{c};'>:</span> "
    "<span>define search ranges (architecture/regularization/"
    "physics), cap max trials, run tuner.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Outputs</span>"
    "<span style='color:{c};'>:</span> "
    "<span>best trial config, best model, tuner report.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Pitfalls</span>"
    "<span style='color:{c};'>:</span> "
    "<span>too broad ranges + too many trials = long runs; keep "
    "search space meaningful and bounded.</span>"
).format(c=PRIMARY)

TAB_INFERENCE_HTML = (
    "<span style='font-weight:700; color:{c};'>Purpose</span>"
    "<span style='color:{c};'>:</span> "
    "<span>run prediction using a trained model + Stage-1 "
    "manifest.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Key actions</span>"
    "<span style='color:{c};'>:</span> "
    "<span>load <code>.keras</code>, load manifest, choose split "
    "(train/val/test/future), optional calibration, export "
    "CSV/plots.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Outputs</span>"
    "<span style='color:{c};'>:</span> "
    "<span>predictions, evaluation summaries, exported artifacts."
    "</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Pitfalls</span>"
    "<span style='color:{c};'>:</span> "
    "<span>model + manifest must match exactly (roles/order/"
    "scaling); mismatches produce invalid outputs.</span>"
).format(c=PRIMARY)

TAB_TRANSFER_HTML = (
    "<span style='font-weight:700; color:{c};'>Purpose</span>"
    "<span style='color:{c};'>:</span> "
    "<span>evaluate cross-city generalization (A→B) under "
    "selected strategies and calibration modes.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Key actions</span>"
    "<span style='color:{c};'>:</span> "
    "<span>select cities, splits, strategies, rescaling mode, "
    "calibration, run report.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Outputs</span>"
    "<span style='color:{c};'>:</span> "
    "<span>transfer matrix (CSV/JSON) + summary panels.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Pitfalls</span>"
    "<span style='color:{c};'>:</span> "
    "<span>semantic feature mismatch can inflate/deflate scores; "
    "use strict checks when auditing.</span>"
).format(c=PRIMARY)

TAB_RESULTS_HTML = (
    "<span style='font-weight:700; color:{c};'>Purpose</span>"
    "<span style='color:{c};'>:</span> "
    "<span>browse all runs under results root and inspect run "
    "artifacts.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Key actions</span>"
    "<span style='color:{c};'>:</span> "
    "<span>filter by city/workflow, open run folders, view key "
    "files, export ZIP archives.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Outputs</span>"
    "<span style='color:{c};'>:</span> "
    "<span>archives for sharing and reproducible audit trails."
    "</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Pitfalls</span>"
    "<span style='color:{c};'>:</span> "
    "<span>mixed results roots can confuse run discovery; keep "
    "a consistent root per project.</span>"
).format(c=PRIMARY)

TAB_MAP_HTML = (
    "<span style='font-weight:700; color:{c};'>Purpose</span>"
    "<span style='color:{c};'>:</span> "
    "<span>spatial inspection and analytics for datasets and "
    "model outputs.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Key actions</span>"
    "<span style='color:{c};'>:</span> "
    "<span>select layers, adjust view controls, explore analytics "
    "panels (sharpness/reliability/inspector).</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Outputs</span>"
    "<span style='color:{c};'>:</span> "
    "<span>visual validation of patterns and diagnostics.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Pitfalls</span>"
    "<span style='color:{c};'>:</span> "
    "<span>if map looks empty, verify dataset selection and that "
    "expected columns exist (lon/lat/value).</span>"
).format(c=PRIMARY)

TAB_TOOLS_HTML = (
    "<span style='font-weight:700; color:{c};'>Purpose</span>"
    "<span style='color:{c};'>:</span> "
    "<span>utilities for reproducibility and debugging.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Key actions</span>"
    "<span style='color:{c};'>:</span> "
    "<span>script generation, inspectors, batch helpers, and "
    "export tools.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Outputs</span>"
    "<span style='color:{c};'>:</span> "
    "<span>reproducible CLI scripts and structured reports.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Pitfalls</span>"
    "<span style='color:{c};'>:</span> "
    "<span>tools may assume a valid results root and consistent "
    "manifest; verify inputs first.</span>"
).format(c=PRIMARY)

TABS_GUIDE_FOOT_HTML = (
    "<span style='font-weight:700; color:{c};'>When debugging</span>"
    "<span style='color:{c};'>:</span> "
    "<span>check </span>"
    "<span style='font-weight:600;'>results root → dataset roles → "
    "manifest → model</span>"
    "<span> in that order. This resolves most issues quickly.</span>"
).format(c=PRIMARY)

TABS_GUIDE_FILES_HTML = (
    "<span style='font-weight:700; color:{c};'>Where files go</span>"
    "<br><br>"
    "<span>All outputs are written under the "
    "<span style='font-weight:600;'>results root</span>. "
    "The exact subfolders can vary by preset/scope, but the "
    "structure below is the mental model to use when auditing "
    "runs.</span><br><br>"
    "<pre style=\"font-family:Consolas,'DejaVu Sans Mono',monospace;"
    " font-size:11px; margin:0;\">"
    "results_root/\n"
    "  city_or_dataset/\n"
    "    manifest.json           # roles/window/scaling/shapes\n"
    "    artifacts/              # preprocessing outputs\n"
    "      *.npz                 # train/val/test arrays\n"
    "      future/               # optional forecast-only NPZ\n"
    "    train/                  # Stage-2 training runs\n"
    "      run_YYYYMMDD_HHMMSS/\n"
    "        model.keras\n"
    "        logs.txt\n"
    "        plots/\n"
    "        metrics.json\n"
    "    tuning/                   # hyper-parameter search\n"
    "      run_YYYYMMDD_HHMMSS/\n"
    "        best_model.keras\n"
    "        trials.csv\n"
    "        report.json\n"
    "    inference/              # predictions & exports\n"
    "      run_YYYYMMDD_HHMMSS/\n"
    "        predictions.csv\n"
    "        plots/\n"
    "        summary.json\n"
    "  xfer/                   # transferability outputs\n"
    "    A_to_B_YYYYMMDD_HHMMSS/\n"
    "      matrix.csv\n"
    "      matrix.json\n"
    "      panels/\n"
    "</pre>"
    "<br>"
    "<span style='font-weight:700; color:{c};'>Tip</span>"
    "<span style='color:{c};'>:</span> "
    "<span>when something looks wrong, inspect "
    "<span style='font-weight:600;'>stage1/manifest</span> first "
    "and confirm the model was trained against the same Stage-1.</span>"
).format(c=PRIMARY)

# --- Troubleshooting page blocks -----------------------------

TROUBLE_HERO_HTML = (
    "This page lists the most common issues and the fastest "
    "fixes. Most problems come from one of four causes:<br>"
    "wrong <span style='font-weight:600;'>results root</span>, "
    "wrong <span style='font-weight:600;'>dataset roles</span>, "
    "stale <span style='font-weight:600;'>Stage-1 manifest</span>, "
    "or a mismatched <span style='font-weight:600;'>model</span>."
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Debug order</span>"
    "<span style='color:{c};'>:</span> "
    "<span>results root → roles → manifest → model.</span>"
).format(c=PRIMARY)

TROUBLE_FIRST_CHECKS_HTML = (
    "<span style='font-weight:700; color:{c};'>Before debugging</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>1</span>"
    "<span style='color:{c};'>.</span> "
    "<span>confirm the "
    "<span style='font-weight:600;'>results root</span> "
    "is the one you expect.</span><br>"
    "<span style='font-weight:700; color:{c};'>2</span>"
    "<span style='color:{c};'>.</span> "
    "<span>confirm a dataset is loaded and roles are mapped.</span><br>"
    "<span style='font-weight:700; color:{c};'>3</span>"
    "<span style='color:{c};'>.</span> "
    "<span>if you changed window/features, rebuild Stage-1.</span><br>"
    "<span style='font-weight:700; color:{c};'>4</span>"
    "<span style='color:{c};'>.</span> "
    "<span>use <span style='font-weight:600;'>Dry run</span> to "
    "verify what would execute without launching long jobs.</span>"
).format(c=PRIMARY)

TROUBLE_DATASET_NOT_LOADED_HTML = (
    "<span style='font-weight:700; color:{c};'>Symptom</span>"
    "<span style='color:{c};'>:</span> "
    "<span>“No dataset loaded” or the library list looks empty.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Cause</span>"
    "<span style='color:{c};'>:</span> "
    "<span>dataset path not selected, wrong results root, or the "
    "CSV does not match expected columns/roles.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Fix</span>"
    "<span style='color:{c};'>:</span> "
    "<span>go to <span style='font-weight:600;'>Data</span>, load a "
    "CSV (or pick one from the library), map roles, then save the "
    "dataset entry. If the library is empty, verify results root.</span>"
).format(c=PRIMARY)

TROUBLE_RESULTS_ROOT_HTML = (
    "<span style='font-weight:700; color:{c};'>Symptom</span>"
    "<span style='color:{c};'>:</span> "
    "<span>runs are “missing”, cities don’t appear, or the dataset "
    "library is unexpectedly empty.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Cause</span>"
    "<span style='color:{c};'>:</span> "
    "<span>the GUI is pointing to a different results root than the "
    "one used for those runs.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Fix</span>"
    "<span style='color:{c};'>:</span> "
    "<span>set the correct results root (top-level location where "
    "all run folders are stored), then refresh Results/Data views.</span>"
).format(c=PRIMARY)

TROUBLE_MANIFEST_MISMATCH_HTML = (
    "<span style='font-weight:700; color:{c};'>Symptom</span>"
    "<span style='color:{c};'>:</span> "
    "<span>inference fails, shapes mismatch, or outputs look wrong "
    "even though the model loads.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Cause</span>"
    "<span style='color:{c};'>:</span> "
    "<span>the <span style='font-weight:600;'>Stage-1 manifest</span> "
    "does not match the model’s expected features/window/scaling.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Fix</span>"
    "<span style='color:{c};'>:</span> "
    "<span>use the manifest created for the same Stage-1 as the model "
    "training run. If you changed roles/window/features, rebuild Stage-1 "
    "and retrain (or rerun inference using the correct pair).</span>"
).format(c=PRIMARY)

TROUBLE_STAGE1_REBUILD_HTML = (
    "<span style='font-weight:700; color:{c};'>When to rebuild Stage-1</span>"
    "<br><br>"
    "<span>Rebuild Stage-1 whenever you change:</span><br>"
    "<span style='font-weight:700; color:{c};'>•</span> "
    "<span>train end / forecast start / horizon / look-back</span><br>"
    "<span style='font-weight:700; color:{c};'>•</span> "
    "<span>feature selection or role mapping</span><br>"
    "<span style='font-weight:700; color:{c};'>•</span> "
    "<span>scaling/normalization choices</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Reason</span>"
    "<span style='color:{c};'>:</span> "
    "<span>Stage-1 defines the contract used by training and inference.</span>"
).format(c=PRIMARY)

TROUBLE_MODEL_LOAD_HTML = (
    "<span style='font-weight:700; color:{c};'>Symptom</span>"
    "<span style='color:{c};'>:</span> "
    "<span><code>.keras</code> model fails to load or errors mention "
    "custom layers/objects.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Cause</span>"
    "<span style='color:{c};'>:</span> "
    "<span>version mismatch (TF/Keras) or missing custom objects "
    "required by the model.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Fix</span>"
    "<span style='color:{c};'>:</span> "
    "<span>use the same environment used for training (TF/Keras), "
    "and ensure FusionLab/GeoPrior code is importable. If you updated "
    "the code, try loading within the same version tag/commit as training.</span>"
).format(c=PRIMARY)

TROUBLE_EMPTY_MAP_HTML = (
    "<span style='font-weight:700; color:{c};'>Symptom</span>"
    "<span style='color:{c};'>:</span> "
    "<span>Map is blank or no points/layers appear.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Cause</span>"
    "<span style='color:{c};'>:</span> "
    "<span>missing coordinate/value roles, wrong selected dataset, or "
    "the current layer is filtered out (scope/time/selection).</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Fix</span>"
    "<span style='color:{c};'>:</span> "
    "<span>confirm lon/lat/value roles in Data, verify the selected "
    "dataset, then enable a simple layer (e.g., points). If still empty, "
    "open Inspector diagnostics in the Map panel.</span>"
).format(c=PRIMARY)

TROUBLE_NO_RUNS_HTML = (
    "<span style='font-weight:700; color:{c};'>Symptom</span>"
    "<span style='color:{c};'>:</span> "
    "<span>Results tab shows “no runs found” or cannot discover cities.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Cause</span>"
    "<span style='color:{c};'>:</span> "
    "<span>wrong results root, missing folder permissions, or runs were "
    "written to a different location.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Fix</span>"
    "<span style='color:{c};'>:</span> "
    "<span>set the correct results root, verify it is writable, then "
    "refresh. Use file explorer to confirm run folders exist on disk.</span>"
).format(c=PRIMARY)

TROUBLE_TUNER_HTML = (
    "<span style='font-weight:700; color:{c};'>Symptom</span>"
    "<span style='color:{c};'>:</span> "
    "<span>tuning is extremely slow or produces unstable results.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Cause</span>"
    "<span style='color:{c};'>:</span> "
    "<span>search space too wide, max trials too high, or heavy physics "
    "constraints for every trial.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Fix</span>"
    "<span style='color:{c};'>:</span> "
    "<span>start with a small, meaningful search space and a low trial "
    "cap. Use short epochs for exploration, then rerun training with the "
    "best config.</span>"
).format(c=PRIMARY)

TROUBLE_XFER_HTML = (
    "<span style='font-weight:700; color:{c};'>Symptom</span>"
    "<span style='color:{c};'>:</span> "
    "<span>transferability scores look unrealistic or inconsistent.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Cause</span>"
    "<span style='color:{c};'>:</span> "
    "<span>feature semantics differ between cities (ordering, scaling, "
    "missing variables), or calibration/rescaling settings are not aligned.</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Fix</span>"
    "<span style='color:{c};'>:</span> "
    "<span>verify both cities use consistent roles and comparable feature "
    "definitions. Prefer strict checks when auditing, and inspect transfer "
    "artifacts in Results.</span>"
).format(c=PRIMARY)

TROUBLE_EXPORT_HTML = (
    "<span style='font-weight:700; color:{c};'>Symptom</span>"
    "<span style='color:{c};'>:</span> "
    "<span>expected CSV/plots are not found after a run.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Cause</span>"
    "<span style='color:{c};'>:</span> "
    "<span>export not enabled, wrong run folder selected, or outputs were "
    "written to a different results root.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Fix</span>"
    "<span style='color:{c};'>:</span> "
    "<span>open Results and locate the run folder; check the run’s summary "
    "and plots/export subfolders. Confirm results root and rerun if needed.</span>"
).format(c=PRIMARY)

TROUBLE_PERF_HTML = (
    "<span style='font-weight:700; color:{c};'>Performance tips</span>"
    "<br><br>"
    "<span style='font-weight:700; color:{c};'>Do</span>"
    "<span style='color:{c};'>:</span> "
    "<span>validate the pipeline with short runs first, reuse Stage-1, "
    "and keep tuning bounded.</span><br><br>"
    "<span style='font-weight:700; color:{c};'>Avoid</span>"
    "<span style='color:{c};'>:</span> "
    "<span>rebuilding Stage-1 repeatedly unless you changed the contract "
    "(roles/window/scaling).</span>"
).format(c=PRIMARY)

TROUBLE_FOOT_HTML = (
    "<span style='font-weight:700; color:{c};'>Still stuck?</span>"
    "<span style='color:{c};'>:</span> "
    "<span>open the bottom console/log panel, capture the error message, "
    "and note the active results root and the manifest/model paths. "
    "Those four pieces usually explain the issue immediately.</span>"
).format(c=PRIMARY)


