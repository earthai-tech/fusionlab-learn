# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# Tools tab for GeoPrior GUI.
#
# Provides a left-hand "Tool Navigator" and a right-hand workspace
# with a QStackedWidget hosting various utility panels related to
# GeoPriorSubsNet (data inspection, manifests, diagnostics,
# environment checks, etc.).
#
# At this stage, the tools are simple placeholders; the goal is to
# stabilise the layout and wiring. Controllers and real logic can be
# added incrementally.

from __future__ import annotations

from pathlib import Path 
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QPushButton,
    QFrame,
    QSizePolicy,
)

from .tools import (
    DatasetExplorerTool,
    EnvironmentCheckTool,
    FeatureInspectorTool,
    ConfigDiffTool,
    Stage1ManagerTool,
    ManifestBrowserTool,
    ReproduceRunHelperTool,
    BuildNPZTool,
    MetricsDashboardTool,
    RunComparisonTool,
    PhysicsDiagnosticsTool,
    IdentifiabilityTool,
)


# ----------------------------------------------------------------------
# Tool specification
# ----------------------------------------------------------------------


@dataclass
class ToolSpec:
    """
    Simple descriptor for a tool entry in the Tools tab navigator.

    Attributes
    ----------
    tool_id : str
        Unique identifier for the tool (used as a key).
    title : str
        Short title displayed in the navigator.
    group : str
        Logical section name (Data & Config, Diagnostics, etc.).
    description : str
        One-line human description shown in the workspace footer.
    factory : Callable[[], QWidget]
        Factory returning a QWidget instance for this tool.
    """

    tool_id: str
    title: str
    group: str
    description: str
    factory: Optional[Callable[["ToolsTab"], QWidget]] = None
    needs_log: bool = True


# ----------------------------------------------------------------------
# Placeholder tool widgets (view-only for now)
# ----------------------------------------------------------------------


class _ToolPlaceholder(QWidget):
    """
    Minimal placeholder widget for a tool.

    This is intentionally dumb: it only shows a title and a few lines of
    explanatory text. Real tools can later replace this with dedicated
    classes (DatasetExplorerTool, MetricsDashboardTool, etc.).
    """

    def __init__(self, title: str, text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title_lbl = QLabel(f"<h2>{title}</h2>", self)
        title_lbl.setTextFormat(Qt.RichText)

        body_lbl = QLabel(text, self)
        body_lbl.setWordWrap(True)
        body_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout.addWidget(title_lbl)
        layout.addWidget(body_lbl)
        layout.addStretch(1)


def _make_placeholder_tool(title: str, desc: str) -> QWidget:
    return _ToolPlaceholder(title, desc)


# ----------------------------------------------------------------------
# Default tools to expose in the tab
# ----------------------------------------------------------------------

def default_tool_specs(
    *,
    app_ctx: object | None = None,
    store: object | None = None,
    geo_cfg: object | None = None,
) -> List[ToolSpec]:

    """
    Return the list of tool specs that populate the navigator.

    This encodes the high-level organisation of the Tools tab:
    grouped sections with individual tools. Later, each placeholder
    can be swapped for a real tool widget.
    """
    def _make_ident_tool() -> QWidget:
        return IdentifiabilityTool(
            app_ctx=app_ctx,
            store=store,
            geo_cfg=geo_cfg,
        )

    return [
        # --- Data & Config ------------------------------------------------
        ToolSpec(
            tool_id="data.dataset_explorer",  # XXX TODO: already Done 
            group="Data & Config",
            title="Dataset explorer",
            description=(
                "Inspect the active dataset: shape, temporal coverage and "
                "per-column missing values."
            ),
            factory=lambda ctx=app_ctx: DatasetExplorerTool(app_ctx=ctx),
            needs_log=False,
        ),
        ToolSpec( # XXX TODO: already Done
            tool_id="data.feature_inspector",
            group="Data & Config",
            title="Feature inspector",
            description=(
                "List features, types, ranges and roles "
                "(inputs, targets, physics variables)."
            ),
            factory=lambda ctx=app_ctx: FeatureInspectorTool(app_ctx=ctx),
            needs_log=False,   # no bottom log; use full height
        ),
        ToolSpec(# XXX TODO: already Done
            tool_id="data.config_diff",
            group="Data & Config",
            title="Config inspector & diff",
            description=(
                "Compare the current GeoPrior GUI configuration with a "
                "config JSON saved from a past run. Highlights changed "
                "keys and shows a tiny bar plot for numeric settings."
            ),
            factory=lambda ctx=app_ctx: ConfigDiffTool(app_ctx=ctx),
            needs_log=False,
        ),        
   
        # --- Runs & Manifests ---------------------------------------------
        ToolSpec(# XXX TODO: already Done
            tool_id="runs.stage1_manager",
            title="Stage-1 manager",
            group="Runs & Manifests",
            description=(
                "Inspect all available Stage-1 manifests per city, "
                "see their configuration and artifacts, and mark which "
                "run the GUI should prefer when training or running "
                "GeoPriorSubsNet."
            ),
            factory=lambda ctx=app_ctx: Stage1ManagerTool(app_ctx=ctx),
            needs_log=False,
        ),
        ToolSpec(# XXX TODO: already Done
            tool_id="runs.manifest_browser",
            title="Manifest browser & validator",
            group="Runs & Manifests",
            description=(
                "Browse train/tune/inference manifests and check their integrity."
            ),
            factory=lambda ctx=app_ctx: ManifestBrowserTool(app_ctx=ctx),
            needs_log=False,
        ),
        ToolSpec(  # XXX TODO: already Done - NPZ builder (inference-ready sequences)
            tool_id="runs.build_npz",
            title="Build NPZ Dataset",
            group="Runs & Manifests",
            description=(
                "Build NPZ sequences for GeoPriorSubsNet from the active "
                "or saved dataset, using either a JSON/manifest config "
                "or manually entered parameters."
            ),
            factory=lambda ctx=app_ctx: BuildNPZTool(app_ctx=ctx),
            needs_log=True,  # show main log console while building NPZ
        ),

        ToolSpec(# XXX TODO: already Done 
            tool_id="runs.reproduce_helper",
            title="Reproduce run helper",
            group="Runs & Manifests",
            description="Generate CLI snippets to reproduce a run.",
            factory=lambda: ReproduceRunHelperTool(app_ctx=app_ctx),
            needs_log=False,  # No need for log; use full space for commands
        ),

        # --- Diagnostics & Plots ------------------------------------------
        ToolSpec(# XXX TODO: already Done 
            tool_id="diag.metrics_dashboard",
            title="Metrics dashboard",
            group="Diagnostics & Plots",
            description=(
                "Visualise performance metrics, per-horizon diagnostics "
                "and, once available, PIT / reliability plots for "
                "GeoPriorSubsNet runs."
            ),
            factory=lambda ctx=app_ctx: MetricsDashboardTool(app_ctx=ctx),
            needs_log=False,  # we use full space for the plots
        ),

        ToolSpec(# XXX TODO: already Done
            tool_id="diag.run_comparison",
            title="Run comparison",
            group="Diagnostics & Plots",
            description=(
                "Compare metrics and high-level configuration across "
                "multiple runs to support ablations and paper figures."
            ),
            factory=lambda ctx=app_ctx: RunComparisonTool(app_ctx=ctx),
            needs_log=False,   # we use our own table + config text area
        ),
                
        ToolSpec( # XXX TODO: already Done
            tool_id="diag.physics_diagnostics",
            title="Physics diagnostics",
            group="Diagnostics & Plots",
            description="Inspect physics residuals and constraint "
                        "violations.",
            factory=lambda ctx=app_ctx: PhysicsDiagnosticsTool(app_ctx=ctx),
            needs_log=False,
        ),

        ToolSpec(
            tool_id="diag.identifiability",
            title="Identifiability (SM3)",
            group="Diagnostics & Plots",
            description=(
                "Run SM3 identifiability diagnostics from a physics "
                "payload (NPZ) with unit-safe conversions, and export "
                "JSON + figures."
            ),
            factory=lambda: _make_ident_tool(),
            needs_log=False,
        ),

        # --- System & Environment -----------------------------------------
        ToolSpec(  # XXX TODO: Done 
            tool_id="sys.env_check",
            group="Data & Config",
            title="Environment check",
            description=(
                "Check Python, TensorFlow version and GPU availability, "
                "including any device overrides."
            ),
            factory=lambda ctx=app_ctx: EnvironmentCheckTool(app_ctx=ctx),
        ),
        ToolSpec(
            tool_id="sys.device_monitor",
            title="GPU / device monitor",
            group="System & Environment",
            description="Show which device GeoPriorSubsNet will use.",
            factory=lambda: _make_placeholder_tool(
                "GPU / device monitor",
                "This panel will expose which device the model is using "
                "(CPU vs GPU), with basic information about memory "
                "usage and user options such as forcing CPU or toggling "
                "mixed precision (when supported)."
            ),
        ),
        ToolSpec(
            tool_id="sys.paths_permissions",
            title="Paths & permissions",
            group="System & Environment",
            description="Check data/results roots and write access.",
            factory=lambda: _make_placeholder_tool(
                "Paths & permissions",
                "Here you will check that data and results directories "
                "exist and are writable, with small helpers to open "
                "them in the file explorer or test write permission."
            ),
        ),

        # --- Advanced -----------------------------------------------------
        ToolSpec(
            tool_id="adv.json_viewer",
            title="JSON viewer (advanced)",
            group="Advanced",
            description="Inspect manifest/config JSON files "
                        "(read-only by default).",
            factory=lambda: _make_placeholder_tool(
                "JSON viewer (advanced)",
                "Read-only viewer for manifest/config JSON files, with "
                "syntax highlighting. Manual editing can be added later "
                "behind an explicit 'expert mode' toggle.",
            ),
        ),
        ToolSpec(
            tool_id="adv.script_generator",
            title="Script / batch generator",
            group="Advanced",
            description="Generate small scripts for batch runs.",
            factory=lambda: _make_placeholder_tool(
                "Script / batch generator",
                "This tool will generate small Python or shell scripts "
                "to run GeoPriorSubsNet in batch mode, useful for "
                "server jobs and reproducible experiments.",
            ),
        ),
    ]


# ----------------------------------------------------------------------
# Tools tab main widget
# ----------------------------------------------------------------------


class ToolsTab(QWidget):
    """
    Main widget for the Tools tab.

    Left side: tool navigator (grouped list).
    Right side: quick-actions row + stacked workspace.
    """

    def __init__(
        self,
        app_ctx: Optional[object] = None,
        store: Optional[object] = None,
        geo_cfg: Optional[object] = None,
        gui_runs_root: Optional[Path] = None,
        parent: Optional[QWidget] = None,
        tool_specs: Optional[List[ToolSpec]] = None,
    ) -> None:

        super().__init__(parent)

        self._app_ctx = app_ctx
        self._geo_cfg = geo_cfg
        self._gui_runs_root = gui_runs_root
        self._store = store

        # Tool specs may want to close over app_ctx / geo_cfg / root
        self._tool_specs = tool_specs or default_tool_specs(
            app_ctx=app_ctx,
            store=store,
            geo_cfg=geo_cfg,
        )
        self._tool_id_to_index: Dict[str, int] = {}

        self._init_ui()


    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left: navigator
        nav_group = QGroupBox("Tool navigator", self)
        nav_layout = QVBoxLayout(nav_group)
        nav_layout.setContentsMargins(8, 8, 8, 8)
        nav_layout.setSpacing(6)

        self._nav_list = QListWidget(nav_group)
        self._nav_list.setSelectionMode(QListWidget.SingleSelection)
        self._nav_list.setSpacing(2)
        self._nav_list.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding
        )

        self._nav_list.currentItemChanged.connect(
            self._on_nav_item_changed
        )

        self._nav_footer = QLabel(
            "Select a tool on the left to get started.", nav_group
        )
        self._nav_footer.setWordWrap(True)
        self._nav_footer.setStyleSheet(
            "color: palette(mid); font-size: 9pt; font-style: italic;"
        )

        nav_layout.addWidget(self._nav_list)
        nav_layout.addWidget(self._nav_footer)

        nav_group.setMinimumWidth(240)
        nav_group.setMaximumWidth(300)

        # Right: workspace
        self._workspace = QStackedWidget(self)
        self._workspace.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

        workspace_layout = QVBoxLayout()
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(8)

        # Quick actions row
        quick_group = QGroupBox("Quick actions", self)
        quick_layout = QHBoxLayout(quick_group)
        quick_layout.setContentsMargins(8, 4, 8, 4)
        quick_layout.setSpacing(6)

        btn_env = QPushButton("Check environment", quick_group)
        btn_env.clicked.connect(
            lambda: self.select_tool("sys.env_check")
        )

        btn_dataset = QPushButton("Inspect dataset", quick_group)
        btn_dataset.clicked.connect(
            lambda: self.select_tool("data.dataset_explorer")
        )
        # (later  might add)
        # btn_features = QPushButton("Inspect features", quick_group)
        # btn_features.clicked.connect(
        #     lambda: self.select_tool("data.feature_inspector")
        # )
        btn_metrics = QPushButton("Open metrics dashboard", quick_group)
        btn_metrics.clicked.connect(
            lambda: self.select_tool("diag.metrics_dashboard")
        )
        btn_ident = QPushButton("Identifiability", quick_group)
        btn_ident.clicked.connect(
            lambda: self.select_tool("diag.identifiability")
        )

        quick_layout.addWidget(btn_env)
        quick_layout.addWidget(btn_dataset)
        quick_layout.addWidget(btn_metrics)
        quick_layout.addWidget(btn_ident)
        quick_layout.addStretch(1)

        # Separator line
        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)


        # Populate tools into stacked widget and navigator
        self._populate_tools()

        workspace_layout.addWidget(quick_group)
        workspace_layout.addWidget(sep)
        workspace_layout.addWidget(self._workspace, stretch=1)
        # workspace_layout.addWidget(self._workspace_footer)

        # Plug into main layout
        main_layout.addWidget(nav_group)
        main_layout.addLayout(workspace_layout, stretch=1)

        # Select first tool by default (if any)
        if self._tool_specs:
            self._nav_list.setCurrentRow(0)

    def _populate_tools(self) -> None:
        """
        Create tool widgets and navigator items from the tool specs.
        """
        # Clear navigator
        self._nav_list.clear()

        # Clear stacked widget pages (QStackedWidget has no .clear())
        for i in reversed(range(self._workspace.count())):
            w = self._workspace.widget(i)
            self._workspace.removeWidget(w)
            w.deleteLater()

        self._tool_id_to_index.clear()

        current_group: Optional[str] = None

        for spec in self._tool_specs:
            # Group header (simple pseudo-header using a disabled item)
            if spec.group != current_group:
                current_group = spec.group
                header_item = QListWidgetItem(spec.group)
                header_item.setFlags(Qt.NoItemFlags)
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
                self._nav_list.addItem(header_item)

            # Actual selectable tool item
            item = QListWidgetItem(f"  {spec.title}")
            item.setData(Qt.UserRole, spec.tool_id)
            self._nav_list.addItem(item)

            # Create and add the corresponding widget
            widget = spec.factory()
            index = self._workspace.addWidget(widget)
            self._tool_id_to_index[spec.tool_id] = index


    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    def _on_nav_item_changed(
        self,
        current: Optional[QListWidgetItem],
        previous: Optional[QListWidgetItem],
    ) -> None:
        """
        React to navigator selection changes.
        """
        if current is None:
            return

        tool_id = current.data(Qt.UserRole)
        if not tool_id:
            # Likely a group header; ignore.
            return

        self._activate_tool(tool_id)

    def _activate_tool(self, tool_id: str) -> None:
        """
        Switch the stacked widget and update footers based on tool_id.
        Also tells the main window whether the bottom log console
        should be visible for this tool.
        """
        index = self._tool_id_to_index.get(tool_id)
        if index is None:
            return

        self._workspace.setCurrentIndex(index)

        # Find the corresponding spec to update descriptions
        spec = next(
            (s for s in self._tool_specs if s.tool_id == tool_id),
            None,
        )
        if spec is not None:
            self._nav_footer.setText(spec.description)
            # self._workspace_footer.setText(spec.description)

            # Per-tool log preference (default True if missing)
            needs_log = getattr(spec, "needs_log", True)

            # Ask the main window to show / hide the console
            if (
                self._app_ctx is not None
                and hasattr(self._app_ctx, "set_console_visible")
            ):
                self._app_ctx.set_console_visible(bool(needs_log))


    # Public API --------------------------------------------------------

    def select_tool(self, tool_id: str) -> None:
        """
        Programmatically select a tool by its id.

        This is used by quick-action buttons and can later be called
        from other tabs (Train/Tune/Inference) to deep-link into the
        Tools tab.
        """
        # Find the matching item in the navigator and select it.
        for row in range(self._nav_list.count()):
            item = self._nav_list.item(row)
            if not item or not item.flags() & Qt.ItemIsSelectable:
                continue
            if item.data(Qt.UserRole) == tool_id:
                self._nav_list.setCurrentRow(row)
                break
