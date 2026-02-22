# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>
#
# geoprior.ui.tools_tab
#
# Tools tab for GeoPrior GUI.
#
# v3.2 UX goals
# -------------
# - Modern left palette: search + favorites + rich rows
# - Right workspace: compact command bar + consistent page chrome
# - Tool wrappers (ToolPageFrame) standardize header/help/refresh
# - Per-tool "needs_log" controls the main bottom console
#
# Notes
# -----
# - This module stays view-only. Real tool logic remains in
#   geoprior/ui/tools/*.py
# - Factories can be either `factory()` or `factory(self)`.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PyQt5.QtCore import Qt, QSettings, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .tools import (
    BuildNPZTool,
    ConfigDiffTool,
    DatasetExplorerTool,
    EnvironmentCheckTool,
    FeatureInspectorTool,
    IdentifiabilityTool,
    ManifestBrowserTool,
    MetricsDashboardTool,
    PhysicsDiagnosticsTool,
    ReproduceRunHelperTool,
    RunComparisonTool,
    Stage1ManagerTool,
    JsonViewerTool, 
    DeviceMonitorTool, 
    PathsPermissionsTool, 
    ScriptGeneratorTool
)

from .icon_utils import try_icon 

# ---------------------------------------------------------------------
# Nav row widget (rich list item)
# ---------------------------------------------------------------------
class ToolNavItem(QWidget):
    def __init__(
        self,
        *,
        title: str,
        desc: str,
        icon: Optional[QIcon] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(10)

        self._icon = QLabel(self)
        self._icon.setObjectName("toolNavIcon")
        self._icon.setFixedWidth(18)

        if icon is not None:
            pm = icon.pixmap(16, 16)
            self._icon.setPixmap(pm)

        txt = QVBoxLayout()
        txt.setContentsMargins(0, 0, 0, 0)
        txt.setSpacing(2)

        self.lbl_title = QLabel(title, self)
        self.lbl_title.setObjectName("toolNavTitle")

        self.lbl_desc = QLabel(desc, self)
        self.lbl_desc.setObjectName("toolNavDesc")
        self.lbl_desc.setWordWrap(True)

        txt.addWidget(self.lbl_title)
        txt.addWidget(self.lbl_desc)

        root.addWidget(self._icon, 0)
        root.addLayout(txt, 1)


# ---------------------------------------------------------------------
# Tool page chrome wrapper
# ---------------------------------------------------------------------
class ToolPageFrame(QFrame):
    """
    Wrapper around a tool widget with consistent chrome:
    title + group chip + help + refresh, then the tool body.
    """

    request_refresh = pyqtSignal()

    def __init__(
        self,
        *,
        title: str,
        group: str,
        desc: str,
        inner: QWidget,
        icon: Optional[QIcon] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("toolPage")
        self._desc = desc
        self._inner = inner

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # --- header row
        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        self._ttl = QLabel(title, self)
        self._ttl.setObjectName("toolPageTitle")

        self._grp = QLabel(group, self)
        self._grp.setObjectName("toolPageGroup")

        hdr.addWidget(self._ttl)
        hdr.addWidget(self._grp)
        hdr.addStretch(1)

        self._btn_refresh = QToolButton(self)
        self._btn_refresh.setAutoRaise(True)
        self._btn_refresh.setObjectName("miniAction")
        self._btn_refresh.setToolTip("Refresh")
        self._btn_refresh.setIcon(
            self.style().standardIcon(
                QStyle.SP_BrowserReload
            )
        )
        self._btn_refresh.clicked.connect(
            self._on_refresh
        )

        self._btn_help = QToolButton(self)
        self._btn_help.setAutoRaise(True)
        self._btn_help.setObjectName("miniAction")
        self._btn_help.setToolTip(desc)
        self._btn_help.setIcon(
            self.style().standardIcon(
                QStyle.SP_MessageBoxInformation
            )
        )

        hdr.addWidget(self._btn_refresh)
        hdr.addWidget(self._btn_help)

        # --- description
        self._desc_lbl = QLabel(desc, self)
        self._desc_lbl.setObjectName("toolPageDesc")
        self._desc_lbl.setWordWrap(True)

        # --- divider
        div = QFrame(self)
        div.setFrameShape(QFrame.HLine)
        div.setFrameShadow(QFrame.Plain)
        div.setObjectName("toolPageDivider")

        # --- body
        body = QFrame(self)
        body.setObjectName("toolPageBody")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(inner, 1)

        root.addLayout(hdr)
        root.addWidget(self._desc_lbl)
        root.addWidget(div)
        root.addWidget(body, 1)

    @property
    def inner(self) -> QWidget:
        return self._inner

    def _on_refresh(self) -> None:
        """
        Best-effort refresh hook: tries common method names.
        """
        w = self._inner

        for name in (
            "refresh",
            "reload",
            "rebuild",
            "update_view",
        ):
            fn = getattr(w, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                return

        self.request_refresh.emit()


# ---------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------
@dataclass
class ToolSpec:
    """
    Descriptor for a tool entry in the Tools tab navigator.
    """

    tool_id: str
    title: str
    group: str
    description: str
    factory: Optional[Callable[["ToolsTab"], QWidget]] = None
    needs_log: bool = True
    favorite: bool = False
    icon_name: str = ""


# ---------------------------------------------------------------------
# Placeholder tools (for future)
# ---------------------------------------------------------------------
class _ToolPlaceholder(QWidget):
    def __init__(
        self,
        title: str,
        text: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        chip = QLabel("Coming soon", self)
        chip.setObjectName("chip")
        chip.setStyleSheet(
            "padding:2px 10px;"
            "border-radius:11px;"
            "background: palette(midlight);"
            "color: palette(text);"
        )

        body = QLabel(text, self)
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        lay.addWidget(chip, 0, Qt.AlignLeft)
        lay.addWidget(body, 0)
        lay.addStretch(1)


def _make_placeholder_tool(title: str, desc: str) -> QWidget:
    # ToolPageFrame already shows title+desc,
    # so placeholder shows only the body text.
    return _ToolPlaceholder(title, desc)

# ---------------------------------------------------------------------
# Default tools list
# ---------------------------------------------------------------------
def default_tool_specs(
    *,
    app_ctx: object | None = None,
    store: object | None = None,
    geo_cfg: object | None = None,
) -> List[ToolSpec]:
    def _make_ident_tool() -> QWidget:
        return IdentifiabilityTool(
            app_ctx=app_ctx,
            store=store,
            geo_cfg=geo_cfg,
        )

    return [
        ToolSpec(
            tool_id="data.dataset_explorer",
            group="Data & Config",
            title="Dataset explorer",
            description=(
                "Inspect the active dataset: shape, "
                "coverage, missing values."
            ),
            factory=lambda _t: DatasetExplorerTool(
                app_ctx=app_ctx
            ),
            needs_log=False,
            icon_name="inspect_data.svg",
        ),
        ToolSpec(
            tool_id="data.feature_inspector",
            group="Data & Config",
            title="Feature inspector",
            description=(
                "List features, types, ranges and roles "
                "(inputs, targets, physics vars)."
            ),
            factory=lambda _t: FeatureInspectorTool(
                app_ctx=app_ctx
            ),
            needs_log=False,
            icon_name ="feature_inspector.svg", 
        ),
        ToolSpec(
            tool_id="data.config_diff",
            group="Data & Config",
            title="Config inspector & diff",
            description=(
                "Compare current GUI config with a "
                "saved JSON config/manifest."
            ),
            factory=lambda _t: ConfigDiffTool(
                app_ctx=app_ctx
            ),
            needs_log=False,
            icon_name="config_diff.svg",
        ),
        ToolSpec(
            tool_id="runs.stage1_manager",
            group="Runs & Manifests",
            title="Stage-1 manager",
            description=(
                "Browse Stage-1 manifests per city, "
                "inspect artifacts and select preferred."
            ),
            factory=lambda _t: Stage1ManagerTool(
                app_ctx=app_ctx
            ),
            needs_log=False,
            icon_name="stage1_preprocess.svg", 
        ),
        ToolSpec(
            tool_id="runs.manifest_browser",
            group="Runs & Manifests",
            title="Manifest browser & validator",
            description=(
                "Browse train/tune/inference manifests "
                "and validate integrity."
            ),
            factory=lambda _t: ManifestBrowserTool(
                app_ctx=app_ctx
            ),
            needs_log=False,
            icon_name="manifest.svg",
        ),
        ToolSpec(
            tool_id="runs.build_npz",
            group="Runs & Manifests",
            title="Build NPZ dataset",
            description=(
                "Build inference-ready NPZ sequences from "
                "active/saved dataset and config."
            ),
            factory=lambda _t: BuildNPZTool(
                app_ctx=app_ctx
            ),
            needs_log=True,
            icon_name="build.svg", 
        ),
        ToolSpec(
            tool_id="runs.reproduce_helper",
            group="Runs & Manifests",
            title="Reproduce run helper",
            description=(
                "Generate CLI snippets to reproduce a run."
            ),
            factory=lambda _t: ReproduceRunHelperTool(
                app_ctx=app_ctx
            ),
            needs_log=False,
            icon_name="reproduce_helper.svg"
        ),
        ToolSpec(
            tool_id="diag.metrics_dashboard",
            group="Diagnostics & Plots",
            title="Metrics dashboard",
            description=(
                "Visualise metrics and diagnostics "
                "(PIT/reliability when available)."
            ),
            factory=lambda _t: MetricsDashboardTool(
                app_ctx=app_ctx
            ),
            needs_log=False,
            icon_name="metric_dashboard.svg",
        ),
        ToolSpec(
            tool_id="diag.run_comparison",
            group="Diagnostics & Plots",
            title="Run comparison",
            description=(
                "Compare metrics/config across runs for "
                "ablations and paper figures."
            ),
            factory=lambda _t: RunComparisonTool(
                app_ctx=app_ctx
            ),
            needs_log=False,
            icon_name="run_comparison.svg"
        ),
        ToolSpec(
            tool_id="diag.physics_diagnostics",
            group="Diagnostics & Plots",
            title="Physics diagnostics",
            description=(
                "Inspect physics residuals and constraint "
                "violations."
            ),
            factory=lambda _t: PhysicsDiagnosticsTool(
                app_ctx=app_ctx
            ),
            needs_log=False,
            icon_name ="diagnostic.svg"
        ),
        ToolSpec(
            tool_id="diag.identifiability",
            group="Diagnostics & Plots",
            title="Identifiability (SM3)",
            description=(
                "Run SM3 identifiability diagnostics from "
                "a physics payload (NPZ)."
            ),
            factory=lambda _t: _make_ident_tool(),
            needs_log=False,
            icon_name="identifiability.svg",
        ),
        ToolSpec(
            tool_id="sys.env_check",
            group="System & Environment",
            title="Environment check",
            description=(
                "Check Python/TensorFlow/GPU availability "
                "and device overrides."
            ),
            factory=lambda _t: EnvironmentCheckTool(
                app_ctx=app_ctx
            ),
            needs_log=True,
            icon_name="system.svg",
        ),
        ToolSpec(
            tool_id="sys.device_monitor",
            group="System & Environment",
            title="GPU / device monitor",
            description="Show which device will be used.",
            factory=lambda _t: DeviceMonitorTool(app_ctx=app_ctx),
            needs_log=False,
            icon_name="device_monitor.svg",
        ),
        ToolSpec(
            tool_id="sys.paths_permissions",
            group="System & Environment",
            title="Paths & permissions",
            description=(
                "Check data/results roots and write access."
            ),
            factory=lambda _t: PathsPermissionsTool(
                app_ctx=app_ctx
            ),
            needs_log=False,
            icon_name="permission.svg",
        ),
        ToolSpec(
            tool_id="adv.json_viewer",
            group="Advanced",
            title="JSON viewer (advanced)",
            description=(
                "Inspect manifest/config JSON "
                "(read-only by default)."
            ),
            factory=lambda _t: JsonViewerTool(app_ctx=app_ctx),
            needs_log=False,
            icon_name="json.svg",
        ),
        ToolSpec(
            tool_id="adv.script_generator",
            group="Advanced",
            title="Script / batch generator",
            description="Generate small scripts for batch runs.",
            factory=lambda _t: ScriptGeneratorTool(
                app_ctx=app_ctx,
                store=store,
            ),
            needs_log=False,
            icon_name="script.svg",
        ),
    ]

# ---------------------------------------------------------------------
# Tools tab main widget
# ---------------------------------------------------------------------
class ToolsTab(QWidget):
    """
    Tools tab: left palette + right workspace.
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

        self._settings = QSettings("fusionlab", "geoprior")

        self._tool_specs = tool_specs or default_tool_specs(
            app_ctx=app_ctx,
            store=store,
            geo_cfg=geo_cfg,
        )

        self._tool_id_to_index: Dict[str, int] = {}
        self._tool_id_to_row: Dict[str, int] = {}
        self._row_to_group: Dict[int, str] = {}
        self._group_to_row: Dict[str, int] = {}
        self._pin_ids: set[str] = set()
        self._recent_ids: List[str] = []


        self._fav_ids: set[str] = set()

        self._init_ui()
        
        self._load_favorites()
        self._load_pins()
        self._load_recents()
        
        self._nav_refreshing = False
        self._page_cache: Dict[str, ToolPageFrame] = {}
        
        self._populate_tools()
        self._select_first_tool()



    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        split = QSplitter(Qt.Horizontal, self)
        split.setObjectName("toolsSplit")
        root.addWidget(split, 1)

        # -------------------------
        # Left palette
        # -------------------------
        left = QFrame(self)
        left.setObjectName("toolsNav")
        split.addWidget(left)

        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        ttl = QLabel("Tools", left)
        ttl.setObjectName("toolsNavTitle")

        self._btn_favs = QToolButton(left)
        self._btn_favs.setCheckable(True)
        self._btn_favs.setAutoRaise(True)
        self._btn_favs.setObjectName("miniAction")
        self._btn_favs.setText("★")
        self._btn_favs.setToolTip("Show favorites only")

        hdr.addWidget(ttl)
        hdr.addStretch(1)
        hdr.addWidget(self._btn_favs)
        ll.addLayout(hdr)

        self._nav_search = QLineEdit(left)
        self._nav_search.setObjectName("toolsNavSearch")
        self._nav_search.setPlaceholderText("Search tools…")
        ll.addWidget(self._nav_search)

        self._nav_list = QListWidget(left)
        self._nav_list.setObjectName("toolsNavList")
        self._nav_list.setSelectionMode(
            QListWidget.SingleSelection
        )
        self._nav_list.setSpacing(3)
        self._nav_list.setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self._nav_list.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding,
        )
        self._nav_list.setMinimumWidth(260)
        
        ll.addWidget(self._nav_list, 1)

        self._nav_footer = QLabel(
            "Select a tool to get started.",
            left,
        )
        self._nav_footer.setWordWrap(True)
        self._nav_footer.setObjectName("toolsNavFooter")
        ll.addWidget(self._nav_footer)

        # -------------------------
        # Right workspace
        # -------------------------
        right = QFrame(self)
        right.setObjectName("toolsWorkspace")
        split.addWidget(right)

        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        self._cmd = self._build_command_bar(right)
        rl.addWidget(self._cmd)

        self._workspace = QStackedWidget(right)
        self._workspace.setObjectName("toolsStack")
        self._workspace.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        rl.addWidget(self._workspace, 1)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([300, 900])

        # signals
        self._nav_list.currentItemChanged.connect(
            self._on_nav_item_changed
        )
        self._nav_list.customContextMenuRequested.connect(
            self._on_nav_menu
        )
        self._nav_search.textChanged.connect(
            self._apply_nav_filter
        )
        self._btn_favs.toggled.connect(
            lambda _on: self._apply_nav_filter(
                self._nav_search.text()
            )
        )
        
    def _load_pins(self) -> None:
        v = self._settings.value("tools.pins", [])
        ids: List[str] = []
        if isinstance(v, list):
            for s in v:
                if isinstance(s, str) and s.strip():
                    ids.append(s.strip())
        self._pin_ids = set(ids)
    
    def _save_pins(self) -> None:
        self._settings.setValue("tools.pins", sorted(self._pin_ids))
    
    def _toggle_pin(self, tool_id: str) -> None:
        if tool_id in self._pin_ids:
            self._pin_ids.remove(tool_id)
        else:
            self._pin_ids.add(tool_id)
        self._save_pins()
        self._populate_tools()
    
    def _load_recents(self) -> None:
        v = self._settings.value("tools.recents", [])
        ids: List[str] = []
        if isinstance(v, list):
            for s in v:
                if isinstance(s, str) and s.strip():
                    ids.append(s.strip())
        # keep unique, preserve order
        seen = set()
        out: List[str] = []
        for x in ids:
            if x not in seen:
                out.append(x)
                seen.add(x)
        self._recent_ids = out[:8]
    
    def _save_recents(self) -> None:
        self._settings.setValue("tools.recents", self._recent_ids[:8])
    
    def _push_recent(self, tool_id: str) -> None:
        if not tool_id:
            return
        cur = [x for x in self._recent_ids if x != tool_id]
        cur.insert(0, tool_id)
        self._recent_ids = cur[:8]
        self._save_recents()

    def _build_command_bar(self, parent: QWidget) -> QWidget:
        bar = QFrame(parent)
        bar.setObjectName("toolsCmdBar")

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        lab = QLabel("Quick", bar)
        lab.setObjectName("toolsCmdTitle")

        lay.addWidget(lab)

        lay.addWidget(
            self._make_quick_btn(
                "Dataset",
                QStyle.SP_FileDialogContentsView,
                "Inspect dataset",
                "data.dataset_explorer",
                svg_icon="inspect_data.svg", 
            )
        )
        lay.addWidget(
            self._make_quick_btn(
                "Config",
                QStyle.SP_FileDialogDetailedView,
                "Config inspector & diff",
                "data.config_diff",
                svg_icon="config_diff.svg", 
            )
        )
        lay.addWidget(
            self._make_quick_btn(
                "Metrics",
                QStyle.SP_ComputerIcon,
                "Open metrics dashboard",
                "diag.metrics_dashboard",
                svg_icon="metric_dashboard.svg", 
            )
        )
        lay.addWidget(
            self._make_quick_btn(
                "Ident",
                QStyle.SP_MessageBoxInformation,
                "Identifiability (SM3)",
                "diag.identifiability",
                svg_icon="identifiability.svg", 
            )
        )

        lay.addStretch(1)
        
        self._cmd_search = QLineEdit(bar)
        self._cmd_search.setObjectName("toolsCmdSearch")
        self._cmd_search.setPlaceholderText(
            "Type a command… (e.g., metrics) and press Enter"
            )
        self._cmd_search.returnPressed.connect(
            self._on_command_enter
            )
        lay.addWidget(self._cmd_search, 1)

        self._btn_more = QToolButton(bar)
        self._btn_more.setAutoRaise(True)
        self._btn_more.setObjectName("miniAction")
        self._btn_more.setText("More ▾")
        self._btn_more.setPopupMode(
            QToolButton.InstantPopup
        )

        self._more_menu = QMenu(self._btn_more)
        self._btn_more.setMenu(self._more_menu)

        lay.addWidget(self._btn_more)

        return bar

    def _make_quick_btn(
        self,
        text: str,
        sp: QStyle.StandardPixmap,
        tip: str,
        tool_id: str,
        *,
        svg_icon: Optional[str] = None,
        icon: Optional[QIcon] = None,
    ) -> QToolButton:
        """
        Create a small toolbar-like button.
    
        Icon priority:
        1) explicit `icon` (QIcon)
        2) `svg_icon` resolved via try_icon("name.svg")
        3) Qt standard pixmap (`sp`)
        """
        b = QToolButton(self)
        b.setAutoRaise(True)
        b.setObjectName("miniAction")
        b.setText(text)
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
    
        ico: Optional[QIcon] = None
        if icon is not None:
            ico = icon
        elif svg_icon:
            ico = try_icon(svg_icon)
    
        if ico is not None and not ico.isNull():
            b.setIcon(ico)
        else:
            b.setIcon(self.style().standardIcon(sp))
    
        b.clicked.connect(lambda: self.select_tool(tool_id))
        return b

    # ------------------------------------------------------------------
    # Populate
    # ------------------------------------------------------------------
    def _tool_icon(self, spec: ToolSpec) -> Optional[QIcon]:
        # 1) Prefer custom SVG from icons/
        if spec.icon_name:
            ico = try_icon(spec.icon_name)
            if ico is not None:
                return ico
    
        # 2) Fallback to standard Qt icons by category
        s = self.style()
        tid = spec.tool_id
    
        if tid.startswith("data."):
            return s.standardIcon(QStyle.SP_FileDialogContentsView)
        if tid.startswith("runs."):
            return s.standardIcon(QStyle.SP_DirIcon)
        if tid.startswith("diag."):
            return s.standardIcon(QStyle.SP_ComputerIcon)
        if tid.startswith("sys."):
            return s.standardIcon(QStyle.SP_DriveNetIcon)
        if tid.startswith("adv."):
            return s.standardIcon(QStyle.SP_FileIcon)
        return None


    def _populate_tools(self) -> None:
        self._nav_list.blockSignals(True)
    
        self._nav_list.clear()
        self._tool_id_to_index.clear()
        self._tool_id_to_row.clear()
        self._row_to_group.clear()
        self._group_to_row.clear()
    
        for i in reversed(range(self._workspace.count())):
            w = self._workspace.widget(i)
            self._workspace.removeWidget(w)

        self._more_menu.clear()
    
        # refresh spec flags from settings
        for s in self._tool_specs:
            s.favorite = s.tool_id in self._fav_ids
    
        # local helpers
        def _add_group_header(text: str) -> None:
            gh = QListWidgetItem(text)
            gh.setFlags(Qt.NoItemFlags)
            gh.setData(Qt.UserRole, "")
            gh.setData(Qt.UserRole + 1, "group")
            font = gh.font()
            font.setBold(True)
            gh.setFont(font)
            self._nav_list.addItem(gh)
    
        def _add_tool_item(spec: ToolSpec) -> None:
            it = QListWidgetItem(self._nav_list)
            it.setData(Qt.UserRole, spec.tool_id)
            it.setData(Qt.UserRole + 1, "tool")
            it.setToolTip(spec.description)
    
            row = self._nav_list.count() - 1
            self._tool_id_to_row[spec.tool_id] = row
            self._row_to_group[row] = spec.group
    
            icon = self._tool_icon(spec)
            navw = ToolNavItem(
                title=spec.title,
                desc=spec.description,
                icon=icon,
                parent=self._nav_list,
            )
            it.setSizeHint(navw.sizeHint())
            self._nav_list.setItemWidget(it, navw)
    
            # workspace page
            page = self._page_cache.get(spec.tool_id)

            if page is None:
                inner = self._build_tool_widget(spec)
                page = ToolPageFrame(
                    title=spec.title,
                    group=spec.group,
                    desc=spec.description,
                    inner=inner,
                    icon=icon,
                    parent=self._workspace,
                )
                self._page_cache[spec.tool_id] = page
            
            idx = self._workspace.addWidget(page)
            self._tool_id_to_index[spec.tool_id] = idx

            # command menu item
            act_title = spec.title
            if spec.tool_id in self._pin_ids:
                act_title = "📌 " + act_title
            elif spec.favorite:
                act_title = "★ " + act_title
    
            act = QAction(act_title, self._more_menu)
            act.setToolTip(spec.description)
            if icon is not None:
                act.setIcon(icon)
            act.triggered.connect(
                lambda _c=False, tid=spec.tool_id: self.select_tool(tid)
            )
            self._more_menu.addAction(act)
    
        # Build lookup by id for pin/recent
        by_id = {s.tool_id: s for s in self._tool_specs}
    
        # --- Pinned section
        pinned_specs: List[ToolSpec] = []
        for tid in sorted(self._pin_ids):
            sp = by_id.get(tid)
            if sp is not None:
                pinned_specs.append(sp)
    
        if pinned_specs:
            _add_group_header("Pinned")
            for sp in pinned_specs:
                _add_tool_item(sp)
    
        # --- Recent section
        recent_specs: List[ToolSpec] = []
        for tid in self._recent_ids:
            sp = by_id.get(tid)
            if sp is not None and tid not in self._pin_ids:
                recent_specs.append(sp)
    
        if recent_specs:
            _add_group_header("Recent")
            for sp in recent_specs:
                _add_tool_item(sp)
    
        # --- Normal groups
        current_group: Optional[str] = None
        for spec in self._tool_specs:
            # skip if already rendered in pinned/recent
            if spec.tool_id in self._pin_ids:
                continue
            if spec.tool_id in self._recent_ids:
                # already shown in recent (unless pinned)
                continue
    
            if spec.group != current_group:
                current_group = spec.group
                _add_group_header(spec.group)
    
            _add_tool_item(spec)
    
        self._nav_list.blockSignals(False)
    
        # Apply filter state (search + favorites)
        self._apply_nav_filter(self._nav_search.text())


    def _build_tool_widget(self, spec: ToolSpec) -> QWidget:
        if spec.factory is None:
            return _make_placeholder_tool(
                spec.title,
                spec.description,
            )

        try:
            w = spec.factory(self)
            if isinstance(w, QWidget):
                return w
        except TypeError:
            pass
        except Exception:
            pass

        try:
            w = spec.factory()  # type: ignore[misc]
            if isinstance(w, QWidget):
                return w
        except Exception:
            pass

        return _make_placeholder_tool(
            spec.title,
            spec.description,
        )

    # ------------------------------------------------------------------
    # Favorites (persist)
    # ------------------------------------------------------------------
    def _load_favorites(self) -> None:
        key = "tools.favorites"
        v = self._settings.value(key, [])
        ids: List[str] = []
        if isinstance(v, list):
            for s in v:
                if isinstance(s, str) and s.strip():
                    ids.append(s.strip())
        self._fav_ids = set(ids)

    def _save_favorites(self) -> None:
        key = "tools.favorites"
        self._settings.setValue(key, sorted(self._fav_ids))

    def _toggle_favorite(self, tool_id: str) -> None:
        if tool_id in self._fav_ids:
            self._fav_ids.remove(tool_id)
        else:
            self._fav_ids.add(tool_id)
        self._save_favorites()
        self._populate_tools()

    # ------------------------------------------------------------------
    # Filtering (search + favorites)
    # ------------------------------------------------------------------
    def _apply_nav_filter(self, text: str) -> None:
        key = (text or "").strip().lower()
        fav_only = bool(self._btn_favs.isChecked())

        # Track visibility per group
        group_visible: Dict[str, bool] = {}

        for i in range(self._nav_list.count()):
            it = self._nav_list.item(i)
            if it is None:
                continue

            kind = it.data(Qt.UserRole + 1) or ""
            if kind != "tool":
                continue

            tool_id = it.data(Qt.UserRole) or ""
            spec = next(
                (s for s in self._tool_specs
                 if s.tool_id == tool_id),
                None,
            )
            if spec is None:
                it.setHidden(True)
                continue

            hay = (
                f"{spec.title} {spec.description} "
                f"{spec.group}"
            ).lower()

            ok = (not key) or (key in hay)
            if fav_only:
                ok = ok and spec.favorite

            it.setHidden(not ok)

            if ok:
                group_visible[spec.group] = True

        # Hide group headers if they have no visible tools
        for i in range(self._nav_list.count()):
            it = self._nav_list.item(i)
            if it is None:
                continue
            kind = it.data(Qt.UserRole + 1) or ""
            if kind != "group":
                continue

            grp = it.text()
            it.setHidden(not group_visible.get(grp, False))

        # If current selection is hidden, pick first visible tool
        cur = self._nav_list.currentItem()
        if cur is not None and cur.isHidden():
            self._select_first_tool()

        # If a header is selected (rare), select first tool
        cur = self._nav_list.currentItem()
        if cur is not None:
            if (cur.data(Qt.UserRole + 1) or "") != "tool":
                self._select_first_tool()

    def _select_first_tool(self) -> None:
        for r in range(self._nav_list.count()):
            it = self._nav_list.item(r)
            if it is None or it.isHidden():
                continue
            if (it.data(Qt.UserRole + 1) or "") != "tool":
                continue
            self._nav_list.setCurrentRow(r)
            return

    # ------------------------------------------------------------------
    # Navigator interactions
    # ------------------------------------------------------------------
    def _on_nav_item_changed(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        if current is None:
            return
        if (current.data(Qt.UserRole + 1) or "") != "tool":
            return

        tool_id = current.data(Qt.UserRole) or ""
        if not tool_id:
            return

        self._activate_tool(tool_id)

    def _on_nav_menu(self, pos) -> None:
        row = self._nav_list.row(
            self._nav_list.itemAt(pos)
        )
        if row < 0:
            return

        it = self._nav_list.item(row)
        if it is None:
            return
        if (it.data(Qt.UserRole + 1) or "") != "tool":
            return

        tool_id = it.data(Qt.UserRole) or ""
        if not tool_id:
            return

        spec = next(
            (s for s in self._tool_specs
             if s.tool_id == tool_id),
            None,
        )
        if spec is None:
            return

        m = QMenu(self)
        
        is_pin = tool_id in self._pin_ids
        is_fav = tool_id in self._fav_ids
        
        a_pin = m.addAction("📌 Unpin" if is_pin else "📌 Pin")
        a_fav = m.addAction("★ Unfavorite" if is_fav else "★ Favorite")
        m.addSeparator()
        a_open = m.addAction("Open")
        
        act = m.exec_(self._nav_list.mapToGlobal(pos))
        
        if act == a_pin:
            self._toggle_pin(tool_id)
        elif act == a_fav:
            self._toggle_favorite(tool_id)
        elif act == a_open:
            self.select_tool(tool_id)

            
    def _on_command_enter(self) -> None:
        q = (self._cmd_search.text() or "").strip().lower()
        if not q:
            return
    
        best_id = ""
        best_score = -1
    
        for s in self._tool_specs:
            hay = f"{s.title} {s.description} {s.group}".lower()
            score = 0
            if q == s.tool_id.lower():
                score = 100
            elif q == s.title.lower():
                score = 95
            elif hay.startswith(q):
                score = 80
            elif q in s.title.lower():
                score = 70
            elif q in hay:
                score = 50
    
            # small boost for pinned
            if s.tool_id in self._pin_ids:
                score += 3
            # boost for recents
            if s.tool_id in self._recent_ids:
                score += 2
    
            if score > best_score:
                best_score = score
                best_id = s.tool_id
    
        if best_id:
            self.select_tool(best_id)
            self._cmd_search.selectAll()

    # ------------------------------------------------------------------
    # Activation + console visibility
    # ------------------------------------------------------------------
    def _activate_tool(self, tool_id: str) -> None:
        
        if getattr(self, "_nav_refreshing", False):
            return
        
        idx = self._tool_id_to_index.get(tool_id)
        if idx is None:
            return

        self._workspace.setCurrentIndex(idx)

        spec = next(
            (s for s in self._tool_specs
             if s.tool_id == tool_id),
            None,
        )
        if spec is None:
            return

        self._nav_footer.setText(spec.description)

        needs_log = bool(getattr(spec, "needs_log", True))
        if (
            self._app_ctx is not None
            and hasattr(self._app_ctx, "set_console_visible")
        ):
            try:
                self._app_ctx.set_console_visible(needs_log)
            except Exception:
                pass
            
        self._push_recent(tool_id)
        
        if (self._nav_search.text() or "").strip():
            return
        
        self._nav_refreshing = True
        try:
            self._populate_tools()
            self._restore_tool_selection(tool_id)
        finally:
            self._nav_refreshing = False
            
    def _restore_tool_selection(self, tool_id: str) -> None:
        row = self._tool_id_to_row.get(tool_id, -1)
        idx = self._tool_id_to_index.get(tool_id)
    
        if idx is not None:
            self._workspace.setCurrentIndex(idx)
    
        if row >= 0:
            self._nav_list.blockSignals(True)
            try:
                self._nav_list.setCurrentRow(row)
            finally:
                self._nav_list.blockSignals(False)
    
        spec = next(
            (s for s in self._tool_specs if s.tool_id == tool_id),
            None,
        )
        if spec is not None:
            self._nav_footer.setText(spec.description)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_tool(self, tool_id: str) -> None:
        row = self._tool_id_to_row.get(tool_id, -1)
        if row < 0:
            return
        self._nav_list.setCurrentRow(row)
        it = self._nav_list.item(row)
        if it is not None:
            self._nav_list.scrollToItem(it)
