# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
stage1_workspace.readiness

Readiness panel for Stage-1 (preprocess) workspace.

This widget is "view-first":
- it does not mutate configuration
- it does not decide Stage-1 actions authoritatively
- the controller (app.py) pushes context + scan results + compat

The widget renders:
- input/status checklist (PASS/WARN/FAIL)
- compatibility diff (current config vs manifest snapshot)
- a decision preview (reuse vs rebuild) driven by options
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape as _esc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

Json = Dict[str, Any]
PathLike = Union[str, Path]


@dataclass(frozen=True)
class ReadinessContext:
    city: str = ""
    csv_path: str = ""
    runs_root: str = ""
    stage1_dir: str = ""


@dataclass(frozen=True)
class Stage1Scan:
    """
    Result of scanning the Stage-1 run directory.

    The controller should fill this with best-effort signals:
    - found: whether an existing Stage-1 directory is detected
    - run_dir: the candidate Stage-1 directory
    - manifest_path: resolved manifest.json path (if any)
    - audit_path: resolved scaling audit path (if any)
    - artifacts_dir: artifacts/ dir path (if any)
    - missing: human labels for missing expected files
    """
    found: bool = False
    run_dir: str = ""
    manifest_path: str = ""
    audit_path: str = ""
    artifacts_dir: str = ""
    missing: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CompatDiff:
    key: str
    current: str
    candidate: str


@dataclass(frozen=True)
class CompatibilityResult:
    """
    Compatibility result between current config and Stage-1 manifest.

    evaluated:
        False means no comparison was attempted yet.
    match:
        True means "safe to reuse" under your policy.
    diffs:
        List of key diffs to explain mismatch.
    note:
        Optional extra message from controller logic.
    """
    evaluated: bool = False
    match: bool = False
    diffs: Tuple[CompatDiff, ...] = ()
    note: str = ""


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _to_file_url(p: PathLike) -> str:
    s = _as_str(p).strip()
    if not s:
        return ""
    try:
        return QUrl.fromLocalFile(s).toString()
    except Exception:
        return ""


def _path_exists(p: str) -> Optional[bool]:
    """
    Best-effort existence check.

    Returns:
        True/False if p looks like a local path, else None.
    """
    s = _as_str(p).strip()
    if not s:
        return False
    try:
        return Path(s).exists()
    except Exception:
        return None


def _kv_row(k: str, v_html: str) -> str:
    kk = _esc(k)
    vv = v_html if v_html else "-"
    return (
        "<tr>"
        "<td style='padding:2px 10px 2px 0;"
        "font-weight:600;white-space:nowrap;'>"
        f"{kk}</td>"
        f"<td style='padding:2px 0;'>{vv}</td>"
        "</tr>"
    )


def _badge(status: str) -> str:
    """
    status: pass|warn|fail|info
    """
    s = (status or "").lower().strip()
    if s == "pass":
        return (
            "<span style='display:inline-block;"
            "padding:1px 8px;border-radius:10px;"
            "background:#1b5e20;color:white;"
            "font-weight:600;'>PASS</span>"
        )
    if s == "warn":
        return (
            "<span style='display:inline-block;"
            "padding:1px 8px;border-radius:10px;"
            "background:#e65100;color:white;"
            "font-weight:600;'>WARN</span>"
        )
    if s == "fail":
        return (
            "<span style='display:inline-block;"
            "padding:1px 8px;border-radius:10px;"
            "background:#b71c1c;color:white;"
            "font-weight:600;'>FAIL</span>"
        )
    return (
        "<span style='display:inline-block;"
        "padding:1px 8px;border-radius:10px;"
        "background:#455a64;color:white;"
        "font-weight:600;'>INFO</span>"
    )


def _link_or_text(path: str) -> str:
    p = _as_str(path)
    if not p:
        return "-"
    url = _to_file_url(p)
    txt = _esc(p)
    if not url:
        return txt
    return f"<a href='{_esc(url)}'>{txt}</a>"


@dataclass
class _CheckItem:
    label: str
    status: str
    message: str = ""
    detail: str = ""


class Stage1Readiness(QWidget):
    """
    Stage-1 readiness panel.

    Public API:
    - clear()
    - set_context(...)
    - set_options(...)
    - set_stage1_scan(...)
    - set_compatibility(...)
    - set_status(...)
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._ctx = ReadinessContext()
        self._options: Dict[str, Any] = {}
        self._scan = Stage1Scan()
        self._compat = CompatibilityResult()
        self._status: str = ""

        self._build_ui()
        self._render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._ctx = ReadinessContext()
        self._options = {}
        self._scan = Stage1Scan()
        self._compat = CompatibilityResult()
        self._status = ""
        self._render()

    def set_context(
        self,
        *,
        city: str,
        csv_path: Optional[PathLike],
        runs_root: Optional[PathLike],
        stage1_dir: Optional[PathLike] = None,
    ) -> None:
        self._ctx = ReadinessContext(
            city=_as_str(city),
            csv_path=_as_str(csv_path),
            runs_root=_as_str(runs_root),
            stage1_dir=_as_str(stage1_dir),
        )
        self._render()

    def set_options(self, options: Optional[Dict[str, Any]]) -> None:
        self._options = options if isinstance(options, dict) else {}
        self._render()

    def set_stage1_scan(self, scan: Optional[Stage1Scan]) -> None:
        self._scan = scan if isinstance(scan, Stage1Scan) else Stage1Scan()
        self._render()

    def set_compatibility(
        self,
        compat: Optional[CompatibilityResult],
    ) -> None:
        self._compat = (
            compat if isinstance(compat, CompatibilityResult)
            else CompatibilityResult()
        )
        self._render()

    def set_status(self, text: str) -> None:
        self._status = _as_str(text)
        self._render()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        hdr = QHBoxLayout()
        hdr.setSpacing(10)

        self.lbl_title = QLabel("Stage-1 Readiness")
        self.lbl_title.setObjectName("stage1ReadinessTitle")

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("stage1ReadinessStatus")

        hdr.addWidget(self.lbl_title)
        hdr.addStretch(1)
        hdr.addWidget(self.lbl_status)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)

        layout.addLayout(hdr)
        layout.addWidget(self.browser, 1)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def _render(self) -> None:
        self.lbl_status.setText(self._status or "")

        html = []
        html.append("<html><body>")
        html.append(self._render_context())
        html.append(self._render_options())
        html.append(self._render_checklist())
        html.append(self._render_compat())
        html.append(self._render_decision_preview())
        html.append("</body></html>")

        self.browser.setHtml("".join(html))

    def _render_context(self) -> str:
        c = self._ctx
        rows = []
        rows.append(_kv_row("City", _esc(c.city or "-")))
        rows.append(_kv_row("Dataset", _link_or_text(c.csv_path)))
        rows.append(_kv_row("Results root", _link_or_text(c.runs_root)))
        rows.append(_kv_row("Stage-1 dir", _link_or_text(c.stage1_dir)))
        return (
            "<h3 style='margin:6px 0 6px 0;'>Context</h3>"
            "<table>"
            + "".join(rows)
            + "</table>"
            "<hr/>"
        )

    def _render_options(self) -> str:
        o = self._options or {}

        def b(key: str, default: bool = False) -> str:
            v = o.get(key, default)
            return "ON" if bool(v) else "OFF"

        rows = []
        rows.append(
            _kv_row(
                "clean_stage1_dir",
                _esc(b("clean_stage1_dir", False)),
            )
        )
        rows.append(
            _kv_row(
                "build_future_npz",
                _esc(b("build_future_npz", False)),
            )
        )
        rows.append(
            _kv_row(
                "stage1_auto_reuse_if_match",
                _esc(b("stage1_auto_reuse_if_match", True)),
            )
        )
        rows.append(
            _kv_row(
                "stage1_force_rebuild_if_mismatch",
                _esc(b("stage1_force_rebuild_if_mismatch", True)),
            )
        )

        return (
            "<h3 style='margin:6px 0 6px 0;'>Policy (options)</h3>"
            "<table>"
            + "".join(rows)
            + "</table>"
            "<hr/>"
        )

    def _render_checklist(self) -> str:
        checks = self._build_checks()

        rows = []
        for it in checks:
            msg = _esc(it.message or "")
            det = _esc(it.detail or "")
            cell = msg
            if det:
                cell = (
                    f"{msg}<br/>"
                    "<span style='color:#546e7a;'>"
                    f"{det}</span>"
                )
            rows.append(
                "<tr>"
                "<td style='padding:4px 10px 4px 0;"
                "white-space:nowrap;'>"
                f"{_badge(it.status)}"
                "</td>"
                "<td style='padding:4px 10px 4px 0;"
                "font-weight:600;'>"
                f"{_esc(it.label)}</td>"
                f"<td style='padding:4px 0;'>{cell}</td>"
                "</tr>"
            )

        return (
            "<h3 style='margin:6px 0 6px 0;'>Checklist</h3>"
            "<table style='border-collapse:collapse;'>"
            + "".join(rows)
            + "</table>"
            "<hr/>"
        )

    def _build_checks(self) -> List[_CheckItem]:
        c = self._ctx
        scan = self._scan

        checks: List[_CheckItem] = []

        # Inputs
        if c.city.strip():
            checks.append(
                _CheckItem(
                    label="City selected",
                    status="pass",
                    message=c.city,
                )
            )
        else:
            checks.append(
                _CheckItem(
                    label="City selected",
                    status="fail",
                    message="No city selected.",
                )
            )

        if c.csv_path.strip():
            ex = _path_exists(c.csv_path)
            st = "pass" if ex is True else "warn"
            if ex is False:
                st = "fail"
            checks.append(
                _CheckItem(
                    label="Dataset path",
                    status=st,
                    message=c.csv_path,
                    detail=self._exists_detail(ex),
                )
            )
        else:
            checks.append(
                _CheckItem(
                    label="Dataset path",
                    status="fail",
                    message="No dataset selected.",
                )
            )

        if c.runs_root.strip():
            ex = _path_exists(c.runs_root)
            st = "pass" if ex is True else "warn"
            if ex is False:
                st = "fail"
            checks.append(
                _CheckItem(
                    label="Results root",
                    status=st,
                    message=c.runs_root,
                    detail=self._exists_detail(ex),
                )
            )
        else:
            checks.append(
                _CheckItem(
                    label="Results root",
                    status="warn",
                    message="No results root set.",
                )
            )

        # Existing Stage-1 run
        if scan.found and scan.run_dir:
            ex = _path_exists(scan.run_dir)
            st = "pass" if ex is True else "warn"
            if ex is False:
                st = "fail"
            msg = scan.run_dir
            det = self._exists_detail(ex)
            if scan.missing:
                st = "warn" if st == "pass" else st
                det2 = ", ".join(scan.missing)
                det = (det + " | " if det else "") + f"missing: {det2}"
            checks.append(
                _CheckItem(
                    label="Stage-1 run detected",
                    status=st,
                    message=msg,
                    detail=det,
                )
            )
        else:
            checks.append(
                _CheckItem(
                    label="Stage-1 run detected",
                    status="info",
                    message="No existing Stage-1 run found.",
                )
            )

        # Manifest presence
        if scan.manifest_path:
            ex = _path_exists(scan.manifest_path)
            st = "pass" if ex is True else "warn"
            if ex is False:
                st = "fail"
            checks.append(
                _CheckItem(
                    label="manifest.json",
                    status=st,
                    message=scan.manifest_path,
                    detail=self._exists_detail(ex),
                )
            )
        else:
            checks.append(
                _CheckItem(
                    label="manifest.json",
                    status="info",
                    message="Manifest path not resolved yet.",
                )
            )

        # Audit presence (optional)
        if scan.audit_path:
            ex = _path_exists(scan.audit_path)
            st = "pass" if ex is True else "warn"
            if ex is False:
                st = "warn"
            checks.append(
                _CheckItem(
                    label="stage1 scaling audit",
                    status=st,
                    message=scan.audit_path,
                    detail=self._exists_detail(ex),
                )
            )
        else:
            checks.append(
                _CheckItem(
                    label="stage1 scaling audit",
                    status="info",
                    message="Audit not resolved (optional).",
                )
            )

        return checks

    def _exists_detail(self, ex: Optional[bool]) -> str:
        if ex is True:
            return "exists"
        if ex is False:
            return "missing"
        return "unknown"

    def _render_compat(self) -> str:
        compat = self._compat
        if not compat.evaluated:
            return (
                "<h3 style='margin:6px 0 6px 0;'>Compatibility</h3>"
                "<p>No compatibility check yet.</p>"
                "<hr/>"
            )

        head = "MATCH" if compat.match else "MISMATCH"
        b = _badge("pass" if compat.match else "warn")
        note = _esc(compat.note or "")

        rows = []
        rows.append(
            _kv_row(
                "Result",
                f"{b} <span style='font-weight:600;'>{head}</span>",
            )
        )
        if note:
            rows.append(_kv_row("Note", note))

        diffs = list(compat.diffs or ())
        if not diffs and compat.match:
            diffs_html = "<p>No diffs detected.</p>"
        elif not diffs:
            diffs_html = "<p>No diff details provided.</p>"
        else:
            d_rows = []
            for d in diffs[:25]:
                d_rows.append(
                    "<tr>"
                    "<td style='padding:3px 10px 3px 0;"
                    "font-weight:600;white-space:nowrap;'>"
                    f"{_esc(d.key)}</td>"
                    "<td style='padding:3px 10px 3px 0;'>"
                    f"{_esc(d.current)}</td>"
                    "<td style='padding:3px 0;'>"
                    f"{_esc(d.candidate)}</td>"
                    "</tr>"
                )
            more = ""
            if len(diffs) > 25:
                more = (
                    "<p style='color:#546e7a;'>"
                    f"Showing 25 of {len(diffs)} diffs."
                    "</p>"
                )
            diffs_html = (
                "<table style='border-collapse:collapse;'>"
                "<tr>"
                "<th style='text-align:left;padding:3px 10px 3px 0;'>Key</th>"
                "<th style='text-align:left;padding:3px 10px 3px 0;'>"
                "Current</th>"
                "<th style='text-align:left;padding:3px 0;'>Candidate</th>"
                "</tr>"
                + "".join(d_rows)
                + "</table>"
                + more
            )

        return (
            "<h3 style='margin:6px 0 6px 0;'>Compatibility</h3>"
            "<table>" + "".join(rows) + "</table>"
            "<h4 style='margin:10px 0 6px 0;'>Diffs</h4>"
            + diffs_html 
            +
            "<hr/>"
        )

    def _render_decision_preview(self) -> str:
        """
        Preview the likely action when clicking "Run Stage-1".

        This is a UI hint only. The controller remains the source of truth.
        """
        o = self._options or {}
        scan = self._scan
        compat = self._compat

        auto_reuse = bool(o.get("stage1_auto_reuse_if_match", True))
        force_rebuild = bool(o.get("stage1_force_rebuild_if_mismatch", True))
        clean_dir = bool(o.get("clean_stage1_dir", False))

        action = "unknown"
        reason = "No decision preview available yet."

        if not scan.found:
            action = "build"
            reason = "No existing Stage-1 run was detected."
        else:
            if compat.evaluated:
                if compat.match:
                    if auto_reuse:
                        action = "reuse"
                        reason = (
                            "Existing Stage-1 matches current config, "
                            "and auto-reuse is ON."
                        )
                    else:
                        action = "build"
                        reason = (
                            "Existing Stage-1 matches current config, "
                            "but auto-reuse is OFF."
                        )
                else:
                    if force_rebuild:
                        action = "rebuild"
                        reason = (
                            "Existing Stage-1 mismatches current config, "
                            "and force rebuild is ON."
                        )
                    else:
                        action = "build"
                        reason = (
                            "Existing Stage-1 mismatches current config, "
                            "and force rebuild is OFF."
                        )
            else:
                action = "unknown"
                reason = (
                    "Existing Stage-1 run detected, but no compatibility "
                    "check was performed."
                )

        if clean_dir and action in ("build", "rebuild"):
            reason = reason + " Clean dir is ON."

        badge_map = {
            "reuse": _badge("pass"),
            "build": _badge("info"),
            "rebuild": _badge("warn"),
            "unknown": _badge("info"),
        }
        b = badge_map.get(action, _badge("info"))
        label = action.upper()

        path_html = ""
        if scan.run_dir:
            path_html = (
                "<p style='margin:6px 0 0 0;'>"
                f"Candidate: {_link_or_text(scan.run_dir)}"
                "</p>"
            )

        return (
            "<h3 style='margin:6px 0 6px 0;'>Decision preview</h3>"
            f"<p style='margin:0 0 6px 0;'>{b} "
            f"<span style='font-weight:600;'>{label}</span>"
            "</p>"
            f"<p style='margin:0 0 6px 0;'>{_esc(reason)}</p>"
            + path_html
        )
