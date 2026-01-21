# geoprior/about/dialog.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ..styles import FLAB_STYLE_SHEET
from .qss import about_qss
from .pages import (
    build_cite_page,
    build_overview_page,
    build_quickstart_page,
    build_tabs_page,
    build_troubleshoot_page,
)
from .pages import content
from .widgets import (
    add_nav_item,
    make_header,
    make_nav,
)


def show_about_dialog(parent: QWidget | None = None) -> None:
    dlg = AboutDialog(parent)
    dlg.exec_()


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setObjectName("aboutDialog")
        self.setWindowTitle("About GeoPrior Forecaster")
        self.setModal(True)
        self.setMinimumWidth(860)

        self.setStyleSheet(
            FLAB_STYLE_SHEET + about_qss()
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        head = make_header(
            self,
            title=f"{content.APP_NAME} [{content.APP_VERSION}]",
            tagline=content.TAGLINE,
            chips=["Help Center", "v3.2"],
            svg_icon="geoprior_logo.ico",
        )
        root.addWidget(head)

        links = QLabel(self)
        links.setObjectName("aboutLinks")
        links.setTextFormat(Qt.RichText)
        links.setOpenExternalLinks(True)
        links.setText(
            (
                f'<a href="{content.DOCS_URL}">User guide</a> · '
                f'<a href="{content.GITHUB_URL}">Source</a> · '
                f'<a href="{content.PORTFOLIO_URL}">Portfolio</a>'
            )
        )
        root.addWidget(links)

        sysi = QLabel(content.system_info_text(), self)
        sysi.setObjectName("aboutFootnote")
        sysi.setWordWrap(True)
        root.addWidget(sysi)

        row = QHBoxLayout()
        row.setSpacing(10)

        self._nav = make_nav(self)
        self._stack = QStackedWidget(self)

        self._stack.addWidget(build_overview_page(self._stack))
        self._stack.addWidget(build_quickstart_page(self._stack))
        self._stack.addWidget(build_tabs_page(self._stack))
        self._stack.addWidget(build_troubleshoot_page(self._stack))
        self._stack.addWidget(build_cite_page(self._stack))

        add_nav_item(
            self._nav,
            "Overview",
            svg="info.svg",
            sp=QStyle.SP_MessageBoxInformation,
        )
        add_nav_item(
            self._nav,
            "Quickstart",
            svg="rocket.svg",
            sp=QStyle.SP_ArrowRight,
        )
        add_nav_item(
            self._nav,
            "Tabs guide",
            svg="tabs.svg",
            sp=QStyle.SP_FileDialogDetailedView,
        )
        add_nav_item(
            self._nav,
            "Troubleshooting",
            svg="bug.svg",
            sp=QStyle.SP_MessageBoxWarning,
        )
        add_nav_item(
            self._nav,
            "Cite",
            svg="quote.svg",
            sp=QStyle.SP_FileDialogInfoView,
        )

        self._nav.currentRowChanged.connect(
            self._stack.setCurrentIndex
        )
        self._nav.setCurrentRow(0)

        row.addWidget(self._nav, 0)
        row.addWidget(self._stack, 1)
        root.addLayout(row, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.Close,
            self,
        )
        btns.rejected.connect(self.reject)
        root.addWidget(btns)
