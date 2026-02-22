# geoprior/about/__init__.py
# -*- coding: utf-8 -*-

from .dialog import AboutDialog, show_about_dialog
from .pages.content import DOCS_URL

__all__ = [
    "AboutDialog",
    "show_about_dialog",
    "DOCS_URL",
]
