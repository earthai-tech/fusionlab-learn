import sys
import traceback
import faulthandler

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QToolTip

def auto_set_ui_fonts(app: QApplication, tooltip: bool = True):
    """
    Detect platform and set a good default UI font for the app,
    and optionally a tooltip font.
    """
    plat = sys.platform
    # Windows
    if plat.startswith("win"):
        ui_font = QFont("Segoe UI", 9)
        tt_font = QFont("Segoe UI", 9)
    # macOS
    elif plat == "darwin":
        ui_font = QFont("San Francisco", 11)  # modern macOS system font
        tt_font = QFont("Helvetica Neue", 9)
    # Linux (fallback to something common)
    else:
        ui_font = QFont("Ubuntu", 10)         # Ubuntu’s system font
        tt_font = QFont("Ubuntu Mono", 9)

    app.setFont(ui_font)
    if tooltip:
        QToolTip.setFont(tt_font)



def enable_qt_crash_handler(app: QApplication, keep_gui_alive: bool = False) -> None:
    """
    Enable comprehensive exception and crash handling for a Qt application.

    - Installs a C-level fault handler for native crashes (segfaults, aborts).
    - Overrides sys.excepthook to print full Python tracebacks and optionally quit the GUI.

    Parameters
    ----------
    app : QApplication
        The Qt application instance to quit on unhandled exceptions.
    keep_gui_alive : bool, default False
        If True, the GUI will remain open after an unhandled exception.
        If False, QApplication.quit() is called after printing the traceback.
    """
    # 1) Catch native crashes (segfaults, etc.)
    faulthandler.enable()

    # 2) Install Python exception hook
    def qt_excepthook(exctype, value, tb):
        # Print the full Python traceback
        traceback.print_exception(exctype, value, tb)
        # Quit the application unless user opted to keep it alive
        if not keep_gui_alive:
            app.quit()

    sys.excepthook = qt_excepthook

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     enable_qt_crash_handler(app, keep_gui_alive=False)
#     # … theme / stylesheet logic …
#     gui = MiniForecaster(theme="fusionlab")
#     gui.show()
#     sys.exit(app.exec_())
    