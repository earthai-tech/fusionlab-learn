from __future__ import annotations 

from pathlib import Path 

from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import ( 
    QSplashScreen, 
    QProgressBar, 
    QLabel, 
    QVBoxLayout, 
    QWidget, 
    QApplication
)

__all__= ['LoadingSplash', 'MovieSplashScreen']


class LoadingSplash(QSplashScreen):
    """
    Splash screen with logo + progress bar + percentage text.
    """

    def __init__(self, logo_path: Path, parent: QWidget | None = None) -> None:
        pixmap = QPixmap(str(logo_path))
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        self.setMask(pixmap.mask())

        # progress bar at the bottom
        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(True)  # shows "NN %"
        self._progress.setFormat("%p%")
        self._progress.setFixedHeight(18)

        # optional text just above the bar
        self._label = QLabel("Starting GeoPrior-3.0 Forecaster…", self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: white;")

        # simple manual layout inside the splash
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addStretch(1)
        layout.addWidget(self._label)
        layout.addWidget(self._progress)

    def set_progress(self, value: int, message: str | None = None) -> None:
        """Update percent + optional message and let Qt repaint."""
        self._progress.setValue(value)
        if message is not None:
            self._label.setText(message)
        QApplication.processEvents()


class MovieSplashScreen(QSplashScreen):
    """
    Simple splash screen that can play an animated GIF (QMovie).
    """
    def __init__(self, movie: QMovie, parent=None):
        # Start with a blank pixmap; we update frames via the movie
        super().__init__(QPixmap(), parent)
        self._movie = movie
        self._movie.frameChanged.connect(self._on_frame_changed)
        self._movie.start()

    def _on_frame_changed(self, _frame: int) -> None:
        self.setPixmap(self._movie.currentPixmap())
        self.update()


# ----------------------------------------------------------------------
# Entry point helper
# ----------------------------------------------------------------------
# def launch_geoprior_gui(theme: str = "fusionlab") -> None:
#     # Create the Qt application
#     cfg = GeoPriorConfig.from_nat_config()
#     app = QApplication(sys.argv)

#     # Cross-platform tweaks
#     auto_set_ui_fonts(app)                              # DPI-aware fonts + Fusion style
#     enable_qt_crash_handler(app, keep_gui_alive=False)  # nice tracebacks if something dies

#     scale = float(getattr(cfg, "ui_font_scale", 1.0))
#     if scale != 1.0:
#         f = app.font()
#         f.setPointSizeF(max(6.0, f.pointSizeF() * scale))
#         app.setFont(f)
    
#     gui = GeoPriorForecaster(theme=theme)
#     gui.show()

#     sys.exit(app.exec_())

# # ----------------------------------------------------------------------
# def launch_geoprior_gui(theme: str = "fusionlab") -> None:
#     app = QApplication(sys.argv)
#     auto_set_ui_fonts(app)

#     # --- 1) Show splash screen immediately ---
#     # Option A: static PNG logo
#     # splash_pix = QPixmap(":/img/geoprior_splash.png")  # if you use a Qt resource
#     # splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)

#     # Option B: animated GIF spinner (put your .gif in a known path)
#     movie = QMovie(str(Path(__file__).with_name("geoprior_splash.gif")))
#     splash = MovieSplashScreen(movie)
#     splash.setWindowFlag(Qt.WindowStaysOnTopHint)
#     splash.show()
#     app.processEvents()  # let Qt paint the splash

#     # --- 2) Build the main GUI while splash is visible ---
#     gui = GeoPriorForecaster(theme=theme)

#     # (Optional) you can show a status text on the splash:
#     # splash.showMessage(
#     #     "Initializing GeoPrior-3.0 Forecaster…",
#     #     Qt.AlignBottom | Qt.AlignCenter,
#     #     Qt.white,
#     # )
#     # app.processEvents()

#     # --- 3) Hide splash and show main window ---
#     splash.finish(gui)
#     gui.show()

#     sys.exit(app.exec_())

# if __name__ == "__main__":
#     QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
#     launch_geoprior_gui()