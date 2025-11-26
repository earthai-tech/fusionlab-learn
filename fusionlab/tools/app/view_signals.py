
from PyQt5.QtCore import pyqtSignal, QObject

class _VisualizerSignals(QObject):                   
    figure_saved = pyqtSignal(str)                   
VIS_SIGNALS = _VisualizerSignals()    