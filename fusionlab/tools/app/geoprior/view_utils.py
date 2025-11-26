import os  
from typing import Dict, Any
from ..view_signals import VIS_SIGNALS

# ---------------------------------------------------------------------
# Optional bridge: notify GUI when a forecast-view PNG is saved
# ---------------------------------------------------------------------
def _emit_figure_saved_if_gui(png_path: str) -> None:
    """
    Best-effort bridge from backend -> Qt GUI.

    If the GeoPrior GUI is running it will have connected
    VIS_SIGNALS.figure_saved to its preview dialog. In plain
    CLI runs this is a no-op.
    """

    VIS_SIGNALS.figure_saved.emit(os.path.abspath(png_path))

def _notify_gui_forecast_views(run_output_path: str, city_name: str) -> None:
    """
    Look for the subsidence-view PNGs and emit a preview signal
    for each one that exists.
    """
    base = os.path.join(run_output_path, f"{city_name}_subsidence_view")
    candidates = [
        base + "_eval.png",
        base + "_future.png",
        # optional fallback in case naming changes later:
        base + ".png",
    ]

    for path in candidates:
        if os.path.exists(path):
            _emit_figure_saved_if_gui(path)


def _notify_gui_xfer_view(result: Dict[str, Any] | None) -> None:
    """
    Bridge XferViewJob → Qt GUI.

    If `result` contains a `png_path` and we are running under the
    Qt GUI, emit VIS_SIGNALS.figure_saved so the main window can
    pop up the ImagePreviewDialog.

    Parameters
    ----------
    result : dict or None
        Result dictionary returned by XferViewJob.run(), expected
        to contain the key 'png_path'.
    """
    if not isinstance(result, dict):
        return

    png_path = result.get("png_path")
    if not png_path:
        return

    # Reuse the common helper: checks exists / Qt presence, then emit.
    _emit_figure_saved_if_gui(png_path)
