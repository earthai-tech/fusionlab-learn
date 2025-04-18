
from __future__ import annotations 

import warnings
import logging
from ._fusionlog import fusionlog

def initialize_logging(
    config_path: str | None = None,
    use_default_logger: bool = True,
    verbose: bool = False
) -> None:
    """
    Initialize FusionLab structured logging.

    Attempts to load the logging configuration via `fusionlog.load_configuration`.
    On failure, emits a warning and falls back to a basic console logger.

    Parameters
    ----------
    config_path : str | None
        Path to a YAML/INI logging config. If None, uses default/basic config.
    use_default_logger : bool
        Whether to install the default FusionLab logger if config_path is None.
    verbose : bool
        If True, prints out which config file is being used.
    """
    try:
        fusionlog.load_configuration(
            config_path=config_path,
            use_default_logger=use_default_logger,
            verbose=verbose
        )
    except Exception as e:
        warnings.warn(
            f"FusionLab logging initialization failed: {e}. "
            "Falling back to basic console logging.",
            RuntimeWarning
        )
        # Basic fallback
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
