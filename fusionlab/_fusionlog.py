# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Track bugs and issues, and handle all FusionLab exceptions.

This module provides a logging utility class `fusionlog` to configure and manage
logging across the `FusionLab` package. It supports various logging configurations,
including YAML and INI formats, and offers methods to set up default loggers,
retrieve named loggers, and configure log file outputs.

The module also includes helper functions to set up logging with environment
variable substitutions, enhancing flexibility in different deployment scenarios.
"""

import os
import yaml
import logging
import logging.config
from string import Template # Noqa 
from typing import Optional


__all__ = ["fusionlog"]


class fusionlog:
    @staticmethod
    def load_configuration(
        config_path: Optional[str] = None,
        use_default_logger: bool = True,
        verbose: bool = False
    ) -> None:
        if not config_path:
            if use_default_logger:
                fusionlog.set_default_logger()
            else:
                logging.basicConfig()
            return

        if verbose:
            print(f"Configuring logging with: {config_path}")

        if config_path.endswith((".yaml", ".yml")):
            fusionlog._configure_from_yaml(config_path, verbose)
        elif config_path.endswith(".ini"):
            logging.config.fileConfig(config_path, disable_existing_loggers=False)
        else:
            logging.warning(
                f"Unsupported logging configuration format: {config_path}"
            )

    @staticmethod
    def _configure_from_yaml(yaml_path: str, verbose: bool = False) -> None:
        full_path = os.path.abspath(yaml_path)
        if not os.path.exists(full_path):
            logging.error(f"The YAML config file {full_path} does not exist.")
            raise FileNotFoundError(f"The YAML config file {full_path} does not exist.")

        if verbose:
            print(f"Loading YAML config from {full_path}")

        try:
            with open(full_path, "rt") as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML config file: {e}")
            raise

    @staticmethod
    def set_default_logger() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @staticmethod
    def get_fusionlab_logger(logger_name: str = '') -> logging.Logger:
        return logging.getLogger(logger_name)

    @staticmethod
    def set_logger_output(
        log_filename: str = "fusionlab.log",
        date_format: str = '%Y-%m-%d %H:%M:%S',
        file_mode: str = "w",
        format_: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level: int = logging.DEBUG
    ) -> None:
        handler = logging.FileHandler(log_filename, mode=file_mode)
        handler.setLevel(level)
        formatter = logging.Formatter(format_, datefmt=date_format)
        handler.setFormatter(formatter)

        logger = fusionlog.get_fusionlab_logger()
        logger.setLevel(level)
        logger.addHandler(handler)

        logger.handlers = list(set(logger.handlers))


class OncePerMessageFilter(logging.Filter):
    """Filter that lets each distinct log message through exactly once."""
    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:   # noqa: D401
        key = record.getMessage()
        if key in self._seen:
            return False          # suppress duplicate
        self._seen.add(key)
        return True


def setup_logging(config_path: str = 'path/to/_flog.yml') -> None:
    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f.read())

    log_path = os.getenv('LOG_PATH', config.get('default_log_path', '/fallback/path'))

    if 'handlers' in config:
        for handler in config['handlers'].values():
            if 'filename' in handler:
                handler['filename'] = handler['filename'].replace('${LOG_PATH}', log_path)

    logging.config.dictConfig(config)


if __name__ == '__main__':
    """
    Entry point for testing the `fusionlog` module.

    Prints the absolute path of the `fusionlog` module.
    """
    print(os.path.abspath(fusionlog.__name__))
