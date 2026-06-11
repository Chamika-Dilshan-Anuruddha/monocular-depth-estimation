""" Sturctured logging setup for the depth estimation project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s:%(line)d - %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str, level: Optional[str] = None,  log_file: Optional[str | Path] = None) -> logging.Logger:

    from depth_estimation.config import settings as _settings

    effective_level = (level or _settings.log_level).upper()
    effective_file = log_file or _settings.log_file

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    logger.setLevel(effective_level)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(effective_level)
    logger.addHandler(sh)

    if effective_file:
        fh = logging.FileHandler(Path(effective_file), encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(effective_level)
        logger.addHandler(fh)
    
    logger.propagate = False
    return logger




