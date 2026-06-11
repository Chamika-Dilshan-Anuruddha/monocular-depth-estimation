"""Abstract base class that every depth-estimator backend mush implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from depth_estimation.utils import get_logger

log = get_logger(__name__)

ImageLike = Union[np.ndarray, Image.Image, str, Path]


class BaseDepthEstimator(ABC):
    """Common interface for MiDaS and Depth Anyting v2 backends."""

    def __init__(self, device: torch.device | None = None) -> None:
        from depth_estimation.utils import get_device

        self.device = device or get_device()
        self._model_loaded = False
        log.debug("%s initialised on %s", self.__class__.__name__, self.device)

    # ----- Lifecycle ---------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Download weights (if needed) and move model to 'self.device'."""

    def ensure_loaded(self) -> None:
        if not self._model_loaded:
            log.info("Loading %s ...", self.__class__.__name__)
            self.load()
            self._model_loaded = True
            log.info("%s ready.", self.__class__.__name__)

    # ----- Core inference -----------------------------------------------------------
    
    @abstractmethod
    def _infer(self, image: np.ndarray) -> np.ndarray:
        """Run model inrefence on a preprocessed RGB uint8 array."""

    def predict(self, image: ImageLike) -> np.ndarray:
        """Public entry-point: accepts PIL, ndarray, or a file path."""

        self.ensure_loaded()
        rgb = self._to_rgb_array(image)
        depth = self._infer(rgb)
        return depth.astype(np.float32)
    
    # ----- Helpers ------------------------------------------------------------------

    @staticmethod
    def _to_rgb_array(image: ImageLike) -> np.ndarray:
        if isinstance(image, (str,Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(image)}")
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return image.astype(np.uint8)
    

    # ---- Dunder -------------------------------------------------------------------

    def __repr__(self) -> str:
        """Calls when object is printed."""
        status = "loaded" if self._model_loaded else "not loaded"
        return f"{self.__class__.__name__}(device={self.device}, {status})"
        