"""Depth Anything v2 monocular depth estimation."""

from __future__ import annotations

import numpy as np
import torch

from depth_estimation.config import DepthAnythingSize, settings
from depth_estimation.model.base_estimator import BaseDepthEstimator
from depth_estimation.utils import get_logger

log = get_logger(__name__)

# Huggingface model IDs for each size varient
_HF_MODEL_IDS: dict[str, str] = {
    "vits" : "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb" : "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl" : "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthAnythingV2Estimator(BaseDepthEstimator):

    def __init__(self, size: DepthAnythingSize | str = DepthAnythingSize.SMALL, device: torch.device | None = None) -> None:
        super().__init__(device=device)
        self.size = (size.value if isinstance(size, DepthAnythingSize) else size)
        self._pipe = None

    # ----- Lifecycle ----------------------------------------------------------------------------------------------------

    def load(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as e:
            raise ImportError(
                "transformers is required for Depth Anything v2.",
                "Install it with: pip install transformers"
            ) from e

        model_id = _HF_MODEL_IDS[self.size]
        log.info("Loading Depth Anything v2 (%s) from HuggingFace ...", model_id)

        device_arg: int | str
        if self.device.type == "cuda":
            device_arg = self.device.index or 0
        elif self.device.type == "mps":
            device_arg = "mps"
        else:
            device_arg = -1 #cpu

        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=device_arg
        )
        log.info("Depth Anything v2 (%s) loaded successfully.", self.size)

    # ----- Inference ----------------------------------------------------------------------------------------------------

    def _infer(self,image: np.ndarray) -> np.ndarray:
        assert self._pipe is not None

        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        result = self._pipe(pil_image)
        depth_pil: PILImage.Image = result["depth"]

        depth = np.array(depth_pil, dtype=np.float32)

        # Normalize to [0,1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return depth
    
    
    # ----- Properties ---------------------------------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return f"DepthAnythingV2/{self.size}"
    
    def __repr__(self) -> str:
        status = "loaded" if self._model_loaded else "not loaded"
        return (
            f"DepthAnythingV2Estimator(size={self.size}), "
            f"device={self.device}, {status}"
        )
